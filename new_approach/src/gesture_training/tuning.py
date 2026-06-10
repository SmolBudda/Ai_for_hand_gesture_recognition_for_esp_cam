from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any
import hashlib
import json
import logging

import joblib
import numpy as np
import pandas as pd
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from .io import read_json, write_json
from .logging import console
from .paths import FEATURES_DIR, MODELS_DIR, REPORTS_DIR, TUNING_DIR, ensure_dirs
from .reports import (
    plot_tuning_method_comparison,
    plot_tuning_parameter_effects,
    plot_tuning_size_vs_score,
    plot_tuning_successive_rounds,
)
from .training import train_random_forest


LOGGER = logging.getLogger(__name__)


DEFAULT_STRUCTURAL_GRID: dict[str, list[Any]] = {
    "n_estimators": [100],
    "max_depth": [8, 12, 16, 20, None],
    "min_samples_leaf": [1, 2, 5, 10],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", 0.5],
    "ccp_alpha": [0.0],
}


def _unique_preserving_order(values: list[Any]) -> list[Any]:
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _range_values(name: str, spec: dict[str, Any]) -> list[int | float]:
    missing = {"min", "max", "count"} - spec.keys()
    if missing:
        raise ValueError(f"Grid range for {name} is missing: {', '.join(sorted(missing))}")
    count = int(spec["count"])
    if count < 1:
        raise ValueError(f"Grid range for {name} must have count >= 1")
    minimum = float(spec["min"])
    maximum = float(spec["max"])
    if minimum > maximum:
        raise ValueError(f"Grid range for {name} must have min <= max")

    value_type = spec.get("type")
    if value_type is None:
        value_type = "int" if isinstance(spec["min"], int) and isinstance(spec["max"], int) else "float"
    if value_type not in {"int", "float"}:
        raise ValueError(f"Grid range for {name} type must be 'int' or 'float'")

    values = np.linspace(minimum, maximum, count).tolist()
    if value_type == "float":
        return [float(value) for value in values]

    rounded = _unique_preserving_order([int(round(value)) for value in values])
    if len(rounded) != count:
        raise ValueError(f"Grid range for {name} produced {len(rounded)} distinct integers, expected {count}")
    return rounded


def tuning_grid_from_config(config: dict[str, Any] | None) -> dict[str, list[Any]]:
    if not config:
        return DEFAULT_STRUCTURAL_GRID
    source = config.get("parameters", config.get("grid", config))
    resolved: dict[str, list[Any]] = {}
    for name, spec in source.items():
        if isinstance(spec, list):
            values = spec
        elif isinstance(spec, dict) and "values" in spec:
            values = spec["values"]
        elif isinstance(spec, dict):
            values = _range_values(name, spec)
        else:
            raise ValueError(f"Grid parameter {name} must be a list, values object, or range object")
        if isinstance(spec, dict) and name == "max_depth" and bool(spec.get("include_unlimited", spec.get("include_none", False))):
            values = [*values, None]
        if not values:
            raise ValueError(f"Grid parameter {name} must define at least one value")
        resolved[name] = _unique_preserving_order(values)
    return resolved


def grid_config_for_method(config: dict[str, Any] | None, method: str) -> dict[str, Any] | None:
    if not config:
        return None
    if "parameters" in config:
        return config
    return config.get(method) or config.get("grid")


def make_tree_stages(start_trees: int, final_trees: int, rounds: int) -> tuple[int, ...]:
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if start_trees < 1 or final_trees < 1:
        raise ValueError("start_trees and final_trees must be >= 1")
    if rounds == 1:
        return (int(final_trees),)
    if start_trees > final_trees:
        raise ValueError("start_trees must be <= final_trees")
    ratio = (final_trees / start_trees) ** (1 / (rounds - 1))
    stages = [max(1, int(round(start_trees * (ratio**index)))) for index in range(rounds)]
    stages[0] = int(start_trees)
    stages[-1] = int(final_trees)
    for index in range(1, len(stages)):
        if stages[index] <= stages[index - 1]:
            stages[index] = stages[index - 1] + 1
    stages[-1] = int(final_trees)
    return tuple(stages)


def tree_stages_from_config(config: dict[str, Any]) -> tuple[int, ...]:
    if "tree_stages" in config:
        return tuple(int(value) for value in config["tree_stages"])
    return make_tree_stages(
        start_trees=int(config.get("start_trees", 50)),
        final_trees=int(config.get("final_trees", 400)),
        rounds=int(config.get("rounds", 3)),
    )


@dataclass(frozen=True)
class TuningPaths:
    results_csv: Path
    summary_json: Path


def _load_arrays(features_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.load(features_dir / "x_train.npy"),
        np.load(features_dir / "y_train.npy"),
        np.load(features_dir / "x_val.npy"),
        np.load(features_dir / "y_val.npy"),
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _cache_payload(
    method: str,
    features_dir: Path,
    subset_fraction: float,
    seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    input_files = ["x_train.npy", "y_train.npy", "x_val.npy", "y_val.npy", "label_map.json"]
    return {
        "method": method,
        "subset_fraction": subset_fraction,
        "seed": seed,
        "config": config,
        "inputs": {name: _hash_file(features_dir / name) for name in input_files},
    }


def _progress() -> Progress:
    return Progress(
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} models"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def _stratified_indices(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("subset_fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label)
        sample_count = max(1, int(round(len(label_indices) * fraction)))
        sample_count = min(sample_count, len(label_indices))
        selected.extend(rng.choice(label_indices, size=sample_count, replace=False).tolist())
    return np.array(sorted(selected), dtype=int)


def make_tuning_subset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    subset_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(y_train) == 0 or len(y_val) == 0:
        raise ValueError("Tuning requires non-empty train and validation feature arrays")
    train_idx = _stratified_indices(y_train, subset_fraction, seed)
    val_idx = _stratified_indices(y_val, subset_fraction, seed + 1)
    return x_train[train_idx], y_train[train_idx], x_val[val_idx], y_val[val_idx]


def _grid_values(grid: dict[str, list[Any]] | None = None) -> list[dict[str, Any]]:
    source = grid or DEFAULT_STRUCTURAL_GRID
    keys = list(source.keys())
    return [dict(zip(keys, values, strict=True)) for values in product(*(source[key] for key in keys))]


def _model_stats(model: RandomForestClassifier) -> dict[str, float | int]:
    depths = [tree.tree_.max_depth for tree in model.estimators_]
    nodes = [tree.tree_.node_count for tree in model.estimators_]
    return {
        "tree_count": int(len(model.estimators_)),
        "avg_tree_depth": float(np.mean(depths)) if depths else 0.0,
        "max_tree_depth": int(max(depths)) if depths else 0,
        "total_nodes": int(sum(nodes)),
        "avg_nodes": float(np.mean(nodes)) if nodes else 0.0,
    }


def _fit_and_score(
    params: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    method: str,
    round_index: int,
) -> dict[str, Any]:
    start = perf_counter()
    model = RandomForestClassifier(
        **params,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    elapsed = perf_counter() - start
    stats = _model_stats(model)
    return {
        "method": method,
        "round": round_index,
        "elapsed_seconds": elapsed,
        "val_macro_f1": float(f1_score(y_val, pred, average="macro", zero_division=0)),
        "val_accuracy": float(accuracy_score(y_val, pred)),
        "n_estimators": int(params["n_estimators"]),
        "max_depth": params.get("max_depth"),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "min_samples_split": int(params["min_samples_split"]),
        "max_features": params.get("max_features"),
        "ccp_alpha": float(params.get("ccp_alpha", 0.0)),
        **stats,
    }


def _write_results(rows: list[dict[str, Any]], output_dir: Path, filename: str, summary_name: str, cache_payload: dict[str, Any]) -> TuningPaths:
    ensure_dirs(output_dir)
    df = pd.DataFrame(rows).sort_values(["val_macro_f1", "val_accuracy", "total_nodes"], ascending=[False, False, True])
    results_csv = output_dir / filename
    summary_json = output_dir / summary_name
    df.to_csv(results_csv, index=False)
    best = df.iloc[0].to_dict() if len(df) else {}
    write_json(
        summary_json,
        {
            "best": best,
            "row_count": int(len(df)),
            "cache": cache_payload,
        },
    )
    return TuningPaths(results_csv=results_csv, summary_json=summary_json)


def tune_grid(
    features_dir: Path = FEATURES_DIR,
    output_dir: Path = TUNING_DIR,
    reports_dir: Path = REPORTS_DIR,
    subset_fraction: float = 0.10,
    seed: int = 42,
    n_estimators: int | None = None,
    grid_config: dict[str, Any] | None = None,
    max_combinations: int | None = None,
    max_runtime_seconds: float | None = None,
    force: bool = False,
) -> TuningPaths:
    ensure_dirs(output_dir, reports_dir)
    structural_grid = tuning_grid_from_config(grid_config)
    if n_estimators is not None and "n_estimators" not in structural_grid:
        structural_grid = {**structural_grid, "n_estimators": [n_estimators]}
    grid = _grid_values(structural_grid)
    if max_combinations is not None:
        grid = grid[:max_combinations]
    config = {
        "grid": structural_grid,
        "max_combinations": max_combinations,
        "max_runtime_seconds": max_runtime_seconds,
    }
    payload = _cache_payload("grid", features_dir, subset_fraction, seed, config)
    cache_key = _json_key(payload)
    results_csv = output_dir / "grid_results.csv"
    summary_json = output_dir / "grid_summary.json"
    manifest_path = output_dir / "grid_cache_manifest.json"
    if not force and manifest_path.exists() and results_csv.exists():
        manifest = read_json(manifest_path)
        if manifest.get("cache_key") == cache_key:
            LOGGER.info("Grid tuning cache hit: %s", results_csv)
            return TuningPaths(results_csv, summary_json)

    x_train, y_train, x_val, y_val = _load_arrays(features_dir)
    xs_train, ys_train, xs_val, ys_val = make_tuning_subset(x_train, y_train, x_val, y_val, subset_fraction, seed)
    LOGGER.info("Grid tuning %s configs on %s train / %s val rows", len(grid), len(ys_train), len(ys_val))

    rows: list[dict[str, Any]] = []
    start = perf_counter()
    with _progress() as progress:
        task_id = progress.add_task("Grid tuning", total=len(grid))
        for params in grid:
            row = _fit_and_score(
                params,
                xs_train,
                ys_train,
                xs_val,
                ys_val,
                seed,
                "grid",
                1,
            )
            rows.append(row)
            progress.advance(task_id)
            if max_runtime_seconds is not None and perf_counter() - start >= max_runtime_seconds:
                LOGGER.info("Grid tuning time budget reached after %s models", len(rows))
                break
    paths = _write_results(rows, output_dir, "grid_results.csv", "grid_summary.json", {**payload, "cache_key": cache_key})
    write_json(manifest_path, {"cache_key": cache_key, **payload})
    plot_tuning_parameter_effects(paths.results_csv, reports_dir, prefix="grid")
    return paths


def tune_successive(
    features_dir: Path = FEATURES_DIR,
    output_dir: Path = TUNING_DIR,
    reports_dir: Path = REPORTS_DIR,
    subset_fraction: float = 0.10,
    seed: int = 42,
    tree_stages: tuple[int, ...] = (50, 150, 400),
    keep_fraction: float = 0.25,
    grid_config: dict[str, Any] | None = None,
    max_configs: int | None = None,
    max_runtime_seconds: float | None = None,
    force: bool = False,
) -> TuningPaths:
    ensure_dirs(output_dir, reports_dir)
    structural_grid = tuning_grid_from_config(grid_config)
    configured_tree_stages = tuple(int(value) for value in structural_grid.pop("n_estimators", list(tree_stages)))
    tree_stages = configured_tree_stages
    candidates = _grid_values(structural_grid)
    if max_configs is not None:
        candidates = candidates[:max_configs]
    config = {
        "tree_stages": list(tree_stages),
        "keep_fraction": keep_fraction,
        "grid": structural_grid,
        "max_configs": max_configs,
        "max_runtime_seconds": max_runtime_seconds,
    }
    payload = _cache_payload("successive", features_dir, subset_fraction, seed, config)
    cache_key = _json_key(payload)
    results_csv = output_dir / "successive_results.csv"
    summary_json = output_dir / "successive_summary.json"
    manifest_path = output_dir / "successive_cache_manifest.json"
    if not force and manifest_path.exists() and results_csv.exists():
        manifest = read_json(manifest_path)
        if manifest.get("cache_key") == cache_key:
            LOGGER.info("Successive tuning cache hit: %s", results_csv)
            return TuningPaths(results_csv, summary_json)

    x_train, y_train, x_val, y_val = _load_arrays(features_dir)
    xs_train, ys_train, xs_val, ys_val = make_tuning_subset(x_train, y_train, x_val, y_val, subset_fraction, seed)
    LOGGER.info("Successive tuning %s configs on %s train / %s val rows", len(candidates), len(ys_train), len(ys_val))

    rows: list[dict[str, Any]] = []
    active = candidates
    total_models = sum(max(1, int(np.ceil(len(candidates) * (keep_fraction ** stage)))) for stage in range(len(tree_stages)))
    start = perf_counter()
    with _progress() as progress:
        task_id = progress.add_task("Successive tuning", total=total_models)
        for round_index, n_estimators in enumerate(tree_stages, start=1):
            round_rows = []
            for params in active:
                row = _fit_and_score(
                    {**params, "n_estimators": n_estimators},
                    xs_train,
                    ys_train,
                    xs_val,
                    ys_val,
                    seed,
                    "successive",
                    round_index,
                )
                rows.append(row)
                round_rows.append(row)
                progress.advance(task_id)
                if max_runtime_seconds is not None and perf_counter() - start >= max_runtime_seconds:
                    LOGGER.info("Successive tuning time budget reached after %s models", len(rows))
                    break
            if not round_rows:
                break
            ranked = sorted(round_rows, key=lambda item: (item["val_macro_f1"], item["val_accuracy"], -item["total_nodes"]), reverse=True)
            keep_count = max(1, int(np.ceil(len(ranked) * keep_fraction)))
            active = [
                {
                    "max_depth": row["max_depth"],
                    "min_samples_leaf": row["min_samples_leaf"],
                    "min_samples_split": row["min_samples_split"],
                    "max_features": row["max_features"],
                    "ccp_alpha": row["ccp_alpha"],
                }
                for row in ranked[:keep_count]
            ]
            if max_runtime_seconds is not None and perf_counter() - start >= max_runtime_seconds:
                break
    paths = _write_results(rows, output_dir, "successive_results.csv", "successive_summary.json", {**payload, "cache_key": cache_key})
    write_json(manifest_path, {"cache_key": cache_key, **payload})
    plot_tuning_successive_rounds(paths.results_csv, reports_dir / "tuning_successive_rounds.png")
    return paths


def _load_best_row(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No tuning rows found in {path}")
    return df.sort_values(["val_macro_f1", "val_accuracy", "total_nodes"], ascending=[False, False, True]).iloc[0].to_dict()


def choose_best_hyperparameters(output_dir: Path = TUNING_DIR, tolerance: float = 0.002) -> Path:
    grid_path = output_dir / "grid_results.csv"
    successive_path = output_dir / "successive_results.csv"
    candidates = []
    if grid_path.exists():
        candidates.append(_load_best_row(grid_path))
    if successive_path.exists():
        candidates.append(_load_best_row(successive_path))
    if not candidates:
        raise FileNotFoundError("No tuning results found. Run tune-grid and/or tune-successive first.")
    candidates = sorted(candidates, key=lambda row: (row["val_macro_f1"], row["val_accuracy"]), reverse=True)
    best = candidates[0]
    for candidate in candidates[1:]:
        if abs(float(best["val_macro_f1"]) - float(candidate["val_macro_f1"])) <= tolerance:
            if int(candidate["total_nodes"]) < int(best["total_nodes"]):
                best = candidate
    params = {
        "method": best["method"],
        "validation_macro_f1": float(best["val_macro_f1"]),
        "validation_accuracy": float(best["val_accuracy"]),
        "total_nodes": int(best["total_nodes"]),
        "hyperparameters": {
            "n_estimators": int(best["n_estimators"]),
            "max_depth": None if pd.isna(best["max_depth"]) else int(best["max_depth"]),
            "min_samples_leaf": int(best["min_samples_leaf"]),
            "min_samples_split": int(best["min_samples_split"]),
            "max_features": None if pd.isna(best["max_features"]) or best["max_features"] == "None" else best["max_features"],
            "ccp_alpha": float(best.get("ccp_alpha", 0.0)),
        },
    }
    if isinstance(params["hyperparameters"]["max_features"], str):
        try:
            params["hyperparameters"]["max_features"] = float(params["hyperparameters"]["max_features"])
        except ValueError:
            pass
    path = output_dir / "best_hyperparameters.json"
    write_json(path, params)
    return path


def compare_tuning_methods(output_dir: Path = TUNING_DIR, reports_dir: Path = REPORTS_DIR) -> None:
    frames = []
    for name in ("grid_results.csv", "successive_results.csv"):
        path = output_dir / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    ensure_dirs(reports_dir)
    combined = reports_dir / "tuning_all_results.csv"
    df.to_csv(combined, index=False)
    plot_tuning_method_comparison(combined, reports_dir / "tuning_method_comparison.png")
    plot_tuning_size_vs_score(combined, reports_dir / "tuning_model_size_vs_score.png")


def retrain_best(
    features_dir: Path = FEATURES_DIR,
    output_dir: Path = MODELS_DIR,
    tuning_dir: Path = TUNING_DIR,
    seed: int = 42,
    n_estimators_override: int | None = None,
    export_tree: bool = False,
) -> tuple[Path, Path]:
    best_path = choose_best_hyperparameters(tuning_dir)
    best = read_json(best_path)
    params = best["hyperparameters"]
    if n_estimators_override is not None:
        params["n_estimators"] = n_estimators_override
    tuned_dir = output_dir / "tuned"
    model_path, metrics_path = train_random_forest(
        features_dir=features_dir,
        output_dir=tuned_dir,
        seed=seed,
        export_tree=export_tree,
        **params,
    )
    write_json(
        tuned_dir / "tuned_training_summary.json",
        {
            "source_best_hyperparameters": best,
            "model_path": model_path.as_posix(),
            "metrics_path": metrics_path.as_posix(),
        },
    )
    return model_path, metrics_path
