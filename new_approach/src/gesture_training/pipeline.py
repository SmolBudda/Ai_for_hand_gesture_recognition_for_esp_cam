from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

from .cache import dataset_fingerprint, hash_existing_files, run_cached_stage, stage_signature
from .io import read_json
from .paths import ARTIFACTS_DIR, DEFAULT_DATASET_DIR
from .stages.landmarks.runner import extract_landmarks
from .stages.preprocess.runner import preprocess_landmarks
from .stages.split.runner import create_split
from .stages.train.runner import train_random_forest
from .stages.tuning.runner import compare_tuning_methods, retrain_best, tree_stages_from_config, tune_grid, tune_successive


LOGGER = logging.getLogger(__name__)
CONFIG_DIR = Path("configs/pipeline")


def _code(path: str) -> Path:
    return Path("src/gesture_training") / path


def _stage_code(stage: str) -> Path:
    return Path("src/gesture_training/stages") / stage / "runner.py"


def _config(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{name}.json"
    return read_json(path) if path.exists() else {}


class PipelinePaths:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.splits = root / "01_splits"
        self.landmarks = root / "02_landmarks"
        self.features = root / "03_features"
        self.models = root / "04_models"
        self.reports = root / "05_reports"
        self.tuning = root / "06_tuning"


def run_cached_pipeline(
    artifact_root: Path = ARTIFACTS_DIR,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    limit_per_class: int | None = None,
    hand_landmarker_model: Path = Path("models/hand_landmarker.task"),
    seed: int = 42,
    n_estimators: int = 300,
    max_depth: int | None = 18,
    min_samples_leaf: int = 2,
    min_samples_split: int = 2,
    max_features: str | float | None = "sqrt",
    ccp_alpha: float = 0.0,
    export_tree: bool = False,
    drop_zero_hand: bool = True,
    with_tuning: bool = False,
    tuning_budget_minutes: float = 15.0,
    force: bool = False,
) -> tuple[Path, Path]:
    paths = PipelinePaths(artifact_root)
    split_defaults = _config("split")
    landmark_defaults = _config("landmarks")
    preprocess_defaults = _config("preprocess")
    train_defaults = _config("train")
    tuning_defaults = _config("tuning")

    seed = seed if seed is not None else int(split_defaults.get("seed", train_defaults.get("seed", 42)))
    n_estimators = n_estimators if n_estimators is not None else int(train_defaults.get("n_estimators", 300))
    max_depth = max_depth if max_depth is not None else train_defaults.get("max_depth", 18)
    min_samples_leaf = min_samples_leaf if min_samples_leaf is not None else int(train_defaults.get("min_samples_leaf", 2))
    min_samples_split = min_samples_split if min_samples_split is not None else int(train_defaults.get("min_samples_split", 2))
    max_features = max_features if max_features is not None else train_defaults.get("max_features", "sqrt")
    ccp_alpha = ccp_alpha if ccp_alpha is not None else float(train_defaults.get("ccp_alpha", 0.0))
    export_tree = export_tree if export_tree is not None else bool(train_defaults.get("export_tree", False))
    drop_zero_hand = drop_zero_hand if drop_zero_hand is not None else bool(preprocess_defaults.get("drop_zero_hand", True))
    tuning_budget_minutes = tuning_budget_minutes if tuning_budget_minutes is not None else float(tuning_defaults.get("tuning_budget_minutes", 15.0))

    split_outputs = [paths.splits / "splits.csv", paths.splits / "split_summary.json"]
    ratios = split_defaults.get("ratios", {"train": 0.70, "val": 0.15, "test": 0.15})
    folds = int(split_defaults.get("folds", 5))
    split_config = {
        "dataset_dir": dataset_dir.as_posix(),
        "limit_per_class": limit_per_class,
        "seed": seed,
        "folds": folds,
        "ratios": ratios,
        "config_file": (CONFIG_DIR / "split.json").as_posix(),
    }
    split_sig = stage_signature(
        "split",
        split_config,
        {"dataset": dataset_fingerprint(dataset_dir)},
        [_code("split.py"), _stage_code("split"), CONFIG_DIR / "split.json"],
    )
    dirty = run_cached_stage(
        "split",
        paths.splits,
        split_outputs,
        split_sig,
        split_config,
        lambda: create_split(
            dataset_dir,
            paths.splits,
            train_ratio=float(ratios["train"]),
            val_ratio=float(ratios["val"]),
            test_ratio=float(ratios["test"]),
            seed=seed,
            folds=folds,
            limit_per_class=limit_per_class,
        ),
        force=force,
    )

    landmark_outputs = [paths.landmarks / "landmarks.jsonl", paths.landmarks / "extraction_diagnostics.json"]
    landmark_config = {
        "hand_landmarker_model": hand_landmarker_model.as_posix(),
        "max_num_hands": int(landmark_defaults.get("max_num_hands", 2)),
        "model_complexity": int(landmark_defaults.get("model_complexity", 1)),
        "min_detection_confidence": float(landmark_defaults.get("min_detection_confidence", 0.5)),
        "min_tracking_confidence": float(landmark_defaults.get("min_tracking_confidence", 0.5)),
        "config_file": (CONFIG_DIR / "landmarks.json").as_posix(),
    }
    landmark_sig = stage_signature(
        "landmarks",
        landmark_config,
        {"files": hash_existing_files([paths.splits / "splits.csv", hand_landmarker_model])},
        [_code("landmarks.py"), _stage_code("landmarks"), _code("cache.py"), CONFIG_DIR / "landmarks.json"],
    )
    stage_ran = run_cached_stage(
        "landmarks",
        paths.landmarks,
        landmark_outputs,
        landmark_sig,
        landmark_config,
        lambda: extract_landmarks(
            paths.splits / "splits.csv",
            paths.landmarks,
            hand_landmarker_model=hand_landmarker_model,
            max_num_hands=landmark_config["max_num_hands"],
            model_complexity=landmark_config["model_complexity"],
            min_detection_confidence=landmark_config["min_detection_confidence"],
            min_tracking_confidence=landmark_config["min_tracking_confidence"],
        ),
        force=force,
        downstream_dirty=dirty,
    )
    dirty = dirty or stage_ran

    feature_outputs = [
        paths.features / "x_train.npy",
        paths.features / "y_train.npy",
        paths.features / "x_val.npy",
        paths.features / "y_val.npy",
        paths.features / "x_test.npy",
        paths.features / "y_test.npy",
        paths.features / "preprocessor.joblib",
        paths.features / "label_map.json",
        paths.features / "feature_summary.json",
        paths.features / "feature_metadata.csv",
    ]
    preprocess_config = {"drop_zero_hand": drop_zero_hand, "config_file": (CONFIG_DIR / "preprocess.json").as_posix()}
    preprocess_sig = stage_signature(
        "preprocess",
        preprocess_config,
        {"files": hash_existing_files([paths.landmarks / "landmarks.jsonl"])},
        [_code("features.py"), _stage_code("preprocess"), CONFIG_DIR / "preprocess.json"],
    )
    stage_ran = run_cached_stage(
        "preprocess",
        paths.features,
        feature_outputs,
        preprocess_sig,
        preprocess_config,
        lambda: preprocess_landmarks(paths.landmarks / "landmarks.jsonl", paths.features, drop_zero_hand=drop_zero_hand),
        force=force,
        downstream_dirty=dirty,
    )
    dirty = dirty or stage_ran

    model_outputs = [
        paths.models / "random_forest.joblib",
        paths.models / "pipeline_bundle.joblib",
        paths.models / "metrics.json",
        paths.models / "model_metadata.json",
    ]
    train_config: dict[str, Any] = {
        "seed": seed,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        "ccp_alpha": ccp_alpha,
        "export_tree": export_tree,
        "config_file": (CONFIG_DIR / "train.json").as_posix(),
    }
    train_sig = stage_signature(
        "train",
        train_config,
        {
            "files": hash_existing_files(
                [
                    paths.features / "x_train.npy",
                    paths.features / "y_train.npy",
                    paths.features / "x_val.npy",
                    paths.features / "y_val.npy",
                    paths.features / "x_test.npy",
                    paths.features / "y_test.npy",
                    paths.features / "label_map.json",
                    paths.features / "feature_summary.json",
                    paths.features / "preprocessor.joblib",
                ]
            )
        },
        [_code("training.py"), _stage_code("train"), _code("reports.py"), CONFIG_DIR / "train.json"],
    )
    stage_ran = run_cached_stage(
        "train",
        paths.models,
        model_outputs,
        train_sig,
        train_config,
        lambda: train_random_forest(
            paths.features,
            paths.models,
            paths.reports,
            seed=seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            ccp_alpha=ccp_alpha,
            export_tree=export_tree,
        ),
        force=force,
        downstream_dirty=dirty,
    )
    dirty = dirty or stage_ran
    model_path = paths.models / "random_forest.joblib"
    metrics_path = paths.models / "metrics.json"

    if with_tuning:
        per_method_seconds = max(60.0, tuning_budget_minutes * 60 / 2)
        tune_grid(
            paths.features,
            paths.tuning,
            paths.reports,
            subset_fraction=float(tuning_defaults.get("subset_fraction", 0.10)),
            seed=seed,
            grid_config=tuning_defaults.get("grid_search", tuning_defaults.get("grid")),
            max_runtime_seconds=per_method_seconds,
            force=dirty,
        )
        successive_config = tuning_defaults.get("halving_grid_search", tuning_defaults.get("successive", {}))
        tune_successive(
            paths.features,
            paths.tuning,
            paths.reports,
            subset_fraction=float(tuning_defaults.get("subset_fraction", 0.10)),
            seed=seed,
            tree_stages=tree_stages_from_config(successive_config),
            keep_fraction=float(successive_config.get("keep_fraction", 0.25)),
            grid_config=successive_config,
            max_runtime_seconds=per_method_seconds,
            force=dirty,
        )
        compare_tuning_methods(paths.tuning, paths.reports)
        model_path, metrics_path = retrain_best(paths.features, paths.models, paths.tuning, seed=seed, export_tree=export_tree)

    LOGGER.info("Cached pipeline complete")
    return model_path, metrics_path
