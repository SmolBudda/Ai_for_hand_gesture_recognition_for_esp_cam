from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .features import preprocess_landmarks
from .landmarks import extract_landmarks
from .logging import configure_logging, console
from .io import read_json
from .paths import ARTIFACTS_DIR, DEFAULT_DATASET_DIR, FEATURES_DIR, LANDMARKS_DIR, MODELS_DIR, REPORTS_DIR, SPLITS_DIR, TUNING_DIR
from .pipeline import run_cached_pipeline
from .realtime import run_webcam
from .split import create_split
from .training import train_random_forest
from .tuning import compare_tuning_methods, grid_config_for_method, retrain_best, tree_stages_from_config, tune_grid, tune_successive


app = typer.Typer(help="Random Forest gesture training pipeline from MediaPipe hand landmarks.")


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging.")] = False) -> None:
    configure_logging(verbose=verbose)


@app.command("split")
def split_command(
    dataset_dir: Annotated[Path, typer.Option(help="Class-folder image dataset root.")] = DEFAULT_DATASET_DIR,
    output_dir: Annotated[Path, typer.Option(help="Split artifact directory.")] = SPLITS_DIR,
    limit_per_class: Annotated[int | None, typer.Option(help="Optional class cap for smoke runs.")] = None,
    seed: Annotated[int, typer.Option(help="Deterministic split seed.")] = 42,
    folds: Annotated[int, typer.Option(help="Fold metadata count for later CV.")] = 5,
) -> None:
    csv_path, summary_path = create_split(dataset_dir, output_dir, seed=seed, folds=folds, limit_per_class=limit_per_class)
    console.print(f"[green]Split complete[/green]: {csv_path} ({summary_path})")


@app.command("extract-landmarks")
def extract_landmarks_command(
    split_csv: Annotated[Path, typer.Option(help="Split CSV path.")] = SPLITS_DIR / "splits.csv",
    output_dir: Annotated[Path, typer.Option(help="Landmark artifact directory.")] = LANDMARKS_DIR,
    hand_landmarker_model: Annotated[Path, typer.Option(help="MediaPipe Tasks .task model path when legacy solutions API is unavailable.")] = Path("models/hand_landmarker.task"),
    min_detection_confidence: Annotated[float, typer.Option(help="MediaPipe detection threshold.")] = 0.5,
) -> None:
    jsonl_path, diagnostics_path = extract_landmarks(
        split_csv=split_csv,
        output_dir=output_dir,
        hand_landmarker_model=hand_landmarker_model,
        min_detection_confidence=min_detection_confidence,
    )
    console.print(f"[green]Landmark extraction complete[/green]: {jsonl_path} ({diagnostics_path})")


@app.command("preprocess")
def preprocess_command(
    landmarks_jsonl: Annotated[Path, typer.Option(help="Landmark JSONL path.")] = LANDMARKS_DIR / "landmarks.jsonl",
    output_dir: Annotated[Path, typer.Option(help="Feature artifact directory.")] = FEATURES_DIR,
    drop_zero_hand: Annotated[bool, typer.Option(help="Drop records where MediaPipe detected zero hands.")] = True,
) -> None:
    summary_path, preprocessor_path = preprocess_landmarks(landmarks_jsonl, output_dir, drop_zero_hand=drop_zero_hand)
    console.print(f"[green]Preprocessing complete[/green]: {summary_path} ({preprocessor_path})")


@app.command("train")
def train_command(
    features_dir: Annotated[Path, typer.Option(help="Feature artifact directory.")] = FEATURES_DIR,
    output_dir: Annotated[Path, typer.Option(help="Model artifact directory.")] = MODELS_DIR,
    n_estimators: Annotated[int, typer.Option(help="Random Forest tree count.")] = 300,
    max_depth: Annotated[int | None, typer.Option(help="Random Forest max depth.")] = 18,
    min_samples_leaf: Annotated[int, typer.Option(help="Minimum samples per leaf.")] = 2,
    min_samples_split: Annotated[int, typer.Option(help="Minimum samples required to split a node.")] = 2,
    max_features: Annotated[str, typer.Option(help="Features considered per split: sqrt, log2, none, or float.")] = "sqrt",
    ccp_alpha: Annotated[float, typer.Option(help="Cost-complexity pruning alpha.")] = 0.0,
    seed: Annotated[int, typer.Option(help="Model seed.")] = 42,
    export_tree: Annotated[bool, typer.Option(help="Export first tree as compact JSON.")] = False,
) -> None:
    parsed_max_features = _parse_max_features(max_features)
    model_path, metrics_path = train_random_forest(
        features_dir=features_dir,
        output_dir=output_dir,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=parsed_max_features,
        ccp_alpha=ccp_alpha,
        seed=seed,
        export_tree=export_tree,
    )
    console.print(f"[green]Training complete[/green]: {model_path} ({metrics_path})")


def _parse_max_features(value: str) -> str | float | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null"}:
        return None
    if normalized in {"sqrt", "log2"}:
        return normalized
    try:
        return float(normalized)
    except ValueError as exc:
        raise typer.BadParameter("max_features must be sqrt, log2, none, or a float like 0.5") from exc


@app.command("tune-grid")
def tune_grid_command(
    features_dir: Annotated[Path, typer.Option(help="Feature artifact directory.")] = FEATURES_DIR,
    output_dir: Annotated[Path, typer.Option(help="Tuning artifact directory.")] = TUNING_DIR,
    reports_dir: Annotated[Path, typer.Option(help="Report/chart artifact directory.")] = REPORTS_DIR,
    subset_fraction: Annotated[float, typer.Option(help="Stratified fraction of train/val rows used for tuning.")] = 0.10,
    n_estimators: Annotated[int | None, typer.Option(help="Fallback tree count if tuning config does not define n_estimators.")] = None,
    max_combinations: Annotated[int | None, typer.Option(help="Optional cap for quick smoke runs.")] = None,
    max_runtime_minutes: Annotated[float | None, typer.Option(help="Optional time budget for this tuning method.")] = 15.0,
    seed: Annotated[int, typer.Option(help="Tuning seed.")] = 42,
    force: Annotated[bool, typer.Option(help="Ignore tuning cache and retrain.")] = False,
) -> None:
    tuning_config = read_json(Path("configs/pipeline/tuning.json"))
    paths = tune_grid(
        features_dir=features_dir,
        output_dir=output_dir,
        reports_dir=reports_dir,
        subset_fraction=subset_fraction,
        seed=seed,
        n_estimators=n_estimators,
        grid_config=grid_config_for_method(tuning_config, "grid_search"),
        max_combinations=max_combinations,
        max_runtime_seconds=None if max_runtime_minutes is None else max_runtime_minutes * 60,
        force=force,
    )
    compare_tuning_methods(output_dir, reports_dir)
    console.print(f"[green]Grid tuning complete[/green]: {paths.results_csv} ({paths.summary_json})")


@app.command("tune-successive")
def tune_successive_command(
    features_dir: Annotated[Path, typer.Option(help="Feature artifact directory.")] = FEATURES_DIR,
    output_dir: Annotated[Path, typer.Option(help="Tuning artifact directory.")] = TUNING_DIR,
    reports_dir: Annotated[Path, typer.Option(help="Report/chart artifact directory.")] = REPORTS_DIR,
    subset_fraction: Annotated[float, typer.Option(help="Stratified fraction of train/val rows used for tuning.")] = 0.10,
    max_configs: Annotated[int | None, typer.Option(help="Optional cap for quick smoke runs.")] = None,
    max_runtime_minutes: Annotated[float | None, typer.Option(help="Optional time budget for this tuning method.")] = 15.0,
    seed: Annotated[int, typer.Option(help="Tuning seed.")] = 42,
    force: Annotated[bool, typer.Option(help="Ignore tuning cache and retrain.")] = False,
) -> None:
    tuning_config = read_json(Path("configs/pipeline/tuning.json"))
    successive_config = tuning_config.get("halving_grid_search", tuning_config.get("successive", {}))
    paths = tune_successive(
        features_dir=features_dir,
        output_dir=output_dir,
        reports_dir=reports_dir,
        subset_fraction=subset_fraction,
        seed=seed,
        tree_stages=tree_stages_from_config(successive_config),
        keep_fraction=float(successive_config.get("keep_fraction", 0.25)),
        grid_config=grid_config_for_method(tuning_config, "halving_grid_search"),
        max_configs=max_configs,
        max_runtime_seconds=None if max_runtime_minutes is None else max_runtime_minutes * 60,
        force=force,
    )
    compare_tuning_methods(output_dir, reports_dir)
    console.print(f"[green]Successive tuning complete[/green]: {paths.results_csv} ({paths.summary_json})")


@app.command("compare-tuning")
def compare_tuning_command(
    output_dir: Annotated[Path, typer.Option(help="Tuning artifact directory containing grid/successive CSVs.")] = TUNING_DIR,
    reports_dir: Annotated[Path, typer.Option(help="Report/chart artifact directory.")] = REPORTS_DIR,
) -> None:
    compare_tuning_methods(output_dir, reports_dir)
    console.print(f"[green]Tuning comparison complete[/green]: {reports_dir / 'tuning_method_comparison.png'}")


@app.command("retrain-best")
def retrain_best_command(
    features_dir: Annotated[Path, typer.Option(help="Feature artifact directory.")] = FEATURES_DIR,
    seed: Annotated[int, typer.Option(help="Model seed.")] = 42,
    n_estimators_override: Annotated[int | None, typer.Option(help="Override tuned tree count for smoke runs.")] = None,
    export_tree: Annotated[bool, typer.Option(help="Export first tree as compact JSON.")] = False,
) -> None:
    model_path, metrics_path = retrain_best(
        features_dir=features_dir,
        seed=seed,
        n_estimators_override=n_estimators_override,
        export_tree=export_tree,
    )
    console.print(f"[green]Tuned retrain complete[/green]: {model_path} ({metrics_path})")


@app.command("webcam")
def webcam_command(
    model_bundle: Annotated[Path, typer.Option(help="Model bundle produced by train/repro.")] = MODELS_DIR / "pipeline_bundle.joblib",
    preprocessor: Annotated[Path, typer.Option(help="Train-fitted preprocessor produced by preprocess/repro.")] = FEATURES_DIR / "preprocessor.joblib",
    camera_index: Annotated[int, typer.Option(help="Webcam device index.")] = 0,
    width: Annotated[int | None, typer.Option(help="Requested camera width.")] = None,
    height: Annotated[int | None, typer.Option(help="Requested camera height.")] = None,
    mirror: Annotated[bool, typer.Option(help="Mirror the webcam image before detection.")] = True,
    hand_landmarker_model: Annotated[Path, typer.Option(help="MediaPipe Tasks .task model path.")] = Path("models/hand_landmarker.task"),
    min_detection_confidence: Annotated[float, typer.Option(help="MediaPipe detection threshold.")] = 0.5,
    min_tracking_confidence: Annotated[float, typer.Option(help="MediaPipe tracking threshold.")] = 0.5,
    model_complexity: Annotated[int, typer.Option(help="MediaPipe Hands model complexity.")] = 1,
    fullscreen: Annotated[bool, typer.Option(help="Display preview in fullscreen at highest available resolution.")] = False,
) -> None:
    run_webcam(
        model_bundle_path=model_bundle,
        preprocessor_path=preprocessor,
        camera_index=camera_index,
        width=width,
        height=height,
        mirror=mirror,
        hand_landmarker_model=hand_landmarker_model,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        model_complexity=model_complexity,
        fullscreen=fullscreen,
    )


@app.command("run-pipeline")
def run_pipeline_command(
    artifact_root: Annotated[Path, typer.Option(help="Root directory for generated pipeline artifacts.")] = ARTIFACTS_DIR,
    dataset_dir: Annotated[Path, typer.Option(help="Class-folder image dataset root.")] = DEFAULT_DATASET_DIR,
    limit_per_class: Annotated[int | None, typer.Option(help="Optional class cap for smoke runs.")] = None,
    hand_landmarker_model: Annotated[Path, typer.Option(help="MediaPipe Tasks .task model path when legacy solutions API is unavailable.")] = Path("models/hand_landmarker.task"),
    seed: Annotated[int, typer.Option(help="Shared split/model seed.")] = 42,
    n_estimators: Annotated[int, typer.Option(help="Random Forest tree count.")] = 300,
    max_depth: Annotated[int | None, typer.Option(help="Random Forest max depth.")] = 18,
    min_samples_leaf: Annotated[int, typer.Option(help="Minimum samples per leaf.")] = 2,
    min_samples_split: Annotated[int, typer.Option(help="Minimum samples required to split a node.")] = 2,
    max_features: Annotated[str, typer.Option(help="Features considered per split: sqrt, log2, none, or float.")] = "sqrt",
    ccp_alpha: Annotated[float, typer.Option(help="Cost-complexity pruning alpha.")] = 0.0,
    export_tree: Annotated[bool, typer.Option(help="Export first tree as compact JSON.")] = False,
    drop_zero_hand: Annotated[bool, typer.Option(help="Drop records where MediaPipe detected zero hands.")] = True,
    with_tuning: Annotated[bool, typer.Option(help="Run base model, tuning on subset, then tuned retrain.")] = False,
    tuning_budget_minutes: Annotated[float, typer.Option(help="Total budget split across grid and successive tuning.")] = 15.0,
    force: Annotated[bool, typer.Option(help="Ignore stage cache and rerun every pipeline step.")] = False,
) -> None:
    model_path, metrics_path = run_cached_pipeline(
        artifact_root=artifact_root,
        dataset_dir=dataset_dir,
        limit_per_class=limit_per_class,
        hand_landmarker_model=hand_landmarker_model,
        seed=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=_parse_max_features(max_features),
        ccp_alpha=ccp_alpha,
        export_tree=export_tree,
        drop_zero_hand=drop_zero_hand,
        with_tuning=with_tuning,
        tuning_budget_minutes=tuning_budget_minutes,
        force=force,
    )
    console.print(f"[bold green]Pipeline complete[/bold green]: {model_path} ({metrics_path})")


@app.command("repro")
def repro_command(
    artifact_root: Annotated[Path, typer.Option(help="Root directory for generated pipeline artifacts.")] = ARTIFACTS_DIR,
    dataset_dir: Annotated[Path, typer.Option(help="Class-folder image dataset root.")] = DEFAULT_DATASET_DIR,
    limit_per_class: Annotated[int | None, typer.Option(help="Optional class cap for smoke runs.")] = None,
    hand_landmarker_model: Annotated[Path, typer.Option(help="MediaPipe Tasks .task model path when legacy solutions API is unavailable.")] = Path("models/hand_landmarker.task"),
    seed: Annotated[int, typer.Option(help="Shared split/model seed.")] = 42,
    with_tuning: Annotated[bool, typer.Option(help="Run base model, tuning on subset, then tuned retrain.")] = False,
    tuning_budget_minutes: Annotated[float, typer.Option(help="Total budget split across grid and successive tuning.")] = 15.0,
    force: Annotated[bool, typer.Option(help="Ignore stage cache and rerun every pipeline step.")] = False,
) -> None:
    model_path, metrics_path = run_cached_pipeline(
        artifact_root=artifact_root,
        dataset_dir=dataset_dir,
        limit_per_class=limit_per_class,
        hand_landmarker_model=hand_landmarker_model,
        seed=seed,
        with_tuning=with_tuning,
        tuning_budget_minutes=tuning_budget_minutes,
        force=force,
    )
    console.print(f"[bold green]Repro complete[/bold green]: {model_path} ({metrics_path})")
