from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import joblib
import numpy as np
import pandas as pd
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .io import read_json, write_json
from .logging import console
from .paths import FEATURES_DIR, MODELS_DIR, REPORTS_DIR, ensure_dirs
from .reports import plot_class_counts, plot_confusion_matrix, plot_feature_importance, save_detection_diagnostics


LOGGER = logging.getLogger(__name__)


def _train_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )


def _load_split_arrays(features_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    return np.load(features_dir / f"x_{split}.npy"), np.load(features_dir / f"y_{split}.npy")


def _export_tree(model: RandomForestClassifier, feature_names: list[str], output_path: Path) -> None:
    tree = model.estimators_[0].tree_
    classes = model.classes_.tolist()
    nodes = []
    for node_id in range(tree.node_count):
        feature_index = int(tree.feature[node_id])
        nodes.append(
            {
                "id": node_id,
                "left": int(tree.children_left[node_id]),
                "right": int(tree.children_right[node_id]),
                "feature": None if feature_index < 0 else feature_names[feature_index],
                "threshold": float(tree.threshold[node_id]),
                "value": tree.value[node_id].reshape(-1).astype(float).tolist(),
            }
        )
    write_json(output_path, {"classes": classes, "nodes": nodes})


def train_random_forest(
    features_dir: Path = FEATURES_DIR,
    output_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    seed: int = 42,
    n_estimators: int = 300,
    max_depth: int | None = 18,
    min_samples_leaf: int = 2,
    min_samples_split: int = 2,
    max_features: str | float | None = "sqrt",
    ccp_alpha: float = 0.0,
    class_weight: str | None = "balanced_subsample",
    export_tree: bool = False,
) -> tuple[Path, Path]:
    ensure_dirs(output_dir, reports_dir)
    x_train, y_train = _load_split_arrays(features_dir, "train")
    x_val, y_val = _load_split_arrays(features_dir, "val")
    x_test, y_test = _load_split_arrays(features_dir, "test")
    summary = read_json(features_dir / "feature_summary.json")
    label_map = read_json(features_dir / "label_map.json")
    labels = [label_map[str(index)] for index in range(len(label_map))]
    feature_names = summary["feature_names"]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        ccp_alpha=ccp_alpha,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=seed,
    )
    with _train_progress() as progress:
        task_id = progress.add_task("Training and evaluating Random Forest", total=3)
        model.fit(x_train, y_train)
        progress.advance(task_id)
        val_pred = model.predict(x_val) if len(x_val) else np.array([], dtype=int)
        progress.advance(task_id)
        test_pred = model.predict(x_test) if len(x_test) else np.array([], dtype=int)
        progress.advance(task_id)

    val_accuracy = float(accuracy_score(y_val, val_pred)) if len(y_val) else None
    test_accuracy = float(accuracy_score(y_test, test_pred)) if len(y_test) else None
    test_macro_f1 = float(f1_score(y_test, test_pred, average="macro", zero_division=0)) if len(y_test) else None
    report = classification_report(y_test, test_pred, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0) if len(y_test) else {}

    model_path = output_dir / "random_forest.joblib"
    joblib.dump(model, model_path)
    joblib.dump({"model": model, "label_map": label_map, "feature_names": feature_names}, output_dir / "pipeline_bundle.joblib")
    metrics = {
        "validation": {"accuracy": val_accuracy},
        "test": {"accuracy": test_accuracy, "macro_f1": test_macro_f1, "classification_report": report},
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "max_features": max_features,
            "ccp_alpha": ccp_alpha,
            "class_weight": class_weight,
            "random_state": seed,
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(
        output_dir / "model_metadata.json",
        {
            "model_type": "RandomForestClassifier",
            "model_path": model_path.as_posix(),
            "label_map": label_map,
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "preprocessor_path": (features_dir / "preprocessor.joblib").as_posix(),
            "metrics_path": (output_dir / "metrics.json").as_posix(),
        },
    )
    pd.DataFrame(report).transpose().to_csv(reports_dir / "classification_report.csv")
    if len(y_val):
        plot_confusion_matrix(y_val, val_pred, labels, reports_dir / "validation_confusion_matrix.png")
    plot_feature_importance(model.feature_importances_, feature_names, reports_dir / "top_feature_importance.png")
    metadata_csv = features_dir / "feature_metadata.csv"
    if metadata_csv.exists():
        plot_class_counts(metadata_csv, reports_dir / "class_counts.png")
        save_detection_diagnostics(metadata_csv, reports_dir)
    if export_tree:
        _export_tree(model, feature_names, output_dir / "tree_0.json")

    LOGGER.info("Saved Random Forest model to %s", model_path)
    return model_path, output_dir / "metrics.json"
