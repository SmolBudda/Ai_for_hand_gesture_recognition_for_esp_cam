from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .paths import REPORTS_DIR, ensure_dirs


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], output_path: Path) -> None:
    ensure_dirs(output_path.parent)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(max(8, len(labels) * 0.6), max(6, len(labels) * 0.5)))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_feature_importance(importances: np.ndarray, feature_names: list[str], output_path: Path, top_n: int = 30) -> None:
    ensure_dirs(output_path.parent)
    order = np.argsort(importances)[::-1][:top_n]
    df = pd.DataFrame({"feature": [feature_names[i] for i in order], "importance": importances[order]})
    plt.figure(figsize=(10, max(5, len(df) * 0.28)))
    sns.barplot(data=df, x="importance", y="feature", color="#4C78A8")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_class_counts(metadata_csv: Path, output_path: Path) -> None:
    ensure_dirs(output_path.parent)
    df = pd.read_csv(metadata_csv)
    counts = df.groupby(["split", "label"]).size().reset_index(name="count")
    plt.figure(figsize=(12, 7))
    sns.barplot(data=counts, x="label", y="count", hue="split")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_detection_diagnostics(metadata_csv: Path, output_dir: Path = REPORTS_DIR) -> Path:
    ensure_dirs(output_dir)
    df = pd.read_csv(metadata_csv)
    diagnostics = (
        df.groupby(["label", "detection_count"]).size().reset_index(name="count").sort_values(["label", "detection_count"])
    )
    path = output_dir / "detection_diagnostics.csv"
    diagnostics.to_csv(path, index=False)
    return path


def plot_tuning_parameter_effects(results_csv: Path, output_dir: Path = REPORTS_DIR, prefix: str = "grid") -> None:
    ensure_dirs(output_dir)
    df = pd.read_csv(results_csv)
    if df.empty:
        return
    for parameter in ("n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features", "ccp_alpha"):
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=df, x=parameter, y="val_macro_f1", color="#72B7B2")
        sns.stripplot(data=df, x=parameter, y="val_macro_f1", color="#222222", size=3, alpha=0.5)
        plt.xlabel(parameter)
        plt.ylabel("Validation macro F1")
        plt.tight_layout()
        plt.savefig(output_dir / f"tuning_{prefix}_{parameter}.png", dpi=160)
        plt.close()


def plot_tuning_successive_rounds(results_csv: Path, output_path: Path) -> None:
    ensure_dirs(output_path.parent)
    df = pd.read_csv(results_csv)
    if df.empty:
        return
    summary = df.groupby("round", as_index=False)["val_macro_f1"].agg(["max", "mean"]).reset_index()
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=summary, x="round", y="max", marker="o", label="best")
    sns.lineplot(data=summary, x="round", y="mean", marker="o", label="mean")
    plt.xlabel("Successive-halving round")
    plt.ylabel("Validation macro F1")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_tuning_method_comparison(results_csv: Path, output_path: Path) -> None:
    ensure_dirs(output_path.parent)
    df = pd.read_csv(results_csv)
    if df.empty:
        return
    best = df.sort_values(["val_macro_f1", "val_accuracy"], ascending=[False, False]).groupby("method", as_index=False).first()
    time_summary = df.groupby("method", as_index=False)["elapsed_seconds"].sum().rename(columns={"elapsed_seconds": "total_seconds"})
    best = best.merge(time_summary, on="method", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sns.barplot(data=best, x="method", y="val_macro_f1", ax=axes[0], color="#4C78A8")
    axes[0].set_ylabel("Best validation macro F1")
    axes[0].set_xlabel("Method")
    sns.barplot(data=best, x="method", y="total_seconds", ax=axes[1], color="#F58518")
    axes[1].set_ylabel("Total tuning seconds")
    axes[1].set_xlabel("Method")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_tuning_size_vs_score(results_csv: Path, output_path: Path) -> None:
    ensure_dirs(output_path.parent)
    df = pd.read_csv(results_csv)
    if df.empty:
        return
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="total_nodes", y="val_macro_f1", hue="method", size="n_estimators", alpha=0.75)
    plt.xlabel("Total tree nodes")
    plt.ylabel("Validation macro F1")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
