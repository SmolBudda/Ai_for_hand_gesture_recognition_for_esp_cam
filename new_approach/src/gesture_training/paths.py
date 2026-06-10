from __future__ import annotations

from pathlib import Path


DEFAULT_DATASET_DIR = Path("tiny_HaGRID/full_set")
ARTIFACTS_DIR = Path("artifacts")
SPLITS_DIR = ARTIFACTS_DIR / "01_splits"
LANDMARKS_DIR = ARTIFACTS_DIR / "02_landmarks"
FEATURES_DIR = ARTIFACTS_DIR / "03_features"
MODELS_DIR = ARTIFACTS_DIR / "04_models"
REPORTS_DIR = ARTIFACTS_DIR / "05_reports"
TUNING_DIR = ARTIFACTS_DIR / "06_tuning"
LOGS_DIR = Path("logs")


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
