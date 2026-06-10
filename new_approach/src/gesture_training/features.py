from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from .io import read_jsonl, write_json
from .logging import console
from .paths import FEATURES_DIR, ensure_dirs


LOGGER = logging.getLogger(__name__)
HAND_SLOTS = ("left", "right")
LANDMARK_COUNT = 21
AXES = ("x", "y", "z")


def _progress() -> Progress:
    return Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} records"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


@dataclass
class FeatureBuildResult:
    features: np.ndarray
    labels: np.ndarray
    splits: np.ndarray
    feature_names: list[str]
    metadata: list[dict[str, Any]]


def feature_names() -> list[str]:
    names: list[str] = []
    for slot in HAND_SLOTS:
        for point_index in range(LANDMARK_COUNT):
            for axis in AXES:
                names.append(f"{slot}_lm{point_index}_{axis}")
        names.extend([f"{slot}_present", f"{slot}_score"])
    return names


def _normalize_landmarks(landmarks: list[dict[str, float]]) -> np.ndarray:
    if len(landmarks) != LANDMARK_COUNT:
        raise ValueError(f"Expected {LANDMARK_COUNT} landmarks, got {len(landmarks)}")
    values = np.array([[point["x"], point["y"], point["z"]] for point in landmarks], dtype=np.float32)
    values = values - values[0]
    scale = float(np.max(np.linalg.norm(values[:, :2], axis=1)))
    if scale <= 1e-8:
        scale = 1.0
    return values / scale


def canonicalize_hands(hands: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    slots: dict[str, dict[str, Any] | None] = {"left": None, "right": None}
    overflow: list[dict[str, Any]] = []
    for hand in sorted(hands, key=lambda item: float(item.get("score") or 0.0), reverse=True):
        handedness = str(hand.get("handedness") or "").lower()
        if handedness in slots and slots[handedness] is None:
            slots[handedness] = hand
        else:
            overflow.append(hand)
    for hand in overflow:
        empty_slots = [slot for slot, value in slots.items() if value is None]
        if empty_slots:
            slots[empty_slots[0]] = hand
    return slots


def record_to_feature(record: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    slots = canonicalize_hands(record.get("hands", []))
    values: list[float] = []
    slot_metadata: dict[str, Any] = {}
    for slot in HAND_SLOTS:
        hand = slots[slot]
        if hand is None:
            values.extend([0.0] * (LANDMARK_COUNT * len(AXES)))
            values.extend([0.0, 0.0])
            slot_metadata[slot] = {"present": False, "score": 0.0, "handedness": None}
            continue
        normalized = _normalize_landmarks(hand.get("landmarks", []))
        values.extend(normalized.reshape(-1).astype(float).tolist())
        score = float(hand.get("score") or 0.0)
        values.extend([1.0, score])
        slot_metadata[slot] = {"present": True, "score": score, "handedness": hand.get("handedness")}
    return np.array(values, dtype=np.float32), slot_metadata


def build_feature_matrix(records: list[dict[str, Any]]) -> FeatureBuildResult:
    names = feature_names()
    rows: list[np.ndarray] = []
    labels: list[str] = []
    splits: list[str] = []
    metadata: list[dict[str, Any]] = []
    with _progress() as progress:
        task_id = progress.add_task("Building features", total=len(records))
        for record in records:
            row, slot_metadata = record_to_feature(record)
            rows.append(row)
            labels.append(record["label"])
            splits.append(record["split"])
            metadata.append(
                {
                    "image_path": record["image_path"],
                    "label": record["label"],
                    "split": record["split"],
                    "fold_id": record.get("fold_id"),
                    "detection_count": record.get("detection_count", 0),
                    "slots": slot_metadata,
                }
            )
            progress.advance(task_id)
    if not rows:
        raise ValueError("No landmark records available for preprocessing")
    return FeatureBuildResult(
        features=np.vstack(rows),
        labels=np.array(labels),
        splits=np.array(splits),
        feature_names=names,
        metadata=metadata,
    )


def _drop_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0, "by_split": {}, "by_label": {}, "by_label_split": {}}
    df = pd.DataFrame(
        {
            "split": record.get("split"),
            "label": record.get("label"),
        }
        for record in records
    )
    return {
        "count": int(len(df)),
        "by_split": {str(key): int(value) for key, value in df.groupby("split").size().items()},
        "by_label": {str(key): int(value) for key, value in df.groupby("label").size().items()},
        "by_label_split": {
            f"{label}:{split}": int(value)
            for (label, split), value in df.groupby(["label", "split"]).size().items()
        },
    }


def preprocess_landmarks(
    landmarks_jsonl: Path,
    output_dir: Path = FEATURES_DIR,
    drop_zero_hand: bool = True,
) -> tuple[Path, Path]:
    ensure_dirs(output_dir)
    records = read_jsonl(landmarks_jsonl)
    input_rows = len(records)
    dropped_records = [record for record in records if int(record.get("detection_count") or 0) == 0] if drop_zero_hand else []
    if drop_zero_hand:
        records = [record for record in records if int(record.get("detection_count") or 0) > 0]
    LOGGER.info("Preprocessing %s records; dropped %s zero-hand records", len(records), len(dropped_records))
    built = build_feature_matrix(records)

    train_mask = built.splits == "train"
    if not np.any(train_mask):
        raise ValueError("Cannot fit scaler: no train rows found")

    scaler = StandardScaler()
    scaler.fit(built.features[train_mask])
    features_scaled = scaler.transform(built.features).astype(np.float32)

    label_encoder = LabelEncoder()
    label_encoder.fit(built.labels[train_mask])
    unknown_labels = sorted(set(built.labels) - set(label_encoder.classes_))
    if unknown_labels:
        raise ValueError(f"Labels appear outside train split: {unknown_labels}")
    y_all = label_encoder.transform(built.labels)

    for split_name in ("train", "val", "test"):
        mask = built.splits == split_name
        np.save(output_dir / f"x_{split_name}.npy", features_scaled[mask])
        np.save(output_dir / f"y_{split_name}.npy", y_all[mask])

    metadata_df = pd.DataFrame(built.metadata)
    metadata_df.to_csv(output_dir / "feature_metadata.csv", index=False)
    label_map = {str(index): label for index, label in enumerate(label_encoder.classes_.tolist())}
    summary = {
        "input_rows": int(input_rows),
        "rows": int(len(built.labels)),
        "kept_rows": int(len(built.labels)),
        "dropped_zero_hand_rows": int(len(dropped_records)),
        "dropped_zero_hand": _drop_summary(dropped_records),
        "feature_count": int(len(built.feature_names)),
        "feature_names": built.feature_names,
        "label_map": label_map,
        "counts_by_split": {split: int((built.splits == split).sum()) for split in sorted(set(built.splits))},
        "preprocessing": {
            "hand_slots": list(HAND_SLOTS),
            "missing_hand_fill": 0.0,
            "normalization": "wrist_relative_max_xy_distance",
            "global_scaler": "StandardScaler fitted on train rows only",
            "drop_zero_hand": drop_zero_hand,
        },
    }
    write_json(output_dir / "feature_summary.json", summary)
    write_json(output_dir / "label_map.json", label_map)
    joblib.dump({"scaler": scaler, "feature_names": built.feature_names}, output_dir / "preprocessor.joblib")
    LOGGER.info("Wrote feature arrays and train-fitted preprocessor to %s", output_dir)
    return output_dir / "feature_summary.json", output_dir / "preprocessor.joblib"
