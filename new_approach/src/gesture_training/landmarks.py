from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import logging
import os

import pandas as pd
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from .cache import suppress_native_stderr_patterns
from .io import write_json, write_jsonl
from .logging import console
from .paths import LANDMARKS_DIR, ensure_dirs


LOGGER = logging.getLogger(__name__)


def _progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} images"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def _empty_record(row: pd.Series, error: str | None = None) -> dict[str, Any]:
    return {
        "image_path": row["image_path"],
        "label": row["label"],
        "split": row["split"],
        "fold_id": int(row["fold_id"]),
        "image_size": {"width": None, "height": None},
        "detection_count": 0,
        "hands": [],
        "error": error,
    }


def _record_from_hands(row: pd.Series, width: int, height: int, hands: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image_path": row["image_path"],
        "label": row["label"],
        "split": row["split"],
        "fold_id": int(row["fold_id"]),
        "image_size": {"width": int(width), "height": int(height)},
        "detection_count": len(hands),
        "hands": hands,
        "error": None,
    }


def _extract_from_image_legacy(image_path: Path, row: pd.Series, hands_model: Any, cv2: Any) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        return _empty_record(row, "image_read_failed")
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands_model.process(rgb)
    handedness = result.multi_handedness or []
    landmarks = result.multi_hand_landmarks or []
    hands = []
    for index, hand_landmarks in enumerate(landmarks[:2]):
        classification = handedness[index].classification[0] if index < len(handedness) else None
        hands.append(
            {
                "index": index,
                "handedness": classification.label if classification else None,
                "score": float(classification.score) if classification else 0.0,
                "landmarks": [
                    {"x": float(point.x), "y": float(point.y), "z": float(point.z)}
                    for point in hand_landmarks.landmark
                ],
            }
        )
    return _record_from_hands(row, width, height, hands)


def _extract_from_image_tasks(image_path: Path, row: pd.Series, landmarker: Any, mp: Any, cv2: Any) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        return _empty_record(row, "image_read_failed")
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    hands = []
    handedness = result.handedness or []
    landmarks = result.hand_landmarks or []
    for index, hand_landmarks in enumerate(landmarks[:2]):
        category = handedness[index][0] if index < len(handedness) and handedness[index] else None
        hands.append(
            {
                "index": index,
                "handedness": category.category_name if category else None,
                "score": float(category.score) if category else 0.0,
                "landmarks": [
                    {"x": float(point.x), "y": float(point.y), "z": float(point.z)}
                    for point in hand_landmarks
                ],
            }
        )
    return _record_from_hands(row, width, height, hands)


def extract_landmarks(
    split_csv: Path,
    output_dir: Path = LANDMARKS_DIR,
    hand_landmarker_model: Path = Path("models/hand_landmarker.task"),
    max_num_hands: int = 2,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> tuple[Path, Path]:
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("MEDIAPIPE_DISABLE_TELEMETRY", "1")
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe/OpenCV dependencies are required for landmark extraction") from exc

    ensure_dirs(output_dir)
    df = pd.read_csv(split_csv)
    jsonl_path = output_dir / "landmarks.jsonl"
    diagnostics_path = output_dir / "extraction_diagnostics.json"
    diagnostics: dict[str, Any] = {
        "total_images": int(len(df)),
        "errors": Counter(),
        "detections_by_label": defaultdict(Counter),
        "detections_by_split": defaultdict(Counter),
    }

    records = []
    total_images = int(len(df))
    LOGGER.info("Starting MediaPipe landmark extraction for %s images", total_images)

    if hasattr(mp, "solutions"):
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        ) as hands_model:
            with _progress() as progress:
                task_id = progress.add_task("Extracting landmarks", total=total_images)
                for _, row in df.iterrows():
                    record = _extract_from_image_legacy(Path(row["image_path"]), row, hands_model, cv2)
                    records.append(record)
                    key = str(record["detection_count"])
                    diagnostics["detections_by_label"][record["label"]][key] += 1
                    diagnostics["detections_by_split"][record["split"]][key] += 1
                    if record["error"]:
                        diagnostics["errors"][record["error"]] += 1
                    progress.advance(task_id)
    else:
        if not hand_landmarker_model.exists():
            raise FileNotFoundError(
                "MediaPipe Tasks API requires a hand landmarker model asset. "
                f"Expected {hand_landmarker_model}. Pass --hand-landmarker-model with a valid .task file."
            )
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(hand_landmarker_model)),
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.IMAGE,
        )
        with suppress_native_stderr_patterns(
            (
                "Failed to send to clearcut",
                "landmark_projection_calculator.cc",
                "init-domain.cc",
                "gl_context.cc",
                "Created TensorFlow Lite XNNPACK delegate",
                "inference_feedback_manager.cc",
            )
        ):
            with vision.HandLandmarker.create_from_options(options) as landmarker:
                with _progress() as progress:
                    task_id = progress.add_task("Extracting landmarks", total=total_images)
                    for _, row in df.iterrows():
                        record = _extract_from_image_tasks(Path(row["image_path"]), row, landmarker, mp, cv2)
                        records.append(record)
                        key = str(record["detection_count"])
                        diagnostics["detections_by_label"][record["label"]][key] += 1
                        diagnostics["detections_by_split"][record["split"]][key] += 1
                        if record["error"]:
                            diagnostics["errors"][record["error"]] += 1
                        progress.advance(task_id)

    write_jsonl(jsonl_path, records)
    serializable_diagnostics = {
        "total_images": diagnostics["total_images"],
        "errors": dict(diagnostics["errors"]),
        "detections_by_label": {label: dict(counts) for label, counts in diagnostics["detections_by_label"].items()},
        "detections_by_split": {split: dict(counts) for split, counts in diagnostics["detections_by_split"].items()},
    }
    write_json(diagnostics_path, serializable_diagnostics)
    LOGGER.info("Wrote %s landmark records to %s", len(records), jsonl_path)
    return jsonl_path, diagnostics_path
