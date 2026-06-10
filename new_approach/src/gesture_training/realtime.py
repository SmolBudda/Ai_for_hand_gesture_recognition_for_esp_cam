from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import logging
import os

import joblib
import numpy as np

from .cache import suppress_native_stderr_patterns
from .features import record_to_feature
from .paths import FEATURES_DIR, MODELS_DIR


LOGGER = logging.getLogger(__name__)
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


@dataclass(frozen=True)
class Prediction:
    label: str | None
    confidence: float
    probabilities: dict[str, float]
    detection_count: int


def _label_map_to_list(label_map: dict[str, Any]) -> list[str]:
    return [str(label_map[str(index)]) for index in range(len(label_map))]


def load_realtime_pipeline(model_bundle_path: Path, preprocessor_path: Path) -> tuple[Any, Any, list[str]]:
    if not model_bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_bundle_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
    bundle = joblib.load(model_bundle_path)
    preprocessor = joblib.load(preprocessor_path)
    model = bundle["model"]
    scaler = preprocessor["scaler"]
    labels = _label_map_to_list(bundle["label_map"])
    return model, scaler, labels


def predict_record(record: dict[str, Any], model: Any, scaler: Any, labels: list[str]) -> Prediction:
    detection_count = int(record.get("detection_count") or 0)
    if detection_count == 0:
        return Prediction(label=None, confidence=0.0, probabilities={}, detection_count=0)
    features, _ = record_to_feature(record)
    scaled = scaler.transform(features.reshape(1, -1)).astype(np.float32)
    class_index = int(model.predict(scaled)[0])
    label = labels[class_index]
    probabilities: dict[str, float] = {}
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(scaled)[0]
        probabilities = {labels[index]: float(value) for index, value in enumerate(proba)}
        confidence = float(proba[class_index])
    return Prediction(label=label, confidence=confidence, probabilities=probabilities, detection_count=detection_count)


def _record_from_legacy_result(result: Any, width: int, height: int) -> dict[str, Any]:
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
    return {
        "image_path": "webcam",
        "label": "",
        "split": "realtime",
        "fold_id": 0,
        "image_size": {"width": int(width), "height": int(height)},
        "detection_count": len(hands),
        "hands": hands,
        "error": None,
    }


def _record_from_tasks_result(result: Any, width: int, height: int) -> dict[str, Any]:
    handedness = result.handedness or []
    landmarks = result.hand_landmarks or []
    hands = []
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
    return {
        "image_path": "webcam",
        "label": "",
        "split": "realtime",
        "fold_id": 0,
        "image_size": {"width": int(width), "height": int(height)},
        "detection_count": len(hands),
        "hands": hands,
        "error": None,
    }


def _draw_record_landmarks(cv2: Any, frame: Any, record: dict[str, Any]) -> None:
    height, width = frame.shape[:2]
    for hand in record.get("hands", []):
        points = []
        for landmark in hand.get("landmarks", []):
            x = int(float(landmark["x"]) * width)
            y = int(float(landmark["y"]) * height)
            points.append((x, y))
        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], (90, 180, 255), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 3, (40, 220, 80), -1, cv2.LINE_AA)


def _put_status(cv2: Any, frame: Any, prediction: Prediction, fps: float) -> None:
    if prediction.label is None:
        text = f"No hand detected | {fps:.1f} FPS"
        color = (60, 60, 255)
    else:
        text = f"{prediction.label}  {prediction.confidence:.2f} | hands: {prediction.detection_count} | {fps:.1f} FPS"
        color = (40, 180, 40)
    cv2.rectangle(frame, (8, 8), (min(frame.shape[1] - 8, 620), 52), (0, 0, 0), -1)
    cv2.putText(frame, text, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, "Press q or Esc to quit", (18, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)


def run_webcam(
    model_bundle_path: Path = MODELS_DIR / "pipeline_bundle.joblib",
    preprocessor_path: Path = FEATURES_DIR / "preprocessor.joblib",
    camera_index: int = 0,
    width: int | None = None,
    height: int | None = None,
    mirror: bool = True,
    hand_landmarker_model: Path = Path("models/hand_landmarker.task"),
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_complexity: int = 1,
    fullscreen: bool = False,
) -> None:
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("MEDIAPIPE_DISABLE_TELEMETRY", "1")
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("Realtime webcam inference requires OpenCV and MediaPipe") from exc

    model, scaler, labels = load_realtime_pipeline(model_bundle_path, preprocessor_path)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam at camera index {camera_index}")
    
    # Request maximum resolution if fullscreen is enabled and custom dimensions not specified
    if fullscreen and width is None and height is None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # 4K width
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)  # 4K height
    
    if width is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fps = 0.0
    previous = perf_counter()
    
    # Set up window for fullscreen if requested
    window_name = "Gesture realtime inference"
    if fullscreen:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    try:
        if hasattr(mp, "solutions"):
            mp_hands = mp.solutions.hands
            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            ) as hands_model:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        LOGGER.warning("Failed to read webcam frame")
                        break
                    if mirror:
                        frame = cv2.flip(frame, 1)
                    height_px, width_px = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    result = hands_model.process(rgb)
                    rgb.flags.writeable = True
                    record = _record_from_legacy_result(result, width_px, height_px)
                    prediction = predict_record(record, model, scaler, labels)
                    _draw_record_landmarks(cv2, frame, record)
                    fps, previous = _show_frame(cv2, frame, prediction, fps, previous, fullscreen=fullscreen, window_name=window_name)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
        else:
            if not hand_landmarker_model.exists():
                raise FileNotFoundError(
                    "MediaPipe Tasks realtime inference requires a hand landmarker model asset. "
                    f"Expected {hand_landmarker_model}."
                )
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(hand_landmarker_model)),
                num_hands=2,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_tracking_confidence,
                min_tracking_confidence=min_tracking_confidence,
                running_mode=vision.RunningMode.VIDEO,
            )
            with suppress_native_stderr_patterns(
                (
                    "Failed to send to clearcut",
                    "Created TensorFlow Lite XNNPACK delegate",
                    "inference_feedback_manager.cc",
                    "landmark_projection_calculator.cc",
                    "init-domain.cc",
                    "gl_context.cc",
                )
            ):
                with vision.HandLandmarker.create_from_options(options) as landmarker:
                    start = perf_counter()
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            LOGGER.warning("Failed to read webcam frame")
                            break
                        if mirror:
                            frame = cv2.flip(frame, 1)
                        height_px, width_px = frame.shape[:2]
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        timestamp_ms = int((perf_counter() - start) * 1000)
                        result = landmarker.detect_for_video(mp_image, timestamp_ms)
                        record = _record_from_tasks_result(result, width_px, height_px)
                        prediction = predict_record(record, model, scaler, labels)
                        _draw_record_landmarks(cv2, frame, record)
                        fps, previous = _show_frame(cv2, frame, prediction, fps, previous, fullscreen=fullscreen, window_name=window_name)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def _show_frame(cv2: Any, frame: Any, prediction: Prediction, fps: float, previous: float, fullscreen: bool = False, window_name: str = "Gesture realtime inference") -> tuple[float, float]:
    now = perf_counter()
    elapsed = max(1e-6, now - previous)
    next_fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps else 1.0 / elapsed
    _put_status(cv2, frame, prediction, next_fps)
    cv2.imshow(window_name, frame)
    return next_fps, now
