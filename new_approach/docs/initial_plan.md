# Random Forest MediaPipe Gesture Training Pipeline

## Summary

Create a Python project, managed by `uv`, for training a Random Forest gesture classifier from `tiny_HaGRID/full_set`. The pipeline splits the image dataset, extracts up to two MediaPipe hands per image into JSONL, preprocesses landmarks without data leakage, trains and evaluates a scikit-learn Random Forest model, and saves metrics, charts, and artifacts in clear step folders with colorful logging.

## Key Changes

- Add/update `pyproject.toml` with Python `>=3.10,<3.14`, a console script named `gesture-pipeline`, and dependencies for MediaPipe, scikit-learn, pandas, numpy, OpenCV/Pillow, seaborn, matplotlib, rich, typer, joblib, and pytest.
- Add a `src/gesture_training/` package with CLI commands:
  - `split`: stratified `train/val/test` split from class folders, default `70/15/15`, seed fixed, plus `fold_id` metadata for later CV/tuning.
  - `extract-landmarks`: run MediaPipe Hands with `max_num_hands=2`; write JSONL records containing image path, label, split, fold, image size, detection count, handedness, scores, and 21 `(x, y, z)` landmarks per detected hand.
  - `preprocess`: build fixed two-hand feature rows, fit preprocessing only on train, transform val/test using saved preprocessing artifacts.
  - `train`: train a Random Forest classifier and save `.joblib`, label map, preprocessing artifacts, config, metrics, and plots.
  - `run-pipeline`: execute all steps in order.
- Use folders:
  - `artifacts/01_splits/`
  - `artifacts/02_landmarks/`
  - `artifacts/03_features/`
  - `artifacts/04_models/`
  - `artifacts/05_reports/`
  - `logs/`

## Data And Modeling Decisions

- Use fixed two-hand input: `left_hand_landmarks`, `right_hand_landmarks`, `left_present`, `right_present`, and detection confidence features.
- Missing hands use zeros plus mask/presence features. No synthetic filler from `no_gesture`, because that risks injecting misleading hand shapes and possible leakage.
- Normalize landmarks by hand geometry before scaling: wrist-relative coordinates and scale by max wrist-to-landmark distance per hand when present.
- Fit any global scaler only on train rows and reuse it for val/test. Although Random Forest does not require scaling, saving the normalized/scaled feature pipeline keeps inference deterministic and protects against leakage.
- Preserve MediaPipe handedness, ordering hands into canonical left/right slots. If two hands of same handedness are detected, keep the highest-confidence detection per handedness when possible and fill the remaining slot only when no matching hand exists.
- First model defaults:
  - `RandomForestClassifier`
  - `n_estimators=300`
  - `max_depth=18`
  - `min_samples_leaf=2`
  - `class_weight="balanced_subsample"`
  - `n_jobs=-1`
  - fixed random seed.
- Save a model metadata JSON that records feature names/order, class labels, preprocessing parameters, and Random Forest hyperparameters. Optionally export a compact tree structure JSON for later ESP-oriented conversion experiments.

## Charts And Reports

- Validation confusion matrix with seaborn.
- Test metrics: accuracy, macro F1, per-class precision/recall/F1.
- Feature importance plot for the top features.
- Dataset diagnostics:
  - class counts per split,
  - one-hand vs two-hand detection counts per label,
  - MediaPipe failure count per label.
- Save both PNG charts and machine-readable JSON/CSV summaries.

## Test Plan

- Unit tests for:
  - deterministic stratified split with no image overlap,
  - scaler/preprocessor fitted only on train rows,
  - one-hand samples producing zero-filled missing hand plus correct mask,
  - two-hand samples producing stable left/right slot ordering,
  - JSONL schema validity.
- Smoke test command on a small sample limit, e.g. `--limit-per-class 5`, to verify the full pipeline without processing all 37,173 images.
- Final verification commands:
  - `uv sync`
  - `uv run gesture-pipeline --help`
  - `uv run gesture-pipeline run-pipeline --limit-per-class 5`
  - optionally full run without limit.

## Assumptions

- Dataset source is `tiny_HaGRID/full_set`, with the 15 class folders currently present.
- `holy` and `take_picture` are expected two-hand classes, but all classes may contain 0, 1, or 2 MediaPipe detections.
- This milestone trains a Random Forest classifier from MediaPipe landmarks, not raw images.
- ESP deployment will require a later conversion/inference strategy for both landmark extraction and Random Forest prediction.
- Existing dirty git state, especially deleted historical files, must not be reverted.
