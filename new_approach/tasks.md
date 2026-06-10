# Gesture Training Pipeline Tasks

## Task 1: Project scaffold and dependencies

- [x] Add/update `pyproject.toml` with `uv` project metadata, Python `>=3.10,<3.14`, console script `gesture-pipeline`, runtime dependencies, and test dependencies.
- [x] Use Random Forest/scikit-learn dependencies, not TensorFlow.
- [x] Create `src/gesture_training/` package with CLI, configuration, logging, and shared path helpers.
- [x] Add `.gitignore` entries for generated artifacts, logs, virtual environments, and Python caches.

## Task 2: Dataset split command

- [x] Implement a deterministic stratified `train/val/test` split from `tiny_HaGRID/full_set`.
- [x] Default split ratios to `70/15/15`, seed to `42`, and future fold count to `5`.
- [x] Support `--limit-per-class` for smoke runs.
- [x] Save split metadata under `artifacts/01_splits/` as CSV and JSON summary.

## Task 3: MediaPipe JSONL extraction command

- [x] Run MediaPipe Hands on each split image with `max_num_hands=2`.
- [x] Store JSONL records under `artifacts/02_landmarks/`.
- [x] Include image path, label, split, fold, image size, detection count, handedness, scores, and landmarks.
- [x] Save extraction diagnostics with failures and one-hand/two-hand counts per class.

## Task 4: Feature preprocessing and scaler persistence

- [x] Convert each JSONL record to fixed two-hand features with canonical left/right slots.
- [x] Zero-fill missing hand landmarks and add presence/confidence features.
- [x] Apply wrist-relative, per-hand scale normalization before optional global scaling.
- [x] Fit preprocessing/scaler only on train rows and reuse it for val/test.
- [x] Save feature arrays, labels, feature names, preprocessing artifact, label map, and summary under `artifacts/03_features/`.

## Task 5: Random Forest training and export

- [x] Train a `RandomForestClassifier` suitable as the first landmark-based gesture baseline.
- [x] Use fixed seed, balanced class weighting, and explicit default hyperparameters.
- [x] Save `.joblib`, model metadata JSON, and final metrics under `artifacts/04_models/`.
- [x] Save optional tree export JSON for later ESP conversion experiments.

## Task 6: Charts, metrics, reports, and logging

- [x] Add colorful logging for each pipeline step.
- [x] Generate seaborn validation confusion matrix.
- [x] Generate top feature importance chart.
- [x] Save test metrics, per-class metrics, class-count plots, and detection diagnostics under `artifacts/05_reports/`.

## Task 7: Unit tests and smoke pipeline

- [x] Add tests for split determinism, no split overlap, preprocessing masks, train-only scaler fitting, and JSONL schema.
- [x] Verify `uv run gesture-pipeline --help`.
- [x] Verify `uv run gesture-pipeline run-pipeline --limit-per-class 5`.

## Task 8: README usage instructions

- [x] Add concise README instructions for setup, smoke run, full run, outputs, and next steps toward webcam/ESP deployment.

## Task 9: Hyperparameter tuning

- [x] Add cached grid search on a stratified feature subset.
- [x] Add cached successive-halving tuning on a stratified feature subset.
- [x] Add best-configuration selection and full-data retraining.
- [x] Add tuning comparison charts and README commands.

## Task 10: DVC-like pipeline cache

- [x] Add stage manifests and hashing for split, landmark extraction, preprocessing, and training.
- [x] Make `run-pipeline` skip unchanged stages and rerun from the first changed stage.
- [x] Add `repro` command as a DVC-style cached pipeline alias.
- [x] Drop zero-hand samples during preprocessing by default.
- [x] Add progress output for split, landmark extraction, preprocessing, tuning, and training.
- [x] Suppress MediaPipe Clearcut telemetry noise.
