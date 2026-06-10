# Gesture Training Pipeline

This project trains a scikit-learn Random Forest gesture classifier from `tiny_HaGRID/full_set` using MediaPipe hand landmarks. The pipeline creates deterministic splits, extracts up to two hands per image, preprocesses fixed left/right hand features without data leakage, trains a Random Forest baseline, and exports joblib/model metadata artifacts.

## Setup

Use a Python version supported by the project range:

```bash
uv sync
```

The project requires Python `>=3.10,<3.14`; `uv` should create or use a compatible interpreter.

MediaPipe `0.10.35+` uses the Tasks API in this environment. Put a hand landmarker model at `models/hand_landmarker.task`, or pass `--hand-landmarker-model /path/to/hand_landmarker.task` to `extract-landmarks` and `run-pipeline`.

## Smoke Run

```bash
uv run gesture-pipeline --help
uv run gesture-pipeline run-pipeline --limit-per-class 5
```

The smoke run processes five images per class and writes artifacts under `artifacts/`.

## Full Run

```bash
uv run gesture-pipeline run-pipeline
```

`run-pipeline` is cached stage by stage. It skips unchanged split, landmark extraction, preprocessing, and training stages, and reruns from the first changed stage onward. Use `--force` to rerun everything.

The DVC-style alias is:

```bash
uv run gesture-pipeline repro
```

Stage configs live in `configs/pipeline/`, and stage source entrypoints live in `src/gesture_training/stages/`.

For smoke/testing runs, write to a separate artifact root so full results are not overwritten:

```bash
uv run gesture-pipeline repro --artifact-root tmp/smoke_artifacts --limit-per-class 1 --force
uv run gesture-pipeline repro --artifact-root tmp/smoke_artifacts --limit-per-class 1
```

To run the base model, tune on a 10% feature subset, then retrain the best Random Forest on the full training split:

```bash
uv run gesture-pipeline run-pipeline --with-tuning
```

`--with-tuning` runs the base model, spends about 15 minutes total across the two tuning methods by default, then retrains the best tuned model on the full train split. Adjust it with:

```bash
uv run gesture-pipeline run-pipeline --with-tuning --tuning-budget-minutes 20
```

Individual steps are also available:

```bash
uv run gesture-pipeline split
uv run gesture-pipeline extract-landmarks
uv run gesture-pipeline extract-landmarks --hand-landmarker-model models/hand_landmarker.task
uv run gesture-pipeline preprocess
uv run gesture-pipeline train
```

## Hyperparameter Tuning

Tuning reuses `artifacts/03_features/`, so it does not rerun MediaPipe. Run the full pipeline or at least `split`, `extract-landmarks`, and `preprocess` first.

Grid search tests every combination from `configs/pipeline/tuning.json` on a stratified subset:

```bash
uv run gesture-pipeline tune-grid --subset-fraction 0.10
```

Successive tuning starts broad with small forests, keeps the strongest configurations, then increases tree count:

```bash
uv run gesture-pipeline tune-successive --subset-fraction 0.10
```

Tune the searched Random Forest parameters in `configs/pipeline/tuning.json`. `grid_search` and `halving_grid_search` each have their own `parameters` section with the same shape. Numeric parameters use evenly spaced ranges:

```json
"min_samples_leaf": {
  "min": 1,
  "max": 10,
  "count": 4,
  "type": "int"
}
```

`max_depth` can also include unlimited trees with `include_unlimited`:

```json
"max_depth": {
  "min": 8,
  "max": 20,
  "count": 4,
  "type": "int",
  "include_unlimited": true
}
```

Parameters that need fixed choices can use explicit values, including strings, floats, and `null`:

```json
"max_features": {
  "values": ["sqrt", "log2", 0.5]
}
```

Both methods support `n_estimators` and `ccp_alpha`. In `grid_search`, `n_estimators` is part of the full parameter product. In `halving_grid_search`, `n_estimators` defines the tree-count stages used while weaker configurations are removed.

Each standalone tuning command has a default 15-minute budget:

```bash
uv run gesture-pipeline tune-grid --subset-fraction 0.10 --max-runtime-minutes 10
uv run gesture-pipeline tune-successive --subset-fraction 0.10 --max-runtime-minutes 10
```

Retrain the best configuration from either tuning method on the full training data:

```bash
uv run gesture-pipeline retrain-best
```

Quick smoke versions:

```bash
uv run gesture-pipeline tune-grid --subset-fraction 0.02 --max-combinations 3
uv run gesture-pipeline tune-successive --subset-fraction 0.02 --max-configs 6
uv run gesture-pipeline retrain-best --n-estimators-override 10
```

## Outputs

- `artifacts/01_splits/`: split CSV and JSON summary
- `artifacts/02_landmarks/`: MediaPipe JSONL and detection diagnostics
- `artifacts/03_features/`: `x_*.npy`, `y_*.npy`, train-fitted scaler, label map, feature metadata
- `artifacts/04_models/`: `random_forest.joblib`, `pipeline_bundle.joblib`, model metadata, metrics, optional `tree_0.json`
- `artifacts/05_reports/`: seaborn charts, class counts, confusion matrix, classification reports
- `artifacts/06_tuning/`: cached tuning results, manifests, and `best_hyperparameters.json`

## Feature Contract

Each sample uses a fixed two-hand vector:

- left hand normalized landmarks, `left_present`, `left_score`
- right hand normalized landmarks, `right_present`, `right_score`

Missing hands are zero-filled and marked with presence `0`. Landmark normalization is wrist-relative with per-hand scale normalization before a `StandardScaler` is fit on train rows only.

Records where MediaPipe detects zero hands are dropped during preprocessing by default because all-zero landmark vectors create contradictory labels. This does not require rerunning landmark extraction; rerun only:

```bash
uv run gesture-pipeline preprocess
uv run gesture-pipeline train
```

or use the cached pipeline:

```bash
uv run gesture-pipeline repro
```

## Next Steps

For webcam or ESP deployment, keep the preprocessing contract identical: extract or provide the same 21-point landmarks per hand, apply the saved scaler from `artifacts/03_features/preprocessor.joblib`, then run the Random Forest. The optional `--export-tree` output is only a starting point for later ESP-oriented conversion experiments; landmark extraction on-device still needs a separate strategy.
