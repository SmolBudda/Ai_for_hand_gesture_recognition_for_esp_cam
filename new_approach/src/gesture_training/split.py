from __future__ import annotations

from collections import Counter
from pathlib import Path
import logging
import random

import pandas as pd
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .logging import console
from .io import write_json
from .paths import SPLITS_DIR, ensure_dirs


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOGGER = logging.getLogger(__name__)


def _progress() -> Progress:
    return Progress(
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} classes"),
        TimeElapsedColumn(),
        console=console,
    )


def _split_counts(n_items: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    raw = [n_items * train_ratio, n_items * val_ratio, n_items * (1.0 - train_ratio - val_ratio)]
    counts = [int(value) for value in raw]
    remainder = n_items - sum(counts)
    for index in sorted(range(3), key=lambda item: raw[item] - counts[item], reverse=True)[:remainder]:
        counts[index] += 1
    if n_items >= 3:
        for index, count in enumerate(counts):
            if count == 0:
                donor = max(range(3), key=lambda item: counts[item])
                counts[donor] -= 1
                counts[index] += 1
    train_count, val_count, test_count = counts
    return train_count, val_count, test_count


def create_split(
    dataset_dir: Path,
    output_dir: Path = SPLITS_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    folds: int = 5,
    limit_per_class: int | None = None,
) -> tuple[Path, Path]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    ensure_dirs(output_dir)
    rows: list[dict[str, object]] = []
    rng = random.Random(seed)

    class_dirs = [path for path in sorted(dataset_dir.iterdir()) if path.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found in {dataset_dir}")

    with _progress() as progress:
        task_id = progress.add_task("Creating split", total=len(class_dirs))
        for class_dir in class_dirs:
            images = sorted(
                path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            rng.shuffle(images)
            if limit_per_class is not None:
                images = images[:limit_per_class]
            train_count, val_count, _ = _split_counts(len(images), train_ratio, val_ratio)
            assignments = (
                [("train", path) for path in images[:train_count]]
                + [("val", path) for path in images[train_count : train_count + val_count]]
                + [("test", path) for path in images[train_count + val_count :]]
            )
            for index, (split_name, image_path) in enumerate(assignments):
                rows.append(
                    {
                        "image_path": image_path.as_posix(),
                        "label": class_dir.name,
                        "split": split_name,
                        "fold_id": index % folds,
                    }
                )
            progress.advance(task_id)

    if not rows:
        raise ValueError(f"No images found under {dataset_dir}")

    df = pd.DataFrame(rows).sort_values(["label", "split", "image_path"]).reset_index(drop=True)
    csv_path = output_dir / "splits.csv"
    summary_path = output_dir / "split_summary.json"
    df.to_csv(csv_path, index=False)
    summary = {
        "dataset_dir": dataset_dir.as_posix(),
        "seed": seed,
        "folds": folds,
        "limit_per_class": limit_per_class,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "total_images": int(len(df)),
        "classes": sorted(df["label"].unique().tolist()),
        "counts_by_split": df.groupby("split").size().to_dict(),
        "counts_by_label_split": {
            f"{label}:{split_name}": int(count)
            for (label, split_name), count in df.groupby(["label", "split"]).size().items()
        },
        "fold_counts": {str(key): value for key, value in Counter(df["fold_id"]).items()},
    }
    write_json(summary_path, summary)
    LOGGER.info("Wrote %s split rows to %s", len(df), csv_path)
    return csv_path, summary_path
