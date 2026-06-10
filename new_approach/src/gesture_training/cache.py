from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable
import hashlib
import json
import logging
import os
import sys
import threading

from .io import read_json, write_json
from .paths import ensure_dirs
from .split import IMAGE_EXTENSIONS


LOGGER = logging.getLogger(__name__)


def _stable_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")


def hash_payload(payload: Any) -> str:
    return hashlib.sha256(_stable_json(payload)).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_existing_files(paths: list[Path]) -> dict[str, str | None]:
    return {path.as_posix(): hash_file(path) if path.exists() else None for path in paths}


def hash_code_files(paths: list[Path]) -> dict[str, str | None]:
    return hash_existing_files(paths)


def dataset_fingerprint(dataset_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            stat = path.stat()
            files.append(
                {
                    "path": path.relative_to(dataset_dir).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {"dataset_dir": dataset_dir.as_posix(), "file_count": len(files), "files": files}


def stage_signature(
    name: str,
    config: dict[str, Any],
    inputs: dict[str, Any],
    code_files: list[Path],
) -> str:
    return hash_payload(
        {
            "stage": name,
            "config": config,
            "inputs": inputs,
            "code": hash_code_files(code_files),
        }
    )


def manifest_path(stage_dir: Path) -> Path:
    return stage_dir / "stage_manifest.json"


def stage_is_current(stage_dir: Path, outputs: list[Path], signature: str) -> bool:
    path = manifest_path(stage_dir)
    if not path.exists() or any(not output.exists() for output in outputs):
        return False
    try:
        manifest = read_json(path)
    except json.JSONDecodeError:
        return False
    return manifest.get("signature") == signature


def write_stage_manifest(stage_dir: Path, signature: str, config: dict[str, Any], outputs: list[Path]) -> None:
    ensure_dirs(stage_dir)
    write_json(
        manifest_path(stage_dir),
        {
            "signature": signature,
            "config": config,
            "outputs": [output.as_posix() for output in outputs],
        },
    )


def run_cached_stage(
    name: str,
    stage_dir: Path,
    outputs: list[Path],
    signature: str,
    config: dict[str, Any],
    runner: Callable[[], tuple[Path, ...] | None],
    force: bool = False,
    downstream_dirty: bool = False,
) -> bool:
    if not force and not downstream_dirty and stage_is_current(stage_dir, outputs, signature):
        LOGGER.info("Cache hit for %s; using %s", name, stage_dir)
        return False
    LOGGER.info("Running %s", name)
    produced = runner()
    write_stage_manifest(stage_dir, signature, config, list(produced or outputs))
    return True


@contextmanager
def suppress_native_stderr_patterns(patterns: tuple[str, ...]):
    """Filter noisy native stderr lines while preserving other stderr output."""
    read_fd, write_fd = os.pipe()
    original_fd = os.dup(2)
    stop = threading.Event()

    def forward() -> None:
        with os.fdopen(read_fd, "rb", closefd=True) as reader:
            for raw in iter(reader.readline, b""):
                text = raw.decode("utf-8", errors="replace")
                if any(pattern in text for pattern in patterns):
                    continue
                if "Source Location Trace" in text or "portable_clearcut_uploader.cc" in text:
                    continue
                os.write(original_fd, raw)
                if stop.is_set():
                    break

    thread = threading.Thread(target=forward, daemon=True)
    thread.start()
    try:
        os.dup2(write_fd, 2)
        yield
    finally:
        os.dup2(original_fd, 2)
        os.close(write_fd)
        stop.set()
        thread.join(timeout=1)
        os.close(original_fd)
        sys.stderr.flush()
