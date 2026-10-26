"""Reproducibility metadata for long-running geospatial workflows."""

from __future__ import annotations

import hashlib
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .config import PipelineConfig


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def workflow_provenance(
    config: PipelineConfig,
    *,
    config_path: str | Path | None,
    inputs: dict[str, str | Path],
    hash_inputs: bool = False,
) -> dict[str, Any]:
    """Describe the software, configuration, and inputs behind one workflow run."""

    input_records: dict[str, dict[str, Any]] = {}
    for name, value in inputs.items():
        path = Path(value).resolve()
        record: dict[str, Any] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }
        if hash_inputs:
            record["sha256"] = sha256_file(path)
        input_records[name] = record

    return {
        "tool": "forest-change",
        "version": __version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config_path": None if config_path is None else str(Path(config_path).resolve()),
        "config": asdict(config),
        "inputs_hashed": hash_inputs,
        "inputs": input_records,
    }
