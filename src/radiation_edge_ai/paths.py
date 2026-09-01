"""Project storage paths with support for keeping heavy assets off the system drive."""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_from_env(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    return fallback.resolve()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    return _resolve_from_env("RADEDGE_DATA_ROOT", project_root() / "data")


def model_root() -> Path:
    return _resolve_from_env("RADEDGE_MODEL_ROOT", project_root() / "models")


def cache_root() -> Path:
    return _resolve_from_env("RADEDGE_CACHE_ROOT", project_root() / ".cache" / "radedge")


def ensure_storage_dirs() -> dict[str, Path]:
    roots = {
        "data": data_root(),
        "models": model_root(),
        "cache": cache_root(),
    }
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
    return roots
