"""Provenance-aware cache verification for future DNA-fiber reference runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from radiation_edge_ai.audit_utils import canonical_json_sha256, read_json_object, sha256_file


@dataclass(frozen=True)
class CacheIdentity:
    input_sha256: str
    model_sha256: str
    preprocessing_sha256: str
    runtime_provider: str
    runtime_version: str
    expected_shape: tuple[int, ...]
    expected_dtype: str = "float32"

    def digest(self) -> str:
        return canonical_json_sha256(asdict(self))


def write_cache_metadata(
    array_path: Path, metadata_path: Path, identity: CacheIdentity
) -> dict[str, Any]:
    if not array_path.is_file():
        raise FileNotFoundError(array_path)
    array = np.load(array_path, allow_pickle=False)
    if tuple(array.shape) != tuple(identity.expected_shape):
        raise RuntimeError(
            f"Cached array shape mismatch: {array.shape} != {identity.expected_shape}"
        )
    if str(array.dtype) != identity.expected_dtype:
        raise RuntimeError(
            f"Cached array dtype mismatch: {array.dtype} != {identity.expected_dtype}"
        )
    if not np.isfinite(array).all():
        raise RuntimeError("Cached array contains non-finite values")
    if metadata_path.exists():
        raise RuntimeError(f"Refusing to overwrite cache metadata: {metadata_path}")
    payload = {
        "schema": "radedge.dnai.cache-provenance.v1",
        "identity": asdict(identity),
        "identity_sha256": identity.digest(),
        "output_file": array_path.name,
        "output_sha256": sha256_file(array_path),
    }
    metadata_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def verify_cached_array(
    array_path: Path,
    metadata_path: Path,
    expected: CacheIdentity,
) -> np.ndarray:
    if not array_path.is_file():
        raise FileNotFoundError(array_path)
    if not metadata_path.is_file():
        raise RuntimeError(f"Legacy/unverified cache: provenance metadata missing for {array_path}")
    metadata = read_json_object(metadata_path)
    if metadata.get("schema") != "radedge.dnai.cache-provenance.v1":
        raise RuntimeError("Unsupported cache provenance schema")
    if metadata.get("identity_sha256") != expected.digest():
        raise RuntimeError(
            "Cached output identity does not match input/model/preprocessing/runtime identity"
        )
    if metadata.get("identity") != asdict(expected):
        raise RuntimeError("Cached output identity payload differs from expected identity")
    if metadata.get("output_sha256") != sha256_file(array_path):
        raise RuntimeError("Cached output bytes changed after provenance metadata was written")
    array = np.load(array_path, allow_pickle=False)
    if tuple(array.shape) != tuple(expected.expected_shape):
        raise RuntimeError("Cached output shape mismatch")
    if str(array.dtype) != expected.expected_dtype:
        raise RuntimeError("Cached output dtype mismatch")
    if not np.isfinite(array).all():
        raise RuntimeError("Cached output contains non-finite values")
    return np.ascontiguousarray(array)
