"""Deterministic offline control plane for Radiation Edge AI.

This module intentionally uses only the Python standard library plus the
project's lightweight path helpers. It performs no neural inference, no KL720
hardware access, and no assay-specific image processing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from radiation_edge_ai.paths import cache_root, data_root, model_root


class ControlPlaneError(RuntimeError):
    """Raised when a control-plane operation cannot be completed safely."""


@dataclass(frozen=True)
class AssaySpec:
    assay_id: str
    title: str
    endpoint: str
    input_kind: str
    required_metadata: tuple[str, ...]
    approved_backends: tuple[str, ...]
    data_path_hints: tuple[str, ...]
    model_path_hints: tuple[str, ...]
    interpretation_guard: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_ASSAYS = {
    "nasa-53bp1-r1-v2": AssaySpec(
        assay_id="nasa-53bp1-r1-v2",
        title="NASA BPS OSD-366 53BP1 R1 v2",
        endpoint="sample-level aggregate latent 53BP1 burden",
        input_kind="single preprocessed nucleus image",
        required_metadata=(
            "sample_id",
            "source_name",
            "radiation_branch",
            "dose_gy",
            "timepoint_hr",
        ),
        approved_backends=("cpu-onnx", "kl720"),
        data_path_hints=("nasa_bps", "nasa-bps", "osd366"),
        model_path_hints=("nasa_bps", "nasa-bps", "osd366"),
        interpretation_guard=(
            "Per-nucleus output is a latent continuous burden score, not an "
            "individually supervised focus count. Biological interpretation "
            "belongs at the aggregate endpoint."
        ),
    ),
    "dnai-fiber-v3": AssaySpec(
        assay_id="dnai-fiber-v3",
        title="DNAi MobileOne-S1 512 v3",
        endpoint="post-processed DNA-fiber objects and tract measurements",
        input_kind="single microscopy field",
        required_metadata=(
            "sample_id",
            "pixel_size_um",
            "channel_semantics",
        ),
        approved_backends=("cpu-onnx", "kl720"),
        data_path_hints=("dnai_public_v2",),
        model_path_hints=("dnai/unet_mobileone_s1",),
        interpretation_guard=(
            "Pixel agreement is not a surrogate for fiber-object fidelity. "
            "Object reconstruction and downstream tract measurements require "
            "their own QC."
        ),
    ),
}


def list_assays() -> tuple[AssaySpec, ...]:
    return tuple(_ASSAYS[key] for key in sorted(_ASSAYS))


def get_assay(assay_id: str) -> AssaySpec:
    try:
        return _ASSAYS[assay_id]
    except KeyError as exc:
        known = ", ".join(sorted(_ASSAYS))
        raise ControlPlaneError(f"Unknown assay {assay_id!r}; choose one of: {known}") from exc


def _path_report(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "readable": os.access(path, os.R_OK) if path.exists() else False,
        "writable": os.access(path, os.W_OK) if path.exists() else False,
    }


def _asset_probe(root: Path, hints: tuple[str, ...]) -> dict[str, object]:
    candidates = [(root / hint).resolve() for hint in hints]
    found = [str(path) for path in candidates if path.exists()]
    return {
        "found": bool(found),
        "found_paths": found,
        "candidate_paths": [str(path) for path in candidates],
    }


def _kl720_python_candidate() -> Path:
    configured = os.environ.get("RADEDGE_KL720_PYTHON")
    if configured:
        return Path(configured).expanduser().resolve()
    home = Path.home()
    if os.name == "nt":
        return (home / "venvs" / "kneron720" / "Scripts" / "python.exe").resolve()
    return (home / "venvs" / "kneron720" / "bin" / "python").resolve()


def collect_doctor_report() -> dict[str, object]:
    """Collect read-only runtime and asset diagnostics.

    This function never creates directories, imports kp, or scans USB devices.
    """

    data = data_root()
    models = model_root()
    cache = cache_root()
    kl720_python = _kl720_python_candidate()

    assays = {}
    for assay in list_assays():
        assays[assay.assay_id] = {
            "data": _asset_probe(data, assay.data_path_hints),
            "models": _asset_probe(models, assay.model_path_hints),
        }

    return {
        "schema_version": 1,
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "storage": {
            "data": _path_report(data),
            "models": _path_report(models),
            "cache": _path_report(cache),
        },
        "optional_runtime": {
            "onnxruntime_importable": importlib.util.find_spec("onnxruntime") is not None,
            "kl720_python": {
                "path": str(kl720_python),
                "exists": kl720_python.is_file(),
                "source": (
                    "RADEDGE_KL720_PYTHON"
                    if os.environ.get("RADEDGE_KL720_PYTHON")
                    else "conventional_user_venv_probe"
                ),
                "hardware_scan_performed": False,
            },
        },
        "assays": assays,
        "mutating_checks_performed": False,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is not allowed: {value}")


def _reject_nonfinite(value: Any, *, label: str) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ControlPlaneError(f"Non-finite value in {label}: {value!r}")
        return
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_nonfinite(child, label=label)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _reject_nonfinite(child, label=label)
        return
    raise ControlPlaneError(f"Unsupported JSON value in {label}: {type(value).__name__}")


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ControlPlaneError(f"Could not read strict JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ControlPlaneError(f"Expected a JSON object: {path}")
    _reject_nonfinite(value, label=str(path))
    return value


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    _reject_nonfinite(value, label="canonical payload")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ControlPlaneError(f"{label} is not a readable file: {resolved}")
    return resolved


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_metadata(spec: AssaySpec, metadata: Mapping[str, Any]) -> None:
    missing = [
        name
        for name in spec.required_metadata
        if name not in metadata or metadata[name] is None or str(metadata[name]).strip() == ""
    ]
    if missing:
        fields = ", ".join(missing)
        raise ControlPlaneError(f"{spec.assay_id} metadata missing required fields: {fields}")


def _identity_core(
    *,
    spec: AssaySpec,
    backend: str,
    input_record: Mapping[str, object],
    model_record: Mapping[str, object],
    metadata_record: Mapping[str, object],
    config_record: Optional[Mapping[str, object]],  # noqa: UP045
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "assay_id": spec.assay_id,
        "backend": backend,
        "artifacts": {
            "input_sha256": input_record["sha256"],
            "input_size_bytes": input_record["size_bytes"],
            "model_sha256": model_record["sha256"],
            "model_size_bytes": model_record["size_bytes"],
            "metadata_sha256": metadata_record["sha256"],
            "config_sha256": config_record["sha256"] if config_record else None,
        },
        "metadata": dict(metadata),
        "config": dict(config),
    }


def create_run_plan(
    *,
    assay_id: str,
    backend: str,
    input_path: Path,
    model_path: Path,
    metadata_path: Path,
    output_dir: Path,
    config_path: Optional[Path] = None,  # noqa: UP045
) -> Path:
    """Create an immutable content-addressed plan without executing inference."""

    spec = get_assay(assay_id)
    if backend not in spec.approved_backends:
        approved = ", ".join(spec.approved_backends)
        raise ControlPlaneError(
            f"Backend {backend!r} is not approved for {assay_id}; choose: {approved}"
        )

    input_file = _require_file(input_path, label="input")
    model_file = _require_file(model_path, label="model")
    metadata_file = _require_file(metadata_path, label="metadata")
    config_file = _require_file(config_path, label="config") if config_path else None

    metadata = load_json_object(metadata_file)
    config = load_json_object(config_file) if config_file else {}
    _validate_metadata(spec, metadata)

    input_record = _artifact_record(input_file)
    model_record = _artifact_record(model_file)
    metadata_record = _artifact_record(metadata_file)
    config_record = _artifact_record(config_file) if config_file else None

    identity = _identity_core(
        spec=spec,
        backend=backend,
        input_record=input_record,
        model_record=model_record,
        metadata_record=metadata_record,
        config_record=config_record,
        metadata=metadata,
        config=config,
    )
    fingerprint = sha256_json(identity)
    run_id = f"r1-{fingerprint[:16]}"

    plan = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "planned",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_fingerprint_sha256": fingerprint,
        "assay": {
            "assay_id": spec.assay_id,
            "title": spec.title,
            "endpoint": spec.endpoint,
            "input_kind": spec.input_kind,
            "interpretation_guard": spec.interpretation_guard,
        },
        "backend": backend,
        "artifacts": {
            "input": input_record,
            "model": model_record,
            "metadata": metadata_record,
            "config": config_record,
        },
        "metadata": metadata,
        "config": config,
        "execution": {
            "performed": False,
            "note": ("v0.1 control-plane plan only; no neural or hardware inference executed"),
        },
    }

    run_root = output_dir.expanduser().resolve() / run_id
    plan_path = run_root / "run_plan.json"
    if plan_path.exists():
        existing = load_json_object(plan_path)
        if existing.get("plan_fingerprint_sha256") == fingerprint:
            return plan_path
        raise ControlPlaneError(f"Conflicting existing run plan: {plan_path}")
    if run_root.exists() and any(run_root.iterdir()):
        raise ControlPlaneError(f"Refusing non-empty run directory without plan: {run_root}")

    run_root.mkdir(parents=True, exist_ok=True)
    temp_path = run_root / "run_plan.json.tmp"
    temp_path.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(plan_path)
    return plan_path


def _identity_core_from_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    assay = plan.get("assay")
    artifacts = plan.get("artifacts")
    if not isinstance(assay, Mapping) or not isinstance(artifacts, Mapping):
        raise ControlPlaneError("Malformed run plan: assay/artifacts must be objects")

    assay_id = assay.get("assay_id")
    backend = plan.get("backend")
    if not isinstance(assay_id, str) or not isinstance(backend, str):
        raise ControlPlaneError("Malformed run plan: assay_id/backend must be strings")
    spec = get_assay(assay_id)

    input_record = artifacts.get("input")
    model_record = artifacts.get("model")
    metadata_record = artifacts.get("metadata")
    config_record = artifacts.get("config")
    for label, record in (
        ("input", input_record),
        ("model", model_record),
        ("metadata", metadata_record),
    ):
        if not isinstance(record, Mapping):
            raise ControlPlaneError(f"Malformed run plan: {label} artifact missing")

    metadata = plan.get("metadata")
    config = plan.get("config")
    if not isinstance(metadata, Mapping) or not isinstance(config, Mapping):
        raise ControlPlaneError("Malformed run plan: metadata/config must be objects")

    return _identity_core(
        spec=spec,
        backend=backend,
        input_record=input_record,
        model_record=model_record,
        metadata_record=metadata_record,
        config_record=config_record if isinstance(config_record, Mapping) else None,
        metadata=metadata,
        config=config,
    )


def _verify_artifact(label: str, record: Mapping[str, Any]) -> dict[str, object]:
    path_value = record.get("path")
    expected_hash = record.get("sha256")
    expected_size = record.get("size_bytes")
    if not isinstance(path_value, str) or not isinstance(expected_hash, str):
        raise ControlPlaneError(f"Malformed {label} artifact record")
    if not isinstance(expected_size, int):
        raise ControlPlaneError(f"Malformed {label} size record")

    path = Path(path_value)
    if not path.is_file():
        return {
            "label": label,
            "ok": False,
            "path": str(path),
            "reason": "missing",
        }

    observed_hash = sha256_file(path)
    observed_size = path.stat().st_size
    ok = observed_hash == expected_hash and observed_size == expected_size
    return {
        "label": label,
        "ok": ok,
        "path": str(path),
        "expected_sha256": expected_hash,
        "observed_sha256": observed_hash,
        "expected_size_bytes": expected_size,
        "observed_size_bytes": observed_size,
    }


def verify_run_plan(
    plan_path: Path,
    *,
    check_artifacts: bool = True,
) -> dict[str, object]:
    """Verify plan identity and, by default, all referenced file contents."""

    path = plan_path.expanduser().resolve()
    plan = load_json_object(path)
    if plan.get("schema_version") != 1:
        raise ControlPlaneError(f"Unsupported run-plan schema: {plan.get('schema_version')!r}")

    expected_fingerprint = plan.get("plan_fingerprint_sha256")
    if not isinstance(expected_fingerprint, str):
        raise ControlPlaneError("Run plan is missing plan_fingerprint_sha256")

    identity = _identity_core_from_plan(plan)
    observed_fingerprint = sha256_json(identity)
    fingerprint_ok = observed_fingerprint == expected_fingerprint

    artifact_results = []
    if check_artifacts:
        artifacts = plan.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ControlPlaneError("Malformed run plan artifacts")
        for label in ("input", "model", "metadata", "config"):
            record = artifacts.get(label)
            if record is None:
                continue
            if not isinstance(record, Mapping):
                raise ControlPlaneError(f"Malformed {label} artifact record")
            artifact_results.append(_verify_artifact(label, record))

    artifacts_ok = all(bool(result["ok"]) for result in artifact_results)
    ok = fingerprint_ok and (artifacts_ok if check_artifacts else True)
    return {
        "schema_version": 1,
        "plan_path": str(path),
        "run_id": plan.get("run_id"),
        "fingerprint_ok": fingerprint_ok,
        "expected_fingerprint_sha256": expected_fingerprint,
        "observed_fingerprint_sha256": observed_fingerprint,
        "artifact_check_performed": check_artifacts,
        "artifacts_ok": artifacts_ok if check_artifacts else None,
        "artifacts": artifact_results,
        "ok": ok,
    }
