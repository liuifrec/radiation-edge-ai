"""Execution and verification for the offline Radiation Edge AI control plane.

The first execution backend is deliberately narrow: a preprocessed ``.npy``
float32 tensor is executed against a frozen ONNX model through an isolated
Python interpreter with ONNX Runtime.  This module itself does not import
ONNX Runtime, so the lightweight project environment remains isolated.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from radiation_edge_ai.control import (
    ControlPlaneError,
    load_json_object,
    sha256_file,
    verify_run_plan,
)


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _resolve_onnx_python(explicit: Optional[Path]) -> Path:  # noqa: UP045
    candidate: Optional[Path] = explicit  # noqa: UP045
    if candidate is None:
        configured = os.environ.get("RADEDGE_ONNX_PYTHON")
        if configured:
            candidate = Path(configured)
    if candidate is None and importlib.util.find_spec("onnxruntime") is not None:
        candidate = Path(sys.executable)
    if candidate is None:
        raise ControlPlaneError(
            "No ONNX Runtime interpreter is available. Set RADEDGE_ONNX_PYTHON "
            "or pass --onnx-python to a Python environment containing onnxruntime."
        )

    resolved = candidate.expanduser().resolve()
    if not resolved.is_file():
        raise ControlPlaneError(f"ONNX Runtime Python interpreter not found: {resolved}")
    return resolved


def _resolve_kl720_python(explicit: Optional[Path]) -> Path:  # noqa: UP045
    candidate: Optional[Path] = explicit  # noqa: UP045
    if candidate is None:
        configured = os.environ.get("RADEDGE_KL720_PYTHON")
        if configured:
            candidate = Path(configured)
    if candidate is None:
        home = Path.home()
        if os.name == "nt":
            candidate = home / "venvs" / "kneron720" / "Scripts" / "python.exe"
        else:
            candidate = home / "venvs" / "kneron720" / "bin" / "python"

    resolved = candidate.expanduser().resolve()
    if not resolved.is_file():
        raise ControlPlaneError(
            f"KL720 Python interpreter not found: {resolved}. "
            "Set RADEDGE_KL720_PYTHON or pass --kl720-python."
        )
    return resolved


def _resolve_kl720_port(explicit: Optional[int]) -> Optional[int]:  # noqa: UP045
    if explicit is not None:
        if explicit < 0:
            raise ControlPlaneError("KL720 USB port must be non-negative")
        return explicit

    configured = os.environ.get("RADEDGE_KL720_PORT")
    if not configured:
        return None
    try:
        value = int(configured)
    except ValueError as exc:
        raise ControlPlaneError(
            f"RADEDGE_KL720_PORT is not an integer: {configured!r}"
        ) from exc
    if value < 0:
        raise ControlPlaneError("RADEDGE_KL720_PORT must be non-negative")
    return value

def _npy_metadata(path: Path) -> dict[str, object]:
    try:
        value = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise ControlPlaneError(f"Could not read NumPy output {path}: {exc}") from exc
    if not isinstance(value, np.ndarray):
        raise ControlPlaneError(f"Expected ndarray in {path}")
    return {
        "dtype": str(value.dtype),
        "shape": [int(x) for x in value.shape],
    }


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ControlPlaneError(f"Malformed {label}: expected object")
    return value


def _bounded(text: str, limit: int = 4000) -> str:
    value = text.strip()
    if len(value) <= limit:
        return value
    return value[-limit:]


def _execute_cpu_onnx_run_plan(
    plan_path: Path,
    *,
    onnx_python: Optional[Path] = None,  # noqa: UP045
) -> Path:
    """Execute one ``cpu-onnx`` run plan and persist an immutable run record.

    The input transport contract for v0.1b is a preprocessed float32 ``.npy``
    tensor.  No image preprocessing or biological endpoint reconstruction is
    performed here.
    """

    plan_file = plan_path.expanduser().resolve()
    plan_verification = verify_run_plan(plan_file, check_artifacts=True)
    if not plan_verification["ok"]:
        raise ControlPlaneError("Run plan verification failed; refusing execution")

    plan = load_json_object(plan_file)
    if plan.get("backend") != "cpu-onnx":
        raise ControlPlaneError(
            f"v0.1b execution supports only backend 'cpu-onnx', got {plan.get('backend')!r}"
        )

    artifacts = _require_mapping(plan.get("artifacts"), label="plan artifacts")
    input_record = _require_mapping(artifacts.get("input"), label="input artifact")
    model_record = _require_mapping(artifacts.get("model"), label="model artifact")

    input_path = Path(str(input_record.get("path", ""))).expanduser().resolve()
    model_path = Path(str(model_record.get("path", ""))).expanduser().resolve()
    if input_path.suffix.lower() != ".npy":
        raise ControlPlaneError(
            "cpu-onnx v0.1b requires a preprocessed .npy tensor input; "
            f"got {input_path.name!r}"
        )
    if model_path.suffix.lower() != ".onnx":
        raise ControlPlaneError(f"cpu-onnx requires an .onnx model; got {model_path.name!r}")

    run_root = plan_file.parent
    record_path = run_root / "run_record.json"
    output_path = run_root / "raw_output.npy"
    manifest_path = run_root / "onnx_worker_manifest.json"

    if record_path.exists():
        existing = verify_run_record(record_path, check_source_artifacts=True)
        if existing["ok"]:
            return record_path
        raise ControlPlaneError(f"Existing run record failed verification: {record_path}")
    orphaned = [path for path in (output_path, manifest_path) if path.exists()]
    if orphaned:
        joined = ", ".join(str(path) for path in orphaned)
        raise ControlPlaneError(f"Refusing to overwrite orphan execution artifacts: {joined}")

    python_path = _resolve_onnx_python(onnx_python)
    worker_path = Path(__file__).with_name("_onnx_cpu_worker.py").resolve()
    if not worker_path.is_file():
        raise ControlPlaneError(f"ONNX worker script missing: {worker_path}")

    command = [
        str(python_path),
        str(worker_path),
        "--model",
        str(model_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--manifest",
        str(manifest_path),
    ]

    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ControlPlaneError(f"ONNX worker could not complete: {exc}") from exc
    wall_seconds = time.perf_counter() - start

    if completed.returncode != 0:
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        detail = _bounded(completed.stderr or completed.stdout)
        raise ControlPlaneError(
            f"ONNX worker failed with exit code {completed.returncode}: {detail}"
        )
    if not output_path.is_file() or not manifest_path.is_file():
        raise ControlPlaneError("ONNX worker exited successfully without required output artifacts")

    worker = load_json_object(manifest_path)
    if worker.get("status") != "complete" or worker.get("backend") != "cpu-onnx":
        raise ControlPlaneError("ONNX worker manifest is not a complete cpu-onnx execution")

    worker_input = _require_mapping(worker.get("input"), label="worker input")
    worker_model = _require_mapping(worker.get("model"), label="worker model")
    worker_output = _require_mapping(worker.get("output"), label="worker output")
    if worker_input.get("sha256") != input_record.get("sha256"):
        raise ControlPlaneError("Worker input hash does not match the frozen run plan")
    if worker_model.get("sha256") != model_record.get("sha256"):
        raise ControlPlaneError("Worker model hash does not match the frozen run plan")

    output_record = _artifact_record(output_path)
    output_record.update(_npy_metadata(output_path))
    if worker_output.get("sha256") != output_record["sha256"]:
        raise ControlPlaneError("Worker output hash does not match persisted raw_output.npy")
    if worker_output.get("dtype") != output_record["dtype"]:
        raise ControlPlaneError("Worker output dtype does not match persisted raw_output.npy")
    if worker_output.get("shape") != output_record["shape"]:
        raise ControlPlaneError("Worker output shape does not match persisted raw_output.npy")

    manifest_record = _artifact_record(manifest_path)
    runtime = _require_mapping(worker.get("runtime"), label="worker runtime")
    timing = _require_mapping(worker.get("timing"), label="worker timing")

    ended_utc = datetime.now(timezone.utc).isoformat()

    record = {
        "schema_version": 2,
        "record_type": "run_record",
        "run_id": plan.get("run_id"),
        "assay_id": _require_mapping(
            plan.get("assay"), label="plan assay"
        ).get("assay_id"),
        "status": "complete",
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "completed_utc": ended_utc,
        "backend": "cpu-onnx",
        "plan": {
            "path": str(plan_file),
            "sha256": sha256_file(plan_file),
            "plan_fingerprint_sha256": plan.get("plan_fingerprint_sha256"),
        },
        "source_artifacts": {
            "input": dict(input_record),
            "model": dict(model_record),
        },
        "runtime": dict(runtime),
        "timing": {
            **dict(timing),
            "worker_subprocess_wall_seconds": wall_seconds,
        },
        "raw_output": output_record,
        "worker_manifest": manifest_record,
        "scientific_interpretation": {
            "endpoint_reconstruction_performed": False,
            "biological_acceptance_evaluated": False,
            "note": (
                "Raw model execution only. The output is not, by itself, a validated "
                "biological endpoint or biological-fidelity result."
            ),
        },
    }

    temp_path = run_root / "run_record.json.tmp"
    temp_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(record_path)
    return record_path


def _execute_kl720_run_plan(
    plan_path: Path,
    *,
    kl720_python: Optional[Path] = None,  # noqa: UP045
    kl720_port: Optional[int] = None,  # noqa: UP045
    kl720_timeout_ms: int = 10000,
) -> Path:
    """Execute one preprocessed tensor with a frozen NEF on physical KL720."""

    plan_file = plan_path.expanduser().resolve()
    plan_verification = verify_run_plan(plan_file, check_artifacts=True)
    if not plan_verification["ok"]:
        raise ControlPlaneError("Run plan verification failed; refusing execution")

    plan = load_json_object(plan_file)
    if plan.get("backend") != "kl720":
        raise ControlPlaneError(
            f"KL720 executor received backend {plan.get('backend')!r}"
        )
    if kl720_timeout_ms <= 0:
        raise ControlPlaneError("KL720 timeout must be positive")

    artifacts = _require_mapping(plan.get("artifacts"), label="plan artifacts")
    input_record = _require_mapping(artifacts.get("input"), label="input artifact")
    model_record = _require_mapping(artifacts.get("model"), label="model artifact")

    input_path = Path(str(input_record.get("path", ""))).expanduser().resolve()
    model_path = Path(str(model_record.get("path", ""))).expanduser().resolve()
    if input_path.suffix.lower() != ".npy":
        raise ControlPlaneError(
            "kl720 v0.1c requires a preprocessed .npy tensor input; "
            f"got {input_path.name!r}"
        )
    if model_path.suffix.lower() != ".nef":
        raise ControlPlaneError(
            f"kl720 v0.1c requires a .nef model; got {model_path.name!r}"
        )

    run_root = plan_file.parent
    record_path = run_root / "run_record.json"
    output_path = run_root / "raw_output.npy"
    manifest_path = run_root / "kl720_worker_manifest.json"

    if record_path.exists():
        existing = verify_run_record(record_path, check_source_artifacts=True)
        if existing["ok"]:
            return record_path
        raise ControlPlaneError(
            f"Existing run record failed verification: {record_path}"
        )

    orphaned = [path for path in (output_path, manifest_path) if path.exists()]
    if orphaned:
        joined = ", ".join(str(path) for path in orphaned)
        raise ControlPlaneError(
            f"Refusing to overwrite orphan execution artifacts: {joined}"
        )

    python_path = _resolve_kl720_python(kl720_python)
    port = _resolve_kl720_port(kl720_port)
    worker_path = Path(__file__).with_name("_kl720_worker.py").resolve()
    if not worker_path.is_file():
        raise ControlPlaneError(f"KL720 worker script missing: {worker_path}")

    command = [
        str(python_path),
        str(worker_path),
        "--model",
        str(model_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--manifest",
        str(manifest_path),
        "--timeout-ms",
        str(int(kl720_timeout_ms)),
    ]
    if port is not None:
        command.extend(["--port", str(port)])

    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ControlPlaneError(f"KL720 worker could not complete: {exc}") from exc
    wall_seconds = time.perf_counter() - start

    if completed.returncode != 0:
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        detail = _bounded(completed.stderr or completed.stdout)
        raise ControlPlaneError(
            f"KL720 worker failed with exit code {completed.returncode}: {detail}"
        )
    if not output_path.is_file() or not manifest_path.is_file():
        raise ControlPlaneError(
            "KL720 worker exited successfully without required output artifacts"
        )

    worker = load_json_object(manifest_path)
    if worker.get("status") != "complete" or worker.get("backend") != "kl720":
        raise ControlPlaneError(
            "KL720 worker manifest is not a complete kl720 execution"
        )

    worker_input = _require_mapping(worker.get("input"), label="worker input")
    worker_model = _require_mapping(worker.get("model"), label="worker model")
    worker_output = _require_mapping(worker.get("output"), label="worker output")
    if worker_input.get("sha256") != input_record.get("sha256"):
        raise ControlPlaneError(
            "Worker input hash does not match the frozen run plan"
        )
    if worker_model.get("sha256") != model_record.get("sha256"):
        raise ControlPlaneError(
            "Worker model hash does not match the frozen run plan"
        )

    runtime = _require_mapping(worker.get("runtime"), label="worker runtime")
    if runtime.get("product_id") != 0x720:
        raise ControlPlaneError("KL720 worker did not record product_id 0x720")
    selected_port = runtime.get("selected_usb_port")
    if not isinstance(selected_port, int) or selected_port < 0:
        raise ControlPlaneError("KL720 worker recorded an invalid USB port")

    output_record = _artifact_record(output_path)
    output_record.update(_npy_metadata(output_path))
    if worker_output.get("sha256") != output_record["sha256"]:
        raise ControlPlaneError(
            "Worker output hash does not match persisted raw_output.npy"
        )
    if worker_output.get("dtype") != output_record["dtype"]:
        raise ControlPlaneError(
            "Worker output dtype does not match persisted raw_output.npy"
        )
    if worker_output.get("shape") != output_record["shape"]:
        raise ControlPlaneError(
            "Worker output shape does not match persisted raw_output.npy"
        )

    manifest_record = _artifact_record(manifest_path)
    timing = _require_mapping(worker.get("timing"), label="worker timing")

    ended_utc = datetime.now(timezone.utc).isoformat()

    record = {
        "schema_version": 2,
        "record_type": "run_record",
        "run_id": plan.get("run_id"),
        "assay_id": _require_mapping(
            plan.get("assay"), label="plan assay"
        ).get("assay_id"),
        "status": "complete",
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "completed_utc": ended_utc,
        "backend": "kl720",
        "plan": {
            "path": str(plan_file),
            "sha256": sha256_file(plan_file),
            "plan_fingerprint_sha256": plan.get("plan_fingerprint_sha256"),
        },
        "source_artifacts": {
            "input": dict(input_record),
            "model": dict(model_record),
        },
        "runtime": dict(runtime),
        "timing": {
            **dict(timing),
            "worker_subprocess_wall_seconds": wall_seconds,
        },
        "raw_output": output_record,
        "worker_manifest": manifest_record,
        "scientific_interpretation": {
            "endpoint_reconstruction_performed": False,
            "biological_acceptance_evaluated": False,
            "note": (
                "Raw physical KL720 model execution only. The output is not, "
                "by itself, a validated biological endpoint or a new "
                "biological-fidelity result."
            ),
        },
    }

    temp_path = run_root / "run_record.json.tmp"
    temp_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(record_path)
    return record_path


def execute_run_plan(
    plan_path: Path,
    *,
    onnx_python: Optional[Path] = None,  # noqa: UP045
    kl720_python: Optional[Path] = None,  # noqa: UP045
    kl720_port: Optional[int] = None,  # noqa: UP045
    kl720_timeout_ms: int = 10000,
) -> Path:
    """Dispatch a frozen run plan to its isolated approved backend."""

    plan_file = plan_path.expanduser().resolve()
    plan = load_json_object(plan_file)
    backend = plan.get("backend")
    if backend == "cpu-onnx":
        return _execute_cpu_onnx_run_plan(
            plan_file,
            onnx_python=onnx_python,
        )
    if backend == "kl720":
        return _execute_kl720_run_plan(
            plan_file,
            kl720_python=kl720_python,
            kl720_port=kl720_port,
            kl720_timeout_ms=kl720_timeout_ms,
        )
    raise ControlPlaneError(f"No execution adapter for backend {backend!r}")

def _verify_file_record(record: Mapping[str, Any], *, label: str) -> dict[str, object]:
    path_value = record.get("path")
    expected_hash = record.get("sha256")
    expected_size = record.get("size_bytes")
    if not isinstance(path_value, str) or not isinstance(expected_hash, str):
        raise ControlPlaneError(f"Malformed {label} artifact record")
    if not isinstance(expected_size, int):
        raise ControlPlaneError(f"Malformed {label} artifact size")

    path = Path(path_value)
    if not path.is_file():
        return {"ok": False, "path": str(path), "reason": "missing"}
    observed_hash = sha256_file(path)
    observed_size = path.stat().st_size
    return {
        "ok": observed_hash == expected_hash and observed_size == expected_size,
        "path": str(path),
        "expected_sha256": expected_hash,
        "observed_sha256": observed_hash,
        "expected_size_bytes": expected_size,
        "observed_size_bytes": observed_size,
    }


def verify_run_record(
    record_path: Path,
    *,
    check_source_artifacts: bool = True,
) -> dict[str, object]:
    """Verify a completed run record, its plan, worker manifest, and raw output."""

    path = record_path.expanduser().resolve()
    record = load_json_object(path)

    record_schema_version = record.get("schema_version")
    if record_schema_version not in (1, 2):
        raise ControlPlaneError(
            "Unsupported run-record schema_version: "
            f"{record_schema_version!r}"
        )
    legacy_schema_v1 = record_schema_version == 1

    if record.get("record_type") != "run_record":
        raise ControlPlaneError(f"Not a supported run record: {path}")
    if record.get("status") != "complete":
        raise ControlPlaneError(f"Run record is not complete: {path}")

    plan_record = _require_mapping(record.get("plan"), label="run-record plan")
    plan_path_value = plan_record.get("path")
    expected_plan_hash = plan_record.get("sha256")
    expected_plan_fingerprint = plan_record.get("plan_fingerprint_sha256")
    if (
        not isinstance(plan_path_value, str)
        or not isinstance(expected_plan_hash, str)
        or not isinstance(expected_plan_fingerprint, str)
    ):
        raise ControlPlaneError("Malformed run-record plan identity")
    plan_path = Path(plan_path_value)
    plan_hash_ok = plan_path.is_file() and sha256_file(plan_path) == expected_plan_hash

    plan_backend_ok = False
    plan_fingerprint_ok = False
    assay_id_ok = False
    if plan_path.is_file():
        persisted_plan = load_json_object(plan_path)
        plan_backend_ok = persisted_plan.get("backend") == record.get("backend")
        plan_fingerprint_ok = (
            persisted_plan.get("plan_fingerprint_sha256")
            == expected_plan_fingerprint
        )
        persisted_assay = _require_mapping(
            persisted_plan.get("assay"), label="persisted plan assay"
        )
        record_assay_id = record.get("assay_id")
        if legacy_schema_v1 and record_assay_id is None:
            assay_id_ok = True
        else:
            assay_id_ok = (
                isinstance(record_assay_id, str)
                and record_assay_id == persisted_assay.get("assay_id")
            )

    plan_report: dict[str, object]
    if plan_path.is_file():
        plan_report = verify_run_plan(plan_path, check_artifacts=check_source_artifacts)
    else:
        plan_report = {"ok": False, "run_id": None}
    run_id_ok = plan_report.get("run_id") == record.get("run_id")

    timestamps_ok = False
    started_value = record.get("started_utc")
    ended_value = record.get("ended_utc")

    if (
        legacy_schema_v1
        and started_value is None
        and ended_value is None
    ):
        completed_value = record.get("completed_utc")
        if isinstance(completed_value, str):
            try:
                completed_dt = datetime.fromisoformat(completed_value)
            except ValueError:
                pass
            else:
                timestamps_ok = completed_dt.tzinfo is not None

    elif isinstance(started_value, str) and isinstance(ended_value, str):
        try:
            started_dt = datetime.fromisoformat(started_value)
            ended_dt = datetime.fromisoformat(ended_value)
        except ValueError:
            pass
        else:
            timestamps_ok = bool(
                started_dt.tzinfo is not None
                and ended_dt.tzinfo is not None
                and ended_dt >= started_dt
                and record.get("completed_utc") == ended_value
            )

    output_record = _require_mapping(record.get("raw_output"), label="raw output")
    output_report = _verify_file_record(output_record, label="raw output")
    output_metadata_ok = False
    output_path = Path(str(output_record.get("path", "")))
    if output_report["ok"]:
        observed = _npy_metadata(output_path)
        output_metadata_ok = (
            observed.get("dtype") == output_record.get("dtype")
            and observed.get("shape") == output_record.get("shape")
        )

    manifest_record = _require_mapping(record.get("worker_manifest"), label="worker manifest")
    manifest_report = _verify_file_record(manifest_record, label="worker manifest")
    manifest_semantics_ok = False
    if manifest_report["ok"]:
        manifest = load_json_object(Path(str(manifest_record["path"])))
        manifest_output = _require_mapping(manifest.get("output"), label="manifest output")
        source_artifacts = _require_mapping(
            record.get("source_artifacts"), label="run-record source artifacts"
        )
        source_input = _require_mapping(source_artifacts.get("input"), label="source input")
        source_model = _require_mapping(source_artifacts.get("model"), label="source model")
        manifest_input = _require_mapping(manifest.get("input"), label="manifest input")
        manifest_model = _require_mapping(manifest.get("model"), label="manifest model")
        manifest_semantics_ok = (
            manifest.get("status") == "complete"
            and manifest.get("backend") == record.get("backend")
            and manifest_input.get("sha256") == source_input.get("sha256")
            and manifest_model.get("sha256") == source_model.get("sha256")
            and manifest_output.get("sha256") == output_record.get("sha256")
            and manifest_output.get("dtype") == output_record.get("dtype")
            and manifest_output.get("shape") == output_record.get("shape")
        )

    ok = bool(
        plan_hash_ok
        and plan_backend_ok
        and plan_fingerprint_ok
        and assay_id_ok
        and timestamps_ok
        and plan_report.get("ok")
        and run_id_ok
        and output_report["ok"]
        and output_metadata_ok
        and manifest_report["ok"]
        and manifest_semantics_ok
    )
    return {
        "kind": "run_record",
        "record_path": str(path),
        "run_id": record.get("run_id"),
        "plan_hash_ok": plan_hash_ok,
        "plan_backend_ok": plan_backend_ok,
        "plan_fingerprint_ok": plan_fingerprint_ok,
        "assay_id_ok": assay_id_ok,
        "timestamps_ok": timestamps_ok,
        "plan_ok": bool(plan_report.get("ok")),
        "run_id_ok": run_id_ok,
        "source_artifact_check_performed": check_source_artifacts,
        "raw_output_ok": bool(output_report["ok"] and output_metadata_ok),
        "worker_manifest_ok": bool(manifest_report["ok"] and manifest_semantics_ok),
        "schema_version": record_schema_version,
        "ok": ok,
    }


def verify_target(path: Path, *, check_artifacts: bool = True) -> dict[str, object]:
    """Verify a supported Radiation Edge AI control-plane artifact."""

    target = path.expanduser().resolve()
    value = load_json_object(target)
    if value.get("record_type") == "run_record":
        return verify_run_record(target, check_source_artifacts=check_artifacts)

    if value.get("record_type") == "batch_plan":
        from radiation_edge_ai.batch import verify_batch_plan

        return verify_batch_plan(
            target,
            check_artifacts=check_artifacts,
        )

    if value.get("record_type") == "batch_record":
        from radiation_edge_ai.batch import verify_batch_record

        return verify_batch_record(
            target,
            check_artifacts=check_artifacts,
        )

    if value.get("record_type") == "measurement_plan":
        from radiation_edge_ai.measurement import verify_measurement_plan

        return verify_measurement_plan(
            target,
            check_source_artifact=check_artifacts,
        )

    if value.get("record_type") == "measurement_record":
        from radiation_edge_ai.measurement import verify_measurement_record

        return verify_measurement_record(
            target,
            check_artifacts=check_artifacts,
        )

    if value.get("report_type") == "nasa_batch_prediction_report":
        from radiation_edge_ai.nasa_predictions import (
            verify_nasa_batch_prediction_report,
        )

        return verify_nasa_batch_prediction_report(
            target,
            check_source_artifact=check_artifacts,
        )

    if value.get("report_type") == "nasa_endpoint_report":
        from radiation_edge_ai.nasa_endpoint import verify_nasa_endpoint_report

        return verify_nasa_endpoint_report(
            target,
            check_source_artifact=check_artifacts,
        )

    report = dict(verify_run_plan(target, check_artifacts=check_artifacts))
    report["kind"] = "run_plan"
    return report
