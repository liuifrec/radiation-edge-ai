from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from radiation_edge_ai.control import ControlPlaneError, create_run_plan, sha256_file
from radiation_edge_ai.execution import execute_run_plan, verify_run_record, verify_target


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _make_plan(tmp_path: Path, *, suffix: str = ".npy") -> Path:
    input_path = tmp_path / f"input{suffix}"
    if suffix == ".npy":
        np.save(input_path, np.zeros((1, 3, 256, 256), dtype=np.float32), allow_pickle=False)
    else:
        input_path.write_bytes(b"not-npy")
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"synthetic-onnx-placeholder")
    metadata_path = _write_json(
        tmp_path / "metadata.json",
        {
            "sample_id": "SMOKE",
            "source_name": "SYNTHETIC",
            "radiation_branch": "X-ray",
            "dose_gy": 0.0,
            "timepoint_hr": 4,
        },
    )
    return create_run_plan(
        assay_id="nasa-53bp1-r1-v2",
        backend="cpu-onnx",
        input_path=input_path,
        model_path=model_path,
        metadata_path=metadata_path,
        output_dir=tmp_path / "runs",
    )


def _fake_worker(command: list[str], **_: object) -> SimpleNamespace:
    def value(flag: str) -> Path:
        return Path(command[command.index(flag) + 1])

    model = value("--model")
    input_path = value("--input")
    output = value("--output")
    manifest = value("--manifest")
    result = np.array([1.25], dtype=np.float32)
    with output.open("wb") as handle:
        np.save(handle, result, allow_pickle=False)
    worker = {
        "schema_version": 1,
        "status": "complete",
        "backend": "cpu-onnx",
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "model": {"path": str(model), "sha256": sha256_file(model)},
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
            "dtype": "float32",
            "shape": [1],
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": "test",
            "numpy_version": np.__version__,
            "onnxruntime_version": "test",
            "requested_provider": "CPUExecutionProvider",
            "active_providers": ["CPUExecutionProvider"],
        },
        "timing": {"session_create_seconds": 0.01, "inference_seconds": 0.02},
    }
    manifest.write_text(json.dumps(worker), encoding="utf-8")
    return SimpleNamespace(returncode=0, stdout="ok", stderr="")


def test_execute_and_verify_run_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan_path = _make_plan(tmp_path)
    monkeypatch.setattr(subprocess, "run", _fake_worker)
    record_path = execute_run_plan(plan_path, onnx_python=Path(sys.executable))
    report = verify_run_record(record_path)
    assert report["ok"] is True
    assert report["raw_output_ok"] is True
    assert report["worker_manifest_ok"] is True
    assert verify_target(record_path)["kind"] == "run_record"


def test_execution_is_idempotent_after_verified_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = _make_plan(tmp_path)
    calls = {"count": 0}

    def counted(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls["count"] += 1
        return _fake_worker(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", counted)
    first = execute_run_plan(plan_path, onnx_python=Path(sys.executable))
    second = execute_run_plan(plan_path, onnx_python=Path(sys.executable))
    assert first == second
    assert calls["count"] == 1


def test_mutated_output_is_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan_path = _make_plan(tmp_path)
    monkeypatch.setattr(subprocess, "run", _fake_worker)
    record_path = execute_run_plan(plan_path, onnx_python=Path(sys.executable))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    output = Path(record["raw_output"]["path"])
    with output.open("wb") as handle:
        np.save(handle, np.array([9.0], dtype=np.float32), allow_pickle=False)
    assert verify_run_record(record_path)["ok"] is False


def test_non_npy_input_is_rejected_before_worker(tmp_path: Path) -> None:
    plan_path = _make_plan(tmp_path, suffix=".bin")
    with pytest.raises(ControlPlaneError, match="preprocessed .npy"):
        execute_run_plan(plan_path, onnx_python=Path(sys.executable))
