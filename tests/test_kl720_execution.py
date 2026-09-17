from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from radiation_edge_ai.control import ControlPlaneError, create_run_plan, sha256_file
from radiation_edge_ai.execution import execute_run_plan, verify_run_record


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _make_plan(tmp_path: Path, *, model_suffix: str = ".nef") -> Path:
    input_path = tmp_path / "input.npy"
    np.save(
        input_path,
        np.zeros((1, 3, 256, 256), dtype=np.float32),
        allow_pickle=False,
    )
    model_path = tmp_path / f"model{model_suffix}"
    model_path.write_bytes(b"synthetic-nef-placeholder")
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
        backend="kl720",
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

    with output.open("wb") as handle:
        np.save(handle, np.array([1.5], dtype=np.float32), allow_pickle=False)

    worker = {
        "schema_version": 1,
        "status": "complete",
        "backend": "kl720",
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
            "kp_module": "test-kp",
            "selected_usb_port": 81,
            "product_id": 0x720,
            "model_id": 32770,
        },
        "timing": {
            "model_load_seconds": 0.01,
            "host_pack_seconds": 0.01,
            "inference_send_receive_seconds": 0.02,
        },
        "kl720_input_pack": {
            "radix": 7,
            "scale": 1.0,
            "layout": "4W4C8B",
        },
    }
    manifest.write_text(json.dumps(worker), encoding="utf-8")
    return SimpleNamespace(returncode=0, stdout="ok", stderr="")


def test_execute_and_verify_kl720_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = _make_plan(tmp_path)
    seen: dict[str, object] = {}

    def fake(command: list[str], **kwargs: object) -> SimpleNamespace:
        seen["command"] = command
        return _fake_worker(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake)
    record_path = execute_run_plan(
        plan_path,
        kl720_python=Path(sys.executable),
        kl720_port=81,
    )

    report = verify_run_record(record_path)
    assert report["ok"] is True

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["backend"] == "kl720"
    assert record["runtime"]["selected_usb_port"] == 81

    command = seen["command"]
    assert isinstance(command, list)
    assert "_kl720_worker.py" in str(command[1])
    assert command[command.index("--port") + 1] == "81"


def test_kl720_execution_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = _make_plan(tmp_path)
    calls = {"count": 0}

    def fake(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls["count"] += 1
        return _fake_worker(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake)
    first = execute_run_plan(
        plan_path,
        kl720_python=Path(sys.executable),
        kl720_port=81,
    )
    second = execute_run_plan(
        plan_path,
        kl720_python=Path(sys.executable),
        kl720_port=81,
    )
    assert first == second
    assert calls["count"] == 1


def test_kl720_rejects_non_nef_model(tmp_path: Path) -> None:
    plan_path = _make_plan(tmp_path, model_suffix=".onnx")
    with pytest.raises(ControlPlaneError, match=r"\.nef"):
        execute_run_plan(
            plan_path,
            kl720_python=Path(sys.executable),
            kl720_port=81,
        )
