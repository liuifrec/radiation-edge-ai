from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from radiation_edge_ai.application import (
    execute_assay_manifest,
)
from radiation_edge_ai.batch import (
    create_batch_plan,
    execute_batch_plan,
    verify_batch_plan,
    verify_batch_record,
)
from radiation_edge_ai.control import (
    ControlPlaneError,
    sha256_file,
)
from radiation_edge_ai.nasa_predictions import (
    BURDEN_COLUMN,
    create_nasa_batch_prediction_report,
    verify_nasa_batch_prediction_report,
)
from radiation_edge_ai.transaction import (
    execute_batch_measurement_transaction,
    verify_batch_measurement_transaction,
)


def _write_manifest(
    root: Path,
    *,
    backend: str = "cpu-onnx",
    duplicate: bool = False,
    include_sample_name: bool = True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)

    model = root / "model.onnx"
    model.write_bytes(b"synthetic-onnx-model")

    inputs = []
    for index in range(2):
        input_path = root / f"nucleus_{index}.npy"
        np.save(
            input_path,
            np.full(
                (1, 3, 256, 256),
                index,
                dtype=np.float32,
            ),
            allow_pickle=False,
        )
        inputs.append(input_path)

    items = []
    for index, input_path in enumerate(inputs):
        sample_id = "N1" if duplicate else f"N{index + 1}"

        metadata = {
            "sample_id": sample_id,
            "source_name": "SRC",
            "radiation_branch": "X-ray",
            "dose_gy": 0 if index == 0 else 0.3,
            "timepoint_hr": 4,
        }

        if include_sample_name:
            metadata["sample_name"] = "SAMPLE_A"

        items.append(
            {
                "input": str(input_path),
                "metadata": metadata,
            }
        )

    manifest = {
        "schema_version": 1,
        "assay_id": "nasa-53bp1-r1-v2",
        "backend": backend,
        "model": str(model),
        "items": items,
    }

    path = root / "batch_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return path


def test_create_verify_and_reuse_batch_plan(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path / "source"
    )

    plan_path = create_batch_plan(
        manifest_path=manifest,
        output_dir=tmp_path / "plans",
    )

    report = verify_batch_plan(plan_path)

    assert report["ok"] is True
    assert report["fingerprint_ok"] is True
    assert report["source_manifest_ok"] is True
    assert report["model_ok"] is True
    assert report["items_ok"] is True
    assert report["manifest_semantics_ok"] is True
    assert report["n_items"] == 2

    second = create_batch_plan(
        manifest_path=manifest,
        output_dir=tmp_path / "plans",
    )

    assert second == plan_path


def test_batch_identity_is_path_independent(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first_manifest = _write_manifest(first_root)

    second_root.mkdir(parents=True)
    shutil.copyfile(
        first_root / "model.onnx",
        second_root / "model.onnx",
    )
    shutil.copyfile(
        first_root / "nucleus_0.npy",
        second_root / "nucleus_0.npy",
    )
    shutil.copyfile(
        first_root / "nucleus_1.npy",
        second_root / "nucleus_1.npy",
    )

    first_value = json.loads(
        first_manifest.read_text(encoding="utf-8")
    )

    second_value = dict(first_value)
    second_value["model"] = str(
        second_root / "model.onnx"
    )
    second_value["items"] = [
        {
            **item,
            "input": str(
                second_root / f"nucleus_{index}.npy"
            ),
        }
        for index, item in enumerate(
            first_value["items"]
        )
    ]

    second_manifest = (
        second_root / "batch_manifest.json"
    )
    second_manifest.write_text(
        json.dumps(second_value, indent=2),
        encoding="utf-8",
    )

    first_plan = create_batch_plan(
        manifest_path=first_manifest,
        output_dir=tmp_path / "plans-a",
    )
    second_plan = create_batch_plan(
        manifest_path=second_manifest,
        output_dir=tmp_path / "plans-b",
    )

    assert first_plan.parent.name == second_plan.parent.name


def test_duplicate_sample_id_is_rejected(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path / "source",
        duplicate=True,
    )

    with pytest.raises(
        ControlPlaneError,
        match="Duplicate batch sample_id",
    ):
        create_batch_plan(
            manifest_path=manifest,
            output_dir=tmp_path / "plans",
        )


def test_kl720_batch_is_not_enabled_in_v0_4(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path / "source",
        backend="kl720",
    )

    with pytest.raises(
        ControlPlaneError,
        match="intentionally limited",
    ):
        create_batch_plan(
            manifest_path=manifest,
            output_dir=tmp_path / "plans",
        )


def test_missing_sample_name_is_rejected(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path / "source",
        include_sample_name=False,
    )

    with pytest.raises(
        ControlPlaneError,
        match="sample_name",
    ):
        create_batch_plan(
            manifest_path=manifest,
            output_dir=tmp_path / "plans",
        )


def test_input_mutation_is_detected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    manifest = _write_manifest(source)

    plan_path = create_batch_plan(
        manifest_path=manifest,
        output_dir=tmp_path / "plans",
    )

    np.save(
        source / "nucleus_0.npy",
        np.ones(
            (1, 3, 256, 256),
            dtype=np.float32,
        ),
        allow_pickle=False,
    )

    report = verify_batch_plan(plan_path)

    assert report["items_ok"] is False
    assert report["manifest_semantics_ok"] is False
    assert report["ok"] is False



def test_relative_paths_resolve_from_manifest_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    manifest = _write_manifest(source)

    value = json.loads(
        manifest.read_text(encoding="utf-8")
    )
    value["model"] = "model.onnx"

    for index, item in enumerate(value["items"]):
        item["input"] = f"nucleus_{index}.npy"

    manifest.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )

    plan_path = create_batch_plan(
        manifest_path=manifest,
        output_dir=tmp_path / "plans",
    )

    report = verify_batch_plan(plan_path)

    assert report["ok"] is True
    assert report["model_ok"] is True
    assert report["items_ok"] is True
    assert report["manifest_semantics_ok"] is True



def test_verify_target_recognizes_batch_plan(
    tmp_path: Path,
) -> None:
    from radiation_edge_ai.execution import verify_target

    manifest = _write_manifest(
        tmp_path / "source"
    )

    plan_path = create_batch_plan(
        manifest_path=manifest,
        output_dir=tmp_path / "plans",
    )

    report = verify_target(plan_path)

    assert report["kind"] == "batch_plan"
    assert report["fingerprint_ok"] is True
    assert report["source_manifest_ok"] is True
    assert report["model_ok"] is True
    assert report["items_ok"] is True
    assert report["ok"] is True


def test_cli_batch_plan_and_verify(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from radiation_edge_ai.cli import main

    manifest = _write_manifest(
        tmp_path / "source"
    )
    output_dir = tmp_path / "plans"

    status = main(
        [
            "batch-plan",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert status == 0

    plan_path = Path(
        capsys.readouterr().out.strip()
    )
    assert plan_path.is_file()

    status = main(
        ["verify", str(plan_path)]
    )
    assert status == 0

    output = capsys.readouterr().out

    assert "kind: batch_plan" in output
    assert "batch_id: b1-" in output
    assert "fingerprint: PASS" in output
    assert "source manifest: PASS" in output
    assert "model: PASS" in output
    assert "items: PASS" in output
    assert "manifest semantics: PASS" in output
    assert "batch items: 2" in output
    assert "verification: PASS" in output



def _fake_onnx_worker(
    command: list[str],
    **_: object,
) -> SimpleNamespace:
    def value(flag: str) -> Path:
        return Path(
            command[command.index(flag) + 1]
        )

    model = value("--model")
    input_path = value("--input")
    output = value("--output")
    manifest = value("--manifest")

    tensor = np.load(
        input_path,
        allow_pickle=False,
    )

    result = np.array(
        [float(np.mean(tensor)) + 0.5],
        dtype=np.float32,
    )

    with output.open("wb") as handle:
        np.save(
            handle,
            result,
            allow_pickle=False,
        )

    worker = {
        "schema_version": 1,
        "status": "complete",
        "backend": "cpu-onnx",
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
        },
        "model": {
            "path": str(model),
            "sha256": sha256_file(model),
        },
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
            "active_providers": [
                "CPUExecutionProvider"
            ],
        },
        "timing": {
            "session_create_seconds": 0.01,
            "inference_seconds": 0.02,
        },
    }

    manifest.write_text(
        json.dumps(worker),
        encoding="utf-8",
    )

    return SimpleNamespace(
        returncode=0,
        stdout="ok",
        stderr="",
    )


def test_execute_batch_plan_and_verify_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(
        tmp_path / "source"
    )

    plan_path = create_batch_plan(
        manifest_path=manifest,
        output_dir=tmp_path / "plans",
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_onnx_worker,
    )

    record_path = execute_batch_plan(
        plan_path,
        onnx_python=Path(sys.executable),
    )

    report = verify_batch_record(record_path)

    assert report["ok"] is True
    assert report["batch_plan_ok"] is True
    assert report["runs_ok"] is True
    assert report["cross_artifact_bindings_ok"] is True
    assert report["n_items"] == 2
    assert report["n_completed_runs"] == 2


def test_batch_execution_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(
        tmp_path / "source"
    )

    plan_path = create_batch_plan(
        manifest_path=manifest,
        output_dir=tmp_path / "plans",
    )

    calls = {"count": 0}

    def counted(
        command: list[str],
        **kwargs: object,
    ) -> SimpleNamespace:
        calls["count"] += 1
        return _fake_onnx_worker(
            command,
            **kwargs,
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        counted,
    )

    first = execute_batch_plan(
        plan_path,
        onnx_python=Path(sys.executable),
    )
    second = execute_batch_plan(
        plan_path,
        onnx_python=Path(sys.executable),
    )

    assert second == first
    assert calls["count"] == 2


def test_verify_target_recognizes_batch_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from radiation_edge_ai.execution import verify_target

    manifest = _write_manifest(
        tmp_path / "source"
    )

    plan_path = create_batch_plan(
        manifest_path=manifest,
        output_dir=tmp_path / "plans",
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_onnx_worker,
    )

    record_path = execute_batch_plan(
        plan_path,
        onnx_python=Path(sys.executable),
    )

    report = verify_target(record_path)

    assert report["kind"] == "batch_record"
    assert report["ok"] is True


def test_cli_batch_run_and_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from radiation_edge_ai.cli import main

    manifest = _write_manifest(
        tmp_path / "source"
    )

    plan_path = create_batch_plan(
        manifest_path=manifest,
        output_dir=tmp_path / "plans",
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_onnx_worker,
    )

    status = main(
        [
            "batch-run",
            str(plan_path),
            "--onnx-python",
            sys.executable,
        ]
    )
    assert status == 0

    record_path = Path(
        capsys.readouterr().out.strip()
    )
    assert record_path.is_file()

    status = main(
        ["verify", str(record_path)]
    )
    assert status == 0

    output = capsys.readouterr().out

    assert "kind: batch_record" in output
    assert "batch_id: b1-" in output
    assert "fingerprint: PASS" in output
    assert "batch plan: PASS" in output
    assert "child runs: PASS" in output
    assert "cross-artifact bindings: PASS" in output
    assert "batch runs: 2 / 2" in output
    assert "verification: PASS" in output



def _completed_synthetic_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    manifest = _write_manifest(
        tmp_path / "source"
    )

    plan_path = create_batch_plan(
        manifest_path=manifest,
        output_dir=tmp_path / "plans",
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_onnx_worker,
    )

    return execute_batch_plan(
        plan_path,
        onnx_python=Path(sys.executable),
    )


def test_create_verify_batch_prediction_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_path = _completed_synthetic_batch(
        tmp_path,
        monkeypatch,
    )

    report_path = create_nasa_batch_prediction_report(
        batch_record_path=record_path,
        output_dir=tmp_path / "predictions",
    )

    report = verify_nasa_batch_prediction_report(
        report_path
    )

    assert report["ok"] is True
    assert report["source_batch_record_ok"] is True
    assert report["prediction_table_ok"] is True
    assert report["derivation_ok"] is True
    assert report["n_rows"] == 2
    assert report["burden_column"] == BURDEN_COLUMN

    value = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    table_path = Path(
        value["output"]["prediction_table"]["path"]
    )

    with table_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["sample_id"] == "N1"
    assert float(rows[0][BURDEN_COLUMN]) == pytest.approx(0.5)
    assert rows[1]["sample_id"] == "N2"
    assert float(rows[1][BURDEN_COLUMN]) == pytest.approx(1.5)


def test_batch_prediction_projection_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_path = _completed_synthetic_batch(
        tmp_path,
        monkeypatch,
    )

    first = create_nasa_batch_prediction_report(
        batch_record_path=record_path,
        output_dir=tmp_path / "predictions",
    )

    second = create_nasa_batch_prediction_report(
        batch_record_path=record_path,
        output_dir=tmp_path / "predictions",
    )

    assert second == first


def test_mutated_prediction_table_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_path = _completed_synthetic_batch(
        tmp_path,
        monkeypatch,
    )

    report_path = create_nasa_batch_prediction_report(
        batch_record_path=record_path,
        output_dir=tmp_path / "predictions",
    )

    value = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    table_path = Path(
        value["output"]["prediction_table"]["path"]
    )

    with table_path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write("\n")

    report = verify_nasa_batch_prediction_report(
        report_path
    )

    assert report["prediction_table_ok"] is False
    assert report["derivation_ok"] is False
    assert report["ok"] is False


def test_verify_target_recognizes_prediction_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from radiation_edge_ai.execution import verify_target

    record_path = _completed_synthetic_batch(
        tmp_path,
        monkeypatch,
    )

    report_path = create_nasa_batch_prediction_report(
        batch_record_path=record_path,
        output_dir=tmp_path / "predictions",
    )

    report = verify_target(report_path)

    assert report["kind"] == "nasa_batch_prediction_report"
    assert report["ok"] is True


def test_cli_batch_predictions_and_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from radiation_edge_ai.cli import main

    record_path = _completed_synthetic_batch(
        tmp_path,
        monkeypatch,
    )

    status = main(
        [
            "batch-predictions",
            str(record_path),
            "--output-dir",
            str(tmp_path / "predictions"),
        ]
    )

    assert status == 0

    report_path = Path(
        capsys.readouterr().out.strip()
    )

    assert report_path.is_file()

    status = main(
        ["verify", str(report_path)]
    )

    assert status == 0

    output = capsys.readouterr().out

    assert "kind: nasa_batch_prediction_report" in output
    assert "prediction_id: p1-" in output
    assert "fingerprint: PASS" in output
    assert "source batch record: PASS" in output
    assert "prediction table: PASS" in output
    assert "derivation: PASS" in output
    assert "prediction rows: 2" in output
    assert "burden column: cpu_onnx_burden" in output
    assert "verification: PASS" in output



def _write_measurement_valid_manifest(
    root: Path,
) -> Path:
    path = _write_manifest(root)

    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    for item in value["items"]:
        item["metadata"]["sample_name"] = "SAMPLE_A"
        item["metadata"]["dose_gy"] = 0.0
        item["metadata"]["timepoint_hr"] = 4.0

    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )

    return path


def test_execute_batch_measurement_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_measurement_valid_manifest(
        tmp_path / "source"
    )

    plan_path = create_batch_plan(
        manifest_path=manifest,
        output_dir=tmp_path / "plans",
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_onnx_worker,
    )

    record_path = execute_batch_measurement_transaction(
        plan_path,
        output_dir=tmp_path / "transactions",
        onnx_python=Path(sys.executable),
    )

    report = verify_batch_measurement_transaction(
        record_path
    )

    assert report["ok"] is True
    assert report["transaction_fingerprint_ok"] is True
    assert report["record_fingerprint_ok"] is True
    assert report["identity_bindings_ok"] is True
    assert report["artifacts_ok"] is True
    assert report["stages_ok"] is True
    assert report["cross_artifact_bindings_ok"] is True
    assert report["n_items"] == 2
    assert report["n_prediction_rows"] == 2
    assert report["n_nuclei"] == 2
    assert report["n_samples"] == 1


def test_batch_measurement_transaction_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_measurement_valid_manifest(
        tmp_path / "source"
    )

    plan_path = create_batch_plan(
        manifest_path=manifest,
        output_dir=tmp_path / "plans",
    )

    calls = {"count": 0}

    def counted(
        command: list[str],
        **kwargs: object,
    ) -> SimpleNamespace:
        calls["count"] += 1
        return _fake_onnx_worker(
            command,
            **kwargs,
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        counted,
    )

    first = execute_batch_measurement_transaction(
        plan_path,
        output_dir=tmp_path / "transactions",
        onnx_python=Path(sys.executable),
    )

    second = execute_batch_measurement_transaction(
        plan_path,
        output_dir=tmp_path / "transactions",
        onnx_python=Path(sys.executable),
    )

    assert second == first

    # Two nuclei execute only during the first transaction.
    assert calls["count"] == 2


def test_verify_target_recognizes_batch_measurement_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from radiation_edge_ai.execution import verify_target

    manifest = _write_measurement_valid_manifest(
        tmp_path / "source"
    )

    plan_path = create_batch_plan(
        manifest_path=manifest,
        output_dir=tmp_path / "plans",
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_onnx_worker,
    )

    record_path = execute_batch_measurement_transaction(
        plan_path,
        output_dir=tmp_path / "transactions",
        onnx_python=Path(sys.executable),
    )

    report = verify_target(record_path)

    assert report["kind"] == "batch_measurement_transaction"
    assert report["ok"] is True


def test_cli_batch_measure_and_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from radiation_edge_ai.cli import main

    manifest = _write_measurement_valid_manifest(
        tmp_path / "source"
    )

    plan_path = create_batch_plan(
        manifest_path=manifest,
        output_dir=tmp_path / "plans",
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_onnx_worker,
    )

    status = main(
        [
            "batch-measure",
            str(plan_path),
            "--output-dir",
            str(tmp_path / "transactions"),
            "--onnx-python",
            sys.executable,
        ]
    )

    assert status == 0

    record_path = Path(
        capsys.readouterr().out.strip()
    )

    assert record_path.is_file()

    status = main(
        ["verify", str(record_path)]
    )

    assert status == 0

    output = capsys.readouterr().out

    assert "kind: batch_measurement_transaction" in output
    assert "transaction_id: t1-" in output
    assert "transaction fingerprint: PASS" in output
    assert "record fingerprint: PASS" in output
    assert "identity bindings: PASS" in output
    assert "artifacts: PASS" in output
    assert "component stages: PASS" in output
    assert "cross-artifact bindings: PASS" in output
    assert (
        "transaction counts: "
        "2 items / 2 predictions / 2 nuclei / 1 samples"
        in output
    )
    assert "verification: PASS" in output


def test_transaction_scope_preserves_scientific_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_measurement_valid_manifest(
        tmp_path / "source"
    )

    plan_path = create_batch_plan(
        manifest_path=manifest,
        output_dir=tmp_path / "plans",
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_onnx_worker,
    )

    record_path = execute_batch_measurement_transaction(
        plan_path,
        output_dir=tmp_path / "transactions",
        onnx_python=Path(sys.executable),
    )

    record = json.loads(
        record_path.read_text(encoding="utf-8")
    )

    scope = record["scientific_scope"]

    assert scope["batch_inference_artifact_verified"] is True
    assert scope["prediction_table_artifact_verified"] is True
    assert scope["endpoint_reconstruction_artifact_verified"] is True
    assert scope["per_nucleus_focus_count_interpretation"] is False
    assert scope["biological_reference_read"] is False
    assert scope["biological_acceptance_evaluated"] is False
    assert scope["hardware_access_performed"] is False



def test_execute_assay_manifest_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_measurement_valid_manifest(
        tmp_path / "source"
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_onnx_worker,
    )

    summary = execute_assay_manifest(
        manifest,
        output_dir=tmp_path / "application",
        onnx_python=Path(sys.executable),
    )

    assert summary["kind"] == "assay_run_summary"
    assert summary["status"] == "complete"
    assert summary["verification"] == "PASS"
    assert summary["assay_id"] == "nasa-53bp1-r1-v2"

    counts = summary["counts"]

    assert counts == {
        "n_items": 2,
        "n_prediction_rows": 2,
        "n_nuclei": 2,
        "n_samples": 1,
    }

    endpoints = summary["sample_endpoints"]

    assert len(endpoints) == 1
    assert endpoints[0]["sample_name"] == "SAMPLE_A"
    assert endpoints[0]["n_nuclei"] == 2
    assert endpoints[0]["mean_latent_burden"] == pytest.approx(
        1.0
    )

    scope = summary["scientific_scope"]

    assert scope[
        "per_nucleus_focus_count_interpretation"
    ] is False
    assert scope["biological_reference_read"] is False
    assert scope[
        "biological_acceptance_evaluated"
    ] is False
    assert scope["hardware_access_performed"] is False


def test_assay_manifest_execution_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_measurement_valid_manifest(
        tmp_path / "source"
    )

    calls = {"count": 0}

    def counted(
        command: list[str],
        **kwargs: object,
    ) -> SimpleNamespace:
        calls["count"] += 1
        return _fake_onnx_worker(
            command,
            **kwargs,
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        counted,
    )

    first = execute_assay_manifest(
        manifest,
        output_dir=tmp_path / "application",
        onnx_python=Path(sys.executable),
    )

    second = execute_assay_manifest(
        manifest,
        output_dir=tmp_path / "application",
        onnx_python=Path(sys.executable),
    )

    assert second == first

    # Two synthetic nuclei execute only once.
    assert calls["count"] == 2


def test_cli_assay_run_human_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from radiation_edge_ai.cli import main

    manifest = _write_measurement_valid_manifest(
        tmp_path / "source"
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_onnx_worker,
    )

    status = main(
        [
            "assay-run",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "application"),
            "--onnx-python",
            sys.executable,
        ]
    )

    assert status == 0

    output = capsys.readouterr().out

    assert "Radiation Edge AI - assay run" in output
    assert "transaction: t1-" in output
    assert (
        "counts: 2 items / 2 predictions / "
        "2 nuclei / 1 samples"
        in output
    )
    assert (
        "SAMPLE_A: n=2, mean_latent_burden=1.0"
        in output
    )
    assert (
        "per-nucleus focus-count interpretation: False"
        in output
    )
    assert "biological reference read: False" in output
    assert (
        "biological acceptance evaluated: False"
        in output
    )
    assert "hardware access performed: False" in output
    assert "verification: PASS" in output


def test_cli_assay_run_json_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from radiation_edge_ai.cli import main

    manifest = _write_measurement_valid_manifest(
        tmp_path / "source"
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_onnx_worker,
    )

    status = main(
        [
            "assay-run",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "application"),
            "--onnx-python",
            sys.executable,
            "--json",
        ]
    )

    assert status == 0

    value = json.loads(
        capsys.readouterr().out
    )

    assert value["kind"] == "assay_run_summary"
    assert value["verification"] == "PASS"
    assert value["counts"]["n_nuclei"] == 2
    assert value["counts"]["n_samples"] == 1
    assert len(value["sample_endpoints"]) == 1
