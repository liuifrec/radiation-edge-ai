from __future__ import annotations

import json
from pathlib import Path

import pytest

from radiation_edge_ai.control import (
    ControlPlaneError,
)
from radiation_edge_ai.dna_fiber import (
    application,
)


def _write_json(
    path: Path,
    value: object,
) -> Path:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
        )
        handle.write("\n")

    return path


def _manifest_value(
    model: Path,
    validation: Path,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "assay_id": (
            "dnai-fiber-v3"
        ),
        "application_adapter": (
            "dnai-tile-stitch-object-v1"
        ),
        "backend": "cpu-onnx",
        "input_mode": (
            "frozen-deployment-windows"
        ),
        "model": {
            "path": str(model),
            "sha256": "a" * 64,
            "size_bytes": 1,
        },
        "validation_manifest": {
            "path": str(validation),
            "sha256": "b" * 64,
            "size_bytes": 1,
        },
        "image_index": 0,
        "metadata": {
            "sample_id": "sample-1",
            "pixel_size_um": 0.26,
            "channel_semantics": (
                "synthetic red/green"
            ),
        },
    }


def _field_value(
) -> dict[str, object]:
    return {
        "field_id": "df1-test",
        "counts": {
            "n_windows": 9,
            "n_fibers_all": 3,
            "n_fibers_valid": 2,
        },
        "measurements": {
            "mean_valid_ratio": 1.0,
            "median_valid_ratio": 1.0,
            "mean_valid_length_um": 2.0,
            "total_valid_length_um": 4.0,
        },
        "scientific_scope": {
            "preprocessed_frozen_window_inputs": True,
            "raw_microscopy_preprocessing_performed": False,
            "floating_fp512_inference_performed": True,
            "gaussian_stitching_performed": True,
            "fiber_object_reconstruction_performed": True,
            "human_annotations_read": False,
            "fp1024_reference_read": False,
            "biological_fidelity_evaluated": False,
            "kl720_hardware_access_performed": False,
        },
    }


def _transaction_value(
) -> dict[str, object]:
    return {
        "transaction_id": (
            "dt1-test"
        ),
        "scientific_scope": {
            "field_record_verified": True,
            "prediction_semantics": "synthetic",
            "endpoint_semantics": "synthetic",
            "raw_microscopy_preprocessing_performed": False,
            "human_annotations_read": False,
            "fp1024_reference_read": False,
            "biological_fidelity_evaluated": False,
            "kl720_hardware_access_performed": False,
            "result_packaging_performed": False,
        },
    }


def test_dnai_application_requires_external_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = (
        tmp_path
        / "manifest.json"
    )

    model = tmp_path / "model.onnx"
    validation = (
        tmp_path
        / "validation.json"
    )

    value = _manifest_value(
        model,
        validation,
    )

    _write_json(
        manifest,
        value,
    )

    monkeypatch.setattr(
        application,
        "validate_dnai_application_manifest",
        lambda _path: value,
    )

    with pytest.raises(
        ControlPlaneError,
        match=(
            "requires --onnx-python"
        ),
    ):
        application.execute_dnai_assay_manifest(
            manifest,
            output_dir=(
                tmp_path / "out"
            ),
        )


def test_dnai_application_reuses_verified_field_and_wraps_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = (
        tmp_path
        / "manifest.json"
    )

    model = tmp_path / "model.onnx"
    validation = (
        tmp_path
        / "validation.json"
    )

    value = _manifest_value(
        model,
        validation,
    )

    _write_json(
        manifest,
        value,
    )

    runtime_python = (
        tmp_path / "python.exe"
    )
    runtime_python.write_bytes(
        b"x"
    )

    field = _write_json(
        tmp_path
        / "existing"
        / "field_result.json",
        _field_value(),
    )

    transaction = _write_json(
        tmp_path
        / "transaction_record.json",
        _transaction_value(),
    )

    calls: dict[str, object] = {}

    monkeypatch.setattr(
        application,
        "validate_dnai_application_manifest",
        lambda _path: value,
    )

    monkeypatch.setattr(
        application,
        "_matching_existing_field_record",
        lambda _root, _manifest: field,
    )

    def fail_worker(
        *_args: object,
        **_kwargs: object,
    ) -> Path:
        raise AssertionError(
            "worker must not run when a verified field is reusable"
        )

    monkeypatch.setattr(
        application,
        "_run_field_worker",
        fail_worker,
    )

    monkeypatch.setattr(
        application,
        "verify_dnai_fiber_field_record",
        lambda *_args, **_kwargs: {
            "ok": True
        },
    )

    def fake_create_transaction(
        manifest_path: Path,
        field_path: Path,
        *,
        output_dir: Path,
    ) -> Path:
        calls["manifest"] = (
            manifest_path
        )
        calls["field"] = field_path
        calls["output_dir"] = (
            output_dir
        )
        return transaction

    monkeypatch.setattr(
        application,
        "create_dnai_fiber_measurement_transaction",
        fake_create_transaction,
    )

    monkeypatch.setattr(
        application,
        "verify_dnai_fiber_measurement_transaction",
        lambda *_args, **_kwargs: {
            "ok": True
        },
    )

    result = (
        application.execute_dnai_assay_manifest(
            manifest,
            output_dir=(
                tmp_path / "app"
            ),
            onnx_python=(
                runtime_python
            ),
        )
    )

    assert result["assay_id"] == (
        "dnai-fiber-v3"
    )
    assert result["verification"] == "PASS"
    assert result["field_execution"] == "reused"

    ids = result["ids"]
    assert isinstance(ids, dict)
    assert ids["field_id"] == "df1-test"
    assert (
        ids["transaction_id"]
        == "dt1-test"
    )

    counts = result["counts"]
    assert isinstance(counts, dict)
    assert counts["n_windows"] == 9
    assert counts[
        "n_fibers_valid"
    ] == 2

    assert calls["manifest"] == (
        manifest.resolve()
    )
    assert calls["field"] == field
    assert calls[
        "output_dir"
    ] == (
        tmp_path
        / "app"
        / "transactions"
    ).resolve()


def test_dnai_application_executes_worker_when_no_reusable_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = (
        tmp_path
        / "manifest.json"
    )

    model = tmp_path / "model.onnx"
    validation = (
        tmp_path
        / "validation.json"
    )

    value = _manifest_value(
        model,
        validation,
    )

    _write_json(
        manifest,
        value,
    )

    runtime_python = (
        tmp_path / "python.exe"
    )
    runtime_python.write_bytes(
        b"x"
    )

    field = _write_json(
        tmp_path
        / "new"
        / "field_result.json",
        _field_value(),
    )

    transaction = _write_json(
        tmp_path
        / "transaction_record.json",
        _transaction_value(),
    )

    calls: dict[str, object] = {}

    monkeypatch.setattr(
        application,
        "validate_dnai_application_manifest",
        lambda _path: value,
    )

    monkeypatch.setattr(
        application,
        "_matching_existing_field_record",
        lambda _root, _manifest: None,
    )

    def fake_worker(
        _manifest: dict[str, object],
        *,
        runtime_python: Path,
        output_root: Path,
    ) -> Path:
        calls["runtime_python"] = (
            runtime_python
        )
        calls["field_output"] = (
            output_root
        )
        return field

    monkeypatch.setattr(
        application,
        "_run_field_worker",
        fake_worker,
    )

    monkeypatch.setattr(
        application,
        "verify_dnai_fiber_field_record",
        lambda *_args, **_kwargs: {
            "ok": True
        },
    )

    monkeypatch.setattr(
        application,
        "create_dnai_fiber_measurement_transaction",
        lambda *_args, **_kwargs: transaction,
    )

    monkeypatch.setattr(
        application,
        "verify_dnai_fiber_measurement_transaction",
        lambda *_args, **_kwargs: {
            "ok": True
        },
    )

    result = (
        application.execute_dnai_assay_manifest(
            manifest,
            output_dir=(
                tmp_path / "app"
            ),
            onnx_python=(
                runtime_python
            ),
        )
    )

    assert (
        result["field_execution"]
        == "executed"
    )

    assert calls[
        "runtime_python"
    ] == runtime_python.resolve()

    assert calls[
        "field_output"
    ] == (
        tmp_path
        / "app"
        / "fields"
    ).resolve()


def test_dnai_application_rejects_unverified_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = (
        tmp_path
        / "manifest.json"
    )

    model = tmp_path / "model.onnx"
    validation = (
        tmp_path
        / "validation.json"
    )

    value = _manifest_value(
        model,
        validation,
    )

    _write_json(
        manifest,
        value,
    )

    runtime_python = (
        tmp_path / "python.exe"
    )
    runtime_python.write_bytes(
        b"x"
    )

    field = _write_json(
        tmp_path
        / "field_result.json",
        _field_value(),
    )

    monkeypatch.setattr(
        application,
        "validate_dnai_application_manifest",
        lambda _path: value,
    )

    monkeypatch.setattr(
        application,
        "_matching_existing_field_record",
        lambda _root, _manifest: field,
    )

    monkeypatch.setattr(
        application,
        "verify_dnai_fiber_field_record",
        lambda *_args, **_kwargs: {
            "ok": False
        },
    )

    with pytest.raises(
        ControlPlaneError,
        match=(
            "field record failed verification"
        ),
    ):
        application.execute_dnai_assay_manifest(
            manifest,
            output_dir=(
                tmp_path / "app"
            ),
            onnx_python=(
                runtime_python
            ),
        )
