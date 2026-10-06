from __future__ import annotations

import json
from pathlib import Path

import pytest

from radiation_edge_ai.control import (
    ControlPlaneError,
)
from radiation_edge_ai.dna_fiber import (
    transaction_v2,
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


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: str = "kl720",
) -> tuple[
    Path,
    Path,
]:
    validation_sha = (
        "a" * 64
    )

    manifest_path = (
        tmp_path
        / "application_manifest.json"
    )

    field_path = (
        tmp_path
        / "field_result.json"
    )

    _write_json(
        manifest_path,
        {
            "synthetic": True
        },
    )

    field = {
        "schema_version": 2,
        "record_type": (
            "dnai_fiber_field_record"
        ),
        "status": "complete",
        "field_id": (
            "df2-0123456789abcdef"
        ),
        "identity": {
            "backend": "kl720",
            "model_sha256": (
                transaction_v2
                .EXPECTED_NEF_SHA256
            ),
            "validation_manifest_sha256": (
                validation_sha
            ),
            "image_index": 0,
            "sample_id": (
                "sample-1"
            ),
        },
        "counts": {
            "n_windows": 9,
            "n_fibers_all": 18,
            "n_fibers_valid": 14,
        },
        "measurements": {
            "mean_valid_ratio": 1.0,
            "median_valid_ratio": 1.0,
            "mean_valid_length_um": 16.0,
            "total_valid_length_um": 224.0,
        },
    }

    _write_json(
        field_path,
        field,
    )

    manifest = {
        "schema_version": 1,
        "assay_id": (
            transaction_v2.ASSAY_ID
        ),
        "application_adapter": (
            transaction_v2
            .APPLICATION_ADAPTER
        ),
        "backend": backend,
        "input_mode": (
            transaction_v2
            .INPUT_MODE
        ),
        "image_index": 0,
        "model": {
            "path": "synthetic.nef",
            "sha256": (
                transaction_v2
                .EXPECTED_NEF_SHA256
            ),
            "size_bytes": 1,
        },
        "validation_manifest": {
            "path": (
                "synthetic-validation.json"
            ),
            "sha256": (
                validation_sha
            ),
            "size_bytes": 1,
        },
        "metadata": {
            "sample_id": (
                "sample-1"
            ),
            "pixel_size_um": 0.26,
            "channel_semantics": (
                "synthetic red/green"
            ),
        },
    }

    monkeypatch.setattr(
        transaction_v2,
        "validate_dnai_application_manifest",
        lambda _path: manifest,
    )

    monkeypatch.setattr(
        transaction_v2,
        "verify_dnai_fiber_physical_field_record",
        lambda *_args, **_kwargs: {
            "ok": True
        },
    )

    return (
        manifest_path,
        field_path,
    )


def test_create_and_verify_dt2_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        manifest,
        field,
    ) = _prepare(
        tmp_path,
        monkeypatch,
    )

    record_path = (
        transaction_v2
        .create_dnai_fiber_physical_measurement_transaction(
            manifest,
            field,
            output_dir=(
                tmp_path
                / "transactions"
            ),
        )
    )

    value = json.loads(
        record_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        value[
            "transaction_id"
        ]
        .startswith(
            "dt2-"
        )
    )

    assert (
        value[
            "transaction_identity"
        ][
            "backend"
        ]
        == "kl720"
    )

    assert (
        value[
            "scientific_scope"
        ][
            "kl720_hardware_access_performed"
        ]
        is True
    )

    assert (
        value[
            "scientific_scope"
        ][
            "biological_fidelity_evaluated"
        ]
        is False
    )

    report = (
        transaction_v2
        .verify_dnai_fiber_physical_measurement_transaction(
            record_path,
            check_artifacts=True,
        )
    )

    assert report[
        "ok"
    ] is True


def test_dt2_rejects_cpu_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        manifest,
        field,
    ) = _prepare(
        tmp_path,
        monkeypatch,
        backend="cpu-onnx",
    )

    with pytest.raises(
        ControlPlaneError,
        match="kl720",
    ):
        (
            transaction_v2
            .create_dnai_fiber_physical_measurement_transaction(
                manifest,
                field,
                output_dir=(
                    tmp_path
                    / "transactions"
                ),
            )
        )


def test_dt2_detects_scope_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        manifest,
        field,
    ) = _prepare(
        tmp_path,
        monkeypatch,
    )

    record_path = (
        transaction_v2
        .create_dnai_fiber_physical_measurement_transaction(
            manifest,
            field,
            output_dir=(
                tmp_path
                / "transactions"
            ),
        )
    )

    value = json.loads(
        record_path.read_text(
            encoding="utf-8"
        )
    )

    value[
        "scientific_scope"
    ][
        "biological_fidelity_evaluated"
    ] = True

    _write_json(
        record_path,
        value,
    )

    report = (
        transaction_v2
        .verify_dnai_fiber_physical_measurement_transaction(
            record_path,
            check_artifacts=True,
        )
    )

    assert report[
        "scope_ok"
    ] is False

    assert report[
        "ok"
    ] is False
