from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from radiation_edge_ai.control import (
    ControlPlaneError,
    sha256_file,
)
from radiation_edge_ai.dna_fiber import (
    reporting,
)
from radiation_edge_ai.dna_fiber.transaction import (
    EXPECTED_TRANSACTION_SCOPE,
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
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")

    return path


def _artifact(
    path: Path,
) -> dict[str, object]:
    return {
        "path": str(
            path.resolve()
        ),
        "sha256": (
            sha256_file(
                path.resolve()
            )
        ),
        "size_bytes": (
            path.stat().st_size
        ),
    }


def _synthetic_source(
    tmp_path: Path,
) -> tuple[Path, Path]:
    fibers = (
        tmp_path
        / "field"
        / "valid_fibers.csv"
    )

    fibers.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fibers.write_text(
        (
            "img_name,ratio,length\n"
            "sample-1,1.0,10.0\n"
            "sample-1,2.0,20.0\n"
        ),
        encoding="utf-8",
    )

    counts = {
        "n_windows": 9,
        "n_fibers_all": 3,
        "n_fibers_valid": 2,
    }

    measurements = {
        "mean_valid_ratio": 1.5,
        "median_valid_ratio": 1.5,
        "mean_valid_length_um": 3.9,
        "total_valid_length_um": 7.8,
    }

    field = _write_json(
        tmp_path
        / "field"
        / "field_result.json",
        {
            "schema_version": 1,
            "record_type": (
                "dnai_fiber_field_record"
            ),
            "status": "complete",
            "field_id": "df1-test",
            "identity": {
                "sample_id": "sample-1",
                "image_index": 0,
                "model_sha256": (
                    "a" * 64
                ),
                "validation_manifest_sha256": (
                    "b" * 64
                ),
                "input_semantics": (
                    "frozen preprocessed and ImageNet-normalized "
                    "DNAi deployment tensors"
                ),
            },
            "counts": counts,
            "measurements": (
                measurements
            ),
            "artifacts": {
                "valid_fibers_csv": (
                    _artifact(
                        fibers
                    )
                ),
            },
        },
    )

    transaction = _write_json(
        tmp_path
        / "transaction"
        / "transaction_record.json",
        {
            "schema_version": 1,
            "record_type": (
                "dnai_fiber_measurement_transaction"
            ),
            "status": "complete",
            "transaction_id": (
                "dt1-test"
            ),
            "transaction_fingerprint_sha256": (
                "c" * 64
            ),
            "record_fingerprint_sha256": (
                "d" * 64
            ),
            "transaction_identity": {
                "assay_id": (
                    "dnai-fiber-v3"
                ),
                "field_id": (
                    "df1-test"
                ),
                "field_record_sha256": (
                    sha256_file(
                        field
                    )
                ),
                "field_record_size_bytes": (
                    field.stat().st_size
                ),
                "sample_id": (
                    "sample-1"
                ),
                "image_index": 0,
                "model_sha256": (
                    "a" * 64
                ),
                "validation_manifest_sha256": (
                    "b" * 64
                ),
            },
            "field_record": (
                _artifact(
                    field
                )
            ),
            "counts": counts,
            "measurements": (
                measurements
            ),
            "scientific_scope": (
                EXPECTED_TRANSACTION_SCOPE
            ),
        },
    )

    return transaction, fibers


def test_dnai_package_is_portable_and_self_verifying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, _fibers = (
        _synthetic_source(
            tmp_path
        )
    )

    monkeypatch.setattr(
        reporting,
        "verify_dnai_fiber_measurement_transaction",
        lambda *_args, **_kwargs: {
            "ok": True
        },
    )

    manifest = (
        reporting.create_dnai_fiber_result_package(
            transaction,
            output_dir=(
                tmp_path
                / "packages"
            ),
        )
    )

    report = (
        reporting.verify_dnai_fiber_result_package(
            manifest
        )
    )

    assert report["ok"] is True
    assert (
        report["identity_semantics_ok"]
        is True
    )
    assert report["files_ok"] is True
    assert (
        report["derivation_ok"]
        is True
    )
    assert (
        report["csv_binding_ok"]
        is True
    )
    assert report["n_windows"] == 9
    assert (
        report["n_fibers_valid"]
        == 2
    )

    package_value = json.loads(
        manifest.read_text(
            encoding="utf-8"
        )
    )

    text = json.dumps(
        package_value
    )

    assert str(
        transaction.resolve()
    ) not in text

    relocated_root = (
        tmp_path
        / "relocated"
        / manifest.parent.name
    )

    relocated_root.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copytree(
        manifest.parent,
        relocated_root,
    )

    relocated_report = (
        reporting.verify_dnai_fiber_result_package(
            relocated_root
            / "package_manifest.json"
        )
    )

    assert (
        relocated_report["ok"]
        is True
    )


def test_dnai_package_creation_reuses_verified_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, _fibers = (
        _synthetic_source(
            tmp_path
        )
    )

    monkeypatch.setattr(
        reporting,
        "verify_dnai_fiber_measurement_transaction",
        lambda *_args, **_kwargs: {
            "ok": True
        },
    )

    output_dir = (
        tmp_path / "packages"
    )

    first = (
        reporting.create_dnai_fiber_result_package(
            transaction,
            output_dir=output_dir,
        )
    )

    second = (
        reporting.create_dnai_fiber_result_package(
            transaction,
            output_dir=output_dir,
        )
    )

    assert second == first


def test_dnai_package_detects_copied_fiber_table_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, _fibers = (
        _synthetic_source(
            tmp_path
        )
    )

    monkeypatch.setattr(
        reporting,
        "verify_dnai_fiber_measurement_transaction",
        lambda *_args, **_kwargs: {
            "ok": True
        },
    )

    manifest = (
        reporting.create_dnai_fiber_result_package(
            transaction,
            output_dir=(
                tmp_path
                / "packages"
            ),
        )
    )

    fibers_copy = (
        manifest.parent
        / "valid_fibers.csv"
    )

    with fibers_copy.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "sample-1,9.0,90.0\n"
        )

    report = (
        reporting.verify_dnai_fiber_result_package(
            manifest
        )
    )

    assert report["files_ok"] is False
    assert report["ok"] is False


def test_dnai_package_refuses_unverified_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, _fibers = (
        _synthetic_source(
            tmp_path
        )
    )

    monkeypatch.setattr(
        reporting,
        "verify_dnai_fiber_measurement_transaction",
        lambda *_args, **_kwargs: {
            "ok": False
        },
    )

    with pytest.raises(
        ControlPlaneError,
        match=(
            "failed full verification"
        ),
    ):
        reporting.create_dnai_fiber_result_package(
            transaction,
            output_dir=(
                tmp_path
                / "packages"
            ),
        )
