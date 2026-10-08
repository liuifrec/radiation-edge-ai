from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from radiation_edge_ai.control import ControlPlaneError, sha256_file
from radiation_edge_ai.dna_fiber import reporting
from radiation_edge_ai.dna_fiber.transaction_v2 import (
    EXPECTED_TRANSACTION_SCOPE as PHYSICAL_SCOPE,
)


def _write_json(path: Path, value: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return path


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path.resolve()),
        "size_bytes": path.stat().st_size,
    }


def _physical_source(tmp_path: Path) -> Path:
    fiber_path = tmp_path / "source" / "valid_fibers.csv"
    fiber_path.parent.mkdir(parents=True, exist_ok=True)
    fiber_path.write_text(
        "Fiber ID,First analog (um),Second analog (um),Ratio\n"
        "8,4.0,8.0,2.0\n",
        encoding="utf-8",
    )

    counts = {
        "n_windows": 9,
        "n_fibers_all": 3,
        "n_fibers_valid": 1,
    }
    measurements = {
        "mean_valid_ratio": 2.0,
        "median_valid_ratio": 2.0,
        "mean_valid_length_um": 12.0,
        "total_valid_length_um": 12.0,
    }

    field = _write_json(
        tmp_path / "source" / "field_result.json",
        {
            "schema_version": 2,
            "record_type": "dnai_fiber_field_record",
            "field_id": "df2-test",
            "identity": {
                "sample_id": "sample-1",
                "image_index": 0,
                "model_sha256": "a" * 64,
                "validation_manifest_sha256": "b" * 64,
                "backend": "kl720",
                "input_semantics": "frozen preprocessed deployment windows",
            },
            "counts": counts,
            "measurements": measurements,
            "artifacts": {"valid_fibers_csv": _artifact(fiber_path)},
        },
    )

    transaction = _write_json(
        tmp_path / "source" / "transaction_record.json",
        {
            "schema_version": 2,
            "record_type": "dnai_fiber_measurement_transaction",
            "transaction_id": "dt2-test",
            "transaction_fingerprint_sha256": "c" * 64,
            "record_fingerprint_sha256": "d" * 64,
            "transaction_identity": {
                "schema_version": 2,
                "assay_id": "dnai-fiber-v3",
                "backend": "kl720",
                "field_id": "df2-test",
                "field_record_sha256": sha256_file(field),
                "field_record_size_bytes": field.stat().st_size,
                "sample_id": "sample-1",
                "image_index": 0,
                "model_sha256": "a" * 64,
                "validation_manifest_sha256": "b" * 64,
            },
            "field_record": _artifact(field),
            "counts": counts,
            "measurements": measurements,
            "scientific_scope": dict(PHYSICAL_SCOPE),
        },
    )
    return transaction


def test_physical_dnai_package_portable_and_source_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _physical_source(tmp_path)
    calls: list[bool] = []

    def fake_verify(_path: Path, *, check_artifacts: bool) -> dict[str, bool]:
        calls.append(check_artifacts)
        return {"ok": True}

    monkeypatch.setattr(
        reporting, "verify_dnai_fiber_physical_measurement_transaction",
        fake_verify,
    )
    manifest_path = reporting.create_dnai_fiber_result_package(
        source, output_dir=tmp_path / "packages",
    )
    assert calls == [True]
    verification = reporting.verify_dnai_fiber_result_package(manifest_path)
    assert verification["ok"] is True
    assert verification["scientific_scope_ok"] is True
    assert verification["csv_binding_ok"] is True

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest["identity"]
    assert manifest["schema_version"] == 1
    assert manifest["package_id"].startswith("drp1-")
    assert identity["source_transaction_schema_version"] == 2
    assert identity["execution_backend"] == "kl720"
    assert identity["source_scientific_scope"] == PHYSICAL_SCOPE
    assert identity["source_transaction_sha256"] == sha256_file(source)

    report_text = (manifest_path.parent / "report.md").read_text(
        encoding="utf-8"
    )
    assert "physical KL720 application path" in report_text
    assert "floating FP512 application path" not in report_text
    assert str(source.resolve()) not in json.dumps(manifest)

    relocated = tmp_path / "relocated" / manifest_path.parent.name
    relocated.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(manifest_path.parent, relocated)
    relocated_path = relocated / "package_manifest.json"
    assert reporting.verify_dnai_fiber_result_package(relocated_path)["ok"] is True

    with (relocated / "valid_fibers.csv").open("a", encoding="utf-8") as handle:
        handle.write("9,5.0,5.0,1.0\n")
    tampered = reporting.verify_dnai_fiber_result_package(relocated_path)
    assert tampered["files_ok"] is False
    assert tampered["ok"] is False
    assert reporting.verify_dnai_fiber_result_package(manifest_path)["ok"] is True


def test_unknown_transaction_schema_is_fail_closed(
    tmp_path: Path,
) -> None:
    source = _physical_source(tmp_path)
    transaction = json.loads(source.read_text(encoding="utf-8"))
    transaction["schema_version"] = 3
    source.write_text(json.dumps(transaction), encoding="utf-8")

    with pytest.raises(ControlPlaneError, match="schema_version"):
        reporting.create_dnai_fiber_result_package(
            source, output_dir=tmp_path / "packages",
        )
