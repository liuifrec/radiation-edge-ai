from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from radiation_edge_ai.assay_runtime import (
    DNAI_ASSAY_ID,
    NASA_ASSAY_ID,
    create_registered_assay_result_package,
    execute_registered_assay_manifest,
    get_assay_runtime_spec,
    list_assay_runtime_specs,
    verify_registered_assay_result_package,
)
from radiation_edge_ai.control import (
    ControlPlaneError,
    list_assays,
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
        )
        handle.write("\n")

    return path


def test_runtime_registry_covers_canonical_assays(
) -> None:
    canonical = {
        spec.assay_id
        for spec in list_assays()
    }

    runtime = {
        spec.assay_id
        for spec in list_assay_runtime_specs()
    }

    assert runtime == canonical
    assert runtime == {
        DNAI_ASSAY_ID,
        NASA_ASSAY_ID,
    }

    nasa = get_assay_runtime_spec(
        NASA_ASSAY_ID
    )

    assert nasa.manifest_run_supported is True
    assert nasa.result_package_supported is True
    assert (
        nasa.package_source_record_type
        == "batch_measurement_transaction"
    )

    dnai = get_assay_runtime_spec(
        DNAI_ASSAY_ID
    )

    assert dnai.manifest_run_supported is False
    assert dnai.result_package_supported is False
    assert (
        dnai.package_source_record_type
        == "dnai_fiber_measurement_transaction"
    )

    assert (
        "Pixel agreement is not a surrogate"
        in dnai.scientific_boundary
    )


def test_nasa_manifest_dispatch_preserves_existing_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from radiation_edge_ai import application

    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "assay_id": NASA_ASSAY_ID,
        },
    )

    output_dir = tmp_path / "application"
    onnx_python = tmp_path / "python.exe"

    calls: dict[str, object] = {}

    def fake_execute(
        manifest_path: Path,
        *,
        output_dir: Path,
        onnx_python: Optional[Path] = None,  # noqa: UP045
    ) -> dict[str, object]:
        calls["manifest"] = manifest_path
        calls["output_dir"] = output_dir
        calls["onnx_python"] = onnx_python

        return {
            "kind": "assay_run_summary",
            "assay_id": NASA_ASSAY_ID,
            "verification": "PASS",
        }

    monkeypatch.setattr(
        application,
        "execute_assay_manifest",
        fake_execute,
    )

    result = execute_registered_assay_manifest(
        manifest,
        output_dir=output_dir,
        onnx_python=onnx_python,
    )

    assert result["assay_id"] == NASA_ASSAY_ID
    assert result["verification"] == "PASS"

    assert calls == {
        "manifest": manifest,
        "output_dir": output_dir,
        "onnx_python": onnx_python,
    }


def test_dnai_manifest_is_registered_but_fails_closed(
    tmp_path: Path,
) -> None:
    manifest = _write_json(
        tmp_path / "dnai_manifest.json",
        {
            "schema_version": 1,
            "assay_id": DNAI_ASSAY_ID,
        },
    )

    with pytest.raises(
        ControlPlaneError,
        match="registered but manifest execution is not enabled",
    ) as exc:
        execute_registered_assay_manifest(
            manifest,
            output_dir=tmp_path / "out",
        )

    message = str(exc.value)

    assert "dnai-tile-stitch-object-v1" in message
    assert (
        "Pixel agreement is not a surrogate"
        in message
    )


def test_unknown_manifest_assay_is_rejected(
    tmp_path: Path,
) -> None:
    manifest = _write_json(
        tmp_path / "unknown.json",
        {
            "schema_version": 1,
            "assay_id": "unknown-assay",
        },
    )

    with pytest.raises(
        ControlPlaneError,
        match="Unknown assay",
    ):
        execute_registered_assay_manifest(
            manifest,
            output_dir=tmp_path / "out",
        )


def test_nasa_result_packaging_dispatches_existing_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from radiation_edge_ai import reporting

    source = _write_json(
        tmp_path / "transaction_record.json",
        {
            "schema_version": 1,
            "record_type": (
                "batch_measurement_transaction"
            ),
            "transaction_identity": {
                "assay_id": NASA_ASSAY_ID,
            },
        },
    )

    output_dir = tmp_path / "packages"
    expected = (
        output_dir
        / "rp1-test"
        / "package_manifest.json"
    )

    calls: dict[str, object] = {}

    def fake_create(
        transaction_path: Path,
        *,
        output_dir: Path,
    ) -> Path:
        calls["source"] = transaction_path
        calls["output_dir"] = output_dir
        return expected

    monkeypatch.setattr(
        reporting,
        "create_assay_result_package",
        fake_create,
    )

    result = create_registered_assay_result_package(
        source,
        output_dir=output_dir,
    )

    assert result == expected

    assert calls == {
        "source": source.resolve(),
        "output_dir": output_dir,
    }


def test_dnai_result_packaging_is_registered_but_fails_closed(
    tmp_path: Path,
) -> None:
    source = _write_json(
        tmp_path / "dnai_transaction.json",
        {
            "schema_version": 1,
            "record_type": (
                "dnai_fiber_measurement_transaction"
            ),
            "assay_id": DNAI_ASSAY_ID,
        },
    )

    with pytest.raises(
        ControlPlaneError,
        match="registered but result packaging is not enabled",
    ) as exc:
        create_registered_assay_result_package(
            source,
            output_dir=tmp_path / "packages",
        )

    assert (
        "dnai-fiber-measurement-package-v1"
        in str(exc.value)
    )


def test_nasa_package_verification_dispatches_existing_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from radiation_edge_ai import reporting

    manifest = _write_json(
        tmp_path
        / "rp1-test"
        / "package_manifest.json",
        {
            "schema_version": 1,
            "record_type": "assay_result_package",
            "identity": {
                "assay_id": NASA_ASSAY_ID,
            },
        },
    )

    calls: dict[str, object] = {}

    def fake_verify(
        manifest_path: Path,
    ) -> dict[str, object]:
        calls["manifest"] = manifest_path

        return {
            "kind": "assay_result_package",
            "package_id": "rp1-test",
            "ok": True,
        }

    monkeypatch.setattr(
        reporting,
        "verify_assay_result_package",
        fake_verify,
    )

    report = verify_registered_assay_result_package(
        manifest
    )

    assert report["ok"] is True
    assert report["assay_id"] == NASA_ASSAY_ID
    assert (
        report["runtime_adapter"]
        == "nasa-sample-aggregate-package-v1"
    )

    assert calls["manifest"] == manifest.resolve()


def test_dnai_package_verifier_fails_closed_until_enabled(
    tmp_path: Path,
) -> None:
    manifest = _write_json(
        tmp_path
        / "rp1-dnai"
        / "package_manifest.json",
        {
            "schema_version": 1,
            "record_type": "assay_result_package",
            "identity": {
                "assay_id": DNAI_ASSAY_ID,
            },
        },
    )

    with pytest.raises(
        ControlPlaneError,
        match="has no enabled result-package verifier",
    ):
        verify_registered_assay_result_package(
            manifest
        )


def test_generic_verify_target_routes_package_through_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import radiation_edge_ai.assay_runtime as runtime
    from radiation_edge_ai.execution import verify_target

    manifest = _write_json(
        tmp_path
        / "rp1-test"
        / "package_manifest.json",
        {
            "schema_version": 1,
            "record_type": "assay_result_package",
            "identity": {
                "assay_id": NASA_ASSAY_ID,
            },
        },
    )

    calls: dict[str, object] = {}

    def fake_verify(
        manifest_path: Path,
    ) -> dict[str, object]:
        calls["manifest"] = manifest_path

        return {
            "kind": "assay_result_package",
            "package_id": "rp1-test",
            "assay_id": NASA_ASSAY_ID,
            "runtime_adapter": (
                "nasa-sample-aggregate-package-v1"
            ),
            "ok": True,
        }

    monkeypatch.setattr(
        runtime,
        "verify_registered_assay_result_package",
        fake_verify,
    )

    report = verify_target(
        manifest
    )

    assert report["ok"] is True
    assert report["assay_id"] == NASA_ASSAY_ID
    assert calls["manifest"] == manifest.resolve()
