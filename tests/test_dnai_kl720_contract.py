from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from radiation_edge_ai import assay_runtime, cli
from radiation_edge_ai.control import (
    ControlPlaneError,
)
from radiation_edge_ai.dna_fiber import (
    application,
    transaction,
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


def _validation_value() -> dict[str, object]:
    return {
        "dnai_commit": (
            transaction.DNAI_COMMIT
        ),
        "tile": [1, 3, 512, 512],
        "full_image_shape": [
            1,
            3,
            1024,
            1024,
        ],
        "overlap": 0.50,
        "stride": 256,
        "blend": "gaussian",
        "windows_per_image": 9,
        "records": [
            {
                "image_index": 0,
                "window_index": index,
                "sample_id": "sample-1",
            }
            for index in range(9)
        ],
    }


def _kl720_manifest(
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
        "backend": "kl720",
        "input_mode": (
            "frozen-deployment-windows"
        ),
        "model": {
            "path": str(model),
            "sha256": (
                transaction.EXPECTED_NEF_SHA256
            ),
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


def test_kl720_manifest_contract_accepts_frozen_nef(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = (
        tmp_path
        / "manifest.json"
    )

    validation_path = (
        tmp_path
        / "validation.json"
    )

    model_path = (
        tmp_path
        / "model.nef"
    )

    manifest = _kl720_manifest(
        model_path,
        validation_path,
    )

    validation = (
        _validation_value()
    )

    def fake_load(
        path: Path,
    ) -> dict[str, object]:
        resolved = (
            path
            .expanduser()
            .resolve()
        )

        if resolved == (
            manifest_path.resolve()
        ):
            return manifest

        if resolved == (
            validation_path.resolve()
        ):
            return validation

        raise AssertionError(
            f"Unexpected JSON path: {resolved}"
        )

    monkeypatch.setattr(
        transaction,
        "load_json_object",
        fake_load,
    )

    monkeypatch.setattr(
        transaction,
        "_artifact_bytes_ok",
        lambda _value: True,
    )

    result = (
        transaction
        .validate_dnai_application_manifest(
            manifest_path
        )
    )

    assert result["backend"] == "kl720"

    model = result["model"]
    assert isinstance(model, dict)

    assert (
        model["sha256"]
        == transaction.EXPECTED_NEF_SHA256
    )


def test_kl720_manifest_rejects_wrong_model_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = (
        tmp_path
        / "manifest.json"
    )

    validation_path = (
        tmp_path
        / "validation.json"
    )

    model_path = (
        tmp_path
        / "model.nef"
    )

    manifest = _kl720_manifest(
        model_path,
        validation_path,
    )

    model = manifest["model"]
    assert isinstance(model, dict)

    model["sha256"] = "0" * 64

    def fake_load(
        path: Path,
    ) -> dict[str, object]:
        if (
            path
            .expanduser()
            .resolve()
            == manifest_path.resolve()
        ):
            return manifest

        return _validation_value()

    monkeypatch.setattr(
        transaction,
        "load_json_object",
        fake_load,
    )

    monkeypatch.setattr(
        transaction,
        "_artifact_bytes_ok",
        lambda _value: True,
    )

    with pytest.raises(
        ControlPlaneError,
        match="model SHA256",
    ):
        transaction.validate_dnai_application_manifest(
            manifest_path
        )


def test_kl720_application_dispatches_physical_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = (
        tmp_path
        / "manifest.json"
    )

    _write_json(
        manifest_path,
        {
            "synthetic": True,
        },
    )

    manifest = {
        "backend": "kl720",
    }

    monkeypatch.setattr(
        application,
        "validate_dnai_application_manifest",
        lambda _path: manifest,
    )

    scientific_python = (
        tmp_path
        / "scientific-python.exe"
    )

    scientific_python.write_bytes(
        b"x"
    )

    kneron_python = (
        tmp_path
        / "kneron-python.exe"
    )

    kneron_python.write_bytes(
        b"x"
    )

    root = (
        tmp_path
        / "out"
    )

    field_path = (
        root
        / "fields"
        / "df2-synthetic"
        / "field_result.json"
    )

    transaction_path = (
        root
        / "transactions"
        / "dt2-synthetic"
        / "transaction_record.json"
    )

    _write_json(
        field_path,
        {
            "field_id": (
                "df2-synthetic"
            ),
            "counts": {
                "n_windows": 9,
                "n_fibers_valid": 14,
            },
            "measurements": {
                "mean_valid_ratio": 1.0,
            },
            "scientific_scope": {
                "kl720_hardware_access_performed": True,
                "biological_fidelity_evaluated": False,
            },
        },
    )

    _write_json(
        transaction_path,
        {
            "transaction_id": (
                "dt2-synthetic"
            ),
            "scientific_scope": {
                "field_record_verified": True,
                "kl720_hardware_access_performed": True,
                "biological_fidelity_evaluated": False,
            },
        },
    )

    monkeypatch.setattr(
        application,
        "_matching_existing_field_record",
        lambda _root, _manifest: None,
    )

    calls: dict[str, object] = {}

    def fake_worker(
        manifest_value: dict[str, object],
        *,
        runtime_python: Path,
        output_root: Path,
        kl720_python: Optional[Path] = None,  # noqa: UP045
        kl720_port: Optional[int] = None,  # noqa: UP045
        kl720_timeout_ms: int = 10000,
    ) -> Path:
        calls["manifest"] = (
            manifest_value
        )
        calls["runtime_python"] = (
            runtime_python
        )
        calls["output_root"] = (
            output_root
        )
        calls["kl720_python"] = (
            kl720_python
        )
        calls["kl720_port"] = (
            kl720_port
        )
        calls["kl720_timeout_ms"] = (
            kl720_timeout_ms
        )

        return field_path

    monkeypatch.setattr(
        application,
        "_run_field_worker",
        fake_worker,
    )

    monkeypatch.setattr(
        application,
        "verify_dnai_fiber_physical_field_record",
        lambda *_args, **_kwargs: {
            "ok": True,
        },
    )

    monkeypatch.setattr(
        application,
        "create_dnai_fiber_physical_measurement_transaction",
        lambda *_args, **_kwargs: (
            transaction_path
        ),
    )

    monkeypatch.setattr(
        application,
        "verify_dnai_fiber_physical_measurement_transaction",
        lambda *_args, **_kwargs: {
            "ok": True,
        },
    )

    def unexpected_v1(
        *_args: object,
        **_kwargs: object,
    ) -> object:
        raise AssertionError(
            "KL720 application used the historical v1 path"
        )

    monkeypatch.setattr(
        application,
        "verify_dnai_fiber_field_record",
        unexpected_v1,
    )

    monkeypatch.setattr(
        application,
        "create_dnai_fiber_measurement_transaction",
        unexpected_v1,
    )

    monkeypatch.setattr(
        application,
        "verify_dnai_fiber_measurement_transaction",
        unexpected_v1,
    )

    result = (
        application.execute_dnai_assay_manifest(
            manifest_path,
            output_dir=root,
            onnx_python=(
                scientific_python
            ),
            kl720_python=(
                kneron_python
            ),
            kl720_port=81,
            kl720_timeout_ms=12345,
        )
    )

    assert (
        result[
            "verification"
        ]
        == "PASS"
    )

    assert (
        result[
            "field_execution"
        ]
        == "executed"
    )

    assert (
        result[
            "ids"
        ][
            "field_id"
        ]
        == "df2-synthetic"
    )

    assert (
        result[
            "ids"
        ][
            "transaction_id"
        ]
        == "dt2-synthetic"
    )

    assert calls == {
        "manifest": manifest,
        "runtime_python": (
            scientific_python.resolve()
        ),
        "output_root": (
            root
            / "fields"
        ).resolve(),
        "kl720_python": (
            kneron_python
        ),
        "kl720_port": 81,
        "kl720_timeout_ms": 12345,
    }


def test_kl720_application_requires_scientific_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = (
        tmp_path
        / "manifest.json"
    )

    manifest_path.write_text(
        "{}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        application,
        "validate_dnai_application_manifest",
        lambda _path: {
            "backend": "kl720",
        },
    )

    kneron_python = (
        tmp_path
        / "kneron-python.exe"
    )

    kneron_python.write_bytes(
        b"x"
    )

    with pytest.raises(
        ControlPlaneError,
        match="--onnx-python",
    ):
        (
            application.execute_dnai_assay_manifest(
                manifest_path,
                output_dir=(
                    tmp_path
                    / "out"
                ),
                kl720_python=(
                    kneron_python
                ),
                kl720_port=81,
            )
        )


def test_runtime_registry_forwards_kl720_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from radiation_edge_ai.dna_fiber import (
        application as dnai_application,
    )

    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "assay_id": (
                "dnai-fiber-v3"
            ),
        },
    )

    kneron_python = (
        tmp_path
        / "kneron-python.exe"
    )

    calls: dict[str, object] = {}

    def fake_execute(
        manifest_path: Path,
        *,
        output_dir: Path,
        onnx_python: Optional[Path] = None,  # noqa: UP045
        kl720_python: Optional[Path] = None,  # noqa: UP045
        kl720_port: Optional[int] = None,  # noqa: UP045
        kl720_timeout_ms: int = 10000,
    ) -> dict[str, object]:
        calls["manifest"] = (
            manifest_path
        )
        calls["output_dir"] = (
            output_dir
        )
        calls["onnx_python"] = (
            onnx_python
        )
        calls["kl720_python"] = (
            kl720_python
        )
        calls["kl720_port"] = (
            kl720_port
        )
        calls["kl720_timeout_ms"] = (
            kl720_timeout_ms
        )

        return {
            "kind": (
                "assay_run_summary"
            ),
            "assay_id": (
                "dnai-fiber-v3"
            ),
            "verification": "PASS",
        }

    monkeypatch.setattr(
        dnai_application,
        "execute_dnai_assay_manifest",
        fake_execute,
    )

    result = (
        assay_runtime
        .execute_registered_assay_manifest(
            manifest,
            output_dir=(
                tmp_path / "out"
            ),
            kl720_python=(
                kneron_python
            ),
            kl720_port=81,
            kl720_timeout_ms=12345,
        )
    )

    assert result["verification"] == "PASS"

    assert calls == {
        "manifest": manifest,
        "output_dir": (
            tmp_path / "out"
        ),
        "onnx_python": None,
        "kl720_python": (
            kneron_python
        ),
        "kl720_port": 81,
        "kl720_timeout_ms": 12345,
    }


def test_cli_assay_run_forwards_kl720_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = (
        tmp_path
        / "manifest.json"
    )

    kneron_python = (
        tmp_path
        / "kneron-python.exe"
    )

    calls: dict[str, object] = {}

    def fake_execute(
        manifest_path: Path,
        *,
        output_dir: Path,
        onnx_python: Optional[Path] = None,  # noqa: UP045
        kl720_python: Optional[Path] = None,  # noqa: UP045
        kl720_port: Optional[int] = None,  # noqa: UP045
        kl720_timeout_ms: int = 10000,
    ) -> dict[str, object]:
        calls["manifest"] = (
            manifest_path
        )
        calls["output_dir"] = (
            output_dir
        )
        calls["onnx_python"] = (
            onnx_python
        )
        calls["kl720_python"] = (
            kl720_python
        )
        calls["kl720_port"] = (
            kl720_port
        )
        calls["kl720_timeout_ms"] = (
            kl720_timeout_ms
        )

        return {
            "kind": (
                "assay_run_summary"
            ),
            "assay_id": (
                "dnai-fiber-v3"
            ),
            "status": "complete",
        }

    monkeypatch.setattr(
        cli,
        "execute_registered_assay_manifest",
        fake_execute,
    )

    exit_code = cli.main(
        [
            "assay-run",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(
                tmp_path / "out"
            ),
            "--kl720-python",
            str(kneron_python),
            "--kl720-port",
            "81",
            "--kl720-timeout-ms",
            "12345",
            "--json",
        ]
    )

    assert exit_code == 0

    output = (
        capsys.readouterr().out
    )

    parsed = json.loads(
        output
    )

    assert (
        parsed["assay_id"]
        == "dnai-fiber-v3"
    )

    assert calls == {
        "manifest": manifest,
        "output_dir": (
            tmp_path / "out"
        ),
        "onnx_python": None,
        "kl720_python": (
            kneron_python
        ),
        "kl720_port": 81,
        "kl720_timeout_ms": 12345,
    }
