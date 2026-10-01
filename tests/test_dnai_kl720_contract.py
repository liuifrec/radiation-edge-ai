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


def test_kl720_application_contract_fails_closed_before_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = (
        tmp_path
        / "manifest.json"
    )

    value = {
        "backend": "kl720",
    }

    monkeypatch.setattr(
        application,
        "validate_dnai_application_manifest",
        lambda _path: value,
    )

    kneron_python = (
        tmp_path
        / "python.exe"
    )

    kneron_python.write_bytes(
        b"x"
    )

    with pytest.raises(
        ControlPlaneError,
        match=(
            "physical field execution "
            "is not enabled yet"
        ),
    ):
        application.execute_dnai_assay_manifest(
            manifest_path,
            output_dir=(
                tmp_path / "out"
            ),
            kl720_python=(
                kneron_python
            ),
            kl720_port=81,
            kl720_timeout_ms=10000,
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
