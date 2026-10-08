from __future__ import annotations

import json
from pathlib import Path

import pytest

from radiation_edge_ai.control import (
    sha256_json,
)
from radiation_edge_ai.dna_fiber import (
    field_record_v2,
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


def _artifact(
    path: Path,
    *,
    sha: str,
    size: int = 1,
) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha,
        "size_bytes": size,
    }


def _record(
    tmp_path: Path,
) -> tuple[
    Path,
    dict[str, object],
]:
    coords = [
        (0, 0),
        (0, 256),
        (0, 512),
        (256, 0),
        (256, 256),
        (256, 512),
        (512, 0),
        (512, 256),
        (512, 512),
    ]

    windows = []

    outputs = []

    for index, (y, x) in enumerate(
        coords
    ):
        tensor_sha = (
            f"{index + 1:064x}"
        )

        output_sha = (
            f"{index + 101:064x}"
        )

        run_path = (
            tmp_path
            / f"run_{index}.json"
        )

        run = {
            "backend": "kl720",
            "assay_id": (
                "dnai-fiber-v3"
            ),
            "source_artifacts": {
                "input": {
                    "sha256": (
                        tensor_sha
                    ),
                },
                "model": {
                    "sha256": (
                        field_record_v2
                        .EXPECTED_NEF_SHA256
                    ),
                },
            },
            "raw_output": {
                "sha256": (
                    output_sha
                ),
                "dtype": "float32",
                "shape": [
                    1,
                    3,
                    512,
                    512,
                ],
            },
            "runtime": {
                "selected_usb_port": 81,
                "product_id": 0x720,
                "firmware": (
                    "KDP2 Comp/F"
                ),
                "model_id": 32769,
            },
        }

        _write_json(
            run_path,
            run,
        )

        windows.append(
            {
                "window_index": (
                    index
                ),
                "y": y,
                "x": x,
                "tensor_sha256": (
                    tensor_sha
                ),
                "tensor_size_bytes": (
                    1000 + index
                ),
            }
        )

        outputs.append(
            {
                "window_index": (
                    index
                ),
                "y": y,
                "x": x,
                "input": _artifact(
                    tmp_path
                    / f"input_{index}.npy",
                    sha=tensor_sha,
                    size=(
                        1000 + index
                    ),
                ),
                "run_record": _artifact(
                    run_path,
                    sha=(
                        f"{index + 201:064x}"
                    ),
                ),
                "output": _artifact(
                    tmp_path
                    / f"output_{index}.npy",
                    sha=output_sha,
                ),
                "inference_ms": 100.0,
            }
        )

    validation_sha = "e" * 64

    identity = {
        "schema_version": 2,
        "record_type": (
            "dnai_fiber_field_record"
        ),
        "assay_id": (
            "dnai-fiber-v3"
        ),
        "application_adapter": (
            "dnai-tile-stitch-object-v1"
        ),
        "backend": "kl720",
        "dnai_commit": (
            field_record_v2
            .DNAI_COMMIT
        ),
        "model_sha256": (
            field_record_v2
            .EXPECTED_NEF_SHA256
        ),
        "validation_manifest_sha256": (
            validation_sha
        ),
        "image_index": 0,
        "sample_id": "sample-1",
        "key": "sample-1",
        "input_semantics": (
            "frozen preprocessed and ImageNet-normalized "
            "DNAi deployment tensors"
        ),
        "deployment_policy": {
            "tile": [
                1,
                3,
                512,
                512,
            ],
            "overlap": 0.50,
            "stride": 256,
            "blend": "gaussian",
            "sigma_scale": 0.125,
            "pixel_size_um": 0.26,
        },
        "windows": windows,
    }

    fingerprint = sha256_json(
        identity
    )

    record = {
        "schema_version": 2,
        "record_type": (
            "dnai_fiber_field_record"
        ),
        "status": "complete",
        "field_id": (
            "df2-"
            + fingerprint[:16]
        ),
        "field_fingerprint_sha256": (
            fingerprint
        ),
        "identity": identity,
        "source_artifacts": {
            "model": _artifact(
                tmp_path / "model.nef",
                sha=(
                    field_record_v2
                    .EXPECTED_NEF_SHA256
                ),
            ),
            "validation_manifest": (
                _artifact(
                    tmp_path
                    / "validation.json",
                    sha=validation_sha,
                )
            ),
        },
        "runtime": {
            "execution_backend": (
                "kl720"
            ),
            "execution_adapter": (
                "generic-run-plan-v1"
            ),
            "selected_usb_ports": [
                81
            ],
            "product_ids": [
                0x720
            ],
            "firmwares": [
                "KDP2 Comp/F"
            ],
            "model_ids": [
                32769
            ],
        },
        "counts": {
            "n_windows": 9,
            "n_fibers_all": 18,
            "n_fibers_valid": 14,
        },
        "measurements": {
            "mean_valid_ratio": 1.0,
            "median_valid_ratio": 1.0,
            "mean_valid_length_um": 10.0,
            "total_valid_length_um": 140.0,
        },
        "window_outputs": outputs,
        "artifacts": {
            "stitched_probabilities": (
                _artifact(
                    tmp_path
                    / "stitched.npz",
                    sha="a" * 64,
                )
            ),
            "segmentation": (
                _artifact(
                    tmp_path
                    / "segmentation.png",
                    sha="b" * 64,
                )
            ),
            "valid_fibers_csv": (
                _artifact(
                    tmp_path
                    / "valid_fibers.csv",
                    sha="c" * 64,
                )
            ),
        },
        "scientific_scope": (
            dict(
                field_record_v2
                .EXPECTED_SCOPE
            )
        ),
    }

    path = _write_json(
        tmp_path / "field.json",
        record,
    )

    return path, record


def _permit_synthetic_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        field_record_v2,
        "_artifact_ok",
        lambda *_args, **_kwargs: True,
    )

    monkeypatch.setattr(
        field_record_v2,
        "verify_run_record",
        lambda *_args, **_kwargs: {
            "ok": True
        },
    )


def test_physical_field_record_contract_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = _record(
        tmp_path
    )

    _permit_synthetic_artifacts(
        monkeypatch
    )

    report = (
        field_record_v2
        .verify_dnai_fiber_physical_field_record(
            path,
            check_artifacts=True,
        )
    )

    assert report["ok"] is True
    assert (
        report["identity_semantics_ok"]
        is True
    )
    assert (
        report["window_artifacts_ok"]
        is True
    )
    assert report["runtime_ok"] is True
    assert (
        report["scientific_scope_ok"]
        is True
    )


def test_physical_field_rejects_cpu_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, record = _record(
        tmp_path
    )

    identity = record["identity"]
    assert isinstance(
        identity,
        dict,
    )

    identity["backend"] = (
        "cpu-onnx"
    )

    fingerprint = sha256_json(
        identity
    )

    record[
        "field_fingerprint_sha256"
    ] = fingerprint

    record["field_id"] = (
        "df2-"
        + fingerprint[:16]
    )

    _write_json(
        path,
        record,
    )

    _permit_synthetic_artifacts(
        monkeypatch
    )

    report = (
        field_record_v2
        .verify_dnai_fiber_physical_field_record(
            path,
        )
    )

    assert report["ok"] is False
    assert (
        report["identity_semantics_ok"]
        is False
    )


def test_physical_field_rejects_wrong_run_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = _record(
        tmp_path
    )

    run_path = (
        tmp_path / "run_4.json"
    )

    run = json.loads(
        run_path.read_text(
            encoding="utf-8"
        )
    )

    run["backend"] = (
        "cpu-onnx"
    )

    _write_json(
        run_path,
        run,
    )

    _permit_synthetic_artifacts(
        monkeypatch
    )

    report = (
        field_record_v2
        .verify_dnai_fiber_physical_field_record(
            path,
        )
    )

    assert report["ok"] is False
    assert (
        report["window_artifacts_ok"]
        is False
    )


def test_physical_field_rejects_biological_fidelity_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, record = _record(
        tmp_path
    )

    scope = record[
        "scientific_scope"
    ]

    assert isinstance(scope, dict)

    scope[
        "biological_fidelity_evaluated"
    ] = True

    _write_json(
        path,
        record,
    )

    _permit_synthetic_artifacts(
        monkeypatch
    )

    report = (
        field_record_v2
        .verify_dnai_fiber_physical_field_record(
            path,
        )
    )

    assert report["ok"] is False
    assert (
        report["scientific_scope_ok"]
        is False
    )
