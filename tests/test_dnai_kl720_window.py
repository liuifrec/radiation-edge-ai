from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from radiation_edge_ai.control import (
    ControlPlaneError,
    sha256_file,
)
from radiation_edge_ai.dna_fiber import (
    kl720_window,
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


def _make_tensor(
    path: Path,
) -> Path:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        path,
        np.zeros(
            (1, 3, 512, 512),
            dtype=np.float32,
        ),
        allow_pickle=False,
    )

    return path


def test_verified_window_composes_generic_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = _make_tensor(
        tmp_path / "window.npy"
    )

    model = (
        tmp_path / "model.nef"
    )

    model.write_bytes(
        b"synthetic-nef"
    )

    model_sha = sha256_file(
        model
    )

    monkeypatch.setattr(
        kl720_window,
        "EXPECTED_NEF_SHA256",
        model_sha,
    )

    output = _make_tensor(
        tmp_path / "raw_output.npy"
    )

    run_record = _write_json(
        tmp_path / "run_record.json",
        {
            "backend": "kl720",
            "assay_id": (
                "dnai-fiber-v3"
            ),
            "source_artifacts": {
                "input": {
                    "sha256": (
                        sha256_file(
                            tensor
                        )
                    ),
                },
                "model": {
                    "sha256": (
                        model_sha
                    ),
                },
            },
            "raw_output": {
                "path": str(
                    output.resolve()
                ),
                "sha256": (
                    sha256_file(
                        output
                    )
                ),
                "size_bytes": (
                    output.stat().st_size
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
            },
            "timing": {
                "inference_send_receive_seconds": (
                    0.125
                ),
            },
        },
    )

    plan = _write_json(
        tmp_path / "run_plan.json",
        {},
    )

    calls: dict[str, object] = {}

    def fake_create(
        **kwargs: object,
    ) -> Path:
        calls["create"] = (
            kwargs
        )
        return plan

    def fake_execute(
        plan_path: Path,
        **kwargs: object,
    ) -> Path:
        calls["execute_plan"] = (
            plan_path
        )
        calls["execute"] = (
            kwargs
        )
        return run_record

    monkeypatch.setattr(
        kl720_window,
        "create_run_plan",
        fake_create,
    )

    monkeypatch.setattr(
        kl720_window,
        "execute_run_plan",
        fake_execute,
    )

    monkeypatch.setattr(
        kl720_window,
        "verify_run_record",
        lambda *_args, **_kwargs: {
            "ok": True
        },
    )

    kneron_python = (
        tmp_path / "python.exe"
    )

    result = (
        kl720_window
        .execute_verified_kl720_window(
            tensor_path=tensor,
            model_path=model,
            sample_id="sample-1",
            pixel_size_um=0.26,
            channel_semantics=(
                "synthetic red/green"
            ),
            image_index=0,
            window_index=4,
            y=256,
            x=256,
            output_dir=(
                tmp_path / "physical"
            ),
            kl720_python=(
                kneron_python
            ),
            kl720_port=81,
            kl720_timeout_ms=12345,
        )
    )

    assert result[
        "window_index"
    ] == 4

    assert result["y"] == 256
    assert result["x"] == 256

    assert result[
        "inference_ms"
    ] == pytest.approx(
        125.0
    )

    create = calls["create"]
    assert isinstance(
        create,
        dict,
    )

    assert create[
        "assay_id"
    ] == "dnai-fiber-v3"

    assert create[
        "backend"
    ] == "kl720"

    execute = calls["execute"]
    assert isinstance(
        execute,
        dict,
    )

    assert execute[
        "kl720_python"
    ] == kneron_python

    assert execute[
        "kl720_port"
    ] == 81

    assert execute[
        "kl720_timeout_ms"
    ] == 12345


def test_verified_window_rejects_wrong_model(
    tmp_path: Path,
) -> None:
    tensor = _make_tensor(
        tmp_path / "window.npy"
    )

    model = (
        tmp_path / "model.nef"
    )

    model.write_bytes(
        b"wrong-nef"
    )

    with pytest.raises(
        ControlPlaneError,
        match="frozen NEF",
    ):
        (
            kl720_window
            .execute_verified_kl720_window(
                tensor_path=tensor,
                model_path=model,
                sample_id="sample-1",
                pixel_size_um=0.26,
                channel_semantics="red/green",
                image_index=0,
                window_index=0,
                y=0,
                x=0,
                output_dir=tmp_path,
            )
        )


def test_verified_window_rejects_wrong_tensor_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = (
        tmp_path / "window.npy"
    )

    np.save(
        tensor,
        np.zeros(
            (1, 3, 256, 256),
            dtype=np.float32,
        ),
        allow_pickle=False,
    )

    model = (
        tmp_path / "model.nef"
    )

    model.write_bytes(
        b"synthetic-nef"
    )

    monkeypatch.setattr(
        kl720_window,
        "EXPECTED_NEF_SHA256",
        sha256_file(
            model
        ),
    )

    with pytest.raises(
        ControlPlaneError,
        match="512",
    ):
        (
            kl720_window
            .execute_verified_kl720_window(
                tensor_path=tensor,
                model_path=model,
                sample_id="sample-1",
                pixel_size_um=0.26,
                channel_semantics="red/green",
                image_index=0,
                window_index=0,
                y=0,
                x=0,
                output_dir=tmp_path,
            )
        )
