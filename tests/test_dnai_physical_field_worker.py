from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from radiation_edge_ai.control import (
    sha256_file,
)
from radiation_edge_ai.dna_fiber import (
    _physical_field_worker as worker,
)


class _FakeTensor:
    def __init__(
        self,
        value: np.ndarray,
    ) -> None:
        self.value = value

    def cpu(
        self,
    ) -> _FakeTensor:
        return self

    def numpy(
        self,
    ) -> np.ndarray:
        return self.value


class _FakeFrame:
    def to_csv(
        self,
        path: Path,
        *,
        index: bool,
    ) -> None:
        assert index is False

        Path(path).write_text(
            "Image,Length,Ratio\n"
            "sample-1,10.0,1.0\n",
            encoding="utf-8",
        )


class _FakeFiber:
    ratio = 1.0
    length = 10.0


class _FakeFibers:
    def __init__(
        self,
        n: int,
    ) -> None:
        self.items = [
            _FakeFiber()
            for _ in range(n)
        ]

    def __len__(
        self,
    ) -> int:
        return len(
            self.items
        )

    def __iter__(
        self,
    ):
        return iter(
            self.items
        )

    def valid_copy(
        self,
    ) -> _FakeFibers:
        return _FakeFibers(
            len(
                self.items
            )
        )

    def to_df(
        self,
        *,
        pixel_size: float,
        img_name: str,
    ) -> _FakeFrame:
        assert pixel_size == 0.26
        assert img_name == "sample-1"
        return _FakeFrame()


class _FakeCv2:
    @staticmethod
    def imwrite(
        path: str,
        value: np.ndarray,
    ) -> bool:
        assert value.shape == (
            1024,
            1024,
        )

        Path(path).write_bytes(
            b"synthetic-png"
        )

        return True


class _FakeTorch:
    float32 = object()
    __version__ = "synthetic"

    @staticmethod
    def from_numpy(
        value: np.ndarray,
    ) -> np.ndarray:
        return value


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


def _make_validation(
    tmp_path: Path,
) -> tuple[
    Path,
    list[Path],
]:
    records = []
    tensors = []

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

    root = (
        tmp_path / "validation"
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, (y, x) in enumerate(
        coords
    ):
        tensor = (
            root
            / f"window_{index}.npy"
        )

        np.save(
            tensor,
            np.zeros(
                (
                    1,
                    3,
                    512,
                    512,
                ),
                dtype=np.float32,
            ),
            allow_pickle=False,
        )

        tensors.append(
            tensor
        )

        records.append(
            {
                "image_index": 0,
                "window_index": (
                    index
                ),
                "sample_id": (
                    "sample-1"
                ),
                "key": "sample-1",
                "y": y,
                "x": x,
                "tensor": (
                    tensor.name
                ),
            }
        )

    manifest = {
        "dnai_commit": (
            worker.DNAI_COMMIT
        ),
        "tile": [
            1,
            3,
            512,
            512,
        ],
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
        "records": records,
    }

    manifest_path = (
        root / "manifest.json"
    )

    _write_json(
        manifest_path,
        manifest,
    )

    return (
        manifest_path,
        tensors,
    )


def _fake_science(
) -> dict[str, object]:
    def importance(
        *_args: object,
        **_kwargs: object,
    ) -> _FakeTensor:
        return _FakeTensor(
            np.ones(
                (
                    512,
                    512,
                ),
                dtype=np.float32,
            )
        )

    def segmentation(
        _value: object,
    ) -> np.ndarray:
        return np.zeros(
            (
                1024,
                1024,
            ),
            dtype=np.uint8,
        )

    def refine(
        _value: np.ndarray,
    ) -> _FakeFibers:
        return _FakeFibers(
            2
        )

    return {
        "cv2": _FakeCv2(),
        "torch": _FakeTorch(),
        "compute_importance_map": (
            importance
        ),
        "probas_to_segmentation": (
            segmentation
        ),
        "refine_segmentation": (
            refine
        ),
    }


def test_physical_worker_assembles_df2_without_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        manifest,
        _tensors,
    ) = _make_validation(
        tmp_path
    )

    model = (
        tmp_path / "model.nef"
    )

    model.write_bytes(
        b"synthetic-nef"
    )

    monkeypatch.setattr(
        worker,
        "EXPECTED_NEF_SHA256",
        sha256_file(
            model
        ),
    )

    monkeypatch.setattr(
        worker,
        "_load_science_runtime",
        _fake_science,
    )

    calls = []

    def fake_window(
        *,
        tensor_path: Path,
        window_index: int,
        y: int,
        x: int,
        output_dir: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        calls.append(
            window_index
        )

        run_root = (
            output_dir
            / f"run_{window_index}"
        )

        run_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = (
            run_root
            / "raw_output.npy"
        )

        logits = np.zeros(
            (
                1,
                3,
                512,
                512,
            ),
            dtype=np.float32,
        )

        logits[
            :,
            window_index % 3,
            :,
            :,
        ] = 1.0

        np.save(
            output,
            logits,
            allow_pickle=False,
        )

        run_record = (
            run_root
            / "run_record.json"
        )

        _write_json(
            run_record,
            {
                "runtime": {
                    "selected_usb_port": 81,
                    "product_id": 0x720,
                    "firmware": (
                        "KDP2 Comp/F"
                    ),
                    "model_id": 32769,
                }
            },
        )

        return {
            "window_index": (
                window_index
            ),
            "y": y,
            "x": x,
            "input": {
                "path": str(
                    tensor_path
                ),
                "sha256": (
                    sha256_file(
                        tensor_path
                    )
                ),
                "size_bytes": (
                    tensor_path.stat().st_size
                ),
            },
            "run_record": {
                "path": str(
                    run_record
                ),
                "sha256": (
                    sha256_file(
                        run_record
                    )
                ),
                "size_bytes": (
                    run_record.stat().st_size
                ),
            },
            "output": {
                "path": str(
                    output
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
            "inference_ms": (
                100.0
                + window_index
            ),
        }

    monkeypatch.setattr(
        worker,
        "execute_verified_kl720_window",
        fake_window,
    )

    monkeypatch.setattr(
        worker,
        "verify_dnai_fiber_physical_field_record",
        lambda *_args, **_kwargs: {
            "ok": True
        },
    )

    monkeypatch.setattr(
        worker,
        "_package_version",
        lambda name: (
            "synthetic-"
            + name
        ),
    )

    record_path = (
        worker.execute_physical_field(
            model_path=model,
            validation_manifest=(
                manifest
            ),
            image_index=0,
            output_root=(
                tmp_path / "out"
            ),
            kl720_port=81,
        )
    )

    assert calls == list(
        range(9)
    )

    result = json.loads(
        record_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        result["schema_version"]
        == 2
    )

    assert (
        result["field_id"]
        .startswith(
            "df2-"
        )
    )

    assert (
        result["identity"][
            "backend"
        ]
        == "kl720"
    )

    assert (
        result["counts"][
            "n_windows"
        ]
        == 9
    )

    assert (
        result["counts"][
            "n_fibers_valid"
        ]
        == 2
    )

    assert (
        result["runtime"][
            "selected_usb_ports"
        ]
        == [81]
    )

    assert (
        result["runtime"][
            "product_ids"
        ]
        == [0x720]
    )

    assert (
        result[
            "scientific_scope"
        ][
            "biological_fidelity_evaluated"
        ]
        is False
    )

    assert (
        len(
            result[
                "window_outputs"
            ]
        )
        == 9
    )


def test_physical_worker_rejects_changed_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        manifest,
        _tensors,
    ) = _make_validation(
        tmp_path
    )

    value = json.loads(
        manifest.read_text(
            encoding="utf-8"
        )
    )

    value["records"][4][
        "x"
    ] = 300

    _write_json(
        manifest,
        value,
    )

    model = (
        tmp_path / "model.nef"
    )

    model.write_bytes(
        b"synthetic-nef"
    )

    monkeypatch.setattr(
        worker,
        "EXPECTED_NEF_SHA256",
        sha256_file(
            model
        ),
    )

    with pytest.raises(
        Exception,
        match="3x3",
    ):
        worker.execute_physical_field(
            model_path=model,
            validation_manifest=(
                manifest
            ),
            image_index=0,
            output_root=(
                tmp_path / "out"
            ),
        )
