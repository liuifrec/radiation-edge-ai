"""Execute one frozen DNAi field through physical KL720 inference.

This v0.11 worker is additive. It does not modify or replace the frozen
v0.9/v0.10 floating-CPU worker.

Nine frozen, already-preprocessed 512x512 deployment tensors are executed
through the generic verified KL720 run-plan layer. The returned logits are
then processed using the same softmax, Gaussian overlap stitching,
segmentation, and fiber reconstruction semantics as the legacy FP512 path.

This is an application-integration execution path. It does not read human
annotations or the FP1024 biological reference and does not, by itself,
evaluate biological deployment fidelity.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from radiation_edge_ai.control import (
    ControlPlaneError,
    load_json_object,
    sha256_file,
    sha256_json,
)
from radiation_edge_ai.dna_fiber.field_record_v2 import (
    APPLICATION_ADAPTER,
    ASSAY_ID,
    BACKEND,
    DNAI_COMMIT,
    EXPECTED_NEF_SHA256,
    EXPECTED_SCOPE,
    verify_dnai_fiber_physical_field_record,
)
from radiation_edge_ai.dna_fiber.kl720_window import (
    execute_verified_kl720_window,
)

TILE = 512
OVERLAP = 0.50
STRIDE = 256
SIGMA_SCALE = 0.125
PIXEL_SIZE_UM = 0.26

EXPECTED_TENSOR = (
    1,
    3,
    TILE,
    TILE,
)

EXPECTED_FULL_IMAGE = (
    1,
    3,
    1024,
    1024,
)

EXPECTED_COORDS = {
    (0, 0),
    (0, 256),
    (0, 512),
    (256, 0),
    (256, 256),
    (256, 512),
    (512, 0),
    (512, 256),
    (512, 512),
}


def _artifact_record(
    path: Path,
) -> dict[str, object]:
    resolved = (
        path
        .expanduser()
        .resolve()
    )

    return {
        "path": str(resolved),
        "sha256": sha256_file(
            resolved
        ),
        "size_bytes": (
            resolved.stat().st_size
        ),
    }


def _atomic_write_json(
    path: Path,
    value: dict[str, object],
) -> None:
    temp = path.with_name(
        path.name + ".tmp"
    )

    with temp.open(
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

    temp.replace(path)


def _atomic_save_npz(
    path: Path,
    **values: np.ndarray,
) -> None:
    temp = path.with_name(
        path.name + ".tmp"
    )

    with temp.open(
        "wb",
    ) as handle:
        np.savez_compressed(
            handle,
            **values,
        )

    temp.replace(path)


def _finite_mean(
    values: list[float],
) -> Optional[float]:  # noqa: UP045
    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(
            array
        )
    ]

    if array.size == 0:
        return None

    return float(
        array.mean()
    )


def _finite_median(
    values: list[float],
) -> Optional[float]:  # noqa: UP045
    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(
            array
        )
    ]

    if array.size == 0:
        return None

    return float(
        np.median(
            array
        )
    )


def _softmax_np(
    array: np.ndarray,
) -> np.ndarray:
    work = array.astype(
        np.float64,
        copy=False,
    )

    shifted = (
        work
        - np.max(
            work,
            axis=1,
            keepdims=True,
        )
    )

    exponential = np.exp(
        shifted
    )

    probabilities = (
        exponential
        / np.sum(
            exponential,
            axis=1,
            keepdims=True,
        )
    )

    return probabilities.astype(
        np.float32
    )


def _package_version(
    name: str,
) -> Optional[str]:  # noqa: UP045
    try:
        return importlib.metadata.version(
            name
        )
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_science_runtime(
) -> dict[str, Any]:
    """Load heavy DNAi scientific dependencies only in the worker runtime."""

    import cv2
    import torch
    from dnafiber.inference import (
        probas_to_segmentation,
    )
    from dnafiber.postprocess import (
        refine_segmentation,
    )
    from monai.data.utils import (
        compute_importance_map,
    )

    return {
        "cv2": cv2,
        "torch": torch,
        "probas_to_segmentation": (
            probas_to_segmentation
        ),
        "refine_segmentation": (
            refine_segmentation
        ),
        "compute_importance_map": (
            compute_importance_map
        ),
    }


def _resolve_records(
    manifest_path: Path,
    *,
    image_index: int,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[Path],
    str,
    str,
]:
    manifest = load_json_object(
        manifest_path
    )

    if (
        manifest.get("dnai_commit")
        != DNAI_COMMIT
    ):
        raise ControlPlaneError(
            "DNAi commit mismatch"
        )

    if tuple(
        manifest.get(
            "tile",
            (),
        )
    ) != EXPECTED_TENSOR:
        raise ControlPlaneError(
            "Unexpected DNAi tile policy"
        )

    if tuple(
        manifest.get(
            "full_image_shape",
            (),
        )
    ) != EXPECTED_FULL_IMAGE:
        raise ControlPlaneError(
            "Unexpected DNAi full-image shape"
        )

    if (
        float(
            manifest.get(
                "overlap",
                -1.0,
            )
        )
        != OVERLAP
    ):
        raise ControlPlaneError(
            "Unexpected DNAi overlap policy"
        )

    if (
        int(
            manifest.get(
                "stride",
                -1,
            )
        )
        != STRIDE
    ):
        raise ControlPlaneError(
            "Unexpected DNAi stride policy"
        )

    if (
        manifest.get(
            "blend"
        )
        != "gaussian"
    ):
        raise ControlPlaneError(
            "Unexpected DNAi stitching policy"
        )

    if (
        int(
            manifest.get(
                "windows_per_image",
                -1,
            )
        )
        != 9
    ):
        raise ControlPlaneError(
            "Unexpected DNAi windows-per-image policy"
        )

    raw_records = manifest.get(
        "records"
    )

    if not isinstance(
        raw_records,
        list,
    ):
        raise ControlPlaneError(
            "DNAi validation records are malformed"
        )

    records = [
        record
        for record in raw_records
        if (
            isinstance(
                record,
                dict,
            )
            and int(
                record.get(
                    "image_index",
                    -1,
                )
            )
            == image_index
        )
    ]

    if len(records) != 9:
        raise ControlPlaneError(
            "Expected exactly 9 DNAi windows "
            f"for image {image_index}; got {len(records)}"
        )

    records.sort(
        key=lambda item: int(
            item["window_index"]
        )
    )

    indices = [
        int(
            record[
                "window_index"
            ]
        )
        for record in records
    ]

    if indices != list(
        range(9)
    ):
        raise ControlPlaneError(
            "DNAi window indices are not 0..8"
        )

    sample_ids = {
        str(
            record["sample_id"]
        )
        for record in records
    }

    keys = {
        str(
            record["key"]
        )
        for record in records
    }

    if len(sample_ids) != 1:
        raise ControlPlaneError(
            "Multiple DNAi sample IDs in one field"
        )

    if len(keys) != 1:
        raise ControlPlaneError(
            "Multiple DNAi keys in one field"
        )

    sample_id = next(
        iter(sample_ids)
    )

    key = next(
        iter(keys)
    )

    coords = {
        (
            int(
                record["y"]
            ),
            int(
                record["x"]
            ),
        )
        for record in records
    }

    if coords != EXPECTED_COORDS:
        raise ControlPlaneError(
            "Frozen DNAi 3x3 deployment grid changed"
        )

    validation_root = (
        manifest_path.parent
    ).resolve()

    tensor_paths: list[Path] = []

    for record in records:
        tensor_value = record.get(
            "tensor"
        )

        if (
            not isinstance(
                tensor_value,
                str,
            )
            or not tensor_value
        ):
            raise ControlPlaneError(
                "DNAi validation tensor path is malformed"
            )

        tensor_path = (
            validation_root
            / Path(
                tensor_value
            )
        ).resolve()

        try:
            tensor_path.relative_to(
                validation_root
            )
        except ValueError as exc:
            raise ControlPlaneError(
                "DNAi tensor path escapes validation root"
            ) from exc

        if not tensor_path.is_file():
            raise ControlPlaneError(
                f"DNAi tensor missing: {tensor_path}"
            )

        tensor = np.load(
            tensor_path,
            allow_pickle=False,
        )

        if (
            tensor.shape
            != EXPECTED_TENSOR
            or tensor.dtype
            != np.float32
            or not np.isfinite(
                tensor
            ).all()
        ):
            raise ControlPlaneError(
                "Frozen DNAi window must be finite "
                "float32 [1,3,512,512]"
            )

        tensor_paths.append(
            tensor_path
        )

    return (
        manifest,
        records,
        tensor_paths,
        sample_id,
        key,
    )


def _build_identity(
    *,
    model_path: Path,
    validation_manifest: Path,
    image_index: int,
    records: list[dict[str, object]],
    tensor_paths: list[Path],
    sample_id: str,
    key: str,
) -> dict[str, object]:
    windows = []

    for record, tensor_path in zip(
        records,
        tensor_paths,
    ):
        windows.append(
            {
                "window_index": int(
                    record[
                        "window_index"
                    ]
                ),
                "y": int(
                    record["y"]
                ),
                "x": int(
                    record["x"]
                ),
                "tensor_sha256": (
                    sha256_file(
                        tensor_path
                    )
                ),
                "tensor_size_bytes": (
                    tensor_path.stat().st_size
                ),
            }
        )

    return {
        "schema_version": 2,
        "record_type": (
            "dnai_fiber_field_record"
        ),
        "assay_id": ASSAY_ID,
        "application_adapter": (
            APPLICATION_ADAPTER
        ),
        "backend": BACKEND,
        "dnai_commit": DNAI_COMMIT,
        "model_sha256": (
            sha256_file(
                model_path
            )
        ),
        "validation_manifest_sha256": (
            sha256_file(
                validation_manifest
            )
        ),
        "image_index": (
            image_index
        ),
        "sample_id": (
            sample_id
        ),
        "key": key,
        "input_semantics": (
            "frozen preprocessed and ImageNet-normalized "
            "DNAi deployment tensors"
        ),
        "deployment_policy": {
            "tile": [
                1,
                3,
                TILE,
                TILE,
            ],
            "overlap": (
                OVERLAP
            ),
            "stride": (
                STRIDE
            ),
            "blend": (
                "gaussian"
            ),
            "sigma_scale": (
                SIGMA_SCALE
            ),
            "pixel_size_um": (
                PIXEL_SIZE_UM
            ),
        },
        "windows": windows,
    }


def execute_physical_field(
    *,
    model_path: Path,
    validation_manifest: Path,
    image_index: int,
    output_root: Path,
    kl720_python: Optional[Path] = None,  # noqa: UP045
    kl720_port: Optional[int] = None,  # noqa: UP045
    kl720_timeout_ms: int = 10000,
) -> Path:
    """Execute one complete frozen DNAi field through physical KL720."""

    model = (
        model_path
        .expanduser()
        .resolve()
    )

    manifest_path = (
        validation_manifest
        .expanduser()
        .resolve()
    )

    root = (
        output_root
        .expanduser()
        .resolve()
    )

    if not model.is_file():
        raise ControlPlaneError(
            f"DNAi KL720 model missing: {model}"
        )

    if model.suffix.lower() != ".nef":
        raise ControlPlaneError(
            "DNAi physical field requires a .nef model"
        )

    if (
        sha256_file(
            model
        )
        != EXPECTED_NEF_SHA256
    ):
        raise ControlPlaneError(
            "DNAi physical field model is not the frozen NEF"
        )

    if not manifest_path.is_file():
        raise ControlPlaneError(
            "DNAi validation manifest is missing"
        )

    if image_index < 0:
        raise ControlPlaneError(
            "DNAi image index must be non-negative"
        )

    if kl720_timeout_ms <= 0:
        raise ControlPlaneError(
            "KL720 timeout must be positive"
        )

    (
        _manifest,
        records,
        tensor_paths,
        sample_id,
        key,
    ) = _resolve_records(
        manifest_path,
        image_index=image_index,
    )

    identity = _build_identity(
        model_path=model,
        validation_manifest=(
            manifest_path
        ),
        image_index=image_index,
        records=records,
        tensor_paths=(
            tensor_paths
        ),
        sample_id=sample_id,
        key=key,
    )

    fingerprint = sha256_json(
        identity
    )

    field_id = (
        "df2-"
        + fingerprint[:16]
    )

    field_root = (
        root / field_id
    )

    record_path = (
        field_root
        / "field_result.json"
    )

    if record_path.exists():
        report = (
            verify_dnai_fiber_physical_field_record(
                record_path,
                check_artifacts=True,
            )
        )

        if report.get(
            "ok"
        ):
            return record_path

        raise ControlPlaneError(
            "Existing DNAi physical field record "
            "failed verification"
        )

    if (
        field_root.exists()
        and any(
            field_root.iterdir()
        )
    ):
        raise ControlPlaneError(
            "Refusing non-empty incomplete "
            f"DNAi physical field directory: {field_root}"
        )

    science = (
        _load_science_runtime()
    )

    torch = science[
        "torch"
    ]

    compute_importance_map = (
        science[
            "compute_importance_map"
        ]
    )

    probas_to_segmentation = (
        science[
            "probas_to_segmentation"
        ]
    )

    refine_segmentation = (
        science[
            "refine_segmentation"
        ]
    )

    cv2 = science[
        "cv2"
    ]

    execution_root = (
        root
        / "_kl720_runs"
        / field_id
    )

    importance = (
        compute_importance_map(
            (
                TILE,
                TILE,
            ),
            mode="gaussian",
            sigma_scale=(
                SIGMA_SCALE
            ),
            device="cpu",
            dtype=torch.float32,
        )
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    importance = np.squeeze(
        importance
    )

    if importance.shape != (
        TILE,
        TILE,
    ):
        raise ControlPlaneError(
            "Unexpected Gaussian importance-map shape"
        )

    accumulator = np.zeros(
        EXPECTED_FULL_IMAGE,
        dtype=np.float64,
    )

    weight_sum = np.zeros(
        (
            1,
            1,
            EXPECTED_FULL_IMAGE[-2],
            EXPECTED_FULL_IMAGE[-1],
        ),
        dtype=np.float64,
    )

    window_outputs = []
    inference_ms: list[float] = []

    ports = set()
    product_ids = set()
    firmwares = set()
    model_ids = set()

    for record, tensor_path in zip(
        records,
        tensor_paths,
    ):
        window_index = int(
            record[
                "window_index"
            ]
        )

        y0 = int(
            record["y"]
        )

        x0 = int(
            record["x"]
        )

        binding = (
            execute_verified_kl720_window(
                tensor_path=tensor_path,
                model_path=model,
                sample_id=(
                    sample_id
                ),
                pixel_size_um=(
                    PIXEL_SIZE_UM
                ),
                channel_semantics=(
                    "frozen DNAi red/green "
                    "deployment tensor"
                ),
                image_index=(
                    image_index
                ),
                window_index=(
                    window_index
                ),
                y=y0,
                x=x0,
                output_dir=(
                    execution_root
                ),
                kl720_python=(
                    kl720_python
                ),
                kl720_port=(
                    kl720_port
                ),
                kl720_timeout_ms=(
                    kl720_timeout_ms
                ),
            )
        )

        output_artifact = (
            binding.get(
                "output"
            )
        )

        run_artifact = (
            binding.get(
                "run_record"
            )
        )

        if (
            not isinstance(
                output_artifact,
                dict,
            )
            or not isinstance(
                run_artifact,
                dict,
            )
        ):
            raise ControlPlaneError(
                "KL720 window adapter returned "
                "malformed artifact bindings"
            )

        logits_path = Path(
            str(
                output_artifact.get(
                    "path",
                    "",
                )
            )
        ).expanduser().resolve()

        logits = np.load(
            logits_path,
            allow_pickle=False,
        )

        if (
            logits.shape
            != EXPECTED_TENSOR
            or logits.dtype
            != np.float32
            or not np.isfinite(
                logits
            ).all()
        ):
            raise ControlPlaneError(
                "Verified KL720 logits changed "
                "the DNAi tensor contract"
            )

        probabilities = (
            _softmax_np(
                logits
            )
        )

        y1 = y0 + TILE
        x1 = x0 + TILE

        accumulator[
            :,
            :,
            y0:y1,
            x0:x1,
        ] += (
            probabilities
            * importance[
                None,
                None,
                :,
                :,
            ]
        )

        weight_sum[
            :,
            :,
            y0:y1,
            x0:x1,
        ] += importance[
            None,
            None,
            :,
            :,
        ]

        elapsed = binding.get(
            "inference_ms"
        )

        if (
            isinstance(
                elapsed,
                bool,
            )
            or not isinstance(
                elapsed,
                (int, float),
            )
            or not math.isfinite(
                float(
                    elapsed
                )
            )
            or float(
                elapsed
            )
            < 0.0
        ):
            raise ControlPlaneError(
                "KL720 window inference timing is invalid"
            )

        inference_ms.append(
            float(
                elapsed
            )
        )

        window_outputs.append(
            binding
        )

        run_path = Path(
            str(
                run_artifact.get(
                    "path",
                    "",
                )
            )
        ).expanduser().resolve()

        run = load_json_object(
            run_path
        )

        runtime = run.get(
            "runtime"
        )

        if not isinstance(
            runtime,
            dict,
        ):
            raise ControlPlaneError(
                "KL720 run runtime is malformed"
            )

        port = runtime.get(
            "selected_usb_port"
        )

        product_id = runtime.get(
            "product_id"
        )

        model_id = runtime.get(
            "model_id"
        )

        firmware = runtime.get(
            "firmware"
        )

        if (
            not isinstance(
                port,
                int,
            )
            or not isinstance(
                product_id,
                int,
            )
            or not isinstance(
                model_id,
                int,
            )
        ):
            raise ControlPlaneError(
                "KL720 runtime device identity is malformed"
            )

        ports.add(
            port
        )

        product_ids.add(
            product_id
        )

        model_ids.add(
            model_id
        )

        if firmware is not None:
            if not isinstance(
                firmware,
                str,
            ):
                raise ControlPlaneError(
                    "KL720 firmware identity is malformed"
                )

            firmwares.add(
                firmware
            )

    if np.any(
        weight_sum <= 0
    ):
        raise ControlPlaneError(
            "Gaussian stitching left uncovered pixels"
        )

    stitched = (
        accumulator
        / weight_sum
    ).astype(
        np.float32
    )

    post_started = (
        time.perf_counter()
    )

    segmentation = (
        probas_to_segmentation(
            torch.from_numpy(
                stitched
            )
        )
    )

    segmentation = np.asarray(
        segmentation,
        dtype=np.uint8,
    )

    if segmentation.shape != (
        1024,
        1024,
    ):
        raise ControlPlaneError(
            "Unexpected stitched segmentation shape"
        )

    fibers_all = (
        refine_segmentation(
            segmentation
        )
    )

    fibers_valid = (
        fibers_all.valid_copy()
    )

    postprocess_ms = (
        time.perf_counter()
        - post_started
    ) * 1000.0

    field_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    stitched_path = (
        field_root
        / "kl720_stitched_probabilities.npz"
    )

    _atomic_save_npz(
        stitched_path,
        probabilities=(
            stitched[0]
        ),
    )

    segmentation_path = (
        field_root
        / "kl720_segmentation_class_ids.png"
    )

    if not cv2.imwrite(
        str(
            segmentation_path
        ),
        segmentation,
    ):
        raise ControlPlaneError(
            "Could not write DNAi segmentation PNG"
        )

    fibers_csv = (
        field_root
        / "valid_fibers.csv"
    )

    fiber_df = (
        fibers_valid.to_df(
            pixel_size=(
                PIXEL_SIZE_UM
            ),
            img_name=key,
        )
    )

    fiber_df.to_csv(
        fibers_csv,
        index=False,
    )

    valid_list = list(
        fibers_valid
    )

    ratios = [
        float(
            fiber.ratio
        )
        for fiber in valid_list
        if math.isfinite(
            float(
                fiber.ratio
            )
        )
    ]

    lengths_um = [
        float(
            fiber.length
        )
        * PIXEL_SIZE_UM
        for fiber in valid_list
    ]

    runtime = {
        "execution_backend": (
            BACKEND
        ),
        "execution_adapter": (
            "generic-run-plan-v1"
        ),
        "selected_usb_ports": (
            sorted(
                ports
            )
        ),
        "product_ids": (
            sorted(
                product_ids
            )
        ),
        "firmwares": (
            sorted(
                firmwares
            )
        ),
        "model_ids": (
            sorted(
                model_ids
            )
        ),
        "scientific_python_executable": (
            sys.executable
        ),
        "scientific_python_version": (
            platform.python_version()
        ),
        "numpy_version": (
            np.__version__
        ),
        "torch_version": (
            getattr(
                torch,
                "__version__",
                None,
            )
        ),
        "dnafiber_version": (
            _package_version(
                "dnafiber"
            )
        ),
        "monai_version": (
            _package_version(
                "monai"
            )
        ),
    }

    result = {
        "schema_version": 2,
        "record_type": (
            "dnai_fiber_field_record"
        ),
        "status": "complete",
        "field_id": (
            field_id
        ),
        "field_fingerprint_sha256": (
            fingerprint
        ),
        "identity": identity,
        "source_artifacts": {
            "model": (
                _artifact_record(
                    model
                )
            ),
            "validation_manifest": (
                _artifact_record(
                    manifest_path
                )
            ),
        },
        "runtime": runtime,
        "counts": {
            "n_windows": 9,
            "n_fibers_all": (
                len(
                    fibers_all
                )
            ),
            "n_fibers_valid": (
                len(
                    valid_list
                )
            ),
        },
        "measurements": {
            "mean_valid_ratio": (
                _finite_mean(
                    ratios
                )
            ),
            "median_valid_ratio": (
                _finite_median(
                    ratios
                )
            ),
            "mean_valid_length_um": (
                _finite_mean(
                    lengths_um
                )
            ),
            "total_valid_length_um": (
                float(
                    np.sum(
                        lengths_um
                    )
                )
                if lengths_um
                else 0.0
            ),
        },
        "timing": {
            "window_inference_ms": (
                inference_ms
            ),
            "mean_window_inference_ms": (
                float(
                    np.mean(
                        inference_ms
                    )
                )
            ),
            "total_window_inference_ms": (
                float(
                    np.sum(
                        inference_ms
                    )
                )
            ),
            "postprocess_ms": (
                postprocess_ms
            ),
        },
        "window_outputs": (
            window_outputs
        ),
        "artifacts": {
            "stitched_probabilities": (
                _artifact_record(
                    stitched_path
                )
            ),
            "segmentation": (
                _artifact_record(
                    segmentation_path
                )
            ),
            "valid_fibers_csv": (
                _artifact_record(
                    fibers_csv
                )
            ),
        },
        "scientific_scope": (
            dict(
                EXPECTED_SCOPE
            )
        ),
        "interpretation": (
            "Physical KL720 application-integration field result. "
            "Nine frozen deployment windows were executed through "
            "verified generic KL720 run records and passed through "
            "the frozen DNAi stitching and fiber-reconstruction "
            "operations. This record does not, by itself, establish "
            "biological deployment fidelity."
        ),
    }

    _atomic_write_json(
        record_path,
        result,
    )

    verification = (
        verify_dnai_fiber_physical_field_record(
            record_path,
            check_artifacts=True,
        )
    )

    if not verification.get(
        "ok"
    ):
        raise ControlPlaneError(
            "New DNAi physical field failed self-verification"
        )

    return record_path


def build_parser(
) -> argparse.ArgumentParser:
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--model",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--validation-manifest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--image-index",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--kl720-python",
        type=Path,
    )

    parser.add_argument(
        "--kl720-port",
        type=int,
    )

    parser.add_argument(
        "--kl720-timeout-ms",
        type=int,
        default=10000,
    )

    return parser


def main(
    argv: Optional[list[str]] = None,  # noqa: UP045
) -> int:
    args = (
        build_parser()
        .parse_args(
            argv
        )
    )

    record_path = (
        execute_physical_field(
            model_path=(
                args.model
            ),
            validation_manifest=(
                args.validation_manifest
            ),
            image_index=(
                args.image_index
            ),
            output_root=(
                args.output_root
            ),
            kl720_python=(
                args.kl720_python
            ),
            kl720_port=(
                args.kl720_port
            ),
            kl720_timeout_ms=(
                args.kl720_timeout_ms
            ),
        )
    )

    record = load_json_object(
        record_path
    )

    print(
        json.dumps(
            {
                "status": (
                    "complete"
                ),
                "field_id": (
                    record[
                        "field_id"
                    ]
                ),
                "sample_id": (
                    record[
                        "identity"
                    ][
                        "sample_id"
                    ]
                ),
                "record": str(
                    record_path
                ),
                "n_fibers_valid": (
                    record[
                        "counts"
                    ][
                        "n_fibers_valid"
                    ]
                ),
            }
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
