"""Execute one frozen DNAi FP512 microscopy field.

This worker is intentionally assay-specific. It consumes the frozen
preprocessed/normalized 512x512 validation tensors, runs the verified optimized
FP512 ONNX model, stitches the nine windows with the frozen 50%-overlap Gaussian
policy, and performs DNAi segmentation plus fiber reconstruction.

It does not read human annotations, the FP1024 biological reference, or KL720
hardware outputs, and it does not evaluate biological deployment fidelity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
import torch
from dnafiber.inference import probas_to_segmentation
from dnafiber.postprocess import refine_segmentation
from monai.data.utils import compute_importance_map

ASSAY_ID = "dnai-fiber-v3"
DNAI_COMMIT = "fcf20c7d6eb385675ff7d07da4fdf471589ce0cf"

EXPECTED_ONNX_SHA256 = (
    "a901d1b309a9a0e5026febd5070252787e4b110a2a7d2828e16a19364d6094d0"
)

TILE = 512
OVERLAP = 0.50
STRIDE = 256
SIGMA_SCALE = 0.125
PIXEL_SIZE_UM = 0.26

EXPECTED_TENSOR = (1, 3, TILE, TILE)
EXPECTED_FULL_IMAGE = (1, 3, 1024, 1024)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def canonical_json_sha256(
    value: dict[str, object],
) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def read_json_object(
    path: Path,
) -> dict[str, object]:
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return value


def atomic_save_npy(
    path: Path,
    value: np.ndarray,
) -> None:
    temp = path.with_name(
        path.name + ".tmp"
    )

    with temp.open("wb") as handle:
        np.save(
            handle,
            value,
            allow_pickle=False,
        )

    temp.replace(path)


def atomic_save_npz(
    path: Path,
    **values: np.ndarray,
) -> None:
    temp = path.with_name(
        path.name + ".tmp"
    )

    with temp.open("wb") as handle:
        np.savez_compressed(
            handle,
            **values,
        )

    temp.replace(path)


def atomic_write_json(
    path: Path,
    value: dict[str, object],
) -> None:
    temp = path.with_name(
        path.name + ".tmp"
    )

    temp.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temp.replace(path)


def finite_mean(
    values: list[float],
) -> Optional[float]:  # noqa: UP045
    array = np.asarray(
        values,
        dtype=float,
    )
    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return None

    return float(
        array.mean()
    )


def finite_median(
    values: list[float],
) -> Optional[float]:  # noqa: UP045
    array = np.asarray(
        values,
        dtype=float,
    )
    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return None

    return float(
        np.median(array)
    )


def softmax_np(
    array: np.ndarray,
) -> np.ndarray:
    work = array.astype(
        np.float64,
        copy=False,
    )

    shifted = work - np.max(
        work,
        axis=1,
        keepdims=True,
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


def artifact_record(
    path: Path,
) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

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

    return parser


def main() -> int:
    args = build_parser().parse_args()

    model_path = (
        args.model
        .expanduser()
        .resolve()
    )

    manifest_path = (
        args.validation_manifest
        .expanduser()
        .resolve()
    )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    if not model_path.is_file():
        raise RuntimeError(
            f"Model missing: {model_path}"
        )

    if not manifest_path.is_file():
        raise RuntimeError(
            f"Validation manifest missing: {manifest_path}"
        )

    model_sha = sha256_file(
        model_path
    )

    if model_sha != EXPECTED_ONNX_SHA256:
        raise RuntimeError(
            "Unexpected optimized FP512 ONNX SHA256: "
            f"{model_sha}"
        )

    manifest = read_json_object(
        manifest_path
    )

    if (
        manifest.get("dnai_commit")
        != DNAI_COMMIT
    ):
        raise RuntimeError(
            "DNAi commit mismatch"
        )

    if tuple(
        manifest.get("tile", ())
    ) != EXPECTED_TENSOR:
        raise RuntimeError(
            f"Unexpected tile policy: {manifest.get('tile')}"
        )

    if tuple(
        manifest.get(
            "full_image_shape",
            (),
        )
    ) != EXPECTED_FULL_IMAGE:
        raise RuntimeError(
            "Unexpected full-image shape"
        )

    if float(
        manifest.get(
            "overlap",
            -1.0,
        )
    ) != OVERLAP:
        raise RuntimeError(
            "Unexpected overlap policy"
        )

    if int(
        manifest.get(
            "stride",
            -1,
        )
    ) != STRIDE:
        raise RuntimeError(
            "Unexpected stride policy"
        )

    if (
        manifest.get("blend")
        != "gaussian"
    ):
        raise RuntimeError(
            "Unexpected stitching mode"
        )

    if int(
        manifest.get(
            "windows_per_image",
            -1,
        )
    ) != 9:
        raise RuntimeError(
            "Unexpected windows-per-image policy"
        )

    raw_records = manifest.get(
        "records"
    )

    if not isinstance(
        raw_records,
        list,
    ):
        raise TypeError(
            "Manifest records are malformed"
        )

    records = [
        record
        for record in raw_records
        if (
            isinstance(record, dict)
            and int(
                record.get(
                    "image_index",
                    -1,
                )
            )
            == args.image_index
        )
    ]

    if len(records) != 9:
        raise RuntimeError(
            "Expected exactly 9 windows for image "
            f"{args.image_index}, got {len(records)}"
        )

    records.sort(
        key=lambda item: int(
            item["window_index"]
        )
    )

    window_indices = [
        int(record["window_index"])
        for record in records
    ]

    if window_indices != list(
        range(9)
    ):
        raise RuntimeError(
            "Window indices are not 0..8"
        )

    sample_ids = {
        str(record["sample_id"])
        for record in records
    }

    keys = {
        str(record["key"])
        for record in records
    }

    if len(sample_ids) != 1:
        raise RuntimeError(
            "Multiple sample IDs in one field"
        )

    if len(keys) != 1:
        raise RuntimeError(
            "Multiple keys in one field"
        )

    sample_id = next(
        iter(sample_ids)
    )
    key = next(
        iter(keys)
    )

    observed_coords = {
        (
            int(record["y"]),
            int(record["x"]),
        )
        for record in records
    }

    if observed_coords != EXPECTED_COORDS:
        raise RuntimeError(
            "Frozen 3x3 deployment grid changed"
        )

    validation_root = (
        manifest_path.parent
    ).resolve()

    window_identity = []
    tensor_paths = []

    for record in records:
        relative = Path(
            str(record["tensor"])
        )

        tensor_path = (
            validation_root
            / relative
        ).resolve()

        try:
            tensor_path.relative_to(
                validation_root
            )
        except ValueError as exc:
            raise RuntimeError(
                "Tensor path escapes validation root"
            ) from exc

        if not tensor_path.is_file():
            raise RuntimeError(
                f"Tensor missing: {tensor_path}"
            )

        tensor = np.load(
            tensor_path,
            allow_pickle=False,
        )

        if tensor.shape != EXPECTED_TENSOR:
            raise RuntimeError(
                "Unexpected tensor shape "
                f"{tensor.shape}: {tensor_path}"
            )

        if tensor.dtype != np.float32:
            raise RuntimeError(
                "Unexpected tensor dtype "
                f"{tensor.dtype}: {tensor_path}"
            )

        if not np.isfinite(
            tensor
        ).all():
            raise RuntimeError(
                f"Non-finite tensor: {tensor_path}"
            )

        tensor_paths.append(
            tensor_path
        )

        window_identity.append(
            {
                "window_index": int(
                    record["window_index"]
                ),
                "y": int(record["y"]),
                "x": int(record["x"]),
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

    identity = {
        "schema_version": 1,
        "record_type": (
            "dnai_fiber_field_record"
        ),
        "assay_id": ASSAY_ID,
        "dnai_commit": DNAI_COMMIT,
        "model_sha256": model_sha,
        "validation_manifest_sha256": (
            sha256_file(
                manifest_path
            )
        ),
        "image_index": (
            args.image_index
        ),
        "sample_id": sample_id,
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
            "overlap": OVERLAP,
            "stride": STRIDE,
            "blend": "gaussian",
            "sigma_scale": (
                SIGMA_SCALE
            ),
            "pixel_size_um": (
                PIXEL_SIZE_UM
            ),
        },
        "windows": window_identity,
    }

    fingerprint = (
        canonical_json_sha256(
            identity
        )
    )

    field_id = (
        f"df1-{fingerprint[:16]}"
    )

    field_root = (
        output_root / field_id
    )

    if field_root.exists():
        raise RuntimeError(
            "Content-addressed DNAi field directory "
            f"already exists: {field_root}"
        )

    field_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    windows_root = (
        field_root / "window_logits"
    )

    windows_root.mkdir()

    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )

    session_start = time.perf_counter()

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=[
            "CPUExecutionProvider"
        ],
    )

    session_seconds = (
        time.perf_counter()
        - session_start
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if (
        len(inputs) != 1
        or len(outputs) != 1
    ):
        raise RuntimeError(
            "Expected exactly one ONNX input and output"
        )

    input_node = inputs[0]
    output_node = outputs[0]

    if list(
        input_node.shape
    ) != list(EXPECTED_TENSOR):
        raise RuntimeError(
            "Unexpected model input shape: "
            f"{input_node.shape}"
        )

    if list(
        output_node.shape
    ) != list(EXPECTED_TENSOR):
        raise RuntimeError(
            "Unexpected model output shape: "
            f"{output_node.shape}"
        )

    importance = compute_importance_map(
        (TILE, TILE),
        mode="gaussian",
        sigma_scale=SIGMA_SCALE,
        device="cpu",
        dtype=torch.float32,
    ).cpu().numpy().astype(
        np.float32,
        copy=False,
    )

    importance = np.squeeze(
        importance
    )

    if importance.shape != (
        TILE,
        TILE,
    ):
        raise RuntimeError(
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
    inference_ms = []

    for record, tensor_path in zip(
        records,
        tensor_paths,
    ):
        tensor = np.load(
            tensor_path,
            allow_pickle=False,
        )

        tensor = np.ascontiguousarray(
            tensor,
            dtype=np.float32,
        )

        started = time.perf_counter()

        result = session.run(
            [output_node.name],
            {
                input_node.name: tensor
            },
        )[0]

        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        logits = np.asarray(
            result,
            dtype=np.float32,
        )

        if logits.shape != EXPECTED_TENSOR:
            raise RuntimeError(
                "Unexpected ONNX output shape "
                f"{logits.shape}"
            )

        if not np.isfinite(
            logits
        ).all():
            raise RuntimeError(
                "Non-finite ONNX output"
            )

        window_index = int(
            record["window_index"]
        )

        stem = (
            f"{args.image_index:02d}_"
            f"{window_index:02d}_"
            f"{sample_id}"
        )

        output_path = (
            windows_root
            / f"{stem}.npy"
        )

        atomic_save_npy(
            output_path,
            np.ascontiguousarray(
                logits
            ),
        )

        probabilities = softmax_np(
            logits
        )

        y0 = int(record["y"])
        x0 = int(record["x"])
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

        inference_ms.append(
            elapsed_ms
        )

        window_outputs.append(
            {
                "window_index": (
                    window_index
                ),
                "y": y0,
                "x": x0,
                "input": (
                    artifact_record(
                        tensor_path
                    )
                ),
                "output": (
                    artifact_record(
                        output_path
                    )
                ),
                "inference_ms": (
                    elapsed_ms
                ),
            }
        )

    if np.any(
        weight_sum <= 0
    ):
        raise RuntimeError(
            "Gaussian stitching left uncovered pixels"
        )

    stitched = (
        accumulator
        / weight_sum
    ).astype(
        np.float32
    )

    stitched_path = (
        field_root
        / "floating512_stitched_probabilities.npz"
    )

    atomic_save_npz(
        stitched_path,
        probabilities=stitched[0],
    )

    post_start = time.perf_counter()

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
        raise RuntimeError(
            "Unexpected stitched segmentation shape "
            f"{segmentation.shape}"
        )

    fibers_all = refine_segmentation(
        segmentation
    )

    fibers_valid = (
        fibers_all.valid_copy()
    )

    postprocess_ms = (
        time.perf_counter()
        - post_start
    ) * 1000.0

    segmentation_path = (
        field_root
        / "floating512_segmentation_class_ids.png"
    )

    if not cv2.imwrite(
        str(segmentation_path),
        segmentation,
    ):
        raise RuntimeError(
            "Could not write segmentation PNG"
        )

    fibers_csv = (
        field_root
        / "valid_fibers.csv"
    )

    fiber_df = fibers_valid.to_df(
        pixel_size=PIXEL_SIZE_UM,
        img_name=key,
    )

    fiber_df.to_csv(
        fibers_csv,
        index=False,
    )

    valid_list = list(
        fibers_valid
    )

    ratios = [
        float(fiber.ratio)
        for fiber in valid_list
        if math.isfinite(
            float(fiber.ratio)
        )
    ]

    lengths_um = [
        float(fiber.length)
        * PIXEL_SIZE_UM
        for fiber in valid_list
    ]

    artifacts = {
        "stitched_probabilities": (
            artifact_record(
                stitched_path
            )
        ),
        "segmentation": (
            artifact_record(
                segmentation_path
            )
        ),
        "valid_fibers_csv": (
            artifact_record(
                fibers_csv
            )
        ),
    }

    timing = {
        "session_create_ms": (
            session_seconds
            * 1000.0
        ),
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
    }

    result = {
        "schema_version": 1,
        "record_type": (
            "dnai_fiber_field_record"
        ),
        "status": "complete",
        "field_id": field_id,
        "field_fingerprint_sha256": (
            fingerprint
        ),
        "identity": identity,
        "source_artifacts": {
            "model": (
                artifact_record(
                    model_path
                )
            ),
            "validation_manifest": (
                artifact_record(
                    manifest_path
                )
            ),
        },
        "runtime": {
            "python_executable": (
                sys.executable
            ),
            "python_version": (
                platform.python_version()
            ),
            "numpy_version": (
                np.__version__
            ),
            "onnxruntime_version": (
                ort.__version__
            ),
            "torch_version": (
                torch.__version__
            ),
            "dnafiber_version": (
                importlib.metadata.version(
                    "dnafiber"
                )
            ),
            "provider": (
                "CPUExecutionProvider"
            ),
            "active_providers": (
                session.get_providers()
            ),
        },
        "counts": {
            "n_windows": 9,
            "n_fibers_all": (
                len(fibers_all)
            ),
            "n_fibers_valid": (
                len(valid_list)
            ),
        },
        "measurements": {
            "mean_valid_ratio": (
                finite_mean(
                    ratios
                )
            ),
            "median_valid_ratio": (
                finite_median(
                    ratios
                )
            ),
            "mean_valid_length_um": (
                finite_mean(
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
        "timing": timing,
        "window_outputs": (
            window_outputs
        ),
        "artifacts": artifacts,
        "scientific_scope": {
            "preprocessed_frozen_window_inputs": True,
            "raw_microscopy_preprocessing_performed": False,
            "floating_fp512_inference_performed": True,
            "gaussian_stitching_performed": True,
            "fiber_object_reconstruction_performed": True,
            "human_annotations_read": False,
            "fp1024_reference_read": False,
            "biological_fidelity_evaluated": False,
            "kl720_hardware_access_performed": False,
        },
        "interpretation": (
            "Application-integration field result only. "
            "This record does not establish biological deployment "
            "fidelity or KL720 equivalence."
        ),
    }

    record_path = (
        field_root
        / "field_result.json"
    )

    atomic_write_json(
        record_path,
        result,
    )

    print(
        json.dumps(
            {
                "status": "complete",
                "field_id": field_id,
                "sample_id": sample_id,
                "record": str(
                    record_path
                ),
                "n_fibers_valid": (
                    len(valid_list)
                ),
            }
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
