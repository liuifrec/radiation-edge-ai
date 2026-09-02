"""Compare original and Kneron-optimized DNAi ONNX models on real microscopy.

This gate is intentionally run before INT8 quantization. It asks whether the
floating-point Kneron optimizer itself changes DNAi decisions on the frozen
public inter-grader panel.

For each 1024x1024 public image we:
1. apply DNAi's exact microscopy preprocessing (0.26 um/pixel, clarity=1.0),
2. apply DNAi's ImageNet tensor normalization,
3. run each fixed-shape ONNX model through MONAI 512x512 sliding-window
   inference with 25% overlap and Gaussian blending,
4. compare stitched probability maps, class decisions, DNAi segmentation, and
   simple reconstructed-fiber summaries.

A center 512x512 crop is also compared at raw-logit level so the optimizer's
numerical perturbation can be measured directly on real data.

No INT8 quantization or KL720 execution happens in this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from monai.inferers import SlidingWindowInferer

from dnafiber.data.utils import load_image
from dnafiber.inference import probas_to_segmentation, transform
from dnafiber.postprocess import refine_segmentation

DNAI_COMMIT = "fcf20c7d6eb385675ff7d07da4fdf471589ce0cf"
PIXEL_SIZE_UM = 0.26
CLARITY = 1.0
DEFAULT_TILE = 512
DEFAULT_OVERLAP = 0.25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = int(a.sum()) + int(b.sum())
    return 1.0 if denom == 0 else 2.0 * float(np.logical_and(a, b).sum()) / denom


def array_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref64 = reference.astype(np.float64, copy=False)
    cand64 = candidate.astype(np.float64, copy=False)
    diff = cand64 - ref64
    abs_diff = np.abs(diff)
    return {
        "max_abs_error": float(abs_diff.max()),
        "mean_abs_error": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff**2))),
    }


def finite_mean(values) -> float | None:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    return None if vals.size == 0 else float(vals.mean())


def make_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    # Disable ORT graph rewrites so this comparison reflects the supplied ONNX
    # graphs rather than a second optimizer applied by ONNX Runtime.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


class OrtSoftmaxModel:
    """Callable adapter for MONAI SlidingWindowInferer.

    DNAi's normal execution unit applies Softmax before the sliding-window
    inferer blends windows. Reproduce that boundary exactly here.
    """

    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu":
            raise RuntimeError("ORT parity adapter expects CPU tensors")
        if tensor.dtype != torch.float32:
            tensor = tensor.to(dtype=torch.float32)
        if tensor.shape[0] != 1:
            raise RuntimeError(
                "Fixed-batch ONNX requires sw_batch_size=1; "
                f"received batch {tensor.shape[0]}"
            )
        logits = self.session.run(
            [self.output_name], {self.input_name: tensor.numpy()}
        )[0]
        return torch.softmax(torch.from_numpy(logits), dim=1)


def raw_logits(session: ort.InferenceSession, patch: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: patch})[0]


def validate_fixed_shape(session: ort.InferenceSession, tile: int, label: str) -> None:
    shape = list(session.get_inputs()[0].shape)
    expected = [1, 3, tile, tile]
    if shape != expected:
        raise RuntimeError(f"{label} input shape {shape}; expected {expected}")


def center_crop(tensor: torch.Tensor, tile: int) -> np.ndarray:
    _, _, height, width = tensor.shape
    if height < tile or width < tile:
        raise RuntimeError(f"Image tensor {height}x{width} is smaller than tile {tile}")
    y0 = (height - tile) // 2
    x0 = (width - tile) // 2
    return tensor[:, :, y0 : y0 + tile, x0 : x0 + tile].numpy()


def segmentation_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    return {
        "pixel_agreement": float(np.mean(reference == candidate)),
        "dice_fg": binary_dice(reference > 0, candidate > 0),
        "dice_red": binary_dice(reference == 1, candidate == 1),
        "dice_green": binary_dice(reference == 2, candidate == 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    default_data_root = os.environ.get(
        "RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"
    )
    default_model_root = Path(
        os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models")
    )
    default_original = (
        default_model_root
        / "dnai"
        / "unet_mobileone_s1"
        / "onnx"
        / "unet_mobileone_s1_512x512_opset11.onnx"
    )
    default_optimized = (
        default_model_root
        / "dnai"
        / "unet_mobileone_s1"
        / "kneron"
        / "unet_mobileone_s1_512x512_opset11_optimized.onnx"
    )
    default_output = (
        default_model_root
        / "dnai"
        / "unet_mobileone_s1"
        / "kneron"
        / "optimizer_parity_512_real"
    )

    parser.add_argument("--original", default=str(default_original))
    parser.add_argument("--optimized", default=str(default_optimized))
    parser.add_argument("--data-root", default=default_data_root)
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-stitched-argmax", type=float, default=0.999)
    parser.add_argument("--min-seg-dice-fg", type=float, default=0.995)
    args = parser.parse_args()

    original_path = Path(args.original)
    optimized_path = Path(args.optimized)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not original_path.is_file():
        raise FileNotFoundError(original_path)
    if not optimized_path.is_file():
        raise FileNotFoundError(optimized_path)
    if args.tile <= 0:
        raise SystemExit("--tile must be positive")
    if not 0.0 <= args.overlap < 1.0:
        raise SystemExit("--overlap must be in [0, 1)")

    data_root = Path(args.data_root)
    dataset_root = data_root / "dnai_public_v2" / "extracted" / "zenodo"
    manifest_path = data_root / "dnai_public_v2" / "r1_intergrader20" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dnai_commit") != DNAI_COMMIT:
        raise RuntimeError("Manifest/code revision mismatch")
    records = manifest["records"]
    if args.limit is not None:
        records = records[: args.limit]

    print("Radiation Edge AI - DNAi real-image Kneron optimizer parity")
    print(f"Original:  {original_path}")
    print(f"Optimized: {optimized_path}")
    print(f"Manifest:  {manifest_path}")
    print(f"Images:    {len(records)}")
    print(f"Tile:      {args.tile}x{args.tile}")
    print(f"Overlap:   {args.overlap}")
    print("DNAi preprocessing: pixel_size=0.26 um, clarity=1.0, reverse_channel=False")
    print("MONAI blending: gaussian; sw_batch_size=1")
    print("")

    print("[Load ONNX Runtime sessions]")
    original_session = make_session(original_path)
    optimized_session = make_session(optimized_path)
    validate_fixed_shape(original_session, args.tile, "original")
    validate_fixed_shape(optimized_session, args.tile, "optimized")
    original_model = OrtSoftmaxModel(original_session)
    optimized_model = OrtSoftmaxModel(optimized_session)
    print(f" - ORT: {ort.__version__}")
    print(f" - original input/output: {original_model.input_name} -> {original_model.output_name}")
    print(f" - optimized input/output: {optimized_model.input_name} -> {optimized_model.output_name}")

    inferer = SlidingWindowInferer(
        roi_size=(args.tile, args.tile),
        sw_batch_size=1,
        overlap=args.overlap,
        mode="gaussian",
        sw_device=torch.device("cpu"),
        device=torch.device("cpu"),
        progress=False,
    )

    rows: list[dict] = []
    start_all = time.perf_counter()

    for position, record in enumerate(records, start=1):
        source = dataset_root / record["source"]
        if sha256_file(source) != record["source_sha256"]:
            raise RuntimeError(f"Source checksum mismatch: {source}")

        image = load_image(
            source,
            reverse_channel=False,
            pixel_size=PIXEL_SIZE_UM,
            verbose=False,
            clarity=CLARITY,
        )
        tensor = transform(image=image)["image"].unsqueeze(0).to(dtype=torch.float32)
        if tuple(tensor.shape) != (1, 3, 1024, 1024):
            raise RuntimeError(f"Expected normalized tensor 1x3x1024x1024, got {tuple(tensor.shape)}")

        center = center_crop(tensor, args.tile)
        t0 = time.perf_counter()
        center_ref = raw_logits(original_session, center)
        center_opt = raw_logits(optimized_session, center)
        center_seconds = time.perf_counter() - t0
        center_metrics = array_metrics(center_ref, center_opt)
        center_argmax = float(
            np.mean(center_ref.argmax(axis=1) == center_opt.argmax(axis=1))
        )

        t0 = time.perf_counter()
        with torch.inference_mode():
            stitched_ref = inferer(tensor, original_model)
            stitched_opt = inferer(tensor, optimized_model)
        stitched_seconds = time.perf_counter() - t0

        ref_np = stitched_ref.numpy().astype(np.float32, copy=False)
        opt_np = stitched_opt.numpy().astype(np.float32, copy=False)
        stitched_metrics = array_metrics(ref_np, opt_np)
        stitched_argmax = float(
            np.mean(ref_np.argmax(axis=1) == opt_np.argmax(axis=1))
        )

        seg_ref = probas_to_segmentation(stitched_ref)
        seg_opt = probas_to_segmentation(stitched_opt)
        seg_metrics = segmentation_metrics(seg_ref, seg_opt)

        fibers_ref = refine_segmentation(seg_ref).valid_copy()
        fibers_opt = refine_segmentation(seg_opt).valid_copy()
        mean_ratio_ref = finite_mean(fibers_ref.ratios)
        mean_ratio_opt = finite_mean(fibers_opt.ratios)
        mean_ratio_abs_diff = None
        if mean_ratio_ref is not None and mean_ratio_opt is not None:
            mean_ratio_abs_diff = abs(mean_ratio_opt - mean_ratio_ref)

        row = {
            "index": record["index"],
            "sample_id": record["sample_id"],
            "key": record["key"],
            "center_logit_max_abs_error": center_metrics["max_abs_error"],
            "center_logit_mean_abs_error": center_metrics["mean_abs_error"],
            "center_logit_rmse": center_metrics["rmse"],
            "center_argmax_agreement": center_argmax,
            "stitched_probability_max_abs_error": stitched_metrics["max_abs_error"],
            "stitched_probability_mean_abs_error": stitched_metrics["mean_abs_error"],
            "stitched_probability_rmse": stitched_metrics["rmse"],
            "stitched_argmax_agreement": stitched_argmax,
            **seg_metrics,
            "reference_valid_fibers": len(fibers_ref),
            "optimized_valid_fibers": len(fibers_opt),
            "valid_fiber_count_difference": len(fibers_opt) - len(fibers_ref),
            "reference_mean_ratio": mean_ratio_ref,
            "optimized_mean_ratio": mean_ratio_opt,
            "mean_ratio_abs_difference": mean_ratio_abs_diff,
            "center_compare_seconds": center_seconds,
            "stitched_compare_seconds": stitched_seconds,
        }
        rows.append(row)

        print(
            f"[{position:02d}/{len(records):02d}] {record['sample_id']} "
            f"center_argmax={center_argmax:.8f} "
            f"stitched_argmax={stitched_argmax:.8f} "
            f"seg_dice_fg={seg_metrics['dice_fg']:.8f} "
            f"fibers={len(fibers_ref)}/{len(fibers_opt)}"
        )

    csv_path = output_dir / "per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    min_center_argmax = min(row["center_argmax_agreement"] for row in rows)
    min_stitched_argmax = min(row["stitched_argmax_agreement"] for row in rows)
    min_seg_dice_fg = min(row["dice_fg"] for row in rows)
    mean_stitched_argmax = float(np.mean([row["stitched_argmax_agreement"] for row in rows]))
    mean_seg_dice_fg = float(np.mean([row["dice_fg"] for row in rows]))
    max_center_logit_error = max(row["center_logit_max_abs_error"] for row in rows)
    max_stitched_probability_error = max(
        row["stitched_probability_max_abs_error"] for row in rows
    )
    fiber_count_exact_fraction = float(
        np.mean([row["reference_valid_fibers"] == row["optimized_valid_fibers"] for row in rows])
    )

    parity_pass = (
        min_stitched_argmax >= args.min_stitched_argmax
        and min_seg_dice_fg >= args.min_seg_dice_fg
    )

    summary = {
        "dnai_commit": DNAI_COMMIT,
        "original_onnx": str(original_path),
        "original_sha256": sha256_file(original_path),
        "optimized_onnx": str(optimized_path),
        "optimized_sha256": sha256_file(optimized_path),
        "manifest": str(manifest_path),
        "n_images": len(rows),
        "tile": args.tile,
        "overlap": args.overlap,
        "preprocessing": {
            "pixel_size_um": PIXEL_SIZE_UM,
            "clarity": CLARITY,
            "reverse_channel": False,
            "tensor_transform": "DNAi inference.transform (ImageNet normalization)",
            "sliding_window_mode": "gaussian",
            "sliding_window_batch": 1,
        },
        "metrics": {
            "max_center_logit_abs_error": max_center_logit_error,
            "min_center_argmax_agreement": min_center_argmax,
            "max_stitched_probability_abs_error": max_stitched_probability_error,
            "mean_stitched_argmax_agreement": mean_stitched_argmax,
            "min_stitched_argmax_agreement": min_stitched_argmax,
            "mean_segmentation_foreground_dice": mean_seg_dice_fg,
            "min_segmentation_foreground_dice": min_seg_dice_fg,
            "fiber_count_exact_fraction": fiber_count_exact_fraction,
        },
        "thresholds": {
            "min_stitched_argmax_agreement": args.min_stitched_argmax,
            "min_segmentation_foreground_dice": args.min_seg_dice_fg,
        },
        "pass": parity_pass,
        "elapsed_seconds": time.perf_counter() - start_all,
        "per_image_csv": str(csv_path),
    }
    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("REAL-IMAGE OPTIMIZER PARITY COMPLETE: YES")
    print(f"REAL-IMAGE OPTIMIZER PARITY READY: {'YES' if parity_pass else 'NO'}")
    print(f"max center-logit abs error: {max_center_logit_error:.8g}")
    print(f"min center argmax agreement: {min_center_argmax:.8f}")
    print(f"max stitched-probability abs error: {max_stitched_probability_error:.8g}")
    print(f"mean stitched argmax agreement: {mean_stitched_argmax:.8f}")
    print(f"min stitched argmax agreement: {min_stitched_argmax:.8f}")
    print(f"mean segmentation foreground Dice: {mean_seg_dice_fg:.8f}")
    print(f"min segmentation foreground Dice: {min_seg_dice_fg:.8f}")
    print(f"fiber-count exact fraction: {fiber_count_exact_fraction:.3f}")
    print(f"Report: {report_path}")
    print(f"Per-image: {csv_path}")

    if not parity_pass:
        raise SystemExit("Kneron optimized ONNX did not pass the real-image parity gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
