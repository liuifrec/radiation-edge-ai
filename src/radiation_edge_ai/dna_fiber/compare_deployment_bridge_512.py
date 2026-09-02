"""Validate the DNAi 512x512 deployment bridge before INT8 quantization.

This experiment separates two effects that must not be conflated in a
biological-fidelity paper:

A. Window adaptation
   strict-FP32 PyTorch at the original 1024x1024 window versus the same
   strict-FP32 PyTorch model at the KL720-compatible 512x512 sliding window.

B. Runtime / graph conversion
   strict-FP32 PyTorch 512 versus original ONNX 512 versus Kneron-optimized
   ONNX 512 on the same real microscopy images.

The frozen 20-image DNAi inter-grader panel and DNAi's exact preprocessing are
used. The script also recomputes AI-vs-human segmentation and fiber-detection
metrics for the 1024 and 512 PyTorch conditions, so a change in technical
window size can be judged at the biological-analysis level rather than only by
pixel agreement.

No INT8 quantization or physical KL720 inference happens here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path

logging.getLogger("streamlit").setLevel(logging.ERROR)

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer

from dnafiber.data.utils import load_image
from dnafiber.inference import probas_to_segmentation, transform
from dnafiber.model.autopadDPT import AutoPad
from dnafiber.model.models_zoo import Models
from dnafiber.model.utils import _get_model
from dnafiber.postprocess import refine_segmentation

DNAI_COMMIT = "fcf20c7d6eb385675ff7d07da4fdf471589ce0cf"
MODEL_REVISION = Models.UNET_MOBILEONE_S1
PIXEL_SIZE_UM = 0.26
CLARITY = 1.0
GRADERS = ("H1", "H2", "H3", "H4")
MATCH_IOU = 0.5
DEFAULT_TILE = 512
DEFAULT_OVERLAP = 0.25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state_dict(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key in sorted(model.state_dict()):
        tensor = model.state_dict()[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def binary_dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = int(a.sum()) + int(b.sum())
    return 1.0 if denom == 0 else 2.0 * float(np.logical_and(a, b).sum()) / denom


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int(np.logical_or(a, b).sum())
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum()) / union


def segmentation_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    return {
        "pixel_agreement": float(np.mean(reference == candidate)),
        "dice_fg": binary_dice(reference > 0, candidate > 0),
        "dice_red": binary_dice(reference == 1, candidate == 1),
        "dice_green": binary_dice(reference == 2, candidate == 2),
    }


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


def read_human_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read annotation: {path}")
    if image.ndim < 3:
        raise RuntimeError(f"Expected RGB annotation mask: {path}")
    image = image[:, :, :3][:, :, ::-1]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[image[:, :, 0] > 150] = 1
    mask[image[:, :, 1] > 150] = 2
    return mask


def mask_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    return {
        "dice_fg": binary_dice(a > 0, b > 0),
        "iou_fg": binary_iou(a > 0, b > 0),
        "dice_red": binary_dice(a == 1, b == 1),
        "dice_green": binary_dice(a == 2, b == 2),
    }


def match_fibers(predicted, reference, threshold: float = MATCH_IOU) -> dict[str, float]:
    pred = list(predicted)
    ref = list(reference)
    candidates = []
    for i, p in enumerate(pred):
        for j, r in enumerate(ref):
            iou = float(p.bbox_iou(r))
            if iou >= threshold:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True)
    used_p: set[int] = set()
    used_r: set[int] = set()
    matches = []
    for iou, i, j in candidates:
        if i in used_p or j in used_r:
            continue
        used_p.add(i)
        used_r.add(j)
        matches.append((pred[i], ref[j], iou))

    n_match = len(matches)
    precision = n_match / len(pred) if pred else (1.0 if not ref else 0.0)
    recall = n_match / len(ref) if ref else (1.0 if not pred else 0.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    ratio_abs = []
    total_abs_um = []
    for p, r, _iou in matches:
        if math.isfinite(float(p.ratio)) and math.isfinite(float(r.ratio)):
            ratio_abs.append(abs(float(p.ratio) - float(r.ratio)))
        total_abs_um.append(abs(float(p.length) - float(r.length)) * PIXEL_SIZE_UM)

    return {
        "matched_fibers": n_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_bbox_iou": finite_mean(iou for _p, _r, iou in matches),
        "matched_ratio_mae": finite_mean(ratio_abs),
        "matched_total_length_mae_um": finite_mean(total_abs_um),
    }


def make_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


class OrtSoftmaxModel:
    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu":
            raise RuntimeError("ORT adapter expects CPU tensors")
        if tensor.dtype != torch.float32:
            tensor = tensor.to(dtype=torch.float32)
        if tensor.shape[0] != 1:
            raise RuntimeError("Fixed-batch ONNX requires sw_batch_size=1")
        logits = self.session.run(
            [self.output_name], {self.input_name: tensor.numpy()}
        )[0]
        return torch.softmax(torch.from_numpy(logits), dim=1)


def validate_fixed_shape(session: ort.InferenceSession, tile: int, label: str) -> None:
    shape = list(session.get_inputs()[0].shape)
    expected = [1, 3, tile, tile]
    if shape != expected:
        raise RuntimeError(f"{label} input shape {shape}; expected {expected}")


def center_crop_tensor(tensor: torch.Tensor, tile: int) -> torch.Tensor:
    _, _, height, width = tensor.shape
    y0 = (height - tile) // 2
    x0 = (width - tile) // 2
    return tensor[:, :, y0 : y0 + tile, x0 : x0 + tile]


def run_gpu_sliding(
    tensor_cpu: torch.Tensor,
    inferer: SlidingWindowInferer,
    exec_unit: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = inferer(tensor_cpu.to(device), exec_unit)
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if output.dtype != torch.float32:
        raise RuntimeError(f"Strict-FP32 violation: {output.dtype}")
    return output.detach().cpu(), elapsed_ms


def load_baseline_1024(sample_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    prob_path = sample_dir / "probabilities_fp32.npz"
    seg_path = sample_dir / "segmentation_class_ids.png"
    if not prob_path.is_file() or not seg_path.is_file():
        raise FileNotFoundError(
            f"Missing frozen 1024 baseline for {sample_dir.name}; expected {prob_path} and {seg_path}"
        )
    with np.load(prob_path) as payload:
        probs = payload["probabilities"].astype(np.float32, copy=False)
    if probs.shape != (3, 1024, 1024):
        raise RuntimeError(f"Unexpected 1024 baseline probability shape: {probs.shape}")
    segmentation = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
    if segmentation is None or segmentation.shape != (1024, 1024):
        raise RuntimeError(f"Unexpected 1024 baseline segmentation: {seg_path}")
    return probs[None, ...], segmentation.astype(np.uint8, copy=False)


def prefixed(prefix: str, values: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    default_data_root = os.environ.get(
        "RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"
    )
    default_model_root = Path(
        os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models")
    )
    default_original = (
        default_model_root / "dnai" / "unet_mobileone_s1" / "onnx"
        / "unet_mobileone_s1_512x512_opset11.onnx"
    )
    default_optimized = (
        default_model_root / "dnai" / "unet_mobileone_s1" / "kneron"
        / "unet_mobileone_s1_512x512_opset11_optimized.onnx"
    )
    default_output = (
        default_model_root / "dnai" / "unet_mobileone_s1" / "kneron"
        / "deployment_bridge_512_real"
    )

    parser.add_argument("--data-root", default=default_data_root)
    parser.add_argument("--original", default=str(default_original))
    parser.add_argument("--optimized", default=str(default_optimized))
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-framework-argmax", type=float, default=0.9999)
    parser.add_argument("--min-framework-dice", type=float, default=0.995)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the PyTorch deployment-bridge reference")

    data_root = Path(args.data_root)
    dataset_root = data_root / "dnai_public_v2" / "extracted" / "zenodo"
    panel_root = data_root / "dnai_public_v2" / "r1_intergrader20"
    manifest_path = panel_root / "manifest.json"
    baseline_root = panel_root / "r1_s1_strict_fp32_preprocessed"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_path = Path(args.original)
    optimized_path = Path(args.optimized)
    for path in (original_path, optimized_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dnai_commit") != DNAI_COMMIT:
        raise RuntimeError("Manifest/code revision mismatch")
    records = manifest["records"]
    if args.limit is not None:
        records = records[: args.limit]

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("Radiation Edge AI - DNAi 512 deployment bridge")
    print(f"Images: {len(records)}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"torch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
    print(f"Tile: {args.tile}x{args.tile}; overlap={args.overlap}; gaussian blending")
    print("Precision: strict float32; autocast disabled")
    print("Preprocessing: DNAi load_image pixel_size=0.26 um, clarity=1.0")
    print(f"Original ONNX:  {original_path}")
    print(f"Optimized ONNX: {optimized_path}")
    print("")

    print("[Load PyTorch S1]")
    model = _get_model(MODEL_REVISION).to(device=device, dtype=torch.float32).eval()
    model_hash = sha256_state_dict(model)
    print(f" - parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f" - state_dict SHA256: {model_hash}")
    gpu_exec = AutoPad(nn.Sequential(model, nn.Softmax(dim=1)), 32).to(device).eval()
    gpu_inferer = SlidingWindowInferer(
        roi_size=(args.tile, args.tile),
        sw_batch_size=1,
        overlap=args.overlap,
        mode="gaussian",
        sw_device=device,
        device=device,
        progress=False,
    )

    print("[Load ONNX Runtime sessions]")
    original_session = make_session(original_path)
    optimized_session = make_session(optimized_path)
    validate_fixed_shape(original_session, args.tile, "original")
    validate_fixed_shape(optimized_session, args.tile, "optimized")
    original_ort = OrtSoftmaxModel(original_session)
    optimized_ort = OrtSoftmaxModel(optimized_session)
    cpu_inferer = SlidingWindowInferer(
        roi_size=(args.tile, args.tile),
        sw_batch_size=1,
        overlap=args.overlap,
        mode="gaussian",
        sw_device=torch.device("cpu"),
        device=torch.device("cpu"),
        progress=False,
    )
    print(f" - ORT: {ort.__version__}; provider=CPUExecutionProvider")
    print("")

    image_rows: list[dict] = []
    human_rows: list[dict] = []
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
        tensor_cpu = transform(image=image)["image"].unsqueeze(0).to(dtype=torch.float32)
        if tuple(tensor_cpu.shape) != (1, 3, 1024, 1024):
            raise RuntimeError(f"Unexpected processed tensor: {tuple(tensor_cpu.shape)}")

        baseline_probs, baseline_seg = load_baseline_1024(
            baseline_root / record["sample_id"]
        )
        baseline_fibers = refine_segmentation(baseline_seg).valid_copy()

        center_cpu = center_crop_tensor(tensor_cpu, args.tile)
        center_gpu = center_cpu.to(device)
        with torch.inference_mode():
            torch_center_logits = model(center_gpu).detach().cpu().numpy()
        onnx_center_logits = original_session.run(
            [original_session.get_outputs()[0].name],
            {original_session.get_inputs()[0].name: center_cpu.numpy()},
        )[0]
        optimized_center_logits = optimized_session.run(
            [optimized_session.get_outputs()[0].name],
            {optimized_session.get_inputs()[0].name: center_cpu.numpy()},
        )[0]

        gpu_probs, gpu_ms = run_gpu_sliding(tensor_cpu, gpu_inferer, gpu_exec, device)
        with torch.inference_mode():
            t0 = time.perf_counter()
            onnx_probs = cpu_inferer(tensor_cpu, original_ort)
            onnx_ms = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            optimized_probs = cpu_inferer(tensor_cpu, optimized_ort)
            optimized_ms = (time.perf_counter() - t0) * 1000.0

        gpu_np = gpu_probs.numpy().astype(np.float32, copy=False)
        onnx_np = onnx_probs.numpy().astype(np.float32, copy=False)
        optimized_np = optimized_probs.numpy().astype(np.float32, copy=False)

        gpu_seg = probas_to_segmentation(gpu_probs)
        onnx_seg = probas_to_segmentation(onnx_probs)
        optimized_seg = probas_to_segmentation(optimized_probs)
        gpu_fibers = refine_segmentation(gpu_seg).valid_copy()
        onnx_fibers = refine_segmentation(onnx_seg).valid_copy()
        optimized_fibers = refine_segmentation(optimized_seg).valid_copy()

        window_argmax = float(
            np.mean(baseline_probs.argmax(axis=1) == gpu_np.argmax(axis=1))
        )
        gpu_onnx_argmax = float(
            np.mean(gpu_np.argmax(axis=1) == onnx_np.argmax(axis=1))
        )
        gpu_opt_argmax = float(
            np.mean(gpu_np.argmax(axis=1) == optimized_np.argmax(axis=1))
        )

        row = {
            "index": record["index"],
            "sample_id": record["sample_id"],
            "key": record["key"],
            "window_1024_vs_512_argmax_agreement": window_argmax,
            **prefixed("window_1024_vs_512_seg", segmentation_metrics(baseline_seg, gpu_seg)),
            "window_1024_fibers": len(baseline_fibers),
            "window_512_fibers": len(gpu_fibers),
            "window_fiber_count_difference": len(gpu_fibers) - len(baseline_fibers),
            "window_1024_mean_ratio": finite_mean(baseline_fibers.ratios),
            "window_512_mean_ratio": finite_mean(gpu_fibers.ratios),
            "gpu_vs_onnx_center_logit_max_abs_error": array_metrics(torch_center_logits, onnx_center_logits)["max_abs_error"],
            "gpu_vs_onnx_center_argmax_agreement": float(np.mean(torch_center_logits.argmax(axis=1) == onnx_center_logits.argmax(axis=1))),
            "gpu_vs_optimized_center_logit_max_abs_error": array_metrics(torch_center_logits, optimized_center_logits)["max_abs_error"],
            "gpu_vs_optimized_center_argmax_agreement": float(np.mean(torch_center_logits.argmax(axis=1) == optimized_center_logits.argmax(axis=1))),
            "gpu_vs_onnx_stitched_probability_max_abs_error": array_metrics(gpu_np, onnx_np)["max_abs_error"],
            "gpu_vs_onnx_stitched_argmax_agreement": gpu_onnx_argmax,
            **prefixed("gpu_vs_onnx_seg", segmentation_metrics(gpu_seg, onnx_seg)),
            "gpu_vs_onnx_fiber_count_difference": len(onnx_fibers) - len(gpu_fibers),
            "gpu_vs_optimized_stitched_probability_max_abs_error": array_metrics(gpu_np, optimized_np)["max_abs_error"],
            "gpu_vs_optimized_stitched_argmax_agreement": gpu_opt_argmax,
            **prefixed("gpu_vs_optimized_seg", segmentation_metrics(gpu_seg, optimized_seg)),
            "gpu_vs_optimized_fiber_count_difference": len(optimized_fibers) - len(gpu_fibers),
            "gpu_512_inference_ms": gpu_ms,
            "onnx_512_cpu_inference_ms": onnx_ms,
            "optimized_onnx_512_cpu_inference_ms": optimized_ms,
        }
        image_rows.append(row)

        for grader in GRADERS:
            annotation = dataset_root / record[f"{grader}_annotation"]
            if sha256_file(annotation) != record[f"{grader}_sha256"]:
                raise RuntimeError(f"Annotation checksum mismatch: {annotation}")
            human_mask = read_human_mask(annotation)
            human_fibers = refine_segmentation(human_mask).valid_copy()
            metrics_1024 = mask_metrics(baseline_seg, human_mask)
            metrics_512 = mask_metrics(gpu_seg, human_mask)
            fiber_1024 = match_fibers(baseline_fibers, human_fibers)
            fiber_512 = match_fibers(gpu_fibers, human_fibers)
            human_rows.append(
                {
                    "index": record["index"],
                    "sample_id": record["sample_id"],
                    "key": record["key"],
                    "grader": grader,
                    "ai1024_dice_fg": metrics_1024["dice_fg"],
                    "ai512_dice_fg": metrics_512["dice_fg"],
                    "dice_fg_difference_512_minus_1024": metrics_512["dice_fg"] - metrics_1024["dice_fg"],
                    "ai1024_detection_f1": fiber_1024["f1"],
                    "ai512_detection_f1": fiber_512["f1"],
                    "detection_f1_difference_512_minus_1024": fiber_512["f1"] - fiber_1024["f1"],
                    "ai1024_valid_fibers": len(baseline_fibers),
                    "ai512_valid_fibers": len(gpu_fibers),
                    "human_valid_fibers": len(human_fibers),
                }
            )

        print(
            f"[{position:02d}/{len(records):02d}] {record['sample_id']} "
            f"window_dice={row['window_1024_vs_512_seg_dice_fg']:.6f} "
            f"GPU/ONNX={gpu_onnx_argmax:.8f} "
            f"GPU/KOPT={gpu_opt_argmax:.8f} "
            f"fibers1024/512={len(baseline_fibers)}/{len(gpu_fibers)}"
        )

    image_csv = output_dir / "per_image.csv"
    with image_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(image_rows)

    human_csv = output_dir / "ai_vs_human.csv"
    with human_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(human_rows[0].keys()))
        writer.writeheader()
        writer.writerows(human_rows)

    min_gpu_onnx_argmax = min(r["gpu_vs_onnx_stitched_argmax_agreement"] for r in image_rows)
    min_gpu_onnx_dice = min(r["gpu_vs_onnx_seg_dice_fg"] for r in image_rows)
    min_gpu_opt_argmax = min(r["gpu_vs_optimized_stitched_argmax_agreement"] for r in image_rows)
    min_gpu_opt_dice = min(r["gpu_vs_optimized_seg_dice_fg"] for r in image_rows)
    framework_pass = (
        min_gpu_onnx_argmax >= args.min_framework_argmax
        and min_gpu_onnx_dice >= args.min_framework_dice
        and min_gpu_opt_argmax >= args.min_framework_argmax
        and min_gpu_opt_dice >= args.min_framework_dice
    )

    mean_window_dice = float(np.mean([r["window_1024_vs_512_seg_dice_fg"] for r in image_rows]))
    min_window_dice = min(r["window_1024_vs_512_seg_dice_fg"] for r in image_rows)
    mean_window_argmax = float(np.mean([r["window_1024_vs_512_argmax_agreement"] for r in image_rows]))
    window_fiber_exact = float(
        np.mean([r["window_fiber_count_difference"] == 0 for r in image_rows])
    )
    mean_human_dice_1024 = float(np.mean([r["ai1024_dice_fg"] for r in human_rows]))
    mean_human_dice_512 = float(np.mean([r["ai512_dice_fg"] for r in human_rows]))
    mean_human_f1_1024 = float(np.mean([r["ai1024_detection_f1"] for r in human_rows]))
    mean_human_f1_512 = float(np.mean([r["ai512_detection_f1"] for r in human_rows]))

    summary = {
        "dnai_commit": DNAI_COMMIT,
        "model": MODEL_REVISION.value,
        "model_state_dict_sha256": model_hash,
        "n_images": len(image_rows),
        "n_ai_human_comparisons": len(human_rows),
        "deployment_tile": args.tile,
        "overlap": args.overlap,
        "window_adaptation": {
            "mean_1024_vs_512_argmax_agreement": mean_window_argmax,
            "mean_1024_vs_512_foreground_dice": mean_window_dice,
            "min_1024_vs_512_foreground_dice": min_window_dice,
            "fiber_count_exact_fraction": window_fiber_exact,
            "mean_ai_vs_human_foreground_dice_1024": mean_human_dice_1024,
            "mean_ai_vs_human_foreground_dice_512": mean_human_dice_512,
            "difference_dice_512_minus_1024": mean_human_dice_512 - mean_human_dice_1024,
            "mean_ai_vs_human_detection_f1_1024": mean_human_f1_1024,
            "mean_ai_vs_human_detection_f1_512": mean_human_f1_512,
            "difference_f1_512_minus_1024": mean_human_f1_512 - mean_human_f1_1024,
        },
        "framework_parity": {
            "min_gpu_vs_original_onnx_argmax_agreement": min_gpu_onnx_argmax,
            "min_gpu_vs_original_onnx_foreground_dice": min_gpu_onnx_dice,
            "min_gpu_vs_kneron_optimized_argmax_agreement": min_gpu_opt_argmax,
            "min_gpu_vs_kneron_optimized_foreground_dice": min_gpu_opt_dice,
            "threshold_argmax": args.min_framework_argmax,
            "threshold_foreground_dice": args.min_framework_dice,
            "pass": framework_pass,
        },
        "mean_latency_ms": {
            "pytorch_gpu_512": float(np.mean([r["gpu_512_inference_ms"] for r in image_rows])),
            "original_onnx_cpu_512": float(np.mean([r["onnx_512_cpu_inference_ms"] for r in image_rows])),
            "optimized_onnx_cpu_512": float(np.mean([r["optimized_onnx_512_cpu_inference_ms"] for r in image_rows])),
        },
        "elapsed_seconds": time.perf_counter() - start_all,
        "per_image_csv": str(image_csv),
        "ai_vs_human_csv": str(human_csv),
    }
    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("DNAI 512 DEPLOYMENT BRIDGE COMPLETE: YES")
    print(f"FRAMEWORK PARITY READY: {'YES' if framework_pass else 'NO'}")
    print("")
    print("[Window adaptation: PyTorch FP32 1024 -> 512]")
    print(f"mean argmax agreement: {mean_window_argmax:.8f}")
    print(f"mean foreground Dice: {mean_window_dice:.8f}")
    print(f"min foreground Dice: {min_window_dice:.8f}")
    print(f"fiber-count exact fraction: {window_fiber_exact:.3f}")
    print(f"AI-vs-human foreground Dice: {mean_human_dice_1024:.4f} -> {mean_human_dice_512:.4f} ({mean_human_dice_512 - mean_human_dice_1024:+.4f})")
    print(f"AI-vs-human detection F1: {mean_human_f1_1024:.4f} -> {mean_human_f1_512:.4f} ({mean_human_f1_512 - mean_human_f1_1024:+.4f})")
    print("")
    print("[Framework parity at 512]")
    print(f"min GPU vs original ONNX argmax: {min_gpu_onnx_argmax:.8f}")
    print(f"min GPU vs original ONNX foreground Dice: {min_gpu_onnx_dice:.8f}")
    print(f"min GPU vs Kneron-optimized argmax: {min_gpu_opt_argmax:.8f}")
    print(f"min GPU vs Kneron-optimized foreground Dice: {min_gpu_opt_dice:.8f}")
    print("")
    print("[Mean latency, not an accelerator benchmark]")
    print(f"PyTorch GPU 512: {summary['mean_latency_ms']['pytorch_gpu_512']:.2f} ms/image")
    print(f"Original ONNX CPU 512: {summary['mean_latency_ms']['original_onnx_cpu_512']:.2f} ms/image")
    print(f"Optimized ONNX CPU 512: {summary['mean_latency_ms']['optimized_onnx_cpu_512']:.2f} ms/image")
    print(f"Report: {report_path}")
    print(f"Per-image: {image_csv}")
    print(f"AI-vs-human: {human_csv}")

    if not framework_pass:
        raise SystemExit("512 deployment framework parity gate did not pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
