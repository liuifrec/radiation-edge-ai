"""Run the frozen 20-image DNAi R1 baseline in strict FP32.

The public inter-grader JPEG tiles are already 1024x1024 processed tiles. This
runner therefore starts from those pixels, applies DNAi's model normalization,
runs MobileOne-S1 with the same 1024x1024 Gaussian sliding-window path and
25% overlap, converts probabilities to DNAi's dilated 3-class segmentation,
and performs the unchanged CPU fiber reconstruction.

Unlike DNAi's normal high-end CUDA path, this benchmark intentionally does NOT
use torch.autocast. R1 is therefore a true FP32 teacher/reference for later
ONNX and KL720 INT8 equivalence tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer

from dnafiber.inference import probas_to_segmentation, transform
from dnafiber.model.autopadDPT import AutoPad
from dnafiber.model.models_zoo import Models
from dnafiber.model.utils import _get_model
from dnafiber.postprocess import refine_segmentation

DNAI_COMMIT = "fcf20c7d6eb385675ff7d07da4fdf471589ce0cf"
MODEL_REVISION = Models.UNET_MOBILEONE_S1
PIXEL_SIZE_UM = 0.26
GRADERS = ("H1", "H2", "H3", "H4")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state_dict(model: nn.Module) -> str:
    """Hash model weights deterministically, independent of HF cache layout."""
    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_human_mask(path: Path) -> np.ndarray:
    """Match DNAi stats/intergrader.py human-mask decoding semantics."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV could not read annotation: {path}")
    if image.ndim == 2:
        raise RuntimeError(f"Expected RGB annotation mask, got grayscale: {path}")
    image = image[:, :, :3][:, :, ::-1]  # BGR -> RGB
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[image[:, :, 0] > 150] = 1
    mask[image[:, :, 1] > 150] = 2
    return mask


def binary_dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    return 2.0 * float(np.logical_and(a, b).sum()) / denom


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum()) / union


def segmentation_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return {
        "dice_fg": binary_dice(pred > 0, truth > 0),
        "iou_fg": binary_iou(pred > 0, truth > 0),
        "dice_red": binary_dice(pred == 1, truth == 1),
        "iou_red": binary_iou(pred == 1, truth == 1),
        "dice_green": binary_dice(pred == 2, truth == 2),
        "iou_green": binary_iou(pred == 2, truth == 2),
    }


def finite_mean(values) -> float | None:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(values.mean())


def finite_median(values) -> float | None:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.median(values))


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    if not cv2.imwrite(str(path), mask.astype(np.uint8)):
        raise RuntimeError(f"Failed to write mask: {path}")


def write_preview_png(path: Path, mask: np.ndarray) -> None:
    # OpenCV writes BGR: class 1 red, class 2 green.
    preview = np.zeros((*mask.shape, 3), dtype=np.uint8)
    preview[mask == 1] = (0, 0, 255)
    preview[mask == 2] = (0, 255, 0)
    if not cv2.imwrite(str(path), preview):
        raise RuntimeError(f"Failed to write preview: {path}")


def strict_infer(
    tensor: torch.Tensor,
    inferer: SlidingWindowInferer,
    exec_unit: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.inference_mode():
        # Deliberately no torch.autocast here: R1 is strict float32.
        probs = inferer(tensor, exec_unit)
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    if probs.dtype != torch.float32:
        raise RuntimeError(f"Strict-FP32 violation: output dtype is {probs.dtype}")
    return probs, elapsed_ms, peak_gib


def main() -> int:
    parser = argparse.ArgumentParser()
    default_data_root = os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data")
    parser.add_argument("--data-root", default=default_data_root)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the strict-FP32 R1 baseline")

    data_root = Path(args.data_root)
    dataset_root = data_root / "dnai_public_v2" / "extracted" / "zenodo"
    r1_root = data_root / "dnai_public_v2" / "r1_intergrader20"
    manifest_path = r1_root / "manifest.json"
    output_root = r1_root / "strict_fp32_mobileone_s1"
    output_root.mkdir(parents=True, exist_ok=True)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"R1 manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dnai_commit") != DNAI_COMMIT:
        raise RuntimeError(
            f"Manifest DNAi commit {manifest.get('dnai_commit')} does not match runner {DNAI_COMMIT}"
        )
    records = manifest["records"]
    if args.limit is not None:
        records = records[: args.limit]

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("Radiation Edge AI - DNAi R1 strict-FP32 baseline")
    print(f"Manifest: {manifest_path}")
    print(f"Images: {len(records)}")
    print(f"torch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Model: {MODEL_REVISION.value}")
    print("Precision: strict float32 (autocast disabled)")
    print("Input policy: public intergrader JPEG tiles are used as stored; DNAi model normalization only")
    print("")

    print("[Load pretrained model]")
    load_start = time.perf_counter()
    model = _get_model(MODEL_REVISION).to(device=device, dtype=torch.float32).eval()
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("Strict-FP32 violation: model contains non-float32 parameters")
    model_hash = sha256_state_dict(model)
    print(f" - Success ({time.perf_counter() - load_start:.2f}s)")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"state_dict SHA256: {model_hash}")

    # Match the DNAi high-end graph except for autocast. AutoPad is a no-op for
    # 1024x1024, but keeping it makes the reference graph explicit.
    exec_unit = AutoPad(nn.Sequential(model, nn.Softmax(dim=1)), 32).to(device).eval()
    inferer = SlidingWindowInferer(
        roi_size=(1024, 1024),
        sw_batch_size=4,
        overlap=0.25,
        mode="gaussian",
        sw_device=device,
        device=device,
        progress=False,
    )

    print("[Warm-up]")
    warmup = torch.zeros((1, 3, 1024, 1024), dtype=torch.float32)
    _warm_probs, _, _ = strict_infer(warmup, inferer, exec_unit, device)
    del _warm_probs
    print(" - Success")
    print("")

    summary_rows: list[dict] = []
    grader_rows: list[dict] = []

    for position, record in enumerate(records, start=1):
        sample_id = record["sample_id"]
        sample_dir = output_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        done_marker = sample_dir / "provenance.json"
        if done_marker.exists() and not args.overwrite:
            print(f"[{position:02d}/{len(records):02d}] {sample_id}: SKIP (already complete)")
            previous = json.loads(done_marker.read_text(encoding="utf-8"))
            summary_rows.append(previous["summary"])
            grader_rows.extend(previous.get("grader_metrics", []))
            continue

        source = dataset_root / record["source"]
        if sha256_file(source) != record["source_sha256"]:
            raise RuntimeError(f"Source checksum mismatch: {source}")

        image = read_rgb(source)
        height, width = image.shape[:2]
        if (height, width) != (record["height"], record["width"]):
            raise RuntimeError(f"Source dimensions changed: {source}")
        if (height, width) != (1024, 1024):
            raise RuntimeError(f"R1 expects 1024x1024 tiles, got {width}x{height}: {source}")

        tensor = transform(image=image)["image"].unsqueeze(0).to(dtype=torch.float32)
        if tensor.dtype != torch.float32:
            raise RuntimeError(f"Strict-FP32 violation: transformed input is {tensor.dtype}")

        probs, infer_ms, peak_gib = strict_infer(tensor, inferer, exec_unit, device)
        probs_np = probs[0].detach().cpu().numpy().astype(np.float32, copy=False)

        post_start = time.perf_counter()
        segmentation = probas_to_segmentation(probs)
        fibers = refine_segmentation(segmentation)
        valid_fibers = fibers.valid_copy()
        post_ms = (time.perf_counter() - post_start) * 1000.0

        np.savez_compressed(sample_dir / "probabilities_fp32.npz", probabilities=probs_np)
        write_mask_png(sample_dir / "segmentation_class_ids.png", segmentation)
        write_preview_png(sample_dir / "segmentation_preview.png", segmentation)

        fiber_labelmap = valid_fibers.get_labelmap(height, width, fiber_width=1)
        write_mask_png(sample_dir / "valid_fiber_labelmap.png", fiber_labelmap)
        valid_fibers.to_pickle(sample_dir / "valid_fibers.pkl")
        fiber_df = valid_fibers.to_df(pixel_size=PIXEL_SIZE_UM, img_name=record["key"])
        fiber_df.to_csv(sample_dir / "valid_fibers.csv", index=False)

        summary = {
            "index": record["index"],
            "sample_id": sample_id,
            "key": record["key"],
            "source": record["source"],
            "source_sha256": record["source_sha256"],
            "width": width,
            "height": height,
            "pixel_size_um": PIXEL_SIZE_UM,
            "inference_ms": infer_ms,
            "postprocess_ms": post_ms,
            "peak_cuda_gib": peak_gib,
            "n_fibers_all": len(fibers),
            "n_fibers_valid": len(valid_fibers),
            "mean_valid_ratio": finite_mean(valid_fibers.ratios),
            "median_valid_ratio": finite_median(valid_fibers.ratios),
            "mean_valid_length_px": finite_mean(valid_fibers.lengths),
            "red_foreground_px": int((segmentation == 1).sum()),
            "green_foreground_px": int((segmentation == 2).sum()),
        }

        per_grader = []
        for grader in GRADERS:
            annotation = dataset_root / record[f"{grader}_annotation"]
            if sha256_file(annotation) != record[f"{grader}_sha256"]:
                raise RuntimeError(f"Annotation checksum mismatch: {annotation}")
            human_mask = read_human_mask(annotation)
            metrics = segmentation_metrics(segmentation, human_mask)
            human_fibers = refine_segmentation(human_mask).valid_copy()
            grader_row = {
                "index": record["index"],
                "sample_id": sample_id,
                "key": record["key"],
                "grader": grader,
                **metrics,
                "ai_valid_fibers": len(valid_fibers),
                "human_valid_fibers": len(human_fibers),
                "ai_mean_ratio": finite_mean(valid_fibers.ratios),
                "human_mean_ratio": finite_mean(human_fibers.ratios),
            }
            grader_rows.append(grader_row)
            per_grader.append(grader_row)

        provenance = {
            "protocol": "R1 strict-FP32 MobileOne-S1",
            "dnai_commit": DNAI_COMMIT,
            "dnafiber_version": importlib.metadata.version("dnafiber"),
            "model_revision": MODEL_REVISION.value,
            "model_state_dict_sha256": model_hash,
            "manifest_sha256": sha256_file(manifest_path),
            "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "precision": "float32",
            "autocast": False,
            "model_normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "sliding_window": {
                "roi_size": [1024, 1024],
                "overlap": 0.25,
                "mode": "gaussian",
                "sw_batch_size": 4,
            },
            "postprocess": "DNAi probas_to_segmentation + refine_segmentation",
            "error_detection_model": False,
            "tta": False,
            "summary": summary,
            "grader_metrics": per_grader,
        }
        done_marker.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        summary_rows.append(summary)

        mean_dice = float(np.mean([row["dice_fg"] for row in per_grader]))
        print(
            f"[{position:02d}/{len(records):02d}] {sample_id}: "
            f"{infer_ms:7.2f} ms, valid fibers={len(valid_fibers):3d}, "
            f"mean human Dice(FG)={mean_dice:.3f}"
        )

        del probs, tensor
        torch.cuda.empty_cache()

    summary_path = output_root / "summary.csv"
    grader_path = output_root / "grader_metrics.csv"
    run_path = output_root / "run_provenance.json"

    if summary_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    if grader_rows:
        with grader_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(grader_rows[0].keys()))
            writer.writeheader()
            writer.writerows(grader_rows)

    inference_times = np.asarray([row["inference_ms"] for row in summary_rows], dtype=float)
    valid_counts = np.asarray([row["n_fibers_valid"] for row in summary_rows], dtype=float)
    foreground_dice = np.asarray([row["dice_fg"] for row in grader_rows], dtype=float)

    run_provenance = {
        "protocol": "R1 strict-FP32 MobileOne-S1",
        "complete": len(summary_rows) == len(records),
        "n_images": len(summary_rows),
        "dnai_commit": DNAI_COMMIT,
        "model_revision": MODEL_REVISION.value,
        "model_state_dict_sha256": model_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "precision": "float32",
        "autocast": False,
        "tta": False,
        "error_detection_model": False,
        "mean_inference_ms": float(inference_times.mean()) if inference_times.size else None,
        "median_inference_ms": float(np.median(inference_times)) if inference_times.size else None,
        "mean_valid_fibers_per_image": float(valid_counts.mean()) if valid_counts.size else None,
        "mean_ai_vs_human_foreground_dice": float(foreground_dice.mean()) if foreground_dice.size else None,
        "outputs": {
            "summary_csv": str(summary_path),
            "grader_metrics_csv": str(grader_path),
        },
    }
    run_path.write_text(json.dumps(run_provenance, indent=2), encoding="utf-8")

    print("")
    print("R1 STRICT FP32 BASELINE COMPLETE: YES")
    print(f"Images: {len(summary_rows)}")
    if inference_times.size:
        print(f"Mean inference:   {inference_times.mean():.2f} ms/image")
        print(f"Median inference: {np.median(inference_times):.2f} ms/image")
    if valid_counts.size:
        print(f"Mean valid fibers: {valid_counts.mean():.2f}/image")
    if foreground_dice.size:
        print(f"Mean AI-vs-human foreground Dice: {foreground_dice.mean():.4f}")
    print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
