"""Run the DNAi paper's UNet-MobileOne S1 baseline on the frozen 20-image panel.

This is the corrected strict-FP32 reference for the compact S1 model. It keeps
DNAi's image preprocessing (background correction/contrast normalization at
0.26 um/pixel, clarity=1.0), model normalization, 1024x1024 inference, softmax,
segmentation conversion, and CPU fiber reconstruction, while explicitly
turning CUDA autocast off.

The previous diagnostic runner that skipped image preprocessing is retained for
provenance but must not be used as the biological reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import logging
import math
import os
import platform
import time
from pathlib import Path

# DNAi imports transitively load Streamlit UI modules. Silence bare-script UI
# noise without changing DNAi behavior.
logging.getLogger("streamlit").setLevel(logging.ERROR)

import cv2
import numpy as np
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


def read_human_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read annotation: {path}")
    if image.ndim < 3:
        raise RuntimeError(f"Expected RGB annotation mask: {path}")
    image = image[:, :, :3][:, :, ::-1]  # BGR -> RGB
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[image[:, :, 0] > 150] = 1
    mask[image[:, :, 1] > 150] = 2
    return mask


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


def mask_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    return {
        "dice_fg": binary_dice(a > 0, b > 0),
        "iou_fg": binary_iou(a > 0, b > 0),
        "dice_red": binary_dice(a == 1, b == 1),
        "iou_red": binary_iou(a == 1, b == 1),
        "dice_green": binary_dice(a == 2, b == 2),
        "iou_green": binary_iou(a == 2, b == 2),
    }


def finite_mean(values):
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    return None if values.size == 0 else float(values.mean())


def finite_median(values):
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    return None if values.size == 0 else float(np.median(values))


def write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed writing {path}")


def write_rgb_png(path: Path, image: np.ndarray) -> None:
    write_png(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def write_mask_preview(path: Path, mask: np.ndarray) -> None:
    preview = np.zeros((*mask.shape, 3), dtype=np.uint8)
    preview[mask == 1] = (0, 0, 255)
    preview[mask == 2] = (0, 255, 0)
    write_png(path, preview)


def strict_infer(tensor, inferer, exec_unit, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.inference_mode():
        probs = inferer(tensor, exec_unit)  # deliberately no autocast
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    if probs.dtype != torch.float32:
        raise RuntimeError(f"Strict-FP32 violation: output dtype={probs.dtype}")
    return probs, elapsed_ms, peak_gib


def match_fibers(predicted, reference, threshold=MATCH_IOU):
    pred = list(predicted)
    ref = list(reference)
    candidates = []
    for i, p in enumerate(pred):
        for j, r in enumerate(ref):
            iou = float(p.bbox_iou(r))
            if iou >= threshold:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True)
    used_p, used_r, matches = set(), set(), []
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
    first_abs_um = []
    second_abs_um = []
    total_abs_um = []
    for p, r, _iou in matches:
        if math.isfinite(float(p.ratio)) and math.isfinite(float(r.ratio)):
            ratio_abs.append(abs(float(p.ratio) - float(r.ratio)))
        first_abs_um.append(abs(float(p.red) - float(r.red)) * PIXEL_SIZE_UM)
        second_abs_um.append(abs(float(p.green) - float(r.green)) * PIXEL_SIZE_UM)
        total_abs_um.append(abs(float(p.length) - float(r.length)) * PIXEL_SIZE_UM)

    return {
        "matched_fibers": n_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_bbox_iou": finite_mean(iou for _p, _r, iou in matches),
        "matched_ratio_mae": finite_mean(ratio_abs),
        "matched_first_analog_mae_um": finite_mean(first_abs_um),
        "matched_second_analog_mae_um": finite_mean(second_abs_um),
        "matched_total_length_mae_um": finite_mean(total_abs_um),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    default_data_root = os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data")
    parser.add_argument("--data-root", default=default_data_root)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    data_root = Path(args.data_root)
    dataset_root = data_root / "dnai_public_v2" / "extracted" / "zenodo"
    panel_root = data_root / "dnai_public_v2" / "r1_intergrader20"
    manifest_path = panel_root / "manifest.json"
    output_root = panel_root / "r1_s1_strict_fp32_preprocessed"
    output_root.mkdir(parents=True, exist_ok=True)

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

    print("Radiation Edge AI - DNAi S1 strict-FP32 corrected baseline")
    print(f"Manifest: {manifest_path}")
    print(f"Images: {len(records)}")
    print(f"torch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Model: {MODEL_REVISION.value}")
    print("Precision: strict float32 (autocast disabled)")
    print(f"DNAi image preprocessing: ENABLED (pixel_size={PIXEL_SIZE_UM} um, clarity={CLARITY})")
    print("TTA: disabled; error detection: disabled (paper single-model base condition)")
    print("")

    print("[Load pretrained model]")
    t0 = time.perf_counter()
    model = _get_model(MODEL_REVISION).to(device=device, dtype=torch.float32).eval()
    model_hash = sha256_state_dict(model)
    print(f" - Success ({time.perf_counter() - t0:.2f}s)")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"state_dict SHA256: {model_hash}")

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
    warm_probs, _, _ = strict_infer(warmup, inferer, exec_unit, device)
    del warm_probs
    print(" - Success")
    print("")

    summary_rows = []
    ai_human_rows = []
    human_human_rows = []

    for position, record in enumerate(records, start=1):
        sample_id = record["sample_id"]
        sample_dir = output_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        done = sample_dir / "provenance.json"

        if done.exists() and not args.overwrite:
            previous = json.loads(done.read_text(encoding="utf-8"))
            summary_rows.append(previous["summary"])
            ai_human_rows.extend(previous["ai_vs_human"])
            human_human_rows.extend(previous["human_vs_human"])
            print(f"[{position:02d}/{len(records):02d}] {sample_id}: SKIP (already complete)")
            continue

        source = dataset_root / record["source"]
        if sha256_file(source) != record["source_sha256"]:
            raise RuntimeError(f"Source checksum mismatch: {source}")

        preprocess_start = time.perf_counter()
        image = load_image(
            source,
            reverse_channel=False,
            pixel_size=PIXEL_SIZE_UM,
            verbose=False,
            clarity=CLARITY,
        )
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
        height, width = image.shape[:2]
        if (height, width) != (1024, 1024):
            raise RuntimeError(f"Expected 1024x1024 processed tile, got {width}x{height}: {source}")

        write_rgb_png(sample_dir / "preprocessed_input.png", image)
        tensor = transform(image=image)["image"].unsqueeze(0).to(dtype=torch.float32)
        probs, infer_ms, peak_gib = strict_infer(tensor, inferer, exec_unit, device)
        probs_np = probs[0].detach().cpu().numpy().astype(np.float32, copy=False)

        post_start = time.perf_counter()
        segmentation = probas_to_segmentation(probs)
        fibers_all = refine_segmentation(segmentation)
        fibers_valid = fibers_all.valid_copy()
        post_ms = (time.perf_counter() - post_start) * 1000.0

        np.savez_compressed(sample_dir / "probabilities_fp32.npz", probabilities=probs_np)
        write_png(sample_dir / "segmentation_class_ids.png", segmentation.astype(np.uint8))
        write_mask_preview(sample_dir / "segmentation_preview.png", segmentation)
        valid_labelmap = fibers_valid.get_labelmap(height, width, fiber_width=1)
        write_png(sample_dir / "valid_fiber_labelmap.png", valid_labelmap)
        fibers_valid.to_pickle(sample_dir / "valid_fibers.pkl")
        fibers_valid.to_df(pixel_size=PIXEL_SIZE_UM, img_name=record["key"]).to_csv(
            sample_dir / "valid_fibers.csv", index=False
        )

        human_masks = {}
        human_fibers = {}
        per_ai_human = []
        for grader in GRADERS:
            annotation = dataset_root / record[f"{grader}_annotation"]
            if sha256_file(annotation) != record[f"{grader}_sha256"]:
                raise RuntimeError(f"Annotation checksum mismatch: {annotation}")
            mask = read_human_mask(annotation)
            hfibers = refine_segmentation(mask).valid_copy()
            human_masks[grader] = mask
            human_fibers[grader] = hfibers

            row = {
                "index": record["index"],
                "sample_id": sample_id,
                "key": record["key"],
                "grader": grader,
                **mask_metrics(segmentation, mask),
                **match_fibers(fibers_valid, hfibers),
                "ai_valid_fibers": len(fibers_valid),
                "human_valid_fibers": len(hfibers),
                "ai_mean_ratio": finite_mean(fibers_valid.ratios),
                "human_mean_ratio": finite_mean(hfibers.ratios),
            }
            ai_human_rows.append(row)
            per_ai_human.append(row)

        per_human_human = []
        for grader_a, grader_b in itertools.combinations(GRADERS, 2):
            row = {
                "index": record["index"],
                "sample_id": sample_id,
                "key": record["key"],
                "grader_a": grader_a,
                "grader_b": grader_b,
                **mask_metrics(human_masks[grader_a], human_masks[grader_b]),
                **match_fibers(human_fibers[grader_a], human_fibers[grader_b]),
                "grader_a_valid_fibers": len(human_fibers[grader_a]),
                "grader_b_valid_fibers": len(human_fibers[grader_b]),
            }
            human_human_rows.append(row)
            per_human_human.append(row)

        summary = {
            "index": record["index"],
            "sample_id": sample_id,
            "key": record["key"],
            "source": record["source"],
            "preprocess_ms": preprocess_ms,
            "inference_ms": infer_ms,
            "postprocess_ms": post_ms,
            "peak_cuda_gib": peak_gib,
            "n_fibers_all": len(fibers_all),
            "n_fibers_valid": len(fibers_valid),
            "mean_valid_ratio": finite_mean(fibers_valid.ratios),
            "median_valid_ratio": finite_median(fibers_valid.ratios),
            "mean_ai_human_fg_dice": finite_mean(r["dice_fg"] for r in per_ai_human),
            "mean_ai_human_detection_f1": finite_mean(r["f1"] for r in per_ai_human),
            "mean_human_human_fg_dice": finite_mean(r["dice_fg"] for r in per_human_human),
            "mean_human_human_detection_f1": finite_mean(r["f1"] for r in per_human_human),
        }
        summary_rows.append(summary)

        provenance = {
            "protocol": "R1-S1 strict-FP32 corrected",
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
            "image_preprocessing": {
                "dnafiber_load_image": True,
                "pixel_size_um": PIXEL_SIZE_UM,
                "clarity": CLARITY,
            },
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
            "tta": False,
            "error_detection_model": False,
            "fiber_match_iou_threshold": MATCH_IOU,
            "summary": summary,
            "ai_vs_human": per_ai_human,
            "human_vs_human": per_human_human,
        }
        done.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

        print(
            f"[{position:02d}/{len(records):02d}] {sample_id}: "
            f"prep={preprocess_ms:6.1f} ms, infer={infer_ms:6.1f} ms, "
            f"valid={len(fibers_valid):3d}, "
            f"AI-human F1={summary['mean_ai_human_detection_f1']:.3f}, "
            f"human-human F1={summary['mean_human_human_detection_f1']:.3f}"
        )

        del probs, tensor
        torch.cuda.empty_cache()

    def write_csv(path, rows):
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary_path = output_root / "summary.csv"
    ai_human_path = output_root / "ai_vs_human.csv"
    human_human_path = output_root / "human_vs_human.csv"
    write_csv(summary_path, summary_rows)
    write_csv(ai_human_path, ai_human_rows)
    write_csv(human_human_path, human_human_rows)

    infer_times = np.asarray([r["inference_ms"] for r in summary_rows], dtype=float)
    valid_counts = np.asarray([r["n_fibers_valid"] for r in summary_rows], dtype=float)
    ai_f1 = np.asarray([r["f1"] for r in ai_human_rows], dtype=float)
    hh_f1 = np.asarray([r["f1"] for r in human_human_rows], dtype=float)
    ai_dice = np.asarray([r["dice_fg"] for r in ai_human_rows], dtype=float)
    hh_dice = np.asarray([r["dice_fg"] for r in human_human_rows], dtype=float)

    run_provenance = {
        "protocol": "R1-S1 strict-FP32 corrected",
        "complete": len(summary_rows) == len(records),
        "n_images": len(summary_rows),
        "dnai_commit": DNAI_COMMIT,
        "model_revision": MODEL_REVISION.value,
        "model_state_dict_sha256": model_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "preprocessing_enabled": True,
        "pixel_size_um": PIXEL_SIZE_UM,
        "clarity": CLARITY,
        "precision": "float32",
        "autocast": False,
        "tta": False,
        "error_detection_model": False,
        "mean_inference_ms": float(infer_times.mean()) if infer_times.size else None,
        "median_inference_ms": float(np.median(infer_times)) if infer_times.size else None,
        "mean_valid_fibers_per_image": float(valid_counts.mean()) if valid_counts.size else None,
        "mean_ai_human_detection_f1": float(ai_f1.mean()) if ai_f1.size else None,
        "mean_human_human_detection_f1": float(hh_f1.mean()) if hh_f1.size else None,
        "mean_ai_human_foreground_dice": float(ai_dice.mean()) if ai_dice.size else None,
        "mean_human_human_foreground_dice": float(hh_dice.mean()) if hh_dice.size else None,
    }
    (output_root / "run_provenance.json").write_text(
        json.dumps(run_provenance, indent=2), encoding="utf-8"
    )

    print("")
    print("R1-S1 CORRECTED STRICT FP32 COMPLETE: YES")
    print(f"Images: {len(summary_rows)}")
    if infer_times.size:
        print(f"Mean inference: {infer_times.mean():.2f} ms/image")
    if valid_counts.size:
        print(f"Mean valid fibers: {valid_counts.mean():.2f}/image")
    if ai_f1.size:
        print(f"Mean AI-vs-human detection F1: {ai_f1.mean():.4f}")
    if hh_f1.size:
        print(f"Mean human-vs-human detection F1: {hh_f1.mean():.4f}")
    if ai_dice.size:
        print(f"Mean AI-vs-human foreground Dice: {ai_dice.mean():.4f}")
    if hh_dice.size:
        print(f"Mean human-vs-human foreground Dice: {hh_dice.mean():.4f}")
    print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
