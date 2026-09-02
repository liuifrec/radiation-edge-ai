"""Sweep 512x512 sliding-window overlap for DNAi deployment fidelity.

The KL720-compatible MobileOne-S1 input is fixed at 512x512. This script keeps
that model/window fixed and varies only MONAI sliding-window overlap to find a
tiling strategy that best preserves the frozen strict-FP32 1024x1024 DNAi
baseline on the public 20-image inter-grader panel.

For each overlap it reports:
- 1024-vs-512 stitched argmax agreement and segmentation Dice
- exact reconstructed-fiber count fraction
- AI-vs-human foreground Dice and fiber-detection F1
- strict-FP32 GPU latency

No ONNX, Kneron optimization, quantization, or KL720 execution is performed.
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
TILE = 512


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


def finite_mean(values) -> float | None:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    return None if vals.size == 0 else float(vals.mean())


def read_human_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read annotation: {path}")
    image = image[:, :, :3][:, :, ::-1]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[image[:, :, 0] > 150] = 1
    mask[image[:, :, 1] > 150] = 2
    return mask


def match_fibers(predicted, reference, threshold: float = MATCH_IOU) -> float:
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
    n_match = 0
    for _iou, i, j in candidates:
        if i in used_p or j in used_r:
            continue
        used_p.add(i)
        used_r.add(j)
        n_match += 1
    precision = n_match / len(pred) if pred else (1.0 if not ref else 0.0)
    recall = n_match / len(ref) if ref else (1.0 if not pred else 0.0)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def load_baseline(sample_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    prob_path = sample_dir / "probabilities_fp32.npz"
    seg_path = sample_dir / "segmentation_class_ids.png"
    with np.load(prob_path) as payload:
        probs = payload["probabilities"].astype(np.float32, copy=False)
    seg = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
    if probs.shape != (3, 1024, 1024) or seg is None or seg.shape != (1024, 1024):
        raise RuntimeError(f"Invalid frozen baseline in {sample_dir}")
    return probs[None, ...], seg.astype(np.uint8, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    default_data_root = os.environ.get(
        "RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"
    )
    default_model_root = Path(
        os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models")
    )
    default_output = (
        default_model_root / "dnai" / "unet_mobileone_s1" / "kneron"
        / "overlap_sweep_512_real"
    )
    parser.add_argument("--data-root", default=default_data_root)
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--overlaps", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    for overlap in args.overlaps:
        if not 0.0 <= overlap < 1.0:
            raise SystemExit(f"Invalid overlap: {overlap}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    data_root = Path(args.data_root)
    dataset_root = data_root / "dnai_public_v2" / "extracted" / "zenodo"
    panel_root = data_root / "dnai_public_v2" / "r1_intergrader20"
    manifest_path = panel_root / "manifest.json"
    baseline_root = panel_root / "r1_s1_strict_fp32_preprocessed"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print("Radiation Edge AI - DNAi 512 overlap fidelity sweep")
    print(f"Images: {len(records)}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Overlaps: {args.overlaps}")
    print("Tile: 512x512; gaussian blending; strict FP32; sw_batch_size=1")
    print("")

    model = _get_model(MODEL_REVISION).to(device=device, dtype=torch.float32).eval()
    exec_unit = AutoPad(nn.Sequential(model, nn.Softmax(dim=1)), 32).to(device).eval()

    image_rows: list[dict] = []
    human_rows: list[dict] = []

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
        baseline_probs, baseline_seg = load_baseline(baseline_root / record["sample_id"])
        baseline_fibers = refine_segmentation(baseline_seg).valid_copy()

        human_masks = {}
        human_fibers = {}
        for grader in GRADERS:
            annotation = dataset_root / record[f"{grader}_annotation"]
            if sha256_file(annotation) != record[f"{grader}_sha256"]:
                raise RuntimeError(f"Annotation checksum mismatch: {annotation}")
            human_masks[grader] = read_human_mask(annotation)
            human_fibers[grader] = refine_segmentation(human_masks[grader]).valid_copy()

        parts = []
        for overlap in args.overlaps:
            inferer = SlidingWindowInferer(
                roi_size=(TILE, TILE),
                sw_batch_size=1,
                overlap=overlap,
                mode="gaussian",
                sw_device=device,
                device=device,
                progress=False,
            )
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.inference_mode():
                probs = inferer(tensor.to(device), exec_unit)
            torch.cuda.synchronize(device)
            latency_ms = (time.perf_counter() - start) * 1000.0
            probs_cpu = probs.detach().cpu()
            probs_np = probs_cpu.numpy().astype(np.float32, copy=False)
            seg = probas_to_segmentation(probs_cpu)
            fibers = refine_segmentation(seg).valid_copy()

            argmax_agreement = float(
                np.mean(baseline_probs.argmax(axis=1) == probs_np.argmax(axis=1))
            )
            dice_fg = binary_dice(baseline_seg > 0, seg > 0)
            row = {
                "index": record["index"],
                "sample_id": record["sample_id"],
                "key": record["key"],
                "overlap": overlap,
                "argmax_agreement_vs_1024": argmax_agreement,
                "foreground_dice_vs_1024": dice_fg,
                "baseline_valid_fibers": len(baseline_fibers),
                "deployment_valid_fibers": len(fibers),
                "fiber_count_difference": len(fibers) - len(baseline_fibers),
                "baseline_mean_ratio": finite_mean(baseline_fibers.ratios),
                "deployment_mean_ratio": finite_mean(fibers.ratios),
                "latency_ms": latency_ms,
            }
            image_rows.append(row)

            for grader in GRADERS:
                human_rows.append(
                    {
                        "index": record["index"],
                        "sample_id": record["sample_id"],
                        "key": record["key"],
                        "overlap": overlap,
                        "grader": grader,
                        "foreground_dice": binary_dice(seg > 0, human_masks[grader] > 0),
                        "detection_f1": match_fibers(fibers, human_fibers[grader]),
                        "ai_valid_fibers": len(fibers),
                        "human_valid_fibers": len(human_fibers[grader]),
                    }
                )
            parts.append(
                f"ov={overlap:.2f} dice={dice_fg:.4f} fibers={len(baseline_fibers)}/{len(fibers)}"
            )

        print(f"[{position:02d}/{len(records):02d}] {record['sample_id']} " + " | ".join(parts))

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

    summaries = []
    for overlap in args.overlaps:
        ir = [row for row in image_rows if row["overlap"] == overlap]
        hr = [row for row in human_rows if row["overlap"] == overlap]
        summary = {
            "overlap": overlap,
            "mean_argmax_agreement_vs_1024": float(np.mean([r["argmax_agreement_vs_1024"] for r in ir])),
            "mean_foreground_dice_vs_1024": float(np.mean([r["foreground_dice_vs_1024"] for r in ir])),
            "min_foreground_dice_vs_1024": float(np.min([r["foreground_dice_vs_1024"] for r in ir])),
            "fiber_count_exact_fraction": float(np.mean([r["fiber_count_difference"] == 0 for r in ir])),
            "mean_ai_vs_human_foreground_dice": float(np.mean([r["foreground_dice"] for r in hr])),
            "mean_ai_vs_human_detection_f1": float(np.mean([r["detection_f1"] for r in hr])),
            "mean_latency_ms": float(np.mean([r["latency_ms"] for r in ir])),
        }
        summaries.append(summary)

    # Rank fidelity first; latency is reported but not used to silently trade
    # away biological agreement.
    best = max(
        summaries,
        key=lambda s: (
            s["mean_ai_vs_human_detection_f1"],
            s["mean_ai_vs_human_foreground_dice"],
            s["mean_foreground_dice_vs_1024"],
            s["fiber_count_exact_fraction"],
        ),
    )

    report = {
        "dnai_commit": DNAI_COMMIT,
        "model": MODEL_REVISION.value,
        "tile": TILE,
        "n_images": len(records),
        "overlaps": args.overlaps,
        "summaries": summaries,
        "best_by_biological_fidelity": best,
        "per_image_csv": str(image_csv),
        "ai_vs_human_csv": str(human_csv),
    }
    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("")
    print("DNAI 512 OVERLAP SWEEP COMPLETE: YES")
    for s in summaries:
        print(
            f"overlap={s['overlap']:.2f} "
            f"meanDice1024={s['mean_foreground_dice_vs_1024']:.6f} "
            f"minDice1024={s['min_foreground_dice_vs_1024']:.6f} "
            f"fiberExact={s['fiber_count_exact_fraction']:.3f} "
            f"humanDice={s['mean_ai_vs_human_foreground_dice']:.4f} "
            f"humanF1={s['mean_ai_vs_human_detection_f1']:.4f} "
            f"latency={s['mean_latency_ms']:.2f}ms"
        )
    print(f"BEST OVERLAP BY BIOLOGICAL FIDELITY: {best['overlap']:.2f}")
    print(f"Report: {report_path}")
    print(f"Per-image: {image_csv}")
    print(f"AI-vs-human: {human_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
