"""Evaluate stitched KL720 BIE DNA-fiber biological fidelity.

Run this script in the Windows DNAi environment after Kneron's Docker BIE
simulator has produced all nine 512x512 outputs for one or more frozen
1024x1024 validation images.

For each complete image, this script:
1. applies softmax to each Kneron floating-point and fixed-point BIE output,
2. stitches nine windows using the frozen 512x512 / 50% overlap / Gaussian
   policy (MONAI's default gaussian importance map, sigma_scale=0.125),
3. runs DNAi segmentation and fiber reconstruction,
4. compares floating vs INT8 BIE at probability, mask, and fiber levels,
5. compares both against the frozen strict-FP32 1024 DNAi baseline and H1-H4
   human annotations.

This is the biological-fidelity gate before NEF compilation. The BIE simulator
is software; latency here is not physical KL720 latency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from monai.data.utils import compute_importance_map

from dnafiber.inference import probas_to_segmentation
from dnafiber.postprocess import refine_segmentation

DNAI_COMMIT = "fcf20c7d6eb385675ff7d07da4fdf471589ce0cf"
PIXEL_SIZE_UM = 0.26
TILE = 512
OVERLAP = 0.50
SIGMA_SCALE = 0.125
GRADERS = ("H1", "H2", "H3", "H4")
MATCH_IOU = 0.5
EXPECTED_OUTPUT = (1, 3, TILE, TILE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def softmax_np(array: np.ndarray) -> np.ndarray:
    arr = array.astype(np.float64, copy=False)
    shifted = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32)


def binary_dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = int(a.sum()) + int(b.sum())
    return 1.0 if denom == 0 else 2.0 * float(np.logical_and(a, b).sum()) / denom


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


def match_fibers(predicted, reference, threshold: float = MATCH_IOU) -> dict[str, float | int | None]:
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
    total_length_abs_um = []
    for p, r, _iou in matches:
        p_ratio = float(p.ratio)
        r_ratio = float(r.ratio)
        if math.isfinite(p_ratio) and math.isfinite(r_ratio):
            ratio_abs.append(abs(p_ratio - r_ratio))
        total_length_abs_um.append(abs(float(p.length) - float(r.length)) * PIXEL_SIZE_UM)

    return {
        "matched": n_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_bbox_iou": finite_mean(iou for _p, _r, iou in matches),
        "ratio_mae": finite_mean(ratio_abs),
        "total_length_mae_um": finite_mean(total_length_abs_um),
    }


def load_kneron_output(path: Path) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.shape != EXPECTED_OUTPUT:
        raise RuntimeError(f"Unexpected Kneron output shape {array.shape}: {path}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"Non-finite output: {path}")
    return np.ascontiguousarray(array, dtype=np.float32)


def stitch_probabilities(
    records: list[dict],
    output_dir: Path,
    prefix: str,
    image_height: int = 1024,
    image_width: int = 1024,
) -> np.ndarray:
    importance = compute_importance_map(
        (TILE, TILE),
        mode="gaussian",
        sigma_scale=SIGMA_SCALE,
        device="cpu",
        dtype=torch.float32,
    ).cpu().numpy().astype(np.float32, copy=False)
    if importance.ndim != 2:
        importance = np.squeeze(importance)
    if importance.shape != (TILE, TILE):
        raise RuntimeError(f"Unexpected MONAI importance-map shape: {importance.shape}")

    accumulator = np.zeros((1, 3, image_height, image_width), dtype=np.float64)
    weight_sum = np.zeros((1, 1, image_height, image_width), dtype=np.float64)

    for record in sorted(records, key=lambda r: r["window_index"]):
        stem = f"{record['image_index']:02d}_{record['window_index']:02d}_{record['sample_id']}"
        logits = load_kneron_output(output_dir / f"{stem}.npy")
        probs = softmax_np(logits)
        y0 = int(record["y"])
        x0 = int(record["x"])
        y1 = y0 + TILE
        x1 = x0 + TILE
        accumulator[:, :, y0:y1, x0:x1] += probs * importance[None, None, :, :]
        weight_sum[:, :, y0:y1, x0:x1] += importance[None, None, :, :]

    if np.any(weight_sum <= 0):
        raise RuntimeError("Stitching produced uncovered pixels")
    return (accumulator / weight_sum).astype(np.float32)


def load_frozen_1024(sample_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    prob_path = sample_dir / "probabilities_fp32.npz"
    seg_path = sample_dir / "segmentation_class_ids.png"
    if not prob_path.is_file() or not seg_path.is_file():
        raise FileNotFoundError(f"Frozen 1024 baseline missing: {sample_dir}")
    with np.load(prob_path) as payload:
        probs = payload["probabilities"].astype(np.float32, copy=False)
    if probs.shape != (3, 1024, 1024):
        raise RuntimeError(f"Unexpected frozen probability shape: {probs.shape}")
    seg = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
    if seg is None or seg.shape != (1024, 1024):
        raise RuntimeError(f"Unexpected frozen segmentation: {seg_path}")
    return probs[None, ...], seg.astype(np.uint8, copy=False)


def prefixed(prefix: str, values: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    default_data_root = Path(
        os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data")
    )
    default_model_root = Path(
        os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models")
    )
    int8_root = (
        default_model_root / "dnai" / "unet_mobileone_s1" / "kneron" / "int8_512"
    )
    parser.add_argument(
        "--validation-manifest",
        default=str(int8_root / "validation_512_overlap50" / "validation_windows_manifest.json"),
    )
    parser.add_argument(
        "--bie-run-dir",
        default=str(int8_root / "bie_validation"),
    )
    parser.add_argument(
        "--output",
        default=str(int8_root / "biological_fidelity"),
    )
    parser.add_argument("--data-root", default=str(default_data_root))
    parser.add_argument("--limit-images", type=int, default=None)
    args = parser.parse_args()

    validation_manifest_path = Path(args.validation_manifest)
    bie_run_dir = Path(args.bie_run_dir)
    floating_dir = bie_run_dir / "floating_outputs"
    fixed_dir = bie_run_dir / "bie_outputs"
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    dataset_root = data_root / "dnai_public_v2" / "extracted" / "zenodo"
    panel_root = data_root / "dnai_public_v2" / "r1_intergrader20"
    panel_manifest_path = panel_root / "manifest.json"
    baseline_root = panel_root / "r1_s1_strict_fp32_preprocessed"

    validation_manifest = json.loads(validation_manifest_path.read_text(encoding="utf-8"))
    if validation_manifest.get("dnai_commit") != DNAI_COMMIT:
        raise RuntimeError("Validation manifest/code revision mismatch")
    if validation_manifest.get("overlap") != OVERLAP:
        raise RuntimeError(f"Expected overlap {OVERLAP}")

    panel_manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    if panel_manifest.get("dnai_commit") != DNAI_COMMIT:
        raise RuntimeError("Panel manifest/code revision mismatch")
    panel_by_sample = {r["sample_id"]: r for r in panel_manifest["records"]}

    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in validation_manifest["records"]:
        grouped[int(record["image_index"])].append(record)

    complete_groups = []
    for image_index in sorted(grouped):
        records = grouped[image_index]
        if len(records) != 9:
            continue
        all_present = True
        for record in records:
            stem = f"{record['image_index']:02d}_{record['window_index']:02d}_{record['sample_id']}"
            if not (floating_dir / f"{stem}.npy").is_file():
                all_present = False
            if not (fixed_dir / f"{stem}.npy").is_file():
                all_present = False
        if all_present:
            complete_groups.append((image_index, records))

    if args.limit_images is not None:
        complete_groups = complete_groups[: args.limit_images]
    if not complete_groups:
        raise RuntimeError(
            "No complete image has all nine floating and BIE outputs. "
            f"Checked {floating_dir} and {fixed_dir}."
        )

    print("Radiation Edge AI - stitched DNAi INT8 biological fidelity")
    print(f"Complete images available: {len(complete_groups)}")
    print("Stitching: 512x512, overlap=0.50, Gaussian, sigma_scale=0.125")
    print("Comparisons: frozen 1024 FP32 vs Kneron floating 512 vs INT8 BIE 512 vs H1-H4")
    print("")

    image_rows: list[dict] = []
    human_rows: list[dict] = []

    for position, (_image_index, records) in enumerate(complete_groups, start=1):
        sample_id = records[0]["sample_id"]
        panel_record = panel_by_sample.get(sample_id)
        if panel_record is None:
            raise RuntimeError(f"Sample not found in panel manifest: {sample_id}")

        fp_probs = stitch_probabilities(records, floating_dir, "floating")
        bie_probs = stitch_probabilities(records, fixed_dir, "bie")
        baseline_probs, baseline_seg = load_frozen_1024(baseline_root / sample_id)

        fp_tensor = torch.from_numpy(fp_probs)
        bie_tensor = torch.from_numpy(bie_probs)
        fp_seg = probas_to_segmentation(fp_tensor)
        bie_seg = probas_to_segmentation(bie_tensor)

        baseline_fibers = refine_segmentation(baseline_seg).valid_copy()
        fp_fibers = refine_segmentation(fp_seg).valid_copy()
        bie_fibers = refine_segmentation(bie_seg).valid_copy()

        prob_metrics = array_metrics(fp_probs, bie_probs)
        fp_bie_argmax = float(
            np.mean(fp_probs.argmax(axis=1) == bie_probs.argmax(axis=1))
        )
        fp_bie_seg = segmentation_metrics(fp_seg, bie_seg)
        fp_bie_fibers = match_fibers(bie_fibers, fp_fibers)
        baseline_fp_seg = segmentation_metrics(baseline_seg, fp_seg)
        baseline_bie_seg = segmentation_metrics(baseline_seg, bie_seg)

        sample_dir = output_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sample_dir / "floating_stitched_probabilities.npz", probabilities=fp_probs[0])
        np.savez_compressed(sample_dir / "bie_stitched_probabilities.npz", probabilities=bie_probs[0])
        cv2.imwrite(str(sample_dir / "floating_segmentation_class_ids.png"), fp_seg.astype(np.uint8))
        cv2.imwrite(str(sample_dir / "bie_segmentation_class_ids.png"), bie_seg.astype(np.uint8))

        row = {
            "sample_id": sample_id,
            "key": records[0]["key"],
            "fp_vs_bie_probability_max_abs_error": prob_metrics["max_abs_error"],
            "fp_vs_bie_probability_mean_abs_error": prob_metrics["mean_abs_error"],
            "fp_vs_bie_probability_rmse": prob_metrics["rmse"],
            "fp_vs_bie_argmax_agreement": fp_bie_argmax,
            **prefixed("fp_vs_bie_seg", fp_bie_seg),
            "baseline_1024_valid_fibers": len(baseline_fibers),
            "floating_512_valid_fibers": len(fp_fibers),
            "bie_512_valid_fibers": len(bie_fibers),
            "bie_minus_floating_fiber_count": len(bie_fibers) - len(fp_fibers),
            "bie_minus_baseline_fiber_count": len(bie_fibers) - len(baseline_fibers),
            **prefixed("bie_vs_floating_fiber", fp_bie_fibers),
            "floating_mean_ratio": finite_mean(fp_fibers.ratios),
            "bie_mean_ratio": finite_mean(bie_fibers.ratios),
            **prefixed("baseline1024_vs_floating512_seg", baseline_fp_seg),
            **prefixed("baseline1024_vs_bie512_seg", baseline_bie_seg),
        }
        image_rows.append(row)

        for grader in GRADERS:
            annotation = dataset_root / panel_record[f"{grader}_annotation"]
            if sha256_file(annotation) != panel_record[f"{grader}_sha256"]:
                raise RuntimeError(f"Annotation checksum mismatch: {annotation}")
            human_seg = read_human_mask(annotation)
            human_fibers = refine_segmentation(human_seg).valid_copy()

            for condition, seg, fibers in (
                ("baseline1024", baseline_seg, baseline_fibers),
                ("floating512", fp_seg, fp_fibers),
                ("bie512", bie_seg, bie_fibers),
            ):
                human_rows.append(
                    {
                        "sample_id": sample_id,
                        "key": records[0]["key"],
                        "grader": grader,
                        "condition": condition,
                        "foreground_dice": binary_dice(seg > 0, human_seg > 0),
                        "detection_f1": match_fibers(fibers, human_fibers)["f1"],
                        "ai_valid_fibers": len(fibers),
                        "human_valid_fibers": len(human_fibers),
                    }
                )

        print(
            f"[{position:02d}/{len(complete_groups):02d}] {sample_id} "
            f"FP/BIE segDice={fp_bie_seg['dice_fg']:.6f} "
            f"argmax={fp_bie_argmax:.8f} "
            f"fibers 1024/FP512/BIE={len(baseline_fibers)}/{len(fp_fibers)}/{len(bie_fibers)} "
            f"fiberMatchF1={fp_bie_fibers['f1']:.4f}"
        )

    per_image_csv = output_root / "per_image.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(image_rows)

    human_csv = output_root / "ai_vs_human.csv"
    with human_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(human_rows[0].keys()))
        writer.writeheader()
        writer.writerows(human_rows)

    def condition_mean(condition: str, key: str) -> float:
        vals = [float(r[key]) for r in human_rows if r["condition"] == condition]
        return float(np.mean(vals))

    summary = {
        "dnai_commit": DNAI_COMMIT,
        "deployment": {
            "tile": TILE,
            "overlap": OVERLAP,
            "blend": "gaussian",
            "sigma_scale": SIGMA_SCALE,
        },
        "n_complete_images": len(image_rows),
        "fp_vs_bie": {
            "mean_argmax_agreement": float(np.mean([r["fp_vs_bie_argmax_agreement"] for r in image_rows])),
            "min_argmax_agreement": float(np.min([r["fp_vs_bie_argmax_agreement"] for r in image_rows])),
            "mean_foreground_dice": float(np.mean([r["fp_vs_bie_seg_dice_fg"] for r in image_rows])),
            "min_foreground_dice": float(np.min([r["fp_vs_bie_seg_dice_fg"] for r in image_rows])),
            "fiber_count_exact_fraction": float(np.mean([r["bie_minus_floating_fiber_count"] == 0 for r in image_rows])),
            "mean_fiber_match_f1": float(np.mean([r["bie_vs_floating_fiber_f1"] for r in image_rows])),
            "mean_fiber_ratio_mae": finite_mean(r["bie_vs_floating_fiber_ratio_mae"] for r in image_rows),
            "mean_fiber_total_length_mae_um": finite_mean(r["bie_vs_floating_fiber_total_length_mae_um"] for r in image_rows),
        },
        "sanity_1024_to_floating512": {
            "mean_foreground_dice": float(np.mean([r["baseline1024_vs_floating512_seg_dice_fg"] for r in image_rows])),
            "min_foreground_dice": float(np.min([r["baseline1024_vs_floating512_seg_dice_fg"] for r in image_rows])),
        },
        "ai_vs_human": {
            "baseline1024_foreground_dice": condition_mean("baseline1024", "foreground_dice"),
            "floating512_foreground_dice": condition_mean("floating512", "foreground_dice"),
            "bie512_foreground_dice": condition_mean("bie512", "foreground_dice"),
            "baseline1024_detection_f1": condition_mean("baseline1024", "detection_f1"),
            "floating512_detection_f1": condition_mean("floating512", "detection_f1"),
            "bie512_detection_f1": condition_mean("bie512", "detection_f1"),
        },
        "per_image_csv": str(per_image_csv),
        "ai_vs_human_csv": str(human_csv),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fp_bie = summary["fp_vs_bie"]
    humans = summary["ai_vs_human"]
    sanity = summary["sanity_1024_to_floating512"]

    print("")
    print("DNAI INT8 BIOLOGICAL FIDELITY EVALUATION COMPLETE: YES")
    print(f"Complete images: {len(image_rows)}")
    print("")
    print("[FP 512 vs INT8 BIE 512]")
    print(f"mean/min stitched argmax agreement: {fp_bie['mean_argmax_agreement']:.8f} / {fp_bie['min_argmax_agreement']:.8f}")
    print(f"mean/min foreground Dice: {fp_bie['mean_foreground_dice']:.8f} / {fp_bie['min_foreground_dice']:.8f}")
    print(f"fiber-count exact fraction: {fp_bie['fiber_count_exact_fraction']:.3f}")
    print(f"mean fiber-match F1: {fp_bie['mean_fiber_match_f1']:.4f}")
    print(f"mean matched-fiber ratio MAE: {fp_bie['mean_fiber_ratio_mae']}")
    print(f"mean matched-fiber total-length MAE: {fp_bie['mean_fiber_total_length_mae_um']} um")
    print("")
    print("[Stitching sanity: frozen 1024 vs Kneron floating 512]")
    print(f"mean/min foreground Dice: {sanity['mean_foreground_dice']:.8f} / {sanity['min_foreground_dice']:.8f}")
    print("")
    print("[AI vs human]")
    print(
        "foreground Dice: "
        f"1024={humans['baseline1024_foreground_dice']:.4f} "
        f"FP512={humans['floating512_foreground_dice']:.4f} "
        f"BIE512={humans['bie512_foreground_dice']:.4f}"
    )
    print(
        "detection F1: "
        f"1024={humans['baseline1024_detection_f1']:.4f} "
        f"FP512={humans['floating512_detection_f1']:.4f} "
        f"BIE512={humans['bie512_detection_f1']:.4f}"
    )
    print(f"Summary: {summary_path}")
    print(f"Per-image: {per_image_csv}")
    print(f"AI-vs-human: {human_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
