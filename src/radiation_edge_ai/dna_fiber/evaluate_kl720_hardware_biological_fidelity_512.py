"""Evaluate full physical-KL720 DNA-fiber biological fidelity on the frozen 20-image panel.

Run this script in the DNAi Python environment after ``run_kl720_dnai_panel.py``
has generated all 180 physical 512x512 outputs. The script uses the same frozen
validation tensors to build a cached floating-point reference with the verified
Kneron-optimized 512 ONNX model, then stitches both reference and physical
outputs with the frozen 50%-overlap Gaussian policy.

The main question is biological rather than merely numerical: does the physical
KL720 preserve fiber detection and quantitative endpoints relative to the
floating 512 reference, the frozen strict-FP32 1024 baseline, and H1-H4 human
annotations?

The script also summarizes human support for object-level disagreements. Human
annotations are descriptive evidence, not infallible ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

from dnafiber.inference import probas_to_segmentation
from dnafiber.postprocess import refine_segmentation

try:
    from .evaluate_bie_biological_fidelity_512 import (
        DNAI_COMMIT,
        GRADERS,
        MATCH_IOU,
        OVERLAP,
        SIGMA_SCALE,
        TILE,
        array_metrics,
        binary_dice,
        finite_mean,
        load_frozen_1024,
        match_fibers,
        read_human_mask,
        segmentation_metrics,
        sha256_file,
        softmax_np,
        stitch_probabilities,
    )
except ImportError:
    from evaluate_bie_biological_fidelity_512 import (
        DNAI_COMMIT,
        GRADERS,
        MATCH_IOU,
        OVERLAP,
        SIGMA_SCALE,
        TILE,
        array_metrics,
        binary_dice,
        finite_mean,
        load_frozen_1024,
        match_fibers,
        read_human_mask,
        segmentation_metrics,
        sha256_file,
        softmax_np,
        stitch_probabilities,
    )

EXPECTED_NEf_SHA256 = "4b3dfec9a61c99e186dd4b8482fa5b06e6a4958f325ed4a0db0546f1dcab2bfc"
EXPECTED_ONNX_SHA256 = "a901d1b309a9a0e5026febd5070252787e4b110a2a7d2828e16a19364d6094d0"
EXPECTED_TENSOR = (1, 3, 512, 512)
EXPECTED_PANEL_WINDOWS = 180
PIXEL_SIZE_UM = 0.26


def make_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_shape = list(session.get_inputs()[0].shape)
    output_shape = list(session.get_outputs()[0].shape)
    if input_shape != list(EXPECTED_TENSOR):
        raise RuntimeError(f"Optimized ONNX input shape {input_shape}; expected {EXPECTED_TENSOR}")
    if output_shape != list(EXPECTED_TENSOR):
        raise RuntimeError(f"Optimized ONNX output shape {output_shape}; expected {EXPECTED_TENSOR}")
    return session


def stem_for(record: dict) -> str:
    return f"{int(record['image_index']):02d}_{int(record['window_index']):02d}_{record['sample_id']}"


def valid_array(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        array = np.load(path, allow_pickle=False)
    except Exception:
        return False
    return (
        array.shape == EXPECTED_TENSOR
        and array.dtype == np.float32
        and bool(np.isfinite(array).all())
    )


def build_floating_reference(
    session: ort.InferenceSession,
    records: list[dict],
    validation_root: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    per_window = []
    ran = 0
    reused = 0
    started = time.perf_counter()

    for position, record in enumerate(records, start=1):
        stem = stem_for(record)
        output_path = output_dir / f"{stem}.npy"
        if not overwrite and valid_array(output_path):
            reused += 1
            continue

        input_path = validation_root / record["tensor"]
        array = np.load(input_path, allow_pickle=False)
        if array.shape != EXPECTED_TENSOR or array.dtype != np.float32:
            raise RuntimeError(f"Invalid validation tensor {array.shape} {array.dtype}: {input_path}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"Non-finite validation tensor: {input_path}")
        array = np.ascontiguousarray(array, dtype=np.float32)

        t0 = time.perf_counter()
        result = session.run([output_name], {input_name: array})[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        result = np.asarray(result, dtype=np.float32)
        if result.shape != EXPECTED_TENSOR or not np.isfinite(result).all():
            raise RuntimeError(f"Invalid ONNX reference output {result.shape}: {stem}")
        np.save(output_path, np.ascontiguousarray(result), allow_pickle=False)
        per_window.append(elapsed_ms)
        ran += 1

        if position == 1 or position % 30 == 0 or position == len(records):
            print(f"  floating reference [{position:03d}/{len(records):03d}] {record['sample_id']} w{record['window_index']} {elapsed_ms:.1f} ms")

    missing = [stem_for(r) for r in records if not valid_array(output_dir / f"{stem_for(r)}.npy")]
    if missing:
        raise RuntimeError(f"Floating reference incomplete; missing {len(missing)} windows")

    return {
        "ran": ran,
        "reused": reused,
        "elapsed_seconds": time.perf_counter() - started,
        "mean_inference_ms": float(np.mean(per_window)) if per_window else None,
    }


def greedy_matches(reference, candidate, threshold: float = MATCH_IOU) -> list[tuple[int, int, float]]:
    ref = list(reference)
    cand = list(candidate)
    candidates: list[tuple[float, int, int]] = []
    for i, r in enumerate(ref):
        for j, c in enumerate(cand):
            iou = float(r.bbox_iou(c))
            if iou >= threshold:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True)

    used_ref: set[int] = set()
    used_cand: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, i, j in candidates:
        if i in used_ref or j in used_cand:
            continue
        used_ref.add(i)
        used_cand.add(j)
        matches.append((i, j, iou))
    return matches


def support_label(count: int) -> str:
    if count >= 3:
        return "human-supported"
    if count >= 1:
        return "ambiguous"
    return "no-human-match"


def human_support_for_fibers(fibers, human_fibers: dict[str, list]) -> list[dict]:
    fibers = list(fibers)
    by_grader: dict[str, dict[int, float]] = {}
    for grader, h_fibers in human_fibers.items():
        matches = greedy_matches(fibers, h_fibers)
        by_grader[grader] = {i: iou for i, _j, iou in matches}

    rows = []
    for i, fiber in enumerate(fibers):
        grader_ious = {grader: by_grader[grader].get(i) for grader in GRADERS}
        count = sum(value is not None for value in grader_ious.values())
        rows.append(
            {
                "index": i,
                "length_um": float(fiber.length) * PIXEL_SIZE_UM,
                "ratio": float(fiber.ratio),
                "human_support_count": count,
                "human_support_label": support_label(count),
                "grader_ious": grader_ious,
            }
        )
    return rows


def prefixed(prefix: str, values: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def condition_mean(rows: list[dict], condition: str, key: str, sample_ids: set[str] | None = None) -> float:
    values = [
        float(row[key])
        for row in rows
        if row["condition"] == condition and (sample_ids is None or row["sample_id"] in sample_ids)
    ]
    if not values:
        return float("nan")
    return float(np.mean(values))


def aggregate_image_metrics(rows: list[dict], sample_ids: set[str] | None = None) -> dict:
    chosen = [row for row in rows if sample_ids is None or row["sample_id"] in sample_ids]
    if not chosen:
        return {}
    return {
        "n_images": len(chosen),
        "mean_argmax_agreement": float(np.mean([r["floating_vs_physical_argmax_agreement"] for r in chosen])),
        "min_argmax_agreement": float(np.min([r["floating_vs_physical_argmax_agreement"] for r in chosen])),
        "mean_foreground_dice": float(np.mean([r["floating_vs_physical_seg_dice_fg"] for r in chosen])),
        "min_foreground_dice": float(np.min([r["floating_vs_physical_seg_dice_fg"] for r in chosen])),
        "fiber_count_exact_fraction": float(np.mean([r["physical_minus_floating_fiber_count"] == 0 for r in chosen])),
        "mean_fiber_match_f1": float(np.mean([r["physical_vs_floating_fiber_f1"] for r in chosen])),
        "mean_fiber_ratio_mae": finite_mean(r["physical_vs_floating_fiber_ratio_mae"] for r in chosen),
        "mean_fiber_total_length_mae_um": finite_mean(r["physical_vs_floating_fiber_total_length_mae_um"] for r in chosen),
        "mean_1024_vs_floating512_foreground_dice": float(np.mean([r["baseline1024_vs_floating512_seg_dice_fg"] for r in chosen])),
        "min_1024_vs_floating512_foreground_dice": float(np.min([r["baseline1024_vs_floating512_seg_dice_fg"] for r in chosen])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    default_data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    default_model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    kneron_root = default_model_root / "dnai" / "unet_mobileone_s1" / "kneron"
    v3_root = kneron_root / "int8_512_v3_pct100"
    parser.add_argument(
        "--validation-manifest",
        default=str(kneron_root / "int8_512" / "validation_512_overlap50" / "validation_windows_manifest.json"),
    )
    parser.add_argument(
        "--optimized-onnx",
        default=str(kneron_root / "unet_mobileone_s1_512x512_opset11_optimized.onnx"),
    )
    parser.add_argument("--hardware-panel", default=str(v3_root / "hardware_panel"))
    parser.add_argument("--output", default=str(v3_root / "hardware_biological_fidelity"))
    parser.add_argument("--data-root", default=str(default_data_root))
    parser.add_argument("--overwrite-floating", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.validation_manifest).resolve()
    onnx_path = Path(args.optimized_onnx).resolve()
    hardware_panel = Path(args.hardware_panel).resolve()
    physical_dir = hardware_panel / "physical_outputs"
    hardware_summary_path = hardware_panel / "summary.json"
    output_root = Path(args.output).resolve()
    floating_dir = output_root / "floating_reference_outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    for path in (manifest_path, onnx_path, hardware_summary_path, physical_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    onnx_sha = sha256_file(onnx_path)
    if onnx_sha.lower() != EXPECTED_ONNX_SHA256:
        raise RuntimeError(f"Unexpected optimized ONNX SHA256: {onnx_sha}")

    hardware_summary = json.loads(hardware_summary_path.read_text(encoding="utf-8"))
    if str(hardware_summary.get("nef_sha256", "")).lower() != EXPECTED_NEf_SHA256:
        raise RuntimeError(
            "Hardware panel was not generated from the frozen verified v3 NEF: "
            f"{hardware_summary.get('nef_sha256')}"
        )
    if not hardware_summary.get("complete"):
        raise RuntimeError("Hardware panel summary is not complete")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dnai_commit") != DNAI_COMMIT:
        raise RuntimeError("Validation manifest/code revision mismatch")
    if tuple(manifest.get("tile", ())) != EXPECTED_TENSOR:
        raise RuntimeError(f"Validation tile mismatch: {manifest.get('tile')}")
    if float(manifest.get("overlap", -1)) != OVERLAP:
        raise RuntimeError(f"Expected frozen overlap={OVERLAP}")
    records = list(manifest.get("records", []))
    if len(records) != EXPECTED_PANEL_WINDOWS:
        raise RuntimeError(f"Expected {EXPECTED_PANEL_WINDOWS} frozen windows, got {len(records)}")

    missing_physical = [stem_for(r) for r in records if not valid_array(physical_dir / f"{stem_for(r)}.npy")]
    if missing_physical:
        raise RuntimeError(f"Physical panel incomplete; missing {len(missing_physical)} outputs")

    data_root = Path(args.data_root).resolve()
    dataset_root = data_root / "dnai_public_v2" / "extracted" / "zenodo"
    panel_root = data_root / "dnai_public_v2" / "r1_intergrader20"
    panel_manifest_path = panel_root / "manifest.json"
    baseline_root = panel_root / "r1_s1_strict_fp32_preprocessed"
    panel_manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    if panel_manifest.get("dnai_commit") != DNAI_COMMIT:
        raise RuntimeError("Panel manifest/code revision mismatch")
    panel_by_sample = {row["sample_id"]: row for row in panel_manifest["records"]}

    print("Radiation Edge AI - full physical KL720 DNAi biological fidelity")
    print(f"Physical windows: {len(records)}")
    print(f"Physical panel: {physical_dir}")
    print(f"Optimized FP512 ONNX: {onnx_path}")
    print(f"ONNX SHA256: {onnx_sha}")
    print(f"Hardware NEF SHA256: {hardware_summary['nef_sha256']}")
    print(
        "Physical timing: "
        f"mean={hardware_summary.get('mean_hardware_send_receive_ms')} ms/window, "
        f"throughput={hardware_summary.get('estimated_device_windows_per_second')} windows/s"
    )
    print("")

    print("[Build/reuse floating 512 ONNX reference]")
    session = make_session(onnx_path)
    floating_build = build_floating_reference(
        session,
        records,
        manifest_path.parent,
        floating_dir,
        args.overwrite_floating,
    )
    print(
        f"  complete: ran={floating_build['ran']} reused={floating_build['reused']} "
        f"mean={floating_build['mean_inference_ms']} ms"
    )
    print("")

    # Direct per-window numerical comparison before stitching.
    window_rows = []
    for record in records:
        stem = stem_for(record)
        floating = np.load(floating_dir / f"{stem}.npy", allow_pickle=False)
        physical = np.load(physical_dir / f"{stem}.npy", allow_pickle=False)
        m = array_metrics(floating, physical)
        floating_class = floating.argmax(axis=1)
        physical_class = physical.argmax(axis=1)
        disagree = int(np.count_nonzero(floating_class != physical_class))
        total = int(floating_class.size)
        window_rows.append(
            {
                "image_index": int(record["image_index"]),
                "window_index": int(record["window_index"]),
                "sample_id": record["sample_id"],
                "max_abs_error": m["max_abs_error"],
                "mean_abs_error": m["mean_abs_error"],
                "rmse": m["rmse"],
                "argmax_agreement": float(np.mean(floating_class == physical_class)),
                "disagreement_pixels": disagree,
                "total_pixels": total,
            }
        )

    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        grouped[int(record["image_index"])].append(record)
    if len(grouped) != 20 or any(len(group) != 9 for group in grouped.values()):
        raise RuntimeError("Frozen validation panel is not 20 complete 9-window images")

    image_rows: list[dict] = []
    human_rows: list[dict] = []
    mismatch_rows: list[dict] = []
    nonempty_samples: set[str] = set()

    for position, image_index in enumerate(sorted(grouped), start=1):
        image_records = grouped[image_index]
        sample_id = str(image_records[0]["sample_id"])
        panel_record = panel_by_sample.get(sample_id)
        if panel_record is None:
            raise RuntimeError(f"Frozen sample not found in panel manifest: {sample_id}")

        floating_probs = stitch_probabilities(image_records, floating_dir, "floating")
        physical_probs = stitch_probabilities(image_records, physical_dir, "physical")
        baseline_probs, baseline_seg = load_frozen_1024(baseline_root / sample_id)

        floating_seg = probas_to_segmentation(torch.from_numpy(floating_probs))
        physical_seg = probas_to_segmentation(torch.from_numpy(physical_probs))
        baseline_fibers = list(refine_segmentation(baseline_seg).valid_copy())
        floating_fibers = list(refine_segmentation(floating_seg).valid_copy())
        physical_fibers = list(refine_segmentation(physical_seg).valid_copy())

        prob_metrics = array_metrics(floating_probs, physical_probs)
        floating_physical_argmax = float(
            np.mean(floating_probs.argmax(axis=1) == physical_probs.argmax(axis=1))
        )
        floating_physical_seg = segmentation_metrics(floating_seg, physical_seg)
        physical_floating_fibers = match_fibers(physical_fibers, floating_fibers)
        baseline_floating_seg = segmentation_metrics(baseline_seg, floating_seg)
        baseline_physical_seg = segmentation_metrics(baseline_seg, physical_seg)

        human_fibers: dict[str, list] = {}
        any_human_fibers = False
        for grader in GRADERS:
            annotation = dataset_root / panel_record[f"{grader}_annotation"]
            if sha256_file(annotation) != panel_record[f"{grader}_sha256"]:
                raise RuntimeError(f"Annotation checksum mismatch: {annotation}")
            human_seg = read_human_mask(annotation)
            h_fibers = list(refine_segmentation(human_seg).valid_copy())
            human_fibers[grader] = h_fibers
            any_human_fibers = any_human_fibers or bool(h_fibers)

            for condition, seg, fibers in (
                ("baseline1024", baseline_seg, baseline_fibers),
                ("floating512", floating_seg, floating_fibers),
                ("physicalKL720", physical_seg, physical_fibers),
            ):
                human_rows.append(
                    {
                        "sample_id": sample_id,
                        "key": image_records[0]["key"],
                        "grader": grader,
                        "condition": condition,
                        "foreground_dice": binary_dice(seg > 0, human_seg > 0),
                        "detection_f1": match_fibers(fibers, h_fibers)["f1"],
                        "ai_valid_fibers": len(fibers),
                        "human_valid_fibers": len(h_fibers),
                    }
                )

        if baseline_fibers or floating_fibers or physical_fibers or any_human_fibers:
            nonempty_samples.add(sample_id)

        direct_matches = greedy_matches(floating_fibers, physical_fibers)
        matched_fp = {i for i, _j, _iou in direct_matches}
        matched_hw = {j for _i, j, _iou in direct_matches}
        fp_support = human_support_for_fibers(floating_fibers, human_fibers)
        hw_support = human_support_for_fibers(physical_fibers, human_fibers)

        for i, row in enumerate(fp_support):
            if i in matched_fp:
                continue
            mismatch_rows.append(
                {
                    "sample_id": sample_id,
                    "kind": "floating512_missed_by_physical",
                    "fiber_index": i,
                    "length_um": row["length_um"],
                    "ratio": row["ratio"],
                    "human_support_count": row["human_support_count"],
                    "human_support_label": row["human_support_label"],
                }
            )
        for j, row in enumerate(hw_support):
            if j in matched_hw:
                continue
            mismatch_rows.append(
                {
                    "sample_id": sample_id,
                    "kind": "physical_only",
                    "fiber_index": j,
                    "length_um": row["length_um"],
                    "ratio": row["ratio"],
                    "human_support_count": row["human_support_count"],
                    "human_support_label": row["human_support_label"],
                }
            )

        sample_dir = output_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sample_dir / "floating512_stitched_probabilities.npz",
            probabilities=floating_probs[0],
        )
        np.savez_compressed(
            sample_dir / "physical_kl720_stitched_probabilities.npz",
            probabilities=physical_probs[0],
        )
        cv2.imwrite(str(sample_dir / "floating512_segmentation_class_ids.png"), floating_seg.astype(np.uint8))
        cv2.imwrite(str(sample_dir / "physical_kl720_segmentation_class_ids.png"), physical_seg.astype(np.uint8))

        image_rows.append(
            {
                "sample_id": sample_id,
                "key": image_records[0]["key"],
                "floating_vs_physical_probability_max_abs_error": prob_metrics["max_abs_error"],
                "floating_vs_physical_probability_mean_abs_error": prob_metrics["mean_abs_error"],
                "floating_vs_physical_probability_rmse": prob_metrics["rmse"],
                "floating_vs_physical_argmax_agreement": floating_physical_argmax,
                **prefixed("floating_vs_physical_seg", floating_physical_seg),
                "baseline_1024_valid_fibers": len(baseline_fibers),
                "floating_512_valid_fibers": len(floating_fibers),
                "physical_kl720_valid_fibers": len(physical_fibers),
                "physical_minus_floating_fiber_count": len(physical_fibers) - len(floating_fibers),
                "physical_minus_baseline_fiber_count": len(physical_fibers) - len(baseline_fibers),
                **prefixed("physical_vs_floating_fiber", physical_floating_fibers),
                "floating_mean_ratio": finite_mean(f.ratio for f in floating_fibers),
                "physical_mean_ratio": finite_mean(f.ratio for f in physical_fibers),
                **prefixed("baseline1024_vs_floating512_seg", baseline_floating_seg),
                **prefixed("baseline1024_vs_physical_seg", baseline_physical_seg),
                "reference_nonempty": sample_id in nonempty_samples,
            }
        )

        print(
            f"[{position:02d}/20] {sample_id} "
            f"FP512/HW segDice={floating_physical_seg['dice_fg']:.6f} "
            f"argmax={floating_physical_argmax:.8f} "
            f"fibers 1024/FP512/HW={len(baseline_fibers)}/{len(floating_fibers)}/{len(physical_fibers)} "
            f"fiberMatchF1={physical_floating_fibers['f1']:.4f}"
        )

    # Persist tables.
    window_csv = output_root / "per_window_numerical.csv"
    with window_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(window_rows[0].keys()))
        writer.writeheader()
        writer.writerows(window_rows)

    image_csv = output_root / "per_image.csv"
    with image_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(image_rows)

    human_csv = output_root / "ai_vs_human.csv"
    with human_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(human_rows[0].keys()))
        writer.writeheader()
        writer.writerows(human_rows)

    mismatch_csv = output_root / "fiber_disagreements_human_support.csv"
    if mismatch_rows:
        with mismatch_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(mismatch_rows[0].keys()))
            writer.writeheader()
            writer.writerows(mismatch_rows)
    else:
        mismatch_csv.write_text("No unmatched floating512/physicalKL720 fibers.\n", encoding="utf-8")

    all_metrics = aggregate_image_metrics(image_rows)
    nonempty_metrics = aggregate_image_metrics(image_rows, nonempty_samples)

    mismatch_counts = {}
    for kind in ("floating512_missed_by_physical", "physical_only"):
        counter = Counter(
            int(row["human_support_count"])
            for row in mismatch_rows
            if row["kind"] == kind
        )
        mismatch_counts[kind] = {str(support): int(counter.get(support, 0)) for support in range(5)}

    window_summary = {
        "mean_argmax_agreement": float(np.mean([r["argmax_agreement"] for r in window_rows])),
        "min_argmax_agreement": float(np.min([r["argmax_agreement"] for r in window_rows])),
        "disagreement_pixels": int(sum(r["disagreement_pixels"] for r in window_rows)),
        "total_pixels": int(sum(r["total_pixels"] for r in window_rows)),
        "max_raw_logit_abs_error": float(np.max([r["max_abs_error"] for r in window_rows])),
        "mean_raw_logit_rmse": float(np.mean([r["rmse"] for r in window_rows])),
    }

    summary = {
        "dnai_commit": DNAI_COMMIT,
        "optimized_onnx": str(onnx_path),
        "optimized_onnx_sha256": onnx_sha,
        "hardware_panel": str(hardware_panel),
        "hardware_nef_sha256": hardware_summary["nef_sha256"],
        "n_windows": len(records),
        "n_images": len(image_rows),
        "n_nonempty_images": len(nonempty_samples),
        "deployment": {
            "tile": TILE,
            "overlap": OVERLAP,
            "blend": "gaussian",
            "sigma_scale": SIGMA_SCALE,
        },
        "physical_performance": {
            "mean_host_pack_ms_per_window": hardware_summary.get("mean_host_pack_ms"),
            "mean_send_receive_ms_per_window": hardware_summary.get("mean_hardware_send_receive_ms"),
            "median_send_receive_ms_per_window": hardware_summary.get("median_hardware_send_receive_ms"),
            "windows_per_second": hardware_summary.get("estimated_device_windows_per_second"),
            "panel_wall_seconds": hardware_summary.get("elapsed_seconds"),
            "estimated_1024_fields_per_second_from_9_windows": (
                float(hardware_summary["estimated_device_windows_per_second"]) / 9.0
                if hardware_summary.get("estimated_device_windows_per_second")
                else None
            ),
        },
        "floating_reference_build": floating_build,
        "per_window_floating_vs_physical": window_summary,
        "per_image_all20": all_metrics,
        "per_image_nonempty": nonempty_metrics,
        "ai_vs_human_all20": {
            "baseline1024_foreground_dice": condition_mean(human_rows, "baseline1024", "foreground_dice"),
            "floating512_foreground_dice": condition_mean(human_rows, "floating512", "foreground_dice"),
            "physicalKL720_foreground_dice": condition_mean(human_rows, "physicalKL720", "foreground_dice"),
            "baseline1024_detection_f1": condition_mean(human_rows, "baseline1024", "detection_f1"),
            "floating512_detection_f1": condition_mean(human_rows, "floating512", "detection_f1"),
            "physicalKL720_detection_f1": condition_mean(human_rows, "physicalKL720", "detection_f1"),
        },
        "ai_vs_human_nonempty": {
            "baseline1024_foreground_dice": condition_mean(human_rows, "baseline1024", "foreground_dice", nonempty_samples),
            "floating512_foreground_dice": condition_mean(human_rows, "floating512", "foreground_dice", nonempty_samples),
            "physicalKL720_foreground_dice": condition_mean(human_rows, "physicalKL720", "foreground_dice", nonempty_samples),
            "baseline1024_detection_f1": condition_mean(human_rows, "baseline1024", "detection_f1", nonempty_samples),
            "floating512_detection_f1": condition_mean(human_rows, "floating512", "detection_f1", nonempty_samples),
            "physicalKL720_detection_f1": condition_mean(human_rows, "physicalKL720", "detection_f1", nonempty_samples),
        },
        "unmatched_fiber_human_support_counts": mismatch_counts,
        "nonempty_sample_ids": sorted(nonempty_samples),
        "tables": {
            "per_window": str(window_csv),
            "per_image": str(image_csv),
            "ai_vs_human": str(human_csv),
            "fiber_disagreements": str(mismatch_csv),
        },
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("PHYSICAL KL720 BIOLOGICAL FIDELITY EVALUATION COMPLETE: YES")
    print(f"Images: {len(image_rows)}; non-empty: {len(nonempty_samples)}")
    print("")
    print("[Per-window optimized FP512 vs physical KL720]")
    print(
        f"mean/min argmax: {window_summary['mean_argmax_agreement']:.8f} / "
        f"{window_summary['min_argmax_agreement']:.8f}"
    )
    print(
        f"disagreement pixels: {window_summary['disagreement_pixels']}/"
        f"{window_summary['total_pixels']}"
    )
    print(f"max raw-logit abs error: {window_summary['max_raw_logit_abs_error']:.8g}")
    print("")
    print("[Stitched FP512 vs physical KL720 - all 20]")
    print(
        f"mean/min foreground Dice: {all_metrics['mean_foreground_dice']:.8f} / "
        f"{all_metrics['min_foreground_dice']:.8f}"
    )
    print(f"fiber-count exact fraction: {all_metrics['fiber_count_exact_fraction']:.3f}")
    print(f"mean fiber-match F1: {all_metrics['mean_fiber_match_f1']:.4f}")
    print(f"mean matched-fiber ratio MAE: {all_metrics['mean_fiber_ratio_mae']}")
    print(f"mean matched-fiber total-length MAE: {all_metrics['mean_fiber_total_length_mae_um']} um")
    print("")
    print("[AI vs human - all 20]")
    h = summary["ai_vs_human_all20"]
    print(
        "foreground Dice: "
        f"1024={h['baseline1024_foreground_dice']:.4f} "
        f"FP512={h['floating512_foreground_dice']:.4f} "
        f"KL720={h['physicalKL720_foreground_dice']:.4f}"
    )
    print(
        "detection F1: "
        f"1024={h['baseline1024_detection_f1']:.4f} "
        f"FP512={h['floating512_detection_f1']:.4f} "
        f"KL720={h['physicalKL720_detection_f1']:.4f}"
    )
    print("")
    print("[Unmatched-fiber human support]")
    for kind, counts in mismatch_counts.items():
        print(
            f"{kind}: "
            + ", ".join(f"{support}/4={counts[str(support)]}" for support in range(5))
        )
    print("")
    print(f"Summary: {summary_path}")
    print(f"Per-image: {image_csv}")
    print(f"AI-vs-human: {human_csv}")
    print(f"Fiber disagreements: {mismatch_csv}")
    print("")
    print(
        "NEXT GATE: decide whether v3 can be frozen for the DNA-fiber deployment case; "
        "base the decision on full-panel phenotype fidelity and human-supported object changes, "
        "not pixel metrics alone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
