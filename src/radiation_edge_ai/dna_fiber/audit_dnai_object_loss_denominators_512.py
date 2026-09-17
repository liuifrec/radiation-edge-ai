"""Denominator-aware object-loss audit for completed DNAi characterization.

NO neural inference. NO KL720 execution. NO PTQ/model selection.

This post-hoc audit reads only persisted stitched segmentations and frozen human
annotations. It reports object-level physical-vs-floating mismatch rates with
explicit denominators by length and human-support strata.

The historical characterization remains unchanged and primary. Results from this
audit are descriptive sensitivity evidence, not confirmatory acceptance gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dnafiber.postprocess import refine_segmentation

from radiation_edge_ai.audit_utils import refuse_nonempty_output_dir, sha256_file, write_csv_rows
from radiation_edge_ai.dna_fiber.evaluate_bie_biological_fidelity_512 import (
    DNAI_COMMIT,
    GRADERS,
    MATCH_IOU,
    load_frozen_1024,
    read_human_mask,
)
from radiation_edge_ai.dna_fiber.matching_sensitivity import maximum_cardinality_iou_matching

PIXEL_SIZE_UM = 0.26
EXPECTED_IMAGES = 20
EXPECTED_ONNX_SHA256 = "a901d1b309a9a0e5026febd5070252787e4b110a2a7d2828e16a19364d6094d0"
EXPECTED_NEF_SHA256 = "4b3dfec9a61c99e186dd4b8482fa5b06e6a4958f325ed4a0db0546f1dcab2bfc"
LENGTH_BINS = ("[0,10)", "[10,20)", "[20,30)", "[30,inf)")


def length_bin(length_um: float) -> str:
    if length_um < 10:
        return "[0,10)"
    if length_um < 20:
        return "[10,20)"
    if length_um < 30:
        return "[20,30)"
    return "[30,inf)"


def read_seg(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    seg = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if seg is None or seg.ndim != 2:
        raise RuntimeError(f"Invalid persisted segmentation: {path}")
    return np.asarray(seg, dtype=np.uint8)


def fibers(seg: np.ndarray) -> list[Any]:
    return list(refine_segmentation(seg).valid_copy())


def human_support(objects: Sequence[Any], human: dict[str, list[Any]]) -> list[int]:
    support = [0] * len(objects)
    for grader in GRADERS:
        matches = maximum_cardinality_iou_matching(objects, human[grader], MATCH_IOU)
        for i, _j, _iou in matches:
            support[i] += 1
    return support


def support_stratum(count: int) -> str:
    if count >= 3:
        return "support_ge3"
    if count >= 1:
        return "support_1_2"
    return "support_0"


def summarize(rows: list[dict[str, Any]], object_type: str) -> list[dict[str, Any]]:
    chosen = [row for row in rows if row["object_type"] == object_type]
    out: list[dict[str, Any]] = []
    strata = [
        ("all", lambda r: True),
        ("support_ge1", lambda r: int(r["human_support_count"]) >= 1),
        ("support_ge3", lambda r: int(r["human_support_count"]) >= 3),
        ("support_0", lambda r: int(r["human_support_count"]) == 0),
        ("support_1_2", lambda r: 1 <= int(r["human_support_count"]) <= 2),
    ]
    for bin_name in list(LENGTH_BINS) + ["ALL"]:
        base = (
            chosen if bin_name == "ALL" else [r for r in chosen if r["length_bin_um"] == bin_name]
        )
        for stratum_name, predicate in strata:
            denom = [r for r in base if predicate(r)]
            total = len(denom)
            unmatched = sum(int(r["unmatched_to_other"]) == 1 for r in denom)
            out.append(
                {
                    "object_type": object_type,
                    "length_bin_um": bin_name,
                    "human_support_stratum": stratum_name,
                    "n_total_objects": total,
                    "n_unmatched_objects": unmatched,
                    "unmatched_fraction": (unmatched / total) if total else None,
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    default_data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    default_model_root = Path(
        os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models")
    )
    kneron_root = default_model_root / "dnai" / "unet_mobileone_s1" / "kneron"
    v3_root = kneron_root / "int8_512_v3_pct100"
    parser.add_argument(
        "--characterization-dir", default=str(v3_root / "hardware_biological_fidelity")
    )
    parser.add_argument("--data-root", default=str(default_data_root))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    characterization = Path(args.characterization_dir).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    refuse_nonempty_output_dir(output_dir)

    legacy_summary_path = characterization / "summary.json"
    legacy_per_image_path = characterization / "per_image.csv"
    if not legacy_summary_path.is_file() or not legacy_per_image_path.is_file():
        raise FileNotFoundError("Completed DNAi characterization not found")

    legacy = json.loads(legacy_summary_path.read_text(encoding="utf-8"))
    if legacy.get("dnai_commit") != DNAI_COMMIT:
        raise RuntimeError("DNAi commit mismatch")
    if str(legacy.get("optimized_onnx_sha256", "")).lower() != EXPECTED_ONNX_SHA256:
        raise RuntimeError("Optimized ONNX identity mismatch")
    if str(legacy.get("hardware_nef_sha256", "")).lower() != EXPECTED_NEF_SHA256:
        raise RuntimeError("NEF identity mismatch")

    with legacy_per_image_path.open("r", newline="", encoding="utf-8-sig") as handle:
        legacy_rows = list(csv.DictReader(handle))
    if len(legacy_rows) != EXPECTED_IMAGES:
        raise RuntimeError(f"Expected 20 images, got {len(legacy_rows)}")
    sample_ids = [row["sample_id"] for row in legacy_rows]
    if len(set(sample_ids)) != EXPECTED_IMAGES:
        raise RuntimeError("Duplicate sample IDs")

    panel_root = data_root / "dnai_public_v2" / "r1_intergrader20"
    dataset_root = data_root / "dnai_public_v2" / "extracted" / "zenodo"
    panel_manifest_path = panel_root / "manifest.json"
    baseline_root = panel_root / "r1_s1_strict_fp32_preprocessed"
    panel_manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    if panel_manifest.get("dnai_commit") != DNAI_COMMIT:
        raise RuntimeError("Panel manifest DNAi commit mismatch")
    panel_by_sample = {str(row["sample_id"]): row for row in panel_manifest["records"]}

    per_fiber: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    hashes: list[dict[str, Any]] = []

    for sample_id in sample_ids:
        record = panel_by_sample.get(sample_id)
        if record is None:
            raise RuntimeError(f"Missing panel record: {sample_id}")

        floating_path = characterization / sample_id / "floating512_segmentation_class_ids.png"
        physical_path = characterization / sample_id / "physical_kl720_segmentation_class_ids.png"
        floating_seg = read_seg(floating_path)
        physical_seg = read_seg(physical_path)
        if floating_seg.shape != physical_seg.shape:
            raise RuntimeError(f"Floating/physical geometry mismatch: {sample_id}")

        _baseline_prob, baseline_seg = load_frozen_1024(baseline_root / sample_id)
        baseline_objects = fibers(baseline_seg)
        floating_objects = fibers(floating_seg)
        physical_objects = fibers(physical_seg)

        human: dict[str, list[Any]] = {}
        for grader in GRADERS:
            annotation = dataset_root / record[f"{grader}_annotation"]
            observed = sha256_file(annotation)
            expected = str(record[f"{grader}_sha256"])
            if observed != expected:
                raise RuntimeError(f"Annotation checksum mismatch: {annotation}")
            human_seg = read_human_mask(annotation)
            if human_seg.shape != floating_seg.shape:
                raise RuntimeError(f"Human geometry mismatch: {sample_id}/{grader}")
            human[grader] = fibers(human_seg)
            hashes.append(
                {
                    "sample_id": sample_id,
                    "artifact": f"{grader}_annotation",
                    "sha256": observed,
                    "path": str(annotation),
                }
            )

        matches = maximum_cardinality_iou_matching(floating_objects, physical_objects, MATCH_IOU)
        matched_fp = {i for i, _j, _iou in matches}
        matched_hw = {j for _i, j, _iou in matches}
        fp_support = human_support(floating_objects, human)
        hw_support = human_support(physical_objects, human)
        reference_defined_nonempty = bool(baseline_objects) or any(human[g] for g in GRADERS)

        per_image.append(
            {
                "sample_id": sample_id,
                "reference_defined_nonempty": int(reference_defined_nonempty),
                "n_baseline1024": len(baseline_objects),
                "n_floating512": len(floating_objects),
                "n_physical": len(physical_objects),
                "n_matched": len(matches),
                "n_floating_unmatched": len(floating_objects) - len(matches),
                "n_physical_unmatched": len(physical_objects) - len(matches),
            }
        )

        for i, fiber in enumerate(floating_objects):
            length_um = float(fiber.length) * PIXEL_SIZE_UM
            per_fiber.append(
                {
                    "sample_id": sample_id,
                    "reference_defined_nonempty": int(reference_defined_nonempty),
                    "object_type": "floating512_reference",
                    "object_index": i,
                    "length_um": length_um,
                    "length_bin_um": length_bin(length_um),
                    "ratio": float(fiber.ratio),
                    "human_support_count": fp_support[i],
                    "human_support_stratum": support_stratum(fp_support[i]),
                    "unmatched_to_other": int(i not in matched_fp),
                }
            )
        for j, fiber in enumerate(physical_objects):
            length_um = float(fiber.length) * PIXEL_SIZE_UM
            per_fiber.append(
                {
                    "sample_id": sample_id,
                    "reference_defined_nonempty": int(reference_defined_nonempty),
                    "object_type": "physical_kl720",
                    "object_index": j,
                    "length_um": length_um,
                    "length_bin_um": length_bin(length_um),
                    "ratio": float(fiber.ratio),
                    "human_support_count": hw_support[j],
                    "human_support_stratum": support_stratum(hw_support[j]),
                    "unmatched_to_other": int(j not in matched_hw),
                }
            )

        hashes.extend(
            [
                {
                    "sample_id": sample_id,
                    "artifact": "floating512_segmentation",
                    "sha256": sha256_file(floating_path),
                    "path": str(floating_path),
                },
                {
                    "sample_id": sample_id,
                    "artifact": "physical_kl720_segmentation",
                    "sha256": sha256_file(physical_path),
                    "path": str(physical_path),
                },
            ]
        )

    rate_rows = summarize(per_fiber, "floating512_reference") + summarize(
        per_fiber, "physical_kl720"
    )
    hi_rows = [
        r
        for r in rate_rows
        if r["object_type"] == "floating512_reference"
        and r["human_support_stratum"] == "support_ge3"
    ]
    hi_by_bin = {r["length_bin_um"]: r for r in hi_rows}
    all_hi = hi_by_bin["ALL"]

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_READ_ONLY_DNAI_DENOMINATOR_AUDIT",
        "role": "post-hoc descriptive sensitivity analysis; not confirmatory",
        "neural_inference_performed": False,
        "physical_hardware_executed": False,
        "model_or_ptq_selection_authorized": False,
        "matching_algorithm": "maximum_cardinality_then_total_iou",
        "iou_threshold": MATCH_IOU,
        "pixel_size_um": PIXEL_SIZE_UM,
        "legacy_summary_sha256": sha256_file(legacy_summary_path),
        "n_images": len(per_image),
        "n_reference_defined_nonempty_images": sum(
            int(r["reference_defined_nonempty"]) for r in per_image
        ),
        "floating512": {
            "n_total": sum(r["object_type"] == "floating512_reference" for r in per_fiber),
            "n_unmatched": sum(
                r["object_type"] == "floating512_reference" and int(r["unmatched_to_other"]) == 1
                for r in per_fiber
            ),
        },
        "physical_kl720": {
            "n_total": sum(r["object_type"] == "physical_kl720" for r in per_fiber),
            "n_unmatched": sum(
                r["object_type"] == "physical_kl720" and int(r["unmatched_to_other"]) == 1
                for r in per_fiber
            ),
        },
        "high_human_support_floating_reference": {
            "definition": "human_support_count >= 3 of H1-H4",
            "n_total": all_hi["n_total_objects"],
            "n_unmatched": all_hi["n_unmatched_objects"],
            "unmatched_fraction": all_hi["unmatched_fraction"],
            "by_length_bin": {
                b: {
                    "n_total": hi_by_bin[b]["n_total_objects"],
                    "n_unmatched": hi_by_bin[b]["n_unmatched_objects"],
                    "unmatched_fraction": hi_by_bin[b]["unmatched_fraction"],
                }
                for b in LENGTH_BINS
            },
        },
        "interpretation_guard": "Describe length dependence from denominator-aware rates, not unmatched counts alone.",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "per_fiber_denominators.csv", per_fiber)
    write_csv_rows(output_dir / "length_support_mismatch_rates.csv", rate_rows)
    write_csv_rows(output_dir / "per_image_object_counts.csv", per_image)
    write_csv_rows(output_dir / "artifact_hash_inventory.csv", hashes)
    out_summary = output_dir / "dnai_denominator_audit_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    h = summary["high_human_support_floating_reference"]
    print("DNAi denominator-aware object-loss audit: COMPLETE")
    print(
        f"floating reference objects unmatched: {summary['floating512']['n_unmatched']}/{summary['floating512']['n_total']}"
    )
    print(
        f"physical objects unmatched: {summary['physical_kl720']['n_unmatched']}/{summary['physical_kl720']['n_total']}"
    )
    print(
        "high-human-support floating reference objects unmatched: "
        f"{h['n_unmatched']}/{h['n_total']} ({h['unmatched_fraction']:.6f})"
    )
    for b in LENGTH_BINS:
        x = h["by_length_bin"][b]
        rate = "NA" if x["unmatched_fraction"] is None else f"{x['unmatched_fraction']:.6f}"
        print(f"  {b}: unmatched/total={x['n_unmatched']}/{x['n_total']} rate={rate}")
    print("neural inference performed: NO")
    print("physical hardware executed: NO")
    print(f"summary: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
