"""Read-only sensitivity audit of completed DNAi physical characterization.

This script performs NO neural inference and NO KL720 execution. It reads the
already-persisted stitched segmentation masks and the frozen FP1024/human
references, then compares the historical greedy bbox-IoU matching with a
post-hoc maximum-cardinality matching sensitivity analysis.

The historical greedy result remains the primary legacy characterization. This
audit must not be used to select a new PTQ candidate or redefine a confirmatory
endpoint after outcomes are known.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
from radiation_edge_ai.dna_fiber.matching_sensitivity import (
    greedy_iou_matching,
    matching_summary,
    maximum_cardinality_iou_matching,
    reference_defined_nonempty,
)

# The legacy evaluator hard-coded 0.26 um/pixel.
PIXEL_SIZE_UM = 0.26
EXPECTED_ONNX_SHA256 = "a901d1b309a9a0e5026febd5070252787e4b110a2a7d2828e16a19364d6094d0"
EXPECTED_NEF_SHA256 = "4b3dfec9a61c99e186dd4b8482fa5b06e6a4958f325ed4a0db0546f1dcab2bfc"
EXPECTED_IMAGES = 20


def _read_segmentation(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2:
        raise RuntimeError(f"Invalid persisted segmentation: {path}")
    if not np.isfinite(image).all():
        raise RuntimeError(f"Non-finite persisted segmentation: {path}")
    return np.asarray(image, dtype=np.uint8)


def _fibers(seg: np.ndarray) -> list[Any]:
    return list(refine_segmentation(seg).valid_copy())


def _finite_mean(values: Sequence[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _endpoint_errors(
    reference: Sequence[Any],
    candidate: Sequence[Any],
    matches: Sequence[tuple[int, int, float]],
) -> dict[str, Any]:
    if not matches:
        return {
            "n_matched": 0,
            "ratio_mae": None,
            "total_length_mae_um": None,
            "mean_iou": None,
        }
    ratio = [abs(float(reference[i].ratio) - float(candidate[j].ratio)) for i, j, _iou in matches]
    length = [
        abs(float(reference[i].length) - float(candidate[j].length)) * PIXEL_SIZE_UM
        for i, j, _iou in matches
    ]
    return {
        "n_matched": len(matches),
        "ratio_mae": float(np.mean(ratio)),
        "total_length_mae_um": float(np.mean(length)),
        "mean_iou": float(np.mean([iou for _i, _j, iou in matches])),
    }


def _support_counts(
    objects: Sequence[Any],
    human: dict[str, list[Any]],
    matcher: Callable[[Sequence[Any], Sequence[Any], float], list[tuple[int, int, float]]],
) -> list[int]:
    supported = [0] * len(objects)
    for grader in GRADERS:
        for i, _j, _iou in matcher(objects, human[grader], MATCH_IOU):
            supported[i] += 1
    return supported


def _support_hist(rows: Sequence[dict[str, Any]], kind: str, algorithm: str) -> dict[str, int]:
    counter = Counter(
        int(row["human_support_count"])
        for row in rows
        if row["kind"] == kind and row["matching_algorithm"] == algorithm
    )
    return {str(i): int(counter.get(i, 0)) for i in range(5)}


def _length_bin(length_um: float) -> str:
    if length_um < 10:
        return "[0,10)"
    if length_um < 20:
        return "[10,20)"
    if length_um < 30:
        return "[20,30)"
    return "[30,inf)"


def _pooled_matching(per_image: Sequence[dict[str, Any]], prefix: str) -> dict[str, Any]:
    n_ref = sum(int(row["floating_fibers"]) for row in per_image)
    n_cand = sum(int(row["physical_fibers"]) for row in per_image)
    n_match = sum(int(row[f"{prefix}_n_matched"]) for row in per_image)
    precision = n_match / n_cand if n_cand else (1.0 if n_ref == 0 else 0.0)
    recall = n_match / n_ref if n_ref else (1.0 if n_cand == 0 else 0.0)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "n_floating_fibers": n_ref,
        "n_physical_fibers": n_cand,
        "n_matched": n_match,
        "n_floating_unmatched": n_ref - n_match,
        "n_physical_unmatched": n_cand - n_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_per_image_f1": float(np.mean([float(row[f"{prefix}_f1"]) for row in per_image])),
        "mean_per_image_ratio_mae": _finite_mean(
            [
                row[f"{prefix}_ratio_mae"]
                for row in per_image
                if row[f"{prefix}_ratio_mae"] is not None
            ]
        ),
        "mean_per_image_total_length_mae_um": _finite_mean(
            [
                row[f"{prefix}_total_length_mae_um"]
                for row in per_image
                if row[f"{prefix}_total_length_mae_um"] is not None
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    default_data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    default_model_root = Path(
        os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models")
    )
    kneron_root = default_model_root / "dnai" / "unet_mobileone_s1" / "kneron"
    v3_root = kneron_root / "int8_512_v3_pct100"
    parser.add_argument(
        "--characterization-dir",
        default=str(v3_root / "hardware_biological_fidelity"),
    )
    parser.add_argument("--data-root", default=str(default_data_root))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    characterization = Path(args.characterization_dir).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    refuse_nonempty_output_dir(output_dir)

    summary_path = characterization / "summary.json"
    per_image_path = characterization / "per_image.csv"
    if not summary_path.is_file() or not per_image_path.is_file():
        raise FileNotFoundError("Completed legacy DNAi characterization files are missing")

    legacy = json.loads(summary_path.read_text(encoding="utf-8"))
    if legacy.get("dnai_commit") != DNAI_COMMIT:
        raise RuntimeError("Legacy characterization DNAi commit mismatch")
    if str(legacy.get("optimized_onnx_sha256", "")).lower() != EXPECTED_ONNX_SHA256:
        raise RuntimeError("Legacy characterization optimized ONNX identity mismatch")
    if str(legacy.get("hardware_nef_sha256", "")).lower() != EXPECTED_NEF_SHA256:
        raise RuntimeError("Legacy characterization NEF identity mismatch")
    if int(legacy.get("n_images", -1)) != EXPECTED_IMAGES:
        raise RuntimeError("Legacy characterization is not the expected 20-image panel")

    with per_image_path.open("r", newline="", encoding="utf-8-sig") as handle:
        legacy_rows = list(csv.DictReader(handle))
    if len(legacy_rows) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} legacy per-image rows, got {len(legacy_rows)}"
        )
    sample_ids = [str(row["sample_id"]) for row in legacy_rows]
    if len(set(sample_ids)) != EXPECTED_IMAGES:
        raise RuntimeError("Duplicate sample_id in legacy per-image table")

    panel_root = data_root / "dnai_public_v2" / "r1_intergrader20"
    dataset_root = data_root / "dnai_public_v2" / "extracted" / "zenodo"
    panel_manifest_path = panel_root / "manifest.json"
    baseline_root = panel_root / "r1_s1_strict_fp32_preprocessed"
    if not panel_manifest_path.is_file():
        raise FileNotFoundError(panel_manifest_path)
    panel_manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    if panel_manifest.get("dnai_commit") != DNAI_COMMIT:
        raise RuntimeError("Panel manifest DNAi commit mismatch")
    panel_by_sample = {str(row["sample_id"]): row for row in panel_manifest.get("records", [])}

    per_image: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    artifact_hashes: list[dict[str, Any]] = []

    algorithms = {
        "legacy_greedy": greedy_iou_matching,
        "max_cardinality_total_iou": maximum_cardinality_iou_matching,
    }

    for sample_id in sample_ids:
        panel_record = panel_by_sample.get(sample_id)
        if panel_record is None:
            raise RuntimeError(f"Sample absent from panel manifest: {sample_id}")

        floating_path = characterization / sample_id / "floating512_segmentation_class_ids.png"
        physical_path = characterization / sample_id / "physical_kl720_segmentation_class_ids.png"
        floating_seg = _read_segmentation(floating_path)
        physical_seg = _read_segmentation(physical_path)
        if floating_seg.shape != physical_seg.shape:
            raise RuntimeError(f"Persisted segmentation shape mismatch for {sample_id}")

        _baseline_probs, baseline_seg = load_frozen_1024(baseline_root / sample_id)
        if baseline_seg.shape != floating_seg.shape:
            raise RuntimeError(f"Baseline/persisted segmentation shape mismatch for {sample_id}")

        floating_fibers = _fibers(floating_seg)
        physical_fibers = _fibers(physical_seg)
        baseline_fibers = _fibers(baseline_seg)

        human: dict[str, list[Any]] = {}
        human_counts: list[int] = []
        for grader in GRADERS:
            annotation = dataset_root / panel_record[f"{grader}_annotation"]
            expected_sha = str(panel_record[f"{grader}_sha256"])
            observed_sha = sha256_file(annotation)
            if observed_sha != expected_sha:
                raise RuntimeError(f"Human annotation checksum mismatch: {annotation}")
            human_seg = read_human_mask(annotation)
            if human_seg.shape != floating_seg.shape:
                raise RuntimeError(
                    f"Human/persisted segmentation shape mismatch for {sample_id}/{grader}"
                )
            human[grader] = _fibers(human_seg)
            human_counts.append(len(human[grader]))
            artifact_hashes.append(
                {
                    "sample_id": sample_id,
                    "artifact": f"{grader}_annotation",
                    "path": str(annotation),
                    "sha256": observed_sha,
                }
            )

        reference_nonempty = reference_defined_nonempty(len(baseline_fibers), human_counts)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "reference_defined_nonempty": int(reference_nonempty),
            "baseline1024_fibers": len(baseline_fibers),
            "floating_fibers": len(floating_fibers),
            "physical_fibers": len(physical_fibers),
            **{f"{grader}_human_fibers": len(human[grader]) for grader in GRADERS},
        }

        artifact_hashes.extend(
            [
                {
                    "sample_id": sample_id,
                    "artifact": "floating512_segmentation",
                    "path": str(floating_path),
                    "sha256": sha256_file(floating_path),
                },
                {
                    "sample_id": sample_id,
                    "artifact": "physical_kl720_segmentation",
                    "path": str(physical_path),
                    "sha256": sha256_file(physical_path),
                },
            ]
        )

        for algorithm, matcher in algorithms.items():
            matches = matcher(floating_fibers, physical_fibers, MATCH_IOU)
            ms = matching_summary(len(floating_fibers), len(physical_fibers), matches)
            ep = _endpoint_errors(floating_fibers, physical_fibers, matches)
            for key, value in ms.items():
                row[f"{algorithm}_{key}"] = value
            row[f"{algorithm}_ratio_mae"] = ep["ratio_mae"]
            row[f"{algorithm}_total_length_mae_um"] = ep["total_length_mae_um"]
            row[f"{algorithm}_mean_iou"] = ep["mean_iou"]

            matched_floating = {i for i, _j, _iou in matches}
            matched_physical = {j for _i, j, _iou in matches}
            floating_support = _support_counts(floating_fibers, human, matcher)
            physical_support = _support_counts(physical_fibers, human, matcher)

            for i, fiber in enumerate(floating_fibers):
                if i in matched_floating:
                    continue
                disagreements.append(
                    {
                        "sample_id": sample_id,
                        "matching_algorithm": algorithm,
                        "kind": "floating512_missed_by_physical",
                        "fiber_index": i,
                        "length_um": float(fiber.length) * PIXEL_SIZE_UM,
                        "length_bin_um": _length_bin(float(fiber.length) * PIXEL_SIZE_UM),
                        "ratio": float(fiber.ratio),
                        "human_support_count": floating_support[i],
                    }
                )
            for j, fiber in enumerate(physical_fibers):
                if j in matched_physical:
                    continue
                disagreements.append(
                    {
                        "sample_id": sample_id,
                        "matching_algorithm": algorithm,
                        "kind": "physical_only",
                        "fiber_index": j,
                        "length_um": float(fiber.length) * PIXEL_SIZE_UM,
                        "length_bin_um": _length_bin(float(fiber.length) * PIXEL_SIZE_UM),
                        "ratio": float(fiber.ratio),
                        "human_support_count": physical_support[j],
                    }
                )

        per_image.append(row)

    ref_subset = [row for row in per_image if int(row["reference_defined_nonempty"]) == 1]
    if not ref_subset:
        raise RuntimeError("Reference-defined nonempty subset unexpectedly empty")

    summary: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_READ_ONLY_DNAI_CHARACTERIZATION_SENSITIVITY_AUDIT",
        "role": "post-hoc sensitivity analysis of development/deployment characterization; not confirmatory",
        "neural_inference_performed": False,
        "physical_hardware_executed": False,
        "ptq_or_model_selection_authorized": False,
        "legacy_characterization": {
            "summary_path": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "per_image_path": str(per_image_path),
            "per_image_sha256": sha256_file(per_image_path),
            "optimized_onnx_sha256": legacy["optimized_onnx_sha256"],
            "hardware_nef_sha256": legacy["hardware_nef_sha256"],
            "n_images": len(per_image),
        },
        "matching": {
            "iou_threshold": MATCH_IOU,
            "historical_primary": "legacy_greedy",
            "posthoc_sensitivity": "max_cardinality_total_iou",
        },
        "reference_population": {
            "definition": "baseline1024 has >=1 valid fiber OR at least one H1-H4 grader has >=1 valid fiber; physical/floating detections are not used",
            "n_all_images": len(per_image),
            "n_reference_defined_nonempty_images": len(ref_subset),
            "reference_defined_empty_sample_ids": [
                row["sample_id"] for row in per_image if not int(row["reference_defined_nonempty"])
            ],
        },
        "all20": {
            "legacy_greedy": _pooled_matching(per_image, "legacy_greedy"),
            "max_cardinality_total_iou": _pooled_matching(per_image, "max_cardinality_total_iou"),
        },
        "reference_defined_nonempty": {
            "legacy_greedy": _pooled_matching(ref_subset, "legacy_greedy"),
            "max_cardinality_total_iou": _pooled_matching(ref_subset, "max_cardinality_total_iou"),
        },
        "unmatched_human_support_counts": {
            algorithm: {
                "floating512_missed_by_physical": _support_hist(
                    disagreements, "floating512_missed_by_physical", algorithm
                ),
                "physical_only": _support_hist(disagreements, "physical_only", algorithm),
            }
            for algorithm in algorithms
        },
    }

    length_rows: list[dict[str, Any]] = []
    bins = ("[0,10)", "[10,20)", "[20,30)", "[30,inf)")
    for algorithm in algorithms:
        for kind in ("floating512_missed_by_physical", "physical_only"):
            subset = [
                row
                for row in disagreements
                if row["matching_algorithm"] == algorithm and row["kind"] == kind
            ]
            for bin_name in bins:
                b = [row for row in subset if row["length_bin_um"] == bin_name]
                length_rows.append(
                    {
                        "matching_algorithm": algorithm,
                        "kind": kind,
                        "length_bin_um": bin_name,
                        "n_unmatched": len(b),
                        "n_human_supported_ge1": sum(
                            int(row["human_support_count"]) >= 1 for row in b
                        ),
                        "n_human_supported_ge3": sum(
                            int(row["human_support_count"]) >= 3 for row in b
                        ),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "per_image_matching_sensitivity.csv", per_image)
    write_csv_rows(output_dir / "unmatched_fiber_sensitivity.csv", disagreements)
    write_csv_rows(output_dir / "length_dependent_unmatched_summary.csv", length_rows)
    write_csv_rows(output_dir / "artifact_hash_inventory.csv", artifact_hashes)
    summary_out = output_dir / "dnai_characterization_sensitivity_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    print("DNAi characterization sensitivity audit: COMPLETE")
    print(f"images: {len(per_image)}")
    print(f"reference-defined nonempty images: {len(ref_subset)}/{len(per_image)}")
    g = summary["all20"]["legacy_greedy"]
    m = summary["all20"]["max_cardinality_total_iou"]
    print(f"all20 pooled fiber F1 legacy/max-cardinality: {g['f1']:.6f} / {m['f1']:.6f}")
    print(f"all20 matched fibers legacy/max-cardinality: {g['n_matched']} / {m['n_matched']}")
    print(
        "legacy floating-missed human support >=3: "
        + str(
            sum(
                int(v)
                for k, v in summary["unmatched_human_support_counts"]["legacy_greedy"][
                    "floating512_missed_by_physical"
                ].items()
                if int(k) >= 3
            )
        )
    )
    print(
        "max-cardinality floating-missed human support >=3: "
        + str(
            sum(
                int(v)
                for k, v in summary["unmatched_human_support_counts"]["max_cardinality_total_iou"][
                    "floating512_missed_by_physical"
                ].items()
                if int(k) >= 3
            )
        )
    )
    print("neural inference performed: NO")
    print("physical hardware executed: NO")
    print(f"summary: {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
