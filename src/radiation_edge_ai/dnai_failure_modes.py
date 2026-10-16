"""Verified, read-only failure-mode diagnostics for frozen FP512/KL720 DNAi fields.

The diagnostic never performs neural inference, replaces a segmentation,
re-matches fibers using new thresholds, or changes historical evidence.
Reconstruction/CSV replay and source verification reuse the v0.12 gates.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from radiation_edge_ai import dnai_fiber_instances as inst
from radiation_edge_ai import dnai_paired_fields as paired
from radiation_edge_ai.control import ControlPlaneError, sha256_json

BORDER_DIAGNOSTIC_PX = 32
ROI_PADDING_PX = 4
SCIENCE_WORKER_TIMEOUT_SECONDS = 180
REPORT_TYPE = "dnai_fp512_kl720_failure_mode_audit"


def _dilate_three_class_argmax(raw: np.ndarray) -> np.ndarray:
    """Exact uint8 3x3 grayscale max dilation used by frozen DNAi.

    Upstream DNAi fcf20c7d6: torch argmax (ties choose lowest class), cast to
    uint8, then cv2.morphologyEx(MORPH_DILATE, ones((3,3),uint8)).
    This is NOT independent per-class dilation: green class 2 wins over red 1.
    """
    if raw.ndim != 2 or raw.dtype != np.uint8 or np.any(raw > 2):
        raise ControlPlaneError("Morphology replay requires a 2D uint8 three-class argmax")
    height, width = raw.shape
    padded = np.pad(raw, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    dilated = np.zeros_like(raw)
    for dy in range(3):
        for dx in range(3):
            np.maximum(dilated, padded[dy:dy + height, dx:dx + width], out=dilated)
    return dilated


def _replay_saved_morphology(probabilities: np.ndarray, saved: np.ndarray, label: str) -> np.ndarray:
    """Fail closed if the bound PNG is not exactly the frozen argmax+dilation."""
    if (
        probabilities.ndim != 3
        or probabilities.shape != (3,) + saved.shape
        or probabilities.dtype != np.float32
        or not np.isfinite(probabilities).all()
    ):
        raise ControlPlaneError(f"{label}: invalid stitched probability contract")
    raw = np.argmax(probabilities, axis=0).astype(np.uint8)
    regenerated = _dilate_three_class_argmax(raw)
    if not np.array_equal(regenerated, saved):
        changed = int(np.count_nonzero(regenerated != saved))
        raise ControlPlaneError(
            f"{label}: frozen DNAi argmax+3x3 morphology fails to reproduce PNG "
            f"({changed} disagreeing pixels)"
        )
    return raw


def _read_saved_segmentation(field: Mapping[str, Any]) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise ControlPlaneError("Scientific Python requires cv2 for DNAi PNG decoding") from exc
    raw = paired._verified_bytes(field, "segmentation")
    seg = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if (
        seg is None
        or seg.shape != paired.SEGMENTATION_SHAPE
        or seg.dtype != np.uint8
        or np.any(seg > 2)
    ):
        raise ControlPlaneError("Bound DNAi segmentation PNG is not uint8 [1024,1024] class 0..2")
    return seg


def _geometry_and_support(fiber: Any, shape: tuple[int, int]) -> tuple[list[int], np.ndarray]:
    """Use the upstream reconstructed fiber's own colored pixel support."""
    try:
        bbox = [int(value) for value in fiber.bbox]
        if len(bbox) != 4 or any(
            isinstance(value, (bool, str)) or int(value) != value
            for value in fiber.bbox
        ):
            raise ValueError("invalid integral bbox")
        data = np.asarray(fiber.data)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ControlPlaneError("Invalid reconstructed DNAi fiber geometry") from exc
    x, y, width, height = bbox
    if (
        x < 0 or y < 0 or width <= 0 or height <= 0
        or x + width > shape[1] or y + height > shape[0]
        or data.shape != (height, width)
        or not np.issubdtype(data.dtype, np.integer)
        or np.any(data > 2) or np.any(data < 0)
        or not np.any(data > 0)
    ):
        raise ControlPlaneError("Reconstructed DNAi fiber support violates field geometry")
    return bbox, data


def _margin_summary(values: np.ndarray) -> dict[str, float] | None:
    if not values.size:
        return None
    return {
        "mean": float(np.mean(values, dtype=np.float64)),
        "median": float(np.median(values)),
        "fraction_nonpositive": float(np.mean(values <= 0)),
    }


def _probability_margins(
    fp: np.ndarray, kl: np.ndarray, support: np.ndarray, class_id: int
) -> dict[str, Any]:
    """Class-specific margins measured on the SAME frozen source-fiber pixels."""
    if not np.any(support):
        return {"against_background": None, "against_strongest_competitor": None}

    def assess(other_class: int | None) -> dict[str, Any]:
        def values(prob: np.ndarray) -> np.ndarray:
            p_class = prob[class_id][support].astype(np.float64)
            if other_class is None:
                competitors = [index for index in range(3) if index != class_id]
                competing = np.maximum(
                    prob[competitors[0]][support].astype(np.float64),
                    prob[competitors[1]][support].astype(np.float64),
                )
            else:
                competing = prob[other_class][support].astype(np.float64)
            return p_class - competing

        left = values(fp)
        right = values(kl)
        return {
            "fp512": _margin_summary(left),
            "kl720": _margin_summary(right),
            "delta_kl720_minus_fp512_mean": float(np.mean(right - left, dtype=np.float64)),
        }

    return {
        "against_background": assess(0),
        "against_strongest_competitor": assess(None),
    }


def _components_touching_source(
    binary: np.ndarray, source_support: np.ndarray
) -> dict[str, int]:
    labels, n = paired._label_components8(binary)
    touching = np.unique(labels[source_support])
    return {
        "roi_components": n,
        "components_touching_source_support": int(np.count_nonzero(touching)),
    }


def _measure_fiber(
    fiber: Any,
    *,
    side: str,
    row: int,
    local_fiber_id: int,
    matched_row: int | None,
    fp_final: np.ndarray,
    kl_final: np.ndarray,
    fp_raw: np.ndarray,
    kl_raw: np.ndarray,
    fp_probs: np.ndarray,
    kl_probs: np.ndarray,
) -> dict[str, Any]:
    """Directional support/fragmentation audit; margins always KL720 minus FP512."""
    if side not in ("fp512", "kl720"):
        raise ControlPlaneError("Unknown DNAi source side")
    shape = fp_final.shape
    if (
        any(image.shape != shape for image in (kl_final, fp_raw, kl_raw))
        or fp_probs.shape != (3,) + shape or kl_probs.shape != (3,) + shape
    ):
        raise ControlPlaneError("DNAi failure-mode field shapes do not agree")
    bbox, fiber_data = _geometry_and_support(fiber, shape)
    x, y, width, height = bbox
    rx0, ry0 = max(0, x - ROI_PADDING_PX), max(0, y - ROI_PADDING_PX)
    rx1 = min(shape[1], x + width + ROI_PADDING_PX)
    ry1 = min(shape[0], y + height + ROI_PADDING_PX)
    ys, xs = slice(ry0, ry1), slice(rx0, rx1)
    support = np.zeros((ry1 - ry0, rx1 - rx0), dtype=np.uint8)
    support[y - ry0:y - ry0 + height, x - rx0:x - rx0 + width] = fiber_data

    source_raw, target_raw = (fp_raw, kl_raw) if side == "fp512" else (kl_raw, fp_raw)
    source_final, target_final = (
        (fp_final, kl_final) if side == "fp512" else (kl_final, fp_final)
    )
    source_r = source_raw[ys, xs]
    target_r = target_raw[ys, xs]
    source_f = source_final[ys, xs]
    target_f = target_final[ys, xs]
    n_support = int(np.count_nonzero(support))
    border_distance = min(x, y, shape[1] - x - width, shape[0] - y - height)

    classes = []
    for class_id in (1, 2):
        pixels = support == class_id
        n = int(np.count_nonzero(pixels))
        target_raw_count = int(np.count_nonzero((target_r == class_id) & pixels))
        target_final_count = int(np.count_nonzero((target_f == class_id) & pixels))
        target_roi_count = int(np.count_nonzero(target_f == class_id))
        component_proxy = {
            "source_raw": _components_touching_source(source_r == class_id, pixels),
            "target_raw": _components_touching_source(target_r == class_id, pixels),
            "source_after_dilation": _components_touching_source(source_f == class_id, pixels),
            "target_after_dilation": _components_touching_source(target_f == class_id, pixels),
        }
        classes.append({
            "class_id": class_id,
            "source_fiber_support_pixels": n,
            "source_raw_same_class_on_support": int(
                np.count_nonzero((source_r == class_id) & pixels)
            ),
            "source_final_same_class_on_support": int(
                np.count_nonzero((source_f == class_id) & pixels)
            ),
            "target_raw_same_class_on_support": target_raw_count,
            "target_final_same_class_on_support": target_final_count,
            "target_raw_foreground_on_support": int(
                np.count_nonzero((target_r != 0) & pixels)
            ),
            "target_final_foreground_on_support": int(
                np.count_nonzero((target_f != 0) & pixels)
            ),
            "target_raw_class_retention": (target_raw_count / n if n else None),
            "target_final_class_retention": (target_final_count / n if n else None),
            "target_final_class_absent_on_support": bool(n and target_final_count == 0),
            "target_final_class_absent_in_padded_roi": bool(n and target_roi_count == 0),
            "raw_to_dilated_target_class_pixel_delta_on_support": (
                target_final_count - target_raw_count
            ),
            "source_final_class_pixels_in_padded_roi": int(
                np.count_nonzero(source_f == class_id)
            ),
            "target_final_class_pixels_in_padded_roi": target_roi_count,
            "probability_margins": _probability_margins(
                fp_probs[:, ys, xs], kl_probs[:, ys, xs], pixels, class_id
            ),
            "eight_connected_segmentation_component_proxy": component_proxy,
        })

    foreground = support != 0
    fg_components = {
        "source_raw": _components_touching_source(source_r != 0, foreground),
        "target_raw": _components_touching_source(target_r != 0, foreground),
        "source_after_dilation": _components_touching_source(source_f != 0, foreground),
        "target_after_dilation": _components_touching_source(target_f != 0, foreground),
    }
    return {
        "source_side": side,
        "target_side": "kl720" if side == "fp512" else "fp512",
        "reconstructed_row": row,
        "local_fiber_id": local_fiber_id,
        "primary_matched": matched_row is not None,
        "primary_partner_row": matched_row,
        "bbox_xywh": bbox,
        "padded_roi_xyxy_exclusive": [rx0, ry0, rx1, ry1],
        "boundary_distance_px": border_distance,
        "within_32px_of_image_boundary": border_distance <= BORDER_DIAGNOSTIC_PX,
        "fiber_colored_support_pixels": n_support,
        "target_raw_foreground_retention": (
            int(np.count_nonzero((target_r != 0) & foreground)) / n_support
        ),
        "target_final_foreground_retention": (
            int(np.count_nonzero((target_f != 0) & foreground)) / n_support
        ),
        "per_class": classes,
        "foreground_eight_connected_component_proxy": fg_components,
    }


def _boundary_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    distances = [int(item["boundary_distance_px"]) for item in rows]
    return {
        "n": len(rows),
        "n_within_32px": sum(dist <= BORDER_DIAGNOSTIC_PX for dist in distances),
        "distances_px": distances,
    }


def _disappearance_summary(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    classes = [
        item for row in rows if not row["primary_matched"] for item in row["per_class"]
        if item["source_fiber_support_pixels"] > 0
    ]
    return {
        "unmatched_source_classes_tested": len(classes),
        "n_zero_target_class_on_support": sum(
            item["target_final_class_absent_on_support"] for item in classes
        ),
        "n_zero_target_class_in_padded_roi": sum(
            item["target_final_class_absent_in_padded_roi"] for item in classes
        ),
    }


def audit_dnai_failure_modes(fp512_path: Path, kl720_path: Path) -> dict[str, Any]:
    """Read, fully verify, reconstruct, and inspect frozen evidence only."""
    paths = [Path(fp512_path).expanduser().resolve(), Path(kl720_path).expanduser().resolve()]
    fields, hashes, identity = inst._verified_field_pair(*paths)
    try:
        version = importlib.metadata.version("dnafiber")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ControlPlaneError("Scientific Python does not contain frozen dnafiber") from exc
    if fields[0]["runtime"].get("dnafiber_version") != fields[1]["runtime"].get(
        "dnafiber_version"
    ):
        raise ControlPlaneError("Frozen DNAi scientific versions differ")
    reconstructed = [inst._reconstruct_verified_field(field, version) for field in fields]
    if reconstructed[0]["pixel_size_um"] != reconstructed[1]["pixel_size_um"]:
        raise ControlPlaneError("DNAi pixel-size policy differs across frozen fields")

    matching = inst._matching_report(
        reconstructed[0]["fibers"], reconstructed[1]["fibers"],
        reconstructed[0]["fiber_ids"], reconstructed[1]["fiber_ids"],
        reconstructed[0]["pixel_size_um"],
    )
    fp_final, kl_final = [_read_saved_segmentation(field) for field in fields]
    fp_probs, kl_probs = [
        paired._load_probabilities(paired._verified_bytes(field, "stitched_probabilities"))
        for field in fields
    ]
    fp_raw = _replay_saved_morphology(fp_probs, fp_final, "FP512")
    kl_raw = _replay_saved_morphology(kl_probs, kl_final, "KL720")

    fp_to_kl = {
        pair["fp512_row"]: pair["kl720_row"]
        for pair in matching["primary"]["matched_pairs"]
    }
    kl_to_fp = {
        pair["kl720_row"]: pair["fp512_row"]
        for pair in matching["primary"]["matched_pairs"]
    }
    shared = {
        "fp_final": fp_final, "kl_final": kl_final,
        "fp_raw": fp_raw, "kl_raw": kl_raw,
        "fp_probs": fp_probs, "kl_probs": kl_probs,
    }
    fp_rows = [
        _measure_fiber(
            fiber, side="fp512", row=i,
            local_fiber_id=reconstructed[0]["fiber_ids"][i],
            matched_row=fp_to_kl.get(i), **shared
        )
        for i, fiber in enumerate(reconstructed[0]["fibers"])
    ]
    kl_rows = [
        _measure_fiber(
            fiber, side="kl720", row=i,
            local_fiber_id=reconstructed[1]["fiber_ids"][i],
            matched_row=kl_to_fp.get(i), **shared
        )
        for i, fiber in enumerate(reconstructed[1]["fibers"])
    ]
    # Recheck all bound artifacts after the operation, not just JSON metadata.
    for path, digest in zip(paths, hashes):
        if inst.sha256_file(path) != digest:
            raise ControlPlaneError("Frozen DNAi source field changed during failure-mode audit")
        verified = inst.verify_target(path, check_artifacts=True)
        if not isinstance(verified, Mapping) or verified.get("ok") is not True:
            raise ControlPlaneError("Frozen DNAi source artifacts changed during failure-mode audit")
        if inst.sha256_file(path) != digest:
            raise ControlPlaneError("Frozen DNAi source field changed during final verification")

    return {
        "schema_version": 1,
        "report_type": REPORT_TYPE,
        "source_fp512": {
            "path": str(paths[0]), "sha256": hashes[0], "field_id": fields[0]["field_id"],
        },
        "source_kl720": {
            "path": str(paths[1]), "sha256": hashes[1], "field_id": fields[1]["field_id"],
        },
        "matched_input_sha256": sha256_json(identity),
        "shared_input": {
            "sample_id": identity["sample_id"], "key": identity["key"],
            "image_index": identity["image_index"], "n_windows": 9,
        },
        "scientific_runtime": {"dnafiber_version": version},
        "fixed_diagnostic_definitions": {
            "morphology": "argmax class ID followed by 3x3 grayscale max dilation",
            "morphology_upstream_commit": identity["dnai_commit"],
            "roi_padding_px": ROI_PADDING_PX,
            "image_boundary_distance_cutoff_px": BORDER_DIAGNOSTIC_PX,
            "fragmentation": "8-connected raw/final segmentation component PROXIES in fixed ROI",
            "support": "reconstructed fiber's own colored data pixels (not whole bbox)",
            "matching": "unchanged primary greedy reconstructed-bbox IoU >= 0.5",
        },
        "morphology_reproduced_exactly": {"fp512": True, "kl720": True},
        "historical_instance_matching": matching,
        "boundary_association": {
            "fp512_primary_matched": _boundary_group(
                [r for r in fp_rows if r["primary_matched"]]
            ),
            "fp512_primary_unmatched": _boundary_group(
                [r for r in fp_rows if not r["primary_matched"]]
            ),
            "kl720_primary_matched": _boundary_group(
                [r for r in kl_rows if r["primary_matched"]]
            ),
            "kl720_primary_unmatched": _boundary_group(
                [r for r in kl_rows if not r["primary_matched"]]
            ),
        },
        "unmatched_class_disappearance": {
            "fp512_source": _disappearance_summary(fp_rows),
            "kl720_source": _disappearance_summary(kl_rows),
        },
        "per_fiber": {"fp512_source": fp_rows, "kl720_source": kl_rows},
        "scientific_scope": {
            "sources_and_artifacts_verified_before_and_after": True,
            "frozen_valid_csv_reproduced": True,
            "morphological_replay_gated_against_saved_segmentations": True,
            "same_historical_matching_threshold": True,
            "model_inference_performed": False,
            "hardware_access_performed": False,
            "matching_or_segmentation_reoptimized": False,
            "human_annotations_read": False,
            "biological_fidelity_evaluated": False,
            "fragmentation_components_are_fibers": False,
            "boundary_association_implies_causality": False,
            "quantization_or_stitching_cause_established": False,
            "ephemeral_diagnostics_only": True,
        },
    }


def run_scientific_failure_mode_worker(
    fp512_path: Path, kl720_path: Path, *, scientific_python: Path
) -> dict[str, Any]:
    """Use the explicit scientific env, leaving the control plane lightweight."""
    executable = Path(scientific_python).expanduser().resolve()
    if not executable.is_file():
        raise ControlPlaneError(f"Scientific Python executable missing: {executable}")
    source_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    paths = [Path(fp512_path).expanduser().resolve(), Path(kl720_path).expanduser().resolve()]
    cmd = [
        str(executable), "-B", "-m", "radiation_edge_ai.dnai_failure_modes",
        "--fp512", str(paths[0]), "--kl720", str(paths[1]),
    ]
    try:
        process = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            encoding="utf-8", timeout=SCIENCE_WORKER_TIMEOUT_SECONDS, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ControlPlaneError(f"DNAi failure-mode scientific worker failed: {exc}") from exc
    if process.returncode != 0:
        raise ControlPlaneError(
            f"DNAi failure-mode scientific worker exited {process.returncode}: "
            f"{process.stderr[-2000:]}"
        )
    try:
        report = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise ControlPlaneError("DNAi failure-mode worker returned non-JSON output") from exc
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("report_type") != REPORT_TYPE
        or report.get("morphology_reproduced_exactly") != {"fp512": True, "kl720": True}
        or not isinstance(report.get("scientific_scope"), dict)
        or report["scientific_scope"].get("sources_and_artifacts_verified_before_and_after")
        is not True
        or not isinstance(report.get("source_fp512"), dict)
        or not isinstance(report.get("source_kl720"), dict)
        or report["source_fp512"].get("path") != str(paths[0])
        or report["source_kl720"].get("path") != str(paths[1])
    ):
        raise ControlPlaneError("DNAi failure-mode scientific worker returned incompatible report")
    return report


def _worker_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only frozen DNAi failure-mode worker")
    parser.add_argument("--fp512", type=Path, required=True)
    parser.add_argument("--kl720", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(
            audit_dnai_failure_modes(args.fp512, args.kl720),
            indent=2, ensure_ascii=False, allow_nan=False,
        ))
        return 0
    except (ControlPlaneError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"DNAi failure-mode audit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_worker_main())
