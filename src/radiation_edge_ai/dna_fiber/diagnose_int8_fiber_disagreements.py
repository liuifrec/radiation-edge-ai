"""Diagnose which DNA fibers are changed across INT8 PTQ variants.

This is a cheap post-hoc diagnostic that uses already-generated stitched
segmentations. It does NOT run Kneron simulation and does NOT change any model.

For one frozen validation image, it reconstructs fibers from the common Kneron
floating-point 512 segmentation, from each available BIE variant, and from the
four H1-H4 human annotations. It then reports, for every floating-point fiber:

- whether each INT8 variant retains a matching fiber (bbox IoU >= 0.5),
- how many human graders contain a matching fiber,
- fiber length and red/green ratio.

It also reports BIE-only fibers and their human support. This distinguishes
quantization failures from cases where INT8 suppresses an FP32-specific object
that is weakly supported by humans.

Human annotations are not treated as infallible ground truth; support counts
are descriptive evidence only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dnafiber.postprocess import refine_segmentation

GRADERS = ("H1", "H2", "H3", "H4")
MATCH_IOU = 0.5
PIXEL_SIZE_UM = 0.26


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


@dataclass(frozen=True)
class Match:
    reference_index: int
    candidate_index: int
    iou: float


def read_segmentation(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read segmentation: {path}")
    if image.shape != (1024, 1024):
        raise RuntimeError(f"Unexpected segmentation shape {image.shape}: {path}")
    return image.astype(np.uint8, copy=False)


def greedy_matches(reference, candidate, threshold: float = MATCH_IOU) -> list[Match]:
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
    matches: list[Match] = []
    for iou, i, j in candidates:
        if i in used_ref or j in used_cand:
            continue
        used_ref.add(i)
        used_cand.add(j)
        matches.append(Match(i, j, iou))
    return matches


def support_label(count: int) -> str:
    if count >= 3:
        return "human-supported"
    if count >= 1:
        return "ambiguous"
    return "no-human-match"


def main() -> int:
    parser = argparse.ArgumentParser()
    default_data_root = Path(
        os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data")
    )
    default_model_root = Path(
        os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models")
    )
    parser.add_argument("--sample-id", default="1__tile_4")
    parser.add_argument("--data-root", default=str(default_data_root))
    parser.add_argument("--model-root", default=str(default_model_root))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    model_root = Path(args.model_root)
    kneron_root = model_root / "dnai" / "unet_mobileone_s1" / "kneron"
    panel_root = data_root / "dnai_public_v2" / "r1_intergrader20"
    dataset_root = data_root / "dnai_public_v2" / "extracted" / "zenodo"
    panel_manifest = json.loads((panel_root / "manifest.json").read_text(encoding="utf-8"))
    panel_by_sample = {r["sample_id"]: r for r in panel_manifest["records"]}
    record = panel_by_sample.get(args.sample_id)
    if record is None:
        raise RuntimeError(f"Unknown frozen-panel sample: {args.sample_id}")

    version_dirs = {
        "v1": kneron_root / "int8_512" / "biological_fidelity" / args.sample_id,
        "v2": kneron_root / "int8_512_v2" / "biological_fidelity" / args.sample_id,
        "v3": kneron_root / "int8_512_v3_pct100" / "biological_fidelity" / args.sample_id,
        "v4": kneron_root / "int8_512_v4_bias" / "biological_fidelity" / args.sample_id,
    }
    available = {
        name: path
        for name, path in version_dirs.items()
        if (path / "bie_segmentation_class_ids.png").is_file()
        and (path / "floating_segmentation_class_ids.png").is_file()
    }
    if not available:
        raise RuntimeError("No completed biological-fidelity variant directories found")

    floating_masks = {
        name: read_segmentation(path / "floating_segmentation_class_ids.png")
        for name, path in available.items()
    }
    first_name = next(iter(floating_masks))
    floating_seg = floating_masks[first_name]
    for name, mask in floating_masks.items():
        if not np.array_equal(mask, floating_seg):
            raise RuntimeError(
                f"Floating segmentation differs between {first_name} and {name}; "
                "comparison is not controlled"
            )

    fp_fibers = list(refine_segmentation(floating_seg).valid_copy())
    bie_fibers = {
        name: list(
            refine_segmentation(
                read_segmentation(path / "bie_segmentation_class_ids.png")
            ).valid_copy()
        )
        for name, path in available.items()
    }

    human_fibers = {}
    for grader in GRADERS:
        annotation = dataset_root / record[f"{grader}_annotation"]
        if sha256_file(annotation) != record[f"{grader}_sha256"]:
            raise RuntimeError(f"Annotation checksum mismatch: {annotation}")
        human_seg = read_human_mask(annotation)
        human_fibers[grader] = list(refine_segmentation(human_seg).valid_copy())

    fp_to_bie: dict[str, dict[int, Match]] = {}
    bie_to_fp: dict[str, dict[int, Match]] = {}
    for name, fibers in bie_fibers.items():
        matches = greedy_matches(fp_fibers, fibers)
        fp_to_bie[name] = {m.reference_index: m for m in matches}
        bie_to_fp[name] = {m.candidate_index: m for m in matches}

    fp_to_human: dict[str, dict[int, Match]] = {}
    for grader, fibers in human_fibers.items():
        matches = greedy_matches(fp_fibers, fibers)
        fp_to_human[grader] = {m.reference_index: m for m in matches}

    output_root = (
        Path(args.output)
        if args.output
        else kneron_root / "int8_fiber_diagnostics" / args.sample_id
    )
    output_root.mkdir(parents=True, exist_ok=True)

    fp_rows = []
    for i, fiber in enumerate(fp_fibers):
        grader_ious = {
            grader: fp_to_human[grader][i].iou if i in fp_to_human[grader] else None
            for grader in GRADERS
        }
        human_support = sum(value is not None for value in grader_ious.values())
        row = {
            "fp_fiber_index": i,
            "length_px": float(fiber.length),
            "length_um": float(fiber.length) * PIXEL_SIZE_UM,
            "ratio": float(fiber.ratio),
            "human_support_count": human_support,
            "human_support_label": support_label(human_support),
        }
        for grader in GRADERS:
            row[f"{grader}_matched"] = grader_ious[grader] is not None
            row[f"{grader}_iou"] = grader_ious[grader]
        for name in available:
            match = fp_to_bie[name].get(i)
            row[f"{name}_retained"] = match is not None
            row[f"{name}_iou"] = match.iou if match else None
        fp_rows.append(row)

    fp_csv = output_root / "fp_fiber_retention_and_human_support.csv"
    with fp_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fp_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fp_rows)

    extra_rows = []
    for name, fibers in bie_fibers.items():
        bie_to_human = {}
        for grader, h_fibers in human_fibers.items():
            matches = greedy_matches(fibers, h_fibers)
            bie_to_human[grader] = {m.reference_index: m for m in matches}
        for j, fiber in enumerate(fibers):
            if j in bie_to_fp[name]:
                continue
            grader_ious = {
                grader: bie_to_human[grader][j].iou if j in bie_to_human[grader] else None
                for grader in GRADERS
            }
            human_support = sum(value is not None for value in grader_ious.values())
            row = {
                "version": name,
                "bie_fiber_index": j,
                "length_px": float(fiber.length),
                "length_um": float(fiber.length) * PIXEL_SIZE_UM,
                "ratio": float(fiber.ratio),
                "human_support_count": human_support,
                "human_support_label": support_label(human_support),
            }
            for grader in GRADERS:
                row[f"{grader}_matched"] = grader_ious[grader] is not None
                row[f"{grader}_iou"] = grader_ious[grader]
            extra_rows.append(row)

    extra_csv = output_root / "bie_only_fibers_and_human_support.csv"
    if extra_rows:
        with extra_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(extra_rows[0].keys()))
            writer.writeheader()
            writer.writerows(extra_rows)
    else:
        extra_csv.write_text("No BIE-only fibers across available variants.\n", encoding="utf-8")

    print("Radiation Edge AI - INT8 fiber disagreement diagnosis")
    print(f"Sample: {args.sample_id}")
    print(f"FP512 valid fibers: {len(fp_fibers)}")
    print("Available variants: " + ", ".join(available))
    print("")

    for name, fibers in bie_fibers.items():
        n_match = len(fp_to_bie[name])
        n_extra = len(fibers) - n_match
        n_missed = len(fp_fibers) - n_match
        print(
            f"{name}: BIE={len(fibers)} matched={n_match} "
            f"BIE-only={n_extra} FP-missed={n_missed}"
        )

    print("")
    latest = "v4" if "v4" in available else list(available)[-1]
    missed_indices = [
        i for i in range(len(fp_fibers)) if i not in fp_to_bie[latest]
    ]
    print(f"[{latest} FP-missed fibers]")
    if not missed_indices:
        print("none")
    else:
        for i in missed_indices:
            row = fp_rows[i]
            retained = ",".join(
                name for name in available if row[f"{name}_retained"]
            ) or "none"
            print(
                f"FP#{i:02d} length={row['length_um']:.3f}um "
                f"ratio={row['ratio']:.4f} humans={row['human_support_count']}/4 "
                f"({row['human_support_label']}) retained_by={retained}"
            )

    print("")
    print("[Human support among all FP512 fibers]")
    for support in range(5):
        count = sum(r["human_support_count"] == support for r in fp_rows)
        print(f"support {support}/4: {count}")

    print("")
    print(f"FP-fiber table: {fp_csv}")
    print(f"BIE-only table: {extra_csv}")
    print("")
    print(
        "Interpretation rule: loss of an FP fiber with 3-4/4 human support is a "
        "stronger biological concern than suppression of a 0/4-supported FP object."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
