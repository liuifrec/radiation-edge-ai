"""Crosswalk FP-missed and BIE-only fibers through the same human targets.

This is a cheap post-hoc diagnostic. It uses already-generated stitched
segmentations and the frozen H1-H4 annotations; it performs no Kneron
simulation and no model conversion.

A hard FP-vs-BIE bbox-IoU threshold can turn one geometry-shifted biological
object into an apparent false negative plus false positive. For each requested
PTQ variant, this helper therefore asks whether unmatched FP and BIE fibers are
matched to the *same human fiber* within each grader. Shared human targets are
reported together with the direct FP-vs-BIE bbox IoU.

This does not make the human annotations infallible ground truth. It is a
diagnostic for distinguishing true object replacement from threshold-sensitive
geometry drift.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dnafiber.postprocess import refine_segmentation

from radiation_edge_ai.dna_fiber.diagnose_int8_fiber_disagreements import (
    GRADERS,
    MATCH_IOU,
    greedy_matches,
    read_human_mask,
    read_segmentation,
    sha256_file,
)


VARIANT_DIRS = {
    "v1": "int8_512",
    "v2": "int8_512_v2",
    "v3": "int8_512_v3_pct100",
    "v4": "int8_512_v4_bias",
}


def parse_variants(value: str) -> list[str]:
    variants = [token.strip() for token in value.split(",") if token.strip()]
    unknown = [item for item in variants if item not in VARIANT_DIRS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown variants: {unknown}")
    if not variants:
        raise argparse.ArgumentTypeError("No variants selected")
    return variants


def main() -> int:
    parser = argparse.ArgumentParser()
    default_data_root = Path(
        os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data")
    )
    default_model_root = Path(
        os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models")
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--variants", default="v3,v4")
    parser.add_argument("--data-root", default=str(default_data_root))
    parser.add_argument("--model-root", default=str(default_model_root))
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    data_root = Path(args.data_root)
    model_root = Path(args.model_root)
    kneron_root = model_root / "dnai" / "unet_mobileone_s1" / "kneron"
    panel_root = data_root / "dnai_public_v2" / "r1_intergrader20"
    dataset_root = data_root / "dnai_public_v2" / "extracted" / "zenodo"

    panel_manifest = json.loads((panel_root / "manifest.json").read_text(encoding="utf-8"))
    panel_by_sample = {record["sample_id"]: record for record in panel_manifest["records"]}
    record = panel_by_sample.get(args.sample_id)
    if record is None:
        raise RuntimeError(f"Unknown frozen-panel sample: {args.sample_id}")

    available: dict[str, Path] = {}
    for variant in variants:
        root = kneron_root / VARIANT_DIRS[variant] / "biological_fidelity" / args.sample_id
        if (root / "floating_segmentation_class_ids.png").is_file() and (
            root / "bie_segmentation_class_ids.png"
        ).is_file():
            available[variant] = root
    if not available:
        raise RuntimeError("None of the requested variants has completed stitched outputs")

    floating_masks = {
        name: read_segmentation(root / "floating_segmentation_class_ids.png")
        for name, root in available.items()
    }
    first = next(iter(floating_masks))
    floating_seg = floating_masks[first]
    for name, mask in floating_masks.items():
        if not (mask == floating_seg).all():
            raise RuntimeError(f"Floating segmentation differs between {first} and {name}")

    fp_fibers = list(refine_segmentation(floating_seg).valid_copy())

    human_fibers = {}
    for grader in GRADERS:
        annotation = dataset_root / record[f"{grader}_annotation"]
        if sha256_file(annotation) != record[f"{grader}_sha256"]:
            raise RuntimeError(f"Annotation checksum mismatch: {annotation}")
        human_fibers[grader] = list(refine_segmentation(read_human_mask(annotation)).valid_copy())

    fp_human_maps = {}
    for grader, fibers in human_fibers.items():
        matches = greedy_matches(fp_fibers, fibers)
        fp_human_maps[grader] = {match.reference_index: match.candidate_index for match in matches}

    print("Radiation Edge AI - unmatched-fiber human-target crosswalk")
    print(f"Sample: {args.sample_id}")
    print(f"Strict FP/BIE match threshold: bbox IoU >= {MATCH_IOU:.2f}")
    print("Variants: " + ", ".join(available))

    for variant, root in available.items():
        bie_fibers = list(
            refine_segmentation(read_segmentation(root / "bie_segmentation_class_ids.png")).valid_copy()
        )
        strict = greedy_matches(fp_fibers, bie_fibers)
        matched_fp = {match.reference_index for match in strict}
        matched_bie = {match.candidate_index for match in strict}
        missed_fp = [index for index in range(len(fp_fibers)) if index not in matched_fp]
        extra_bie = [index for index in range(len(bie_fibers)) if index not in matched_bie]

        bie_human_maps = {}
        for grader, fibers in human_fibers.items():
            matches = greedy_matches(bie_fibers, fibers)
            bie_human_maps[grader] = {
                match.reference_index: match.candidate_index for match in matches
            }

        candidates = []
        for fp_index in missed_fp:
            for bie_index in extra_bie:
                shared = []
                for grader in GRADERS:
                    fp_target = fp_human_maps[grader].get(fp_index)
                    bie_target = bie_human_maps[grader].get(bie_index)
                    if fp_target is not None and fp_target == bie_target:
                        shared.append(f"{grader}#{fp_target}")
                direct_iou = float(fp_fibers[fp_index].bbox_iou(bie_fibers[bie_index]))
                if shared or direct_iou >= 0.10:
                    candidates.append(
                        (
                            len(shared),
                            direct_iou,
                            fp_index,
                            bie_index,
                            shared,
                        )
                    )

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

        print("")
        print(
            f"[{variant}] strict-matched={len(strict)} FP-missed={len(missed_fp)} "
            f"BIE-only={len(extra_bie)}"
        )
        if not missed_fp and not extra_bie:
            print("No unmatched fibers.")
            continue

        if candidates:
            print("Candidate crosswalks among unmatched fibers:")
            for shared_count, direct_iou, fp_index, bie_index, shared in candidates:
                fp = fp_fibers[fp_index]
                bie = bie_fibers[bie_index]
                shared_text = ",".join(shared) if shared else "none"
                print(
                    f"  FP#{fp_index:02d} <-> BIE#{bie_index:02d} "
                    f"shared-human-targets={shared_count}/4 [{shared_text}] "
                    f"directIoU={direct_iou:.3f} "
                    f"FP(len={float(fp.length)*0.26:.3f}um ratio={float(fp.ratio):.4f}) "
                    f"BIE(len={float(bie.length)*0.26:.3f}um ratio={float(bie.ratio):.4f})"
                )
        else:
            print("No unmatched FP/BIE pair shares a human target and none has direct IoU >= 0.10.")

    print("")
    print(
        "Interpretation: if an FP-missed and BIE-only pair shares the same human target "
        "in multiple graders, the strict IoU=0.5 bookkeeping may be counting one "
        "geometry-shifted biological object as both a miss and a new object."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
