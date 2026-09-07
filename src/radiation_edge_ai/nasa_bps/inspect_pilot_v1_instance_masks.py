"""Diagnose target-nucleus instance labels in NASA BPS pilot-v1 MASK crops.

The BPS MASK TIFFs are instance-label images, not simple binary masks.  A crop
can therefore contain the intended nucleus plus fragments of neighboring nuclei.
This script tests whether the final object index encoded in the filename maps to
the intended MASK label and quantifies contamination from other labels.

No model training or image modification is performed.  Run with the existing
DNAi Python environment (numpy + OpenCV).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "OpenCV is required. Run with "
        "D:\\radiation-edge-ai-data\\envs\\dnai311\\Scripts\\python.exe"
    ) from exc

FROZEN_MANIFEST_SHA256 = "41ff2e4d64577f17c976dea5bbc17221fe63be9e2e7e2291309b520027fbe725"
MASK_RE = re.compile(
    r"^(?P<plate>P\d+)_(?P<acquisition>\d+)-(?P<well>[A-Za-z]+\d+)_"
    r"(?P<field>\d+)_(?P<object>\d+)_MASK\.tif$",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV could not read MASK TIFF: {path}")
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim != 2:
        raise RuntimeError(f"Expected 2D MASK TIFF, got shape={image.shape}: {path}")
    return image


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    meta_root = data_root / "nasa_bps_microscopy" / "metadata" / "pilot_v1"
    pilot_root = data_root / "nasa_bps_microscopy" / "pilot_v1"
    parser.add_argument("--manifest", default=str(meta_root / "pilot_v1_manifest.csv"))
    parser.add_argument("--mask-dir", default=str(pilot_root / "images" / "mask"))
    parser.add_argument("--output-dir", default=str(pilot_root / "mask_qc"))
    parser.add_argument("--expected-manifest-sha256", default=FROZEN_MANIFEST_SHA256)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    mask_dir = Path(args.mask_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_sha = sha256_file(manifest_path)
    if args.expected_manifest_sha256 and manifest_sha != args.expected_manifest_sha256:
        raise RuntimeError(
            f"Frozen manifest SHA256 mismatch: expected {args.expected_manifest_sha256}, "
            f"got {manifest_sha}"
        )

    rows = read_csv(manifest_path)
    results: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        filename = row["mask_filename"]
        path = mask_dir / filename
        try:
            match = MASK_RE.match(Path(filename).name)
            if match is None:
                raise ValueError(f"Unrecognized MASK filename: {filename}")
            expected_label = int(match.group("object"))
            mask = load_mask(path)
            labels, counts = np.unique(mask, return_counts=True)
            nonzero = [(int(label), int(count)) for label, count in zip(labels, counts) if int(label) != 0]
            nonzero_labels = [label for label, _ in nonzero]
            union_px = int(sum(count for _, count in nonzero))
            largest_label = max(nonzero, key=lambda x: x[1])[0] if nonzero else 0
            largest_px = max((count for _, count in nonzero), default=0)

            expected_present = expected_label in nonzero_labels
            target_px = 0
            centroid_y = None
            centroid_x = None
            center_offset = None
            touches_border = False
            target_fraction_union = None
            contamination_fraction = None
            if expected_present:
                target = mask == expected_label
                target_px = int(target.sum())
                ys, xs = np.nonzero(target)
                centroid_y = float(np.mean(ys))
                centroid_x = float(np.mean(xs))
                cy = (mask.shape[0] - 1) / 2.0
                cx = (mask.shape[1] - 1) / 2.0
                diagonal = float(np.hypot(mask.shape[0], mask.shape[1]))
                center_offset = float(np.hypot(centroid_y - cy, centroid_x - cx) / diagonal)
                touches_border = bool(
                    np.any(target[0, :])
                    or np.any(target[-1, :])
                    or np.any(target[:, 0])
                    or np.any(target[:, -1])
                )
                target_fraction_union = float(target_px / union_px) if union_px else None
                contamination_fraction = float((union_px - target_px) / union_px) if union_px else None

            center_label = int(mask[mask.shape[0] // 2, mask.shape[1] // 2])
            results.append(
                {
                    "sample_id": sample_id,
                    "mask_filename": filename,
                    "source_name": row["source_name"],
                    "particle_type": row["particle_type"],
                    "dose_Gy": row["dose_Gy"],
                    "hr_post_exposure": row["hr_post_exposure"],
                    "height": int(mask.shape[0]),
                    "width": int(mask.shape[1]),
                    "expected_object_label": expected_label,
                    "expected_label_present": expected_present,
                    "n_nonzero_labels": len(nonzero_labels),
                    "nonzero_labels": ";".join(str(v) for v in nonzero_labels),
                    "largest_label": largest_label,
                    "expected_is_largest": bool(expected_present and expected_label == largest_label),
                    "center_pixel_label": center_label,
                    "center_pixel_matches_expected": bool(center_label == expected_label),
                    "union_mask_px": union_px,
                    "target_mask_px": target_px,
                    "largest_mask_px": largest_px,
                    "target_fraction_of_union": target_fraction_union,
                    "other_label_contamination_fraction": contamination_fraction,
                    "target_centroid_y": centroid_y,
                    "target_centroid_x": centroid_x,
                    "target_center_offset_fraction_diagonal": center_offset,
                    "target_touches_border": touches_border,
                }
            )
        except Exception as exc:
            errors.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})

        if index == 1 or index % 250 == 0 or index == len(rows):
            print(f"[{index:>5}/{len(rows)}] errors={len(errors)}")

    per_nucleus_path = output_dir / "instance_mask_diagnostics.csv"
    write_csv(per_nucleus_path, results)
    if errors:
        write_csv(output_dir / "errors.csv", errors)

    n = len(results)
    present = sum(bool(r["expected_label_present"]) for r in results)
    largest = sum(bool(r["expected_is_largest"]) for r in results)
    center_match = sum(bool(r["center_pixel_matches_expected"]) for r in results)
    multi = sum(int(r["n_nonzero_labels"]) > 1 for r in results)
    single = sum(int(r["n_nonzero_labels"]) == 1 for r in results)
    border = sum(bool(r["target_touches_border"]) for r in results if r["expected_label_present"])
    label_counts = Counter(int(r["n_nonzero_labels"]) for r in results)
    contamination = [
        float(r["other_label_contamination_fraction"])
        for r in results
        if r["other_label_contamination_fraction"] is not None
    ]
    center_offsets = [
        float(r["target_center_offset_fraction_diagonal"])
        for r in results
        if r["target_center_offset_fraction_diagonal"] is not None
    ]

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_version": "NASA_BPS_53BP1_PILOT_V1",
        "manifest_sha256": manifest_sha,
        "n_requested": len(rows),
        "n_successful": n,
        "n_errors": len(errors),
        "expected_object_label_present": present,
        "expected_object_label_present_fraction": present / n if n else 0.0,
        "expected_object_label_is_largest": largest,
        "expected_object_label_is_largest_fraction": largest / n if n else 0.0,
        "center_pixel_matches_expected": center_match,
        "center_pixel_matches_expected_fraction": center_match / n if n else 0.0,
        "single_nonzero_label_masks": single,
        "multiple_nonzero_label_masks": multi,
        "nonzero_label_count_distribution": dict(sorted(label_counts.items())),
        "target_touches_border": border,
        "contamination_fraction_median": percentile(contamination, 50),
        "contamination_fraction_p95": percentile(contamination, 95),
        "contamination_fraction_max": max(contamination) if contamination else None,
        "target_center_offset_fraction_diagonal_median": percentile(center_offsets, 50),
        "target_center_offset_fraction_diagonal_p95": percentile(center_offsets, 95),
        "interpretation": [
            "MASK TIFFs are treated as instance-label images rather than binary masks.",
            "The expected target label is the final object index encoded before _MASK.tif.",
            "If expected-label presence is 100%, use mask == expected_object_label for the target nucleus.",
            "Do not use mask > 0 for target-nucleus intensity or foci endpoints when neighboring labels are present.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Radiation Edge AI - NASA BPS pilot-v1 instance-mask diagnostic")
    print(f"Manifest SHA256: {manifest_sha}")
    print(f"successful masks: {n}/{len(rows)}")
    print(f"errors: {len(errors)}")
    print("")
    print("[Target-label mapping]")
    print(f"expected filename object label present: {present}/{n}")
    print(f"expected label is largest nonzero instance: {largest}/{n}")
    print(f"center pixel matches expected label: {center_match}/{n}")
    print("")
    print("[Neighbor-label contamination]")
    print(f"single-label masks: {single}/{n}")
    print(f"multi-label masks: {multi}/{n}")
    print(f"nonzero label count distribution: {dict(sorted(label_counts.items()))}")
    if contamination:
        print(
            "other-label contamination fraction median/p95/max: "
            f"{percentile(contamination, 50):.6f}/"
            f"{percentile(contamination, 95):.6f}/"
            f"{max(contamination):.6f}"
        )
    print(f"target touches crop border: {border}/{present}")
    if center_offsets:
        print(
            "target centroid offset (fraction of crop diagonal) median/p95: "
            f"{percentile(center_offsets, 50):.6f}/{percentile(center_offsets, 95):.6f}"
        )
    print("")
    print(f"Per-nucleus diagnostics: {per_nucleus_path}")
    print(f"Summary: {summary_path}")
    print("")

    if errors or present != n:
        print("NASA BPS PILOT V1 TARGET-MASK RESOLUTION COMPLETE: NO")
        print(
            "NEXT GATE: inspect missing/ambiguous target labels before deriving any target-nucleus "
            "53BP1 endpoint."
        )
        return 1

    print("NASA BPS PILOT V1 TARGET-MASK RESOLUTION COMPLETE: YES")
    print("Target nucleus policy: MASK == filename object index; do not use MASK > 0.")
    print(
        "NEXT GATE: regenerate target-only visual/quantitative QC and then freeze the 53BP1 "
        "focus-map reference endpoint on native-scale 256x256 padded inputs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
