"""Rebuild NASA BPS pilot-v1 QC using the resolved target instance only.

The BPS MASK TIFFs are instance-label images.  The intended nucleus is the
instance whose integer label equals the final object index encoded in the
filename, e.g. ``..._001_014_MASK.tif`` -> ``mask == 14``.

This script replaces the exploratory ``mask > 0`` QC with a target-only pass:
- verifies the frozen pilot manifest SHA256;
- reads FITC/DAPI/MASK triplets unchanged;
- isolates only ``MASK == filename_object_index``;
- records target-only intensity/geometry statistics per nucleus;
- summarizes each source x radiation-condition cell before condition-level QC;
- chooses one representative nucleus per source x condition deterministically;
- writes adaptive and fixed-scale contact sheets with only the target nucleus
  visible and only the target-instance boundary outlined;
- verifies that every native crop fits the frozen 256 x 256 padded canvas.

These are descriptive pre-modeling diagnostics only.  No 53BP1 focus threshold,
foci count, training label, or biological-equivalence claim is created here.
Run with the existing DNAi Python 3.11 environment (numpy + OpenCV).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import defaultdict
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
CANVAS = 256
SOURCES = (
    "BALBCF1",
    "BALBCM1",
    "BALBCM2",
    "C57BLF1",
    "C57BLM1",
    "C57BLM3",
)
CONDITIONS = (
    ("Fe", "0.0", "4"),
    ("Fe", "0.0", "24"),
    ("Fe", "0.0", "48"),
    ("Fe", "0.82", "4"),
    ("Fe", "0.82", "24"),
    ("Fe", "0.82", "48"),
    ("X-ray", "0.0", "4"),
    ("X-ray", "0.0", "24"),
    ("X-ray", "0.0", "48"),
    ("X-ray", "1.0", "4"),
    ("X-ray", "1.0", "24"),
    ("X-ray", "1.0", "48"),
)
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


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV could not read TIFF: {path}")
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim != 2:
        raise RuntimeError(f"Expected 2D single-channel TIFF, got {image.shape}: {path}")
    return image


def target_label_from_filename(filename: str) -> int:
    match = MASK_RE.match(Path(filename).name)
    if match is None:
        raise ValueError(f"Unrecognized MASK filename: {filename}")
    return int(match.group("object"))


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values).reshape(-1).astype(np.float64, copy=False)
    if values.size == 0:
        raise RuntimeError("Cannot summarize an empty target mask")
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "p995": float(np.percentile(values, 99.5)),
        "max": float(np.max(values)),
    }


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def condition_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row["particle_type"]),
        str(row["dose_Gy"]),
        str(row["hr_post_exposure"]),
    )


def robust_scale(image: np.ndarray, target: np.ndarray) -> np.ndarray:
    values = image[target]
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.5))
    if high <= low:
        low = float(np.min(values))
        high = float(np.max(values))
    if high <= low:
        out = np.zeros(image.shape, dtype=np.uint8)
    else:
        scaled = (image.astype(np.float32) - low) * (255.0 / (high - low))
        out = np.clip(scaled, 0, 255).astype(np.uint8)
    out[~target] = 0
    return out


def fixed_scale(image: np.ndarray, target: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        out = np.zeros(image.shape, dtype=np.uint8)
    else:
        scaled = (image.astype(np.float32) - low) * (255.0 / (high - low))
        out = np.clip(scaled, 0, 255).astype(np.uint8)
    out[~target] = 0
    return out


def target_boundary(target: np.ndarray) -> np.ndarray:
    binary = target.astype(np.uint8) * 255
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.subtract(binary, eroded) > 0


def composite(fitc8: np.ndarray, dapi8: np.ndarray, target: np.ndarray) -> np.ndarray:
    out = np.zeros((*fitc8.shape, 3), dtype=np.uint8)
    # OpenCV BGR ordering: B=DAPI, G=FITC, R=target boundary.
    out[:, :, 0] = dapi8
    out[:, :, 1] = fitc8
    out[target_boundary(target), 2] = 255
    return out


def to_canvas(panel: np.ndarray) -> np.ndarray:
    height, width = panel.shape[:2]
    if height > CANVAS or width > CANVAS:
        raise RuntimeError(
            f"Native crop {height}x{width} exceeds frozen {CANVAS}x{CANVAS} canvas"
        )
    canvas = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
    y0 = (CANVAS - height) // 2
    x0 = (CANVAS - width) // 2
    canvas[y0 : y0 + height, x0 : x0 + width] = panel
    return canvas


def label_canvas(canvas: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.45) -> None:
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )


def choose_representatives(
    manifest_by_id: dict[str, dict[str, str]],
    nucleus_rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    groups: dict[tuple[str, tuple[str, str, str]], list[dict[str, object]]] = defaultdict(list)
    for row in nucleus_rows:
        source = str(row["source_name"])
        cond = condition_key(row)
        if source in SOURCES and cond in CONDITIONS:
            groups[(source, cond)].append(row)

    selected: list[dict[str, str]] = []
    missing: list[str] = []
    for cond in CONDITIONS:
        for source in SOURCES:
            subset = groups.get((source, cond), [])
            if not subset:
                missing.append(f"{source}|{cond}")
                continue
            areas = np.asarray([float(r["target_area_px"]) for r in subset], dtype=np.float64)
            p99s = np.asarray([float(r["fitc_target_p99"]) for r in subset], dtype=np.float64)
            median_area = float(np.median(areas))
            median_p99 = float(np.median(p99s))
            mad_area = float(np.median(np.abs(areas - median_area))) or 1.0
            mad_p99 = float(np.median(np.abs(p99s - median_p99))) or 1.0

            def score(row: dict[str, object]) -> tuple[float, str]:
                area_z = abs(float(row["target_area_px"]) - median_area) / mad_area
                p99_z = abs(float(row["fitc_target_p99"]) - median_p99) / mad_p99
                return area_z + p99_z, str(row["sample_id"])

            chosen_qc = min(subset, key=score)
            chosen = dict(manifest_by_id[str(chosen_qc["sample_id"])])
            chosen["representative_score"] = f"{score(chosen_qc)[0]:.8f}"
            selected.append(chosen)

    if missing:
        raise RuntimeError(f"Missing representative cells: {missing}")
    return selected


def derive_fixed_limits(selected: list[dict[str, str]], image_dir: Path) -> dict[str, tuple[float, float]]:
    values_by_channel: dict[str, list[np.ndarray]] = {"fitc": [], "dapi": []}
    for row in selected:
        mask = load_gray(image_dir / "mask" / row["mask_filename"])
        label = target_label_from_filename(row["mask_filename"])
        target = mask == label
        if not np.any(target):
            raise RuntimeError(f"Target label {label} absent: {row['mask_filename']}")
        for channel, key in (("fitc", "fitc_filename"), ("dapi", "dapi_filename")):
            image = load_gray(image_dir / channel / row[key])
            vals = image[target]
            values_by_channel[channel].append(vals.astype(np.float32, copy=False))

    limits: dict[str, tuple[float, float]] = {}
    for channel in ("fitc", "dapi"):
        values = np.concatenate(values_by_channel[channel])
        limits[channel] = (
            float(np.percentile(values, 1.0)),
            float(np.percentile(values, 99.5)),
        )
    return limits


def render_sheet(
    selected: list[dict[str, str]],
    image_dir: Path,
    mode: str,
    fixed_limits: dict[str, tuple[float, float]] | None,
) -> np.ndarray:
    header_h = 48
    row_label_w = 150
    tile = CANVAS
    gap = 4
    width = row_label_w + len(SOURCES) * (tile + gap) + gap
    height = header_h + len(CONDITIONS) * (tile + gap) + gap
    sheet = np.zeros((height, width, 3), dtype=np.uint8)

    for col, source in enumerate(SOURCES):
        x = row_label_w + col * (tile + gap) + 4
        label_canvas(sheet, source, (x + 4, 28), scale=0.48)

    selected_map = {(row["source_name"], condition_key(row)): row for row in selected}
    for r, cond in enumerate(CONDITIONS):
        y = header_h + r * (tile + gap) + gap
        particle, dose, hour = cond
        label_canvas(sheet, f"{particle} {dose} Gy", (6, y + 24), scale=0.45)
        label_canvas(sheet, f"{hour} h", (6, y + 46), scale=0.45)
        for c, source in enumerate(SOURCES):
            row = selected_map[(source, cond)]
            fitc = load_gray(image_dir / "fitc" / row["fitc_filename"])
            dapi = load_gray(image_dir / "dapi" / row["dapi_filename"])
            mask = load_gray(image_dir / "mask" / row["mask_filename"])
            if fitc.shape != dapi.shape or fitc.shape != mask.shape:
                raise RuntimeError(f"Triplet shape mismatch for {row['sample_id']}")
            label = target_label_from_filename(row["mask_filename"])
            target = mask == label
            if not np.any(target):
                raise RuntimeError(f"Target label {label} absent for {row['sample_id']}")
            if mode == "adaptive":
                fitc8 = robust_scale(fitc, target)
                dapi8 = robust_scale(dapi, target)
            elif mode == "fixed":
                if fixed_limits is None:
                    raise RuntimeError("fixed mode requires fixed display limits")
                fitc8 = fixed_scale(fitc, target, *fixed_limits["fitc"])
                dapi8 = fixed_scale(dapi, target, *fixed_limits["dapi"])
            else:
                raise ValueError(mode)
            panel = to_canvas(composite(fitc8, dapi8, target))
            x = row_label_w + c * (tile + gap) + gap
            sheet[y : y + tile, x : x + tile] = panel
            label_canvas(sheet, row["sample_id"], (x + 4, y + tile - 8), scale=0.34)
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    metadata = data_root / "nasa_bps_microscopy" / "metadata" / "pilot_v1"
    pilot = data_root / "nasa_bps_microscopy" / "pilot_v1"
    parser.add_argument("--manifest", default=str(metadata / "pilot_v1_manifest.csv"))
    parser.add_argument("--image-dir", default=str(pilot / "images"))
    parser.add_argument("--output-dir", default=str(pilot / "target_qc"))
    parser.add_argument("--expected-manifest-sha256", default=FROZEN_MANIFEST_SHA256)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    image_dir = Path(args.image_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_sha = sha256_file(manifest_path)
    if args.expected_manifest_sha256 and manifest_sha != args.expected_manifest_sha256:
        raise RuntimeError(
            f"Frozen pilot-v1 manifest SHA256 mismatch: expected {args.expected_manifest_sha256}, "
            f"got {manifest_sha}"
        )

    manifest_rows = read_csv(manifest_path)
    manifest_by_id = {row["sample_id"]: row for row in manifest_rows}
    nucleus_rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    heights: list[int] = []
    widths: list[int] = []

    print("Radiation Edge AI - rebuild NASA BPS pilot-v1 target-only QC")
    print(f"Manifest SHA256: {manifest_sha}")
    print(f"Nuclei: {len(manifest_rows)}")
    print("")

    for index, row in enumerate(manifest_rows, start=1):
        sample_id = row["sample_id"]
        try:
            fitc = load_gray(image_dir / "fitc" / row["fitc_filename"])
            dapi = load_gray(image_dir / "dapi" / row["dapi_filename"])
            mask = load_gray(image_dir / "mask" / row["mask_filename"])
            if fitc.shape != dapi.shape or fitc.shape != mask.shape:
                raise RuntimeError(
                    f"Triplet shape mismatch: FITC={fitc.shape}, DAPI={dapi.shape}, MASK={mask.shape}"
                )
            label = target_label_from_filename(row["mask_filename"])
            target = mask == label
            if not np.any(target):
                raise RuntimeError(f"Expected target label {label} absent")
            center_label = int(mask[mask.shape[0] // 2, mask.shape[1] // 2])
            if center_label != label:
                raise RuntimeError(f"Center pixel label {center_label} != expected target {label}")

            nonzero_px = int(np.sum(mask > 0))
            target_px = int(np.sum(target))
            neighbor_px = nonzero_px - target_px
            fitc_stats = stats(fitc[target])
            dapi_stats = stats(dapi[target])
            ys, xs = np.nonzero(target)
            target_h = int(np.max(ys) - np.min(ys) + 1)
            target_w = int(np.max(xs) - np.min(xs) + 1)
            touches_border = bool(
                np.any(target[0, :])
                or np.any(target[-1, :])
                or np.any(target[:, 0])
                or np.any(target[:, -1])
            )
            heights.append(int(mask.shape[0]))
            widths.append(int(mask.shape[1]))

            nucleus_rows.append(
                {
                    "sample_id": sample_id,
                    "source_name": row["source_name"],
                    "sample_name": row["sample_name"],
                    "strain": row["strain"],
                    "sex": row["sex"],
                    "particle_type": row["particle_type"],
                    "dose_Gy": row["dose_Gy"],
                    "hr_post_exposure": row["hr_post_exposure"],
                    "height": int(mask.shape[0]),
                    "width": int(mask.shape[1]),
                    "target_label": label,
                    "target_area_px": target_px,
                    "target_bbox_height_px": target_h,
                    "target_bbox_width_px": target_w,
                    "target_touches_crop_border": touches_border,
                    "nonzero_union_area_px": nonzero_px,
                    "neighbor_area_px": neighbor_px,
                    "neighbor_fraction_of_union": float(neighbor_px / nonzero_px) if nonzero_px else 0.0,
                    "fitc_target_mean": fitc_stats["mean"],
                    "fitc_target_p50": fitc_stats["p50"],
                    "fitc_target_p95": fitc_stats["p95"],
                    "fitc_target_p99": fitc_stats["p99"],
                    "fitc_target_p995": fitc_stats["p995"],
                    "fitc_target_max": fitc_stats["max"],
                    "fitc_target_integrated": float(np.sum(fitc[target].astype(np.float64))),
                    "dapi_target_mean": dapi_stats["mean"],
                    "dapi_target_p99": dapi_stats["p99"],
                }
            )
        except Exception as exc:
            errors.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})

        if index == 1 or index % 250 == 0 or index == len(manifest_rows):
            print(f"[{index:>5}/{len(manifest_rows)}] errors={len(errors)}")

    per_nucleus_path = output_dir / "per_nucleus_target_stats.csv"
    write_csv(per_nucleus_path, nucleus_rows)
    if errors:
        write_csv(output_dir / "errors.csv", errors)

    # First summarize the 30 nuclei inside each biological source x condition.
    source_groups: dict[tuple[str, tuple[str, str, str]], list[dict[str, object]]] = defaultdict(list)
    for row in nucleus_rows:
        source_groups[(str(row["source_name"]), condition_key(row))].append(row)

    source_condition_rows: list[dict[str, object]] = []
    for (source, cond), subset in sorted(
        source_groups.items(), key=lambda item: (item[0][1][0], float(item[0][1][1]), float(item[0][1][2]), item[0][0])
    ):
        source_condition_rows.append(
            {
                "source_name": source,
                "particle_type": cond[0],
                "dose_Gy": cond[1],
                "hr_post_exposure": cond[2],
                "n_nuclei": len(subset),
                "median_target_area_px": median([float(r["target_area_px"]) for r in subset]),
                "median_fitc_target_mean": median([float(r["fitc_target_mean"]) for r in subset]),
                "median_fitc_target_p99": median([float(r["fitc_target_p99"]) for r in subset]),
                "median_dapi_target_mean": median([float(r["dapi_target_mean"]) for r in subset]),
            }
        )

    source_condition_path = output_dir / "source_condition_target_stats.csv"
    write_csv(source_condition_path, source_condition_rows)

    # Then summarize across the six independent Source Names, avoiding n=180 language.
    condition_groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in source_condition_rows:
        condition_groups[condition_key(row)].append(row)

    condition_rows: list[dict[str, object]] = []
    for cond, subset in sorted(
        condition_groups.items(), key=lambda item: (item[0][0], float(item[0][1]), float(item[0][2]))
    ):
        condition_rows.append(
            {
                "particle_type": cond[0],
                "dose_Gy": cond[1],
                "hr_post_exposure": cond[2],
                "n_sources": len(subset),
                "n_nuclei_total": int(sum(int(r["n_nuclei"]) for r in subset)),
                "mean_of_source_median_target_area_px": mean([float(r["median_target_area_px"]) for r in subset]),
                "median_of_source_median_target_area_px": median([float(r["median_target_area_px"]) for r in subset]),
                "mean_of_source_median_fitc_target_mean": mean([float(r["median_fitc_target_mean"]) for r in subset]),
                "median_of_source_median_fitc_target_mean": median([float(r["median_fitc_target_mean"]) for r in subset]),
                "mean_of_source_median_fitc_target_p99": mean([float(r["median_fitc_target_p99"]) for r in subset]),
                "median_of_source_median_fitc_target_p99": median([float(r["median_fitc_target_p99"]) for r in subset]),
            }
        )

    condition_path = output_dir / "condition_target_descriptive_stats.csv"
    write_csv(condition_path, condition_rows)

    selected = choose_representatives(manifest_by_id, nucleus_rows)
    fixed_limits = derive_fixed_limits(selected, image_dir)
    adaptive = render_sheet(selected, image_dir, "adaptive", None)
    fixed = render_sheet(selected, image_dir, "fixed", fixed_limits)
    adaptive_path = output_dir / "pilot_v1_target_only_contact_sheet_adaptive.png"
    fixed_path = output_dir / "pilot_v1_target_only_contact_sheet_fixed.png"
    if not cv2.imwrite(str(adaptive_path), adaptive):
        raise RuntimeError(f"Could not write {adaptive_path}")
    if not cv2.imwrite(str(fixed_path), fixed):
        raise RuntimeError(f"Could not write {fixed_path}")

    representative_path = output_dir / "representative_target_nuclei.csv"
    write_csv(representative_path, [dict(row) for row in selected])

    max_dims = np.maximum(np.asarray(heights), np.asarray(widths)) if heights else np.asarray([])
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_version": "NASA_BPS_53BP1_PILOT_V1",
        "manifest_sha256": manifest_sha,
        "target_nucleus_policy": "MASK == filename object index",
        "n_requested": len(manifest_rows),
        "n_successful": len(nucleus_rows),
        "n_errors": len(errors),
        "n_sources": len({str(r["source_name"]) for r in nucleus_rows}),
        "n_conditions": len({condition_key(r) for r in nucleus_rows}),
        "all_native_crops_fit_256": bool(max_dims.size and np.all(max_dims <= CANVAS)),
        "max_native_crop_dimension": int(np.max(max_dims)) if max_dims.size else None,
        "targets_touching_crop_border": int(sum(bool(r["target_touches_crop_border"]) for r in nucleus_rows)),
        "fixed_display_limits": {
            channel: {"p01": limits[0], "p995": limits[1]}
            for channel, limits in fixed_limits.items()
        },
        "outputs": {
            "per_nucleus_target_stats": str(per_nucleus_path),
            "source_condition_target_stats": str(source_condition_path),
            "condition_target_descriptive_stats": str(condition_path),
            "adaptive_target_contact_sheet": str(adaptive_path),
            "fixed_target_contact_sheet": str(fixed_path),
            "representative_target_nuclei": str(representative_path),
        },
        "notes": [
            "Only the filename-resolved target instance is used for target-nucleus statistics.",
            "Pixels belonging to neighboring MASK labels are excluded.",
            "Condition summaries are built from Source Name summaries; the biological n is six sources, not 180 nuclei.",
            "Raw fluorescence summaries remain descriptive QC and are not treated as a validated 53BP1 foci endpoint.",
            "No focus threshold or pseudo-label is inferred by this script.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("[Target-only QC]")
    print(f"successful nuclei: {len(nucleus_rows)}/{len(manifest_rows)}")
    print(f"errors: {len(errors)}")
    print(f"target nuclei touching crop border: {summary['targets_touching_crop_border']}/{len(nucleus_rows)}")
    print(f"maximum native crop dimension: {summary['max_native_crop_dimension']}")
    print(f"all native crops fit 256x256: {'YES' if summary['all_native_crops_fit_256'] else 'NO'}")
    print("")
    print("[Condition descriptive QC from six Source Name summaries]")
    for row in condition_rows:
        print(
            f"{row['particle_type']:<5} {row['dose_Gy']:>4} Gy {row['hr_post_exposure']:>3} h "
            f"sources={row['n_sources']} nuclei={row['n_nuclei_total']} "
            f"source-median FITC mean={row['median_of_source_median_fitc_target_mean']:.3f} "
            f"p99={row['median_of_source_median_fitc_target_p99']:.3f}"
        )
    print("")
    print("[Target-only visual QC]")
    print(f"FITC fixed p01/p99.5: {fixed_limits['fitc'][0]:.3f}/{fixed_limits['fitc'][1]:.3f}")
    print(f"DAPI fixed p01/p99.5: {fixed_limits['dapi'][0]:.3f}/{fixed_limits['dapi'][1]:.3f}")
    print(f"Adaptive sheet: {adaptive_path}")
    print(f"Fixed sheet: {fixed_path}")
    print(f"Summary: {summary_path}")
    print("")

    if errors:
        print("NASA BPS PILOT V1 TARGET-ONLY QC COMPLETE: NO")
        return 1

    print("NASA BPS PILOT V1 TARGET-ONLY QC COMPLETE: YES")
    print("Target nucleus policy frozen for pilot v1: MASK == filename object index.")
    print(
        "NEXT GATE: inspect the corrected target-only contact sheets and descriptive statistics; "
        "then freeze the 256x256 reference endpoint/model policy before training."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
