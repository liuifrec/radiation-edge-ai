"""Create representative visual contact sheets for NASA BPS 53BP1 pilot v1.

This pre-modeling diagnostic selects one representative nucleus from every
source x radiation-condition cell in the frozen pilot, then renders two
contact sheets without changing spatial scale:

- adaptive: FITC and DAPI are contrast-scaled independently per nucleus so
  puncta/morphology can be inspected across dim and bright images;
- fixed: one shared FITC and DAPI display scale is used across the selected
  panel so gross acquisition/intensity differences remain visible.

Every nucleus is centered on a 256 x 256 canvas at native pixel resolution.
The NASA BPS pilot-v1 crops are expected to fit within 256 x 256; the script
fails rather than silently resizing if any selected nucleus is larger.

Display convention:
- green = FITC / 53BP1
- blue  = DAPI
- red   = nucleus MASK boundary

Representative selection uses the already-generated QC table. Within each
source x condition cell, the chosen nucleus is closest to the cell medians in
nucleus area and FITC p99, avoiding obvious intensity/size outliers while
remaining deterministic.

The PNG contact sheets are local analysis artifacts under RADEDGE_DATA_ROOT;
they are not intended to be committed to the public repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "OpenCV is required. Run with the existing DNAi Python 3.11 environment: "
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


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim != 2:
        raise RuntimeError(f"Expected single-channel image, got {image.shape}: {path}")
    return image


def robust_scale(image: np.ndarray, mask: np.ndarray, low_q: float = 1.0, high_q: float = 99.5) -> np.ndarray:
    pixels = image[mask]
    if pixels.size == 0:
        pixels = image.reshape(-1)
    low = float(np.percentile(pixels, low_q))
    high = float(np.percentile(pixels, high_q))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(image))
        high = float(np.max(image))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (image.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def fixed_scale(image: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (image.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def boundary_from_mask(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    return cv2.subtract(binary, eroded) > 0


def to_canvas(rgb_bgr: np.ndarray) -> np.ndarray:
    height, width = rgb_bgr.shape[:2]
    if height > CANVAS or width > CANVAS:
        raise RuntimeError(
            f"Selected nucleus {height}x{width} exceeds frozen {CANVAS}x{CANVAS} native-scale canvas"
        )
    canvas = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
    y0 = (CANVAS - height) // 2
    x0 = (CANVAS - width) // 2
    canvas[y0 : y0 + height, x0 : x0 + width] = rgb_bgr
    return canvas


def composite(fitc8: np.ndarray, dapi8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*fitc8.shape, 3), dtype=np.uint8)
    # OpenCV BGR ordering: B=DAPI, G=FITC, R=mask boundary.
    out[:, :, 0] = dapi8
    out[:, :, 1] = fitc8
    boundary = boundary_from_mask(mask)
    out[boundary, 2] = 255
    return out


def condition_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["particle_type"], row["dose_Gy"], row["hr_post_exposure"])


def choose_representatives(
    manifest_rows: list[dict[str, str]], qc_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    manifest_by_id = {row["sample_id"]: row for row in manifest_rows}
    groups: dict[tuple[str, tuple[str, str, str]], list[dict[str, str]]] = defaultdict(list)
    for row in qc_rows:
        sample_id = row["sample_id"]
        manifest = manifest_by_id.get(sample_id)
        if manifest is None:
            continue
        source = manifest["source_name"]
        cond = condition_key(manifest)
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
            areas = np.asarray([float(row["nucleus_area_px"]) for row in subset], dtype=np.float64)
            p99s = np.asarray([float(row["fitc_inside_p99"]) for row in subset], dtype=np.float64)
            median_area = float(np.median(areas))
            median_p99 = float(np.median(p99s))
            mad_area = float(np.median(np.abs(areas - median_area))) or 1.0
            mad_p99 = float(np.median(np.abs(p99s - median_p99))) or 1.0

            def score(row: dict[str, str]) -> tuple[float, str]:
                area_z = abs(float(row["nucleus_area_px"]) - median_area) / mad_area
                p99_z = abs(float(row["fitc_inside_p99"]) - median_p99) / mad_p99
                return (area_z + p99_z, row["sample_id"])

            chosen_qc = min(subset, key=score)
            chosen = dict(manifest_by_id[chosen_qc["sample_id"]])
            chosen["representative_score"] = f"{score(chosen_qc)[0]:.8f}"
            selected.append(chosen)

    if missing:
        raise RuntimeError(f"Missing representative cells: {missing}")
    return selected


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
                raise RuntimeError(f"Triplet mismatch for {row['sample_id']}")
            nucleus = mask > 0
            if mode == "adaptive":
                fitc8 = robust_scale(fitc, nucleus)
                dapi8 = robust_scale(dapi, nucleus)
            elif mode == "fixed":
                if fixed_limits is None:
                    raise RuntimeError("fixed mode requires fixed_limits")
                fitc8 = fixed_scale(fitc, *fixed_limits["fitc"])
                dapi8 = fixed_scale(dapi, *fixed_limits["dapi"])
            else:
                raise ValueError(mode)

            panel = to_canvas(composite(fitc8, dapi8, mask))
            x = row_label_w + c * (tile + gap) + gap
            sheet[y : y + tile, x : x + tile] = panel
            label_canvas(sheet, row["sample_id"], (x + 4, y + tile - 8), scale=0.34)

    return sheet


def derive_fixed_limits(selected: list[dict[str, str]], image_dir: Path) -> dict[str, tuple[float, float]]:
    sampled: dict[str, list[np.ndarray]] = {"fitc": [], "dapi": []}
    for row in selected:
        mask = load_gray(image_dir / "mask" / row["mask_filename"]) > 0
        for channel, filename_key in (("fitc", "fitc_filename"), ("dapi", "dapi_filename")):
            image = load_gray(image_dir / channel / row[filename_key])
            values = image[mask]
            if values.size > 5000:
                step = max(1, values.size // 5000)
                values = values[::step]
            sampled[channel].append(values.astype(np.float32, copy=False))

    limits: dict[str, tuple[float, float]] = {}
    for channel in ("fitc", "dapi"):
        values = np.concatenate(sampled[channel])
        low = float(np.percentile(values, 1.0))
        high = float(np.percentile(values, 99.5))
        limits[channel] = (low, high)
    return limits


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    metadata = data_root / "nasa_bps_microscopy" / "metadata" / "pilot_v1"
    pilot = data_root / "nasa_bps_microscopy" / "pilot_v1"
    parser.add_argument("--manifest", default=str(metadata / "pilot_v1_manifest.csv"))
    parser.add_argument("--qc", default=str(pilot / "qc" / "per_nucleus_stats.csv"))
    parser.add_argument("--image-dir", default=str(pilot / "images"))
    parser.add_argument("--output-dir", default=str(pilot / "visual_qc"))
    parser.add_argument("--expected-manifest-sha256", default=FROZEN_MANIFEST_SHA256)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    qc_path = Path(args.qc).resolve()
    image_dir = Path(args.image_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_sha = sha256_file(manifest_path)
    if args.expected_manifest_sha256 and manifest_sha != args.expected_manifest_sha256:
        raise RuntimeError(
            f"Frozen pilot-v1 manifest SHA256 mismatch: expected {args.expected_manifest_sha256}, got {manifest_sha}"
        )

    manifest_rows = read_csv(manifest_path)
    qc_rows = read_csv(qc_path)
    selected = choose_representatives(manifest_rows, qc_rows)
    fixed_limits = derive_fixed_limits(selected, image_dir)

    adaptive = render_sheet(selected, image_dir, "adaptive", None)
    fixed = render_sheet(selected, image_dir, "fixed", fixed_limits)

    adaptive_path = output_dir / "pilot_v1_contact_sheet_adaptive.png"
    fixed_path = output_dir / "pilot_v1_contact_sheet_fixed.png"
    if not cv2.imwrite(str(adaptive_path), adaptive):
        raise RuntimeError(f"Could not write {adaptive_path}")
    if not cv2.imwrite(str(fixed_path), fixed):
        raise RuntimeError(f"Could not write {fixed_path}")

    representative_path = output_dir / "representative_nuclei.csv"
    with representative_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0].keys()))
        writer.writeheader()
        writer.writerows(selected)

    # Compact geometry statement over the entire already-generated QC table.
    heights = np.asarray([int(row["height"]) for row in qc_rows], dtype=np.int32)
    widths = np.asarray([int(row["width"]) for row in qc_rows], dtype=np.int32)
    max_dims = np.maximum(heights, widths)
    mask_previews = sorted({row.get("mask_unique_preview", "") for row in qc_rows})

    summary = {
        "pilot_version": "NASA_BPS_53BP1_PILOT_V1",
        "manifest_sha256": manifest_sha,
        "n_representatives": len(selected),
        "geometry": {
            "height_min": int(np.min(heights)),
            "height_median": float(np.median(heights)),
            "height_max": int(np.max(heights)),
            "width_min": int(np.min(widths)),
            "width_median": float(np.median(widths)),
            "width_max": int(np.max(widths)),
            "max_dimension_max": int(np.max(max_dims)),
            "all_fit_256_native_scale": bool(np.all(max_dims <= CANVAS)),
        },
        "fixed_display_limits": {
            channel: {"p01": limits[0], "p995": limits[1]}
            for channel, limits in fixed_limits.items()
        },
        "mask_unique_previews": mask_previews,
        "adaptive_contact_sheet": str(adaptive_path),
        "fixed_contact_sheet": str(fixed_path),
        "representative_manifest": str(representative_path),
        "display_note": "Contact sheets are visual QC only; no focus-count threshold is inferred from display scaling.",
    }
    summary_path = output_dir / "visual_qc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Radiation Edge AI - NASA BPS pilot-v1 visual QC")
    print(f"Manifest SHA256: {manifest_sha}")
    print(f"Representatives: {len(selected)} = {len(SOURCES)} sources x {len(CONDITIONS)} conditions")
    print("")
    print("[Geometry]")
    print(
        f"height min/median/max: {np.min(heights)}/{np.median(heights):.1f}/{np.max(heights)}"
    )
    print(
        f"width  min/median/max: {np.min(widths)}/{np.median(widths):.1f}/{np.max(widths)}"
    )
    print(f"maximum observed dimension: {np.max(max_dims)}")
    print(f"all 2160 crops fit 256x256 at native scale: {'YES' if np.all(max_dims <= CANVAS) else 'NO'}")
    print("")
    print("[Fixed display scale from representative masked pixels]")
    print(f"FITC p01/p99.5: {fixed_limits['fitc'][0]:.3f}/{fixed_limits['fitc'][1]:.3f}")
    print(f"DAPI p01/p99.5: {fixed_limits['dapi'][0]:.3f}/{fixed_limits['dapi'][1]:.3f}")
    print("")
    print("NASA BPS PILOT V1 VISUAL QC COMPLETE: YES")
    print(f"Adaptive sheet: {adaptive_path}")
    print(f"Fixed sheet: {fixed_path}")
    print(f"Representative nuclei: {representative_path}")
    print(f"Summary: {summary_path}")
    print("")
    print(
        "NEXT GATE: inspect/upload the adaptive and fixed contact sheets, then freeze the 53BP1 "
        "reference endpoint and compact 256x256 model input policy."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
