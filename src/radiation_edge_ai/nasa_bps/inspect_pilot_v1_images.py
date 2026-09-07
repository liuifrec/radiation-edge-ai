"""Inspect the downloaded NASA BPS 53BP1 pilot-v1 TIFF triplets.

This script performs a pre-modeling quality-control pass over the frozen pilot:
- confirms FITC/DAPI/MASK files exist and are readable;
- records shape, dtype and intensity distributions per file;
- verifies that each nucleus triplet is shape-aligned;
- summarizes nucleus-mask area and descriptive FITC/DAPI intensity inside masks;
- aggregates descriptive statistics by radiation condition.

The intensity summaries are descriptive QC only. They are not treated as a
validated 53BP1 foci-counting endpoint and are not used to choose thresholds.

OpenCV is used with IMREAD_UNCHANGED so 16-bit TIFF values remain unscaled.
Run this with an environment that already has numpy + opencv-python available
(e.g. the existing DNAi Python 3.11 environment in this project).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "OpenCV is required for TIFF QC. Run this script with the existing DNAi Python 3.11 "
        "environment (D:\\radiation-edge-ai-data\\envs\\dnai311\\Scripts\\python.exe)."
    ) from exc

FROZEN_MANIFEST_SHA256 = "41ff2e4d64577f17c976dea5bbc17221fe63be9e2e7e2291309b520027fbe725"


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
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[:, :, 0]
        else:
            raise RuntimeError(f"Expected a single-channel TIFF, got shape={image.shape}: {path}")
    if image.ndim != 2:
        raise RuntimeError(f"Expected 2D TIFF, got shape={image.shape}: {path}")
    return image


def finite_stats(values: np.ndarray) -> dict[str, float | None]:
    flat = np.asarray(values).reshape(-1)
    if flat.size == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "p01": None,
            "p50": None,
            "p99": None,
            "p999": None,
            "max": None,
        }
    flat64 = flat.astype(np.float64, copy=False)
    return {
        "mean": float(np.mean(flat64)),
        "std": float(np.std(flat64)),
        "min": float(np.min(flat64)),
        "p01": float(np.percentile(flat64, 1.0)),
        "p50": float(np.percentile(flat64, 50.0)),
        "p99": float(np.percentile(flat64, 99.0)),
        "p999": float(np.percentile(flat64, 99.9)),
        "max": float(np.max(flat64)),
    }


def image_stats(sample_id: str, channel: str, filename: str, path: Path, image: np.ndarray) -> dict[str, object]:
    stats = finite_stats(image)
    return {
        "sample_id": sample_id,
        "channel": channel,
        "filename": filename,
        "local_path": str(path),
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "dtype": str(image.dtype),
        "n_pixels": int(image.size),
        "nonzero_fraction": float(np.mean(image != 0)),
        **stats,
    }


def safe_mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_median(values: list[float]) -> float | None:
    return None if not values else float(np.median(np.asarray(values, dtype=np.float64)))


def condition_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row["particle_type"]),
        str(row["dose_Gy"]),
        str(row["hr_post_exposure"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    default_data_root = Path(
        os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data")
    )
    default_metadata = default_data_root / "nasa_bps_microscopy" / "metadata" / "pilot_v1"
    default_pilot = default_data_root / "nasa_bps_microscopy" / "pilot_v1"
    parser.add_argument("--manifest", default=str(default_metadata / "pilot_v1_manifest.csv"))
    parser.add_argument("--image-dir", default=str(default_pilot / "images"))
    parser.add_argument("--output-dir", default=str(default_pilot / "qc"))
    parser.add_argument("--limit-nuclei", type=int, default=None)
    parser.add_argument("--expected-manifest-sha256", default=FROZEN_MANIFEST_SHA256)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    image_dir = Path(args.image_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest_sha = sha256_file(manifest_path)
    if args.expected_manifest_sha256 and manifest_sha != args.expected_manifest_sha256:
        raise RuntimeError(
            "Frozen NASA BPS pilot-v1 manifest SHA256 mismatch: "
            f"expected {args.expected_manifest_sha256}, got {manifest_sha}"
        )

    rows = read_csv(manifest_path)
    if args.limit_nuclei is not None:
        if args.limit_nuclei <= 0:
            raise ValueError("--limit-nuclei must be positive")
        rows = rows[: args.limit_nuclei]

    print("Radiation Edge AI - inspect NASA BPS 53BP1 pilot v1 images")
    print(f"Manifest SHA256: {manifest_sha}")
    print(f"Nuclei to inspect: {len(rows)}")
    print(f"Images: {image_dir}")
    print("")

    per_file_rows: list[dict[str, object]] = []
    nucleus_rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    shape_counter: dict[str, Counter[str]] = defaultdict(Counter)
    dtype_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        paths = {
            "fitc": image_dir / "fitc" / row["fitc_filename"],
            "dapi": image_dir / "dapi" / row["dapi_filename"],
            "mask": image_dir / "mask" / row["mask_filename"],
        }
        try:
            for channel, path in paths.items():
                if not path.is_file():
                    raise FileNotFoundError(path)
            fitc = load_gray(paths["fitc"])
            dapi = load_gray(paths["dapi"])
            mask = load_gray(paths["mask"])
            if fitc.shape != dapi.shape or fitc.shape != mask.shape:
                raise RuntimeError(
                    f"Triplet shape mismatch: FITC={fitc.shape}, DAPI={dapi.shape}, MASK={mask.shape}"
                )

            images = {"fitc": fitc, "dapi": dapi, "mask": mask}
            filenames = {
                "fitc": row["fitc_filename"],
                "dapi": row["dapi_filename"],
                "mask": row["mask_filename"],
            }
            for channel, image in images.items():
                per_file_rows.append(
                    image_stats(sample_id, channel, filenames[channel], paths[channel], image)
                )
                shape_counter[channel][f"{image.shape[0]}x{image.shape[1]}"] += 1
                dtype_counter[channel][str(image.dtype)] += 1

            nucleus_mask = mask > 0
            nucleus_pixels = int(nucleus_mask.sum())
            total_pixels = int(mask.size)
            fitc_inside = fitc[nucleus_mask]
            fitc_outside = fitc[~nucleus_mask]
            dapi_inside = dapi[nucleus_mask]

            fitc_inside_stats = finite_stats(fitc_inside)
            fitc_outside_stats = finite_stats(fitc_outside)
            dapi_inside_stats = finite_stats(dapi_inside)
            mask_values = np.unique(mask)

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
                    "height": int(fitc.shape[0]),
                    "width": int(fitc.shape[1]),
                    "fitc_dtype": str(fitc.dtype),
                    "dapi_dtype": str(dapi.dtype),
                    "mask_dtype": str(mask.dtype),
                    "mask_unique_count": int(mask_values.size),
                    "mask_unique_preview": ";".join(str(v) for v in mask_values[:12]),
                    "nucleus_area_px": nucleus_pixels,
                    "nucleus_fraction": nucleus_pixels / total_pixels if total_pixels else 0.0,
                    "fitc_inside_mean": fitc_inside_stats["mean"],
                    "fitc_inside_p50": fitc_inside_stats["p50"],
                    "fitc_inside_p99": fitc_inside_stats["p99"],
                    "fitc_inside_p999": fitc_inside_stats["p999"],
                    "fitc_inside_max": fitc_inside_stats["max"],
                    "fitc_inside_integrated": float(np.sum(fitc_inside.astype(np.float64))),
                    "fitc_outside_mean": fitc_outside_stats["mean"],
                    "fitc_outside_p99": fitc_outside_stats["p99"],
                    "dapi_inside_mean": dapi_inside_stats["mean"],
                    "dapi_inside_p99": dapi_inside_stats["p99"],
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "sample_id": sample_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        if index == 1 or index % 250 == 0 or index == len(rows):
            print(f"[{index:>5}/{len(rows)}] errors={len(errors)}")

    condition_groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in nucleus_rows:
        condition_groups[condition_key(row)].append(row)

    condition_rows: list[dict[str, object]] = []
    for condition, subset in sorted(
        condition_groups.items(), key=lambda item: (item[0][0], float(item[0][1]), float(item[0][2]))
    ):
        condition_rows.append(
            {
                "particle_type": condition[0],
                "dose_Gy": condition[1],
                "hr_post_exposure": condition[2],
                "n_nuclei": len(subset),
                "n_sources": len({str(r["source_name"]) for r in subset}),
                "n_strains": len({str(r["strain"]) for r in subset}),
                "median_nucleus_area_px": safe_median([float(r["nucleus_area_px"]) for r in subset]),
                "mean_fitc_inside_mean": safe_mean([float(r["fitc_inside_mean"]) for r in subset]),
                "median_fitc_inside_mean": safe_median([float(r["fitc_inside_mean"]) for r in subset]),
                "mean_fitc_inside_p99": safe_mean([float(r["fitc_inside_p99"]) for r in subset]),
                "median_fitc_inside_p99": safe_median([float(r["fitc_inside_p99"]) for r in subset]),
                "mean_dapi_inside_mean": safe_mean([float(r["dapi_inside_mean"]) for r in subset]),
            }
        )

    per_file_path = output_dir / "per_file_stats.csv"
    per_nucleus_path = output_dir / "per_nucleus_stats.csv"
    condition_path = output_dir / "condition_descriptive_stats.csv"
    errors_path = output_dir / "errors.csv"
    write_csv(per_file_path, per_file_rows)
    write_csv(per_nucleus_path, nucleus_rows)
    write_csv(condition_path, condition_rows)
    if errors:
        write_csv(errors_path, errors)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_version": "NASA_BPS_53BP1_PILOT_V1",
        "manifest_sha256": manifest_sha,
        "n_nuclei_requested": len(rows),
        "n_nuclei_successful": len(nucleus_rows),
        "n_nuclei_failed": len(errors),
        "shape_counts": {channel: dict(counter) for channel, counter in shape_counter.items()},
        "dtype_counts": {channel: dict(counter) for channel, counter in dtype_counter.items()},
        "per_file_stats": str(per_file_path),
        "per_nucleus_stats": str(per_nucleus_path),
        "condition_descriptive_stats": str(condition_path),
        "notes": [
            "TIFFs were read unchanged; no intensity normalization was applied.",
            "Condition-level intensity summaries are descriptive QC, not a validated foci endpoint.",
            "No threshold-based focus counts are generated by this script.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("[Image geometry and dtype]")
    for channel in ("fitc", "dapi", "mask"):
        print(f"{channel.upper()} shapes: {dict(shape_counter[channel])}")
        print(f"{channel.upper()} dtypes: {dict(dtype_counter[channel])}")
    print("")
    print("[Condition descriptive QC]")
    for row in condition_rows:
        print(
            f"{row['particle_type']:<5} {row['dose_Gy']:>4} Gy {row['hr_post_exposure']:>3} h "
            f"n={row['n_nuclei']:<3} sources={row['n_sources']} "
            f"FITC-in mean={row['mean_fitc_inside_mean']:.3f} p99={row['mean_fitc_inside_p99']:.3f}"
        )
    print("")
    print(f"successful nuclei: {len(nucleus_rows)}/{len(rows)}")
    print(f"errors: {len(errors)}")
    print(f"summary: {summary_path}")
    print(f"per nucleus: {per_nucleus_path}")
    print(f"condition QC: {condition_path}")
    print("")

    if errors:
        print("NASA BPS PILOT V1 IMAGE QC COMPLETE: NO")
        for row in errors[:10]:
            print(f"FAILED {row['sample_id']}: {row['error']}")
        return 1

    print("NASA BPS PILOT V1 IMAGE QC COMPLETE: YES")
    print(
        "NEXT GATE: use the observed geometry/bit depth and nucleus masks to choose the "
        "reference 53BP1 endpoint/model; do not set foci thresholds from raw intensity QC alone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
