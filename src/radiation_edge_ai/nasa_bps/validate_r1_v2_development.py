"""Validate the frozen NASA BPS R1 v2 DEVELOPMENT image set before training.

This is a development-only QC gate. It reads the frozen 7,200-nucleus
R1 v2 development manifest and the corresponding local FITC/DAPI/MASK TIFFs.
It never reads the final-holdout manifest, final-holdout images, or phenotype
outcomes.

Checks performed for every development nucleus:
- frozen development-manifest SHA256 and expected cohort structure;
- FITC/DAPI/MASK files exist and decode as 2D images;
- FITC, DAPI and MASK native shapes agree;
- native geometry fits the frozen 256x256 no-resize canvas;
- FITC/DAPI are uint16 and MASK is integer-valued;
- the filename object index is present as a label in the MASK;
- the crop center belongs to that target label;
- the target label does not touch the native crop border;
- robust p1/p99.5 FITC/DAPI normalization has finite non-negative range.

The target-mask checks are QC only. MASK remains excluded from the deployable
R1 model input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np

FROZEN_DEV_MANIFEST_SHA256 = "029d156250f1b73a6cc2b556b88a8e3b6900366a6a2b36244b699d2d9c43ab01"
EXPECTED_NUCLEI = 7200
EXPECTED_BAGS = 72
EXPECTED_NUCLEI_PER_BAG = 100
EXPECTED_V1_MEMBERS = 2160
CANVAS = 256
P_LOW = 1.0
P_HIGH = 99.5

DEV_SOURCES = {
    "BALBCF1",
    "BALBCM1",
    "BALBCM2",
    "C57BLF1",
    "C57BLM1",
    "C57BLM3",
}
FINAL_HOLDOUT_SOURCES = {"BALBCF2", "C57BLF2", "C57BLF3"}

FILENAME_RE = re.compile(
    r"^(?P<plate>P\d+)_(?P<acquisition>\d+)-(?P<well>[A-Za-z]+\d+)_"
    r"(?P<field>\d+)_(?P<object>\d+)_(?P<channel>proj|DAPI|MASK)\.tif$",
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_object_index(filename: str) -> int:
    match = FILENAME_RE.match(Path(filename).name)
    if match is None:
        raise ValueError(f"Unrecognized NASA BPS filename: {filename}")
    return int(match.group("object"))


def percentile_range(image: np.ndarray) -> tuple[float, float, float]:
    x = image.astype(np.float32, copy=False)
    lo = float(np.percentile(x, P_LOW))
    hi = float(np.percentile(x, P_HIGH))
    return lo, hi, hi - lo


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode TIFF: {path}")
    return image


def inspect_one(row: dict[str, str], image_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_id": row["sample_id"],
        "nucleus_key": row["nucleus_key"],
        "source_name": row["source_name"],
        "sample_name": row["sample_name"],
        "particle_type": row["particle_type"],
        "dose_Gy": row["dose_Gy"],
        "hr_post_exposure": row["hr_post_exposure"],
        "v1_member": row["v1_member"],
        "fitc_filename": row["fitc_filename"],
        "dapi_filename": row["dapi_filename"],
        "mask_filename": row["mask_filename"],
        "ok": 0,
        "error": "",
    }
    try:
        fitc_path = image_root / "fitc" / row["fitc_filename"]
        dapi_path = image_root / "dapi" / row["dapi_filename"]
        mask_path = image_root / "mask" / row["mask_filename"]
        for path in (fitc_path, dapi_path, mask_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        fitc = load_image(fitc_path)
        dapi = load_image(dapi_path)
        mask = load_image(mask_path)
        if fitc.ndim != 2 or dapi.ndim != 2 or mask.ndim != 2:
            raise RuntimeError(
                f"Expected 2D TIFFs; FITC={fitc.shape} DAPI={dapi.shape} MASK={mask.shape}"
            )
        if fitc.shape != dapi.shape or fitc.shape != mask.shape:
            raise RuntimeError(
                f"Shape mismatch; FITC={fitc.shape} DAPI={dapi.shape} MASK={mask.shape}"
            )

        h, w = fitc.shape
        target_label = parse_object_index(row["fitc_filename"])
        mask_integer = np.issubdtype(mask.dtype, np.integer)
        target = mask == target_label
        target_present = bool(np.any(target))
        center_label = int(mask[h // 2, w // 2]) if mask_integer else -1
        center_matches = bool(mask_integer and center_label == target_label)
        border_touches = bool(
            target_present
            and (
                np.any(target[0, :])
                or np.any(target[-1, :])
                or np.any(target[:, 0])
                or np.any(target[:, -1])
            )
        )
        nonzero_labels = np.unique(mask[mask != 0]) if mask_integer else np.array([])
        fitc_lo, fitc_hi, fitc_range = percentile_range(fitc)
        dapi_lo, dapi_hi, dapi_range = percentile_range(dapi)

        result.update(
            {
                "height": h,
                "width": w,
                "fitc_dtype": str(fitc.dtype),
                "dapi_dtype": str(dapi.dtype),
                "mask_dtype": str(mask.dtype),
                "fits_256": int(h <= CANVAS and w <= CANVAS),
                "target_label": target_label,
                "target_present": int(target_present),
                "center_label": center_label,
                "center_matches_target": int(center_matches),
                "target_touches_border": int(border_touches),
                "target_pixels": int(np.count_nonzero(target)),
                "n_nonzero_mask_labels": int(len(nonzero_labels)),
                "multilabel_mask": int(len(nonzero_labels) > 1),
                "fitc_p1": fitc_lo,
                "fitc_p99_5": fitc_hi,
                "fitc_robust_range": fitc_range,
                "dapi_p1": dapi_lo,
                "dapi_p99_5": dapi_hi,
                "dapi_robust_range": dapi_range,
                "fitc_range_valid": int(math.isfinite(fitc_range) and fitc_range >= 0.0),
                "dapi_range_valid": int(math.isfinite(dapi_range) and dapi_range >= 0.0),
            }
        )

        failures = []
        if fitc.dtype != np.uint16:
            failures.append(f"FITC dtype={fitc.dtype}, expected uint16")
        if dapi.dtype != np.uint16:
            failures.append(f"DAPI dtype={dapi.dtype}, expected uint16")
        if not mask_integer:
            failures.append(f"MASK dtype={mask.dtype} is not integer")
        if h > CANVAS or w > CANVAS:
            failures.append(f"native geometry {h}x{w} exceeds {CANVAS}x{CANVAS}")
        if not target_present:
            failures.append(f"target label {target_label} absent from MASK")
        if not center_matches:
            failures.append(f"center label {center_label} != target label {target_label}")
        if border_touches:
            failures.append(f"target label {target_label} touches crop border")
        if not (math.isfinite(fitc_range) and fitc_range >= 0.0):
            failures.append("FITC robust percentile range invalid")
        if not (math.isfinite(dapi_range) and dapi_range >= 0.0):
            failures.append("DAPI robust percentile range invalid")

        if failures:
            result["error"] = "; ".join(failures)
            return result

        result["ok"] = 1
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def stat_triplet(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0.0, "max": 0}
    return {
        "min": int(min(values)),
        "median": float(median(values)),
        "max": int(max(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"
    parser.add_argument(
        "--manifest",
        default=str(metadata_root / "r1_v2_freeze" / "r1_v2_development_manifest_100.csv"),
    )
    parser.add_argument(
        "--image-root",
        default=str(data_root / "nasa_bps_microscopy" / "r1_v2_development" / "images"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(data_root / "nasa_bps_microscopy" / "r1_v2_development" / "qc"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    image_root = Path(args.image_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != FROZEN_DEV_MANIFEST_SHA256:
        raise RuntimeError(
            "Frozen R1 v2 development manifest SHA256 mismatch: "
            f"expected {FROZEN_DEV_MANIFEST_SHA256}, got {manifest_sha}"
        )

    rows = read_csv(manifest_path)
    required = {
        "sample_id",
        "nucleus_key",
        "source_name",
        "sample_name",
        "particle_type",
        "dose_Gy",
        "hr_post_exposure",
        "fitc_filename",
        "dapi_filename",
        "mask_filename",
        "v1_member",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise RuntimeError(f"Development manifest missing columns: {missing}")
    if len(rows) != EXPECTED_NUCLEI:
        raise RuntimeError(f"Development nuclei={len(rows)}; expected {EXPECTED_NUCLEI}")

    sources = {row["source_name"] for row in rows}
    if sources != DEV_SOURCES:
        raise RuntimeError(f"Unexpected development Source Names: {sorted(sources)}")
    if sources & FINAL_HOLDOUT_SOURCES:
        raise RuntimeError("Final holdout Source Name leaked into development manifest")

    bag_counts = Counter(row["sample_name"] for row in rows)
    if len(bag_counts) != EXPECTED_BAGS:
        raise RuntimeError(f"Development bags={len(bag_counts)}; expected {EXPECTED_BAGS}")
    bad_bags = {name: n for name, n in bag_counts.items() if n != EXPECTED_NUCLEI_PER_BAG}
    if bad_bags:
        raise RuntimeError(f"Development bags not size {EXPECTED_NUCLEI_PER_BAG}: {bad_bags}")
    v1_members = sum(str(row["v1_member"]).strip() == "1" for row in rows)
    if v1_members != EXPECTED_V1_MEMBERS:
        raise RuntimeError(f"v1_member rows={v1_members}; expected {EXPECTED_V1_MEMBERS}")

    cv2.setNumThreads(0)
    print("Radiation Edge AI - validate NASA BPS R1 v2 DEVELOPMENT images")
    print("Final holdout remains untouched: no holdout manifest/images/phenotypes are read.")
    print(f"Development manifest SHA256: {manifest_sha}")
    print(f"Development nuclei: {len(rows)}")
    print(f"Development bags: {len(bag_counts)} x {EXPECTED_NUCLEI_PER_BAG} nuclei")
    print(f"Pilot-v1 members preserved: {v1_members}")
    print(f"Image root: {image_root}")
    print(f"Workers: {max(1, args.workers)}")
    print("")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(inspect_one, row, image_root) for row in rows]
        for i, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if i == 1 or i % 500 == 0 or i == len(futures):
                failures = sum(int(r.get("ok", 0)) == 0 for r in results)
                print(f"[{i:>4}/{len(futures)}] QC failures={failures}")

    results.sort(key=lambda r: str(r["sample_id"]))
    qc_path = output_dir / "r1_v2_development_triplet_qc.csv"
    write_csv(qc_path, results)

    failures = [r for r in results if int(r.get("ok", 0)) != 1]
    heights = [int(r["height"]) for r in results if "height" in r]
    widths = [int(r["width"]) for r in results if "width" in r]
    target_pixels = [int(r["target_pixels"]) for r in results if "target_pixels" in r]
    fitc_dtypes = Counter(str(r.get("fitc_dtype", "ERROR")) for r in results)
    dapi_dtypes = Counter(str(r.get("dapi_dtype", "ERROR")) for r in results)
    mask_dtypes = Counter(str(r.get("mask_dtype", "ERROR")) for r in results)
    target_missing = sum(int(r.get("target_present", 0)) == 0 for r in results)
    center_mismatch = sum(int(r.get("center_matches_target", 0)) == 0 for r in results)
    border_touch = sum(int(r.get("target_touches_border", 0)) == 1 for r in results)
    multilabel = sum(int(r.get("multilabel_mask", 0)) == 1 for r in results)
    oversize = sum(int(r.get("fits_256", 0)) == 0 for r in results)
    zero_fitc_range = sum(float(r.get("fitc_robust_range", -1.0)) == 0.0 for r in results)
    zero_dapi_range = sum(float(r.get("dapi_robust_range", -1.0)) == 0.0 for r in results)

    all_pass = len(failures) == 0
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "NASA_BPS_R1_V2_DEVELOPMENT_ONLY",
        "final_holdout_accessed": False,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "image_root": str(image_root),
        "n_nuclei": len(rows),
        "n_bags": len(bag_counts),
        "nuclei_per_bag": EXPECTED_NUCLEI_PER_BAG,
        "v1_members": v1_members,
        "n_qc_failures": len(failures),
        "native_height": stat_triplet(heights),
        "native_width": stat_triplet(widths),
        "target_pixels": stat_triplet(target_pixels),
        "fitc_dtype_counts": dict(fitc_dtypes),
        "dapi_dtype_counts": dict(dapi_dtypes),
        "mask_dtype_counts": dict(mask_dtypes),
        "target_label_missing": target_missing,
        "center_target_mismatch": center_mismatch,
        "target_border_touch": border_touch,
        "multilabel_masks": multilabel,
        "native_geometry_exceeds_256": oversize,
        "zero_fitc_robust_range": zero_fitc_range,
        "zero_dapi_robust_range": zero_dapi_range,
        "development_qc_pass": all_pass,
        "qc_csv": str(qc_path),
        "qc_csv_sha256": sha256_file(qc_path),
        "first_failures": [
            {
                "sample_id": r.get("sample_id", ""),
                "nucleus_key": r.get("nucleus_key", ""),
                "error": r.get("error", ""),
            }
            for r in failures[:50]
        ],
    }
    summary_path = output_dir / "r1_v2_development_qc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("[Development QC summary]")
    print(f"nuclei passing all hard checks: {len(rows) - len(failures)}/{len(rows)}")
    print(
        "native height min/median/max: "
        f"{summary['native_height']['min']}/{summary['native_height']['median']}/{summary['native_height']['max']}"
    )
    print(
        "native width min/median/max: "
        f"{summary['native_width']['min']}/{summary['native_width']['median']}/{summary['native_width']['max']}"
    )
    print(f"all fit native 256x256 canvas: {'YES' if oversize == 0 else 'NO'}")
    print(f"FITC dtype counts: {dict(fitc_dtypes)}")
    print(f"DAPI dtype counts: {dict(dapi_dtypes)}")
    print(f"MASK dtype counts: {dict(mask_dtypes)}")
    print(f"target label present: {len(rows) - target_missing}/{len(rows)}")
    print(f"crop center matches target label: {len(rows) - center_mismatch}/{len(rows)}")
    print(f"target touches crop border: {border_touch}/{len(rows)}")
    print(f"multi-label MASKs: {multilabel}/{len(rows)}")
    print(
        "target pixels min/median/max: "
        f"{summary['target_pixels']['min']}/{summary['target_pixels']['median']}/{summary['target_pixels']['max']}"
    )
    print(f"zero FITC robust ranges: {zero_fitc_range}/{len(rows)}")
    print(f"zero DAPI robust ranges: {zero_dapi_range}/{len(rows)}")
    print(f"QC table: {qc_path}")
    print(f"Summary: {summary_path}")
    print("")

    if not all_pass:
        print("NASA BPS R1 V2 DEVELOPMENT QC COMPLETE: NO")
        for row in failures[:10]:
            print(f"FAILED {row['sample_id']} {row['nucleus_key']}: {row['error']}")
        print("NEXT GATE: resolve hard development-image QC failures before freezing R1 v2 training policy.")
        return 1

    print("NASA BPS R1 V2 DEVELOPMENT QC COMPLETE: YES")
    print("FINAL HOLDOUT STATUS: UNTOUCHED")
    print(
        "NEXT GATE: freeze the R1 v2 contrast-aware objective, development-CV decision rule, "
        "and fixed training schedule before any final-holdout access."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
