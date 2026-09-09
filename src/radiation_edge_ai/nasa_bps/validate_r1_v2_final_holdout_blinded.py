"""Phenotype-blind hard QC for the frozen NASA BPS R1 v2 FINAL HOLDOUT.

This script is allowed to inspect only the already materialized holdout image
triplets plus the frozen blinded manifest and repaired-checkpoint identity. It
must run before any holdout phenotype/reference table is joined.

The per-nucleus hard checks are exactly the development QC checks implemented in
validate_r1_v2_development.py:
- FITC/DAPI/MASK decode as aligned 2D images;
- FITC/DAPI uint16, MASK integer-valued;
- native geometry <= 256x256 (no resize policy);
- filename-specific target label present in MASK and at crop center;
- target label does not touch the native crop border;
- robust p1/p99.5 normalization ranges are finite.

Multi-label MASKs are reported but are not a failure because the target remains
the filename-specific object label. MASK is QC-only and is never model input.

If every frozen nucleus passes, this script copies the blinded manifest byte-for-
byte to a QC1 path and reports its SHA256 as the QC-passed holdout identity. If
geometry-only failures occur, no QC1 manifest is frozen; a deterministic,
phenotype-blind geometry-repair gate must be run next.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import cv2

FROZEN_REPAIRED_CHECKPOINT_SHA256 = (
    "2b07c67d6e50e7a36e26e0052feebc826b518e6efae863ba312ba9761a4eb3d5"
)
FROZEN_BLINDED_HOLDOUT_MANIFEST_SHA256 = (
    "40f015f7f8a8acb71289c1fe91218253e0b518c7e98f920511a5792dd5b6862d"
)
EXPECTED_NUCLEI = 2058
EXPECTED_BAGS = 22
EXPECTED_CONTRASTS = 11
EXPECTED_SOURCE_NUCLEI = {
    "BALBCF2": 858,
    "C57BLF2": 600,
    "C57BLF3": 600,
}
EXPECTED_SOURCE_CONTRASTS = {
    "BALBCF2": 5,
    "C57BLF2": 3,
    "C57BLF3": 3,
}
EXPECTED_HOLDOUT_STATUS = "FINAL_BLINDED_DO_NOT_JOIN_PHENOTYPES"
FORBIDDEN_COLUMN_TOKENS = (
    "nfoci",
    "phenotype",
    "reference",
    "target_",
    "ground_truth",
    "prediction",
    "predicted",
)
EXPOSED_DOSE = {"Fe": "0.82", "X-ray": "1.0"}


def load_dev_qc_module(script_dir: Path):
    path = script_dir / "validate_r1_v2_development.py"
    spec = importlib.util.spec_from_file_location("nasa_bps_dev_qc_for_holdout", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load development QC implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def norm_number(value: str) -> str:
    number = float(str(value).strip())
    if number.is_integer():
        return f"{number:.1f}"
    return format(number, "g")


def norm_hour(value: str) -> str:
    number = float(str(value).strip())
    return str(int(number)) if number.is_integer() else format(number, "g")


def validate_blinded_structure(rows: list[dict[str, str]]) -> tuple[Counter, Counter, dict[str, int]]:
    required = {
        "sample_id",
        "nucleus_key",
        "source_name",
        "sample_name",
        "strain",
        "sex",
        "particle_type",
        "dose_Gy",
        "hr_post_exposure",
        "fitc_filename",
        "dapi_filename",
        "mask_filename",
        "holdout_status",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise RuntimeError(f"Blinded holdout manifest missing columns: {missing}")

    forbidden = sorted(
        column
        for column in rows[0]
        if any(token in column.lower() for token in FORBIDDEN_COLUMN_TOKENS)
    )
    if forbidden:
        raise RuntimeError(
            "Blinded holdout manifest contains forbidden outcome/reference columns: "
            + ", ".join(forbidden)
        )

    if len(rows) != EXPECTED_NUCLEI:
        raise RuntimeError(f"Holdout nuclei={len(rows)}; expected {EXPECTED_NUCLEI}")

    source_counts = Counter(row["source_name"] for row in rows)
    if dict(source_counts) != EXPECTED_SOURCE_NUCLEI:
        raise RuntimeError(
            f"Unexpected holdout source counts: {dict(source_counts)}; expected {EXPECTED_SOURCE_NUCLEI}"
        )
    if {row["sex"] for row in rows} != {"Female"}:
        raise RuntimeError(f"Unexpected holdout sex values: {sorted({row['sex'] for row in rows})}")

    statuses = {row["holdout_status"] for row in rows}
    if statuses != {EXPECTED_HOLDOUT_STATUS}:
        raise RuntimeError(f"Unexpected holdout_status values: {sorted(statuses)}")

    bag_counts = Counter(row["sample_name"] for row in rows)
    if len(bag_counts) != EXPECTED_BAGS:
        raise RuntimeError(f"Holdout bags={len(bag_counts)}; expected {EXPECTED_BAGS}")
    if min(bag_counts.values()) < 1 or max(bag_counts.values()) > 100:
        raise RuntimeError(
            f"Unexpected holdout bag size range: {min(bag_counts.values())}/{max(bag_counts.values())}"
        )

    # One sample/plate-well bag per Source x branch x dose x time condition.
    sample_condition: dict[tuple[str, str, str, str], str] = {}
    for row in rows:
        key = (
            row["source_name"],
            row["particle_type"],
            norm_number(row["dose_Gy"]),
            norm_hour(row["hr_post_exposure"]),
        )
        prior = sample_condition.setdefault(key, row["sample_name"])
        if prior != row["sample_name"]:
            raise RuntimeError(f"Condition {key} maps to multiple Sample Names")

    contrast_counts: Counter[str] = Counter()
    for source in EXPECTED_SOURCE_NUCLEI:
        for branch, dose in EXPOSED_DOSE.items():
            for hour in ("4", "24", "48"):
                sham = (source, branch, "0.0", hour)
                exposed = (source, branch, norm_number(dose), hour)
                if sham in sample_condition and exposed in sample_condition:
                    contrast_counts[source] += 1

    if dict(contrast_counts) != EXPECTED_SOURCE_CONTRASTS:
        raise RuntimeError(
            f"Unexpected complete matched-contrast structure: {dict(contrast_counts)}; expected {EXPECTED_SOURCE_CONTRASTS}"
        )
    if sum(contrast_counts.values()) != EXPECTED_CONTRASTS:
        raise RuntimeError("Unexpected total number of complete holdout contrasts")

    return source_counts, bag_counts, dict(contrast_counts)


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
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"
    script_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--manifest",
        default=str(
            metadata_root
            / "r1_v2_freeze"
            / "r1_v2_final_holdout_manifest_blinded.csv"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            model_root
            / "nasa_bps_53bp1"
            / "r1_v2_final_candidate"
            / "r1_countnet_v2_final_bn_recalibrated.pt"
        ),
    )
    parser.add_argument(
        "--image-root",
        default=str(
            data_root
            / "nasa_bps_microscopy"
            / "r1_v2_final_holdout_blinded"
            / "images"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            data_root
            / "nasa_bps_microscopy"
            / "r1_v2_final_holdout_blinded"
            / "qc"
        ),
    )
    parser.add_argument(
        "--qc-manifest",
        default=str(
            metadata_root
            / "r1_v2_freeze"
            / "r1_v2_final_holdout_manifest_blinded_qc1.csv"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    image_root = Path(args.image_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    qc_manifest_path = Path(args.qc_manifest).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (manifest_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != FROZEN_REPAIRED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Repaired final checkpoint SHA256 mismatch: "
            f"expected {FROZEN_REPAIRED_CHECKPOINT_SHA256}, got {checkpoint_sha}"
        )
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != FROZEN_BLINDED_HOLDOUT_MANIFEST_SHA256:
        raise RuntimeError(
            "Blinded holdout manifest SHA256 mismatch: "
            f"expected {FROZEN_BLINDED_HOLDOUT_MANIFEST_SHA256}, got {manifest_sha}"
        )

    rows = read_csv(manifest_path)
    source_counts, bag_counts, contrast_counts = validate_blinded_structure(rows)
    dev_qc = load_dev_qc_module(script_dir)

    cv2.setNumThreads(0)
    print("Radiation Edge AI - validate NASA BPS R1 v2 FINAL HOLDOUT images")
    print("PHENOTYPE-BLIND HARD QC ONLY")
    print(f"repaired checkpoint SHA256 pinned: {checkpoint_sha}")
    print(f"blinded holdout manifest SHA256: {manifest_sha}")
    print(f"holdout nuclei: {len(rows)}")
    print(f"holdout bags: {len(bag_counts)}")
    print(f"source nuclei: {dict(source_counts)}")
    print(f"complete matched contrasts: {sum(contrast_counts.values())} {contrast_counts}")
    print(f"image root: {image_root}")
    print(f"workers: {max(1, args.workers)}")
    print("NASA phenotype/reference tables read: NO")
    print("")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(dev_qc.inspect_one, row, image_root) for row in rows]
        for i, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if i == 1 or i % 250 == 0 or i == len(futures):
                failures = sum(int(row.get("ok", 0)) == 0 for row in results)
                print(f"[{i:>4}/{len(futures)}] QC failures={failures}")

    results.sort(key=lambda row: str(row["sample_id"]))
    qc_table_path = output_dir / "r1_v2_final_holdout_triplet_qc.csv"
    write_csv(qc_table_path, results)

    failures = [row for row in results if int(row.get("ok", 0)) != 1]
    geometry_only_failures = [
        row
        for row in failures
        if "native geometry" in str(row.get("error", ""))
        and ";" not in str(row.get("error", ""))
    ]
    non_geometry_failures = [row for row in failures if row not in geometry_only_failures]

    heights = [int(row["height"]) for row in results if "height" in row]
    widths = [int(row["width"]) for row in results if "width" in row]
    target_pixels = [int(row["target_pixels"]) for row in results if "target_pixels" in row]
    fitc_dtypes = Counter(str(row.get("fitc_dtype", "ERROR")) for row in results)
    dapi_dtypes = Counter(str(row.get("dapi_dtype", "ERROR")) for row in results)
    mask_dtypes = Counter(str(row.get("mask_dtype", "ERROR")) for row in results)
    target_present = sum(int(row.get("target_present", 0)) == 1 for row in results)
    center_matches = sum(int(row.get("center_matches_target", 0)) == 1 for row in results)
    border_touch = sum(int(row.get("target_touches_border", 0)) == 1 for row in results)
    multilabel = sum(int(row.get("multilabel_mask", 0)) == 1 for row in results)
    zero_fitc_ranges = sum(float(row.get("fitc_robust_range", -1.0)) == 0.0 for row in results)
    zero_dapi_ranges = sum(float(row.get("dapi_robust_range", -1.0)) == 0.0 for row in results)

    qc_manifest_sha = ""
    if not failures:
        qc_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, qc_manifest_path)
        qc_manifest_sha = sha256_file(qc_manifest_path)
        if qc_manifest_sha != manifest_sha:
            raise RuntimeError(
                "Byte-for-byte QC1 manifest copy changed SHA256 unexpectedly: "
                f"source={manifest_sha} qc1={qc_manifest_sha}"
            )
    elif qc_manifest_path.exists():
        # Never leave a stale apparently-passing QC manifest after a failed run.
        qc_manifest_path.unlink()

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "R1_v2_final_holdout_blinded_hard_qc",
        "phenotype_blind": True,
        "phenotype_tables_read": False,
        "model_predictions_computed": False,
        "repaired_checkpoint_sha256": checkpoint_sha,
        "blinded_holdout_manifest_sha256": manifest_sha,
        "holdout_nuclei": len(rows),
        "holdout_bags": len(bag_counts),
        "source_nuclei": dict(source_counts),
        "complete_matched_contrasts": contrast_counts,
        "nuclei_passing_all_hard_checks": len(rows) - len(failures),
        "hard_failures": len(failures),
        "geometry_only_failures": len(geometry_only_failures),
        "non_geometry_failures": len(non_geometry_failures),
        "height": stat_triplet(heights),
        "width": stat_triplet(widths),
        "all_fit_256_native_scale": len(failures) == 0 or not any(
            "native geometry" in str(row.get("error", "")) for row in failures
        ),
        "fitc_dtype_counts": dict(fitc_dtypes),
        "dapi_dtype_counts": dict(dapi_dtypes),
        "mask_dtype_counts": dict(mask_dtypes),
        "target_label_present": target_present,
        "center_matches_target": center_matches,
        "target_touches_border": border_touch,
        "multilabel_masks": multilabel,
        "target_pixels": stat_triplet(target_pixels),
        "zero_fitc_robust_ranges": zero_fitc_ranges,
        "zero_dapi_robust_ranges": zero_dapi_ranges,
        "qc_table": str(qc_table_path),
        "qc_passed_manifest": str(qc_manifest_path) if not failures else "",
        "qc_passed_manifest_sha256": qc_manifest_sha,
        "hard_qc_pass": len(failures) == 0,
    }
    summary_path = output_dir / "r1_v2_final_holdout_qc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    h = summary["height"]
    w = summary["width"]
    tp = summary["target_pixels"]
    print("")
    print("[Phenotype-blind final-holdout QC summary]")
    print(f"nuclei passing all hard checks: {len(rows) - len(failures)}/{len(rows)}")
    print(f"native height min/median/max: {h['min']}/{h['median']}/{h['max']}")
    print(f"native width min/median/max: {w['min']}/{w['median']}/{w['max']}")
    print(f"all fit native 256x256 canvas: {'YES' if summary['all_fit_256_native_scale'] else 'NO'}")
    print(f"FITC dtype counts: {dict(fitc_dtypes)}")
    print(f"DAPI dtype counts: {dict(dapi_dtypes)}")
    print(f"MASK dtype counts: {dict(mask_dtypes)}")
    print(f"target label present: {target_present}/{len(rows)}")
    print(f"crop center matches target label: {center_matches}/{len(rows)}")
    print(f"target touches crop border: {border_touch}/{len(rows)}")
    print(f"multi-label MASKs: {multilabel}/{len(rows)}")
    print(f"target pixels min/median/max: {tp['min']}/{tp['median']}/{tp['max']}")
    print(f"zero FITC robust ranges: {zero_fitc_ranges}/{len(rows)}")
    print(f"zero DAPI robust ranges: {zero_dapi_ranges}/{len(rows)}")
    print(f"QC table: {qc_table_path}")
    print(f"Summary: {summary_path}")
    print("NASA phenotype/reference tables read: NO")
    print("FINAL HOLDOUT PHENOTYPE STATUS: BLINDED")

    if failures:
        print("")
        print("NASA BPS R1 V2 FINAL HOLDOUT HARD QC COMPLETE: NO")
        print(f"geometry-only failures: {len(geometry_only_failures)}")
        print(f"non-geometry failures: {len(non_geometry_failures)}")
        for row in failures[:30]:
            print(f"FAILED {row['sample_id']} {row['nucleus_key']}: {row['error']}")
        if len(failures) > 30:
            print(f"... {len(failures) - 30} additional failures")
        if non_geometry_failures:
            print("NEXT GATE: STOP confirmatory evaluation and resolve the non-geometry protocol deviation while remaining phenotype-blind.")
        else:
            print("NEXT GATE: phenotype-blind deterministic geometry replacement only; do not join phenotype outcomes.")
        return 2

    print(f"QC-passed holdout manifest: {qc_manifest_path}")
    print(f"QC-passed holdout manifest SHA256: {qc_manifest_sha}")
    print("NASA BPS R1 V2 FINAL HOLDOUT HARD QC COMPLETE: YES")
    print("NEXT GATE: freeze the one-time evaluator with the repaired checkpoint and QC-passed manifest hashes before phenotype unblinding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
