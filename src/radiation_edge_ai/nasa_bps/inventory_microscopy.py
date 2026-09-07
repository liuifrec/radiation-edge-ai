"""Fetch and inventory NASA BPS microscopy metadata without downloading images.

The NASA Biological and Physical Sciences (BPS) Microscopy Benchmark is hosted
as public AWS Open Data. This utility intentionally downloads only the two
small metadata CSV files used to describe the FITC/53BP1 and DAPI/MASK image
collections. It does not bulk-download microscopy images.

The first goal is to freeze metadata provenance and identify biologically
informative dose x particle x post-exposure-time contrasts before selecting a
small pilot subset for KL720 development.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

FITC_META_URL = (
    "https://nasa-bps-training-data.s3.us-west-2.amazonaws.com/"
    "Microscopy/train/meta.csv"
)
DAPI_MASK_META_URL = (
    "https://nasa-bps-training-data.s3.us-west-2.amazonaws.com/"
    "Microscopy/DAPI_MASK_images/meta_DAPI_MASK.csv"
)
REQUIRED_COLUMNS = ("filename", "dose_Gy", "particle_type", "hr_post_exposure")
IDENTIFIER_HINTS = (
    "strain",
    "mouse",
    "animal",
    "plate",
    "well",
    "sample",
    "subject",
    "group",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, path: Path, refresh: bool) -> None:
    if path.is_file() and not refresh:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "radiation-edge-ai/0.1 NASA-BPS-metadata-inventory"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"Empty response from {url}")
    path.write_bytes(payload)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fieldnames:
        raise RuntimeError(f"CSV has no header: {path}")
    return fieldnames, rows


def validate_required(fieldnames: list[str], label: str) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise RuntimeError(
            f"{label} metadata missing expected columns {missing}; fields={fieldnames}"
        )


def maybe_number(value: str) -> object:
    text = value.strip()
    if text == "":
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return number


def sorted_unique(rows: list[dict[str, str]], column: str) -> list[object]:
    values = {maybe_number(str(row.get(column, ""))) for row in rows}
    return sorted(values, key=lambda value: (isinstance(value, str), str(value)))


def extension_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        suffix = Path(str(row.get("filename", ""))).suffix.lower() or "<none>"
        counter[suffix] += 1
    return dict(sorted(counter.items()))


def candidate_identifier_columns(fieldnames: list[str]) -> list[str]:
    result = []
    for field in fieldnames:
        lowered = field.lower()
        if any(token in lowered for token in IDENTIFIER_HINTS):
            result.append(field)
    return result


def condition_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    counter: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        key = (
            str(row.get("dose_Gy", "")).strip(),
            str(row.get("particle_type", "")).strip(),
            str(row.get("hr_post_exposure", "")).strip(),
        )
        counter[key] += 1

    result = []
    for (dose, particle, hours), count in counter.items():
        result.append(
            {
                "dose_Gy": maybe_number(dose),
                "particle_type": particle,
                "hr_post_exposure": maybe_number(hours),
                "n_images": count,
            }
        )
    result.sort(
        key=lambda row: (
            str(row["particle_type"]),
            str(row["dose_Gy"]),
            str(row["hr_post_exposure"]),
        )
    )
    return result


def write_condition_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dose_Gy", "particle_type", "hr_post_exposure", "n_images"),
        )
        writer.writeheader()
        writer.writerows(rows)


def dataset_summary(
    label: str,
    source_url: str,
    local_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> dict[str, object]:
    return {
        "label": label,
        "source_url": source_url,
        "local_path": str(local_path),
        "sha256": sha256_file(local_path),
        "n_rows": len(rows),
        "columns": fieldnames,
        "candidate_identifier_columns": candidate_identifier_columns(fieldnames),
        "dose_Gy_values": sorted_unique(rows, "dose_Gy"),
        "particle_type_values": sorted_unique(rows, "particle_type"),
        "hr_post_exposure_values": sorted_unique(rows, "hr_post_exposure"),
        "filename_extensions": extension_counts(rows),
        "n_conditions": len(condition_rows(rows)),
    }


def print_summary(summary: dict[str, object]) -> None:
    print(f"[{summary['label']}]")
    print(f"rows: {summary['n_rows']}")
    print(f"SHA256: {summary['sha256']}")
    print("columns: " + ", ".join(str(v) for v in summary["columns"]))
    print(f"dose_Gy: {summary['dose_Gy_values']}")
    print(f"particle_type: {summary['particle_type_values']}")
    print(f"hr_post_exposure: {summary['hr_post_exposure_values']}")
    print(f"candidate identifiers: {summary['candidate_identifier_columns']}")
    print(f"condition cells: {summary['n_conditions']}")
    print("")


def main() -> int:
    parser = argparse.ArgumentParser()
    default_data_root = Path(
        os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data")
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_data_root / "nasa_bps_microscopy" / "metadata"),
    )
    parser.add_argument("--fitc-meta-url", default=FITC_META_URL)
    parser.add_argument("--dapi-mask-meta-url", default=DAPI_MASK_META_URL)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download metadata even if local copies already exist",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fitc_path = output_dir / "meta.csv"
    dapi_path = output_dir / "meta_DAPI_MASK.csv"

    print("Radiation Edge AI - NASA BPS microscopy metadata inventory")
    print("Metadata only: no microscopy images will be downloaded.")
    print(f"Output: {output_dir}")
    print("")

    print("[Fetch]")
    fetch(args.fitc_meta_url, fitc_path, args.refresh)
    print(f"FITC/53BP1: {fitc_path}")
    fetch(args.dapi_mask_meta_url, dapi_path, args.refresh)
    print(f"DAPI/MASK: {dapi_path}")
    print("")

    fitc_fields, fitc_rows = read_csv(fitc_path)
    dapi_fields, dapi_rows = read_csv(dapi_path)
    validate_required(fitc_fields, "FITC/53BP1")
    validate_required(dapi_fields, "DAPI/MASK")

    fitc_summary = dataset_summary(
        "FITC/53BP1", args.fitc_meta_url, fitc_path, fitc_fields, fitc_rows
    )
    dapi_summary = dataset_summary(
        "DAPI/MASK", args.dapi_mask_meta_url, dapi_path, dapi_fields, dapi_rows
    )

    fitc_conditions = condition_rows(fitc_rows)
    dapi_conditions = condition_rows(dapi_rows)
    fitc_condition_path = output_dir / "fitc_53bp1_condition_counts.csv"
    dapi_condition_path = output_dir / "dapi_mask_condition_counts.csv"
    write_condition_csv(fitc_condition_path, fitc_conditions)
    write_condition_csv(dapi_condition_path, dapi_conditions)

    print_summary(fitc_summary)
    print_summary(dapi_summary)

    inventory = {
        "accessed_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_s3_prefix": "s3://nasa-bps-training-data/Microscopy/",
        "dataset_registry": "https://registry.opendata.aws/bps_microscopy/",
        "osdr_study": "OSD-366",
        "osdr_doi": "10.26030/v8w4-rg83",
        "fitc_53bp1": fitc_summary,
        "dapi_mask": dapi_summary,
        "fitc_condition_counts_csv": str(fitc_condition_path),
        "dapi_mask_condition_counts_csv": str(dapi_condition_path),
        "notes": [
            "This inventory intentionally downloads metadata only.",
            "Do not infer sham/irradiated status from particle_type because particle_type may be populated for 0 Gy rows.",
            "Resolve plate/mouse/strain grouping before freezing train/validation/test splits.",
        ],
    }
    inventory_path = output_dir / "inventory_summary.json"
    inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    print("NASA BPS METADATA INVENTORY COMPLETE: YES")
    print(f"Inventory: {inventory_path}")
    print(f"FITC conditions: {fitc_condition_path}")
    print(f"DAPI/MASK conditions: {dapi_condition_path}")
    print("")
    print(
        "NEXT GATE: inspect condition counts and identifier columns, then freeze a small "
        "radiation-native pilot without plate/mouse/strain leakage."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
