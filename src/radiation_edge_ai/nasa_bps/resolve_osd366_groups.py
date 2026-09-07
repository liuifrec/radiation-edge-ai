"""Resolve NASA BPS microscopy filenames against OSD-366 sample metadata.

This metadata-only utility addresses the leakage-control gate for the NASA BPS
53BP1 case.  The benchmark CSVs expose only filename, dose, particle type, and
post-exposure time, while the parent OSD-366 ISA metadata contain strain,
plate, plate-well, and sample/source information.

The script:
1. parses benchmark filenames such as
   P242_73665006707-A6_001_001_proj.tif;
2. fixes the FITC/DAPI/MASK crosswalk by stripping terminal channel suffixes;
3. downloads only the small OSD-366 ISA metadata ZIP (unless supplied locally);
4. finds the study sample table containing plate/plate-well/strain columns;
5. crosswalks benchmark plate + well to OSD-366 sample metadata;
6. reports whether source/sample/strain identifiers are suitable independent
   grouping keys for train/validation/test splitting.

No microscopy images are downloaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OSD_ACCESSION = "OSD-366"
ISA_ZIP_NAME = "OSD-366_metadata_OSD-366-ISA.zip"
OSDR_FILE_RECORD_URL = (
    "https://visualization.osdr.nasa.gov/biodata/api/v2/dataset/OSD-366/file/"
    "OSD-366_metadata_OSD-366-ISA.zip/"
)
FALLBACK_DOWNLOAD_URLS = (
    "https://genelab-data.ndc.nasa.gov/datamanager/file/Home/genelab/"
    "genelab-data/GLDS-366/metadata/OSD-366_metadata_OSD-366-ISA.zip",
    "https://genelab-data.ndc.nasa.gov/datamanager/file/Home/genelab/"
    "genelab-data/OSD-366/metadata/OSD-366_metadata_OSD-366-ISA.zip",
)

FILENAME_RE = re.compile(
    r"^(?P<plate>P\d+)_(?P<acquisition>\d+)-(?P<well>[A-Za-z]+\d+)_"
    r"(?P<field>\d+)_(?P<object>\d+)_(?P<channel>proj|DAPI|MASK)\.tif$",
    re.IGNORECASE,
)

TARGET_COLUMNS = {
    "source_name": ("Source Name",),
    "sample_name": ("Sample Name",),
    "sex": ("Characteristics: Sex",),
    "strain": ("Factor Value: Strain",),
    "time": ("Factor Value: Time of Sample Collection After Treatment",),
    "radiation": ("Factor Value: Ionizing Radiation",),
    "dose": ("Factor Value: Absorbed Radiation Dose",),
    "group_id": ("Factor Value: Group ID",),
    "plate": ("Comment: plate", "Comment[plate]"),
    "plate_well": ("Comment: plate well", "Comment[plate well]"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def parse_filename(filename: str) -> dict[str, str]:
    match = FILENAME_RE.match(Path(filename).name)
    if match is None:
        raise ValueError(f"Unrecognized NASA BPS filename: {filename}")
    result = match.groupdict()
    result["plate"] = result["plate"].upper()
    result["well"] = result["well"].upper()
    result["channel"] = result["channel"].lower()
    result["plate_acquisition"] = f"{result['plate']}_{result['acquisition']}"
    result["nucleus_key"] = (
        f"{result['plate']}_{result['acquisition']}-{result['well']}_"
        f"{result['field']}_{result['object']}"
    )
    return result


def norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def norm_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def find_column(headers: list[str], aliases: tuple[str, ...]) -> int | None:
    normalized = {norm_header(alias) for alias in aliases}
    for index, header in enumerate(headers):
        if norm_header(header) in normalized:
            return index
    return None


def iter_strings(obj: Any):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                yield key, value
            else:
                yield from iter_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_strings(value)


def discover_download_url() -> str | None:
    request = urllib.request.Request(
        OSDR_FILE_RECORD_URL,
        headers={"User-Agent": "radiation-edge-ai/0.1 OSD-366-metadata-crosswalk"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
    except Exception:
        return None
    try:
        obj = json.loads(payload.decode("utf-8"))
    except Exception:
        return None

    candidates: list[str] = []
    for key, value in iter_strings(obj):
        lower_key = key.lower()
        if value.startswith("http") and (
            "url" in lower_key or "download" in lower_key or ISA_ZIP_NAME in value
        ):
            candidates.append(value)
    for value in candidates:
        if ISA_ZIP_NAME in value:
            return value
    return candidates[0] if candidates else None


def download_file(url: str, path: Path) -> bool:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "radiation-edge-ai/0.1 OSD-366-metadata-crosswalk"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
    except Exception:
        return False
    if len(payload) < 100 or not payload.startswith(b"PK"):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return True


def ensure_isa_zip(path: Path, refresh: bool) -> tuple[Path, str]:
    if path.is_file() and not refresh:
        return path, "existing"

    discovered = discover_download_url()
    candidates = ([discovered] if discovered else []) + list(FALLBACK_DOWNLOAD_URLS)
    tried: list[str] = []
    for url in candidates:
        if not url or url in tried:
            continue
        tried.append(url)
        if download_file(url, path):
            return path, url

    raise RuntimeError(
        "Could not download OSD-366 ISA metadata automatically. Download "
        f"{ISA_ZIP_NAME} from the OSD-366 Files panel and rerun with --isa-zip PATH. "
        f"Tried: {tried}"
    )


def choose_sample_table(zip_path: Path) -> tuple[str, list[str], list[list[str]]]:
    with zipfile.ZipFile(zip_path) as archive:
        candidates = [
            name for name in archive.namelist()
            if Path(name).name.lower().startswith("s_") and name.lower().endswith(".txt")
        ]
        if not candidates:
            raise RuntimeError("No s_*.txt sample table found in OSD-366 ISA ZIP")

        scored: list[tuple[int, str, list[str], list[list[str]]]] = []
        for name in candidates:
            raw = archive.read(name)
            text = raw.decode("utf-8-sig", errors="replace")
            rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
            if not rows:
                continue
            headers = rows[0]
            score = sum(
                find_column(headers, TARGET_COLUMNS[key]) is not None
                for key in ("strain", "plate", "plate_well", "source_name", "sample_name")
            )
            scored.append((score, name, headers, rows[1:]))

    if not scored:
        raise RuntimeError("Could not parse any OSD-366 sample table")
    scored.sort(key=lambda item: item[0], reverse=True)
    score, name, headers, rows = scored[0]
    if score < 3:
        raise RuntimeError(
            f"Best sample table {name} does not expose enough grouping columns; score={score}"
        )
    return name, headers, rows


def extract_osd_records(headers: list[str], rows: list[list[str]]) -> list[dict[str, str]]:
    indices = {key: find_column(headers, aliases) for key, aliases in TARGET_COLUMNS.items()}
    missing = [key for key in ("strain", "plate", "plate_well") if indices[key] is None]
    if missing:
        raise RuntimeError(f"OSD sample table missing required columns: {missing}")

    records: list[dict[str, str]] = []
    for row in rows:
        record: dict[str, str] = {}
        for key, index in indices.items():
            if index is None or index >= len(row):
                record[key] = ""
            else:
                record[key] = norm_text(row[index])
        if record["plate"] in {"", "Not Applicable", "N/A", "NA"}:
            continue
        if record["plate_well"] in {"", "Not Applicable", "N/A", "NA"}:
            continue
        record["plate"] = record["plate"].upper()
        record["plate_well"] = record["plate_well"].upper().replace(" ", "")
        records.append(record)
    return records


def compact_values(values) -> list[str]:
    return sorted({value for value in values if value and value != "Not Applicable"})


def counter_summary(counter: Counter) -> dict[str, object]:
    if not counter:
        return {"n_groups": 0, "min": None, "median": None, "max": None}
    sizes = sorted(counter.values())
    return {
        "n_groups": len(counter),
        "min": sizes[0],
        "median": sizes[len(sizes) // 2],
        "max": sizes[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    default_root = Path(
        os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data")
    ) / "nasa_bps_microscopy" / "metadata"
    parser.add_argument("--metadata-dir", default=str(default_root))
    parser.add_argument("--isa-zip", default=None)
    parser.add_argument("--refresh-isa", action="store_true")
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir).resolve()
    fitc_path = metadata_dir / "meta.csv"
    dapi_path = metadata_dir / "meta_DAPI_MASK.csv"
    for path in (fitc_path, dapi_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    isa_path = (
        Path(args.isa_zip).resolve()
        if args.isa_zip
        else metadata_dir / "osd366" / ISA_ZIP_NAME
    )
    isa_path, isa_source = ensure_isa_zip(isa_path, args.refresh_isa)

    fitc_rows = read_csv(fitc_path)
    dapi_rows = read_csv(dapi_path)

    fitc_parsed = []
    for row in fitc_rows:
        parsed = parse_filename(row["filename"])
        fitc_parsed.append({**row, **parsed})
    dapi_parsed = []
    for row in dapi_rows:
        parsed = parse_filename(row["filename"])
        dapi_parsed.append({**row, **parsed})

    fitc_keys = Counter(row["nucleus_key"] for row in fitc_parsed)
    dapi_keys = Counter(row["nucleus_key"] for row in dapi_parsed)
    shared_keys = set(fitc_keys) & set(dapi_keys)
    fitc_with_both_dapi_mask = 0
    dapi_channels: dict[str, set[str]] = defaultdict(set)
    for row in dapi_parsed:
        dapi_channels[row["nucleus_key"]].add(row["channel"])
    for key in fitc_keys:
        if {"dapi", "mask"}.issubset(dapi_channels.get(key, set())):
            fitc_with_both_dapi_mask += 1

    table_name, headers, table_rows = choose_sample_table(isa_path)
    osd_records = extract_osd_records(headers, table_rows)

    osd_by_plate_well: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for record in osd_records:
        osd_by_plate_well[(record["plate"], record["plate_well"])].append(record)

    image_plate_wells = {(row["plate"], row["well"]) for row in fitc_parsed}
    matched_plate_wells = image_plate_wells & set(osd_by_plate_well)

    image_crosswalk_rows: list[dict[str, str]] = []
    n_unmatched_images = 0
    n_ambiguous_images = 0
    for row in fitc_parsed:
        candidates = osd_by_plate_well.get((row["plate"], row["well"]), [])
        if not candidates:
            n_unmatched_images += 1
            continue
        identities = {
            (
                candidate.get("source_name", ""),
                candidate.get("sample_name", ""),
                candidate.get("strain", ""),
                candidate.get("sex", ""),
            )
            for candidate in candidates
        }
        if len(identities) > 1:
            n_ambiguous_images += 1
        candidate = candidates[0]
        image_crosswalk_rows.append(
            {
                "filename": row["filename"],
                "plate": row["plate"],
                "acquisition": row["acquisition"],
                "plate_acquisition": row["plate_acquisition"],
                "well": row["well"],
                "field": row["field"],
                "object": row["object"],
                "dose_Gy": row["dose_Gy"],
                "particle_type": row["particle_type"],
                "hr_post_exposure": row["hr_post_exposure"],
                "osd_source_name": candidate.get("source_name", ""),
                "osd_sample_name": candidate.get("sample_name", ""),
                "osd_strain": candidate.get("strain", ""),
                "osd_sex": candidate.get("sex", ""),
                "osd_group_id": candidate.get("group_id", ""),
                "osd_radiation": candidate.get("radiation", ""),
                "osd_dose": candidate.get("dose", ""),
                "osd_time": candidate.get("time", ""),
                "n_osd_candidates_for_plate_well": str(len(candidates)),
                "n_identity_candidates": str(len(identities)),
            }
        )

    crosswalk_path = metadata_dir / "fitc_53bp1_osd366_crosswalk.csv"
    if image_crosswalk_rows:
        with crosswalk_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(image_crosswalk_rows[0].keys()))
            writer.writeheader()
            writer.writerows(image_crosswalk_rows)

    plate_counts = Counter(row["plate"] for row in fitc_parsed)
    acquisition_counts = Counter(row["plate_acquisition"] for row in fitc_parsed)
    plate_well_counts = Counter((row["plate"], row["well"]) for row in fitc_parsed)
    source_counts = Counter(
        row["osd_source_name"] for row in image_crosswalk_rows if row["osd_source_name"]
    )
    sample_counts = Counter(
        row["osd_sample_name"] for row in image_crosswalk_rows if row["osd_sample_name"]
    )
    strain_counts = Counter(
        row["osd_strain"] for row in image_crosswalk_rows if row["osd_strain"]
    )

    plate_identity_sets = defaultdict(set)
    for record in osd_records:
        plate_identity_sets[(record["plate"], record["plate_well"])].add(
            (record.get("source_name", ""), record.get("sample_name", ""), record.get("strain", ""))
        )
    ambiguous_plate_wells = {
        key: sorted(values)
        for key, values in plate_identity_sets.items()
        if len(values) > 1 and key in image_plate_wells
    }

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fitc_metadata_sha256": sha256_file(fitc_path),
        "dapi_mask_metadata_sha256": sha256_file(dapi_path),
        "osd366_isa_zip": str(isa_path),
        "osd366_isa_zip_sha256": sha256_file(isa_path),
        "osd366_isa_source": isa_source,
        "osd366_sample_table": table_name,
        "benchmark_filename_parse": {
            "fitc_rows": len(fitc_parsed),
            "dapi_mask_rows": len(dapi_parsed),
            "unique_plate_codes": len(plate_counts),
            "unique_plate_acquisitions": len(acquisition_counts),
            "unique_plate_wells": len(plate_well_counts),
        },
        "channel_crosswalk": {
            "fitc_unique_nucleus_keys": len(fitc_keys),
            "dapi_mask_unique_nucleus_keys": len(dapi_keys),
            "shared_nucleus_keys": len(shared_keys),
            "fitc_keys_with_both_dapi_and_mask": fitc_with_both_dapi_mask,
        },
        "osd_sample_records_with_plate_well": len(osd_records),
        "osd_unique_plate_wells": len(osd_by_plate_well),
        "benchmark_plate_wells_matched_to_osd": len(matched_plate_wells),
        "benchmark_unique_plate_wells": len(image_plate_wells),
        "fitc_images_crosswalked": len(image_crosswalk_rows),
        "fitc_images_unmatched": n_unmatched_images,
        "fitc_images_with_ambiguous_identity": n_ambiguous_images,
        "ambiguous_benchmark_plate_wells": len(ambiguous_plate_wells),
        "group_cardinality": {
            "plate": counter_summary(plate_counts),
            "plate_acquisition": counter_summary(acquisition_counts),
            "plate_well": counter_summary(plate_well_counts),
            "osd_source_name": counter_summary(source_counts),
            "osd_sample_name": counter_summary(sample_counts),
            "osd_strain": counter_summary(strain_counts),
        },
        "osd_values": {
            "source_names": compact_values(r.get("source_name", "") for r in osd_records),
            "sample_names": compact_values(r.get("sample_name", "") for r in osd_records),
            "strains": compact_values(r.get("strain", "") for r in osd_records),
            "sexes": compact_values(r.get("sex", "") for r in osd_records),
        },
        "crosswalk_csv": str(crosswalk_path),
    }

    summary_path = metadata_dir / "osd366_group_resolution.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Radiation Edge AI - NASA BPS / OSD-366 group resolution")
    print("Metadata only: no microscopy images are downloaded.")
    print("")
    print("[Corrected channel crosswalk]")
    print(f"FITC unique nucleus keys: {len(fitc_keys)}")
    print(f"DAPI/MASK unique nucleus keys: {len(dapi_keys)}")
    print(f"shared nucleus keys: {len(shared_keys)}")
    print(f"FITC keys with both DAPI+MASK: {fitc_with_both_dapi_mask}")
    print("")
    print("[Filename grouping]")
    print(f"plate codes (P###): {len(plate_counts)}")
    print(f"plate+acquisition IDs: {len(acquisition_counts)}")
    print(f"plate+well groups: {len(plate_well_counts)}")
    print("")
    print("[OSD-366 ISA]")
    print(f"ISA ZIP: {isa_path}")
    print(f"SHA256: {sha256_file(isa_path)}")
    print(f"sample table: {table_name}")
    print(f"records with plate+well: {len(osd_records)}")
    print(f"unique OSD plate+well keys: {len(osd_by_plate_well)}")
    print("")
    print("[Benchmark -> OSD-366 crosswalk]")
    print(f"benchmark unique plate+well keys: {len(image_plate_wells)}")
    print(f"matched plate+well keys: {len(matched_plate_wells)}")
    print(f"FITC images crosswalked: {len(image_crosswalk_rows)}/{len(fitc_parsed)}")
    print(f"unmatched FITC images: {n_unmatched_images}")
    print(f"images with ambiguous OSD identity: {n_ambiguous_images}")
    print(f"ambiguous benchmark plate+well keys: {len(ambiguous_plate_wells)}")
    print("")
    print("[Resolved biological groups in crosswalked FITC]")
    print(f"source names: {len(source_counts)}")
    print(f"sample names: {len(sample_counts)}")
    print(f"strains: {len(strain_counts)}")
    if strain_counts:
        print("strain image counts:")
        for strain, count in strain_counts.most_common():
            print(f"  {strain}: {count}")
    print("")
    print("NASA BPS / OSD-366 GROUP RESOLUTION COMPLETE: YES")
    print(f"Summary: {summary_path}")
    if image_crosswalk_rows:
        print(f"Crosswalk: {crosswalk_path}")
    print("")
    print(
        "NEXT GATE: choose the highest-level biologically independent grouping key that is "
        "unambiguous across plate+well records (prefer individual source/cell line over plate; "
        "use strain-held-out evaluation as an additional generalization test)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
