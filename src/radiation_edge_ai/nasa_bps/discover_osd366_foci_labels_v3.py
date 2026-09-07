"""Precise OSD-366 53BP1/foci-label discovery using filename-only evidence.

The v2 discovery deliberately scanned broad assay-table context, but that can
make every immunostaining image archive appear to have strong ``53bp1`` or
``pheno`` evidence simply because those words occur elsewhere on the same ISA
row.  This v3 pass separates three things that must not be conflated:

1. terms present in the *candidate filename itself*;
2. the assay/table context in which that filename is referenced; and
3. historical phenotype filenames known to have been removed from OSD-366 v2.

Only filename-level evidence is used to call a current foci/phenotype table a
strong candidate.  ISA tables are parsed by header so we can report actual
Data File / Processed Data / Raw Data references without inheriting unrelated
text from the rest of the row.

This is metadata-only.  It downloads no microscopy archive and no phenotype
candidate file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ACCESSION = "OSD-366"
OSDR_FILES_URL = "https://visualization.osdr.nasa.gov/biodata/api/v2/dataset/OSD-366/files/"
USER_AGENT = "radiation-edge-ai/0.1 OSD-366-foci-label-discovery-v3"

HISTORICAL_REMOVED = (
    "GLDS-366_Histology_raw_pheno_V3.csv",
    "GLDS-366_Histology_processed_pheno_V2.csv",
)

TABULAR_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json"}
IMAGE_SUFFIXES = {".zip", ".tif", ".tiff", ".ics"}
KNOWN_SUFFIXES = TABULAR_SUFFIXES | IMAGE_SUFFIXES

STRONG_FILENAME_TERMS = (
    "nfoci",
    "foci",
    "focus",
    "53bp1",
    "histology",
    "pheno",
    "phenotype",
    "focipergray",
    "foci_per_gray",
    "fpg",
    "background",
    "bgd",
    "rif",
)

FILE_COLUMN_TERMS = (
    "data file",
    "raw data",
    "processed data",
    "image file",
    "file name",
    "filename",
)

IMAGING_CONTEXT_TERMS = (
    "molecular-cellular-imaging",
    "molecular cellular imaging",
    "immunostaining",
    "zeiss",
)


def get_json(url: str, timeout: int = 120) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def iter_dicts(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from iter_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_dicts(value)


def filename_from_url(value: str) -> str | None:
    try:
        parsed = urllib.parse.urlparse(value)
    except Exception:
        return None
    query = urllib.parse.parse_qs(parsed.query)
    for key in ("file", "filename", "name"):
        values = query.get(key)
        if values:
            candidate = urllib.parse.unquote(values[0]).replace("\\", "/").split("/")[-1]
            if Path(candidate).suffix.lower() in KNOWN_SUFFIXES:
                return candidate
    basename = urllib.parse.unquote(parsed.path).replace("\\", "/").rstrip("/").split("/")[-1]
    if Path(basename).suffix.lower() in KNOWN_SUFFIXES:
        return basename
    return None


def plausible_filename(value: str) -> str | None:
    value = value.strip().strip('"').strip("'")
    if not value:
        return None
    if value.startswith("http://") or value.startswith("https://"):
        return filename_from_url(value)
    candidate = urllib.parse.unquote(value).replace("\\", "/").rstrip("/").split("/")[-1]
    if Path(candidate).suffix.lower() in KNOWN_SUFFIXES:
        return candidate
    return None


def pick_filename(record: dict[str, Any]) -> str | None:
    ranked: list[tuple[int, int, str]] = []
    for key, value in record.items():
        if not isinstance(value, str):
            continue
        candidate = plausible_filename(value)
        if not candidate:
            continue
        key_lower = str(key).lower()
        if key_lower in {"filename", "file_name", "file name", "name"}:
            priority = 0
        elif key_lower == "url":
            priority = 1
        elif "file" in key_lower:
            priority = 2
        else:
            priority = 3
        ranked.append((priority, len(candidate), candidate))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][2]


def pick_direct_url(record: dict[str, Any]) -> str:
    for key, value in record.items():
        if isinstance(value, str) and str(key).lower() == "url" and value.startswith("http"):
            return value
    for key, value in record.items():
        if not isinstance(value, str) or not value.startswith("http"):
            continue
        key_lower = str(key).lower()
        if key_lower == "rest_url" or key_lower.endswith("rest_url"):
            continue
        if "download" in key_lower or "remote" in key_lower:
            return value
    return ""


def term_hits(text: str, terms: tuple[str, ...]) -> list[str]:
    lower = text.lower()
    return sorted({term for term in terms if term in lower})


def collect_current_files(payload: Any) -> list[dict[str, Any]]:
    """Return one record per current file, scoring filename only."""
    by_name: dict[str, dict[str, Any]] = {}
    for record in iter_dicts(payload):
        filename = pick_filename(record)
        if not filename:
            continue
        suffix = Path(filename).suffix.lower()
        if suffix not in KNOWN_SUFFIXES:
            continue
        hits = term_hits(filename, STRONG_FILENAME_TERMS)
        row = {
            "filename": filename,
            "suffix": suffix,
            "is_tabular": suffix in TABULAR_SUFFIXES,
            "is_image_archive": suffix in IMAGE_SUFFIXES,
            "filename_strong_hits": ";".join(hits),
            "direct_url": pick_direct_url(record),
        }
        previous = by_name.get(filename)
        if previous is None or (not previous["direct_url"] and row["direct_url"]):
            by_name[filename] = row
    return sorted(by_name.values(), key=lambda row: str(row["filename"]))


def is_file_column(header: str) -> bool:
    lower = header.strip().lower()
    return any(term in lower for term in FILE_COLUMN_TERMS)


def is_imaging_context(member_name: str, headers: list[str]) -> bool:
    text = (member_name + " " + " ".join(headers)).lower()
    return any(term in text for term in IMAGING_CONTEXT_TERMS)


def scan_isa_data_file_references(isa_zip: Path) -> list[dict[str, Any]]:
    refs: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not isa_zip.is_file():
        return []

    with zipfile.ZipFile(isa_zip, "r") as archive:
        for info in archive.infolist():
            if Path(info.filename).suffix.lower() != ".txt":
                continue
            try:
                text = archive.read(info).decode("utf-8", errors="replace")
            except Exception:
                continue
            reader = csv.reader(text.splitlines(), delimiter="\t")
            try:
                headers = next(reader)
            except StopIteration:
                continue
            file_columns = [index for index, header in enumerate(headers) if is_file_column(header)]
            if not file_columns:
                continue
            imaging_context = is_imaging_context(info.filename, headers)

            for line_no, row in enumerate(reader, start=2):
                for index in file_columns:
                    if index >= len(row):
                        continue
                    raw = row[index].strip()
                    candidate = plausible_filename(raw)
                    if not candidate:
                        continue
                    suffix = Path(candidate).suffix.lower()
                    hits = term_hits(candidate, STRONG_FILENAME_TERMS)
                    header = headers[index].strip()
                    key = (info.filename, header, candidate)
                    if key in refs:
                        continue
                    refs[key] = {
                        "isa_member": info.filename,
                        "line": line_no,
                        "column": header,
                        "filename": candidate,
                        "suffix": suffix,
                        "is_tabular": suffix in TABULAR_SUFFIXES,
                        "is_image_archive": suffix in IMAGE_SUFFIXES,
                        "imaging_assay_context": imaging_context,
                        "filename_strong_hits": ";".join(hits),
                    }
    return sorted(
        refs.values(),
        key=lambda row: (str(row["isa_member"]), str(row["column"]), str(row["filename"])),
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"
    parser.add_argument(
        "--isa-zip",
        default=str(metadata_root / "osd366" / "OSD-366_metadata_OSD-366-ISA.zip"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(metadata_root / "foci_label_discovery_v3"),
    )
    args = parser.parse_args()

    isa_zip = Path(args.isa_zip).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Radiation Edge AI - OSD-366 foci-label discovery v3")
    print("Metadata only: filename-level evidence only; no candidate data files are downloaded.")
    print("")

    payload = get_json(OSDR_FILES_URL)
    current = collect_current_files(payload)
    current_names = {str(row["filename"]) for row in current}
    current_tabular = [row for row in current if bool(row["is_tabular"])]
    current_strong_tabular = [
        row for row in current_tabular if bool(row["filename_strong_hits"])
    ]

    isa_refs = scan_isa_data_file_references(isa_zip)
    isa_tabular = [row for row in isa_refs if bool(row["is_tabular"])]
    isa_imaging_tabular = [
        row for row in isa_tabular if bool(row["imaging_assay_context"])
    ]
    isa_strong_tabular = [
        row for row in isa_tabular if bool(row["filename_strong_hits"])
    ]

    print("[Current OSDR filename-only inventory]")
    print(f"parsed current files: {len(current)}")
    print(f"current tabular files: {len(current_tabular)}")
    print(f"strong foci/phenotype tabular filenames: {len(current_strong_tabular)}")
    if current_strong_tabular:
        for row in current_strong_tabular[:50]:
            print(f"  {row['filename']}  hits={row['filename_strong_hits']}")
    else:
        print("  none")
    print("")

    print("[Historical Version-1 phenotype leads]")
    for filename in HISTORICAL_REMOVED:
        state = "PRESENT" if filename in current_names else "NOT PRESENT in current Version-2 listing"
        print(f"  {filename}: {state}")
    print("")

    print("[ISA data-file references parsed by column header]")
    print(f"all data-file references: {len(isa_refs)}")
    print(f"tabular references: {len(isa_tabular)}")
    print(f"tabular references in imaging-assay context: {len(isa_imaging_tabular)}")
    print(f"strong foci/phenotype tabular filenames: {len(isa_strong_tabular)}")
    print("")

    print("[Imaging-assay tabular references]")
    if isa_imaging_tabular:
        for row in isa_imaging_tabular[:100]:
            print(
                f"  {row['isa_member']} :: {row['column']} :: {row['filename']}  "
                f"hits={row['filename_strong_hits'] or '-'}"
            )
        if len(isa_imaging_tabular) > 100:
            print(f"  ... {len(isa_imaging_tabular) - 100} additional rows written to CSV")
    else:
        print("  none")
    print("")

    fields = [
        "isa_member",
        "line",
        "column",
        "filename",
        "suffix",
        "is_tabular",
        "is_image_archive",
        "imaging_assay_context",
        "filename_strong_hits",
    ]
    write_csv(output_dir / "isa_data_file_references.csv", isa_refs, fields)
    write_csv(output_dir / "isa_imaging_tabular_references.csv", isa_imaging_tabular, fields)
    write_csv(
        output_dir / "current_strong_tabular_candidates.csv",
        current_strong_tabular,
        [
            "filename",
            "suffix",
            "is_tabular",
            "is_image_archive",
            "filename_strong_hits",
            "direct_url",
        ],
    )

    name_counter = Counter(row["filename"] for row in isa_imaging_tabular)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "accession": ACCESSION,
        "osdr_files_url": OSDR_FILES_URL,
        "isa_zip": str(isa_zip),
        "current_file_count": len(current),
        "current_tabular_file_count": len(current_tabular),
        "current_strong_tabular_filename_count": len(current_strong_tabular),
        "isa_data_file_reference_count": len(isa_refs),
        "isa_tabular_reference_count": len(isa_tabular),
        "isa_imaging_tabular_reference_count": len(isa_imaging_tabular),
        "isa_strong_tabular_filename_count": len(isa_strong_tabular),
        "imaging_tabular_filename_counts": dict(sorted(name_counter.items())),
        "historical_removed": [
            {
                "filename": filename,
                "present_in_current_version2_listing": filename in current_names,
            }
            for filename in HISTORICAL_REMOVED
        ],
        "interpretation": [
            "Strong-candidate status is based on the filename itself, never unrelated assay-row text.",
            "Imaging-assay context is reported separately and is not evidence of per-nucleus foci labels.",
            "The two historical Histology phenotype CSVs remain leads only if absent from the current Version-2 listing.",
            "If no strong current/ISA tabular filenames remain, stop broad current-file searching and move to targeted historical-file recovery or an independently defined reference endpoint.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("OSD-366 FOCI-LABEL DISCOVERY V3 COMPLETE: YES")
    print(f"Output: {output_dir}")
    if not current_strong_tabular and not isa_strong_tabular:
        print(
            "NEXT GATE: no filename-level current foci/phenotype table was found; "
            "try targeted recovery of the two removed Version-1 Histology phenotype CSVs, "
            "then freeze an independent focus-map endpoint if recovery fails."
        )
    else:
        print(
            "NEXT GATE: inspect/fetch only the filename-level strong tabular candidate(s) "
            "and test identifiers against the frozen BPS nucleus manifest."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
