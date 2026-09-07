"""Refine discovery of recoverable OSD-366 53BP1/foci label tables.

This second-pass diagnostic corrects two problems in the broad v1 discovery:

1. OSDR direct-download URLs can encode the real file name in the ``file=``
   query parameter, so treating the URL basename as the file name produces
   strings such as ``download?source=datamanager&file=...``.
2. Broad terms such as ``immunostaining`` and ``processed`` match thousands of
   image ZIPs and obscure the much smaller set of useful tabular phenotype
   candidates.

The script therefore:
- parses true file names from current OSDR file records and URL query strings;
- focuses the primary report on tabular files containing foci/phenotype terms;
- separately inventories imaging ZIPs without presenting them as label tables;
- scans the cached OSD-366 ISA ZIP for file references in assay tables;
- reports the two known historical histology phenotype files as removed leads;
- downloads no microscopy archive and no candidate phenotype file.

Run with the project .venv.  Outputs are metadata-only CSV/JSON reports under
RADEDGE_DATA_ROOT.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ACCESSION = "OSD-366"
OSDR_FILES_URL = "https://visualization.osdr.nasa.gov/biodata/api/v2/dataset/OSD-366/files/"
USER_AGENT = "radiation-edge-ai/0.1 OSD-366-foci-label-discovery-v2"

HISTORICAL_REMOVED = (
    "GLDS-366_Histology_raw_pheno_V3.csv",
    "GLDS-366_Histology_processed_pheno_V2.csv",
)

TABULAR_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json"}
IMAGE_ARCHIVE_SUFFIXES = {".zip", ".tif", ".tiff", ".ics"}

STRONG_TERMS = (
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
WEAK_TERMS = ("immunostaining", "processed")


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


def iter_strings(obj: Any) -> Iterable[tuple[str, str]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                yield str(key), value
            else:
                yield from iter_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_strings(value)


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
            if candidate:
                return candidate
    basename = urllib.parse.unquote(parsed.path).replace("\\", "/").rstrip("/").split("/")[-1]
    if "." in basename:
        return basename
    return None


def plausible_filename(value: str) -> str | None:
    if value.startswith("http://") or value.startswith("https://"):
        return filename_from_url(value)
    candidate = urllib.parse.unquote(value).replace("\\", "/").rstrip("/").split("/")[-1]
    suffix = Path(candidate).suffix.lower()
    if suffix in TABULAR_SUFFIXES | IMAGE_ARCHIVE_SUFFIXES:
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
        priority = 0
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
    return [term for term in terms if term in lower]


def compact_record_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in record.items():
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def collect_current_files(payload: Any) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for record in iter_dicts(payload):
        filename = pick_filename(record)
        if not filename:
            continue
        suffix = Path(filename).suffix.lower()
        if suffix not in TABULAR_SUFFIXES | IMAGE_ARCHIVE_SUFFIXES:
            continue
        text = filename + " " + compact_record_text(record)
        strong = sorted(set(term_hits(text, STRONG_TERMS)))
        weak = sorted(set(term_hits(text, WEAK_TERMS)))
        row = {
            "filename": filename,
            "suffix": suffix,
            "is_tabular": suffix in TABULAR_SUFFIXES,
            "is_image_archive": suffix in IMAGE_ARCHIVE_SUFFIXES,
            "strong_hits": ";".join(strong),
            "weak_hits": ";".join(weak),
            "direct_url": pick_direct_url(record),
            "record_preview": compact_record_text(record)[:1200],
        }
        previous = by_name.get(filename)
        if previous is None or (bool(strong), bool(weak)) > (
            bool(previous["strong_hits"]),
            bool(previous["weak_hits"]),
        ):
            by_name[filename] = row
    return sorted(by_name.values(), key=lambda row: str(row["filename"]))


def scan_isa_references(isa_zip: Path) -> list[dict[str, Any]]:
    refs: dict[tuple[str, str], dict[str, Any]] = {}
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
            for line_no, line in enumerate(text.splitlines(), start=1):
                cells = line.split("\t")
                for cell in cells:
                    candidate = plausible_filename(cell.strip())
                    if not candidate:
                        continue
                    strong = sorted(set(term_hits(candidate + " " + line, STRONG_TERMS)))
                    weak = sorted(set(term_hits(candidate + " " + line, WEAK_TERMS)))
                    key = (info.filename, candidate)
                    if key not in refs:
                        refs[key] = {
                            "isa_member": info.filename,
                            "line": line_no,
                            "filename": candidate,
                            "suffix": Path(candidate).suffix.lower(),
                            "strong_hits": ";".join(strong),
                            "weak_hits": ";".join(weak),
                            "line_preview": re.sub(r"\s+", " ", line)[:1000],
                        }
    return sorted(refs.values(), key=lambda row: (str(row["isa_member"]), str(row["filename"])))


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
        default=str(metadata_root / "foci_label_discovery_v2"),
    )
    args = parser.parse_args()

    isa_zip = Path(args.isa_zip).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Radiation Edge AI - OSD-366 foci-label discovery v2")
    print("Metadata only: no microscopy archives or phenotype candidates are downloaded.")
    print("")

    payload = get_json(OSDR_FILES_URL)
    current = collect_current_files(payload)
    current_names = {str(row["filename"]) for row in current}

    tabular_strong = [
        row for row in current if bool(row["is_tabular"]) and bool(row["strong_hits"])
    ]
    tabular_weak_only = [
        row
        for row in current
        if bool(row["is_tabular"]) and not row["strong_hits"] and bool(row["weak_hits"])
    ]
    imaging = [
        row
        for row in current
        if bool(row["is_image_archive"]) and (row["strong_hits"] or row["weak_hits"])
    ]

    isa_refs = scan_isa_references(isa_zip)
    isa_relevant = [row for row in isa_refs if row["strong_hits"] or row["weak_hits"]]

    print(f"[Current OSDR inventory] parsed files: {len(current)}")
    print(f"strong foci/phenotype tabular candidates: {len(tabular_strong)}")
    print(f"weak-only tabular candidates: {len(tabular_weak_only)}")
    print(f"imaging archives matching broad terms: {len(imaging)}")
    print("")

    print("[Historical Version-1 phenotype leads]")
    for filename in HISTORICAL_REMOVED:
        state = "PRESENT" if filename in current_names else "NOT PRESENT in current Version-2 listing"
        print(f"  {filename}: {state}")
    print("")

    print("[Strong current tabular candidates]")
    if tabular_strong:
        for row in tabular_strong[:50]:
            print(f"  {row['filename']}  hits={row['strong_hits']}")
    else:
        print("  none")
    print("")

    print("[ISA-referenced foci/phenotype/immunostaining files]")
    if isa_relevant:
        for row in isa_relevant[:50]:
            print(
                f"  {row['isa_member']} :: {row['filename']}  "
                f"strong={row['strong_hits'] or '-'} weak={row['weak_hits'] or '-'}"
            )
        if len(isa_relevant) > 50:
            print(f"  ... {len(isa_relevant) - 50} additional references written to CSV")
    else:
        print("  none")
    print("")

    write_csv(
        output_dir / "current_strong_tabular_candidates.csv",
        tabular_strong,
        [
            "filename",
            "suffix",
            "is_tabular",
            "is_image_archive",
            "strong_hits",
            "weak_hits",
            "direct_url",
            "record_preview",
        ],
    )
    write_csv(
        output_dir / "current_weak_tabular_candidates.csv",
        tabular_weak_only,
        [
            "filename",
            "suffix",
            "is_tabular",
            "is_image_archive",
            "strong_hits",
            "weak_hits",
            "direct_url",
            "record_preview",
        ],
    )
    write_csv(
        output_dir / "isa_file_references.csv",
        isa_refs,
        [
            "isa_member",
            "line",
            "filename",
            "suffix",
            "strong_hits",
            "weak_hits",
            "line_preview",
        ],
    )
    write_csv(
        output_dir / "matching_imaging_archives.csv",
        imaging,
        [
            "filename",
            "suffix",
            "is_tabular",
            "is_image_archive",
            "strong_hits",
            "weak_hits",
            "direct_url",
            "record_preview",
        ],
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "accession": ACCESSION,
        "osdr_files_url": OSDR_FILES_URL,
        "isa_zip": str(isa_zip),
        "parsed_current_file_count": len(current),
        "strong_tabular_candidate_count": len(tabular_strong),
        "weak_tabular_candidate_count": len(tabular_weak_only),
        "matching_imaging_archive_count": len(imaging),
        "historical_removed": [
            {
                "filename": filename,
                "present_in_current_version2_listing": filename in current_names,
            }
            for filename in HISTORICAL_REMOVED
        ],
        "isa_reference_count": len(isa_refs),
        "isa_relevant_reference_count": len(isa_relevant),
        "strong_candidate_term_counts": dict(
            Counter(
                hit
                for row in tabular_strong
                for hit in str(row["strong_hits"]).split(";")
                if hit
            )
        ),
        "interpretation": [
            "Image ZIPs are not treated as candidate foci-label tables merely because their names contain immunostaining/processed.",
            "The two historical GLDS-366 Histology phenotype CSV files are known Version-1 leads and are expected to be absent from the current Version-2 file listing.",
            "If no strong current tabular candidate exists, next try to recover the removed Version-1 phenotype CSVs or inspect whether current processed-image archives contain embedded tabular outputs before constructing new pseudo-labels.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("OSD-366 FOCI-LABEL DISCOVERY V2 COMPLETE: YES")
    print(f"Output: {output_dir}")
    if tabular_strong:
        print("NEXT GATE: inspect/fetch only the strongest small tabular candidate(s) and test identifiers against the frozen BPS manifest.")
    else:
        print("NEXT GATE: no current strong tabular label candidate was found; probe the removed Version-1 histology phenotype CSVs and/or inspect current processed-image archive contents without bulk download.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
