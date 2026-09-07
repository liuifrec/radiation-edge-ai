"""Discover recoverable 53BP1 foci/nfoci label tables for OSD-366.

This is a metadata-only discovery step. It does not download microscopy images.
It searches three places for foci-related labels that could provide an
independent reference endpoint for the NASA BPS microscopy pilot:

1. the already-downloaded OSD-366 ISA metadata ZIP;
2. local NASA BPS/OSD metadata files under RADEDGE_DATA_ROOT;
3. the current NASA OSDR Biological Data API dataset file listing.

The script ranks current OSDR files by names/metadata containing terms such as
``nfoci``, ``foci``, ``histology``, ``pheno``, ``53bp1``, and ``immunostaining``.
It also reports two historical OSD-366 phenotype files that NASA's current
version history says were removed in Version 2:

- GLDS-366_Histology_raw_pheno_V3.csv
- GLDS-366_Histology_processed_pheno_V2.csv

Those historical names are treated as leads only; their contents are not
assumed available or authoritative until recovered and inspected.

Outputs are small CSV/JSON inventories only. No candidate data file is fetched
automatically; a later gate can explicitly download a selected tabular file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ACCESSION = "OSD-366"
OSDR_FILES_URL = "https://visualization.osdr.nasa.gov/biodata/api/v2/dataset/OSD-366/files/"
USER_AGENT = "radiation-edge-ai/0.1 OSD-366-foci-label-discovery"

HISTORICAL_FILES = (
    "GLDS-366_Histology_raw_pheno_V3.csv",
    "GLDS-366_Histology_processed_pheno_V2.csv",
)

KEYWORDS = (
    "nfoci",
    "foci",
    "focus",
    "53bp1",
    "histology",
    "pheno",
    "phenotype",
    "immunostaining",
    "processed",
)

TEXT_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".xml", ".yaml", ".yml"}


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def keyword_hits(text: str) -> list[str]:
    lower = text.lower()
    return [keyword for keyword in KEYWORDS if keyword in lower]


def compact_record_text(record: dict[str, Any]) -> str:
    values: list[str] = []
    for key, value in record.items():
        if isinstance(value, (str, int, float, bool)):
            values.append(f"{key}={value}")
    return " | ".join(values)


def pick_filename(record: dict[str, Any]) -> str | None:
    candidates: list[str] = []
    for key, value in record.items():
        if not isinstance(value, str):
            continue
        key_lower = str(key).lower()
        if "file" in key_lower or "name" in key_lower or key_lower == "url":
            basename = value.replace("\\", "/").rstrip("/").split("/")[-1]
            if basename:
                candidates.append(basename)
    if not candidates:
        return None
    # Prefer strings with a plausible file extension.
    with_ext = [name for name in candidates if "." in name]
    if with_ext:
        return min(with_ext, key=len)
    return min(candidates, key=len)


def pick_direct_url(record: dict[str, Any]) -> str | None:
    # Current OSDR file records use URL for the actual download and REST_URL for
    # the JSON record. Never treat REST_URL as the data file.
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
    return None


def find_size(record: dict[str, Any]) -> str:
    for key, value in record.items():
        key_lower = str(key).lower()
        if any(token in key_lower for token in ("size", "bytes", "length")):
            if isinstance(value, (str, int, float)):
                return str(value)
    return ""


def scan_osdr_listing(payload: Any) -> tuple[list[dict[str, Any]], set[str]]:
    by_name: dict[str, dict[str, Any]] = {}
    all_names: set[str] = set()

    for record in iter_dicts(payload):
        filename = pick_filename(record)
        if filename is None:
            continue
        all_names.add(filename)
        text = compact_record_text(record)
        hits = keyword_hits(filename + " " + text)
        historical = filename in HISTORICAL_FILES
        if not hits and not historical:
            continue

        score = 0
        lower = filename.lower()
        if "nfoci" in lower:
            score += 100
        if "foci" in lower or "focus" in lower:
            score += 80
        if "histology" in lower:
            score += 60
        if "pheno" in lower:
            score += 50
        if "53bp1" in lower:
            score += 50
        if "immunostaining" in lower:
            score += 20
        if "processed" in lower:
            score += 10
        suffix = Path(filename).suffix.lower()
        if suffix in {".csv", ".tsv", ".txt"}:
            score += 25
        if suffix in {".zip", ".tif", ".tiff", ".ics"}:
            score -= 20
        if historical:
            score += 200

        row = {
            "score": score,
            "filename": filename,
            "keyword_hits": ";".join(sorted(set(hits))),
            "historical_known_name": historical,
            "size_field": find_size(record),
            "direct_url": pick_direct_url(record) or "",
            "record_preview": text[:1000],
        }
        previous = by_name.get(filename)
        if previous is None or int(row["score"]) > int(previous["score"]):
            by_name[filename] = row

    rows = sorted(by_name.values(), key=lambda row: (-int(row["score"]), str(row["filename"])))
    return rows, all_names


def text_keyword_context(text: str, limit: int = 4) -> list[str]:
    contexts: list[str] = []
    for keyword in KEYWORDS:
        for match in re.finditer(re.escape(keyword), text, flags=re.IGNORECASE):
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 180)
            snippet = re.sub(r"\s+", " ", text[start:end]).strip()
            if snippet and snippet not in contexts:
                contexts.append(snippet)
            if len(contexts) >= limit:
                return contexts
    return contexts


def scan_isa_zip(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with zipfile.ZipFile(path, "r") as archive:
        for info in archive.infolist():
            suffix = Path(info.filename).suffix.lower()
            if suffix not in TEXT_SUFFIXES:
                continue
            try:
                raw = archive.read(info)
                text = raw.decode("utf-8", errors="replace")
            except Exception as exc:
                rows.append(
                    {
                        "source": "isa_zip",
                        "path": info.filename,
                        "keyword_hits": "",
                        "contexts": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            hits = keyword_hits(info.filename + "\n" + text)
            if not hits:
                continue
            rows.append(
                {
                    "source": "isa_zip",
                    "path": info.filename,
                    "keyword_hits": ";".join(sorted(set(hits))),
                    "contexts": " || ".join(text_keyword_context(text)),
                    "error": "",
                }
            )
    return rows


def scan_local_metadata(root: Path, exclude: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            if path.resolve().is_relative_to(exclude.resolve()):
                continue
        except AttributeError:
            # Python 3.8 compatibility fallback.
            try:
                path.resolve().relative_to(exclude.resolve())
                continue
            except ValueError:
                pass
        try:
            if path.stat().st_size > 50 * 1024 * 1024:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            rows.append(
                {
                    "source": "local_metadata",
                    "path": str(path),
                    "keyword_hits": "",
                    "contexts": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        hits = keyword_hits(path.name + "\n" + text)
        if not hits:
            continue
        rows.append(
            {
                "source": "local_metadata",
                "path": str(path),
                "keyword_hits": ";".join(sorted(set(hits))),
                "contexts": " || ".join(text_keyword_context(text)),
                "error": "",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    nasa_root = data_root / "nasa_bps_microscopy"
    metadata_root = nasa_root / "metadata"
    default_isa = metadata_root / "osd366" / "OSD-366_metadata_OSD-366-ISA.zip"
    default_output = metadata_root / "foci_label_discovery"
    parser.add_argument("--isa-zip", default=str(default_isa))
    parser.add_argument("--metadata-root", default=str(metadata_root))
    parser.add_argument("--output-dir", default=str(default_output))
    args = parser.parse_args()

    isa_zip = Path(args.isa_zip).resolve()
    metadata_root = Path(args.metadata_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Radiation Edge AI - discover OSD-366 53BP1 foci-label sources")
    print("Metadata only: no microscopy images or candidate phenotype files are downloaded.")
    print("")

    local_hits = scan_isa_zip(isa_zip)
    local_hits.extend(scan_local_metadata(metadata_root, output_dir))

    print(f"[Local metadata] keyword-hit files/ZIP members: {len(local_hits)}")
    for row in local_hits[:20]:
        print(f"  {row['source']}: {row['path']} -> {row['keyword_hits']}")
    if len(local_hits) > 20:
        print(f"  ... {len(local_hits) - 20} additional local hits written to CSV")
    print("")

    print(f"[OSDR API] {OSDR_FILES_URL}")
    payload = get_json(OSDR_FILES_URL)
    candidates, all_names = scan_osdr_listing(payload)
    print(f"current file names discovered: {len(all_names)}")
    print(f"foci/phenotype candidate files: {len(candidates)}")
    print("")

    for historical in HISTORICAL_FILES:
        status = "PRESENT in current listing" if historical in all_names else "NOT PRESENT in current listing"
        print(f"[Historical lead] {historical}: {status}")
    print("")

    print("[Top current candidates]")
    for row in candidates[:30]:
        print(
            f"score={int(row['score']):>3}  {row['filename']}  "
            f"hits={row['keyword_hits']}"
        )
    if not candidates:
        print("  none")
    print("")

    write_csv(
        output_dir / "local_keyword_hits.csv",
        local_hits,
        ["source", "path", "keyword_hits", "contexts", "error"],
    )
    write_csv(
        output_dir / "osdr_candidate_files.csv",
        candidates,
        [
            "score",
            "filename",
            "keyword_hits",
            "historical_known_name",
            "size_field",
            "direct_url",
            "record_preview",
        ],
    )

    keyword_counter = Counter()
    for row in candidates:
        for hit in str(row["keyword_hits"]).split(";"):
            if hit:
                keyword_counter[hit] += 1

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "accession": ACCESSION,
        "osdr_files_url": OSDR_FILES_URL,
        "isa_zip": str(isa_zip),
        "local_keyword_hit_count": len(local_hits),
        "current_file_name_count": len(all_names),
        "candidate_file_count": len(candidates),
        "candidate_keyword_counts": dict(sorted(keyword_counter.items())),
        "historical_leads": [
            {
                "filename": filename,
                "present_in_current_listing": filename in all_names,
            }
            for filename in HISTORICAL_FILES
        ],
        "top_candidates": candidates[:50],
        "interpretation": [
            "Current OSDR Version 2 is authoritative for currently distributed files.",
            "Historical removed phenotype filenames are discovery leads only.",
            "Do not infer per-nucleus nfoci labels from condition metadata alone.",
            "Do not download raw image ZIPs at this gate.",
            "If a small tabular foci/phenotype candidate is found, inspect its columns and identifiers before any crosswalk to the BPS nucleus crops.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("OSD-366 FOCI-LABEL DISCOVERY COMPLETE: YES")
    print(f"Candidates: {output_dir / 'osdr_candidate_files.csv'}")
    print(f"Local hits: {output_dir / 'local_keyword_hits.csv'}")
    print(f"Summary: {summary_path}")
    print("")
    print(
        "NEXT GATE: inspect the top candidate file names/metadata. If a small tabular foci/nfoci "
        "file is identified, fetch only that file and test whether its identifiers can be mapped "
        "to the frozen BPS nucleus manifest."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
