"""Fetch and inspect the three current OSD-366 immunostaining phenotype files.

This is the first deliberately narrow data-recovery step after foci-label
discovery v3.  It downloads only the three small tabular/text files currently
listed by NASA OSDR:

- LSDS-111_immunostaining_Phenotypes_description.txt
- LSDS-111_immunostaining_Processed_pheno_V2.csv
- LSDS-111_immunostaining_Raw_pheno_V3.csv

It does NOT download any microscopy ZIP archive.

For the two CSV files the script reports row/column counts, candidate foci
columns, example values, and generic identifier overlap against the frozen NASA
BPS pilot-v1 manifest.  The overlap diagnostic deliberately avoids assuming
that the phenotype tables are per-nucleus; it tests whether columns map to
nucleus keys, image filenames, source names, sample names, plate/well, or other
manifest identifiers before any biological use is allowed.

Run with the project .venv; only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ACCESSION = "OSD-366"
OSDR_FILES_URL = "https://visualization.osdr.nasa.gov/biodata/api/v2/dataset/OSD-366/files/"
USER_AGENT = "radiation-edge-ai/0.1 OSD-366-current-phenotype-inspect"

TARGET_FILES = (
    "LSDS-111_immunostaining_Phenotypes_description.txt",
    "LSDS-111_immunostaining_Processed_pheno_V2.csv",
    "LSDS-111_immunostaining_Raw_pheno_V3.csv",
)

MAX_BYTES_PER_FILE = 50 * 1024 * 1024
FROZEN_MANIFEST_SHA256 = "41ff2e4d64577f17c976dea5bbc17221fe63be9e2e7e2291309b520027fbe725"

FOCI_TERMS = (
    "nfoci",
    "foci",
    "focus",
    "53bp1",
    "background",
    "bgd",
    "focipergray",
    "foci_per_gray",
    "fpg",
    "rif",
    "area",
    "size",
    "intensity",
)

ID_TERMS = (
    "file",
    "image",
    "name",
    "id",
    "plate",
    "well",
    "sample",
    "source",
    "mouse",
    "animal",
    "strain",
    "cell",
    "nucleus",
    "field",
    "object",
    "acquisition",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_bytes(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > MAX_BYTES_PER_FILE:
            raise RuntimeError(
                f"Refusing unexpectedly large phenotype file ({length} bytes): {url}"
            )
        payload = response.read(MAX_BYTES_PER_FILE + 1)
    if len(payload) > MAX_BYTES_PER_FILE:
        raise RuntimeError(
            f"Refusing unexpectedly large phenotype file (> {MAX_BYTES_PER_FILE} bytes): {url}"
        )
    return payload


def get_json(url: str) -> Any:
    return json.loads(get_bytes(url).decode("utf-8"))


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
            if candidate:
                return candidate
    basename = urllib.parse.unquote(parsed.path).replace("\\", "/").rstrip("/").split("/")[-1]
    return basename or None


def plausible_filename(value: str) -> str | None:
    if value.startswith("http://") or value.startswith("https://"):
        return filename_from_url(value)
    candidate = urllib.parse.unquote(value).replace("\\", "/").rstrip("/").split("/")[-1]
    if candidate in TARGET_FILES:
        return candidate
    return None


def record_filename(record: dict[str, Any]) -> str | None:
    ranked: list[tuple[int, int, str]] = []
    for key, value in record.items():
        if not isinstance(value, str):
            continue
        candidate = plausible_filename(value)
        if candidate not in TARGET_FILES:
            continue
        kl = str(key).lower()
        priority = 0 if kl in {"filename", "file_name", "name"} else 1 if kl == "url" else 2
        ranked.append((priority, len(candidate), candidate))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][2]


def record_download_url(record: dict[str, Any]) -> str | None:
    for key, value in record.items():
        if isinstance(value, str) and str(key).lower() == "url" and value.startswith("http"):
            return value
    for key, value in record.items():
        if not isinstance(value, str) or not value.startswith("http"):
            continue
        kl = str(key).lower()
        if kl == "rest_url" or kl.endswith("rest_url"):
            continue
        if "download" in kl or "remote" in kl:
            return value
    return None


def discover_targets(payload: Any) -> dict[str, str]:
    found: dict[str, str] = {}
    for record in iter_dicts(payload):
        filename = record_filename(record)
        if filename is None:
            continue
        url = record_download_url(record)
        if url:
            found[filename] = url
    return found


def decode_text(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError(f"Frozen manifest is empty: {path}")
    return rows


def normalize(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("\\", "/")
    text = urllib.parse.unquote(text)
    text = re.sub(r"\s+", "", text)
    return text


def basename_stem(value: str) -> str:
    name = normalize(value).split("/")[-1]
    for suffix in ("_proj.tif", "_dapi.tif", "_mask.tif", ".tif", ".tiff", ".csv", ".txt"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def manifest_value_sets(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    fields = (
        "sample_id",
        "nucleus_key",
        "source_name",
        "sample_name",
        "strain",
        "sex",
        "plate",
        "acquisition",
        "well",
        "fitc_filename",
        "dapi_filename",
        "mask_filename",
    )
    for field in fields:
        values = {normalize(row.get(field, "")) for row in rows if row.get(field, "").strip()}
        if values:
            sets[field] = values

    sets["plate_well"] = {
        normalize(f"{row.get('plate','')}_{row.get('well','')}")
        for row in rows
        if row.get("plate", "").strip() and row.get("well", "").strip()
    }
    sets["image_stem"] = {
        basename_stem(row[field])
        for row in rows
        for field in ("fitc_filename", "dapi_filename", "mask_filename")
        if row.get(field, "").strip()
    }
    return sets


def sniff_csv(text: str) -> tuple[list[dict[str, str]], list[str], str]:
    sample = text[:65536]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    rows = [dict(row) for row in reader]
    columns = list(reader.fieldnames or [])
    return rows, columns, delimiter


def candidate_columns(columns: list[str], terms: tuple[str, ...]) -> list[str]:
    result = []
    for column in columns:
        lower = column.lower().replace(" ", "").replace("_", "")
        if any(term.replace("_", "") in lower for term in terms):
            result.append(column)
    return result


def summarize_values(rows: list[dict[str, str]], column: str, limit: int = 8) -> dict[str, Any]:
    values = [str(row.get(column, "")).strip() for row in rows if str(row.get(column, "")).strip()]
    counts = Counter(values)
    numeric = []
    for value in values:
        try:
            numeric.append(float(value))
        except ValueError:
            pass
    summary: dict[str, Any] = {
        "nonempty": len(values),
        "unique": len(counts),
        "examples": [value for value, _count in counts.most_common(limit)],
    }
    if numeric and len(numeric) == len(values):
        numeric_sorted = sorted(numeric)
        summary.update(
            {
                "numeric_min": numeric_sorted[0],
                "numeric_median": numeric_sorted[len(numeric_sorted) // 2],
                "numeric_max": numeric_sorted[-1],
            }
        )
    return summary


def crosswalk_candidates(
    rows: list[dict[str, str]], columns: list[str], manifest_sets: dict[str, set[str]]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for column in columns:
        raw_values = [str(row.get(column, "")).strip() for row in rows]
        values = {normalize(value) for value in raw_values if value}
        stems = {basename_stem(value) for value in raw_values if value}
        if not values:
            continue
        for manifest_field, target_values in manifest_sets.items():
            if manifest_field == "image_stem":
                overlap = stems & target_values
                n_source = len(stems)
            else:
                overlap = values & target_values
                n_source = len(values)
            if not overlap:
                continue
            results.append(
                {
                    "phenotype_column": column,
                    "manifest_field": manifest_field,
                    "phenotype_unique_values": n_source,
                    "manifest_unique_values": len(target_values),
                    "overlap_unique_values": len(overlap),
                    "phenotype_overlap_fraction": len(overlap) / n_source if n_source else 0.0,
                    "example_overlap": ";".join(sorted(overlap)[:8]),
                }
            )
    results.sort(
        key=lambda row: (
            -float(row["phenotype_overlap_fraction"]),
            -int(row["overlap_unique_values"]),
            str(row["phenotype_column"]),
        )
    )
    return results


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
        "--manifest",
        default=str(metadata_root / "pilot_v1" / "pilot_v1_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(metadata_root / "current_phenotypes"),
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--expected-manifest-sha256", default=FROZEN_MANIFEST_SHA256)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    raw_dir = output_dir / "files"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_sha = sha256_file(manifest_path)
    if args.expected_manifest_sha256 and manifest_sha != args.expected_manifest_sha256:
        raise RuntimeError(
            f"Frozen pilot manifest SHA256 mismatch: expected {args.expected_manifest_sha256}, got {manifest_sha}"
        )
    manifest_rows = read_manifest(manifest_path)
    manifest_sets = manifest_value_sets(manifest_rows)

    print("Radiation Edge AI - fetch/inspect current OSD-366 phenotype tables")
    print("Only three small text/CSV files are allowed; no microscopy archive is downloaded.")
    print(f"Frozen pilot manifest SHA256: {manifest_sha}")
    print("")

    listing = get_json(OSDR_FILES_URL)
    urls = discover_targets(listing)
    missing = [name for name in TARGET_FILES if name not in urls]
    if missing:
        raise RuntimeError(f"Target phenotype files missing from current OSDR listing: {missing}")

    downloaded: list[dict[str, Any]] = []
    table_summaries: list[dict[str, Any]] = []
    all_crosswalks: list[dict[str, Any]] = []

    for filename in TARGET_FILES:
        path = raw_dir / filename
        url = urls[filename]
        if path.is_file() and not args.refresh:
            payload = path.read_bytes()
            source = "reused"
        else:
            payload = get_bytes(url)
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(payload)
            tmp.replace(path)
            source = "downloaded"

        text = decode_text(payload)
        downloaded.append(
            {
                "filename": filename,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "source": source,
                "url": url,
            }
        )
        print(f"[File] {filename}")
        print(f"  {source}: {len(payload)} bytes")
        print(f"  SHA256: {sha256_bytes(payload)}")

        if filename.lower().endswith(".txt"):
            lines = [line.rstrip() for line in text.splitlines() if line.strip()]
            print("  description preview:")
            for line in lines[:20]:
                print(f"    {line[:240]}")
            table_summaries.append(
                {
                    "filename": filename,
                    "rows": len(lines),
                    "columns": "",
                    "foci_columns": "",
                    "identifier_columns": "",
                    "delimiter": "text",
                }
            )
            print("")
            continue

        rows, columns, delimiter = sniff_csv(text)
        foci_cols = candidate_columns(columns, FOCI_TERMS)
        id_cols = candidate_columns(columns, ID_TERMS)
        crosswalks = crosswalk_candidates(rows, columns, manifest_sets)
        for row in crosswalks:
            row["filename"] = filename
        all_crosswalks.extend(crosswalks)

        table_summaries.append(
            {
                "filename": filename,
                "rows": len(rows),
                "columns": ";".join(columns),
                "foci_columns": ";".join(foci_cols),
                "identifier_columns": ";".join(id_cols),
                "delimiter": repr(delimiter),
            }
        )

        print(f"  rows: {len(rows)}")
        print(f"  columns ({len(columns)}): {', '.join(columns)}")
        print(f"  foci/phenotype-like columns: {foci_cols or 'none'}")
        print(f"  identifier-like columns: {id_cols or 'none'}")
        for column in foci_cols[:12]:
            stats = summarize_values(rows, column)
            print(
                f"    {column}: nonempty={stats['nonempty']} unique={stats['unique']} "
                f"examples={stats['examples']}"
            )
        if crosswalks:
            print("  strongest manifest identifier overlaps:")
            for hit in crosswalks[:12]:
                print(
                    f"    {hit['phenotype_column']} -> {hit['manifest_field']}: "
                    f"overlap={hit['overlap_unique_values']} "
                    f"fraction={hit['phenotype_overlap_fraction']:.3f}"
                )
        else:
            print("  manifest identifier overlaps: none detected")
        print("")

    write_csv(
        output_dir / "download_inventory.csv",
        downloaded,
        ["filename", "bytes", "sha256", "source", "url"],
    )
    write_csv(
        output_dir / "table_summary.csv",
        table_summaries,
        ["filename", "rows", "columns", "foci_columns", "identifier_columns", "delimiter"],
    )
    write_csv(
        output_dir / "manifest_crosswalk_candidates.csv",
        all_crosswalks,
        [
            "filename",
            "phenotype_column",
            "manifest_field",
            "phenotype_unique_values",
            "manifest_unique_values",
            "overlap_unique_values",
            "phenotype_overlap_fraction",
            "example_overlap",
        ],
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "accession": ACCESSION,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "allowed_files": list(TARGET_FILES),
        "downloaded": downloaded,
        "tables": table_summaries,
        "crosswalk_candidate_count": len(all_crosswalks),
        "top_crosswalk_candidates": all_crosswalks[:50],
        "interpretation_guardrails": [
            "Presence of a foci-like column does not prove per-nucleus correspondence.",
            "Do not use phenotype values as training labels until an identifier crosswalk to the BPS nucleus crop or a defensible aggregate level is demonstrated.",
            "The frozen BPS pilot remains grouped by Source Name; any phenotype validation must preserve this biological grouping.",
            "No microscopy archive is downloaded by this script.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("OSD-366 CURRENT PHENOTYPE INSPECTION COMPLETE: YES")
    print(f"Files: {raw_dir}")
    print(f"Inventory: {output_dir / 'download_inventory.csv'}")
    print(f"Table summary: {output_dir / 'table_summary.csv'}")
    print(f"Crosswalk candidates: {output_dir / 'manifest_crosswalk_candidates.csv'}")
    print(f"Summary: {summary_path}")
    print("")
    print(
        "NEXT GATE: decide whether Raw/Processed phenotype rows map to individual BPS nuclei, "
        "fields/wells, or only aggregate samples. Freeze R1 only after that level is explicit."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
