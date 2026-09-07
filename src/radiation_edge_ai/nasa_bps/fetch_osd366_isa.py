"""Fetch the OSD-366 ISA metadata ZIP through NASA's current public APIs.

This helper exists because historical GeneLab download URLs can move even while
OSD-366 remains public. It first asks the current OSDR Biological Data API for
the dataset file listing, which is the documented discovery route for direct
file download URLs. If that fails, it falls back to the legacy GeneLab file
listing API and extracts a matching remote URL.

Only the small OSD-366 ISA metadata ZIP is downloaded. No microscopy images are
fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ACCESSION = "OSD-366"
ISA_NAME = "OSD-366_metadata_OSD-366-ISA.zip"
OSDR_FILES_URL = (
    "https://visualization.osdr.nasa.gov/biodata/api/v2/dataset/OSD-366/files/"
)
LEGACY_FILES_URL = (
    "https://genelab-data.ndc.nasa.gov/genelab/data/glds/files/366?page={page}&size=25"
)
GENELAB_BASE = "https://genelab-data.ndc.nasa.gov"
OSDR_DOWNLOAD_BASE = "https://osdr.nasa.gov"
USER_AGENT = "radiation-edge-ai/0.1 OSD-366-ISA-fetch"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_bytes(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def get_json(url: str) -> Any:
    return json.loads(get_bytes(url).decode("utf-8"))


def iter_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from iter_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_dicts(value)


def iter_strings(obj: Any):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                yield str(key), value
            else:
                yield from iter_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_strings(value)


def normalize_download_candidate(key: str, value: str) -> str | None:
    """Normalize one OSDR/GeneLab URL field into a downloadable URL.

    Current OSDR file records expose both:
      REST_URL -> JSON metadata for the file record
      URL      -> actual downloadable file

    The REST_URL must therefore never be selected as the binary download URL.
    """
    key_lower = key.lower()
    if key_lower == "rest_url" or key_lower.endswith("rest_url"):
        return None
    if value.startswith("http"):
        return value
    if value.startswith("/geode-py/"):
        return OSDR_DOWNLOAD_BASE + value
    if value.startswith("/datamanager/"):
        return GENELAB_BASE + value
    return None


def matching_direct_url(obj: Any) -> str | None:
    """Find the actual downloadable URL for the ISA ZIP.

    Prefer the OSDR record's exact ``URL`` field. ``REST_URL`` is an API JSON
    endpoint and deliberately excluded.
    """
    for record in iter_dicts(obj):
        strings = [(str(key), value) for key, value in record.items() if isinstance(value, str)]
        mentions_target = any(ISA_NAME in urllib.parse.unquote(value) for _key, value in strings)
        if not mentions_target:
            continue

        # OSDR's current API uses an exact URL field for the downloadable file.
        for key, value in strings:
            if key.lower() == "url":
                candidate = normalize_download_candidate(key, value)
                if candidate and ISA_NAME in urllib.parse.unquote(candidate):
                    return candidate

        # Then prefer explicit download/remote URL fields.
        ranked: list[tuple[int, int, str]] = []
        for key, value in strings:
            candidate = normalize_download_candidate(key, value)
            if not candidate or ISA_NAME not in urllib.parse.unquote(candidate):
                continue
            key_lower = key.lower()
            priority = 0 if "download" in key_lower else 1 if "remote" in key_lower else 2
            ranked.append((priority, len(candidate), candidate))
        if ranked:
            ranked.sort()
            return ranked[0][2]

    # Fallback: scan the whole response, still excluding REST_URL.
    candidates: list[tuple[int, int, str]] = []
    for key, value in iter_strings(obj):
        candidate = normalize_download_candidate(key, value)
        if candidate is None or ISA_NAME not in urllib.parse.unquote(candidate):
            continue
        key_lower = key.lower()
        priority = 0 if key_lower == "url" else 1 if "download" in key_lower else 2
        candidates.append((priority, len(candidate), candidate))
    if candidates:
        candidates.sort()
        return candidates[0][2]
    return None


def discover_from_osdr() -> tuple[str | None, str]:
    try:
        payload = get_json(OSDR_FILES_URL)
    except Exception as exc:
        return None, f"OSDR file listing failed: {type(exc).__name__}: {exc}"
    url = matching_direct_url(payload)
    if url:
        return url, "OSDR Biological Data API dataset file listing"
    return None, "OSDR file listing returned JSON but target ISA ZIP/direct URL was not found"


def discover_from_legacy() -> tuple[str | None, str]:
    notes = []
    for page in range(0, 50):
        url = LEGACY_FILES_URL.format(page=page)
        try:
            payload = get_json(url)
        except Exception as exc:
            notes.append(f"page {page}: {type(exc).__name__}: {exc}")
            break

        direct = matching_direct_url(payload)
        if direct:
            return direct, f"legacy GeneLab file listing page {page}"

        # Historical responses often expose a path such as
        # /datamanager/file/Home/... in a field named remote_url.
        for key, value in iter_strings(payload):
            if ISA_NAME not in value:
                continue
            candidate = normalize_download_candidate(key, value)
            if candidate:
                return candidate, f"legacy GeneLab URL page {page}"

        text = json.dumps(payload)
        if ISA_NAME not in text and page > 0 and len(text) < 1000:
            break

    return None, "legacy GeneLab discovery failed" + ("; " + "; ".join(notes) if notes else "")


def resolve_json_file_record(payload: bytes, original_url: str) -> str | None:
    """Recover if an OSDR REST metadata endpoint was passed accidentally."""
    try:
        obj = json.loads(payload.decode("utf-8"))
    except Exception:
        return None
    candidate = matching_direct_url(obj)
    if candidate and candidate.rstrip("/") != original_url.rstrip("/"):
        return candidate
    return None


def download_zip(url: str, destination: Path) -> str:
    payload = get_bytes(url, timeout=120)
    effective_url = url

    if not payload.startswith(b"PK"):
        recovered = resolve_json_file_record(payload, url)
        if recovered:
            print(f"[REST metadata] resolved direct file URL: {recovered}")
            effective_url = recovered
            payload = get_bytes(recovered, timeout=120)

    if len(payload) < 100 or not payload.startswith(b"PK"):
        head = payload[:80]
        raise RuntimeError(
            "NASA URL did not return a ZIP archive. "
            f"received {len(payload)} bytes; first bytes={head!r}; url={effective_url}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return effective_url


def main() -> int:
    parser = argparse.ArgumentParser()
    default_root = Path(
        os.environ.get("RADEDGE_DATA_ROOT", r"D:\\radiation-edge-ai-data\\data")
    ) / "nasa_bps_microscopy" / "metadata" / "osd366"
    parser.add_argument("--output", default=str(default_root / ISA_NAME))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    destination = Path(args.output).resolve()
    if destination.is_file() and not args.refresh:
        print("NASA OSD-366 ISA metadata already present")
        print(f"Path: {destination}")
        print(f"SHA256: {sha256_file(destination)}")
        return 0

    print("Radiation Edge AI - fetch OSD-366 ISA metadata")
    print("Metadata only: no microscopy images will be downloaded.")
    print(f"Target: {destination}")
    print("")

    direct_url, source = discover_from_osdr()
    if direct_url is None:
        print(f"[OSDR API] {source}")
        direct_url, source = discover_from_legacy()
    else:
        print("[OSDR API] target ISA record found")

    if direct_url is None:
        raise RuntimeError(
            "Could not discover a direct NASA download URL for "
            f"{ISA_NAME}. Current OSDR and legacy GeneLab file listings were tried."
        )

    print(f"[Discovery] {source}")
    print(f"[Download] {direct_url}")
    effective_url = download_zip(direct_url, destination)
    if effective_url != direct_url:
        print(f"[Effective download] {effective_url}")
    print("")
    print("OSD-366 ISA FETCH COMPLETE: YES")
    print(f"Path: {destination}")
    print(f"Bytes: {destination.stat().st_size}")
    print(f"SHA256: {sha256_file(destination)}")
    print("")
    print("NEXT: rerun resolve_osd366_groups.py; it will reuse this local ZIP.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
