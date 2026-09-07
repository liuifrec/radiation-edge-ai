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


def matching_direct_url(obj: Any) -> str | None:
    """Find a direct URL for the ISA ZIP in a heterogeneous API response."""
    # Prefer a URL living in the same record/dict as the target filename.
    for record in iter_dicts(obj):
        strings = [(str(key), value) for key, value in record.items() if isinstance(value, str)]
        if not any(ISA_NAME in value for _key, value in strings):
            continue
        urls = [
            value
            for key, value in strings
            if value.startswith("http")
            and ("url" in key.lower() or "download" in key.lower() or ISA_NAME in value)
        ]
        if urls:
            urls.sort(key=lambda value: (ISA_NAME not in value, len(value)))
            return urls[0]

    # Otherwise accept any direct-looking URL containing the target filename.
    candidates = []
    for key, value in iter_strings(obj):
        if value.startswith("http") and ISA_NAME in urllib.parse.unquote(value):
            candidates.append(("download" not in key.lower() and "url" not in key.lower(), len(value), value))
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
            if value.startswith("/"):
                return GENELAB_BASE + value, f"legacy GeneLab remote_url page {page}"
            if value.startswith("http"):
                return value, f"legacy GeneLab URL page {page}"

        # Stop when the response clearly contains no study files / no next page.
        text = json.dumps(payload)
        if ISA_NAME not in text and page > 0 and len(text) < 1000:
            break

    return None, "legacy GeneLab discovery failed" + ("; " + "; ".join(notes) if notes else "")


def download_zip(url: str, destination: Path) -> None:
    payload = get_bytes(url, timeout=120)
    if len(payload) < 100 or not payload.startswith(b"PK"):
        head = payload[:80]
        raise RuntimeError(
            "NASA URL did not return a ZIP archive. "
            f"received {len(payload)} bytes; first bytes={head!r}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    default_root = Path(
        os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data")
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
    download_zip(direct_url, destination)
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
