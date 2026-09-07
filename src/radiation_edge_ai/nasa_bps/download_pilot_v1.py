"""Download only the frozen NASA BPS 53BP1 pilot-v1 image triplets.

The pilot manifest is frozen independently by ``freeze_pilot_v1.py``. This
utility consumes that manifest and downloads only its FITC/53BP1, DAPI and MASK
TIFF files from the public NASA BPS AWS Open Data bucket.

The downloader is intentionally conservative:
- the frozen manifest SHA256 is checked before any network transfer;
- downloads are resumable and written atomically via ``.part`` files;
- every local TIFF is checked for a TIFF magic header;
- SHA256 and byte size are recorded for every downloaded file;
- raw images remain under RADEDGE_DATA_ROOT and are never committed.

No training or model inference happens here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

FROZEN_MANIFEST_SHA256 = "41ff2e4d64577f17c976dea5bbc17221fe63be9e2e7e2291309b520027fbe725"
FITC_BASE_URL = (
    "https://nasa-bps-training-data.s3.us-west-2.amazonaws.com/Microscopy/train/"
)
DAPI_MASK_BASE_URL = (
    "https://nasa-bps-training-data.s3.us-west-2.amazonaws.com/Microscopy/DAPI_MASK_images/"
)
USER_AGENT = "radiation-edge-ai/0.1 NASA-BPS-pilot-v1"
TIFF_MAGICS = (b"II*\x00", b"MM\x00*")
CHANNEL_ORDER = {"fitc": 0, "dapi": 1, "mask": 2}


@dataclass(frozen=True)
class DownloadTask:
    sample_id: str
    channel: str
    filename: str
    url: str
    path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_tiff(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 8:
        return False
    with path.open("rb") as handle:
        return handle.read(4) in TIFF_MAGICS


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError(f"Pilot manifest is empty: {path}")
    required = {"sample_id", "fitc_filename", "dapi_filename", "mask_filename"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise RuntimeError(f"Pilot manifest missing columns: {missing}")
    return rows


def object_url(base: str, filename: str) -> str:
    return base + urllib.parse.quote(filename, safe="-_.()")


def build_tasks(rows: list[dict[str, str]], output_dir: Path) -> list[DownloadTask]:
    tasks: list[DownloadTask] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        sample_id = row["sample_id"]
        specs = (
            ("fitc", row["fitc_filename"], FITC_BASE_URL),
            ("dapi", row["dapi_filename"], DAPI_MASK_BASE_URL),
            ("mask", row["mask_filename"], DAPI_MASK_BASE_URL),
        )
        for channel, filename, base in specs:
            key = (channel, filename)
            if key in seen:
                raise RuntimeError(f"Duplicate pilot file task: {channel} {filename}")
            seen.add(key)
            tasks.append(
                DownloadTask(
                    sample_id=sample_id,
                    channel=channel,
                    filename=filename,
                    url=object_url(base, filename),
                    path=output_dir / channel / filename,
                )
            )
    return tasks


def inspect_existing(task: DownloadTask) -> dict[str, object] | None:
    if not is_tiff(task.path):
        return None
    return {
        "sample_id": task.sample_id,
        "channel": task.channel,
        "filename": task.filename,
        "url": task.url,
        "local_path": str(task.path),
        "status": "existing",
        "bytes": task.path.stat().st_size,
        "sha256": sha256_file(task.path),
        "error": "",
    }


def download_once(task: DownloadTask, timeout: int) -> tuple[int, str]:
    task.path.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".part.{os.getpid()}.{threading.get_ident()}"
    temporary = task.path.with_name(task.path.name + suffix)
    request = urllib.request.Request(task.url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    n_bytes = 0
    first4 = b""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                if len(first4) < 4:
                    first4 = (first4 + chunk)[:4]
                out.write(chunk)
                digest.update(chunk)
                n_bytes += len(chunk)
        if first4 not in TIFF_MAGICS:
            raise RuntimeError(
                f"NASA object was not a TIFF: received {n_bytes} bytes, first4={first4!r}"
            )
        if n_bytes < 8:
            raise RuntimeError(f"NASA object is unexpectedly small: {n_bytes} bytes")
        os.replace(temporary, task.path)
        return n_bytes, digest.hexdigest()
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def run_task(task: DownloadTask, retries: int, timeout: int, overwrite: bool) -> dict[str, object]:
    if not overwrite:
        existing = inspect_existing(task)
        if existing is not None:
            return existing

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            n_bytes, digest = download_once(task, timeout)
            return {
                "sample_id": task.sample_id,
                "channel": task.channel,
                "filename": task.filename,
                "url": task.url,
                "local_path": str(task.path),
                "status": "downloaded",
                "bytes": n_bytes,
                "sha256": digest,
                "error": "",
            }
        except Exception as exc:  # network errors are recorded per object
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 4))

    return {
        "sample_id": task.sample_id,
        "channel": task.channel,
        "filename": task.filename,
        "url": task.url,
        "local_path": str(task.path),
        "status": "failed",
        "bytes": 0,
        "sha256": "",
        "error": last_error,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    default_data_root = Path(
        os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data")
    )
    default_metadata = default_data_root / "nasa_bps_microscopy" / "metadata" / "pilot_v1"
    default_output = default_data_root / "nasa_bps_microscopy" / "pilot_v1" / "images"
    parser.add_argument(
        "--manifest",
        default=str(default_metadata / "pilot_v1_manifest.csv"),
    )
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--limit-nuclei", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-manifest-sha256", default=FROZEN_MANIFEST_SHA256)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    actual_manifest_sha = sha256_file(manifest_path)
    if args.expected_manifest_sha256 and actual_manifest_sha != args.expected_manifest_sha256:
        raise RuntimeError(
            "Frozen NASA BPS pilot-v1 manifest SHA256 mismatch: "
            f"expected {args.expected_manifest_sha256}, got {actual_manifest_sha}"
        )

    rows = read_manifest(manifest_path)
    if args.limit_nuclei is not None:
        if args.limit_nuclei <= 0:
            raise ValueError("--limit-nuclei must be positive")
        rows = rows[: args.limit_nuclei]
    tasks = build_tasks(rows, output_dir)

    print("Radiation Edge AI - download NASA BPS 53BP1 pilot v1")
    print(f"Manifest: {manifest_path}")
    print(f"Manifest SHA256: {actual_manifest_sha}")
    print(f"Selected nuclei: {len(rows)}")
    print(f"Expected TIFF files: {len(tasks)}")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.workers}")
    print("")

    results: list[dict[str, object]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(run_task, task, args.retries, args.timeout, args.overwrite): task
            for task in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed == 1 or completed % 100 == 0 or completed == len(tasks):
                n_failed = sum(r["status"] == "failed" for r in results)
                print(f"[{completed:>5}/{len(tasks)}] failures={n_failed}")

    results.sort(
        key=lambda row: (
            row["sample_id"],
            CHANNEL_ORDER.get(str(row["channel"]), 99),
            row["filename"],
        )
    )
    inventory_path = output_dir.parent / "download_inventory.csv"
    write_csv(inventory_path, results)

    successful = [row for row in results if row["status"] != "failed"]
    failed = [row for row in results if row["status"] == "failed"]
    total_bytes = sum(int(row["bytes"]) for row in successful)
    counts = {
        channel: sum(
            row["channel"] == channel and row["status"] != "failed" for row in results
        )
        for channel in ("fitc", "dapi", "mask")
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_version": "NASA_BPS_53BP1_PILOT_V1",
        "manifest": str(manifest_path),
        "manifest_sha256": actual_manifest_sha,
        "n_nuclei_requested": len(rows),
        "n_files_expected": len(tasks),
        "n_files_successful": len(successful),
        "n_files_failed": len(failed),
        "successful_by_channel": counts,
        "total_bytes": total_bytes,
        "output_dir": str(output_dir),
        "download_inventory": str(inventory_path),
        "download_inventory_sha256": sha256_file(inventory_path),
        "fitc_base_url": FITC_BASE_URL,
        "dapi_mask_base_url": DAPI_MASK_BASE_URL,
        "failures": [
            {"sample_id": r["sample_id"], "channel": r["channel"], "filename": r["filename"], "error": r["error"]}
            for r in failed[:100]
        ],
    }
    summary_path = output_dir.parent / "download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("[Download summary]")
    print(f"successful files: {len(successful)}/{len(tasks)}")
    print(f"FITC: {counts['fitc']}/{len(rows)}")
    print(f"DAPI: {counts['dapi']}/{len(rows)}")
    print(f"MASK: {counts['mask']}/{len(rows)}")
    print(f"total bytes: {total_bytes:,}")
    print(f"inventory: {inventory_path}")
    print(f"summary: {summary_path}")
    print("")

    if failed:
        print("NASA BPS PILOT V1 DOWNLOAD COMPLETE: NO")
        print("Rerun the same command; successful files are resumable and will be reused.")
        for row in failed[:10]:
            print(f"FAILED {row['channel']} {row['filename']}: {row['error']}")
        return 1

    print("NASA BPS PILOT V1 DOWNLOAD COMPLETE: YES")
    print("NEXT GATE: inspect TIFF dimensions, bit depth, MASK pairing, and intensity distributions before modeling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
