"""Download only the frozen NASA BPS R1 v2 DEVELOPMENT image triplets.

This utility consumes the frozen 7,200-nucleus R1 v2 development manifest and
materializes a self-contained development image tree.  The final blinded holdout
manifest is deliberately not read.

Important behavior:
- validate the frozen development manifest SHA256 before any transfer;
- validate the frozen pilot-v1 manifest SHA256;
- assert that all 2,160 pilot-v1 nuclei are preserved in the 7,200-nucleus set;
- reuse already downloaded pilot-v1 FITC/DAPI/MASK TIFFs by NTFS hard link when
  possible, falling back to local copy if hard-link creation is unavailable;
- download only files not already reusable from pilot v1;
- resumable atomic network downloads via temporary .part files;
- verify TIFF magic headers and record byte size + SHA256 for every file;
- never read, materialize or inspect the final R1 v2 holdout cohort.

No phenotype tables are read and no model is trained here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FROZEN_DEV_MANIFEST_SHA256 = "029d156250f1b73a6cc2b556b88a8e3b6900366a6a2b36244b699d2d9c43ab01"
FROZEN_V1_MANIFEST_SHA256 = "41ff2e4d64577f17c976dea5bbc17221fe63be9e2e7e2291309b520027fbe725"
EXPECTED_DEV_NUCLEI = 7200
EXPECTED_V1_NUCLEI = 2160
EXPECTED_NEW_NUCLEI = EXPECTED_DEV_NUCLEI - EXPECTED_V1_NUCLEI
EXPECTED_FILES = EXPECTED_DEV_NUCLEI * 3
EXPECTED_REUSED_FILES = EXPECTED_V1_NUCLEI * 3
EXPECTED_NETWORK_FILES = EXPECTED_NEW_NUCLEI * 3

DEV_SOURCES = {
    "BALBCF1",
    "BALBCM1",
    "BALBCM2",
    "C57BLF1",
    "C57BLM1",
    "C57BLM3",
}
FINAL_HOLDOUT_SOURCES = {"BALBCF2", "C57BLF2", "C57BLF3"}

FITC_BASE_URL = "https://nasa-bps-training-data.s3.us-west-2.amazonaws.com/Microscopy/train/"
DAPI_MASK_BASE_URL = "https://nasa-bps-training-data.s3.us-west-2.amazonaws.com/Microscopy/DAPI_MASK_images/"
USER_AGENT = "radiation-edge-ai/0.1 NASA-BPS-R1-v2-development"
TIFF_MAGICS = (b"II*\x00", b"MM\x00*")
CHANNEL_ORDER = {"fitc": 0, "dapi": 1, "mask": 2}


@dataclass(frozen=True)
class DownloadTask:
    sample_id: str
    nucleus_key: str
    channel: str
    filename: str
    url: str
    target_path: Path
    v1_source_path: Path | None


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


def object_url(base: str, filename: str) -> str:
    return base + urllib.parse.quote(filename, safe="-_.()")


def validate_manifests(
    dev_rows: list[dict[str, str]],
    v1_rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    required_dev = {
        "sample_id",
        "nucleus_key",
        "source_name",
        "fitc_filename",
        "dapi_filename",
        "mask_filename",
        "v1_member",
        "v1_sample_id",
    }
    required_v1 = {
        "sample_id",
        "nucleus_key",
        "source_name",
        "fitc_filename",
        "dapi_filename",
        "mask_filename",
    }
    missing = sorted(required_dev - set(dev_rows[0]))
    if missing:
        raise RuntimeError(f"Development manifest missing columns: {missing}")
    missing = sorted(required_v1 - set(v1_rows[0]))
    if missing:
        raise RuntimeError(f"Pilot-v1 manifest missing columns: {missing}")

    if len(dev_rows) != EXPECTED_DEV_NUCLEI:
        raise RuntimeError(f"Development nuclei={len(dev_rows)}; expected {EXPECTED_DEV_NUCLEI}")
    if len(v1_rows) != EXPECTED_V1_NUCLEI:
        raise RuntimeError(f"Pilot-v1 nuclei={len(v1_rows)}; expected {EXPECTED_V1_NUCLEI}")

    sources = {row["source_name"] for row in dev_rows}
    if sources != DEV_SOURCES:
        raise RuntimeError(f"Unexpected development Source Names: {sorted(sources)}")
    if sources & FINAL_HOLDOUT_SOURCES:
        raise RuntimeError("Final holdout Source Name leaked into development manifest")

    dev_by_key = {row["nucleus_key"]: row for row in dev_rows}
    v1_by_key = {row["nucleus_key"]: row for row in v1_rows}
    if len(dev_by_key) != len(dev_rows):
        raise RuntimeError("Development manifest contains duplicate nucleus_key values")
    if len(v1_by_key) != len(v1_rows):
        raise RuntimeError("Pilot-v1 manifest contains duplicate nucleus_key values")

    missing_v1 = sorted(set(v1_by_key) - set(dev_by_key))
    if missing_v1:
        raise RuntimeError(f"Frozen development manifest lost {len(missing_v1)} v1 nuclei")

    claimed_v1 = [row for row in dev_rows if str(row["v1_member"]).strip() == "1"]
    if len(claimed_v1) != EXPECTED_V1_NUCLEI:
        raise RuntimeError(
            f"Development v1_member rows={len(claimed_v1)}; expected {EXPECTED_V1_NUCLEI}"
        )

    for row in claimed_v1:
        key = row["nucleus_key"]
        old = v1_by_key.get(key)
        if old is None:
            raise RuntimeError(f"v1_member nucleus absent from frozen pilot manifest: {key}")
        if row["v1_sample_id"] != old["sample_id"]:
            raise RuntimeError(
                f"v1_sample_id mismatch for {key}: dev={row['v1_sample_id']} old={old['sample_id']}"
            )
        for field in ("fitc_filename", "dapi_filename", "mask_filename"):
            if row[field] != old[field]:
                raise RuntimeError(
                    f"Frozen filename mismatch for {key} {field}: dev={row[field]} old={old[field]}"
                )

    return v1_by_key


def build_tasks(
    dev_rows: list[dict[str, str]],
    v1_by_key: dict[str, dict[str, str]],
    output_dir: Path,
    pilot_image_root: Path,
) -> list[DownloadTask]:
    tasks: list[DownloadTask] = []
    seen: set[tuple[str, str]] = set()
    for row in dev_rows:
        key = row["nucleus_key"]
        old = v1_by_key.get(key)
        specs = (
            ("fitc", row["fitc_filename"], FITC_BASE_URL),
            ("dapi", row["dapi_filename"], DAPI_MASK_BASE_URL),
            ("mask", row["mask_filename"], DAPI_MASK_BASE_URL),
        )
        for channel, filename, base in specs:
            file_key = (channel, filename)
            if file_key in seen:
                raise RuntimeError(f"Duplicate file task: {channel} {filename}")
            seen.add(file_key)
            source_path = None
            if old is not None:
                source_path = pilot_image_root / channel / filename
            tasks.append(
                DownloadTask(
                    sample_id=row["sample_id"],
                    nucleus_key=key,
                    channel=channel,
                    filename=filename,
                    url=object_url(base, filename),
                    target_path=output_dir / channel / filename,
                    v1_source_path=source_path,
                )
            )
    if len(tasks) != EXPECTED_FILES:
        raise RuntimeError(f"Development file tasks={len(tasks)}; expected {EXPECTED_FILES}")
    return tasks


def result_row(
    task: DownloadTask,
    *,
    status: str,
    n_bytes: int,
    digest: str,
    error: str = "",
) -> dict[str, Any]:
    return {
        "sample_id": task.sample_id,
        "nucleus_key": task.nucleus_key,
        "channel": task.channel,
        "filename": task.filename,
        "url": task.url,
        "local_path": str(task.target_path),
        "v1_source_path": str(task.v1_source_path) if task.v1_source_path else "",
        "status": status,
        "bytes": n_bytes,
        "sha256": digest,
        "error": error,
    }


def reuse_v1(task: DownloadTask) -> dict[str, Any] | None:
    source = task.v1_source_path
    if source is None or not is_tiff(source):
        return None
    task.target_path.parent.mkdir(parents=True, exist_ok=True)
    if task.target_path.exists():
        task.target_path.unlink()
    try:
        os.link(source, task.target_path)
        status = "linked_v1"
    except OSError:
        shutil.copy2(source, task.target_path)
        status = "copied_v1"
    if not is_tiff(task.target_path):
        raise RuntimeError(f"Local v1 reuse produced invalid TIFF: {task.target_path}")
    return result_row(
        task,
        status=status,
        n_bytes=task.target_path.stat().st_size,
        digest=sha256_file(task.target_path),
    )


def download_once(task: DownloadTask, timeout: int) -> tuple[int, str]:
    task.target_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".part.{os.getpid()}.{threading.get_ident()}"
    temporary = task.target_path.with_name(task.target_path.name + suffix)
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
        if first4 not in TIFF_MAGICS or n_bytes < 8:
            raise RuntimeError(
                f"NASA object is not a valid TIFF: bytes={n_bytes} first4={first4!r}"
            )
        os.replace(temporary, task.target_path)
        return n_bytes, digest.hexdigest()
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def run_task(
    task: DownloadTask,
    *,
    retries: int,
    timeout: int,
    overwrite: bool,
) -> dict[str, Any]:
    if not overwrite and is_tiff(task.target_path):
        return result_row(
            task,
            status="existing",
            n_bytes=task.target_path.stat().st_size,
            digest=sha256_file(task.target_path),
        )

    if task.v1_source_path is not None:
        reused = reuse_v1(task)
        if reused is not None:
            return reused

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            n_bytes, digest = download_once(task, timeout)
            status = "downloaded_repair_v1" if task.v1_source_path is not None else "downloaded_new"
            return result_row(
                task,
                status=status,
                n_bytes=n_bytes,
                digest=digest,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 4))

    return result_row(task, status="failed", n_bytes=0, digest="", error=last_error)


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"
    parser.add_argument(
        "--development-manifest",
        default=str(metadata_root / "r1_v2_freeze" / "r1_v2_development_manifest_100.csv"),
    )
    parser.add_argument(
        "--v1-manifest",
        default=str(metadata_root / "pilot_v1" / "pilot_v1_manifest.csv"),
    )
    parser.add_argument(
        "--pilot-image-root",
        default=str(data_root / "nasa_bps_microscopy" / "pilot_v1" / "images"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(data_root / "nasa_bps_microscopy" / "r1_v2_development" / "images"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dev_manifest_path = Path(args.development_manifest).resolve()
    v1_manifest_path = Path(args.v1_manifest).resolve()
    pilot_image_root = Path(args.pilot_image_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    for path in (dev_manifest_path, v1_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not pilot_image_root.is_dir():
        raise FileNotFoundError(pilot_image_root)

    dev_sha = sha256_file(dev_manifest_path)
    v1_sha = sha256_file(v1_manifest_path)
    if dev_sha != FROZEN_DEV_MANIFEST_SHA256:
        raise RuntimeError(
            f"Frozen R1 v2 development manifest SHA256 mismatch: expected {FROZEN_DEV_MANIFEST_SHA256}, got {dev_sha}"
        )
    if v1_sha != FROZEN_V1_MANIFEST_SHA256:
        raise RuntimeError(
            f"Frozen pilot-v1 manifest SHA256 mismatch: expected {FROZEN_V1_MANIFEST_SHA256}, got {v1_sha}"
        )

    dev_rows = read_csv(dev_manifest_path)
    v1_rows = read_csv(v1_manifest_path)
    v1_by_key = validate_manifests(dev_rows, v1_rows)
    tasks = build_tasks(dev_rows, v1_by_key, output_dir, pilot_image_root)

    n_reuse_tasks = sum(task.v1_source_path is not None for task in tasks)
    n_network_tasks = len(tasks) - n_reuse_tasks
    if n_reuse_tasks != EXPECTED_REUSED_FILES:
        raise RuntimeError(
            f"Reusable v1 file tasks={n_reuse_tasks}; expected {EXPECTED_REUSED_FILES}"
        )
    if n_network_tasks != EXPECTED_NETWORK_FILES:
        raise RuntimeError(
            f"New network file tasks={n_network_tasks}; expected {EXPECTED_NETWORK_FILES}"
        )

    print("Radiation Edge AI - download NASA BPS R1 v2 DEVELOPMENT images")
    print("Final holdout remains untouched: this script does not read its manifest.")
    print(f"Development manifest SHA256: {dev_sha}")
    print(f"Development nuclei: {len(dev_rows)}")
    print(f"Pilot-v1 nuclei reused: {EXPECTED_V1_NUCLEI}")
    print(f"Additional development nuclei: {EXPECTED_NEW_NUCLEI}")
    print(f"Expected TIFF files: {len(tasks)}")
    print(f"Reusable local v1 files: {n_reuse_tasks}")
    print(f"Network files if local reuse is complete: {n_network_tasks}")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.workers}")
    print("")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                run_task,
                task,
                retries=max(1, args.retries),
                timeout=max(1, args.timeout),
                overwrite=args.overwrite,
            ): task
            for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if completed == 1 or completed % 250 == 0 or completed == len(tasks):
                failures = sum(row["status"] == "failed" for row in results)
                print(f"[{completed:>5}/{len(tasks)}] failures={failures}")

    results.sort(
        key=lambda row: (
            row["sample_id"],
            CHANNEL_ORDER.get(str(row["channel"]), 99),
            row["filename"],
        )
    )

    output_parent = output_dir.parent
    inventory_path = output_parent / "download_inventory.csv"
    write_csv(inventory_path, results)

    failures = [row for row in results if row["status"] == "failed"]
    counts = Counter(str(row["status"]) for row in results)
    channel_ok = {
        channel: sum(
            row["channel"] == channel and row["status"] != "failed" for row in results
        )
        for channel in ("fitc", "dapi", "mask")
    }
    total_bytes_logical = sum(int(row["bytes"]) for row in results if row["status"] != "failed")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "NASA_BPS_R1_V2_DEVELOPMENT_ONLY",
        "final_holdout_touched": False,
        "development_manifest": str(dev_manifest_path),
        "development_manifest_sha256": dev_sha,
        "v1_manifest": str(v1_manifest_path),
        "v1_manifest_sha256": v1_sha,
        "n_development_nuclei": len(dev_rows),
        "n_v1_nuclei_reused": EXPECTED_V1_NUCLEI,
        "n_additional_development_nuclei": EXPECTED_NEW_NUCLEI,
        "n_files_expected": len(tasks),
        "n_files_successful": len(results) - len(failures),
        "n_files_failed": len(failures),
        "status_counts": dict(counts),
        "successful_by_channel": channel_ok,
        "logical_total_bytes": total_bytes_logical,
        "output_dir": str(output_dir),
        "download_inventory": str(inventory_path),
        "download_inventory_sha256": sha256_file(inventory_path),
        "notes": [
            "Pilot-v1 files are hard-linked into the R1 v2 development tree where possible; local copy is fallback only.",
            "Only additional development files are expected to require network transfer when the frozen pilot-v1 tree is intact.",
            "The final blinded holdout manifest, holdout images and holdout phenotypes are not read by this script.",
        ],
        "failures": [
            {
                "sample_id": row["sample_id"],
                "channel": row["channel"],
                "filename": row["filename"],
                "error": row["error"],
            }
            for row in failures[:100]
        ],
    }
    summary_path = output_parent / "download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("[Download/materialization summary]")
    print(f"successful files: {len(results) - len(failures)}/{len(tasks)}")
    print(f"FITC: {channel_ok['fitc']}/{len(dev_rows)}")
    print(f"DAPI: {channel_ok['dapi']}/{len(dev_rows)}")
    print(f"MASK: {channel_ok['mask']}/{len(dev_rows)}")
    print(f"status counts: {dict(counts)}")
    print(f"inventory: {inventory_path}")
    print(f"summary: {summary_path}")
    print("")

    if failures:
        print("NASA BPS R1 V2 DEVELOPMENT DOWNLOAD COMPLETE: NO")
        print("Rerun the same command; completed files are reused.")
        for row in failures[:10]:
            print(f"FAILED {row['channel']} {row['filename']}: {row['error']}")
        return 1

    print("NASA BPS R1 V2 DEVELOPMENT DOWNLOAD COMPLETE: YES")
    print("FINAL HOLDOUT STATUS: UNTOUCHED")
    print(
        "NEXT GATE: validate all 7,200 development FITC/DAPI/MASK triplets and freeze the R1 v2 contrast-aware training policy before any holdout access."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
