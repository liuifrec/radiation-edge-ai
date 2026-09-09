"""Materialize the frozen NASA BPS R1 v2 FINAL HOLDOUT images phenotype-blind.

This script is the first permitted final-holdout access after the all-development
FP32 candidate and its development-only BatchNorm repair were frozen.

It deliberately reads only:
- the frozen blinded final-holdout manifest;
- the already frozen repaired FP32 checkpoint identity (SHA256 only);
- public NASA BPS image objects named by the manifest.

It does NOT read any NASA phenotype table, avg_nfoci values, development
reference table, model predictions, or final-holdout biological outcomes.

Behavior:
- pin and verify the repaired FP32 checkpoint SHA256 before any image transfer;
- pin and verify the blinded holdout-manifest SHA256;
- reject manifests containing phenotype/reference/target columns;
- assert the exact 2,058-nucleus / 22-bag / three-source blinded cohort;
- download FITC/DAPI/MASK TIFFs exactly as frozen in the manifest;
- use atomic .part downloads and resume from already valid TIFFs;
- verify TIFF magic, byte size and SHA256 for every materialized file;
- write only transfer provenance, never phenotype values.

The model checkpoint is not executed here. MASK remains QC-only and will be
used by the next phenotype-blind QC gate.
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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FROZEN_REPAIRED_CHECKPOINT_SHA256 = (
    "2b07c67d6e50e7a36e26e0052feebc826b518e6efae863ba312ba9761a4eb3d5"
)
FROZEN_BLINDED_HOLDOUT_MANIFEST_SHA256 = (
    "40f015f7f8a8acb71289c1fe91218253e0b518c7e98f920511a5792dd5b6862d"
)
EXPECTED_NUCLEI = 2058
EXPECTED_BAGS = 22
EXPECTED_FILES = EXPECTED_NUCLEI * 3
EXPECTED_SOURCE_NUCLEI = {
    "BALBCF2": 858,
    "C57BLF2": 600,
    "C57BLF3": 600,
}
EXPECTED_HOLDOUT_STATUS = "FINAL_BLINDED_DO_NOT_JOIN_PHENOTYPES"

# Reject any manifest that accidentally carries outcome/reference information.
FORBIDDEN_COLUMN_TOKENS = (
    "nfoci",
    "phenotype",
    "reference",
    "target_",
    "ground_truth",
    "prediction",
    "predicted",
)

FITC_BASE_URL = "https://nasa-bps-training-data.s3.us-west-2.amazonaws.com/Microscopy/train/"
DAPI_MASK_BASE_URL = "https://nasa-bps-training-data.s3.us-west-2.amazonaws.com/Microscopy/DAPI_MASK_images/"
USER_AGENT = "radiation-edge-ai/0.1 NASA-BPS-R1-v2-final-holdout-blinded"
TIFF_MAGICS = (b"II*\x00", b"MM\x00*")
CHANNEL_ORDER = {"fitc": 0, "dapi": 1, "mask": 2}


@dataclass(frozen=True)
class DownloadTask:
    sample_id: str
    nucleus_key: str
    source_name: str
    sample_name: str
    channel: str
    filename: str
    url: str
    target_path: Path


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


def validate_blinded_manifest(rows: list[dict[str, str]]) -> None:
    required = {
        "sample_id",
        "nucleus_key",
        "source_name",
        "sample_name",
        "strain",
        "sex",
        "particle_type",
        "dose_Gy",
        "hr_post_exposure",
        "fitc_filename",
        "dapi_filename",
        "mask_filename",
        "holdout_status",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise RuntimeError(f"Blinded holdout manifest missing columns: {missing}")

    forbidden = sorted(
        column
        for column in rows[0]
        if any(token in column.lower() for token in FORBIDDEN_COLUMN_TOKENS)
    )
    if forbidden:
        raise RuntimeError(
            "Blinded holdout manifest contains forbidden outcome/reference columns: "
            + ", ".join(forbidden)
        )

    if len(rows) != EXPECTED_NUCLEI:
        raise RuntimeError(f"Holdout nuclei={len(rows)}; expected {EXPECTED_NUCLEI}")

    source_counts = Counter(row["source_name"] for row in rows)
    if dict(source_counts) != EXPECTED_SOURCE_NUCLEI:
        raise RuntimeError(
            f"Unexpected holdout source counts: {dict(source_counts)}; expected {EXPECTED_SOURCE_NUCLEI}"
        )

    bag_counts = Counter(row["sample_name"] for row in rows)
    if len(bag_counts) != EXPECTED_BAGS:
        raise RuntimeError(f"Holdout bags={len(bag_counts)}; expected {EXPECTED_BAGS}")
    if min(bag_counts.values()) < 1 or max(bag_counts.values()) > 100:
        raise RuntimeError(
            f"Unexpected holdout bag sizes: min={min(bag_counts.values())} max={max(bag_counts.values())}"
        )

    keys = [row["nucleus_key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("Blinded holdout manifest contains duplicate nucleus_key values")

    statuses = {row["holdout_status"] for row in rows}
    if statuses != {EXPECTED_HOLDOUT_STATUS}:
        raise RuntimeError(f"Unexpected holdout_status values: {sorted(statuses)}")


def build_tasks(rows: list[dict[str, str]], output_dir: Path) -> list[DownloadTask]:
    tasks: list[DownloadTask] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        specs = (
            ("fitc", row["fitc_filename"], FITC_BASE_URL),
            ("dapi", row["dapi_filename"], DAPI_MASK_BASE_URL),
            ("mask", row["mask_filename"], DAPI_MASK_BASE_URL),
        )
        for channel, filename, base in specs:
            file_key = (channel, filename)
            if file_key in seen:
                raise RuntimeError(f"Duplicate holdout file task: {channel} {filename}")
            seen.add(file_key)
            tasks.append(
                DownloadTask(
                    sample_id=row["sample_id"],
                    nucleus_key=row["nucleus_key"],
                    source_name=row["source_name"],
                    sample_name=row["sample_name"],
                    channel=channel,
                    filename=filename,
                    url=object_url(base, filename),
                    target_path=output_dir / channel / filename,
                )
            )
    if len(tasks) != EXPECTED_FILES:
        raise RuntimeError(f"Holdout file tasks={len(tasks)}; expected {EXPECTED_FILES}")
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
        "source_name": task.source_name,
        "sample_name": task.sample_name,
        "channel": task.channel,
        "filename": task.filename,
        "url": task.url,
        "local_path": str(task.target_path),
        "status": status,
        "bytes": n_bytes,
        "sha256": digest,
        "error": error,
    }


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

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            n_bytes, digest = download_once(task, timeout)
            return result_row(
                task,
                status="downloaded",
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
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"

    parser.add_argument(
        "--manifest",
        default=str(
            metadata_root
            / "r1_v2_freeze"
            / "r1_v2_final_holdout_manifest_blinded.csv"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            model_root
            / "nasa_bps_53bp1"
            / "r1_v2_final_candidate"
            / "r1_countnet_v2_final_bn_recalibrated.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            data_root
            / "nasa_bps_microscopy"
            / "r1_v2_final_holdout_blinded"
            / "images"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()

    for path in (manifest_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != FROZEN_REPAIRED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Repaired final checkpoint SHA256 mismatch: "
            f"expected {FROZEN_REPAIRED_CHECKPOINT_SHA256}, got {checkpoint_sha}"
        )
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != FROZEN_BLINDED_HOLDOUT_MANIFEST_SHA256:
        raise RuntimeError(
            "Blinded holdout manifest SHA256 mismatch: "
            f"expected {FROZEN_BLINDED_HOLDOUT_MANIFEST_SHA256}, got {manifest_sha}"
        )

    rows = read_csv(manifest_path)
    validate_blinded_manifest(rows)
    tasks = build_tasks(rows, output_dir)

    bag_counts = Counter(row["sample_name"] for row in rows)
    source_counts = Counter(row["source_name"] for row in rows)

    print("Radiation Edge AI - materialize NASA BPS R1 v2 FINAL HOLDOUT images")
    print("PHENOTYPE-BLIND IMAGE ACCESS ONLY")
    print(f"repaired checkpoint SHA256 pinned: {checkpoint_sha}")
    print(f"blinded holdout manifest SHA256: {manifest_sha}")
    print(f"holdout nuclei: {len(rows)}")
    print(f"holdout bags: {len(bag_counts)}")
    print(f"source nuclei: {dict(source_counts)}")
    print(f"expected TIFF files: {len(tasks)}")
    print(f"output: {output_dir}")
    print(f"workers: {max(1, args.workers)}")
    print("NASA phenotype/reference tables read: NO")
    print("")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(
                run_task,
                task,
                retries=max(1, args.retries),
                timeout=max(1, args.timeout),
                overwrite=args.overwrite,
            ): task
            for task in tasks
        }
        for i, future in enumerate(as_completed(future_map), start=1):
            results.append(future.result())
            if i == 1 or i % 250 == 0 or i == len(tasks):
                failures = sum(row["status"] == "failed" for row in results)
                print(f"[{i:>5}/{len(tasks)}] failures={failures}")

    results.sort(
        key=lambda row: (
            row["sample_id"],
            CHANNEL_ORDER.get(str(row["channel"]), 99),
            row["filename"],
        )
    )
    failed = [row for row in results if row["status"] == "failed"]
    channel_ok = Counter(row["channel"] for row in results if row["status"] != "failed")
    status_counts = Counter(row["status"] for row in results)

    materialization_root = output_dir.parent
    inventory_path = materialization_root / "download_inventory.csv"
    summary_path = materialization_root / "download_summary.json"
    write_csv(inventory_path, results)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "R1_v2_final_holdout_blinded_image_materialization",
        "phenotype_blind": True,
        "phenotype_tables_read": False,
        "model_predictions_computed": False,
        "repaired_checkpoint_sha256": checkpoint_sha,
        "blinded_holdout_manifest_sha256": manifest_sha,
        "holdout_nuclei": len(rows),
        "holdout_bags": len(bag_counts),
        "source_nuclei": dict(source_counts),
        "expected_files": len(tasks),
        "successful_files": len(results) - len(failed),
        "failed_files": len(failed),
        "channel_success_counts": dict(channel_ok),
        "status_counts": dict(status_counts),
        "image_root": str(output_dir),
        "inventory": str(inventory_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("[Phenotype-blind materialization summary]")
    print(f"successful files: {len(results) - len(failed)}/{len(tasks)}")
    print(f"FITC: {channel_ok['fitc']}/{EXPECTED_NUCLEI}")
    print(f"DAPI: {channel_ok['dapi']}/{EXPECTED_NUCLEI}")
    print(f"MASK: {channel_ok['mask']}/{EXPECTED_NUCLEI}")
    print(f"status counts: {dict(status_counts)}")
    print(f"inventory: {inventory_path}")
    print(f"summary: {summary_path}")
    print("NASA phenotype/reference tables read: NO")
    print("FINAL HOLDOUT PHENOTYPE STATUS: BLINDED")

    if failed:
        print(f"MATERIALIZATION COMPLETE: NO ({len(failed)} failed files)")
        for row in failed[:30]:
            print(f"FAILED {row['channel']} {row['filename']}: {row['error']}")
        if len(failed) > 30:
            print(f"... {len(failed) - 30} additional failures")
        print("NEXT GATE: rerun this resumable command; do not inspect phenotype outcomes.")
        return 2

    if len(results) != EXPECTED_FILES or any(channel_ok[c] != EXPECTED_NUCLEI for c in CHANNEL_ORDER):
        raise RuntimeError("Materialized file counts do not match the frozen blinded manifest")

    print("NASA BPS R1 V2 FINAL HOLDOUT IMAGE MATERIALIZATION COMPLETE: YES")
    print("NEXT GATE: run phenotype-blind FITC/DAPI/MASK hard QC before any phenotype unblinding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
