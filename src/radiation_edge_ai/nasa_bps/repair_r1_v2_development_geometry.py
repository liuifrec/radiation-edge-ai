"""Repair rare >256x256 nuclei in the frozen NASA BPS R1 v2 DEVELOPMENT set.

This utility is intentionally narrow. It reads only the frozen R1 v2 development
manifest, its QC table, the public benchmark crosswalk/DAPI-MASK metadata, and
local DEVELOPMENT images. It never reads the final-holdout manifest, final-
holdout images, or phenotype outcomes.

For each hard QC failure, the script requires the failure to be geometry-only
(native crop exceeds the frozen 256x256 canvas) and requires the failed nucleus
not to be a pilot-v1 member. It then walks the same deterministic v1 ranking for
that Source Name x condition cell, considers only previously unselected nuclei,
downloads candidate FITC/DAPI/MASK triplets into an audit cache, and accepts the
first candidate that passes the same hard image checks used by the development
QC gate.

The original 7,200-row manifest is preserved for audit. A repaired manifest is
written separately, keeping exactly 100 nuclei in every one of the 72 bags and
preserving all 2,160 pilot-v1 nuclei. No phenotype data are read and no model is
trained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ORIGINAL_DEV_MANIFEST_SHA256 = "029d156250f1b73a6cc2b556b88a8e3b6900366a6a2b36244b699d2d9c43ab01"
EXPECTED_NUCLEI = 7200
EXPECTED_BAGS = 72
EXPECTED_PER_BAG = 100
EXPECTED_V1 = 2160
CANVAS = 256
P_LOW = 1.0
P_HIGH = 99.5
SELECTION_SEED = "nasa-bps-pilot-v1-720"

FITC_BASE_URL = "https://nasa-bps-training-data.s3.us-west-2.amazonaws.com/Microscopy/train/"
DAPI_MASK_BASE_URL = "https://nasa-bps-training-data.s3.us-west-2.amazonaws.com/Microscopy/DAPI_MASK_images/"
USER_AGENT = "radiation-edge-ai/0.1 NASA-BPS-R1-v2-geometry-repair"
TIFF_MAGICS = (b"II*\x00", b"MM\x00*")

DEV_SOURCES = {
    "BALBCF1",
    "BALBCM1",
    "BALBCM2",
    "C57BLF1",
    "C57BLM1",
    "C57BLM3",
}
FINAL_HOLDOUT_SOURCES = {"BALBCF2", "C57BLF2", "C57BLF3"}

FILENAME_RE = re.compile(
    r"^(?P<plate>P\d+)_(?P<acquisition>\d+)-(?P<well>[A-Za-z]+\d+)_"
    r"(?P<field>\d+)_(?P<object>\d+)_(?P<channel>proj|DAPI|MASK)\.tif$",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def normalize_number(value: str) -> str:
    x = float(str(value).strip())
    if x.is_integer():
        return f"{x:.1f}"
    return format(x, "g")


def normalize_hour(value: str) -> str:
    x = float(str(value).strip())
    return str(int(x)) if x.is_integer() else format(x, "g")


def condition_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row["particle_type"]).strip(),
        normalize_number(row["dose_Gy"]),
        normalize_hour(row["hr_post_exposure"]),
    )


def parse_filename(filename: str) -> dict[str, str]:
    match = FILENAME_RE.match(Path(filename).name)
    if match is None:
        raise ValueError(f"Unrecognized NASA BPS filename: {filename}")
    result = match.groupdict()
    result["plate"] = result["plate"].upper()
    result["well"] = result["well"].upper()
    result["channel"] = result["channel"].lower()
    result["nucleus_key"] = (
        f"{result['plate']}_{result['acquisition']}-{result['well']}_"
        f"{result['field']}_{result['object']}"
    )
    return result


def deterministic_rank(nucleus_key: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}|{nucleus_key}".encode("utf-8")).hexdigest()


def is_tiff(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 8:
        return False
    with path.open("rb") as handle:
        return handle.read(4) in TIFF_MAGICS


def object_url(base: str, filename: str) -> str:
    return base + urllib.parse.quote(filename, safe="-_.()")


def download_file(url: str, path: Path, *, retries: int, timeout: int) -> None:
    if is_tiff(path):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""
    for attempt in range(1, retries + 1):
        temporary = path.with_name(
            path.name + f".part.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            first4 = b""
            n_bytes = 0
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    if len(first4) < 4:
                        first4 = (first4 + chunk)[:4]
                    out.write(chunk)
                    n_bytes += len(chunk)
            if first4 not in TIFF_MAGICS or n_bytes < 8:
                raise RuntimeError(
                    f"Downloaded object is not a valid TIFF: bytes={n_bytes} first4={first4!r}"
                )
            os.replace(temporary, path)
            return
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 4))
    raise RuntimeError(f"Download failed after {retries} attempts: {url}: {last_error}")


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode TIFF: {path}")
    return image


def inspect_triplet(fitc_path: Path, dapi_path: Path, mask_path: Path, target_label: int) -> tuple[bool, str, dict[str, Any]]:
    fitc = load_image(fitc_path)
    dapi = load_image(dapi_path)
    mask = load_image(mask_path)
    meta: dict[str, Any] = {
        "fitc_shape": list(fitc.shape),
        "dapi_shape": list(dapi.shape),
        "mask_shape": list(mask.shape),
        "fitc_dtype": str(fitc.dtype),
        "dapi_dtype": str(dapi.dtype),
        "mask_dtype": str(mask.dtype),
    }
    failures: list[str] = []
    if fitc.ndim != 2 or dapi.ndim != 2 or mask.ndim != 2:
        failures.append("non-2D image")
        return False, "; ".join(failures), meta
    if fitc.shape != dapi.shape or fitc.shape != mask.shape:
        failures.append("shape mismatch")
        return False, "; ".join(failures), meta

    h, w = fitc.shape
    meta["height"] = int(h)
    meta["width"] = int(w)
    if fitc.dtype != np.uint16:
        failures.append(f"FITC dtype={fitc.dtype}")
    if dapi.dtype != np.uint16:
        failures.append(f"DAPI dtype={dapi.dtype}")
    if not np.issubdtype(mask.dtype, np.integer):
        failures.append(f"MASK dtype={mask.dtype} not integer")
    if h > CANVAS or w > CANVAS:
        failures.append(f"native geometry {h}x{w} exceeds {CANVAS}x{CANVAS}")

    target = mask == target_label
    target_present = bool(np.any(target))
    center_label = int(mask[h // 2, w // 2]) if np.issubdtype(mask.dtype, np.integer) else -1
    border_touches = bool(
        target_present
        and (
            np.any(target[0, :])
            or np.any(target[-1, :])
            or np.any(target[:, 0])
            or np.any(target[:, -1])
        )
    )
    meta["target_present"] = int(target_present)
    meta["center_label"] = center_label
    meta["target_touches_border"] = int(border_touches)
    if not target_present:
        failures.append(f"target label {target_label} absent")
    if center_label != target_label:
        failures.append(f"center label {center_label} != target {target_label}")
    if border_touches:
        failures.append(f"target label {target_label} touches border")

    for channel, image in (("FITC", fitc), ("DAPI", dapi)):
        x = image.astype(np.float32, copy=False)
        lo = float(np.percentile(x, P_LOW))
        hi = float(np.percentile(x, P_HIGH))
        span = hi - lo
        meta[f"{channel.lower()}_robust_range"] = span
        if not math.isfinite(span) or span < 0:
            failures.append(f"{channel} robust range invalid")

    return len(failures) == 0, "; ".join(failures), meta


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"
    development_root = data_root / "nasa_bps_microscopy" / "r1_v2_development"
    parser.add_argument(
        "--manifest",
        default=str(metadata_root / "r1_v2_freeze" / "r1_v2_development_manifest_100.csv"),
    )
    parser.add_argument(
        "--qc-table",
        default=str(development_root / "qc" / "r1_v2_development_triplet_qc.csv"),
    )
    parser.add_argument(
        "--crosswalk",
        default=str(metadata_root / "fitc_53bp1_osd366_crosswalk.csv"),
    )
    parser.add_argument(
        "--dapi-mask-meta",
        default=str(metadata_root / "meta_DAPI_MASK.csv"),
    )
    parser.add_argument(
        "--image-root",
        default=str(development_root / "images"),
    )
    parser.add_argument(
        "--candidate-cache",
        default=str(development_root / "repair_candidates"),
    )
    parser.add_argument(
        "--output-manifest",
        default=str(metadata_root / "r1_v2_freeze" / "r1_v2_development_manifest_100_qc1.csv"),
    )
    parser.add_argument(
        "--summary",
        default=str(metadata_root / "r1_v2_freeze" / "r1_v2_development_geometry_repair_summary.json"),
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    qc_path = Path(args.qc_table).resolve()
    crosswalk_path = Path(args.crosswalk).resolve()
    dapi_path = Path(args.dapi_mask_meta).resolve()
    image_root = Path(args.image_root).resolve()
    candidate_cache = Path(args.candidate_cache).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    summary_path = Path(args.summary).resolve()

    for path in (manifest_path, qc_path, crosswalk_path, dapi_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != ORIGINAL_DEV_MANIFEST_SHA256:
        raise RuntimeError(
            f"Original development manifest SHA256 mismatch: expected {ORIGINAL_DEV_MANIFEST_SHA256}, got {manifest_sha}"
        )

    manifest = read_csv(manifest_path)
    qc_rows = read_csv(qc_path)
    crosswalk = read_csv(crosswalk_path)
    dapi_rows = read_csv(dapi_path)

    if len(manifest) != EXPECTED_NUCLEI:
        raise RuntimeError(f"Development nuclei={len(manifest)}; expected {EXPECTED_NUCLEI}")
    if {row["source_name"] for row in manifest} != DEV_SOURCES:
        raise RuntimeError("Unexpected Source Name set in development manifest")
    if {row["source_name"] for row in manifest} & FINAL_HOLDOUT_SOURCES:
        raise RuntimeError("Final holdout Source Name leaked into development manifest")

    failed_qc = [row for row in qc_rows if str(row.get("ok", "0")).strip() != "1"]
    if not failed_qc:
        raise RuntimeError("QC table contains no failures; repair is unnecessary")
    for row in failed_qc:
        error = row.get("error", "")
        if "native geometry" not in error or "exceeds 256x256" not in error:
            raise RuntimeError(f"Refusing non-geometry repair for {row.get('sample_id')}: {error}")
        if str(row.get("v1_member", "0")).strip() == "1":
            raise RuntimeError(f"Geometry failure is a pilot-v1 member; automatic replacement forbidden: {row['sample_id']}")

    manifest_by_sample_id = {row["sample_id"]: row for row in manifest}
    selected_keys = {row["nucleus_key"] for row in manifest}
    if len(selected_keys) != EXPECTED_NUCLEI:
        raise RuntimeError("Development manifest contains duplicate nucleus_key values")

    partners: dict[str, dict[str, str]] = defaultdict(dict)
    for row in dapi_rows:
        parsed = parse_filename(row["filename"])
        if parsed["channel"] in {"dapi", "mask"}:
            partners[parsed["nucleus_key"]][parsed["channel"]] = row["filename"]

    candidates_by_cell: dict[tuple[str, tuple[str, str, str]], list[dict[str, str]]] = defaultdict(list)
    for row in crosswalk:
        source = row.get("osd_source_name", "")
        if source not in DEV_SOURCES:
            continue
        parsed = parse_filename(row["filename"])
        pair = partners.get(parsed["nucleus_key"], {})
        if "dapi" not in pair or "mask" not in pair:
            continue
        enriched = dict(row)
        enriched["nucleus_key"] = parsed["nucleus_key"]
        enriched["dapi_filename"] = pair["dapi"]
        enriched["mask_filename"] = pair["mask"]
        enriched["rank_sha256"] = deterministic_rank(parsed["nucleus_key"])
        candidates_by_cell[(source, condition_key(row))].append(enriched)

    for key in candidates_by_cell:
        candidates_by_cell[key].sort(key=lambda row: (row["rank_sha256"], row["nucleus_key"]))

    repaired = [dict(row) for row in manifest]
    row_index = {row["sample_id"]: i for i, row in enumerate(repaired)}
    replacements: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    print("Radiation Edge AI - repair NASA BPS R1 v2 DEVELOPMENT geometry outliers")
    print("Final holdout remains untouched: no holdout manifest/images/phenotypes are read.")
    print(f"Original manifest SHA256: {manifest_sha}")
    print(f"Geometry failures to replace: {len(failed_qc)}")
    print("")

    for fail in sorted(failed_qc, key=lambda row: row["sample_id"]):
        sample_id = fail["sample_id"]
        old = manifest_by_sample_id[sample_id]
        source = old["source_name"]
        cell = condition_key(old)
        pool = candidates_by_cell.get((source, cell), [])
        if not pool:
            raise RuntimeError(f"No replacement candidate pool for {source} {cell}")

        accepted: dict[str, str] | None = None
        accepted_meta: dict[str, Any] | None = None
        for candidate in pool:
            key = candidate["nucleus_key"]
            if key in selected_keys:
                continue
            parsed = parse_filename(candidate["filename"])
            fitc_cache = candidate_cache / "fitc" / candidate["filename"]
            dapi_cache = candidate_cache / "dapi" / candidate["dapi_filename"]
            mask_cache = candidate_cache / "mask" / candidate["mask_filename"]
            download_file(object_url(FITC_BASE_URL, candidate["filename"]), fitc_cache, retries=args.retries, timeout=args.timeout)
            download_file(object_url(DAPI_MASK_BASE_URL, candidate["dapi_filename"]), dapi_cache, retries=args.retries, timeout=args.timeout)
            download_file(object_url(DAPI_MASK_BASE_URL, candidate["mask_filename"]), mask_cache, retries=args.retries, timeout=args.timeout)
            ok, reason, meta = inspect_triplet(
                fitc_cache,
                dapi_cache,
                mask_cache,
                int(parsed["object"]),
            )
            if not ok:
                rejected.append(
                    {
                        "replacing_sample_id": sample_id,
                        "candidate_nucleus_key": key,
                        "reason": reason,
                        **meta,
                    }
                )
                print(f"  reject {sample_id} -> {key}: {reason}")
                continue
            accepted = candidate
            accepted_meta = meta
            break

        if accepted is None or accepted_meta is None:
            raise RuntimeError(f"Could not find a QC-passing replacement for {sample_id}")

        parsed = parse_filename(accepted["filename"])
        new_row = dict(old)
        new_row.update(
            {
                "nucleus_key": accepted["nucleus_key"],
                "source_name": accepted["osd_source_name"],
                "sample_name": accepted["osd_sample_name"],
                "strain": accepted["osd_strain"],
                "sex": accepted["osd_sex"],
                "dose_Gy": normalize_number(accepted["dose_Gy"]),
                "particle_type": accepted["particle_type"],
                "hr_post_exposure": normalize_hour(accepted["hr_post_exposure"]),
                "plate": parsed["plate"],
                "acquisition": parsed["acquisition"],
                "well": parsed["well"],
                "fitc_filename": accepted["filename"],
                "dapi_filename": accepted["dapi_filename"],
                "mask_filename": accepted["mask_filename"],
                "selection_rank_sha256": accepted["rank_sha256"],
                "v1_member": "0",
                "v1_sample_id": "",
            }
        )
        for field in ("source_name", "sample_name", "strain", "sex", "dose_Gy", "particle_type", "hr_post_exposure"):
            if field in old and field in new_row and field in {"source_name", "sample_name", "strain", "sex", "particle_type"}:
                if str(old[field]) != str(new_row[field]):
                    raise RuntimeError(
                        f"Replacement changed invariant {field} for {sample_id}: {old[field]!r} -> {new_row[field]!r}"
                    )
        if normalize_number(old["dose_Gy"]) != normalize_number(new_row["dose_Gy"]):
            raise RuntimeError(f"Replacement changed dose for {sample_id}")
        if normalize_hour(old["hr_post_exposure"]) != normalize_hour(new_row["hr_post_exposure"]):
            raise RuntimeError(f"Replacement changed timepoint for {sample_id}")

        for channel, filename in (
            ("fitc", accepted["filename"]),
            ("dapi", accepted["dapi_filename"]),
            ("mask", accepted["mask_filename"]),
        ):
            src = candidate_cache / channel / filename
            dst = image_root / channel / filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)

        repaired[row_index[sample_id]] = new_row
        selected_keys.remove(old["nucleus_key"])
        selected_keys.add(accepted["nucleus_key"])
        replacements.append(
            {
                "sample_id": sample_id,
                "old_nucleus_key": old["nucleus_key"],
                "old_error": fail.get("error", ""),
                "new_nucleus_key": accepted["nucleus_key"],
                "new_fitc_filename": accepted["filename"],
                "new_rank_sha256": accepted["rank_sha256"],
                **accepted_meta,
            }
        )
        print(
            f"  replace {sample_id}: {old['nucleus_key']} -> {accepted['nucleus_key']} "
            f"({accepted_meta['height']}x{accepted_meta['width']})"
        )

    if len(repaired) != EXPECTED_NUCLEI:
        raise RuntimeError("Repaired manifest row count changed")
    bag_counts = Counter(row["sample_name"] for row in repaired)
    if len(bag_counts) != EXPECTED_BAGS or any(n != EXPECTED_PER_BAG for n in bag_counts.values()):
        raise RuntimeError(f"Repaired bag structure invalid: {dict(bag_counts)}")
    if sum(str(row.get("v1_member", "0")).strip() == "1" for row in repaired) != EXPECTED_V1:
        raise RuntimeError("Repaired manifest did not preserve all 2,160 v1 members")
    if len({row["nucleus_key"] for row in repaired}) != EXPECTED_NUCLEI:
        raise RuntimeError("Repaired manifest contains duplicate nucleus keys")

    write_csv(output_manifest, repaired)
    repaired_sha = sha256_file(output_manifest)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repair_version": "r1_v2_development_geometry_qc1",
        "original_manifest": str(manifest_path),
        "original_manifest_sha256": manifest_sha,
        "repaired_manifest": str(output_manifest),
        "repaired_manifest_sha256": repaired_sha,
        "n_replacements": len(replacements),
        "n_rejected_candidates_before_acceptance": len(rejected),
        "replacements": replacements,
        "rejected_candidates": rejected,
        "n_nuclei": len(repaired),
        "n_bags": len(bag_counts),
        "n_v1_members": EXPECTED_V1,
        "final_holdout_status": "UNTOUCHED",
        "notes": [
            "Only geometry-only development QC failures were eligible for automatic replacement.",
            "Replacement candidates follow the same deterministic v1 ranking and are chosen from previously unselected nuclei in the same Source Name x condition cell.",
            "The original manifest remains unchanged for audit.",
            "No phenotype tables or final-holdout resources were read.",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("[Repair summary]")
    print(f"replacements: {len(replacements)}")
    print(f"rejected candidates before acceptance: {len(rejected)}")
    print(f"repaired nuclei: {len(repaired)}")
    print(f"bags: {len(bag_counts)} x {EXPECTED_PER_BAG}")
    print(f"pilot-v1 members preserved: {EXPECTED_V1}")
    print(f"repaired manifest: {output_manifest}")
    print(f"repaired manifest SHA256: {repaired_sha}")
    print(f"summary: {summary_path}")
    print("")
    print("NASA BPS R1 V2 DEVELOPMENT GEOMETRY REPAIR COMPLETE: YES")
    print("FINAL HOLDOUT STATUS: UNTOUCHED")
    print("NEXT GATE: rerun the development QC validator against the repaired manifest; freeze its SHA256 only after 7200/7200 pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
