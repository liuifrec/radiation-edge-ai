"""Build the frozen 20-image DNAi R1 inter-grader manifest.

This script discovers the 20 image keys present in H1/H2/H3/H4 ``common``
annotations, resolves the matching source image for each key, and records
SHA-256 hashes so the benchmark inputs are immutable and auditable.

Run this with the dedicated DNAi Python environment, not the KL720 Python 3.9
runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2

DNAI_COMMIT = "fcf20c7d6eb385675ff7d07da4fdf471589ce0cf"
ZENODO_RECORD = "18868353"
ZENODO_DOI = "10.5281/zenodo.18868353"
ZENODO_ARCHIVE_MD5 = "52f205d5d6c81f2f0a9fb016d5fa534f"
GRADERS = ("H1", "H2", "H3", "H4")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def natural_key(relative: Path) -> tuple[int, int, str]:
    try:
        group = int(relative.parts[0])
    except (ValueError, IndexError):
        group = 10**9
    stem = relative.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    tile = int(digits) if digits else 10**9
    return group, tile, relative.as_posix()


def source_score(path: Path, relative: Path) -> int:
    parts = [part.lower() for part in path.parts]
    score = 0
    if "annotations" in parts:
        return -10_000
    if "intergrader" in parts:
        score += 100
    if "images" in parts or "image" in parts:
        score += 50
    if "common" in parts:
        score += 25
    if "test" in parts:
        score += 10
    if path.parent.name == relative.parent.name:
        score += 15
    if "mask" in parts or "masks" in parts:
        score -= 500
    return score


def resolve_source(dataset_root: Path, relative: Path, all_images: list[Path]) -> Path:
    candidates = [
        path
        for path in all_images
        if path.name == relative.name
        and path.parent.name == relative.parent.name
        and "annotations" not in [part.lower() for part in path.parts]
    ]
    if not candidates:
        raise RuntimeError(
            f"No source-image candidate found for {relative.as_posix()}. "
            "The dataset layout needs manual inspection."
        )

    ranked = sorted(candidates, key=lambda p: (-source_score(p, relative), str(p)))
    best_score = source_score(ranked[0], relative)
    best = [p for p in ranked if source_score(p, relative) == best_score]

    if len(best) == 1:
        return best[0]

    # Duplicate copies are acceptable only when they are byte-identical.
    hashes = {sha256(path) for path in best}
    if len(hashes) == 1:
        return best[0]

    details = "\n".join(f"  score={source_score(p, relative):4d}  {p}" for p in ranked)
    raise RuntimeError(
        f"Ambiguous non-identical source images for {relative.as_posix()}:\n{details}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    default_data_root = os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data")
    parser.add_argument("--data-root", default=default_data_root)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    dataset_root = data_root / "dnai_public_v2" / "extracted" / "zenodo"
    annotation_root = dataset_root / "intergrader" / "annotations"
    output_root = data_root / "dnai_public_v2" / "r1_intergrader20"
    output_root.mkdir(parents=True, exist_ok=True)

    if not annotation_root.exists():
        raise FileNotFoundError(f"Inter-grader annotation root not found: {annotation_root}")

    h1_common = annotation_root / "H1" / "common"
    keys = sorted(
        [p.relative_to(h1_common) for p in h1_common.rglob("*.png")],
        key=natural_key,
    )
    if len(keys) != 20:
        raise RuntimeError(f"Expected exactly 20 H1/common annotations, found {len(keys)}")

    # Ensure the panel is genuinely shared by all four graders.
    for relative in keys:
        missing = [
            grader
            for grader in GRADERS
            if not (annotation_root / grader / "common" / relative).is_file()
        ]
        if missing:
            raise RuntimeError(f"{relative.as_posix()} is missing from graders: {missing}")

    all_images = [
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]

    records = []
    print("DNAi R1 manifest builder")
    print(f"Dataset root: {dataset_root}")
    print(f"Shared inter-grader keys: {len(keys)}")
    print("")

    for index, relative in enumerate(keys, start=1):
        source = resolve_source(dataset_root, relative, all_images)
        image = cv2.imread(str(source), cv2.IMREAD_COLOR_RGB)
        if image is None:
            raise RuntimeError(f"OpenCV could not read source image: {source}")
        height, width = image.shape[:2]

        record = {
            "index": index,
            "sample_id": f"{relative.parts[0]}__{relative.stem}",
            "key": relative.as_posix(),
            "source": source.relative_to(dataset_root).as_posix(),
            "source_sha256": sha256(source),
            "width": width,
            "height": height,
            "pixel_size_um": 0.26,
        }
        for grader in GRADERS:
            annotation = annotation_root / grader / "common" / relative
            record[f"{grader}_annotation"] = annotation.relative_to(dataset_root).as_posix()
            record[f"{grader}_sha256"] = sha256(annotation)
        records.append(record)
        print(
            f"[{index:02d}/20] {record['key']:<18} "
            f"{width}x{height}  <-  {record['source']}"
        )

    manifest = {
        "name": "DNAi R1 inter-grader 20-image reference panel",
        "purpose": "Strict-FP32 teacher/reference panel before ONNX and KL720 conversion",
        "dataset": {
            "zenodo_record": ZENODO_RECORD,
            "doi": ZENODO_DOI,
            "archive_md5": ZENODO_ARCHIVE_MD5,
        },
        "dnai_commit": DNAI_COMMIT,
        "model": "UNET_MOBILEONE_S1",
        "pixel_size_um": 0.26,
        "graders": list(GRADERS),
        "n_images": len(records),
        "records": records,
    }

    json_path = output_root / "manifest.json"
    csv_path = output_root / "manifest.csv"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    fieldnames = list(records[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("")
    print("R1 MANIFEST READY: YES")
    print(f"Images: {len(records)}")
    print(f"JSON:   {json_path}")
    print(f"CSV:    {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
