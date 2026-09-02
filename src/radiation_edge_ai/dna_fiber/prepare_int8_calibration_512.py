"""Prepare deterministic DNAi 512x512 PTQ calibration tensors.

This script intentionally keeps the frozen 20-image inter-grader panel out of
calibration. It searches the public DNAi dataset for image files whose path
contains a training token, verifies that none match the frozen validation
sources by SHA256, applies DNAi's microscopy preprocessing and tensor
normalization, then saves deterministic 512x512 float32 NCHW crops for Kneron
`ModelConfig.analysis`.

The output directory is an artifact and must not be committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

from dnafiber.data.utils import load_image
from dnafiber.inference import transform

DNAI_COMMIT = "fcf20c7d6eb385675ff7d07da4fdf471589ce0cf"
PIXEL_SIZE_UM = 0.26
CLARITY = 1.0
TILE = 512
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
TRAIN_TOKENS = {"train", "training"}
EXCLUDE_TOKENS = {"annotation", "annotations", "mask", "masks", "label", "labels", "intergrader"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_token(path: Path, tokens: set[str]) -> bool:
    parts = {part.lower() for part in path.parts}
    stem_parts = set(path.stem.lower().replace("-", "_").split("_"))
    return bool((parts | stem_parts) & tokens)


def deterministic_crop(tensor: np.ndarray, rng: random.Random) -> tuple[np.ndarray, int, int]:
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
        raise RuntimeError(f"Expected NCHW 1x3xHxW, got {tensor.shape}")
    height, width = tensor.shape[-2:]
    if height < TILE or width < TILE:
        raise RuntimeError(f"Processed image too small for {TILE}: {width}x{height}")
    y0 = 0 if height == TILE else rng.randint(0, height - TILE)
    x0 = 0 if width == TILE else rng.randint(0, width - TILE)
    crop = tensor[:, :, y0 : y0 + TILE, x0 : x0 + TILE]
    return np.ascontiguousarray(crop, dtype=np.float32), y0, x0


def main() -> int:
    parser = argparse.ArgumentParser()
    default_data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    default_model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    default_output = default_model_root / "dnai" / "unet_mobileone_s1" / "calibration_512"
    parser.add_argument("--data-root", default=str(default_data_root))
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=720)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    dataset_root = data_root / "dnai_public_v2" / "extracted" / "zenodo"
    manifest_path = data_root / "dnai_public_v2" / "r1_intergrader20" / "manifest.json"
    output_dir = Path(args.output)
    tensor_dir = output_dir / "tensors"

    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dnai_commit") != DNAI_COMMIT:
        raise RuntimeError("Manifest/code revision mismatch")
    validation_hashes = {record["source_sha256"] for record in manifest["records"]}

    candidates = []
    for path in dataset_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = path.relative_to(dataset_root)
        if not has_token(rel, TRAIN_TOKENS):
            continue
        if has_token(rel, EXCLUDE_TOKENS):
            continue
        candidates.append(path)

    candidates.sort(key=lambda p: p.as_posix().lower())
    if not candidates:
        top_dirs = sorted({p.parts[0] for p in dataset_root.rglob("*") if p.relative_to(dataset_root).parts})
        raise RuntimeError(
            "No training-image candidates found using train/training path tokens. "
            f"Top-level entries seen: {top_dirs[:50]}"
        )

    rng = random.Random(args.seed)
    shuffled = candidates[:]
    rng.shuffle(shuffled)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output is not empty: {output_dir}; use --overwrite to replace")
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in tensor_dir.glob("*.npy"):
            path.unlink()

    print("Radiation Edge AI - DNAi INT8 calibration preparation")
    print(f"Dataset: {dataset_root}")
    print(f"Training-image candidates: {len(candidates)}")
    print(f"Requested tensors: {args.count}")
    print(f"Seed: {args.seed}")
    print("Validation exclusion: frozen 20 source SHA256 values")
    print("Preprocessing: load_image(pixel_size=0.26, clarity=1.0, reverse_channel=False) + DNAi transform")
    print("")

    records = []
    for source in shuffled:
        source_hash = sha256_file(source)
        if source_hash in validation_hashes:
            continue
        image = load_image(
            source,
            reverse_channel=False,
            pixel_size=PIXEL_SIZE_UM,
            verbose=False,
            clarity=CLARITY,
        )
        tensor = transform(image=image)["image"].unsqueeze(0).numpy().astype(np.float32, copy=False)
        try:
            crop, y0, x0 = deterministic_crop(tensor, rng)
        except RuntimeError:
            continue
        index = len(records)
        tensor_name = f"cal_{index:03d}.npy"
        tensor_path = tensor_dir / tensor_name
        np.save(tensor_path, crop, allow_pickle=False)
        records.append(
            {
                "index": index,
                "source": source.relative_to(dataset_root).as_posix(),
                "source_sha256": source_hash,
                "processed_shape": list(tensor.shape),
                "crop_y": y0,
                "crop_x": x0,
                "tensor": f"tensors/{tensor_name}",
                "tensor_shape": list(crop.shape),
                "tensor_dtype": str(crop.dtype),
                "tensor_min": float(crop.min()),
                "tensor_max": float(crop.max()),
                "tensor_mean": float(crop.mean()),
                "tensor_std": float(crop.std()),
            }
        )
        print(f"[{len(records):03d}/{args.count:03d}] {records[-1]['source']} crop=({y0},{x0})")
        if len(records) >= args.count:
            break

    if len(records) < args.count:
        raise RuntimeError(f"Only produced {len(records)} usable non-validation calibration tensors")

    payload = {
        "dnai_commit": DNAI_COMMIT,
        "purpose": "KL720 INT8 PTQ calibration only; frozen inter-grader20 excluded",
        "input_name": "input",
        "tile": [1, 3, TILE, TILE],
        "pixel_size_um": PIXEL_SIZE_UM,
        "clarity": CLARITY,
        "reverse_channel": False,
        "seed": args.seed,
        "n_candidates": len(candidates),
        "n_tensors": len(records),
        "records": records,
    }
    manifest_out = output_dir / "manifest.json"
    manifest_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("")
    print("INT8 CALIBRATION TENSORS READY: YES")
    print(f"Tensors: {tensor_dir}")
    print(f"Manifest: {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
