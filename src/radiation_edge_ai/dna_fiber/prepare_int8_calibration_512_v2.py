"""Prepare stratified, fiber-aware DNAi 512x512 PTQ calibration tensors.

INT8 v1 used 64 random crops and passed the hardware/simulator gates but lost
biological fidelity on the first complete image (14 -> 10 valid fibers). This
v2 calibration keeps the frozen 20-image inter-grader panel excluded and uses
more representative public-training data:

- default 192 UNIQUE source images,
- stratified across train/images subgroups when available (e.g. V0/S1/Origins),
- one crop per source image,
- approximately half ordinary random crops and half fluorescence-rich crops,
- exact DNAi preprocessing before crop selection and saving.

The goal is PTQ range calibration, not supervised retraining. No labels or
validation images are used. Outputs are artifacts and must not be committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
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
EXCLUDE_TOKENS = {
    "annotation",
    "annotations",
    "mask",
    "masks",
    "label",
    "labels",
    "intergrader",
}


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


def infer_stratum(relative_path: Path) -> str:
    """Use the directory immediately below train/images when present."""
    lowered = [part.lower() for part in relative_path.parts]
    for i, part in enumerate(lowered[:-1]):
        if part == "images" and i > 0 and lowered[i - 1] in TRAIN_TOKENS:
            if i + 1 < len(relative_path.parts) - 1:
                return relative_path.parts[i + 1]
    # Fallback: first directory after any train/training token.
    for i, part in enumerate(lowered[:-1]):
        if part in TRAIN_TOKENS and i + 1 < len(relative_path.parts) - 1:
            candidate = relative_path.parts[i + 1]
            if candidate.lower() != "images":
                return candidate
            if i + 2 < len(relative_path.parts) - 1:
                return relative_path.parts[i + 2]
    return "unstratified"


def crop_at(tensor: np.ndarray, y0: int, x0: int) -> np.ndarray:
    crop = tensor[:, :, y0 : y0 + TILE, x0 : x0 + TILE]
    return np.ascontiguousarray(crop, dtype=np.float32)


def random_position(height: int, width: int, rng: random.Random) -> tuple[int, int]:
    if height < TILE or width < TILE:
        raise RuntimeError(f"Processed image too small for {TILE}: {width}x{height}")
    y0 = 0 if height == TILE else rng.randint(0, height - TILE)
    x0 = 0 if width == TILE else rng.randint(0, width - TILE)
    return y0, x0


def signal_score(crop: np.ndarray) -> float:
    """Heuristic score for thin bright fluorescence signal after DNAi normalization.

    DNAi's ImageNet-normalized background is typically negative while bright
    red/green fiber pixels are positive. We deliberately combine moderate and
    high positive-pixel fractions with channel variability so calibration sees
    rare high-activation tracts without discarding ordinary background.
    """
    pixels = crop[0]
    pixel_max = pixels.max(axis=0)
    moderate = float(np.mean(pixel_max > 0.5))
    high = float(np.mean(pixel_max > 1.5))
    spread = float(np.std(pixels))
    return moderate + 2.0 * high + 0.05 * spread


def choose_signal_crop(
    tensor: np.ndarray,
    rng: random.Random,
    n_candidates: int,
) -> tuple[np.ndarray, int, int, float]:
    height, width = tensor.shape[-2:]
    candidates: list[tuple[float, int, int, np.ndarray]] = []

    # Include deterministic deployment-like anchor positions when possible.
    anchors_y = sorted({0, max(0, (height - TILE) // 2), max(0, height - TILE)})
    anchors_x = sorted({0, max(0, (width - TILE) // 2), max(0, width - TILE)})
    for y0 in anchors_y:
        for x0 in anchors_x:
            crop = crop_at(tensor, y0, x0)
            candidates.append((signal_score(crop), y0, x0, crop))

    # Add random candidates to avoid always favoring image centers/corners.
    for _ in range(max(0, n_candidates - len(candidates))):
        y0, x0 = random_position(height, width, rng)
        crop = crop_at(tensor, y0, x0)
        candidates.append((signal_score(crop), y0, x0, crop))

    score, y0, x0, crop = max(candidates, key=lambda item: item[0])
    return crop, y0, x0, score


def balanced_source_order(
    by_stratum: dict[str, list[Path]],
    rng: random.Random,
    count: int,
) -> list[tuple[str, Path]]:
    """Round-robin shuffled strata so no abundant subgroup dominates PTQ."""
    pools: dict[str, list[Path]] = {}
    for stratum, paths in sorted(by_stratum.items()):
        shuffled = paths[:]
        rng.shuffle(shuffled)
        pools[stratum] = shuffled

    ordered: list[tuple[str, Path]] = []
    strata = list(sorted(pools))
    offsets = {s: 0 for s in strata}
    while len(ordered) < count:
        progress = False
        for stratum in strata:
            idx = offsets[stratum]
            if idx < len(pools[stratum]):
                ordered.append((stratum, pools[stratum][idx]))
                offsets[stratum] += 1
                progress = True
                if len(ordered) >= count:
                    break
        if not progress:
            break
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    default_data_root = Path(
        os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data")
    )
    default_model_root = Path(
        os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models")
    )
    default_output = (
        default_model_root / "dnai" / "unet_mobileone_s1" / "calibration_512_v2"
    )
    parser.add_argument("--data-root", default=str(default_data_root))
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--count", type=int, default=192)
    parser.add_argument("--seed", type=int, default=7202)
    parser.add_argument("--signal-fraction", type=float, default=0.5)
    parser.add_argument("--signal-candidates", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if not 0.0 <= args.signal_fraction <= 1.0:
        raise SystemExit("--signal-fraction must be between 0 and 1")
    if args.signal_candidates <= 0:
        raise SystemExit("--signal-candidates must be positive")

    data_root = Path(args.data_root)
    dataset_root = data_root / "dnai_public_v2" / "extracted" / "zenodo"
    validation_manifest_path = data_root / "dnai_public_v2" / "r1_intergrader20" / "manifest.json"
    output_dir = Path(args.output)
    tensor_dir = output_dir / "tensors"

    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    validation_manifest = json.loads(validation_manifest_path.read_text(encoding="utf-8"))
    if validation_manifest.get("dnai_commit") != DNAI_COMMIT:
        raise RuntimeError("Manifest/code revision mismatch")
    validation_hashes = {record["source_sha256"] for record in validation_manifest["records"]}

    candidates: list[Path] = []
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
        raise RuntimeError("No public training-image candidates found")

    # Hash once up front so validation exclusion is explicit and auditable.
    eligible: list[tuple[Path, str]] = []
    excluded_by_validation_hash = 0
    for path in candidates:
        digest = sha256_file(path)
        if digest in validation_hashes:
            excluded_by_validation_hash += 1
            continue
        eligible.append((path, digest))

    if len(eligible) < args.count:
        raise RuntimeError(
            f"Need {args.count} unique non-validation source images, found {len(eligible)}"
        )

    hash_by_path = {path: digest for path, digest in eligible}
    by_stratum: dict[str, list[Path]] = defaultdict(list)
    for path, _digest in eligible:
        rel = path.relative_to(dataset_root)
        by_stratum[infer_stratum(rel)].append(path)

    rng = random.Random(args.seed)
    source_order = balanced_source_order(by_stratum, rng, args.count)
    if len(source_order) < args.count:
        raise RuntimeError(f"Balanced selection produced only {len(source_order)} sources")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise RuntimeError(f"Output is not empty: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    n_signal_target = int(round(args.count * args.signal_fraction))
    modes = ["signal"] * n_signal_target + ["random"] * (args.count - n_signal_target)
    # Shuffle modes independently so each stratum receives a mixture over the
    # round-robin source order rather than all early sources using one mode.
    mode_rng = random.Random(args.seed + 1)
    mode_rng.shuffle(modes)

    print("Radiation Edge AI - DNAi INT8 calibration v2")
    print(f"Dataset: {dataset_root}")
    print(f"Training-image candidates: {len(candidates)}")
    print(f"Eligible unique non-validation images: {len(eligible)}")
    print(f"Validation-source hashes excluded: {excluded_by_validation_hash}")
    print(f"Requested tensors: {args.count}")
    print(f"Signal-rich fraction: {args.signal_fraction:.2f}")
    print(f"Signal candidate crops/source: {args.signal_candidates}")
    print(f"Seed: {args.seed}")
    print("Observed strata: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(by_stratum.items())))
    print("Preprocessing: exact DNAi load_image(pixel_size=0.26, clarity=1.0, reverse_channel=False) + transform")
    print("Frozen inter-grader20 used for calibration: NO")
    print("")

    records = []
    stratum_counts: dict[str, int] = defaultdict(int)
    mode_counts: dict[str, int] = defaultdict(int)

    for index, ((stratum, source), mode) in enumerate(zip(source_order, modes)):
        image = load_image(
            source,
            reverse_channel=False,
            pixel_size=PIXEL_SIZE_UM,
            verbose=False,
            clarity=CLARITY,
        )
        tensor = (
            transform(image=image)["image"]
            .unsqueeze(0)
            .numpy()
            .astype(np.float32, copy=False)
        )
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
            raise RuntimeError(f"Expected NCHW 1x3xHxW, got {tensor.shape}: {source}")
        height, width = tensor.shape[-2:]
        if height < TILE or width < TILE:
            raise RuntimeError(f"Processed image too small: {tensor.shape}: {source}")

        if mode == "signal":
            crop, y0, x0, score = choose_signal_crop(
                tensor, rng, args.signal_candidates
            )
        else:
            y0, x0 = random_position(height, width, rng)
            crop = crop_at(tensor, y0, x0)
            score = signal_score(crop)

        tensor_name = f"cal_v2_{index:03d}.npy"
        tensor_path = tensor_dir / tensor_name
        np.save(tensor_path, crop, allow_pickle=False)
        rel = source.relative_to(dataset_root).as_posix()
        source_hash = hash_by_path[source]

        record = {
            "index": index,
            "source": rel,
            "source_sha256": source_hash,
            "stratum": stratum,
            "selection_mode": mode,
            "signal_score": score,
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
        records.append(record)
        stratum_counts[stratum] += 1
        mode_counts[mode] += 1
        print(
            f"[{index + 1:03d}/{args.count:03d}] {stratum:<12} {mode:<6} "
            f"score={score:.5f} {rel} crop=({y0},{x0})"
        )

    payload = {
        "dnai_commit": DNAI_COMMIT,
        "purpose": (
            "KL720 INT8 PTQ v2 calibration; stratified unique public-training "
            "sources with mixed random/fiber-rich crops; frozen inter-grader20 excluded"
        ),
        "calibration_version": 2,
        "input_name": "input",
        "tile": [1, 3, TILE, TILE],
        "pixel_size_um": PIXEL_SIZE_UM,
        "clarity": CLARITY,
        "reverse_channel": False,
        "seed": args.seed,
        "n_training_candidates": len(candidates),
        "n_eligible_nonvalidation_sources": len(eligible),
        "n_tensors": len(records),
        "unique_source_images": len({r["source_sha256"] for r in records}),
        "signal_fraction_requested": args.signal_fraction,
        "signal_candidate_crops": args.signal_candidates,
        "selection_mode_counts": dict(sorted(mode_counts.items())),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "validation_panel_used_for_calibration": False,
        "records": records,
    }
    manifest_out = output_dir / "manifest.json"
    manifest_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("")
    print("INT8 CALIBRATION V2 TENSORS READY: YES")
    print(f"Tensors: {len(records)}")
    print(f"Unique source images: {payload['unique_source_images']}")
    print("Strata: " + ", ".join(f"{k}={v}" for k, v in payload["stratum_counts"].items()))
    print("Modes: " + ", ".join(f"{k}={v}" for k, v in payload["selection_mode_counts"].items()))
    print(f"Tensor directory: {tensor_dir}")
    print(f"Manifest: {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
