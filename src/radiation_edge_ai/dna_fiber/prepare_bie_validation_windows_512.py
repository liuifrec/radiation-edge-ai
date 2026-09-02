"""Prepare the frozen DNAi validation windows for KL720 BIE inference.

Run in the DNAi Windows environment. The frozen 20-image inter-grader panel is
preprocessed exactly as DNAi expects and split into the deployment policy that
was selected before quantization: 512x512 windows, 50% overlap. For a
1024x1024 image this yields 9 normalized float32 NCHW inputs.

These tensors are validation-only and must never be fed back into PTQ analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from dnafiber.data.utils import load_image
from dnafiber.inference import transform

DNAI_COMMIT = "fcf20c7d6eb385675ff7d07da4fdf471589ce0cf"
PIXEL_SIZE_UM = 0.26
CLARITY = 1.0
TILE = 512
OVERLAP = 0.50
STRIDE = int(TILE * (1.0 - OVERLAP))
EXPECTED_IMAGE = (1, 3, 1024, 1024)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def starts(length: int) -> list[int]:
    if length < TILE:
        raise RuntimeError(f"Dimension {length} is smaller than tile {TILE}")
    positions = list(range(0, max(length - TILE + 1, 1), STRIDE))
    last = length - TILE
    if positions[-1] != last:
        positions.append(last)
    return positions


def main() -> int:
    parser = argparse.ArgumentParser()
    default_data_root = Path(
        os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data")
    )
    default_model_root = Path(
        os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models")
    )
    default_output = (
        default_model_root
        / "dnai"
        / "unet_mobileone_s1"
        / "kneron"
        / "int8_512"
        / "validation_512_overlap50"
    )
    parser.add_argument("--data-root", default=str(default_data_root))
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    dataset_root = data_root / "dnai_public_v2" / "extracted" / "zenodo"
    panel_root = data_root / "dnai_public_v2" / "r1_intergrader20"
    manifest_path = panel_root / "manifest.json"
    output_root = Path(args.output)
    input_root = output_root / "inputs"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dnai_commit") != DNAI_COMMIT:
        raise RuntimeError("Manifest/code revision mismatch")
    records = manifest["records"]
    if args.limit is not None:
        records = records[: args.limit]

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output is not empty: {output_root}; use --overwrite")
    output_root.mkdir(parents=True, exist_ok=True)
    input_root.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in input_root.glob("*.npy"):
            path.unlink()

    print("Radiation Edge AI - frozen DNAi BIE validation window preparation")
    print(f"Images: {len(records)}")
    print("Policy: 512x512, 50% overlap, stride=256")
    print("Preprocessing: DNAi load_image + ImageNet normalization")
    print("Role: validation only; excluded from calibration")
    print("")

    out_records = []
    for image_index, record in enumerate(records):
        source = dataset_root / record["source"]
        source_hash = sha256_file(source)
        if source_hash != record["source_sha256"]:
            raise RuntimeError(f"Source checksum mismatch: {source}")

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
        if tuple(tensor.shape) != EXPECTED_IMAGE:
            raise RuntimeError(f"Expected {EXPECTED_IMAGE}, got {tensor.shape}: {source}")

        y_positions = starts(tensor.shape[-2])
        x_positions = starts(tensor.shape[-1])
        window_index = 0
        for y0 in y_positions:
            for x0 in x_positions:
                patch = np.ascontiguousarray(
                    tensor[:, :, y0 : y0 + TILE, x0 : x0 + TILE],
                    dtype=np.float32,
                )
                name = f"{image_index:02d}_{window_index:02d}_{record['sample_id']}.npy"
                path = input_root / name
                np.save(path, patch, allow_pickle=False)
                out_records.append(
                    {
                        "image_index": image_index,
                        "window_index": window_index,
                        "sample_id": record["sample_id"],
                        "key": record["key"],
                        "source": record["source"],
                        "source_sha256": source_hash,
                        "y": y0,
                        "x": x0,
                        "tensor": f"inputs/{name}",
                        "shape": list(patch.shape),
                        "dtype": str(patch.dtype),
                    }
                )
                window_index += 1

        if window_index != 9:
            raise RuntimeError(
                f"Expected 9 deployment windows for {record['sample_id']}, got {window_index}"
            )
        print(f"[{image_index + 1:02d}/{len(records):02d}] {record['sample_id']}: 9 windows")

    payload = {
        "dnai_commit": DNAI_COMMIT,
        "source_manifest": str(manifest_path),
        "purpose": "frozen inter-grader20 BIE validation only; never calibration",
        "input_name": "input",
        "tile": [1, 3, TILE, TILE],
        "full_image_shape": list(EXPECTED_IMAGE),
        "overlap": OVERLAP,
        "stride": STRIDE,
        "blend": "gaussian",
        "pixel_size_um": PIXEL_SIZE_UM,
        "clarity": CLARITY,
        "reverse_channel": False,
        "n_images": len(records),
        "n_windows": len(out_records),
        "windows_per_image": 9,
        "records": out_records,
    }
    out_manifest = output_root / "validation_windows_manifest.json"
    out_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("")
    print("BIE VALIDATION WINDOWS READY: YES")
    print(f"Images: {len(records)}")
    print(f"Windows: {len(out_records)}")
    print(f"Inputs: {input_root}")
    print(f"Manifest: {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
