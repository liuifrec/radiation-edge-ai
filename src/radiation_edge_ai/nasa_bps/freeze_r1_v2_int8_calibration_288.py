"""Freeze and verify the 288-tensor NASA BPS R1 v2 INT8 calibration panel.

This performs no model inference and reads no final-holdout data. It pins the
observed calibration manifest SHA256, verifies every recorded tensor SHA256 and
shape/dtype, then writes one immutable calibration-freeze JSON for Kneron PTQ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

FROZEN_CALIBRATION_MANIFEST_SHA256 = (
    "78c9ed83e1992907264dfeec9ef6e2a434868bd3f299672f610a42a0018b5d4e"
)
FROZEN_ONNX_SHA256 = (
    "a65718a8f07d5bcd8707bae78d8f730d56047db21e575fa55ba5396adc49a07e"
)
FROZEN_ONNX_FREEZE_SHA256 = (
    "e31c64050244ea93cbb3ea57433e0f281218bbbd352ee66d058e98bb3e36fe06"
)
REPAIRED_DEV_MANIFEST_SHA256 = (
    "23e21fdc8bf00a2814b64c56223cfd3d8e96eda71bdbff5d8ec74dae218cf5a9"
)
EXPECTED_TENSORS = 288
EXPECTED_BAGS = 72
EXPECTED_PER_BAG = 4
EXPECTED_SHAPE = (1, 3, 256, 256)
EXPECTED_SOURCE_COUNTS = {
    "BALBCF1": 48,
    "BALBCM1": 48,
    "BALBCM2": 48,
    "C57BLF1": 48,
    "C57BLM1": 48,
    "C57BLM3": 48,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return obj


def main() -> int:
    parser = argparse.ArgumentParser()
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    deployment_root = model_root / "nasa_bps_53bp1" / "r1_v2_deployment"
    parser.add_argument(
        "--manifest",
        default=str(deployment_root / "int8_calibration_288" / "manifest.json"),
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    calibration_dir = manifest_path.parent
    freeze_path = calibration_dir / "int8_calibration_freeze.json"

    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != FROZEN_CALIBRATION_MANIFEST_SHA256:
        raise RuntimeError(
            "Calibration manifest SHA256 mismatch: "
            f"expected {FROZEN_CALIBRATION_MANIFEST_SHA256}, got {manifest_sha}"
        )
    manifest = read_json(manifest_path)

    checks = {
        "status": "FROZEN_DEVELOPMENT_ONLY_INT8_CALIBRATION_PANEL",
        "repaired_development_manifest_sha256": REPAIRED_DEV_MANIFEST_SHA256,
        "frozen_onnx_sha256": FROZEN_ONNX_SHA256,
        "onnx_deployment_candidate_freeze_sha256": FROZEN_ONNX_FREEZE_SHA256,
        "selection_seed": 7203,
        "development_nuclei": 7200,
        "development_bags": EXPECTED_BAGS,
        "development_nuclei_per_bag": 100,
        "selected_per_bag": EXPECTED_PER_BAG,
        "n_tensors": EXPECTED_TENSORS,
        "final_holdout_images_used_for_calibration": False,
        "final_holdout_outcomes_used_for_calibration": False,
        "phenotype_values_used_for_selection": False,
        "model_predictions_used_for_selection": False,
        "image_intensity_used_for_selection": False,
        "mask_used_as_model_input": False,
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"Calibration manifest invariant mismatch for {key}: "
                f"expected {expected!r}, got {manifest.get(key)!r}"
            )
    if tuple(manifest.get("tile", ())) != EXPECTED_SHAPE:
        raise RuntimeError(f"Unexpected calibration tile: {manifest.get('tile')!r}")
    if manifest.get("dtype") != "float32":
        raise RuntimeError(f"Unexpected calibration dtype: {manifest.get('dtype')!r}")
    if manifest.get("source_counts") != EXPECTED_SOURCE_COUNTS:
        raise RuntimeError(f"Unexpected calibration source counts: {manifest.get('source_counts')!r}")

    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_TENSORS:
        raise RuntimeError(f"Calibration records={len(records) if isinstance(records, list) else None}; expected {EXPECTED_TENSORS}")
    if len({record.get("sample_id") for record in records}) != EXPECTED_TENSORS:
        raise RuntimeError("Calibration sample_id values are not unique")
    bag_counts: dict[str, int] = {}
    tensor_identities: list[dict[str, Any]] = []
    global_min = float("inf")
    global_max = float("-inf")

    for index, record in enumerate(records):
        sample_name = str(record.get("sample_name", ""))
        bag_counts[sample_name] = bag_counts.get(sample_name, 0) + 1
        tensor_rel = record.get("tensor")
        if not tensor_rel:
            raise RuntimeError(f"Calibration record {index} has no tensor path")
        tensor_path = calibration_dir / str(tensor_rel)
        if not tensor_path.is_file():
            raise FileNotFoundError(tensor_path)
        tensor_sha = sha256_file(tensor_path)
        if tensor_sha != record.get("tensor_sha256"):
            raise RuntimeError(
                f"Calibration tensor SHA256 mismatch for {tensor_path}: "
                f"manifest={record.get('tensor_sha256')} actual={tensor_sha}"
            )
        array = np.load(tensor_path, allow_pickle=False)
        if array.shape != EXPECTED_SHAPE:
            raise RuntimeError(f"Unexpected calibration tensor shape {array.shape}: {tensor_path}")
        if array.dtype != np.float32:
            raise RuntimeError(f"Unexpected calibration tensor dtype {array.dtype}: {tensor_path}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"Non-finite calibration tensor: {tensor_path}")
        if not np.all(array[:, 2, :, :] == 0.0):
            raise RuntimeError(f"Third calibration channel is not zero: {tensor_path}")
        amin = float(array.min())
        amax = float(array.max())
        global_min = min(global_min, amin)
        global_max = max(global_max, amax)
        tensor_identities.append(
            {
                "index": index,
                "sample_id": record.get("sample_id"),
                "sample_name": sample_name,
                "tensor": str(tensor_rel),
                "sha256": tensor_sha,
                "size_bytes": tensor_path.stat().st_size,
            }
        )

    if len(bag_counts) != EXPECTED_BAGS or set(bag_counts.values()) != {EXPECTED_PER_BAG}:
        raise RuntimeError(
            f"Calibration bag representation is not exactly {EXPECTED_PER_BAG} per {EXPECTED_BAGS} bags: {bag_counts}"
        )

    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_VERIFIED_DEVELOPMENT_ONLY_INT8_CALIBRATION_PANEL",
        "calibration_manifest_sha256": manifest_sha,
        "repaired_development_manifest_sha256": REPAIRED_DEV_MANIFEST_SHA256,
        "onnx_sha256": FROZEN_ONNX_SHA256,
        "onnx_deployment_candidate_freeze_sha256": FROZEN_ONNX_FREEZE_SHA256,
        "selection_seed": 7203,
        "n_tensors": EXPECTED_TENSORS,
        "bags": EXPECTED_BAGS,
        "selected_per_bag": EXPECTED_PER_BAG,
        "shape": list(EXPECTED_SHAPE),
        "dtype": "float32",
        "source_counts": EXPECTED_SOURCE_COUNTS,
        "global_min": global_min,
        "global_max": global_max,
        "tensor_identities": tensor_identities,
        "all_tensor_hashes_verified": True,
        "final_holdout_used_for_calibration": False,
        "phenotype_or_prediction_guided_selection": False,
        "authorized_ptq_config": {
            "platform": "720",
            "datapath_range_method": "percentage",
            "percentage": 0.999,
            "percentage_16b": 0.999999,
            "optimize": 0,
        },
    }

    if freeze_path.exists():
        existing = read_json(freeze_path)
        old = dict(existing)
        new = dict(freeze)
        old.pop("created_utc", None)
        new.pop("created_utc", None)
        if old != new:
            raise RuntimeError(f"Existing calibration freeze differs from verified tensors: {freeze_path}")
        print("Radiation Edge AI - NASA BPS R1 v2 INT8 calibration freeze")
        print("existing calibration freeze verified without overwrite: YES")
    else:
        freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
        print("Radiation Edge AI - NASA BPS R1 v2 INT8 calibration freeze")
        print("new calibration freeze written: YES")

    print(f"calibration manifest SHA256: {manifest_sha}")
    print(f"tensor hashes verified: {EXPECTED_TENSORS}/{EXPECTED_TENSORS}")
    print(f"bags represented: {len(bag_counts)}/{EXPECTED_BAGS}; per bag={EXPECTED_PER_BAG}")
    print(f"global tensor range: [{global_min:.6f}, {global_max:.6f}]")
    print("final holdout used for calibration: NO")
    print(f"freeze: {freeze_path}")
    print(f"freeze SHA256: {sha256_file(freeze_path)}")
    print("NASA BPS R1 V2 INT8 CALIBRATION FROZEN: YES")
    print("NEXT GATE: run the frozen ONNX through Kneron optimizer/IP evaluation before PTQ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
