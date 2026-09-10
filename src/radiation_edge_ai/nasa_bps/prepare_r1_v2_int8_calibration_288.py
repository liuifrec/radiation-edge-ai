"""Materialize the frozen 288-nucleus NASA BPS R1 v2 INT8 calibration panel.

Selection is deterministic, phenotype-blind, intensity-blind, and uses only the
7,200-nucleus repaired development manifest: four nuclei from each of 72 bags,
ranked by SHA256('7203|<sample_id>|<nucleus_key>').

The final holdout is never read. Each selected nucleus is transformed with the
unchanged R1 preprocessing into float32 NCHW [1,3,256,256] and saved as .npy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from radiation_edge_ai.nasa_bps import train_pilot_v1_r1_mil as v1

REPAIRED_DEV_MANIFEST_SHA256 = (
    "23e21fdc8bf00a2814b64c56223cfd3d8e96eda71bdbff5d8ec74dae218cf5a9"
)
FROZEN_ONNX_SHA256 = (
    "a65718a8f07d5bcd8707bae78d8f730d56047db21e575fa55ba5396adc49a07e"
)
SEED = 7203
EXPECTED_NUCLEI = 7200
EXPECTED_BAGS = 72
EXPECTED_PER_BAG = 100
SELECT_PER_BAG = 4
EXPECTED_SELECTED = EXPECTED_BAGS * SELECT_PER_BAG
EXPECTED_SOURCES = {
    "BALBCF1": 12,
    "BALBCM1": 12,
    "BALBCM2": 12,
    "C57BLF1": 12,
    "C57BLM1": 12,
    "C57BLM3": 12,
}
FORBIDDEN_TOKENS = (
    "nfoci",
    "phenotype",
    "reference",
    "prediction",
    "predicted",
    "residual",
    "ground_truth",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def selection_digest(row: dict[str, str]) -> str:
    payload = f"{SEED}|{row['sample_id']}|{row['nucleus_key']}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_manifest(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
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
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise RuntimeError(f"Development manifest missing columns: {missing}")
    forbidden = sorted(
        column
        for column in rows[0]
        if any(token in column.lower() for token in FORBIDDEN_TOKENS)
    )
    if forbidden:
        raise RuntimeError(
            "Development manifest unexpectedly contains outcome/model columns: "
            + ", ".join(forbidden)
        )
    if len(rows) != EXPECTED_NUCLEI:
        raise RuntimeError(f"Development nuclei={len(rows)}; expected {EXPECTED_NUCLEI}")
    if len({row['sample_id'] for row in rows}) != EXPECTED_NUCLEI:
        raise RuntimeError("Duplicate sample_id in development manifest")
    if len({row['nucleus_key'] for row in rows}) != EXPECTED_NUCLEI:
        raise RuntimeError("Duplicate nucleus_key in development manifest")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["sample_name"]].append(row)
    if len(grouped) != EXPECTED_BAGS:
        raise RuntimeError(f"Development bags={len(grouped)}; expected {EXPECTED_BAGS}")
    for sample_name, subset in grouped.items():
        if len(subset) != EXPECTED_PER_BAG:
            raise RuntimeError(
                f"Development bag {sample_name!r} has {len(subset)} nuclei; expected {EXPECTED_PER_BAG}"
            )
        invariant = (
            "source_name",
            "strain",
            "sex",
            "particle_type",
            "dose_Gy",
            "hr_post_exposure",
        )
        for field in invariant:
            values = {row[field] for row in subset}
            if len(values) != 1:
                raise RuntimeError(
                    f"Development bag {sample_name!r} is not invariant for {field}: {sorted(values)}"
                )

    source_bags = Counter(subset[0]["source_name"] for subset in grouped.values())
    if dict(sorted(source_bags.items())) != dict(sorted(EXPECTED_SOURCES.items())):
        raise RuntimeError(
            f"Unexpected development source/bag structure: {dict(source_bags)}"
        )
    return grouped


def verify_onnx_freeze(path: Path, onnx_path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(
            f"ONNX deployment candidate is not frozen yet: {path}. "
            "Run freeze_r1_v2_onnx_deployment_candidate first."
        )
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("status") != "FROZEN_ONNX_DEPLOYMENT_CANDIDATE_AFTER_FP32_PARITY_PASS":
        raise RuntimeError(f"Unexpected ONNX freeze status: {obj.get('status')!r}")
    if obj.get("onnx_sha256") != FROZEN_ONNX_SHA256:
        raise RuntimeError("ONNX freeze points to a different ONNX model")
    actual_onnx = sha256_file(onnx_path)
    if actual_onnx != FROZEN_ONNX_SHA256:
        raise RuntimeError(
            f"ONNX SHA256 mismatch: expected {FROZEN_ONNX_SHA256}, got {actual_onnx}"
        )
    return sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"
    deployment_root = model_root / "nasa_bps_53bp1" / "r1_v2_deployment"
    onnx_dir = deployment_root / "onnx"
    parser.add_argument(
        "--manifest",
        default=str(metadata_root / "r1_v2_freeze" / "r1_v2_development_manifest_100_qc1.csv"),
    )
    parser.add_argument(
        "--image-root",
        default=str(data_root / "nasa_bps_microscopy" / "r1_v2_development" / "images"),
    )
    parser.add_argument(
        "--onnx",
        default=str(onnx_dir / "r1_countnet_v2_final_fp32_1x3x256x256_opset11.onnx"),
    )
    parser.add_argument(
        "--onnx-freeze",
        default=str(onnx_dir / "onnx_deployment_candidate_freeze.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(deployment_root / "int8_calibration_288"),
    )
    parser.add_argument("--cache-items", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    image_root = Path(args.image_root).resolve()
    onnx_path = Path(args.onnx).resolve()
    onnx_freeze_path = Path(args.onnx_freeze).resolve()
    output_dir = Path(args.output_dir).resolve()
    tensor_dir = output_dir / "tensors"

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != REPAIRED_DEV_MANIFEST_SHA256:
        raise RuntimeError(
            f"Repaired development manifest SHA256 mismatch: expected {REPAIRED_DEV_MANIFEST_SHA256}, got {manifest_sha}"
        )
    onnx_freeze_sha = verify_onnx_freeze(onnx_freeze_path, onnx_path)

    rows = read_csv(manifest_path)
    grouped = validate_manifest(rows)

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise RuntimeError(f"Calibration output is not empty: {output_dir}; refusing overwrite")
        shutil.rmtree(output_dir)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    selected: list[tuple[dict[str, str], str, int]] = []
    for sample_name, subset in sorted(grouped.items()):
        ranked = sorted(
            ((selection_digest(row), row) for row in subset),
            key=lambda item: (item[0], item[1]["sample_id"]),
        )
        for rank, (digest, row) in enumerate(ranked[:SELECT_PER_BAG], start=1):
            selected.append((row, digest, rank))

    if len(selected) != EXPECTED_SELECTED:
        raise RuntimeError(f"Selected calibration nuclei={len(selected)}; expected {EXPECTED_SELECTED}")
    selected_counts = Counter(row["sample_name"] for row, _digest, _rank in selected)
    if set(selected_counts.values()) != {SELECT_PER_BAG} or len(selected_counts) != EXPECTED_BAGS:
        raise RuntimeError("Calibration selection is not exactly four nuclei per development bag")

    print("Radiation Edge AI - NASA BPS R1 v2 INT8 calibration materializer")
    print(f"development manifest SHA256: {manifest_sha}")
    print(f"frozen ONNX SHA256: {FROZEN_ONNX_SHA256}")
    print(f"ONNX freeze SHA256: {onnx_freeze_sha}")
    print(f"development nuclei/bags: {EXPECTED_NUCLEI}/{EXPECTED_BAGS}")
    print(f"selection: {SELECT_PER_BAG} nuclei per bag -> {EXPECTED_SELECTED} tensors")
    print(f"selection seed: {SEED}")
    print("selection uses phenotype values: NO")
    print("selection uses model predictions: NO")
    print("selection uses image intensity: NO")
    print("final holdout used for calibration: NO")
    print("preprocessing: frozen p1/p99.5 FITC+DAPI normalization + native-scale center pad + zero third channel")

    cache = v1.NativeImageCache(image_root=image_root, max_items=args.cache_items)
    records: list[dict[str, Any]] = []
    global_min = float("inf")
    global_max = float("-inf")
    source_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()

    for index, (row, digest, rank) in enumerate(selected):
        packed = v1.pad_and_pack(cache.get(row)).astype(np.float32, copy=False)
        tensor = np.ascontiguousarray(packed[None, ...], dtype=np.float32)
        if tensor.shape != (1, 3, 256, 256):
            raise RuntimeError(f"Unexpected calibration tensor shape {tensor.shape}: {row['sample_id']}")
        if tensor.dtype != np.float32 or not np.isfinite(tensor).all():
            raise RuntimeError(f"Invalid calibration tensor: {row['sample_id']}")
        if not np.all(tensor[:, 2, :, :] == 0.0):
            raise RuntimeError(f"Third calibration channel is not identically zero: {row['sample_id']}")

        name = f"r1v2_cal_{index:03d}.npy"
        path = tensor_dir / name
        np.save(path, tensor, allow_pickle=False)
        tensor_sha = sha256_file(path)
        global_min = min(global_min, float(tensor.min()))
        global_max = max(global_max, float(tensor.max()))
        source_counts[row["source_name"]] += 1
        condition_key = "|".join(
            (
                row["particle_type"],
                row["dose_Gy"],
                row["hr_post_exposure"],
            )
        )
        condition_counts[condition_key] += 1

        records.append(
            {
                "index": index,
                "sample_id": row["sample_id"],
                "nucleus_key": row["nucleus_key"],
                "source_name": row["source_name"],
                "sample_name": row["sample_name"],
                "strain": row["strain"],
                "sex": row["sex"],
                "particle_type": row["particle_type"],
                "dose_Gy": row["dose_Gy"],
                "hr_post_exposure": row["hr_post_exposure"],
                "selection_rank_within_bag": rank,
                "selection_sha256": digest,
                "tensor": f"tensors/{name}",
                "tensor_sha256": tensor_sha,
                "tensor_shape": list(tensor.shape),
                "tensor_dtype": str(tensor.dtype),
                "tensor_min": float(tensor.min()),
                "tensor_max": float(tensor.max()),
                "tensor_mean": float(tensor.mean()),
                "tensor_std": float(tensor.std()),
            }
        )
        if (index + 1) % 36 == 0 or index + 1 == EXPECTED_SELECTED:
            print(f"calibration tensors: {index + 1}/{EXPECTED_SELECTED}")

    expected_source_nuclei = {source: bags * SELECT_PER_BAG for source, bags in EXPECTED_SOURCES.items()}
    if dict(sorted(source_counts.items())) != dict(sorted(expected_source_nuclei.items())):
        raise RuntimeError(f"Unexpected calibration source counts: {dict(source_counts)}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_DEVELOPMENT_ONLY_INT8_CALIBRATION_PANEL",
        "purpose": "KL720 post-training quantization calibration for NASA BPS R1 v2 53BP1 burden regressor",
        "repaired_development_manifest_sha256": manifest_sha,
        "frozen_onnx_sha256": FROZEN_ONNX_SHA256,
        "onnx_deployment_candidate_freeze_sha256": onnx_freeze_sha,
        "input_name": "input",
        "tile": [1, 3, 256, 256],
        "dtype": "float32",
        "selection_seed": SEED,
        "selection_rule": "four per exact sample_name bag; rank by SHA256('7203|<sample_id>|<nucleus_key>'); choose first four",
        "development_nuclei": EXPECTED_NUCLEI,
        "development_bags": EXPECTED_BAGS,
        "development_nuclei_per_bag": EXPECTED_PER_BAG,
        "selected_per_bag": SELECT_PER_BAG,
        "n_tensors": len(records),
        "source_counts": dict(sorted(source_counts.items())),
        "condition_counts": dict(sorted(condition_counts.items())),
        "global_min": global_min,
        "global_max": global_max,
        "preprocessing": "R1 frozen p1/p99.5 FITC+DAPI normalization; no resize; native-scale center pad to 256; zero third channel",
        "mask_used_as_model_input": False,
        "phenotype_values_used_for_selection": False,
        "model_predictions_used_for_selection": False,
        "image_intensity_used_for_selection": False,
        "final_holdout_images_used_for_calibration": False,
        "final_holdout_outcomes_used_for_calibration": False,
        "records": records,
    }
    manifest_path_out = output_dir / "manifest.json"
    manifest_path_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("")
    print("NASA BPS R1 V2 DEVELOPMENT-ONLY INT8 CALIBRATION READY: YES")
    print(f"tensors: {len(records)}")
    print(f"bags represented: {len(selected_counts)}/{EXPECTED_BAGS}")
    print(f"per bag: {SELECT_PER_BAG}")
    print("source counts: " + ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items())))
    print(f"global tensor range: [{global_min:.6f}, {global_max:.6f}]")
    print(f"tensor directory: {tensor_dir}")
    print(f"manifest: {manifest_path_out}")
    print(f"manifest SHA256: {sha256_file(manifest_path_out)}")
    print("final holdout used for calibration: NO")
    print("NEXT GATE: run KL720 floating-point optimization/IP evaluation on the frozen ONNX, then quantize with this exact calibration manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
