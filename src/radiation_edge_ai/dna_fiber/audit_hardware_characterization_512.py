"""Read-only completion-state audit for the physical DNAi 512 panel.

This performs no inference and never overwrites the historical hardware panel
or biological-characterization directories.  It inventories frozen inputs and
physical outputs by manifest identity, hashes every existing array, and reports
whether a prior hardware biological-fidelity summary already exists.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from radiation_edge_ai.audit_utils import (
    read_json_object,
    refuse_nonempty_output_dir,
    sha256_file,
    write_csv_rows,
)

EXPECTED_NEF_SHA256 = "4b3dfec9a61c99e186dd4b8482fa5b06e6a4958f325ed4a0db0546f1dcab2bfc"
EXPECTED_ONNX_SHA256 = "a901d1b309a9a0e5026febd5070252787e4b110a2a7d2828e16a19364d6094d0"
EXPECTED_WINDOWS = 180
EXPECTED_IMAGES = 20
EXPECTED_WINDOWS_PER_IMAGE = 9
EXPECTED_SHAPE = (1, 3, 512, 512)


def stem_for(record: dict[str, Any]) -> str:
    return (
        f"{int(record['image_index']):02d}_{int(record['window_index']):02d}_{record['sample_id']}"
    )


def inspect_array(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        return {
            "exists": True,
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
            "sha256": sha256_file(path),
        }
    valid = (
        tuple(array.shape) == EXPECTED_SHAPE
        and array.dtype == np.float32
        and bool(np.isfinite(array).all())
    )
    return {
        "exists": True,
        "valid": bool(valid),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    kneron_root = model_root / "dnai" / "unet_mobileone_s1" / "kneron"
    v3_root = kneron_root / "int8_512_v3_pct100"
    parser.add_argument(
        "--manifest",
        default=str(
            kneron_root
            / "int8_512"
            / "validation_512_overlap50"
            / "validation_windows_manifest.json"
        ),
    )
    parser.add_argument("--hardware-panel", default=str(v3_root / "hardware_panel"))
    parser.add_argument(
        "--optimized-onnx",
        default=str(kneron_root / "unet_mobileone_s1_512x512_opset11_optimized.onnx"),
    )
    parser.add_argument(
        "--characterization-dir", default=str(v3_root / "hardware_biological_fidelity")
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    panel_root = Path(args.hardware_panel).resolve()
    physical_dir = panel_root / "physical_outputs"
    summary_path = panel_root / "summary.json"
    onnx_path = Path(args.optimized_onnx).resolve()
    characterization_dir = Path(args.characterization_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    refuse_nonempty_output_dir(output_dir)

    required = [
        {
            "label": "validation manifest",
            "path": str(manifest_path),
            "exists": manifest_path.is_file(),
        },
        {
            "label": "hardware panel summary",
            "path": str(summary_path),
            "exists": summary_path.is_file(),
        },
        {
            "label": "physical output directory",
            "path": str(physical_dir),
            "exists": physical_dir.is_dir(),
        },
        {
            "label": "optimized ONNX",
            "path": str(onnx_path),
            "exists": onnx_path.is_file(),
            "expected_sha256": EXPECTED_ONNX_SHA256,
        },
    ]
    missing = [row for row in required if not row["exists"]]
    if missing:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "BLOCKED_MISSING_LOCAL_DNAI_ARTIFACTS",
            "required_artifacts": required,
            "instruction": "Restore or point the audit at these existing frozen artifacts; do not regenerate them from this audit.",
        }
        (output_dir / "missing_artifacts.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print("DNAi completion-state audit: BLOCKED - missing local artifacts")
        for row in missing:
            print(f"MISSING: {row['label']}: {row['path']}")
        print(f"actionable inventory: {output_dir / 'missing_artifacts.json'}")
        return 2

    onnx_sha = sha256_file(onnx_path)
    if onnx_sha.lower() != EXPECTED_ONNX_SHA256:
        raise RuntimeError(f"Optimized ONNX SHA mismatch: {onnx_sha}")
    summary = read_json_object(summary_path)
    if str(summary.get("nef_sha256", "")).lower() != EXPECTED_NEF_SHA256:
        raise RuntimeError(f"Hardware panel NEF SHA mismatch: {summary.get('nef_sha256')}")
    if summary.get("complete") is not True:
        raise RuntimeError("Hardware panel summary is not complete")

    manifest = read_json_object(manifest_path)
    records = list(manifest.get("records") or [])
    if len(records) != EXPECTED_WINDOWS:
        raise RuntimeError(f"Expected {EXPECTED_WINDOWS} manifest windows, got {len(records)}")
    if tuple(manifest.get("tile") or ()) != EXPECTED_SHAPE:
        raise RuntimeError(f"Manifest tile differs from {EXPECTED_SHAPE}: {manifest.get('tile')}")
    if float(manifest.get("overlap", -1)) != 0.5:
        raise RuntimeError("Manifest overlap is not frozen 0.5")

    image_counts = Counter(int(row["image_index"]) for row in records)
    if len(image_counts) != EXPECTED_IMAGES or set(image_counts.values()) != {
        EXPECTED_WINDOWS_PER_IMAGE
    }:
        raise RuntimeError(f"Manifest is not 20 x 9 windows: {dict(image_counts)}")

    validation_root = manifest_path.parent
    inventory: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for record in sorted(
        records, key=lambda row: (int(row["image_index"]), int(row["window_index"]))
    ):
        for field in ("sample_id", "tensor", "key", "image_index", "window_index"):
            if field not in record:
                raise RuntimeError(f"Manifest record missing {field}")
        stem = stem_for(record)
        input_path = validation_root / str(record["tensor"])
        output_path = physical_dir / f"{stem}.npy"
        input_meta = inspect_array(input_path)
        output_meta = inspect_array(output_path)
        sample_ids.add(str(record["sample_id"]))
        inventory.append(
            {
                "image_index": int(record["image_index"]),
                "window_index": int(record["window_index"]),
                "sample_id": str(record["sample_id"]),
                "key": str(record["key"]),
                "input_tensor": str(input_path),
                "input_exists": input_meta.get("exists"),
                "input_valid": input_meta.get("valid"),
                "input_sha256": input_meta.get("sha256"),
                "physical_output": str(output_path),
                "physical_exists": output_meta.get("exists"),
                "physical_valid": output_meta.get("valid"),
                "physical_sha256": output_meta.get("sha256"),
            }
        )

    missing_input = [row for row in inventory if not row["input_exists"]]
    invalid_input = [row for row in inventory if row["input_exists"] and not row["input_valid"]]
    missing_output = [row for row in inventory if not row["physical_exists"]]
    invalid_output = [
        row for row in inventory if row["physical_exists"] and not row["physical_valid"]
    ]

    existing_characterization = characterization_dir / "summary.json"
    characterization = {
        "directory": str(characterization_dir),
        "summary_exists": existing_characterization.is_file(),
        "summary_sha256": sha256_file(existing_characterization)
        if existing_characterization.is_file()
        else None,
        "legacy_cache_provenance_status": (
            "UNVERIFIED_BY_CURRENT_EVALUATOR_SCHEMA: cached floating outputs are accepted by shape/dtype/finiteness only"
        ),
        "panel_role": "development/deployment characterization; not an untouched confirmatory holdout",
    }

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_READ_ONLY_DNAI_PHYSICAL_PANEL_INVENTORY",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "hardware_summary": str(summary_path),
        "hardware_summary_sha256": sha256_file(summary_path),
        "optimized_onnx_sha256": onnx_sha,
        "nef_sha256_from_hardware_summary": summary["nef_sha256"],
        "n_windows": len(inventory),
        "n_samples": len(sample_ids),
        "missing_inputs": len(missing_input),
        "invalid_inputs": len(invalid_input),
        "missing_physical_outputs": len(missing_output),
        "invalid_physical_outputs": len(invalid_output),
        "all_180_inputs_valid": not missing_input and not invalid_input,
        "all_180_physical_outputs_valid": not missing_output and not invalid_output,
        "existing_characterization": characterization,
        "next_characterization_policy": {
            "tile": 512,
            "overlap": 0.5,
            "blend": "gaussian",
            "legacy_greedy_matching_retained": True,
            "maximum_cardinality_matching_is_sensitivity_only": True,
            "physical_panel_may_not_select_a_new_ptq_candidate": True,
            "reference_defined_nonempty_must_not_use_physical_detections": True,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "dnai_physical_panel_checksum_inventory.csv", inventory)
    (output_dir / "dnai_completion_state_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )

    print("DNAi physical-panel completion-state audit: COMPLETE")
    print(f"windows/samples: {len(inventory)}/{len(sample_ids)}")
    print(
        f"valid physical outputs: {len(inventory) - len(missing_output) - len(invalid_output)}/{len(inventory)}"
    )
    print(
        f"existing full hardware biological-fidelity summary: {characterization['summary_exists']}"
    )
    print("panel role: development/deployment characterization")
    print(f"audit output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
