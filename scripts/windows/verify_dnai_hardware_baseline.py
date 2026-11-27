"""Compare a fresh three-field physical DNAi v3 panel with frozen FP512 and KL720 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np

SAMPLES = ("1__tile_4", "2__tile_9", "16__tile_5")
NEF = "4b3dfec9a61c99e186dd4b8482fa5b06e6a4958f325ed4a0db0546f1dcab2bfc"
MODEL_BASE = Path("models/dnai/unet_mobileone_s1/kneron")
V3 = MODEL_BASE / "int8_512_v3_pct100"
MANIFEST = MODEL_BASE / "int8_512/validation_512_overlap50/validation_windows_manifest.json"
WINDOWS = Path("audits/dnai_physical_panel_20260916/dnai_physical_panel_checksum_inventory.csv")
MASKS = Path("audits/dnai_characterization_sensitivity_20260916/artifact_hash_inventory.csv")
FIELD_METRICS = V3 / "hardware_biological_fidelity/per_image.csv"
FROZEN = {
    MANIFEST: "fa1eed5231bc9362231eb59f34ea7bf45fa6012e584ea2af1deda0282ab01711",
    WINDOWS: "54473ad94c06b7f2cfca24d6de724986928ac25e17907a80bad3da01d7dd8e58",
    MASKS: "d6ee8f5f0abcffe65a25947b0bec3e425b20d8e3b5c1dbb15fa8891de25d9b9d",
    FIELD_METRICS: "0ba76bd7ef768d7baea2ab899f2ac04a6e43f69002debad38862eaa531abdb91",
}
PHYSICAL_PROBABILITIES = {
    "1__tile_4": "959a9e7715a0b38c24bd4d9817c2ea38ad88544e40e3c9979a4ce856db5935c0",
    "2__tile_9": "856fcc82e7ad4cb2607302ba44fcdcb8db8ac4749ed2ca4da76f2d96482edf0c",
    "16__tile_5": "fec13c28c71844401a1096cb26053d39e4d39debaed22cdcda7e6b80ad6f0eb9",
}


def digest(path):
    with path.open("rb") as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def checked(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Frozen SHA256 mismatch: {path}")
    return path


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def fresh_rows(summary, log, manifest_rows, run, manifest):
    if not (
        summary["complete"] is True
        and summary["nef_sha256"] == NEF
        and summary["model_id"] == 32769
        and summary["selected_samples"] == list(SAMPLES)
        and summary["n_existing_skipped"] == 0
        and not summary["missing_outputs"]
        and summary["n_ran"] == summary["selected_windows"] == summary["n_valid_outputs"] == 27
        and Path(summary["validation_manifest"]).resolve() == manifest.resolve()
        and Path(summary["physical_output_dir"]).resolve() == (run / "physical_outputs").resolve()
        and Path(summary["per_window_csv"]).resolve() == (run / "per_window.csv").resolve()
        and len(log) == len(manifest_rows) == 27
    ):
        raise ValueError("Expected a complete, fresh physical v3 run of all three fields")
    for index, (row, record) in enumerate(zip(log, manifest_rows), 1):
        name = Path(record["tensor"]).name
        timing = float(row["hardware_send_receive_ms"])
        if not (
            row["status"] == "ran"
            and int(row["position"]) == index
            and row["sample_id"] == record["sample_id"]
            and row["key"] == record["key"]
            and int(row["image_index"]) == record["image_index"]
            and int(row["window_index"]) == record["window_index"]
            and Path(row["input_tensor"]).resolve()
            == (manifest.parent / record["tensor"]).resolve()
            and Path(row["physical_output"]).resolve()
            == (run / "physical_outputs" / name).resolve()
            and math.isfinite(timing)
            and timing > 0
        ):
            raise ValueError(f"Invalid physical execution row for {name}")


def compare_window(live, historical):
    a, b = (np.load(path, allow_pickle=False) for path in (live, historical))
    if any(
        x.shape != (1, 3, 512, 512) or x.dtype != np.float32 or not np.isfinite(x).all()
        for x in (a, b)
    ):
        raise ValueError(f"Invalid physical logits for {live.name}")
    return {
        "array_equal": bool(np.array_equal(a, b)),
        "max_abs_difference": float(np.max(np.abs(a.astype(np.float64) - b))),
        "different_values": int(np.count_nonzero(a != b)),
    }


def instances(fibers, pixel_size):
    data = []
    valid_index = 0
    for index, fiber in enumerate(fibers):
        valid = bool(fiber.is_valid)
        ratio = float(fiber.ratio)
        data.append(
            {
                "object_index": index,
                "valid_index": valid_index if valid else None,
                "bbox_xywh": list(map(int, fiber.bbox)),
                "length_um": float(fiber.length) * pixel_size,
                "red_length_um": float(fiber.red) * pixel_size,
                "green_length_um": float(fiber.green) * pixel_size,
                "green_red_ratio": ratio if math.isfinite(ratio) else None,
            }
        )
        valid_index += int(valid)
    return data


def field(sample, records, root, run, output, mask_hashes, expected):
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    import cv2
    import torch
    from dnafiber.inference import probas_to_segmentation
    from dnafiber.postprocess import refine_segmentation

    from radiation_edge_ai.dna_fiber import evaluate_kl720_hardware_biological_fidelity_512 as assay
    from radiation_edge_ai.dnai_window_origin import REPLAY_ATOL

    reference = root / V3 / "hardware_biological_fidelity" / sample
    masks = {}
    for label in ("floating512", "physical_kl720"):
        path = reference / f"{label}_segmentation_class_ids.png"
        sha = mask_hashes[(sample, f"{label}_segmentation")]
        masks[label] = cv2.imread(str(checked(path, sha)), cv2.IMREAD_UNCHANGED)
        if masks[label] is None or masks[label].shape != (1024, 1024):
            raise ValueError(f"Invalid reference segmentation: {path}")

    prob = assay.stitch_probabilities(records, run / "physical_outputs", "physical")
    ref_prob = checked(
        reference / "physical_kl720_stitched_probabilities.npz", PHYSICAL_PROBABILITIES[sample]
    )
    with np.load(ref_prob, allow_pickle=False) as archive:
        old = archive["probabilities"]
        if old.shape != prob[0].shape or not np.isfinite(old).all():
            raise ValueError(f"Invalid probabilities: {ref_prob}")
        max_error = float(np.max(np.abs(old - prob[0])))
    seg = probas_to_segmentation(torch.from_numpy(prob))
    mask_equal = bool(np.array_equal(seg, masks["physical_kl720"]))
    fp, hw = (refine_segmentation(mask) for mask in (masks["floating512"], seg))
    fp_valid, hw_valid = list(fp.valid_copy()), list(hw.valid_copy())
    seg_metrics = assay.segmentation_metrics(masks["floating512"], seg)
    matches = assay.greedy_matches(hw_valid, fp_valid)
    match_metrics = assay.match_fibers(hw_valid, fp_valid)
    actual = {
        "floating_512_valid_fibers": len(fp_valid),
        "physical_kl720_valid_fibers": len(hw_valid),
        "physical_minus_floating_fiber_count": len(hw_valid) - len(fp_valid),
        **{"physical_vs_floating_fiber_" + k: v for k, v in match_metrics.items()},
        **{"floating_vs_physical_seg_" + k: v for k, v in seg_metrics.items()},
    }
    frozen = {key: float(expected[key]) for key in actual}
    exact_metrics = {key: actual[key] == v for key, v in frozen.items()}
    target = output / sample
    target.mkdir()
    png = target / "physical_kl720_segmentation_class_ids.png"
    if not cv2.imwrite(str(png), seg):
        raise OSError(f"Could not save mask: {png}")
    prob_path = target / "physical_kl720_stitched_probabilities.npz"
    np.savez_compressed(prob_path, probabilities=prob[0])
    return {
        "sample_id": sample,
        "passed": mask_equal and max_error <= REPLAY_ATOL and all(exact_metrics.values()),
        "postprocess": {
            "pixel_size_um": assay.PIXEL_SIZE_UM,
            "tile": assay.TILE,
            "overlap": assay.OVERLAP,
            "blend": "gaussian",
            "sigma_scale": assay.SIGMA_SCALE,
            "bbox_match_iou": assay.MATCH_IOU,
        },
        "frozen_physical_mask_equal": mask_equal,
        "physical_probability_replay_max_abs": max_error,
        "physical_probability_replay_limit": REPLAY_ATOL,
        "historical_metrics": frozen,
        "measured_metrics": actual,
        "metric_exact_agreement": exact_metrics,
        "physical_instances": instances(hw, assay.PIXEL_SIZE_UM),
        "fp512_instances": instances(fp, assay.PIXEL_SIZE_UM),
        "matched_valid_instances": [
            {"physical_valid_index": i, "fp512_valid_index": j, "bbox_iou": iou}
            for i, j, iou in matches
        ],
        "frozen_mask_hashes": {k: mask_hashes[(sample, k + "_segmentation")] for k in masks},
        "frozen_physical_probability_sha256": PHYSICAL_PROBABILITIES[sample],
        "saved_mask_sha256": digest(png),
        "saved_probability_sha256": digest(prob_path),
    }


def verify(root, run):
    root, run = root.resolve(), run.resolve()
    if any(
        path == run or path in run.parents
        for path in (root / "models", root / "cache", root / "audits", root / "data")
    ):
        raise ValueError("Select a fresh physical run outside frozen evidence directories")
    for path, sha in FROZEN.items():
        checked(root / path, sha)
    manifest = root / MANIFEST
    records = [r for r in json.loads(manifest.read_text())["records"] if r["sample_id"] in SAMPLES]
    summary = json.loads((run / "summary.json").read_text())
    log = rows(run / "per_window.csv")
    fresh_rows(summary, log, records, run, manifest)
    checked(Path(summary["nef"]), NEF)
    inventory = {(r["sample_id"], int(r["window_index"])): r for r in rows(root / WINDOWS)}
    mask_hashes = {(r["sample_id"], r["artifact"]): r["sha256"] for r in rows(root / MASKS)}
    metrics = {r["sample_id"]: r for r in rows(root / FIELD_METRICS)}
    windows = []
    for row, record in zip(log, records):
        sample, index = record["sample_id"], record["window_index"]
        evidence = inventory[(sample, index)]
        name = Path(record["tensor"]).name
        checked(manifest.parent / record["tensor"], evidence["input_sha256"])
        historical = root / V3 / "hardware_panel/physical_outputs" / name
        checked(historical, evidence["physical_sha256"])
        live = run / "physical_outputs" / name
        windows.append(
            {
                "sample_id": sample,
                "window_index": index,
                "input_sha256": evidence["input_sha256"],
                "frozen_physical_sha256": evidence["physical_sha256"],
                "new_physical_sha256": digest(live),
                "hardware_send_receive_ms": float(row["hardware_send_receive_ms"]),
                **compare_window(live, historical),
            }
        )

    destination = run / "comparison"
    if destination.exists():
        raise FileExistsError(f"Comparison already exists: {destination}")
    # Preserve partial output on exceptions while keeping a retry possible.
    output = run / ("comparison_in_progress_" + uuid4().hex)
    if output.resolve().parent != run:
        raise ValueError("Working comparison directory is outside the run")
    output.mkdir(exist_ok=False)
    fields = [
        field(
            sample,
            [r for r in records if r["sample_id"] == sample],
            root,
            run,
            output,
            mask_hashes,
            metrics[sample],
        )
        for sample in SAMPLES
    ]
    receipt = {
        "schema": "dnai_physical_baseline_comparison_v1",
        "passed": all(w["array_equal"] for w in windows) and all(f["passed"] for f in fields),
        "model_nef_sha256": NEF,
        "run_summary_sha256": digest(run / "summary.json"),
        "per_window_csv_sha256": digest(run / "per_window.csv"),
        "frozen_evidence_sha256": {p.as_posix(): h for p, h in FROZEN.items()},
        "windows": windows,
        "fields": fields,
    }
    receipt_path = output / "comparison.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output.rename(destination)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("D:/radiation-edge-ai-data"))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    result = verify(args.data_root, args.run_dir)
    print(f"{args.run_dir / 'comparison/comparison.json'}: passed={result['passed']}")
    sys.exit(0 if result["passed"] else 1)
