"""Create and audit the development-only BatchNorm-repaired R1 v2 final candidate.

This script implements docs/NASA_BPS_R1_V2_BN_REPAIR_POLICY.md.

It loads the already frozen all-development FP32 checkpoint, changes only
BatchNorm running state using the deterministic six-source development-only
recalibration procedure established by diagnose_r1_v2_final_candidate_bn.py,
serializes a NEW checkpoint, reloads it in eval mode, and verifies that:

- the parent checkpoint identity is exact;
- all learned/non-BN-running tensors are bit-identical to the parent;
- BatchNorm running state actually changed;
- the serialized repaired checkpoint reproduces the in-memory repaired metrics;
- no final-holdout manifest, image, or phenotype outcome is read.

The original checkpoint is never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

FROZEN_PARENT_CHECKPOINT_SHA256 = (
    "6ee614030df15a44c616fa0229c7c23399ba59b2bc017eed103c7b21531d5f58"
)
REPAIRED_DEV_MANIFEST_SHA256 = (
    "23e21fdc8bf00a2814b64c56223cfd3d8e96eda71bdbff5d8ec74dae218cf5a9"
)
FROZEN_SAMPLE_REFERENCE_SHA256 = (
    "fcbf1339805e02a7ca9cec36b476438b0512b222bd70ac2f90ed75c238a0b01f"
)
EXPECTED_BAGS = 72
EXPECTED_NUCLEI = 7200
METRIC_TOLERANCE = 1e-6


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_bn_running_state_key(key: str) -> bool:
    return key.endswith(".running_mean") or key.endswith(".running_var") or key.endswith(
        ".num_batches_tracked"
    )


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def compare_state_dicts(
    parent: dict[str, torch.Tensor], repaired: dict[str, torch.Tensor]
) -> dict[str, Any]:
    if set(parent) != set(repaired):
        missing = sorted(set(parent) - set(repaired))
        extra = sorted(set(repaired) - set(parent))
        raise RuntimeError(f"State-dict keys changed: missing={missing} extra={extra}")

    non_bn_changed: list[str] = []
    bn_changed: list[str] = []
    bn_unchanged: list[str] = []

    for key in sorted(parent):
        a = parent[key].detach().cpu()
        b = repaired[key].detach().cpu()
        equal = torch.equal(a, b)
        if is_bn_running_state_key(key):
            (bn_unchanged if equal else bn_changed).append(key)
        elif not equal:
            non_bn_changed.append(key)

    return {
        "non_bn_running_state_changed": non_bn_changed,
        "bn_running_state_changed": bn_changed,
        "bn_running_state_unchanged": bn_unchanged,
        "learned_and_other_state_bit_identical": len(non_bn_changed) == 0,
        "bn_running_state_change_detected": len(bn_changed) > 0,
    }


def metric_close(a: dict[str, float], b: dict[str, float], tol: float) -> tuple[bool, dict[str, float]]:
    keys = sorted(set(a) | set(b))
    diffs: dict[str, float] = {}
    ok = True
    for key in keys:
        if key not in a or key not in b:
            ok = False
            diffs[key] = float("inf")
            continue
        diff = abs(float(a[key]) - float(b[key]))
        diffs[key] = diff
        if diff > tol:
            ok = False
    return ok, diffs


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    script_dir = Path(__file__).resolve().parent
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"
    pilot_root = metadata_root / "pilot_v1"

    parser.add_argument(
        "--parent-checkpoint",
        default=str(
            model_root
            / "nasa_bps_53bp1"
            / "r1_v2_final_candidate"
            / "r1_countnet_v2_final.pt"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(
            metadata_root
            / "r1_v2_freeze"
            / "r1_v2_development_manifest_100_qc1.csv"
        ),
    )
    parser.add_argument(
        "--reference",
        default=str(
            pilot_root
            / "reference_phenotypes_v2"
            / "pilot_v1_sample_reference_frozen.csv"
        ),
    )
    parser.add_argument(
        "--reference-freeze",
        default=str(
            pilot_root / "reference_phenotypes_v2" / "reference_freeze_v2.json"
        ),
    )
    parser.add_argument(
        "--image-root",
        default=str(
            data_root / "nasa_bps_microscopy" / "r1_v2_development" / "images"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(model_root / "nasa_bps_53bp1" / "r1_v2_final_candidate"),
    )
    parser.add_argument("--cache-items", type=int, default=2048)
    args = parser.parse_args()

    parent_path = Path(args.parent_checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    reference_path = Path(args.reference).resolve()
    freeze_path = Path(args.reference_freeze).resolve()
    image_root = Path(args.image_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    for path in (parent_path, manifest_path, reference_path, freeze_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    parent_sha = sha256_file(parent_path)
    if parent_sha != FROZEN_PARENT_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Parent checkpoint SHA256 mismatch: expected {FROZEN_PARENT_CHECKPOINT_SHA256}, got {parent_sha}"
        )
    if sha256_file(manifest_path) != REPAIRED_DEV_MANIFEST_SHA256:
        raise RuntimeError("Development manifest SHA256 mismatch")
    if sha256_file(reference_path) != FROZEN_SAMPLE_REFERENCE_SHA256:
        raise RuntimeError("Frozen sample-reference SHA256 mismatch")

    r1v2 = load_module(
        "nasa_bps_r1_v2_impl_bn_repair", script_dir / "train_r1_v2_contrast.py"
    )
    diag = load_module(
        "nasa_bps_r1_v2_bn_diag_repair",
        script_dir / "diagnose_r1_v2_final_candidate_bn.py",
    )
    v1 = r1v2.load_v1_module()
    _ = r1v2.validate_reference_freeze(freeze_path)

    bags = r1v2.build_bags(
        v1, r1v2.read_csv(manifest_path), r1v2.read_csv(reference_path)
    )
    if len(bags) != EXPECTED_BAGS or sum(len(b.rows) for b in bags) != EXPECTED_NUCLEI:
        raise RuntimeError("Unexpected development cohort size")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = v1.NativeImageCache(image_root=image_root, max_items=args.cache_items)
    parent_obj = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent_state = {
        key: value.detach().cpu().clone()
        for key, value in parent_obj["model_state_dict"].items()
    }

    print("Radiation Edge AI - repair R1 v2 FINAL candidate BatchNorm running state")
    print("DEVELOPMENT ONLY: FINAL HOLDOUT STATUS = UNTOUCHED")
    print(f"device: {device}")
    print(f"parent checkpoint SHA256: {parent_sha}")
    print(f"development bags/nuclei: {len(bags)}/{sum(len(b.rows) for b in bags)}")
    print("")

    model = diag.checkpoint_model(v1, parent_path, device)
    recalibration = diag.recalibrate_batchnorm(v1, model, bags, cache, device)
    repaired_state = cpu_state_dict(model)
    state_audit = compare_state_dicts(parent_state, repaired_state)

    print("[State-change audit]")
    print(
        "learned/other tensors bit-identical: "
        f"{'YES' if state_audit['learned_and_other_state_bit_identical'] else 'NO'}"
    )
    print(
        "BatchNorm running-state change detected: "
        f"{'YES' if state_audit['bn_running_state_change_detected'] else 'NO'}"
    )
    print(f"BN running-state keys changed: {len(state_audit['bn_running_state_changed'])}")
    if not state_audit["learned_and_other_state_bit_identical"]:
        raise RuntimeError(
            "Repair modified non-BatchNorm-running-state tensors: "
            + ", ".join(state_audit["non_bn_running_state_changed"][:20])
        )
    if not state_audit["bn_running_state_change_detected"]:
        raise RuntimeError("No BatchNorm running-state tensor changed")

    inmem_bag, inmem_delta = diag.evaluate_with_mode(
        r1v2, v1, model, bags, cache, device, mode="eval"
    )
    inmem_metrics = diag.metric_summary(inmem_bag, inmem_delta)
    diag.print_metrics("in-memory repaired eval-mode", inmem_metrics)

    repaired_checkpoint_path = output_dir / "r1_countnet_v2_final_bn_recalibrated.pt"
    output_dir.mkdir(parents=True, exist_ok=True)

    repaired_obj = copy.deepcopy(parent_obj)
    repaired_obj["candidate_name"] = "NASA_BPS_R1_V2_FINAL_FP32_BN_RECALIBRATED"
    repaired_obj["model_state_dict"] = repaired_state
    repaired_obj["parent_checkpoint_sha256"] = parent_sha
    repaired_obj["bn_repair_policy"] = "docs/NASA_BPS_R1_V2_BN_REPAIR_POLICY.md"
    repaired_obj["bn_repair"] = {
        "development_only": True,
        "final_holdout_read": False,
        "development_manifest_sha256": REPAIRED_DEV_MANIFEST_SHA256,
        "calibration_nuclei": recalibration["calibration_nuclei"],
        "mixed_batch_size": recalibration["mixed_batch_size"],
        "calibration_batches": recalibration["calibration_batches"],
        "batchnorm_layers": recalibration["batchnorm_layers"],
        "running_stat_rule": recalibration["running_stat_rule"],
        "learned_and_other_state_bit_identical_to_parent": True,
        "allowed_state_changes": [
            "BatchNorm running_mean",
            "BatchNorm running_var",
            "BatchNorm num_batches_tracked",
        ],
        "augmentation": False,
    }
    repaired_obj["final_holdout_used_for_training_or_selection"] = False

    torch.save(repaired_obj, repaired_checkpoint_path)
    repaired_sha = sha256_file(repaired_checkpoint_path)

    # Reload the serialized checkpoint and repeat eval-mode audit.
    reloaded_model = diag.checkpoint_model(v1, repaired_checkpoint_path, device)
    reload_state = cpu_state_dict(reloaded_model)
    serialization_state_equal = all(
        torch.equal(repaired_state[key], reload_state[key]) for key in repaired_state
    )
    if not serialization_state_equal:
        raise RuntimeError("Reloaded repaired checkpoint state differs from serialized state")

    reload_bag, reload_delta = diag.evaluate_with_mode(
        r1v2, v1, reloaded_model, bags, cache, device, mode="eval"
    )
    reload_metrics = diag.metric_summary(reload_bag, reload_delta)
    diag.print_metrics("reloaded repaired eval-mode", reload_metrics)

    metrics_match, metric_diffs = metric_close(
        inmem_metrics, reload_metrics, METRIC_TOLERANCE
    )
    repair_audit_pass = bool(
        state_audit["learned_and_other_state_bit_identical"]
        and state_audit["bn_running_state_change_detected"]
        and serialization_state_equal
        and metrics_match
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "final_holdout_read": False,
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": parent_sha,
        "repaired_checkpoint": str(repaired_checkpoint_path),
        "repaired_checkpoint_sha256": repaired_sha,
        "development_manifest_sha256": REPAIRED_DEV_MANIFEST_SHA256,
        "sample_reference_sha256": FROZEN_SAMPLE_REFERENCE_SHA256,
        "recalibration": recalibration,
        "state_audit": state_audit,
        "in_memory_repaired_eval_metrics": inmem_metrics,
        "reloaded_repaired_eval_metrics": reload_metrics,
        "metric_tolerance": METRIC_TOLERANCE,
        "metric_absolute_differences": metric_diffs,
        "metrics_reproduced_after_reload": metrics_match,
        "serialized_state_exactly_reloaded": serialization_state_equal,
        "repair_audit_pass": repair_audit_pass,
        "checkpoint_modified_parent": False,
    }
    summary_path = output_dir / "bn_repair_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("[Repaired final candidate identity]")
    print(f"checkpoint: {repaired_checkpoint_path}")
    print(f"checkpoint SHA256: {repaired_sha}")
    print(f"serialized state exactly reloaded: {'YES' if serialization_state_equal else 'NO'}")
    print(f"reloaded metrics reproduce in-memory repair: {'YES' if metrics_match else 'NO'}")
    print(f"REPAIR AUDIT ALL PASS: {'YES' if repair_audit_pass else 'NO'}")
    print(f"summary: {summary_path}")
    print("PARENT CHECKPOINT MODIFIED: NO")
    print("FINAL HOLDOUT STATUS: UNTOUCHED")
    if repair_audit_pass:
        print(
            "NEXT GATE: pin the repaired checkpoint SHA256 in phenotype-blind final-holdout materialization/QC code before downloading any holdout images."
        )
        return 0

    print("NEXT GATE: keep final holdout closed and resolve the failed repair audit using development data only.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
