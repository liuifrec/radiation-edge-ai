"""Diagnose the NASA BPS R1 v2 final candidate before any holdout access.

The all-development final candidate trained with a low aggregate loss but its
post-training development-fit audit was unexpectedly poor in model.eval() mode.
Because R1CountNetV1 contains BatchNorm layers, this script tests whether the
discrepancy is caused by BatchNorm running statistics rather than learned
weights.

This is DEVELOPMENT ONLY. It reads no final-holdout manifest, image, or
phenotype outcome and never overwrites the frozen checkpoint.

It reports three views of the exact frozen checkpoint:
1. frozen eval-mode inference (the deployment-relevant behavior);
2. train-mode/batch-stat inference on a disposable clone (diagnostic only);
3. eval-mode inference after deterministic, development-only BatchNorm running-
   statistic recalibration on a disposable clone using mixed-condition batches.

If (2) and/or (3) recover the expected development fit while (1) is poor, the
problem is BatchNorm state calibration and must be resolved before opening the
final holdout. Any repaired checkpoint must be frozen under a documented policy
amendment before holdout access.
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

import numpy as np
import torch
from torch import nn

FROZEN_FINAL_CHECKPOINT_SHA256 = (
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
MIXED_BATCH_SIZE = 100


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


def checkpoint_model(v1: Any, checkpoint_path: Path, device: torch.device) -> nn.Module:
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if obj.get("model_name") != "R1CountNetV1":
        raise RuntimeError(f"Unexpected checkpoint model: {obj.get('model_name')!r}")
    model = v1.R1CountNetV1()
    model.load_state_dict(obj["model_state_dict"], strict=True)
    return model.to(device)


def evaluate_with_mode(
    r1v2: Any,
    v1: Any,
    model: nn.Module,
    bags: list[Any],
    cache: Any,
    device: torch.device,
    *,
    mode: str,
) -> tuple[dict[str, float], dict[str, float]]:
    if mode not in {"eval", "train_batch_stats"}:
        raise ValueError(mode)

    bag_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(r1v2.SEED)
    if mode == "eval":
        model.eval()
    else:
        model.train()

    with torch.no_grad():
        for bag in bags:
            x = v1.load_bag_tensor(bag, cache, augment=False, rng=rng).to(device)
            pred = model(x).detach().cpu().numpy().astype(np.float64)
            bag_rows.append(
                {
                    "fold": mode,
                    "sample_name": bag.sample_name,
                    "source_name": bag.source_name,
                    "strain": bag.strain,
                    "sex": bag.sex,
                    "particle_type": bag.particle_type,
                    "dose_Gy": bag.dose_Gy,
                    "hr_post_exposure": bag.hr_post_exposure,
                    "nasa_avg_nfoci": bag.target_avg_nfoci,
                    "nasa_avg_foci_no_outl": bag.target_avg_foci_no_outl,
                    "predicted_bag_mean": float(np.mean(pred)),
                }
            )

    bag_metrics = v1.regression_metrics(bag_rows)
    delta_rows = v1.biological_deltas(bag_rows)
    delta_metrics = v1.delta_metrics(delta_rows)
    return bag_metrics, delta_metrics


def mixed_row_order(bags: list[Any]) -> list[dict[str, str]]:
    # Interleave nucleus rank across all 72 bags so every calibration batch is
    # condition-mixed rather than one homogeneous Source x condition bag.
    ordered_bags = sorted(bags, key=lambda bag: bag.sample_name)
    sizes = {len(bag.rows) for bag in ordered_bags}
    if sizes != {100}:
        raise RuntimeError(f"Expected all development bags to contain 100 nuclei, got {sorted(sizes)}")
    rows: list[dict[str, str]] = []
    for nucleus_rank in range(100):
        for bag in ordered_bags:
            rows.append(bag.rows[nucleus_rank])
    if len(rows) != EXPECTED_NUCLEI:
        raise RuntimeError(f"Mixed calibration rows={len(rows)}; expected {EXPECTED_NUCLEI}")
    return rows


def recalibrate_batchnorm(
    v1: Any,
    model: nn.Module,
    bags: list[Any],
    cache: Any,
    device: torch.device,
) -> dict[str, Any]:
    bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    if not bn_layers:
        raise RuntimeError("No BatchNorm2d layers found")

    old_momentum = [m.momentum for m in bn_layers]
    for m in bn_layers:
        m.reset_running_stats()
        # Cumulative moving average across deterministic mixed batches.
        m.momentum = None

    model.train()
    rows = mixed_row_order(bags)
    batches = 0
    with torch.no_grad():
        for start in range(0, len(rows), MIXED_BATCH_SIZE):
            chunk = rows[start : start + MIXED_BATCH_SIZE]
            images = [v1.pad_and_pack(cache.get(row)) for row in chunk]
            x = torch.from_numpy(np.stack(images, axis=0)).to(device)
            _ = model(x)
            batches += 1

    for m, momentum in zip(bn_layers, old_momentum, strict=True):
        m.momentum = momentum
    model.eval()
    return {
        "batchnorm_layers": len(bn_layers),
        "calibration_nuclei": len(rows),
        "mixed_batch_size": MIXED_BATCH_SIZE,
        "calibration_batches": batches,
        "running_stat_rule": "reset then cumulative moving average over deterministic mixed-condition batches",
    }


def metric_summary(bag: dict[str, float], delta: dict[str, float]) -> dict[str, float]:
    return {
        "bag_mae": float(bag["mae"]),
        "bag_rmse": float(bag["rmse"]),
        "bag_pearson": float(bag["pearson"]),
        "bag_spearman": float(bag["spearman"]),
        "delta_mae": float(delta["delta_mae"]),
        "delta_pearson": float(delta["delta_pearson"]),
        "delta_spearman": float(delta["delta_spearman"]),
        "direction_agreement": float(delta["direction_agreement_fraction"]),
    }


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(
        f"{label}: bag MAE={metrics['bag_mae']:.4f} "
        f"Spearman={metrics['bag_spearman']:.4f} "
        f"delta Spearman={metrics['delta_spearman']:.4f} "
        f"direction={metrics['direction_agreement']:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    repo_script_dir = Path(__file__).resolve().parent
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"
    pilot_root = metadata_root / "pilot_v1"

    parser.add_argument(
        "--checkpoint",
        default=str(model_root / "nasa_bps_53bp1" / "r1_v2_final_candidate" / "r1_countnet_v2_final.pt"),
    )
    parser.add_argument(
        "--manifest",
        default=str(metadata_root / "r1_v2_freeze" / "r1_v2_development_manifest_100_qc1.csv"),
    )
    parser.add_argument(
        "--reference",
        default=str(pilot_root / "reference_phenotypes_v2" / "pilot_v1_sample_reference_frozen.csv"),
    )
    parser.add_argument(
        "--reference-freeze",
        default=str(pilot_root / "reference_phenotypes_v2" / "reference_freeze_v2.json"),
    )
    parser.add_argument(
        "--image-root",
        default=str(data_root / "nasa_bps_microscopy" / "r1_v2_development" / "images"),
    )
    parser.add_argument(
        "--output",
        default=str(model_root / "nasa_bps_53bp1" / "r1_v2_final_candidate" / "bn_diagnostic.json"),
    )
    parser.add_argument("--cache-items", type=int, default=2048)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    reference_path = Path(args.reference).resolve()
    freeze_path = Path(args.reference_freeze).resolve()
    image_root = Path(args.image_root).resolve()
    output_path = Path(args.output).resolve()

    for path in (checkpoint_path, manifest_path, reference_path, freeze_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != FROZEN_FINAL_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Final checkpoint SHA256 mismatch: expected {FROZEN_FINAL_CHECKPOINT_SHA256}, got {checkpoint_sha}"
        )
    if sha256_file(manifest_path) != REPAIRED_DEV_MANIFEST_SHA256:
        raise RuntimeError("Development manifest SHA256 mismatch")
    if sha256_file(reference_path) != FROZEN_SAMPLE_REFERENCE_SHA256:
        raise RuntimeError("Sample reference SHA256 mismatch")

    r1v2 = load_module("nasa_bps_r1_v2_impl_bn_diag", repo_script_dir / "train_r1_v2_contrast.py")
    v1 = r1v2.load_v1_module()
    _ = r1v2.validate_reference_freeze(freeze_path)
    bags = r1v2.build_bags(v1, r1v2.read_csv(manifest_path), r1v2.read_csv(reference_path))
    if len(bags) != EXPECTED_BAGS or sum(len(b.rows) for b in bags) != EXPECTED_NUCLEI:
        raise RuntimeError("Unexpected development cohort size")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = v1.NativeImageCache(image_root=image_root, max_items=args.cache_items)

    print("Radiation Edge AI - diagnose R1 v2 FINAL candidate BatchNorm state")
    print("DEVELOPMENT ONLY: FINAL HOLDOUT STATUS = UNTOUCHED")
    print(f"device: {device}")
    print(f"checkpoint SHA256: {checkpoint_sha}")
    print(f"bags/nuclei: {len(bags)}/{sum(len(b.rows) for b in bags)}")
    print("")

    frozen_model = checkpoint_model(v1, checkpoint_path, device)
    eval_bag, eval_delta = evaluate_with_mode(
        r1v2, v1, frozen_model, bags, cache, device, mode="eval"
    )
    frozen_eval = metric_summary(eval_bag, eval_delta)
    print_metrics("frozen eval-mode", frozen_eval)

    train_clone = checkpoint_model(v1, checkpoint_path, device)
    train_bag, train_delta = evaluate_with_mode(
        r1v2, v1, train_clone, bags, cache, device, mode="train_batch_stats"
    )
    train_stats = metric_summary(train_bag, train_delta)
    print_metrics("disposable train-mode batch-stats", train_stats)

    recal_model = checkpoint_model(v1, checkpoint_path, device)
    recalibration = recalibrate_batchnorm(v1, recal_model, bags, cache, device)
    recal_bag, recal_delta = evaluate_with_mode(
        r1v2, v1, recal_model, bags, cache, device, mode="eval"
    )
    recal_eval = metric_summary(recal_bag, recal_delta)
    print_metrics("disposable BN-recalibrated eval-mode", recal_eval)

    bn_gap_supported = bool(
        train_stats["bag_mae"] < frozen_eval["bag_mae"] * 0.5
        or recal_eval["bag_mae"] < frozen_eval["bag_mae"] * 0.5
    )

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "final_holdout_read": False,
        "frozen_checkpoint_sha256": checkpoint_sha,
        "frozen_eval_mode": frozen_eval,
        "disposable_train_mode_batch_stats": train_stats,
        "disposable_bn_recalibrated_eval_mode": recal_eval,
        "bn_recalibration": recalibration,
        "batchnorm_state_gap_supported": bn_gap_supported,
        "checkpoint_modified": False,
        "interpretation": (
            "A large recovery with train-mode batch statistics and/or deterministic development-only BN recalibration indicates a BatchNorm running-statistics problem, not necessarily a failure of the learned convolutional weights. No holdout should be opened until a deployment-valid FP32 checkpoint is frozen."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("")
    print(f"BatchNorm state gap supported: {'YES' if bn_gap_supported else 'NO'}")
    print(f"Diagnostic: {output_path}")
    print("FROZEN CHECKPOINT MODIFIED: NO")
    print("FINAL HOLDOUT STATUS: UNTOUCHED")
    if bn_gap_supported:
        print("NEXT GATE: document a development-only BatchNorm-state repair policy, create and freeze a repaired FP32 checkpoint, and repeat this audit BEFORE holdout image access.")
    else:
        print("NEXT GATE: investigate the all-development candidate using development data only; do not open the final holdout yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
