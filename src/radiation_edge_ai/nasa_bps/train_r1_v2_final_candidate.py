"""Train the single all-development NASA BPS R1 v2 FP32 final candidate.

This script is intentionally holdout-blind. It verifies that the already-frozen
six-source R1 v2 development-CV gate passed, then trains one final candidate on
all six development Source Names using the exact frozen R1 v2 architecture,
input policy, contrast-aware objective, optimizer schedule, and seed.

It never reads the final-holdout manifest, holdout images, or holdout phenotype
outcomes. The epoch-60 checkpoint SHA256 is the identity that must be pinned
before any final-holdout image access.

Run without --train first for a dry run, then rerun with --train.
"""

from __future__ import annotations

import argparse
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

REPAIRED_DEV_MANIFEST_SHA256 = (
    "23e21fdc8bf00a2814b64c56223cfd3d8e96eda71bdbff5d8ec74dae218cf5a9"
)
FROZEN_SAMPLE_REFERENCE_SHA256 = (
    "fcbf1339805e02a7ca9cec36b476438b0512b222bd70ac2f90ed75c238a0b01f"
)
EXPECTED_SOURCES = 6
EXPECTED_BAGS = 72
EXPECTED_NUCLEI = 7200
EXPECTED_PAIRS = 36
EXPECTED_NUCLEI_PER_BAG = 100


def load_r1v2_module():
    path = Path(__file__).with_name("train_r1_v2_contrast.py")
    spec = importlib.util.spec_from_file_location("nasa_bps_r1_v2_impl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load R1-v2 implementation: {path}")
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


def build_all_development_pairs(r1v2: Any, bags: list[Any]) -> list[tuple[Any, Any]]:
    lookup = r1v2.bag_lookup(bags)
    sources = sorted({bag.source_name for bag in bags})
    if len(sources) != EXPECTED_SOURCES:
        raise RuntimeError(f"Development sources={len(sources)}; expected {EXPECTED_SOURCES}")

    pairs: list[tuple[Any, Any]] = []
    for source in sources:
        for branch, exposed_dose in r1v2.EXPOSED_DOSE.items():
            for hour in ("4", "24", "48"):
                sham_key = (source, branch, "0.0", hour)
                exp_key = (source, branch, r1v2.norm_number(exposed_dose), hour)
                if sham_key not in lookup or exp_key not in lookup:
                    raise RuntimeError(f"Missing all-development pair: {sham_key} / {exp_key}")
                pairs.append((lookup[sham_key], lookup[exp_key]))

    if len(pairs) != EXPECTED_PAIRS:
        raise RuntimeError(f"Matched pairs={len(pairs)}; expected {EXPECTED_PAIRS}")
    used = [bag.sample_name for pair in pairs for bag in pair]
    if len(used) != EXPECTED_BAGS or len(set(used)) != EXPECTED_BAGS:
        raise RuntimeError(
            "All-development matched-pair epoch does not cover each of 72 bags exactly once"
        )
    return pairs


def validate_development_gate(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    gate = obj.get("development_gate")
    if not isinstance(gate, dict) or not gate.get("all_pass", False):
        raise RuntimeError(
            "R1 v2 development gate is not recorded as all-pass; final candidate training is not authorized"
        )
    if obj.get("final_holdout_read") is not False:
        raise RuntimeError("Development summary no longer records final_holdout_read=false")
    return obj


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"
    pilot_root = metadata_root / "pilot_v1"

    parser.add_argument(
        "--manifest",
        default=str(metadata_root / "r1_v2_freeze" / "r1_v2_development_manifest_100_qc1.csv"),
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
        default=str(pilot_root / "reference_phenotypes_v2" / "reference_freeze_v2.json"),
    )
    parser.add_argument(
        "--development-summary",
        default=str(model_root / "nasa_bps_53bp1" / "r1_v2_contrast" / "r1_v2_development_oof_summary.json"),
    )
    parser.add_argument(
        "--image-root",
        default=str(data_root / "nasa_bps_microscopy" / "r1_v2_development" / "images"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(model_root / "nasa_bps_53bp1" / "r1_v2_final_candidate"),
    )
    parser.add_argument("--cache-items", type=int, default=2048)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the frozen all-development final candidate. Without this flag only dry-run validation is performed.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    reference_path = Path(args.reference).resolve()
    freeze_path = Path(args.reference_freeze).resolve()
    dev_summary_path = Path(args.development_summary).resolve()
    image_root = Path(args.image_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    for path in (manifest_path, reference_path, freeze_path, dev_summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != REPAIRED_DEV_MANIFEST_SHA256:
        raise RuntimeError(
            f"Repaired development manifest SHA256 mismatch: expected {REPAIRED_DEV_MANIFEST_SHA256}, got {manifest_sha}"
        )
    reference_sha = sha256_file(reference_path)
    if reference_sha != FROZEN_SAMPLE_REFERENCE_SHA256:
        raise RuntimeError(
            f"Frozen sample-reference SHA256 mismatch: expected {FROZEN_SAMPLE_REFERENCE_SHA256}, got {reference_sha}"
        )

    development_summary = validate_development_gate(dev_summary_path)
    r1v2 = load_r1v2_module()
    v1 = r1v2.load_v1_module()
    freeze = r1v2.validate_reference_freeze(freeze_path)
    manifest = r1v2.read_csv(manifest_path)
    refs = r1v2.read_csv(reference_path)
    bags = r1v2.build_bags(v1, manifest, refs)
    pairs = build_all_development_pairs(r1v2, bags)

    if len(bags) != EXPECTED_BAGS:
        raise RuntimeError(f"Development bags={len(bags)}; expected {EXPECTED_BAGS}")
    if sum(len(bag.rows) for bag in bags) != EXPECTED_NUCLEI:
        raise RuntimeError("Unexpected all-development nucleus count")
    if any(len(bag.rows) != EXPECTED_NUCLEI_PER_BAG for bag in bags):
        raise RuntimeError("At least one all-development bag is not size 100")

    sources = sorted({bag.source_name for bag in bags})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = v1.NativeImageCache(image_root=image_root, max_items=args.cache_items)

    print("Radiation Edge AI - NASA BPS R1 v2 ALL-DEVELOPMENT final FP32 candidate")
    print(f"mode: {'TRAIN' if args.train else 'DRY RUN'}")
    print("FINAL HOLDOUT STATUS: UNTOUCHED")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")
    print(f"development gate all pass: {development_summary['development_gate']['all_pass']}")
    print(f"development manifest SHA256: {manifest_sha}")
    print(f"sample reference SHA256: {reference_sha}")
    print(f"reference freeze: {freeze.get('reference_freeze_version')}")
    print(f"sources: {len(sources)} {sources}")
    print(f"bags: {len(bags)}")
    print(f"nuclei: {sum(len(bag.rows) for bag in bags)}")
    print(f"matched training pairs/epoch: {len(pairs)}")
    print(
        f"schedule: {r1v2.EPOCHS} epochs AdamW lr={r1v2.LEARNING_RATE} wd={r1v2.WEIGHT_DECAY} seed={r1v2.SEED}"
    )
    print(
        f"loss: L_abs + {r1v2.DELTA_LOSS_WEIGHT:g} * L_delta; SmoothL1 beta={r1v2.SMOOTHL1_BETA}"
    )
    print("")

    print("[Full development geometry validation]")
    geometry = v1.validate_all_geometry(bags, cache)
    print(
        f"nuclei={geometry['n_nuclei']} "
        f"height={geometry['height_min']}/{geometry['height_median']}/{geometry['height_max']} "
        f"width={geometry['width_min']}/{geometry['width_median']}/{geometry['width_max']}"
    )
    print(f"all fit 256x256 at native scale: {'YES' if geometry['all_fit_256_native_scale'] else 'NO'}")
    print("")

    r1v2.seed_everything(r1v2.SEED)
    smoke_model = v1.R1CountNetV1().to(device)
    smoke_rng = np.random.default_rng(r1v2.SEED)
    sham, exp = pairs[0]
    sham_x = v1.load_bag_tensor(sham, cache, augment=False, rng=smoke_rng).to(device)
    exp_x = v1.load_bag_tensor(exp, cache, augment=False, rng=smoke_rng).to(device)
    sham_target = torch.tensor(sham.target_avg_nfoci, dtype=torch.float32, device=device)
    exp_target = torch.tensor(exp.target_avg_nfoci, dtype=torch.float32, device=device)
    with torch.no_grad():
        total0, abs0, delta0, sham0, exp0 = r1v2.pair_loss(
            smoke_model, sham_x, exp_x, sham_target, exp_target
        )
    print("[Model/pair smoke]")
    print(f"model: R1CountNetV1")
    print(f"parameters: {v1.parameter_count(smoke_model):,}")
    print(f"bag input: {tuple(sham_x.shape)}")
    print(f"first pair: {sham.source_name} {sham.particle_type} {sham.hr_post_exposure}h")
    print(
        f"initial means sham/exposed: {float(sham0):.4f}/{float(exp0):.4f}; "
        f"targets: {float(sham_target):.4f}/{float(exp_target):.4f}"
    )
    print(
        f"initial loss total/abs/delta: {float(total0):.5f}/{float(abs0):.5f}/{float(delta0):.5f}"
    )
    print("")

    dry_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "train" if args.train else "dry_run",
        "final_holdout_read": False,
        "development_gate_all_pass": True,
        "development_manifest_sha256": manifest_sha,
        "sample_reference_sha256": reference_sha,
        "reference_freeze_version": freeze.get("reference_freeze_version"),
        "sources": sources,
        "bags": len(bags),
        "nuclei": sum(len(bag.rows) for bag in bags),
        "matched_pairs_per_epoch": len(pairs),
        "model": "R1CountNetV1",
        "parameter_count": v1.parameter_count(smoke_model),
        "epochs": r1v2.EPOCHS,
        "learning_rate": r1v2.LEARNING_RATE,
        "weight_decay": r1v2.WEIGHT_DECAY,
        "seed": r1v2.SEED,
        "objective": "L_abs + 1.0 * L_delta; SmoothL1 beta=0.5",
        "geometry": geometry,
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.train:
        (output_dir / "dry_run_summary.json").write_text(
            json.dumps(dry_summary, indent=2), encoding="utf-8"
        )
        print("NASA BPS R1 V2 FINAL CANDIDATE DRY RUN COMPLETE: YES")
        print("No model was trained.")
        print("FINAL HOLDOUT STATUS: UNTOUCHED")
        print("NEXT GATE: rerun the same command with --train; freeze the epoch-60 checkpoint SHA256 before any holdout image access.")
        return 0

    del smoke_model, sham_x, exp_x
    if device.type == "cuda":
        torch.cuda.empty_cache()

    r1v2.seed_everything(r1v2.SEED)
    model = v1.R1CountNetV1().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=r1v2.LEARNING_RATE, weight_decay=r1v2.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=r1v2.EPOCHS
    )
    rng = np.random.default_rng(r1v2.SEED)

    history: list[dict[str, Any]] = []
    print("[Training all six development Source Names]")
    for epoch in range(1, r1v2.EPOCHS + 1):
        model.train()
        order = rng.permutation(len(pairs))
        totals: list[float] = []
        absolutes: list[float] = []
        deltas: list[float] = []
        for idx in order:
            sham, exp = pairs[int(idx)]
            sham_x = v1.load_bag_tensor(sham, cache, augment=True, rng=rng).to(device)
            exp_x = v1.load_bag_tensor(exp, cache, augment=True, rng=rng).to(device)
            sham_target = torch.tensor(sham.target_avg_nfoci, dtype=torch.float32, device=device)
            exp_target = torch.tensor(exp.target_avg_nfoci, dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            total, abs_loss, delta_loss, _, _ = r1v2.pair_loss(
                model, sham_x, exp_x, sham_target, exp_target
            )
            total.backward()
            optimizer.step()

            totals.append(float(total.detach().cpu()))
            absolutes.append(float(abs_loss.detach().cpu()))
            deltas.append(float(delta_loss.detach().cpu()))

        scheduler.step()
        record = {
            "epoch": epoch,
            "train_total_loss_mean": float(np.mean(totals)),
            "train_absolute_loss_mean": float(np.mean(absolutes)),
            "train_delta_loss_mean": float(np.mean(deltas)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        if epoch == 1 or epoch % 10 == 0 or epoch == r1v2.EPOCHS:
            print(
                f"  epoch {epoch:3d}/{r1v2.EPOCHS}: "
                f"total={record['train_total_loss_mean']:.5f} "
                f"abs={record['train_absolute_loss_mean']:.5f} "
                f"delta={record['train_delta_loss_mean']:.5f} "
                f"lr={record['learning_rate']:.3g}"
            )

    checkpoint_path = output_dir / "r1_countnet_v2_final.pt"
    checkpoint = {
        "model_name": "R1CountNetV1",
        "candidate_name": "NASA_BPS_R1_V2_FINAL_FP32",
        "model_state_dict": model.state_dict(),
        "development_manifest_sha256": manifest_sha,
        "sample_reference_sha256": reference_sha,
        "train_sources": sources,
        "bags": len(bags),
        "nuclei": sum(len(bag.rows) for bag in bags),
        "matched_pairs_per_epoch": len(pairs),
        "input_policy": {
            "canvas": 256,
            "channels": ["FITC_p1_p99.5", "DAPI_p1_p99.5", "ZERO"],
            "resize": False,
            "padding": "center_zero",
        },
        "objective": "L_abs + 1.0 * L_delta; SmoothL1 beta=0.5",
        "epochs": r1v2.EPOCHS,
        "learning_rate": r1v2.LEARNING_RATE,
        "weight_decay": r1v2.WEIGHT_DECAY,
        "seed": r1v2.SEED,
        "parameter_count": v1.parameter_count(model),
        "final_holdout_used_for_training_or_selection": False,
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    r1v2.write_csv(output_dir / "training_history.csv", history)

    # Development-fit outputs are audit-only. They are not a new model-selection gate.
    fit_bag_rows, fit_nucleus_rows = r1v2.evaluate_bags(
        v1, model, bags, cache, device, "all_development_fit"
    )
    r1v2.write_csv(output_dir / "development_fit_bag_predictions.csv", fit_bag_rows)
    r1v2.write_csv(output_dir / "development_fit_nucleus_predictions.csv", fit_nucleus_rows)
    fit_metrics = v1.regression_metrics(fit_bag_rows)
    fit_deltas = v1.biological_deltas(fit_bag_rows)
    fit_delta_metrics = v1.delta_metrics(fit_deltas)
    r1v2.write_csv(output_dir / "development_fit_matched_radiation_deltas.csv", fit_deltas)

    summary = {
        **dry_summary,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "train",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "development_fit_metrics_audit_only": fit_metrics,
        "development_fit_delta_metrics_audit_only": fit_delta_metrics,
        "final_holdout_read": False,
        "checkpoint_frozen_before_holdout_access": True,
    }
    (output_dir / "final_candidate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("")
    print("[Final candidate identity]")
    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint SHA256: {checkpoint_sha}")
    print(
        f"development-fit audit only: MAE={fit_metrics['mae']:.4f} "
        f"Spearman={fit_metrics['spearman']:.4f} "
        f"delta Spearman={fit_delta_metrics['delta_spearman']:.4f}"
    )
    print("")
    print("NASA BPS R1 V2 FINAL CANDIDATE TRAINING COMPLETE: YES")
    print("FINAL HOLDOUT STATUS: UNTOUCHED")
    print("NEXT GATE: pin this checkpoint SHA256 in the holdout materialization/evaluation code, then open holdout images for phenotype-blind QC only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
