"""Train the frozen NASA BPS R1 v2 contrast-aware DEVELOPMENT model.

R1 v2 keeps the R1-v1 compact CNN and native-scale 256x256 input policy, but
uses 100 nuclei per sample bag and adds a matched sham/exposed contrast loss.

The final holdout remains untouched. This script reads only:
- the repaired six-source R1 v2 development manifest;
- the already frozen 72-sample NASA aggregate reference from pilot v1;
- the already frozen reference metadata;
- local R1 v2 DEVELOPMENT FITC/DAPI images.

Run without --train first. The dry run validates the repaired manifest,
reference/sample joins, all 7,200 FITC/DAPI geometries, the three development-CV
folds, all matched training pairs, and one model/pair forward pass. Then rerun
with --train to fit the fixed 60-epoch policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

REPAIRED_DEV_MANIFEST_SHA256 = (
    "23e21fdc8bf00a2814b64c56223cfd3d8e96eda71bdbff5d8ec74dae218cf5a9"
)
FROZEN_V1_MANIFEST_SHA256 = (
    "41ff2e4d64577f17c976dea5bbc17221fe63be9e2e7e2291309b520027fbe725"
)
FROZEN_RAW_PHENOTYPE_SHA256 = (
    "d28dc5d9c53221c66a90074a92783fd4708ea83db0fc1a9c767e47500e643d6a"
)
EXPECTED_NUCLEI = 7200
EXPECTED_BAGS = 72
EXPECTED_NUCLEI_PER_BAG = 100
EXPECTED_V1_MEMBERS = 2160
SEED = 720
EPOCHS = 60
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
SMOOTHL1_BETA = 0.5
DELTA_LOSS_WEIGHT = 1.0

FOLD_FIELDS = (
    "fold_female_role",
    "fold_male_1_role",
    "fold_male_2_role",
)
EXPOSED_DOSE = {"Fe": "0.82", "X-ray": "1.0"}

# Predeclared development-CV gate from docs/NASA_BPS_R1_V2_CONTRAST_POLICY.md
GATE = {
    "overall_bag_spearman_min": 0.50,
    "weakest_fold_bag_spearman_min": 0.25,
    "overall_bag_mae_max": 0.6654,
    "delta_spearman_min": 0.70,
    "direction_agreement_min": 30.0 / 36.0,
    "four_hour_direction_min": 1.0,
    "late_24_48_direction_min": 17.0 / 24.0,
    "peak_time_recovery_min": 10.0 / 12.0,
}


def load_v1_module():
    path = Path(__file__).with_name("train_pilot_v1_r1_mil.py")
    spec = importlib.util.spec_from_file_location("nasa_bps_r1_v1_impl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load R1-v1 implementation: {path}")
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def norm_number(value: str) -> str:
    x = float(str(value).strip())
    if x.is_integer():
        return f"{x:.1f}"
    return format(x, "g")


def norm_hour(value: str) -> str:
    x = float(str(value).strip())
    return str(int(x)) if x.is_integer() else format(x, "g")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_reference_freeze(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not obj.get("reference_level_frozen", False):
        raise RuntimeError("NASA aggregate reference is not marked frozen")
    if obj.get("reference_level") != "sample_plate_well_aggregate":
        raise RuntimeError(f"Unexpected reference level: {obj.get('reference_level')!r}")
    if obj.get("primary_reference_endpoint") != "avg_nfoci":
        raise RuntimeError(
            f"Unexpected primary reference endpoint: {obj.get('primary_reference_endpoint')!r}"
        )
    if obj.get("manifest_sha256") != FROZEN_V1_MANIFEST_SHA256:
        raise RuntimeError(
            "Reference freeze no longer points to the frozen pilot-v1 manifest: "
            f"{obj.get('manifest_sha256')}"
        )
    if obj.get("raw_phenotype_sha256") != FROZEN_RAW_PHENOTYPE_SHA256:
        raise RuntimeError(
            "Reference freeze raw phenotype hash mismatch: "
            f"{obj.get('raw_phenotype_sha256')}"
        )
    return obj


def build_bags(v1: Any, manifest: list[dict[str, str]], refs: list[dict[str, str]]):
    if len(manifest) != EXPECTED_NUCLEI:
        raise RuntimeError(f"Development nuclei={len(manifest)}; expected {EXPECTED_NUCLEI}")

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
        "v1_member",
        *FOLD_FIELDS,
    }
    missing = sorted(required - set(manifest[0]))
    if missing:
        raise RuntimeError(f"Development manifest missing columns: {missing}")

    if sum(str(row["v1_member"]).strip() == "1" for row in manifest) != EXPECTED_V1_MEMBERS:
        raise RuntimeError("Development manifest no longer preserves exactly 2,160 R1-v1 nuclei")

    ref_by_sample = {row["sample_name"]: row for row in refs}
    if len(ref_by_sample) != EXPECTED_BAGS:
        raise RuntimeError(
            f"Frozen sample-reference rows={len(ref_by_sample)}; expected {EXPECTED_BAGS}"
        )

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        grouped[row["sample_name"]].append(row)
    if len(grouped) != EXPECTED_BAGS:
        raise RuntimeError(f"Development bags={len(grouped)}; expected {EXPECTED_BAGS}")

    bags = []
    invariants = (
        "source_name",
        "strain",
        "sex",
        "particle_type",
        "dose_Gy",
        "hr_post_exposure",
        *FOLD_FIELDS,
    )
    for sample_name, rows in sorted(grouped.items()):
        if len(rows) != EXPECTED_NUCLEI_PER_BAG:
            raise RuntimeError(
                f"Sample {sample_name} has {len(rows)} nuclei; expected {EXPECTED_NUCLEI_PER_BAG}"
            )
        first = rows[0]
        for field in invariants:
            values = {str(row[field]) for row in rows}
            if len(values) != 1:
                raise RuntimeError(
                    f"Sample {sample_name} is not invariant for {field}: {sorted(values)}"
                )
        ref = ref_by_sample.get(sample_name)
        if ref is None:
            raise RuntimeError(f"Development sample absent from frozen reference: {sample_name}")
        if ref.get("reference_level") != "sample_plate_well_aggregate":
            raise RuntimeError(
                f"Unexpected reference level for {sample_name}: {ref.get('reference_level')!r}"
            )

        # The repaired development manifest must still refer to exactly the same
        # biological sample/condition as the frozen aggregate reference.
        for manifest_field, ref_field, normalizer in (
            ("source_name", "source_name", str),
            ("particle_type", "particle_type", str),
            ("dose_Gy", "dose_Gy", norm_number),
            ("hr_post_exposure", "hr_post_exposure", norm_hour),
        ):
            left = normalizer(first[manifest_field])
            right = normalizer(ref[ref_field])
            if left != right:
                raise RuntimeError(
                    f"Reference mismatch for {sample_name} {manifest_field}: dev={left!r} ref={right!r}"
                )

        bags.append(
            v1.Bag(
                sample_name=sample_name,
                source_name=first["source_name"],
                strain=first["strain"],
                sex=first["sex"],
                particle_type=first["particle_type"],
                dose_Gy=norm_number(first["dose_Gy"]),
                hr_post_exposure=norm_hour(first["hr_post_exposure"]),
                target_avg_nfoci=float(ref["nasa_avg_nfoci"]),
                target_avg_foci_no_outl=float(ref["nasa_avg_foci_no_outl"]),
                rows=sorted(rows, key=lambda row: row["sample_id"]),
                fold_roles={field: first[field] for field in FOLD_FIELDS},
            )
        )
    return bags


def bag_lookup(bags: list[Any]) -> dict[tuple[str, str, str, str], Any]:
    lookup: dict[tuple[str, str, str, str], Any] = {}
    for bag in bags:
        key = (
            bag.source_name,
            bag.particle_type,
            norm_number(bag.dose_Gy),
            norm_hour(bag.hr_post_exposure),
        )
        if key in lookup:
            raise RuntimeError(f"Duplicate development bag key: {key}")
        lookup[key] = bag
    return lookup


def build_training_pairs(train_bags: list[Any]) -> list[tuple[Any, Any]]:
    lookup = bag_lookup(train_bags)
    sources = sorted({bag.source_name for bag in train_bags})
    pairs: list[tuple[Any, Any]] = []
    for source in sources:
        for branch, exposed_dose in EXPOSED_DOSE.items():
            for hour in ("4", "24", "48"):
                sham_key = (source, branch, "0.0", hour)
                exp_key = (source, branch, norm_number(exposed_dose), hour)
                if sham_key not in lookup or exp_key not in lookup:
                    raise RuntimeError(f"Missing matched training contrast: {sham_key} / {exp_key}")
                pairs.append((lookup[sham_key], lookup[exp_key]))
    if len(sources) != 4 or len(pairs) != 24:
        raise RuntimeError(
            f"Unexpected training pair structure: sources={len(sources)} pairs={len(pairs)}"
        )
    used = [bag.sample_name for pair in pairs for bag in pair]
    if len(used) != 48 or len(set(used)) != 48:
        raise RuntimeError("Matched-pair epoch does not cover each of 48 training bags exactly once")
    return pairs


def pair_loss(
    model: torch.nn.Module,
    sham_x: torch.Tensor,
    exp_x: torch.Tensor,
    sham_target: torch.Tensor,
    exp_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sham_mean = model(sham_x).mean()
    exp_mean = model(exp_x).mean()
    abs_loss = 0.5 * (
        F.smooth_l1_loss(sham_mean, sham_target, beta=SMOOTHL1_BETA)
        + F.smooth_l1_loss(exp_mean, exp_target, beta=SMOOTHL1_BETA)
    )
    target_delta = exp_target - sham_target
    pred_delta = exp_mean - sham_mean
    delta_loss = F.smooth_l1_loss(pred_delta, target_delta, beta=SMOOTHL1_BETA)
    total = abs_loss + DELTA_LOSS_WEIGHT * delta_loss
    return total, abs_loss, delta_loss, sham_mean, exp_mean


def evaluate_bags(v1: Any, model: torch.nn.Module, bags: list[Any], cache: Any, device: torch.device, fold_name: str):
    bag_rows, nucleus_rows = v1.evaluate_bags(model, bags, cache, device, fold_name)
    for row in bag_rows:
        if "n_pilot_nuclei" in row:
            row["n_development_nuclei"] = row.pop("n_pilot_nuclei")
    return bag_rows, nucleus_rows


def train_fold(
    v1: Any,
    fold_field: str,
    bags: list[Any],
    cache: Any,
    device: torch.device,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fold_name = fold_field.removesuffix("_role")
    train_bags = [bag for bag in bags if bag.fold_roles[fold_field] == "train"]
    test_bags = [bag for bag in bags if bag.fold_roles[fold_field] == "test"]
    if len(train_bags) != 48 or len(test_bags) != 24:
        raise RuntimeError(
            f"Unexpected development split for {fold_name}: train={len(train_bags)} test={len(test_bags)}"
        )
    pairs = build_training_pairs(train_bags)

    fold_seed = SEED + FOLD_FIELDS.index(fold_field) * 1000
    seed_everything(fold_seed)
    model = v1.R1CountNetV1().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    rng = np.random.default_rng(fold_seed)

    history: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
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
            total, abs_loss, delta_loss, _, _ = pair_loss(
                model, sham_x, exp_x, sham_target, exp_target
            )
            total.backward()
            optimizer.step()

            totals.append(float(total.detach().cpu()))
            absolutes.append(float(abs_loss.detach().cpu()))
            deltas.append(float(delta_loss.detach().cpu()))

            del sham_x, exp_x, total, abs_loss, delta_loss

        scheduler.step()
        row = {
            "fold": fold_name,
            "epoch": epoch,
            "train_total_loss_mean": float(np.mean(totals)),
            "train_absolute_loss_mean": float(np.mean(absolutes)),
            "train_delta_loss_mean": float(np.mean(deltas)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(
                f"  {fold_name} epoch {epoch:3d}/{EPOCHS}: "
                f"total={row['train_total_loss_mean']:.5f} "
                f"abs={row['train_absolute_loss_mean']:.5f} "
                f"delta={row['train_delta_loss_mean']:.5f} "
                f"lr={row['learning_rate']:.3g}"
            )

    fold_dir = output_dir / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": "R1CountNetV1_backbone_R1V2_contrast",
        "fold": fold_name,
        "model_state_dict": model.state_dict(),
        "manifest_sha256": REPAIRED_DEV_MANIFEST_SHA256,
        "input_policy": {
            "canvas": v1.CANVAS,
            "channels": ["FITC_p1_p99.5", "DAPI_p1_p99.5", "ZERO"],
            "resize": False,
            "padding": "center_zero",
        },
        "target": "NASA sample-level avg_nfoci",
        "loss": {
            "absolute": "0.5*(SmoothL1(sham_mean, sham_target)+SmoothL1(exposed_mean, exposed_target))",
            "contrast": "SmoothL1((exposed_mean-sham_mean),(exposed_target-sham_target))",
            "beta": SMOOTHL1_BETA,
            "delta_weight": DELTA_LOSS_WEIGHT,
        },
        "bag_size": EXPECTED_NUCLEI_PER_BAG,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": fold_seed,
        "train_sources": sorted({bag.source_name for bag in train_bags}),
        "development_test_sources": sorted({bag.source_name for bag in test_bags}),
        "development_test_used_for_early_stopping": False,
        "final_holdout_used": False,
    }
    torch.save(checkpoint, fold_dir / "r1_countnet_v2_contrast.pt")
    write_csv(fold_dir / "training_history.csv", history)

    test_bag_rows, test_nucleus_rows = evaluate_bags(
        v1, model, test_bags, cache, device, fold_name
    )
    metrics = v1.regression_metrics(test_bag_rows)
    write_csv(fold_dir / "development_test_bag_predictions.csv", test_bag_rows)
    write_csv(fold_dir / "development_test_nucleus_predictions.csv", test_nucleus_rows)
    (fold_dir / "development_test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(
        f"  {fold_name} DEV-TEST: MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} "
        f"Pearson={metrics['pearson']:.4f} Spearman={metrics['spearman']:.4f}"
    )
    return test_bag_rows, test_nucleus_rows, metrics


def direction_fraction(rows: list[dict[str, Any]], hours: set[str]) -> tuple[int, int, float]:
    subset = [row for row in rows if str(row["timepoint_hr"]) in hours]
    passed = sum(int(row["direction_agreement"]) for row in subset)
    total = len(subset)
    return passed, total, float(passed / total) if total else float("nan")


def peak_time_recovery(rows: list[dict[str, Any]]) -> tuple[int, int, float]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source_name"]), str(row["branch"]))].append(row)
    recovered = 0
    for _, subset in grouped.items():
        if len(subset) != 3:
            raise RuntimeError("Each source x branch must have exactly 3 time-point deltas")
        nasa_peak = max(subset, key=lambda row: float(row["nasa_delta"]))["timepoint_hr"]
        pred_peak = max(subset, key=lambda row: float(row["pred_delta"]))["timepoint_hr"]
        recovered += int(str(nasa_peak) == str(pred_peak))
    total = len(grouped)
    return recovered, total, float(recovered / total) if total else float("nan")


def evaluate_gate(
    overall: dict[str, float],
    fold_metrics: dict[str, dict[str, float]],
    delta_summary: dict[str, float],
    delta_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    weakest_fold_spearman = min(float(m["spearman"]) for m in fold_metrics.values())
    four_ok, four_n, four_fraction = direction_fraction(delta_rows, {"4"})
    late_ok, late_n, late_fraction = direction_fraction(delta_rows, {"24", "48"})
    peak_ok, peak_n, peak_fraction = peak_time_recovery(delta_rows)

    values = {
        "overall_bag_spearman": float(overall["spearman"]),
        "weakest_fold_bag_spearman": weakest_fold_spearman,
        "overall_bag_mae": float(overall["mae"]),
        "delta_spearman": float(delta_summary["delta_spearman"]),
        "direction_agreement": float(delta_summary["direction_agreement_fraction"]),
        "four_hour_direction": four_fraction,
        "late_24_48_direction": late_fraction,
        "peak_time_recovery": peak_fraction,
    }
    checks = {
        "overall_bag_spearman": values["overall_bag_spearman"] >= GATE["overall_bag_spearman_min"],
        "weakest_fold_bag_spearman": values["weakest_fold_bag_spearman"] >= GATE["weakest_fold_bag_spearman_min"],
        "overall_bag_mae": values["overall_bag_mae"] <= GATE["overall_bag_mae_max"],
        "delta_spearman": values["delta_spearman"] >= GATE["delta_spearman_min"],
        "direction_agreement": values["direction_agreement"] >= GATE["direction_agreement_min"],
        "four_hour_direction": values["four_hour_direction"] >= GATE["four_hour_direction_min"],
        "late_24_48_direction": values["late_24_48_direction"] >= GATE["late_24_48_direction_min"],
        "peak_time_recovery": values["peak_time_recovery"] >= GATE["peak_time_recovery_min"],
    }
    return {
        "thresholds": GATE,
        "values": values,
        "checks": checks,
        "detail_counts": {
            "four_hour_direction": f"{four_ok}/{four_n}",
            "late_24_48_direction": f"{late_ok}/{late_n}",
            "peak_time_recovery": f"{peak_ok}/{peak_n}",
        },
        "all_pass": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"
    parser.add_argument(
        "--manifest",
        default=str(metadata_root / "r1_v2_freeze" / "r1_v2_development_manifest_100_qc1.csv"),
    )
    parser.add_argument(
        "--reference",
        default=str(
            metadata_root
            / "pilot_v1"
            / "reference_phenotypes_v2"
            / "pilot_v1_sample_reference_frozen.csv"
        ),
    )
    parser.add_argument(
        "--reference-freeze",
        default=str(
            metadata_root
            / "pilot_v1"
            / "reference_phenotypes_v2"
            / "reference_freeze_v2.json"
        ),
    )
    parser.add_argument(
        "--image-root",
        default=str(data_root / "nasa_bps_microscopy" / "r1_v2_development" / "images"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(model_root / "nasa_bps_53bp1" / "r1_v2_contrast"),
    )
    parser.add_argument("--cache-items", type=int, default=2048)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the fixed 60-epoch three-fold development-CV policy. Without this flag only dry-run validation occurs.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    reference_path = Path(args.reference).resolve()
    freeze_path = Path(args.reference_freeze).resolve()
    image_root = Path(args.image_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    for path in (manifest_path, reference_path, freeze_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != REPAIRED_DEV_MANIFEST_SHA256:
        raise RuntimeError(
            "Repaired R1 v2 development manifest SHA256 mismatch: "
            f"expected {REPAIRED_DEV_MANIFEST_SHA256}, got {manifest_sha}"
        )
    reference_sha = sha256_file(reference_path)
    freeze = validate_reference_freeze(freeze_path)
    manifest = read_csv(manifest_path)
    refs = read_csv(reference_path)

    v1 = load_v1_module()
    # Reuse only the implemented R1-v1 image/model utilities; v2 data checks are
    # handled above and below. These constants are used by its geometry/eval helpers.
    v1.EXPECTED_NUCLEI = EXPECTED_NUCLEI
    v1.EXPECTED_NUCLEI_PER_SAMPLE = EXPECTED_NUCLEI_PER_BAG
    v1.SEED = SEED
    bags = build_bags(v1, manifest, refs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = v1.NativeImageCache(image_root=image_root, max_items=args.cache_items)

    print("Radiation Edge AI - NASA BPS R1 v2 contrast-aware DEVELOPMENT model")
    print(f"mode: {'TRAIN' if args.train else 'DRY RUN'}")
    print("FINAL HOLDOUT STATUS: UNTOUCHED")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")
    print(f"development manifest SHA256: {manifest_sha}")
    print(f"frozen sample reference SHA256: {reference_sha}")
    print(f"reference freeze: {freeze.get('reference_freeze_version')}")
    print(f"bags: {len(bags)}")
    print(f"nuclei: {sum(len(bag.rows) for bag in bags)}")
    print(f"nuclei/bag: {EXPECTED_NUCLEI_PER_BAG}")
    print(f"model schedule: {EPOCHS} epochs AdamW lr={LEARNING_RATE:g} wd={WEIGHT_DECAY:g}")
    print(f"loss: L_abs + {DELTA_LOSS_WEIGHT:g} * L_delta; SmoothL1 beta={SMOOTHL1_BETA:g}")
    print("")

    print("[Development-CV folds]")
    for field in FOLD_FIELDS:
        train_bags = [bag for bag in bags if bag.fold_roles[field] == "train"]
        test_bags = [bag for bag in bags if bag.fold_roles[field] == "test"]
        pairs = build_training_pairs(train_bags)
        print(
            f"{field.removesuffix('_role')}: train={len(train_bags)} bags / {len(pairs)} matched pairs "
            f"{sorted({bag.source_name for bag in train_bags})}; "
            f"dev-test={len(test_bags)} bags {sorted({bag.source_name for bag in test_bags})}"
        )
    print("")

    print("[Full FITC/DAPI geometry validation]")
    geometry = v1.validate_all_geometry(bags, cache)
    print(
        f"nuclei={geometry['n_nuclei']} "
        f"height={geometry['height_min']}/{geometry['height_median']}/{geometry['height_max']} "
        f"width={geometry['width_min']}/{geometry['width_median']}/{geometry['width_max']}"
    )
    print(f"all fit 256x256 at native scale: {'YES' if geometry['all_fit_256_native_scale'] else 'NO'}")
    print(f"dtype counts: {geometry['dtype_counts']}")
    print("")

    seed_everything(SEED)
    smoke_model = v1.R1CountNetV1().to(device)
    smoke_rng = np.random.default_rng(SEED)
    first_train = [bag for bag in bags if bag.fold_roles[FOLD_FIELDS[0]] == "train"]
    smoke_pairs = build_training_pairs(first_train)
    sham, exp = smoke_pairs[0]
    sham_x = v1.load_bag_tensor(sham, cache, augment=False, rng=smoke_rng).to(device)
    exp_x = v1.load_bag_tensor(exp, cache, augment=False, rng=smoke_rng).to(device)
    sham_target = torch.tensor(sham.target_avg_nfoci, dtype=torch.float32, device=device)
    exp_target = torch.tensor(exp.target_avg_nfoci, dtype=torch.float32, device=device)
    with torch.no_grad():
        total, abs_loss, delta_loss, sham_mean, exp_mean = pair_loss(
            smoke_model, sham_x, exp_x, sham_target, exp_target
        )
    print("[Model/pair smoke]")
    print("model: R1CountNetV1 backbone (unchanged from R1 v1)")
    print(f"parameters: {v1.parameter_count(smoke_model):,}")
    print(f"single bag input: {tuple(sham_x.shape)}")
    print(f"matched pair: {sham.source_name} {sham.particle_type} {sham.hr_post_exposure}h")
    print(
        f"initial sham/exposed means: {float(sham_mean.cpu()):.4f}/{float(exp_mean.cpu()):.4f} "
        f"targets: {float(sham_target.cpu()):.4f}/{float(exp_target.cpu()):.4f}"
    )
    print(
        f"initial losses total/abs/delta: {float(total.cpu()):.5f}/"
        f"{float(abs_loss.cpu()):.5f}/{float(delta_loss.cpu()):.5f}"
    )
    print("")

    dry_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "train" if args.train else "dry_run",
        "manifest_sha256": manifest_sha,
        "sample_reference_sha256": reference_sha,
        "raw_phenotype_sha256": FROZEN_RAW_PHENOTYPE_SHA256,
        "bags": len(bags),
        "nuclei": sum(len(bag.rows) for bag in bags),
        "bag_size": EXPECTED_NUCLEI_PER_BAG,
        "geometry": geometry,
        "model": "R1CountNetV1_backbone_R1V2_contrast",
        "parameter_count": v1.parameter_count(smoke_model),
        "objective": {
            "smooth_l1_beta": SMOOTHL1_BETA,
            "delta_loss_weight": DELTA_LOSS_WEIGHT,
        },
        "schedule": {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
        },
        "development_gate": GATE,
        "final_holdout_read": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.train:
        (output_dir / "dry_run_summary.json").write_text(
            json.dumps(dry_summary, indent=2), encoding="utf-8"
        )
        print("NASA BPS R1 V2 DRY RUN COMPLETE: YES")
        print("No model was trained.")
        print("FINAL HOLDOUT STATUS: UNTOUCHED")
        print("NEXT GATE: rerun the same command with --train for the fixed 60-epoch development-CV run.")
        return 0

    (output_dir / "training_config.json").write_text(
        json.dumps(dry_summary, indent=2), encoding="utf-8"
    )

    print("[Training three development-CV folds]")
    all_oof_bags: list[dict[str, Any]] = []
    all_oof_nuclei: list[dict[str, Any]] = []
    fold_metrics: dict[str, dict[str, float]] = {}
    for field in FOLD_FIELDS:
        bag_rows, nucleus_rows, metrics = train_fold(
            v1, field, bags, cache, device, output_dir
        )
        all_oof_bags.extend(bag_rows)
        all_oof_nuclei.extend(nucleus_rows)
        fold_metrics[field.removesuffix("_role")] = metrics

    if len(all_oof_bags) != EXPECTED_BAGS:
        raise RuntimeError(f"OOF development bags={len(all_oof_bags)}; expected {EXPECTED_BAGS}")
    if len(all_oof_nuclei) != EXPECTED_NUCLEI:
        raise RuntimeError(
            f"OOF development nuclei={len(all_oof_nuclei)}; expected {EXPECTED_NUCLEI}"
        )

    all_oof_bags.sort(key=lambda row: (str(row["source_name"]), str(row["sample_name"])))
    all_oof_nuclei.sort(key=lambda row: str(row["sample_id"]))
    write_csv(output_dir / "oof_development_bag_predictions.csv", all_oof_bags)
    write_csv(output_dir / "oof_development_nucleus_predictions.csv", all_oof_nuclei)

    overall = v1.regression_metrics(all_oof_bags)
    delta_rows = v1.biological_deltas(all_oof_bags)
    delta_summary = v1.delta_metrics(delta_rows)
    write_csv(output_dir / "oof_development_matched_radiation_deltas.csv", delta_rows)

    gate = evaluate_gate(overall, fold_metrics, delta_summary, delta_rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "R1CountNetV1_backbone_R1V2_contrast",
        "fold_metrics": fold_metrics,
        "oof_bag_metrics": overall,
        "oof_biological_delta_metrics": delta_summary,
        "development_gate": gate,
        "final_holdout_read": False,
        "interpretation_guardrails": [
            "NASA avg_nfoci is sample-level aggregate supervision, not a per-nucleus label.",
            "The three historical source-held-out folds are development CV, not untouched final validation.",
            "The final holdout remains unopened by this script regardless of gate outcome.",
            "No KL720 or quantization equivalence is implied by this FP32 development result.",
        ],
    }
    (output_dir / "r1_v2_development_oof_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    four_ok, four_n, four_fraction = direction_fraction(delta_rows, {"4"})
    late_ok, late_n, late_fraction = direction_fraction(delta_rows, {"24", "48"})
    peak_ok, peak_n, peak_fraction = peak_time_recovery(delta_rows)

    print("")
    print("[Out-of-fold R1 v2 DEVELOPMENT summary]")
    print(
        f"bags={int(overall['n_bags'])} MAE={overall['mae']:.4f} RMSE={overall['rmse']:.4f} "
        f"Pearson={overall['pearson']:.4f} Spearman={overall['spearman']:.4f}"
    )
    print(
        f"matched radiation contrasts={int(delta_summary['n_matched_contrasts'])} "
        f"direction={delta_summary['direction_agreement_fraction']:.3f} "
        f"delta MAE={delta_summary['delta_mae']:.4f} "
        f"delta Spearman={delta_summary['delta_spearman']:.4f}"
    )
    print(f"4h direction: {four_ok}/{four_n} = {four_fraction:.3f}")
    print(f"24h+48h direction: {late_ok}/{late_n} = {late_fraction:.3f}")
    print(f"NASA peak-time recovered: {peak_ok}/{peak_n} = {peak_fraction:.3f}")
    print("")
    print("[Predeclared development gate]")
    for name, passed in gate["checks"].items():
        value = gate["values"][name]
        print(f"{name}: {value:.4f} -> {'PASS' if passed else 'FAIL'}")
    print(f"DEVELOPMENT GATE ALL PASS: {'YES' if gate['all_pass'] else 'NO'}")
    print("FINAL HOLDOUT STATUS: UNTOUCHED")
    print("")
    print("NASA BPS R1 V2 TRAINING COMPLETE: YES")
    print(f"Output: {output_dir}")
    if gate["all_pass"]:
        print("NEXT GATE: freeze the one-time final-holdout evaluation procedure before opening holdout images/phenotypes.")
    else:
        print("NEXT GATE: keep final holdout closed and revise only within the six-source development cohort.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
