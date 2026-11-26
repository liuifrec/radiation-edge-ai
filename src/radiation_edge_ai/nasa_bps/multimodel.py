"""Train and export candidate NASA models using the existing source-grouped folds.

This is a new common recipe, not a reproduction of the frozen R1 v2 training:
each training step uses a deterministic subsample of each matched sample bag;
final BatchNorm statistics use only mixed training bags; evaluation uses every
nucleus. Only the frozen NASA sample means supervise the
model. Historical preprocessing, contrast loss, and metrics are reused without
changing the frozen scripts. Neither development metrics nor ONNX parity qualify
a model for KL720 deployment.

Listing, metadata validation, and image SHA256 verification need no PyTorch.
Training needs the scientific environment; export additionally needs ONNX and
ONNX Runtime. No weights or datasets are downloaded by this entry point.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from .model_factory import MODEL_NAMES, checkpoint_identity, create_model, model_contract

FOLD_FIELDS = ("fold_female_role", "fold_male_1_role", "fold_male_2_role")
RECIPE_ID = "nasa_grouped_minibag_contrast_v1"
# These are the existing repaired development manifest and frozen sample reference.
MANIFEST_SHA256 = "23e21fdc8bf00a2814b64c56223cfd3d8e96eda71bdbff5d8ec74dae218cf5a9"
REFERENCE_SHA256 = "fcbf1339805e02a7ca9cec36b476438b0512b222bd70ac2f90ed75c238a0b01f"
V1_MANIFEST_SHA256 = "41ff2e4d64577f17c976dea5bbc17221fe63be9e2e7e2291309b520027fbe725"
RAW_PHENOTYPE_SHA256 = "d28dc5d9c53221c66a90074a92783fd4708ea83db0fc1a9c767e47500e643d6a"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing an empty result CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), **checkpoint_identity(path)}


def _check_sha(path: Path, expected: str) -> dict[str, Any]:
    identity = _identity(path)
    if identity["sha256"] != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {identity['sha256']} != {expected}")
    return identity


def _filename(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Expected a plain image filename, got {value!r}")
    return value


def validate_grouping(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Validate supplied frozen fold roles; never generate replacement folds."""
    if not rows:
        raise ValueError("Development manifest is empty")
    required = {"sample_id", "nucleus_key", "sample_name", "source_name",
                "fitc_filename", "dapi_filename", *FOLD_FIELDS}
    samples: dict[str, str] = {}
    roles: dict[str, dict[str, str]] = defaultdict(dict)
    seen = {field: set() for field in ("sample_id", "nucleus_key",
                                      "fitc_filename", "dapi_filename")}
    for row in rows:
        if any(not row.get(key) for key in required):
            raise ValueError("Manifest has missing required fields or empty identifiers")
        for field, values in seen.items():
            if row[field] in values:
                raise ValueError(f"Duplicate {field}: {row[field]}")
            values.add(row[field])
        _filename(row["fitc_filename"])
        _filename(row["dapi_filename"])
        source, sample = row["source_name"], row["sample_name"]
        if sample in samples and samples[sample] != source:
            raise ValueError(f"Sample spans source groups: {sample}")
        samples[sample] = source
        for fold in FOLD_FIELDS:
            role = row[fold]
            if role not in {"train", "test"}:
                raise ValueError(f"Unexpected {fold} role {role!r}")
            if fold in roles[source] and roles[source][fold] != role:
                raise ValueError(f"Source leakage in {fold}: {source}")
            roles[source][fold] = role
    if any(sum(value == "test" for value in item.values()) != 1
           for item in roles.values()):
        raise ValueError("Each development source must be held out in exactly one frozen fold")
    folds = {}
    for fold in FOLD_FIELDS:
        train = sorted(source for source, values in roles.items() if values[fold] == "train")
        test = sorted(source for source, values in roles.items() if values[fold] == "test")
        if not train or not test:
            raise ValueError(f"Empty train or test source set: {fold}")
        folds[fold] = {"train_sources": train, "test_sources": test,
                       "train_bags": sum(source in train for source in samples.values()),
                       "test_bags": sum(source in test for source in samples.values())}
    return {"nuclei": len(rows), "bags": len(samples), "sources": len(roles), "folds": folds}


def verify_images(rows: list[dict[str, str]], image_root: Path,
                  inventory: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    """Record current TIFF bytes, checking prior receipts when explicitly supplied.

    The historical geometry repair did not update download_inventory.csv. A new
    measured snapshot must not be described as historical image verification.
    Supplying an inventory makes every missing or mismatched receipt fatal.
    """
    expected = {}
    for item in inventory or []:
        if item.get("channel") not in {"fitc", "dapi"}:
            continue
        key = (item.get("sample_id"), item["channel"])
        if key in expected:
            raise ValueError(f"Duplicate download inventory record: {key}")
        expected[key] = item
    verified = []
    for row in rows:
        for channel in ("fitc", "dapi"):
            filename = _filename(row[f"{channel}_filename"])
            path = image_root / channel / filename
            if inventory is None:
                identity = _identity(path)
            else:
                item = expected.get((row["sample_id"], channel))
                if item is None or item.get("filename") != filename:
                    raise ValueError(f"Missing or mismatched inventory record: {row['sample_id']} {channel}")
                digest = item.get("sha256", "")
                if (item.get("status") == "failed" or len(digest) != 64
                        or any(c not in "0123456789abcdef" for c in digest)):
                    raise ValueError(f"Invalid image receipt: {filename}")
                identity = _check_sha(path, digest)
                if int(item["bytes"]) != identity["size_bytes"]:
                    raise ValueError(f"Image byte count mismatch: {filename}")
            verified.append({"sample_id": row["sample_id"], "nucleus_key": row["nucleus_key"],
                             "sample_name": row["sample_name"], "source_name": row["source_name"],
                             "channel": channel, "filename": filename,
                             "sha256": identity["sha256"], "bytes": identity["size_bytes"]})
    return verified


def validate_geometry(rows: list[dict[str, str]], image_root: Path) -> dict[str, Any]:
    """Reject bad geometry and nonfinite pixels before any model is trained."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NASA preflight needs numpy and OpenCV to inspect TIFF geometry") from exc
    heights, widths = [], []
    for row in rows:
        images = [cv2.imread(str(image_root / channel / row[f"{channel}_filename"]),
                             cv2.IMREAD_UNCHANGED) for channel in ("fitc", "dapi")]
        if any(image is None for image in images):
            raise ValueError(f"Unreadable FITC/DAPI image: {row['sample_id']}")
        fitc, dapi = images
        if fitc.ndim != 2 or fitc.shape != dapi.shape:
            raise ValueError(f"FITC/DAPI geometry mismatch: {row['sample_id']}")
        height, width = fitc.shape
        if not (0 < height <= 256 and 0 < width <= 256):
            raise ValueError(f"Native crop does not fit 256x256; resizing is forbidden: {row['sample_id']}")
        if any(not np.isfinite(image).all() for image in images):
            raise ValueError(f"Nonfinite image pixels: {row['sample_id']}")
        heights.append(height)
        widths.append(width)
    return {"nuclei": len(rows), "height_min": min(heights), "height_max": max(heights),
            "width_min": min(widths), "width_max": max(widths),
            "all_pixels_finite": True, "all_fit_256_native_scale": True}


def select_bag_rows(rows: list[dict[str, str]], count: int, *, seed: int,
                    purpose: str) -> list[dict[str, str]]:
    """Sample without replacement; selection is independent of model RNG usage."""
    if not 1 <= count <= len(rows):
        raise ValueError(f"Nuclei per bag must be between 1 and {len(rows)}")
    def rank(row: dict[str, str]) -> str:
        return hashlib.sha256(f"{seed}|{purpose}|{row['nucleus_key']}".encode()).hexdigest()
    chosen = sorted(rows, key=lambda row: (rank(row), row["sample_id"]))[:count]
    return sorted(chosen, key=lambda row: row["sample_id"])


def calibration_rows(rows: list[dict[str, str]], fold: str, *, seed: int,
                     per_bag: int) -> list[dict[str, str]]:
    """Create a fold-specific calibration subset containing only training sources."""
    grouping = validate_grouping(rows)
    if fold not in FOLD_FIELDS:
        raise ValueError(f"Unknown fold: {fold}")
    train_sources = set(grouping["folds"][fold]["train_sources"])
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row[fold] == "train" and row["source_name"] in train_sources:
            grouped[row["sample_name"]].append(row)
    selected = []
    for _, bag_rows in sorted(grouped.items()):
        for row in select_bag_rows(bag_rows, per_bag, seed=seed, purpose="ptq_calibration"):
            selected.append({**row, "dataset_role": "development_train_ptq_calibration",
                             "calibration_fold": fold})
    return selected


def _dependencies() -> dict[str, Any]:
    packages = {"numpy": "numpy", "torch": "torch", "torchvision": "torchvision",
                "cv2": "opencv-python", "onnx": "onnx", "onnxruntime": "onnxruntime"}
    result = {}
    for module, distribution in packages.items():
        available = importlib.util.find_spec(module) is not None
        try:
            version = metadata.version(distribution) if available else None
        except metadata.PackageNotFoundError:
            version = "distribution name unavailable"
        result[module] = {"available": available, "version": version}
    return result


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, str]],
                                                list[dict[str, Any]]]:
    """Pin the existing development membership, references, roles, and image bytes."""
    manifest = _check_sha(args.manifest, MANIFEST_SHA256)
    reference = _check_sha(args.reference, REFERENCE_SHA256)
    freeze = json.loads(args.reference_freeze.read_text(encoding="utf-8"))
    expected = {"reference_level_frozen": True, "reference_level": "sample_plate_well_aggregate",
                "primary_reference_endpoint": "avg_nfoci", "manifest_sha256": V1_MANIFEST_SHA256,
                "raw_phenotype_sha256": RAW_PHENOTYPE_SHA256}
    if any(freeze.get(key) != value for key, value in expected.items()):
        raise ValueError("NASA reference freeze identity or aggregate endpoint changed")
    rows = _read_csv(args.manifest)
    grouping = validate_grouping(rows)
    if (grouping["nuclei"], grouping["bags"], grouping["sources"]) != (7200, 72, 6):
        raise ValueError("Expected the repaired 7,200-nucleus, 72-bag, six-source development cohort")
    if any((len(fold["train_sources"]), len(fold["test_sources"]),
            fold["train_bags"], fold["test_bags"]) != (4, 2, 48, 24)
           for fold in grouping["folds"].values()):
        raise ValueError("Frozen folds must contain 48 training and 24 held-out sample bags")
    verified = verify_images(rows, args.image_root,
                             _read_csv(args.inventory) if args.inventory else None)
    geometry = validate_geometry(rows, args.image_root)
    report = {"schema_version": 1, "recipe_id": RECIPE_ID,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "dataset_role": "development_only", "final_holdout_touched": False,
              "manifest": manifest, "reference": reference,
              "reference_freeze": _identity(args.reference_freeze),
              "download_inventory": _identity(args.inventory) if args.inventory else None,
              "image_hash_provenance": "checked_against_supplied_receipts" if args.inventory else
                  "newly_measured_snapshot_not_historical_image_verification",
              "image_root": str(args.image_root.resolve()),
              "hashed_fitc_dapi_files": len(verified), "geometry": geometry, **grouping,
              "dependencies": _dependencies(),
              "dependency_note": "Availability only; train/export also check actual imports.",
              "qualification": "candidate_only_no_accuracy_or_device_qualification"}
    return report, rows, verified


def _runtime():
    try:
        import numpy as np
        import torch

        from . import train_pilot_v1_r1_mil as v1
        from . import train_r1_v2_contrast as v2
    except ImportError as exc:
        raise RuntimeError("Training needs numpy, torch and OpenCV in the scientific environment") from exc
    return np, torch, v1, v2


def train_one_epoch(v1: Any, v2: Any, model: Any, pairs: list[Any], cache: Any,
                    optimizer: Any, device: Any, *, seed: int, epoch: int,
                    nuclei_per_bag: int) -> dict[str, float]:
    """Use aggregate targets once per matched pair, never individual pseudo-counts."""
    import numpy as np
    import torch
    model.train()
    rng = np.random.default_rng(seed + epoch)
    totals = np.zeros(3, dtype=np.float64)
    for pair_index in rng.permutation(len(pairs)):
        sham, exposed = pairs[int(pair_index)]
        inputs = []
        for bag in (sham, exposed):
            subset = select_bag_rows(bag.rows, nuclei_per_bag, seed=seed,
                                     purpose=f"train_epoch_{epoch}")
            minibag = replace(bag, rows=subset)
            inputs.append(v1.load_bag_tensor(minibag, cache, augment=True, rng=rng).to(device))
        targets = [torch.tensor(bag.target_avg_nfoci, device=device, dtype=torch.float32)
                   for bag in (sham, exposed)]
        optimizer.zero_grad(set_to_none=True)
        total, absolute, delta, _, _ = v2.pair_loss(model, *inputs, *targets)
        if not bool(torch.isfinite(total)):
            raise RuntimeError(f"Nonfinite aggregate loss in epoch {epoch}")
        total.backward()
        optimizer.step()
        totals += [float(total.detach()), float(absolute.detach()), float(delta.detach())]
    return dict(zip(("loss", "absolute_loss", "contrast_loss"),
                    (float(value) / len(pairs) for value in totals)))


def evaluate_bags_chunked(v1: Any, model: Any, bags: list[Any], cache: Any,
                          device: Any, fold: str, chunk_size: int):
    """Evaluate all nuclei, then compute each sample's arithmetic-mean prediction."""
    import numpy as np
    import torch
    if chunk_size < 1:
        raise ValueError("Evaluation chunk size must be positive")
    model.eval()
    bag_rows, nucleus_rows = [], []
    with torch.no_grad():
        for bag in bags:
            values = []
            for start in range(0, len(bag.rows), chunk_size):
                chunk = replace(bag, rows=bag.rows[start:start + chunk_size])
                x = v1.load_bag_tensor(chunk, cache, augment=False,
                                       rng=np.random.default_rng(0)).to(device)
                predictions = model(x).detach().cpu().numpy().astype(np.float64)
                if predictions.shape != (len(chunk.rows),) or not np.isfinite(predictions).all():
                    raise RuntimeError("Invalid per-nucleus model output during evaluation")
                values.extend(predictions.tolist())
            mean = float(np.mean(values))
            shared = {"fold": fold, "sample_name": bag.sample_name,
                      "source_name": bag.source_name, "strain": bag.strain, "sex": bag.sex,
                      "particle_type": bag.particle_type, "dose_Gy": bag.dose_Gy,
                      "hr_post_exposure": bag.hr_post_exposure}
            bag_rows.append({**shared, "n_development_nuclei": len(values),
                             "nasa_avg_nfoci": bag.target_avg_nfoci,
                             "nasa_avg_foci_no_outl": bag.target_avg_foci_no_outl,
                             "predicted_bag_mean": mean, "residual": mean - bag.target_avg_nfoci})
            for row, value in zip(bag.rows, values):
                nucleus_rows.append({**shared, "sample_id": row["sample_id"],
                                     "nucleus_key": row["nucleus_key"],
                                     "predicted_continuous_burden": value,
                                     "label_semantics": "unlabeled_individual_prediction_calibrated_only_by_sample_mean"})
    return bag_rows, nucleus_rows


def recalibrate_batchnorm(v1: Any, model: Any, bags: list[Any], cache: Any,
                          device: Any, fold: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Update BN buffers using one nucleus from each training bag per batch.

    All 48 training bags contribute to each of 100 equally sized batches. This
    adapts the historical mixed-bag BN repair to each development fold; held-out
    sources never enter the recalibration. Learned parameters remain unchanged.
    """
    import numpy as np
    import torch
    source_roles: dict[str, set[str]] = defaultdict(set)
    for bag in bags:
        source_roles[bag.source_name].add(bag.fold_roles[fold])
    if any(len(roles) != 1 or not roles <= {"train", "test"}
           for roles in source_roles.values()):
        raise ValueError("Source leakage or invalid role in BatchNorm recalibration")
    training = sorted((bag for bag in bags if bag.fold_roles[fold] == "train"),
                      key=lambda bag: bag.sample_name)
    sizes = {len(bag.rows) for bag in training}
    if not training or len(sizes) != 1 or 0 in sizes:
        raise ValueError("BatchNorm recalibration needs equally sized, nonempty training bags")
    if any(row.get(fold) != "train" or row.get("source_name") != bag.source_name
           for bag in training for row in bag.rows):
        raise ValueError("Non-training nucleus or mismatched source in BatchNorm recalibration")
    ordered = [sorted(bag.rows, key=lambda row: row["sample_id"]) for bag in training]
    batches = [[rows[rank] for rows in ordered] for rank in range(next(iter(sizes)))]
    selected = [{**row, "dataset_role": "development_train_bn_calibration",
                 "calibration_fold": fold} for batch in batches for row in batch]
    bn_layers = [(name, module) for name, module in model.named_modules()
                 if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    if not bn_layers or any(not module.track_running_stats for _, module in bn_layers):
        raise ValueError("This recipe requires BatchNorm layers with tracked running statistics")
    before = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    allowed = {f"{name}." + suffix if name else suffix for name, _ in bn_layers
               for suffix in ("running_mean", "running_var", "num_batches_tracked")}
    momenta = [module.momentum for _, module in bn_layers]
    model.eval()
    try:
        for _, module in bn_layers:
            module.reset_running_stats()
            module.momentum = None
            module.train()
        with torch.no_grad():
            for batch in batches:
                x = np.stack([v1.pad_and_pack(cache.get(row)) for row in batch], axis=0)
                output = model(torch.from_numpy(x).to(device))
                if not bool(torch.isfinite(output).all()):
                    raise RuntimeError("Nonfinite output during training-only BN recalibration")
    finally:
        for (_, module), momentum in zip(bn_layers, momenta):
            module.momentum = momentum
        model.eval()
    changed = [name for name, tensor in model.state_dict().items()
               if not torch.equal(before[name], tensor.detach().cpu())]
    if set(changed) - allowed:
        raise RuntimeError("BatchNorm recalibration changed a learned parameter or other protected state")
    if not changed:
        raise RuntimeError("BatchNorm recalibration did not change any running state")
    identity = hashlib.sha256("\n".join(row["sample_id"] for row in selected).encode()).hexdigest()
    return {"dataset_role": "development_train_bn_calibration", "fold": fold,
            "sources": sorted({bag.source_name for bag in training}),
            "sample_names": [bag.sample_name for bag in training],
            "calibration_nuclei": len(selected), "mixed_batch_size": len(training),
            "calibration_batches": len(batches), "batchnorm_layers": len(bn_layers),
            "row_order": "nucleus_rank_major_sorted_sample_name_minor",
            "row_order_sha256": identity, "augmentation": False,
            "running_stat_rule": "reset_then_cumulative_average_momentum_none",
            "learned_parameters_bit_identical": True, "other_state_bit_identical": True,
            "changed_running_state": changed, "held_out_sources_used": []}, selected


def _metrics(v1: Any, bags: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = v1.biological_deltas(bags)
    return {"bag": v1.regression_metrics(bags), "matched_contrast": v1.delta_metrics(deltas),
            "per_source": {source: v1.regression_metrics(
                [row for row in bags if row["source_name"] == source])
                for source in sorted({row["source_name"] for row in bags})},
            "qualification": "development_metrics_only"}


def _fresh_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite a run: {args.out}")
    _, torch, v1, v2 = _runtime()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly for a smoke run")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    report, rows, verified = preflight(args)
    bags = v2.build_bags(v1, rows, _read_csv(args.reference))
    v2.validate_reference_freeze(args.reference_freeze)
    cache = v1.NativeImageCache(args.image_root, max_items=512)
    pairs_by_fold = {fold: v2.build_training_pairs([bag for bag in bags
                                                  if bag.fold_roles[fold] == "train"])
                     for fold in args.folds}
    # Validate all requested architectures before creating a partially populated run.
    for architecture in args.models:
        create_model(architecture, pretrained=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    _fresh_directory(args.out)
    _write_csv(args.out / "verified_image_inventory.csv", verified)
    report["verified_image_inventory"] = _identity(args.out / "verified_image_inventory.csv")
    _write_json(args.out / "preflight.json", report)
    image_hashes = {(row["sample_id"], row["channel"]): row["sha256"] for row in verified}
    code = {name: _identity(Path(__file__).with_name(name)) for name in
            ("multimodel.py", "model_factory.py", "train_pilot_v1_r1_mil.py", "train_r1_v2_contrast.py")}
    recipe = {"id": RECIPE_ID, "epochs": args.epochs, "learning_rate": 3e-4,
              "weight_decay": 1e-4, "optimizer": "AdamW", "scheduler": "CosineAnnealingLR",
              "train_nuclei_per_bag": args.train_nuclei_per_bag,
              "eval_nuclei_per_bag": 100, "eval_chunk_size": args.eval_chunk_size,
              "absolute_smooth_l1_beta": 0.5, "contrast_loss_weight": 1.0,
              "selection": "last_fixed_epoch_no_held_out_selection",
              "batchnorm_recalibration": "mandatory_train_only_48_bags_x_100_rank_batches",
              "batchnorm_augmentation": False, "batchnorm_momentum": None,
              "individual_nucleus_labels": False, "pretrained_backbone": False,
              "historical_checkpoint": "not_loaded_architecture_only_comparator"}
    runtime = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
               "cudnn": torch.backends.cudnn.version(), "device": str(device),
               "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    benchmark = {"schema_version": 1, "recipe": recipe, "seed": args.seed,
                 "models": args.models, "folds": args.folds, "dataset": report,
                 "implementation": code, "runtime": runtime, "status": "running"}
    _write_json(args.out / "benchmark.json", benchmark)
    comparison = []
    for architecture in args.models:
        all_bags = []
        for fold in args.folds:
            fold_seed = args.seed + FOLD_FIELDS.index(fold) * 1000
            v2.seed_everything(fold_seed)
            model = create_model(architecture, pretrained=False).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            output = args.out / architecture / fold
            _fresh_directory(output)
            calibration = calibration_rows(rows, fold, seed=fold_seed,
                                            per_bag=args.calibration_nuclei_per_bag)
            for row in calibration:
                for channel in ("fitc", "dapi"):
                    row[f"{channel}_sha256"] = image_hashes[(row["sample_id"], channel)]
            _write_csv(output / "calibration_manifest.csv", calibration)
            run = {"schema_version": 1, "status": "training", "architecture": architecture,
                   "contract": model_contract(architecture), "fold": fold, "seed": fold_seed,
                   "recipe": recipe, "dataset": report, "implementation": code,
                   "runtime": runtime,
                   "source_split": report["folds"][fold],
                   "calibration": _identity(output / "calibration_manifest.csv"),
                   "qualification": "candidate_only_no_accuracy_or_device_qualification"}
            _write_json(output / "run.json", run)
            history = []
            for epoch in range(1, args.epochs + 1):
                losses = train_one_epoch(v1, v2, model, pairs_by_fold[fold], cache,
                                         optimizer, device, seed=fold_seed, epoch=epoch,
                                         nuclei_per_bag=args.train_nuclei_per_bag)
                history.append({"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                                **losses})
                scheduler.step()
                _write_csv(output / "training_history.csv", history)
                print(f"{architecture} {fold} epoch {epoch}/{args.epochs} loss={losses['loss']:.6f}",
                      flush=True)
            state_metadata = {"architecture": architecture, "fold": fold, "recipe_id": RECIPE_ID,
                              "manifest_sha256": MANIFEST_SHA256, "seed": fold_seed}
            parent = output / "trained_before_bn.pt"
            torch.save({**state_metadata, "state_dict": {key: value.detach().cpu()
                        for key, value in model.state_dict().items()}}, parent)
            run.update(status="trained_pending_bn_calibration", parent_checkpoint=_identity(parent))
            _write_json(output / "run.json", run)
            optimizer.zero_grad(set_to_none=True)
            # Training is complete; release Adam state before BN and reload
            # temporarily allocate their own working tensors and model copy.
            del optimizer, scheduler
            bn_audit, bn_rows = recalibrate_batchnorm(v1, model, bags, cache, device, fold)
            _write_csv(output / "bn_calibration_manifest.csv", bn_rows)
            bn_audit["manifest"] = _identity(output / "bn_calibration_manifest.csv")
            run["batchnorm_recalibration"] = bn_audit
            checkpoint = output / "model.pt"
            torch.save({**state_metadata, "parent_checkpoint_sha256": run["parent_checkpoint"]["sha256"],
                        "batchnorm_recalibration": bn_audit,
                        "state_dict": {key: value.detach().cpu()
                                       for key, value in model.state_dict().items()}}, checkpoint)
            run.update(status="trained_pending_evaluation", checkpoint=_identity(checkpoint))
            # Verify the serialized final state before evaluating any held-out bag.
            reloaded = create_model(architecture, pretrained=False).to(device).eval()
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            reloaded.load_state_dict(saved["state_dict"], strict=True)
            probe = torch.from_numpy(v1.pad_and_pack(cache.get(calibration[0]))[None]).to(device)
            with torch.no_grad():
                expected, actual = model(probe), reloaded(probe)
                error = float(torch.max(torch.abs(expected - actual)))
                if not bool(torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)):
                    raise RuntimeError("Serialized BN-calibrated checkpoint failed reload parity")
            run["serialization_reload"] = {"passed": True, "sample_id": calibration[0]["sample_id"],
                                            "dataset_role": "development_train_probe",
                                            "max_absolute_error": error, "atol": 1e-6, "rtol": 1e-5}
            del model
            model = reloaded
            del reloaded, saved, probe, expected, actual
            _write_json(output / "run.json", run)
            held_out = [bag for bag in bags if bag.fold_roles[fold] == "test"]
            bag_rows, nuclei = evaluate_bags_chunked(v1, model, held_out, cache,
                                                    device, fold, args.eval_chunk_size)
            _write_csv(output / "bag_predictions.csv", bag_rows)
            _write_csv(output / "nucleus_predictions.csv", nuclei)
            _write_csv(output / "matched_contrasts.csv", v1.biological_deltas(bag_rows))
            verify_images(rows, args.image_root, verified)
            run.update(status="complete", metrics=_metrics(v1, bag_rows),
                       image_snapshot_reverified_after_fold=True)
            _write_json(output / "run.json", run)
            all_bags.extend(bag_rows)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        _write_csv(args.out / architecture / "oof_bag_predictions.csv", all_bags)
        metrics = _metrics(v1, all_bags)
        complete = set(args.folds) == set(FOLD_FIELDS)
        _write_json(args.out / architecture / "oof_metrics.json",
                    {"all_frozen_folds_complete": complete, "folds": args.folds, **metrics})
        comparison.append({"architecture": architecture, "recipe_id": RECIPE_ID,
                           "n_folds": len(args.folds), "all_frozen_folds_complete": complete,
                           "bag_mae": metrics["bag"]["mae"], "bag_rmse": metrics["bag"]["rmse"],
                           "bag_spearman": metrics["bag"]["spearman"],
                           "contrast_mae": metrics["matched_contrast"]["delta_mae"],
                           "contrast_direction_agreement": metrics["matched_contrast"]["direction_agreement_fraction"],
                           "qualification": "development_candidate_only"})
        _write_csv(args.out / "comparison.csv", _json_safe(comparison))
    benchmark["status"] = "complete"
    benchmark["comparison"] = _identity(args.out / "comparison.csv")
    _write_json(args.out / "benchmark.json", benchmark)
    return {"output": str(args.out.resolve()), "status": "complete", "recipe": recipe,
            "qualification": "development_metrics_only_pending_onnx_ptq_and_device_validation"}


def export(args: argparse.Namespace) -> dict[str, Any]:
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite an export: {args.out}")
    np, torch, v1, _ = _runtime()
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Export requires onnx and onnxruntime in the scientific environment") from exc
    run = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    if run.get("status") != "complete" or run["recipe"]["id"] != RECIPE_ID:
        raise ValueError("Export requires a completed new-recipe fold run")
    bn_audit = run.get("batchnorm_recalibration", {})
    if (not bn_audit.get("learned_parameters_bit_identical")
            or not bn_audit.get("other_state_bit_identical")
            or not bn_audit.get("changed_running_state")
            or bn_audit.get("held_out_sources_used") != []
            or not run.get("serialization_reload", {}).get("passed")):
        raise ValueError("Export requires the mandatory training-only BN audit and checkpoint reload parity")
    checkpoint_path = args.run_dir / "model.pt"
    checkpoint = _check_sha(checkpoint_path, run["checkpoint"]["sha256"])
    parent = _check_sha(args.run_dir / "trained_before_bn.pt", run["parent_checkpoint"]["sha256"])
    _check_sha(args.run_dir / "calibration_manifest.csv", run["calibration"]["sha256"])
    _check_sha(args.run_dir / "bn_calibration_manifest.csv", bn_audit["manifest"]["sha256"])
    for name in ("model_factory.py", "train_pilot_v1_r1_mil.py"):
        _check_sha(Path(__file__).with_name(name), run["implementation"][name]["sha256"])
    architecture, fold = run["architecture"], run["fold"]
    if run["contract"] != model_contract(architecture):
        raise ValueError("Model input/output contract changed since training")
    calibration = _read_csv(args.run_dir / "calibration_manifest.csv")
    train_sources = set(run["source_split"]["train_sources"])
    test_sources = set(run["source_split"]["test_sources"])
    if train_sources & test_sources or any(
        row.get(fold) != "train" or row.get("calibration_fold") != fold
        or row.get("dataset_role") != "development_train_ptq_calibration"
        or row["source_name"] not in train_sources or row["source_name"] in test_sources
        for row in calibration
    ):
        raise ValueError("PTQ calibration contains a non-training source or incorrect role")
    bn_rows = _read_csv(args.run_dir / "bn_calibration_manifest.csv")
    if (bn_audit.get("fold") != fold or bn_audit.get("sources") != sorted(train_sources)
            or len(bn_rows) != bn_audit.get("calibration_nuclei")
            or any(row.get(fold) != "train" or row.get("calibration_fold") != fold
                   or row.get("dataset_role") != "development_train_bn_calibration"
                   or row["source_name"] not in train_sources or row["source_name"] in test_sources
                   for row in bn_rows)):
        raise ValueError("BatchNorm calibration contains a non-training source or incorrect role")
    image_root = args.image_root or Path(run["dataset"]["image_root"])
    for row in calibration:
        for channel in ("fitc", "dapi"):
            _check_sha(image_root / channel / _filename(row[f"{channel}_filename"]),
                       row[f"{channel}_sha256"])
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise RuntimeError("Export requires a PyTorch version supporting weights_only=True")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (state["architecture"], state["fold"], state["recipe_id"], state["manifest_sha256"]) != (
            architecture, fold, RECIPE_ID, MANIFEST_SHA256):
        raise ValueError("Checkpoint does not match the recorded architecture, fold, or NASA manifest")
    if (state.get("parent_checkpoint_sha256") != parent["sha256"]
            or state.get("batchnorm_recalibration") != bn_audit):
        raise ValueError("Checkpoint lacks the recorded BN calibration and parent checkpoint identity")
    model = create_model(architecture, pretrained=False).cpu().eval()
    model.load_state_dict(state["state_dict"], strict=True)
    cache = v1.NativeImageCache(image_root, max_items=4)
    example = torch.from_numpy(v1.pad_and_pack(cache.get(calibration[0]))[None])
    _fresh_directory(args.out)
    onnx_path = args.out / f"{architecture}_{fold}_fp32_opset11.onnx"
    options = {"opset_version": 11, "input_names": ["input"], "output_names": ["burden"],
               "do_constant_folding": True, "training": torch.onnx.TrainingMode.EVAL}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False
    torch.onnx.export(model, example, str(onnx_path), **options)
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    calibration_dir = args.out / "calibration_nchw"
    calibration_dir.mkdir()
    parity_rows, arrays = [], []
    with torch.no_grad():
        for index, row in enumerate(calibration):
            x = v1.pad_and_pack(cache.get(row))[None].astype(np.float32)
            array_path = calibration_dir / f"{index:04d}.npy"
            np.save(array_path, x, allow_pickle=False)
            expected = model(torch.from_numpy(x)).numpy()
            actual = session.run(["burden"], {"input": x})[0]
            passed = bool(expected.shape == actual.shape == (1,)
                          and np.isfinite(expected).all() and np.isfinite(actual).all()
                          and np.allclose(expected, actual, atol=1e-5, rtol=1e-4))
            parity_rows.append({"sample_id": row["sample_id"], "source_name": row["source_name"],
                                "pytorch": float(expected.reshape(-1)[0]),
                                "onnx": float(actual.reshape(-1)[0]),
                                "absolute_error": float(np.max(np.abs(expected - actual))),
                                "parity_pass": passed})
            arrays.append({**row, "array_file": array_path.relative_to(args.out).as_posix(),
                           "array_sha256": checkpoint_identity(array_path)["sha256"]})
    _write_csv(args.out / "calibration_manifest.csv", arrays)
    _write_csv(args.out / "onnx_parity.csv", parity_rows)
    passed = all(row["parity_pass"] for row in parity_rows)
    result = {"schema_version": 1, "architecture": architecture, "fold": fold,
              "checkpoint": checkpoint, "parent_checkpoint": parent,
              "batchnorm_recalibration": bn_audit, "training_run": _identity(args.run_dir / "run.json"),
              "onnx": _identity(onnx_path), "contract": run["contract"],
              "dataset_role": "development_train_ptq_calibration", "source_split": run["source_split"],
              "calibration": _identity(args.out / "calibration_manifest.csv"),
              "calibration_arrays": len(arrays), "parity_pass": passed,
              "parity_atol": 1e-5, "parity_rtol": 1e-4,
              "max_absolute_error": max(row["absolute_error"] for row in parity_rows),
              "dependencies": _dependencies(),
              "qualification": "fp32_export_only_pending_int8_compile_accuracy_and_kl720_execution"}
    _write_json(args.out / "export.json", result)
    if not passed:
        raise RuntimeError(f"ONNX parity failed; inspect {args.out / 'export.json'}")
    return result


def _positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _data_arguments(parser: argparse.ArgumentParser) -> None:
    root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    data = root / "nasa_bps_microscopy"
    reference = data / "metadata" / "pilot_v1" / "reference_phenotypes_v2"
    parser.add_argument("--manifest", type=Path, default=data / "metadata" / "r1_v2_freeze" /
                        "r1_v2_development_manifest_100_qc1.csv")
    parser.add_argument("--reference", type=Path, default=reference / "pilot_v1_sample_reference_frozen.csv")
    parser.add_argument("--reference-freeze", type=Path, default=reference / "reference_freeze_v2.json")
    parser.add_argument("--image-root", type=Path, default=data / "r1_v2_development" / "images")
    parser.add_argument("--inventory", type=Path,
                        help="Optional matching repaired-data SHA receipts; mismatches are fatal. "
                             "Default records a fresh snapshot because historical QC repair did not update receipts.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List candidates and contracts without importing PyTorch")
    check = commands.add_parser("preflight", help="Verify frozen NASA membership, grouping, and image SHA256s")
    _data_arguments(check)
    fit = commands.add_parser("train", help="Train new models on frozen development source folds")
    _data_arguments(fit)
    fit.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    fit.add_argument("--folds", nargs="+", choices=FOLD_FIELDS, default=list(FOLD_FIELDS))
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--epochs", type=_positive, default=60)
    fit.add_argument("--seed", type=int, default=720)
    fit.add_argument("--train-nuclei-per-bag", type=_positive, default=16)
    fit.add_argument("--eval-chunk-size", type=_positive, default=16)
    fit.add_argument("--calibration-nuclei-per-bag", type=_positive, default=4)
    fit.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    convert = commands.add_parser("export", help="Export one fold to fixed batch-one ONNX and training-only PTQ arrays")
    convert.add_argument("--run-dir", type=Path, required=True)
    convert.add_argument("--out", type=Path, required=True)
    convert.add_argument("--image-root", type=Path, help="Optional relocated copy; every calibration TIFF SHA is checked")
    args = parser.parse_args(argv)
    if args.command == "train":
        if len(set(args.models)) != len(args.models) or len(set(args.folds)) != len(args.folds):
            parser.error("Models and folds must be unique")
        if args.train_nuclei_per_bag > 100 or args.calibration_nuclei_per_bag > 100:
            parser.error("The frozen development bags contain 100 nuclei each")
        if args.seed < 0 or args.seed > 2**32 - 2001:
            parser.error("Seed must permit all three fold seeds in NumPy's uint32 range")
    try:
        if args.command == "list":
            result = {"models": [model_contract(name) for name in MODEL_NAMES], "recipe_id": RECIPE_ID}
        elif args.command == "preflight":
            result, _, _ = preflight(args)
        elif args.command == "train":
            result = train(args)
        else:
            result = export(args)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(f"NASA multimodel {args.command} failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_json_safe(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
