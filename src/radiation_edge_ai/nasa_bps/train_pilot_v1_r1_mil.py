"""Train the frozen NASA BPS pilot-v1 R1 aggregate-label baseline.

R1 v1 is a deliberately conservative multiple-instance / aggregate-label
regression baseline. NASA ``avg_nfoci`` is a sample/plate-well mean computed
from hundreds of nuclei, whereas the frozen pilot contains 30 nucleus crops per
sample. Therefore supervision is applied ONLY after averaging the 30 model
predictions within a sample bag. The NASA sample mean is never copied onto an
individual nucleus as if it were a per-nucleus ground-truth count.

Input policy (frozen in docs/NASA_BPS_R1_WEAK_SUPERVISION_POLICY.md):
- FITC/53BP1 + DAPI only; MASK is not a model input;
- preserve native pixel scale; no resize;
- center-pad native crops to 256x256;
- per-image robust p1/p99.5 normalization on the native unpadded crop;
- 3-channel packing = FITC, DAPI, zeros;
- only 90-degree rotations/flips during training.

Model policy:
- compact KL720-friendly Conv-BN-ReLU CNN;
- non-negative continuous per-nucleus burden output;
- bag prediction = arithmetic mean of 30 nucleus outputs;
- SmoothL1 loss against NASA sample-level ``avg_nfoci``;
- three already-frozen Source-Name-held-out outer folds;
- fixed schedule; no outer-test early stopping or test-driven tuning.

Run without ``--train`` first. The default dry run validates the entire pilot,
checks every FITC/DAPI geometry, verifies the reference freeze, and performs one
forward pass. Then rerun with ``--train`` to fit all three outer folds and write
out-of-fold bag/nucleus predictions and radiation-effect deltas.

Use the existing DNAi Python environment (torch + OpenCV are already present).
No network access is used and no raw data are committed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FROZEN_MANIFEST_SHA256 = "41ff2e4d64577f17c976dea5bbc17221fe63be9e2e7e2291309b520027fbe725"
EXPECTED_NUCLEI = 2160
EXPECTED_SAMPLES = 72
EXPECTED_NUCLEI_PER_SAMPLE = 30
CANVAS = 256
P_LOW = 1.0
P_HIGH = 99.5
SEED = 720

FOLD_FIELDS = (
    "fold_female_role",
    "fold_male_1_role",
    "fold_male_2_role",
)

EXPOSED_DOSE = {
    "Fe": "0.82",
    "X-ray": "1.0",
}


@dataclass
class Bag:
    sample_name: str
    source_name: str
    strain: str
    sex: str
    particle_type: str
    dose_Gy: str
    hr_post_exposure: str
    target_avg_nfoci: float
    target_avg_foci_no_outl: float
    rows: list[dict[str, str]]
    fold_roles: dict[str, str]


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
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
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


def load_reference_freeze(path: Path, manifest_sha: str) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not obj.get("reference_level_frozen", False):
        raise RuntimeError(f"Reference freeze is not marked frozen: {path}")
    if obj.get("reference_level") != "sample_plate_well_aggregate":
        raise RuntimeError(
            f"Unexpected reference level {obj.get('reference_level')!r}; expected sample_plate_well_aggregate"
        )
    if obj.get("primary_reference_endpoint") != "avg_nfoci":
        raise RuntimeError(
            f"Unexpected primary endpoint {obj.get('primary_reference_endpoint')!r}; expected avg_nfoci"
        )
    if obj.get("manifest_sha256") != manifest_sha:
        raise RuntimeError(
            "Reference freeze manifest hash does not match frozen pilot manifest: "
            f"freeze={obj.get('manifest_sha256')} pilot={manifest_sha}"
        )
    return obj


def build_bags(
    manifest: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
) -> list[Bag]:
    if len(manifest) != EXPECTED_NUCLEI:
        raise RuntimeError(f"Manifest nuclei={len(manifest)}; expected {EXPECTED_NUCLEI}")

    ref_by_sample = {row["sample_name"]: row for row in reference_rows}
    if len(ref_by_sample) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Reference unique Sample Names={len(ref_by_sample)}; expected {EXPECTED_SAMPLES}"
        )

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        grouped[row["sample_name"]].append(row)

    if len(grouped) != EXPECTED_SAMPLES:
        raise RuntimeError(f"Manifest unique Sample Names={len(grouped)}; expected {EXPECTED_SAMPLES}")

    bags: list[Bag] = []
    invariants = (
        "source_name",
        "strain",
        "sex",
        "particle_type",
        "dose_Gy",
        "hr_post_exposure",
    )
    for sample_name, rows in sorted(grouped.items()):
        if len(rows) != EXPECTED_NUCLEI_PER_SAMPLE:
            raise RuntimeError(
                f"Sample {sample_name!r} has {len(rows)} pilot nuclei; expected {EXPECTED_NUCLEI_PER_SAMPLE}"
            )
        first = rows[0]
        for field in invariants + FOLD_FIELDS:
            values = {row.get(field, "") for row in rows}
            if len(values) != 1:
                raise RuntimeError(
                    f"Sample {sample_name!r} not invariant for {field}: {sorted(values)}"
                )
        if sample_name not in ref_by_sample:
            raise RuntimeError(f"Sample missing from frozen reference: {sample_name}")
        ref = ref_by_sample[sample_name]
        if ref.get("reference_level") != "sample_plate_well_aggregate":
            raise RuntimeError(
                f"Sample {sample_name!r} has unexpected reference_level={ref.get('reference_level')!r}"
            )
        bags.append(
            Bag(
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


def robust_normalize(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected 2D microscopy crop, got shape={image.shape}")
    x = image.astype(np.float32, copy=False)
    lo = float(np.percentile(x, P_LOW))
    hi = float(np.percentile(x, P_HIGH))
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError("Non-finite percentile in microscopy image")
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    np.clip(x, 0.0, 1.0, out=x)
    return x


class NativeImageCache:
    """Small LRU cache of robust-normalized native FITC+DAPI crops."""

    def __init__(self, image_root: Path, max_items: int = 1024):
        self.image_root = image_root
        self.max_items = max(0, int(max_items))
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def _load(self, row: dict[str, str]) -> np.ndarray:
        fitc_path = self.image_root / "fitc" / row["fitc_filename"]
        dapi_path = self.image_root / "dapi" / row["dapi_filename"]
        for path in (fitc_path, dapi_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        fitc = cv2.imread(str(fitc_path), cv2.IMREAD_UNCHANGED)
        dapi = cv2.imread(str(dapi_path), cv2.IMREAD_UNCHANGED)
        if fitc is None or dapi is None:
            raise RuntimeError(f"Could not read FITC/DAPI for {row['sample_id']}")
        if fitc.ndim != 2 or dapi.ndim != 2:
            raise RuntimeError(
                f"Expected 2D TIFFs for {row['sample_id']}: FITC={fitc.shape}, DAPI={dapi.shape}"
            )
        if fitc.shape != dapi.shape:
            raise RuntimeError(
                f"FITC/DAPI shape mismatch for {row['sample_id']}: {fitc.shape} vs {dapi.shape}"
            )
        h, w = fitc.shape
        if h > CANVAS or w > CANVAS:
            raise RuntimeError(
                f"Native crop exceeds frozen {CANVAS}x{CANVAS} canvas for {row['sample_id']}: {h}x{w}; resize forbidden"
            )
        return np.stack((robust_normalize(fitc), robust_normalize(dapi)), axis=0)

    def get(self, row: dict[str, str]) -> np.ndarray:
        key = row["sample_id"]
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        value = self._load(row)
        if self.max_items > 0:
            self.cache[key] = value
            while len(self.cache) > self.max_items:
                self.cache.popitem(last=False)
        return value


def pad_and_pack(native: np.ndarray) -> np.ndarray:
    if native.shape[0] != 2:
        raise ValueError(f"Expected 2 native channels, got {native.shape}")
    _, h, w = native.shape
    out = np.zeros((3, CANVAS, CANVAS), dtype=np.float32)
    y0 = (CANVAS - h) // 2
    x0 = (CANVAS - w) // 2
    out[0:2, y0 : y0 + h, x0 : x0 + w] = native
    return out


def augment_geometric(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.integers(0, 4))
    if k:
        x = np.rot90(x, k=k, axes=(1, 2))
    if bool(rng.integers(0, 2)):
        x = x[:, :, ::-1]
    if bool(rng.integers(0, 2)):
        x = x[:, ::-1, :]
    return np.ascontiguousarray(x)


def load_bag_tensor(
    bag: Bag,
    cache: NativeImageCache,
    *,
    augment: bool,
    rng: np.random.Generator,
) -> torch.Tensor:
    images: list[np.ndarray] = []
    for row in bag.rows:
        x = pad_and_pack(cache.get(row))
        if augment:
            x = augment_geometric(x, rng)
        images.append(x)
    return torch.from_numpy(np.stack(images, axis=0))


class ConvStage(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class R1CountNetV1(nn.Module):
    """Small deployment-oriented continuous 53BP1 burden regressor."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvStage(3, 24),
            ConvStage(24, 32),
            ConvStage(32, 48),
            ConvStage(48, 64),
            ConvStage(64, 96),
            ConvStage(96, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(128, 1)
        self.nonnegative = nn.ReLU()
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.head.bias, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        x = self.nonnegative(x)
        return x[:, 0]


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def regression_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    y = np.array([float(row["nasa_avg_nfoci"]) for row in rows], dtype=np.float64)
    p = np.array([float(row["predicted_bag_mean"]) for row in rows], dtype=np.float64)
    err = p - y
    return {
        "n_bags": float(len(rows)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "pearson": correlation(p, y),
        "spearman": correlation(average_ranks(p), average_ranks(y)),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate_bags(
    model: nn.Module,
    bags: list[Bag],
    cache: NativeImageCache,
    device: torch.device,
    fold_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    bag_rows: list[dict[str, Any]] = []
    nucleus_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    with torch.no_grad():
        for bag in bags:
            x = load_bag_tensor(bag, cache, augment=False, rng=rng).to(device)
            pred = model(x).detach().cpu().numpy().astype(np.float64)
            bag_mean = float(np.mean(pred))
            bag_rows.append(
                {
                    "fold": fold_name,
                    "sample_name": bag.sample_name,
                    "source_name": bag.source_name,
                    "strain": bag.strain,
                    "sex": bag.sex,
                    "particle_type": bag.particle_type,
                    "dose_Gy": bag.dose_Gy,
                    "hr_post_exposure": bag.hr_post_exposure,
                    "n_pilot_nuclei": len(bag.rows),
                    "nasa_avg_nfoci": bag.target_avg_nfoci,
                    "nasa_avg_foci_no_outl": bag.target_avg_foci_no_outl,
                    "predicted_bag_mean": bag_mean,
                    "residual": bag_mean - bag.target_avg_nfoci,
                }
            )
            for row, value in zip(bag.rows, pred, strict=True):
                nucleus_rows.append(
                    {
                        "fold": fold_name,
                        "sample_id": row["sample_id"],
                        "nucleus_key": row["nucleus_key"],
                        "sample_name": bag.sample_name,
                        "source_name": bag.source_name,
                        "strain": bag.strain,
                        "sex": bag.sex,
                        "particle_type": bag.particle_type,
                        "dose_Gy": bag.dose_Gy,
                        "hr_post_exposure": bag.hr_post_exposure,
                        "predicted_continuous_burden": float(value),
                        "label_semantics": "unlabeled_individual_prediction_calibrated_only_by_sample_mean",
                    }
                )
    return bag_rows, nucleus_rows


def biological_deltas(oof_bags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in oof_bags:
        lookup[(
            str(row["source_name"]),
            str(row["particle_type"]),
            norm_number(str(row["dose_Gy"])),
            norm_hour(str(row["hr_post_exposure"])),
        )] = row

    rows: list[dict[str, Any]] = []
    sources = sorted({str(row["source_name"]) for row in oof_bags})
    for source in sources:
        for branch, exposed_dose in EXPOSED_DOSE.items():
            for hour in ("4", "24", "48"):
                sham = lookup[(source, branch, "0.0", hour)]
                exp = lookup[(source, branch, norm_number(exposed_dose), hour)]
                nasa_delta = float(exp["nasa_avg_nfoci"]) - float(sham["nasa_avg_nfoci"])
                pred_delta = float(exp["predicted_bag_mean"]) - float(sham["predicted_bag_mean"])
                rows.append(
                    {
                        "source_name": source,
                        "strain": exp["strain"],
                        "sex": exp["sex"],
                        "branch": branch,
                        "timepoint_hr": hour,
                        "exposed_dose_Gy": norm_number(exposed_dose),
                        "nasa_sham": float(sham["nasa_avg_nfoci"]),
                        "nasa_exposed": float(exp["nasa_avg_nfoci"]),
                        "nasa_delta": nasa_delta,
                        "pred_sham": float(sham["predicted_bag_mean"]),
                        "pred_exposed": float(exp["predicted_bag_mean"]),
                        "pred_delta": pred_delta,
                        "direction_agreement": int(np.sign(nasa_delta) == np.sign(pred_delta)),
                    }
                )
    return rows


def delta_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    nasa = np.array([float(row["nasa_delta"]) for row in rows], dtype=np.float64)
    pred = np.array([float(row["pred_delta"]) for row in rows], dtype=np.float64)
    return {
        "n_matched_contrasts": float(len(rows)),
        "direction_agreement_fraction": float(
            np.mean([int(row["direction_agreement"]) for row in rows])
        ),
        "delta_mae": float(np.mean(np.abs(pred - nasa))),
        "delta_pearson": correlation(pred, nasa),
        "delta_spearman": correlation(average_ranks(pred), average_ranks(nasa)),
    }


def validate_all_geometry(bags: list[Bag], cache: NativeImageCache) -> dict[str, Any]:
    heights: list[int] = []
    widths: list[int] = []
    dtypes: defaultdict[str, int] = defaultdict(int)
    seen = 0
    for bag in bags:
        for row in bag.rows:
            fitc_path = cache.image_root / "fitc" / row["fitc_filename"]
            dapi_path = cache.image_root / "dapi" / row["dapi_filename"]
            fitc = cv2.imread(str(fitc_path), cv2.IMREAD_UNCHANGED)
            dapi = cv2.imread(str(dapi_path), cv2.IMREAD_UNCHANGED)
            if fitc is None or dapi is None:
                raise RuntimeError(f"Unreadable FITC/DAPI for {row['sample_id']}")
            if fitc.shape != dapi.shape or fitc.ndim != 2:
                raise RuntimeError(
                    f"Geometry mismatch {row['sample_id']}: FITC={fitc.shape}, DAPI={dapi.shape}"
                )
            h, w = fitc.shape
            if h > CANVAS or w > CANVAS:
                raise RuntimeError(
                    f"Crop exceeds frozen canvas {row['sample_id']}: {h}x{w}; resize forbidden"
                )
            heights.append(h)
            widths.append(w)
            dtypes[f"FITC:{fitc.dtype}"] += 1
            dtypes[f"DAPI:{dapi.dtype}"] += 1
            seen += 1
    return {
        "n_nuclei": seen,
        "height_min": min(heights),
        "height_median": float(median(heights)),
        "height_max": max(heights),
        "width_min": min(widths),
        "width_median": float(median(widths)),
        "width_max": max(widths),
        "dtype_counts": dict(dtypes),
        "all_fit_256_native_scale": True,
    }


def train_fold(
    fold_field: str,
    bags: list[Bag],
    cache: NativeImageCache,
    device: torch.device,
    output_dir: Path,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fold_name = fold_field.removesuffix("_role")
    train_bags = [bag for bag in bags if bag.fold_roles[fold_field] == "train"]
    test_bags = [bag for bag in bags if bag.fold_roles[fold_field] == "test"]
    if len(train_bags) != 48 or len(test_bags) != 24:
        raise RuntimeError(
            f"Unexpected split for {fold_name}: train={len(train_bags)} test={len(test_bags)}"
        )

    fold_seed = SEED + FOLD_FIELDS.index(fold_field) * 1000
    seed_everything(fold_seed)
    model = R1CountNetV1().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    rng = np.random.default_rng(fold_seed)

    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_bags))
        losses: list[float] = []
        for index in order:
            bag = train_bags[int(index)]
            x = load_bag_tensor(bag, cache, augment=True, rng=rng).to(device)
            target = torch.tensor(bag.target_avg_nfoci, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            per_nucleus = model(x)
            bag_mean = per_nucleus.mean()
            loss = F.smooth_l1_loss(bag_mean, target, beta=0.5)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        record = {
            "fold": fold_name,
            "epoch": epoch,
            "train_bag_loss_mean": float(np.mean(losses)),
            "train_bag_loss_median": float(np.median(losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"  {fold_name} epoch {epoch:3d}/{epochs}: "
                f"mean_loss={record['train_bag_loss_mean']:.5f} "
                f"lr={record['learning_rate']:.3g}"
            )

    fold_dir = output_dir / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": "R1CountNetV1",
        "fold": fold_name,
        "model_state_dict": model.state_dict(),
        "input_policy": {
            "canvas": CANVAS,
            "channels": ["FITC_p1_p99.5", "DAPI_p1_p99.5", "ZERO"],
            "resize": False,
            "padding": "center_zero",
        },
        "target": "NASA sample-level avg_nfoci",
        "loss": "SmoothL1(mean(per_nucleus_prediction), avg_nfoci), beta=0.5",
        "seed": fold_seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "parameter_count": parameter_count(model),
        "train_sources": sorted({bag.source_name for bag in train_bags}),
        "test_sources": sorted({bag.source_name for bag in test_bags}),
    }
    torch.save(checkpoint, fold_dir / "r1_countnet_v1.pt")
    write_csv(fold_dir / "training_history.csv", history)

    test_bag_rows, test_nucleus_rows = evaluate_bags(
        model, test_bags, cache, device, fold_name
    )
    metrics = regression_metrics(test_bag_rows)
    write_csv(fold_dir / "test_bag_predictions.csv", test_bag_rows)
    write_csv(fold_dir / "test_nucleus_predictions.csv", test_nucleus_rows)
    (fold_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(
        f"  {fold_name} TEST: MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} "
        f"Pearson={metrics['pearson']:.4f} Spearman={metrics['spearman']:.4f}"
    )
    return test_bag_rows, test_nucleus_rows, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata" / "pilot_v1"
    parser.add_argument(
        "--manifest",
        default=str(metadata_root / "pilot_v1_manifest.csv"),
    )
    parser.add_argument(
        "--reference",
        default=str(
            metadata_root
            / "reference_phenotypes_v2"
            / "pilot_v1_sample_reference_frozen.csv"
        ),
    )
    parser.add_argument(
        "--reference-freeze",
        default=str(metadata_root / "reference_phenotypes_v2" / "reference_freeze_v2.json"),
    )
    parser.add_argument(
        "--image-root",
        default=str(data_root / "nasa_bps_microscopy" / "pilot_v1" / "images"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(model_root / "nasa_bps_53bp1" / "r1_v1_mil"),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cache-items", type=int, default=1024)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Actually train all three outer folds. Without this flag only validation/dry-run is performed.",
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
    if manifest_sha != FROZEN_MANIFEST_SHA256:
        raise RuntimeError(
            f"Frozen pilot manifest SHA256 mismatch: expected {FROZEN_MANIFEST_SHA256}, got {manifest_sha}"
        )
    freeze = load_reference_freeze(freeze_path, manifest_sha)
    manifest = read_csv(manifest_path)
    reference = read_csv(reference_path)
    bags = build_bags(manifest, reference)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = NativeImageCache(image_root=image_root, max_items=args.cache_items)

    print("Radiation Edge AI - NASA BPS R1 v1 aggregate-label baseline")
    print(f"mode: {'TRAIN' if args.train else 'DRY RUN'}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")
    print(f"manifest SHA256: {manifest_sha}")
    print(f"reference freeze: {freeze.get('reference_freeze_version')}")
    print(f"bags: {len(bags)}")
    print(f"nuclei: {sum(len(bag.rows) for bag in bags)}")
    print(f"nuclei/bag: {EXPECTED_NUCLEI_PER_SAMPLE}")
    print(
        "NASA full-sample nuclei min/median/max: "
        f"{freeze.get('nasa_num_nuc_min')}/{freeze.get('nasa_num_nuc_median')}/{freeze.get('nasa_num_nuc_max')}"
    )
    targets = [bag.target_avg_nfoci for bag in bags]
    print(
        f"avg_nfoci target min/median/max: {min(targets):.4f}/{median(targets):.4f}/{max(targets):.4f}"
    )
    print("")

    print("[Frozen outer folds]")
    for field in FOLD_FIELDS:
        train_sources = sorted({bag.source_name for bag in bags if bag.fold_roles[field] == "train"})
        test_sources = sorted({bag.source_name for bag in bags if bag.fold_roles[field] == "test"})
        train_bags = sum(bag.fold_roles[field] == "train" for bag in bags)
        test_bags = sum(bag.fold_roles[field] == "test" for bag in bags)
        print(
            f"{field.removesuffix('_role')}: train={train_bags} bags {train_sources}; "
            f"test={test_bags} bags {test_sources}"
        )
    print("")

    print("[Full geometry validation]")
    geometry = validate_all_geometry(bags, cache)
    print(
        f"nuclei={geometry['n_nuclei']} "
        f"height={geometry['height_min']}/{geometry['height_median']}/{geometry['height_max']} "
        f"width={geometry['width_min']}/{geometry['width_median']}/{geometry['width_max']}"
    )
    print(f"all fit 256x256 at native scale: {'YES' if geometry['all_fit_256_native_scale'] else 'NO'}")
    print(f"dtype counts: {geometry['dtype_counts']}")
    print("")

    seed_everything(SEED)
    smoke_model = R1CountNetV1().to(device)
    smoke_rng = np.random.default_rng(SEED)
    smoke_x = load_bag_tensor(bags[0], cache, augment=False, rng=smoke_rng).to(device)
    with torch.no_grad():
        smoke_pred = smoke_model(smoke_x)
    print("[Model smoke]")
    print(f"model: R1CountNetV1")
    print(f"parameters: {parameter_count(smoke_model):,}")
    print(f"input: {tuple(smoke_x.shape)}")
    print(f"output: {tuple(smoke_pred.shape)}")
    print(f"initial bag mean: {float(smoke_pred.mean().cpu()):.4f}")
    print("")

    dry_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "train" if args.train else "dry_run",
        "manifest_sha256": manifest_sha,
        "reference_freeze_version": freeze.get("reference_freeze_version"),
        "reference_level": freeze.get("reference_level"),
        "primary_reference_endpoint": freeze.get("primary_reference_endpoint"),
        "bags": len(bags),
        "nuclei": sum(len(bag.rows) for bag in bags),
        "geometry": geometry,
        "model": "R1CountNetV1",
        "parameter_count": parameter_count(smoke_model),
        "input_policy": {
            "canvas": CANVAS,
            "resize": False,
            "p_low": P_LOW,
            "p_high": P_HIGH,
            "channels": ["FITC", "DAPI", "ZERO"],
        },
        "supervision": "bag_mean_only",
        "target": "NASA avg_nfoci",
    }

    if not args.train:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "dry_run_summary.json").write_text(
            json.dumps(dry_summary, indent=2), encoding="utf-8"
        )
        print("NASA BPS R1 V1 DRY RUN COMPLETE: YES")
        print("No model was trained.")
        print(
            "NEXT GATE: rerun the same command with --train to fit the three frozen source-held-out folds."
        )
        return 0

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_config.json").write_text(
        json.dumps(
            {
                **dry_summary,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "seed": SEED,
                "test_set_used_for_early_stopping": False,
                "outer_test_tuning_allowed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[Training three frozen outer folds]")
    all_oof_bags: list[dict[str, Any]] = []
    all_oof_nuclei: list[dict[str, Any]] = []
    fold_metrics: dict[str, Any] = {}
    for fold_field in FOLD_FIELDS:
        bag_rows, nucleus_rows, metrics = train_fold(
            fold_field,
            bags,
            cache,
            device,
            output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        all_oof_bags.extend(bag_rows)
        all_oof_nuclei.extend(nucleus_rows)
        fold_metrics[fold_field.removesuffix("_role")] = metrics

    if len(all_oof_bags) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Out-of-fold bag rows={len(all_oof_bags)}; expected {EXPECTED_SAMPLES}"
        )
    if len(all_oof_nuclei) != EXPECTED_NUCLEI:
        raise RuntimeError(
            f"Out-of-fold nucleus rows={len(all_oof_nuclei)}; expected {EXPECTED_NUCLEI}"
        )

    all_oof_bags.sort(key=lambda row: (str(row["source_name"]), str(row["sample_name"])))
    all_oof_nuclei.sort(key=lambda row: str(row["sample_id"]))
    write_csv(output_dir / "oof_bag_predictions.csv", all_oof_bags)
    write_csv(output_dir / "oof_nucleus_predictions.csv", all_oof_nuclei)

    overall = regression_metrics(all_oof_bags)
    delta_rows = biological_deltas(all_oof_bags)
    write_csv(output_dir / "oof_matched_radiation_deltas.csv", delta_rows)
    delta_summary = delta_metrics(delta_rows)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "R1CountNetV1",
        "fold_metrics": fold_metrics,
        "oof_bag_metrics": overall,
        "oof_biological_delta_metrics": delta_summary,
        "interpretation_guardrails": [
            "NASA avg_nfoci is sample-level aggregate supervision, not a per-nucleus label.",
            "Per-nucleus outputs are continuous expected burden scores calibrated only through bag means.",
            "Biological conclusions are evaluated on held-out Source Names and matched branch-specific sham/exposed contrasts.",
            "No KL720 or quantization equivalence is implied by this FP32 training result.",
        ],
    }
    (output_dir / "r1_v1_oof_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("")
    print("[Out-of-fold R1 v1 summary]")
    print(
        f"bags={int(overall['n_bags'])} MAE={overall['mae']:.4f} RMSE={overall['rmse']:.4f} "
        f"Pearson={overall['pearson']:.4f} Spearman={overall['spearman']:.4f}"
    )
    print(
        f"matched radiation contrasts={int(delta_summary['n_matched_contrasts'])} "
        f"direction agreement={delta_summary['direction_agreement_fraction']:.3f} "
        f"delta MAE={delta_summary['delta_mae']:.4f} "
        f"delta Spearman={delta_summary['delta_spearman']:.4f}"
    )
    print("")
    print("NASA BPS R1 V1 TRAINING COMPLETE: YES")
    print(f"Output: {output_dir}")
    print(
        "NEXT GATE: inspect held-out bag/delta fidelity before deciding whether R1 v1 is sufficient or a spatial weakly supervised R1 v2 is warranted."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
