"""One-time confirmatory evaluator for the frozen NASA BPS R1 v2 final holdout.

The evaluator is frozen only after phenotype-blind hard QC and the permitted
geometry-only repair have completed.  It pins the exact repaired FP32 model,
QC1 holdout manifest, and NASA raw phenotype table identities.

Default mode is PRE-FLIGHT and does not read the phenotype table.  It verifies
model/manifest identities, reruns the identical hard image QC, loads the frozen
model, and performs phenotype-blind inference without printing or writing the
prediction values.  Only an explicit --unblind run may hash/read the raw NASA
phenotype table, join avg_nfoci by exact Sample Name, and write the predeclared
confirmatory outputs.

No model training, BatchNorm recalibration, threshold tuning, bag resampling, or
phenotype-dependent repair is implemented here.  A completed confirmatory run
cannot be overwritten by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
import torch

FROZEN_REPAIRED_CHECKPOINT_SHA256 = (
    "2b07c67d6e50e7a36e26e0052feebc826b518e6efae863ba312ba9761a4eb3d5"
)
FROZEN_QC1_HOLDOUT_MANIFEST_SHA256 = (
    "f89962f96bec05930c07149717a90bd682016a2778bbfae6b827425159aa0643"
)
FROZEN_RAW_PHENOTYPE_SHA256 = (
    "d28dc5d9c53221c66a90074a92783fd4708ea83db0fc1a9c767e47500e643d6a"
)
EXPECTED_NUCLEI = 2058
EXPECTED_BAGS = 22
EXPECTED_CONTRASTS = 11
EXPECTED_SOURCE_NUCLEI = {"BALBCF2": 858, "C57BLF2": 600, "C57BLF3": 600}
EXPECTED_SOURCE_CONTRASTS = {"BALBCF2": 5, "C57BLF2": 3, "C57BLF3": 3}
EXPECTED_SOURCE_DIRECTION_MIN = {"BALBCF2": 4, "C57BLF2": 2, "C57BLF3": 2}
EXPECTED_HOLDOUT_STATUS = "FINAL_BLINDED_DO_NOT_JOIN_PHENOTYPES"
EXPOSED_DOSE = {"Fe": "0.82", "X-ray": "1.0"}
SEED = 720

GATE = {
    "bag_spearman_min": 0.50,
    "delta_spearman_min": 0.60,
    "direction_count_min": 9,
    "direction_count_total": 11,
    "four_hour_direction_min": 4,
    "four_hour_direction_total": 4,
    "late_direction_min": 5,
    "late_direction_total": 7,
    "peak_recovery_min": 3,
    "peak_recovery_total": 3,
    "source_direction_min_counts": EXPECTED_SOURCE_DIRECTION_MIN,
}

FORBIDDEN_MANIFEST_TOKENS = (
    "nfoci",
    "phenotype",
    "reference",
    "ground_truth",
    "prediction",
    "predicted",
)
RAW_REQUIRED = (
    "Sample Name",
    "UserID",
    "plate",
    "plate_well",
    "Strain",
    "Gender",
    "radiation",
    "dose..Gy.",
    "timepoint..hr.",
    "num_nuc",
    "avg_nfoci",
)


@dataclass
class HoldoutBag:
    sample_name: str
    source_name: str
    strain: str
    sex: str
    particle_type: str
    dose_Gy: str
    hr_post_exposure: str
    plate: str
    well: str
    rows: list[dict[str, str]]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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


def norm_text(value: str) -> str:
    return "".join(str(value).strip().lower().split())


def norm_number(value: str) -> str:
    x = float(str(value).strip())
    if x.is_integer():
        return f"{x:.1f}"
    return format(x, "g")


def norm_hour(value: str) -> str:
    x = float(str(value).strip())
    return str(int(x)) if x.is_integer() else format(x, "g")


def norm_radiation(value: str) -> str:
    text = norm_text(value).replace("_", "-")
    aliases = {
        "xray": "x-ray",
        "x-ray": "x-ray",
        "x-rays": "x-ray",
        "fe": "fe",
        "iron": "fe",
        "56fe": "fe",
    }
    return aliases.get(text, text)


def norm_sex(value: str) -> str:
    text = norm_text(value)
    return {"f": "female", "female": "female", "m": "male", "male": "male"}.get(
        text, text
    )


def validate_manifest(rows: list[dict[str, str]]) -> list[HoldoutBag]:
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
        "holdout_status",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise RuntimeError(f"QC1 holdout manifest missing columns: {missing}")
    forbidden = sorted(
        column
        for column in rows[0]
        if any(token in column.lower() for token in FORBIDDEN_MANIFEST_TOKENS)
    )
    if forbidden:
        raise RuntimeError(f"QC1 holdout manifest contains outcome columns: {forbidden}")
    if len(rows) != EXPECTED_NUCLEI:
        raise RuntimeError(f"Holdout nuclei={len(rows)}; expected {EXPECTED_NUCLEI}")
    if len({row["sample_id"] for row in rows}) != EXPECTED_NUCLEI:
        raise RuntimeError("Duplicate sample_id in QC1 holdout manifest")
    if len({row["nucleus_key"] for row in rows}) != EXPECTED_NUCLEI:
        raise RuntimeError("Duplicate nucleus_key in QC1 holdout manifest")
    if {row["holdout_status"] for row in rows} != {EXPECTED_HOLDOUT_STATUS}:
        raise RuntimeError("Unexpected holdout_status in QC1 manifest")

    source_counts = Counter(row["source_name"] for row in rows)
    if dict(source_counts) != EXPECTED_SOURCE_NUCLEI:
        raise RuntimeError(
            f"Unexpected holdout source counts: {dict(source_counts)}; expected {EXPECTED_SOURCE_NUCLEI}"
        )
    if {row["sex"] for row in rows} != {"Female"}:
        raise RuntimeError("Final holdout is no longer the frozen all-female three-source cohort")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["sample_name"]].append(row)
    if len(grouped) != EXPECTED_BAGS:
        raise RuntimeError(f"Holdout bags={len(grouped)}; expected {EXPECTED_BAGS}")

    bags: list[HoldoutBag] = []
    invariants = ("source_name", "strain", "sex", "particle_type", "dose_Gy", "hr_post_exposure")
    for sample_name, subset in sorted(grouped.items()):
        first = subset[0]
        for field in invariants:
            values = {str(row[field]) for row in subset}
            if len(values) != 1:
                raise RuntimeError(f"Bag {sample_name} is not invariant for {field}: {sorted(values)}")
        plates = {str(row.get("plate", "")) for row in subset}
        wells = {str(row.get("well", "")) for row in subset}
        if len(plates) != 1 or len(wells) != 1:
            raise RuntimeError(f"Bag {sample_name} is not invariant for plate/well")
        bags.append(
            HoldoutBag(
                sample_name=sample_name,
                source_name=first["source_name"],
                strain=first["strain"],
                sex=first["sex"],
                particle_type=first["particle_type"],
                dose_Gy=norm_number(first["dose_Gy"]),
                hr_post_exposure=norm_hour(first["hr_post_exposure"]),
                plate=next(iter(plates)),
                well=next(iter(wells)),
                rows=sorted(subset, key=lambda row: row["sample_id"]),
            )
        )

    contrast_counts: Counter[str] = Counter()
    lookup = {(b.source_name, b.particle_type, b.dose_Gy, b.hr_post_exposure): b for b in bags}
    for source in EXPECTED_SOURCE_NUCLEI:
        for branch, exposed in EXPOSED_DOSE.items():
            for hour in ("4", "24", "48"):
                sham = (source, branch, "0.0", hour)
                exp = (source, branch, norm_number(exposed), hour)
                if sham in lookup and exp in lookup:
                    contrast_counts[source] += 1
    if dict(contrast_counts) != EXPECTED_SOURCE_CONTRASTS:
        raise RuntimeError(
            f"Unexpected matched-contrast structure: {dict(contrast_counts)}; expected {EXPECTED_SOURCE_CONTRASTS}"
        )
    if sum(contrast_counts.values()) != EXPECTED_CONTRASTS:
        raise RuntimeError("Unexpected total matched contrast count")
    return bags


def hard_qc(dev_qc: Any, rows: list[dict[str, str]], image_root: Path) -> dict[str, Any]:
    cv2.setNumThreads(0)
    results = [dev_qc.inspect_one(row, image_root) for row in rows]
    failures = [row for row in results if int(row.get("ok", 0)) != 1]
    if failures:
        first = failures[0]
        raise RuntimeError(
            f"QC1 manifest no longer passes hard image QC: {len(failures)} failure(s); "
            f"first={first.get('sample_id')} {first.get('error')}"
        )
    return {
        "nuclei": len(results),
        "all_pass": True,
        "height_min": min(int(row["height"]) for row in results),
        "height_median": float(median(int(row["height"]) for row in results)),
        "height_max": max(int(row["height"]) for row in results),
        "width_min": min(int(row["width"]) for row in results),
        "width_median": float(median(int(row["width"]) for row in results)),
        "width_max": max(int(row["width"]) for row in results),
        "multilabel_masks": sum(int(row.get("multilabel_mask", 0)) for row in results),
    }


def load_model(v1: Any, checkpoint_path: Path, device: torch.device):
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if obj.get("candidate_name") != "NASA_BPS_R1_V2_FINAL_FP32_BN_RECALIBRATED":
        raise RuntimeError(f"Unexpected final candidate identity: {obj.get('candidate_name')!r}")
    if obj.get("final_holdout_used_for_training_or_selection") is not False:
        raise RuntimeError("Checkpoint no longer records holdout exclusion from training/selection")
    model = v1.R1CountNetV1().to(device)
    model.load_state_dict(obj["model_state_dict"], strict=True)
    model.eval()
    if v1.parameter_count(model) != 215833:
        raise RuntimeError(f"Unexpected model parameter count: {v1.parameter_count(model)}")
    return model


def infer_blinded(v1: Any, model: Any, bags: list[HoldoutBag], image_root: Path, device: torch.device, cache_items: int):
    cache = v1.NativeImageCache(image_root=image_root, max_items=cache_items)
    bag_rows: list[dict[str, Any]] = []
    nucleus_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for bag in bags:
            images = [v1.pad_and_pack(cache.get(row)) for row in bag.rows]
            x = torch.from_numpy(np.stack(images, axis=0)).to(device)
            pred = model(x).detach().cpu().numpy().astype(np.float64)
            if len(pred) != len(bag.rows) or not np.all(np.isfinite(pred)):
                raise RuntimeError(f"Non-finite or incomplete predictions for bag {bag.sample_name}")
            bag_mean = float(np.mean(pred))
            bag_rows.append(
                {
                    "sample_name": bag.sample_name,
                    "source_name": bag.source_name,
                    "strain": bag.strain,
                    "sex": bag.sex,
                    "particle_type": bag.particle_type,
                    "dose_Gy": bag.dose_Gy,
                    "hr_post_exposure": bag.hr_post_exposure,
                    "n_holdout_nuclei": len(bag.rows),
                    "predicted_bag_mean": bag_mean,
                }
            )
            for row, value in zip(bag.rows, pred, strict=True):
                nucleus_rows.append(
                    {
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
                        "label_semantics": "latent_continuous_burden_not_individually_supervised",
                    }
                )
    if len(bag_rows) != EXPECTED_BAGS or len(nucleus_rows) != EXPECTED_NUCLEI:
        raise RuntimeError("Unexpected phenotype-blind inference output size")
    return bag_rows, nucleus_rows


def join_raw_phenotypes(raw_rows: list[dict[str, str]], bags: list[HoldoutBag]) -> dict[str, dict[str, Any]]:
    missing = [column for column in RAW_REQUIRED if column not in raw_rows[0]]
    if missing:
        raise RuntimeError(f"Raw phenotype table missing required columns: {missing}")
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        by_sample[norm_text(row["Sample Name"])].append(row)

    references: dict[str, dict[str, Any]] = {}
    checks = (
        ("source_name", "UserID", norm_text),
        ("strain", "Strain", norm_text),
        ("sex", "Gender", norm_sex),
        ("plate", "plate", norm_text),
        ("well", "plate_well", norm_text),
        ("particle_type", "radiation", norm_radiation),
        ("dose_Gy", "dose..Gy.", norm_number),
        ("hr_post_exposure", "timepoint..hr.", norm_hour),
    )
    for bag in bags:
        matches = by_sample.get(norm_text(bag.sample_name), [])
        if len(matches) != 1:
            raise RuntimeError(
                f"Holdout Sample Name {bag.sample_name!r} matched {len(matches)} raw phenotype rows; expected exactly 1"
            )
        nasa = matches[0]
        for bag_field, raw_field, normalizer in checks:
            left = normalizer(getattr(bag, bag_field))
            right = normalizer(nasa.get(raw_field, ""))
            if left and right and left != right:
                raise RuntimeError(
                    f"Phenotype metadata mismatch for {bag.sample_name}: {bag_field}={left!r} != {raw_field}={right!r}"
                )
        target = float(nasa["avg_nfoci"])
        num_nuc = int(float(nasa["num_nuc"]))
        if not math.isfinite(target) or num_nuc <= 0:
            raise RuntimeError(f"Invalid NASA aggregate reference for {bag.sample_name}")
        references[bag.sample_name] = {
            "nasa_avg_nfoci": target,
            "nasa_num_nuc": num_nuc,
        }
    if len(references) != EXPECTED_BAGS:
        raise RuntimeError("Did not resolve exactly 22 holdout aggregate references")
    return references


def attach_references(bag_rows: list[dict[str, Any]], references: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in bag_rows:
        ref = references[str(row["sample_name"])]
        out = dict(row)
        out.update(ref)
        out["residual"] = float(out["predicted_bag_mean"]) - float(out["nasa_avg_nfoci"])
        rows.append(out)
    return rows


def build_deltas(bag_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (str(row["source_name"]), str(row["particle_type"]), norm_number(row["dose_Gy"]), norm_hour(row["hr_post_exposure"])): row
        for row in bag_rows
    }
    rows: list[dict[str, Any]] = []
    for source in EXPECTED_SOURCE_NUCLEI:
        for branch, exposed in EXPOSED_DOSE.items():
            for hour in ("4", "24", "48"):
                sham = lookup.get((source, branch, "0.0", hour))
                exp = lookup.get((source, branch, norm_number(exposed), hour))
                if sham is None or exp is None:
                    continue
                nasa_delta = float(exp["nasa_avg_nfoci"]) - float(sham["nasa_avg_nfoci"])
                pred_delta = float(exp["predicted_bag_mean"]) - float(sham["predicted_bag_mean"])
                rows.append(
                    {
                        "source_name": source,
                        "strain": exp["strain"],
                        "sex": exp["sex"],
                        "branch": branch,
                        "timepoint_hr": hour,
                        "exposed_dose_Gy": norm_number(exposed),
                        "nasa_sham": float(sham["nasa_avg_nfoci"]),
                        "nasa_exposed": float(exp["nasa_avg_nfoci"]),
                        "nasa_delta": nasa_delta,
                        "pred_sham": float(sham["predicted_bag_mean"]),
                        "pred_exposed": float(exp["predicted_bag_mean"]),
                        "pred_delta": pred_delta,
                        "direction_agreement": int(np.sign(nasa_delta) == np.sign(pred_delta)),
                    }
                )
    counts = Counter(str(row["source_name"]) for row in rows)
    if len(rows) != EXPECTED_CONTRASTS or dict(counts) != EXPECTED_SOURCE_CONTRASTS:
        raise RuntimeError(f"Final matched deltas do not reproduce frozen 11-contrast structure: {dict(counts)}")
    return rows


def direction_rows(delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "all", delta_rows)]
    for hour in ("4", "24", "48"):
        groups.append(("timepoint_hr", hour, [r for r in delta_rows if str(r["timepoint_hr"]) == hour]))
    groups.append(("timepoint_hr", "24+48", [r for r in delta_rows if str(r["timepoint_hr"]) in {"24", "48"}]))
    for branch in ("Fe", "X-ray"):
        groups.append(("branch", branch, [r for r in delta_rows if str(r["branch"]) == branch]))
    for source in EXPECTED_SOURCE_NUCLEI:
        groups.append(("source_name", source, [r for r in delta_rows if str(r["source_name"]) == source]))
    for level, value, subset in groups:
        passed = sum(int(r["direction_agreement"]) for r in subset)
        summaries.append(
            {
                "group_type": level,
                "group": value,
                "correct_directions": passed,
                "total_contrasts": len(subset),
                "fraction": float(passed / len(subset)) if subset else float("nan"),
            }
        )
    return summaries


def peak_rows(delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        grouped[(str(row["source_name"]), str(row["branch"]))].append(row)
    rows: list[dict[str, Any]] = []
    for (source, branch), subset in sorted(grouped.items()):
        if len(subset) != 3:
            continue
        subset = sorted(subset, key=lambda row: (4, 24, 48).index(int(row["timepoint_hr"])))
        nasa_peak = max(subset, key=lambda row: float(row["nasa_delta"]))
        pred_peak = max(subset, key=lambda row: float(row["pred_delta"]))
        rows.append(
            {
                "source_name": source,
                "branch": branch,
                "nasa_peak_time_hr": str(nasa_peak["timepoint_hr"]),
                "pred_peak_time_hr": str(pred_peak["timepoint_hr"]),
                "peak_time_recovered": int(str(nasa_peak["timepoint_hr"]) == str(pred_peak["timepoint_hr"])),
                "nasa_peak_delta": float(nasa_peak["nasa_delta"]),
                "pred_peak_delta": float(pred_peak["pred_delta"]),
            }
        )
    if len(rows) != 3:
        raise RuntimeError(f"Complete 4/24/48 source x branch series={len(rows)}; expected 3")
    return rows


def source_residual_rows(v1: Any, bag_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in EXPECTED_SOURCE_NUCLEI:
        subset = [row for row in bag_rows if str(row["source_name"]) == source]
        residuals = np.array([float(row["residual"]) for row in subset], dtype=np.float64)
        rows.append(
            {
                "source_name": source,
                "n_bags": len(subset),
                "mae": float(np.mean(np.abs(residuals))),
                "rmse": float(np.sqrt(np.mean(residuals * residuals))),
                "mean_residual": float(np.mean(residuals)),
                "median_residual": float(np.median(residuals)),
                "bag_spearman": float(v1.regression_metrics(subset)["spearman"]),
            }
        )
    return rows


def evaluate_gate(v1: Any, bag_rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]], peaks: list[dict[str, Any]]):
    bag_metrics = v1.regression_metrics(bag_rows)
    delta_metrics = v1.delta_metrics(delta_rows)
    four = [r for r in delta_rows if str(r["timepoint_hr"]) == "4"]
    late = [r for r in delta_rows if str(r["timepoint_hr"]) in {"24", "48"}]
    four_ok = sum(int(r["direction_agreement"]) for r in four)
    late_ok = sum(int(r["direction_agreement"]) for r in late)
    overall_ok = sum(int(r["direction_agreement"]) for r in delta_rows)
    peak_ok = sum(int(r["peak_time_recovered"]) for r in peaks)
    source_counts = {
        source: sum(int(r["direction_agreement"]) for r in delta_rows if str(r["source_name"]) == source)
        for source in EXPECTED_SOURCE_NUCLEI
    }
    checks = {
        "bag_spearman_ge_0_50": float(bag_metrics["spearman"]) >= GATE["bag_spearman_min"],
        "delta_spearman_ge_0_60": float(delta_metrics["delta_spearman"]) >= GATE["delta_spearman_min"],
        "overall_direction_ge_9_of_11": len(delta_rows) == 11 and overall_ok >= 9,
        "four_hour_direction_eq_4_of_4": len(four) == 4 and four_ok == 4,
        "late_24_48_direction_ge_5_of_7": len(late) == 7 and late_ok >= 5,
        "peak_time_recovery_eq_3_of_3": len(peaks) == 3 and peak_ok == 3,
        "every_source_direction_minimum": all(
            source_counts[source] >= EXPECTED_SOURCE_DIRECTION_MIN[source]
            for source in EXPECTED_SOURCE_DIRECTION_MIN
        ),
    }
    detail = {
        "bag_metrics": bag_metrics,
        "delta_metrics": delta_metrics,
        "direction_counts": {
            "overall": f"{overall_ok}/11",
            "4h": f"{four_ok}/4",
            "24+48h": f"{late_ok}/7",
            "by_source": {
                source: f"{source_counts[source]}/{EXPECTED_SOURCE_CONTRASTS[source]}"
                for source in EXPECTED_SOURCE_NUCLEI
            },
        },
        "peak_time_recovery": f"{peak_ok}/3",
        "checks": checks,
        "all_pass": all(checks.values()),
    }
    return detail


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"
    holdout_root = data_root / "nasa_bps_microscopy" / "r1_v2_final_holdout_blinded"
    script_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--manifest",
        default=str(metadata_root / "r1_v2_freeze" / "r1_v2_final_holdout_manifest_blinded_qc1.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(model_root / "nasa_bps_53bp1" / "r1_v2_final_candidate" / "r1_countnet_v2_final_bn_recalibrated.pt"),
    )
    parser.add_argument("--image-root", default=str(holdout_root / "images"))
    parser.add_argument(
        "--raw-phenotypes",
        default=str(metadata_root / "current_phenotypes" / "files" / "LSDS-111_immunostaining_Raw_pheno_V3.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(model_root / "nasa_bps_53bp1" / "r1_v2_final_holdout_confirmation"),
    )
    parser.add_argument("--cache-items", type=int, default=512)
    parser.add_argument(
        "--unblind",
        action="store_true",
        help="Explicitly open the frozen raw phenotype table and execute the one-time confirmatory evaluation. Default is phenotype-blind preflight.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    image_root = Path(args.image_root).resolve()
    raw_path = Path(args.raw_phenotypes).resolve()
    output_dir = Path(args.output_dir).resolve()
    evaluator_source_sha = sha256_file(Path(__file__).resolve())

    for path in (manifest_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    checkpoint_sha = sha256_file(checkpoint_path)
    manifest_sha = sha256_file(manifest_path)
    if checkpoint_sha != FROZEN_REPAIRED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Final repaired FP32 SHA256 mismatch: {checkpoint_sha}")
    if manifest_sha != FROZEN_QC1_HOLDOUT_MANIFEST_SHA256:
        raise RuntimeError(f"QC1 holdout manifest SHA256 mismatch: {manifest_sha}")

    manifest = read_csv(manifest_path)
    bags = validate_manifest(manifest)
    v1 = load_module("nasa_bps_r1_v1_final_holdout_eval", script_dir / "train_pilot_v1_r1_mil.py")
    dev_qc = load_module("nasa_bps_dev_qc_final_holdout_eval", script_dir / "validate_r1_v2_development.py")
    v1.seed_everything(SEED)
    qc = hard_qc(dev_qc, manifest, image_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(v1, checkpoint_path, device)

    print("Radiation Edge AI - NASA BPS R1 v2 FINAL HOLDOUT confirmatory evaluator")
    print(f"mode: {'ONE-TIME UNBLIND' if args.unblind else 'PHENOTYPE-BLIND PRE-FLIGHT'}")
    print(f"evaluator source SHA256: {evaluator_source_sha}")
    print(f"repaired FP32 SHA256: {checkpoint_sha}")
    print(f"QC1 holdout manifest SHA256: {manifest_sha}")
    print(f"raw phenotype expected SHA256 pinned: {FROZEN_RAW_PHENOTYPE_SHA256}")
    print(f"holdout nuclei/bags/contrasts: {EXPECTED_NUCLEI}/{EXPECTED_BAGS}/{EXPECTED_CONTRASTS}")
    print(f"hard QC recheck: {qc['nuclei']}/{qc['nuclei']} PASS")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Run phenotype-blind inference first so any model/image failure happens before unblinding.
    blinded_bags, nucleus_rows = infer_blinded(v1, model, bags, image_root, device, args.cache_items)
    print(f"phenotype-blind inference: {len(nucleus_rows)} nuclei -> {len(blinded_bags)} bags; finite outputs YES")

    preflight = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "unblind" if args.unblind else "preflight",
        "evaluator_source_sha256": evaluator_source_sha,
        "repaired_checkpoint_sha256": checkpoint_sha,
        "qc1_holdout_manifest_sha256": manifest_sha,
        "raw_phenotype_expected_sha256": FROZEN_RAW_PHENOTYPE_SHA256,
        "raw_phenotype_read": False,
        "holdout_nuclei": EXPECTED_NUCLEI,
        "holdout_bags": EXPECTED_BAGS,
        "holdout_matched_contrasts": EXPECTED_CONTRASTS,
        "hard_qc": qc,
        "phenotype_blind_inference_complete": True,
        "prediction_values_written": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = output_dir / "final_holdout_evaluator_preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2), encoding="utf-8")

    if not args.unblind:
        print("NASA phenotype/reference tables read: NO")
        print("FINAL HOLDOUT PHENOTYPE STATUS: BLINDED")
        print(f"preflight summary: {preflight_path}")
        print("NASA BPS R1 V2 FINAL HOLDOUT EVALUATOR PRE-FLIGHT COMPLETE: YES")
        print("NEXT GATE: verify this frozen evaluator identity, then rerun exactly once with --unblind.")
        return 0

    complete_marker = output_dir / "FINAL_HOLDOUT_CONFIRMATION_COMPLETE.json"
    start_marker = output_dir / "FINAL_HOLDOUT_UNBLINDING_STARTED.json"
    if complete_marker.exists():
        raise RuntimeError(
            f"Final holdout confirmation already completed; refusing overwrite/re-evaluation: {complete_marker}"
        )
    if start_marker.exists():
        raise RuntimeError(
            "A prior unblinding start marker already exists without a completed result. "
            "Do not silently rerun or modify the evaluator; inspect/document the interrupted run first."
        )
    start_marker.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "evaluator_source_sha256": evaluator_source_sha,
                "repaired_checkpoint_sha256": checkpoint_sha,
                "qc1_holdout_manifest_sha256": manifest_sha,
                "raw_phenotype_expected_sha256": FROZEN_RAW_PHENOTYPE_SHA256,
                "status": "UNBLINDING_STARTED_NO_TUNING_PERMITTED",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print("[ONE-TIME PHENOTYPE UNBLINDING]")
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    raw_sha = sha256_file(raw_path)
    if raw_sha != FROZEN_RAW_PHENOTYPE_SHA256:
        raise RuntimeError(
            f"Raw phenotype SHA256 mismatch: expected {FROZEN_RAW_PHENOTYPE_SHA256}, got {raw_sha}"
        )
    raw_rows = read_csv(raw_path)
    references = join_raw_phenotypes(raw_rows, bags)
    print(f"raw phenotype SHA256 verified: {raw_sha}")
    print(f"holdout aggregate references joined by exact Sample Name: {len(references)}/{EXPECTED_BAGS}")

    bag_rows = attach_references(blinded_bags, references)
    delta_rows = build_deltas(bag_rows)
    peaks = peak_rows(delta_rows)
    directions = direction_rows(delta_rows)
    source_residuals = source_residual_rows(v1, bag_rows)
    gate = evaluate_gate(v1, bag_rows, delta_rows, peaks)

    write_csv(output_dir / "final_holdout_bag_predictions.csv", bag_rows)
    write_csv(output_dir / "final_holdout_nucleus_latent_burden.csv", nucleus_rows)
    write_csv(output_dir / "final_holdout_matched_radiation_deltas.csv", delta_rows)
    write_csv(output_dir / "final_holdout_direction_summary.csv", directions)
    write_csv(output_dir / "final_holdout_peak_time_recovery.csv", peaks)
    write_csv(output_dir / "final_holdout_source_residuals.csv", source_residuals)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": "COMPLETE_ONE_TIME_CONFIRMATORY_HOLDOUT",
        "evaluator_source_sha256": evaluator_source_sha,
        "repaired_checkpoint_sha256": checkpoint_sha,
        "qc1_holdout_manifest_sha256": manifest_sha,
        "raw_phenotype_sha256": raw_sha,
        "raw_phenotype_read": True,
        "primary_reference_endpoint": "NASA sample/plate-well aggregate avg_nfoci",
        "per_nucleus_prediction_semantics": "continuous latent 53BP1 burden; not individually supervised and not validated as discrete focus count",
        "biological_n_semantics": "Source Name/sample structure; nuclei are not independent biological replicates",
        "bag_count": len(bag_rows),
        "nucleus_count": len(nucleus_rows),
        "matched_contrast_count": len(delta_rows),
        "gate_thresholds": GATE,
        "gate_result": gate,
        "source_residuals": source_residuals,
        "no_holdout_outcome_used_for_tuning": True,
        "model_retrained_after_unblinding": False,
        "model_recalibrated_after_unblinding": False,
        "thresholds_changed_after_unblinding": False,
        "advance_to_onnx_int8_kl720": bool(gate["all_pass"]),
        "interpretation": (
            "FP32 confirmatory biological-fidelity gate PASSED; the exact frozen FP32 model may advance to ONNX/INT8/KL720 deployment-equivalence testing."
            if gate["all_pass"]
            else "FP32 confirmatory biological-fidelity gate FAILED; report as a failed confirmatory evaluation. This holdout must not be reused as an untouched tuning set."
        ),
    }
    summary_path = output_dir / "final_holdout_confirmation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    complete_marker.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    bm = gate["bag_metrics"]
    dm = gate["delta_metrics"]
    print("")
    print("[Confirmatory FP32 final-holdout result]")
    print(
        f"bag MAE/RMSE/Pearson/Spearman: {bm['mae']:.4f} / {bm['rmse']:.4f} / "
        f"{bm['pearson']:.4f} / {bm['spearman']:.4f}"
    )
    print(
        f"delta MAE/Pearson/Spearman: {dm['delta_mae']:.4f} / {dm['delta_pearson']:.4f} / "
        f"{dm['delta_spearman']:.4f}"
    )
    print(f"direction overall: {gate['direction_counts']['overall']}")
    print(f"direction 4 h: {gate['direction_counts']['4h']}")
    print(f"direction 24+48 h: {gate['direction_counts']['24+48h']}")
    print(f"direction by Source Name: {gate['direction_counts']['by_source']}")
    print(f"peak-time recovery: {gate['peak_time_recovery']}")
    print("")
    for name, passed in gate["checks"].items():
        print(f"GATE {name}: {'PASS' if passed else 'FAIL'}")
    print(f"ALL SEVEN PREDECLARED FINAL-HOLDOUT CRITERIA PASS: {'YES' if gate['all_pass'] else 'NO'}")
    print("NO HOLDOUT OUTCOME USED FOR TUNING: YES")
    print(f"summary: {summary_path}")
    print("FINAL HOLDOUT PHENOTYPE STATUS: UNBLINDED - CONFIRMATORY EVALUATION COMPLETE")
    if gate["all_pass"]:
        print("NEXT GATE: freeze this FP32 result as deployment reference; proceed to ONNX numerical parity, then INT8/BIE/NEF/KL720 equivalence.")
    else:
        print("NEXT GATE: report failed confirmatory FP32 evaluation; do not tune R1 v2 on this holdout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
