"""Evaluate frozen physical KL720 predictions against the original seven gates.

This is the terminal biological deployment-equivalence evaluator for NASA BPS
R1 v2. It performs NO hardware inference, NO preprocessing, NO quantization,
and NO model conversion. The complete physical KL720 prediction set was frozen
outcome-blind before this evaluator is allowed to read any biological target or
reference output.

Targets are taken only from the immutable FP32 confirmatory deployment
reference; the raw NASA phenotype table is not reopened. The already accepted
BIE result may be read only for descriptive physical-vs-BIE numerical
comparison. No new physical-vs-BIE numerical acceptance threshold is introduced
post hoc. The seven biological gates are exactly the original predeclared gates.

A failed result is reported as a deployment failure. PTQ/BIE/NEF/model/threshold
retuning from this holdout is not authorized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

FROZEN_PHYSICAL_PREDICTIONS_FREEZE_SHA256 = (
    "e4e7d9150e2e33479e2ec8894e583f0d09a3d890a6a97fbd356a1a0fb78d2b30"
)
FROZEN_PHYSICAL_PREDICTIONS_SHA256 = (
    "91ea895a16cd97067af63b5c5c4434e972629bfda4c9582ad456a79e007b6d46"
)
FROZEN_PHYSICAL_INFERENCE_SUMMARY_SHA256 = (
    "7147ba4de4ba44fa64ed6b414e990970cd08ac83ea5e4b993ed38e9ad0a93aba"
)
FROZEN_NEF_SHA256 = (
    "d04f02855a8a82ae7cb5ff48af91eee69c341142d1097469890385bd23ee097b"
)
FROZEN_NEF_DEPLOYMENT_FREEZE_SHA256 = (
    "f818b9d912017a1d8fdc4aafd4bf321b8bf5b84e8214e6ef0ba029b48b0b90bb"
)
FROZEN_PHYSICAL_SMOKE_PASS_FREEZE_SHA256 = (
    "224b754235ebcd22eb053ca1f279043f3b74e9e3b461ed1f8dad26beddfa679d"
)
FROZEN_QC1_HOLDOUT_MANIFEST_SHA256 = (
    "f89962f96bec05930c07149717a90bd682016a2778bbfae6b827425159aa0643"
)
FROZEN_FP32_REFERENCE_FREEZE_SHA256 = (
    "abb58026d4f798554ef7e44346a989499242fcbd89d8c6629a1c4209d6c16ce3"
)
FROZEN_BIE_EQUIVALENCE_FREEZE_SHA256 = (
    "e901d4be044c5ea5507b3b60643043eb8aa0af5643a0b7fba2f9bdd7c588bb0a"
)
FROZEN_BIE_EQUIVALENCE_SUMMARY_SHA256 = (
    "c15776c6820a4aa28fff8e18c16b576164ce3ed2cad61ba2057efdef79800ca4"
)

EXPECTED_PLATFORM = 720
EXPECTED_MODEL_ID = 32770
EXPECTED_MODEL_VERSION = "8b29"
EXPECTED_USB_PORT = 81
EXPECTED_NUCLEI = 2058
EXPECTED_BAGS = 22
EXPECTED_CONTRASTS = 11
EXPECTED_SOURCE_NUCLEI = {"BALBCF2": 858, "C57BLF2": 600, "C57BLF3": 600}
EXPECTED_SOURCE_CONTRASTS = {"BALBCF2": 5, "C57BLF2": 3, "C57BLF3": 3}
EXPECTED_SOURCE_DIRECTION_MIN = {"BALBCF2": 4, "C57BLF2": 2, "C57BLF3": 2}
EXPOSED_DOSE = {"Fe": "0.82", "X-ray": "1.0"}

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

EXPECTED_FP32_DIRECTION_COUNTS = {
    "overall": "10/11",
    "4h": "4/4",
    "24+48h": "6/7",
    "by_source": {
        "BALBCF2": "4/5",
        "C57BLF2": "3/3",
        "C57BLF3": "3/3",
    },
}
EXPECTED_FP32_PEAK = "3/3"
EXPECTED_OUTPUT_SEMANTICS = (
    "scalar latent continuous 53BP1 burden; not per-nucleus focus count"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return obj


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
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


def norm_number(value: Any) -> str:
    x = float(str(value).strip())
    if x.is_integer():
        return f"{x:.1f}"
    return format(x, "g")


def norm_hour(value: Any) -> str:
    x = float(str(value).strip())
    return str(int(x)) if x.is_integer() else format(x, "g")


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


def error_stats(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if ref.shape != cand.shape or ref.size == 0:
        raise RuntimeError(f"Cannot compare shapes {ref.shape} and {cand.shape}")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(cand)):
        raise RuntimeError("Non-finite value in numerical comparison")
    diff = cand - ref
    absolute = np.abs(diff)
    return {
        "n": int(ref.size),
        "exact_array_equality": bool(np.array_equal(ref, cand)),
        "max_abs_error": float(absolute.max()),
        "mean_abs_error": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "pearson": correlation(ref, cand),
    }


def regression_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([float(row["nasa_avg_nfoci"]) for row in rows], dtype=np.float64)
    p = np.asarray([float(row["predicted_bag_mean"]) for row in rows], dtype=np.float64)
    err = p - y
    return {
        "n_bags": int(len(rows)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "pearson": correlation(y, p),
        "spearman": correlation(average_ranks(y), average_ranks(p)),
    }


def delta_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nasa = np.asarray([float(row["nasa_delta"]) for row in rows], dtype=np.float64)
    pred = np.asarray([float(row["pred_delta"]) for row in rows], dtype=np.float64)
    return {
        "n_matched_contrasts": int(len(rows)),
        "direction_agreement_fraction": float(np.mean(np.sign(nasa) == np.sign(pred))),
        "delta_mae": float(np.mean(np.abs(pred - nasa))),
        "delta_pearson": correlation(nasa, pred),
        "delta_spearman": correlation(average_ranks(nasa), average_ranks(pred)),
    }


def verify_physical_freeze(path: Path) -> dict[str, Any]:
    check_sha(path, FROZEN_PHYSICAL_PREDICTIONS_FREEZE_SHA256, "Physical prediction freeze")
    obj = read_json(path)
    expected = {
        "status": "FROZEN_OUTCOME_BLIND_FULL_PHYSICAL_KL720_HOLDOUT_PREDICTIONS_BEFORE_BIOLOGICAL_EVALUATION",
        "nef_sha256": FROZEN_NEF_SHA256,
        "nef_deployment_freeze_sha256": FROZEN_NEF_DEPLOYMENT_FREEZE_SHA256,
        "physical_smoke_pass_freeze_sha256": FROZEN_PHYSICAL_SMOKE_PASS_FREEZE_SHA256,
        "qc1_holdout_manifest_sha256": FROZEN_QC1_HOLDOUT_MANIFEST_SHA256,
        "physical_holdout_inference_summary_sha256": FROZEN_PHYSICAL_INFERENCE_SUMMARY_SHA256,
        "physical_holdout_nucleus_predictions_sha256": FROZEN_PHYSICAL_PREDICTIONS_SHA256,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "usb_port": EXPECTED_USB_PORT,
        "nuclei": EXPECTED_NUCLEI,
        "bags": EXPECTED_BAGS,
        "source_nuclei": EXPECTED_SOURCE_NUCLEI,
        "output_semantics": EXPECTED_OUTPUT_SEMANTICS,
        "raw_phenotype_table_read_by_freeze": False,
        "fp32_or_bie_holdout_reference_outputs_read_by_freeze": False,
        "biological_targets_or_gate_results_read_by_freeze": False,
        "ptq_bie_or_nef_changed": False,
        "post_holdout_ptq_retuning_authorized": False,
        "authorized_next_stage": "EVALUATE_FROZEN_PHYSICAL_PREDICTIONS_AGAINST_ORIGINAL_SEVEN_BIOLOGICAL_GATES",
    }
    for key, value in expected.items():
        if obj.get(key) != value:
            raise RuntimeError(
                f"Physical freeze invariant mismatch for {key}: expected {value!r}, got {obj.get(key)!r}"
            )
    return obj


def verify_fp32_reference(freeze_path: Path, confirmation_dir: Path) -> dict[str, Any]:
    check_sha(freeze_path, FROZEN_FP32_REFERENCE_FREEZE_SHA256, "FP32 deployment reference freeze")
    freeze = read_json(freeze_path)
    if freeze.get("status") != "FROZEN_FP32_DEPLOYMENT_REFERENCE_AFTER_CONFIRMATORY_PASS":
        raise RuntimeError("Unexpected FP32 deployment-reference status")
    biological = freeze.get("biological_reference") or {}
    if biological.get("nuclei") != EXPECTED_NUCLEI:
        raise RuntimeError("FP32 reference nucleus count mismatch")
    if biological.get("bags") != EXPECTED_BAGS:
        raise RuntimeError("FP32 reference bag count mismatch")
    if biological.get("matched_contrasts") != EXPECTED_CONTRASTS:
        raise RuntimeError("FP32 reference contrast count mismatch")
    if biological.get("direction_counts") != EXPECTED_FP32_DIRECTION_COUNTS:
        raise RuntimeError("FP32 reference direction signature mismatch")
    if biological.get("peak_time_recovery") != EXPECTED_FP32_PEAK:
        raise RuntimeError("FP32 reference peak signature mismatch")
    if biological.get("all_seven_original_gates_pass") is not True:
        raise RuntimeError("FP32 reference did not pass all seven gates")
    reference_files = freeze.get("reference_files") or {}
    required = (
        "final_holdout_confirmation_summary.json",
        "final_holdout_bag_predictions.csv",
        "final_holdout_nucleus_latent_burden.csv",
    )
    for name in required:
        meta = reference_files.get(name)
        candidate = confirmation_dir / name
        if not isinstance(meta, dict) or not candidate.is_file():
            raise RuntimeError(f"Frozen FP32 reference file missing: {name}")
        if sha256_file(candidate) != meta.get("sha256"):
            raise RuntimeError(f"Frozen FP32 reference file changed: {name}")
    return freeze


def verify_bie_reference(freeze_path: Path, equivalence_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    check_sha(freeze_path, FROZEN_BIE_EQUIVALENCE_FREEZE_SHA256, "BIE equivalence freeze")
    freeze = read_json(freeze_path)
    if freeze.get("status") != "FROZEN_KL720_INT8_BIE_BIOLOGICAL_EQUIVALENCE_PASS_BEFORE_NEF":
        raise RuntimeError("Unexpected BIE equivalence freeze status")
    if freeze.get("all_seven_original_biological_gates_pass") is not True:
        raise RuntimeError("Accepted BIE reference did not pass all seven gates")
    if freeze.get("exact_fp32_direction_peak_signature_reproduced") is not True:
        raise RuntimeError("Accepted BIE reference did not reproduce FP32 direction/peak signature")
    if freeze.get("ptq_retuning_after_this_freeze_authorized") is not False:
        raise RuntimeError("BIE equivalence freeze permits PTQ retuning")
    summary_path = equivalence_dir / "bie_holdout_equivalence_summary.json"
    check_sha(summary_path, FROZEN_BIE_EQUIVALENCE_SUMMARY_SHA256, "BIE equivalence summary")
    summary = read_json(summary_path)
    outputs = summary.get("outputs") or {}
    for name in ("bie_holdout_nucleus_predictions.csv", "bie_holdout_bag_predictions.csv"):
        meta = outputs.get(name)
        candidate = equivalence_dir / name
        if not isinstance(meta, dict) or not candidate.is_file():
            raise RuntimeError(f"Frozen BIE reference output missing: {name}")
        if sha256_file(candidate) != meta.get("sha256"):
            raise RuntimeError(f"Frozen BIE reference output changed: {name}")
    return freeze, summary


def build_bag_rows(
    physical_rows: list[dict[str, str]],
    reference_bags: dict[str, dict[str, str]],
    bie_bags: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in physical_rows:
        grouped[row["sample_name"]].append(row)
    if len(grouped) != EXPECTED_BAGS:
        raise RuntimeError(f"Physical result bag count mismatch: {len(grouped)}")
    result: list[dict[str, Any]] = []
    for sample_name, subset in sorted(grouped.items()):
        ref = reference_bags.get(sample_name)
        bie = bie_bags.get(sample_name)
        if ref is None or bie is None:
            raise RuntimeError(f"Missing frozen reference bag: {sample_name}")
        physical_mean = float(np.mean([float(row["physical_kl720_burden"]) for row in subset]))
        bie_mean = float(bie["predicted_bag_mean"])
        out = {
            "sample_name": sample_name,
            "source_name": ref["source_name"],
            "strain": ref["strain"],
            "sex": ref["sex"],
            "particle_type": ref["particle_type"],
            "dose_Gy": norm_number(ref["dose_Gy"]),
            "hr_post_exposure": norm_hour(ref["hr_post_exposure"]),
            "n_holdout_nuclei": len(subset),
            "predicted_bag_mean": physical_mean,
            "bie_reference_bag_mean": bie_mean,
            "fp32_reference_bag_mean": float(ref["predicted_bag_mean"]),
            "nasa_avg_nfoci": float(ref["nasa_avg_nfoci"]),
            "nasa_num_nuc": int(float(ref["nasa_num_nuc"])),
        }
        out["physical_minus_bie_bag"] = physical_mean - bie_mean
        out["physical_minus_fp32_bag"] = physical_mean - out["fp32_reference_bag_mean"]
        out["physical_residual"] = physical_mean - out["nasa_avg_nfoci"]
        result.append(out)
    return result


def build_deltas(bag_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["source_name"]),
            str(row["particle_type"]),
            norm_number(row["dose_Gy"]),
            norm_hour(row["hr_post_exposure"]),
        ): row
        for row in bag_rows
    }
    rows: list[dict[str, Any]] = []
    for source in EXPECTED_SOURCE_NUCLEI:
        for branch in ("Fe", "X-ray"):
            exposed = norm_number(EXPOSED_DOSE[branch])
            for hour in ("4", "24", "48"):
                sham = lookup.get((source, branch, "0.0", hour))
                exp = lookup.get((source, branch, exposed, hour))
                if sham is None or exp is None:
                    continue
                nasa_delta = float(exp["nasa_avg_nfoci"]) - float(sham["nasa_avg_nfoci"])
                pred_delta = float(exp["predicted_bag_mean"]) - float(sham["predicted_bag_mean"])
                rows.append(
                    {
                        "source_name": source,
                        "branch": branch,
                        "timepoint_hr": hour,
                        "sham_sample_name": sham["sample_name"],
                        "exposed_sample_name": exp["sample_name"],
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
        raise RuntimeError(
            f"Physical matched deltas do not reproduce frozen 11-contrast structure: {dict(counts)}"
        )
    return rows


def build_peaks(delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        grouped[(str(row["source_name"]), str(row["branch"]))].append(row)
    rows: list[dict[str, Any]] = []
    for (source, branch), subset in sorted(grouped.items()):
        hours = {str(row["timepoint_hr"]) for row in subset}
        if hours != {"4", "24", "48"}:
            continue
        nasa_peak = max(subset, key=lambda row: float(row["nasa_delta"]))
        pred_peak = max(subset, key=lambda row: float(row["pred_delta"]))
        rows.append(
            {
                "source_name": source,
                "branch": branch,
                "nasa_peak_time_hr": str(nasa_peak["timepoint_hr"]),
                "pred_peak_time_hr": str(pred_peak["timepoint_hr"]),
                "peak_time_recovered": int(
                    str(nasa_peak["timepoint_hr"]) == str(pred_peak["timepoint_hr"])
                ),
                "nasa_peak_delta": float(nasa_peak["nasa_delta"]),
                "pred_peak_delta": float(pred_peak["pred_delta"]),
            }
        )
    if len(rows) != 3:
        raise RuntimeError(f"Expected exactly three complete source x branch peak series; got {len(rows)}")
    return rows


def evaluate_gate(
    bag_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    peaks: list[dict[str, Any]],
) -> dict[str, Any]:
    bag = regression_metrics(bag_rows)
    delta = delta_metrics(delta_rows)
    four = [row for row in delta_rows if str(row["timepoint_hr"]) == "4"]
    late = [row for row in delta_rows if str(row["timepoint_hr"]) in {"24", "48"}]
    four_ok = sum(int(row["direction_agreement"]) for row in four)
    late_ok = sum(int(row["direction_agreement"]) for row in late)
    overall_ok = sum(int(row["direction_agreement"]) for row in delta_rows)
    peak_ok = sum(int(row["peak_time_recovered"]) for row in peaks)
    source_counts = {
        source: sum(
            int(row["direction_agreement"])
            for row in delta_rows
            if str(row["source_name"]) == source
        )
        for source in EXPECTED_SOURCE_NUCLEI
    }
    checks = {
        "bag_spearman_ge_0_50": float(bag["spearman"]) >= GATE["bag_spearman_min"],
        "delta_spearman_ge_0_60": float(delta["delta_spearman"]) >= GATE["delta_spearman_min"],
        "overall_direction_ge_9_of_11": len(delta_rows) == 11 and overall_ok >= 9,
        "four_hour_direction_eq_4_of_4": len(four) == 4 and four_ok == 4,
        "late_24_48_direction_ge_5_of_7": len(late) == 7 and late_ok >= 5,
        "peak_time_recovery_eq_3_of_3": len(peaks) == 3 and peak_ok == 3,
        "every_source_direction_minimum": all(
            source_counts[source] >= EXPECTED_SOURCE_DIRECTION_MIN[source]
            for source in EXPECTED_SOURCE_DIRECTION_MIN
        ),
    }
    directions = {
        "overall": f"{overall_ok}/11",
        "4h": f"{four_ok}/4",
        "24+48h": f"{late_ok}/7",
        "by_source": {
            source: f"{source_counts[source]}/{EXPECTED_SOURCE_CONTRASTS[source]}"
            for source in EXPECTED_SOURCE_NUCLEI
        },
    }
    return {
        "bag_metrics": bag,
        "delta_metrics": delta,
        "direction_counts": directions,
        "peak_time_recovery": f"{peak_ok}/3",
        "checks": checks,
        "all_pass": all(checks.values()),
        "exact_fp32_direction_signature_reproduced": directions == EXPECTED_FP32_DIRECTION_COUNTS,
        "exact_fp32_peak_signature_reproduced": f"{peak_ok}/3" == EXPECTED_FP32_PEAK,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    deployment_root = model_root / "nasa_bps_53bp1" / "r1_v2_deployment"
    physical_root = deployment_root / "kneron_nef" / "physical_holdout_hardware"
    confirmation_dir_default = model_root / "nasa_bps_53bp1" / "r1_v2_final_holdout_confirmation"
    bie_dir_default = deployment_root / "kneron_int8" / "bie_holdout_equivalence"

    parser.add_argument("--physical-predictions", default=str(physical_root / "physical_holdout_nucleus_predictions.csv"))
    parser.add_argument("--physical-inference-summary", default=str(physical_root / "physical_holdout_inference_summary.json"))
    parser.add_argument("--physical-predictions-freeze", default=str(physical_root / "physical_holdout_predictions_freeze.json"))
    parser.add_argument("--fp32-reference-freeze", default=str(confirmation_dir_default / "fp32_deployment_reference_freeze.json"))
    parser.add_argument("--confirmation-dir", default=str(confirmation_dir_default))
    parser.add_argument("--bie-equivalence-freeze", default=str(bie_dir_default / "bie_holdout_equivalence_freeze.json"))
    parser.add_argument("--bie-equivalence-dir", default=str(bie_dir_default))
    parser.add_argument("--output-dir", default=str(physical_root / "biological_equivalence"))
    args = parser.parse_args()

    physical_path = Path(args.physical_predictions).resolve()
    physical_summary_path = Path(args.physical_inference_summary).resolve()
    physical_freeze_path = Path(args.physical_predictions_freeze).resolve()
    fp32_freeze_path = Path(args.fp32_reference_freeze).resolve()
    confirmation_dir = Path(args.confirmation_dir).resolve()
    bie_freeze_path = Path(args.bie_equivalence_freeze).resolve()
    bie_dir = Path(args.bie_equivalence_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    verify_physical_freeze(physical_freeze_path)
    check_sha(physical_path, FROZEN_PHYSICAL_PREDICTIONS_SHA256, "Physical prediction CSV")
    check_sha(physical_summary_path, FROZEN_PHYSICAL_INFERENCE_SUMMARY_SHA256, "Physical inference summary")
    verify_fp32_reference(fp32_freeze_path, confirmation_dir)
    _bie_freeze, bie_summary = verify_bie_reference(bie_freeze_path, bie_dir)

    physical_rows = read_csv(physical_path)
    fp32_nucleus_rows = read_csv(confirmation_dir / "final_holdout_nucleus_latent_burden.csv")
    fp32_bag_rows = read_csv(confirmation_dir / "final_holdout_bag_predictions.csv")
    bie_nucleus_rows = read_csv(bie_dir / "bie_holdout_nucleus_predictions.csv")
    bie_bag_rows = read_csv(bie_dir / "bie_holdout_bag_predictions.csv")

    if len(physical_rows) != EXPECTED_NUCLEI:
        raise RuntimeError(f"Physical nuclei={len(physical_rows)}; expected {EXPECTED_NUCLEI}")
    if len(fp32_nucleus_rows) != EXPECTED_NUCLEI or len(bie_nucleus_rows) != EXPECTED_NUCLEI:
        raise RuntimeError("Frozen nucleus reference count mismatch")
    if len(fp32_bag_rows) != EXPECTED_BAGS or len(bie_bag_rows) != EXPECTED_BAGS:
        raise RuntimeError("Frozen bag reference count mismatch")

    physical_by_id = {row["sample_id"]: row for row in physical_rows}
    fp32_by_id = {row["sample_id"]: row for row in fp32_nucleus_rows}
    bie_by_id = {row["sample_id"]: row for row in bie_nucleus_rows}
    if len(physical_by_id) != EXPECTED_NUCLEI or len(fp32_by_id) != EXPECTED_NUCLEI or len(bie_by_id) != EXPECTED_NUCLEI:
        raise RuntimeError("Duplicate sample_id in frozen nucleus results")
    if set(physical_by_id) != set(fp32_by_id) or set(physical_by_id) != set(bie_by_id):
        raise RuntimeError("Physical/FP32/BIE nucleus identities differ")

    physical_values: list[float] = []
    fp32_values: list[float] = []
    bie_values: list[float] = []
    for sample_id in sorted(physical_by_id):
        physical = float(physical_by_id[sample_id]["physical_kl720_burden"])
        fp32 = float(fp32_by_id[sample_id]["predicted_continuous_burden"])
        bie = float(bie_by_id[sample_id]["bie_burden"])
        if not all(math.isfinite(v) for v in (physical, fp32, bie)):
            raise RuntimeError(f"Non-finite frozen burden at {sample_id}")
        physical_values.append(physical)
        fp32_values.append(fp32)
        bie_values.append(bie)

    fp32_bags = {row["sample_name"]: row for row in fp32_bag_rows}
    bie_bags = {row["sample_name"]: row for row in bie_bag_rows}
    if len(fp32_bags) != EXPECTED_BAGS or len(bie_bags) != EXPECTED_BAGS:
        raise RuntimeError("Duplicate sample_name in frozen bag references")

    bag_rows = build_bag_rows(physical_rows, fp32_bags, bie_bags)
    delta_rows = build_deltas(bag_rows)
    peak_rows = build_peaks(delta_rows)
    gate = evaluate_gate(bag_rows, delta_rows, peak_rows)

    physical_vs_fp32_nucleus = error_stats(fp32_values, physical_values)
    physical_vs_bie_nucleus = error_stats(bie_values, physical_values)
    fp32_bag_values = [float(row["fp32_reference_bag_mean"]) for row in bag_rows]
    bie_bag_values = [float(row["bie_reference_bag_mean"]) for row in bag_rows]
    physical_bag_values = [float(row["predicted_bag_mean"]) for row in bag_rows]
    physical_vs_fp32_bag = error_stats(fp32_bag_values, physical_bag_values)
    physical_vs_bie_bag = error_stats(bie_bag_values, physical_bag_values)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "physical_holdout_biological_equivalence_summary.json"
    bag_path = output_dir / "physical_holdout_bag_predictions.csv"
    delta_path = output_dir / "physical_holdout_matched_radiation_deltas.csv"
    peak_path = output_dir / "physical_holdout_peak_time_recovery.csv"
    if any(path.exists() for path in (summary_path, bag_path, delta_path, peak_path)):
        raise RuntimeError(f"Physical biological-equivalence result already exists; refusing overwrite: {output_dir}")

    write_csv(bag_path, bag_rows)
    write_csv(delta_path, delta_rows)
    write_csv(peak_path, peak_rows)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_FROZEN_PHYSICAL_KL720_BIOLOGICAL_DEPLOYMENT_EQUIVALENCE",
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "physical_holdout_predictions_freeze_sha256": FROZEN_PHYSICAL_PREDICTIONS_FREEZE_SHA256,
        "physical_holdout_nucleus_predictions_sha256": FROZEN_PHYSICAL_PREDICTIONS_SHA256,
        "physical_holdout_inference_summary_sha256": FROZEN_PHYSICAL_INFERENCE_SUMMARY_SHA256,
        "nef_sha256": FROZEN_NEF_SHA256,
        "nef_deployment_freeze_sha256": FROZEN_NEF_DEPLOYMENT_FREEZE_SHA256,
        "physical_smoke_pass_freeze_sha256": FROZEN_PHYSICAL_SMOKE_PASS_FREEZE_SHA256,
        "qc1_holdout_manifest_sha256": FROZEN_QC1_HOLDOUT_MANIFEST_SHA256,
        "fp32_deployment_reference_freeze_sha256": FROZEN_FP32_REFERENCE_FREEZE_SHA256,
        "bie_equivalence_freeze_sha256": FROZEN_BIE_EQUIVALENCE_FREEZE_SHA256,
        "bie_equivalence_summary_sha256": FROZEN_BIE_EQUIVALENCE_SUMMARY_SHA256,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "usb_port": EXPECTED_USB_PORT,
        "nuclei": EXPECTED_NUCLEI,
        "bags": EXPECTED_BAGS,
        "matched_contrasts": EXPECTED_CONTRASTS,
        "output_semantics": EXPECTED_OUTPUT_SEMANTICS,
        "targets_source": "immutable FP32 confirmatory reference outputs frozen before deployment conversion",
        "raw_phenotype_table_read": False,
        "protocol_deviation_carried_forward": "documented non-outcome metadata-normalization implementation recovery",
        "physical_predictions_frozen_outcome_blind_before_this_evaluation": True,
        "ptq_bie_nef_or_model_changed_after_physical_predictions": False,
        "post_holdout_ptq_retuning_authorized": False,
        "original_predeclared_biological_gate": GATE,
        "physical_vs_fp32_nucleus_descriptive_only": physical_vs_fp32_nucleus,
        "physical_vs_bie_nucleus_descriptive_only_no_posthoc_gate": physical_vs_bie_nucleus,
        "physical_vs_fp32_bag_descriptive_only": physical_vs_fp32_bag,
        "physical_vs_bie_bag_descriptive_only_no_posthoc_gate": physical_vs_bie_bag,
        "accepted_bie_biological_gate_reference": bie_summary.get("biological_gate"),
        "biological_gate": gate,
        "all_seven_original_biological_gates_pass": bool(gate["all_pass"]),
        "exact_fp32_direction_peak_signature_reproduced": bool(
            gate["exact_fp32_direction_signature_reproduced"]
            and gate["exact_fp32_peak_signature_reproduced"]
        ),
        "outputs": {
            bag_path.name: {"sha256": sha256_file(bag_path), "size_bytes": bag_path.stat().st_size},
            delta_path.name: {"sha256": sha256_file(delta_path), "size_bytes": delta_path.stat().st_size},
            peak_path.name: {"sha256": sha256_file(peak_path), "size_bytes": peak_path.stat().st_size},
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    bm = gate["bag_metrics"]
    dm = gate["delta_metrics"]
    print("Radiation Edge AI - NASA BPS R1 v2 FINAL physical KL720 biological equivalence")
    print(f"physical prediction freeze SHA256: {FROZEN_PHYSICAL_PREDICTIONS_FREEZE_SHA256}")
    print(f"physical prediction CSV SHA256: {FROZEN_PHYSICAL_PREDICTIONS_SHA256}")
    print(f"FP32 deployment reference freeze SHA256: {FROZEN_FP32_REFERENCE_FREEZE_SHA256}")
    print(f"BIE equivalence freeze SHA256: {FROZEN_BIE_EQUIVALENCE_FREEZE_SHA256}")
    print("raw phenotype table read: NO")
    print("physical predictions frozen before outcome/reference read: YES")
    print("")
    print("[Physical KL720 biological deployment-equivalence result]")
    print(
        "bag MAE/RMSE/Pearson/Spearman: "
        f"{bm['mae']:.4f} / {bm['rmse']:.4f} / {bm['pearson']:.4f} / {bm['spearman']:.4f}"
    )
    print(
        "delta MAE/Pearson/Spearman: "
        f"{dm['delta_mae']:.4f} / {dm['delta_pearson']:.4f} / {dm['delta_spearman']:.4f}"
    )
    print(f"direction overall: {gate['direction_counts']['overall']}")
    print(f"direction 4 h: {gate['direction_counts']['4h']}")
    print(f"direction 24+48 h: {gate['direction_counts']['24+48h']}")
    print(f"direction by Source Name: {gate['direction_counts']['by_source']}")
    print(f"peak-time recovery: {gate['peak_time_recovery']}")
    print("")
    print("[Physical vs accepted BIE; descriptive only, no post-hoc gate]")
    print(f"nucleus exact equality: {'YES' if physical_vs_bie_nucleus['exact_array_equality'] else 'NO'}")
    print(f"nucleus max abs error: {physical_vs_bie_nucleus['max_abs_error']:.10g}")
    print(f"bag max abs error: {physical_vs_bie_bag['max_abs_error']:.10g}")
    print("")
    for name, passed in gate["checks"].items():
        print(f"GATE {name}: {'PASS' if passed else 'FAIL'}")
    print(f"ALL SEVEN ORIGINAL BIOLOGICAL GATES PASS: {'YES' if gate['all_pass'] else 'NO'}")
    print(
        "EXACT FP32 DIRECTION/PEAK SIGNATURE REPRODUCED: "
        + ("YES" if summary["exact_fp32_direction_peak_signature_reproduced"] else "NO")
    )
    print("PTQ/BIE/NEF/model/threshold retuning from this holdout authorized: NO")
    print(f"summary: {summary_path}")
    print(f"summary SHA256: {sha256_file(summary_path)}")
    if gate["all_pass"]:
        print("NASA BPS R1 V2 PHYSICAL KL720 BIOLOGICAL DEPLOYMENT-EQUIVALENCE PASS: YES")
        print("NEXT GATE: freeze this final physical biological result; no model/PTQ/threshold changes.")
    else:
        print("NASA BPS R1 V2 PHYSICAL KL720 BIOLOGICAL DEPLOYMENT-EQUIVALENCE PASS: NO")
        print("DEPLOYMENT STATUS: physical biological-equivalence gate failed; holdout-guided retuning is not authorized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
