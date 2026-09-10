"""Evaluate the frozen NASA BPS R1 v2 KL720 INT8 BIE on the frozen holdout.

This script is Python-3.7-compatible for the pinned Kneron Toolchain image. It
runs the frozen 2,058-nucleus deployment-equivalence holdout through both the
Kneron floating-point ONNX simulator and the fixed-point BIE simulator, then
applies the same seven predeclared biological gates used by the confirmatory
FP32 evaluation.

Important invariants:
- exact optimized ONNX and BIE identities are pinned;
- exact pre-holdout BIE freeze identity is pinned;
- exact QC1 holdout manifest identity is pinned;
- exact FP32 deployment-reference freeze identity is pinned;
- raw NASA phenotype metadata are NOT reopened;
- targets come only from the already-frozen FP32 confirmation outputs;
- no PTQ/calibration parameter is exposed or tunable here;
- a failed biological-equivalence result is reported, not repaired by tuning.

The per-nucleus output remains a latent continuous 53BP1 burden score, not an
individually supervised focus count.
"""

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import ktc
import numpy as np

FROZEN_OPTIMIZED_ONNX_SHA256 = (
    "c3a6aa5ed80280b0286f3b1aebf5f77f44dc60f569cc42fe2ab00ba7edde820c"
)
FROZEN_BIE_SHA256 = (
    "c52b8a78c595347667bad7a34950af92adbe734179b913646d06513a88fc1dd8"
)
FROZEN_BIE_FREEZE_SHA256 = (
    "63192365e77aa269d5ec792f2b184f8ada6f1603c451da47b1d1718ceb3dcc57"
)
FROZEN_PTQ_SUMMARY_SHA256 = (
    "98f0e096c059b8f7312bd9cb14c4733d5901183c0ffd712fd996f248db43fde1"
)
FROZEN_QC1_HOLDOUT_MANIFEST_SHA256 = (
    "f89962f96bec05930c07149717a90bd682016a2778bbfae6b827425159aa0643"
)
FROZEN_FP32_REFERENCE_FREEZE_SHA256 = (
    "abb58026d4f798554ef7e44346a989499242fcbd89d8c6629a1c4209d6c16ce3"
)
EXPECTED_TOOLCHAIN_VERSION = "kneron/toolchain:v0.33.1"
EXPECTED_PLATFORM = 720
EXPECTED_MODEL_ID = 32770
EXPECTED_MODEL_VERSION = "8b29"
EXPECTED_NUCLEI = 2058
EXPECTED_BAGS = 22
EXPECTED_CONTRASTS = 11
EXPECTED_SOURCE_NUCLEI = {"BALBCF2": 858, "C57BLF2": 600, "C57BLF3": 600}
EXPECTED_SOURCE_CONTRASTS = {"BALBCF2": 5, "C57BLF2": 3, "C57BLF3": 3}
EXPECTED_SOURCE_DIRECTION_MIN = {"BALBCF2": 4, "C57BLF2": 2, "C57BLF3": 2}
EXPECTED_HOLDOUT_STATUS = "FINAL_BLINDED_DO_NOT_JOIN_PHENOTYPES"
EXPOSED_DOSE = {"Fe": "0.82", "X-ray": "1.0"}
EXPECTED_INPUT_SHAPE = (1, 3, 256, 256)
CANVAS = 256
P_LOW = 1.0
P_HIGH = 99.5

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

FORBIDDEN_MANIFEST_TOKENS = (
    "nfoci",
    "phenotype",
    "reference",
    "ground_truth",
    "prediction",
    "predicted",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError("Expected JSON object: {}".format(path))
    return obj


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError("CSV is empty: {}".format(path))
    return rows


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("Refusing to write empty CSV: {}".format(path))
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def check_sha(path, expected, label):
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            "{} SHA256 mismatch: expected {}, got {}".format(label, expected, actual)
        )
    return actual


def toolchain_version():
    for candidate in (Path("/workspace/version.txt"), Path("/workspace/VERSION")):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace").strip()
    return None


def norm_number(value):
    x = float(str(value).strip())
    if x.is_integer():
        return "{:.1f}".format(x)
    return format(x, "g")


def norm_hour(value):
    x = float(str(value).strip())
    return str(int(x)) if x.is_integer() else format(x, "g")


def average_ranks(values):
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


def correlation(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def error_stats(reference, candidate):
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if ref.shape != cand.shape or ref.size == 0:
        raise RuntimeError("Cannot compare shapes {} and {}".format(ref.shape, cand.shape))
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(cand)):
        raise RuntimeError("Non-finite value in numerical comparison")
    diff = cand - ref
    absolute = np.abs(diff)
    return {
        "n": int(ref.size),
        "max_abs_error": float(absolute.max()),
        "mean_abs_error": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "pearson": correlation(ref, cand),
    }


def regression_metrics(rows):
    y = np.array([float(row["nasa_avg_nfoci"]) for row in rows], dtype=np.float64)
    p = np.array([float(row["predicted_bag_mean"]) for row in rows], dtype=np.float64)
    err = p - y
    return {
        "n_bags": int(len(rows)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "pearson": correlation(y, p),
        "spearman": correlation(average_ranks(y), average_ranks(p)),
    }


def delta_metrics(rows):
    nasa = np.array([float(row["nasa_delta"]) for row in rows], dtype=np.float64)
    pred = np.array([float(row["pred_delta"]) for row in rows], dtype=np.float64)
    return {
        "n_matched_contrasts": int(len(rows)),
        "direction_agreement_fraction": float(
            np.mean(np.sign(nasa) == np.sign(pred))
        ),
        "delta_mae": float(np.mean(np.abs(pred - nasa))),
        "delta_pearson": correlation(nasa, pred),
        "delta_spearman": correlation(average_ranks(nasa), average_ranks(pred)),
    }


def robust_normalize(image):
    if image.ndim != 2:
        raise RuntimeError("Expected 2D microscopy crop, got {}".format(image.shape))
    x = image.astype(np.float32, copy=False)
    lo = float(np.percentile(x, P_LOW))
    hi = float(np.percentile(x, P_HIGH))
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise RuntimeError("Non-finite normalization percentile")
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    np.clip(x, 0.0, 1.0, out=x)
    return x


def load_tensor(row, image_root):
    fitc_path = image_root / "fitc" / row["fitc_filename"]
    dapi_path = image_root / "dapi" / row["dapi_filename"]
    for path in (fitc_path, dapi_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    fitc = cv2.imread(str(fitc_path), cv2.IMREAD_UNCHANGED)
    dapi = cv2.imread(str(dapi_path), cv2.IMREAD_UNCHANGED)
    if fitc is None or dapi is None:
        raise RuntimeError("Could not read FITC/DAPI for {}".format(row["sample_id"]))
    if fitc.ndim != 2 or dapi.ndim != 2 or fitc.shape != dapi.shape:
        raise RuntimeError(
            "Invalid FITC/DAPI geometry for {}: {} / {}".format(
                row["sample_id"], getattr(fitc, "shape", None), getattr(dapi, "shape", None)
            )
        )
    h, w = fitc.shape
    if h > CANVAS or w > CANVAS:
        raise RuntimeError(
            "Native crop exceeds frozen 256x256 canvas for {}: {}x{}; resize forbidden".format(
                row["sample_id"], h, w
            )
        )
    out = np.zeros(EXPECTED_INPUT_SHAPE, dtype=np.float32)
    y0 = (CANVAS - h) // 2
    x0 = (CANVAS - w) // 2
    out[0, 0, y0 : y0 + h, x0 : x0 + w] = robust_normalize(fitc)
    out[0, 1, y0 : y0 + h, x0 : x0 + w] = robust_normalize(dapi)
    if not np.all(out[:, 2, :, :] == 0.0):
        raise RuntimeError("Third channel is not identically zero")
    return np.ascontiguousarray(out, dtype=np.float32)


def scalar_output(result, label):
    if not isinstance(result, (list, tuple)) or len(result) != 1:
        raise RuntimeError(
            "{} expected one output tensor, got {} len={}".format(
                label,
                type(result),
                len(result) if hasattr(result, "__len__") else "NA",
            )
        )
    array = np.asarray(result[0], dtype=np.float32).reshape(-1)
    if array.size != 1 or not np.isfinite(array).all():
        raise RuntimeError("{} invalid scalar output shape/value: {}".format(label, array.shape))
    return float(array[0]), tuple(np.asarray(result[0]).shape)


def validate_holdout_manifest(rows):
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
        raise RuntimeError("QC1 manifest missing columns: {}".format(missing))
    forbidden = sorted(
        column
        for column in rows[0]
        if any(token in column.lower() for token in FORBIDDEN_MANIFEST_TOKENS)
    )
    if forbidden:
        raise RuntimeError("QC1 manifest contains outcome columns: {}".format(forbidden))
    if len(rows) != EXPECTED_NUCLEI:
        raise RuntimeError("Holdout nuclei={}; expected {}".format(len(rows), EXPECTED_NUCLEI))
    if len({row["sample_id"] for row in rows}) != EXPECTED_NUCLEI:
        raise RuntimeError("Duplicate sample_id in QC1 holdout manifest")
    if len({row["nucleus_key"] for row in rows}) != EXPECTED_NUCLEI:
        raise RuntimeError("Duplicate nucleus_key in QC1 holdout manifest")
    if {row["holdout_status"] for row in rows} != {EXPECTED_HOLDOUT_STATUS}:
        raise RuntimeError("Unexpected holdout_status in QC1 manifest")
    source_counts = Counter(row["source_name"] for row in rows)
    if dict(source_counts) != EXPECTED_SOURCE_NUCLEI:
        raise RuntimeError("Unexpected holdout source counts: {}".format(dict(source_counts)))
    if {row["sex"] for row in rows} != {"Female"}:
        raise RuntimeError("Frozen all-female holdout structure changed")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sample_name"]].append(row)
    if len(grouped) != EXPECTED_BAGS:
        raise RuntimeError("Holdout bags={}; expected 22".format(len(grouped)))
    return grouped


def verify_bie_freeze(path):
    check_sha(path, FROZEN_BIE_FREEZE_SHA256, "BIE deployment freeze")
    obj = read_json(path)
    checks = {
        "status": "FROZEN_KL720_INT8_BIE_DEPLOYMENT_CANDIDATE_BEFORE_HOLDOUT",
        "optimized_onnx_sha256": FROZEN_OPTIMIZED_ONNX_SHA256,
        "bie_sha256": FROZEN_BIE_SHA256,
        "ptq_summary_sha256": FROZEN_PTQ_SUMMARY_SHA256,
        "toolchain_version": EXPECTED_TOOLCHAIN_VERSION,
        "platform": "720",
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "final_holdout_used_for_calibration": False,
        "final_holdout_read_before_this_freeze": False,
        "ptq_tuning_after_holdout_authorized": False,
        "authorized_next_stage": "FIXED_BIE_DEPLOYMENT_EQUIVALENCE_HOLDOUT",
    }
    for key, expected in checks.items():
        if obj.get(key) != expected:
            raise RuntimeError(
                "BIE freeze invariant mismatch for {}: expected {!r}, got {!r}".format(
                    key, expected, obj.get(key)
                )
            )
    return obj


def verify_fp32_reference(freeze_path, confirmation_dir):
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
        raise RuntimeError("FP32 deployment reference did not pass all seven gates")

    reference_files = freeze.get("reference_files") or {}
    required = (
        "final_holdout_confirmation_summary.json",
        "FINAL_HOLDOUT_CONFIRMATION_COMPLETE.json",
        "FINAL_HOLDOUT_RECOVERY_COMPLETE.json",
        "final_holdout_bag_predictions.csv",
        "final_holdout_nucleus_latent_burden.csv",
        "final_holdout_matched_radiation_deltas.csv",
        "final_holdout_direction_summary.csv",
        "final_holdout_peak_time_recovery.csv",
        "final_holdout_source_residuals.csv",
    )
    for name in required:
        meta = reference_files.get(name)
        path = confirmation_dir / name
        if not isinstance(meta, dict) or not path.is_file():
            raise RuntimeError("Frozen FP32 reference file missing: {}".format(name))
        if sha256_file(path) != meta.get("sha256"):
            raise RuntimeError("Frozen FP32 reference file changed: {}".format(name))
    return freeze


def build_bag_rows(nucleus_rows, reference_bags):
    grouped = defaultdict(list)
    for row in nucleus_rows:
        grouped[row["sample_name"]].append(row)
    if len(grouped) != EXPECTED_BAGS:
        raise RuntimeError("BIE result bag count mismatch")
    result = []
    for sample_name, subset in sorted(grouped.items()):
        ref = reference_bags.get(sample_name)
        if ref is None:
            raise RuntimeError("Missing frozen FP32 bag reference: {}".format(sample_name))
        bie_mean = float(np.mean([float(row["bie_burden"]) for row in subset]))
        kfp_mean = float(np.mean([float(row["kneron_fp_burden"]) for row in subset]))
        out = {
            "sample_name": sample_name,
            "source_name": ref["source_name"],
            "strain": ref["strain"],
            "sex": ref["sex"],
            "particle_type": ref["particle_type"],
            "dose_Gy": norm_number(ref["dose_Gy"]),
            "hr_post_exposure": norm_hour(ref["hr_post_exposure"]),
            "n_holdout_nuclei": len(subset),
            "predicted_bag_mean": bie_mean,
            "kneron_fp_bag_mean": kfp_mean,
            "fp32_reference_bag_mean": float(ref["predicted_bag_mean"]),
            "nasa_avg_nfoci": float(ref["nasa_avg_nfoci"]),
            "nasa_num_nuc": int(float(ref["nasa_num_nuc"])),
        }
        out["bie_minus_fp32_bag"] = bie_mean - out["fp32_reference_bag_mean"]
        out["bie_residual"] = bie_mean - out["nasa_avg_nfoci"]
        result.append(out)
    return result


def build_deltas(bag_rows):
    lookup = {
        (
            str(row["source_name"]),
            str(row["particle_type"]),
            norm_number(row["dose_Gy"]),
            norm_hour(row["hr_post_exposure"]),
        ): row
        for row in bag_rows
    }
    rows = []
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
            "BIE matched deltas do not reproduce frozen 11-contrast structure: {}".format(dict(counts))
        )
    return rows


def build_peaks(delta_rows):
    grouped = defaultdict(list)
    for row in delta_rows:
        grouped[(row["source_name"], row["branch"])].append(row)
    rows = []
    for (source, branch), subset in sorted(grouped.items()):
        hours = {str(row["timepoint_hr"]) for row in subset}
        if hours != {"4", "24", "48"}:
            continue
        subset = sorted(subset, key=lambda row: ("4", "24", "48").index(str(row["timepoint_hr"])))
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
        raise RuntimeError("Expected exactly three complete source x branch peak series; got {}".format(len(rows)))
    return rows


def evaluate_gate(bag_rows, delta_rows, peaks):
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
        "overall": "{}/11".format(overall_ok),
        "4h": "{}/4".format(four_ok),
        "24+48h": "{}/7".format(late_ok),
        "by_source": {
            source: "{}/{}".format(source_counts[source], EXPECTED_SOURCE_CONTRASTS[source])
            for source in EXPECTED_SOURCE_NUCLEI
        },
    }
    return {
        "bag_metrics": bag,
        "delta_metrics": delta,
        "direction_counts": directions,
        "peak_time_recovery": "{}/3".format(peak_ok),
        "checks": checks,
        "all_pass": all(checks.values()),
        "exact_fp32_direction_signature_reproduced": directions == EXPECTED_FP32_DIRECTION_COUNTS,
        "exact_fp32_peak_signature_reproduced": "{}/3".format(peak_ok) == EXPECTED_FP32_PEAK,
    }


def run_preflight(rows, image_root, onnx_path, bie_path):
    row = sorted(rows, key=lambda item: item["sample_id"])[0]
    tensor = load_tensor(row, image_root)
    started = time.perf_counter()
    fp = ktc.kneron_inference(
        [tensor], onnx_file=str(onnx_path), input_names=["input"]
    )
    fp_ms = (time.perf_counter() - started) * 1000.0
    _fp_value, fp_shape = scalar_output(fp, "Kneron floating ONNX")
    started = time.perf_counter()
    fixed = ktc.kneron_inference(
        [tensor],
        bie_file=str(bie_path),
        input_names=["input"],
        platform=EXPECTED_PLATFORM,
    )
    fixed_ms = (time.perf_counter() - started) * 1000.0
    _fixed_value, fixed_shape = scalar_output(fixed, "Kneron fixed BIE")
    print("BIE HOLDOUT PREFLIGHT COMPLETE: YES")
    print("sample tested: {}".format(row["sample_id"]))
    print("input shape/dtype: {} / {}".format(tensor.shape, tensor.dtype))
    print("Kneron FP output shape: {}".format(fp_shape))
    print("Kneron BIE output shape: {}".format(fixed_shape))
    print("Kneron FP simulator time: {:.1f} ms".format(fp_ms))
    print("Kneron BIE simulator time: {:.1f} ms".format(fixed_ms))
    print("outcome/reference files read by preflight: NO")
    print("PTQ/BIE parameters changed: NO")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--bie", required=True)
    parser.add_argument("--bie-freeze", required=True)
    parser.add_argument("--holdout-manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--fp32-reference-freeze", required=True)
    parser.add_argument("--confirmation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    onnx_path = Path(args.onnx).resolve()
    bie_path = Path(args.bie).resolve()
    bie_freeze_path = Path(args.bie_freeze).resolve()
    manifest_path = Path(args.holdout_manifest).resolve()
    image_root = Path(args.image_root).resolve()
    fp32_freeze_path = Path(args.fp32_reference_freeze).resolve()
    confirmation_dir = Path(args.confirmation_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    check_sha(onnx_path, FROZEN_OPTIMIZED_ONNX_SHA256, "Optimized ONNX")
    check_sha(bie_path, FROZEN_BIE_SHA256, "INT8 BIE")
    verify_bie_freeze(bie_freeze_path)
    check_sha(manifest_path, FROZEN_QC1_HOLDOUT_MANIFEST_SHA256, "QC1 holdout manifest")
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    version = toolchain_version()
    if version != EXPECTED_TOOLCHAIN_VERSION:
        raise RuntimeError(
            "Toolchain version mismatch: expected {!r}, got {!r}".format(
                EXPECTED_TOOLCHAIN_VERSION, version
            )
        )

    rows = read_csv(manifest_path)
    grouped = validate_holdout_manifest(rows)
    cv2.setNumThreads(0)

    print("Radiation Edge AI - NASA BPS R1 v2 KL720 BIE holdout equivalence")
    print("optimized ONNX SHA256: {}".format(FROZEN_OPTIMIZED_ONNX_SHA256))
    print("BIE SHA256: {}".format(FROZEN_BIE_SHA256))
    print("BIE freeze SHA256: {}".format(FROZEN_BIE_FREEZE_SHA256))
    print("QC1 holdout manifest SHA256: {}".format(FROZEN_QC1_HOLDOUT_MANIFEST_SHA256))
    print("toolchain version: {}".format(version))
    print("holdout nuclei/bags: {}/{}".format(len(rows), len(grouped)))
    print("preprocessing: p1/p99.5 FITC+DAPI; no resize; center pad 256; zero third channel")
    print("MASK used as model input: NO")
    print("PTQ tuning after holdout authorized: NO")

    if args.preflight_only:
        run_preflight(rows, image_root, onnx_path, bie_path)
        return 0

    if not confirmation_dir.is_dir():
        raise FileNotFoundError(confirmation_dir)
    verify_fp32_reference(fp32_freeze_path, confirmation_dir)
    reference_nucleus_rows = read_csv(
        confirmation_dir / "final_holdout_nucleus_latent_burden.csv"
    )
    reference_bag_rows = read_csv(
        confirmation_dir / "final_holdout_bag_predictions.csv"
    )
    if len(reference_nucleus_rows) != EXPECTED_NUCLEI:
        raise RuntimeError("Frozen FP32 nucleus reference count mismatch")
    if len(reference_bag_rows) != EXPECTED_BAGS:
        raise RuntimeError("Frozen FP32 bag reference count mismatch")
    reference_nuclei = {row["sample_id"]: row for row in reference_nucleus_rows}
    reference_bags = {row["sample_name"]: row for row in reference_bag_rows}
    if len(reference_nuclei) != EXPECTED_NUCLEI or len(reference_bags) != EXPECTED_BAGS:
        raise RuntimeError("Duplicate identifier in frozen FP32 reference outputs")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "bie_holdout_equivalence_summary.json"
    if summary_path.exists():
        raise RuntimeError("BIE holdout result already exists; refusing overwrite: {}".format(summary_path))

    nucleus_rows = []
    kfp_values = []
    bie_values = []
    fp32_values = []
    started_all = time.perf_counter()
    total_fp_ms = 0.0
    total_bie_ms = 0.0

    ordered_rows = sorted(rows, key=lambda row: (row["sample_name"], row["sample_id"]))
    for index, row in enumerate(ordered_rows, start=1):
        ref = reference_nuclei.get(row["sample_id"])
        if ref is None:
            raise RuntimeError("Sample missing frozen FP32 nucleus reference: {}".format(row["sample_id"]))
        tensor = load_tensor(row, image_root)

        t0 = time.perf_counter()
        fp_result = ktc.kneron_inference(
            [tensor], onnx_file=str(onnx_path), input_names=["input"]
        )
        fp_ms = (time.perf_counter() - t0) * 1000.0
        kfp, _fp_shape = scalar_output(fp_result, "Kneron floating ONNX")

        t0 = time.perf_counter()
        bie_result = ktc.kneron_inference(
            [tensor],
            bie_file=str(bie_path),
            input_names=["input"],
            platform=EXPECTED_PLATFORM,
        )
        bie_ms = (time.perf_counter() - t0) * 1000.0
        bie, _bie_shape = scalar_output(bie_result, "Kneron fixed BIE")

        fp32 = float(ref["predicted_continuous_burden"])
        if not math.isfinite(fp32):
            raise RuntimeError("Non-finite frozen FP32 burden: {}".format(row["sample_id"]))
        nucleus_rows.append(
            {
                "sample_id": row["sample_id"],
                "nucleus_key": row["nucleus_key"],
                "sample_name": row["sample_name"],
                "source_name": row["source_name"],
                "strain": row["strain"],
                "sex": row["sex"],
                "particle_type": row["particle_type"],
                "dose_Gy": norm_number(row["dose_Gy"]),
                "hr_post_exposure": norm_hour(row["hr_post_exposure"]),
                "fp32_reference_burden": fp32,
                "kneron_fp_burden": kfp,
                "bie_burden": bie,
                "kneron_fp_minus_fp32": kfp - fp32,
                "bie_minus_kneron_fp": bie - kfp,
                "bie_minus_fp32": bie - fp32,
                "kneron_fp_ms": fp_ms,
                "bie_ms": bie_ms,
                "label_semantics": "latent_continuous_burden_not_individually_supervised",
            }
        )
        fp32_values.append(fp32)
        kfp_values.append(kfp)
        bie_values.append(bie)
        total_fp_ms += fp_ms
        total_bie_ms += bie_ms
        if index % 50 == 0 or index == EXPECTED_NUCLEI:
            print("holdout inference: {}/{}".format(index, EXPECTED_NUCLEI))

    if len(nucleus_rows) != EXPECTED_NUCLEI:
        raise RuntimeError("Incomplete BIE holdout inference")

    bag_rows = build_bag_rows(nucleus_rows, reference_bags)
    delta_rows = build_deltas(bag_rows)
    peak_rows = build_peaks(delta_rows)
    gate = evaluate_gate(bag_rows, delta_rows, peak_rows)

    fp32_vs_kfp = error_stats(fp32_values, kfp_values)
    kfp_vs_bie = error_stats(kfp_values, bie_values)
    fp32_vs_bie = error_stats(fp32_values, bie_values)
    fp32_bag = [float(row["fp32_reference_bag_mean"]) for row in bag_rows]
    kfp_bag = [float(row["kneron_fp_bag_mean"]) for row in bag_rows]
    bie_bag = [float(row["predicted_bag_mean"]) for row in bag_rows]
    bag_fp32_vs_kfp = error_stats(fp32_bag, kfp_bag)
    bag_kfp_vs_bie = error_stats(kfp_bag, bie_bag)
    bag_fp32_vs_bie = error_stats(fp32_bag, bie_bag)

    nucleus_path = output_dir / "bie_holdout_nucleus_predictions.csv"
    bag_path = output_dir / "bie_holdout_bag_predictions.csv"
    delta_path = output_dir / "bie_holdout_matched_radiation_deltas.csv"
    peak_path = output_dir / "bie_holdout_peak_time_recovery.csv"
    write_csv(nucleus_path, nucleus_rows)
    write_csv(bag_path, bag_rows)
    write_csv(delta_path, delta_rows)
    write_csv(peak_path, peak_rows)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_FROZEN_KL720_BIE_DEPLOYMENT_EQUIVALENCE_HOLDOUT",
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "optimized_onnx_sha256": FROZEN_OPTIMIZED_ONNX_SHA256,
        "bie_sha256": FROZEN_BIE_SHA256,
        "bie_deployment_freeze_sha256": FROZEN_BIE_FREEZE_SHA256,
        "ptq_summary_sha256": FROZEN_PTQ_SUMMARY_SHA256,
        "qc1_holdout_manifest_sha256": FROZEN_QC1_HOLDOUT_MANIFEST_SHA256,
        "fp32_deployment_reference_freeze_sha256": FROZEN_FP32_REFERENCE_FREEZE_SHA256,
        "toolchain_version": version,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "nuclei": EXPECTED_NUCLEI,
        "bags": EXPECTED_BAGS,
        "matched_contrasts": EXPECTED_CONTRASTS,
        "output_semantics": "scalar latent continuous 53BP1 burden; not per-nucleus focus count",
        "preprocessing": "p1/p99.5 FITC+DAPI native crop normalization; no resize; center pad 256; zero third channel",
        "mask_used_as_model_input": False,
        "raw_phenotype_table_read": False,
        "targets_source": "immutable FP32 confirmatory reference outputs frozen before deployment conversion",
        "protocol_deviation_carried_forward": "documented non-outcome metadata-normalization implementation recovery",
        "ptq_tuning_after_holdout": False,
        "post_holdout_ptq_tuning_authorized": False,
        "nucleus_numerical_comparisons_descriptive_only": {
            "frozen_fp32_vs_kneron_floating": fp32_vs_kfp,
            "kneron_floating_vs_bie": kfp_vs_bie,
            "frozen_fp32_vs_bie": fp32_vs_bie,
        },
        "bag_numerical_comparisons_descriptive_only": {
            "frozen_fp32_vs_kneron_floating": bag_fp32_vs_kfp,
            "kneron_floating_vs_bie": bag_kfp_vs_bie,
            "frozen_fp32_vs_bie": bag_fp32_vs_bie,
        },
        "biological_gate": gate,
        "all_seven_original_biological_gates_pass": bool(gate["all_pass"]),
        "exact_fp32_biological_signature_reproduced": bool(
            gate["exact_fp32_direction_signature_reproduced"]
            and gate["exact_fp32_peak_signature_reproduced"]
        ),
        "mean_kneron_fp_simulator_ms_per_nucleus": total_fp_ms / EXPECTED_NUCLEI,
        "mean_bie_simulator_ms_per_nucleus": total_bie_ms / EXPECTED_NUCLEI,
        "elapsed_seconds": time.perf_counter() - started_all,
        "outputs": {
            nucleus_path.name: {"sha256": sha256_file(nucleus_path), "size_bytes": nucleus_path.stat().st_size},
            bag_path.name: {"sha256": sha256_file(bag_path), "size_bytes": bag_path.stat().st_size},
            delta_path.name: {"sha256": sha256_file(delta_path), "size_bytes": delta_path.stat().st_size},
            peak_path.name: {"sha256": sha256_file(peak_path), "size_bytes": peak_path.stat().st_size},
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    bm = gate["bag_metrics"]
    dm = gate["delta_metrics"]
    print("")
    print("[KL720 INT8 BIE biological deployment-equivalence result]")
    print(
        "bag MAE/RMSE/Pearson/Spearman: {:.4f} / {:.4f} / {:.4f} / {:.4f}".format(
            bm["mae"], bm["rmse"], bm["pearson"], bm["spearman"]
        )
    )
    print(
        "delta MAE/Pearson/Spearman: {:.4f} / {:.4f} / {:.4f}".format(
            dm["delta_mae"], dm["delta_pearson"], dm["delta_spearman"]
        )
    )
    print("direction overall: {}".format(gate["direction_counts"]["overall"]))
    print("direction 4 h: {}".format(gate["direction_counts"]["4h"]))
    print("direction 24+48 h: {}".format(gate["direction_counts"]["24+48h"]))
    print("direction by Source Name: {}".format(gate["direction_counts"]["by_source"]))
    print("peak-time recovery: {}".format(gate["peak_time_recovery"]))
    print("")
    for name, passed in gate["checks"].items():
        print("GATE {}: {}".format(name, "PASS" if passed else "FAIL"))
    print(
        "ALL SEVEN ORIGINAL BIOLOGICAL GATES PASS: {}".format(
            "YES" if gate["all_pass"] else "NO"
        )
    )
    print(
        "EXACT FP32 DIRECTION/PEAK SIGNATURE REPRODUCED: {}".format(
            "YES"
            if summary["exact_fp32_biological_signature_reproduced"]
            else "NO"
        )
    )
    print("raw phenotype table read: NO")
    print("PTQ tuning after holdout: NO")
    print("summary: {}".format(summary_path))
    print("summary SHA256: {}".format(sha256_file(summary_path)))
    if gate["all_pass"]:
        print("NASA BPS R1 V2 KL720 INT8 BIE BIOLOGICAL EQUIVALENCE PASS: YES")
        print("NEXT GATE: compile this exact frozen BIE to NEF; no PTQ retuning.")
    else:
        print("NASA BPS R1 V2 KL720 INT8 BIE BIOLOGICAL EQUIVALENCE PASS: NO")
        print("DEPLOYMENT STATUS: BIE biological-equivalence gate failed; PTQ retuning from this holdout is not authorized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
