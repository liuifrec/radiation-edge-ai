"""Diagnose held-out biological fidelity of NASA BPS R1 v1.

This utility is intentionally post-hoc descriptive only. It does not retrain,
retune, recalibrate, or alter the frozen outer folds. It reads the out-of-fold
bag predictions and matched sham/exposed contrasts produced by
``train_pilot_v1_r1_mil.py`` and localizes where the first aggregate-label
baseline succeeds or fails.

Reports:
- overall and per-fold bag-level regression metrics;
- per-held-out-source bag-level metrics;
- matched radiation-delta fidelity by fold, branch, timepoint and source;
- the exact direction-disagreement contrasts;
- whether the model preserves the NASA temporal ordering of matched radiation
  effects within each Source Name x branch;
- a compact recommendation gate for whether it is scientifically reasonable to
  proceed directly to KL720 deployment or first develop a stronger FP32 R1.

No model selection or test-set tuning is performed. The diagnostic only reads
existing R1 v1 OOF outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_BAGS = 72
EXPECTED_DELTAS = 36
EXPECTED_SOURCES = 6


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


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


def fmt(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.4f}"


def bag_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    y = np.array([float(row["nasa_avg_nfoci"]) for row in rows], dtype=np.float64)
    p = np.array([float(row["predicted_bag_mean"]) for row in rows], dtype=np.float64)
    err = p - y
    return {
        "n": float(len(rows)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "pearson": correlation(p, y),
        "spearman": correlation(average_ranks(p), average_ranks(y)),
        "bias": float(np.mean(err)),
    }


def delta_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    nasa = np.array([float(row["nasa_delta"]) for row in rows], dtype=np.float64)
    pred = np.array([float(row["pred_delta"]) for row in rows], dtype=np.float64)
    direction = np.array([int(row["direction_agreement"]) for row in rows], dtype=np.float64)
    err = pred - nasa
    return {
        "n": float(len(rows)),
        "direction": float(np.mean(direction)),
        "delta_mae": float(np.mean(np.abs(err))),
        "delta_rmse": float(np.sqrt(np.mean(err * err))),
        "pearson": correlation(pred, nasa),
        "spearman": correlation(average_ranks(pred), average_ranks(nasa)),
        "bias": float(np.mean(err)),
    }


def print_bag_metric_line(label: str, rows: list[dict[str, str]]) -> dict[str, float]:
    m = bag_metrics(rows)
    print(
        f"{label:<28s} n={int(m['n']):2d} "
        f"MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} "
        f"Pearson={fmt(m['pearson'])} Spearman={fmt(m['spearman'])} bias={m['bias']:+.4f}"
    )
    return m


def print_delta_metric_line(label: str, rows: list[dict[str, str]]) -> dict[str, float]:
    m = delta_metrics(rows)
    print(
        f"{label:<28s} n={int(m['n']):2d} "
        f"dir={m['direction']:.3f} MAE={m['delta_mae']:.4f} "
        f"Pearson={fmt(m['pearson'])} Spearman={fmt(m['spearman'])} bias={m['bias']:+.4f}"
    )
    return m


def temporal_ordering(delta_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in delta_rows:
        grouped[(row["source_name"], row["branch"])].append(row)

    out: list[dict[str, Any]] = []
    expected_hours = {"4", "24", "48"}
    for (source, branch), rows in sorted(grouped.items()):
        by_hour = {str(row["timepoint_hr"]): row for row in rows}
        if set(by_hour) != expected_hours:
            raise RuntimeError(
                f"Unexpected timepoint coverage for {source}/{branch}: {sorted(by_hour)}"
            )
        nasa = [float(by_hour[h]["nasa_delta"]) for h in ("4", "24", "48")]
        pred = [float(by_hour[h]["pred_delta"]) for h in ("4", "24", "48")]
        nasa_order = tuple(np.argsort(-np.asarray(nasa), kind="mergesort").tolist())
        pred_order = tuple(np.argsort(-np.asarray(pred), kind="mergesort").tolist())
        out.append(
            {
                "source_name": source,
                "branch": branch,
                "nasa_4h": nasa[0],
                "nasa_24h": nasa[1],
                "nasa_48h": nasa[2],
                "pred_4h": pred[0],
                "pred_24h": pred[1],
                "pred_48h": pred[2],
                "nasa_peak_hour": ("4", "24", "48")[int(np.argmax(nasa))],
                "pred_peak_hour": ("4", "24", "48")[int(np.argmax(pred))],
                "peak_hour_agreement": int(int(np.argmax(nasa)) == int(np.argmax(pred))),
                "full_order_agreement": int(nasa_order == pred_order),
            }
        )
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    model_root = Path(
        os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models")
    )
    default_dir = model_root / "nasa_bps_53bp1" / "r1_v1_mil"
    parser.add_argument("--r1-dir", default=str(default_dir))
    args = parser.parse_args()

    r1_dir = Path(args.r1_dir).resolve()
    bag_path = r1_dir / "oof_bag_predictions.csv"
    delta_path = r1_dir / "oof_matched_radiation_deltas.csv"
    summary_path = r1_dir / "r1_v1_oof_summary.json"
    for path in (bag_path, delta_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    bag_rows = read_csv(bag_path)
    delta_rows = read_csv(delta_path)
    if len(bag_rows) != EXPECTED_BAGS:
        raise RuntimeError(f"OOF bags={len(bag_rows)}; expected {EXPECTED_BAGS}")
    if len(delta_rows) != EXPECTED_DELTAS:
        raise RuntimeError(f"Matched deltas={len(delta_rows)}; expected {EXPECTED_DELTAS}")
    sources = sorted({row["source_name"] for row in bag_rows})
    if len(sources) != EXPECTED_SOURCES:
        raise RuntimeError(f"OOF sources={len(sources)}; expected {EXPECTED_SOURCES}")

    print("Radiation Edge AI - NASA BPS R1 v1 held-out biological fidelity diagnostic")
    print(f"R1 directory: {r1_dir}")
    print(f"OOF bags: {len(bag_rows)}")
    print(f"Matched sham/exposed contrasts: {len(delta_rows)}")
    print("")

    report: dict[str, Any] = {
        "r1_dir": str(r1_dir),
        "n_oof_bags": len(bag_rows),
        "n_matched_contrasts": len(delta_rows),
    }

    print("[Bag-level held-out fidelity]")
    report["bag_overall"] = print_bag_metric_line("overall", bag_rows)
    by_fold: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in bag_rows:
        by_fold[row["fold"]].append(row)
    report["bag_by_fold"] = {}
    for fold in sorted(by_fold):
        report["bag_by_fold"][fold] = print_bag_metric_line(fold, by_fold[fold])
    print("")

    print("[Bag-level fidelity by held-out Source Name]")
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in bag_rows:
        by_source[row["source_name"]].append(row)
    report["bag_by_source"] = {}
    for source in sources:
        report["bag_by_source"][source] = print_bag_metric_line(source, by_source[source])
    print("")

    print("[Matched radiation-effect fidelity]")
    report["delta_overall"] = print_delta_metric_line("overall", delta_rows)

    bag_fold_by_source = {row["source_name"]: row["fold"] for row in bag_rows}
    delta_by_fold: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in delta_rows:
        delta_by_fold[bag_fold_by_source[row["source_name"]]].append(row)
    report["delta_by_fold"] = {}
    for fold in sorted(delta_by_fold):
        report["delta_by_fold"][fold] = print_delta_metric_line(fold, delta_by_fold[fold])
    print("")

    print("[Matched effects by branch and time]")
    report["delta_by_branch_time"] = {}
    for branch in ("Fe", "X-ray"):
        for hour in ("4", "24", "48"):
            subset = [
                row
                for row in delta_rows
                if row["branch"] == branch and str(row["timepoint_hr"]) == hour
            ]
            label = f"{branch} {hour}h"
            report["delta_by_branch_time"][label] = print_delta_metric_line(label, subset)
    print("")

    print("[Matched effects by Source Name]")
    report["delta_by_source"] = {}
    for source in sources:
        subset = [row for row in delta_rows if row["source_name"] == source]
        report["delta_by_source"][source] = print_delta_metric_line(source, subset)
    print("")

    disagreements = [row for row in delta_rows if int(row["direction_agreement"]) == 0]
    disagreements.sort(
        key=lambda row: abs(float(row["pred_delta"]) - float(row["nasa_delta"])),
        reverse=True,
    )
    print(f"[Direction disagreements] {len(disagreements)}/{len(delta_rows)}")
    for row in disagreements:
        print(
            f"  {row['source_name']:8s} {row['branch']:5s} {row['timepoint_hr']:>2s}h "
            f"NASA={float(row['nasa_delta']):+.3f} pred={float(row['pred_delta']):+.3f} "
            f"error={float(row['pred_delta']) - float(row['nasa_delta']):+.3f}"
        )
    print("")

    temporal = temporal_ordering(delta_rows)
    peak_fraction = float(np.mean([row["peak_hour_agreement"] for row in temporal]))
    order_fraction = float(np.mean([row["full_order_agreement"] for row in temporal]))
    report["temporal_ordering"] = {
        "n_source_branch_series": len(temporal),
        "peak_hour_agreement_fraction": peak_fraction,
        "full_4_24_48_order_agreement_fraction": order_fraction,
        "rows": temporal,
    }
    print("[Temporal radiation-effect ordering]")
    print(f"source x branch series: {len(temporal)}")
    print(f"NASA peak-time recovered: {peak_fraction:.3f}")
    print(f"full 4/24/48 ordering recovered: {order_fraction:.3f}")
    for row in temporal:
        print(
            f"  {row['source_name']:8s} {row['branch']:5s} "
            f"NASA peak={row['nasa_peak_hour']:>2s}h pred peak={row['pred_peak_hour']:>2s}h "
            f"peak_ok={row['peak_hour_agreement']} order_ok={row['full_order_agreement']}"
        )
    print("")

    # This is a conservative deployment-readiness gate, not a tuning objective.
    # A weak FP32 reference should not be sent to quantization merely because it compiles.
    overall_bag = report["bag_overall"]
    overall_delta = report["delta_overall"]
    fold_spearman = [m["spearman"] for m in report["bag_by_fold"].values()]
    weakest_fold_spearman = min(fold_spearman)
    direct_to_kl720 = bool(
        math.isfinite(overall_bag["spearman"])
        and overall_bag["spearman"] >= 0.60
        and overall_delta["direction"] >= 0.85
        and math.isfinite(overall_delta["spearman"])
        and overall_delta["spearman"] >= 0.70
        and weakest_fold_spearman >= 0.40
    )
    report["deployment_readiness"] = {
        "direct_to_kl720_recommended": direct_to_kl720,
        "gate_note": (
            "Conservative descriptive gate only; thresholds were not used for training or model selection. "
            "If false, preserve R1 v1 as the frozen baseline and develop a biologically stronger FP32 R1 v2 before quantization."
        ),
        "weakest_fold_spearman": weakest_fold_spearman,
    }

    out_path = r1_dir / "r1_v1_oof_fidelity_diagnostic.json"
    write_json(out_path, report)

    print("[Decision gate]")
    print(f"weakest fold Spearman: {fmt(weakest_fold_spearman)}")
    print(f"direct-to-KL720 recommended: {'YES' if direct_to_kl720 else 'NO'}")
    print(f"Diagnostic: {out_path}")
    print("")
    if direct_to_kl720:
        print("NEXT GATE: freeze/export the FP32 R1 v1 inference graph and begin ONNX/KL720 parity work.")
    else:
        print("NEXT GATE: preserve R1 v1 as the frozen baseline and design R1 v2 to improve held-out biological fidelity before quantization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
