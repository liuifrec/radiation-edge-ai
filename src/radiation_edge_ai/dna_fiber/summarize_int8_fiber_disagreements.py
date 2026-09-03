"""Summarize already-generated INT8 fiber disagreement diagnostics.

This helper is intentionally cheap: it reads the CSV files written by
``diagnose_int8_fiber_disagreements.py`` and prints, for every available PTQ
variant, both FP512 fibers that were missed and BIE-only fibers, including
human-grader support. It performs no Kneron inference and no image processing.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def support_label(count: int) -> str:
    if count >= 3:
        return "human-supported"
    if count >= 1:
        return "ambiguous"
    return "no-human-match"


def main() -> int:
    parser = argparse.ArgumentParser()
    default_model_root = Path(
        os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models")
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--model-root", default=str(default_model_root))
    args = parser.parse_args()

    diagnostic_root = (
        Path(args.model_root)
        / "dnai"
        / "unet_mobileone_s1"
        / "kneron"
        / "int8_fiber_diagnostics"
        / args.sample_id
    )
    fp_csv = diagnostic_root / "fp_fiber_retention_and_human_support.csv"
    extra_csv = diagnostic_root / "bie_only_fibers_and_human_support.csv"
    if not fp_csv.is_file():
        raise FileNotFoundError(fp_csv)

    with fp_csv.open(newline="", encoding="utf-8") as handle:
        fp_rows = list(csv.DictReader(handle))
    if not fp_rows:
        raise RuntimeError(f"No FP fiber rows in {fp_csv}")

    retained_columns = [
        key for key in fp_rows[0].keys() if key.endswith("_retained")
    ]
    versions = [key[: -len("_retained")] for key in retained_columns]
    if not versions:
        raise RuntimeError("No PTQ retention columns found")

    extra_rows = []
    if extra_csv.is_file():
        text = extra_csv.read_text(encoding="utf-8").strip()
        if text and not text.startswith("No BIE-only fibers"):
            with extra_csv.open(newline="", encoding="utf-8") as handle:
                extra_rows = list(csv.DictReader(handle))

    print("Radiation Edge AI - concise INT8 fiber disagreement summary")
    print(f"Sample: {args.sample_id}")
    print(f"FP512 valid fibers: {len(fp_rows)}")
    print("Variants: " + ", ".join(versions))

    for version in versions:
        missed = [row for row in fp_rows if not as_bool(row[f"{version}_retained"])]
        extras = [row for row in extra_rows if row.get("version") == version]
        print("")
        print(f"[{version}] FP-missed={len(missed)} BIE-only={len(extras)}")
        if missed:
            print("FP-missed fibers:")
            for row in missed:
                support = int(row["human_support_count"])
                print(
                    f"  FP#{int(row['fp_fiber_index']):02d} "
                    f"length={float(row['length_um']):.3f}um "
                    f"ratio={float(row['ratio']):.4f} "
                    f"humans={support}/4 ({support_label(support)})"
                )
        else:
            print("FP-missed fibers: none")

        if extras:
            print("BIE-only fibers:")
            for row in extras:
                support = int(row["human_support_count"])
                print(
                    f"  BIE#{int(row['bie_fiber_index']):02d} "
                    f"length={float(row['length_um']):.3f}um "
                    f"ratio={float(row['ratio']):.4f} "
                    f"humans={support}/4 ({support_label(support)})"
                )
        else:
            print("BIE-only fibers: none")

    print("")
    print(
        "Interpretation: prefer a variant that preserves 3-4/4-supported FP fibers "
        "without creating 0/4-supported BIE-only objects."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
