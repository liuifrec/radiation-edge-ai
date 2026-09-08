"""Plan a leakage-safe NASA BPS R1 v2 expansion before any new training.

R1 v1 used six Source Names, 12 branch/time/dose conditions, and 30 nucleus
crops per Source Name x condition bag.  Its held-out diagnostic showed that the
acute 4 h radiation response was detected reliably, but absolute calibration
and late 24/48 h effects were not yet robust enough for KL720 quantization.

Because the v1 outer folds have now been inspected, they should be treated as a
development benchmark rather than an untouched final test.  This metadata-only
utility asks whether the remaining BPS Source Names can provide a new untouched
holdout and whether larger bags (for example 100 nuclei per sample) are feasible.

It reports:
- the six v1 development Source Names and all remaining untouched Source Names;
- coverage of the exact 12 frozen v1 conditions in every Source Name;
- completeness of the six biologically matched sham/exposed contrasts
  (Fe and X-ray at 4, 24, and 48 h);
- available FITC image counts per Source Name x condition and the maximum common
  deterministic bag size that could be supported without replacement;
- extra low-dose conditions outside the v1 condition set that may be reserved
  for later generalization tests.

No images are downloaded, no model is trained, and no split is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

FROZEN_MANIFEST_SHA256 = "41ff2e4d64577f17c976dea5bbc17221fe63be9e2e7e2291309b520027fbe725"


def sha256_file(path: Path) -> str:
    import hashlib

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
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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
    if not math.isfinite(x):
        raise ValueError(f"Non-finite number: {value!r}")
    if x.is_integer():
        return f"{x:.1f}"
    return format(x, "g")


def norm_hour(value: str) -> str:
    x = float(str(value).strip())
    if not math.isfinite(x):
        raise ValueError(f"Non-finite hour: {value!r}")
    return str(int(x)) if x.is_integer() else format(x, "g")


def condition_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row["particle_type"]).strip(),
        norm_number(row["dose_Gy"]),
        norm_hour(row["hr_post_exposure"]),
    )


def condition_label(key: tuple[str, str, str]) -> str:
    branch, dose, hour = key
    return f"{branch}|{dose}Gy|{hour}h"


def sort_condition(key: tuple[str, str, str]) -> tuple[str, float, float]:
    return (key[0], float(key[1]), float(key[2]))


def matched_contrast_specs(
    pilot_conditions: set[tuple[str, str, str]],
) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    exposed = {"Fe": "0.82", "X-ray": "1.0"}
    for branch, dose in exposed.items():
        for hour in ("4", "24", "48"):
            sham = (branch, "0.0", hour)
            exp = (branch, dose, hour)
            if sham not in pilot_conditions or exp not in pilot_conditions:
                raise RuntimeError(
                    f"Frozen pilot condition set missing required pair: {sham} / {exp}"
                )
            specs.append(
                {
                    "branch": branch,
                    "hour": hour,
                    "sham": condition_label(sham),
                    "exposed": condition_label(exp),
                }
            )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"
    parser.add_argument(
        "--crosswalk",
        default=str(metadata_root / "fitc_53bp1_osd366_crosswalk.csv"),
    )
    parser.add_argument(
        "--pilot-manifest",
        default=str(metadata_root / "pilot_v1" / "pilot_v1_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(metadata_root / "r1_v2_planning"),
    )
    args = parser.parse_args()

    crosswalk_path = Path(args.crosswalk).resolve()
    manifest_path = Path(args.pilot_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (crosswalk_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != FROZEN_MANIFEST_SHA256:
        raise RuntimeError(
            "Frozen pilot manifest SHA256 mismatch: "
            f"expected {FROZEN_MANIFEST_SHA256}, got {manifest_sha}"
        )

    crosswalk = read_csv(crosswalk_path)
    manifest = read_csv(manifest_path)

    required_crosswalk = {
        "filename",
        "dose_Gy",
        "particle_type",
        "hr_post_exposure",
        "osd_source_name",
        "osd_sample_name",
        "osd_strain",
        "osd_sex",
    }
    missing = sorted(required_crosswalk - set(crosswalk[0]))
    if missing:
        raise RuntimeError(f"Crosswalk missing required columns: {missing}")

    required_manifest = {
        "source_name",
        "sample_name",
        "dose_Gy",
        "particle_type",
        "hr_post_exposure",
    }
    missing = sorted(required_manifest - set(manifest[0]))
    if missing:
        raise RuntimeError(f"Pilot manifest missing required columns: {missing}")

    pilot_sources = sorted({row["source_name"] for row in manifest})
    pilot_conditions = {condition_key(row) for row in manifest}
    if len(pilot_sources) != 6:
        raise RuntimeError(f"Frozen pilot Source Names={len(pilot_sources)}; expected 6")
    if len(pilot_conditions) != 12:
        raise RuntimeError(f"Frozen pilot conditions={len(pilot_conditions)}; expected 12")

    all_sources = sorted({row["osd_source_name"] for row in crosswalk if row["osd_source_name"]})
    untouched_sources = sorted(set(all_sources) - set(pilot_sources))

    by_source_condition: dict[
        tuple[str, tuple[str, str, str]], list[dict[str, str]]
    ] = defaultdict(list)
    for row in crosswalk:
        source = row["osd_source_name"]
        if not source:
            continue
        by_source_condition[(source, condition_key(row))].append(row)

    all_conditions = sorted(
        {condition_key(row) for row in crosswalk}, key=sort_condition
    )
    extra_conditions = [key for key in all_conditions if key not in pilot_conditions]
    specs = matched_contrast_specs(pilot_conditions)

    source_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []

    for source in all_sources:
        source_subset = [row for row in crosswalk if row["osd_source_name"] == source]
        strains = sorted({row["osd_strain"] for row in source_subset if row["osd_strain"]})
        sexes = sorted({row["osd_sex"] for row in source_subset if row["osd_sex"]})

        counts: dict[tuple[str, str, str], int] = {}
        sample_sets: dict[tuple[str, str, str], set[str]] = {}
        for cond in pilot_conditions:
            rows = by_source_condition.get((source, cond), [])
            counts[cond] = len(rows)
            sample_sets[cond] = {row["osd_sample_name"] for row in rows if row["osd_sample_name"]}
            coverage_rows.append(
                {
                    "source_name": source,
                    "role": "v1_development" if source in pilot_sources else "untouched",
                    "strain": ";".join(strains),
                    "sex": ";".join(sexes),
                    "particle_type": cond[0],
                    "dose_Gy": cond[1],
                    "hr_post_exposure": cond[2],
                    "n_fitc_images": len(rows),
                    "n_sample_names": len(sample_sets[cond]),
                    "sample_names": ";".join(sorted(sample_sets[cond])),
                    "condition_present": int(bool(rows)),
                }
            )

        present = [cond for cond in pilot_conditions if counts[cond] > 0]
        missing_conditions = [cond for cond in pilot_conditions if counts[cond] == 0]
        ambiguous_cells = [
            cond for cond in pilot_conditions if counts[cond] > 0 and len(sample_sets[cond]) != 1
        ]
        available_counts = [counts[cond] for cond in pilot_conditions if counts[cond] > 0]

        complete_contrasts = 0
        for spec in specs:
            sham_key = next(k for k in pilot_conditions if condition_label(k) == spec["sham"])
            exp_key = next(k for k in pilot_conditions if condition_label(k) == spec["exposed"])
            sham_n = counts[sham_key]
            exp_n = counts[exp_key]
            complete = sham_n > 0 and exp_n > 0
            complete_contrasts += int(complete)
            contrast_rows.append(
                {
                    "source_name": source,
                    "role": "v1_development" if source in pilot_sources else "untouched",
                    "strain": ";".join(strains),
                    "sex": ";".join(sexes),
                    "branch": spec["branch"],
                    "timepoint_hr": spec["hour"],
                    "sham_condition": spec["sham"],
                    "exposed_condition": spec["exposed"],
                    "sham_images": sham_n,
                    "exposed_images": exp_n,
                    "matched_contrast_complete": int(complete),
                    "max_balanced_images_per_arm": min(sham_n, exp_n) if complete else 0,
                }
            )

        extra_present = [
            cond for cond in extra_conditions if by_source_condition.get((source, cond), [])
        ]
        source_rows.append(
            {
                "source_name": source,
                "role": "v1_development" if source in pilot_sources else "untouched",
                "strain": ";".join(strains),
                "sex": ";".join(sexes),
                "pilot_conditions_present": len(present),
                "pilot_conditions_total": len(pilot_conditions),
                "complete_12_condition_panel": int(len(present) == len(pilot_conditions)),
                "matched_contrasts_complete": complete_contrasts,
                "matched_contrasts_total": len(specs),
                "min_images_present_cell": min(available_counts) if available_counts else 0,
                "median_images_present_cell": median(available_counts) if available_counts else 0,
                "max_images_present_cell": max(available_counts) if available_counts else 0,
                "max_common_bag_size_all_12": (
                    min(counts.values()) if all(counts[cond] > 0 for cond in pilot_conditions) else 0
                ),
                "ambiguous_source_condition_cells": len(ambiguous_cells),
                "missing_pilot_conditions": ";".join(
                    condition_label(cond) for cond in sorted(missing_conditions, key=sort_condition)
                ),
                "extra_conditions_present": ";".join(
                    condition_label(cond) for cond in sorted(extra_present, key=sort_condition)
                ),
            }
        )

    development_rows = [row for row in source_rows if row["role"] == "v1_development"]
    untouched_rows = [row for row in source_rows if row["role"] == "untouched"]

    if any(int(row["ambiguous_source_condition_cells"]) > 0 for row in source_rows):
        ambiguous_warning = True
    else:
        ambiguous_warning = False

    dev_common_bag = min(
        int(row["max_common_bag_size_all_12"])
        for row in development_rows
        if int(row["max_common_bag_size_all_12"]) > 0
    )
    untouched_complete = [
        row for row in untouched_rows if int(row["complete_12_condition_panel"]) == 1
    ]
    untouched_pair_complete = [
        row for row in untouched_rows if int(row["matched_contrasts_complete"]) == len(specs)
    ]

    candidate_bag_sizes = [30, 60, 100, 150, 200]
    bag_size_feasibility: list[dict[str, Any]] = []
    for size in candidate_bag_sizes:
        bag_size_feasibility.append(
            {
                "bag_size": size,
                "all_six_v1_development_sources_support_full_12": int(dev_common_bag >= size),
                "n_untouched_sources_supporting_full_12": sum(
                    int(row["max_common_bag_size_all_12"]) >= size
                    for row in untouched_rows
                ),
                "untouched_sources_supporting_full_12": ";".join(
                    row["source_name"]
                    for row in untouched_rows
                    if int(row["max_common_bag_size_all_12"]) >= size
                ),
            }
        )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "crosswalk": str(crosswalk_path),
        "pilot_manifest": str(manifest_path),
        "pilot_manifest_sha256": manifest_sha,
        "v1_development_sources": pilot_sources,
        "untouched_sources": untouched_sources,
        "pilot_conditions": [condition_label(k) for k in sorted(pilot_conditions, key=sort_condition)],
        "extra_conditions": [condition_label(k) for k in extra_conditions],
        "v1_development_max_common_bag_size_all_12": dev_common_bag,
        "untouched_complete_12_condition_sources": [row["source_name"] for row in untouched_complete],
        "untouched_complete_six_contrast_sources": [row["source_name"] for row in untouched_pair_complete],
        "ambiguous_source_condition_warning": ambiguous_warning,
        "recommended_next_step": (
            "If at least one untouched Source Name has all six matched sham/exposed contrasts, reserve it as a new final holdout. "
            "Use the six v1 sources for R1 v2 development only. Increase bag size only to a value supported without replacement by all development cells; 100 is a useful target if feasible."
        ),
    }

    source_path = output_dir / "r1_v2_source_summary.csv"
    coverage_path = output_dir / "r1_v2_source_condition_coverage.csv"
    contrast_path = output_dir / "r1_v2_matched_contrast_coverage.csv"
    bag_path = output_dir / "r1_v2_bag_size_feasibility.csv"
    summary_path = output_dir / "r1_v2_planning_summary.json"
    write_csv(source_path, source_rows)
    write_csv(coverage_path, coverage_rows)
    write_csv(contrast_path, contrast_rows)
    write_csv(bag_path, bag_size_feasibility)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Radiation Edge AI - NASA BPS R1 v2 expansion / untouched-holdout planner")
    print("Metadata only: no images are downloaded and no model is trained.")
    print("")
    print(f"v1 development Source Names ({len(pilot_sources)}): {', '.join(pilot_sources)}")
    print(f"untouched Source Names ({len(untouched_sources)}): {', '.join(untouched_sources) or 'NONE'}")
    print(f"frozen v1 conditions: {len(pilot_conditions)}")
    print(f"extra BPS conditions outside v1: {len(extra_conditions)}")
    print("")

    print("[Untouched Source Name coverage of frozen 12-condition panel]")
    for row in untouched_rows:
        print(
            f"{row['source_name']:8s} strain={row['strain']:<10s} sex={row['sex'] or '?':<6s} "
            f"conditions={row['pilot_conditions_present']}/12 "
            f"matched_contrasts={row['matched_contrasts_complete']}/6 "
            f"images/cell={row['min_images_present_cell']}/{row['median_images_present_cell']}/{row['max_images_present_cell']} "
            f"common12={row['max_common_bag_size_all_12']}"
        )
        if row["missing_pilot_conditions"]:
            print(f"  missing: {row['missing_pilot_conditions']}")
    print("")

    print("[Development bag-size feasibility]")
    print(f"maximum common bag size across all six v1 sources x 12 conditions: {dev_common_bag}")
    for row in bag_size_feasibility:
        print(
            f"bag={row['bag_size']:3d}: six-source development full12="
            f"{'YES' if row['all_six_v1_development_sources_support_full_12'] else 'NO'}; "
            f"untouched full12 sources={row['n_untouched_sources_supporting_full_12']} "
            f"[{row['untouched_sources_supporting_full_12']}]"
        )
    print("")

    print("[Untouched final-holdout candidates]")
    print(
        "complete 12-condition sources: "
        + (", ".join(row["source_name"] for row in untouched_complete) or "NONE")
    )
    print(
        "complete six matched-contrast sources: "
        + (", ".join(row["source_name"] for row in untouched_pair_complete) or "NONE")
    )
    if ambiguous_warning:
        print("WARNING: one or more Source Name x condition cells map to multiple Sample Names; inspect before freezing expansion.")
    print("")

    print("NASA BPS R1 V2 PLANNING COMPLETE: YES")
    print(f"Source summary: {source_path}")
    print(f"Condition coverage: {coverage_path}")
    print(f"Matched contrasts: {contrast_path}")
    print(f"Bag-size feasibility: {bag_path}")
    print(f"Summary: {summary_path}")
    print("")
    print(
        "NEXT GATE: reserve any eligible untouched Source Name(s) as final holdout, then freeze an expanded development manifest and R1 v2 contrast-aware objective without looking at final-holdout outcomes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
