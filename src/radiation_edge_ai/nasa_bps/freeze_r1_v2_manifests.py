"""Freeze NASA BPS R1 v2 development and blinded final-holdout manifests.

This is a metadata-only gate.  It does not download microscopy images, read
NASA phenotype tables, or train a model.

Design frozen here
------------------
Development cohort
- the same six Source Names used by pilot v1;
- the same 12 Fe/X-ray sham/exposed x 4/24/48 h condition cells;
- 100 paired FITC/DAPI/MASK nuclei per Source Name x condition;
- the *same* deterministic ranking seed as pilot v1, so every original v1
  top-30 nucleus must be contained in the expanded top-100 bag;
- the original three source-held-out folds are retained, but are explicitly
  development-CV folds because their v1 outcomes have already been inspected.

Final holdout cohort
- the three Source Names never used in v1: BALBCF2, C57BLF2, C57BLF3;
- only their available cells from the same frozen 12-condition panel;
- up to 100 paired nuclei per available Source Name x condition, selected by a
  new deterministic holdout seed;
- phenotype/reference values are deliberately NOT read or written here.

The three holdout sources are incomplete individually but complementary:
BALBCF2 covers 10/12 conditions (5/6 matched contrasts), C57BLF2 covers the
complete X-ray branch (3/6 contrasts), and C57BLF3 covers the complete Fe branch
(3/6 contrasts).  Their union therefore provides an untouched partial panel
without pretending that any one source has all 12 conditions.

The final-holdout phenotype table should not be joined or inspected until the
R1 v2 architecture, objective, training schedule and development-CV decision
rule are frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FROZEN_V1_MANIFEST_SHA256 = (
    "41ff2e4d64577f17c976dea5bbc17221fe63be9e2e7e2291309b520027fbe725"
)
V1_SELECTION_SEED = "nasa-bps-pilot-v1-720"
R1_V2_HOLDOUT_SEED = "nasa-bps-r1-v2-final-holdout-720"

DEV_N_PER_SOURCE_CONDITION = 100
HOLDOUT_MAX_N_PER_SOURCE_CONDITION = 100
EXPECTED_V1_NUCLEI = 2160
EXPECTED_DEV_NUCLEI = 6 * 12 * DEV_N_PER_SOURCE_CONDITION

DEV_SOURCES = (
    "BALBCF1",
    "BALBCM1",
    "BALBCM2",
    "C57BLF1",
    "C57BLM1",
    "C57BLM3",
)

FINAL_HOLDOUT_SOURCES = (
    "BALBCF2",
    "C57BLF2",
    "C57BLF3",
)

DEV_FOLDS = {
    "fold_female": ("BALBCF1", "C57BLF1"),
    "fold_male_1": ("BALBCM1", "C57BLM1"),
    "fold_male_2": ("BALBCM2", "C57BLM3"),
}

R1_CONDITIONS = (
    ("Fe", "0.0", "4"),
    ("Fe", "0.0", "24"),
    ("Fe", "0.0", "48"),
    ("Fe", "0.82", "4"),
    ("Fe", "0.82", "24"),
    ("Fe", "0.82", "48"),
    ("X-ray", "0.0", "4"),
    ("X-ray", "0.0", "24"),
    ("X-ray", "0.0", "48"),
    ("X-ray", "1.0", "4"),
    ("X-ray", "1.0", "24"),
    ("X-ray", "1.0", "48"),
)

EXPECTED_HOLDOUT_MISSING = {
    "BALBCF2": {
        ("X-ray", "0.0", "48"),
        ("X-ray", "1.0", "48"),
    },
    "C57BLF2": {
        ("Fe", "0.0", "4"),
        ("Fe", "0.0", "24"),
        ("Fe", "0.0", "48"),
        ("Fe", "0.82", "4"),
        ("Fe", "0.82", "24"),
        ("Fe", "0.82", "48"),
    },
    "C57BLF3": {
        ("X-ray", "0.0", "4"),
        ("X-ray", "0.0", "24"),
        ("X-ray", "0.0", "48"),
        ("X-ray", "1.0", "4"),
        ("X-ray", "1.0", "24"),
        ("X-ray", "1.0", "48"),
    },
}

FILENAME_RE = re.compile(
    r"^(?P<plate>P\d+)_(?P<acquisition>\d+)-(?P<well>[A-Za-z]+\d+)_"
    r"(?P<field>\d+)_(?P<object>\d+)_(?P<channel>proj|DAPI|MASK)\.tif$",
    re.IGNORECASE,
)


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
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_number(value: str) -> str:
    number = float(str(value).strip())
    if number.is_integer():
        return f"{number:.1f}"
    return format(number, "g")


def normalize_hour(value: str) -> str:
    number = float(str(value).strip())
    return str(int(number)) if number.is_integer() else format(number, "g")


def condition_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row["particle_type"]).strip(),
        normalize_number(row["dose_Gy"]),
        normalize_hour(row["hr_post_exposure"]),
    )


def condition_label(cond: tuple[str, str, str]) -> str:
    return f"{cond[0]}|{cond[1]}Gy|{cond[2]}h"


def parse_filename(filename: str) -> dict[str, str]:
    match = FILENAME_RE.match(Path(filename).name)
    if match is None:
        raise ValueError(f"Unrecognized NASA BPS filename: {filename}")
    result = match.groupdict()
    result["plate"] = result["plate"].upper()
    result["well"] = result["well"].upper()
    result["channel"] = result["channel"].lower()
    result["nucleus_key"] = (
        f"{result['plate']}_{result['acquisition']}-{result['well']}_"
        f"{result['field']}_{result['object']}"
    )
    return result


def deterministic_rank(seed: str, nucleus_key: str) -> str:
    return hashlib.sha256(f"{seed}|{nucleus_key}".encode("utf-8")).hexdigest()


def matched_contrast_specs() -> list[tuple[str, str, tuple[str, str, str], tuple[str, str, str]]]:
    specs = []
    exposed = {"Fe": "0.82", "X-ray": "1.0"}
    for branch, dose in exposed.items():
        for hour in ("4", "24", "48"):
            specs.append(
                (
                    branch,
                    hour,
                    (branch, "0.0", hour),
                    (branch, dose, hour),
                )
            )
    return specs


def enrich_candidates(
    crosswalk: list[dict[str, str]],
    dapi_meta: list[dict[str, str]],
) -> tuple[dict[tuple[str, tuple[str, str, str]], list[dict[str, str]]], int]:
    partners: dict[str, dict[str, str]] = defaultdict(dict)
    for row in dapi_meta:
        parsed = parse_filename(row["filename"])
        if parsed["channel"] in {"dapi", "mask"}:
            partners[parsed["nucleus_key"]][parsed["channel"]] = row["filename"]

    allowed_sources = set(DEV_SOURCES) | set(FINAL_HOLDOUT_SOURCES)
    allowed_conditions = set(R1_CONDITIONS)
    eligible: dict[tuple[str, tuple[str, str, str]], list[dict[str, str]]] = defaultdict(list)
    missing_pairs = 0

    for row in crosswalk:
        source = row["osd_source_name"]
        if source not in allowed_sources:
            continue
        cond = condition_key(row)
        if cond not in allowed_conditions:
            continue
        parsed = parse_filename(row["filename"])
        pair = partners.get(parsed["nucleus_key"], {})
        if "dapi" not in pair or "mask" not in pair:
            missing_pairs += 1
            continue
        enriched = dict(row)
        enriched["nucleus_key"] = parsed["nucleus_key"]
        enriched["dapi_filename"] = pair["dapi"]
        enriched["mask_filename"] = pair["mask"]
        eligible[(source, cond)].append(enriched)
    return eligible, missing_pairs


def validate_single_sample_per_cell(
    eligible: dict[tuple[str, tuple[str, str, str]], list[dict[str, str]]]
) -> None:
    bad: list[str] = []
    for (source, cond), rows in sorted(eligible.items()):
        if not rows:
            continue
        sample_names = sorted({row["osd_sample_name"] for row in rows if row["osd_sample_name"]})
        if len(sample_names) != 1:
            bad.append(f"{source} {condition_label(cond)} -> Sample Names={sample_names}")
    if bad:
        raise RuntimeError(
            "One or more Source Name x condition cells do not map to exactly one Sample Name: "
            + "; ".join(bad[:20])
        )


def build_record(
    row: dict[str, str],
    *,
    cohort: str,
    index: int,
    rank_sha256: str,
    v1_by_nucleus: dict[str, dict[str, str]],
) -> dict[str, Any]:
    source = row["osd_source_name"]
    old = v1_by_nucleus.get(row["nucleus_key"])
    record: dict[str, Any] = {
        "cohort": cohort,
        "manifest_index": index,
        "sample_id": (
            f"BPSR1V2D_{index:05d}" if cohort == "development" else f"BPSR1V2H_{index:05d}"
        ),
        "nucleus_key": row["nucleus_key"],
        "source_name": source,
        "sample_name": row["osd_sample_name"],
        "strain": row["osd_strain"],
        "sex": row["osd_sex"],
        "dose_Gy": normalize_number(row["dose_Gy"]),
        "particle_type": row["particle_type"],
        "hr_post_exposure": normalize_hour(row["hr_post_exposure"]),
        "plate": row["plate"],
        "acquisition": row["acquisition"],
        "well": row["well"],
        "fitc_filename": row["filename"],
        "dapi_filename": row["dapi_filename"],
        "mask_filename": row["mask_filename"],
        "selection_rank_sha256": rank_sha256,
        "v1_member": int(old is not None),
        "v1_sample_id": old["sample_id"] if old is not None else "",
    }
    if cohort == "development":
        for fold_name, held_out in DEV_FOLDS.items():
            record[f"{fold_name}_role"] = "test" if source in held_out else "train"
    else:
        record["holdout_status"] = "FINAL_BLINDED_DO_NOT_JOIN_PHENOTYPES"
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"
    parser.add_argument(
        "--crosswalk",
        default=str(metadata_root / "fitc_53bp1_osd366_crosswalk.csv"),
    )
    parser.add_argument(
        "--dapi-mask-meta",
        default=str(metadata_root / "meta_DAPI_MASK.csv"),
    )
    parser.add_argument(
        "--v1-manifest",
        default=str(metadata_root / "pilot_v1" / "pilot_v1_manifest.csv"),
    )
    parser.add_argument(
        "--v1-summary",
        default=str(metadata_root / "pilot_v1" / "pilot_v1_summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(metadata_root / "r1_v2_freeze"),
    )
    parser.add_argument("--dev-bag-size", type=int, default=DEV_N_PER_SOURCE_CONDITION)
    parser.add_argument(
        "--holdout-max-bag-size",
        type=int,
        default=HOLDOUT_MAX_N_PER_SOURCE_CONDITION,
    )
    args = parser.parse_args()

    if args.dev_bag_size != DEV_N_PER_SOURCE_CONDITION:
        raise ValueError(
            f"R1 v2 freeze requires development bag size {DEV_N_PER_SOURCE_CONDITION}; got {args.dev_bag_size}"
        )
    if args.holdout_max_bag_size != HOLDOUT_MAX_N_PER_SOURCE_CONDITION:
        raise ValueError(
            "R1 v2 freeze requires holdout maximum bag size "
            f"{HOLDOUT_MAX_N_PER_SOURCE_CONDITION}; got {args.holdout_max_bag_size}"
        )

    crosswalk_path = Path(args.crosswalk).resolve()
    dapi_path = Path(args.dapi_mask_meta).resolve()
    v1_manifest_path = Path(args.v1_manifest).resolve()
    v1_summary_path = Path(args.v1_summary).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (crosswalk_path, dapi_path, v1_manifest_path, v1_summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    v1_manifest_sha = sha256_file(v1_manifest_path)
    if v1_manifest_sha != FROZEN_V1_MANIFEST_SHA256:
        raise RuntimeError(
            f"Frozen v1 manifest SHA256 mismatch: expected {FROZEN_V1_MANIFEST_SHA256}, got {v1_manifest_sha}"
        )

    v1_summary = json.loads(v1_summary_path.read_text(encoding="utf-8"))
    crosswalk_sha = sha256_file(crosswalk_path)
    dapi_sha = sha256_file(dapi_path)
    if v1_summary.get("crosswalk_sha256") != crosswalk_sha:
        raise RuntimeError(
            "Current crosswalk does not match the metadata frozen with pilot v1: "
            f"v1={v1_summary.get('crosswalk_sha256')} current={crosswalk_sha}"
        )
    if v1_summary.get("dapi_mask_metadata_sha256") != dapi_sha:
        raise RuntimeError(
            "Current DAPI/MASK metadata does not match the metadata frozen with pilot v1: "
            f"v1={v1_summary.get('dapi_mask_metadata_sha256')} current={dapi_sha}"
        )

    crosswalk = read_csv(crosswalk_path)
    dapi_meta = read_csv(dapi_path)
    v1_manifest = read_csv(v1_manifest_path)
    if len(v1_manifest) != EXPECTED_V1_NUCLEI:
        raise RuntimeError(f"v1 manifest nuclei={len(v1_manifest)}; expected {EXPECTED_V1_NUCLEI}")

    v1_sources = tuple(sorted({row["source_name"] for row in v1_manifest}))
    if set(v1_sources) != set(DEV_SOURCES):
        raise RuntimeError(f"v1 Source Names changed: {v1_sources}")
    v1_conditions = {condition_key(row) for row in v1_manifest}
    if v1_conditions != set(R1_CONDITIONS):
        raise RuntimeError("v1 condition panel does not match the frozen R1 v2 condition panel")

    v1_by_nucleus = {row["nucleus_key"]: row for row in v1_manifest}
    if len(v1_by_nucleus) != EXPECTED_V1_NUCLEI:
        raise RuntimeError("v1 manifest contains duplicate nucleus_key values")

    eligible, missing_pairs = enrich_candidates(crosswalk, dapi_meta)
    validate_single_sample_per_cell(eligible)

    # Freeze expanded six-source development cohort.  Use the original v1 seed
    # so top-30 continuity is mathematically guaranteed and then verified below.
    dev_records: list[dict[str, Any]] = []
    dev_cell_counts: dict[str, int] = {}
    index = 0
    for source in DEV_SOURCES:
        for cond in R1_CONDITIONS:
            candidates = eligible.get((source, cond), [])
            ranked = sorted(
                candidates,
                key=lambda row: (
                    deterministic_rank(V1_SELECTION_SEED, row["nucleus_key"]),
                    row["nucleus_key"],
                ),
            )
            if len(ranked) < DEV_N_PER_SOURCE_CONDITION:
                raise RuntimeError(
                    f"Development cell {source} {condition_label(cond)} has {len(ranked)} paired nuclei; "
                    f"need {DEV_N_PER_SOURCE_CONDITION}"
                )
            chosen = ranked[:DEV_N_PER_SOURCE_CONDITION]
            dev_cell_counts[f"{source}|{condition_label(cond)}"] = len(chosen)
            for row in chosen:
                index += 1
                rank = deterministic_rank(V1_SELECTION_SEED, row["nucleus_key"])
                dev_records.append(
                    build_record(
                        row,
                        cohort="development",
                        index=index,
                        rank_sha256=rank,
                        v1_by_nucleus=v1_by_nucleus,
                    )
                )

    if len(dev_records) != EXPECTED_DEV_NUCLEI:
        raise RuntimeError(
            f"Expanded development nuclei={len(dev_records)}; expected {EXPECTED_DEV_NUCLEI}"
        )

    dev_keys = {row["nucleus_key"] for row in dev_records}
    missing_v1 = sorted(set(v1_by_nucleus) - dev_keys)
    if missing_v1:
        raise RuntimeError(
            f"Expanded top-100 bags failed to contain {len(missing_v1)} v1 nuclei; first={missing_v1[:5]}"
        )
    v1_members_in_dev = sum(int(row["v1_member"]) for row in dev_records)
    if v1_members_in_dev != EXPECTED_V1_NUCLEI:
        raise RuntimeError(
            f"Expanded development manifest marks {v1_members_in_dev} v1 members; expected {EXPECTED_V1_NUCLEI}"
        )

    # Freeze untouched partial holdout without reading any phenotype table.
    holdout_records: list[dict[str, Any]] = []
    holdout_cell_rows: list[dict[str, Any]] = []
    holdout_index = 0
    for source in FINAL_HOLDOUT_SOURCES:
        observed_present = {
            cond for cond in R1_CONDITIONS if eligible.get((source, cond), [])
        }
        observed_missing = set(R1_CONDITIONS) - observed_present
        if observed_missing != EXPECTED_HOLDOUT_MISSING[source]:
            raise RuntimeError(
                f"Holdout coverage drift for {source}: expected missing "
                f"{sorted(condition_label(c) for c in EXPECTED_HOLDOUT_MISSING[source])}, got "
                f"{sorted(condition_label(c) for c in observed_missing)}"
            )

        for cond in R1_CONDITIONS:
            candidates = eligible.get((source, cond), [])
            if not candidates:
                holdout_cell_rows.append(
                    {
                        "source_name": source,
                        "particle_type": cond[0],
                        "dose_Gy": cond[1],
                        "hr_post_exposure": cond[2],
                        "condition_present": 0,
                        "available_paired_nuclei": 0,
                        "selected_nuclei": 0,
                    }
                )
                continue

            ranked = sorted(
                candidates,
                key=lambda row: (
                    deterministic_rank(R1_V2_HOLDOUT_SEED, row["nucleus_key"]),
                    row["nucleus_key"],
                ),
            )
            chosen = ranked[:HOLDOUT_MAX_N_PER_SOURCE_CONDITION]
            holdout_cell_rows.append(
                {
                    "source_name": source,
                    "particle_type": cond[0],
                    "dose_Gy": cond[1],
                    "hr_post_exposure": cond[2],
                    "condition_present": 1,
                    "available_paired_nuclei": len(ranked),
                    "selected_nuclei": len(chosen),
                }
            )
            for row in chosen:
                holdout_index += 1
                rank = deterministic_rank(R1_V2_HOLDOUT_SEED, row["nucleus_key"])
                holdout_records.append(
                    build_record(
                        row,
                        cohort="final_holdout_blinded",
                        index=holdout_index,
                        rank_sha256=rank,
                        v1_by_nucleus=v1_by_nucleus,
                    )
                )

    holdout_keys = {row["nucleus_key"] for row in holdout_records}
    if dev_keys & holdout_keys:
        raise RuntimeError("Development and final-holdout manifests overlap in nucleus_key")
    if any(int(row["v1_member"]) for row in holdout_records):
        raise RuntimeError("Final holdout unexpectedly contains pilot-v1 nuclei")

    contrast_rows: list[dict[str, Any]] = []
    contrast_complete_total = 0
    for source in FINAL_HOLDOUT_SOURCES:
        for branch, hour, sham, exposed in matched_contrast_specs():
            sham_rows = [
                row
                for row in holdout_cell_rows
                if row["source_name"] == source
                and row["particle_type"] == sham[0]
                and row["dose_Gy"] == sham[1]
                and row["hr_post_exposure"] == sham[2]
            ]
            exp_rows = [
                row
                for row in holdout_cell_rows
                if row["source_name"] == source
                and row["particle_type"] == exposed[0]
                and row["dose_Gy"] == exposed[1]
                and row["hr_post_exposure"] == exposed[2]
            ]
            if len(sham_rows) != 1 or len(exp_rows) != 1:
                raise RuntimeError("Internal holdout contrast coverage error")
            complete = int(
                int(sham_rows[0]["selected_nuclei"]) > 0
                and int(exp_rows[0]["selected_nuclei"]) > 0
            )
            contrast_complete_total += complete
            contrast_rows.append(
                {
                    "source_name": source,
                    "branch": branch,
                    "timepoint_hr": hour,
                    "sham_condition": condition_label(sham),
                    "exposed_condition": condition_label(exposed),
                    "sham_selected_nuclei": int(sham_rows[0]["selected_nuclei"]),
                    "exposed_selected_nuclei": int(exp_rows[0]["selected_nuclei"]),
                    "contrast_complete": complete,
                    "phenotype_status": "BLINDED_NOT_JOINED",
                }
            )

    if contrast_complete_total != 11:
        raise RuntimeError(
            f"Untouched holdout complete matched contrasts={contrast_complete_total}; expected 11"
        )

    dev_path = output_dir / "r1_v2_development_manifest_100.csv"
    holdout_path = output_dir / "r1_v2_final_holdout_manifest_blinded.csv"
    holdout_cells_path = output_dir / "r1_v2_final_holdout_condition_coverage.csv"
    holdout_contrasts_path = output_dir / "r1_v2_final_holdout_contrast_coverage.csv"
    write_csv(dev_path, dev_records)
    write_csv(holdout_path, holdout_records)
    write_csv(holdout_cells_path, holdout_cell_rows)
    write_csv(holdout_contrasts_path, contrast_rows)

    dev_by_source = Counter(str(row["source_name"]) for row in dev_records)
    holdout_by_source = Counter(str(row["source_name"]) for row in holdout_records)
    holdout_selected_counts = [
        int(row["selected_nuclei"])
        for row in holdout_cell_rows
        if int(row["condition_present"]) == 1
    ]

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_version": "NASA_BPS_R1_V2_MANIFEST_FREEZE_V1",
        "metadata_only": True,
        "phenotype_tables_read": False,
        "v1_manifest_sha256": v1_manifest_sha,
        "crosswalk_sha256": crosswalk_sha,
        "dapi_mask_metadata_sha256": dapi_sha,
        "development": {
            "sources": list(DEV_SOURCES),
            "conditions": [condition_label(cond) for cond in R1_CONDITIONS],
            "n_per_source_condition": DEV_N_PER_SOURCE_CONDITION,
            "n_nuclei": len(dev_records),
            "selection_seed": V1_SELECTION_SEED,
            "v1_nuclei_preserved": v1_members_in_dev,
            "counts_by_source": dict(dev_by_source),
            "folds_status": "development_cv_only_not_untouched_final_test",
            "folds": {name: list(values) for name, values in DEV_FOLDS.items()},
            "manifest": str(dev_path),
            "manifest_sha256": sha256_file(dev_path),
        },
        "final_holdout": {
            "sources": list(FINAL_HOLDOUT_SOURCES),
            "status": "BLINDED_DO_NOT_JOIN_PHENOTYPES_UNTIL_R1_V2_MODEL_POLICY_FROZEN",
            "max_n_per_available_source_condition": HOLDOUT_MAX_N_PER_SOURCE_CONDITION,
            "n_available_source_condition_cells": sum(
                int(row["condition_present"]) for row in holdout_cell_rows
            ),
            "n_nuclei": len(holdout_records),
            "selection_seed": R1_V2_HOLDOUT_SEED,
            "counts_by_source": dict(holdout_by_source),
            "selected_nuclei_per_present_cell_min": min(holdout_selected_counts),
            "selected_nuclei_per_present_cell_max": max(holdout_selected_counts),
            "complete_matched_contrasts": contrast_complete_total,
            "possible_matched_contrasts": 18,
            "manifest": str(holdout_path),
            "manifest_sha256": sha256_file(holdout_path),
            "condition_coverage": str(holdout_cells_path),
            "contrast_coverage": str(holdout_contrasts_path),
        },
        "missing_paired_nuclei_seen_before_selection": missing_pairs,
        "guardrails": [
            "The six v1 Source Names are now development data; their old held-out outcomes have already been inspected.",
            "R1 v2 development bags contain the exact v1 top-30 nuclei plus 70 additional deterministic nuclei per Source Name x condition.",
            "The three final-holdout Source Names were not used in R1 v1 training or model development.",
            "Final-holdout phenotype/reference values were not read while freezing these manifests.",
            "Do not join or inspect final-holdout Raw_pheno/Processed_pheno outcomes until R1 v2 architecture, objective, schedule and development decision rule are frozen.",
            "Final holdout is a complementary partial panel; do not claim any individual holdout source has all 12 conditions.",
            "All model validation remains grouped at OSD Source Name / sample-bag level; never random-split nuclei as independent biological replicates.",
        ],
    }
    summary_path = output_dir / "r1_v2_manifest_freeze_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Radiation Edge AI - freeze NASA BPS R1 v2 manifests")
    print("Metadata only: no images downloaded, no phenotype tables read, no model trained.")
    print("")
    print("[Development manifest]")
    print(f"sources: {len(DEV_SOURCES)}")
    print(f"conditions/source: {len(R1_CONDITIONS)}")
    print(f"nuclei/source x condition: {DEV_N_PER_SOURCE_CONDITION}")
    print(f"development nuclei: {len(dev_records)}")
    print(f"pilot-v1 nuclei preserved inside expanded manifest: {v1_members_in_dev}/{EXPECTED_V1_NUCLEI}")
    print(f"manifest SHA256: {summary['development']['manifest_sha256']}")
    print("")
    print("[Final holdout manifest - BLINDED]")
    for source in FINAL_HOLDOUT_SOURCES:
        present = sum(
            int(row["condition_present"])
            for row in holdout_cell_rows
            if row["source_name"] == source
        )
        contrasts = sum(
            int(row["contrast_complete"])
            for row in contrast_rows
            if row["source_name"] == source
        )
        nuclei = holdout_by_source[source]
        print(
            f"{source}: conditions={present}/12 matched_contrasts={contrasts}/6 selected_nuclei={nuclei}"
        )
    print(f"holdout nuclei total: {len(holdout_records)}")
    print(f"complete matched contrasts total: {contrast_complete_total}/18")
    print(
        "selected nuclei/present cell min/max: "
        f"{min(holdout_selected_counts)}/{max(holdout_selected_counts)}"
    )
    print(f"manifest SHA256: {summary['final_holdout']['manifest_sha256']}")
    print("")
    print("NASA BPS R1 V2 MANIFEST FREEZE COMPLETE: YES")
    print(f"Development manifest: {dev_path}")
    print(f"Final holdout manifest: {holdout_path}")
    print(f"Holdout condition coverage: {holdout_cells_path}")
    print(f"Holdout contrast coverage: {holdout_contrasts_path}")
    print(f"Summary: {summary_path}")
    print("")
    print(
        "NEXT GATE: download only the additional R1 v2 DEVELOPMENT images first. "
        "Keep the final-holdout image/phenotype outcomes untouched until the R1 v2 model policy is frozen."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
