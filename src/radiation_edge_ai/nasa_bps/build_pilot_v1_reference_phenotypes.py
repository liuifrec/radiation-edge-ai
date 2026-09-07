"""Build the NASA BPS pilot-v1 sample-level 53BP1 reference phenotype table.

This gate makes the analysis level explicit before model training.

The current OSD-366/LSDS-111 raw phenotype table contains one aggregate row per
sample/plate-well rather than one row per segmented nucleus.  The frozen pilot
contains 30 nucleus crops from each of 6 biological sources x 12 radiation
conditions = 72 source-condition sample bags.

This script therefore:
- verifies the frozen pilot-v1 manifest SHA256;
- joins each unique pilot Sample Name to exactly one row of
  LSDS-111_immunostaining_Raw_pheno_V3.csv;
- checks agreement for UserID/source, strain, sex, plate, well, radiation, dose,
  and timepoint whenever those fields are available;
- writes one authoritative sample-level reference row for each pilot bag;
- propagates that reference back to all 2160 pilot nuclei only as an aggregate
  bag label (never as a claimed per-nucleus foci count);
- summarizes NASA avg_nfoci and avg_foci_no_outl across the six biological
  sources for each radiation condition;
- computes matched irradiated-minus-sham deltas within source/radiation/time.

The output is intended to freeze the biological reference hierarchy:
  nucleus image -> model output -> aggregate within sample bag -> compare with
  NASA raw avg_nfoci -> compare higher-level dose/time/radiation conclusions.

No microscopy data or additional NASA files are downloaded.
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
from statistics import median
from typing import Any

FROZEN_MANIFEST_SHA256 = "41ff2e4d64577f17c976dea5bbc17221fe63be9e2e7e2291309b520027fbe725"

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
    "avg_foci_no_outl",
)

REFERENCE_FIELDS = (
    "num_nuc",
    "avg_nfoci",
    "std_nfoci",
    "num_nuc_no_outl",
    "avg_foci_no_outl",
    "std_foci_no_outl",
    "avg_fitc",
    "avg_fitc_bg",
    "avg_nuc_area",
    "avg_nuc_dapi",
    "avg_p2a",
    "spot_fitc_sum",
    "confluence....",
)


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def norm_text(value: str) -> str:
    return "".join(str(value).strip().lower().split())


def norm_number(value: str) -> str:
    text = str(value).strip()
    if text == "":
        return ""
    number = float(text)
    if math.isfinite(number) and number.is_integer():
        return f"{number:.1f}"
    return format(number, "g")


def norm_hour(value: str) -> str:
    text = str(value).strip()
    if text == "":
        return ""
    number = float(text)
    return str(int(number)) if number.is_integer() else format(number, "g")


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
    aliases = {"f": "female", "female": "female", "m": "male", "male": "male"}
    return aliases.get(text, text)


def float_or_none(value: str) -> float | None:
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def unique_manifest_samples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_name"]].append(row)

    unique_rows: list[dict[str, str]] = []
    invariant_fields = (
        "source_name",
        "sample_name",
        "strain",
        "sex",
        "dose_Gy",
        "particle_type",
        "hr_post_exposure",
        "plate",
        "well",
    )
    fold_fields = [field for field in rows[0] if field.endswith("_role")]
    for sample_name, subset in sorted(by_sample.items()):
        first = subset[0]
        for field in invariant_fields + tuple(fold_fields):
            values = {row.get(field, "") for row in subset}
            if len(values) != 1:
                raise RuntimeError(
                    f"Pilot manifest Sample Name {sample_name!r} is not invariant for {field}: {sorted(values)}"
                )
        record = {field: first.get(field, "") for field in invariant_fields + tuple(fold_fields)}
        record["pilot_nuclei"] = str(len(subset))
        unique_rows.append(record)
    return unique_rows


def condition_key(row: dict[str, str], *, raw: bool = False) -> tuple[str, str, str]:
    if raw:
        return (
            norm_radiation(row["radiation"]),
            norm_number(row["dose..Gy."]),
            norm_hour(row["timepoint..hr."]),
        )
    return (
        norm_radiation(row["particle_type"]),
        norm_number(row["dose_Gy"]),
        norm_hour(row["hr_post_exposure"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"
    parser.add_argument(
        "--manifest",
        default=str(metadata_root / "pilot_v1" / "pilot_v1_manifest.csv"),
    )
    parser.add_argument(
        "--raw-phenotypes",
        default=str(
            metadata_root
            / "current_phenotypes"
            / "files"
            / "LSDS-111_immunostaining_Raw_pheno_V3.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(metadata_root / "pilot_v1" / "reference_phenotypes"),
    )
    parser.add_argument("--expected-manifest-sha256", default=FROZEN_MANIFEST_SHA256)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    raw_path = Path(args.raw_phenotypes).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (manifest_path, raw_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest_sha = sha256_file(manifest_path)
    if args.expected_manifest_sha256 and manifest_sha != args.expected_manifest_sha256:
        raise RuntimeError(
            f"Frozen pilot manifest SHA256 mismatch: expected {args.expected_manifest_sha256}, got {manifest_sha}"
        )

    manifest = read_csv(manifest_path)
    raw = read_csv(raw_path)
    missing_columns = [column for column in RAW_REQUIRED if column not in raw[0]]
    if missing_columns:
        raise RuntimeError(f"Raw phenotype table missing required columns: {missing_columns}")

    pilot_samples = unique_manifest_samples(manifest)
    raw_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        raw_by_sample[norm_text(row["Sample Name"])].append(row)

    reference_rows: list[dict[str, Any]] = []
    unmatched: list[str] = []
    ambiguous: list[str] = []
    mismatch_rows: list[dict[str, Any]] = []

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

    for pilot in pilot_samples:
        key = norm_text(pilot["sample_name"])
        matches = raw_by_sample.get(key, [])
        if not matches:
            unmatched.append(pilot["sample_name"])
            continue
        if len(matches) != 1:
            ambiguous.append(pilot["sample_name"])
            continue
        nasa = matches[0]

        mismatches: list[str] = []
        for pilot_field, raw_field, normalizer in checks:
            left = normalizer(pilot.get(pilot_field, ""))
            right = normalizer(nasa.get(raw_field, ""))
            if left and right and left != right:
                mismatches.append(f"{pilot_field}={pilot.get(pilot_field,'')} != {raw_field}={nasa.get(raw_field,'')}")

        record: dict[str, Any] = dict(pilot)
        record.update(
            {
                "nasa_user_id": nasa["UserID"],
                "nasa_sample_name": nasa["Sample Name"],
                "nasa_plate": nasa["plate"],
                "nasa_plate_well": nasa["plate_well"],
                "nasa_radiation": nasa["radiation"],
                "nasa_dose_Gy": nasa["dose..Gy."],
                "nasa_timepoint_hr": nasa["timepoint..hr."],
                "metadata_mismatch_count": len(mismatches),
                "metadata_mismatches": " | ".join(mismatches),
            }
        )
        for field in REFERENCE_FIELDS:
            record[f"nasa_{field}"] = nasa.get(field, "")
        reference_rows.append(record)
        if mismatches:
            mismatch_rows.append(
                {
                    "sample_name": pilot["sample_name"],
                    "mismatch_count": len(mismatches),
                    "mismatches": " | ".join(mismatches),
                }
            )

    # Propagate aggregate bag reference to each nucleus, explicitly labeled as aggregate.
    ref_by_sample = {row["sample_name"]: row for row in reference_rows}
    nucleus_rows: list[dict[str, Any]] = []
    for row in manifest:
        ref = ref_by_sample.get(row["sample_name"])
        if ref is None:
            continue
        nucleus_rows.append(
            {
                "sample_id": row["sample_id"],
                "nucleus_key": row["nucleus_key"],
                "source_name": row["source_name"],
                "sample_name": row["sample_name"],
                "strain": row["strain"],
                "sex": row["sex"],
                "particle_type": row["particle_type"],
                "dose_Gy": row["dose_Gy"],
                "hr_post_exposure": row["hr_post_exposure"],
                "aggregate_reference_level": "sample_plate_well_mean_not_per_nucleus",
                "nasa_num_nuc": ref["nasa_num_nuc"],
                "nasa_avg_nfoci": ref["nasa_avg_nfoci"],
                "nasa_avg_foci_no_outl": ref["nasa_avg_foci_no_outl"],
            }
        )

    # Condition summaries over biological sources (not nuclei).
    by_condition: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in reference_rows:
        by_condition[condition_key(row)].append(row)

    condition_rows: list[dict[str, Any]] = []
    for (radiation, dose, hour), subset in sorted(by_condition.items()):
        nfoci = [float(row["nasa_avg_nfoci"]) for row in subset if str(row["nasa_avg_nfoci"]).strip()]
        no_outl = [
            float(row["nasa_avg_foci_no_outl"])
            for row in subset
            if str(row["nasa_avg_foci_no_outl"]).strip()
        ]
        num_nuc = [int(float(row["nasa_num_nuc"])) for row in subset if str(row["nasa_num_nuc"]).strip()]
        condition_rows.append(
            {
                "radiation": radiation,
                "dose_Gy": dose,
                "timepoint_hr": hour,
                "biological_sources": len(subset),
                "nasa_total_nuclei_used": sum(num_nuc),
                "source_median_avg_nfoci": median(nfoci) if nfoci else "",
                "source_min_avg_nfoci": min(nfoci) if nfoci else "",
                "source_max_avg_nfoci": max(nfoci) if nfoci else "",
                "source_median_avg_foci_no_outl": median(no_outl) if no_outl else "",
            }
        )

    # Matched irradiated-minus-sham deltas within source/radiation/time.
    source_lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in reference_rows:
        radiation, dose, hour = condition_key(row)
        source_lookup[(row["source_name"], radiation, dose, hour)] = row

    delta_rows: list[dict[str, Any]] = []
    irradiated_dose = {"fe": "0.82", "x-ray": "1.0"}
    for source in sorted({row["source_name"] for row in reference_rows}):
        for radiation, dose in irradiated_dose.items():
            for hour in ("4", "24", "48"):
                sham = source_lookup.get((source, radiation, "0.0", hour))
                exposed = source_lookup.get((source, radiation, dose, hour))
                if sham is None or exposed is None:
                    continue
                sham_nfoci = float(sham["nasa_avg_nfoci"])
                exp_nfoci = float(exposed["nasa_avg_nfoci"])
                sham_clean = float(sham["nasa_avg_foci_no_outl"])
                exp_clean = float(exposed["nasa_avg_foci_no_outl"])
                delta_rows.append(
                    {
                        "source_name": source,
                        "strain": exposed["strain"],
                        "sex": exposed["sex"],
                        "radiation": radiation,
                        "dose_Gy": dose,
                        "timepoint_hr": hour,
                        "sham_avg_nfoci": sham_nfoci,
                        "irradiated_avg_nfoci": exp_nfoci,
                        "delta_avg_nfoci": exp_nfoci - sham_nfoci,
                        "sham_avg_foci_no_outl": sham_clean,
                        "irradiated_avg_foci_no_outl": exp_clean,
                        "delta_avg_foci_no_outl": exp_clean - sham_clean,
                    }
                )

    bag_path = output_dir / "pilot_v1_sample_reference_phenotypes.csv"
    nucleus_path = output_dir / "pilot_v1_nucleus_to_aggregate_reference.csv"
    condition_path = output_dir / "pilot_v1_condition_reference_summary.csv"
    delta_path = output_dir / "pilot_v1_matched_radiation_deltas.csv"
    mismatch_path = output_dir / "pilot_v1_reference_metadata_mismatches.csv"

    write_csv(bag_path, reference_rows)
    write_csv(nucleus_path, nucleus_rows)
    write_csv(condition_path, condition_rows)
    write_csv(delta_path, delta_rows)
    write_csv(mismatch_path, mismatch_rows, ["sample_name", "mismatch_count", "mismatches"])

    raw_num_nuc = [int(float(row["nasa_num_nuc"])) for row in reference_rows if str(row["nasa_num_nuc"]).strip()]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_version": "NASA_BPS_53BP1_PILOT_V1",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "raw_phenotype_table": str(raw_path),
        "raw_phenotype_sha256": sha256_file(raw_path),
        "raw_rows": len(raw),
        "pilot_nuclei": len(manifest),
        "pilot_unique_sample_names": len(pilot_samples),
        "matched_unique_sample_names": len(reference_rows),
        "unmatched_sample_names": unmatched,
        "ambiguous_sample_names": ambiguous,
        "metadata_mismatch_samples": len(mismatch_rows),
        "nuclei_with_aggregate_reference": len(nucleus_rows),
        "raw_num_nuc_min": min(raw_num_nuc) if raw_num_nuc else None,
        "raw_num_nuc_median": median(raw_num_nuc) if raw_num_nuc else None,
        "raw_num_nuc_max": max(raw_num_nuc) if raw_num_nuc else None,
        "interpretation": [
            "LSDS-111 Raw_pheno rows are treated as sample/plate-well aggregate measurements, not per-nucleus labels.",
            "NASA avg_nfoci and avg_foci_no_outl may be used as sample-level biological reference targets after exact Sample Name/metadata concordance is demonstrated.",
            "The 30 frozen pilot nuclei per sample form a deterministic image bag; their model outputs may be aggregated and compared with the NASA sample-level phenotype.",
            "Do not copy avg_nfoci onto individual nuclei and evaluate it as if it were a per-nucleus ground-truth count.",
            "Source Name remains the biological grouping unit for train/test separation.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Radiation Edge AI - NASA BPS pilot-v1 reference phenotype crosswalk")
    print(f"Frozen pilot manifest SHA256: {manifest_sha}")
    print(f"Raw phenotype rows: {len(raw)}")
    print(f"Pilot nuclei: {len(manifest)}")
    print(f"Pilot unique Sample Names: {len(pilot_samples)}")
    print("")
    print("[Sample-level crosswalk]")
    print(f"exact unique Sample Name matches: {len(reference_rows)}/{len(pilot_samples)}")
    print(f"unmatched pilot Sample Names: {len(unmatched)}")
    print(f"ambiguous pilot Sample Names: {len(ambiguous)}")
    print(f"samples with metadata mismatches: {len(mismatch_rows)}")
    print(f"pilot nuclei linked to aggregate reference: {len(nucleus_rows)}/{len(manifest)}")
    if raw_num_nuc:
        print(
            "NASA nuclei contributing to each matched raw sample min/median/max: "
            f"{min(raw_num_nuc)}/{median(raw_num_nuc):.1f}/{max(raw_num_nuc)}"
        )
    print("")
    print("[Condition reference: source-level medians]")
    for row in condition_rows:
        print(
            f"{row['radiation']:6} {row['dose_Gy']:>4} Gy {row['timepoint_hr']:>2} h "
            f"sources={row['biological_sources']} NASA_n={row['nasa_total_nuclei_used']} "
            f"median avg_nfoci={float(row['source_median_avg_nfoci']):.4f} "
            f"median no-outl={float(row['source_median_avg_foci_no_outl']):.4f}"
        )
    print("")
    print("NASA BPS PILOT V1 REFERENCE PHENOTYPE CROSSWALK COMPLETE: YES")
    print(f"Sample reference: {bag_path}")
    print(f"Nucleus -> aggregate reference: {nucleus_path}")
    print(f"Condition summary: {condition_path}")
    print(f"Matched radiation deltas: {delta_path}")
    print(f"Summary: {summary_path}")
    print("")

    success = (
        len(reference_rows) == len(pilot_samples)
        and not unmatched
        and not ambiguous
        and not mismatch_rows
        and len(nucleus_rows) == len(manifest)
    )
    if success:
        print("REFERENCE LEVEL FROZEN: sample/plate-well aggregate, not per nucleus.")
        print(
            "NEXT GATE: use the 72 matched sample bags to freeze R1: model per-nucleus 53BP1 burden, "
            "aggregate predictions within each bag, and validate against NASA avg_nfoci/avg_foci_no_outl "
            "under source-held-out folds before KL720 conversion."
        )
        return 0

    print("REFERENCE LEVEL FROZEN: NO")
    print("NEXT GATE: inspect unmatched/ambiguous/mismatched sample records before any training target is used.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
