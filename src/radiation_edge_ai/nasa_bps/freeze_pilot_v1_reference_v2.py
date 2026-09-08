"""Freeze the NASA BPS pilot-v1 aggregate 53BP1 reference hierarchy.

This v2 freeze resolves the two apparent metadata mismatches identified after
exact Sample Name matching:

1. Strain vocabulary is shortened in LSDS-111 Raw_pheno_V3 (for the frozen
   pilot, BALB/cByJ -> BALBC and C57BL/6J -> C57). The full 6,507-sample
   ISA-to-raw join demonstrated deterministic strain aliases.
2. Radiation vocabulary encodes different concepts for sham samples. The ISA
   labels 0-Gy samples as ``sham-irradiated`` whereas the BPS microscopy
   benchmark and LSDS-111 raw phenotype table retain the experimental branch
   label (for this pilot, Fe 600 MeV/n or X-ray) even when dose == 0.

Therefore radiation is validated branch-wise against the frozen BPS manifest,
not as a globally bijective ISA->raw vocabulary mapping.

The script:
- verifies the frozen manifest and raw phenotype hashes;
- requires 72 unique Sample Name joins and 30 pilot nuclei per sample;
- validates source/UserID, sex, plate, well, dose, time, demonstrated strain
  aliases, and experimental radiation branch for every matched sample;
- writes a clean authoritative 72-row sample reference table;
- writes a 2,160-row nucleus-to-aggregate-reference table explicitly marking
  the NASA phenotype as sample/plate-well aggregate, never per-nucleus truth;
- summarizes the NASA reference by condition and matched irradiated-vs-sham
  deltas over biological Source Names;
- emits a freeze JSON documenting the exact semantics and source hashes.

No network access and no microscopy downloads are performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

FROZEN_MANIFEST_SHA256 = "41ff2e4d64577f17c976dea5bbc17221fe63be9e2e7e2291309b520027fbe725"
FROZEN_RAW_PHENO_SHA256 = "d28dc5d9c53221c66a90074a92783fd4708ea83db0fc1a9c767e47500e643d6a"

EXPECTED_PILOT_SAMPLES = 72
EXPECTED_PILOT_NUCLEI = 2160
EXPECTED_NUCLEI_PER_SAMPLE = 30

STRAIN_ALIAS = {
    "BALB/cByJ": "BALBC",
    "C57BL/6J": "C57",
}

RAW_REFERENCE_FIELDS = (
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def norm_text(value: str) -> str:
    return "".join(str(value).strip().lower().split())


def norm_sex(value: str) -> str:
    text = norm_text(value)
    return {"f": "female", "female": "female", "m": "male", "male": "male"}.get(text, text)


def norm_number(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numeric value: {value!r}")
    return f"{number:.12g}"


def norm_hour(value: str) -> str:
    return norm_number(value)


def norm_plate_or_well(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def norm_branch(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())
    if text in {"fe", "56fe", "fe56", "fe600mevn", "fe600mevpern"} or text.startswith("fe600"):
        return "fe"
    if text in {"xray", "xrays", "xrayradiation"} or text.startswith("xray"):
        return "x-ray"
    return text


def unique_manifest_samples(manifest: list[dict[str, str]]) -> list[dict[str, str]]:
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        by_sample[row["sample_name"]].append(row)

    result: list[dict[str, str]] = []
    invariant = (
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
    fold_fields = [key for key in manifest[0] if key.endswith("_role")]

    for sample_name, subset in sorted(by_sample.items()):
        if len(subset) != EXPECTED_NUCLEI_PER_SAMPLE:
            raise RuntimeError(
                f"Pilot Sample Name {sample_name!r} has {len(subset)} nuclei; expected {EXPECTED_NUCLEI_PER_SAMPLE}"
            )
        first = subset[0]
        for field in invariant + tuple(fold_fields):
            values = {row.get(field, "") for row in subset}
            if len(values) != 1:
                raise RuntimeError(
                    f"Pilot Sample Name {sample_name!r} is not invariant for {field}: {sorted(values)}"
                )
        record = {field: first.get(field, "") for field in invariant + tuple(fold_fields)}
        record["pilot_nuclei"] = str(len(subset))
        result.append(record)
    return result


def median_or_blank(values: list[float]) -> float | str:
    return median(values) if values else ""


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
        default=str(metadata_root / "pilot_v1" / "reference_phenotypes_v2"),
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    raw_path = Path(args.raw_phenotypes).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (manifest_path, raw_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest_sha = sha256_file(manifest_path)
    raw_sha = sha256_file(raw_path)
    if manifest_sha != FROZEN_MANIFEST_SHA256:
        raise RuntimeError(
            f"Frozen pilot manifest SHA256 mismatch: expected {FROZEN_MANIFEST_SHA256}, got {manifest_sha}"
        )
    if raw_sha != FROZEN_RAW_PHENO_SHA256:
        raise RuntimeError(
            f"Raw phenotype SHA256 mismatch: expected {FROZEN_RAW_PHENO_SHA256}, got {raw_sha}"
        )

    manifest = read_csv(manifest_path)
    raw = read_csv(raw_path)
    if len(manifest) != EXPECTED_PILOT_NUCLEI:
        raise RuntimeError(f"Pilot manifest has {len(manifest)} nuclei; expected {EXPECTED_PILOT_NUCLEI}")

    pilot_samples = unique_manifest_samples(manifest)
    if len(pilot_samples) != EXPECTED_PILOT_SAMPLES:
        raise RuntimeError(
            f"Pilot has {len(pilot_samples)} unique Sample Names; expected {EXPECTED_PILOT_SAMPLES}"
        )

    raw_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        raw_by_sample[norm_text(row["Sample Name"])].append(row)

    reference_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for pilot in pilot_samples:
        sample_name = pilot["sample_name"]
        matches = raw_by_sample.get(norm_text(sample_name), [])
        if len(matches) != 1:
            failures.append(f"{sample_name}: raw Sample Name matches={len(matches)}")
            continue
        nasa = matches[0]

        checks = {
            "source/UserID": norm_text(pilot["source_name"]) == norm_text(nasa["UserID"]),
            "sex/Gender": norm_sex(pilot["sex"]) == norm_sex(nasa["Gender"]),
            "plate": norm_plate_or_well(pilot["plate"]) == norm_plate_or_well(nasa["plate"]),
            "well": norm_plate_or_well(pilot["well"]) == norm_plate_or_well(nasa["plate_well"]),
            "dose": norm_number(pilot["dose_Gy"]) == norm_number(nasa["dose..Gy."]),
            "time": norm_hour(pilot["hr_post_exposure"]) == norm_hour(nasa["timepoint..hr."]),
            "strain_alias": STRAIN_ALIAS.get(pilot["strain"], pilot["strain"]) == nasa["Strain"],
            "radiation_branch": norm_branch(pilot["particle_type"]) == norm_branch(nasa["radiation"]),
        }
        bad = [name for name, passed in checks.items() if not passed]
        if bad:
            failures.append(
                f"{sample_name}: failed {bad}; pilot={{source:{pilot['source_name']}, strain:{pilot['strain']}, "
                f"sex:{pilot['sex']}, plate:{pilot['plate']}, well:{pilot['well']}, branch:{pilot['particle_type']}, "
                f"dose:{pilot['dose_Gy']}, time:{pilot['hr_post_exposure']}}}; "
                f"NASA={{UserID:{nasa['UserID']}, Strain:{nasa['Strain']}, Gender:{nasa['Gender']}, "
                f"plate:{nasa['plate']}, well:{nasa['plate_well']}, radiation:{nasa['radiation']}, "
                f"dose:{nasa['dose..Gy.']}, time:{nasa['timepoint..hr.']}}}"
            )
            continue

        record: dict[str, Any] = dict(pilot)
        record.update(
            {
                "reference_level": "sample_plate_well_aggregate",
                "reference_primary": "avg_nfoci",
                "reference_secondary": "avg_foci_no_outl",
                "sham_branch_semantics": (
                    "dose_0_branch_labeled_not_exposed" if norm_number(pilot["dose_Gy"]) == "0" else "irradiated_branch"
                ),
                "nasa_user_id": nasa["UserID"],
                "nasa_raw_strain": nasa["Strain"],
                "nasa_raw_radiation": nasa["radiation"],
                "resolved_branch": norm_branch(nasa["radiation"]),
            }
        )
        for field in RAW_REFERENCE_FIELDS:
            record[f"nasa_{field}"] = nasa.get(field, "")
        reference_rows.append(record)

    if failures:
        print("REFERENCE LEVEL FROZEN: NO")
        print(f"validation failures: {len(failures)}")
        for failure in failures[:30]:
            print(f"  {failure}")
        if len(failures) > 30:
            print(f"  ... {len(failures) - 30} additional failures")
        return 1

    if len(reference_rows) != EXPECTED_PILOT_SAMPLES:
        raise RuntimeError(
            f"Validated reference rows={len(reference_rows)}; expected {EXPECTED_PILOT_SAMPLES}"
        )

    ref_by_sample = {row["sample_name"]: row for row in reference_rows}
    nucleus_rows: list[dict[str, Any]] = []
    for row in manifest:
        ref = ref_by_sample[row["sample_name"]]
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

    by_condition: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in reference_rows:
        key = (
            norm_branch(row["particle_type"]),
            norm_number(row["dose_Gy"]),
            norm_hour(row["hr_post_exposure"]),
        )
        by_condition[key].append(row)

    condition_rows: list[dict[str, Any]] = []
    for (branch, dose, hour), subset in sorted(by_condition.items()):
        nfoci = [float(row["nasa_avg_nfoci"]) for row in subset]
        no_outl = [float(row["nasa_avg_foci_no_outl"]) for row in subset]
        n_nuc = [int(float(row["nasa_num_nuc"])) for row in subset]
        condition_rows.append(
            {
                "branch": branch,
                "dose_Gy": dose,
                "timepoint_hr": hour,
                "biological_sources": len(subset),
                "nasa_total_nuclei": sum(n_nuc),
                "source_median_avg_nfoci": median_or_blank(nfoci),
                "source_min_avg_nfoci": min(nfoci),
                "source_max_avg_nfoci": max(nfoci),
                "source_median_avg_foci_no_outl": median_or_blank(no_outl),
            }
        )

    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in reference_rows:
        lookup[(
            row["source_name"],
            norm_branch(row["particle_type"]),
            norm_number(row["dose_Gy"]),
            norm_hour(row["hr_post_exposure"]),
        )] = row

    delta_rows: list[dict[str, Any]] = []
    exposed_dose = {"fe": "0.82", "x-ray": "1"}
    sources = sorted({row["source_name"] for row in reference_rows})
    for source in sources:
        for branch, dose in exposed_dose.items():
            for hour in ("4", "24", "48"):
                sham = lookup[(source, branch, "0", hour)]
                exposed = lookup[(source, branch, dose, hour)]
                delta_rows.append(
                    {
                        "source_name": source,
                        "strain": exposed["strain"],
                        "sex": exposed["sex"],
                        "branch": branch,
                        "dose_Gy": dose,
                        "timepoint_hr": hour,
                        "sham_avg_nfoci": float(sham["nasa_avg_nfoci"]),
                        "irradiated_avg_nfoci": float(exposed["nasa_avg_nfoci"]),
                        "delta_avg_nfoci": float(exposed["nasa_avg_nfoci"]) - float(sham["nasa_avg_nfoci"]),
                        "sham_avg_foci_no_outl": float(sham["nasa_avg_foci_no_outl"]),
                        "irradiated_avg_foci_no_outl": float(exposed["nasa_avg_foci_no_outl"]),
                        "delta_avg_foci_no_outl": float(exposed["nasa_avg_foci_no_outl"])
                        - float(sham["nasa_avg_foci_no_outl"]),
                    }
                )

    sample_path = output_dir / "pilot_v1_sample_reference_frozen.csv"
    nucleus_path = output_dir / "pilot_v1_nucleus_to_aggregate_reference_frozen.csv"
    condition_path = output_dir / "pilot_v1_condition_reference_summary_frozen.csv"
    delta_path = output_dir / "pilot_v1_matched_radiation_deltas_frozen.csv"

    write_csv(sample_path, reference_rows)
    write_csv(nucleus_path, nucleus_rows)
    write_csv(condition_path, condition_rows)
    write_csv(delta_path, delta_rows)

    num_nuc = [int(float(row["nasa_num_nuc"])) for row in reference_rows]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_version": "NASA_BPS_53BP1_PILOT_V1",
        "reference_freeze_version": "v2_branch_aware",
        "reference_level_frozen": True,
        "reference_level": "sample_plate_well_aggregate",
        "primary_reference_endpoint": "avg_nfoci",
        "secondary_reference_endpoint": "avg_foci_no_outl",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "raw_phenotype_table": str(raw_path),
        "raw_phenotype_sha256": raw_sha,
        "pilot_nuclei": len(manifest),
        "pilot_unique_samples": len(reference_rows),
        "nuclei_per_pilot_sample": dict(Counter(row["sample_name"] for row in manifest)),
        "nasa_num_nuc_min": min(num_nuc),
        "nasa_num_nuc_median": median(num_nuc),
        "nasa_num_nuc_max": max(num_nuc),
        "strain_aliases": STRAIN_ALIAS,
        "radiation_semantics": {
            "pilot_BPS_particle_type": "experimental branch label",
            "raw_phenotype_radiation": "experimental branch label",
            "isa_zero_dose": "sham-irradiated",
            "zero_dose_rule": "At dose 0, validate Sample Name/core metadata/dose/time and require raw branch to match BPS particle_type; do not require ISA radiation to be bijective with raw branch.",
            "fe_branch_raw_label": "Fe 600 MeV/n",
            "xray_branch_raw_label": "X-ray",
        },
        "guardrails": [
            "NASA avg_nfoci is a sample/plate-well aggregate over hundreds of nuclei, not a per-nucleus annotation.",
            "Do not assign nasa_avg_nfoci to an individual nucleus as ground-truth focus count.",
            "Model outputs may be aggregated over the 30 pilot nuclei within each sample bag and compared with NASA sample-level endpoints.",
            "Biological validation must remain grouped by OSD Source Name to avoid pseudoreplication.",
            "Raw fluorescence intensity is secondary QC because staining/background intensity shows branch/acquisition effects.",
        ],
        "outputs": {
            "sample_reference": str(sample_path),
            "nucleus_to_aggregate_reference": str(nucleus_path),
            "condition_summary": str(condition_path),
            "matched_radiation_deltas": str(delta_path),
        },
    }
    summary_path = output_dir / "reference_freeze_v2.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Radiation Edge AI - freeze NASA BPS pilot-v1 reference v2")
    print(f"Frozen pilot manifest SHA256: {manifest_sha}")
    print(f"Frozen raw phenotype SHA256: {raw_sha}")
    print(f"unique sample joins validated: {len(reference_rows)}/{EXPECTED_PILOT_SAMPLES}")
    print(f"pilot nuclei linked to aggregate reference: {len(nucleus_rows)}/{EXPECTED_PILOT_NUCLEI}")
    print(f"NASA nuclei/sample min/median/max: {min(num_nuc)}/{median(num_nuc)}/{max(num_nuc)}")
    print("")
    print("[Resolved semantics]")
    print("strain: BALB/cByJ -> BALBC; C57BL/6J -> C57 (globally demonstrated deterministic aliases)")
    print("radiation: validate BPS/raw experimental branch; ISA 0-Gy 'sham-irradiated' is not a branch label")
    print("Fe branch: Fe <-> Fe 600 MeV/n")
    print("X-ray branch: X-ray <-> X-ray")
    print("")
    print("REFERENCE LEVEL FROZEN: YES")
    print("Reference level: sample/plate-well aggregate, not per nucleus")
    print(f"Sample reference: {sample_path}")
    print(f"Nucleus -> aggregate reference: {nucleus_path}")
    print(f"Condition summary: {condition_path}")
    print(f"Matched radiation deltas: {delta_path}")
    print(f"Freeze metadata: {summary_path}")
    print("")
    print("NEXT GATE: freeze the R1 weak-supervision/model policy on native-scale 256x256 nucleus inputs, then begin training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
