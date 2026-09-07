"""Validate semantic alias mappings between OSD-366 ISA metadata and LSDS-111 phenotypes.

The pilot-v1 phenotype crosswalk already matches all 72 Sample Names uniquely, but
strict metadata checks fail for two controlled-vocabulary fields:

- ISA strain names such as ``BALB/cByJ`` and ``C57BL/6J`` versus shortened raw
  phenotype labels such as ``BALBC`` and ``C57``;
- ISA/BPS particle label ``Fe`` versus raw phenotype label ``Fe 600 MeV/n``.

This diagnostic tests whether those are deterministic aliases rather than true
biological inconsistencies.  It joins the full OSD-366 ISA sample table to the
current LSDS-111 raw phenotype table by Sample Name and reports:

- unique/global Sample Name coverage;
- exact agreement for source/UserID, sex, plate, well, dose and time;
- ISA -> raw and raw -> ISA mappings for strain and radiation;
- the same checks restricted to the 72 frozen pilot Sample Names.

No data are modified and no network access is used.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def norm_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def find_header(headers: list[str], aliases: tuple[str, ...]) -> int | None:
    wanted = {norm_header(alias) for alias in aliases}
    for index, header in enumerate(headers):
        if norm_header(header) in wanted:
            return index
    return None


def norm_text(value: str) -> str:
    return "".join(str(value).strip().lower().split())


def norm_sex(value: str) -> str:
    text = norm_text(value)
    return {"f": "female", "female": "female", "m": "male", "male": "male"}.get(text, text)


def norm_number(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return f"{float(text):.12g}"


def norm_hour(value: str) -> str:
    return norm_number(value)


def norm_plate_or_well(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def choose_sample_table(isa_zip: Path) -> tuple[str, list[str], list[list[str]]]:
    with zipfile.ZipFile(isa_zip, "r") as archive:
        candidates = [
            name
            for name in archive.namelist()
            if Path(name).name.lower().startswith("s_") and name.lower().endswith(".txt")
        ]
        if not candidates:
            raise RuntimeError("No s_*.txt sample table found in ISA ZIP")

        scored: list[tuple[int, str, list[str], list[list[str]]]] = []
        for name in candidates:
            text = archive.read(name).decode("utf-8-sig", errors="replace")
            rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
            if not rows:
                continue
            headers = rows[0]
            score = sum(
                find_header(headers, aliases) is not None
                for aliases in (
                    ("Sample Name",),
                    ("Source Name",),
                    ("Factor Value: Strain",),
                    ("Factor Value: Ionizing Radiation",),
                    ("Factor Value: Absorbed Radiation Dose",),
                    ("Factor Value: Time of Sample Collection After Treatment",),
                    ("Comment: plate", "Comment[plate]"),
                    ("Comment: plate well", "Comment[plate well]"),
                )
            )
            scored.append((score, name, headers, rows[1:]))

    if not scored:
        raise RuntimeError("Could not parse any sample table from ISA ZIP")
    scored.sort(key=lambda item: item[0], reverse=True)
    score, name, headers, rows = scored[0]
    if score < 6:
        raise RuntimeError(f"Best sample table lacks required metadata fields: {name}, score={score}")
    return name, headers, rows


def extract_isa_rows(isa_zip: Path) -> tuple[str, list[dict[str, str]]]:
    table_name, headers, rows = choose_sample_table(isa_zip)
    aliases = {
        "sample_name": ("Sample Name",),
        "source_name": ("Source Name",),
        "strain": ("Factor Value: Strain",),
        "sex": ("Characteristics: Sex",),
        "radiation": ("Factor Value: Ionizing Radiation",),
        "dose": ("Factor Value: Absorbed Radiation Dose",),
        "time": ("Factor Value: Time of Sample Collection After Treatment",),
        "plate": ("Comment: plate", "Comment[plate]"),
        "well": ("Comment: plate well", "Comment[plate well]"),
    }
    indices = {key: find_header(headers, values) for key, values in aliases.items()}
    required = ("sample_name", "source_name", "strain", "radiation", "dose", "time", "plate", "well")
    missing = [key for key in required if indices[key] is None]
    if missing:
        raise RuntimeError(f"ISA sample table missing required fields: {missing}")

    extracted: list[dict[str, str]] = []
    for row in rows:
        record: dict[str, str] = {}
        for key, index in indices.items():
            if index is None or index >= len(row):
                record[key] = ""
            else:
                record[key] = str(row[index]).strip()
        if not record["sample_name"]:
            continue
        extracted.append(record)
    return table_name, extracted


def unique_by(rows: list[dict[str, str]], field: str) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = norm_text(row.get(field, ""))
        if key:
            grouped[key].append(row)
    unique = {key: subset[0] for key, subset in grouped.items() if len(subset) == 1}
    duplicate_counts = {key: len(subset) for key, subset in grouped.items() if len(subset) > 1}
    return unique, duplicate_counts


def relation_summary(pairs: Counter[tuple[str, str]]) -> dict[str, Any]:
    left_to_right: dict[str, set[str]] = defaultdict(set)
    right_to_left: dict[str, set[str]] = defaultdict(set)
    for (left, right), _count in pairs.items():
        left_to_right[left].add(right)
        right_to_left[right].add(left)
    return {
        "pairs": [
            {"isa": left, "raw": right, "n": count}
            for (left, right), count in sorted(pairs.items(), key=lambda item: (-item[1], item[0]))
        ],
        "isa_to_raw_is_function": all(len(values) == 1 for values in left_to_right.values()),
        "raw_to_isa_is_function": all(len(values) == 1 for values in right_to_left.values()),
        "isa_values": len(left_to_right),
        "raw_values": len(right_to_left),
    }


def compare_subset(
    sample_names: set[str],
    isa_unique: dict[str, dict[str, str]],
    raw_unique: dict[str, dict[str, str]],
) -> dict[str, Any]:
    common = sorted(sample_names & set(isa_unique) & set(raw_unique))
    core_fields = (
        ("source", "source_name", "UserID", norm_text),
        ("sex", "sex", "Gender", norm_sex),
        ("plate", "plate", "plate", norm_plate_or_well),
        ("well", "well", "plate_well", norm_plate_or_well),
        ("dose", "dose", "dose..Gy.", norm_number),
        ("time", "time", "timepoint..hr.", norm_hour),
    )
    mismatch_counts: Counter[str] = Counter()
    mismatch_examples: list[dict[str, str]] = []
    strain_pairs: Counter[tuple[str, str]] = Counter()
    radiation_pairs: Counter[tuple[str, str]] = Counter()

    for key in common:
        isa = isa_unique[key]
        raw = raw_unique[key]
        for label, isa_field, raw_field, normalizer in core_fields:
            left_raw = isa.get(isa_field, "")
            right_raw = raw.get(raw_field, "")
            left = normalizer(left_raw)
            right = normalizer(right_raw)
            if left and right and left != right:
                mismatch_counts[label] += 1
                if len(mismatch_examples) < 30:
                    mismatch_examples.append(
                        {
                            "sample_name": isa.get("sample_name", ""),
                            "field": label,
                            "isa": left_raw,
                            "raw": right_raw,
                        }
                    )
        strain_pairs[(isa.get("strain", ""), raw.get("Strain", ""))] += 1
        radiation_pairs[(isa.get("radiation", ""), raw.get("radiation", ""))] += 1

    strain_summary = relation_summary(strain_pairs)
    radiation_summary = relation_summary(radiation_pairs)
    core_ok = not mismatch_counts
    aliases_deterministic = (
        strain_summary["isa_to_raw_is_function"]
        and strain_summary["raw_to_isa_is_function"]
        and radiation_summary["isa_to_raw_is_function"]
        and radiation_summary["raw_to_isa_is_function"]
    )

    return {
        "requested_samples": len(sample_names),
        "matched_unique_samples": len(common),
        "missing_from_unique_join": len(sample_names) - len(common),
        "core_metadata_mismatch_counts": dict(mismatch_counts),
        "core_metadata_mismatch_examples": mismatch_examples,
        "strain_mapping": strain_summary,
        "radiation_mapping": radiation_summary,
        "core_metadata_consistent": core_ok,
        "alias_mappings_deterministic": aliases_deterministic,
        "safe_alias_evidence": len(common) == len(sample_names) and core_ok and aliases_deterministic,
    }


def print_mapping(title: str, mapping: dict[str, Any]) -> None:
    print(title)
    for row in mapping["pairs"]:
        print(f"  {row['n']:>5}  ISA={row['isa']!r}  -> raw={row['raw']!r}")
    print(
        "  deterministic ISA->raw: "
        f"{'YES' if mapping['isa_to_raw_is_function'] else 'NO'}; "
        "raw->ISA: "
        f"{'YES' if mapping['raw_to_isa_is_function'] else 'NO'}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"
    parser.add_argument(
        "--isa-zip",
        default=str(metadata_root / "osd366" / "OSD-366_metadata_OSD-366-ISA.zip"),
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
        "--pilot-manifest",
        default=str(metadata_root / "pilot_v1" / "pilot_v1_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(metadata_root / "pilot_v1" / "reference_phenotypes" / "alias_validation_summary.json"),
    )
    args = parser.parse_args()

    isa_zip = Path(args.isa_zip).resolve()
    raw_path = Path(args.raw_phenotypes).resolve()
    pilot_path = Path(args.pilot_manifest).resolve()
    output_path = Path(args.output).resolve()
    for path in (isa_zip, raw_path, pilot_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    table_name, isa_rows = extract_isa_rows(isa_zip)
    raw_rows = read_csv(raw_path)
    pilot_rows = read_csv(pilot_path)

    isa_unique, isa_duplicates = unique_by(isa_rows, "sample_name")
    raw_unique, raw_duplicates = unique_by(raw_rows, "Sample Name")
    all_common = set(isa_unique) & set(raw_unique)
    pilot_names = {norm_text(row["sample_name"]) for row in pilot_rows if row.get("sample_name", "").strip()}

    global_result = compare_subset(all_common, isa_unique, raw_unique)
    pilot_result = compare_subset(pilot_names, isa_unique, raw_unique)

    print("Radiation Edge AI - validate OSD-366 reference metadata aliases")
    print(f"ISA sample table: {table_name}")
    print(f"ISA rows with Sample Name: {len(isa_rows)}")
    print(f"raw phenotype rows: {len(raw_rows)}")
    print(f"unique ISA Sample Names: {len(isa_unique)}")
    print(f"unique raw Sample Names: {len(raw_unique)}")
    print(f"unique Sample Names shared globally: {len(all_common)}")
    print(f"duplicate ISA Sample Names: {len(isa_duplicates)}")
    print(f"duplicate raw Sample Names: {len(raw_duplicates)}")
    print("")

    print("[Global exact-identifier join]")
    print(f"joined unique samples: {global_result['matched_unique_samples']}")
    print(f"core metadata mismatches: {global_result['core_metadata_mismatch_counts'] or 'none'}")
    print_mapping("strain aliases:", global_result["strain_mapping"])
    print_mapping("radiation aliases:", global_result["radiation_mapping"])
    print("")

    print("[Frozen pilot 72-sample join]")
    print(f"pilot unique Sample Names: {len(pilot_names)}")
    print(f"joined unique samples: {pilot_result['matched_unique_samples']}/{len(pilot_names)}")
    print(f"core metadata mismatches: {pilot_result['core_metadata_mismatch_counts'] or 'none'}")
    print_mapping("pilot strain aliases:", pilot_result["strain_mapping"])
    print_mapping("pilot radiation aliases:", pilot_result["radiation_mapping"])
    print("")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "isa_sample_table": table_name,
        "isa_rows": len(isa_rows),
        "raw_rows": len(raw_rows),
        "unique_isa_sample_names": len(isa_unique),
        "unique_raw_sample_names": len(raw_unique),
        "global_unique_shared_sample_names": len(all_common),
        "duplicate_isa_sample_names": len(isa_duplicates),
        "duplicate_raw_sample_names": len(raw_duplicates),
        "global": global_result,
        "pilot": pilot_result,
        "interpretation": [
            "Sample Name is the primary exact join key.",
            "Source/UserID, sex, plate, well, dose and time must agree exactly after notation normalization.",
            "Strain and radiation aliases are accepted only if the observed mappings are deterministic in both directions.",
            "This diagnostic does not change the frozen reference builder or training labels.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Summary: {output_path}")
    if pilot_result["safe_alias_evidence"]:
        print("ALIAS VALIDATION: PILOT SAFE = YES")
        if global_result["safe_alias_evidence"]:
            print("ALIAS VALIDATION: GLOBAL JOIN SAFE = YES")
        else:
            print("ALIAS VALIDATION: GLOBAL JOIN SAFE = NO (pilot may still be valid; inspect global diagnostics)")
        print("NEXT GATE: patch the reference builder with only the empirically demonstrated explicit aliases, then rerun the freeze.")
        return 0

    print("ALIAS VALIDATION: PILOT SAFE = NO")
    print("NEXT GATE: inspect missing joins, core metadata mismatches, or non-deterministic alias mappings before freezing R1.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
