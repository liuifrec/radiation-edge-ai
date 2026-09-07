"""Diagnose systematic metadata mismatches in the NASA BPS pilot-v1 phenotype crosswalk.

The sample-name crosswalk can be exact while one metadata field differs only in
notation (for example strain punctuation or radiation naming).  This diagnostic
repeats the 72 sample-level comparisons and reports, by field:

- strict mismatch counts using the current reference-builder normalizers;
- the most common raw value pairs;
- whether each mismatch disappears under a conservative relaxed normalizer;
- the first examples that remain genuinely discordant after relaxed matching.

No files are modified and no network access is used.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def norm_text(value: str) -> str:
    return "".join(str(value).strip().lower().split())


def norm_number(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    value_f = float(text)
    return f"{value_f:.12g}"


def norm_hour(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    value_f = float(text)
    return f"{value_f:.12g}"


def norm_sex(value: str) -> str:
    text = norm_text(value)
    return {"f": "female", "female": "female", "m": "male", "male": "male"}.get(text, text)


def norm_radiation(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())
    aliases = {
        "xray": "xray",
        "xrays": "xray",
        "xirradiation": "xray",
        "fe": "fe",
        "iron": "fe",
        "56fe": "fe",
        "fe56": "fe",
    }
    return aliases.get(text, text)


def strict_strain(value: str) -> str:
    return norm_text(value)


def relaxed_strain(value: str) -> str:
    # Conservative notation-only relaxation: remove separators/punctuation,
    # retain all alphanumeric strain tokens.
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def strict_plate_or_well(value: str) -> str:
    return norm_text(value)


def relaxed_plate(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def relaxed_well(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())
    match = re.fullmatch(r"([a-z]+)0*(\d+)", text)
    if match:
        return f"{match.group(1)}{int(match.group(2))}"
    return text


def unique_manifest_samples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_name"]].append(row)
    out: list[dict[str, str]] = []
    for sample_name, subset in sorted(by_sample.items()):
        first = subset[0]
        out.append(first)
    return out


CHECKS = (
    ("source_name", "UserID", norm_text, norm_text),
    ("strain", "Strain", strict_strain, relaxed_strain),
    ("sex", "Gender", norm_sex, norm_sex),
    ("plate", "plate", strict_plate_or_well, relaxed_plate),
    ("well", "plate_well", strict_plate_or_well, relaxed_well),
    ("particle_type", "radiation", norm_radiation, norm_radiation),
    ("dose_Gy", "dose..Gy.", norm_number, norm_number),
    ("hr_post_exposure", "timepoint..hr.", norm_hour, norm_hour),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"
    parser.add_argument("--manifest", default=str(metadata_root / "pilot_v1" / "pilot_v1_manifest.csv"))
    parser.add_argument(
        "--raw-phenotypes",
        default=str(metadata_root / "current_phenotypes" / "files" / "LSDS-111_immunostaining_Raw_pheno_V3.csv"),
    )
    args = parser.parse_args()

    manifest = read_csv(Path(args.manifest).resolve())
    raw = read_csv(Path(args.raw_phenotypes).resolve())
    pilot_samples = unique_manifest_samples(manifest)

    raw_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        raw_by_sample[norm_text(row["Sample Name"])].append(row)

    strict_counts: Counter[str] = Counter()
    relaxed_counts: Counter[str] = Counter()
    pair_counts: dict[str, Counter[tuple[str, str]]] = {p: Counter() for p, *_ in CHECKS}
    unresolved_examples: list[tuple[str, str, str, str, str]] = []
    sample_strict_mismatch_counts: Counter[int] = Counter()

    matched = 0
    unmatched = 0
    ambiguous = 0
    for pilot in pilot_samples:
        matches = raw_by_sample.get(norm_text(pilot["sample_name"]), [])
        if not matches:
            unmatched += 1
            continue
        if len(matches) != 1:
            ambiguous += 1
            continue
        matched += 1
        nasa = matches[0]
        sample_mismatch = 0

        for pilot_field, raw_field, strict_norm, relaxed_norm in CHECKS:
            left_raw = pilot.get(pilot_field, "")
            right_raw = nasa.get(raw_field, "")
            left = strict_norm(left_raw)
            right = strict_norm(right_raw)
            if left and right and left != right:
                sample_mismatch += 1
                strict_counts[pilot_field] += 1
                pair_counts[pilot_field][(left_raw, right_raw)] += 1
                left_relaxed = relaxed_norm(left_raw)
                right_relaxed = relaxed_norm(right_raw)
                if left_relaxed and right_relaxed and left_relaxed != right_relaxed:
                    relaxed_counts[pilot_field] += 1
                    if len(unresolved_examples) < 30:
                        unresolved_examples.append(
                            (pilot["sample_name"], pilot_field, left_raw, raw_field, right_raw)
                        )
        sample_strict_mismatch_counts[sample_mismatch] += 1

    print("Radiation Edge AI - diagnose NASA pilot-v1 reference metadata mismatches")
    print(f"pilot unique Sample Names: {len(pilot_samples)}")
    print(f"exact unique Sample Name matches: {matched}/{len(pilot_samples)}")
    print(f"unmatched: {unmatched}")
    print(f"ambiguous: {ambiguous}")
    print("")
    print("[Strict mismatch counts by field]")
    for pilot_field, raw_field, _strict, _relaxed in CHECKS:
        count = strict_counts[pilot_field]
        print(f"{pilot_field:18s} vs {raw_field:18s}: {count}/{matched}")
    print("")
    print(f"strict mismatches per sample distribution: {dict(sorted(sample_strict_mismatch_counts.items()))}")
    print("")

    print("[Most common strict mismatch value pairs]")
    for pilot_field, raw_field, _strict, _relaxed in CHECKS:
        if not pair_counts[pilot_field]:
            continue
        print(f"{pilot_field} vs {raw_field}:")
        for (left, right), count in pair_counts[pilot_field].most_common(12):
            print(f"  {count:>3}  pilot={left!r}  NASA={right!r}")
    print("")

    print("[Mismatch counts after conservative relaxed normalization]")
    for pilot_field, raw_field, _strict, _relaxed in CHECKS:
        count = relaxed_counts[pilot_field]
        if strict_counts[pilot_field] or count:
            resolved = strict_counts[pilot_field] - count
            print(
                f"{pilot_field:18s}: unresolved={count}/{matched} "
                f"resolved_as_notation_only={resolved}"
            )
    print("")

    if unresolved_examples:
        print("[First genuinely unresolved examples]")
        for sample, pilot_field, left, raw_field, right in unresolved_examples:
            print(f"{sample}: {pilot_field}={left!r} != {raw_field}={right!r}")
    else:
        print("No mismatches remain after conservative relaxed normalization.")
    print("")

    if matched == len(pilot_samples) and not any(relaxed_counts.values()):
        print("DIAGNOSIS: all 72 sample matches are biologically consistent; strict failures are notation-only.")
        print("NEXT GATE: update the reference builder to use the demonstrated field-specific normalizers and rerun the freeze.")
        return 0

    print("DIAGNOSIS: one or more metadata fields remain genuinely discordant after notation normalization.")
    print("NEXT GATE: inspect the unresolved field/value pairs before freezing the reference hierarchy.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
