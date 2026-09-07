"""Inspect resolved NASA BPS biological groups before freezing train/validation/test splits.

This utility consumes ``fitc_53bp1_osd366_crosswalk.csv`` produced by
``resolve_osd366_groups.py``.  It asks whether OSD Source Name behaves like an
individual biological source/cell line, how many samples and radiation
conditions each source contributes, and whether a leakage-safe source-level
split is feasible.

No microscopy images are downloaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def condition_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("dose_Gy", "")).strip(),
        str(row.get("particle_type", "")).strip(),
        str(row.get("hr_post_exposure", "")).strip(),
    )


def condition_label(key: tuple[str, str, str]) -> str:
    dose, particle, hours = key
    return f"{particle}|{dose}Gy|{hours}h"


def sorted_conditions(rows: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    keys = {condition_key(row) for row in rows}
    return sorted(keys, key=lambda k: (k[1], float(k[0]), float(k[2])))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    default_root = Path(
        os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data")
    ) / "nasa_bps_microscopy" / "metadata"
    parser.add_argument(
        "--crosswalk",
        default=str(default_root / "fitc_53bp1_osd366_crosswalk.csv"),
    )
    parser.add_argument("--output-dir", default=str(default_root / "resolved_group_diagnostics"))
    args = parser.parse_args()

    crosswalk_path = Path(args.crosswalk).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not crosswalk_path.is_file():
        raise FileNotFoundError(crosswalk_path)

    rows = read_csv(crosswalk_path)
    if not rows:
        raise RuntimeError("Crosswalk is empty")

    required = {
        "osd_source_name",
        "osd_sample_name",
        "osd_strain",
        "osd_sex",
        "plate",
        "acquisition",
        "well",
        "dose_Gy",
        "particle_type",
        "hr_post_exposure",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise RuntimeError(f"Crosswalk missing required columns: {missing}")

    conditions = sorted_conditions(rows)
    sources = sorted({r["osd_source_name"] for r in rows if r["osd_source_name"]})
    samples = sorted({r["osd_sample_name"] for r in rows if r["osd_sample_name"]})

    source_rows: list[dict[str, object]] = []
    source_condition_rows: list[dict[str, object]] = []
    source_integrity: dict[str, dict[str, object]] = {}

    for source in sources:
        subset = [r for r in rows if r["osd_source_name"] == source]
        strains = sorted({r["osd_strain"] for r in subset if r["osd_strain"]})
        sexes = sorted({r["osd_sex"] for r in subset if r["osd_sex"]})
        sample_names = sorted({r["osd_sample_name"] for r in subset if r["osd_sample_name"]})
        plates = sorted({r["plate"] for r in subset if r["plate"]})
        acquisitions = sorted({f"{r['plate']}_{r['acquisition']}" for r in subset})
        wells = sorted({f"{r['plate']}_{r['well']}" for r in subset})
        cond_counter = Counter(condition_key(r) for r in subset)

        row: dict[str, object] = {
            "source_name": source,
            "strain": ";".join(strains),
            "sex": ";".join(sexes),
            "n_images": len(subset),
            "n_samples": len(sample_names),
            "n_plates": len(plates),
            "n_plate_acquisitions": len(acquisitions),
            "n_plate_wells": len(wells),
            "n_conditions": len(cond_counter),
            "single_strain": len(strains) == 1,
            "single_sex": len(sexes) <= 1,
        }
        source_rows.append(row)

        source_integrity[source] = {
            "strains": strains,
            "sexes": sexes,
            "sample_names": sample_names,
            "plates": plates,
            "plate_acquisitions": acquisitions,
            "plate_wells": wells,
            "condition_count": len(cond_counter),
        }

        for condition in conditions:
            source_condition_rows.append(
                {
                    "source_name": source,
                    "strain": ";".join(strains),
                    "sex": ";".join(sexes),
                    "dose_Gy": condition[0],
                    "particle_type": condition[1],
                    "hr_post_exposure": condition[2],
                    "n_images": cond_counter.get(condition, 0),
                }
            )

    sample_to_sources: dict[str, set[str]] = defaultdict(set)
    sample_to_strains: dict[str, set[str]] = defaultdict(set)
    sample_to_conditions: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for row in rows:
        sample = row["osd_sample_name"]
        if not sample:
            continue
        sample_to_sources[sample].add(row["osd_source_name"])
        sample_to_strains[sample].add(row["osd_strain"])
        sample_to_conditions[sample].add(condition_key(row))

    ambiguous_samples = {
        sample: {
            "sources": sorted(values),
            "strains": sorted(sample_to_strains[sample]),
        }
        for sample, values in sample_to_sources.items()
        if len(values) != 1 or len(sample_to_strains[sample]) != 1
    }

    condition_source_counts = []
    for condition in conditions:
        present = []
        for source in sources:
            count = sum(
                1
                for row in rows
                if row["osd_source_name"] == source and condition_key(row) == condition
            )
            if count:
                present.append(source)
        condition_source_counts.append(
            {
                "dose_Gy": condition[0],
                "particle_type": condition[1],
                "hr_post_exposure": condition[2],
                "n_sources": len(present),
                "sources": ";".join(present),
            }
        )

    strain_sources: dict[str, list[str]] = defaultdict(list)
    for row in source_rows:
        strain = str(row["strain"])
        if strain:
            strain_sources[strain].append(str(row["source_name"]))

    all_sources_single_strain = all(bool(r["single_strain"]) for r in source_rows)
    all_samples_single_source = not ambiguous_samples
    minimum_condition_sources = min(int(r["n_sources"]) for r in condition_source_counts)
    complete_sources = [
        str(r["source_name"]) for r in source_rows if int(r["n_conditions"]) == len(conditions)
    ]

    recommendation = {
        "source_name_candidate_for_primary_grouping": bool(
            all_sources_single_strain and all_samples_single_source and len(sources) >= 4
        ),
        "n_sources": len(sources),
        "n_samples": len(samples),
        "n_strains": len(strain_sources),
        "n_conditions": len(conditions),
        "minimum_sources_per_condition": minimum_condition_sources,
        "sources_with_all_conditions": complete_sources,
        "primary_validation": (
            "source-held-out (individual biological source/cell-line level), preferably repeated/grouped CV"
        ),
        "secondary_validation": (
            "strain-held-out as a domain-shift stress test only; with two strains it is not a broad unseen-strain benchmark"
        ),
        "warning": (
            "Do not split individual nuclei or sample wells randomly across train/test; preserve OSD Source Name groups."
        ),
    }

    source_summary_path = output_dir / "source_summary.csv"
    coverage_path = output_dir / "source_condition_coverage.csv"
    condition_path = output_dir / "condition_source_counts.csv"
    summary_path = output_dir / "summary.json"
    write_csv(source_summary_path, source_rows)
    write_csv(coverage_path, source_condition_rows)
    write_csv(condition_path, condition_source_counts)
    summary_path.write_text(
        json.dumps(
            {
                "crosswalk": str(crosswalk_path),
                "source_integrity": source_integrity,
                "ambiguous_samples": ambiguous_samples,
                "strain_sources": dict(strain_sources),
                "recommendation": recommendation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Radiation Edge AI - NASA BPS resolved biological-group diagnostics")
    print("Metadata only: no microscopy images are downloaded.")
    print("")
    print("[Resolved hierarchy]")
    print(f"FITC images: {len(rows)}")
    print(f"OSD Source Names: {len(sources)}")
    print(f"OSD Sample Names: {len(samples)}")
    print(f"strains: {len(strain_sources)}")
    print(f"radiation condition cells: {len(conditions)}")
    print(f"ambiguous Sample Name -> Source/strain mappings: {len(ambiguous_samples)}")
    print("")

    print("[Source Name summary]")
    for row in source_rows:
        print(
            f"{row['source_name']}: strain={row['strain']} sex={row['sex'] or '?'} "
            f"images={row['n_images']} samples={row['n_samples']} "
            f"conditions={row['n_conditions']}/{len(conditions)} plates={row['n_plates']}"
        )
    print("")

    print("[Sources by strain]")
    for strain, values in sorted(strain_sources.items()):
        print(f"{strain}: {len(values)} source(s) -> {', '.join(values)}")
    print("")

    print("[Condition-level source coverage]")
    for row in condition_source_counts:
        print(
            f"{row['particle_type']:<5} {row['dose_Gy']:>4} Gy {row['hr_post_exposure']:>3} h: "
            f"{row['n_sources']}/{len(sources)} source(s)"
        )
    print("")

    print("[Split recommendation]")
    print(
        "Source Name suitable as primary grouping candidate: "
        + ("YES" if recommendation["source_name_candidate_for_primary_grouping"] else "NOT YET")
    )
    print(f"minimum source coverage of any condition: {minimum_condition_sources}/{len(sources)}")
    print(f"sources with all {len(conditions)} conditions: {len(complete_sources)}")
    print("Primary: source-held-out grouped validation; never random nucleus-level splitting.")
    print("Secondary: strain-held-out domain-shift stress test (only two strains in benchmark subset).")
    print("")
    print("NASA BPS RESOLVED GROUP DIAGNOSTICS COMPLETE: YES")
    print(f"Source summary: {source_summary_path}")
    print(f"Coverage matrix: {coverage_path}")
    print(f"Condition source counts: {condition_path}")
    print(f"Summary: {summary_path}")
    print("")
    print(
        "NEXT GATE: if Source Name is internally consistent and condition coverage is adequate, "
        "freeze a small source-held-out radiation-native pilot and only then download those images."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
