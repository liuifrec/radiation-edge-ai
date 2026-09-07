"""Freeze a small leakage-safe NASA BPS 53BP1 pilot without downloading images.

Pilot v1 is deliberately radiation-native and source-held-out. It uses the six
OSD Source Names that cover all 17 benchmark condition cells, then selects a
balanced subset spanning:

- sham versus irradiated;
- X ray versus high-LET Fe;
- early/intermediate/late repair (4/24/48 h);
- both benchmark strains;
- both sexes represented in the six complete biological sources.

The frozen condition panel is:
- Fe: 0 and 0.82 Gy at 4, 24, 48 h;
- X-ray: 0 and 1.0 Gy at 4, 24, 48 h.

Thirty nuclei per source x condition are selected by deterministic SHA256 rank,
for 6 x 12 x 30 = 2160 nuclei when all pairs are available. Only nuclei with a
FITC/53BP1 image plus both DAPI and MASK partners are eligible.

Validation is defined as three paired source-held-out folds. Each held-out pair
contains one BALB/cByJ and one C57BL/6J source. No microscopy images are
fetched by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

SEED = "nasa-bps-pilot-v1-720"
N_PER_SOURCE_CONDITION = 30

COMPLETE_SOURCES = (
    "BALBCF1",
    "BALBCM1",
    "BALBCM2",
    "C57BLF1",
    "C57BLM1",
    "C57BLM3",
)

# Three paired source-held-out folds, balanced by strain within every test fold.
FOLDS = {
    "fold_female": ("BALBCF1", "C57BLF1"),
    "fold_male_1": ("BALBCM1", "C57BLM1"),
    "fold_male_2": ("BALBCM2", "C57BLM3"),
}

PILOT_CONDITIONS = (
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

FILENAME_RE = re.compile(
    r"^(?P<plate>P\d+)_(?P<acquisition>\d+)-(?P<well>[A-Za-z]+\d+)_"
    r"(?P<field>\d+)_(?P<object>\d+)_(?P<channel>proj|DAPI|MASK)\.tif$",
    re.IGNORECASE,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def normalize_number(value: str) -> str:
    text = str(value).strip()
    if text == "":
        return text
    number = float(text)
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


def deterministic_rank(nucleus_key: str) -> str:
    return hashlib.sha256(f"{SEED}|{nucleus_key}".encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    default_root = Path(
        os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data")
    ) / "nasa_bps_microscopy" / "metadata"
    parser.add_argument(
        "--crosswalk",
        default=str(default_root / "fitc_53bp1_osd366_crosswalk.csv"),
    )
    parser.add_argument(
        "--dapi-mask-meta",
        default=str(default_root / "meta_DAPI_MASK.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_root / "pilot_v1"),
    )
    parser.add_argument("--n-per-source-condition", type=int, default=N_PER_SOURCE_CONDITION)
    args = parser.parse_args()

    crosswalk_path = Path(args.crosswalk).resolve()
    dapi_path = Path(args.dapi_mask_meta).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (crosswalk_path, dapi_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    fitc_rows = read_csv(crosswalk_path)
    dapi_rows = read_csv(dapi_path)

    dapi_by_key: dict[str, dict[str, str]] = defaultdict(dict)
    for row in dapi_rows:
        parsed = parse_filename(row["filename"])
        if parsed["channel"] in {"dapi", "mask"}:
            dapi_by_key[parsed["nucleus_key"]][parsed["channel"]] = row["filename"]

    expected_conditions = set(PILOT_CONDITIONS)
    available_sources = {r["osd_source_name"] for r in fitc_rows}
    missing_sources = [source for source in COMPLETE_SOURCES if source not in available_sources]
    if missing_sources:
        raise RuntimeError(f"Expected complete sources missing from crosswalk: {missing_sources}")

    eligible: dict[tuple[str, tuple[str, str, str]], list[dict[str, str]]] = defaultdict(list)
    missing_pair_count = 0
    for row in fitc_rows:
        source = row["osd_source_name"]
        if source not in COMPLETE_SOURCES:
            continue
        cond = condition_key(row)
        if cond not in expected_conditions:
            continue
        parsed = parse_filename(row["filename"])
        partners = dapi_by_key.get(parsed["nucleus_key"], {})
        if "dapi" not in partners or "mask" not in partners:
            missing_pair_count += 1
            continue
        enriched = dict(row)
        enriched["nucleus_key"] = parsed["nucleus_key"]
        enriched["dapi_filename"] = partners["dapi"]
        enriched["mask_filename"] = partners["mask"]
        enriched["rank_sha256"] = deterministic_rank(parsed["nucleus_key"])
        eligible[(source, cond)].append(enriched)

    selected: list[dict[str, str]] = []
    short_cells: list[dict[str, object]] = []
    for source in COMPLETE_SOURCES:
        for cond in PILOT_CONDITIONS:
            candidates = sorted(
                eligible.get((source, cond), []),
                key=lambda row: (row["rank_sha256"], row["nucleus_key"]),
            )
            if len(candidates) < args.n_per_source_condition:
                short_cells.append(
                    {
                        "source_name": source,
                        "particle_type": cond[0],
                        "dose_Gy": cond[1],
                        "hr_post_exposure": cond[2],
                        "available": len(candidates),
                        "requested": args.n_per_source_condition,
                    }
                )
            selected.extend(candidates[: args.n_per_source_condition])

    if short_cells:
        raise RuntimeError(
            "One or more source x condition cells do not contain enough paired nuclei; "
            f"details={short_cells}"
        )

    fold_membership = {}
    for fold_name, held_out in FOLDS.items():
        for source in COMPLETE_SOURCES:
            fold_membership[(fold_name, source)] = "test" if source in held_out else "train"

    manifest_rows: list[dict[str, object]] = []
    for index, row in enumerate(selected, start=1):
        source = row["osd_source_name"]
        record: dict[str, object] = {
            "pilot_index": index,
            "sample_id": f"BPSV1_{index:05d}",
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
            "selection_rank_sha256": row["rank_sha256"],
        }
        for fold_name in FOLDS:
            record[f"{fold_name}_role"] = fold_membership[(fold_name, source)]
        manifest_rows.append(record)

    manifest_path = output_dir / "pilot_v1_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    counts_source = Counter(row["source_name"] for row in manifest_rows)
    counts_condition = Counter(
        (row["particle_type"], row["dose_Gy"], row["hr_post_exposure"])
        for row in manifest_rows
    )
    counts_strain = Counter(row["strain"] for row in manifest_rows)
    counts_sex = Counter(row["sex"] for row in manifest_rows)

    summary = {
        "pilot_version": "NASA_BPS_53BP1_PILOT_V1",
        "selection_seed": SEED,
        "n_per_source_condition": args.n_per_source_condition,
        "complete_sources": list(COMPLETE_SOURCES),
        "pilot_conditions": [
            {"particle_type": p, "dose_Gy": d, "hr_post_exposure": h}
            for p, d, h in PILOT_CONDITIONS
        ],
        "folds": {name: list(values) for name, values in FOLDS.items()},
        "n_selected_nuclei": len(manifest_rows),
        "expected_n_selected": len(COMPLETE_SOURCES) * len(PILOT_CONDITIONS) * args.n_per_source_condition,
        "missing_paired_nuclei_seen_before_selection": missing_pair_count,
        "counts_by_source": dict(counts_source),
        "counts_by_strain": dict(counts_strain),
        "counts_by_sex": dict(counts_sex),
        "counts_by_condition": {
            f"{p}|{d}Gy|{h}h": count for (p, d, h), count in sorted(counts_condition.items())
        },
        "crosswalk_sha256": sha256_file(crosswalk_path),
        "dapi_mask_metadata_sha256": sha256_file(dapi_path),
        "manifest_sha256": sha256_file(manifest_path),
        "notes": [
            "No microscopy images were downloaded while freezing this manifest.",
            "Every selected nucleus has FITC/53BP1, DAPI and MASK filenames.",
            "Train/test roles are grouped strictly by OSD Source Name.",
            "The pilot intentionally uses only the six sources with all 17 original condition cells.",
            "Pilot v1 uses near-1-Gy Fe versus X-ray contrasts plus sham at 4/24/48 h; low-dose 0.1/0.3 Gy conditions are reserved for expansion.",
        ],
    }
    summary_path = output_dir / "pilot_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fold_path = output_dir / "pilot_v1_folds.json"
    fold_path.write_text(
        json.dumps(
            {
                name: {
                    "test_sources": list(test_sources),
                    "train_sources": [s for s in COMPLETE_SOURCES if s not in test_sources],
                }
                for name, test_sources in FOLDS.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Radiation Edge AI - freeze NASA BPS 53BP1 pilot v1")
    print("Manifest only: no microscopy images were downloaded.")
    print("")
    print("[Design]")
    print(f"complete biological sources: {len(COMPLETE_SOURCES)}")
    print(f"radiation condition cells: {len(PILOT_CONDITIONS)}")
    print(f"nuclei per source x condition: {args.n_per_source_condition}")
    print(f"selected nuclei: {len(manifest_rows)}")
    print("")
    print("[Sources]")
    for source in COMPLETE_SOURCES:
        subset = [row for row in manifest_rows if row["source_name"] == source]
        strain = subset[0]["strain"] if subset else "?"
        sex = subset[0]["sex"] if subset else "?"
        print(f"{source}: strain={strain} sex={sex} n={len(subset)}")
    print("")
    print("[Paired source-held-out folds]")
    for name, test_sources in FOLDS.items():
        train_sources = [source for source in COMPLETE_SOURCES if source not in test_sources]
        print(f"{name}: test={','.join(test_sources)} train={','.join(train_sources)}")
    print("")
    print("[Balance]")
    print(f"by strain: {dict(counts_strain)}")
    print(f"by sex: {dict(counts_sex)}")
    print("")
    print("NASA BPS PILOT V1 FREEZE COMPLETE: YES")
    print(f"Manifest: {manifest_path}")
    print(f"Manifest SHA256: {summary['manifest_sha256']}")
    print(f"Folds: {fold_path}")
    print(f"Summary: {summary_path}")
    print("")
    print(
        "NEXT GATE: verify the frozen manifest counts, then download only these paired "
        "FITC/DAPI/MASK images and inspect image dimensions/intensity distributions before modeling."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
