"""Inspect NASA BPS microscopy metadata for hidden grouping structure.

The public benchmark metadata expose only filename, dose_Gy, particle_type and
hr_post_exposure. Before freezing any train/validation/test split we therefore
need to determine whether plate/mouse/sample/strain information is encoded in
filenames or recoverable through a deterministic filename crosswalk.

This utility is metadata-only: it reads the two CSV files downloaded by
``inventory_microscopy.py`` and does not download microscopy images.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED = ("filename", "dose_Gy", "particle_type", "hr_post_exposure")
CHANNEL_WORDS = {
    "fitc",
    "dapi",
    "mask",
    "53bp1",
    "ch1",
    "ch2",
    "channel1",
    "channel2",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        missing = [name for name in REQUIRED if name not in fields]
        if missing:
            raise RuntimeError(f"{path} missing columns {missing}; fields={fields}")
        return [dict(row) for row in reader]


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def condition_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row["dose_Gy"]).strip(),
        str(row["particle_type"]).strip(),
        str(row["hr_post_exposure"]).strip(),
    )


def condition_counts(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    counts = Counter(condition_key(row) for row in rows)
    result = [
        {
            "dose_Gy": dose,
            "particle_type": particle,
            "hr_post_exposure": hours,
            "n_images": count,
        }
        for (dose, particle, hours), count in counts.items()
    ]
    result.sort(key=lambda r: (natural_key(str(r["particle_type"])), float(r["dose_Gy"]), float(r["hr_post_exposure"])))
    return result


def stem(filename: str) -> str:
    return Path(filename).stem


def tokens(filename: str) -> list[str]:
    value = stem(filename)
    # Split on punctuation and letter/number boundaries. This intentionally
    # avoids assuming a particular NASA naming convention.
    coarse = [p for p in re.split(r"[^A-Za-z0-9]+", value) if p]
    result: list[str] = []
    for part in coarse:
        result.extend(p for p in re.findall(r"[A-Za-z]+|\d+", part) if p)
    return result


def normalized_channel_free_stem(filename: str) -> str:
    parts = tokens(filename)
    kept = [part.lower() for part in parts if part.lower() not in CHANNEL_WORDS]
    return "_".join(kept)


def token_position_stats(rows: list[dict[str, str]], max_positions: int = 12) -> list[dict[str, object]]:
    all_tokens = [tokens(str(row["filename"])) for row in rows]
    stats = []
    for position in range(max_positions):
        values = [parts[position] for parts in all_tokens if len(parts) > position]
        if not values:
            break
        counts = Counter(values)
        examples = [value for value, _count in counts.most_common(8)]
        stats.append(
            {
                "token_position": position,
                "n_rows_with_token": len(values),
                "n_unique": len(counts),
                "top_values": examples,
            }
        )
    return stats


def prefix_stats(rows: list[dict[str, str]], max_tokens: int = 6) -> list[dict[str, object]]:
    parsed = [tokens(str(row["filename"])) for row in rows]
    result = []
    for n_tokens in range(1, max_tokens + 1):
        prefixes = Counter(
            "_".join(parts[:n_tokens])
            for parts in parsed
            if len(parts) >= n_tokens
        )
        if not prefixes:
            continue
        sizes = list(prefixes.values())
        result.append(
            {
                "prefix_tokens": n_tokens,
                "n_unique_prefixes": len(prefixes),
                "min_group_size": min(sizes),
                "median_group_size": sorted(sizes)[len(sizes) // 2],
                "max_group_size": max(sizes),
                "largest_groups": [
                    {"prefix": prefix, "n": count}
                    for prefix, count in prefixes.most_common(10)
                ],
            }
        )
    return result


def condition_examples(rows: list[dict[str, str]], n_examples: int) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in rows:
        grouped[condition_key(row)].append(str(row["filename"]))
    output = []
    for key in sorted(grouped, key=lambda k: (natural_key(k[1]), float(k[0]), float(k[2]))):
        names = sorted(grouped[key], key=natural_key)
        output.append(
            {
                "dose_Gy": key[0],
                "particle_type": key[1],
                "hr_post_exposure": key[2],
                "n_images": len(names),
                "examples": names[:n_examples],
            }
        )
    return output


def write_condition_examples(path: Path, rows: list[dict[str, object]], n_examples: int) -> None:
    fields = ["dose_Gy", "particle_type", "hr_post_exposure", "n_images"] + [
        f"filename_{i + 1}" for i in range(n_examples)
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {key: row[key] for key in ("dose_Gy", "particle_type", "hr_post_exposure", "n_images")}
            examples = list(row["examples"])
            for i in range(n_examples):
                record[f"filename_{i + 1}"] = examples[i] if i < len(examples) else ""
            writer.writerow(record)


def crosswalk_stats(fitc_rows: list[dict[str, str]], dapi_rows: list[dict[str, str]]) -> dict[str, object]:
    fitc_names = {str(row["filename"]) for row in fitc_rows}
    dapi_names = {str(row["filename"]) for row in dapi_rows}
    exact = fitc_names & dapi_names

    fitc_norm: dict[str, int] = Counter(normalized_channel_free_stem(name) for name in fitc_names)
    dapi_norm: dict[str, int] = Counter(normalized_channel_free_stem(name) for name in dapi_names)
    shared_norm = set(fitc_norm) & set(dapi_norm)
    one_to_one = [key for key in shared_norm if fitc_norm[key] == 1 and dapi_norm[key] == 1]

    return {
        "exact_filename_intersection": len(exact),
        "normalized_channel_free_shared_keys": len(shared_norm),
        "normalized_channel_free_one_to_one_keys": len(one_to_one),
        "fitc_unique_normalized_keys": len(fitc_norm),
        "dapi_unique_normalized_keys": len(dapi_norm),
        "example_shared_normalized_keys": sorted(shared_norm, key=natural_key)[:20],
    }


def dataset_report(label: str, rows: list[dict[str, str]], n_examples: int) -> dict[str, object]:
    filenames = [str(row["filename"]) for row in rows]
    token_lengths = Counter(len(tokens(name)) for name in filenames)
    return {
        "label": label,
        "n_rows": len(rows),
        "n_unique_filenames": len(set(filenames)),
        "n_conditions": len(condition_counts(rows)),
        "condition_counts": condition_counts(rows),
        "filename_examples": sorted(set(filenames), key=natural_key)[:20],
        "token_length_counts": dict(sorted(token_lengths.items())),
        "token_position_stats": token_position_stats(rows),
        "prefix_stats": prefix_stats(rows),
        "condition_examples": condition_examples(rows, n_examples),
    }


def print_report(report: dict[str, object]) -> None:
    print(f"[{report['label']} condition counts]")
    for row in report["condition_counts"]:
        print(
            f"dose={row['dose_Gy']:>4} Gy  particle={row['particle_type']:<5} "
            f"time={row['hr_post_exposure']:>3} h  n={row['n_images']}"
        )
    print("")
    print(f"[{report['label']} filename examples]")
    for name in report["filename_examples"][:12]:
        print(name)
    print("")
    print(f"[{report['label']} token-position cardinalities]")
    for row in report["token_position_stats"]:
        print(
            f"pos {row['token_position']:>2}: unique={row['n_unique']:<8} "
            f"rows={row['n_rows_with_token']:<8} top={row['top_values']}"
        )
    print("")
    print(f"[{report['label']} prefix grouping]")
    for row in report["prefix_stats"]:
        print(
            f"first {row['prefix_tokens']} token(s): groups={row['n_unique_prefixes']:<8} "
            f"size min/median/max={row['min_group_size']}/{row['median_group_size']}/{row['max_group_size']}"
        )
    print("")


def main() -> int:
    parser = argparse.ArgumentParser()
    default_root = Path(
        os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data")
    ) / "nasa_bps_microscopy" / "metadata"
    parser.add_argument("--metadata-dir", default=str(default_root))
    parser.add_argument("--examples-per-condition", type=int, default=3)
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir).resolve()
    fitc_path = metadata_dir / "meta.csv"
    dapi_path = metadata_dir / "meta_DAPI_MASK.csv"
    for path in (fitc_path, dapi_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    fitc_rows = read_csv(fitc_path)
    dapi_rows = read_csv(dapi_path)
    fitc = dataset_report("FITC/53BP1", fitc_rows, args.examples_per_condition)
    dapi = dataset_report("DAPI/MASK", dapi_rows, args.examples_per_condition)
    crosswalk = crosswalk_stats(fitc_rows, dapi_rows)

    print("Radiation Edge AI - NASA BPS filename/group diagnostics")
    print("Metadata only: no microscopy images will be downloaded.")
    print("")
    print_report(fitc)
    print_report(dapi)
    print("[FITC vs DAPI/MASK filename crosswalk]")
    for key, value in crosswalk.items():
        if key != "example_shared_normalized_keys":
            print(f"{key}: {value}")
    print(f"example shared normalized keys: {crosswalk['example_shared_normalized_keys'][:10]}")
    print("")

    report_path = metadata_dir / "filename_group_diagnostics.json"
    report_path.write_text(
        json.dumps({"fitc_53bp1": fitc, "dapi_mask": dapi, "crosswalk": crosswalk}, indent=2),
        encoding="utf-8",
    )
    write_condition_examples(
        metadata_dir / "fitc_53bp1_condition_examples.csv",
        fitc["condition_examples"],
        args.examples_per_condition,
    )
    write_condition_examples(
        metadata_dir / "dapi_mask_condition_examples.csv",
        dapi["condition_examples"],
        args.examples_per_condition,
    )

    print("NASA BPS FILENAME/GROUP DIAGNOSTICS COMPLETE: YES")
    print(f"Report: {report_path}")
    print("")
    print(
        "NEXT GATE: use filename structure and/or OSD-366 metadata to define an independent "
        "plate/mouse/strain grouping key before selecting the pilot or train/test split."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
