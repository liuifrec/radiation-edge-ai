"""Pure utilities for evidence audits.

This module intentionally has no Kneron, OpenCV, Torch, or hardware imports.
It is used by post-release evidence audits and CPU-only synthetic tests.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np

CONTRAST_FIELDS = (
    "source_name",
    "branch",
    "timepoint_hr",
    "sham_sample_name",
    "exposed_sample_name",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"Expected JSON object: {path}")
    reject_nonfinite_json(obj, label=str(path))
    return obj


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def reject_nonfinite_json(value: Any, *, label: str = "value", path: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, np.integer)):
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise RuntimeError(f"Non-finite JSON number at {label}:{path}: {value!r}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            reject_nonfinite_json(child, label=label, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, child in enumerate(value):
            reject_nonfinite_json(child, label=label, path=f"{path}[{index}]")
        return


def finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid numeric value for {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite numeric value for {label}: {value!r}")
    return result


def finite_int(value: Any, *, label: str) -> int:
    number = finite_float(value, label=label)
    if not number.is_integer():
        raise RuntimeError(f"Expected integer-valued {label}, got {value!r}")
    return int(number)


def require_fields(row: Mapping[str, Any], fields: Iterable[str], *, label: str) -> None:
    missing = [field for field in fields if field not in row or str(row[field]).strip() == ""]
    if missing:
        raise RuntimeError(f"{label} missing required fields: {missing}")


def normalized_hour(value: Any) -> str:
    x = finite_float(value, label="timepoint_hr")
    return str(int(x)) if x.is_integer() else format(x, "g")


def normalized_number(value: Any) -> str:
    x = finite_float(value, label="numeric value")
    if x.is_integer():
        return f"{x:.1f}"
    return format(x, "g")


def enrich_contrast_sample_names(
    delta_rows: Sequence[Mapping[str, Any]],
    bag_rows: Sequence[Mapping[str, Any]],
    *,
    exposed_dose_by_branch: Mapping[str, Any],
    label: str,
) -> list[dict[str, Any]]:
    """Attach canonical sham/exposed sample names using a verified bag table.

    The historical FP32 delta table predates the deployment tables and does not
    contain ``sham_sample_name``/``exposed_sample_name`` columns.  Rather than
    weakening the audit key, reconstruct those identities from the frozen bag
    table and verify any names already present in newer BIE/physical delta
    tables.
    """

    required_bag = (
        "sample_name",
        "source_name",
        "particle_type",
        "dose_Gy",
        "hr_post_exposure",
    )
    lookup: dict[tuple[str, str, str, str], str] = {}
    for bag in bag_rows:
        require_fields(bag, required_bag, label=f"{label} bag row")
        key = (
            str(bag["source_name"]),
            str(bag["particle_type"]),
            normalized_number(bag["dose_Gy"]),
            normalized_hour(bag["hr_post_exposure"]),
        )
        sample_name = str(bag["sample_name"])
        if key in lookup:
            raise RuntimeError(
                f"Duplicate {label} bag identity {key}: {lookup[key]!r} / {sample_name!r}"
            )
        lookup[key] = sample_name

    enriched: list[dict[str, Any]] = []
    for row in delta_rows:
        require_fields(
            row, ("source_name", "branch", "timepoint_hr"), label=f"{label} contrast row"
        )
        source = str(row["source_name"])
        branch = str(row["branch"])
        hour = normalized_hour(row["timepoint_hr"])
        if branch not in exposed_dose_by_branch:
            raise RuntimeError(f"Unknown {label} contrast branch {branch!r}")
        exposed_dose = normalized_number(exposed_dose_by_branch[branch])
        if str(row.get("exposed_dose_Gy", "")).strip():
            observed_dose = normalized_number(row["exposed_dose_Gy"])
            if observed_dose != exposed_dose:
                raise RuntimeError(
                    f"{label} exposed dose mismatch for {(source, branch, hour)}: "
                    f"delta={observed_dose}, policy={exposed_dose}"
                )
        sham_key = (source, branch, normalized_number(0.0), hour)
        exposed_key = (source, branch, exposed_dose, hour)
        if sham_key not in lookup or exposed_key not in lookup:
            raise RuntimeError(
                f"Could not resolve {label} contrast sample identities for "
                f"{(source, branch, hour)}; sham={sham_key in lookup} exposed={exposed_key in lookup}"
            )
        sham_name = lookup[sham_key]
        exposed_name = lookup[exposed_key]
        for field, derived in (
            ("sham_sample_name", sham_name),
            ("exposed_sample_name", exposed_name),
        ):
            existing = str(row.get(field, "")).strip()
            if existing and existing != derived:
                raise RuntimeError(
                    f"{label} {field} mismatch for {(source, branch, hour)}: "
                    f"delta={existing!r}, bag={derived!r}"
                )
        out = dict(row)
        out["sham_sample_name"] = sham_name
        out["exposed_sample_name"] = exposed_name
        enriched.append(out)
    return enriched


def contrast_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    require_fields(row, CONTRAST_FIELDS, label="contrast row")
    return (
        str(row["source_name"]),
        str(row["branch"]),
        normalized_hour(row["timepoint_hr"]),
        str(row["sham_sample_name"]),
        str(row["exposed_sample_name"]),
    )


def unique_index(
    rows: Sequence[Mapping[str, Any]],
    key_fn: Callable[[Mapping[str, Any]], Any],
    *,
    label: str,
) -> dict[Any, Mapping[str, Any]]:
    result: dict[Any, Mapping[str, Any]] = {}
    duplicates: list[Any] = []
    for row in rows:
        key = key_fn(row)
        if key in result:
            duplicates.append(key)
        else:
            result[key] = row
    if duplicates:
        raise RuntimeError(f"Duplicate {label} keys: {duplicates[:5]}")
    return result


def sign(value: float, *, zero_tolerance: float = 0.0) -> int:
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite value passed to sign: {value!r}")
    if abs(value) <= zero_tolerance:
        return 0
    return 1 if value > 0 else -1


def direction_signature(
    rows: Sequence[Mapping[str, Any]], delta_field: str = "pred_delta"
) -> dict[str, Any]:
    overall = 0
    by_source: dict[str, list[int]] = defaultdict(list)
    four: list[int] = []
    late: list[int] = []
    for row in rows:
        nasa = finite_float(row["nasa_delta"], label="nasa_delta")
        pred = finite_float(row[delta_field], label=delta_field)
        agree = int(sign(nasa) == sign(pred))
        overall += agree
        source = str(row["source_name"])
        by_source[source].append(agree)
        hour = normalized_hour(row["timepoint_hr"])
        if hour == "4":
            four.append(agree)
        elif hour in {"24", "48"}:
            late.append(agree)
    return {
        "overall": (overall, len(rows)),
        "4h": (sum(four), len(four)),
        "24+48h": (sum(late), len(late)),
        "by_source": {
            source: (sum(values), len(values)) for source, values in sorted(by_source.items())
        },
    }


def join_contrasts(
    fp32_rows: Sequence[Mapping[str, Any]],
    bie_rows: Sequence[Mapping[str, Any]],
    physical_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fp32 = unique_index(fp32_rows, contrast_key, label="FP32 contrast")
    bie = unique_index(bie_rows, contrast_key, label="BIE contrast")
    physical = unique_index(physical_rows, contrast_key, label="physical contrast")
    keys = set(fp32)
    if set(bie) != keys or set(physical) != keys:
        raise RuntimeError(
            "Contrast key sets differ: "
            f"fp32={len(fp32)} bie={len(bie)} physical={len(physical)}; "
            f"missing_bie={sorted(keys - set(bie))[:5]} extra_bie={sorted(set(bie) - keys)[:5]} "
            f"missing_physical={sorted(keys - set(physical))[:5]} "
            f"extra_physical={sorted(set(physical) - keys)[:5]}"
        )

    joined: list[dict[str, Any]] = []
    fp32_vs_bie_changed: list[tuple[str, str, str, str, str]] = []
    fp32_vs_physical_changed: list[tuple[str, str, str, str, str]] = []
    bie_vs_physical_changed: list[tuple[str, str, str, str, str]] = []

    for key in sorted(keys):
        f = fp32[key]
        b = bie[key]
        p = physical[key]
        nasa_values = [finite_float(row["nasa_delta"], label="nasa_delta") for row in (f, b, p)]
        if max(nasa_values) - min(nasa_values) > 1e-12:
            raise RuntimeError(
                f"NASA delta differs across frozen contrast tables for key={key}: {nasa_values}"
            )
        fd = finite_float(f["pred_delta"], label="FP32 pred_delta")
        bd = finite_float(b["pred_delta"], label="BIE pred_delta")
        pd = finite_float(p["pred_delta"], label="physical pred_delta")
        ns = sign(nasa_values[0])
        fs, bs, ps = sign(fd), sign(bd), sign(pd)
        if fs != bs:
            fp32_vs_bie_changed.append(key)
        if fs != ps:
            fp32_vs_physical_changed.append(key)
        if bs != ps:
            bie_vs_physical_changed.append(key)
        joined.append(
            {
                "source_name": key[0],
                "branch": key[1],
                "timepoint_hr": key[2],
                "sham_sample_name": key[3],
                "exposed_sample_name": key[4],
                "nasa_delta": nasa_values[0],
                "fp32_delta": fd,
                "bie_delta": bd,
                "physical_delta": pd,
                "nasa_sign": ns,
                "fp32_sign": fs,
                "bie_sign": bs,
                "physical_sign": ps,
                "fp32_direction_agreement_with_nasa": int(fs == ns),
                "bie_direction_agreement_with_nasa": int(bs == ns),
                "physical_direction_agreement_with_nasa": int(ps == ns),
                "fp32_distance_to_zero": abs(fd),
                "bie_distance_to_zero": abs(bd),
                "physical_distance_to_zero": abs(pd),
                "fp32_vs_bie_sign_same": int(fs == bs),
                "fp32_vs_physical_sign_same": int(fs == ps),
                "bie_vs_physical_sign_same": int(bs == ps),
            }
        )

    audit = {
        "n_contrasts": len(joined),
        "fp32_vs_bie_changed_contrasts": [list(key) for key in fp32_vs_bie_changed],
        "fp32_vs_physical_changed_contrasts": [list(key) for key in fp32_vs_physical_changed],
        "bie_vs_physical_changed_contrasts": [list(key) for key in bie_vs_physical_changed],
        "actual_fp32_vs_bie_per_contrast_sign_identity": not fp32_vs_bie_changed,
        "actual_fp32_vs_physical_per_contrast_sign_identity": not fp32_vs_physical_changed,
        "actual_bie_vs_physical_per_contrast_sign_identity": not bie_vs_physical_changed,
    }
    return joined, audit


def peak_series_audit(joined_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        grouped[(str(row["source_name"]), str(row["branch"]))].append(row)

    result: list[dict[str, Any]] = []
    for (source, branch), subset in sorted(grouped.items()):
        hours = {normalized_hour(row["timepoint_hr"]) for row in subset}
        complete = hours == {"4", "24", "48"} and len(subset) == 3
        if not complete:
            continue
        record: dict[str, Any] = {"source_name": source, "branch": branch, "complete_4_24_48": True}
        for method, field in (
            ("nasa", "nasa_delta"),
            ("fp32", "fp32_delta"),
            ("bie", "bie_delta"),
            ("physical", "physical_delta"),
        ):
            values = [
                (normalized_hour(row["timepoint_hr"]), finite_float(row[field], label=field))
                for row in subset
            ]
            maximum = max(value for _hour, value in values)
            peak_hours = sorted(hour for hour, value in values if value == maximum)
            record[f"{method}_peak_delta"] = maximum
            record[f"{method}_peak_hours"] = ",".join(peak_hours)
            record[f"{method}_peak_tied"] = len(peak_hours) > 1
        record["fp32_vs_physical_peak_set_same"] = int(
            record["fp32_peak_hours"] == record["physical_peak_hours"]
        )
        record["fp32_vs_bie_peak_set_same"] = int(
            record["fp32_peak_hours"] == record["bie_peak_hours"]
        )
        result.append(record)
    return result


def refuse_nonempty_output_dir(path: Path) -> None:
    if path.exists():
        if path.is_file():
            raise RuntimeError(f"Output path exists as a file: {path}")
        try:
            next(path.iterdir())
        except StopIteration:
            return
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {path}")


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
