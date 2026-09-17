"""Deterministic NASA R1 v2 sample-level endpoint reconstruction.

This module does no neural inference and performs no biological acceptance
testing. It converts frozen per-nucleus latent 53BP1 burden predictions into
the assay's sample-level aggregate endpoint.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from radiation_edge_ai.control import (
    ControlPlaneError,
    get_assay,
    load_json_object,
    sha256_file,
    sha256_json,
)

ASSAY_ID = "nasa-53bp1-r1-v2"
REPORT_TYPE = "nasa_endpoint_report"
AGGREGATION_METHOD = "arithmetic_mean_float64_numpy"
OUTPUT_FIELDS = (
    "sample_name",
    "source_name",
    "strain",
    "sex",
    "radiation_branch",
    "dose_gy",
    "timepoint_hr",
    "n_nuclei",
    "mean_latent_burden",
)


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ControlPlaneError(f"{label} is not a readable file: {resolved}")
    return resolved


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ControlPlaneError(f"Could not read CSV {path}: {exc}") from exc

    if not fields:
        raise ControlPlaneError(f"CSV has no header: {path}")
    if len(fields) != len(set(fields)):
        raise ControlPlaneError(f"CSV contains duplicate column names: {path}")
    if not rows:
        raise ControlPlaneError(f"CSV contains no data rows: {path}")

    return fields, rows


def _finite_float(value: object, *, label: str) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ControlPlaneError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ControlPlaneError(f"{label} is non-finite: {value!r}")
    return result


def _canonical_number(value: object, *, label: str) -> str:
    number = _finite_float(value, label=label)
    if number.is_integer():
        return f"{number:.1f}"
    return format(number, "g")


def _canonical_hour(value: object) -> str:
    number = _finite_float(value, label="timepoint")
    if number.is_integer():
        return str(int(number))
    return format(number, "g")


def _required_columns(burden_column: str) -> tuple[str, ...]:
    return (
        "sample_id",
        "sample_name",
        "source_name",
        "particle_type",
        "dose_Gy",
        "hr_post_exposure",
        burden_column,
    )


def _single_metadata_value(
    rows: list[dict[str, str]],
    column: str,
    *,
    sample_name: str,
    required: bool,
) -> str:
    if not rows:
        raise ControlPlaneError(f"Empty nucleus group for {sample_name}")

    if column not in rows[0]:
        if required:
            raise ControlPlaneError(
                f"Required column {column!r} missing for sample {sample_name}"
            )
        return ""

    values = {str(row.get(column, "")).strip() for row in rows}
    if required and "" in values:
        raise ControlPlaneError(
            f"Blank {column!r} value within sample {sample_name}"
        )
    if len(values) != 1:
        raise ControlPlaneError(
            f"Inconsistent {column!r} values within sample {sample_name}: "
            f"{sorted(values)!r}"
        )
    return next(iter(values))


def _single_numeric_metadata_value(
    rows: list[dict[str, str]],
    column: str,
    *,
    sample_name: str,
) -> str:
    """Require one semantically identical numeric value within a sample."""

    if not rows:
        raise ControlPlaneError(f"Empty nucleus group for {sample_name}")

    if column not in rows[0]:
        raise ControlPlaneError(
            f"Required column {column!r} missing for sample {sample_name}"
        )

    raw_values = [str(row.get(column, "")).strip() for row in rows]
    if any(value == "" for value in raw_values):
        raise ControlPlaneError(
            f"Blank {column!r} value within sample {sample_name}"
        )

    if column == "dose_Gy":
        normalized = {
            _canonical_number(
                value,
                label=f"dose_Gy for {sample_name}",
            )
            for value in raw_values
        }
    elif column == "hr_post_exposure":
        normalized = {
            _canonical_hour(value)
            for value in raw_values
        }
    else:
        raise ControlPlaneError(
            f"Unsupported numeric metadata column: {column!r}"
        )

    if len(normalized) != 1:
        raise ControlPlaneError(
            f"Inconsistent {column!r} values within sample {sample_name}: "
            f"{sorted(normalized)!r}"
        )

    return next(iter(normalized))


def _aggregate_rows(
    rows: list[dict[str, str]],
    *,
    burden_column: str,
) -> list[dict[str, object]]:
    if not burden_column.strip():
        raise ControlPlaneError("burden_column must not be blank")

    required = _required_columns(burden_column)
    missing = [column for column in required if column not in rows[0]]
    if missing:
        raise ControlPlaneError(
            "Prediction CSV is missing required columns: "
            + ", ".join(missing)
        )

    nucleus_ids: set[str] = set()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    for index, row in enumerate(rows, start=2):
        nucleus_id = str(row.get("sample_id", "")).strip()
        sample_name = str(row.get("sample_name", "")).strip()

        if not nucleus_id:
            raise ControlPlaneError(f"Blank sample_id at CSV line {index}")
        if nucleus_id in nucleus_ids:
            raise ControlPlaneError(f"Duplicate sample_id: {nucleus_id}")
        nucleus_ids.add(nucleus_id)

        if not sample_name:
            raise ControlPlaneError(f"Blank sample_name at CSV line {index}")

        _finite_float(
            row.get(burden_column),
            label=f"{burden_column} at CSV line {index}",
        )
        grouped[sample_name].append(row)

    aggregates: list[dict[str, object]] = []

    for sample_name, subset in sorted(grouped.items()):
        source_name = _single_metadata_value(
            subset,
            "source_name",
            sample_name=sample_name,
            required=True,
        )
        branch = _single_metadata_value(
            subset,
            "particle_type",
            sample_name=sample_name,
            required=True,
        )
        dose_value = _single_numeric_metadata_value(
            subset,
            "dose_Gy",
            sample_name=sample_name,
        )
        hour_value = _single_numeric_metadata_value(
            subset,
            "hr_post_exposure",
            sample_name=sample_name,
        )
        strain = _single_metadata_value(
            subset,
            "strain",
            sample_name=sample_name,
            required=False,
        )
        sex = _single_metadata_value(
            subset,
            "sex",
            sample_name=sample_name,
            required=False,
        )

        burdens = np.asarray(
            [
                _finite_float(
                    row[burden_column],
                    label=f"{burden_column} for {sample_name}",
                )
                for row in subset
            ],
            dtype=np.float64,
        )
        mean_burden = float(np.mean(burdens))
        if not math.isfinite(mean_burden):
            raise ControlPlaneError(
                f"Non-finite aggregate burden for sample {sample_name}"
            )

        aggregates.append(
            {
                "sample_name": sample_name,
                "source_name": source_name,
                "strain": strain,
                "sex": sex,
                "radiation_branch": branch,
                "dose_gy": dose_value,
                "timepoint_hr": hour_value,
                "n_nuclei": len(subset),
                "mean_latent_burden": mean_burden,
            }
        )

    return aggregates


def _render_aggregate_csv(rows: list[dict[str, object]]) -> bytes:
    if not rows:
        raise ControlPlaneError("Refusing to render an empty aggregate table")

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(OUTPUT_FIELDS),
        lineterminator="\n",
    )
    writer.writeheader()

    for row in rows:
        rendered = dict(row)
        rendered["mean_latent_burden"] = str(
            float(rendered["mean_latent_burden"])
        )
        writer.writerow(rendered)

    return stream.getvalue().encode("utf-8")


def _validate_aggregate_csv(
    path: Path,
    *,
    expected_samples: int,
    expected_nuclei: int,
) -> bool:
    try:
        fields, rows = _read_csv(path)
    except ControlPlaneError:
        return False

    if fields != list(OUTPUT_FIELDS):
        return False
    if len(rows) != expected_samples:
        return False

    sample_names: set[str] = set()
    nucleus_total = 0

    for row in rows:
        sample_name = str(row.get("sample_name", "")).strip()
        if not sample_name or sample_name in sample_names:
            return False
        sample_names.add(sample_name)

        try:
            count = int(row["n_nuclei"])
            burden = float(row["mean_latent_burden"])
        except (KeyError, TypeError, ValueError):
            return False

        if count <= 0 or not math.isfinite(burden):
            return False
        nucleus_total += count

    return nucleus_total == expected_nuclei


def create_nasa_endpoint_report(
    *,
    predictions_path: Path,
    burden_column: str,
    output_dir: Path,
) -> Path:
    """Aggregate frozen per-nucleus NASA predictions into sample endpoints."""

    spec = get_assay(ASSAY_ID)
    source = _require_file(predictions_path, label="prediction CSV")
    _fields, rows = _read_csv(source)

    aggregates = _aggregate_rows(
        rows,
        burden_column=burden_column,
    )
    csv_bytes = _render_aggregate_csv(aggregates)

    source_record = _artifact_record(source)
    aggregate_sha256 = _sha256_bytes(csv_bytes)

    identity: dict[str, Any] = {
        "schema_version": 1,
        "report_type": REPORT_TYPE,
        "assay_id": ASSAY_ID,
        "source_predictions_sha256": source_record["sha256"],
        "source_predictions_size_bytes": source_record["size_bytes"],
        "burden_column": burden_column,
        "nucleus_identity_column": "sample_id",
        "grouping_column": "sample_name",
        "aggregation_method": AGGREGATION_METHOD,
        "aggregate_csv_sha256": aggregate_sha256,
        "n_nuclei": len(rows),
        "n_samples": len(aggregates),
    }
    fingerprint = sha256_json(identity)
    aggregation_id = f"a1-{fingerprint[:16]}"

    root = output_dir.expanduser().resolve()
    report_dir = root / aggregation_id
    report_path = report_dir / "endpoint_report.json"
    aggregate_path = report_dir / "sample_aggregates.csv"

    if report_path.exists():
        verification = verify_nasa_endpoint_report(report_path)
        if verification["ok"]:
            return report_path
        raise ControlPlaneError(
            f"Existing endpoint report failed verification: {report_path}"
        )

    if report_dir.exists():
        raise ControlPlaneError(
            "Content-addressed endpoint directory already exists without a "
            f"verified report: {report_dir}"
        )

    report_dir.mkdir(parents=True, exist_ok=False)
    aggregate_path.write_bytes(csv_bytes)

    report = {
        "schema_version": 1,
        "report_type": REPORT_TYPE,
        "aggregation_id": aggregation_id,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "aggregation_fingerprint_sha256": fingerprint,
        "identity": identity,
        "assay": {
            "assay_id": spec.assay_id,
            "title": spec.title,
            "endpoint": spec.endpoint,
            "interpretation_guard": spec.interpretation_guard,
        },
        "source_predictions": source_record,
        "configuration": {
            "burden_column": burden_column,
            "nucleus_identity_column": "sample_id",
            "grouping_column": "sample_name",
            "aggregation_method": AGGREGATION_METHOD,
        },
        "counts": {
            "n_nuclei": len(rows),
            "n_samples": len(aggregates),
        },
        "outputs": {
            "sample_aggregates": {
                "path": str(aggregate_path),
                "sha256": sha256_file(aggregate_path),
                "size_bytes": aggregate_path.stat().st_size,
            }
        },
        "scientific_interpretation": {
            "per_nucleus_semantics": (
                "latent continuous 53BP1 burden; not an individually "
                "supervised focus count"
            ),
            "endpoint_reconstruction_performed": True,
            "endpoint": "sample-level mean latent 53BP1 burden",
            "biological_reference_read": False,
            "biological_acceptance_evaluated": False,
        },
    }

    temp_path = report_dir / "endpoint_report.json.tmp"
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    temp_path.replace(report_path)

    return report_path


def _verify_artifact(record: Mapping[str, object]) -> bool:
    path_value = record.get("path")
    expected_hash = record.get("sha256")
    expected_size = record.get("size_bytes")

    if (
        not isinstance(path_value, str)
        or not isinstance(expected_hash, str)
        or not isinstance(expected_size, int)
    ):
        return False

    path = Path(path_value)
    if not path.is_file():
        return False

    return (
        sha256_file(path) == expected_hash
        and path.stat().st_size == expected_size
    )


def verify_nasa_endpoint_report(
    report_path: Path,
    *,
    check_source_artifact: bool = True,
) -> dict[str, object]:
    """Verify one content-addressed NASA endpoint reconstruction report."""

    path = report_path.expanduser().resolve()
    report = load_json_object(path)

    if (
        report.get("schema_version") != 1
        or report.get("report_type") != REPORT_TYPE
    ):
        raise ControlPlaneError(f"Not a supported NASA endpoint report: {path}")

    identity = report.get("identity")
    source_record = report.get("source_predictions")
    outputs = report.get("outputs")
    counts = report.get("counts")

    if not isinstance(identity, dict):
        raise ControlPlaneError("Endpoint report identity is malformed")
    if not isinstance(source_record, dict):
        raise ControlPlaneError("Endpoint report source artifact is malformed")
    if not isinstance(outputs, dict):
        raise ControlPlaneError("Endpoint report outputs are malformed")
    if not isinstance(counts, dict):
        raise ControlPlaneError("Endpoint report counts are malformed")

    aggregate_record = outputs.get("sample_aggregates")
    if not isinstance(aggregate_record, dict):
        raise ControlPlaneError("Endpoint aggregate artifact is malformed")

    fingerprint = report.get("aggregation_fingerprint_sha256")
    aggregation_id = report.get("aggregation_id")

    fingerprint_ok = (
        isinstance(fingerprint, str)
        and sha256_json(identity) == fingerprint
    )
    aggregation_id_ok = (
        isinstance(fingerprint, str)
        and isinstance(aggregation_id, str)
        and aggregation_id == f"a1-{fingerprint[:16]}"
        and path.parent.name == aggregation_id
    )

    identity_bindings_ok = bool(
        identity.get("schema_version") == 1
        and identity.get("report_type") == REPORT_TYPE
        and identity.get("assay_id") == ASSAY_ID
        and identity.get("source_predictions_sha256")
        == source_record.get("sha256")
        and identity.get("source_predictions_size_bytes")
        == source_record.get("size_bytes")
        and identity.get("aggregate_csv_sha256")
        == aggregate_record.get("sha256")
        and identity.get("n_nuclei") == counts.get("n_nuclei")
        and identity.get("n_samples") == counts.get("n_samples")
    )

    source_ok = True
    if check_source_artifact:
        source_ok = _verify_artifact(source_record)

    aggregate_ok = _verify_artifact(aggregate_record)

    structure_ok = False
    aggregate_path_value = aggregate_record.get("path")
    n_samples = counts.get("n_samples")
    n_nuclei = counts.get("n_nuclei")

    if (
        aggregate_ok
        and isinstance(aggregate_path_value, str)
        and isinstance(n_samples, int)
        and isinstance(n_nuclei, int)
    ):
        structure_ok = _validate_aggregate_csv(
            Path(aggregate_path_value),
            expected_samples=n_samples,
            expected_nuclei=n_nuclei,
        )

    ok = bool(
        report.get("status") == "complete"
        and fingerprint_ok
        and aggregation_id_ok
        and identity_bindings_ok
        and source_ok
        and aggregate_ok
        and structure_ok
    )

    return {
        "kind": REPORT_TYPE,
        "aggregation_id": aggregation_id,
        "report_path": str(path),
        "fingerprint_ok": fingerprint_ok,
        "aggregation_id_ok": aggregation_id_ok,
        "identity_bindings_ok": identity_bindings_ok,
        "source_artifact_check_performed": check_source_artifact,
        "source_predictions_ok": source_ok,
        "sample_aggregates_ok": bool(aggregate_ok and structure_ok),
        "n_nuclei": n_nuclei,
        "n_samples": n_samples,
        "ok": ok,
    }
