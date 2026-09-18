"""Deterministic NASA prediction-table projection from a verified batch record.

This stage performs no neural inference. It projects already completed,
verified CPU-ONNX nucleus outputs and frozen batch metadata into the CSV
contract consumed by the existing NASA measurement layer.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from radiation_edge_ai.batch import verify_batch_record
from radiation_edge_ai.control import (
    ControlPlaneError,
    load_json_object,
    sha256_file,
    sha256_json,
)

ASSAY_ID = "nasa-53bp1-r1-v2"

REPORT_SCHEMA_VERSION = 1
REPORT_TYPE = "nasa_batch_prediction_report"
TABLE_SCHEMA_VERSION = 1
BURDEN_COLUMN = "cpu_onnx_burden"

OUTPUT_FIELDS = (
    "sample_id",
    "sample_name",
    "source_name",
    "particle_type",
    "dose_Gy",
    "hr_post_exposure",
    "strain",
    "sex",
    BURDEN_COLUMN,
    "run_id",
    "raw_output_sha256",
)


def _require_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ControlPlaneError(
            f"{label} is not a readable file: {resolved}"
        )
    return resolved


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_artifact(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False

    path_value = value.get("path")
    expected_sha = value.get("sha256")
    expected_size = value.get("size_bytes")

    if (
        not isinstance(path_value, str)
        or not isinstance(expected_sha, str)
        or not isinstance(expected_size, int)
    ):
        return False

    path = Path(path_value)

    return bool(
        path.is_file()
        and sha256_file(path) == expected_sha
        and path.stat().st_size == expected_size
    )


def _require_mapping(
    value: object,
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ControlPlaneError(
            f"{label} must be a JSON object"
        )
    return value


def _required_text(
    value: object,
    *,
    label: str,
) -> str:
    if not isinstance(value, str):
        raise ControlPlaneError(
            f"{label} must be a string"
        )

    result = value.strip()

    if not result:
        raise ControlPlaneError(
            f"{label} must not be blank"
        )

    return result


def _optional_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _number_text(
    value: object,
    *,
    label: str,
) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ControlPlaneError(
            f"{label} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise ControlPlaneError(
            f"{label} must be finite"
        )

    return str(number)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _timestamp_is_aware(value: object) -> bool:
    if not isinstance(value, str):
        return False

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False

    return bool(
        parsed.tzinfo is not None
        and parsed.utcoffset() is not None
    )


def _scientific_scope() -> dict[str, object]:
    return {
        "source_batch_neural_inference_performed": True,
        "neural_inference_performed_by_projection": False,
        "prediction_semantics": (
            "per-nucleus latent continuous 53BP1 burden"
        ),
        "per_nucleus_focus_count_interpretation": False,
        "prediction_table_created": True,
        "endpoint_reconstruction_performed": False,
        "biological_reference_read": False,
        "biological_acceptance_evaluated": False,
        "hardware_access_performed": False,
    }


def _load_scalar_output(
    output_record: Mapping[str, Any],
    *,
    item_id: str,
) -> float:
    path_value = output_record.get("path")

    if not isinstance(path_value, str):
        raise ControlPlaneError(
            f"Raw output path is malformed for {item_id!r}"
        )

    path = Path(path_value)

    try:
        value = np.load(
            path,
            allow_pickle=False,
        )
    except (OSError, ValueError) as exc:
        raise ControlPlaneError(
            f"Could not read raw output for {item_id!r}: {exc}"
        ) from exc

    if not isinstance(value, np.ndarray):
        raise ControlPlaneError(
            f"Raw output is not an ndarray for {item_id!r}"
        )

    if value.size != 1:
        raise ControlPlaneError(
            f"Expected scalar output for {item_id!r}; got {value.shape}"
        )

    if not np.issubdtype(value.dtype, np.floating):
        raise ControlPlaneError(
            f"Expected floating output for {item_id!r}; got {value.dtype}"
        )

    result = float(value.reshape(-1)[0])

    if not math.isfinite(result):
        raise ControlPlaneError(
            f"Non-finite prediction for {item_id!r}"
        )

    return result


def _render_prediction_table(
    batch_record_path: Path,
) -> tuple[bytes, list[dict[str, str]]]:
    record = load_json_object(batch_record_path)

    plan_binding = _require_mapping(
        record.get("batch_plan"),
        label="batch-record plan binding",
    )

    plan_path_value = plan_binding.get("path")

    if not isinstance(plan_path_value, str):
        raise ControlPlaneError(
            "Batch-record plan path is malformed"
        )

    plan = load_json_object(Path(plan_path_value))

    items = plan.get("items")
    runs = record.get("runs")

    if not isinstance(items, list):
        raise ControlPlaneError(
            "Batch-plan items are malformed"
        )

    if not isinstance(runs, list):
        raise ControlPlaneError(
            "Batch-record runs are malformed"
        )

    run_by_item: dict[str, Mapping[str, Any]] = {}

    for raw_run in runs:
        run = _require_mapping(
            raw_run,
            label="batch child run",
        )

        item_id = _required_text(
            run.get("item_id"),
            label="batch child item_id",
        )

        if item_id in run_by_item:
            raise ControlPlaneError(
                f"Duplicate child item_id: {item_id!r}"
            )

        run_by_item[item_id] = run

    rows: list[dict[str, str]] = []
    planned_ids: set[str] = set()

    ordered_items = sorted(
        items,
        key=lambda item: str(
            item.get("item_id", "")
            if isinstance(item, Mapping)
            else ""
        ),
    )

    for raw_item in ordered_items:
        item = _require_mapping(
            raw_item,
            label="batch-plan item",
        )

        item_id = _required_text(
            item.get("item_id"),
            label="batch-plan item_id",
        )

        if item_id in planned_ids:
            raise ControlPlaneError(
                f"Duplicate planned item_id: {item_id!r}"
            )

        planned_ids.add(item_id)

        metadata = _require_mapping(
            item.get("metadata"),
            label=f"metadata for {item_id!r}",
        )

        sample_id = _required_text(
            metadata.get("sample_id"),
            label=f"sample_id for {item_id!r}",
        )

        if sample_id != item_id:
            raise ControlPlaneError(
                f"item_id/sample_id mismatch for {item_id!r}"
            )

        run = run_by_item.get(item_id)

        if run is None:
            raise ControlPlaneError(
                f"Missing completed child run for {item_id!r}"
            )

        raw_output = _require_mapping(
            run.get("raw_output"),
            label=f"raw output for {item_id!r}",
        )

        run_id = _required_text(
            run.get("run_id"),
            label=f"run_id for {item_id!r}",
        )

        raw_sha = _required_text(
            raw_output.get("sha256"),
            label=f"raw-output SHA for {item_id!r}",
        )

        burden = _load_scalar_output(
            raw_output,
            item_id=item_id,
        )

        rows.append(
            {
                "sample_id": sample_id,
                "sample_name": _required_text(
                    metadata.get("sample_name"),
                    label=f"sample_name for {item_id!r}",
                ),
                "source_name": _required_text(
                    metadata.get("source_name"),
                    label=f"source_name for {item_id!r}",
                ),
                "particle_type": _required_text(
                    metadata.get("radiation_branch"),
                    label=f"radiation_branch for {item_id!r}",
                ),
                "dose_Gy": _number_text(
                    metadata.get("dose_gy"),
                    label=f"dose_gy for {item_id!r}",
                ),
                "hr_post_exposure": _number_text(
                    metadata.get("timepoint_hr"),
                    label=f"timepoint_hr for {item_id!r}",
                ),
                "strain": _optional_text(
                    metadata.get("strain")
                ),
                "sex": _optional_text(
                    metadata.get("sex")
                ),
                BURDEN_COLUMN: str(burden),
                "run_id": run_id,
                "raw_output_sha256": raw_sha,
            }
        )

    if planned_ids != set(run_by_item):
        raise ControlPlaneError(
            "Batch plan and completed-run item sets differ"
        )

    stream = io.StringIO(newline="")

    writer = csv.DictWriter(
        stream,
        fieldnames=list(OUTPUT_FIELDS),
        lineterminator="\n",
    )

    writer.writeheader()
    writer.writerows(rows)

    return stream.getvalue().encode("utf-8"), rows


def _identity(
    *,
    source_record: Mapping[str, Any],
    batch_id: str,
    backend: str,
    table_sha256: str,
    n_rows: int,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": REPORT_TYPE,
        "assay_id": ASSAY_ID,
        "batch_id": batch_id,
        "backend": backend,
        "source_batch_record_sha256": source_record["sha256"],
        "source_batch_record_size_bytes": source_record[
            "size_bytes"
        ],
        "table_schema_version": TABLE_SCHEMA_VERSION,
        "burden_column": BURDEN_COLUMN,
        "output_fields": list(OUTPUT_FIELDS),
        "prediction_table_sha256": table_sha256,
        "n_rows": n_rows,
        "scientific_scope": _scientific_scope(),
    }


def create_nasa_batch_prediction_report(
    *,
    batch_record_path: Path,
    output_dir: Path,
) -> Path:
    """Project one verified CPU batch into a deterministic prediction CSV."""

    source = _require_file(
        batch_record_path,
        label="batch record",
    )

    source_verification = verify_batch_record(
        source,
        check_artifacts=True,
    )

    if not source_verification["ok"]:
        raise ControlPlaneError(
            "Batch record failed verification; refusing prediction projection"
        )

    record = load_json_object(source)

    batch_id = _required_text(
        record.get("batch_id"),
        label="batch_id",
    )

    backend = _required_text(
        record.get("backend"),
        label="batch backend",
    )

    if backend != "cpu-onnx":
        raise ControlPlaneError(
            "NASA v0.4 prediction projection currently supports only cpu-onnx"
        )

    table_bytes, rows = _render_prediction_table(source)
    table_sha = _sha256_bytes(table_bytes)

    source_record = _artifact_record(source)

    identity = _identity(
        source_record=source_record,
        batch_id=batch_id,
        backend=backend,
        table_sha256=table_sha,
        n_rows=len(rows),
    )

    fingerprint = sha256_json(identity)
    prediction_id = f"p1-{fingerprint[:16]}"

    root = output_dir.expanduser().resolve()
    report_dir = root / prediction_id

    report_path = report_dir / "prediction_table_report.json"
    table_path = report_dir / "prediction_table.csv"

    if report_path.exists():
        verification = verify_nasa_batch_prediction_report(
            report_path,
            check_source_artifact=True,
        )

        if verification["ok"]:
            return report_path

        raise ControlPlaneError(
            f"Existing prediction report failed verification: {report_path}"
        )

    if report_dir.exists():
        raise ControlPlaneError(
            "Prediction-report directory already exists without a verified "
            f"report: {report_dir}"
        )

    report_dir.mkdir(parents=True, exist_ok=False)

    table_temp = report_dir / "prediction_table.csv.tmp"
    table_temp.write_bytes(table_bytes)
    table_temp.replace(table_path)

    table_record = _artifact_record(table_path)

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": REPORT_TYPE,
        "prediction_id": prediction_id,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_fingerprint_sha256": fingerprint,
        "identity": identity,
        "assay_id": ASSAY_ID,
        "batch_id": batch_id,
        "backend": backend,
        "source_batch_record": {
            **source_record,
            "batch_id": batch_id,
        },
        "output": {
            "prediction_table": table_record,
            "table_schema_version": TABLE_SCHEMA_VERSION,
            "burden_column": BURDEN_COLUMN,
            "fields": list(OUTPUT_FIELDS),
        },
        "counts": {
            "n_rows": len(rows),
        },
        "scientific_scope": _scientific_scope(),
    }

    temp_path = report_dir / "prediction_table_report.json.tmp"

    with temp_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
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


def verify_nasa_batch_prediction_report(
    report_path: Path,
    *,
    check_source_artifact: bool = True,
) -> dict[str, object]:
    """Verify a NASA batch prediction-table derivation."""

    path = report_path.expanduser().resolve()
    report = load_json_object(path)

    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("report_type") != REPORT_TYPE
    ):
        raise ControlPlaneError(
            f"Not a supported NASA prediction report: {path}"
        )

    identity = report.get("identity")
    source = report.get("source_batch_record")
    output = report.get("output")
    counts = report.get("counts")
    scope = report.get("scientific_scope")

    if not isinstance(identity, dict):
        raise ControlPlaneError(
            "Prediction-report identity is malformed"
        )
    if not isinstance(source, dict):
        raise ControlPlaneError(
            "Prediction-report source binding is malformed"
        )
    if not isinstance(output, dict):
        raise ControlPlaneError(
            "Prediction-report output is malformed"
        )
    if not isinstance(counts, dict):
        raise ControlPlaneError(
            "Prediction-report counts are malformed"
        )
    if not isinstance(scope, dict):
        raise ControlPlaneError(
            "Prediction-report scientific scope is malformed"
        )

    table = output.get("prediction_table")

    if not isinstance(table, dict):
        raise ControlPlaneError(
            "Prediction-table artifact is malformed"
        )

    fingerprint = report.get(
        "prediction_fingerprint_sha256"
    )
    prediction_id = report.get("prediction_id")

    fingerprint_ok = bool(
        isinstance(fingerprint, str)
        and sha256_json(identity) == fingerprint
    )

    prediction_id_ok = bool(
        isinstance(fingerprint, str)
        and isinstance(prediction_id, str)
        and prediction_id == f"p1-{fingerprint[:16]}"
        and path.parent.name == prediction_id
    )

    batch_id = report.get("batch_id")
    backend = report.get("backend")
    n_rows = counts.get("n_rows")

    identity_bindings_ok = False

    if (
        isinstance(batch_id, str)
        and isinstance(backend, str)
        and isinstance(n_rows, int)
        and isinstance(table.get("sha256"), str)
    ):
        expected_identity = _identity(
            source_record=source,
            batch_id=batch_id,
            backend=backend,
            table_sha256=table["sha256"],
            n_rows=n_rows,
        )

        identity_bindings_ok = bool(
            identity == expected_identity
            and source.get("batch_id") == batch_id
            and report.get("assay_id") == ASSAY_ID
            and output.get("table_schema_version")
            == TABLE_SCHEMA_VERSION
            and output.get("burden_column") == BURDEN_COLUMN
            and output.get("fields") == list(OUTPUT_FIELDS)
        )

    scope_ok = scope == _scientific_scope()
    timestamp_ok = _timestamp_is_aware(
        report.get("created_utc")
    )

    source_ok = True
    table_ok = True
    derivation_ok = True

    if check_source_artifact:
        source_ok = _verify_artifact(source)
        table_ok = _verify_artifact(table)

        derivation_ok = False

        if source_ok and table_ok:
            source_path_value = source.get("path")
            table_path_value = table.get("path")

            if (
                isinstance(source_path_value, str)
                and isinstance(table_path_value, str)
            ):
                source_report = verify_batch_record(
                    Path(source_path_value),
                    check_artifacts=True,
                )

                if (
                    source_report["ok"]
                    and source_report.get("batch_id")
                    == batch_id
                ):
                    expected_bytes, rows = (
                        _render_prediction_table(
                            Path(source_path_value)
                        )
                    )

                    table_path = Path(table_path_value)

                    derivation_ok = bool(
                        table_path.read_bytes()
                        == expected_bytes
                        and len(rows) == n_rows
                        and _sha256_bytes(expected_bytes)
                        == table.get("sha256")
                    )

    ok = bool(
        report.get("status") == "complete"
        and fingerprint_ok
        and prediction_id_ok
        and identity_bindings_ok
        and scope_ok
        and timestamp_ok
        and source_ok
        and table_ok
        and derivation_ok
    )

    return {
        "kind": REPORT_TYPE,
        "prediction_id": prediction_id,
        "batch_id": batch_id,
        "report_path": str(path),
        "fingerprint_ok": fingerprint_ok,
        "prediction_id_ok": prediction_id_ok,
        "identity_bindings_ok": identity_bindings_ok,
        "scientific_scope_ok": scope_ok,
        "timestamp_ok": timestamp_ok,
        "source_artifact_check_performed": check_source_artifact,
        "source_batch_record_ok": source_ok,
        "prediction_table_ok": table_ok,
        "derivation_ok": derivation_ok,
        "n_rows": n_rows,
        "burden_column": output.get("burden_column"),
        "ok": ok,
    }
