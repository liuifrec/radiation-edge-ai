"""User-facing offline assay execution.

v0.6 provides a thin front door over the existing control-plane layers:

manifest
-> batch plan
-> v0.5 batch-measurement transaction
-> verified scientific summary

This module introduces no neural-inference implementation, prediction
transformation, endpoint calculation, biological-reference lookup, or
biological-acceptance logic.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Optional

from radiation_edge_ai.batch import (
    create_batch_plan,
    verify_batch_plan,
)
from radiation_edge_ai.control import (
    ControlPlaneError,
    load_json_object,
)
from radiation_edge_ai.transaction import (
    execute_batch_measurement_transaction,
    verify_batch_measurement_transaction,
)

ASSAY_ID = "nasa-53bp1-r1-v2"


def _required_mapping(
    value: object,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
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


def _read_sample_endpoints(
    transaction: dict[str, Any],
) -> list[dict[str, object]]:
    aggregate = _required_mapping(
        transaction.get("sample_aggregates"),
        label="sample aggregate artifact",
    )

    path_value = aggregate.get("path")

    if not isinstance(path_value, str):
        raise ControlPlaneError(
            "Sample aggregate path is malformed"
        )

    path = Path(path_value)

    if not path.is_file():
        raise ControlPlaneError(
            f"Sample aggregate file is missing: {path}"
        )

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ControlPlaneError(
            f"Could not read sample aggregate table: {exc}"
        ) from exc

    required = (
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

    missing = [
        field
        for field in required
        if field not in fields
    ]

    if missing:
        raise ControlPlaneError(
            "Sample aggregate table is missing columns: "
            + ", ".join(missing)
        )

    endpoints: list[dict[str, object]] = []

    for row in rows:
        sample_name = str(
            row["sample_name"]
        ).strip()

        if not sample_name:
            raise ControlPlaneError(
                "Sample aggregate contains blank sample_name"
            )

        try:
            n_nuclei = int(row["n_nuclei"])
            mean_burden = float(
                row["mean_latent_burden"]
            )
        except (TypeError, ValueError) as exc:
            raise ControlPlaneError(
                f"Malformed sample endpoint for {sample_name}"
            ) from exc

        if (
            n_nuclei <= 0
            or not math.isfinite(mean_burden)
        ):
            raise ControlPlaneError(
                f"Invalid sample endpoint for {sample_name}"
            )

        endpoints.append(
            {
                "sample_name": sample_name,
                "source_name": str(
                    row["source_name"]
                ).strip(),
                "strain": str(
                    row["strain"]
                ).strip(),
                "sex": str(
                    row["sex"]
                ).strip(),
                "radiation_branch": str(
                    row["radiation_branch"]
                ).strip(),
                "dose_gy": str(
                    row["dose_gy"]
                ).strip(),
                "timepoint_hr": str(
                    row["timepoint_hr"]
                ).strip(),
                "n_nuclei": n_nuclei,
                "mean_latent_burden": mean_burden,
            }
        )

    if not endpoints:
        raise ControlPlaneError(
            "Sample aggregate table contains no endpoints"
        )

    return endpoints


def execute_assay_manifest(
    manifest_path: Path,
    *,
    output_dir: Path,
    onnx_python: Optional[Path] = None,  # noqa: UP045
) -> dict[str, object]:
    """Execute one NASA assay manifest through the verified v0.5 path."""

    root = output_dir.expanduser().resolve()

    plan_path = create_batch_plan(
        manifest_path=manifest_path,
        output_dir=root / "plans",
    )

    plan_verification = verify_batch_plan(
        plan_path,
        check_artifacts=True,
    )

    if not plan_verification["ok"]:
        raise ControlPlaneError(
            "Generated/reused batch plan failed verification"
        )

    transaction_path = (
        execute_batch_measurement_transaction(
            plan_path,
            output_dir=root / "transactions",
            onnx_python=onnx_python,
        )
    )

    transaction_verification = (
        verify_batch_measurement_transaction(
            transaction_path,
            check_artifacts=True,
        )
    )

    if not transaction_verification["ok"]:
        raise ControlPlaneError(
            "Completed assay transaction failed verification"
        )

    transaction = load_json_object(
        transaction_path
    )

    batch_record = _required_mapping(
        transaction.get("batch_record"),
        label="batch record",
    )
    prediction_report = _required_mapping(
        transaction.get("prediction_report"),
        label="prediction report",
    )
    measurement_record = _required_mapping(
        transaction.get("measurement_record"),
        label="measurement record",
    )
    endpoint_report = _required_mapping(
        transaction.get("endpoint_report"),
        label="endpoint report",
    )
    counts = _required_mapping(
        transaction.get("counts"),
        label="transaction counts",
    )
    scope = _required_mapping(
        transaction.get("scientific_scope"),
        label="transaction scientific scope",
    )

    transaction_id = _required_text(
        transaction.get("transaction_id"),
        label="transaction_id",
    )
    batch_id = _required_text(
        batch_record.get("batch_id"),
        label="batch_id",
    )
    prediction_id = _required_text(
        prediction_report.get("prediction_id"),
        label="prediction_id",
    )
    measurement_id = _required_text(
        measurement_record.get("measurement_id"),
        label="measurement_id",
    )
    aggregation_id = _required_text(
        endpoint_report.get("aggregation_id"),
        label="aggregation_id",
    )

    endpoints = _read_sample_endpoints(
        transaction
    )

    n_items = counts.get("n_items")
    n_prediction_rows = counts.get(
        "n_prediction_rows"
    )
    n_nuclei = counts.get("n_nuclei")
    n_samples = counts.get("n_samples")

    if not all(
        isinstance(value, int)
        for value in (
            n_items,
            n_prediction_rows,
            n_nuclei,
            n_samples,
        )
    ):
        raise ControlPlaneError(
            "Transaction counts are malformed"
        )

    if len(endpoints) != n_samples:
        raise ControlPlaneError(
            "Sample summary count does not match "
            "verified transaction count"
        )

    return {
        "kind": "assay_run_summary",
        "assay_id": ASSAY_ID,
        "status": "complete",
        "verification": "PASS",
        "batch_plan_path": str(
            plan_path.resolve()
        ),
        "transaction_record_path": str(
            transaction_path.resolve()
        ),
        "ids": {
            "batch_id": batch_id,
            "transaction_id": transaction_id,
            "prediction_id": prediction_id,
            "measurement_id": measurement_id,
            "aggregation_id": aggregation_id,
        },
        "counts": {
            "n_items": n_items,
            "n_prediction_rows": (
                n_prediction_rows
            ),
            "n_nuclei": n_nuclei,
            "n_samples": n_samples,
        },
        "prediction_semantics": (
            "per-nucleus latent continuous 53BP1 burden"
        ),
        "endpoint_semantics": (
            "sample-level arithmetic mean latent 53BP1 burden"
        ),
        "sample_endpoints": endpoints,
        "scientific_scope": {
            "per_nucleus_focus_count_interpretation": (
                scope.get(
                    "per_nucleus_focus_count_interpretation"
                )
            ),
            "biological_reference_read": (
                scope.get(
                    "biological_reference_read"
                )
            ),
            "biological_acceptance_evaluated": (
                scope.get(
                    "biological_acceptance_evaluated"
                )
            ),
            "hardware_access_performed": (
                scope.get(
                    "hardware_access_performed"
                )
            ),
        },
    }
