"""High-level batch-to-measurement transaction orchestration.

v0.5 composes already validated lower-level control-plane operations:

batch plan
-> batch execution
-> NASA prediction-table projection
-> measurement planning
-> endpoint reconstruction
-> measurement record

This module introduces no new neural inference implementation, scientific
aggregation, biological-reference lookup, or biological acceptance logic.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from radiation_edge_ai.batch import (
    execute_batch_plan,
    verify_batch_plan,
    verify_batch_record,
)
from radiation_edge_ai.control import (
    ControlPlaneError,
    load_json_object,
    sha256_file,
    sha256_json,
)
from radiation_edge_ai.measurement import (
    create_measurement_plan,
    execute_measurement_plan,
    verify_measurement_plan,
    verify_measurement_record,
)
from radiation_edge_ai.nasa_predictions import (
    BURDEN_COLUMN,
    create_nasa_batch_prediction_report,
    verify_nasa_batch_prediction_report,
)

TRANSACTION_SCHEMA_VERSION = 1
TRANSACTION_RECORD_TYPE = "batch_measurement_transaction"

ASSAY_ID = "nasa-53bp1-r1-v2"
SUPPORTED_BACKEND = "cpu-onnx"


def _require_file(
    path: Path,
    *,
    label: str,
) -> Path:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise ControlPlaneError(
            f"{label} is not a readable file: {resolved}"
        )

    return resolved


def _artifact_record(
    path: Path,
) -> dict[str, object]:
    resolved = path.expanduser().resolve()

    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _verify_artifact(
    value: object,
) -> bool:
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


def _timestamp_is_aware(
    value: object,
) -> bool:
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
        "batch_inference_artifact_verified": True,
        "prediction_table_artifact_verified": True,
        "endpoint_reconstruction_artifact_verified": True,
        "prediction_semantics": (
            "per-nucleus latent continuous 53BP1 burden"
        ),
        "endpoint": (
            "sample-level arithmetic mean latent 53BP1 burden"
        ),
        "per_nucleus_focus_count_interpretation": False,
        "biological_reference_read": False,
        "biological_acceptance_evaluated": False,
        "hardware_access_performed": False,
    }


def _transaction_identity(
    *,
    batch_plan: Mapping[str, Any],
    batch_plan_record: Mapping[str, object],
) -> dict[str, Any]:
    batch_id = _required_text(
        batch_plan.get("batch_id"),
        label="batch_id",
    )

    fingerprint = _required_text(
        batch_plan.get("batch_fingerprint_sha256"),
        label="batch fingerprint",
    )

    backend = _required_text(
        batch_plan.get("backend"),
        label="batch backend",
    )

    return {
        "schema_version": TRANSACTION_SCHEMA_VERSION,
        "record_type": TRANSACTION_RECORD_TYPE,
        "assay_id": ASSAY_ID,
        "batch_id": batch_id,
        "backend": backend,
        "source_batch_plan_sha256": (
            batch_plan_record["sha256"]
        ),
        "source_batch_plan_size_bytes": (
            batch_plan_record["size_bytes"]
        ),
        "source_batch_fingerprint_sha256": fingerprint,
        "burden_column": BURDEN_COLUMN,
    }


def _record_identity(
    *,
    transaction_id: str,
    transaction_fingerprint: str,
    batch_plan_record: Mapping[str, object],
    batch_record_record: Mapping[str, object],
    prediction_report_record: Mapping[str, object],
    prediction_table_record: Mapping[str, object],
    measurement_plan_record: Mapping[str, object],
    measurement_record_record: Mapping[str, object],
    endpoint_report_record: Mapping[str, object],
    sample_aggregates_record: Mapping[str, object],
    batch_id: str,
    prediction_id: str,
    measurement_id: str,
    aggregation_id: str,
    n_items: int,
    n_prediction_rows: int,
    n_nuclei: int,
    n_samples: int,
) -> dict[str, Any]:
    return {
        "schema_version": TRANSACTION_SCHEMA_VERSION,
        "record_type": TRANSACTION_RECORD_TYPE,
        "assay_id": ASSAY_ID,
        "transaction_id": transaction_id,
        "transaction_fingerprint_sha256": (
            transaction_fingerprint
        ),
        "batch_id": batch_id,
        "prediction_id": prediction_id,
        "measurement_id": measurement_id,
        "aggregation_id": aggregation_id,
        "artifacts": {
            "batch_plan_sha256": batch_plan_record["sha256"],
            "batch_record_sha256": batch_record_record["sha256"],
            "prediction_report_sha256": (
                prediction_report_record["sha256"]
            ),
            "prediction_table_sha256": (
                prediction_table_record["sha256"]
            ),
            "measurement_plan_sha256": (
                measurement_plan_record["sha256"]
            ),
            "measurement_record_sha256": (
                measurement_record_record["sha256"]
            ),
            "endpoint_report_sha256": (
                endpoint_report_record["sha256"]
            ),
            "sample_aggregates_sha256": (
                sample_aggregates_record["sha256"]
            ),
        },
        "counts": {
            "n_items": n_items,
            "n_prediction_rows": n_prediction_rows,
            "n_nuclei": n_nuclei,
            "n_samples": n_samples,
        },
        "scientific_scope": _scientific_scope(),
    }


def execute_batch_measurement_transaction(
    batch_plan_path: Path,
    *,
    output_dir: Path,
    onnx_python: Optional[Path] = None,  # noqa: UP045
) -> Path:
    """Execute the complete verified NASA CPU batch-to-measurement path."""

    plan_file = _require_file(
        batch_plan_path,
        label="batch plan",
    )

    plan_verification = verify_batch_plan(
        plan_file,
        check_artifacts=True,
    )

    if not plan_verification["ok"]:
        raise ControlPlaneError(
            f"Batch plan failed verification: {plan_file}"
        )

    batch_plan = load_json_object(plan_file)

    if batch_plan.get("backend") != SUPPORTED_BACKEND:
        raise ControlPlaneError(
            "v0.5 batch-measure supports only cpu-onnx"
        )

    batch_plan_record = _artifact_record(plan_file)

    transaction_identity = _transaction_identity(
        batch_plan=batch_plan,
        batch_plan_record=batch_plan_record,
    )

    transaction_fingerprint = sha256_json(
        transaction_identity
    )
    transaction_id = (
        f"t1-{transaction_fingerprint[:16]}"
    )

    root = output_dir.expanduser().resolve()
    transaction_root = root / transaction_id
    transaction_record_path = (
        transaction_root / "transaction_record.json"
    )

    if transaction_record_path.exists():
        verification = verify_batch_measurement_transaction(
            transaction_record_path,
            check_artifacts=True,
        )

        if verification["ok"]:
            return transaction_record_path

        raise ControlPlaneError(
            "Existing transaction record failed verification: "
            f"{transaction_record_path}"
        )

    if (
        transaction_root.exists()
        and any(transaction_root.iterdir())
    ):
        raise ControlPlaneError(
            "Refusing incomplete/non-empty transaction directory "
            f"without a verified record: {transaction_root}"
        )

    # Stage 1: ordinary v0.4 batch execution.
    #
    # execute_batch_plan is itself idempotent. A pre-existing verified
    # batch record may therefore be reused without new neural inference.
    batch_record_path = execute_batch_plan(
        plan_file,
        onnx_python=onnx_python,
    )

    batch_verification = verify_batch_record(
        batch_record_path,
        check_artifacts=True,
    )

    if not batch_verification["ok"]:
        raise ControlPlaneError(
            "Batch record failed verification after execution/reuse"
        )

    transaction_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Stage 2: deterministic v0.4 prediction-table projection.
    prediction_report_path = (
        create_nasa_batch_prediction_report(
            batch_record_path=batch_record_path,
            output_dir=transaction_root / "predictions",
        )
    )

    prediction_verification = (
        verify_nasa_batch_prediction_report(
            prediction_report_path,
            check_source_artifact=True,
        )
    )

    if not prediction_verification["ok"]:
        raise ControlPlaneError(
            "Prediction report failed verification"
        )

    prediction_report = load_json_object(
        prediction_report_path
    )

    prediction_output = _require_mapping(
        prediction_report.get("output"),
        label="prediction output",
    )

    prediction_table = _require_mapping(
        prediction_output.get("prediction_table"),
        label="prediction table artifact",
    )

    prediction_table_path_value = prediction_table.get(
        "path"
    )

    if not isinstance(
        prediction_table_path_value,
        str,
    ):
        raise ControlPlaneError(
            "Prediction table path is malformed"
        )

    prediction_table_path = Path(
        prediction_table_path_value
    )

    # Stage 3: existing v0.3 measurement planning.
    measurement_plan_path = create_measurement_plan(
        predictions_path=prediction_table_path,
        burden_column=BURDEN_COLUMN,
        output_dir=transaction_root / "measurement",
    )

    measurement_plan_verification = (
        verify_measurement_plan(
            measurement_plan_path,
            check_source_artifact=True,
        )
    )

    if not measurement_plan_verification["ok"]:
        raise ControlPlaneError(
            "Measurement plan failed verification"
        )

    # Stage 4: existing v0.3 endpoint reconstruction.
    measurement_record_path = execute_measurement_plan(
        measurement_plan_path
    )

    measurement_verification = (
        verify_measurement_record(
            measurement_record_path,
            check_artifacts=True,
        )
    )

    if not measurement_verification["ok"]:
        raise ControlPlaneError(
            "Measurement record failed verification"
        )

    batch_record = load_json_object(
        batch_record_path
    )
    measurement_plan = load_json_object(
        measurement_plan_path
    )
    measurement_record = load_json_object(
        measurement_record_path
    )

    endpoint_binding = _require_mapping(
        measurement_record.get("endpoint_report"),
        label="measurement endpoint report",
    )

    aggregate_binding = _require_mapping(
        measurement_record.get("sample_aggregates"),
        label="measurement aggregate",
    )

    endpoint_path_value = endpoint_binding.get("path")
    aggregate_path_value = aggregate_binding.get("path")

    if not isinstance(endpoint_path_value, str):
        raise ControlPlaneError(
            "Endpoint report path is malformed"
        )

    if not isinstance(aggregate_path_value, str):
        raise ControlPlaneError(
            "Sample aggregate path is malformed"
        )

    endpoint_report_path = Path(
        endpoint_path_value
    )
    sample_aggregates_path = Path(
        aggregate_path_value
    )

    batch_record_record = _artifact_record(
        batch_record_path
    )
    prediction_report_record = _artifact_record(
        prediction_report_path
    )
    prediction_table_record = _artifact_record(
        prediction_table_path
    )
    measurement_plan_record = _artifact_record(
        measurement_plan_path
    )
    measurement_record_record = _artifact_record(
        measurement_record_path
    )
    endpoint_report_record = _artifact_record(
        endpoint_report_path
    )
    sample_aggregates_record = _artifact_record(
        sample_aggregates_path
    )

    batch_id = _required_text(
        batch_record.get("batch_id"),
        label="batch record batch_id",
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
        endpoint_binding.get("aggregation_id"),
        label="aggregation_id",
    )

    batch_counts = _require_mapping(
        batch_record.get("counts"),
        label="batch counts",
    )
    prediction_counts = _require_mapping(
        prediction_report.get("counts"),
        label="prediction counts",
    )
    measurement_counts = _require_mapping(
        measurement_record.get("counts"),
        label="measurement counts",
    )

    n_items = batch_counts.get("n_items")
    n_prediction_rows = prediction_counts.get(
        "n_rows"
    )
    n_nuclei = measurement_counts.get("n_nuclei")
    n_samples = measurement_counts.get("n_samples")

    if not isinstance(n_items, int):
        raise ControlPlaneError(
            "Batch item count is malformed"
        )

    if not isinstance(n_prediction_rows, int):
        raise ControlPlaneError(
            "Prediction row count is malformed"
        )

    if not isinstance(n_nuclei, int):
        raise ControlPlaneError(
            "Measurement nucleus count is malformed"
        )

    if not isinstance(n_samples, int):
        raise ControlPlaneError(
            "Measurement sample count is malformed"
        )

    # Explicit cross-stage bindings.
    source_batch = _require_mapping(
        prediction_report.get("source_batch_record"),
        label="prediction source batch",
    )

    measurement_source = _require_mapping(
        measurement_plan.get("source_predictions"),
        label="measurement source predictions",
    )

    measurement_plan_binding = _require_mapping(
        measurement_record.get("measurement_plan"),
        label="measurement-plan binding",
    )

    if batch_id != batch_plan.get("batch_id"):
        raise ControlPlaneError(
            "Batch record is not bound to source batch plan"
        )

    if (
        source_batch.get("sha256")
        != batch_record_record["sha256"]
    ):
        raise ControlPlaneError(
            "Prediction report is not bound to batch record"
        )

    if (
        measurement_source.get("sha256")
        != prediction_table_record["sha256"]
    ):
        raise ControlPlaneError(
            "Measurement plan is not bound to prediction table"
        )

    if (
        measurement_plan_binding.get("sha256")
        != measurement_plan_record["sha256"]
    ):
        raise ControlPlaneError(
            "Measurement record is not bound to measurement plan"
        )

    if (
        endpoint_binding.get("sha256")
        != endpoint_report_record["sha256"]
    ):
        raise ControlPlaneError(
            "Measurement record endpoint binding mismatch"
        )

    if (
        aggregate_binding.get("sha256")
        != sample_aggregates_record["sha256"]
    ):
        raise ControlPlaneError(
            "Measurement record aggregate binding mismatch"
        )

    if not (
        n_items
        == n_prediction_rows
        == n_nuclei
    ):
        raise ControlPlaneError(
            "Cross-stage nucleus/item counts are inconsistent"
        )

    started_utc = datetime.now(timezone.utc)

    record_identity = _record_identity(
        transaction_id=transaction_id,
        transaction_fingerprint=transaction_fingerprint,
        batch_plan_record=batch_plan_record,
        batch_record_record=batch_record_record,
        prediction_report_record=prediction_report_record,
        prediction_table_record=prediction_table_record,
        measurement_plan_record=measurement_plan_record,
        measurement_record_record=measurement_record_record,
        endpoint_report_record=endpoint_report_record,
        sample_aggregates_record=sample_aggregates_record,
        batch_id=batch_id,
        prediction_id=prediction_id,
        measurement_id=measurement_id,
        aggregation_id=aggregation_id,
        n_items=n_items,
        n_prediction_rows=n_prediction_rows,
        n_nuclei=n_nuclei,
        n_samples=n_samples,
    )

    record_fingerprint = sha256_json(
        record_identity
    )

    completed_utc = datetime.now(timezone.utc)

    record = {
        "schema_version": TRANSACTION_SCHEMA_VERSION,
        "record_type": TRANSACTION_RECORD_TYPE,
        "transaction_id": transaction_id,
        "status": "complete",
        "started_utc": started_utc.isoformat(),
        "completed_utc": completed_utc.isoformat(),
        "transaction_fingerprint_sha256": (
            transaction_fingerprint
        ),
        "transaction_identity": transaction_identity,
        "record_fingerprint_sha256": (
            record_fingerprint
        ),
        "record_identity": record_identity,
        "source_batch_plan": batch_plan_record,
        "batch_record": {
            **batch_record_record,
            "batch_id": batch_id,
        },
        "prediction_report": {
            **prediction_report_record,
            "prediction_id": prediction_id,
        },
        "prediction_table": prediction_table_record,
        "measurement_plan": {
            **measurement_plan_record,
            "measurement_id": measurement_id,
        },
        "measurement_record": {
            **measurement_record_record,
            "measurement_id": measurement_id,
        },
        "endpoint_report": {
            **endpoint_report_record,
            "aggregation_id": aggregation_id,
        },
        "sample_aggregates": (
            sample_aggregates_record
        ),
        "counts": {
            "n_items": n_items,
            "n_prediction_rows": n_prediction_rows,
            "n_nuclei": n_nuclei,
            "n_samples": n_samples,
        },
        "scientific_scope": _scientific_scope(),
        "interpretation": {
            "note": (
                "High-level orchestration only. Scientific inference, "
                "prediction projection, and endpoint reconstruction are "
                "delegated to previously validated lower-level stages."
            )
        },
    }

    temp_path = (
        transaction_root
        / "transaction_record.json.tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            record,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")

    temp_path.replace(
        transaction_record_path
    )

    return transaction_record_path


def verify_batch_measurement_transaction(
    record_path: Path,
    *,
    check_artifacts: bool = True,
) -> dict[str, object]:
    """Verify one completed v0.5 batch-measurement transaction."""

    path = record_path.expanduser().resolve()
    record = load_json_object(path)

    if (
        record.get("schema_version")
        != TRANSACTION_SCHEMA_VERSION
        or record.get("record_type")
        != TRANSACTION_RECORD_TYPE
    ):
        raise ControlPlaneError(
            f"Not a supported transaction record: {path}"
        )

    transaction_identity = record.get(
        "transaction_identity"
    )
    record_identity = record.get(
        "record_identity"
    )

    if not isinstance(transaction_identity, dict):
        raise ControlPlaneError(
            "Transaction identity is malformed"
        )

    if not isinstance(record_identity, dict):
        raise ControlPlaneError(
            "Transaction record identity is malformed"
        )

    transaction_fingerprint = record.get(
        "transaction_fingerprint_sha256"
    )
    record_fingerprint = record.get(
        "record_fingerprint_sha256"
    )
    transaction_id = record.get(
        "transaction_id"
    )

    transaction_fingerprint_ok = bool(
        isinstance(transaction_fingerprint, str)
        and sha256_json(transaction_identity)
        == transaction_fingerprint
    )

    transaction_id_ok = bool(
        isinstance(transaction_id, str)
        and isinstance(transaction_fingerprint, str)
        and transaction_id
        == f"t1-{transaction_fingerprint[:16]}"
        and path.parent.name == transaction_id
    )

    record_fingerprint_ok = bool(
        isinstance(record_fingerprint, str)
        and sha256_json(record_identity)
        == record_fingerprint
    )

    names = (
        "source_batch_plan",
        "batch_record",
        "prediction_report",
        "prediction_table",
        "measurement_plan",
        "measurement_record",
        "endpoint_report",
        "sample_aggregates",
    )

    artifacts: dict[str, Mapping[str, Any]] = {}

    for name in names:
        value = record.get(name)

        if not isinstance(value, Mapping):
            raise ControlPlaneError(
                f"Transaction artifact {name!r} is malformed"
            )

        artifacts[name] = value

    artifacts_ok = True

    if check_artifacts:
        artifacts_ok = all(
            _verify_artifact(value)
            for value in artifacts.values()
        )

    stages_ok = True
    cross_bindings_ok = True

    if check_artifacts and artifacts_ok:
        batch_plan_path = Path(
            str(artifacts["source_batch_plan"]["path"])
        )
        batch_record_path = Path(
            str(artifacts["batch_record"]["path"])
        )
        prediction_report_path = Path(
            str(artifacts["prediction_report"]["path"])
        )
        measurement_plan_path = Path(
            str(artifacts["measurement_plan"]["path"])
        )
        measurement_record_path = Path(
            str(artifacts["measurement_record"]["path"])
        )

        reports = (
            verify_batch_plan(
                batch_plan_path,
                check_artifacts=True,
            ),
            verify_batch_record(
                batch_record_path,
                check_artifacts=True,
            ),
            verify_nasa_batch_prediction_report(
                prediction_report_path,
                check_source_artifact=True,
            ),
            verify_measurement_plan(
                measurement_plan_path,
                check_source_artifact=True,
            ),
            verify_measurement_record(
                measurement_record_path,
                check_artifacts=True,
            ),
        )

        stages_ok = all(
            bool(stage["ok"])
            for stage in reports
        )

        if stages_ok:
            batch_plan = load_json_object(
                batch_plan_path
            )
            batch_record = load_json_object(
                batch_record_path
            )
            prediction_report = load_json_object(
                prediction_report_path
            )
            measurement_plan = load_json_object(
                measurement_plan_path
            )
            measurement_record = load_json_object(
                measurement_record_path
            )

            prediction_source = _require_mapping(
                prediction_report.get(
                    "source_batch_record"
                ),
                label="prediction source batch",
            )

            measurement_source = _require_mapping(
                measurement_plan.get(
                    "source_predictions"
                ),
                label="measurement source",
            )

            measurement_plan_binding = (
                _require_mapping(
                    measurement_record.get(
                        "measurement_plan"
                    ),
                    label="measurement-plan binding",
                )
            )

            endpoint_binding = _require_mapping(
                measurement_record.get(
                    "endpoint_report"
                ),
                label="endpoint binding",
            )

            aggregate_binding = _require_mapping(
                measurement_record.get(
                    "sample_aggregates"
                ),
                label="aggregate binding",
            )

            cross_bindings_ok = bool(
                batch_plan.get("batch_id")
                == batch_record.get("batch_id")
                == record.get("batch_record", {}).get(
                    "batch_id"
                )
                and prediction_report.get("batch_id")
                == batch_record.get("batch_id")
                and prediction_source.get("sha256")
                == artifacts["batch_record"].get(
                    "sha256"
                )
                and measurement_source.get("sha256")
                == artifacts["prediction_table"].get(
                    "sha256"
                )
                and measurement_plan_binding.get(
                    "sha256"
                )
                == artifacts["measurement_plan"].get(
                    "sha256"
                )
                and endpoint_binding.get("sha256")
                == artifacts["endpoint_report"].get(
                    "sha256"
                )
                and aggregate_binding.get("sha256")
                == artifacts["sample_aggregates"].get(
                    "sha256"
                )
            )

    counts = record.get("counts")

    if not isinstance(counts, Mapping):
        raise ControlPlaneError(
            "Transaction counts are malformed"
        )

    n_items = counts.get("n_items")
    n_prediction_rows = counts.get(
        "n_prediction_rows"
    )
    n_nuclei = counts.get("n_nuclei")
    n_samples = counts.get("n_samples")

    counts_ok = bool(
        isinstance(n_items, int)
        and isinstance(n_prediction_rows, int)
        and isinstance(n_nuclei, int)
        and isinstance(n_samples, int)
        and n_items == n_prediction_rows == n_nuclei
        and n_samples > 0
    )

    scope = record.get("scientific_scope")
    scope_ok = scope == _scientific_scope()

    transaction_identity_bindings_ok = bool(
        transaction_identity.get(
            "schema_version"
        )
        == TRANSACTION_SCHEMA_VERSION
        and transaction_identity.get(
            "record_type"
        )
        == TRANSACTION_RECORD_TYPE
        and transaction_identity.get(
            "assay_id"
        )
        == ASSAY_ID
        and transaction_identity.get(
            "batch_id"
        )
        == artifacts["batch_record"].get(
            "batch_id"
        )
        and transaction_identity.get(
            "backend"
        )
        == SUPPORTED_BACKEND
        and transaction_identity.get(
            "source_batch_plan_sha256"
        )
        == artifacts["source_batch_plan"].get(
            "sha256"
        )
        and transaction_identity.get(
            "source_batch_plan_size_bytes"
        )
        == artifacts["source_batch_plan"].get(
            "size_bytes"
        )
        and transaction_identity.get(
            "burden_column"
        )
        == BURDEN_COLUMN
    )

    expected_record_identity = _record_identity(
        transaction_id=str(transaction_id),
        transaction_fingerprint=str(
            transaction_fingerprint
        ),
        batch_plan_record=artifacts[
            "source_batch_plan"
        ],
        batch_record_record=artifacts[
            "batch_record"
        ],
        prediction_report_record=artifacts[
            "prediction_report"
        ],
        prediction_table_record=artifacts[
            "prediction_table"
        ],
        measurement_plan_record=artifacts[
            "measurement_plan"
        ],
        measurement_record_record=artifacts[
            "measurement_record"
        ],
        endpoint_report_record=artifacts[
            "endpoint_report"
        ],
        sample_aggregates_record=artifacts[
            "sample_aggregates"
        ],
        batch_id=str(
            artifacts["batch_record"].get(
                "batch_id"
            )
        ),
        prediction_id=str(
            artifacts["prediction_report"].get(
                "prediction_id"
            )
        ),
        measurement_id=str(
            artifacts["measurement_record"].get(
                "measurement_id"
            )
        ),
        aggregation_id=str(
            artifacts["endpoint_report"].get(
                "aggregation_id"
            )
        ),
        n_items=int(n_items),
        n_prediction_rows=int(
            n_prediction_rows
        ),
        n_nuclei=int(n_nuclei),
        n_samples=int(n_samples),
    )

    record_identity_bindings_ok = (
        record_identity
        == expected_record_identity
    )

    timestamps_ok = bool(
        _timestamp_is_aware(
            record.get("started_utc")
        )
        and _timestamp_is_aware(
            record.get("completed_utc")
        )
    )

    ok = bool(
        record.get("status") == "complete"
        and transaction_fingerprint_ok
        and transaction_id_ok
        and record_fingerprint_ok
        and transaction_identity_bindings_ok
        and record_identity_bindings_ok
        and counts_ok
        and scope_ok
        and artifacts_ok
        and stages_ok
        and cross_bindings_ok
        and timestamps_ok
    )

    return {
        "kind": TRANSACTION_RECORD_TYPE,
        "transaction_id": transaction_id,
        "record_path": str(path),
        "transaction_fingerprint_ok": (
            transaction_fingerprint_ok
        ),
        "record_fingerprint_ok": (
            record_fingerprint_ok
        ),
        "identity_bindings_ok": bool(
            transaction_identity_bindings_ok
            and record_identity_bindings_ok
        ),
        "artifact_check_performed": (
            check_artifacts
        ),
        "artifacts_ok": (
            artifacts_ok
            if check_artifacts
            else None
        ),
        "stages_ok": (
            stages_ok
            if check_artifacts
            else None
        ),
        "cross_artifact_bindings_ok": (
            cross_bindings_ok
            if check_artifacts
            else None
        ),
        "counts_ok": counts_ok,
        "scientific_scope_ok": scope_ok,
        "n_items": n_items,
        "n_prediction_rows": n_prediction_rows,
        "n_nuclei": n_nuclei,
        "n_samples": n_samples,
        "ok": ok,
    }
