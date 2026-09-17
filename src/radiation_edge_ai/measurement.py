"""Content-addressed measurement planning for Radiation Edge AI.

A measurement plan binds scientific intent to an immutable inference artifact.
Planning performs no neural inference, no endpoint reconstruction, no
biological-reference lookup, and no hardware access.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from radiation_edge_ai.control import (
    ControlPlaneError,
    get_assay,
    load_json_object,
    sha256_file,
    sha256_json,
)
from radiation_edge_ai.nasa_endpoint import (
    AGGREGATION_METHOD,
    ASSAY_ID,
    create_nasa_endpoint_report,
    verify_nasa_endpoint_report,
)

MEASUREMENT_PLAN_SCHEMA_VERSION = 1
MEASUREMENT_PLAN_TYPE = "measurement_plan"

MEASUREMENT_RECORD_SCHEMA_VERSION = 1
MEASUREMENT_RECORD_TYPE = "measurement_record"

NUCLEUS_IDENTITY_COLUMN = "sample_id"
GROUPING_COLUMN = "sample_name"

_REQUIRED_COLUMNS = (
    "sample_id",
    "sample_name",
    "source_name",
    "particle_type",
    "dose_Gy",
    "hr_post_exposure",
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


def _read_prediction_header(path: Path) -> list[str]:
    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ControlPlaneError(
                    f"Prediction CSV is empty: {path}"
                ) from exc
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ControlPlaneError(
            f"Could not read prediction CSV header {path}: {exc}"
        ) from exc

    fields = [str(value).strip() for value in header]

    if not fields or any(not field for field in fields):
        raise ControlPlaneError(
            f"Prediction CSV has an invalid header: {path}"
        )

    if len(fields) != len(set(fields)):
        raise ControlPlaneError(
            f"Prediction CSV contains duplicate columns: {path}"
        )

    return fields


def _validate_prediction_contract(
    path: Path,
    *,
    burden_column: str,
) -> None:
    burden = burden_column.strip()
    if not burden:
        raise ControlPlaneError(
            "burden_column must not be blank"
        )

    fields = _read_prediction_header(path)

    required = (*_REQUIRED_COLUMNS, burden)
    missing = [
        column
        for column in required
        if column not in fields
    ]

    if missing:
        raise ControlPlaneError(
            "Prediction CSV is missing required columns: "
            + ", ".join(missing)
        )


def _scientific_scope() -> dict[str, object]:
    return {
        "input_semantics": (
            "frozen per-nucleus latent continuous 53BP1 burden predictions"
        ),
        "endpoint": (
            "sample-level arithmetic mean latent 53BP1 burden"
        ),
        "per_nucleus_focus_count_interpretation": False,
        "endpoint_reconstruction_performed": False,
        "biological_reference_read": False,
        "biological_acceptance_evaluated": False,
        "hardware_access_performed": False,
    }


def _configuration(
    *,
    burden_column: str,
) -> dict[str, str]:
    return {
        "burden_column": burden_column,
        "nucleus_identity_column": NUCLEUS_IDENTITY_COLUMN,
        "grouping_column": GROUPING_COLUMN,
        "aggregation_method": AGGREGATION_METHOD,
    }


def _identity(
    *,
    source_record: dict[str, object],
    burden_column: str,
) -> dict[str, Any]:
    return {
        "schema_version": MEASUREMENT_PLAN_SCHEMA_VERSION,
        "record_type": MEASUREMENT_PLAN_TYPE,
        "assay_id": ASSAY_ID,
        "source_predictions_sha256": source_record["sha256"],
        "source_predictions_size_bytes": source_record["size_bytes"],
        "configuration": _configuration(
            burden_column=burden_column,
        ),
        "scientific_scope": _scientific_scope(),
    }


def create_measurement_plan(
    *,
    predictions_path: Path,
    burden_column: str,
    output_dir: Path,
) -> Path:
    """Create an immutable content-addressed NASA measurement plan."""

    source = _require_file(
        predictions_path,
        label="prediction CSV",
    )

    burden = burden_column.strip()
    _validate_prediction_contract(
        source,
        burden_column=burden,
    )

    spec = get_assay(ASSAY_ID)
    source_record = _artifact_record(source)

    identity = _identity(
        source_record=source_record,
        burden_column=burden,
    )
    fingerprint = sha256_json(identity)
    measurement_id = f"m1-{fingerprint[:16]}"

    root = output_dir.expanduser().resolve()
    plan_dir = root / measurement_id
    plan_path = plan_dir / "measurement_plan.json"

    if plan_path.exists():
        verification = verify_measurement_plan(plan_path)
        if verification["ok"]:
            return plan_path
        raise ControlPlaneError(
            "Existing measurement plan failed verification: "
            f"{plan_path}"
        )

    if plan_dir.exists():
        raise ControlPlaneError(
            "Content-addressed measurement directory already exists "
            f"without a verified plan: {plan_dir}"
        )

    plan_dir.mkdir(parents=True, exist_ok=False)

    plan = {
        "schema_version": MEASUREMENT_PLAN_SCHEMA_VERSION,
        "record_type": MEASUREMENT_PLAN_TYPE,
        "measurement_id": measurement_id,
        "status": "planned",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_fingerprint_sha256": fingerprint,
        "identity": identity,
        "assay": {
            "assay_id": spec.assay_id,
            "title": spec.title,
            "endpoint": spec.endpoint,
            "interpretation_guard": spec.interpretation_guard,
        },
        "source_predictions": source_record,
        "configuration": _configuration(
            burden_column=burden,
        ),
        "scientific_scope": _scientific_scope(),
    }

    temp_path = plan_dir / "measurement_plan.json.tmp"
    with temp_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            plan,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")

    temp_path.replace(plan_path)
    return plan_path


def _verify_artifact(
    record: dict[str, object],
) -> bool:
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

    return bool(
        sha256_file(path) == expected_hash
        and path.stat().st_size == expected_size
    )


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


def verify_measurement_plan(
    plan_path: Path,
    *,
    check_source_artifact: bool = True,
) -> dict[str, object]:
    """Verify one content-addressed measurement plan."""

    path = plan_path.expanduser().resolve()
    plan = load_json_object(path)

    if (
        plan.get("schema_version")
        != MEASUREMENT_PLAN_SCHEMA_VERSION
        or plan.get("record_type")
        != MEASUREMENT_PLAN_TYPE
    ):
        raise ControlPlaneError(
            f"Not a supported measurement plan: {path}"
        )

    identity = plan.get("identity")
    source = plan.get("source_predictions")
    configuration = plan.get("configuration")
    scope = plan.get("scientific_scope")
    assay = plan.get("assay")

    if not isinstance(identity, dict):
        raise ControlPlaneError(
            "Measurement-plan identity is malformed"
        )
    if not isinstance(source, dict):
        raise ControlPlaneError(
            "Measurement-plan source artifact is malformed"
        )
    if not isinstance(configuration, dict):
        raise ControlPlaneError(
            "Measurement-plan configuration is malformed"
        )
    if not isinstance(scope, dict):
        raise ControlPlaneError(
            "Measurement-plan scientific scope is malformed"
        )
    if not isinstance(assay, dict):
        raise ControlPlaneError(
            "Measurement-plan assay is malformed"
        )

    fingerprint = plan.get(
        "measurement_fingerprint_sha256"
    )
    measurement_id = plan.get("measurement_id")

    fingerprint_ok = bool(
        isinstance(fingerprint, str)
        and sha256_json(identity) == fingerprint
    )

    measurement_id_ok = bool(
        isinstance(fingerprint, str)
        and isinstance(measurement_id, str)
        and measurement_id
        == f"m1-{fingerprint[:16]}"
        and path.parent.name == measurement_id
    )

    expected_configuration = identity.get(
        "configuration"
    )
    expected_scope = identity.get(
        "scientific_scope"
    )

    identity_bindings_ok = bool(
        identity.get("schema_version")
        == MEASUREMENT_PLAN_SCHEMA_VERSION
        and identity.get("record_type")
        == MEASUREMENT_PLAN_TYPE
        and identity.get("assay_id") == ASSAY_ID
        and identity.get(
            "source_predictions_sha256"
        )
        == source.get("sha256")
        and identity.get(
            "source_predictions_size_bytes"
        )
        == source.get("size_bytes")
        and expected_configuration == configuration
        and expected_scope == scope
        and assay.get("assay_id") == ASSAY_ID
    )

    scope_ok = scope == _scientific_scope()

    source_ok = True
    source_contract_ok = True

    if check_source_artifact:
        source_ok = _verify_artifact(source)

        source_path = source.get("path")
        burden_column = configuration.get(
            "burden_column"
        )

        if (
            source_ok
            and isinstance(source_path, str)
            and isinstance(burden_column, str)
        ):
            try:
                _validate_prediction_contract(
                    Path(source_path),
                    burden_column=burden_column,
                )
            except ControlPlaneError:
                source_contract_ok = False
        else:
            source_contract_ok = False

    timestamps_ok = _timestamp_is_aware(
        plan.get("created_utc")
    )

    ok = bool(
        plan.get("status") == "planned"
        and fingerprint_ok
        and measurement_id_ok
        and identity_bindings_ok
        and scope_ok
        and source_ok
        and source_contract_ok
        and timestamps_ok
    )

    return {
        "kind": MEASUREMENT_PLAN_TYPE,
        "measurement_id": measurement_id,
        "plan_path": str(path),
        "fingerprint_ok": fingerprint_ok,
        "measurement_id_ok": measurement_id_ok,
        "identity_bindings_ok": identity_bindings_ok,
        "scientific_scope_ok": scope_ok,
        "timestamps_ok": timestamps_ok,
        "source_artifact_check_performed": (
            check_source_artifact
        ),
        "source_predictions_ok": bool(
            source_ok and source_contract_ok
        ),
        "ok": ok,
    }



def _completed_scientific_scope() -> dict[str, object]:
    return {
        "input_semantics": (
            "frozen per-nucleus latent continuous 53BP1 burden predictions"
        ),
        "endpoint": (
            "sample-level arithmetic mean latent 53BP1 burden"
        ),
        "per_nucleus_focus_count_interpretation": False,
        "endpoint_reconstruction_performed": True,
        "biological_reference_read": False,
        "biological_acceptance_evaluated": False,
        "hardware_access_performed": False,
    }


def _record_identity(
    *,
    plan_sha256: str,
    measurement_id: str,
    measurement_fingerprint_sha256: str,
    endpoint_report_sha256: str,
    aggregation_id: str,
    aggregate_sha256: str,
    n_nuclei: int,
    n_samples: int,
) -> dict[str, Any]:
    return {
        "schema_version": MEASUREMENT_RECORD_SCHEMA_VERSION,
        "record_type": MEASUREMENT_RECORD_TYPE,
        "assay_id": ASSAY_ID,
        "measurement_id": measurement_id,
        "measurement_plan_sha256": plan_sha256,
        "measurement_fingerprint_sha256": (
            measurement_fingerprint_sha256
        ),
        "endpoint_report_sha256": endpoint_report_sha256,
        "aggregation_id": aggregation_id,
        "sample_aggregates_sha256": aggregate_sha256,
        "n_nuclei": n_nuclei,
        "n_samples": n_samples,
        "scientific_scope": _completed_scientific_scope(),
    }


def execute_measurement_plan(
    plan_path: Path,
) -> Path:
    """Execute a verified measurement plan by reconstructing its endpoint.

    This operation performs no neural inference, no hardware access, and no
    biological-reference evaluation.
    """

    plan_file = plan_path.expanduser().resolve()

    verification = verify_measurement_plan(
        plan_file,
        check_source_artifact=True,
    )
    if not verification["ok"]:
        raise ControlPlaneError(
            f"Measurement plan failed verification: {plan_file}"
        )

    plan = load_json_object(plan_file)

    measurement_id = plan.get("measurement_id")
    measurement_fingerprint = plan.get(
        "measurement_fingerprint_sha256"
    )
    source = plan.get("source_predictions")
    configuration = plan.get("configuration")

    if not isinstance(measurement_id, str):
        raise ControlPlaneError(
            "Measurement plan has invalid measurement_id"
        )
    if not isinstance(measurement_fingerprint, str):
        raise ControlPlaneError(
            "Measurement plan has invalid fingerprint"
        )
    if not isinstance(source, dict):
        raise ControlPlaneError(
            "Measurement plan source artifact is malformed"
        )
    if not isinstance(configuration, dict):
        raise ControlPlaneError(
            "Measurement plan configuration is malformed"
        )

    source_path_value = source.get("path")
    burden_column = configuration.get("burden_column")

    if not isinstance(source_path_value, str):
        raise ControlPlaneError(
            "Measurement plan source path is malformed"
        )
    if not isinstance(burden_column, str):
        raise ControlPlaneError(
            "Measurement plan burden column is malformed"
        )

    record_path = plan_file.parent / "measurement_record.json"

    if record_path.exists():
        existing = verify_measurement_record(
            record_path,
            check_artifacts=True,
        )
        if existing["ok"]:
            return record_path
        raise ControlPlaneError(
            "Existing measurement record failed verification: "
            f"{record_path}"
        )

    started_utc = datetime.now(timezone.utc)

    endpoint_root = plan_file.parent / "endpoint"
    endpoint_report_path = create_nasa_endpoint_report(
        predictions_path=Path(source_path_value),
        burden_column=burden_column,
        output_dir=endpoint_root,
    )

    endpoint_verification = verify_nasa_endpoint_report(
        endpoint_report_path,
        check_source_artifact=True,
    )
    if not endpoint_verification["ok"]:
        raise ControlPlaneError(
            "Endpoint report failed verification after reconstruction: "
            f"{endpoint_report_path}"
        )

    endpoint = load_json_object(endpoint_report_path)
    endpoint_source = endpoint.get("source_predictions")
    endpoint_configuration = endpoint.get("configuration")
    endpoint_outputs = endpoint.get("outputs")
    endpoint_counts = endpoint.get("counts")

    if not isinstance(endpoint_source, dict):
        raise ControlPlaneError(
            "Endpoint report source artifact is malformed"
        )
    if not isinstance(endpoint_configuration, dict):
        raise ControlPlaneError(
            "Endpoint report configuration is malformed"
        )
    if not isinstance(endpoint_outputs, dict):
        raise ControlPlaneError(
            "Endpoint report outputs are malformed"
        )
    if not isinstance(endpoint_counts, dict):
        raise ControlPlaneError(
            "Endpoint report counts are malformed"
        )

    aggregate = endpoint_outputs.get("sample_aggregates")
    if not isinstance(aggregate, dict):
        raise ControlPlaneError(
            "Endpoint report aggregate artifact is malformed"
        )

    if endpoint_source.get("sha256") != source.get("sha256"):
        raise ControlPlaneError(
            "Endpoint report is not bound to measurement source predictions"
        )

    expected_endpoint_configuration = {
        "burden_column": configuration.get("burden_column"),
        "nucleus_identity_column": configuration.get(
            "nucleus_identity_column"
        ),
        "grouping_column": configuration.get("grouping_column"),
        "aggregation_method": configuration.get(
            "aggregation_method"
        ),
    }

    if endpoint_configuration != expected_endpoint_configuration:
        raise ControlPlaneError(
            "Endpoint configuration does not match measurement plan"
        )

    aggregation_id = endpoint.get("aggregation_id")
    aggregate_sha256 = aggregate.get("sha256")
    n_nuclei = endpoint_counts.get("n_nuclei")
    n_samples = endpoint_counts.get("n_samples")

    if not isinstance(aggregation_id, str):
        raise ControlPlaneError(
            "Endpoint report has invalid aggregation_id"
        )
    if not isinstance(aggregate_sha256, str):
        raise ControlPlaneError(
            "Endpoint aggregate SHA256 is malformed"
        )
    if not isinstance(n_nuclei, int):
        raise ControlPlaneError(
            "Endpoint nucleus count is malformed"
        )
    if not isinstance(n_samples, int):
        raise ControlPlaneError(
            "Endpoint sample count is malformed"
        )

    plan_sha256 = sha256_file(plan_file)
    endpoint_sha256 = sha256_file(endpoint_report_path)

    identity = _record_identity(
        plan_sha256=plan_sha256,
        measurement_id=measurement_id,
        measurement_fingerprint_sha256=measurement_fingerprint,
        endpoint_report_sha256=endpoint_sha256,
        aggregation_id=aggregation_id,
        aggregate_sha256=aggregate_sha256,
        n_nuclei=n_nuclei,
        n_samples=n_samples,
    )

    record_fingerprint = sha256_json(identity)
    ended_utc = datetime.now(timezone.utc)

    record = {
        "schema_version": MEASUREMENT_RECORD_SCHEMA_VERSION,
        "record_type": MEASUREMENT_RECORD_TYPE,
        "measurement_id": measurement_id,
        "status": "complete",
        "started_utc": started_utc.isoformat(),
        "ended_utc": ended_utc.isoformat(),
        "completed_utc": ended_utc.isoformat(),
        "record_fingerprint_sha256": record_fingerprint,
        "identity": identity,
        "measurement_plan": {
            "path": str(plan_file),
            "sha256": plan_sha256,
            "measurement_fingerprint_sha256": (
                measurement_fingerprint
            ),
        },
        "endpoint_report": {
            "path": str(endpoint_report_path),
            "sha256": endpoint_sha256,
            "aggregation_id": aggregation_id,
        },
        "sample_aggregates": {
            "path": aggregate.get("path"),
            "sha256": aggregate_sha256,
            "size_bytes": aggregate.get("size_bytes"),
        },
        "counts": {
            "n_nuclei": n_nuclei,
            "n_samples": n_samples,
        },
        "scientific_scope": _completed_scientific_scope(),
    }

    temp_path = plan_file.parent / "measurement_record.json.tmp"
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

    temp_path.replace(record_path)
    return record_path


def verify_measurement_record(
    record_path: Path,
    *,
    check_artifacts: bool = True,
) -> dict[str, object]:
    """Verify a completed measurement orchestration record."""

    path = record_path.expanduser().resolve()
    record = load_json_object(path)

    if (
        record.get("schema_version")
        != MEASUREMENT_RECORD_SCHEMA_VERSION
        or record.get("record_type")
        != MEASUREMENT_RECORD_TYPE
    ):
        raise ControlPlaneError(
            f"Not a supported measurement record: {path}"
        )

    identity = record.get("identity")
    plan_record = record.get("measurement_plan")
    endpoint_record = record.get("endpoint_report")
    aggregate_record = record.get("sample_aggregates")
    counts = record.get("counts")
    scope = record.get("scientific_scope")

    if not isinstance(identity, dict):
        raise ControlPlaneError(
            "Measurement-record identity is malformed"
        )
    if not isinstance(plan_record, dict):
        raise ControlPlaneError(
            "Measurement-record plan binding is malformed"
        )
    if not isinstance(endpoint_record, dict):
        raise ControlPlaneError(
            "Measurement-record endpoint binding is malformed"
        )
    if not isinstance(aggregate_record, dict):
        raise ControlPlaneError(
            "Measurement-record aggregate binding is malformed"
        )
    if not isinstance(counts, dict):
        raise ControlPlaneError(
            "Measurement-record counts are malformed"
        )
    if not isinstance(scope, dict):
        raise ControlPlaneError(
            "Measurement-record scientific scope is malformed"
        )

    fingerprint = record.get("record_fingerprint_sha256")
    fingerprint_ok = bool(
        isinstance(fingerprint, str)
        and sha256_json(identity) == fingerprint
    )

    measurement_id = record.get("measurement_id")

    plan_path_value = plan_record.get("path")
    endpoint_path_value = endpoint_record.get("path")
    aggregate_path_value = aggregate_record.get("path")

    plan_ok = False
    endpoint_ok = False
    aggregate_ok = False
    bindings_ok = False

    plan: Optional[dict[str, Any]] = None  # noqa: UP045
    endpoint: Optional[dict[str, Any]] = None  # noqa: UP045

    if isinstance(plan_path_value, str):
        plan_path = Path(plan_path_value)

        if plan_path.is_file():
            plan_hash_ok = (
                sha256_file(plan_path)
                == plan_record.get("sha256")
            )

            if plan_hash_ok:
                plan_verification = verify_measurement_plan(
                    plan_path,
                    check_source_artifact=check_artifacts,
                )
                plan_ok = bool(plan_verification["ok"])
                if plan_ok:
                    plan = load_json_object(plan_path)

    if isinstance(endpoint_path_value, str):
        endpoint_path = Path(endpoint_path_value)

        if endpoint_path.is_file():
            endpoint_hash_ok = (
                sha256_file(endpoint_path)
                == endpoint_record.get("sha256")
            )

            if endpoint_hash_ok:
                endpoint_verification = (
                    verify_nasa_endpoint_report(
                        endpoint_path,
                        check_source_artifact=check_artifacts,
                    )
                )
                endpoint_ok = bool(
                    endpoint_verification["ok"]
                )
                if endpoint_ok:
                    endpoint = load_json_object(
                        endpoint_path
                    )

    if (
        isinstance(aggregate_path_value, str)
        and isinstance(
            aggregate_record.get("sha256"),
            str,
        )
        and isinstance(
            aggregate_record.get("size_bytes"),
            int,
        )
    ):
        aggregate_path = Path(aggregate_path_value)
        aggregate_ok = bool(
            aggregate_path.is_file()
            and sha256_file(aggregate_path)
            == aggregate_record.get("sha256")
            and aggregate_path.stat().st_size
            == aggregate_record.get("size_bytes")
        )

    if plan is not None and endpoint is not None:
        plan_source = plan.get("source_predictions")
        plan_config = plan.get("configuration")

        endpoint_source = endpoint.get(
            "source_predictions"
        )
        endpoint_config = endpoint.get(
            "configuration"
        )
        endpoint_outputs = endpoint.get("outputs")
        endpoint_counts = endpoint.get("counts")

        if (
            isinstance(plan_source, dict)
            and isinstance(plan_config, dict)
            and isinstance(endpoint_source, dict)
            and isinstance(endpoint_config, dict)
            and isinstance(endpoint_outputs, dict)
            and isinstance(endpoint_counts, dict)
        ):
            endpoint_aggregate = endpoint_outputs.get(
                "sample_aggregates"
            )

            if isinstance(endpoint_aggregate, dict):
                bindings_ok = bool(
                    measurement_id
                    == plan.get("measurement_id")
                    and plan_record.get(
                        "measurement_fingerprint_sha256"
                    )
                    == plan.get(
                        "measurement_fingerprint_sha256"
                    )
                    and endpoint_record.get(
                        "aggregation_id"
                    )
                    == endpoint.get("aggregation_id")
                    and plan_source.get("sha256")
                    == endpoint_source.get("sha256")
                    and plan_config
                    == endpoint_config
                    and aggregate_record.get("sha256")
                    == endpoint_aggregate.get("sha256")
                    and counts.get("n_nuclei")
                    == endpoint_counts.get("n_nuclei")
                    and counts.get("n_samples")
                    == endpoint_counts.get("n_samples")
                )

    identity_bindings_ok = bool(
        identity.get("schema_version")
        == MEASUREMENT_RECORD_SCHEMA_VERSION
        and identity.get("record_type")
        == MEASUREMENT_RECORD_TYPE
        and identity.get("assay_id") == ASSAY_ID
        and identity.get("measurement_id")
        == measurement_id
        and identity.get("measurement_plan_sha256")
        == plan_record.get("sha256")
        and identity.get(
            "measurement_fingerprint_sha256"
        )
        == plan_record.get(
            "measurement_fingerprint_sha256"
        )
        and identity.get("endpoint_report_sha256")
        == endpoint_record.get("sha256")
        and identity.get("aggregation_id")
        == endpoint_record.get("aggregation_id")
        and identity.get("sample_aggregates_sha256")
        == aggregate_record.get("sha256")
        and identity.get("n_nuclei")
        == counts.get("n_nuclei")
        and identity.get("n_samples")
        == counts.get("n_samples")
        and identity.get("scientific_scope")
        == scope
    )

    scope_ok = scope == _completed_scientific_scope()

    started = record.get("started_utc")
    ended = record.get("ended_utc")
    completed = record.get("completed_utc")

    timestamps_ok = bool(
        _timestamp_is_aware(started)
        and _timestamp_is_aware(ended)
        and _timestamp_is_aware(completed)
    )

    if timestamps_ok:
        assert isinstance(started, str)
        assert isinstance(ended, str)
        assert isinstance(completed, str)

        started_dt = datetime.fromisoformat(started)
        ended_dt = datetime.fromisoformat(ended)

        timestamps_ok = bool(
            started_dt <= ended_dt
            and completed == ended
        )

    ok = bool(
        record.get("status") == "complete"
        and fingerprint_ok
        and identity_bindings_ok
        and scope_ok
        and timestamps_ok
        and plan_ok
        and endpoint_ok
        and aggregate_ok
        and bindings_ok
    )

    return {
        "kind": MEASUREMENT_RECORD_TYPE,
        "measurement_id": measurement_id,
        "record_path": str(path),
        "fingerprint_ok": fingerprint_ok,
        "identity_bindings_ok": identity_bindings_ok,
        "scientific_scope_ok": scope_ok,
        "timestamps_ok": timestamps_ok,
        "plan_ok": plan_ok,
        "endpoint_report_ok": endpoint_ok,
        "sample_aggregates_ok": aggregate_ok,
        "cross_artifact_bindings_ok": bindings_ok,
        "artifact_check_performed": check_artifacts,
        "n_nuclei": counts.get("n_nuclei"),
        "n_samples": counts.get("n_samples"),
        "ok": ok,
    }
