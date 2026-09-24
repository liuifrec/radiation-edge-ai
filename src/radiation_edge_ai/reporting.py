"""Portable assay-result export.

v0.7 creates a compact, path-independent result package from a fully verified
v0.5 batch-measurement transaction.

The package is a derived presentation/export artifact. It is not a replacement
for the source transaction record and introduces no neural inference,
prediction transformation, endpoint calculation, biological-reference lookup,
or biological-acceptance logic.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from radiation_edge_ai.control import (
    ControlPlaneError,
    load_json_object,
    sha256_file,
    sha256_json,
)
from radiation_edge_ai.transaction import (
    verify_batch_measurement_transaction,
)

PACKAGE_SCHEMA_VERSION = 1
PACKAGE_RECORD_TYPE = "assay_result_package"
ASSAY_ID = "nasa-53bp1-r1-v2"


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


def _portable_artifact_record(
    package_root: Path,
    path: Path,
) -> dict[str, object]:
    resolved_root = package_root.resolve()
    resolved_path = path.resolve()

    try:
        relative = resolved_path.relative_to(
            resolved_root
        )
    except ValueError as exc:
        raise ControlPlaneError(
            "Result-package artifact escaped package root"
        ) from exc

    return {
        "relative_path": relative.as_posix(),
        "sha256": sha256_file(resolved_path),
        "size_bytes": resolved_path.stat().st_size,
    }


def _verify_portable_artifact(
    package_root: Path,
    value: object,
) -> bool:
    if not isinstance(value, Mapping):
        return False

    relative_value = value.get("relative_path")
    expected_sha = value.get("sha256")
    expected_size = value.get("size_bytes")

    if (
        not isinstance(relative_value, str)
        or not isinstance(expected_sha, str)
        or not isinstance(expected_size, int)
    ):
        return False

    relative = Path(relative_value)

    if relative.is_absolute():
        return False

    candidate = (
        package_root / relative
    ).resolve()

    try:
        candidate.relative_to(
            package_root.resolve()
        )
    except ValueError:
        return False

    return bool(
        candidate.is_file()
        and sha256_file(candidate) == expected_sha
        and candidate.stat().st_size == expected_size
    )


def _read_endpoint_rows(
    path: Path,
) -> list[dict[str, object]]:
    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            raw_rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ControlPlaneError(
            f"Could not read endpoint table: {exc}"
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
        name
        for name in required
        if name not in fields
    ]

    if missing:
        raise ControlPlaneError(
            "Endpoint table is missing columns: "
            + ", ".join(missing)
        )

    result: list[dict[str, object]] = []

    for row in raw_rows:
        sample_name = str(
            row["sample_name"]
        ).strip()

        if not sample_name:
            raise ControlPlaneError(
                "Endpoint table contains blank sample_name"
            )

        try:
            n_nuclei = int(row["n_nuclei"])
            burden = float(
                row["mean_latent_burden"]
            )
        except (TypeError, ValueError) as exc:
            raise ControlPlaneError(
                f"Malformed endpoint for {sample_name}"
            ) from exc

        if (
            n_nuclei <= 0
            or not math.isfinite(burden)
        ):
            raise ControlPlaneError(
                f"Invalid endpoint for {sample_name}"
            )

        result.append(
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
                "mean_latent_burden": burden,
            }
        )

    if not result:
        raise ControlPlaneError(
            "Endpoint table contains no sample endpoints"
        )

    return result


def _scientific_scope(
    transaction: Mapping[str, Any],
) -> dict[str, object]:
    scope = _require_mapping(
        transaction.get("scientific_scope"),
        label="transaction scientific scope",
    )

    expected_false = (
        "per_nucleus_focus_count_interpretation",
        "biological_reference_read",
        "biological_acceptance_evaluated",
        "hardware_access_performed",
    )

    for key in expected_false:
        if scope.get(key) is not False:
            raise ControlPlaneError(
                f"Unexpected scientific scope for {key}"
            )

    return {
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


def _component_ids(
    transaction: Mapping[str, Any],
) -> dict[str, str]:
    batch = _require_mapping(
        transaction.get("batch_record"),
        label="batch record",
    )
    prediction = _require_mapping(
        transaction.get("prediction_report"),
        label="prediction report",
    )
    measurement = _require_mapping(
        transaction.get("measurement_record"),
        label="measurement record",
    )
    endpoint = _require_mapping(
        transaction.get("endpoint_report"),
        label="endpoint report",
    )

    return {
        "transaction_id": _required_text(
            transaction.get("transaction_id"),
            label="transaction_id",
        ),
        "batch_id": _required_text(
            batch.get("batch_id"),
            label="batch_id",
        ),
        "prediction_id": _required_text(
            prediction.get("prediction_id"),
            label="prediction_id",
        ),
        "measurement_id": _required_text(
            measurement.get("measurement_id"),
            label="measurement_id",
        ),
        "aggregation_id": _required_text(
            endpoint.get("aggregation_id"),
            label="aggregation_id",
        ),
    }


def _component_hashes(
    transaction: Mapping[str, Any],
) -> dict[str, str]:
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

    hashes: dict[str, str] = {}

    for name in names:
        value = _require_mapping(
            transaction.get(name),
            label=f"transaction artifact {name}",
        )

        hashes[name] = _required_text(
            value.get("sha256"),
            label=f"{name} sha256",
        )

    return hashes


def _counts(
    transaction: Mapping[str, Any],
) -> dict[str, int]:
    value = _require_mapping(
        transaction.get("counts"),
        label="transaction counts",
    )

    keys = (
        "n_items",
        "n_prediction_rows",
        "n_nuclei",
        "n_samples",
    )

    result: dict[str, int] = {}

    for key in keys:
        item = value.get(key)

        if not isinstance(item, int):
            raise ControlPlaneError(
                f"Malformed transaction count: {key}"
            )

        result[key] = item

    if not (
        result["n_items"]
        == result["n_prediction_rows"]
        == result["n_nuclei"]
    ):
        raise ControlPlaneError(
            "Transaction nucleus/item counts are inconsistent"
        )

    if result["n_samples"] <= 0:
        raise ControlPlaneError(
            "Transaction contains no sample endpoints"
        )

    return result


def _build_result(
    *,
    package_id: str,
    ids: Mapping[str, str],
    counts: Mapping[str, int],
    endpoints: list[dict[str, object]],
    scope: Mapping[str, object],
    transaction_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "assay_result",
        "package_id": package_id,
        "assay_id": ASSAY_ID,
        "verification": "PASS",
        "source_transaction": {
            "transaction_id": ids["transaction_id"],
            "sha256": transaction_sha256,
        },
        "ids": dict(ids),
        "counts": dict(counts),
        "prediction_semantics": (
            "per-nucleus latent continuous 53BP1 burden"
        ),
        "endpoint_semantics": (
            "sample-level arithmetic mean latent 53BP1 burden"
        ),
        "sample_endpoints": endpoints,
        "scientific_scope": dict(scope),
        "interpretation": {
            "package_role": (
                "derived portable result/export artifact"
            ),
            "provenance_authority": (
                "source v0.5 transaction record"
            ),
            "biological_validation_result": False,
        },
    }


def _build_provenance(
    *,
    ids: Mapping[str, str],
    component_hashes: Mapping[str, str],
    transaction: Mapping[str, Any],
    transaction_sha256: str,
    transaction_size_bytes: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "assay_result_provenance",
        "assay_id": ASSAY_ID,
        "source_transaction": {
            "transaction_id": ids["transaction_id"],
            "sha256": transaction_sha256,
            "size_bytes": transaction_size_bytes,
            "transaction_fingerprint_sha256": (
                transaction.get(
                    "transaction_fingerprint_sha256"
                )
            ),
            "record_fingerprint_sha256": (
                transaction.get(
                    "record_fingerprint_sha256"
                )
            ),
            "verified_at_export": True,
        },
        "component_ids": dict(ids),
        "component_sha256": dict(component_hashes),
        "path_independent_export": True,
        "upstream_binary_artifacts_included": False,
    }


def _render_markdown(
    result: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    ids = _require_mapping(
        result.get("ids"),
        label="result ids",
    )
    counts = _require_mapping(
        result.get("counts"),
        label="result counts",
    )
    endpoints = result.get("sample_endpoints")
    scope = _require_mapping(
        result.get("scientific_scope"),
        label="result scientific scope",
    )
    source = _require_mapping(
        provenance.get("source_transaction"),
        label="source transaction provenance",
    )

    if not isinstance(endpoints, list):
        raise ControlPlaneError(
            "Result sample endpoints are malformed"
        )

    lines = [
        "# Radiation Edge AI assay result",
        "",
        f"- Assay: `{ASSAY_ID}`",
        f"- Package: `{result['package_id']}`",
        f"- Verification: `{result['verification']}`",
        f"- Transaction: `{ids['transaction_id']}`",
        f"- Batch: `{ids['batch_id']}`",
        f"- Prediction projection: `{ids['prediction_id']}`",
        f"- Measurement: `{ids['measurement_id']}`",
        f"- Aggregation: `{ids['aggregation_id']}`",
        "",
        "## Endpoint",
        "",
        (
            "Per-nucleus model output is a latent continuous 53BP1 "
            "burden score. It is not an individually supervised "
            "focus count."
        ),
        "",
        (
            "The reported assay endpoint is the sample-level "
            "arithmetic mean latent 53BP1 burden."
        ),
        "",
        "| Sample | Branch | Dose (Gy) | Time (h) | N nuclei | Mean latent burden |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for endpoint in endpoints:
        row = _require_mapping(
            endpoint,
            label="sample endpoint",
        )
        lines.append(
            "| "
            f"{row['sample_name']} | "
            f"{row['radiation_branch']} | "
            f"{row['dose_gy']} | "
            f"{row['timepoint_hr']} | "
            f"{row['n_nuclei']} | "
            f"{row['mean_latent_burden']} |"
        )

    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Batch items: {counts['n_items']}",
            (
                "- Prediction rows: "
                f"{counts['n_prediction_rows']}"
            ),
            f"- Nuclei: {counts['n_nuclei']}",
            f"- Sample endpoints: {counts['n_samples']}",
            "",
            "## Scientific scope",
            "",
            (
                "- Per-nucleus focus-count interpretation: "
                f"{scope['per_nucleus_focus_count_interpretation']}"
            ),
            (
                "- Biological reference read during this run: "
                f"{scope['biological_reference_read']}"
            ),
            (
                "- Biological acceptance evaluated during this run: "
                f"{scope['biological_acceptance_evaluated']}"
            ),
            (
                "- KL720 hardware accessed during this run: "
                f"{scope['hardware_access_performed']}"
            ),
            "",
            "## Provenance",
            "",
            (
                "- Source transaction SHA256: "
                f"`{source['sha256']}`"
            ),
            "",
            (
                "This package is a derived presentation/export artifact. "
                "The source transaction record remains the provenance "
                "authority."
            ),
            "",
            (
                "The package does not contain the model, input tensors, "
                "raw microscopy, or biological reference."
            ),
            "",
            (
                "This package is not a new biological-validation, "
                "holdout-performance, or statistical-equivalence result."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def _write_json(
    path: Path,
    value: object,
) -> None:
    with path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def create_assay_result_package(
    transaction_path: Path,
    *,
    output_dir: Path,
) -> Path:
    """Export one fully verified transaction into a portable result package."""

    source = transaction_path.expanduser().resolve()

    if not source.is_file():
        raise ControlPlaneError(
            f"Transaction record does not exist: {source}"
        )

    verification = verify_batch_measurement_transaction(
        source,
        check_artifacts=True,
    )

    if not verification["ok"]:
        raise ControlPlaneError(
            "Source transaction failed full verification"
        )

    transaction = load_json_object(source)

    ids = _component_ids(transaction)
    counts = _counts(transaction)
    scope = _scientific_scope(transaction)
    component_hashes = _component_hashes(
        transaction
    )

    aggregate_record = _require_mapping(
        transaction.get("sample_aggregates"),
        label="sample aggregate artifact",
    )

    aggregate_path_value = aggregate_record.get(
        "path"
    )

    if not isinstance(
        aggregate_path_value,
        str,
    ):
        raise ControlPlaneError(
            "Sample aggregate path is malformed"
        )

    aggregate_source = Path(
        aggregate_path_value
    ).resolve()

    if (
        not aggregate_source.is_file()
        or sha256_file(aggregate_source)
        != component_hashes["sample_aggregates"]
    ):
        raise ControlPlaneError(
            "Source sample aggregate artifact failed verification"
        )

    endpoints = _read_endpoint_rows(
        aggregate_source
    )

    if len(endpoints) != counts["n_samples"]:
        raise ControlPlaneError(
            "Endpoint-row count does not match transaction"
        )

    if (
        sum(
            int(row["n_nuclei"])
            for row in endpoints
        )
        != counts["n_nuclei"]
    ):
        raise ControlPlaneError(
            "Endpoint nucleus total does not match transaction"
        )

    transaction_sha = sha256_file(source)
    transaction_size = source.stat().st_size

    identity = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "record_type": PACKAGE_RECORD_TYPE,
        "assay_id": ASSAY_ID,
        "source_transaction_sha256": transaction_sha,
        "source_transaction_size_bytes": transaction_size,
        "transaction_id": ids["transaction_id"],
        "sample_aggregates_sha256": (
            component_hashes["sample_aggregates"]
        ),
        "n_nuclei": counts["n_nuclei"],
        "n_samples": counts["n_samples"],
        "scientific_scope": scope,
    }

    fingerprint = sha256_json(identity)
    package_id = f"rp1-{fingerprint[:16]}"

    root = output_dir.expanduser().resolve()
    package_root = root / package_id
    manifest_path = (
        package_root / "package_manifest.json"
    )

    if manifest_path.exists():
        report = verify_assay_result_package(
            manifest_path
        )

        if report["ok"]:
            return manifest_path

        raise ControlPlaneError(
            "Existing result package failed verification: "
            f"{manifest_path}"
        )

    if package_root.exists():
        raise ControlPlaneError(
            "Content-addressed result-package directory "
            f"already exists without a verified manifest: {package_root}"
        )

    temp_root = root / f".{package_id}.tmp"

    if temp_root.exists():
        raise ControlPlaneError(
            f"Temporary package directory already exists: {temp_root}"
        )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )
    temp_root.mkdir(
        parents=False,
        exist_ok=False,
    )

    try:
        aggregate_copy = (
            temp_root / "sample_aggregates.csv"
        )

        shutil.copyfile(
            aggregate_source,
            aggregate_copy,
        )

        provenance = _build_provenance(
            ids=ids,
            component_hashes=component_hashes,
            transaction=transaction,
            transaction_sha256=transaction_sha,
            transaction_size_bytes=transaction_size,
        )

        result = _build_result(
            package_id=package_id,
            ids=ids,
            counts=counts,
            endpoints=endpoints,
            scope=scope,
            transaction_sha256=transaction_sha,
        )

        provenance_path = (
            temp_root / "provenance.json"
        )
        result_path = temp_root / "result.json"
        report_path = temp_root / "report.md"

        _write_json(
            provenance_path,
            provenance,
        )
        _write_json(
            result_path,
            result,
        )

        with report_path.open(
            "x",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(
                _render_markdown(
                    result,
                    provenance,
                )
            )

        files = {
            "result": _portable_artifact_record(
                temp_root,
                result_path,
            ),
            "provenance": _portable_artifact_record(
                temp_root,
                provenance_path,
            ),
            "sample_aggregates": (
                _portable_artifact_record(
                    temp_root,
                    aggregate_copy,
                )
            ),
            "report": _portable_artifact_record(
                temp_root,
                report_path,
            ),
        }

        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "record_type": PACKAGE_RECORD_TYPE,
            "package_id": package_id,
            "status": "complete",
            "created_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "package_fingerprint_sha256": fingerprint,
            "identity": identity,
            "source_transaction_verified_at_export": True,
            "files": files,
            "interpretation": {
                "role": (
                    "derived portable result/export artifact"
                ),
                "provenance_authority": (
                    "source v0.5 transaction record"
                ),
                "upstream_binary_artifacts_included": False,
            },
        }

        _write_json(
            temp_root / "package_manifest.json",
            manifest,
        )

        temp_root.replace(
            package_root
        )

    except Exception:
        if temp_root.exists():
            shutil.rmtree(
                temp_root,
                ignore_errors=True,
            )
        raise

    return manifest_path


def verify_assay_result_package(
    manifest_path: Path,
) -> dict[str, object]:
    """Verify portable package integrity and deterministic derivation."""

    path = manifest_path.expanduser().resolve()
    manifest = load_json_object(path)

    if (
        manifest.get("schema_version")
        != PACKAGE_SCHEMA_VERSION
        or manifest.get("record_type")
        != PACKAGE_RECORD_TYPE
    ):
        raise ControlPlaneError(
            f"Not a supported assay result package: {path}"
        )

    if path.name != "package_manifest.json":
        raise ControlPlaneError(
            "Result-package manifest filename is invalid"
        )

    package_root = path.parent
    identity = manifest.get("identity")
    files = manifest.get("files")

    if not isinstance(identity, dict):
        raise ControlPlaneError(
            "Result-package identity is malformed"
        )

    if not isinstance(files, dict):
        raise ControlPlaneError(
            "Result-package file bindings are malformed"
        )

    fingerprint = manifest.get(
        "package_fingerprint_sha256"
    )
    package_id = manifest.get("package_id")

    fingerprint_ok = bool(
        isinstance(fingerprint, str)
        and sha256_json(identity) == fingerprint
    )

    package_id_ok = bool(
        isinstance(package_id, str)
        and isinstance(fingerprint, str)
        and package_id
        == f"rp1-{fingerprint[:16]}"
        and package_root.name == package_id
    )

    expected_file_names = {
        "result",
        "provenance",
        "sample_aggregates",
        "report",
    }

    file_keys_ok = (
        set(files) == expected_file_names
    )

    files_ok = bool(
        file_keys_ok
        and all(
            _verify_portable_artifact(
                package_root,
                files[name],
            )
            for name in expected_file_names
        )
    )

    derivation_ok = False
    scope_ok = False
    counts_ok = False

    n_nuclei: object = None
    n_samples: object = None

    if files_ok:
        result_record = _require_mapping(
            files["result"],
            label="result file record",
        )
        provenance_record = _require_mapping(
            files["provenance"],
            label="provenance file record",
        )
        aggregate_record = _require_mapping(
            files["sample_aggregates"],
            label="aggregate file record",
        )
        report_record = _require_mapping(
            files["report"],
            label="report file record",
        )

        result_path = package_root / str(
            result_record["relative_path"]
        )
        provenance_path = package_root / str(
            provenance_record["relative_path"]
        )
        aggregate_path = package_root / str(
            aggregate_record["relative_path"]
        )
        report_path = package_root / str(
            report_record["relative_path"]
        )

        result = load_json_object(
            result_path
        )
        provenance = load_json_object(
            provenance_path
        )
        endpoints = _read_endpoint_rows(
            aggregate_path
        )

        source = _require_mapping(
            provenance.get("source_transaction"),
            label="source transaction provenance",
        )
        component_ids = _require_mapping(
            provenance.get("component_ids"),
            label="component ids",
        )
        component_hashes = _require_mapping(
            provenance.get("component_sha256"),
            label="component hashes",
        )

        result_ids = _require_mapping(
            result.get("ids"),
            label="result ids",
        )
        result_counts = _require_mapping(
            result.get("counts"),
            label="result counts",
        )
        result_scope = _require_mapping(
            result.get("scientific_scope"),
            label="result scientific scope",
        )

        n_nuclei = result_counts.get(
            "n_nuclei"
        )
        n_samples = result_counts.get(
            "n_samples"
        )

        counts_ok = bool(
            isinstance(n_nuclei, int)
            and isinstance(n_samples, int)
            and n_nuclei
            == sum(
                int(row["n_nuclei"])
                for row in endpoints
            )
            and n_samples == len(endpoints)
            and result_counts.get("n_items")
            == result_counts.get(
                "n_prediction_rows"
            )
            == n_nuclei
        )

        scope_ok = bool(
            result_scope
            == identity.get(
                "scientific_scope"
            )
        )

        expected_result = _build_result(
            package_id=str(package_id),
            ids={
                key: str(value)
                for key, value
                in result_ids.items()
            },
            counts={
                key: int(value)
                for key, value
                in result_counts.items()
            },
            endpoints=endpoints,
            scope=dict(result_scope),
            transaction_sha256=str(
                source.get("sha256")
            ),
        )

        expected_report = _render_markdown(
            expected_result,
            provenance,
        )

        aggregate_hash_ok = bool(
            component_hashes.get(
                "sample_aggregates"
            )
            == aggregate_record.get(
                "sha256"
            )
            == identity.get(
                "sample_aggregates_sha256"
            )
        )

        provenance_binding_ok = bool(
            source.get("transaction_id")
            == identity.get(
                "transaction_id"
            )
            == component_ids.get(
                "transaction_id"
            )
            == result_ids.get(
                "transaction_id"
            )
            and source.get("sha256")
            == identity.get(
                "source_transaction_sha256"
            )
            == result.get(
                "source_transaction",
                {},
            ).get("sha256")
            and source.get("size_bytes")
            == identity.get(
                "source_transaction_size_bytes"
            )
        )

        report_text = report_path.read_text(
            encoding="utf-8"
        )

        derivation_ok = bool(
            result == expected_result
            and report_text == expected_report
            and aggregate_hash_ok
            and provenance_binding_ok
            and result.get("assay_id")
            == ASSAY_ID
            and provenance.get("assay_id")
            == ASSAY_ID
            and identity.get("assay_id")
            == ASSAY_ID
            and result.get("verification")
            == "PASS"
        )

    status_ok = (
        manifest.get("status")
        == "complete"
    )

    role_ok = bool(
        manifest.get(
            "source_transaction_verified_at_export"
        )
        is True
        and _require_mapping(
            manifest.get("interpretation"),
            label="package interpretation",
        ).get("provenance_authority")
        == "source v0.5 transaction record"
    )

    ok = bool(
        status_ok
        and fingerprint_ok
        and package_id_ok
        and files_ok
        and derivation_ok
        and scope_ok
        and counts_ok
        and role_ok
    )

    return {
        "kind": PACKAGE_RECORD_TYPE,
        "package_id": package_id,
        "manifest_path": str(path),
        "fingerprint_ok": fingerprint_ok,
        "package_id_ok": package_id_ok,
        "files_ok": files_ok,
        "derivation_ok": derivation_ok,
        "scientific_scope_ok": scope_ok,
        "counts_ok": counts_ok,
        "source_transaction_verified_at_export": (
            manifest.get(
                "source_transaction_verified_at_export"
            )
            is True
        ),
        "n_nuclei": n_nuclei,
        "n_samples": n_samples,
        "ok": ok,
    }
