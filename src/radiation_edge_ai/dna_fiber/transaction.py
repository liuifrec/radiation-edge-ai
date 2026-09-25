"""DNAi field-measurement transaction provenance.

This layer binds an application manifest to one verified DNAi field record.
It performs no neural inference, image preprocessing, stitching, or fiber
reconstruction.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from radiation_edge_ai.control import (
    ControlPlaneError,
    load_json_object,
    sha256_file,
    sha256_json,
)
from radiation_edge_ai.dna_fiber.field_record import (
    DNAI_COMMIT,
    EXPECTED_MODEL_SHA256,
    verify_dnai_fiber_field_record,
)

SCHEMA_VERSION = 1
RECORD_TYPE = "dnai_fiber_measurement_transaction"
ASSAY_ID = "dnai-fiber-v3"
APPLICATION_ADAPTER = "dnai-tile-stitch-object-v1"
BACKEND = "cpu-onnx"
INPUT_MODE = "frozen-deployment-windows"


EXPECTED_TRANSACTION_SCOPE = {
    "field_record_verified": True,
    "prediction_semantics": (
        "tile-level segmentation reconstructed into a "
        "stitched microscopy-field segmentation"
    ),
    "endpoint_semantics": (
        "post-processed DNA-fiber objects and tract measurements"
    ),
    "raw_microscopy_preprocessing_performed": False,
    "human_annotations_read": False,
    "fp1024_reference_read": False,
    "biological_fidelity_evaluated": False,
    "kl720_hardware_access_performed": False,
    "result_packaging_performed": False,
}


def _artifact_record(
    path: Path,
) -> dict[str, object]:
    resolved = (
        path
        .expanduser()
        .resolve()
    )

    if not resolved.is_file():
        raise ControlPlaneError(
            f"Artifact is not a readable file: {resolved}"
        )

    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _artifact_metadata_ok(
    value: object,
) -> bool:
    if not isinstance(value, dict):
        return False

    path = value.get("path")
    digest = value.get("sha256")
    size = value.get("size_bytes")

    if (
        not isinstance(path, str)
        or not path.strip()
        or not isinstance(digest, str)
        or len(digest) != 64
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
    ):
        return False

    try:
        int(digest, 16)
    except ValueError:
        return False

    return True


def _artifact_bytes_ok(
    value: object,
) -> bool:
    if not _artifact_metadata_ok(
        value
    ):
        return False

    assert isinstance(value, dict)

    path = Path(
        str(value["path"])
    ).expanduser().resolve()

    return bool(
        path.is_file()
        and path.stat().st_size
        == int(value["size_bytes"])
        and sha256_file(path)
        == value["sha256"]
    )


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


def validate_dnai_application_manifest(
    manifest_path: Path,
) -> dict[str, Any]:
    resolved = (
        manifest_path
        .expanduser()
        .resolve()
    )

    manifest = load_json_object(
        resolved
    )

    if (
        manifest.get("schema_version")
        != SCHEMA_VERSION
    ):
        raise ControlPlaneError(
            "Unsupported DNAi application-manifest schema"
        )

    if (
        manifest.get("assay_id")
        != ASSAY_ID
    ):
        raise ControlPlaneError(
            "DNAi application manifest has wrong assay_id"
        )

    if (
        manifest.get(
            "application_adapter"
        )
        != APPLICATION_ADAPTER
    ):
        raise ControlPlaneError(
            "DNAi application manifest has wrong adapter"
        )

    if manifest.get("backend") != BACKEND:
        raise ControlPlaneError(
            "DNAi v0.9 application supports only cpu-onnx"
        )

    if (
        manifest.get("input_mode")
        != INPUT_MODE
    ):
        raise ControlPlaneError(
            "DNAi v0.9 supports only frozen deployment windows"
        )

    image_index = manifest.get(
        "image_index"
    )

    if (
        not isinstance(image_index, int)
        or isinstance(image_index, bool)
        or image_index < 0
    ):
        raise ControlPlaneError(
            "DNAi image_index is malformed"
        )

    model = _required_mapping(
        manifest.get("model"),
        label="model artifact",
    )

    validation = _required_mapping(
        manifest.get(
            "validation_manifest"
        ),
        label="validation-manifest artifact",
    )

    if not _artifact_bytes_ok(model):
        raise ControlPlaneError(
            "DNAi application model artifact failed verification"
        )

    if not _artifact_bytes_ok(
        validation
    ):
        raise ControlPlaneError(
            "DNAi validation-manifest artifact failed verification"
        )

    if (
        model.get("sha256")
        != EXPECTED_MODEL_SHA256
    ):
        raise ControlPlaneError(
            "DNAi application model is not the frozen FP512 ONNX"
        )

    validation_path = Path(
        str(validation["path"])
    ).expanduser().resolve()

    validation_value = load_json_object(
        validation_path
    )

    if (
        validation_value.get(
            "dnai_commit"
        )
        != DNAI_COMMIT
    ):
        raise ControlPlaneError(
            "DNAi validation manifest/code revision mismatch"
        )

    if (
        validation_value.get("tile")
        != [1, 3, 512, 512]
        or validation_value.get(
            "full_image_shape"
        )
        != [1, 3, 1024, 1024]
        or validation_value.get(
            "overlap"
        )
        != 0.50
        or validation_value.get(
            "stride"
        )
        != 256
        or validation_value.get(
            "blend"
        )
        != "gaussian"
        or validation_value.get(
            "windows_per_image"
        )
        != 9
    ):
        raise ControlPlaneError(
            "DNAi validation deployment policy changed"
        )

    records = validation_value.get(
        "records"
    )

    if not isinstance(records, list):
        raise ControlPlaneError(
            "DNAi validation records are malformed"
        )

    selected = [
        record
        for record in records
        if (
            isinstance(record, dict)
            and record.get(
                "image_index"
            )
            == image_index
        )
    ]

    if len(selected) != 9:
        raise ControlPlaneError(
            "Selected DNAi field does not have exactly 9 windows"
        )

    sample_ids = {
        str(record.get("sample_id", ""))
        for record in selected
    }

    if (
        len(sample_ids) != 1
        or not next(iter(sample_ids))
    ):
        raise ControlPlaneError(
            "Selected DNAi field has inconsistent sample_id"
        )

    selected_sample = next(
        iter(sample_ids)
    )

    metadata = _required_mapping(
        manifest.get("metadata"),
        label="DNAi application metadata",
    )

    sample_id = _required_text(
        metadata.get("sample_id"),
        label="metadata sample_id",
    )

    if sample_id != selected_sample:
        raise ControlPlaneError(
            "Application metadata sample_id does not match "
            "validation-manifest field"
        )

    if metadata.get(
        "pixel_size_um"
    ) != 0.26:
        raise ControlPlaneError(
            "DNAi pixel_size_um must remain 0.26"
        )

    _required_text(
        metadata.get(
            "channel_semantics"
        ),
        label="channel_semantics",
    )

    return manifest


def _transaction_identity(
    *,
    application_manifest: dict[str, object],
    application_manifest_record: dict[str, object],
    field_record: dict[str, Any],
    field_record_artifact: dict[str, object],
    worker_artifact: dict[str, object],
) -> dict[str, object]:
    field_identity = _required_mapping(
        field_record.get("identity"),
        label="field identity",
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": RECORD_TYPE,
        "assay_id": ASSAY_ID,
        "application_adapter": (
            APPLICATION_ADAPTER
        ),
        "backend": BACKEND,
        "input_mode": INPUT_MODE,
        "application_manifest_sha256": (
            application_manifest_record[
                "sha256"
            ]
        ),
        "application_manifest_size_bytes": (
            application_manifest_record[
                "size_bytes"
            ]
        ),
        "field_id": field_record.get(
            "field_id"
        ),
        "field_record_sha256": (
            field_record_artifact[
                "sha256"
            ]
        ),
        "field_record_size_bytes": (
            field_record_artifact[
                "size_bytes"
            ]
        ),
        "worker_source_sha256": (
            worker_artifact["sha256"]
        ),
        "model_sha256": (
            field_identity.get(
                "model_sha256"
            )
        ),
        "validation_manifest_sha256": (
            field_identity.get(
                "validation_manifest_sha256"
            )
        ),
        "image_index": (
            application_manifest.get(
                "image_index"
            )
        ),
        "sample_id": (
            field_identity.get(
                "sample_id"
            )
        ),
    }


def _record_identity(
    *,
    transaction_id: str,
    transaction_fingerprint: str,
    transaction_identity: dict[str, object],
    counts: dict[str, object],
    measurements: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": RECORD_TYPE,
        "assay_id": ASSAY_ID,
        "transaction_id": (
            transaction_id
        ),
        "transaction_fingerprint_sha256": (
            transaction_fingerprint
        ),
        "field_id": (
            transaction_identity[
                "field_id"
            ]
        ),
        "field_record_sha256": (
            transaction_identity[
                "field_record_sha256"
            ]
        ),
        "counts": counts,
        "measurements": measurements,
        "scientific_scope": (
            EXPECTED_TRANSACTION_SCOPE
        ),
    }


def create_dnai_fiber_measurement_transaction(
    application_manifest_path: Path,
    field_record_path: Path,
    *,
    output_dir: Path,
) -> Path:
    """Wrap one verified DNAi field record in a transaction."""

    manifest_path = (
        application_manifest_path
        .expanduser()
        .resolve()
    )

    field_path = (
        field_record_path
        .expanduser()
        .resolve()
    )

    application_manifest = (
        validate_dnai_application_manifest(
            manifest_path
        )
    )

    field_verification = (
        verify_dnai_fiber_field_record(
            field_path,
            check_artifacts=True,
        )
    )

    if not field_verification[
        "ok"
    ]:
        raise ControlPlaneError(
            "DNAi field record failed verification"
        )

    field_record = load_json_object(
        field_path
    )

    field_identity = _required_mapping(
        field_record.get("identity"),
        label="field identity",
    )

    model = _required_mapping(
        application_manifest.get(
            "model"
        ),
        label="model artifact",
    )

    validation = _required_mapping(
        application_manifest.get(
            "validation_manifest"
        ),
        label="validation artifact",
    )

    metadata = _required_mapping(
        application_manifest.get(
            "metadata"
        ),
        label="metadata",
    )

    if (
        field_identity.get(
            "model_sha256"
        )
        != model.get("sha256")
        or field_identity.get(
            "validation_manifest_sha256"
        )
        != validation.get("sha256")
        or field_identity.get(
            "image_index"
        )
        != application_manifest.get(
            "image_index"
        )
        or field_identity.get(
            "sample_id"
        )
        != metadata.get(
            "sample_id"
        )
    ):
        raise ControlPlaneError(
            "Application manifest and DNAi field record are not bound "
            "to the same field/model/input manifest"
        )

    application_manifest_record = (
        _artifact_record(
            manifest_path
        )
    )

    field_record_artifact = (
        _artifact_record(
            field_path
        )
    )

    worker_path = (
        Path(__file__)
        .with_name(
            "_field_worker.py"
        )
        .resolve()
    )

    worker_artifact = (
        _artifact_record(
            worker_path
        )
    )

    identity = _transaction_identity(
        application_manifest=(
            application_manifest
        ),
        application_manifest_record=(
            application_manifest_record
        ),
        field_record=field_record,
        field_record_artifact=(
            field_record_artifact
        ),
        worker_artifact=(
            worker_artifact
        ),
    )

    fingerprint = sha256_json(
        identity
    )

    transaction_id = (
        f"dt1-{fingerprint[:16]}"
    )

    root = (
        output_dir
        .expanduser()
        .resolve()
    )

    transaction_root = (
        root / transaction_id
    )

    record_path = (
        transaction_root
        / "transaction_record.json"
    )

    if record_path.exists():
        verification = (
            verify_dnai_fiber_measurement_transaction(
                record_path,
                check_artifacts=True,
            )
        )

        if verification["ok"]:
            return record_path

        raise ControlPlaneError(
            "Existing DNAi transaction failed verification"
        )

    if (
        transaction_root.exists()
        and any(
            transaction_root.iterdir()
        )
    ):
        raise ControlPlaneError(
            "Refusing non-empty incomplete DNAi transaction directory"
        )

    counts = _required_mapping(
        field_record.get("counts"),
        label="field counts",
    )

    measurements = _required_mapping(
        field_record.get(
            "measurements"
        ),
        label="field measurements",
    )

    record_identity = _record_identity(
        transaction_id=transaction_id,
        transaction_fingerprint=(
            fingerprint
        ),
        transaction_identity=identity,
        counts=counts,
        measurements=measurements,
    )

    record_fingerprint = (
        sha256_json(
            record_identity
        )
    )

    transaction = {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "record_type": RECORD_TYPE,
        "status": "complete",
        "transaction_id": (
            transaction_id
        ),
        "transaction_fingerprint_sha256": (
            fingerprint
        ),
        "record_fingerprint_sha256": (
            record_fingerprint
        ),
        "created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "transaction_identity": (
            identity
        ),
        "record_identity": (
            record_identity
        ),
        "application_manifest": (
            application_manifest_record
        ),
        "field_record": (
            field_record_artifact
        ),
        "worker_source": (
            worker_artifact
        ),
        "counts": counts,
        "measurements": (
            measurements
        ),
        "scientific_scope": (
            EXPECTED_TRANSACTION_SCOPE
        ),
        "interpretation": (
            "Application-provenance transaction only. "
            "No new biological-fidelity, deployment-equivalence, "
            "or KL720 hardware result."
        ),
    }

    transaction_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    with record_path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            transaction,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")

    verification = (
        verify_dnai_fiber_measurement_transaction(
            record_path,
            check_artifacts=True,
        )
    )

    if not verification["ok"]:
        raise ControlPlaneError(
            "New DNAi transaction failed self-verification"
        )

    return record_path


def verify_dnai_fiber_measurement_transaction(
    record_path: Path,
    *,
    check_artifacts: bool = True,
) -> dict[str, object]:
    """Verify a DNAi field-measurement transaction."""

    resolved = (
        record_path
        .expanduser()
        .resolve()
    )

    value = load_json_object(
        resolved
    )

    if (
        value.get("record_type")
        != RECORD_TYPE
    ):
        raise ControlPlaneError(
            f"Not a DNAi measurement transaction: {resolved}"
        )

    identity = value.get(
        "transaction_identity"
    )

    if not isinstance(identity, dict):
        identity = {}

    identity_semantics_ok = (
        identity.get("schema_version")
        == SCHEMA_VERSION
        and identity.get(
            "record_type"
        )
        == RECORD_TYPE
        and identity.get(
            "assay_id"
        )
        == ASSAY_ID
        and identity.get(
            "application_adapter"
        )
        == APPLICATION_ADAPTER
        and identity.get(
            "backend"
        )
        == BACKEND
        and identity.get(
            "input_mode"
        )
        == INPUT_MODE
        and identity.get(
            "model_sha256"
        )
        == EXPECTED_MODEL_SHA256
        and isinstance(
            identity.get(
                "field_id"
            ),
            str,
        )
        and isinstance(
            identity.get(
                "sample_id"
            ),
            str,
        )
    )

    expected_transaction_fingerprint = (
        sha256_json(identity)
        if identity
        else ""
    )

    transaction_fingerprint_ok = (
        value.get(
            "transaction_fingerprint_sha256"
        )
        == expected_transaction_fingerprint
    )

    expected_transaction_id = (
        f"dt1-{expected_transaction_fingerprint[:16]}"
        if expected_transaction_fingerprint
        else ""
    )

    transaction_id_ok = (
        value.get(
            "transaction_id"
        )
        == expected_transaction_id
    )

    manifest_artifact = value.get(
        "application_manifest"
    )

    field_artifact = value.get(
        "field_record"
    )

    worker_artifact = value.get(
        "worker_source"
    )

    artifacts_metadata_ok = all(
        _artifact_metadata_ok(
            artifact
        )
        for artifact in (
            manifest_artifact,
            field_artifact,
            worker_artifact,
        )
    )

    artifacts_ok = (
        artifacts_metadata_ok
    )

    if check_artifacts:
        artifacts_ok = all(
            _artifact_bytes_ok(
                artifact
            )
            for artifact in (
                manifest_artifact,
                field_artifact,
                worker_artifact,
            )
        )

    identity_bindings_ok = False
    field_verification_ok = False
    manifest_semantics_ok = False

    if artifacts_metadata_ok:
        assert isinstance(
            manifest_artifact,
            dict,
        )
        assert isinstance(
            field_artifact,
            dict,
        )
        assert isinstance(
            worker_artifact,
            dict,
        )

        identity_bindings_ok = (
            manifest_artifact.get(
                "sha256"
            )
            == identity.get(
                "application_manifest_sha256"
            )
            and manifest_artifact.get(
                "size_bytes"
            )
            == identity.get(
                "application_manifest_size_bytes"
            )
            and field_artifact.get(
                "sha256"
            )
            == identity.get(
                "field_record_sha256"
            )
            and field_artifact.get(
                "size_bytes"
            )
            == identity.get(
                "field_record_size_bytes"
            )
            and worker_artifact.get(
                "sha256"
            )
            == identity.get(
                "worker_source_sha256"
            )
        )

    if check_artifacts and artifacts_ok:
        assert isinstance(
            manifest_artifact,
            dict,
        )
        assert isinstance(
            field_artifact,
            dict,
        )

        manifest_path = Path(
            str(
                manifest_artifact[
                    "path"
                ]
            )
        )

        field_path = Path(
            str(
                field_artifact[
                    "path"
                ]
            )
        )

        try:
            manifest = (
                validate_dnai_application_manifest(
                    manifest_path
                )
            )
            manifest_semantics_ok = True
        except ControlPlaneError:
            manifest = {}
            manifest_semantics_ok = False

        field_report = (
            verify_dnai_fiber_field_record(
                field_path,
                check_artifacts=True,
            )
        )

        field_verification_ok = bool(
            field_report["ok"]
        )

        if (
            manifest_semantics_ok
            and field_verification_ok
        ):
            field_value = load_json_object(
                field_path
            )

            field_identity = (
                _required_mapping(
                    field_value.get(
                        "identity"
                    ),
                    label="field identity",
                )
            )

            metadata = _required_mapping(
                manifest.get(
                    "metadata"
                ),
                label="metadata",
            )

            identity_bindings_ok = bool(
                identity_bindings_ok
                and field_value.get(
                    "field_id"
                )
                == identity.get(
                    "field_id"
                )
                and field_identity.get(
                    "model_sha256"
                )
                == identity.get(
                    "model_sha256"
                )
                and field_identity.get(
                    "validation_manifest_sha256"
                )
                == identity.get(
                    "validation_manifest_sha256"
                )
                and field_identity.get(
                    "image_index"
                )
                == identity.get(
                    "image_index"
                )
                and field_identity.get(
                    "sample_id"
                )
                == identity.get(
                    "sample_id"
                )
                == metadata.get(
                    "sample_id"
                )
            )

    counts = value.get(
        "counts"
    )

    measurements = value.get(
        "measurements"
    )

    summary_semantics_ok = bool(
        isinstance(counts, dict)
        and counts.get(
            "n_windows"
        )
        == 9
        and isinstance(
            counts.get(
                "n_fibers_valid"
            ),
            int,
        )
        and isinstance(
            measurements,
            dict,
        )
        and value.get(
            "scientific_scope"
        )
        == EXPECTED_TRANSACTION_SCOPE
    )

    record_identity = value.get(
        "record_identity"
    )

    record_identity_expected = (
        _record_identity(
            transaction_id=(
                expected_transaction_id
            ),
            transaction_fingerprint=(
                expected_transaction_fingerprint
            ),
            transaction_identity=identity,
            counts=counts
            if isinstance(
                counts,
                dict,
            )
            else {},
            measurements=measurements
            if isinstance(
                measurements,
                dict,
            )
            else {},
        )
    )

    record_identity_ok = (
        record_identity
        == record_identity_expected
    )

    expected_record_fingerprint = (
        sha256_json(
            record_identity_expected
        )
    )

    record_fingerprint_ok = (
        value.get(
            "record_fingerprint_sha256"
        )
        == expected_record_fingerprint
    )

    status_ok = (
        value.get("status")
        == "complete"
    )

    schema_ok = (
        value.get(
            "schema_version"
        )
        == SCHEMA_VERSION
    )

    ok = all(
        (
            schema_ok,
            status_ok,
            identity_semantics_ok,
            transaction_fingerprint_ok,
            transaction_id_ok,
            artifacts_ok,
            identity_bindings_ok,
            summary_semantics_ok,
            record_identity_ok,
            record_fingerprint_ok,
        )
    )

    if check_artifacts:
        ok = bool(
            ok
            and manifest_semantics_ok
            and field_verification_ok
        )

    return {
        "kind": RECORD_TYPE,
        "transaction_id": (
            value.get(
                "transaction_id"
            )
        ),
        "field_id": (
            identity.get(
                "field_id"
            )
        ),
        "schema_ok": schema_ok,
        "status_ok": status_ok,
        "identity_semantics_ok": (
            identity_semantics_ok
        ),
        "transaction_fingerprint_ok": (
            transaction_fingerprint_ok
        ),
        "transaction_id_ok": (
            transaction_id_ok
        ),
        "record_identity_ok": (
            record_identity_ok
        ),
        "record_fingerprint_ok": (
            record_fingerprint_ok
        ),
        "artifact_check_performed": (
            check_artifacts
        ),
        "artifacts_ok": artifacts_ok,
        "manifest_semantics_ok": (
            manifest_semantics_ok
        ),
        "field_verification_ok": (
            field_verification_ok
        ),
        "identity_bindings_ok": (
            identity_bindings_ok
        ),
        "summary_semantics_ok": (
            summary_semantics_ok
        ),
        "n_windows": (
            counts.get("n_windows")
            if isinstance(
                counts,
                dict,
            )
            else None
        ),
        "n_fibers_valid": (
            counts.get(
                "n_fibers_valid"
            )
            if isinstance(
                counts,
                dict,
            )
            else None
        ),
        "ok": ok,
    }
