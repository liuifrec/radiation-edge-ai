"""Physical KL720 DNAi field-measurement transaction provenance.

This v0.11 transaction layer is additive. It binds one validated KL720
application manifest to one verified df2 physical field record.

It performs no neural inference, preprocessing, stitching, fiber
reconstruction, or biological-fidelity evaluation.
"""

from __future__ import annotations

import hashlib
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
from radiation_edge_ai.dna_fiber.field_record_v2 import (
    APPLICATION_ADAPTER,
    ASSAY_ID,
    BACKEND,
    EXPECTED_NEF_SHA256,
    verify_dnai_fiber_physical_field_record,
)
from radiation_edge_ai.dna_fiber.transaction import (
    validate_dnai_application_manifest,
)

SCHEMA_VERSION = 2
RECORD_TYPE = "dnai_fiber_measurement_transaction"
TRANSACTION_ID_PREFIX = "dt2"
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
    "preprocessed_frozen_window_inputs": True,
    "raw_microscopy_preprocessing_performed": False,
    "floating_fp512_inference_performed": False,
    "physical_kl720_inference_performed": True,
    "human_annotations_read": False,
    "fp1024_reference_read": False,
    "biological_fidelity_evaluated": False,
    "kl720_hardware_access_performed": True,
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
        "sha256": sha256_file(
            resolved
        ),
        "size_bytes": (
            resolved.stat().st_size
        ),
    }


def _canonical_lf_bytes(
    path: Path,
) -> bytes:
    try:
        raw = (
            path
            .expanduser()
            .resolve()
            .read_bytes()
        )
    except OSError as exc:
        raise ControlPlaneError(
            f"Could not read source artifact: {path}"
        ) from exc

    return raw.replace(
        b"\r\n",
        b"\n",
    )


def _source_artifact_record(
    path: Path,
) -> dict[str, object]:
    resolved = (
        path
        .expanduser()
        .resolve()
    )

    if not resolved.is_file():
        raise ControlPlaneError(
            f"Source artifact is not readable: {resolved}"
        )

    canonical = (
        _canonical_lf_bytes(
            resolved
        )
    )

    return {
        "path": str(
            resolved
        ),
        "sha256": hashlib.sha256(
            canonical
        ).hexdigest(),
        "size_bytes": len(
            canonical
        ),
    }


def _artifact_metadata_ok(
    value: object,
) -> bool:
    if not isinstance(
        value,
        dict,
    ):
        return False

    path = value.get(
        "path"
    )

    digest = value.get(
        "sha256"
    )

    size = value.get(
        "size_bytes"
    )

    if (
        not isinstance(
            path,
            str,
        )
        or not path.strip()
        or not isinstance(
            digest,
            str,
        )
        or len(
            digest
        )
        != 64
        or not isinstance(
            size,
            int,
        )
        or isinstance(
            size,
            bool,
        )
        or size < 0
    ):
        return False

    try:
        int(
            digest,
            16,
        )
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

    assert isinstance(
        value,
        dict,
    )

    path = Path(
        str(
            value[
                "path"
            ]
        )
    ).expanduser().resolve()

    return bool(
        path.is_file()
        and path.stat().st_size
        == int(
            value[
                "size_bytes"
            ]
        )
        and sha256_file(
            path
        )
        == value[
            "sha256"
        ]
    )


def _source_bytes_ok(
    value: object,
) -> bool:
    if not _artifact_metadata_ok(
        value
    ):
        return False

    assert isinstance(
        value,
        dict,
    )

    path = Path(
        str(
            value[
                "path"
            ]
        )
    ).expanduser().resolve()

    if not path.is_file():
        return False

    try:
        canonical = (
            _canonical_lf_bytes(
                path
            )
        )
    except ControlPlaneError:
        return False

    return bool(
        len(
            canonical
        )
        == int(
            value[
                "size_bytes"
            ]
        )
        and hashlib.sha256(
            canonical
        ).hexdigest()
        == value[
            "sha256"
        ]
    )


def _required_mapping(
    value: object,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(
        value,
        dict,
    ):
        raise ControlPlaneError(
            f"{label} must be a JSON object"
        )

    return value


def _transaction_identity(
    *,
    application_manifest_record: dict[str, object],
    field_record: dict[str, Any],
    field_record_artifact: dict[str, object],
    physical_worker_artifact: dict[str, object],
    window_adapter_artifact: dict[str, object],
) -> dict[str, object]:
    field_identity = (
        _required_mapping(
            field_record.get(
                "identity"
            ),
            label="field identity",
        )
    )

    return {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "record_type": (
            RECORD_TYPE
        ),
        "assay_id": (
            ASSAY_ID
        ),
        "application_adapter": (
            APPLICATION_ADAPTER
        ),
        "backend": (
            BACKEND
        ),
        "input_mode": (
            INPUT_MODE
        ),
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
        "field_id": (
            field_record.get(
                "field_id"
            )
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
        "physical_worker_source_sha256": (
            physical_worker_artifact[
                "sha256"
            ]
        ),
        "window_adapter_source_sha256": (
            window_adapter_artifact[
                "sha256"
            ]
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
            field_identity.get(
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
    counts: dict[str, Any],
    measurements: dict[str, Any],
) -> dict[str, object]:
    return {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "record_type": (
            RECORD_TYPE
        ),
        "assay_id": (
            ASSAY_ID
        ),
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
        "measurements": (
            measurements
        ),
        "scientific_scope": (
            EXPECTED_TRANSACTION_SCOPE
        ),
    }


def create_dnai_fiber_physical_measurement_transaction(
    application_manifest_path: Path,
    field_record_path: Path,
    *,
    output_dir: Path,
) -> Path:
    """Bind one verified df2 field to one dt2 transaction."""

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

    if (
        application_manifest.get(
            "backend"
        )
        != BACKEND
    ):
        raise ControlPlaneError(
            "DNAi physical transaction requires backend kl720"
        )

    field_verification = (
        verify_dnai_fiber_physical_field_record(
            field_path,
            check_artifacts=True,
        )
    )

    if not field_verification.get(
        "ok"
    ):
        raise ControlPlaneError(
            "DNAi physical field record failed verification"
        )

    field_record = (
        load_json_object(
            field_path
        )
    )

    field_identity = (
        _required_mapping(
            field_record.get(
                "identity"
            ),
            label="field identity",
        )
    )

    model = _required_mapping(
        application_manifest.get(
            "model"
        ),
        label="model artifact",
    )

    validation = (
        _required_mapping(
            application_manifest.get(
                "validation_manifest"
            ),
            label="validation artifact",
        )
    )

    metadata = _required_mapping(
        application_manifest.get(
            "metadata"
        ),
        label="metadata",
    )

    if (
        field_identity.get(
            "backend"
        )
        != BACKEND
        or field_identity.get(
            "model_sha256"
        )
        != EXPECTED_NEF_SHA256
        or field_identity.get(
            "model_sha256"
        )
        != model.get(
            "sha256"
        )
        or field_identity.get(
            "validation_manifest_sha256"
        )
        != validation.get(
            "sha256"
        )
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
            "KL720 application manifest and physical field "
            "record are not bound to the same execution"
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

    physical_worker_path = (
        Path(__file__)
        .with_name(
            "_physical_field_worker.py"
        )
        .resolve()
    )

    window_adapter_path = (
        Path(__file__)
        .with_name(
            "kl720_window.py"
        )
        .resolve()
    )

    physical_worker_artifact = (
        _source_artifact_record(
            physical_worker_path
        )
    )

    window_adapter_artifact = (
        _source_artifact_record(
            window_adapter_path
        )
    )

    identity = _transaction_identity(
        application_manifest_record=(
            application_manifest_record
        ),
        field_record=(
            field_record
        ),
        field_record_artifact=(
            field_record_artifact
        ),
        physical_worker_artifact=(
            physical_worker_artifact
        ),
        window_adapter_artifact=(
            window_adapter_artifact
        ),
    )

    fingerprint = sha256_json(
        identity
    )

    transaction_id = (
        f"{TRANSACTION_ID_PREFIX}-"
        f"{fingerprint[:16]}"
    )

    root = (
        output_dir
        .expanduser()
        .resolve()
    )

    transaction_root = (
        root
        / transaction_id
    )

    record_path = (
        transaction_root
        / "transaction_record.json"
    )

    if record_path.exists():
        verification = (
            verify_dnai_fiber_physical_measurement_transaction(
                record_path,
                check_artifacts=True,
            )
        )

        if verification.get(
            "ok"
        ):
            return record_path

        raise ControlPlaneError(
            "Existing DNAi physical transaction "
            "failed verification"
        )

    if (
        transaction_root.exists()
        and any(
            transaction_root.iterdir()
        )
    ):
        raise ControlPlaneError(
            "Refusing non-empty incomplete "
            "DNAi physical transaction directory"
        )

    counts = _required_mapping(
        field_record.get(
            "counts"
        ),
        label="field counts",
    )

    measurements = (
        _required_mapping(
            field_record.get(
                "measurements"
            ),
            label="field measurements",
        )
    )

    record_identity = _record_identity(
        transaction_id=(
            transaction_id
        ),
        transaction_fingerprint=(
            fingerprint
        ),
        transaction_identity=(
            identity
        ),
        counts=counts,
        measurements=(
            measurements
        ),
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
        "record_type": (
            RECORD_TYPE
        ),
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
        "physical_worker_source": (
            physical_worker_artifact
        ),
        "window_adapter_source": (
            window_adapter_artifact
        ),
        "counts": counts,
        "measurements": (
            measurements
        ),
        "scientific_scope": (
            EXPECTED_TRANSACTION_SCOPE
        ),
        "interpretation": (
            "Physical KL720 application-provenance transaction. "
            "This transaction records verified hardware execution "
            "and downstream DNAi field reconstruction, but does not "
            "establish new biological deployment fidelity."
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
        verify_dnai_fiber_physical_measurement_transaction(
            record_path,
            check_artifacts=True,
        )
    )

    if not verification.get(
        "ok"
    ):
        raise ControlPlaneError(
            "New DNAi physical transaction "
            "failed self-verification"
        )

    return record_path


def verify_dnai_fiber_physical_measurement_transaction(
    record_path: Path,
    *,
    check_artifacts: bool = True,
) -> dict[str, object]:
    """Verify one dt2 physical DNAi transaction."""

    resolved = (
        record_path
        .expanduser()
        .resolve()
    )

    value = load_json_object(
        resolved
    )

    if (
        value.get(
            "record_type"
        )
        != RECORD_TYPE
    ):
        raise ControlPlaneError(
            "Not a DNAi measurement transaction: "
            f"{resolved}"
        )

    schema_ok = (
        value.get(
            "schema_version"
        )
        == SCHEMA_VERSION
    )

    status_ok = (
        value.get(
            "status"
        )
        == "complete"
    )

    identity = value.get(
        "transaction_identity"
    )

    if not isinstance(
        identity,
        dict,
    ):
        identity = {}

    identity_semantics_ok = bool(
        identity.get(
            "schema_version"
        )
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
        == EXPECTED_NEF_SHA256
        and isinstance(
            identity.get(
                "field_id"
            ),
            str,
        )
        and str(
            identity.get(
                "field_id"
            )
        ).startswith(
            "df2-"
        )
        and isinstance(
            identity.get(
                "sample_id"
            ),
            str,
        )
    )

    expected_fingerprint = (
        sha256_json(
            identity
        )
        if identity
        else ""
    )

    fingerprint_ok = (
        value.get(
            "transaction_fingerprint_sha256"
        )
        == expected_fingerprint
    )

    expected_transaction_id = (
        f"{TRANSACTION_ID_PREFIX}-"
        f"{expected_fingerprint[:16]}"
        if expected_fingerprint
        else ""
    )

    transaction_id_ok = (
        value.get(
            "transaction_id"
        )
        == expected_transaction_id
    )

    manifest_artifact = (
        value.get(
            "application_manifest"
        )
    )

    field_artifact = (
        value.get(
            "field_record"
        )
    )

    physical_worker_artifact = (
        value.get(
            "physical_worker_source"
        )
    )

    window_adapter_artifact = (
        value.get(
            "window_adapter_source"
        )
    )

    artifacts_metadata_ok = all(
        _artifact_metadata_ok(
            artifact
        )
        for artifact in (
            manifest_artifact,
            field_artifact,
            physical_worker_artifact,
            window_adapter_artifact,
        )
    )

    artifacts_ok = (
        artifacts_metadata_ok
    )

    if (
        check_artifacts
        and artifacts_metadata_ok
    ):
        artifacts_ok = bool(
            _artifact_bytes_ok(
                manifest_artifact
            )
            and _artifact_bytes_ok(
                field_artifact
            )
            and _source_bytes_ok(
                physical_worker_artifact
            )
            and _source_bytes_ok(
                window_adapter_artifact
            )
        )

    identity_bindings_ok = False
    manifest_semantics_ok = False
    field_verification_ok = False

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
            physical_worker_artifact,
            dict,
        )
        assert isinstance(
            window_adapter_artifact,
            dict,
        )

        identity_bindings_ok = bool(
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
            and physical_worker_artifact.get(
                "sha256"
            )
            == identity.get(
                "physical_worker_source_sha256"
            )
            and window_adapter_artifact.get(
                "sha256"
            )
            == identity.get(
                "window_adapter_source_sha256"
            )
        )

    if (
        check_artifacts
        and artifacts_ok
    ):
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

            manifest_semantics_ok = (
                manifest.get(
                    "backend"
                )
                == BACKEND
            )

        except ControlPlaneError:
            manifest = {}
            manifest_semantics_ok = False

        try:
            field_report = (
                verify_dnai_fiber_physical_field_record(
                    field_path,
                    check_artifacts=True,
                )
            )

            field_verification_ok = bool(
                field_report.get(
                    "ok"
                )
            )

        except ControlPlaneError:
            field_verification_ok = False

        if (
            manifest_semantics_ok
            and field_verification_ok
        ):
            field_value = (
                load_json_object(
                    field_path
                )
            )

            field_identity = (
                _required_mapping(
                    field_value.get(
                        "identity"
                    ),
                    label="field identity",
                )
            )

            metadata = (
                _required_mapping(
                    manifest.get(
                        "metadata"
                    ),
                    label="metadata",
                )
            )

            model = _required_mapping(
                manifest.get(
                    "model"
                ),
                label="model",
            )

            validation = (
                _required_mapping(
                    manifest.get(
                        "validation_manifest"
                    ),
                    label="validation",
                )
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
                    "backend"
                )
                == BACKEND
                and field_identity.get(
                    "model_sha256"
                )
                == identity.get(
                    "model_sha256"
                )
                == model.get(
                    "sha256"
                )
                == EXPECTED_NEF_SHA256
                and field_identity.get(
                    "validation_manifest_sha256"
                )
                == identity.get(
                    "validation_manifest_sha256"
                )
                == validation.get(
                    "sha256"
                )
                and field_identity.get(
                    "image_index"
                )
                == identity.get(
                    "image_index"
                )
                == manifest.get(
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

    measurements = (
        value.get(
            "measurements"
        )
    )

    counts_ok = isinstance(
        counts,
        dict,
    )

    measurements_ok = isinstance(
        measurements,
        dict,
    )

    scope_ok = (
        value.get(
            "scientific_scope"
        )
        == EXPECTED_TRANSACTION_SCOPE
    )

    record_identity = value.get(
        "record_identity"
    )

    record_identity_ok = False
    record_fingerprint_ok = False

    if (
        isinstance(
            record_identity,
            dict,
        )
        and counts_ok
        and measurements_ok
        and expected_fingerprint
        and expected_transaction_id
    ):
        assert isinstance(
            counts,
            dict,
        )
        assert isinstance(
            measurements,
            dict,
        )

        expected_record_identity = (
            _record_identity(
                transaction_id=(
                    expected_transaction_id
                ),
                transaction_fingerprint=(
                    expected_fingerprint
                ),
                transaction_identity=(
                    identity
                ),
                counts=counts,
                measurements=(
                    measurements
                ),
            )
        )

        record_identity_ok = (
            record_identity
            == expected_record_identity
        )

        record_fingerprint_ok = (
            value.get(
                "record_fingerprint_sha256"
            )
            == sha256_json(
                expected_record_identity
            )
        )

    created_utc_ok = False

    created_utc = value.get(
        "created_utc"
    )

    if isinstance(
        created_utc,
        str,
    ):
        try:
            created = (
                datetime.fromisoformat(
                    created_utc
                )
            )
        except ValueError:
            pass
        else:
            created_utc_ok = (
                created.tzinfo
                is not None
            )

    ok = bool(
        schema_ok
        and status_ok
        and identity_semantics_ok
        and fingerprint_ok
        and transaction_id_ok
        and artifacts_metadata_ok
        and artifacts_ok
        and identity_bindings_ok
        and manifest_semantics_ok
        and field_verification_ok
        and counts_ok
        and measurements_ok
        and scope_ok
        and record_identity_ok
        and record_fingerprint_ok
        and created_utc_ok
    )

    return {
        "kind": RECORD_TYPE,
        "transaction_id": (
            value.get(
                "transaction_id"
            )
        ),
        "schema_ok": (
            schema_ok
        ),
        "status_ok": (
            status_ok
        ),
        "identity_semantics_ok": (
            identity_semantics_ok
        ),
        "fingerprint_ok": (
            fingerprint_ok
        ),
        "transaction_id_ok": (
            transaction_id_ok
        ),
        "artifacts_metadata_ok": (
            artifacts_metadata_ok
        ),
        "artifacts_ok": (
            artifacts_ok
        ),
        "identity_bindings_ok": (
            identity_bindings_ok
        ),
        "manifest_semantics_ok": (
            manifest_semantics_ok
        ),
        "field_verification_ok": (
            field_verification_ok
        ),
        "scope_ok": (
            scope_ok
        ),
        "record_identity_ok": (
            record_identity_ok
        ),
        "record_fingerprint_ok": (
            record_fingerprint_ok
        ),
        "created_utc_ok": (
            created_utc_ok
        ),
        "ok": ok,
    }
