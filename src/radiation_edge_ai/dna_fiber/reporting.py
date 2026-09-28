"""Portable DNAi assay-result export.

This module derives a compact, path-independent result package from a fully
verified ``dnai_fiber_measurement_transaction``.

Packaging performs no neural inference, image preprocessing, stitching,
segmentation, fiber reconstruction, biological-reference lookup, or
biological-fidelity evaluation. The source transaction remains the provenance
authority.
"""

from __future__ import annotations

import csv
import json
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
from radiation_edge_ai.dna_fiber.transaction import (
    EXPECTED_TRANSACTION_SCOPE,
    verify_dnai_fiber_measurement_transaction,
)

PACKAGE_SCHEMA_VERSION = 1
PACKAGE_RECORD_TYPE = "assay_result_package"
PACKAGE_ADAPTER = "dnai-fiber-measurement-package-v1"
ASSAY_ID = "dnai-fiber-v3"
PACKAGE_ID_PREFIX = "drp1"


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


def _is_sha256(
    value: object,
) -> bool:
    if (
        not isinstance(value, str)
        or len(value) != 64
    ):
        return False

    try:
        int(value, 16)
    except ValueError:
        return False

    return True


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
            "DNAi result-package artifact escaped package root"
        ) from exc

    return {
        "relative_path": relative.as_posix(),
        "sha256": sha256_file(
            resolved_path
        ),
        "size_bytes": (
            resolved_path.stat().st_size
        ),
    }


def _verify_portable_artifact(
    package_root: Path,
    value: object,
) -> bool:
    if not isinstance(value, Mapping):
        return False

    relative_value = value.get(
        "relative_path"
    )
    expected_sha = value.get(
        "sha256"
    )
    expected_size = value.get(
        "size_bytes"
    )

    if (
        not isinstance(
            relative_value,
            str,
        )
        or not _is_sha256(
            expected_sha
        )
        or not isinstance(
            expected_size,
            int,
        )
        or isinstance(
            expected_size,
            bool,
        )
        or expected_size < 0
    ):
        return False

    relative = Path(
        relative_value
    )

    if relative.is_absolute():
        return False

    candidate = (
        package_root
        / relative
    ).resolve()

    try:
        candidate.relative_to(
            package_root.resolve()
        )
    except ValueError:
        return False

    return bool(
        candidate.is_file()
        and candidate.stat().st_size
        == expected_size
        and sha256_file(candidate)
        == expected_sha
    )


def _verified_source_artifact(
    value: object,
    *,
    label: str,
) -> Path:
    artifact = _require_mapping(
        value,
        label=label,
    )

    path_value = _required_text(
        artifact.get("path"),
        label=f"{label} path",
    )

    expected_sha = artifact.get(
        "sha256"
    )
    expected_size = artifact.get(
        "size_bytes"
    )

    if (
        not _is_sha256(
            expected_sha
        )
        or not isinstance(
            expected_size,
            int,
        )
        or isinstance(
            expected_size,
            bool,
        )
        or expected_size < 0
    ):
        raise ControlPlaneError(
            f"{label} metadata is malformed"
        )

    path = Path(
        path_value
    ).expanduser().resolve()

    if (
        not path.is_file()
        or path.stat().st_size
        != expected_size
        or sha256_file(path)
        != expected_sha
    ):
        raise ControlPlaneError(
            f"{label} failed byte verification"
        )

    return path


def _csv_row_count(
    path: Path,
) -> int:
    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(
                handle
            )

            if not reader.fieldnames:
                raise ControlPlaneError(
                    "DNAi valid-fibers table has no header"
                )

            return sum(
                1
                for _row in reader
            )
    except (
        OSError,
        UnicodeError,
        csv.Error,
    ) as exc:
        raise ControlPlaneError(
            f"Could not read DNAi valid-fibers table: {exc}"
        ) from exc


def _source_components(
    transaction: Mapping[str, Any],
) -> dict[str, object]:
    transaction_identity = (
        _require_mapping(
            transaction.get(
                "transaction_identity"
            ),
            label=(
                "DNAi transaction identity"
            ),
        )
    )

    counts = _require_mapping(
        transaction.get("counts"),
        label="DNAi transaction counts",
    )

    measurements = _require_mapping(
        transaction.get(
            "measurements"
        ),
        label=(
            "DNAi transaction measurements"
        ),
    )

    scope = _require_mapping(
        transaction.get(
            "scientific_scope"
        ),
        label=(
            "DNAi transaction scientific scope"
        ),
    )

    if dict(scope) != (
        EXPECTED_TRANSACTION_SCOPE
    ):
        raise ControlPlaneError(
            "DNAi transaction scientific scope changed"
        )

    field_path = (
        _verified_source_artifact(
            transaction.get(
                "field_record"
            ),
            label=(
                "DNAi source field record"
            ),
        )
    )

    field = load_json_object(
        field_path
    )

    field_identity = (
        _require_mapping(
            field.get("identity"),
            label="DNAi field identity",
        )
    )

    field_counts = _require_mapping(
        field.get("counts"),
        label="DNAi field counts",
    )

    field_measurements = (
        _require_mapping(
            field.get(
                "measurements"
            ),
            label=(
                "DNAi field measurements"
            ),
        )
    )

    if (
        dict(field_counts)
        != dict(counts)
        or dict(
            field_measurements
        )
        != dict(
            measurements
        )
    ):
        raise ControlPlaneError(
            "DNAi transaction summary no longer matches field record"
        )

    artifacts = _require_mapping(
        field.get("artifacts"),
        label=(
            "DNAi field derived artifacts"
        ),
    )

    fibers_path = (
        _verified_source_artifact(
            artifacts.get(
                "valid_fibers_csv"
            ),
            label=(
                "DNAi valid-fibers table"
            ),
        )
    )

    field_id = _required_text(
        field.get("field_id"),
        label="DNAi field_id",
    )

    if (
        field_id
        != transaction_identity.get(
            "field_id"
        )
    ):
        raise ControlPlaneError(
            "DNAi field identity does not match transaction"
        )

    n_valid = counts.get(
        "n_fibers_valid"
    )

    if (
        not isinstance(
            n_valid,
            int,
        )
        or isinstance(
            n_valid,
            bool,
        )
        or n_valid < 0
    ):
        raise ControlPlaneError(
            "DNAi valid-fiber count is malformed"
        )

    row_count = _csv_row_count(
        fibers_path
    )

    if row_count != n_valid:
        raise ControlPlaneError(
            "DNAi valid-fibers CSV row count does not match transaction"
        )

    return {
        "transaction_identity": (
            dict(
                transaction_identity
            )
        ),
        "field_path": field_path,
        "field": field,
        "field_identity": (
            dict(
                field_identity
            )
        ),
        "fibers_path": fibers_path,
        "counts": dict(counts),
        "measurements": (
            dict(
                measurements
            )
        ),
        "scientific_scope": (
            dict(scope)
        ),
    }


def _package_identity(
    *,
    transaction: Mapping[str, Any],
    source_transaction_sha256: str,
    source_transaction_size_bytes: int,
    source: Mapping[str, Any],
) -> dict[str, object]:
    transaction_identity = (
        _require_mapping(
            source.get(
                "transaction_identity"
            ),
            label=(
                "source transaction identity"
            ),
        )
    )

    field_identity = (
        _require_mapping(
            source.get(
                "field_identity"
            ),
            label=(
                "source field identity"
            ),
        )
    )

    counts = _require_mapping(
        source.get("counts"),
        label="source counts",
    )

    measurements = (
        _require_mapping(
            source.get(
                "measurements"
            ),
            label=(
                "source measurements"
            ),
        )
    )

    scope = _require_mapping(
        source.get(
            "scientific_scope"
        ),
        label=(
            "source scientific scope"
        ),
    )

    fibers_path = source.get(
        "fibers_path"
    )

    if not isinstance(
        fibers_path,
        Path,
    ):
        raise ControlPlaneError(
            "Source valid-fibers path is malformed"
        )

    return {
        "schema_version": (
            PACKAGE_SCHEMA_VERSION
        ),
        "record_type": (
            PACKAGE_RECORD_TYPE
        ),
        "assay_id": ASSAY_ID,
        "package_adapter": (
            PACKAGE_ADAPTER
        ),
        "source_transaction_sha256": (
            source_transaction_sha256
        ),
        "source_transaction_size_bytes": (
            source_transaction_size_bytes
        ),
        "transaction_id": (
            transaction.get(
                "transaction_id"
            )
        ),
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
        "field_id": (
            transaction_identity.get(
                "field_id"
            )
        ),
        "field_record_sha256": (
            transaction_identity.get(
                "field_record_sha256"
            )
        ),
        "valid_fibers_csv_sha256": (
            sha256_file(
                fibers_path
            )
        ),
        "valid_fibers_csv_size_bytes": (
            fibers_path.stat().st_size
        ),
        "sample_id": (
            transaction_identity.get(
                "sample_id"
            )
        ),
        "image_index": (
            transaction_identity.get(
                "image_index"
            )
        ),
        "model_sha256": (
            transaction_identity.get(
                "model_sha256"
            )
        ),
        "validation_manifest_sha256": (
            transaction_identity.get(
                "validation_manifest_sha256"
            )
        ),
        "counts": dict(counts),
        "measurements": (
            dict(
                measurements
            )
        ),
        "source_scientific_scope": (
            dict(scope)
        ),
        "prediction_semantics": (
            "tile-level segmentation reconstructed into a "
            "stitched microscopy-field segmentation"
        ),
        "endpoint_semantics": (
            "post-processed DNA-fiber objects and tract measurements"
        ),
        "field_input_semantics": (
            field_identity.get(
                "input_semantics"
            )
        ),
    }


def _identity_semantics_ok(
    identity: Mapping[str, Any],
) -> bool:
    counts = identity.get(
        "counts"
    )
    measurements = identity.get(
        "measurements"
    )

    if (
        identity.get(
            "schema_version"
        )
        != PACKAGE_SCHEMA_VERSION
        or identity.get(
            "record_type"
        )
        != PACKAGE_RECORD_TYPE
        or identity.get(
            "assay_id"
        )
        != ASSAY_ID
        or identity.get(
            "package_adapter"
        )
        != PACKAGE_ADAPTER
        or not _is_sha256(
            identity.get(
                "source_transaction_sha256"
            )
        )
        or not _is_sha256(
            identity.get(
                "transaction_fingerprint_sha256"
            )
        )
        or not _is_sha256(
            identity.get(
                "record_fingerprint_sha256"
            )
        )
        or not _is_sha256(
            identity.get(
                "field_record_sha256"
            )
        )
        or not _is_sha256(
            identity.get(
                "valid_fibers_csv_sha256"
            )
        )
        or not _is_sha256(
            identity.get(
                "model_sha256"
            )
        )
        or not _is_sha256(
            identity.get(
                "validation_manifest_sha256"
            )
        )
        or not isinstance(
            identity.get(
                "transaction_id"
            ),
            str,
        )
        or not isinstance(
            identity.get(
                "field_id"
            ),
            str,
        )
        or not isinstance(
            identity.get(
                "sample_id"
            ),
            str,
        )
        or not isinstance(
            identity.get(
                "image_index"
            ),
            int,
        )
        or isinstance(
            identity.get(
                "image_index"
            ),
            bool,
        )
        or not isinstance(
            counts,
            Mapping,
        )
        or not isinstance(
            measurements,
            Mapping,
        )
        or identity.get(
            "source_scientific_scope"
        )
        != EXPECTED_TRANSACTION_SCOPE
    ):
        return False

    return (
        counts.get(
            "n_windows"
        )
        == 9
        and isinstance(
            counts.get(
                "n_fibers_all"
            ),
            int,
        )
        and isinstance(
            counts.get(
                "n_fibers_valid"
            ),
            int,
        )
        and counts.get(
            "n_fibers_valid"
        )
        <= counts.get(
            "n_fibers_all"
        )
    )


def _build_result(
    *,
    package_id: str,
    identity: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": (
            "dnai_fiber_assay_result"
        ),
        "package_id": package_id,
        "assay_id": ASSAY_ID,
        "verification": "PASS",
        "source_transaction": {
            "transaction_id": (
                identity[
                    "transaction_id"
                ]
            ),
            "sha256": (
                identity[
                    "source_transaction_sha256"
                ]
            ),
        },
        "field": {
            "field_id": (
                identity[
                    "field_id"
                ]
            ),
            "sample_id": (
                identity[
                    "sample_id"
                ]
            ),
            "image_index": (
                identity[
                    "image_index"
                ]
            ),
        },
        "counts": dict(
            _require_mapping(
                identity.get(
                    "counts"
                ),
                label=(
                    "package identity counts"
                ),
            )
        ),
        "measurements": dict(
            _require_mapping(
                identity.get(
                    "measurements"
                ),
                label=(
                    "package identity measurements"
                ),
            )
        ),
        "prediction_semantics": (
            identity[
                "prediction_semantics"
            ]
        ),
        "endpoint_semantics": (
            identity[
                "endpoint_semantics"
            ]
        ),
        "scientific_scope": dict(
            _require_mapping(
                identity.get(
                    "source_scientific_scope"
                ),
                label=(
                    "package identity scientific scope"
                ),
            )
        ),
        "interpretation": {
            "package_role": (
                "derived portable result/export artifact"
            ),
            "provenance_authority": (
                "source dnai_fiber_measurement_transaction"
            ),
            "biological_validation_result": False,
            "deployment_equivalence_result": False,
        },
    }


def _build_provenance(
    *,
    identity: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": (
            "dnai_fiber_result_provenance"
        ),
        "assay_id": ASSAY_ID,
        "source_transaction": {
            "transaction_id": (
                identity[
                    "transaction_id"
                ]
            ),
            "sha256": (
                identity[
                    "source_transaction_sha256"
                ]
            ),
            "size_bytes": (
                identity[
                    "source_transaction_size_bytes"
                ]
            ),
            "transaction_fingerprint_sha256": (
                identity[
                    "transaction_fingerprint_sha256"
                ]
            ),
            "record_fingerprint_sha256": (
                identity[
                    "record_fingerprint_sha256"
                ]
            ),
            "verified_at_export": True,
        },
        "source_field": {
            "field_id": (
                identity[
                    "field_id"
                ]
            ),
            "field_record_sha256": (
                identity[
                    "field_record_sha256"
                ]
            ),
            "sample_id": (
                identity[
                    "sample_id"
                ]
            ),
            "image_index": (
                identity[
                    "image_index"
                ]
            ),
            "model_sha256": (
                identity[
                    "model_sha256"
                ]
            ),
            "validation_manifest_sha256": (
                identity[
                    "validation_manifest_sha256"
                ]
            ),
        },
        "valid_fibers_source": {
            "sha256": (
                identity[
                    "valid_fibers_csv_sha256"
                ]
            ),
            "size_bytes": (
                identity[
                    "valid_fibers_csv_size_bytes"
                ]
            ),
        },
        "path_independent_export": True,
        "upstream_binary_artifacts_included": False,
        "inference_performed_during_export": False,
        "biological_fidelity_evaluated_during_export": False,
        "kl720_accessed_during_export": False,
    }


def _render_markdown(
    result: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    field = _require_mapping(
        result.get("field"),
        label="DNAi result field",
    )

    counts = _require_mapping(
        result.get("counts"),
        label="DNAi result counts",
    )

    measurements = (
        _require_mapping(
            result.get(
                "measurements"
            ),
            label=(
                "DNAi result measurements"
            ),
        )
    )

    scope = _require_mapping(
        result.get(
            "scientific_scope"
        ),
        label=(
            "DNAi result scientific scope"
        ),
    )

    source = _require_mapping(
        provenance.get(
            "source_transaction"
        ),
        label=(
            "DNAi source transaction provenance"
        ),
    )

    lines = [
        "# Radiation Edge AI DNAi assay result",
        "",
        f"- Assay: `{ASSAY_ID}`",
        f"- Package: `{result['package_id']}`",
        f"- Verification: `{result['verification']}`",
        f"- Transaction: `{source['transaction_id']}`",
        f"- Field: `{field['field_id']}`",
        f"- Sample: `{field['sample_id']}`",
        f"- Image index: {field['image_index']}",
        "",
        "## Fiber-object endpoint",
        "",
        (
            "The reported endpoint consists of post-processed DNA-fiber "
            "objects and tract measurements reconstructed from the frozen "
            "floating FP512 application path."
        ),
        "",
        f"- Windows: {counts['n_windows']}",
        f"- Reconstructed fibers: {counts['n_fibers_all']}",
        f"- Valid fibers: {counts['n_fibers_valid']}",
        (
            "- Mean valid tract ratio: "
            f"{measurements['mean_valid_ratio']}"
        ),
        (
            "- Median valid tract ratio: "
            f"{measurements['median_valid_ratio']}"
        ),
        (
            "- Mean valid fiber length (um): "
            f"{measurements['mean_valid_length_um']}"
        ),
        (
            "- Total valid fiber length (um): "
            f"{measurements['total_valid_length_um']}"
        ),
        "",
        "## Scientific scope",
        "",
        (
            "- Raw microscopy preprocessing performed in source run: "
            f"{scope['raw_microscopy_preprocessing_performed']}"
        ),
        (
            "- Human annotations read in source run: "
            f"{scope['human_annotations_read']}"
        ),
        (
            "- FP1024 reference read in source run: "
            f"{scope['fp1024_reference_read']}"
        ),
        (
            "- Biological fidelity evaluated in source run: "
            f"{scope['biological_fidelity_evaluated']}"
        ),
        (
            "- KL720 hardware accessed in source run: "
            f"{scope['kl720_hardware_access_performed']}"
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
            "This package is a derived, path-independent presentation/export "
            "artifact. The source DNAi measurement transaction remains the "
            "provenance authority."
        ),
        "",
        (
            "The package contains the validated fiber table and summary, but "
            "does not contain the model, input tensors, stitched probability "
            "array, segmentation image, raw microscopy, human annotations, "
            "or biological reference."
        ),
        "",
        (
            "Packaging performs no inference and is not a new biological-"
            "validation, deployment-equivalence, or KL720 result."
        ),
        "",
    ]

    return "\n".join(
        lines
    )


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


def create_dnai_fiber_result_package(
    transaction_path: Path,
    *,
    output_dir: Path,
) -> Path:
    """Export one fully verified DNAi transaction."""

    source_path = (
        transaction_path
        .expanduser()
        .resolve()
    )

    if not source_path.is_file():
        raise ControlPlaneError(
            "DNAi transaction record does not exist: "
            f"{source_path}"
        )

    verification = (
        verify_dnai_fiber_measurement_transaction(
            source_path,
            check_artifacts=True,
        )
    )

    if not verification["ok"]:
        raise ControlPlaneError(
            "Source DNAi transaction failed full verification"
        )

    transaction = load_json_object(
        source_path
    )

    source = _source_components(
        transaction
    )

    source_sha = sha256_file(
        source_path
    )

    source_size = (
        source_path.stat().st_size
    )

    identity = _package_identity(
        transaction=transaction,
        source_transaction_sha256=(
            source_sha
        ),
        source_transaction_size_bytes=(
            source_size
        ),
        source=source,
    )

    if not _identity_semantics_ok(
        identity
    ):
        raise ControlPlaneError(
            "Derived DNAi package identity is malformed"
        )

    fingerprint = sha256_json(
        identity
    )

    package_id = (
        f"{PACKAGE_ID_PREFIX}-"
        f"{fingerprint[:16]}"
    )

    root = (
        output_dir
        .expanduser()
        .resolve()
    )

    package_root = (
        root / package_id
    )

    manifest_path = (
        package_root
        / "package_manifest.json"
    )

    if manifest_path.exists():
        report = (
            verify_dnai_fiber_result_package(
                manifest_path
            )
        )

        if report["ok"]:
            return manifest_path

        raise ControlPlaneError(
            "Existing DNAi result package failed verification: "
            f"{manifest_path}"
        )

    if package_root.exists():
        raise ControlPlaneError(
            "Content-addressed DNAi package directory already exists "
            f"without a verified manifest: {package_root}"
        )

    temp_root = (
        root
        / f".{package_id}.tmp"
    )

    if temp_root.exists():
        raise ControlPlaneError(
            "Temporary DNAi package directory already exists: "
            f"{temp_root}"
        )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_root.mkdir(
        parents=False,
        exist_ok=False,
    )

    fibers_path = source.get(
        "fibers_path"
    )

    if not isinstance(
        fibers_path,
        Path,
    ):
        raise ControlPlaneError(
            "DNAi valid-fibers source path is malformed"
        )

    try:
        fibers_copy = (
            temp_root
            / "valid_fibers.csv"
        )

        shutil.copyfile(
            fibers_path,
            fibers_copy,
        )

        result = _build_result(
            package_id=package_id,
            identity=identity,
        )

        provenance = (
            _build_provenance(
                identity=identity,
            )
        )

        result_path = (
            temp_root
            / "result.json"
        )

        provenance_path = (
            temp_root
            / "provenance.json"
        )

        report_path = (
            temp_root
            / "report.md"
        )

        _write_json(
            result_path,
            result,
        )

        _write_json(
            provenance_path,
            provenance,
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
            "result": (
                _portable_artifact_record(
                    temp_root,
                    result_path,
                )
            ),
            "provenance": (
                _portable_artifact_record(
                    temp_root,
                    provenance_path,
                )
            ),
            "valid_fibers": (
                _portable_artifact_record(
                    temp_root,
                    fibers_copy,
                )
            ),
            "report": (
                _portable_artifact_record(
                    temp_root,
                    report_path,
                )
            ),
        }

        manifest = {
            "schema_version": (
                PACKAGE_SCHEMA_VERSION
            ),
            "record_type": (
                PACKAGE_RECORD_TYPE
            ),
            "package_id": package_id,
            "status": "complete",
            "created_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "package_fingerprint_sha256": (
                fingerprint
            ),
            "identity": identity,
            "source_transaction_verified_at_export": True,
            "files": files,
            "interpretation": {
                "role": (
                    "derived portable result/export artifact"
                ),
                "provenance_authority": (
                    "source dnai_fiber_measurement_transaction"
                ),
                "upstream_binary_artifacts_included": False,
            },
        }

        _write_json(
            temp_root
            / "package_manifest.json",
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

    report = (
        verify_dnai_fiber_result_package(
            manifest_path
        )
    )

    if not report["ok"]:
        raise ControlPlaneError(
            "New DNAi result package failed self-verification"
        )

    return manifest_path


def verify_dnai_fiber_result_package(
    manifest_path: Path,
) -> dict[str, object]:
    """Verify DNAi portable package integrity and derivation."""

    path = (
        manifest_path
        .expanduser()
        .resolve()
    )

    manifest = load_json_object(
        path
    )

    if (
        manifest.get(
            "schema_version"
        )
        != PACKAGE_SCHEMA_VERSION
        or manifest.get(
            "record_type"
        )
        != PACKAGE_RECORD_TYPE
    ):
        raise ControlPlaneError(
            "Not a supported DNAi assay result package: "
            f"{path}"
        )

    if (
        path.name
        != "package_manifest.json"
    ):
        raise ControlPlaneError(
            "DNAi package manifest filename is invalid"
        )

    package_root = path.parent

    identity = manifest.get(
        "identity"
    )

    files = manifest.get(
        "files"
    )

    if not isinstance(
        identity,
        dict,
    ):
        raise ControlPlaneError(
            "DNAi package identity is malformed"
        )

    if not isinstance(
        files,
        dict,
    ):
        raise ControlPlaneError(
            "DNAi package file bindings are malformed"
        )

    identity_semantics_ok = (
        _identity_semantics_ok(
            identity
        )
    )

    fingerprint = manifest.get(
        "package_fingerprint_sha256"
    )

    package_id = manifest.get(
        "package_id"
    )

    fingerprint_ok = bool(
        isinstance(
            fingerprint,
            str,
        )
        and sha256_json(
            identity
        )
        == fingerprint
    )

    package_id_ok = bool(
        isinstance(
            package_id,
            str,
        )
        and isinstance(
            fingerprint,
            str,
        )
        and package_id
        == (
            f"{PACKAGE_ID_PREFIX}-"
            f"{fingerprint[:16]}"
        )
        and package_root.name
        == package_id
    )

    expected_file_names = {
        "result",
        "provenance",
        "valid_fibers",
        "report",
    }

    file_keys_ok = (
        set(files)
        == expected_file_names
    )

    files_ok = bool(
        file_keys_ok
        and all(
            _verify_portable_artifact(
                package_root,
                files[name],
            )
            for name
            in expected_file_names
        )
    )

    derivation_ok = False
    scientific_scope_ok = False
    counts_ok = False
    csv_binding_ok = False

    n_windows: object = None
    n_fibers_valid: object = None

    if (
        identity_semantics_ok
        and files_ok
    ):
        result_record = (
            _require_mapping(
                files["result"],
                label=(
                    "DNAi result file record"
                ),
            )
        )

        provenance_record = (
            _require_mapping(
                files["provenance"],
                label=(
                    "DNAi provenance file record"
                ),
            )
        )

        fibers_record = (
            _require_mapping(
                files["valid_fibers"],
                label=(
                    "DNAi fiber table file record"
                ),
            )
        )

        report_record = (
            _require_mapping(
                files["report"],
                label=(
                    "DNAi report file record"
                ),
            )
        )

        result_path = (
            package_root
            / str(
                result_record[
                    "relative_path"
                ]
            )
        )

        provenance_path = (
            package_root
            / str(
                provenance_record[
                    "relative_path"
                ]
            )
        )

        fibers_path = (
            package_root
            / str(
                fibers_record[
                    "relative_path"
                ]
            )
        )

        report_path = (
            package_root
            / str(
                report_record[
                    "relative_path"
                ]
            )
        )

        result = load_json_object(
            result_path
        )

        provenance = (
            load_json_object(
                provenance_path
            )
        )

        counts = _require_mapping(
            result.get("counts"),
            label=(
                "DNAi packaged counts"
            ),
        )

        n_windows = counts.get(
            "n_windows"
        )

        n_fibers_valid = (
            counts.get(
                "n_fibers_valid"
            )
        )

        row_count = (
            _csv_row_count(
                fibers_path
            )
        )

        identity_counts = (
            _require_mapping(
                identity.get(
                    "counts"
                ),
                label=(
                    "DNAi package identity counts"
                ),
            )
        )

        counts_ok = bool(
            dict(counts)
            == dict(
                identity_counts
            )
            and n_windows == 9
            and isinstance(
                n_fibers_valid,
                int,
            )
            and not isinstance(
                n_fibers_valid,
                bool,
            )
            and n_fibers_valid
            == row_count
        )

        scientific_scope_ok = bool(
            result.get(
                "scientific_scope"
            )
            == identity.get(
                "source_scientific_scope"
            )
            == EXPECTED_TRANSACTION_SCOPE
        )

        csv_binding_ok = bool(
            fibers_record.get(
                "sha256"
            )
            == identity.get(
                "valid_fibers_csv_sha256"
            )
            and fibers_record.get(
                "size_bytes"
            )
            == identity.get(
                "valid_fibers_csv_size_bytes"
            )
        )

        expected_result = (
            _build_result(
                package_id=str(
                    package_id
                ),
                identity=identity,
            )
        )

        expected_provenance = (
            _build_provenance(
                identity=identity,
            )
        )

        expected_report = (
            _render_markdown(
                expected_result,
                expected_provenance,
            )
        )

        report_text = (
            report_path.read_text(
                encoding="utf-8"
            )
        )

        derivation_ok = bool(
            result
            == expected_result
            and provenance
            == expected_provenance
            and report_text
            == expected_report
            and result.get(
                "assay_id"
            )
            == ASSAY_ID
            and provenance.get(
                "assay_id"
            )
            == ASSAY_ID
            and identity.get(
                "assay_id"
            )
            == ASSAY_ID
            and result.get(
                "verification"
            )
            == "PASS"
        )

    status_ok = (
        manifest.get("status")
        == "complete"
    )

    interpretation = (
        _require_mapping(
            manifest.get(
                "interpretation"
            ),
            label=(
                "DNAi package interpretation"
            ),
        )
    )

    role_ok = bool(
        manifest.get(
            "source_transaction_verified_at_export"
        )
        is True
        and interpretation.get(
            "provenance_authority"
        )
        == (
            "source dnai_fiber_measurement_transaction"
        )
        and interpretation.get(
            "upstream_binary_artifacts_included"
        )
        is False
    )

    ok = all(
        (
            status_ok,
            identity_semantics_ok,
            fingerprint_ok,
            package_id_ok,
            files_ok,
            derivation_ok,
            scientific_scope_ok,
            counts_ok,
            csv_binding_ok,
            role_ok,
        )
    )

    return {
        "kind": (
            PACKAGE_RECORD_TYPE
        ),
        "package_id": (
            package_id
        ),
        "manifest_path": (
            str(path)
        ),
        "identity_semantics_ok": (
            identity_semantics_ok
        ),
        "fingerprint_ok": (
            fingerprint_ok
        ),
        "package_id_ok": (
            package_id_ok
        ),
        "files_ok": files_ok,
        "derivation_ok": (
            derivation_ok
        ),
        "scientific_scope_ok": (
            scientific_scope_ok
        ),
        "counts_ok": counts_ok,
        "csv_binding_ok": (
            csv_binding_ok
        ),
        "source_transaction_verified_at_export": (
            manifest.get(
                "source_transaction_verified_at_export"
            )
            is True
        ),
        "n_windows": n_windows,
        "n_fibers_valid": (
            n_fibers_valid
        ),
        "ok": ok,
    }
