"""Assay-level runtime dispatch for Radiation Edge AI.

This module defines the application-facing multi-assay boundary.

It deliberately does not generalize assay-specific scientific algorithms.
Each registered assay owns its own validated execution, measurement, QC, and
reporting implementation.

v0.8 enables the existing NASA application and result-package adapters through
this registry. DNAi is registered with an explicit scientific contract but
remains execution-disabled until its tiled segmentation, stitching, object
reconstruction, and QC path is integrated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from radiation_edge_ai.control import (
    ControlPlaneError,
    get_assay,
    load_json_object,
)

NASA_ASSAY_ID = "nasa-53bp1-r1-v2"
DNAI_ASSAY_ID = "dnai-fiber-v3"


@dataclass(frozen=True)
class AssayRuntimeSpec:
    """Application capabilities for one registered assay."""

    assay_id: str
    adapter_version: int
    application_adapter: str
    package_adapter: str
    package_source_record_type: str
    manifest_run_supported: bool
    result_package_supported: bool
    prediction_semantics: str
    endpoint_semantics: str
    scientific_boundary: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_RUNTIME_SPECS = {
    NASA_ASSAY_ID: AssayRuntimeSpec(
        assay_id=NASA_ASSAY_ID,
        adapter_version=1,
        application_adapter=(
            "nasa-batch-measurement-v1"
        ),
        package_adapter=(
            "nasa-sample-aggregate-package-v1"
        ),
        package_source_record_type=(
            "batch_measurement_transaction"
        ),
        manifest_run_supported=True,
        result_package_supported=True,
        prediction_semantics=(
            "per-nucleus latent continuous 53BP1 burden"
        ),
        endpoint_semantics=(
            "sample-level arithmetic mean latent 53BP1 burden"
        ),
        scientific_boundary=(
            "Per-nucleus output is not an individually supervised "
            "focus count. Biological interpretation belongs at the "
            "aggregate endpoint."
        ),
    ),
    DNAI_ASSAY_ID: AssayRuntimeSpec(
        assay_id=DNAI_ASSAY_ID,
        adapter_version=1,
        application_adapter=(
            "dnai-tile-stitch-object-v1"
        ),
        package_adapter=(
            "dnai-fiber-measurement-package-v1"
        ),
        package_source_record_type=(
            "dnai_fiber_measurement_transaction"
        ),
        manifest_run_supported=False,
        result_package_supported=False,
        prediction_semantics=(
            "tile-level segmentation reconstructed into a "
            "stitched microscopy-field segmentation"
        ),
        endpoint_semantics=(
            "post-processed DNA-fiber objects and tract measurements"
        ),
        scientific_boundary=(
            "Pixel agreement is not a surrogate for fiber-object "
            "fidelity. Object reconstruction and downstream tract "
            "measurements require assay-specific QC."
        ),
    ),
}


def list_assay_runtime_specs(
) -> tuple[AssayRuntimeSpec, ...]:
    """Return all registered runtime adapters."""

    return tuple(
        _RUNTIME_SPECS[key]
        for key in sorted(_RUNTIME_SPECS)
    )


def get_assay_runtime_spec(
    assay_id: str,
) -> AssayRuntimeSpec:
    """Return the registered runtime contract for an assay."""

    # Require the assay to exist in the canonical assay registry too.
    get_assay(assay_id)

    try:
        return _RUNTIME_SPECS[assay_id]
    except KeyError as exc:
        known = ", ".join(
            sorted(_RUNTIME_SPECS)
        )
        raise ControlPlaneError(
            "No application runtime adapter is registered for "
            f"{assay_id!r}; registered adapters: {known}"
        ) from exc


def _required_assay_id(
    value: object,
    *,
    label: str,
) -> str:
    if not isinstance(value, str):
        raise ControlPlaneError(
            f"{label} must be a string"
        )

    assay_id = value.strip()

    if not assay_id:
        raise ControlPlaneError(
            f"{label} must not be blank"
        )

    return assay_id


def assay_id_from_manifest(
    manifest_path: Path,
) -> str:
    """Read only the assay identity from an application manifest."""

    value = load_json_object(
        manifest_path.expanduser().resolve()
    )

    return _required_assay_id(
        value.get("assay_id"),
        label="manifest assay_id",
    )


def assay_id_from_result_source(
    source_path: Path,
) -> str:
    """Resolve assay identity from a verified-result source artifact."""

    value = load_json_object(
        source_path.expanduser().resolve()
    )

    direct = value.get("assay_id")

    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    transaction_identity = value.get(
        "transaction_identity"
    )

    if isinstance(
        transaction_identity,
        dict,
    ):
        assay_id = transaction_identity.get(
            "assay_id"
        )

        if isinstance(
            assay_id,
            str,
        ) and assay_id.strip():
            return assay_id.strip()

    record_identity = value.get(
        "record_identity"
    )

    if isinstance(
        record_identity,
        dict,
    ):
        assay_id = record_identity.get(
            "assay_id"
        )

        if isinstance(
            assay_id,
            str,
        ) and assay_id.strip():
            return assay_id.strip()

    identity = value.get("identity")

    if isinstance(identity, dict):
        assay_id = identity.get("assay_id")

        if isinstance(
            assay_id,
            str,
        ) and assay_id.strip():
            return assay_id.strip()

    raise ControlPlaneError(
        "Could not resolve assay identity from source artifact"
    )


def execute_registered_assay_manifest(
    manifest_path: Path,
    *,
    output_dir: Path,
    onnx_python: Optional[Path] = None,  # noqa: UP045
) -> dict[str, object]:
    """Dispatch one manifest to its registered assay application adapter."""

    assay_id = assay_id_from_manifest(
        manifest_path
    )

    runtime = get_assay_runtime_spec(
        assay_id
    )

    if not runtime.manifest_run_supported:
        raise ControlPlaneError(
            f"Assay {assay_id!r} is registered but manifest "
            "execution is not enabled. Required adapter: "
            f"{runtime.application_adapter}. "
            f"Scientific boundary: {runtime.scientific_boundary}"
        )

    if assay_id == NASA_ASSAY_ID:
        from radiation_edge_ai.application import (
            execute_assay_manifest,
        )

        summary = execute_assay_manifest(
            manifest_path,
            output_dir=output_dir,
            onnx_python=onnx_python,
        )

        if summary.get("assay_id") != assay_id:
            raise ControlPlaneError(
                "Assay application returned a mismatched assay_id"
            )

        return summary

    raise ControlPlaneError(
        f"No execution dispatch implementation for assay {assay_id!r}"
    )


def create_registered_assay_result_package(
    source_path: Path,
    *,
    output_dir: Path,
) -> Path:
    """Dispatch result packaging according to source assay identity."""

    resolved = source_path.expanduser().resolve()
    value = load_json_object(resolved)

    assay_id = assay_id_from_result_source(
        resolved
    )

    runtime = get_assay_runtime_spec(
        assay_id
    )

    if not runtime.result_package_supported:
        raise ControlPlaneError(
            f"Assay {assay_id!r} is registered but result packaging "
            "is not enabled. Required adapter: "
            f"{runtime.package_adapter}. "
            f"Scientific boundary: {runtime.scientific_boundary}"
        )

    record_type = value.get(
        "record_type"
    )

    if (
        record_type
        != runtime.package_source_record_type
    ):
        raise ControlPlaneError(
            f"Assay {assay_id!r} result packaging expects source "
            f"record_type {runtime.package_source_record_type!r}; "
            f"got {record_type!r}"
        )

    if assay_id == NASA_ASSAY_ID:
        from radiation_edge_ai.reporting import (
            create_assay_result_package,
        )

        return create_assay_result_package(
            resolved,
            output_dir=output_dir,
        )

    raise ControlPlaneError(
        f"No package dispatch implementation for assay {assay_id!r}"
    )


def verify_registered_assay_result_package(
    manifest_path: Path,
) -> dict[str, object]:
    """Dispatch portable-package verification by embedded assay identity."""

    resolved = manifest_path.expanduser().resolve()
    value = load_json_object(resolved)

    if (
        value.get("record_type")
        != "assay_result_package"
    ):
        raise ControlPlaneError(
            f"Not an assay result package: {resolved}"
        )

    identity = value.get("identity")

    if not isinstance(identity, dict):
        raise ControlPlaneError(
            "Result-package identity is malformed"
        )

    assay_id = _required_assay_id(
        identity.get("assay_id"),
        label="result-package assay_id",
    )

    runtime = get_assay_runtime_spec(
        assay_id
    )

    if not runtime.result_package_supported:
        raise ControlPlaneError(
            f"Assay {assay_id!r} has no enabled result-package "
            f"verifier; adapter: {runtime.package_adapter}"
        )

    if assay_id == NASA_ASSAY_ID:
        from radiation_edge_ai.reporting import (
            verify_assay_result_package,
        )

        report = verify_assay_result_package(
            resolved
        )

        report["assay_id"] = assay_id
        report["runtime_adapter"] = (
            runtime.package_adapter
        )

        return report

    raise ControlPlaneError(
        f"No package verification dispatch for assay {assay_id!r}"
    )
