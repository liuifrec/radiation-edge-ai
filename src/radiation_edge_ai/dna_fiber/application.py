"""DNAi offline application execution.

This module is a thin control-plane adapter around the frozen DNAi field
worker. Scientific algorithms remain inside the assay-specific worker.

application manifest
-> verified frozen assets
-> external DNAi Python
-> one field_result.json
-> strict field verification
-> DNAi provenance transaction
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from radiation_edge_ai.control import (
    ControlPlaneError,
    load_json_object,
)
from radiation_edge_ai.dna_fiber.field_record import (
    verify_dnai_fiber_field_record,
)
from radiation_edge_ai.dna_fiber.transaction import (
    create_dnai_fiber_measurement_transaction,
    validate_dnai_application_manifest,
    verify_dnai_fiber_measurement_transaction,
)

ASSAY_ID = "dnai-fiber-v3"
APPLICATION_ADAPTER = "dnai-tile-stitch-object-v1"

PREDICTION_SEMANTICS = (
    "tile-level segmentation reconstructed into a "
    "stitched microscopy-field segmentation"
)

ENDPOINT_SEMANTICS = (
    "post-processed DNA-fiber objects and tract measurements"
)


def _required_runtime_python(
    value: Optional[Path],  # noqa: UP045
) -> Path:
    if value is None:
        raise ControlPlaneError(
            "DNAi manifest execution requires --onnx-python "
            "pointing to the external DNAi runtime"
        )

    resolved = (
        value
        .expanduser()
        .resolve()
    )

    if not resolved.is_file():
        raise ControlPlaneError(
            f"DNAi runtime Python is not a readable file: {resolved}"
        )

    return resolved


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


def _matching_existing_field_record(
    field_root: Path,
    manifest: dict[str, Any],
) -> Optional[Path]:  # noqa: UP045
    """Find one already verified field result matching this manifest."""

    if not field_root.is_dir():
        return None

    model = _required_mapping(
        manifest.get("model"),
        label="model artifact",
    )

    validation = _required_mapping(
        manifest.get(
            "validation_manifest"
        ),
        label="validation artifact",
    )

    metadata = _required_mapping(
        manifest.get("metadata"),
        label="metadata",
    )

    matches: list[Path] = []

    for candidate in sorted(
        field_root.glob(
            "df1-*/field_result.json"
        )
    ):
        try:
            verification = (
                verify_dnai_fiber_field_record(
                    candidate,
                    check_artifacts=True,
                )
            )
        except ControlPlaneError:
            continue

        if not verification["ok"]:
            continue

        value = load_json_object(
            candidate
        )

        identity = value.get(
            "identity"
        )

        if not isinstance(
            identity,
            dict,
        ):
            continue

        if (
            identity.get(
                "model_sha256"
            )
            == model.get(
                "sha256"
            )
            and identity.get(
                "validation_manifest_sha256"
            )
            == validation.get(
                "sha256"
            )
            and identity.get(
                "image_index"
            )
            == manifest.get(
                "image_index"
            )
            and identity.get(
                "sample_id"
            )
            == metadata.get(
                "sample_id"
            )
        ):
            matches.append(
                candidate.resolve()
            )

    if len(matches) > 1:
        raise ControlPlaneError(
            "Multiple verified DNAi field records match one "
            "application manifest"
        )

    if matches:
        return matches[0]

    return None


def _run_field_worker(
    manifest: dict[str, Any],
    *,
    runtime_python: Path,
    output_root: Path,
) -> Path:
    """Launch the assay-specific worker in the external DNAi environment."""

    model = _required_mapping(
        manifest.get("model"),
        label="model artifact",
    )

    validation = _required_mapping(
        manifest.get(
            "validation_manifest"
        ),
        label="validation artifact",
    )

    model_path = Path(
        str(model["path"])
    ).expanduser().resolve()

    validation_path = Path(
        str(validation["path"])
    ).expanduser().resolve()

    image_index = manifest.get(
        "image_index"
    )

    if not isinstance(
        image_index,
        int,
    ) or isinstance(
        image_index,
        bool,
    ):
        raise ControlPlaneError(
            "DNAi image_index is malformed"
        )

    src_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    environment = os.environ.copy()

    existing_pythonpath = environment.get(
        "PYTHONPATH"
    )

    if existing_pythonpath:
        environment["PYTHONPATH"] = (
            str(src_root)
            + os.pathsep
            + existing_pythonpath
        )
    else:
        environment["PYTHONPATH"] = str(
            src_root
        )

    command = [
        str(runtime_python),
        "-m",
        "radiation_edge_ai.dna_fiber._field_worker",
        "--model",
        str(model_path),
        "--validation-manifest",
        str(validation_path),
        "--image-index",
        str(image_index),
        "--output-root",
        str(output_root),
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=environment,
    )

    if completed.returncode != 0:
        stderr = (
            completed.stderr
            or completed.stdout
            or ""
        )

        raise ControlPlaneError(
            "DNAi external field worker failed "
            f"with exit code {completed.returncode}: "
            f"{stderr[-2000:]}"
        )

    worker_result = None

    for line in reversed(
        completed.stdout.splitlines()
    ):
        stripped = line.strip()

        if not stripped:
            continue

        try:
            value = json.loads(
                stripped
            )
        except json.JSONDecodeError:
            continue

        if (
            isinstance(value, dict)
            and value.get("status")
            == "complete"
            and isinstance(
                value.get("record"),
                str,
            )
        ):
            worker_result = value
            break

    if worker_result is None:
        raise ControlPlaneError(
            "DNAi worker completed without a valid result record"
        )

    record_path = Path(
        worker_result["record"]
    ).expanduser().resolve()

    if not record_path.is_file():
        raise ControlPlaneError(
            "DNAi worker-reported field record is missing: "
            f"{record_path}"
        )

    resolved_output_root = (
        output_root
        .expanduser()
        .resolve()
    )

    try:
        record_path.relative_to(
            resolved_output_root
        )
    except ValueError as exc:
        raise ControlPlaneError(
            "DNAi worker returned a field record outside "
            "the requested output root"
        ) from exc

    return record_path


def execute_dnai_assay_manifest(
    manifest_path: Path,
    *,
    output_dir: Path,
    onnx_python: Optional[Path] = None,  # noqa: UP045
) -> dict[str, object]:
    """Execute one validated DNAi field application manifest."""

    resolved_manifest = (
        manifest_path
        .expanduser()
        .resolve()
    )

    manifest = (
        validate_dnai_application_manifest(
            resolved_manifest
        )
    )

    runtime_python = (
        _required_runtime_python(
            onnx_python
        )
    )

    root = (
        output_dir
        .expanduser()
        .resolve()
    )

    field_root = (
        root / "fields"
    )

    field_record_path = (
        _matching_existing_field_record(
            field_root,
            manifest,
        )
    )

    field_execution = "reused"

    if field_record_path is None:
        field_execution = "executed"

        field_record_path = (
            _run_field_worker(
                manifest,
                runtime_python=runtime_python,
                output_root=field_root,
            )
        )

    field_verification = (
        verify_dnai_fiber_field_record(
            field_record_path,
            check_artifacts=True,
        )
    )

    if not field_verification["ok"]:
        raise ControlPlaneError(
            "DNAi field record failed verification after execution/reuse"
        )

    transaction_path = (
        create_dnai_fiber_measurement_transaction(
            resolved_manifest,
            field_record_path,
            output_dir=(
                root
                / "transactions"
            ),
        )
    )

    transaction_verification = (
        verify_dnai_fiber_measurement_transaction(
            transaction_path,
            check_artifacts=True,
        )
    )

    if not transaction_verification[
        "ok"
    ]:
        raise ControlPlaneError(
            "DNAi measurement transaction failed verification"
        )

    field_record = load_json_object(
        field_record_path
    )

    transaction = load_json_object(
        transaction_path
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

    field_scope = _required_mapping(
        field_record.get(
            "scientific_scope"
        ),
        label="field scientific scope",
    )

    transaction_scope = (
        _required_mapping(
            transaction.get(
                "scientific_scope"
            ),
            label=(
                "transaction scientific scope"
            ),
        )
    )

    field_id = field_record.get(
        "field_id"
    )

    transaction_id = (
        transaction.get(
            "transaction_id"
        )
    )

    if (
        not isinstance(field_id, str)
        or not field_id
        or not isinstance(
            transaction_id,
            str,
        )
        or not transaction_id
    ):
        raise ControlPlaneError(
            "Verified DNAi result identifiers are malformed"
        )

    return {
        "kind": "assay_run_summary",
        "assay_id": ASSAY_ID,
        "status": "complete",
        "verification": "PASS",
        "application_adapter": (
            APPLICATION_ADAPTER
        ),
        "application_manifest_path": (
            str(
                resolved_manifest
            )
        ),
        "field_record_path": (
            str(
                field_record_path
            )
        ),
        "transaction_record_path": (
            str(
                transaction_path
            )
        ),
        "field_execution": (
            field_execution
        ),
        "ids": {
            "field_id": field_id,
            "transaction_id": (
                transaction_id
            ),
        },
        "counts": counts,
        "measurements": (
            measurements
        ),
        "prediction_semantics": (
            PREDICTION_SEMANTICS
        ),
        "endpoint_semantics": (
            ENDPOINT_SEMANTICS
        ),
        "scientific_scope": (
            transaction_scope
        ),
        "field_scientific_scope": (
            field_scope
        ),
    }
