"""Verification for backend-aware DNAi physical field records.

This v0.11 layer is additive. It does not replace the frozen v0.9/v0.10
floating-CPU field-record verifier.

A physical field record binds nine already-preprocessed DNAi deployment
windows to nine verified generic KL720 run records, followed by the same
assay-specific stitching and fiber reconstruction semantics.

This module performs verification only. It does not execute inference.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from radiation_edge_ai.control import (
    ControlPlaneError,
    load_json_object,
    sha256_file,
    sha256_json,
)
from radiation_edge_ai.execution import (
    verify_run_record,
)

SCHEMA_VERSION = 2
RECORD_TYPE = "dnai_fiber_field_record"
FIELD_ID_PREFIX = "df2"

ASSAY_ID = "dnai-fiber-v3"
APPLICATION_ADAPTER = "dnai-tile-stitch-object-v1"
BACKEND = "kl720"

DNAI_COMMIT = (
    "fcf20c7d6eb385675ff7d07da4fdf471589ce0cf"
)

EXPECTED_NEF_SHA256 = (
    "4b3dfec9a61c99e186dd4b8482fa5b06"
    "e6a4958f325ed4a0db0546f1dcab2bfc"
)

EXPECTED_TILE = [1, 3, 512, 512]

EXPECTED_COORDS = {
    (0, 0),
    (0, 256),
    (0, 512),
    (256, 0),
    (256, 256),
    (256, 512),
    (512, 0),
    (512, 256),
    (512, 512),
}

EXPECTED_SCOPE = {
    "preprocessed_frozen_window_inputs": True,
    "raw_microscopy_preprocessing_performed": False,
    "floating_fp512_inference_performed": False,
    "physical_kl720_inference_performed": True,
    "gaussian_stitching_performed": True,
    "fiber_object_reconstruction_performed": True,
    "human_annotations_read": False,
    "fp1024_reference_read": False,
    "biological_fidelity_evaluated": False,
    "kl720_hardware_access_performed": True,
}


def _is_sha256(
    value: object,
) -> bool:
    if not isinstance(value, str):
        return False

    if len(value) != 64:
        return False

    try:
        int(value, 16)
    except ValueError:
        return False

    return True


def _nonnegative_int(
    value: object,
) -> bool:
    return bool(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _finite_number(
    value: object,
    *,
    allow_none: bool = False,
) -> bool:
    if value is None:
        return allow_none

    if isinstance(value, bool):
        return False

    if not isinstance(
        value,
        (int, float),
    ):
        return False

    return math.isfinite(
        float(value)
    )


def _artifact_metadata_ok(
    value: object,
) -> bool:
    if not isinstance(value, dict):
        return False

    path = value.get("path")
    digest = value.get("sha256")
    size = value.get("size_bytes")

    return bool(
        isinstance(path, str)
        and path.strip()
        and _is_sha256(digest)
        and _nonnegative_int(size)
    )


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


def _artifact_ok(
    value: object,
    *,
    check_artifacts: bool,
) -> bool:
    if check_artifacts:
        return _artifact_bytes_ok(
            value
        )

    return _artifact_metadata_ok(
        value
    )


def _identity_semantics_ok(
    identity: object,
) -> bool:
    if not isinstance(identity, dict):
        return False

    if (
        identity.get("schema_version")
        != SCHEMA_VERSION
        or identity.get("record_type")
        != RECORD_TYPE
        or identity.get("assay_id")
        != ASSAY_ID
        or identity.get(
            "application_adapter"
        )
        != APPLICATION_ADAPTER
        or identity.get("backend")
        != BACKEND
        or identity.get("dnai_commit")
        != DNAI_COMMIT
        or identity.get("model_sha256")
        != EXPECTED_NEF_SHA256
    ):
        return False

    if not _is_sha256(
        identity.get(
            "validation_manifest_sha256"
        )
    ):
        return False

    if not _nonnegative_int(
        identity.get(
            "image_index"
        )
    ):
        return False

    sample_id = identity.get(
        "sample_id"
    )

    key = identity.get("key")

    if (
        not isinstance(sample_id, str)
        or not sample_id.strip()
        or not isinstance(key, str)
        or not key.strip()
    ):
        return False

    if (
        identity.get(
            "input_semantics"
        )
        != (
            "frozen preprocessed and ImageNet-normalized "
            "DNAi deployment tensors"
        )
    ):
        return False

    policy = identity.get(
        "deployment_policy"
    )

    if not isinstance(policy, dict):
        return False

    if (
        policy.get("tile")
        != EXPECTED_TILE
        or policy.get("overlap")
        != 0.50
        or policy.get("stride")
        != 256
        or policy.get("blend")
        != "gaussian"
        or policy.get("sigma_scale")
        != 0.125
        or policy.get("pixel_size_um")
        != 0.26
    ):
        return False

    windows = identity.get(
        "windows"
    )

    if (
        not isinstance(windows, list)
        or len(windows) != 9
    ):
        return False

    indices = []
    coords = set()

    for window in windows:
        if not isinstance(window, dict):
            return False

        index = window.get(
            "window_index"
        )

        y = window.get("y")
        x = window.get("x")

        if (
            not _nonnegative_int(index)
            or not _nonnegative_int(y)
            or not _nonnegative_int(x)
            or not _is_sha256(
                window.get(
                    "tensor_sha256"
                )
            )
            or not _nonnegative_int(
                window.get(
                    "tensor_size_bytes"
                )
            )
        ):
            return False

        indices.append(
            int(index)
        )

        coords.add(
            (
                int(y),
                int(x),
            )
        )

    return bool(
        sorted(indices)
        == list(range(9))
        and coords == EXPECTED_COORDS
    )


def _source_bindings_ok(
    value: dict[str, object],
    *,
    check_artifacts: bool,
) -> bool:
    identity = value.get(
        "identity"
    )

    sources = value.get(
        "source_artifacts"
    )

    if (
        not isinstance(identity, dict)
        or not isinstance(sources, dict)
    ):
        return False

    model = sources.get("model")

    validation = sources.get(
        "validation_manifest"
    )

    if (
        not _artifact_ok(
            model,
            check_artifacts=check_artifacts,
        )
        or not _artifact_ok(
            validation,
            check_artifacts=check_artifacts,
        )
    ):
        return False

    assert isinstance(model, dict)
    assert isinstance(validation, dict)

    model_path = Path(
        str(model["path"])
    )

    return bool(
        model.get("sha256")
        == identity.get(
            "model_sha256"
        )
        == EXPECTED_NEF_SHA256
        and model_path.suffix.lower()
        == ".nef"
        and validation.get(
            "sha256"
        )
        == identity.get(
            "validation_manifest_sha256"
        )
    )


def _window_execution_bindings(
    value: dict[str, object],
    *,
    check_artifacts: bool,
) -> tuple[
    bool,
    Optional[dict[str, list[object]]],  # noqa: UP045
]:
    identity = value.get(
        "identity"
    )

    outputs = value.get(
        "window_outputs"
    )

    if (
        not isinstance(identity, dict)
        or not isinstance(outputs, list)
        or len(outputs) != 9
    ):
        return False, None

    identity_windows = identity.get(
        "windows"
    )

    if not isinstance(
        identity_windows,
        list,
    ):
        return False, None

    expected = {
        int(window["window_index"]): window
        for window in identity_windows
        if (
            isinstance(window, dict)
            and _nonnegative_int(
                window.get(
                    "window_index"
                )
            )
        )
    }

    if set(expected) != set(
        range(9)
    ):
        return False, None

    observed_indices = set()

    ports = set()
    product_ids = set()
    firmwares = set()
    model_ids = set()

    for output in outputs:
        if not isinstance(
            output,
            dict,
        ):
            return False, None

        index = output.get(
            "window_index"
        )

        y = output.get("y")
        x = output.get("x")

        if (
            not _nonnegative_int(index)
            or int(index) not in expected
            or int(index)
            in observed_indices
            or not _nonnegative_int(y)
            or not _nonnegative_int(x)
        ):
            return False, None

        index_int = int(index)
        observed_indices.add(
            index_int
        )

        identity_window = expected[
            index_int
        ]

        if (
            int(y)
            != int(
                identity_window["y"]
            )
            or int(x)
            != int(
                identity_window["x"]
            )
        ):
            return False, None

        input_artifact = output.get(
            "input"
        )

        run_artifact = output.get(
            "run_record"
        )

        output_artifact = output.get(
            "output"
        )

        if (
            not _artifact_ok(
                input_artifact,
                check_artifacts=check_artifacts,
            )
            or not _artifact_ok(
                run_artifact,
                check_artifacts=check_artifacts,
            )
            or not _artifact_ok(
                output_artifact,
                check_artifacts=check_artifacts,
            )
        ):
            return False, None

        assert isinstance(
            input_artifact,
            dict,
        )

        assert isinstance(
            run_artifact,
            dict,
        )

        assert isinstance(
            output_artifact,
            dict,
        )

        if (
            input_artifact.get(
                "sha256"
            )
            != identity_window.get(
                "tensor_sha256"
            )
            or input_artifact.get(
                "size_bytes"
            )
            != identity_window.get(
                "tensor_size_bytes"
            )
        ):
            return False, None

        inference_ms = output.get(
            "inference_ms"
        )

        if (
            not _finite_number(
                inference_ms
            )
            or float(
                inference_ms
            )
            < 0.0
        ):
            return False, None

        if not check_artifacts:
            continue

        run_path = Path(
            str(
                run_artifact[
                    "path"
                ]
            )
        ).expanduser().resolve()

        try:
            run_report = verify_run_record(
                run_path,
                check_source_artifacts=True,
            )
        except ControlPlaneError:
            return False, None

        if not run_report.get(
            "ok"
        ):
            return False, None

        try:
            run = load_json_object(
                run_path
            )
        except ControlPlaneError:
            return False, None

        run_sources = run.get(
            "source_artifacts"
        )

        raw_output = run.get(
            "raw_output"
        )

        runtime = run.get(
            "runtime"
        )

        if (
            not isinstance(
                run_sources,
                dict,
            )
            or not isinstance(
                raw_output,
                dict,
            )
            or not isinstance(
                runtime,
                dict,
            )
        ):
            return False, None

        run_input = run_sources.get(
            "input"
        )

        run_model = run_sources.get(
            "model"
        )

        if (
            not isinstance(
                run_input,
                dict,
            )
            or not isinstance(
                run_model,
                dict,
            )
        ):
            return False, None

        if (
            run.get("backend")
            != BACKEND
            or run.get("assay_id")
            != ASSAY_ID
            or run_input.get(
                "sha256"
            )
            != input_artifact.get(
                "sha256"
            )
            or run_model.get(
                "sha256"
            )
            != EXPECTED_NEF_SHA256
            or raw_output.get(
                "sha256"
            )
            != output_artifact.get(
                "sha256"
            )
            or raw_output.get(
                "dtype"
            )
            != "float32"
            or raw_output.get(
                "shape"
            )
            != EXPECTED_TILE
        ):
            return False, None

        port = runtime.get(
            "selected_usb_port"
        )

        product_id = runtime.get(
            "product_id"
        )

        model_id = runtime.get(
            "model_id"
        )

        firmware = runtime.get(
            "firmware"
        )

        if (
            not _nonnegative_int(port)
            or product_id != 0x720
            or not _nonnegative_int(
                model_id
            )
        ):
            return False, None

        ports.add(
            int(port)
        )

        product_ids.add(
            int(product_id)
        )

        model_ids.add(
            int(model_id)
        )

        if firmware is not None:
            if not isinstance(
                firmware,
                str,
            ):
                return False, None

            firmwares.add(
                firmware
            )

    if observed_indices != set(
        range(9)
    ):
        return False, None

    if not check_artifacts:
        return True, None

    observed_runtime = {
        "selected_usb_ports": (
            sorted(ports)
        ),
        "product_ids": (
            sorted(product_ids)
        ),
        "firmwares": (
            sorted(firmwares)
        ),
        "model_ids": (
            sorted(model_ids)
        ),
    }

    return True, observed_runtime


def _runtime_ok(
    value: dict[str, object],
    *,
    observed: Optional[  # noqa: UP045
        dict[str, list[object]]
    ],
) -> bool:
    runtime = value.get(
        "runtime"
    )

    if not isinstance(runtime, dict):
        return False

    if (
        runtime.get(
            "execution_backend"
        )
        != BACKEND
        or runtime.get(
            "execution_adapter"
        )
        != "generic-run-plan-v1"
    ):
        return False

    for name in (
        "selected_usb_ports",
        "product_ids",
        "firmwares",
        "model_ids",
    ):
        item = runtime.get(name)

        if not isinstance(
            item,
            list,
        ):
            return False

        if (
            observed is not None
            and item != observed[name]
        ):
            return False

    product_ids = runtime.get(
        "product_ids"
    )

    assert isinstance(
        product_ids,
        list,
    )

    return product_ids == [0x720]


def _derived_artifacts_ok(
    value: dict[str, object],
    *,
    check_artifacts: bool,
) -> bool:
    artifacts = value.get(
        "artifacts"
    )

    if not isinstance(
        artifacts,
        dict,
    ):
        return False

    required = (
        "stitched_probabilities",
        "segmentation",
        "valid_fibers_csv",
    )

    return all(
        _artifact_ok(
            artifacts.get(name),
            check_artifacts=(
                check_artifacts
            ),
        )
        for name in required
    )


def _counts_ok(
    value: dict[str, object],
) -> bool:
    counts = value.get(
        "counts"
    )

    if not isinstance(counts, dict):
        return False

    n_windows = counts.get(
        "n_windows"
    )

    n_all = counts.get(
        "n_fibers_all"
    )

    n_valid = counts.get(
        "n_fibers_valid"
    )

    if (
        n_windows != 9
        or not _nonnegative_int(
            n_all
        )
        or not _nonnegative_int(
            n_valid
        )
    ):
        return False

    return int(n_valid) <= int(
        n_all
    )


def _measurements_ok(
    value: dict[str, object],
) -> bool:
    measurements = value.get(
        "measurements"
    )

    if not isinstance(
        measurements,
        dict,
    ):
        return False

    for name in (
        "mean_valid_ratio",
        "median_valid_ratio",
        "mean_valid_length_um",
    ):
        if not _finite_number(
            measurements.get(name),
            allow_none=True,
        ):
            return False

    total_length = measurements.get(
        "total_valid_length_um"
    )

    return bool(
        _finite_number(
            total_length
        )
        and float(
            total_length
        )
        >= 0.0
    )


def verify_dnai_fiber_physical_field_record(
    record_path: Path,
    *,
    check_artifacts: bool = True,
) -> dict[str, object]:
    """Verify one v0.11 backend-aware physical DNAi field record."""

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
            "Not a DNAi fiber field record: "
            f"{resolved}"
        )

    identity = value.get(
        "identity"
    )

    identity_semantics_ok = (
        _identity_semantics_ok(
            identity
        )
    )

    if isinstance(
        identity,
        dict,
    ):
        expected_fingerprint = (
            sha256_json(
                identity
            )
        )
    else:
        expected_fingerprint = ""

    fingerprint_ok = (
        value.get(
            "field_fingerprint_sha256"
        )
        == expected_fingerprint
    )

    expected_field_id = (
        f"{FIELD_ID_PREFIX}-"
        f"{expected_fingerprint[:16]}"
        if expected_fingerprint
        else ""
    )

    field_id_ok = (
        value.get(
            "field_id"
        )
        == expected_field_id
    )

    source_artifacts_ok = (
        _source_bindings_ok(
            value,
            check_artifacts=(
                check_artifacts
            ),
        )
    )

    (
        window_artifacts_ok,
        observed_runtime,
    ) = _window_execution_bindings(
        value,
        check_artifacts=(
            check_artifacts
        ),
    )

    runtime_ok = _runtime_ok(
        value,
        observed=(
            observed_runtime
        ),
    )

    derived_artifacts_ok = (
        _derived_artifacts_ok(
            value,
            check_artifacts=(
                check_artifacts
            ),
        )
    )

    scientific_scope_ok = (
        value.get(
            "scientific_scope"
        )
        == EXPECTED_SCOPE
    )

    counts_ok = _counts_ok(
        value
    )

    measurements_ok = (
        _measurements_ok(
            value
        )
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
            fingerprint_ok,
            field_id_ok,
            source_artifacts_ok,
            window_artifacts_ok,
            runtime_ok,
            derived_artifacts_ok,
            scientific_scope_ok,
            counts_ok,
            measurements_ok,
        )
    )

    counts = value.get(
        "counts"
    )

    if isinstance(counts, dict):
        n_windows = counts.get(
            "n_windows"
        )

        n_fibers_valid = counts.get(
            "n_fibers_valid"
        )
    else:
        n_windows = None
        n_fibers_valid = None

    return {
        "kind": RECORD_TYPE,
        "field_id": value.get(
            "field_id"
        ),
        "schema_ok": schema_ok,
        "status_ok": status_ok,
        "identity_semantics_ok": (
            identity_semantics_ok
        ),
        "fingerprint_ok": (
            fingerprint_ok
        ),
        "field_id_ok": field_id_ok,
        "artifact_check_performed": (
            check_artifacts
        ),
        "source_artifacts_ok": (
            source_artifacts_ok
        ),
        "window_artifacts_ok": (
            window_artifacts_ok
        ),
        "runtime_ok": runtime_ok,
        "derived_artifacts_ok": (
            derived_artifacts_ok
        ),
        "scientific_scope_ok": (
            scientific_scope_ok
        ),
        "counts_ok": counts_ok,
        "measurements_ok": (
            measurements_ok
        ),
        "n_windows": n_windows,
        "n_fibers_valid": (
            n_fibers_valid
        ),
        "ok": ok,
    }
