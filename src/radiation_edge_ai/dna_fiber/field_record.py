"""Verification for DNAi field application records.

This module is deliberately lightweight. It verifies content-addressed
identity, provenance bindings, scientific-scope declarations, and artifact
hashes. It does not import DNAi, ONNX Runtime, Torch, MONAI, or perform
scientific inference/reconstruction.
"""

from __future__ import annotations

import math
from pathlib import Path

from radiation_edge_ai.control import (
    ControlPlaneError,
    load_json_object,
    sha256_file,
    sha256_json,
)

ASSAY_ID = "dnai-fiber-v3"
DNAI_COMMIT = "fcf20c7d6eb385675ff7d07da4fdf471589ce0cf"

EXPECTED_MODEL_SHA256 = (
    "a901d1b309a9a0e5026febd5070252787e4b110a2a7d2828e16a19364d6094d0"
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
    "floating_fp512_inference_performed": True,
    "gaussian_stitching_performed": True,
    "fiber_object_reconstruction_performed": True,
    "human_annotations_read": False,
    "fp1024_reference_read": False,
    "biological_fidelity_evaluated": False,
    "kl720_hardware_access_performed": False,
}


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str):
        return False

    if len(value) != 64:
        return False

    try:
        int(value, 16)
    except ValueError:
        return False

    return True


def _nonnegative_int(value: object) -> bool:
    return (
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
    sha = value.get("sha256")
    size = value.get("size_bytes")

    return (
        isinstance(path, str)
        and bool(path.strip())
        and _is_sha256(sha)
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

    if not path.is_file():
        return False

    if (
        path.stat().st_size
        != int(value["size_bytes"])
    ):
        return False

    return (
        sha256_file(path)
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
        != 1
    ):
        return False

    if (
        identity.get("record_type")
        != "dnai_fiber_field_record"
    ):
        return False

    if (
        identity.get("assay_id")
        != ASSAY_ID
    ):
        return False

    if (
        identity.get("dnai_commit")
        != DNAI_COMMIT
    ):
        return False

    if (
        identity.get("model_sha256")
        != EXPECTED_MODEL_SHA256
    ):
        return False

    manifest_sha = identity.get(
        "validation_manifest_sha256"
    )

    if not _is_sha256(
        manifest_sha
    ):
        return False

    image_index = identity.get(
        "image_index"
    )

    if not _nonnegative_int(
        image_index
    ):
        return False

    sample_id = identity.get(
        "sample_id"
    )

    key = identity.get(
        "key"
    )

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

    if not isinstance(
        policy,
        dict,
    ):
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
        if not isinstance(
            window,
            dict,
        ):
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
        ):
            return False

        if not _is_sha256(
            window.get(
                "tensor_sha256"
            )
        ):
            return False

        if not _nonnegative_int(
            window.get(
                "tensor_size_bytes"
            )
        ):
            return False

        indices.append(index)
        coords.add(
            (int(y), int(x))
        )

    if sorted(indices) != list(
        range(9)
    ):
        return False

    return coords == EXPECTED_COORDS


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
    manifest = sources.get(
        "validation_manifest"
    )

    if (
        not _artifact_ok(
            model,
            check_artifacts=check_artifacts,
        )
        or not _artifact_ok(
            manifest,
            check_artifacts=check_artifacts,
        )
    ):
        return False

    assert isinstance(model, dict)
    assert isinstance(manifest, dict)

    return (
        model.get("sha256")
        == identity.get(
            "model_sha256"
        )
        and manifest.get("sha256")
        == identity.get(
            "validation_manifest_sha256"
        )
    )


def _window_bindings_ok(
    value: dict[str, object],
    *,
    check_artifacts: bool,
) -> bool:
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
        return False

    identity_windows = identity.get(
        "windows"
    )

    if not isinstance(
        identity_windows,
        list,
    ):
        return False

    expected = {
        int(window["window_index"]): window
        for window in identity_windows
        if isinstance(window, dict)
        and _nonnegative_int(
            window.get(
                "window_index"
            )
        )
    }

    if set(expected) != set(
        range(9)
    ):
        return False

    observed_indices = set()

    for output in outputs:
        if not isinstance(
            output,
            dict,
        ):
            return False

        index = output.get(
            "window_index"
        )

        y = output.get("y")
        x = output.get("x")

        if (
            not _nonnegative_int(index)
            or int(index) not in expected
            or not _nonnegative_int(y)
            or not _nonnegative_int(x)
        ):
            return False

        index_int = int(index)

        if index_int in observed_indices:
            return False

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
            return False

        input_artifact = output.get(
            "input"
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
                output_artifact,
                check_artifacts=check_artifacts,
            )
        ):
            return False

        assert isinstance(
            input_artifact,
            dict,
        )

        if (
            input_artifact.get(
                "sha256"
            )
            != identity_window.get(
                "tensor_sha256"
            )
        ):
            return False

        if (
            input_artifact.get(
                "size_bytes"
            )
            != identity_window.get(
                "tensor_size_bytes"
            )
        ):
            return False

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
            return False

    return observed_indices == set(
        range(9)
    )


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
            check_artifacts=check_artifacts,
        )
        for name in required
    )


def _counts_ok(
    value: dict[str, object],
) -> bool:
    counts = value.get(
        "counts"
    )

    if not isinstance(
        counts,
        dict,
    ):
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

    return (
        _finite_number(
            total_length
        )
        and float(total_length) >= 0.0
    )


def _runtime_ok(
    value: dict[str, object],
) -> bool:
    runtime = value.get(
        "runtime"
    )

    if not isinstance(
        runtime,
        dict,
    ):
        return False

    active = runtime.get(
        "active_providers"
    )

    return (
        runtime.get("provider")
        == "CPUExecutionProvider"
        and isinstance(
            active,
            list,
        )
        and "CPUExecutionProvider"
        in active
    )


def verify_dnai_fiber_field_record(
    record_path: Path,
    *,
    check_artifacts: bool = True,
) -> dict[str, object]:
    """Verify one content-addressed DNAi field record."""

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
        != "dnai_fiber_field_record"
    ):
        raise ControlPlaneError(
            f"Not a DNAi fiber field record: {resolved}"
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
        f"df1-{expected_fingerprint[:16]}"
        if expected_fingerprint
        else ""
    )

    field_id_ok = (
        value.get("field_id")
        == expected_field_id
    )

    source_artifacts_ok = (
        _source_bindings_ok(
            value,
            check_artifacts=check_artifacts,
        )
    )

    window_artifacts_ok = (
        _window_bindings_ok(
            value,
            check_artifacts=check_artifacts,
        )
    )

    derived_artifacts_ok = (
        _derived_artifacts_ok(
            value,
            check_artifacts=check_artifacts,
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

    runtime_ok = _runtime_ok(
        value
    )

    status_ok = (
        value.get("status")
        == "complete"
    )

    schema_ok = (
        value.get(
            "schema_version"
        )
        == 1
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
            derived_artifacts_ok,
            scientific_scope_ok,
            counts_ok,
            measurements_ok,
            runtime_ok,
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
        "kind": (
            "dnai_fiber_field_record"
        ),
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
        "runtime_ok": runtime_ok,
        "n_windows": n_windows,
        "n_fibers_valid": (
            n_fibers_valid
        ),
        "ok": ok,
    }
