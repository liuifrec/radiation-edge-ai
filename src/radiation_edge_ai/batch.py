"""Content-addressed batch planning for Radiation Edge AI.

A batch plan binds one approved model and backend to multiple immutable assay
inputs. Planning performs no neural inference, hardware access, endpoint
reconstruction, biological-reference lookup, or biological acceptance testing.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from radiation_edge_ai.control import (
    ControlPlaneError,
    create_run_plan,
    get_assay,
    load_json_object,
    sha256_file,
    sha256_json,
)
from radiation_edge_ai.execution import (
    execute_run_plan,
    verify_run_record,
)

BATCH_MANIFEST_SCHEMA_VERSION = 1
BATCH_PLAN_SCHEMA_VERSION = 1
BATCH_PLAN_TYPE = "batch_plan"

BATCH_RECORD_SCHEMA_VERSION = 1
BATCH_RECORD_TYPE = "batch_record"

NASA_ASSAY_ID = "nasa-53bp1-r1-v2"

# v0.4 intentionally starts with software-side CPU execution only.
BATCH_SUPPORTED_BACKENDS = ("cpu-onnx",)


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


def _finite_float(
    value: object,
    *,
    label: str,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ControlPlaneError(
            f"{label} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise ControlPlaneError(
            f"{label} must be finite"
        )

    return result


def _normalize_nasa_metadata(
    value: object,
) -> dict[str, object]:
    metadata = _require_mapping(
        value,
        label="batch item metadata",
    )

    normalized: dict[str, object] = {
        "sample_id": _required_text(
            metadata.get("sample_id"),
            label="metadata sample_id",
        ),
        "sample_name": _required_text(
            metadata.get("sample_name"),
            label="metadata sample_name",
        ),
        "source_name": _required_text(
            metadata.get("source_name"),
            label="metadata source_name",
        ),
        "radiation_branch": _required_text(
            metadata.get("radiation_branch"),
            label="metadata radiation_branch",
        ),
        "dose_gy": _finite_float(
            metadata.get("dose_gy"),
            label="metadata dose_gy",
        ),
        "timepoint_hr": _finite_float(
            metadata.get("timepoint_hr"),
            label="metadata timepoint_hr",
        ),
    }

    # Optional descriptors are useful downstream but are not mandatory.
    for field in ("strain", "sex"):
        raw = metadata.get(field)
        if raw is not None:
            normalized[field] = _required_text(
                raw,
                label=f"metadata {field}",
            )

    return normalized


def _scientific_scope() -> dict[str, object]:
    return {
        "input_semantics": (
            "multiple preprocessed single-nucleus tensors for "
            "NASA R1 v2 latent 53BP1 burden inference"
        ),
        "intended_prediction_semantics": (
            "per-nucleus latent continuous 53BP1 burden"
        ),
        "per_nucleus_focus_count_interpretation": False,
        "neural_inference_performed": False,
        "prediction_table_created": False,
        "endpoint_reconstruction_performed": False,
        "biological_reference_read": False,
        "biological_acceptance_evaluated": False,
        "hardware_access_performed": False,
    }


def _normalized_manifest(
    manifest_path: Path,
) -> dict[str, object]:
    source = _require_file(
        manifest_path,
        label="batch manifest",
    )
    manifest = load_json_object(source)
    manifest_dir = source.parent

    if (
        manifest.get("schema_version")
        != BATCH_MANIFEST_SCHEMA_VERSION
    ):
        raise ControlPlaneError(
            "Unsupported batch-manifest schema_version"
        )

    assay_id = _required_text(
        manifest.get("assay_id"),
        label="batch assay_id",
    )
    if assay_id != NASA_ASSAY_ID:
        raise ControlPlaneError(
            "v0.4 batch planning is currently implemented only for "
            f"{NASA_ASSAY_ID!r}; got {assay_id!r}"
        )

    spec = get_assay(assay_id)

    backend = _required_text(
        manifest.get("backend"),
        label="batch backend",
    )

    if backend not in spec.approved_backends:
        raise ControlPlaneError(
            f"Backend {backend!r} is not approved for assay {assay_id!r}"
        )

    if backend not in BATCH_SUPPORTED_BACKENDS:
        supported = ", ".join(BATCH_SUPPORTED_BACKENDS)
        raise ControlPlaneError(
            "v0.4 batch execution is intentionally limited to "
            f"{supported}; got {backend!r}"
        )

    model_value = _required_text(
        manifest.get("model"),
        label="batch model",
    )
    model_reference = Path(model_value).expanduser()
    if not model_reference.is_absolute():
        model_reference = manifest_dir / model_reference

    model_path = _require_file(
        model_reference,
        label="batch model",
    )

    if model_path.suffix.lower() != ".onnx":
        raise ControlPlaneError(
            "cpu-onnx batch model must have .onnx suffix"
        )

    raw_items = manifest.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ControlPlaneError(
            "Batch manifest items must be a non-empty list"
        )

    model_record = _artifact_record(model_path)

    normalized_items: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for index, raw_item in enumerate(raw_items):
        item = _require_mapping(
            raw_item,
            label=f"batch item {index}",
        )

        input_value = _required_text(
            item.get("input"),
            label=f"batch item {index} input",
        )
        input_reference = Path(input_value).expanduser()
        if not input_reference.is_absolute():
            input_reference = manifest_dir / input_reference

        input_path = _require_file(
            input_reference,
            label=f"batch item {index} input",
        )

        if input_path.suffix.lower() != ".npy":
            raise ControlPlaneError(
                "cpu-onnx batch inputs must be preprocessed .npy tensors; "
                f"got {input_path.name!r}"
            )

        metadata = _normalize_nasa_metadata(
            item.get("metadata")
        )

        item_id = str(metadata["sample_id"])
        if item_id in seen_ids:
            raise ControlPlaneError(
                f"Duplicate batch sample_id: {item_id!r}"
            )
        seen_ids.add(item_id)

        normalized_items.append(
            {
                "item_id": item_id,
                "input": _artifact_record(input_path),
                "metadata": metadata,
            }
        )

    # Scientific identity is independent of manifest item order.
    normalized_items.sort(
        key=lambda item: str(item["item_id"])
    )

    identity_items = []
    for item in normalized_items:
        input_record = _require_mapping(
            item["input"],
            label="normalized input artifact",
        )
        identity_items.append(
            {
                "item_id": item["item_id"],
                "input_sha256": input_record["sha256"],
                "input_size_bytes": input_record["size_bytes"],
                "metadata": item["metadata"],
            }
        )

    identity: dict[str, Any] = {
        "schema_version": BATCH_PLAN_SCHEMA_VERSION,
        "record_type": BATCH_PLAN_TYPE,
        "assay_id": assay_id,
        "backend": backend,
        "model_sha256": model_record["sha256"],
        "model_size_bytes": model_record["size_bytes"],
        "n_items": len(normalized_items),
        "items": identity_items,
        "scientific_scope": _scientific_scope(),
    }

    return {
        "source_manifest": _artifact_record(source),
        "assay_id": assay_id,
        "backend": backend,
        "model": model_record,
        "items": normalized_items,
        "identity": identity,
    }


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
    if not path.is_file():
        return False

    return bool(
        sha256_file(path) == expected_sha
        and path.stat().st_size == expected_size
    )


def create_batch_plan(
    *,
    manifest_path: Path,
    output_dir: Path,
) -> Path:
    """Create an immutable content-addressed batch plan."""

    normalized = _normalized_manifest(manifest_path)

    identity = _require_mapping(
        normalized["identity"],
        label="batch identity",
    )
    fingerprint = sha256_json(identity)
    batch_id = f"b1-{fingerprint[:16]}"

    root = output_dir.expanduser().resolve()
    plan_dir = root / batch_id
    plan_path = plan_dir / "batch_plan.json"

    if plan_path.exists():
        verification = verify_batch_plan(plan_path)
        if verification["ok"]:
            return plan_path
        raise ControlPlaneError(
            f"Existing batch plan failed verification: {plan_path}"
        )

    if plan_dir.exists():
        raise ControlPlaneError(
            "Content-addressed batch directory already exists without a "
            f"verified plan: {plan_dir}"
        )

    plan_dir.mkdir(parents=True, exist_ok=False)

    assay_id = str(normalized["assay_id"])
    spec = get_assay(assay_id)

    plan = {
        "schema_version": BATCH_PLAN_SCHEMA_VERSION,
        "record_type": BATCH_PLAN_TYPE,
        "batch_id": batch_id,
        "status": "planned",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "batch_fingerprint_sha256": fingerprint,
        "identity": dict(identity),
        "assay": {
            "assay_id": spec.assay_id,
            "title": spec.title,
            "endpoint": spec.endpoint,
            "interpretation_guard": spec.interpretation_guard,
        },
        "backend": normalized["backend"],
        "source_manifest": normalized["source_manifest"],
        "model": normalized["model"],
        "items": normalized["items"],
        "counts": {
            "n_items": len(normalized["items"]),
        },
        "scientific_scope": _scientific_scope(),
    }

    temp_path = plan_dir / "batch_plan.json.tmp"
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


def verify_batch_plan(
    plan_path: Path,
    *,
    check_artifacts: bool = True,
) -> dict[str, object]:
    """Verify a content-addressed v0.4 batch plan."""

    path = plan_path.expanduser().resolve()
    plan = load_json_object(path)

    if (
        plan.get("schema_version")
        != BATCH_PLAN_SCHEMA_VERSION
        or plan.get("record_type") != BATCH_PLAN_TYPE
    ):
        raise ControlPlaneError(
            f"Not a supported batch plan: {path}"
        )

    identity = plan.get("identity")
    model = plan.get("model")
    items = plan.get("items")
    counts = plan.get("counts")
    scope = plan.get("scientific_scope")
    source_manifest = plan.get("source_manifest")

    if not isinstance(identity, dict):
        raise ControlPlaneError(
            "Batch-plan identity is malformed"
        )
    if not isinstance(model, dict):
        raise ControlPlaneError(
            "Batch-plan model artifact is malformed"
        )
    if not isinstance(items, list):
        raise ControlPlaneError(
            "Batch-plan items are malformed"
        )
    if not isinstance(counts, dict):
        raise ControlPlaneError(
            "Batch-plan counts are malformed"
        )
    if not isinstance(scope, dict):
        raise ControlPlaneError(
            "Batch-plan scientific scope is malformed"
        )
    if not isinstance(source_manifest, dict):
        raise ControlPlaneError(
            "Batch-plan source manifest is malformed"
        )

    fingerprint = plan.get("batch_fingerprint_sha256")
    batch_id = plan.get("batch_id")

    fingerprint_ok = bool(
        isinstance(fingerprint, str)
        and sha256_json(identity) == fingerprint
    )

    batch_id_ok = bool(
        isinstance(fingerprint, str)
        and isinstance(batch_id, str)
        and batch_id == f"b1-{fingerprint[:16]}"
        and path.parent.name == batch_id
    )

    projected_items: list[dict[str, object]] = []
    item_structure_ok = True

    for item in items:
        if not isinstance(item, dict):
            item_structure_ok = False
            break

        input_record = item.get("input")
        metadata = item.get("metadata")
        item_id = item.get("item_id")

        if (
            not isinstance(input_record, dict)
            or not isinstance(metadata, dict)
            or not isinstance(item_id, str)
        ):
            item_structure_ok = False
            break

        projected_items.append(
            {
                "item_id": item_id,
                "input_sha256": input_record.get("sha256"),
                "input_size_bytes": input_record.get(
                    "size_bytes"
                ),
                "metadata": metadata,
            }
        )

    projected_identity = {
        "schema_version": BATCH_PLAN_SCHEMA_VERSION,
        "record_type": BATCH_PLAN_TYPE,
        "assay_id": NASA_ASSAY_ID,
        "backend": plan.get("backend"),
        "model_sha256": model.get("sha256"),
        "model_size_bytes": model.get("size_bytes"),
        "n_items": len(items),
        "items": projected_items,
        "scientific_scope": scope,
    }

    identity_bindings_ok = bool(
        item_structure_ok
        and identity == projected_identity
        and counts.get("n_items") == len(items)
        and plan.get("assay", {}).get("assay_id")
        == NASA_ASSAY_ID
    )

    scope_ok = scope == _scientific_scope()
    timestamps_ok = _timestamp_is_aware(
        plan.get("created_utc")
    )

    source_manifest_ok = True
    model_ok = True
    items_ok = True
    manifest_semantics_ok = True

    if check_artifacts:
        source_manifest_ok = _verify_artifact(
            source_manifest
        )
        model_ok = _verify_artifact(model)

        for item in items:
            if (
                not isinstance(item, dict)
                or not _verify_artifact(item.get("input"))
            ):
                items_ok = False
                break

        if source_manifest_ok:
            source_path = source_manifest.get("path")
            if isinstance(source_path, str):
                try:
                    regenerated = _normalized_manifest(
                        Path(source_path)
                    )
                    regenerated_identity = regenerated[
                        "identity"
                    ]
                    manifest_semantics_ok = (
                        regenerated_identity == identity
                    )
                except ControlPlaneError:
                    manifest_semantics_ok = False
            else:
                manifest_semantics_ok = False
        else:
            manifest_semantics_ok = False

    ok = bool(
        plan.get("status") == "planned"
        and fingerprint_ok
        and batch_id_ok
        and identity_bindings_ok
        and scope_ok
        and timestamps_ok
        and source_manifest_ok
        and model_ok
        and items_ok
        and manifest_semantics_ok
    )

    return {
        "kind": BATCH_PLAN_TYPE,
        "batch_id": batch_id,
        "plan_path": str(path),
        "fingerprint_ok": fingerprint_ok,
        "batch_id_ok": batch_id_ok,
        "identity_bindings_ok": identity_bindings_ok,
        "scientific_scope_ok": scope_ok,
        "timestamps_ok": timestamps_ok,
        "artifact_check_performed": check_artifacts,
        "source_manifest_ok": source_manifest_ok,
        "model_ok": model_ok,
        "items_ok": items_ok,
        "manifest_semantics_ok": manifest_semantics_ok,
        "n_items": counts.get("n_items"),
        "ok": ok,
    }



def _executed_scientific_scope() -> dict[str, object]:
    return {
        "input_semantics": (
            "multiple preprocessed single-nucleus tensors for "
            "NASA R1 v2 latent 53BP1 burden inference"
        ),
        "prediction_semantics": (
            "per-nucleus latent continuous 53BP1 burden"
        ),
        "per_nucleus_focus_count_interpretation": False,
        "neural_inference_performed": True,
        "prediction_table_created": False,
        "endpoint_reconstruction_performed": False,
        "biological_reference_read": False,
        "biological_acceptance_evaluated": False,
        "hardware_access_performed": False,
    }


def _canonical_metadata_bytes(
    metadata: Mapping[str, Any],
) -> bytes:
    return (
        json.dumps(
            dict(metadata),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _materialize_metadata(
    path: Path,
    metadata: Mapping[str, Any],
) -> dict[str, object]:
    expected = _canonical_metadata_bytes(metadata)

    if path.exists():
        if not path.is_file():
            raise ControlPlaneError(
                f"Batch item metadata path is not a file: {path}"
            )
        if path.read_bytes() != expected:
            raise ControlPlaneError(
                "Existing batch item metadata does not match the "
                f"frozen batch plan: {path}"
            )
        return _artifact_record(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_suffix(path.suffix + ".tmp")
    if temp.exists():
        raise ControlPlaneError(
            f"Refusing orphan temporary metadata file: {temp}"
        )

    temp.write_bytes(expected)
    temp.replace(path)
    return _artifact_record(path)


def _batch_record_identity(
    *,
    batch_id: str,
    batch_plan_sha256: str,
    batch_fingerprint_sha256: str,
    backend: str,
    runs: list[dict[str, object]],
) -> dict[str, Any]:
    projected_runs = []

    for run in runs:
        projected_runs.append(
            {
                "item_id": run["item_id"],
                "run_id": run["run_id"],
                "run_plan_sha256": run["run_plan_sha256"],
                "run_record_sha256": run["run_record_sha256"],
                "raw_output_sha256": run["raw_output_sha256"],
            }
        )

    return {
        "schema_version": BATCH_RECORD_SCHEMA_VERSION,
        "record_type": BATCH_RECORD_TYPE,
        "assay_id": NASA_ASSAY_ID,
        "batch_id": batch_id,
        "batch_plan_sha256": batch_plan_sha256,
        "batch_fingerprint_sha256": batch_fingerprint_sha256,
        "backend": backend,
        "n_items": len(projected_runs),
        "runs": projected_runs,
        "scientific_scope": _executed_scientific_scope(),
    }


def execute_batch_plan(
    plan_path: Path,
    *,
    onnx_python: Optional[Path] = None,  # noqa: UP045
) -> Path:
    """Execute a verified CPU-ONNX batch through ordinary run primitives."""

    plan_file = plan_path.expanduser().resolve()

    verification = verify_batch_plan(
        plan_file,
        check_artifacts=True,
    )
    if not verification["ok"]:
        raise ControlPlaneError(
            f"Batch plan failed verification: {plan_file}"
        )

    plan = load_json_object(plan_file)

    if plan.get("backend") != "cpu-onnx":
        raise ControlPlaneError(
            "v0.4 batch execution supports only cpu-onnx"
        )

    batch_id = plan.get("batch_id")
    fingerprint = plan.get("batch_fingerprint_sha256")
    model = plan.get("model")
    items = plan.get("items")

    if not isinstance(batch_id, str):
        raise ControlPlaneError("Batch plan has invalid batch_id")
    if not isinstance(fingerprint, str):
        raise ControlPlaneError(
            "Batch plan has invalid fingerprint"
        )
    if not isinstance(model, dict):
        raise ControlPlaneError(
            "Batch plan model artifact is malformed"
        )
    if not isinstance(items, list):
        raise ControlPlaneError(
            "Batch plan items are malformed"
        )

    model_path_value = model.get("path")
    if not isinstance(model_path_value, str):
        raise ControlPlaneError(
            "Batch plan model path is malformed"
        )
    model_path = Path(model_path_value)

    record_path = plan_file.parent / "batch_record.json"

    if record_path.exists():
        existing = verify_batch_record(
            record_path,
            check_artifacts=True,
        )
        if existing["ok"]:
            return record_path
        raise ControlPlaneError(
            f"Existing batch record failed verification: {record_path}"
        )

    metadata_root = plan_file.parent / "item_metadata"
    runs_root = plan_file.parent / "runs"

    started_utc = datetime.now(timezone.utc)

    run_bindings: list[dict[str, object]] = []
    run_records: list[dict[str, object]] = []

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ControlPlaneError(
                f"Batch item {index} is malformed"
            )

        item_id = item.get("item_id")
        input_record = item.get("input")
        metadata = item.get("metadata")

        if not isinstance(item_id, str):
            raise ControlPlaneError(
                f"Batch item {index} has invalid item_id"
            )
        if not isinstance(input_record, dict):
            raise ControlPlaneError(
                f"Batch item {item_id!r} input is malformed"
            )
        if not isinstance(metadata, dict):
            raise ControlPlaneError(
                f"Batch item {item_id!r} metadata is malformed"
            )

        input_path_value = input_record.get("path")
        if not isinstance(input_path_value, str):
            raise ControlPlaneError(
                f"Batch item {item_id!r} input path is malformed"
            )

        metadata_path = (
            metadata_root / f"item-{index:06d}.json"
        )
        metadata_artifact = _materialize_metadata(
            metadata_path,
            metadata,
        )

        run_plan_path = create_run_plan(
            assay_id=NASA_ASSAY_ID,
            backend="cpu-onnx",
            input_path=Path(input_path_value),
            model_path=model_path,
            metadata_path=metadata_path,
            output_dir=runs_root,
        )

        run_record_path = execute_run_plan(
            run_plan_path,
            onnx_python=onnx_python,
        )

        run_verification = verify_run_record(
            run_record_path,
            check_source_artifacts=True,
        )
        if not run_verification["ok"]:
            raise ControlPlaneError(
                "Completed child run failed verification: "
                f"{run_record_path}"
            )

        run_plan = load_json_object(run_plan_path)
        run_record = load_json_object(run_record_path)

        run_artifacts = run_plan.get("artifacts")
        source_artifacts = run_record.get(
            "source_artifacts"
        )
        raw_output = run_record.get("raw_output")

        if not isinstance(run_artifacts, dict):
            raise ControlPlaneError(
                f"Child run plan is malformed for {item_id!r}"
            )
        if not isinstance(source_artifacts, dict):
            raise ControlPlaneError(
                f"Child run record is malformed for {item_id!r}"
            )
        if not isinstance(raw_output, dict):
            raise ControlPlaneError(
                f"Child raw output is malformed for {item_id!r}"
            )

        child_input = run_artifacts.get("input")
        child_model = run_artifacts.get("model")

        if (
            not isinstance(child_input, dict)
            or not isinstance(child_model, dict)
        ):
            raise ControlPlaneError(
                f"Child run artifacts are malformed for {item_id!r}"
            )

        if (
            child_input.get("sha256")
            != input_record.get("sha256")
        ):
            raise ControlPlaneError(
                f"Child input binding mismatch for {item_id!r}"
            )

        if child_model.get("sha256") != model.get("sha256"):
            raise ControlPlaneError(
                f"Child model binding mismatch for {item_id!r}"
            )

        if run_plan.get("metadata") != metadata:
            raise ControlPlaneError(
                f"Child metadata binding mismatch for {item_id!r}"
            )

        if run_record.get("backend") != "cpu-onnx":
            raise ControlPlaneError(
                f"Unexpected child backend for {item_id!r}"
            )

        if run_record.get("assay_id") != NASA_ASSAY_ID:
            raise ControlPlaneError(
                f"Unexpected child assay for {item_id!r}"
            )

        run_id = run_record.get("run_id")
        raw_output_sha = raw_output.get("sha256")

        if not isinstance(run_id, str):
            raise ControlPlaneError(
                f"Invalid child run_id for {item_id!r}"
            )
        if not isinstance(raw_output_sha, str):
            raise ControlPlaneError(
                f"Invalid child output SHA for {item_id!r}"
            )

        plan_artifact = _artifact_record(run_plan_path)
        record_artifact = _artifact_record(run_record_path)

        run_bindings.append(
            {
                "item_id": item_id,
                "run_id": run_id,
                "run_plan_sha256": plan_artifact["sha256"],
                "run_record_sha256": record_artifact["sha256"],
                "raw_output_sha256": raw_output_sha,
            }
        )

        run_records.append(
            {
                "item_id": item_id,
                "metadata": metadata_artifact,
                "run_id": run_id,
                "run_plan": {
                    **plan_artifact,
                    "plan_fingerprint_sha256": run_plan.get(
                        "plan_fingerprint_sha256"
                    ),
                },
                "run_record": record_artifact,
                "raw_output": dict(raw_output),
            }
        )

    plan_sha = sha256_file(plan_file)

    identity = _batch_record_identity(
        batch_id=batch_id,
        batch_plan_sha256=plan_sha,
        batch_fingerprint_sha256=fingerprint,
        backend="cpu-onnx",
        runs=run_bindings,
    )

    record_fingerprint = sha256_json(identity)
    ended_utc = datetime.now(timezone.utc)

    record = {
        "schema_version": BATCH_RECORD_SCHEMA_VERSION,
        "record_type": BATCH_RECORD_TYPE,
        "batch_id": batch_id,
        "status": "complete",
        "started_utc": started_utc.isoformat(),
        "ended_utc": ended_utc.isoformat(),
        "completed_utc": ended_utc.isoformat(),
        "batch_record_fingerprint_sha256": record_fingerprint,
        "identity": identity,
        "batch_plan": {
            "path": str(plan_file),
            "sha256": plan_sha,
            "size_bytes": plan_file.stat().st_size,
            "batch_fingerprint_sha256": fingerprint,
        },
        "backend": "cpu-onnx",
        "runs": run_records,
        "counts": {
            "n_items": len(run_records),
            "n_completed_runs": len(run_records),
        },
        "scientific_scope": _executed_scientific_scope(),
    }

    temp_path = plan_file.parent / "batch_record.json.tmp"

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


def verify_batch_record(
    record_path: Path,
    *,
    check_artifacts: bool = True,
) -> dict[str, object]:
    """Verify a completed v0.4 batch execution record."""

    path = record_path.expanduser().resolve()
    record = load_json_object(path)

    if (
        record.get("schema_version")
        != BATCH_RECORD_SCHEMA_VERSION
        or record.get("record_type") != BATCH_RECORD_TYPE
    ):
        raise ControlPlaneError(
            f"Not a supported batch record: {path}"
        )

    identity = record.get("identity")
    plan_record = record.get("batch_plan")
    runs = record.get("runs")
    counts = record.get("counts")
    scope = record.get("scientific_scope")

    if not isinstance(identity, dict):
        raise ControlPlaneError(
            "Batch-record identity is malformed"
        )
    if not isinstance(plan_record, dict):
        raise ControlPlaneError(
            "Batch-record plan binding is malformed"
        )
    if not isinstance(runs, list):
        raise ControlPlaneError(
            "Batch-record runs are malformed"
        )
    if not isinstance(counts, dict):
        raise ControlPlaneError(
            "Batch-record counts are malformed"
        )
    if not isinstance(scope, dict):
        raise ControlPlaneError(
            "Batch-record scientific scope is malformed"
        )

    fingerprint = record.get(
        "batch_record_fingerprint_sha256"
    )

    fingerprint_ok = bool(
        isinstance(fingerprint, str)
        and sha256_json(identity) == fingerprint
    )

    plan_ok = False
    plan: Optional[dict[str, Any]] = None  # noqa: UP045

    plan_path_value = plan_record.get("path")
    if isinstance(plan_path_value, str):
        plan_path = Path(plan_path_value)

        if (
            plan_path.is_file()
            and sha256_file(plan_path)
            == plan_record.get("sha256")
            and plan_path.stat().st_size
            == plan_record.get("size_bytes")
        ):
            plan_verification = verify_batch_plan(
                plan_path,
                check_artifacts=check_artifacts,
            )
            plan_ok = bool(plan_verification["ok"])

            if plan_ok:
                plan = load_json_object(plan_path)

    run_bindings: list[dict[str, object]] = []
    runs_ok = True
    cross_bindings_ok = True
    seen_items: set[str] = set()

    plan_items_by_id: dict[str, dict[str, Any]] = {}

    if plan is not None:
        plan_items = plan.get("items")

        if isinstance(plan_items, list):
            for item in plan_items:
                if (
                    isinstance(item, dict)
                    and isinstance(item.get("item_id"), str)
                ):
                    plan_items_by_id[item["item_id"]] = item

    for run in runs:
        if not isinstance(run, dict):
            runs_ok = False
            cross_bindings_ok = False
            break

        item_id = run.get("item_id")
        run_id = run.get("run_id")
        metadata_record = run.get("metadata")
        run_plan_record = run.get("run_plan")
        run_record_record = run.get("run_record")
        raw_output_record = run.get("raw_output")

        if (
            not isinstance(item_id, str)
            or not isinstance(run_id, str)
            or not isinstance(metadata_record, dict)
            or not isinstance(run_plan_record, dict)
            or not isinstance(run_record_record, dict)
            or not isinstance(raw_output_record, dict)
        ):
            runs_ok = False
            cross_bindings_ok = False
            break

        if item_id in seen_items:
            runs_ok = False
            cross_bindings_ok = False
            break
        seen_items.add(item_id)

        artifacts_ok = bool(
            _verify_artifact(metadata_record)
            and _verify_artifact(run_plan_record)
            and _verify_artifact(run_record_record)
            and _verify_artifact(raw_output_record)
        )

        if not artifacts_ok:
            runs_ok = False
            cross_bindings_ok = False
            break

        child_plan_path = Path(
            str(run_plan_record["path"])
        )
        child_record_path = Path(
            str(run_record_record["path"])
        )

        child_report = verify_run_record(
            child_record_path,
            check_source_artifacts=check_artifacts,
        )

        if not child_report["ok"]:
            runs_ok = False
            cross_bindings_ok = False
            break

        child_plan = load_json_object(child_plan_path)
        child_record = load_json_object(child_record_path)

        expected_item = plan_items_by_id.get(item_id)
        if not isinstance(expected_item, dict):
            cross_bindings_ok = False
            break

        expected_input = expected_item.get("input")
        expected_metadata = expected_item.get("metadata")

        child_artifacts = child_plan.get("artifacts")
        child_sources = child_record.get("source_artifacts")
        child_raw = child_record.get("raw_output")

        if (
            not isinstance(expected_input, dict)
            or not isinstance(expected_metadata, dict)
            or not isinstance(child_artifacts, dict)
            or not isinstance(child_sources, dict)
            or not isinstance(child_raw, dict)
        ):
            cross_bindings_ok = False
            break

        child_input = child_artifacts.get("input")
        child_model = child_artifacts.get("model")

        if (
            not isinstance(child_input, dict)
            or not isinstance(child_model, dict)
        ):
            cross_bindings_ok = False
            break

        plan_model = (
            plan.get("model")
            if isinstance(plan, dict)
            else None
        )

        if not isinstance(plan_model, dict):
            cross_bindings_ok = False
            break

        if not (
            run_id == child_record.get("run_id")
            == child_plan.get("run_id")
            and child_plan.get("backend") == "cpu-onnx"
            and child_record.get("backend") == "cpu-onnx"
            and child_record.get("assay_id") == NASA_ASSAY_ID
            and child_plan.get("metadata")
            == expected_metadata
            and child_input.get("sha256")
            == expected_input.get("sha256")
            and child_model.get("sha256")
            == plan_model.get("sha256")
            and child_raw.get("sha256")
            == raw_output_record.get("sha256")
        ):
            cross_bindings_ok = False
            break

        run_bindings.append(
            {
                "item_id": item_id,
                "run_id": run_id,
                "run_plan_sha256": run_plan_record.get(
                    "sha256"
                ),
                "run_record_sha256": run_record_record.get(
                    "sha256"
                ),
                "raw_output_sha256": raw_output_record.get(
                    "sha256"
                ),
            }
        )

    batch_id = record.get("batch_id")
    backend = record.get("backend")

    expected_identity = None

    if (
        isinstance(batch_id, str)
        and isinstance(backend, str)
        and isinstance(plan_record.get("sha256"), str)
        and isinstance(
            plan_record.get("batch_fingerprint_sha256"),
            str,
        )
    ):
        expected_identity = _batch_record_identity(
            batch_id=batch_id,
            batch_plan_sha256=plan_record["sha256"],
            batch_fingerprint_sha256=plan_record[
                "batch_fingerprint_sha256"
            ],
            backend=backend,
            runs=run_bindings,
        )

    identity_bindings_ok = bool(
        expected_identity is not None
        and identity == expected_identity
        and plan is not None
        and batch_id == plan.get("batch_id")
        and backend == plan.get("backend")
        and plan_record.get(
            "batch_fingerprint_sha256"
        )
        == plan.get("batch_fingerprint_sha256")
    )

    counts_ok = bool(
        counts.get("n_items") == len(runs)
        and counts.get("n_completed_runs") == len(runs)
        and (
            plan is not None
            and isinstance(plan.get("counts"), dict)
            and plan["counts"].get("n_items") == len(runs)
        )
    )

    scope_ok = scope == _executed_scientific_scope()

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
        and plan_ok
        and runs_ok
        and cross_bindings_ok
        and identity_bindings_ok
        and counts_ok
        and scope_ok
        and timestamps_ok
    )

    return {
        "kind": BATCH_RECORD_TYPE,
        "batch_id": batch_id,
        "record_path": str(path),
        "fingerprint_ok": fingerprint_ok,
        "batch_plan_ok": plan_ok,
        "runs_ok": runs_ok,
        "cross_artifact_bindings_ok": cross_bindings_ok,
        "identity_bindings_ok": identity_bindings_ok,
        "counts_ok": counts_ok,
        "scientific_scope_ok": scope_ok,
        "timestamps_ok": timestamps_ok,
        "artifact_check_performed": check_artifacts,
        "n_items": counts.get("n_items"),
        "n_completed_runs": counts.get("n_completed_runs"),
        "ok": ok,
    }
