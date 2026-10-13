"""Read-only paired FP512/KL720 DNAi field diagnostics.

All source fields and their artifacts are fully verified before comparison.
Connected foreground components are *segmentation proxies*, not DNA fibers.
No inference, image preprocessing, reconstruction, retuning, or writing occurs.
"""

from __future__ import annotations

import hashlib
import io
import math
from zipfile import BadZipFile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from radiation_edge_ai.control import ControlPlaneError, load_json_object, sha256_file, sha256_json
from radiation_edge_ai.execution import verify_target

SEGMENTATION_SHAPE = (1024, 1024)
PROBABILITIES_SHAPE = (3, 1024, 1024)
COMPONENT_IOU_THRESHOLD = 0.5


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ControlPlaneError(f"{label} must be a JSON object")
    return value


def _paired_input(identity: Mapping[str, Any]) -> dict[str, object]:
    """Extract the shared deployment-input identity, excluding the model/backend."""
    windows = identity.get("windows")
    if not isinstance(windows, list) or len(windows) != 9:
        raise ControlPlaneError("DNAi field must describe exactly nine input windows")
    normalized = []
    for window in windows:
        item = _mapping(window, "DNAi deployment window")
        normalized.append(
            {
                "window_index": item.get("window_index"),
                "x": item.get("x"),
                "y": item.get("y"),
                "tensor_sha256": item.get("tensor_sha256"),
                "tensor_size_bytes": item.get("tensor_size_bytes"),
            }
        )
    normalized.sort(key=lambda item: item["window_index"])
    return {
        "sample_id": identity.get("sample_id"),
        "key": identity.get("key"),
        "image_index": identity.get("image_index"),
        "dnai_commit": identity.get("dnai_commit"),
        "validation_manifest_sha256": identity.get("validation_manifest_sha256"),
        "deployment_policy": identity.get("deployment_policy"),
        "windows": normalized,
    }


def _verified_bytes(field: Mapping[str, Any], name: str) -> bytes:
    artifacts = _mapping(field.get("artifacts"), "DNAi field artifacts")
    artifact = _mapping(artifacts.get(name), f"DNAi {name} artifact")
    file_path = artifact.get("path")
    sha = artifact.get("sha256")
    size = artifact.get("size_bytes")
    if (
        not isinstance(file_path, str)
        or not file_path.strip()
        or not isinstance(sha, str)
        or len(sha) != 64
        or type(size) is not int
        or size < 0
    ):
        raise ControlPlaneError(f"DNAi {name} artifact metadata is invalid")
    path = Path(file_path).expanduser().resolve()
    try:
        raw = path.read_bytes()
    except (OSError, UnicodeError) as exc:
        raise ControlPlaneError(f"Cannot read verified DNAi {name} artifact: {exc}") from exc
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != sha:
        raise ControlPlaneError(f"DNAi {name} artifact failed byte verification")
    return raw


def _load_class_image(raw: bytes) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ControlPlaneError("PNG audit needs Pillow; install radiation-edge-ai[analysis]") from exc
    try:
        with Image.open(io.BytesIO(raw)) as img:
            if img.format != "PNG" or img.mode != "L":
                raise ControlPlaneError("Expected an 8-bit grayscale DNAi PNG")
            values = np.asarray(img, dtype=np.uint8).copy()
    except (OSError, ValueError, SyntaxError) as exc:
        raise ControlPlaneError(f"Could not decode DNAi PNG: {exc}") from exc
    if values.shape != SEGMENTATION_SHAPE or np.any(values > 2):
        raise ControlPlaneError("DNAi segmentation shape or 3-class IDs changed")
    return values


def _load_probabilities(raw: bytes) -> np.ndarray:
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != {"probabilities"}:
                raise ControlPlaneError("DNAi probability NPZ keys changed")
            probabilities = archive["probabilities"]
    except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
        raise ControlPlaneError(f"Could not decode DNAi probabilities: {exc}") from exc
    if (
        probabilities.shape != PROBABILITIES_SHAPE
        or probabilities.dtype != np.float32
        or not np.isfinite(probabilities).all()
    ):
        raise ControlPlaneError("DNAi probability array contract changed")
    return probabilities


def _label_components8(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Deterministic run-length/union-find 8-neighbor labels; no SciPy dependency."""
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    parent = [0]
    previous: list[tuple[int, int, int]] = []

    def root(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for y in range(height):
        padded = np.empty(width + 2, dtype=np.int8)
        padded[0] = 0
        padded[-1] = 0
        padded[1:-1] = mask[y]
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        current: list[tuple[int, int, int]] = []
        prior_index = 0
        for start, end in zip(starts, ends):
            a, b = int(start), int(end)
            run_id = len(parent)
            parent.append(run_id)
            labels[y, a:b] = run_id
            while prior_index < len(previous) and previous[prior_index][1] < a:
                prior_index += 1
            for c, d, prior_id in previous[prior_index:]:
                if c > b:
                    break
                x = root(run_id)
                z = root(prior_id)
                if x != z:
                    parent[max(x, z)] = min(x, z)
            current.append((a, b, run_id))
        previous = current

    if len(parent) == 1:
        return labels, 0
    roots = sorted({root(i) for i in range(1, len(parent))})
    canonical = {value: index + 1 for index, value in enumerate(roots)}
    lookup = np.zeros(len(parent), dtype=np.int32)
    for i in range(1, len(parent)):
        lookup[i] = canonical[root(i)]
    return lookup[labels], len(roots)


def _component_proxy_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    reference, n_reference = _label_components8(left != 0)
    candidate, n_candidate = _label_components8(right != 0)
    reference_sizes = np.bincount(reference.ravel(), minlength=n_reference + 1)
    candidate_sizes = np.bincount(candidate.ravel(), minlength=n_candidate + 1)
    both = (reference != 0) & (candidate != 0)
    combined = (reference[both].astype(np.int64) * (n_candidate + 1)
                + candidate[both].astype(np.int64))
    codes, intersections = np.unique(combined, return_counts=True)
    edges = []
    for code, area in zip(codes, intersections):
        i = int(code // (n_candidate + 1))
        j = int(code % (n_candidate + 1))
        denom = reference_sizes[i] + candidate_sizes[j] - int(area)
        iou = float(int(area) / int(denom))
        if iou >= COMPONENT_IOU_THRESHOLD:
            edges.append((iou, i, j))
    edges.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[dict[str, object]] = []
    for iou, i, j in edges:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        matches.append({"reference_component": i, "kl720_component": j, "iou": iou})
    n_match = len(matches)
    precision = n_match / n_candidate if n_candidate else (1.0 if n_reference == 0 else 0.0)
    recall = n_match / n_reference if n_reference else (1.0 if n_candidate == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "definition": "8-connected nonzero-class segmentation components; NOT DNA-fiber identities",
        "matching": "greedy one-to-one mask-IoU >= 0.5, diagnostic only",
        "n_fp512_components": n_reference,
        "n_kl720_components": n_candidate,
        "n_matched_proxies": n_match,
        "n_fp512_unmatched": n_reference - n_match,
        "n_kl720_unmatched": n_candidate - n_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }


def _segmentation_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    labels = (left.astype(np.int32) * 3 + right.astype(np.int32)).ravel()
    confusion = np.bincount(labels, minlength=9).reshape((3, 3))
    a_foreground = left != 0
    b_foreground = right != 0
    intersection = int(np.count_nonzero(a_foreground & b_foreground))
    union = int(np.count_nonzero(a_foreground | b_foreground))
    a_count = int(np.count_nonzero(a_foreground))
    b_count = int(np.count_nonzero(b_foreground))
    per_class = []
    for class_id in range(3):
        overlap = int(confusion[class_id, class_id])
        reference_pixels = int(np.sum(confusion[class_id, :]))
        physical_pixels = int(np.sum(confusion[:, class_id]))
        denom = reference_pixels + physical_pixels
        iou_denom = denom - overlap
        per_class.append({
            "class_id": class_id,
            "fp512_pixels": reference_pixels,
            "kl720_pixels": physical_pixels,
            "dice": (2.0 * overlap / denom if denom else None),
            "iou": (overlap / iou_denom if iou_denom else None),
        })
    return {
        "per_class": per_class,
        "height": SEGMENTATION_SHAPE[0],
        "width": SEGMENTATION_SHAPE[1],
        "pixel_accuracy": float(np.mean(left == right)),
        "changed_pixels": int(np.count_nonzero(left != right)),
        "confusion_reference_rows_kl720_columns": confusion.tolist(),
        "fp512_foreground_pixels": a_count,
        "kl720_foreground_pixels": b_count,
        "foreground_dice": (2.0 * intersection / (a_count + b_count)
                            if a_count + b_count else None),
        "foreground_iou": (intersection / union if union else None),
    }


def _measurements(field: Mapping[str, Any]) -> dict[str, object]:
    counts = _mapping(field.get("counts"), "DNAi field counts")
    values = _mapping(field.get("measurements"), "DNAi field measurements")
    valid = counts.get("n_fibers_valid")
    if type(valid) is not int or valid < 0:
        raise ControlPlaneError("Invalid verified field count")
    result: dict[str, object] = {"n_valid_fibers": valid}
    for name in ("mean_valid_ratio", "median_valid_ratio", "mean_valid_length_um", "total_valid_length_um"):
        value = values.get(name)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ControlPlaneError(f"Invalid verified DNAi measurement: {name}")
        result[name] = float(value) if value is not None else None
    return result


def compare_dnai_fields(fp512_path: Path, kl720_path: Path) -> dict[str, object]:
    """Verify paired sources, compare their frozen pixel arrays, and return JSON data."""
    paths = [Path(fp512_path).expanduser().resolve(), Path(kl720_path).expanduser().resolve()]
    if paths[0] == paths[1]:
        raise ControlPlaneError("Two distinct DNAi field records are required")
    original_hashes: list[str] = []
    fields: list[dict[str, Any]] = []
    for path, expected_schema in zip(paths, (1, 2)):
        try:
            sha = sha256_file(path)
        except OSError as exc:
            raise ControlPlaneError(f"Cannot read DNAi field record: {path}: {exc}") from exc
        report = verify_target(path, check_artifacts=True)
        if not isinstance(report, Mapping) or report.get("kind") != "dnai_fiber_field_record" or report.get("ok") is not True:
            raise ControlPlaneError(f"DNAi source failed full verification: {path}")
        field = load_json_object(path)
        if (field.get("record_type") != "dnai_fiber_field_record"
                or type(field.get("schema_version")) is not int
                or field["schema_version"] != expected_schema):
            raise ControlPlaneError(f"Unexpected DNAi field schema (expected {expected_schema}): {path}")
        if sha256_file(path) != sha:
            raise ControlPlaneError("DNAi field record changed during comparison")
        fields.append(field)
        original_hashes.append(sha)

    input1 = _paired_input(_mapping(fields[0].get("identity"), "FP512 identity"))
    input2 = _paired_input(_mapping(fields[1].get("identity"), "KL720 identity"))
    if input1 != input2:
        raise ControlPlaneError("DNAi fields do not bind identical frozen input windows and policy")
    if any(input1.get(name) in (None, "") for name in ("sample_id", "key", "image_index", "validation_manifest_sha256")):
        raise ControlPlaneError("DNAi shared input identity is incomplete")

    left_seg = _load_class_image(_verified_bytes(fields[0], "segmentation"))
    right_seg = _load_class_image(_verified_bytes(fields[1], "segmentation"))
    left_probs = _load_probabilities(_verified_bytes(fields[0], "stitched_probabilities"))
    right_probs = _load_probabilities(_verified_bytes(fields[1], "stitched_probabilities"))
    left_measure = _measurements(fields[0])
    right_measure = _measurements(fields[1])
    deltas = {}
    for key in left_measure:
        a, b = left_measure[key], right_measure[key]
        deltas[key] = (b - a if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None)

    for path, sha in zip(paths, original_hashes):
        if sha256_file(path) != sha:
            raise ControlPlaneError("DNAi source field changed during paired comparison")

    return {
        "schema_version": 1,
        "report_type": "dnai_fp512_kl720_paired_field_audit",
        "source_fp512": {"path": str(paths[0]), "sha256": original_hashes[0], "field_id": fields[0]["field_id"]},
        "source_kl720": {"path": str(paths[1]), "sha256": original_hashes[1], "field_id": fields[1]["field_id"]},
        "matched_input_sha256": sha256_json(input1),
        "shared_input": {"sample_id": input1["sample_id"], "key": input1["key"], "image_index": input1["image_index"], "n_windows": 9},
        "probability_metrics": {
            "argmax_agreement": float(np.mean(np.argmax(left_probs, axis=0) == np.argmax(right_probs, axis=0))),
            "mean_absolute_probability_difference": float(np.mean(np.abs(left_probs.astype(np.float64) - right_probs.astype(np.float64)))),
        },
        "segmentation_metrics": _segmentation_metrics(left_seg, right_seg),
        "component_proxy_metrics": _component_proxy_metrics(left_seg, right_seg),
        "field_measurements": {"fp512": left_measure, "kl720": right_measure, "delta_kl720_minus_fp512": deltas},
        "scientific_scope": {
            "source_artifacts_verified": True,
            "inputs_identical": True,
            "model_inference_performed": False,
            "biological_fidelity_evaluated": False,
            "components_are_valid_fiber_instances": False,
            "human_annotations_read": False,
            "ephemeral_diagnostics_only": True,
        },
    }
