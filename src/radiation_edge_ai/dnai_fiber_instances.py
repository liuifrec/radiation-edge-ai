"""Read-only, source-verified reconstructed DNAi fiber correspondence.

Runs the frozen DNAi reconstruction in an explicit scientific Python runtime.
The lightweight control-plane CLI spawns that worker; nothing is inferred,
calibrated, optimized or written into the source evidence directories.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from radiation_edge_ai.control import ControlPlaneError, load_json_object, sha256_file, sha256_json
from radiation_edge_ai.dnai_paired_fields import _mapping, _paired_input, _verified_bytes
from radiation_edge_ai.dna_fiber.matching_sensitivity import (
    greedy_iou_matching,
    matching_summary,
    maximum_cardinality_iou_matching,
)
from radiation_edge_ai.execution import verify_target

MATCH_IOU = 0.5
SCIENCE_WORKER_TIMEOUT_SECONDS = 180


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _matching_report(
    fp_fibers: Sequence[Any],
    kl_fibers: Sequence[Any],
    fp_fiber_ids: Sequence[int],
    kl_fiber_ids: Sequence[int],
    pixel_size_um: float,
) -> dict[str, Any]:
    """Compute objective spatial matching and an exact length-sum identity."""
    if (
        not isinstance(pixel_size_um, (int, float))
        or isinstance(pixel_size_um, bool)
        or not math.isfinite(float(pixel_size_um))
        or pixel_size_um <= 0
    ):
        raise ControlPlaneError("Invalid shared positive DNAi pixel size")
    if len(fp_fibers) != len(fp_fiber_ids) or len(kl_fibers) != len(kl_fiber_ids):
        raise ControlPlaneError("Reconstructed-fiber and CSV row counts differ")
    for ids in (fp_fiber_ids, kl_fiber_ids):
        if any(type(fid) is not int or fid < 0 for fid in ids) or len(set(ids)) != len(ids):
            raise ControlPlaneError("Frozen CSV Fiber IDs must be distinct nonnegative integers")

    fp_lengths = []
    kl_lengths = []
    for group, destination in ((fp_fibers, fp_lengths), (kl_fibers, kl_lengths)):
        for fiber in group:
            value = float(fiber.length) * pixel_size_um
            if not math.isfinite(value) or value < 0.0:
                raise ControlPlaneError("A reconstructed fiber length is invalid")
            destination.append(value)

    primary = greedy_iou_matching(fp_fibers, kl_fibers, threshold=MATCH_IOU)
    sensitivity = maximum_cardinality_iou_matching(fp_fibers, kl_fibers, threshold=MATCH_IOU)

    def describe(
        matches: Sequence[tuple[int, int, float]], *, detail: bool
    ) -> dict[str, Any]:
        ref_used: set[int] = set()
        hw_used: set[int] = set()
        pairs = []
        length_abs = []
        ratio_abs = []
        overlap = []
        matched_length_delta = 0.0
        for i, j, iou in sorted(matches, key=lambda item: (item[0], item[1])):
            if (
                type(i) is not int
                or type(j) is not int
                or i < 0
                or j < 0
                or i >= len(fp_fibers)
                or j >= len(kl_fibers)
                or i in ref_used
                or j in hw_used
                or not math.isfinite(float(iou))
                or not MATCH_IOU <= float(iou) <= 1.0
            ):
                raise ControlPlaneError("Invalid or non-disjoint DNAi fiber matching")
            ref_used.add(i)
            hw_used.add(j)
            fplen = fp_lengths[i]
            hwlen = kl_lengths[j]
            delta = hwlen - fplen
            matched_length_delta += delta
            length_abs.append(abs(delta))
            overlap.append(float(iou))
            fp_ratio = _finite_or_none(fp_fibers[i].ratio)
            kl_ratio = _finite_or_none(kl_fibers[j].ratio)
            ratio_delta = None if fp_ratio is None or kl_ratio is None else kl_ratio - fp_ratio
            if ratio_delta is not None:
                ratio_abs.append(abs(ratio_delta))
            pairs.append(
                {
                    "fp512_row": i,
                    "kl720_row": j,
                    "fp512_local_fiber_id": fp_fiber_ids[i],
                    "kl720_local_fiber_id": kl_fiber_ids[j],
                    "bbox_iou": float(iou),
                    "fp512_length_um": fplen,
                    "kl720_length_um": hwlen,
                    "delta_length_um": delta,
                    "fp512_ratio": fp_ratio,
                    "kl720_ratio": kl_ratio,
                    "delta_ratio": ratio_delta,
                }
            )
        if len(ref_used) != len(matches) or len(hw_used) != len(matches):
            raise ControlPlaneError("Duplicate one-to-one fiber match")
        unmatched_fp = [i for i in range(len(fp_fibers)) if i not in ref_used]
        unmatched_kl = [j for j in range(len(kl_fibers)) if j not in hw_used]
        fp_total = float(sum(fp_lengths))
        kl_total = float(sum(kl_lengths))
        fp_unmatched = float(sum(fp_lengths[i] for i in unmatched_fp))
        kl_unmatched = float(sum(kl_lengths[j] for j in unmatched_kl))
        delta_total = kl_total - fp_total
        recomposed = matched_length_delta + kl_unmatched - fp_unmatched
        if not math.isclose(delta_total, recomposed, abs_tol=1e-8, rel_tol=1e-12):
            raise ControlPlaneError("Fiber total-length decomposition does not reconcile")
        summary: dict[str, Any] = {
            "summary": matching_summary(len(fp_fibers), len(kl_fibers), matches),
            "mean_matched_bbox_iou": sum(overlap) / len(overlap) if overlap else None,
            "matched_length_mae_um": sum(length_abs) / len(length_abs) if length_abs else None,
            "matched_ratio_mae": sum(ratio_abs) / len(ratio_abs) if ratio_abs else None,
            "n_ratio_comparable": len(ratio_abs),
            "matched_pairs": pairs,
            "length_decomposition_um": {
                "fp512_total": fp_total,
                "kl720_total": kl_total,
                "matched_kl720_minus_fp512": matched_length_delta,
                "unmatched_fp512": fp_unmatched,
                "unmatched_kl720": kl_unmatched,
                "kl720_minus_fp512_total": delta_total,
                "reconciled": True,
            },
        }
        if detail:
            summary["unmatched_fp512"] = [
                {"row": i, "local_fiber_id": fp_fiber_ids[i], "length_um": fp_lengths[i],
                 "ratio": _finite_or_none(fp_fibers[i].ratio)}
                for i in unmatched_fp
            ]
            summary["unmatched_kl720"] = [
                {"row": j, "local_fiber_id": kl_fiber_ids[j], "length_um": kl_lengths[j],
                 "ratio": _finite_or_none(kl_fibers[j].ratio)}
                for j in unmatched_kl
            ]
        return summary

    main_report = describe(primary, detail=True)
    sensitivity_report = describe(sensitivity, detail=False)
    return {
        "matching_definition": {
            "objects": "DNAi refine_segmentation(segmentation).valid_copy()",
            "threshold": MATCH_IOU,
            "metric": "reconstructed fiber bounding-box IoU",
            "primary": "existing greedy_iou_matching; ties ordered by ascending row indexes",
            "sensitivity": "existing maximum_cardinality_iou_matching, then total IoU",
            "fiber_ids_are_local_not_cross_runtime_correspondences": True,
        },
        "primary": main_report,
        "maximum_cardinality_sensitivity": {
            **sensitivity_report,
            "same_cardinality": len(primary) == len(sensitivity),
            "same_pair_indices": {(i, j) for i, j, _ in primary}
            == {(i, j) for i, j, _ in sensitivity},
        },
    }


def _verified_field_pair(
    fp512_path: Path, kl720_path: Path
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    paths = [Path(fp512_path).expanduser().resolve(), Path(kl720_path).expanduser().resolve()]
    if paths[0] == paths[1]:
        raise ControlPlaneError("Two distinct DNAi field records are required")
    fields: list[dict[str, Any]] = []
    source_sha: list[str] = []
    for path, schema in zip(paths, (1, 2)):
        try:
            sha = sha256_file(path)
        except OSError as exc:
            raise ControlPlaneError(f"Unreadable DNAi field record: {path}") from exc
        result = verify_target(path, check_artifacts=True)
        if (
            not isinstance(result, Mapping)
            or result.get("kind") != "dnai_fiber_field_record"
            or result.get("ok") is not True
        ):
            raise ControlPlaneError(f"DNAi field failed full verification: {path}")
        field = load_json_object(path)
        if (
            field.get("record_type") != "dnai_fiber_field_record"
            or type(field.get("schema_version")) is not int
            or field["schema_version"] != schema
        ):
            raise ControlPlaneError(f"Expected DNAi field schema {schema}: {path}")
        if sha256_file(path) != sha:
            raise ControlPlaneError("DNAi source record changed during verification")
        source_sha.append(sha)
        fields.append(field)

    identity_fp = _paired_input(_mapping(fields[0].get("identity"), "FP512 identity"))
    identity_kl = _paired_input(_mapping(fields[1].get("identity"), "KL720 identity"))
    if identity_fp != identity_kl:
        raise ControlPlaneError("DNAi fields do not bind identical frozen input windows and policy")
    if any(
        identity_fp.get(key) in (None, "")
        for key in ("sample_id", "key", "image_index", "validation_manifest_sha256")
    ):
        raise ControlPlaneError("DNAi shared field identity is incomplete")
    return fields, source_sha, identity_fp


def _reconstruct_verified_field(field: Mapping[str, Any], dnafiber_version: str) -> dict[str, Any]:
    """Replay only DNAi postprocessing using bound persisted segmentation bytes."""
    try:
        import cv2
        import numpy as np
        import pandas as pd
        from dnafiber.postprocess import refine_segmentation
    except ImportError as exc:
        raise ControlPlaneError(
            "Scientific Python requires cv2, pandas, numpy, dnafiber.postprocess"
        ) from exc

    runtime = _mapping(field.get("runtime"), "DNAi field scientific runtime")
    expected_version = runtime.get("dnafiber_version")
    if not isinstance(expected_version, str) or expected_version != dnafiber_version:
        raise ControlPlaneError("Frozen DNAi reconstruction version mismatch")
    raw = _verified_bytes(field, "segmentation")
    segmentation = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if (
        segmentation is None
        or segmentation.shape != (1024, 1024)
        or segmentation.dtype != np.uint8
        or np.any(segmentation > 2)
    ):
        raise ControlPlaneError("Invalid saved DNAi three-class segmentation PNG")

    all_fibers = refine_segmentation(segmentation)
    valid_copy = all_fibers.valid_copy()
    valid_list = list(valid_copy)
    counts = _mapping(field.get("counts"), "DNAi field reconstruction counts")
    if len(all_fibers) != counts.get("n_fibers_all") or len(valid_list) != counts.get("n_fibers_valid"):
        raise ControlPlaneError("DNAi reconstructed count differs from frozen field")
    identity = _mapping(field.get("identity"), "DNAi reconstruction identity")
    deployment = _mapping(identity.get("deployment_policy"), "DNAi deployment policy")
    pixel_size = deployment.get("pixel_size_um")
    if type(pixel_size) not in (int, float) or not math.isfinite(pixel_size) or pixel_size <= 0:
        raise ControlPlaneError("DNAi pixel-size contract invalid")
    csv_bytes = _verified_bytes(field, "valid_fibers_csv")
    csv_table = pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8-sig")
    regenerated_table = valid_copy.to_df(pixel_size=float(pixel_size), img_name=identity["key"])
    try:
        pd.testing.assert_frame_equal(
            csv_table.reset_index(drop=True),
            regenerated_table.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-10,
            atol=1e-10,
        )
    except AssertionError as exc:
        raise ControlPlaneError("Frozen DNAi fiber CSV reproduction differs") from exc
    if "Fiber ID" not in csv_table.columns:
        raise ControlPlaneError("Frozen DNAi CSV missing Fiber ID column")
    ids = []
    for v in csv_table["Fiber ID"].tolist():
        if isinstance(v, (bool, str)) or int(v) != v:
            raise ControlPlaneError("Invalid DNAi CSV fiber row ID")
        ids.append(int(v))
    lengths = [float(f.length) * float(pixel_size) for f in valid_list]
    total_length = float(sum(lengths))
    reported = _mapping(field.get("measurements"), "DNAi field measurements")
    if not math.isclose(
        total_length, float(reported["total_valid_length_um"]),
        rel_tol=1e-9, abs_tol=1e-8,
    ):
        raise ControlPlaneError("Reconstructed valid-fiber total length changed")
    return {
        "all_count": len(all_fibers),
        "valid_count": len(valid_list),
        "fiber_ids": ids,
        "fibers": valid_list,
        "pixel_size_um": float(pixel_size),
        "csv_reproduced": True,
    }


def compare_dnai_fiber_instances(fp512_path: Path, kl720_path: Path) -> dict[str, Any]:
    """No neural inference: only replay validated DNAi reconstruction and match valid fibers."""
    paths = [Path(fp512_path).expanduser().resolve(), Path(kl720_path).expanduser().resolve()]
    fields, hashes, input_identity = _verified_field_pair(*paths)
    try:
        version = importlib.metadata.version("dnafiber")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ControlPlaneError("DNAi scientific Python does not contain dnafiber") from exc
    if fields[0]["runtime"].get("dnafiber_version") != fields[1]["runtime"].get("dnafiber_version"):
        raise ControlPlaneError("The two frozen DNAi reconstruction versions differ")
    sources = [_reconstruct_verified_field(field, version) for field in fields]
    if sources[0]["pixel_size_um"] != sources[1]["pixel_size_um"]:
        raise ControlPlaneError("DNAi shared pixel size differs")
    matching = _matching_report(
        sources[0]["fibers"], sources[1]["fibers"],
        sources[0]["fiber_ids"], sources[1]["fiber_ids"],
        sources[0]["pixel_size_um"],
    )
    for path, before in zip(paths, hashes):
        if sha256_file(path) != before:
            raise ControlPlaneError("DNAi source record changed during instance matching")
    return {
        "schema_version": 1,
        "report_type": "dnai_fp512_kl720_valid_fiber_instance_audit",
        "source_fp512": {"path": str(paths[0]), "sha256": hashes[0], "field_id": fields[0]["field_id"]},
        "source_kl720": {"path": str(paths[1]), "sha256": hashes[1], "field_id": fields[1]["field_id"]},
        "matched_input_sha256": sha256_json(input_identity),
        "shared_input": {"sample_id": input_identity["sample_id"], "key": input_identity["key"],
                         "image_index": input_identity["image_index"], "n_windows": 9},
        "scientific_runtime": {"dnafiber_version": version},
        "reconstruction": {
            "fp512": {key: sources[0][key] for key in ("all_count", "valid_count", "csv_reproduced")},
            "kl720": {key: sources[1][key] for key in ("all_count", "valid_count", "csv_reproduced")},
        },
        **matching,
        "scientific_scope": {
            "source_artifacts_verified": True,
            "identical_deployment_inputs": True,
            "frozen_csv_reproduced": True,
            "dnafiber_reconstruction_replayed": True,
            "model_inference_performed": False,
            "human_annotations_read": False,
            "biological_fidelity_evaluated": False,
            "unmatched_means_biologically_absent": False,
            "split_merge_classification_performed": False,
            "ephemeral_diagnostics_only": True,
        },
    }


def run_scientific_fiber_worker(
    fp512_path: Path, kl720_path: Path, *, scientific_python: Path
) -> dict[str, Any]:
    """Call the heavy scientific env without installing DNAi into the CLI venv."""
    science_python = Path(scientific_python).expanduser().resolve()
    if not science_python.is_file():
        raise ControlPlaneError(f"Scientific Python executable missing: {science_python}")
    source_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [
        str(science_python), "-B", "-m", "radiation_edge_ai.dnai_fiber_instances",
        "--fp512", str(Path(fp512_path).expanduser().resolve()),
        "--kl720", str(Path(kl720_path).expanduser().resolve()),
    ]
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, encoding="utf-8",
            timeout=SCIENCE_WORKER_TIMEOUT_SECONDS, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ControlPlaneError(f"DNAi scientific worker could not complete: {exc}") from exc
    if proc.returncode != 0:
        raise ControlPlaneError(
            f"DNAi scientific worker failed (code {proc.returncode}): {proc.stderr[-2000:]}"
        )
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ControlPlaneError("DNAi scientific worker returned invalid JSON") from exc
    if not isinstance(report, dict):
        raise ControlPlaneError("DNAi scientific worker returned incompatible audit report")
    fp_source = report.get("source_fp512")
    kl_source = report.get("source_kl720")
    scope = report.get("scientific_scope")
    if (
        report.get("schema_version") != 1
        or report.get("report_type") != "dnai_fp512_kl720_valid_fiber_instance_audit"
        or not isinstance(scope, dict)
        or scope.get("source_artifacts_verified") is not True
        or not isinstance(fp_source, dict)
        or not isinstance(kl_source, dict)
        or fp_source.get("path") != str(Path(fp512_path).expanduser().resolve())
        or kl_source.get("path") != str(Path(kl720_path).expanduser().resolve())
    ):
        raise ControlPlaneError("DNAi scientific worker returned incompatible audit report")
    return report


def _worker_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only verified DNAi reconstruction worker")
    parser.add_argument("--fp512", type=Path, required=True)
    parser.add_argument("--kl720", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = compare_dnai_fiber_instances(args.fp512, args.kl720)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ControlPlaneError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"DNAi instance matching: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_worker_main())
