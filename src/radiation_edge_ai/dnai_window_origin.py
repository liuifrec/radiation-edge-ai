"""Read-only origin audit for already-frozen FP512/KL720 DNAi window logits.

The purpose is to separate *observed* pre-stitch probability differences from
the deterministic effects of the shared Gaussian combination policy. No ONNX
or KL720 inference, optimization, source writes, or matching retuning occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from radiation_edge_ai import dnai_fiber_instances as instances
from radiation_edge_ai import dnai_paired_fields as paired
from radiation_edge_ai.control import ControlPlaneError, sha256_json
from radiation_edge_ai.dnai_failure_modes import _geometry_and_support

REPORT_TYPE = "dnai_fp512_kl720_frozen_window_origin_audit"
REPLAY_ATOL = 5e-7
MARGIN_ATOL = 2e-6
SEAM_DIAGNOSTIC_PX = 32
WORKER_TIMEOUT_SECONDS = 240
TILE = 512
FULL = 1024


def _softmax_frozen(logits: np.ndarray) -> np.ndarray:
    """Same float64 intermediate / float32 output as both frozen field workers."""
    if (
        logits.shape != (1, 3, TILE, TILE)
        or logits.dtype != np.float32
        or not np.isfinite(logits).all()
    ):
        raise ControlPlaneError("Frozen window logits must be finite float32 [1,3,512,512]")
    work = logits.astype(np.float64, copy=False)
    shifted = work - np.max(work, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return (exps / np.sum(exps, axis=1, keepdims=True)).astype(np.float32)


def _verified_window_output(output: Mapping[str, Any]) -> np.ndarray:
    """Read from verified source bytes, not an inferred or reconstructed path."""
    meta = paired._mapping(output.get("output"), "DNAi frozen window output binding")
    path = meta.get("path")
    digest = meta.get("sha256")
    size = meta.get("size_bytes")
    if (
        not isinstance(path, str) or not path.strip()
        or not isinstance(digest, str) or len(digest) != 64
        or type(size) is not int or size < 0
    ):
        raise ControlPlaneError("DNAi frozen window output has invalid source binding")
    try:
        data = Path(path).expanduser().resolve().read_bytes()
    except (OSError, UnicodeError) as exc:
        raise ControlPlaneError(f"Cannot read DNAi frozen window output: {exc}") from exc
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise ControlPlaneError("Frozen window output bytes differ from bound SHA256/size")
    try:
        array = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ControlPlaneError(f"Cannot decode frozen window .npy: {exc}") from exc
    if not isinstance(array, np.ndarray):
        raise ControlPlaneError("Frozen window output must be one .npy ndarray")
    if (
        array.shape != (1, 3, TILE, TILE)
        or array.dtype != np.float32
        or not np.isfinite(array).all()
    ):
        raise ControlPlaneError("Frozen window output tensor contract changed")
    return array


def _importance(policy: Mapping[str, Any]) -> np.ndarray:
    """Obtain exactly the MONAI Gaussian used by both historical workers."""
    if (
        policy.get("tile") != [1, 3, TILE, TILE]
        or policy.get("stride") != 256
        or policy.get("overlap") != 0.5
        or policy.get("blend") != "gaussian"
        or policy.get("sigma_scale") != 0.125
    ):
        raise ControlPlaneError("Frozen DNAi Gaussian stitching policy changed")
    try:
        import torch
        from monai.data.utils import compute_importance_map
    except ImportError as exc:
        raise ControlPlaneError("Scientific Python requires torch and monai") from exc
    weight = compute_importance_map(
        (TILE, TILE), mode="gaussian", sigma_scale=0.125,
        device="cpu", dtype=torch.float32,
    ).cpu().numpy().astype(np.float32, copy=False)
    weight = np.squeeze(weight)
    if (
        weight.shape != (TILE, TILE)
        or not np.isfinite(weight).all()
        or np.any(weight <= 0)
    ):
        raise ControlPlaneError("Frozen MONAI Gaussian importance map is invalid")
    return weight


def _replay_field_windows(
    field: Mapping[str, Any],
    importance: np.ndarray,
) -> dict[str, Any]:
    """Replay the persisted 9 raw outputs and require frozen stitched agreement."""
    identity = paired._mapping(field.get("identity"), "DNAi field identity")
    expected = paired._mapping(
        {w["window_index"]: w for w in identity["windows"]}, "DNAi frozen windows"
    )
    outputs = field.get("window_outputs")
    if not isinstance(outputs, list) or len(outputs) != 9:
        raise ControlPlaneError("DNAi field requires nine frozen raw window outputs")
    if importance.shape != (TILE, TILE) or np.any(importance <= 0):
        raise ControlPlaneError("Frozen Gaussian importance map changed")
    stitched_num = np.zeros((3, FULL, FULL), dtype=np.float64)
    stitched_den = np.zeros((FULL, FULL), dtype=np.float64)
    entries = []
    seen = set()
    # The historical execution order is ascending window index.
    for record in sorted(outputs, key=lambda item: item["window_index"]):
        idx = record.get("window_index")
        if type(idx) is not int or idx not in range(9) or idx in seen:
            raise ControlPlaneError("Frozen DNAi window indices must be unique 0..8")
        seen.add(idx)
        target = paired._mapping(expected.get(idx), "DNAi frozen input window")
        x, y = record.get("x"), record.get("y")
        if (
            type(x) is not int or type(y) is not int
            or x != target.get("x") or y != target.get("y")
            or x not in (0, 256, 512) or y not in (0, 256, 512)
        ):
            raise ControlPlaneError("Frozen window placement differs from input identity")
        inp = paired._mapping(record.get("input"), "DNAi input artifact binding")
        if (
            inp.get("sha256") != target.get("tensor_sha256")
            or inp.get("size_bytes") != target.get("tensor_size_bytes")
        ):
            raise ControlPlaneError("Frozen window input binding differs from shared tensor")
        logits = _verified_window_output(record)
        p = _softmax_frozen(logits)[0]
        stitched_num[:, y:y + TILE, x:x + TILE] += (
            p.astype(np.float64, copy=False) * importance[None, :, :]
        )
        stitched_den[y:y + TILE, x:x + TILE] += importance
        entries.append({
            "window_index": idx, "x": x, "y": y,
            "output_sha256": record["output"]["sha256"],
            "probabilities": p,
        })
    if len(seen) != 9 or np.any(stitched_den <= 0):
        raise ControlPlaneError("Frozen window coverage is incomplete")
    replay = (stitched_num / stitched_den).astype(np.float32)
    saved = paired._load_probabilities(
        paired._verified_bytes(field, "stitched_probabilities")
    )
    error = float(np.max(np.abs(
        replay.astype(np.float64) - saved.astype(np.float64)
    )))
    if error > REPLAY_ATOL:
        raise ControlPlaneError(
            f"Frozen 9-window Gaussian stitch does not reproduce saved NPZ "
            f"(max_abs_error={error:.9g}; fixed limit={REPLAY_ATOL})"
        )
    return {
        "windows": entries,
        "saved": saved,
        "max_abs_replay_error": error,
        "bitwise_identical_to_saved": bool(np.array_equal(replay, saved)),
    }


def _internal_window_edges(windows: Sequence[Mapping[str, Any]], axis: str) -> list[int]:
    """Internal TILE starts/ends, explicitly excluding whole-image boundaries."""
    if axis not in ("x", "y"):
        raise ControlPlaneError("Unknown DNAi window coordinate axis")
    values: set[int] = set()
    for window in windows:
        start = int(window[axis])
        for edge in (start, start + TILE):
            if 0 < edge < FULL:
                values.add(edge)
    return sorted(values)


def _fixed_support_margin(
    fp_windows: Sequence[Mapping[str, Any]],
    kl_windows: Sequence[Mapping[str, Any]],
    fp_saved: np.ndarray,
    kl_saved: np.ndarray,
    importance: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    class_id: int,
    *,
    seam_x: Sequence[int],
    seam_y: Sequence[int],
) -> dict[str, Any]:
    """Exact weighted class-vs-background delta on the SAME source-fiber pixels.

    Margin delta for any global pixel is a convex average of the per-window
    KL720-minus-FP512 deltas when the two runtimes share stitching weights.
    This is an algebraic decomposition, NOT a causal intervention.
    """
    if class_id not in (1, 2) or ys.ndim != 1 or xs.ndim != 1 or len(ys) != len(xs):
        raise ControlPlaneError("Invalid source-support sampling contract")
    n = len(ys)
    if n == 0 or np.any(ys < 0) or np.any(xs < 0):
        raise ControlPlaneError("Source support for margin decomposition is empty/invalid")
    h, w = fp_saved.shape[-2:]
    if (
        fp_saved.shape != kl_saved.shape or fp_saved.shape[0] != 3
        or np.any(ys >= h) or np.any(xs >= w)
    ):
        raise ControlPlaneError("Frozen stitched probability/support shape mismatch")
    if len(fp_windows) != len(kl_windows) or len(fp_windows) == 0:
        raise ControlPlaneError("No paired frozen windows for margin decomposition")

    denominator = np.zeros(n, dtype=np.float64)
    weighted = np.zeros(n, dtype=np.float64)
    coverage = np.zeros(n, dtype=np.int32)
    positive = np.zeros(n, dtype=np.int32)
    negative = np.zeros(n, dtype=np.int32)
    observations: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    tile_h, tile_w = importance.shape
    for left, right in zip(fp_windows, kl_windows):
        idx = left["window_index"]
        if (
            idx != right["window_index"] or left["x"] != right["x"]
            or left["y"] != right["y"]
        ):
            raise ControlPlaneError("Paired window positions/indices differ")
        x0, y0 = left["x"], left["y"]
        mask = (
            (xs >= x0) & (xs < x0 + tile_w)
            & (ys >= y0) & (ys < y0 + tile_h)
        )
        positions = np.flatnonzero(mask)
        if len(positions) == 0:
            continue
        lx, ly = xs[positions] - x0, ys[positions] - y0
        fp_p = left["probabilities"].astype(np.float64, copy=False)
        kl_p = right["probabilities"].astype(np.float64, copy=False)
        delta = (
            (kl_p[class_id, ly, lx] - kl_p[0, ly, lx])
            - (fp_p[class_id, ly, lx] - fp_p[0, ly, lx])
        )
        weight = importance[ly, lx].astype(np.float64)
        denominator[positions] += weight
        weighted[positions] += weight * delta
        coverage[positions] += 1
        positive[positions] += delta > 0
        negative[positions] += delta < 0
        observations.append((idx, positions, delta, weight))
    if np.any(denominator <= 0):
        raise ControlPlaneError("Fiber support not covered by frozen windows")
    if np.any(coverage > 4):
        raise ControlPlaneError("Unexpected overlap multiplicity of frozen windows")
    predicted = weighted / denominator

    a = fp_saved.astype(np.float64, copy=False)
    b = kl_saved.astype(np.float64, copy=False)
    frozen_delta = (
        b[class_id, ys, xs] - b[0, ys, xs]
        - a[class_id, ys, xs] + a[0, ys, xs]
    )
    residual = float(np.max(np.abs(frozen_delta - predicted)))
    if residual > MARGIN_ATOL:
        raise ControlPlaneError(
            f"Per-window margin decomposition does not reconcile saved stitch "
            f"(residual={residual:.9g}, fixed limit={MARGIN_ATOL})"
        )
    by_window = []
    for idx, positions, deltas, weights in observations:
        by_window.append({
            "window_index": int(idx),
            "n_class_support_pixels_in_window": len(positions),
            "mean_pre_stitch_margin_delta": float(np.mean(deltas, dtype=np.float64)),
            "fraction_pre_stitch_margin_delta_negative": float(np.mean(deltas < 0)),
            "mean_weighted_contribution_over_all_class_support_pixels": float(
                np.sum(weights * deltas / denominator[positions], dtype=np.float64) / n
            ),
        })
    recovered_total = float(sum(
        item["mean_weighted_contribution_over_all_class_support_pixels"]
        for item in by_window
    ))
    if abs(recovered_total - float(np.mean(predicted))) > 1e-12:
        raise ControlPlaneError("Weighted window contribution accounting did not reconcile")

    seam_distances = []
    for edge in seam_x:
        seam_distances.append(np.abs(xs.astype(np.int32) - int(edge)))
    for edge in seam_y:
        seam_distances.append(np.abs(ys.astype(np.int32) - int(edge)))
    seam_distance = np.minimum.reduce(seam_distances) if seam_distances else None
    return {
        "class_id": class_id,
        "source_class_support_pixels": n,
        "n_support_pixels_covered_by_1_window": int(np.count_nonzero(coverage == 1)),
        "n_support_pixels_covered_by_2_windows": int(np.count_nonzero(coverage == 2)),
        "n_support_pixels_covered_by_4_windows": int(np.count_nonzero(coverage == 4)),
        "n_multiwindow_pixels_with_opposite_delta_signs": int(np.count_nonzero(
            (coverage > 1) & (positive > 0) & (negative > 0)
        )),
        "n_support_pixels_within_32px_of_internal_window_edge": (
            int(np.count_nonzero(seam_distance <= SEAM_DIAGNOSTIC_PX))
            if seam_distance is not None else 0
        ),
        "minimum_support_pixel_distance_to_internal_window_edge_px": (
            int(seam_distance.min()) if seam_distance is not None else None
        ),
        "saved_stitched_margin_delta_mean": float(np.mean(frozen_delta, dtype=np.float64)),
        "weighted_pre_stitch_margin_delta_mean": float(np.mean(predicted, dtype=np.float64)),
        "maximum_absolute_margin_decomposition_residual": residual,
        "per_window": by_window,
    }


def _source_fiber_reports(
    reconstructed: Mapping[str, Any],
    *,
    side: str,
    partner_rows: Mapping[int, int],
    fp: Mapping[str, Any],
    kl: Mapping[str, Any],
    importance: np.ndarray,
) -> list[dict[str, Any]]:
    entries = []
    seam_x = _internal_window_edges(fp["windows"], "x")
    seam_y = _internal_window_edges(fp["windows"], "y")
    for row, fiber in enumerate(reconstructed["fibers"]):
        bbox, data = _geometry_and_support(fiber, (FULL, FULL))
        x, y, width, height = bbox
        classes = []
        for class_id in (1, 2):
            local_y, local_x = np.where(data == class_id)
            if not len(local_y):
                continue
            classes.append(_fixed_support_margin(
                fp["windows"], kl["windows"], fp["saved"], kl["saved"],
                importance, local_y + y, local_x + x, class_id,
                seam_x=seam_x, seam_y=seam_y,
            ))
        entries.append({
            "source_side": side,
            "reconstructed_row": row,
            "local_fiber_id": reconstructed["fiber_ids"][row],
            "historical_primary_matched": row in partner_rows,
            "primary_partner_row": partner_rows.get(row),
            "bbox_xywh": bbox,
            "image_edge_distance_px": min(x, y, FULL - x - width, FULL - y - height),
            "per_class": classes,
        })
    return entries


def audit_dnai_window_origin(fp512_path: Path, kl720_path: Path) -> dict[str, Any]:
    """Verify frozen evidence and replay only saved logits, stitching and fiber geometry."""
    paths = [Path(fp512_path).expanduser().resolve(), Path(kl720_path).expanduser().resolve()]
    fields, hashes, identity = instances._verified_field_pair(*paths)
    try:
        version = importlib.metadata.version("dnafiber")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ControlPlaneError("Scientific Python needs frozen dnafiber") from exc
    if fields[0]["runtime"].get("dnafiber_version") != fields[1]["runtime"].get(
        "dnafiber_version"
    ):
        raise ControlPlaneError("The two frozen DNAi versions differ")
    reconstructed = [
        instances._reconstruct_verified_field(field, version) for field in fields
    ]
    if reconstructed[0]["pixel_size_um"] != reconstructed[1]["pixel_size_um"]:
        raise ControlPlaneError("Paired frozen DNAi pixel sizes differ")
    matching = instances._matching_report(
        reconstructed[0]["fibers"], reconstructed[1]["fibers"],
        reconstructed[0]["fiber_ids"], reconstructed[1]["fiber_ids"],
        reconstructed[0]["pixel_size_um"],
    )
    importance = _importance(paired._mapping(
        identity["deployment_policy"], "DNAi frozen Gaussian deployment policy"
    ))
    fp, kl = [_replay_field_windows(field, importance) for field in fields]
    for left, right in zip(fp["windows"], kl["windows"]):
        if (left["window_index"], left["x"], left["y"]) != (
            right["window_index"], right["x"], right["y"]
        ):
            raise ControlPlaneError("Frozen per-window output layouts differ")
    pairs = matching["primary"]["matched_pairs"]
    fp_to_kl = {p["fp512_row"]: p["kl720_row"] for p in pairs}
    kl_to_fp = {p["kl720_row"]: p["fp512_row"] for p in pairs}
    summaries = {}
    for side, rep, partners in (
        ("fp512_source", reconstructed[0], fp_to_kl),
        ("kl720_source", reconstructed[1], kl_to_fp),
    ):
        summaries[side] = _source_fiber_reports(
            rep, side=side, partner_rows=partners,
            fp=fp, kl=kl, importance=importance,
        )

    window_pairs = []
    for left, right in zip(fp["windows"], kl["windows"]):
        p = left["probabilities"]
        q = right["probabilities"]
        window_pairs.append({
            "window_index": left["window_index"],
            "x": left["x"], "y": left["y"],
            "fp512_raw_logits_sha256": left["output_sha256"],
            "kl720_raw_logits_sha256": right["output_sha256"],
            "pre_stitch_probability_mae": float(np.mean(
                np.abs(q.astype(np.float64) - p.astype(np.float64)), dtype=np.float64
            )),
            "pre_stitch_argmax_agreement": float(
                np.mean(np.argmax(p, axis=0) == np.argmax(q, axis=0))
            ),
        })

    # Verify all frozen artifact bytes and field metadata a second time.
    for path, digest in zip(paths, hashes):
        if instances.sha256_file(path) != digest:
            raise ControlPlaneError("Frozen field JSON changed during window-origin audit")
        result = instances.verify_target(path, check_artifacts=True)
        if (
            not isinstance(result, Mapping)
            or result.get("kind") != "dnai_fiber_field_record"
            or result.get("ok") is not True
        ):
            raise ControlPlaneError("Frozen field artifacts changed during window-origin audit")
        if instances.sha256_file(path) != digest:
            raise ControlPlaneError("Frozen field JSON changed during final verification")

    return {
        "schema_version": 1, "report_type": REPORT_TYPE,
        "source_fp512": {
            "path": str(paths[0]), "sha256": hashes[0], "field_id": fields[0]["field_id"],
        },
        "source_kl720": {
            "path": str(paths[1]), "sha256": hashes[1], "field_id": fields[1]["field_id"],
        },
        "matched_input_sha256": sha256_json(identity),
        "shared_input": {
            "sample_id": identity["sample_id"], "key": identity["key"],
            "image_index": identity["image_index"], "n_windows": 9,
        },
        "scientific_runtime": {"dnafiber_version": version},
        "frozen_replay": {
            "fp512_max_absolute_probability_error": fp["max_abs_replay_error"],
            "kl720_max_absolute_probability_error": kl["max_abs_replay_error"],
            "fp512_bitwise_identical": fp["bitwise_identical_to_saved"],
            "kl720_bitwise_identical": kl["bitwise_identical_to_saved"],
            "fixed_max_probability_replay_error_limit": REPLAY_ATOL,
            "fixed_max_margin_residual_limit": MARGIN_ATOL,
            "both_replays_passed": True,
        },
        "historical_instance_matching": matching,
        "window_pair_metrics": window_pairs,
        "internal_window_edges_px": {
            "x": _internal_window_edges(fp["windows"], "x"),
            "y": _internal_window_edges(fp["windows"], "y"),
        },
        "per_fiber": summaries,
        "scientific_scope": {
            "frozen_full_artifact_verification_before_and_after": True,
            "frozen_window_outputs_consumed": True,
            "frozen_csv_reproduced": True,
            "identical_frozen_input_windows": True,
            "same_historical_primary_matching": True,
            "window_stitch_replay_passed": True,
            "model_inference_performed": False,
            "hardware_access_performed": False,
            "model_or_matching_retuned": False,
            "source_data_written": False,
            "biological_fidelity_evaluated": False,
            "window_margin_decomposition_is_causal_attribution": False,
            "stitching_or_quantization_cause_established": False,
            "ephemeral_diagnostics_only": True,
        },
    }


def run_scientific_window_origin_worker(
    fp512_path: Path, kl720_path: Path, *, scientific_python: Path,
) -> dict[str, Any]:
    executable = Path(scientific_python).expanduser().resolve()
    if not executable.is_file():
        raise ControlPlaneError(f"DNAi scientific Python missing: {executable}")
    paths = [Path(fp512_path).expanduser().resolve(), Path(kl720_path).expanduser().resolve()]
    env = os.environ.copy()
    source_root = Path(__file__).resolve().parent.parent
    env["PYTHONPATH"] = str(source_root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        str(executable), "-B", "-m", "radiation_edge_ai.dnai_window_origin",
        "--fp512", str(paths[0]), "--kl720", str(paths[1]),
    ]
    try:
        proc = subprocess.run(
            command, env=env, capture_output=True, text=True,
            encoding="utf-8", timeout=WORKER_TIMEOUT_SECONDS, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ControlPlaneError(f"DNAi window-origin worker failed: {exc}") from exc
    if proc.returncode:
        raise ControlPlaneError(
            f"DNAi window-origin worker exit {proc.returncode}: {proc.stderr[-2000:]}"
        )
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ControlPlaneError("DNAi window-origin worker returned invalid JSON") from exc
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("report_type") != REPORT_TYPE
        or report.get("frozen_replay", {}).get("both_replays_passed") is not True
        or report.get("scientific_scope", {}).get(
            "frozen_full_artifact_verification_before_and_after"
        ) is not True
        or report.get("source_fp512", {}).get("path") != str(paths[0])
        or report.get("source_kl720", {}).get("path") != str(paths[1])
    ):
        raise ControlPlaneError("DNAi window-origin worker returned incompatible report")
    return report


def _worker_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only frozen DNAi window-origin audit")
    parser.add_argument("--fp512", type=Path, required=True)
    parser.add_argument("--kl720", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(
            audit_dnai_window_origin(args.fp512, args.kl720),
            indent=2, ensure_ascii=False, allow_nan=False,
        ))
        return 0
    except (ControlPlaneError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"DNAi window-origin audit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_worker_main())
