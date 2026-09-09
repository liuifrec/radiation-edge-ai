"""Export the frozen NASA BPS R1 v2 FP32 model to ONNX and verify parity.

The deployment graph is fixed NCHW [1, 3, 256, 256] -> scalar burden. Frozen
FITC/DAPI normalization, native-scale center padding, zero third channel, bag
aggregation, and biological evaluation remain outside the graph.

This script does not read the raw NASA phenotype table. It requires the frozen
FP32 deployment-reference JSON and uses the already-completed confirmatory
nucleus/bag outputs as the immutable GPU-FP32 reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from radiation_edge_ai.nasa_bps import evaluate_r1_v2_final_holdout_once as final_eval
from radiation_edge_ai.nasa_bps import train_pilot_v1_r1_mil as v1

FROZEN_REPAIRED_CHECKPOINT_SHA256 = (
    "2b07c67d6e50e7a36e26e0052feebc826b518e6efae863ba312ba9761a4eb3d5"
)
FROZEN_QC1_HOLDOUT_MANIFEST_SHA256 = (
    "f89962f96bec05930c07149717a90bd682016a2778bbfae6b827425159aa0643"
)
EXPECTED_INPUT_SHAPE = (1, 3, 256, 256)
EXPECTED_OUTPUT_ELEMENTS = 1
DEFAULT_OPSET = 11

PT_VS_ORT_LIMITS = {
    "max_abs_error": 1e-4,
    "mean_abs_error": 1e-5,
    "rmse": 1e-5,
}
GPU_REFERENCE_VS_ORT_LIMITS = {
    "max_abs_error": 1e-3,
    "mean_abs_error": 1e-4,
    "rmse": 1e-4,
}
EXPECTED_DIRECTION_COUNTS = {
    "overall": "10/11",
    "4h": "4/4",
    "24+48h": "6/7",
    "by_source": {
        "BALBCF2": "4/5",
        "C57BLF2": "3/3",
        "C57BLF3": "3/3",
    },
}
EXPECTED_PEAK = "3/3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return obj


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    final_eval.write_csv(path, rows)


def error_stats(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if ref.shape != cand.shape or ref.size == 0:
        raise RuntimeError(f"Cannot compare arrays with shapes {ref.shape} and {cand.shape}")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(cand)):
        raise RuntimeError("Non-finite value in parity comparison")
    diff = cand - ref
    abs_diff = np.abs(diff)
    return {
        "n": int(ref.size),
        "max_abs_error": float(abs_diff.max()),
        "mean_abs_error": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "pearson": float(np.corrcoef(ref, cand)[0, 1]) if ref.size > 1 else float("nan"),
    }


def within_limits(stats: dict[str, float], limits: dict[str, float]) -> bool:
    return all(float(stats[name]) <= float(limit) for name, limit in limits.items())


def graph_summary(model_proto: onnx.ModelProto) -> dict[str, Any]:
    ops = Counter(node.op_type for node in model_proto.graph.node)

    def dims(value_info: Any) -> list[int | str | None]:
        result: list[int | str | None] = []
        tensor_type = value_info.type.tensor_type
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                result.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                result.append(str(dim.dim_param))
            else:
                result.append(None)
        return result

    return {
        "ir_version": int(model_proto.ir_version),
        "opset_imports": {
            item.domain or "ai.onnx": int(item.version)
            for item in model_proto.opset_import
        },
        "n_nodes": len(model_proto.graph.node),
        "n_initializers": len(model_proto.graph.initializer),
        "operator_counts": dict(sorted(ops.items())),
        "inputs": [
            {"name": value.name, "shape": dims(value)}
            for value in model_proto.graph.input
        ],
        "outputs": [
            {"name": value.name, "shape": dims(value)}
            for value in model_proto.graph.output
        ],
    }


def verify_reference_freeze(
    freeze_path: Path,
    confirmation_dir: Path,
    checkpoint_sha: str,
    manifest_sha: str,
) -> dict[str, Any]:
    if not freeze_path.is_file():
        raise RuntimeError(
            f"FP32 deployment reference is not frozen yet: {freeze_path}. "
            "Run freeze_r1_v2_fp32_deployment_reference first."
        )
    freeze = read_json(freeze_path)
    if freeze.get("status") != "FROZEN_FP32_DEPLOYMENT_REFERENCE_AFTER_CONFIRMATORY_PASS":
        raise RuntimeError(f"Unexpected deployment-freeze status: {freeze.get('status')!r}")
    if freeze.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("Deployment freeze checkpoint identity mismatch")
    if freeze.get("qc1_holdout_manifest_sha256") != manifest_sha:
        raise RuntimeError("Deployment freeze manifest identity mismatch")
    biological = freeze.get("biological_reference", {})
    if biological.get("all_seven_original_gates_pass") is not True:
        raise RuntimeError("FP32 deployment freeze does not authorize ONNX parity")
    if biological.get("direction_counts") != EXPECTED_DIRECTION_COUNTS:
        raise RuntimeError("FP32 deployment freeze direction signature mismatch")
    if biological.get("peak_time_recovery") != EXPECTED_PEAK:
        raise RuntimeError("FP32 deployment freeze peak signature mismatch")

    files = freeze.get("reference_files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("Deployment freeze contains no reference-file identities")
    for name, meta in files.items():
        path = confirmation_dir / name
        if not path.is_file():
            raise RuntimeError(f"Frozen FP32 reference file missing: {path}")
        actual = sha256_file(path)
        expected = meta.get("sha256") if isinstance(meta, dict) else None
        if actual != expected:
            raise RuntimeError(
                f"Frozen FP32 reference file changed: {name}; expected {expected}, got {actual}"
            )
    return freeze


def load_reference_nuclei(path: Path) -> dict[str, dict[str, str]]:
    rows = final_eval.read_csv(path)
    if len(rows) != 2058:
        raise RuntimeError(f"Frozen FP32 nucleus rows={len(rows)}; expected 2058")
    by_id = {row["sample_id"]: row for row in rows}
    if len(by_id) != 2058:
        raise RuntimeError("Duplicate sample_id in frozen FP32 nucleus outputs")
    for row in rows:
        value = float(row["predicted_continuous_burden"])
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite frozen FP32 burden: {row['sample_id']}")
    return by_id


def load_reference_bags(path: Path) -> dict[str, dict[str, str]]:
    rows = final_eval.read_csv(path)
    if len(rows) != 22:
        raise RuntimeError(f"Frozen FP32 bag rows={len(rows)}; expected 22")
    by_name = {row["sample_name"]: row for row in rows}
    if len(by_name) != 22:
        raise RuntimeError("Duplicate sample_name in frozen FP32 bag outputs")
    return by_name


def build_onnx_bags(
    nucleus_rows: list[dict[str, Any]],
    frozen_bags: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in nucleus_rows:
        grouped[str(row["sample_name"])].append(row)
    if len(grouped) != 22:
        raise RuntimeError(f"ONNX bag count={len(grouped)}; expected 22")

    result: list[dict[str, Any]] = []
    for sample_name, subset in sorted(grouped.items()):
        if sample_name not in frozen_bags:
            raise RuntimeError(f"ONNX sample missing frozen bag reference: {sample_name}")
        ref = frozen_bags[sample_name]
        out: dict[str, Any] = {
            "sample_name": sample_name,
            "source_name": ref["source_name"],
            "strain": ref["strain"],
            "sex": ref["sex"],
            "particle_type": ref["particle_type"],
            "dose_Gy": ref["dose_Gy"],
            "hr_post_exposure": ref["hr_post_exposure"],
            "n_holdout_nuclei": len(subset),
            "predicted_bag_mean": float(
                np.mean([float(row["onnx_burden"]) for row in subset])
            ),
            "fp32_reference_bag_mean": float(ref["predicted_bag_mean"]),
            "nasa_avg_nfoci": float(ref["nasa_avg_nfoci"]),
            "nasa_num_nuc": int(float(ref["nasa_num_nuc"])),
        }
        out["onnx_minus_fp32_bag"] = (
            float(out["predicted_bag_mean"])
            - float(out["fp32_reference_bag_mean"])
        )
        out["residual"] = (
            float(out["predicted_bag_mean"]) - float(out["nasa_avg_nfoci"])
        )
        result.append(out)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"
    holdout_root = data_root / "nasa_bps_microscopy" / "r1_v2_final_holdout_blinded"

    parser.add_argument(
        "--checkpoint",
        default=str(
            model_root
            / "nasa_bps_53bp1"
            / "r1_v2_final_candidate"
            / "r1_countnet_v2_final_bn_recalibrated.pt"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(
            metadata_root
            / "r1_v2_freeze"
            / "r1_v2_final_holdout_manifest_blinded_qc1.csv"
        ),
    )
    parser.add_argument("--image-root", default=str(holdout_root / "images"))
    parser.add_argument(
        "--confirmation-dir",
        default=str(
            model_root
            / "nasa_bps_53bp1"
            / "r1_v2_final_holdout_confirmation"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(model_root / "nasa_bps_53bp1" / "r1_v2_deployment" / "onnx"),
    )
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--cache-items", type=int, default=512)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    image_root = Path(args.image_root).resolve()
    confirmation_dir = Path(args.confirmation_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    freeze_path = confirmation_dir / "fp32_deployment_reference_freeze.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.opset != 11:
        raise RuntimeError(
            f"Deployment policy freezes ONNX opset 11; requested {args.opset}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    checkpoint_sha = sha256_file(checkpoint)
    manifest_sha = sha256_file(manifest_path)
    if checkpoint_sha != FROZEN_REPAIRED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Authoritative FP32 SHA256 mismatch: {checkpoint_sha}")
    if manifest_sha != FROZEN_QC1_HOLDOUT_MANIFEST_SHA256:
        raise RuntimeError(f"QC1 manifest SHA256 mismatch: {manifest_sha}")

    freeze = verify_reference_freeze(
        freeze_path, confirmation_dir, checkpoint_sha, manifest_sha
    )
    freeze_sha = sha256_file(freeze_path)

    report_path = output_dir / "r1_countnet_v2_final_fp32_opset11_parity.json"
    nucleus_parity_path = output_dir / "r1_countnet_v2_final_fp32_opset11_nucleus_parity.csv"
    bag_parity_path = output_dir / "r1_countnet_v2_final_fp32_opset11_bag_parity.csv"
    delta_path = output_dir / "r1_countnet_v2_final_fp32_opset11_matched_deltas.csv"
    onnx_path = output_dir / "r1_countnet_v2_final_fp32_1x3x256x256_opset11.onnx"

    if report_path.exists():
        raise RuntimeError(
            f"ONNX parity has already been evaluated; refusing overwrite: {report_path}"
        )

    manifest = final_eval.read_csv(manifest_path)
    bags = final_eval.validate_manifest(manifest)
    if len(bags) != 22:
        raise RuntimeError(f"Validated holdout bags={len(bags)}; expected 22")

    frozen_nuclei = load_reference_nuclei(
        confirmation_dir / "final_holdout_nucleus_latent_burden.csv"
    )
    frozen_bags = load_reference_bags(
        confirmation_dir / "final_holdout_bag_predictions.csv"
    )

    print("Radiation Edge AI - NASA BPS R1 v2 FP32 -> ONNX parity")
    print(f"authoritative FP32 SHA256: {checkpoint_sha}")
    print(f"QC1 holdout manifest SHA256: {manifest_sha}")
    print(f"FP32 deployment-reference freeze SHA256: {freeze_sha}")
    print(f"input shape: {EXPECTED_INPUT_SHAPE}")
    print("output semantics: scalar latent continuous 53BP1 burden")
    print("ONNX opset: 11")
    print("raw phenotype table read by this stage: NO")

    device = torch.device("cpu")
    model = final_eval.load_model(v1, checkpoint, device)
    model.eval()
    dummy = torch.zeros(EXPECTED_INPUT_SHAPE, dtype=torch.float32)

    print("")
    print("[Export fixed-shape ONNX]")
    t0 = time.perf_counter()
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["burden"],
        dynamic_axes=None,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,
    )
    export_seconds = time.perf_counter() - t0
    print(f"export: PASS ({export_seconds:.2f}s)")

    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)
    graph = graph_summary(model_proto)
    if len(graph["inputs"]) != 1 or graph["inputs"][0]["shape"] != list(EXPECTED_INPUT_SHAPE):
        raise RuntimeError(f"Unexpected ONNX input signature: {graph['inputs']}")
    print("ONNX checker: PASS")
    print(f"ONNX SHA256: {sha256_file(onnx_path)}")
    print(f"ONNX nodes: {graph['n_nodes']}")
    print(
        "ONNX operators: "
        + ", ".join(f"{name}={count}" for name, count in graph["operator_counts"].items())
    )

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    if session.get_inputs()[0].name != "input":
        raise RuntimeError(f"Unexpected ORT input name: {session.get_inputs()[0].name}")
    if session.get_outputs()[0].name != "burden":
        raise RuntimeError(f"Unexpected ORT output name: {session.get_outputs()[0].name}")

    print("")
    print("[2058-nucleus FP32 parity]")
    cache = v1.NativeImageCache(image_root=image_root, max_items=args.cache_items)
    nucleus_rows: list[dict[str, Any]] = []
    frozen_gpu_values: list[float] = []
    pytorch_cpu_values: list[float] = []
    onnx_values: list[float] = []
    ordered_manifest = sorted(manifest, key=lambda row: row["sample_id"])

    infer_start = time.perf_counter()
    with torch.inference_mode():
        for index, row in enumerate(ordered_manifest, start=1):
            sample_id = row["sample_id"]
            if sample_id not in frozen_nuclei:
                raise RuntimeError(f"Manifest sample_id missing frozen FP32 output: {sample_id}")
            packed = v1.pad_and_pack(cache.get(row)).astype(np.float32, copy=False)
            batch = np.ascontiguousarray(packed[None, ...], dtype=np.float32)

            pt_value = float(model(torch.from_numpy(batch)).detach().cpu().item())
            ort_raw = session.run(["burden"], {"input": batch})[0]
            ort_flat = np.asarray(ort_raw, dtype=np.float32).reshape(-1)
            if ort_flat.size != EXPECTED_OUTPUT_ELEMENTS:
                raise RuntimeError(
                    f"Unexpected ONNX output for {sample_id}: shape={np.asarray(ort_raw).shape}"
                )
            ort_value = float(ort_flat[0])
            gpu_value = float(
                frozen_nuclei[sample_id]["predicted_continuous_burden"]
            )
            if not all(math.isfinite(v) for v in (pt_value, ort_value, gpu_value)):
                raise RuntimeError(f"Non-finite parity value for {sample_id}")

            frozen_gpu_values.append(gpu_value)
            pytorch_cpu_values.append(pt_value)
            onnx_values.append(ort_value)
            nucleus_rows.append(
                {
                    "sample_id": sample_id,
                    "nucleus_key": row["nucleus_key"],
                    "sample_name": row["sample_name"],
                    "source_name": row["source_name"],
                    "particle_type": row["particle_type"],
                    "dose_Gy": row["dose_Gy"],
                    "hr_post_exposure": row["hr_post_exposure"],
                    "frozen_gpu_fp32_burden": gpu_value,
                    "pytorch_cpu_fp32_burden": pt_value,
                    "onnx_burden": ort_value,
                    "onnx_minus_pytorch_cpu": ort_value - pt_value,
                    "onnx_minus_frozen_gpu": ort_value - gpu_value,
                }
            )
            if index % 250 == 0 or index == len(ordered_manifest):
                print(f"parity nuclei: {index}/{len(ordered_manifest)}")

    inference_seconds = time.perf_counter() - infer_start
    gpu_arr = np.asarray(frozen_gpu_values, dtype=np.float64)
    pt_arr = np.asarray(pytorch_cpu_values, dtype=np.float64)
    ort_arr = np.asarray(onnx_values, dtype=np.float64)

    pt_vs_ort = error_stats(pt_arr, ort_arr)
    gpu_vs_ort = error_stats(gpu_arr, ort_arr)
    gpu_vs_pt = error_stats(gpu_arr, pt_arr)
    numerical_pass = (
        within_limits(pt_vs_ort, PT_VS_ORT_LIMITS)
        and within_limits(gpu_vs_ort, GPU_REFERENCE_VS_ORT_LIMITS)
    )

    print(
        "ONNX vs CPU PyTorch max/mean/RMSE: "
        f"{pt_vs_ort['max_abs_error']:.8g} / "
        f"{pt_vs_ort['mean_abs_error']:.8g} / {pt_vs_ort['rmse']:.8g}"
    )
    print(
        "ONNX vs frozen GPU FP32 max/mean/RMSE: "
        f"{gpu_vs_ort['max_abs_error']:.8g} / "
        f"{gpu_vs_ort['mean_abs_error']:.8g} / {gpu_vs_ort['rmse']:.8g}"
    )
    print(f"predeclared numerical parity gates: {'PASS' if numerical_pass else 'FAIL'}")

    onnx_bags = build_onnx_bags(nucleus_rows, frozen_bags)
    deltas = final_eval.build_deltas(onnx_bags)
    peaks = final_eval.peak_rows(deltas)
    directions = final_eval.direction_rows(deltas)
    source_residuals = final_eval.source_residual_rows(v1, onnx_bags)
    gate = final_eval.evaluate_gate(v1, onnx_bags, deltas, peaks)

    exact_signature_pass = (
        gate.get("all_pass") is True
        and gate.get("direction_counts") == EXPECTED_DIRECTION_COUNTS
        and gate.get("peak_time_recovery") == EXPECTED_PEAK
    )
    biological_pass = bool(gate.get("all_pass")) and exact_signature_pass
    all_pass = numerical_pass and biological_pass

    write_csv(nucleus_parity_path, nucleus_rows)
    write_csv(bag_parity_path, onnx_bags)
    write_csv(delta_path, deltas)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_FP32_ONNX_PARITY_EVALUATION",
        "exporter_source_sha256": sha256_file(Path(__file__).resolve()),
        "authoritative_checkpoint_sha256": checkpoint_sha,
        "qc1_holdout_manifest_sha256": manifest_sha,
        "fp32_deployment_reference_freeze_sha256": freeze_sha,
        "fp32_reference_protocol_deviation_carried_forward": freeze.get(
            "protocol_deviation_carried_forward"
        ),
        "onnx": {
            "path": str(onnx_path),
            "sha256": sha256_file(onnx_path),
            "size_bytes": onnx_path.stat().st_size,
            "opset": 11,
            "fixed_input_shape": list(EXPECTED_INPUT_SHAPE),
            "output_name": "burden",
            "graph": graph,
            "export_seconds": export_seconds,
        },
        "runtime": {
            "torch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "onnxruntime_provider": "CPUExecutionProvider",
            "full_2058_inference_seconds": inference_seconds,
        },
        "accelerator_boundary": {
            "inside_onnx": "R1CountNetV1 neural regressor only",
            "outside_onnx": [
                "TIFF loading",
                "p1/p99.5 FITC and DAPI normalization",
                "native-scale 256x256 center padding",
                "zero third-channel construction",
                "MASK hard QC",
                "bag averaging",
                "branch-specific sham matching",
                "biological metrics and gates",
            ],
        },
        "nucleus_count": len(nucleus_rows),
        "bag_count": len(onnx_bags),
        "matched_contrast_count": len(deltas),
        "numerical_parity": {
            "onnx_vs_cpu_pytorch": {
                "limits": PT_VS_ORT_LIMITS,
                "observed": pt_vs_ort,
                "pass": within_limits(pt_vs_ort, PT_VS_ORT_LIMITS),
            },
            "onnx_vs_frozen_gpu_fp32": {
                "limits": GPU_REFERENCE_VS_ORT_LIMITS,
                "observed": gpu_vs_ort,
                "pass": within_limits(gpu_vs_ort, GPU_REFERENCE_VS_ORT_LIMITS),
            },
            "cpu_pytorch_vs_frozen_gpu_fp32_descriptive": gpu_vs_pt,
            "all_numerical_gates_pass": numerical_pass,
        },
        "biological_parity": {
            "gate_result": gate,
            "required_exact_direction_signature": EXPECTED_DIRECTION_COUNTS,
            "required_peak_time_recovery": EXPECTED_PEAK,
            "exact_signature_pass": exact_signature_pass,
            "all_biological_gates_pass": biological_pass,
            "direction_summary": directions,
            "source_residuals": source_residuals,
        },
        "raw_phenotype_table_read": False,
        "no_quantization_performed": True,
        "overall_onnx_parity_pass": all_pass,
        "authorized_next_stage": (
            "DEVELOPMENT_ONLY_INT8_CALIBRATION_AND_KNERON_OPTIMIZATION"
            if all_pass
            else "BLOCKED"
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("")
    print("[ONNX biological parity]")
    bm = gate["bag_metrics"]
    dm = gate["delta_metrics"]
    print(
        f"bag MAE/RMSE/Pearson/Spearman: {bm['mae']:.4f} / {bm['rmse']:.4f} / "
        f"{bm['pearson']:.4f} / {bm['spearman']:.4f}"
    )
    print(
        f"delta MAE/Pearson/Spearman: {dm['delta_mae']:.4f} / "
        f"{dm['delta_pearson']:.4f} / {dm['delta_spearman']:.4f}"
    )
    print(f"direction overall: {gate['direction_counts']['overall']}")
    print(f"direction 4 h: {gate['direction_counts']['4h']}")
    print(f"direction 24+48 h: {gate['direction_counts']['24+48h']}")
    print(f"direction by Source Name: {gate['direction_counts']['by_source']}")
    print(f"peak-time recovery: {gate['peak_time_recovery']}")
    print(f"all seven original biological gates: {'PASS' if gate['all_pass'] else 'FAIL'}")
    print(f"exact FP32 biological signature reproduced: {'YES' if exact_signature_pass else 'NO'}")
    print(f"OVERALL FP32 -> ONNX PARITY PASS: {'YES' if all_pass else 'NO'}")
    print(f"ONNX: {onnx_path}")
    print(f"ONNX SHA256: {report['onnx']['sha256']}")
    print(f"report: {report_path}")
    print("raw phenotype table read by ONNX stage: NO")

    if not all_pass:
        raise SystemExit(
            "FP32 -> ONNX parity failed a predeclared numerical or biological gate; quantization is blocked."
        )
    print(
        "NEXT GATE: freeze this ONNX artifact; select INT8 calibration data from the 7200-nucleus development set only."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
