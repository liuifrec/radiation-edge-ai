"""Export complete, untrained NASA candidate graphs for early KL720 screening.

These deterministic random-weight graphs answer an architecture-feasibility
question only. Synthetic PyTorch/ONNX parity is not biological accuracy,
successful Kneron compilation, or physical NPU execution. The trained graph
must be exported and screened again before deployment.

Imports are deliberately lightweight so that ``--help`` also works outside
the scientific environment. Execute export in the separate ``.[nasa]``
environment, never by changing the existing Kneron PLUS environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from . import model_factory


SCHEMA_VERSION = 1
PURPOSE = "architecture_feasibility_only"
INPUT_SHAPE = [1, 3, 256, 256]
OUTPUT_SHAPE = [1]
OPSET = 11
PARITY_ATOL = 1e-5
PARITY_RTOL = 1e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Candidate graph export requires NumPy, PyTorch, ONNX and ONNX Runtime "
            "in the separate scientific environment; install the project's .[nasa] "
            "extra there. Do not change the Kneron PLUS environment."
        ) from exc
    return np, torch, onnx, ort


def _versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "torch", "torchvision", "onnx", "onnxruntime"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def _shape(value: Any) -> list[Any]:
    result = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(str(dim.dim_param))
        else:
            result.append(None)
    return result


def graph_summary(model: Any) -> dict[str, Any]:
    """Return the complete exported graph's operators and tensor signatures."""

    initializers = {value.name for value in model.graph.initializer}
    return {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "operators": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "inputs": [
            {"name": value.name, "shape": _shape(value),
             "onnx_element_type": int(value.type.tensor_type.elem_type)}
            for value in model.graph.input if value.name not in initializers
        ],
        "outputs": [
            {"name": value.name, "shape": _shape(value),
             "onnx_element_type": int(value.type.tensor_type.elem_type)}
            for value in model.graph.output
        ],
        "opsets": [
            {"domain": item.domain, "version": int(item.version)}
            for item in model.opset_import
        ],
    }


def _probes(np: Any, seed: int) -> list[tuple[str, Any]]:
    """Create distinct, reproducible shape probes without reading NASA data."""

    zeros = np.zeros(INPUT_SHAPE, dtype=np.float32)
    ramp = zeros.copy()
    ramp[0, 0] = np.linspace(0, 1, 256 * 256, dtype=np.float32).reshape(256, 256)
    ramp[0, 1] = 1 - ramp[0, 0]
    random = np.random.default_rng(seed).random(INPUT_SHAPE).astype(np.float32)
    random[:, 2] = 0
    return [("zeros", zeros), ("opposed_channel_ramps", ramp), ("seeded_uniform", random)]


def _check_signature(summary: dict[str, Any]) -> None:
    if summary["inputs"] != [
        {"name": "nucleus", "shape": INPUT_SHAPE, "onnx_element_type": 1}
    ]:
        raise RuntimeError("Exported ONNX input does not match fixed float32 NASA contract")
    if summary["outputs"] != [
        {"name": "burden", "shape": OUTPUT_SHAPE, "onnx_element_type": 1}
    ]:
        raise RuntimeError("Exported ONNX output does not match the complete scalar burden head")
    if summary["opsets"] != [{"domain": "", "version": OPSET}]:
        raise RuntimeError("Exported ONNX graph does not use only the required ONNX opset 11")


def _export_one(
    architecture: str, directory: Path, seed: int, run_id: str,
    runtime: tuple[Any, Any, Any, Any], versions: dict[str, str],
) -> dict[str, Any]:
    np, torch, onnx, ort = runtime
    directory.mkdir()
    onnx_path = directory / "candidate.onnx"
    probes = _probes(np, seed)

    # Restore the caller's CPU RNG state after deterministic, download-free
    # construction. Every architecture receives the same initialization seed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = model_factory.create_model(architecture, pretrained=False).cpu().eval()
    parameter_count = sum(value.numel() for value in model.parameters())
    with torch.no_grad():
        torch.onnx.export(
            model,
            torch.from_numpy(probes[0][1]),
            str(onnx_path),
            input_names=["nucleus"],
            output_names=["burden"],
            export_params=True,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )

    exported = onnx.load(str(onnx_path), load_external_data=False)
    if any(onnx.external_data_helper.uses_external_data(item)
           for item in exported.graph.initializer):
        raise RuntimeError("Candidate graph must embed all weights in one ONNX file")
    onnx.checker.check_model(exported)
    summary = graph_summary(exported)
    _check_signature(summary)
    onnx_sha256 = _sha256(onnx_path)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    comparisons = []
    for name, values in probes:
        with torch.no_grad():
            reference = model(torch.from_numpy(values)).detach().cpu().numpy()
        actual = session.run(["burden"], {"nucleus": values})[0]
        if list(reference.shape) != OUTPUT_SHAPE or list(actual.shape) != OUTPUT_SHAPE:
            raise RuntimeError("Scalar burden output shape changed during parity checking")
        if not np.isfinite(reference).all() or not np.isfinite(actual).all():
            raise RuntimeError("Non-finite candidate output in synthetic parity check")
        if (reference < 0).any() or (actual < 0).any():
            raise RuntimeError("Candidate output violated the nonnegative burden contract")
        max_abs = float(np.abs(actual.astype(np.float64) - reference.astype(np.float64)).max())
        passed = bool(np.allclose(actual, reference, atol=PARITY_ATOL, rtol=PARITY_RTOL))
        comparison = {
            "probe": name,
            "input_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
            "input_shape": list(values.shape),
            "third_channel_is_zero": bool((values[:, 2] == 0).all()),
            "pytorch_output": reference.tolist(),
            "onnxruntime_output": actual.tolist(),
            "max_absolute_error": max_abs,
            "passed": passed,
        }
        comparisons.append(comparison)
        if not passed:
            raise RuntimeError(
                "Synthetic PyTorch/ONNX parity failed for {} / {} (max abs error {})".format(
                    architecture, name, max_abs
                )
            )
    if _sha256(onnx_path) != onnx_sha256:
        raise RuntimeError("ONNX file changed while its parity was being checked")
    reference_values = [item["pytorch_output"][0] for item in comparisons]
    output_span = float(max(reference_values) - min(reference_values))

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "run_id": run_id,
        "created_utc": _utc(),
        "architecture": architecture,
        "purpose": PURPOSE,
        "trained": False,
        "pretrained_weights": False,
        "input_shape": INPUT_SHAPE,
        "output_shape": OUTPUT_SHAPE,
        "opset": OPSET,
        "onnx_file": "candidate.onnx",
        "onnx_sha256": onnx_sha256,
        "onnx_size_bytes": onnx_path.stat().st_size,
        "seed": seed,
        "parameter_count": parameter_count,
        "source_hashes": {
            "candidate_graphs.py": _sha256(Path(__file__).resolve()),
            "model_factory.py": _sha256(Path(model_factory.__file__).resolve()),
        },
        "dependency_versions": versions,
        "model_contract": model_factory.model_contract(architecture, pretrained=False),
        "graph": summary,
        "onnx_checker": "PASS",
        "parity": {
            "status": "PASS",
            "reference": "PyTorch CPU eval",
            "backend": "ONNX Runtime CPUExecutionProvider",
            "atol": PARITY_ATOL,
            "rtol": PARITY_RTOL,
            "probe_count": len(comparisons),
            "probes": comparisons,
            "maximum_absolute_error": max(item["max_absolute_error"] for item in comparisons),
            "reference_output_span": output_span,
            "constant_reference_outputs": output_span == 0,
            "diagnostic_note": (
                "All synthetic reference outputs are identical at float32 precision. "
                "Parity is a weak numerical diagnostic for this untrained candidate; "
                "the full exported graph still requires vendor screening."
                if output_span == 0 else
                "Synthetic parity checks export behavior only; trained weights and "
                "NASA biological accuracy require separate evaluation."
            ),
        },
        "nasa_data_read": False,
        "biological_accuracy_evaluated": False,
        "kneron_support_evaluated": False,
        "physical_npu_executed": False,
        "trained_graph_requires_new_export_and_screen": True,
    }
    manifest_path = directory / "candidate.json"
    _write_json(manifest_path, report)
    return {
        "architecture": architecture,
        "onnx_file": "{}/candidate.onnx".format(architecture),
        "candidate_manifest": "{}/candidate.json".format(architecture),
        "onnx_sha256": onnx_sha256,
        "candidate_manifest_sha256": _sha256(manifest_path),
    }


def export_candidates(
    out: Path | str, models: Sequence[str] = model_factory.MODEL_NAMES, seed: int = 720,
) -> dict[str, Any]:
    """Export and verify selected candidates into a new directory, never overwrite.

    A failed run retains its diagnostics and any completed model artifacts, but
    its root manifest stays FAILED. It must not be used as a complete campaign.
    """

    selected = list(models)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Choose at least one architecture, with no duplicates")
    for architecture in selected:
        model_factory.get_model_spec(architecture)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer from 0 through 4294967295")
    output_dir = Path(out).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite an existing output directory: {}".format(output_dir))
    runtime = _runtime()
    versions = _versions()
    output_dir.mkdir(parents=True, exist_ok=False)
    index_path = output_dir / "candidate_graphs.json"
    index = {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING",
        "run_id": uuid.uuid4().hex,
        "created_utc": _utc(),
        "purpose": PURPOSE,
        "trained": False,
        "requested_models": selected,
        "seed": seed,
        "input_shape": INPUT_SHAPE,
        "output_shape": OUTPUT_SHAPE,
        "opset": OPSET,
        "models": [],
        "nasa_data_read": False,
        "biological_accuracy_evaluated": False,
        "kneron_support_evaluated": False,
        "physical_npu_executed": False,
    }
    _write_json(index_path, index)
    torch = runtime[1]
    previous_threads = torch.get_num_threads()
    current = None
    try:
        # A small fixed CPU worker count avoids machine-dependent oversubscription
        # and makes this short screening stage usable on ordinary local machines.
        torch.set_num_threads(1)
        for current in selected:
            entry = _export_one(
                current, output_dir / current, seed, index["run_id"], runtime, versions
            )
            index["models"].append(entry)
            _write_json(index_path, index)
        index["status"] = "COMPLETE"
        index["completed_utc"] = _utc()
        _write_json(index_path, index)
    except Exception as exc:
        index["status"] = "FAILED"
        index["failed_utc"] = _utc()
        index["failure"] = {
            "architecture": current,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        _write_json(index_path, index)
        raise
    finally:
        torch.set_num_threads(previous_threads)
    return index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New directory for candidate graphs and manifests")
    parser.add_argument("--models", nargs="+", choices=model_factory.MODEL_NAMES,
                        default=list(model_factory.MODEL_NAMES))
    parser.add_argument("--seed", type=int, default=720)
    args = parser.parse_args(argv)
    try:
        result = export_candidates(args.out, args.models, args.seed)
    except Exception as exc:
        print("Candidate graph export FAILED: {}".format(exc), file=sys.stderr)
        return 1
    print("NASA candidate graph export COMPLETE: {} full models".format(len(result["models"])))
    for model in result["models"]:
        print("{}: ONNX checker + synthetic CPU parity PASS; SHA256 {}".format(
            model["architecture"], model["onnx_sha256"]
        ))
    print("Manifest: {}".format(Path(args.out).expanduser().resolve() / "candidate_graphs.json"))
    print("Random weights only. Biological accuracy and KL720 support remain unevaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
