"""Screen one complete candidate or strict-parity trained NASA graph on the KL720 SDK.

This file runs directly in the historical Python 3.7 toolchain container.  It
does not import the application package, install packages, quantize, compile a
NEF, read NASA images, or communicate with a device.  ``supported`` means only
that fresh IP-evaluator evidence supports an NPU-compute architecture screen.

Evidence formats are documented by Kneron:
https://doc.kneron.com/docs/toolchain/appendix/yolo_example/
https://kneron.github.io/kneopi-documentation/examples/tc/yolov7.html
https://kneron.github.io/kneopi-documentation/examples/tc/pidnet.html
Unrecognized versions remain ``unknown`` and retain their raw evidence.
"""

import argparse
import contextlib
import csv
import hashlib
import importlib
import io
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath


EXIT_CODES = {"supported": 0, "unsupported": 2, "unknown": 3}
ARTIFACT_NAMES = (
    "model_fx_report.json", "model_fx_report.html", "batch_compile.log",
    "backtrace.log", "ioinfo.csv", "ProfileResult.txt", "ip_eval_prof.txt",
)
MODEL_NAMES = ("r1countnet_v1", "mobilenet_v2", "resnet18")
# Export identities and model IDs from the November 18 trained-graph screen.
# Each immutable export binds its checkpoint, training run, ONNX and calibration manifest.
TRAINED_EXPORTS = {
    ("mobilenet_v2", "fold_female_role"): ("40b89d4b349dfa4060e4fd62a31cf27647b6fb04c1bdef93b4f7b84165f32257", 32781),
    ("mobilenet_v2", "fold_male_2_role"): ("bff39f471361a57f25c61d76b20570595348364afc70296e9dc959925814f192", 32782),
    ("resnet18", "fold_female_role"): ("fc12dcf5be78d45b2c221257f4e1f70b5017982d16bec7af17bfe886fce07af5", 32783),
    ("resnet18", "fold_male_1_role"): ("d41c6627bc31f3456e3292e02b453025c4301301cd1d8a042719fe4356b6a34a", 32784),
    ("resnet18", "fold_male_2_role"): ("990032c6d2c318e42cb78d30eb5becb55defc338334cc2fd19822c9df6390a7e", 32785),
}
BOOKKEEPING_OPS = {"Identity", "Reshape", "Flatten", "Squeeze", "Unsqueeze"}
TRAINED_PARITY_SHA256 = {
    ("mobilenet_v2", "fold_female_role"): "bac5774a2038ca22fcc5d51c19d7587156901894792051502359e50247a31af4",
    ("mobilenet_v2", "fold_male_2_role"): "b7f01de72833c46cffe467e3f7d163790782ee831254170b3442f8471ed0cb15",
    ("resnet18", "fold_female_role"): "6c1f230cf9ee2335ae4a0432f4a0cbf891c780ea5aa0e9fcaaba467fa0196dd9",
    ("resnet18", "fold_male_1_role"): "526f70aa7e0c9647e2ba2c264e24a39522883613e814e397eaacabc39e0e08fa",
    ("resnet18", "fold_male_2_role"): "2a9fbd3874598dd269205922035e30eb44027e71d2b4c318238270bca438e60a",
}
SHAPE_OPS = BOOKKEEPING_OPS | {
    "Gather", "Slice", "Concat", "Cast", "Add", "Sub", "Mul", "Div",
}
NEURAL_OPS = {
    "Conv", "ConvTranspose", "Gemm", "MatMul", "Relu", "PRelu", "LeakyRelu",
    "Clip", "Sigmoid", "Tanh", "Softmax", "BatchNormalization", "MaxPool",
    "AveragePool", "GlobalAveragePool", "GlobalMaxPool", "ReduceMean",
    "ReduceSum", "Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Exp", "Log",
    "Resize", "Upsample", "KneronResize", "Gather", "Slice", "Concat",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _identity(path):
    path = Path(path)
    stat = path.stat()
    return {"sha256": sha256_file(path), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def _artifact_files(roots):
    paths = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for name in ARTIFACT_NAMES:
            for path in root.rglob(name):
                if path.is_file() and not path.is_symlink():
                    paths.add(path.resolve())
    return sorted(paths)


def snapshot_artifacts(roots):
    return {str(path): _identity(path) for path in _artifact_files(roots)}


def harvest_artifacts(roots, before, started_ns, output):
    """Copy only evidence created or rewritten since this evaluation started.

    A previous run's file never qualifies merely because no fresh file exists.
    Nanosecond timestamps are deliberately strict: coarse/unreliable metadata
    may yield unknown, but must not accidentally authorize an architecture.
    """
    output = Path(output)
    accepted, rejected = [], []
    for path in _artifact_files(roots):
        identity = _identity(path)
        old = before.get(str(path))
        fresh = identity["mtime_ns"] >= started_ns and (
            old is None or identity["mtime_ns"] > old["mtime_ns"]
        )
        if not fresh:
            rejected.append({"source": str(path), "reason": "not_fresh_for_this_evaluation"})
            continue
        output.mkdir(parents=True, exist_ok=True)
        destination = output / ("{:03d}_{}".format(len(accepted), path.name))
        shutil.copyfile(str(path), str(destination))
        if sha256_file(destination) != identity["sha256"]:
            raise RuntimeError("Vendor artifact changed while being copied: {}".format(path))
        accepted.append(dict(identity, name=path.name, source=str(path), path=str(destination)))
    return accepted, rejected


def _shape(value):
    return [int(dim.dim_value) if dim.HasField("dim_value") else None
            for dim in value.type.tensor_type.shape.dim]


def _numel(shape):
    if shape is None or any(not isinstance(value, int) or value < 1 for value in shape):
        return None
    result = 1
    for value in shape:
        result *= value
    return result


def graph_summary(model):
    initializers = {value.name for value in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializers]
    return {
        "nodes": len(model.graph.node),
        "operators": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "inputs": [{"name": value.name, "shape": _shape(value)} for value in inputs],
        "outputs": [{"name": value.name, "shape": _shape(value)} for value in model.graph.output],
    }


def graph_cpu_context(model, onnx):
    """Build conservative, exact-name evidence for harmless CPU bookkeeping."""
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    graph = inferred.graph
    shapes = {value.name: _shape(value)
              for value in list(graph.input) + list(graph.output) + list(graph.value_info)}
    constants = {}
    for value in graph.initializer:
        shapes[value.name] = list(value.dims)
        try:
            array = onnx.numpy_helper.to_array(value)
            constants[value.name] = array.reshape(-1).tolist() if array.size <= 8 else None
        except Exception:
            constants[value.name] = None
    nodes = []
    for node in graph.node:
        item = {"name": node.name, "op_type": node.op_type, "inputs": list(node.input),
                "outputs": list(node.output), "axis": 0}
        for attribute in node.attribute:
            if attribute.name == "axis":
                item["axis"] = int(attribute.i)
            if node.op_type == "Constant" and attribute.name == "value":
                array = onnx.numpy_helper.to_array(attribute.t)
                constants[node.output[0]] = array.reshape(-1).tolist() if array.size <= 8 else None
                shapes[node.output[0]] = list(array.shape)
        nodes.append(item)
    shape_values = set()
    for node in nodes:
        inputs = [name for name in node["inputs"] if name]
        if node["op_type"] in {"Shape", "Size"} or (
            node["op_type"] in SHAPE_OPS and inputs
            and all(name in shape_values or name in constants for name in inputs)
            and any(name in shape_values for name in inputs)
        ):
            shape_values.update(node["outputs"])
    signature = graph_summary(inferred)
    return {"nodes": nodes, "shapes": shapes, "constants": constants,
            "shape_values": sorted(shape_values), "inputs": signature["inputs"],
            "outputs": signature["outputs"]}


def _cpu_node(name, stated_type, context):
    matches = [node for node in context.get("nodes", [])
               if name and (node["name"] == name or name in node["outputs"])]
    item = {"name": name, "reported_type": stated_type}
    if len(matches) != 1:
        item.update(classification="unresolved", reason="no_unique_exact_graph_node")
        return item
    node = matches[0]
    op = node["op_type"]
    item["op_type"] = op
    item["graph_node_identity"] = node["name"] or "outputs:" + ",".join(node["outputs"])
    if stated_type and stated_type != op:
        item.update(classification="unresolved", reason="report_graph_operator_mismatch")
        return item
    shapes = context.get("shapes", {})
    shape_values = set(context.get("shape_values", []))
    constants = context.get("constants", {})
    inputs, outputs = node["inputs"], node["outputs"]
    reason = None
    if op in {"Shape", "Size", "Constant"}:
        reason = "exact_shape_or_constant_node"
    elif outputs and all(name in shape_values for name in outputs):
        reason = "traced_shape_values_only"
    elif op in BOOKKEEPING_OPS and inputs and len(outputs) == 1:
        input_size = _numel(shapes.get(inputs[0]))
        if input_size is not None and input_size == _numel(shapes.get(outputs[0])):
            reason = "exact_node_preserves_element_values_and_count"
    elif op == "Gather" and len(inputs) == 2 and len(outputs) == 1:
        if (_numel(shapes.get(inputs[0])) == 1
                and _numel(shapes.get(outputs[0])) == 1
                and constants.get(inputs[1]) in ([0], [-1])):
            reason = "scalar_identity_gather"
    if reason:
        item.update(classification="bookkeeping", reason=reason)
    elif op in NEURAL_OPS:
        item.update(classification="neural_compute", reason="cpu_operation_on_model_values")
    else:
        item.update(classification="unresolved", reason="cpu_operation_semantics_not_proven")
    return item


def _positive_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (ValueError, TypeError):
        return None


def _cpu_report(value, context):
    if value == "N/A":
        return [], True
    if not isinstance(value, str) or not value.strip():
        return [], False
    # The documented string groups exact node names under their ONNX type.
    groups = list(re.finditer(r"(?:^|[\n;]\s*)([A-Za-z][A-Za-z0-9_]*):\s*", value))
    if not groups or groups[0].start() != 0:
        return [], False
    nodes = []
    for index, match in enumerate(groups):
        end = groups[index + 1].start() if index + 1 < len(groups) else len(value)
        names = [name.strip() for name in value[match.end():end].split(",")]
        if not names or any(not name or ":" in name for name in names):
            return nodes, False
        nodes.extend(_cpu_node(name, match.group(1), context) for name in names)
    return nodes, True


def _io_report(text, context):
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or any(len(row) < 6 or row[0].strip() not in {"i", "o", "c"}
                       for row in rows):
        return [], False
    if not {"i", "o"}.issubset({row[0].strip() for row in rows}):
        return [], False
    try:
        if any(any(int(value) < 1 for value in row[3:]) for row in rows):
            return [], False
    except ValueError:
        return [], False
    # The documented KL720 CSV stores CHW dimensions without the batch axis.
    # Bind every i/o row to this exact optimized graph. Unknown renamed aliases
    # are deliberately not guessed from a similar-looking unrelated model.
    for kind, key in (("i", "inputs"), ("o", "outputs")):
        expected = context.get(key, [])
        observed = [row for row in rows if row[0].strip() == kind]
        if not expected or len(observed) != len(expected):
            return [], False
        seen_indices = set()
        for row in observed:
            try:
                index = int(row[1])
            except ValueError:
                return [], False
            if index in seen_indices or not 0 <= index < len(expected):
                return [], False
            seen_indices.add(index)
            tensor = expected[index]
            shape = tensor["shape"]
            if kind == "o" and _numel(shape) == 1:
                chw = [1, 1, 1]
            elif len(shape) == 4 and shape[0] == 1:
                chw = shape[1:]
            else:
                return [], False
            if row[2].strip() != tensor["name"] or [int(value) for value in row[3:]] != chw:
                return [], False
    return [_cpu_node(row[2].strip(), None, context) for row in rows
            if row[0].strip() == "c"], True


def _failure_evidence(texts):
    """Recognize actual failure messages, not incidental words or zero errors."""
    failures = []
    for source, text in texts:
        for line in text.splitlines():
            # SDK logs commonly prepend one or more [module]/[severity] tags.
            body = re.sub(r"^\s*(?:\[[^\]\n]+\]\s*)*", "", line)
            explicit = re.match(r"(?:FAILURE\s+for\s+model\b|failed\s+to\s+compile\b)", body, re.IGNORECASE)
            codes = re.findall(r"\bERR\s*:\s*(-?\d+)\b", body, re.IGNORECASE)
            if explicit or any(int(value) != 0 for value in codes):
                failures.append({"source": source, "message": line})
    return failures


def classify_evidence(evaluation_text, artifacts, context, evaluation_completed=True, error=None):
    """Require fresh positive performance AND explicit CPU partition evidence."""
    positive, cpu_sources, cpu_nodes, reasons, invalid = [], [], [], [], []
    texts = [("evaluate_return", evaluation_text or "")]
    for artifact in artifacts:
        text = Path(artifact["path"]).read_text(encoding="utf-8", errors="replace")
        name = artifact["name"]
        texts.append((artifact["path"], text))
        if name == "model_fx_report.json":
            try:
                report = json.loads(text)
            except (ValueError, TypeError):
                invalid.append("invalid_model_fx_report_json")
                continue
            if not isinstance(report, dict):
                invalid.append("model_fx_report_is_not_an_object")
                continue
            fps = _positive_number(report.get("kdp720/ip_eval/fps"))
            if fps is not None:
                positive.append({"source": artifact["path"], "format": "kdp720_json", "estimated_npu_fps": fps})
            nodes, recognized = _cpu_report(report.get("kdp720/ip_eval/cpu_node"), context)
            if recognized:
                cpu_sources.append(artifact["path"])
                cpu_nodes.extend(nodes)
            else:
                invalid.append("unrecognized_target720_cpu_report")
        elif name == "ioinfo.csv":
            nodes, recognized = _io_report(text, context)
            if recognized:
                cpu_sources.append(artifact["path"])
                cpu_nodes.extend(nodes)
            else:
                invalid.append("unrecognized_ioinfo_partition_report")
    for source, text in texts:
        match = re.search(r"\[Evaluation Result\]\s*estimate FPS float\s*=\s*([0-9.eE+-]+)", text)
        if match and _positive_number(match.group(1)) is not None:
            positive.append({"source": source, "format": "kl720_legacy_text", "estimated_npu_fps": float(match.group(1))})
    corpus = "\n".join(text for _, text in texts) + "\n" + (error or "")
    unsupported = re.search(
        r"hardware\s*not\s*support(?:ed)?|\bhw\s+not\s+support|"
        r"unsupported\s+(?:operator|op|node|layer)\s*[:=]\s*[^0\s]|"
        r"(?:operator|node|layer)\s+\S+\s+(?:is\s+)?not supported",
        corpus, re.IGNORECASE,
    )
    cpu_warning = re.findall(r"this model has\s+(\d+)\s+CPU ops", corpus, flags=re.IGNORECASE)
    unique_cpu_names = {node.get("graph_node_identity", node["name"]) for node in cpu_nodes}
    if cpu_warning and max(map(int, cpu_warning)) > len(unique_cpu_names):
        invalid.append("cpu_warning_not_fully_explained_by_partition_evidence")
    neural = [node for node in cpu_nodes if node["classification"] == "neural_compute"]
    unresolved = [node for node in cpu_nodes if node["classification"] == "unresolved"]
    failure_evidence = _failure_evidence(texts)
    if unsupported:
        status = "unsupported"
        reasons.append("explicit_vendor_hardware_rejection")
    elif neural:
        status = "unsupported"
        reasons.append("neural_network_computation_assigned_to_cpu")
    elif failure_evidence:
        status = "unknown"
        reasons.append("vendor_failure_evidence_overrides_partial_positive_report")
    elif not evaluation_completed or error:
        status = "unknown"
        reasons.append("vendor_evaluation_did_not_complete")
    elif not positive or not cpu_sources or unresolved or invalid:
        status = "unknown"
        if not positive:
            reasons.append("no_recognized_positive_target720_evaluation")
        if not cpu_sources:
            reasons.append("no_explicit_fresh_cpu_partition_evidence")
        if unresolved:
            reasons.append("unresolved_cpu_nodes")
        reasons.extend(sorted(set(invalid)))
    else:
        status = "supported"
        reasons.append("fresh_target720_ip_evaluation_with_no_cpu_neural_compute")
    return {"support_status": status, "reasons": reasons,
            "positive_evidence": positive, "cpu_partition_sources": cpu_sources,
            "cpu_nodes": cpu_nodes, "failure_evidence": failure_evidence}


@contextlib.contextmanager
def _vendor_log(path):
    """Capture Python and native subprocess stdout/stderr in the same raw log."""
    sys.stdout.flush()
    sys.stderr.flush()
    with Path(path).open("w", encoding="utf-8") as stream:
        saved = []
        try:
            for fd in (1, 2):
                saved.append((fd, os.dup(fd)))
                os.dup2(stream.fileno(), fd)
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                yield stream
        finally:
            stream.flush()
            for fd, original in saved:
                os.dup2(original, fd)
                os.close(original)


def _load_runtime():
    onnx = importlib.import_module("onnx")
    ktc = importlib.import_module("ktc")
    try:
        optimizer = importlib.import_module("kneronnxopt")
    except ImportError as exc:
        # A broken installed optimizer is not equivalent to an absent package.
        if getattr(exc, "name", None) != "kneronnxopt":
            raise
        optimizer = None
    return onnx, ktc, optimizer


def probe_runtime(args):
    """Check SDK imports and API presence without evaluating any model."""
    output = Path(args.out).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a new receipt before importing vendor code. Existing evidence is
    # never overwritten, including when an SDK import subsequently fails.
    with output.open("x", encoding="utf-8") as handle:
        report = {
            "schema_version": 1, "purpose": "toolchain_runtime_preflight",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "FAILED", "toolchain_image_id": args.toolchain_image_id,
            "python_version": platform.python_version(), "python_executable": sys.executable,
            "dependency_versions": {name: None for name in ("onnx", "ktc", "kneronnxopt")},
            "dependency_paths": {name: None for name in ("onnx", "ktc", "kneronnxopt")},
            "optimizer_backend": None, "error": None,
            "model_evaluation_performed": False,
        }
        try:
            if sys.version_info < (3, 7):
                raise RuntimeError("Kneron screening requires Python 3.7 or later")
            onnx, ktc, optimizer = _load_runtime()
            modules = {"onnx": onnx, "ktc": ktc, "kneronnxopt": optimizer}
            for name, module in modules.items():
                report["dependency_versions"][name] = getattr(module, "__version__", None)
                report["dependency_paths"][name] = getattr(module, "__file__", None)
            config = getattr(ktc, "ModelConfig", None)
            if not callable(config) or not callable(getattr(config, "evaluate", None)):
                raise RuntimeError("Kneron SDK must provide ModelConfig.evaluate")
            if optimizer is not None:
                if not callable(getattr(optimizer, "optimize", None)):
                    raise RuntimeError("Installed kneronnxopt has no callable optimize API")
                report["optimizer_backend"] = "kneronnxopt.optimize"
            else:
                flow = getattr(getattr(ktc, "onnx_optimizer", None), "onnx2onnx_flow", None)
                if not callable(flow):
                    raise RuntimeError("Installed SDK has no supported ONNX optimizer API")
                report["optimizer_backend"] = "ktc.onnx_optimizer.onnx2onnx_flow"
            report["status"] = "PASS"
        except (Exception, SystemExit) as exc:
            report["error"] = "{}: {}".format(type(exc).__name__, exc)
            traceback.print_exc()
        report["exit_code"] = 0 if report["status"] == "PASS" else EXIT_CODES["unknown"]
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("Kneron runtime preflight: {}".format(report["status"]))
    print("Interpreter: {} (Python {})".format(report["python_executable"], report["python_version"]))
    if report["status"] == "PASS":
        print("Optimizer API: {}".format(report["optimizer_backend"]))
    else:
        print("Reason: {}".format(report["error"]))
    print("Runtime receipt: {}".format(output))
    return report["exit_code"]


def _validate_manifest(path, source):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {"schema_version": 1, "purpose": "architecture_feasibility_only",
                "trained": False, "input_shape": [1, 3, 256, 256],
                "output_shape": [1], "opset": 11, "status": "COMPLETE"}
    if not isinstance(manifest, dict):
        raise ValueError("Candidate manifest must be an object")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError("Candidate manifest contract mismatch: {}".format(key))
    if manifest.get("architecture") not in MODEL_NAMES:
        raise ValueError("Unknown candidate architecture")
    if manifest.get("onnx_file") != source.name:
        raise ValueError("Candidate ONNX filename mismatch")
    if manifest.get("onnx_sha256") != sha256_file(source):
        raise ValueError("Candidate ONNX SHA256 mismatch")
    if not isinstance(manifest.get("parity"), dict) or manifest["parity"].get("status") != "PASS":
        raise ValueError("Candidate export has no passing ONNX parity record")
    return manifest


def _validate_trained_export(path, source, receipt_hash, graph_hash):
    export = json.loads(path.read_text(encoding="utf-8-sig"))
    case = (export["architecture"], export["fold"])
    if case not in TRAINED_EXPORTS or TRAINED_EXPORTS[case][0] != receipt_hash:
        raise ValueError("Trained export is outside the five verified campaign exports")
    if (export["parity_pass"] is not True or export["calibration_arrays"] != 192
            or export["parity_atol"] != 1e-5 or export["parity_rtol"] != 1e-4
            or export["dataset_role"] != "development_train_ptq_calibration"):
        raise ValueError("Original strict 192-input training-source parity gate failed")
    if (PureWindowsPath(export["onnx"]["path"]).name != source.name
            or export["onnx"]["sha256"] != graph_hash):
        raise ValueError("Trained ONNX differs from its strict export")
    calibration = path.with_name("calibration_manifest.csv")
    if sha256_file(calibration) != export["calibration"]["sha256"]:
        raise ValueError("Calibration manifest differs from the strict export")
    with calibration.open(encoding="utf-8-sig", newline="") as handle:
        expected = {row["sample_id"]: row["source_name"] for row in csv.DictReader(handle)}
    train = set(export["source_split"]["train_sources"])
    if (len(expected) != 192 or set(expected.values()) != train
            or train.intersection(export["source_split"]["test_sources"])):
        raise ValueError("Calibration members differ from the training sources")
    parity = path.with_name("onnx_parity.csv")
    parity_hash = sha256_file(parity)
    if parity_hash != TRAINED_PARITY_SHA256[case]:
        raise ValueError("Parity table differs from the frozen trained export")
    with parity.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if expected.pop(row["sample_id"], None) != row["source_name"]:
                raise ValueError("Unmatched parity sample identity")
            reference, actual = float(row["pytorch"]), float(row["onnx"])
            # The original exporter uses np.allclose(pytorch, onnx, atol, rtol).
            if (row["parity_pass"] != "True" or not all(map(math.isfinite, (reference, actual)))
                    or abs(reference - actual) > 1e-5 + 1e-4 * abs(actual)):
                raise ValueError("Failed strict PyTorch/ONNX parity for {}".format(row["sample_id"]))
    if expected or sha256_file(parity) != parity_hash:
        raise ValueError("Incomplete or changed parity evidence")
    return {"architecture": case[0], "fold": case[1], "model_id": TRAINED_EXPORTS[case][1],
            "training_run_sha256": export["training_run"]["sha256"],
            "checkpoint_sha256": export["checkpoint"]["sha256"],
            "calibration_manifest_sha256": export["calibration"]["sha256"],
            "original_fp32_parity_pass": True, "original_fp32_parity_sha256": parity_hash}


def run_screen(args):
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    work = output / "work"
    work.mkdir()
    source = Path(args.input).resolve()
    trained = bool(getattr(args, "trained_export", None))
    manifest_path = Path(args.trained_export if trained else args.candidate_manifest).resolve()
    manifest_digest = "export_receipt_sha256" if trained else "candidate_manifest_sha256"
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "trained_fp32_exact_graph_feasibility_only" if trained else "architecture_feasibility_only",
        "trained": trained,
        "platform": args.platform, "model_id": args.model_id, "model_version": args.model_version,
        "architecture": None, "support_status": "unknown", "reasons": [],
        "source_onnx_sha256": None, manifest_digest: None,
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "toolchain_image_id": args.toolchain_image_id,
        "python_version": platform.python_version(), "python_executable": sys.executable,
        "ptq_performed": False, "bie_generated": False, "nef_compiled": False,
        "physical_inference_performed": False, "biological_accuracy_evaluated": False,
        "nasa_data_read": False, "artifacts": [], "rejected_stale_artifacts": [],
        "vendor_evaluation_completed": False,
    }
    error, evaluation, context = None, "", {"nodes": []}
    vendor_path = output / "vendor.log"
    evaluation_path = output / "evaluation.log"
    previous_cwd = Path.cwd()
    started = time.monotonic()
    try:
        report["source_onnx_sha256"] = sha256_file(source)
        report[manifest_digest] = sha256_file(manifest_path)
        if trained:
            report.update(_validate_trained_export(manifest_path, source, report[manifest_digest],
                                                   report["source_onnx_sha256"]))
            if args.toolchain_image_id != "sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041":
                raise ValueError("Trained export requires the pinned campaign image")
            if args.model_id is not None and report["model_id"] != args.model_id:
                raise ValueError("Model ID differs from the trained campaign")
        else:
            manifest = _validate_manifest(manifest_path, source)
            report["architecture"] = manifest["architecture"]
        if args.platform != "720":
            raise ValueError("This adapter recognizes KL720 evidence only")
        if report["model_id"] <= 32768 or not re.fullmatch(r"[0-9a-fA-F]{4}", args.model_version):
            raise ValueError("Custom model ID must exceed 32768; version must be four hex digits")
        with _vendor_log(vendor_path):
            os.chdir(str(work))
            try:
                onnx, ktc, optimizer = _load_runtime()
                report["dependency_versions"] = {
                    "onnx": getattr(onnx, "__version__", None),
                    "ktc": getattr(ktc, "__version__", None),
                    "kneronnxopt": getattr(optimizer, "__version__", None),
                }
                report["toolchain_versions"] = {
                    str(path): path.read_text(encoding="utf-8", errors="replace").strip()
                    for path in (Path("/workspace/version.txt"), Path("/workspace/VERSION"))
                    if path.is_file()
                }
                model = onnx.load(str(source), load_external_data=False)
                if any(value.external_data for value in model.graph.initializer):
                    raise ValueError("Candidate must be a self-contained ONNX file")
                if [(op.domain, op.version) for op in model.opset_import] != [("", 11)]:
                    raise ValueError("Candidate must use the fixed standard ONNX opset 11")
                onnx.checker.check_model(model)
                before_graph = graph_summary(model)
                if (len(before_graph["inputs"]) != 1 or before_graph["inputs"][0]["shape"] != [1, 3, 256, 256]
                        or len(before_graph["outputs"]) != 1 or before_graph["outputs"][0]["shape"] != [1]):
                    raise ValueError("Source graph does not implement the full NASA input/scalar-output contract")
                report["original_graph"] = before_graph
                if trained and (before_graph["inputs"][0]["name"] != "input"
                                or before_graph["outputs"][0]["name"] != "burden"):
                    raise ValueError("Trained graph has an unexpected tensor name")
                if optimizer is not None:
                    model = optimizer.optimize(model)
                    report["optimizer_backend"] = "kneronnxopt.optimize"
                else:
                    flow = getattr(getattr(ktc, "onnx_optimizer", None), "onnx2onnx_flow", None)
                    if flow is None:
                        raise RuntimeError("Installed SDK has no supported ONNX optimizer API")
                    model = flow(model)
                    report["optimizer_backend"] = "ktc.onnx_optimizer.onnx2onnx_flow"
                onnx.checker.check_model(model)
                optimized_graph = graph_summary(model)
                if (len(optimized_graph["inputs"]) != 1 or optimized_graph["inputs"][0]["shape"] != [1, 3, 256, 256]
                        or len(optimized_graph["outputs"]) != 1 or _numel(optimized_graph["outputs"][0]["shape"]) != 1):
                    raise ValueError("Optimized graph changed the NASA input/scalar-output contract")
                report["optimized_graph"] = optimized_graph
                optimized_path = output / "optimized.onnx"
                onnx.save(model, str(optimized_path))
                report["optimized_onnx_sha256"] = sha256_file(optimized_path)
                context = graph_cpu_context(model, onnx)
                roots = [work, Path("/data1/kneron_flow"), Path("/data1/compiler"),
                         Path("/workspace/.tmp"), Path("/tmp")]
                snapshot = snapshot_artifacts(roots)
                evaluation_started_ns = time.time_ns()
                try:
                    with _vendor_log(evaluation_path):
                        try:
                            config = ktc.ModelConfig(report["model_id"], args.model_version, args.platform, onnx_model=model)
                            evaluation_value = config.evaluate()
                            evaluation = "" if evaluation_value is None else str(evaluation_value)
                            print(evaluation)
                            report["vendor_evaluation_completed"] = True
                        except (Exception, SystemExit):
                            traceback.print_exc()
                            raise
                finally:
                    # Retain a complete diagnostic pipeline log, but only the
                    # separately bounded evaluation log is classification evidence.
                    if evaluation_path.is_file():
                        print(evaluation_path.read_text(encoding="utf-8", errors="replace"))
                    report["artifacts"], report["rejected_stale_artifacts"] = harvest_artifacts(
                        roots, snapshot, evaluation_started_ns, output / "artifacts")
            except (Exception, SystemExit):
                traceback.print_exc()
                raise
    except (Exception, SystemExit) as exc:
        error = "{}: {}".format(type(exc).__name__, exc)
        report["error"] = error
    finally:
        os.chdir(str(previous_cwd))
    report["evaluation_return"] = evaluation
    evidence_artifacts = list(report["artifacts"])
    if vendor_path.is_file():
        report["vendor_log"] = dict(_identity(vendor_path), path=str(vendor_path))
    if evaluation_path.is_file():
        report["evaluation_log"] = dict(_identity(evaluation_path), path=str(evaluation_path))
        evidence_artifacts.append({"name": "evaluation.log", "path": str(evaluation_path)})
    report.update(classify_evidence(evaluation, evidence_artifacts, context,
                                   report["vendor_evaluation_completed"], error))
    # The input mount is normally read-only, but direct script usage receives
    # the same final immutability check.
    try:
        if (sha256_file(source) != report["source_onnx_sha256"]
                or sha256_file(manifest_path) != report[manifest_digest]):
            raise RuntimeError("Source graph or candidate manifest changed during screening")
        if report.get("optimized_onnx_sha256") and sha256_file(output / "optimized.onnx") != report["optimized_onnx_sha256"]:
            raise RuntimeError("Optimized graph changed during screening")
        if trained and report.get("original_fp32_parity_pass"):
            if (sha256_file(manifest_path.with_name("calibration_manifest.csv")) != report["calibration_manifest_sha256"]
                    or sha256_file(manifest_path.with_name("onnx_parity.csv")) != report["original_fp32_parity_sha256"]):
                raise RuntimeError("Trained export calibration or parity evidence changed during screening")
    except Exception as exc:
        report.update(support_status="unknown", reasons=["input_identity_changed_or_unavailable"],
                      error="{}: {}".format(type(exc).__name__, exc))
    report["duration_seconds"] = time.monotonic() - started
    report["exit_code"] = EXIT_CODES[report["support_status"]]
    _write_json(output / "screen.json", report)
    print("NASA candidate {}: {}".format(report["architecture"] or "unverified", report["support_status"]))
    print("Reasons: {}".format(", ".join(report["reasons"])))
    print("Report: {}".format(output / "screen.json"))
    return report["exit_code"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    provenance = parser.add_mutually_exclusive_group()
    provenance.add_argument("--candidate-manifest")
    provenance.add_argument("--trained-export", help="Frozen campaign export.json beside its calibration and parity CSVs")
    parser.add_argument("--out", required=True,
                        help="New screen output directory, or new JSON file for --probe-runtime")
    parser.add_argument("--probe-runtime", action="store_true",
                        help="Check installed SDK imports and APIs without evaluating a model")
    parser.add_argument("--platform", default="720")
    parser.add_argument("--model-id", type=int, help="Defaults to the campaign model ID for trained exports or 32771 for candidates")
    parser.add_argument("--model-version", default="0001")
    parser.add_argument("--toolchain-image-id", default=None)
    args = parser.parse_args(argv)
    if args.model_id is None and not args.trained_export:
        args.model_id = 32771
    if not args.probe_runtime and (not args.input or not (args.trained_export or args.candidate_manifest)):
        parser.error("Model screening requires --input and either --candidate-manifest or --trained-export")
    try:
        if args.probe_runtime:
            return probe_runtime(args)
        return run_screen(args)
    except (OSError, ValueError) as exc:
        print("Cannot create fresh screening evidence: {}".format(exc), file=sys.stderr)
        return EXIT_CODES["unknown"]


if __name__ == "__main__":
    raise SystemExit(main())
