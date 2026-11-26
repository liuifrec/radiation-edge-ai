"""Offline evidence-contract tests; fake SDK outputs never claim hardware support."""

import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from radiation_edge_ai.nasa_bps import kneron_candidate_screen as screen


IO_SIGNATURE = {"inputs": [{"name": "input", "shape": [1, 3, 256, 256]}],
                "outputs": [{"name": "score", "shape": [1]}]}


def artifact(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(json.dumps(content) if isinstance(content, dict) else content, encoding="utf-8")
    return {"name": name, "path": str(path)}


def fresh_evaluation_artifact(name, content):
    evidence = artifact(Path.cwd(), name, content)
    # Immediate fake SDK writes can receive a filesystem timestamp just before
    # the evaluation clock. Set the fixture's intended freshness explicitly.
    fresh_ns = time.time_ns() + 1_000_000
    os.utime(evidence["path"], ns=(fresh_ns, fresh_ns))
    return evidence


def report(cpu="N/A", fps="123.4"):
    return {"kdp720/ip_eval/fps": fps, "kdp720/ip_eval/cpu_node": cpu}


def node(name, op, inputs=("features",), outputs=("result",)):
    return {"name": name, "op_type": op, "inputs": list(inputs), "outputs": list(outputs), "axis": 0}


def test_script_remains_python37_compatible_and_lazy():
    source = Path(screen.__file__).read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 7))
    tree = ast.parse(source)
    imports = [name.name for statement in tree.body if isinstance(statement, ast.Import) for name in statement.names]
    assert not {"onnx", "ktc", "torch", "kneronnxopt"}.intersection(imports)


@pytest.mark.parametrize("evaluation", [None, "", "None", "OK", "hardware_supported=True"])
def test_empty_or_unrecognized_success_is_unknown(evaluation):
    result = screen.classify_evidence(evaluation, [], {"nodes": []})
    assert result["support_status"] == "unknown"


def test_positive_report_requires_exact_target_platform(tmp_path):
    evidence = artifact(tmp_path, "model_fx_report.json", {
        "kdp730/ip_eval/fps": "12", "kdp730/ip_eval/cpu_node": "N/A"})
    assert screen.classify_evidence("", [evidence], {"nodes": []})["support_status"] == "unknown"


@pytest.mark.parametrize("fps", ["0", "-3", "NaN", "inf", "bad", None, True, False])
def test_invalid_performance_report_is_unknown(tmp_path, fps):
    evidence = artifact(tmp_path, "model_fx_report.json", report(fps=fps))
    assert screen.classify_evidence("", [evidence], {"nodes": []})["support_status"] == "unknown"


def test_documented_json_with_explicit_no_cpu_is_supported(tmp_path):
    evidence = artifact(tmp_path, "model_fx_report.json", report())
    result = screen.classify_evidence("", [evidence], {"nodes": []})
    assert result["support_status"] == "supported"
    assert result["positive_evidence"][0]["estimated_npu_fps"] == 123.4


def test_no_cpu_information_is_unknown_even_with_positive_fps(tmp_path):
    evidence = artifact(tmp_path, "model_fx_report.json", report(cpu=""))
    assert screen.classify_evidence("", [evidence], {"nodes": []})["support_status"] == "unknown"


def test_named_cpu_pooling_is_unsupported(tmp_path):
    evidence = artifact(tmp_path, "model_fx_report.json", report("AveragePool: /pool"))
    result = screen.classify_evidence("", [evidence], {"nodes": [node("/pool", "AveragePool")]})
    assert result["support_status"] == "unsupported"
    assert result["cpu_nodes"][0]["classification"] == "neural_compute"


def test_shape_gather_and_scalar_head_gather_are_proven_bookkeeping(tmp_path):
    context = {"nodes": [node("/shape_gather", "Gather", ("shape", "index"), ("dim",)),
                         node("/head_gather", "Gather", ("head", "index"), ("score",))],
               "shape_values": ["shape", "dim"], "constants": {"index": [0]},
               "shapes": {"head": [1, 1], "score": [1]}}
    evidence = artifact(tmp_path, "model_fx_report.json", report("Gather: /shape_gather, /head_gather"))
    result = screen.classify_evidence("", [evidence], context)
    assert result["support_status"] == "supported"
    assert [item["reason"] for item in result["cpu_nodes"]] == ["traced_shape_values_only", "scalar_identity_gather"]


def test_feature_gather_is_cpu_computation(tmp_path):
    context = {"nodes": [node("/gather", "Gather", ("feature", "index"))],
               "constants": {"index": [0]}, "shapes": {"feature": [1, 16], "result": [1]}}
    evidence = artifact(tmp_path, "model_fx_report.json", report("Gather: /gather"))
    assert screen.classify_evidence("", [evidence], context)["support_status"] == "unsupported"


def test_unknown_cpu_node_never_inherits_shape_operator_exemption(tmp_path):
    evidence = artifact(tmp_path, "model_fx_report.json", report("Reshape: /absent"))
    assert screen.classify_evidence("", [evidence], {"nodes": []})["support_status"] == "unknown"


def test_cpu_node_type_disagreement_is_unknown(tmp_path):
    evidence = artifact(tmp_path, "model_fx_report.json", report("Shape: /relu"))
    context = {"nodes": [node("/relu", "Relu")]}
    assert screen.classify_evidence("", [evidence], context)["support_status"] == "unknown"


def test_positive_legacy_text_needs_fresh_io_partition(tmp_path):
    evaluation = "[Evaluation Result]\nestimate FPS float = 22.5861\ntotal time = 44.2751 ms"
    assert screen.classify_evidence(evaluation, [], {"nodes": []})["support_status"] == "unknown"
    io = artifact(tmp_path, "ioinfo.csv", "i,0,input,3,256,256\no,0,score,1,1,1\n")
    context = dict(IO_SIGNATURE, nodes=[])
    assert screen.classify_evidence(evaluation, [io], context)["support_status"] == "supported"


@pytest.mark.parametrize("csv_text", [
    "i,0,other_input,3,64,64\no,0,other_output,1000,1,1\n",
    "i,0,input,3,64,64\no,0,score,1,1,1\n",
    "i,0,input,3,256,256\no,0,score,1000,1,1\n",
    "i,0,other_input,3,256,256\no,0,score,1,1,1\n",
    "i,0,input,3,256,256\no,0,other_output,1,1,1\n",
    "i,0,input,3,256,256\ni,0,input,3,256,256\no,0,score,1,1,1\n",
    "i,1,input,3,256,256\no,0,score,1,1,1\n",
])
def test_legacy_io_report_must_bind_whole_optimized_graph(tmp_path, csv_text):
    evaluation = "[Evaluation Result]\nestimate FPS float = 22.5861"
    io = artifact(tmp_path, "ioinfo.csv", csv_text)
    context = dict(IO_SIGNATURE, nodes=[])
    result = screen.classify_evidence(evaluation, [io], context)
    assert result["support_status"] == "unknown"
    assert "unrecognized_ioinfo_partition_report" in result["reasons"]


def test_fresh_unrelated_io_report_also_blocks_positive_json(tmp_path):
    evidence = artifact(tmp_path, "model_fx_report.json", report())
    io = artifact(tmp_path, "ioinfo.csv", "i,0,other_input,3,64,64\no,0,other_output,1000,1,1\n")
    result = screen.classify_evidence("", [evidence, io], dict(IO_SIGNATURE, nodes=[]))
    assert result["support_status"] == "unknown"


def test_unexplained_cpu_warning_blocks_positive_report(tmp_path):
    evidence = artifact(tmp_path, "model_fx_report.json", report())
    text = "***** Warning: this model has 1 CPU ops which may cause different fps *****"
    assert screen.classify_evidence(text, [evidence], {"nodes": []})["support_status"] == "unknown"


def test_cpu_count_deduplicates_graph_node_name_and_output_alias(tmp_path):
    context = dict(IO_SIGNATURE, nodes=[node("/shape", "Shape", ("input",), ("shape_output",))])
    evidence = artifact(tmp_path, "model_fx_report.json", report("Shape: /shape"))
    io = artifact(tmp_path, "ioinfo.csv", "i,0,input,3,256,256\nc,0,shape_output,1,1,4\no,0,score,1,1,1\n")
    text = "Warning: this model has 2 CPU ops"
    result = screen.classify_evidence(text, [evidence, io], context)
    assert result["support_status"] == "unknown"
    assert "cpu_warning_not_fully_explained_by_partition_evidence" in result["reasons"]


def test_hardware_rejection_wins_over_positive_result(tmp_path):
    evidence = artifact(tmp_path, "model_fx_report.json", report())
    result = screen.classify_evidence("HardwareNotSupport: unsupported Conv", [evidence], {"nodes": []})
    assert result["support_status"] == "unsupported"


def test_runtime_failure_does_not_reuse_partial_positive_result(tmp_path):
    evidence = artifact(tmp_path, "model_fx_report.json", report())
    result = screen.classify_evidence("", [evidence], {"nodes": []}, False, "RuntimeError: tool died")
    assert result["support_status"] == "unknown"


@pytest.mark.parametrize("failure", ["FAILURE for model", "failed to compile", "ERR: 4",
                                    "[compiler][error] failed to compile model", "ERR: -1"])
def test_fresh_generic_failure_overrides_positive_json(tmp_path, failure):
    evidence = artifact(tmp_path, "model_fx_report.json", report())
    log = artifact(tmp_path, "batch_compile.log", failure)
    result = screen.classify_evidence("", [evidence, log], {"nodes": []})
    assert result["support_status"] == "unknown"
    assert result["failure_evidence"]


def test_zero_errors_do_not_reject_a_positive_report(tmp_path):
    evidence = artifact(tmp_path, "model_fx_report.json", report())
    log = artifact(tmp_path, "batch_compile.log", "ERR: 0\n0 errors\n0 failures for model\nNo models failed to compile")
    result = screen.classify_evidence("", [evidence, log], {"nodes": []})
    assert result["support_status"] == "supported"
    assert result["failure_evidence"] == []


def test_stale_reports_never_fall_back_to_newest_file(tmp_path):
    source = tmp_path / "vendor"
    source.mkdir()
    stale = source / "model_fx_report.json"
    stale.write_text(json.dumps(report()), encoding="utf-8")
    old_time = time.time_ns() - 10_000_000_000
    os.utime(stale, ns=(old_time, old_time))
    snapshot = screen.snapshot_artifacts([source])
    started = time.time_ns()
    accepted, rejected = screen.harvest_artifacts([source], snapshot, started, tmp_path / "copies")
    assert accepted == []
    assert rejected[0]["reason"] == "not_fresh_for_this_evaluation"
    assert not (tmp_path / "copies").exists()


def test_new_vendor_artifact_is_copied_and_hash_bound(tmp_path):
    source = tmp_path / "vendor"
    source.mkdir()
    snapshot = screen.snapshot_artifacts([source])
    started = time.time_ns()
    evidence = artifact(source, "model_fx_report.json", report())
    os.utime(evidence["path"], ns=(started + 1_000_000, started + 1_000_000))
    accepted, rejected = screen.harvest_artifacts([source], snapshot, started, tmp_path / "copies")
    assert not rejected
    assert accepted[0]["sha256"] == screen.sha256_file(evidence["path"])
    assert Path(accepted[0]["path"]).is_file()


def inputs(tmp_path):
    source = tmp_path / "candidate.onnx"
    source.write_bytes(b"test-only-onnx-placeholder")
    manifest = tmp_path / "candidate.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "architecture": "r1countnet_v1",
        "purpose": "architecture_feasibility_only", "trained": False,
        "input_shape": [1, 3, 256, 256], "output_shape": [1], "opset": 11,
        "status": "COMPLETE", "parity": {"status": "PASS"},
        "onnx_file": source.name, "onnx_sha256": screen.sha256_file(source),
    }), encoding="utf-8")
    args = SimpleNamespace(input=str(source), candidate_manifest=str(manifest), out=str(tmp_path / "out"),
                           platform="720", model_id=32771, model_version="0001",
                           toolchain_image_id="sha256:test-fixture-only")
    return args, source, manifest


def read_screen(args):
    return json.loads((Path(args.out) / "screen.json").read_text(encoding="utf-8"))


def test_missing_sdk_still_writes_unknown_receipt(tmp_path, monkeypatch):
    args, source, manifest = inputs(tmp_path)

    def missing():
        raise ModuleNotFoundError("No module named 'ktc'")

    monkeypatch.setattr(screen, "_load_runtime", missing)
    assert screen.run_screen(args) == 3
    receipt = read_screen(args)
    assert receipt["support_status"] == "unknown"
    assert receipt["source_onnx_sha256"] == screen.sha256_file(source)
    assert receipt["candidate_manifest_sha256"] == screen.sha256_file(manifest)
    assert "No module named 'ktc'" in receipt["error"]
    assert receipt["vendor_log"]["sha256"] == screen.sha256_file(Path(args.out) / "vendor.log")


def test_manifest_hash_mismatch_stops_before_sdk(tmp_path, monkeypatch):
    args, source, _ = inputs(tmp_path)
    source.write_bytes(b"changed")
    monkeypatch.setattr(screen, "_load_runtime", lambda: pytest.fail("SDK must not run"))
    assert screen.run_screen(args) == 3
    assert "SHA256 mismatch" in read_screen(args)["error"]


def test_refuses_existing_output_without_touching_it(tmp_path):
    args, _, _ = inputs(tmp_path)
    Path(args.out).mkdir()
    marker = Path(args.out) / "existing.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        screen.run_screen(args)
    assert marker.read_text(encoding="utf-8") == "preserve"


class Dim:
    def __init__(self, value):
        self.dim_value = value

    def HasField(self, name):
        return name == "dim_value"


def tensor(name, shape):
    return SimpleNamespace(name=name, type=SimpleNamespace(tensor_type=SimpleNamespace(
        shape=SimpleNamespace(dim=[Dim(value) for value in shape]))))


def fake_runtime(monkeypatch, evaluation_callback, optimizer_callback=None, output_name="score"):
    model = SimpleNamespace(graph=SimpleNamespace(
        initializer=[], input=[tensor("input", [1, 3, 256, 256])], output=[tensor(output_name, [1])],
        value_info=[], node=[]), opset_import=[SimpleNamespace(domain="", version=11)])
    onnx = SimpleNamespace(__version__="test-fixture", load=lambda *a, **k: model,
                           checker=SimpleNamespace(check_model=lambda model: None),
                           shape_inference=SimpleNamespace(infer_shapes=lambda model: model),
                           save=lambda model, path: Path(path).write_bytes(b"optimized-test-only"))

    class Config:
        def __init__(self, model_id, version, platform, onnx_model):
            assert model_id > 32768 and version == "0001" and platform == "720"
            assert onnx_model is model

        def evaluate(self):
            return evaluation_callback()

    def optimize(model):
        if optimizer_callback:
            optimizer_callback()
        return model

    ktc = SimpleNamespace(ModelConfig=Config, onnx_optimizer=SimpleNamespace(onnx2onnx_flow=optimize))
    monkeypatch.setattr(screen, "_load_runtime", lambda: (onnx, ktc, None))


def trained_inputs(tmp_path, monkeypatch):
    source = tmp_path / "model.onnx"
    source.write_bytes(b"trained-test-placeholder")
    calibration = tmp_path / "calibration_manifest.csv"
    calibration.write_text("sample_id,source_name\n" + "".join(
        "n{},s{}\n".format(i, i % 4) for i in range(192)), encoding="utf-8")
    parity = tmp_path / "onnx_parity.csv"
    parity.write_text("sample_id,source_name,pytorch,onnx,parity_pass\n" + "".join(
        "n{},s{},1.0,1.0,True\n".format(i, i % 4) for i in range(192)), encoding="utf-8")
    manifest = tmp_path / "export.json"
    manifest.write_text(json.dumps({
        "architecture": "resnet18", "fold": "fold_female_role", "parity_pass": True,
        "calibration_arrays": 192, "parity_atol": 1e-5, "parity_rtol": 1e-4,
        "dataset_role": "development_train_ptq_calibration",
        "onnx": {"path": r"D:\exports\model.onnx", "sha256": screen.sha256_file(source)},
        "training_run": {"sha256": "a" * 64}, "checkpoint": {"sha256": "b" * 64},
        "calibration": {"sha256": screen.sha256_file(calibration)},
        "source_split": {"train_sources": ["s0", "s1", "s2", "s3"], "test_sources": ["s4", "s5"]},
    }), encoding="utf-8")
    monkeypatch.setitem(screen.TRAINED_EXPORTS, ("resnet18", "fold_female_role"),
                        (screen.sha256_file(manifest), 32783))
    monkeypatch.setitem(screen.TRAINED_PARITY_SHA256, ("resnet18", "fold_female_role"),
                        screen.sha256_file(parity))
    return ["--input", str(source), "--trained-export", str(manifest), "--out", str(tmp_path / "out"),
            "--toolchain-image-id", "sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041"]


def test_trained_screen_shares_evaluator_and_binds_export(tmp_path, monkeypatch):
    args = trained_inputs(tmp_path, monkeypatch)
    fake_runtime(monkeypatch, lambda: fresh_evaluation_artifact("model_fx_report.json", report()),
                 output_name="burden")
    assert screen.main(args) == 0
    result = json.loads((tmp_path / "out/screen.json").read_text(encoding="utf-8"))
    assert result["purpose"] == "trained_fp32_exact_graph_feasibility_only"
    assert result["trained"] and result["model_id"] == 32783 and result["original_fp32_parity_pass"]
    assert result["export_receipt_sha256"] == screen.sha256_file(tmp_path / "export.json")
    assert result["original_fp32_parity_sha256"] == screen.sha256_file(tmp_path / "onnx_parity.csv")


@pytest.mark.parametrize("original,replacement", [
    ("n0,s0,1.0,1.0,True", "n0,s0,1.0,1.1,True"),
    ("n0,s0,1.0,1.0,True", "n1,s1,1.0,1.0,True"),
    ("n0,s0,1.0,1.0,True", "n0,s4,1.0,1.0,True"),
])
def test_changed_trained_parity_blocks_vendor(tmp_path, monkeypatch, original, replacement):
    args = trained_inputs(tmp_path, monkeypatch)
    parity = tmp_path / "onnx_parity.csv"
    parity.write_text(parity.read_text().replace(original, replacement), encoding="utf-8")
    monkeypatch.setitem(screen.TRAINED_PARITY_SHA256, ("resnet18", "fold_female_role"),
                        screen.sha256_file(parity))
    monkeypatch.setattr(screen, "_load_runtime", lambda: pytest.fail("Parity must gate the SDK"))
    assert screen.main(args) == 3
    result = json.loads((tmp_path / "out/screen.json").read_text(encoding="utf-8"))
    assert not result["vendor_evaluation_completed"]
    assert ("mobilenet_v2", "fold_male_1_role") not in screen.TRAINED_EXPORTS


def test_complete_fake_sdk_flow_binds_fresh_result(tmp_path, monkeypatch):
    args, _, _ = inputs(tmp_path)

    def evaluate():
        fresh_evaluation_artifact("model_fx_report.json", report())
        print("fresh test-fixture evaluation")
        subprocess.run([sys.executable, "-c", "import os; os.write(2, b'native child log\\n')"], check=True)
        return None

    fake_runtime(monkeypatch, evaluate)
    assert screen.run_screen(args) == 0
    receipt = read_screen(args)
    assert receipt["vendor_evaluation_completed"]
    assert receipt["support_status"] == "supported"
    assert receipt["toolchain_image_id"] == "sha256:test-fixture-only"
    assert receipt["artifacts"][0]["sha256"]
    assert "native child log" in (Path(args.out) / "vendor.log").read_text(encoding="utf-8")
    assert "native child log" in (Path(args.out) / "evaluation.log").read_text(encoding="utf-8")
    assert not receipt["nef_compiled"] and not receipt["physical_inference_performed"]


def test_pre_evaluation_positive_log_cannot_supply_fresh_support(tmp_path, monkeypatch):
    args, _, _ = inputs(tmp_path)

    def optimize():
        print("[Evaluation Result]\nestimate FPS float = 99.3")

    def evaluate():
        fresh_evaluation_artifact("ioinfo.csv", "i,0,input,3,256,256\no,0,score,1,1,1\n")
        return None

    fake_runtime(monkeypatch, evaluate, optimize)
    assert screen.run_screen(args) == 3
    receipt = read_screen(args)
    assert receipt["support_status"] == "unknown"
    assert "estimate FPS" in (Path(args.out) / "vendor.log").read_text(encoding="utf-8")
    assert "estimate FPS" not in (Path(args.out) / "evaluation.log").read_text(encoding="utf-8")
    assert any(item["name"] == "ioinfo.csv" for item in receipt["artifacts"])
    assert receipt["positive_evidence"] == []


@pytest.mark.parametrize("error, expected", [(RuntimeError("tool crashed"), "unknown"),
                                              (RuntimeError("HardwareNotSupport: Conv"), "unsupported"),
                                              (SystemExit(9), "unknown")])
def test_vendor_failures_persist_receipts(tmp_path, monkeypatch, error, expected):
    args, _, _ = inputs(tmp_path)

    def evaluate():
        raise error

    fake_runtime(monkeypatch, evaluate)
    assert screen.run_screen(args) == screen.EXIT_CODES[expected]
    receipt = read_screen(args)
    assert receipt["support_status"] == expected
    assert not receipt["vendor_evaluation_completed"]


def test_vendor_source_mutation_invalidates_otherwise_positive_result(tmp_path, monkeypatch):
    args, source, _ = inputs(tmp_path)

    def evaluate():
        fresh_evaluation_artifact("model_fx_report.json", report())
        source.write_bytes(b"modified during vendor run")

    fake_runtime(monkeypatch, evaluate)
    assert screen.run_screen(args) == 3
    assert read_screen(args)["reasons"] == ["input_identity_changed_or_unavailable"]
