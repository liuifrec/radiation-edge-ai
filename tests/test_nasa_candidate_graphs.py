"""Early graph screening exports real full models without NASA data or the SDK."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from radiation_edge_ai.nasa_bps import candidate_graphs as graphs
from radiation_edge_ai.nasa_bps import model_factory


def test_help_imports_no_scientific_or_vendor_runtime():
    program = """
import builtins
original_import = builtins.__import__
def restricted(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'torch', 'torchvision', 'onnx', 'onnxruntime', 'kp', 'ktc', 'cv2'}:
        raise AssertionError('optional runtime imported: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = restricted
from radiation_edge_ai.nasa_bps.candidate_graphs import main
main(['--help'])
"""
    source = str(Path(graphs.__file__).resolve().parents[2])
    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, {!r})\n{}".format(source, program)],
        text=True, capture_output=True, check=True,
    )
    assert "--out" in result.stdout
    assert all(name in result.stdout for name in model_factory.MODEL_NAMES)


@pytest.mark.parametrize("models,seed", [
    ([], 720),
    (["r1countnet_v1", "r1countnet_v1"], 720),
    (["unknown"], 720),
    (["r1countnet_v1"], -1),
    (["r1countnet_v1"], 2**32),
    (["r1countnet_v1"], True),
])
def test_invalid_requests_fail_before_creating_output_or_loading_runtime(tmp_path, monkeypatch, models, seed):
    def unexpected_runtime():
        pytest.fail("Invalid requests must be rejected before loading scientific dependencies")
    monkeypatch.setattr(graphs, "_runtime", unexpected_runtime)
    out = tmp_path / "invalid"
    with pytest.raises(ValueError):
        graphs.export_candidates(out, models, seed)
    assert not out.exists()


def test_existing_directory_is_rejected_without_modifying_its_contents(tmp_path, monkeypatch):
    sentinel = tmp_path / "sentinel.bin"
    sentinel.write_bytes(b"existing frozen evidence")
    monkeypatch.setattr(graphs, "_runtime", lambda: pytest.fail("Overwrite check must come first"))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        graphs.export_candidates(tmp_path)
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"existing frozen evidence"


def test_failed_export_leaves_failed_campaign_and_restores_cpu_threads(tmp_path, monkeypatch):
    class FakeTorch:
        threads = 7

        @classmethod
        def get_num_threads(cls):
            return cls.threads

        @classmethod
        def set_num_threads(cls, count):
            cls.threads = count

    monkeypatch.setattr(graphs, "_runtime", lambda: (None, FakeTorch, None, None))

    def fail_export(*args, **kwargs):
        raise RuntimeError("Unsupported synthetic export fault")

    monkeypatch.setattr(graphs, "_export_one", fail_export)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="synthetic export fault"):
        graphs.export_candidates(out, ["r1countnet_v1"])
    manifest = json.loads((out / "candidate_graphs.json").read_text())
    assert manifest["status"] == "FAILED"
    assert manifest["models"] == []
    assert manifest["failure"]["architecture"] == "r1countnet_v1"
    assert FakeTorch.threads == 7


def _science(require_torchvision=False):
    pytest.importorskip("torch", reason="Optional real graph test needs the NASA scientific environment")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    if require_torchvision:
        if importlib.util.find_spec("torchvision") is None:
            pytest.skip("Optional full-model export needs torchvision")
        try:
            import torchvision  # noqa: F401
        except (ImportError, RuntimeError) as exc:
            pytest.fail("Installed torchvision is incompatible: {}".format(exc))


@pytest.fixture(scope="module")
def real_exports(tmp_path_factory):
    _science(require_torchvision=True)
    out = tmp_path_factory.mktemp("candidate-parent") / "graphs"
    index = graphs.export_candidates(out)
    return out, index


@pytest.mark.parametrize("architecture", model_factory.MODEL_NAMES)
def test_real_complete_model_exports_and_synthetic_parity(real_exports, architecture):
    out, index = real_exports
    assert index["status"] == "COMPLETE"
    assert index["requested_models"] == list(model_factory.MODEL_NAMES)
    entry = next(item for item in index["models"] if item["architecture"] == architecture)
    manifest_bytes = (out / entry["candidate_manifest"]).read_bytes()
    manifest = json.loads(manifest_bytes)
    assert hashlib.sha256(manifest_bytes).hexdigest() == entry["candidate_manifest_sha256"]
    assert hashlib.sha256((out / entry["onnx_file"]).read_bytes()).hexdigest() == entry["onnx_sha256"]
    assert manifest["onnx_sha256"] == entry["onnx_sha256"]
    assert manifest["run_id"] == index["run_id"]
    assert manifest["input_shape"] == [1, 3, 256, 256]
    assert manifest["output_shape"] == [1]
    assert manifest["opset"] == 11
    assert manifest["status"] == "COMPLETE"
    assert manifest["trained"] is False
    assert manifest["pretrained_weights"] is False
    assert manifest["purpose"] == "architecture_feasibility_only"
    assert manifest["model_contract"]["initialization"]["pretrained_backbone"] is False
    assert manifest["graph"]["operators"]["Conv"] >= 6
    # Whole-model export must retain the actual learnable scalar regression head.
    import onnx
    actual_graph = onnx.load(str(out / entry["onnx_file"]))
    initializer_shapes = {item.name: list(item.dims) for item in actual_graph.graph.initializer}
    expected_features = {"r1countnet_v1": 128, "mobilenet_v2": 1280, "resnet18": 512}
    assert initializer_shapes["head.weight"] == [1, expected_features[architecture]]
    assert initializer_shapes["head.bias"] == [1]
    assert manifest["onnx_checker"] == "PASS"
    assert manifest["parity"]["status"] == "PASS"
    probes = manifest["parity"]["probes"]
    assert len(probes) == 3
    assert len({item["input_sha256"] for item in probes}) == 3
    assert all(item["passed"] and item["third_channel_is_zero"] for item in probes)
    if manifest["parity"]["constant_reference_outputs"]:
        assert manifest["parity"]["reference_output_span"] == 0
        assert "weak numerical diagnostic" in manifest["parity"]["diagnostic_note"]
    assert manifest["nasa_data_read"] is False
    assert manifest["biological_accuracy_evaluated"] is False
    assert manifest["kneron_support_evaluated"] is False
    assert manifest["physical_npu_executed"] is False
    assert manifest["trained_graph_requires_new_export_and_screen"] is True


def test_seeded_export_reproduces_graph_bytes_and_does_not_change_torch_rng(real_exports, tmp_path):
    import torch
    out, index = real_exports
    torch.manual_seed(555)
    expected_rng = torch.random.get_rng_state().clone()
    second = graphs.export_candidates(tmp_path / "repeat", ["r1countnet_v1"], seed=720)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    first = next(item for item in index["models"] if item["architecture"] == "r1countnet_v1")
    assert second["models"][0]["onnx_sha256"] == first["onnx_sha256"]
    first_report = json.loads((out / first["candidate_manifest"]).read_text())
    second_report = json.loads((tmp_path / "repeat" / second["models"][0]["candidate_manifest"]).read_text())
    assert first_report["parity"] == second_report["parity"]


def test_real_parity_failure_cannot_produce_complete_candidate(tmp_path, monkeypatch):
    _science()
    import onnxruntime as ort
    actual_session = ort.InferenceSession

    class CorruptSession:
        def __init__(self, *args, **kwargs):
            self.session = actual_session(*args, **kwargs)

        def run(self, *args, **kwargs):
            result = self.session.run(*args, **kwargs)
            return [result[0] + 100]

    monkeypatch.setattr(ort, "InferenceSession", CorruptSession)
    out = tmp_path / "parity-fault"
    with pytest.raises(RuntimeError, match="parity failed"):
        graphs.export_candidates(out, ["r1countnet_v1"])
    index = json.loads((out / "candidate_graphs.json").read_text())
    assert index["status"] == "FAILED"
    assert not (out / "r1countnet_v1" / "candidate.json").exists()
