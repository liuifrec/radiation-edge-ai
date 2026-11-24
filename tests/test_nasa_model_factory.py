"""Registry checks run in minimal CI; numerical checks use optional torch."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from radiation_edge_ai.nasa_bps import model_factory as factory


def test_registry_and_contract_work_without_optional_training_dependencies():
    code = """
import builtins
original_import = builtins.__import__
def restricted(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision', 'kp', 'onnx', 'cv2'}:
        raise AssertionError('optional dependency imported: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = restricted
from radiation_edge_ai.nasa_bps.model_factory import list_model_specs, model_contract
import json
print(json.dumps([model_contract(x.architecture) for x in list_model_specs()]))
"""
    source = str(Path(factory.__file__).resolve().parents[2])
    program = f"import sys; sys.path.insert(0, {source!r})\n" + code
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    contracts = json.loads(result.stdout)
    assert {value["architecture"] for value in contracts} == set(factory.MODEL_NAMES)
    assert all(value["input"]["deployment_shape"] == [1, 3, 256, 256] for value in contracts)
    assert all(value["output"]["reference_label"] == "sample_plate_well_avg_nfoci"
               for value in contracts)
    assert all(value["initialization"]["pretrained_backbone"] is False for value in contracts)


def test_invalid_model_and_pretrained_selection_rejected_before_import():
    with pytest.raises(ValueError, match="Unknown NASA architecture"):
        factory.create_model("unknown")
    with pytest.raises(ValueError, match="no pretrained"):
        factory.create_model("r1countnet_v1", pretrained=True)
    with pytest.raises(ValueError, match="explicit boolean"):
        factory.model_contract("resnet18", pretrained="false")
    assert factory.model_contract("mobilenet_v2", pretrained=True)["initialization"][
        "pretrained_weights"
    ] == "MobileNet_V2_Weights.IMAGENET1K_V1"


def test_checkpoint_fingerprint_tracks_actual_bytes_without_deserializing(tmp_path):
    path = tmp_path / "model.pt"
    original = b"opaque checkpoint bytes, not a pickle"
    path.write_bytes(original)
    assert factory.checkpoint_identity(path) == {
        "sha256": hashlib.sha256(original).hexdigest(), "size_bytes": len(original)
    }
    path.write_bytes(original + b"changed")
    assert factory.checkpoint_identity(path)["sha256"] != hashlib.sha256(original).hexdigest()


def _torch():
    return pytest.importorskip("torch", reason="optional numerical model test needs torch")


def _require_torchvision():
    if importlib.util.find_spec("torchvision") is None:
        pytest.skip("optional candidate model test needs torchvision")
    try:
        import torchvision  # noqa: F401
    except (ImportError, RuntimeError) as exc:
        pytest.fail(f"Installed torchvision is incompatible: {exc}")


@pytest.mark.parametrize("architecture", factory.MODEL_NAMES)
def test_common_scalar_output_and_gradient_contract(architecture):
    torch = _torch()
    if architecture != "r1countnet_v1":
        _require_torchvision()
    torch.set_num_threads(1)
    torch.manual_seed(720)
    model = factory.create_model(architecture)
    model.train()
    images = torch.rand(2, 3, 256, 256)
    images[:, 2] = 0
    prediction = model(images)
    assert prediction.shape == (2,)
    assert torch.isfinite(prediction).all()
    assert (prediction >= 0).all()
    # Aggregate supervision must propagate into the backbone, not just its head.
    loss = (prediction.mean() - 1.5).square()
    loss.backward()
    gradients = [p.grad for p in model.features.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g).item() > 0 for g in gradients)
    model.eval()
    with torch.no_grad():
        assert model(images[:1]).shape == (1,)


def test_baseline_matches_frozen_architecture_state_and_predictions():
    torch = _torch()
    torch.set_num_threads(1)
    # Execute only the two historical class definitions. Importing the entire
    # frozen training module would unnecessarily require its OpenCV data stack.
    historical = Path(factory.__file__).with_name("train_pilot_v1_r1_mil.py")
    source = ast.parse(historical.read_text(encoding="utf-8"))
    classes = [
        node for node in source.body
        if isinstance(node, ast.ClassDef) and node.name in {"ConvStage", "R1CountNetV1"}
    ]
    assert len(classes) == 2
    namespace = {"nn": torch.nn, "torch": torch}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(historical), "exec"), namespace)
    torch.manual_seed(123)
    reference = namespace["R1CountNetV1"]().eval()
    torch.manual_seed(123)
    candidate = factory.create_model("r1countnet_v1").eval()
    assert sum(p.numel() for p in candidate.parameters()) == 215833
    old_state, new_state = reference.state_dict(), candidate.state_dict()
    assert old_state.keys() == new_state.keys()
    for key in old_state:
        assert torch.equal(old_state[key], new_state[key]), key
    # Existing state dictionaries must load strictly, without remapping keys.
    candidate.load_state_dict(old_state, strict=True)
    images = torch.rand(2, 3, 256, 256)
    images[:, 2] = 0
    with torch.no_grad():
        assert torch.equal(reference(images), candidate(images))
