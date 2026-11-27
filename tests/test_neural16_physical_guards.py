"""Offline guard tests for the ResNet18 physical launcher."""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from radiation_edge_ai.nasa_bps.multimodel_ptq import load_json, save_json, sha256


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/windows"))
    import run_resnet_neural16_usb

    return run_resnet_neural16_usb


def test_rejects_an_unknown_dll_before_any_native_api(launcher, tmp_path, monkeypatch):
    package = tmp_path / "kp"
    dll = package / "lib/libkplus.dll"
    dll.parent.mkdir(parents=True)
    dll.write_bytes(b"unsupported test DLL")
    ref = tmp_path / "reference.json"
    save_json(ref, {"system_info": {"firmware_version": "test"}, "device": {"kn_number": "0x1"},
                    "plus_version": "3.2.0", "plus_dll_sha256": launcher.abi.DLL_SHA})
    monkeypatch.setattr(launcher, "HARDWARE_REF", ref.name)
    monkeypatch.setattr(launcher, "HARDWARE_REF_SHA", sha256(ref))
    kp = SimpleNamespace(__file__=str(package / "__init__.py"),
                         core=SimpleNamespace(get_version=lambda: pytest.fail("Unverified native DLL called")))
    with pytest.raises(ValueError, match="Unexpected PLUS native DLL"):
        launcher.preflight(SimpleNamespace(data=tmp_path), kp)


@pytest.mark.parametrize("scope,key,value", [
    ("row", "input_sha256", "foreign tensor"),
    ("row", "packed_sha256", "foreign packing"),
    ("run", "model_id", 32782),
    ("run", "model_version", "2"),
])
def test_full_replay_requires_exact_positive_smoke_provenance(launcher, tmp_path, scope, key, value):
    reference = dict(sample_id=launcher.transport.SMOKE, word=7520, input_sha256="tensor", sdk_input_sha256="pack")
    row = dict(sample_id=reference["sample_id"], passed=True, fixed_word=7520,
               input_sha256="tensor", packed_sha256="pack")
    device = {"product_id": "0x720", "kn_number": "1"}
    smoke = dict(stage="smoke", passed=True, physical_execution=True, nef_sha256="nef", sim_sha256="sim",
                 rows=[row], device=device, system={"kn_number": "1", "firmware_version": "firmware"},
                 model_id=32783, model_version="1")
    (row if scope == "row" else smoke)[key] = value
    file = tmp_path / "receipt.json"
    save_json(file, smoke)
    with pytest.raises(ValueError, match="different model|qualified positive witness"):
        launcher.passing_smoke(file, sha256(file), "nef", "sim", {}, None, 1, "firmware", reference)


def test_passes_immutable_authenticated_bytes_to_the_model_loader(launcher, tmp_path, monkeypatch):
    nef = tmp_path / "model.nef"
    nef.write_bytes(b"authenticated test model")
    qualified_bytes = nef.read_bytes()
    device_fields = dict(product_id="0x720", kn_number="1")
    device = SimpleNamespace(usb_port_id=5, get_member_variable_dict=lambda: device_fields)
    system = SimpleNamespace(get_member_variable_dict=lambda: {"kn_number": "1", "firmware_version": "firmware"})
    loaded = []

    def preflight(args, kp):
        nef.write_bytes(b"subsequently changed model")
        return qualified_bytes, {}, [object()], 1, "firmware"

    def capture_load(group, payload):
        loaded.append(payload)
        raise RuntimeError("Synthetic loader reached")

    core = SimpleNamespace(connect_devices=lambda usb_port_ids: "mock group", set_timeout=lambda **kwargs: None,
                           get_system_info=lambda group, port: system, load_model=capture_load,
                           disconnect_devices=lambda group: None)
    monkeypatch.setitem(sys.modules, "kp", SimpleNamespace(core=core))
    monkeypatch.setattr(launcher, "preflight", preflight)
    monkeypatch.setattr(launcher.usb, "scan_descriptors", lambda kp: [device])
    monkeypatch.setattr(launcher.usb, "select_kl720", lambda descriptors, port: device)
    args = SimpleNamespace(data=tmp_path, out=tmp_path / "mock-receipt", mode="smoke", execute=True, port=5,
                           audit_sha256="A", compile_sha256="C", sim_sha256="S")
    with pytest.raises(RuntimeError, match="Synthetic loader reached"):
        launcher.run(args)
    assert loaded == [qualified_bytes] and type(loaded[0]) is bytes
    receipt = load_json(args.out / "receipt.json")
    assert receipt["nef_sha256"] == hashlib.sha256(qualified_bytes).hexdigest()
    assert receipt["nef_sha256"] != sha256(nef) and receipt["passed"] is False
