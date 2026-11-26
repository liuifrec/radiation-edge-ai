import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from radiation_edge_ai.hardware import pretrained_models as registry
from radiation_edge_ai.hardware.pretrained_models import (
    STAGE_BINDINGS,
    model_record,
    qualified_nef,
    verified_artifact,
)
from radiation_edge_ai.hardware.spotiflow_transport import check_device, registered_kn


def test_two_weight_sets_share_adapter_and_have_separate_qualifications():
    general = model_record("spotiflow/general")
    synth = model_record("spotiflow/synth_complex")
    assert general["adapter"] == synth["adapter"]
    assert general["artifacts"]["best.pt"]["sha256"] != synth["artifacts"]["best.pt"]["sha256"]
    assert general["threshold"] != synth["threshold"]
    assert general["scientific"]["nasa_independent_biological_validation"] == "not_established"


def test_unqualified_or_changed_artifact_cannot_be_loaded(tmp_path):
    card = deepcopy(model_record("spotiflow/general"))
    entry = card["artifacts"]["best.pt"]
    path = tmp_path / entry["path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verified_artifact(card, "best.pt", tmp_path)
    card["engineering"]["native"] = {"status": "failed"}
    with pytest.raises(ValueError, match="native"):
        qualified_nef(card, tmp_path)


def test_alternate_boundary_uses_its_own_graph_and_scientific_history():
    original = model_record("spotiflow/general")
    split = model_record("spotiflow/general", variant="separate_heads")
    assert original["artifacts"]["best.pt"] == split["artifacts"]["best.pt"]
    assert original["scientific"] == split["scientific"]
    assert original["artifacts"]["onnx"] != split["artifacts"]["onnx"]
    assert original["artifacts"]["onnx_parity_receipt"] != split["artifacts"]["onnx_parity_receipt"]
    assert "head_substitution" not in split["artifacts"]
    with pytest.raises(KeyError):
        model_record("spotiflow/synth_complex", variant="separate_heads")
    higher = model_record("spotiflow/general", variant="separate_heads_int16")
    assert higher["artifacts"]["onnx"] == split["artifacts"]["onnx"]
    assert higher["artifacts"]["bie"] != split["artifacts"]["bie"]
    assert higher["analysis_overrides"] == {"model_out_bitwidth_mode": "int16"}


def test_even_hash_verified_receipts_must_qualify_same_nef(tmp_path):
    # Construct two small, locally verifiable receipts for different NEFs.
    def save(name, content):
        (tmp_path / name).write_bytes(content)
        return {"path": name, "sha256": hashlib.sha256(content).hexdigest()}

    card = model_record("spotiflow/general", variant="separate_heads")
    artifacts = card["artifacts"]
    artifacts["nef"] = save("model.nef", b"test fixture")
    artifacts["bie"] = save("bie", b"bie")
    cases = [{"case_id": str(i), "role": "readout", "tensor_sha256": f"input-{i}"} for i in range(63)]
    artifacts["panel"] = save("panel", json.dumps({"rows": cases}).encode())
    decisions = []
    packed = {"available": True, "sha256": hashlib.sha256(b"packed fixture input").hexdigest()}
    for key, stage in (("bie_outputs", "bie_readout"), ("nef_outputs", "nef_simulator")):
        rows = [{"case_id": c["case_id"], "input_tensor_sha256": c["tensor_sha256"],
                 "raw_heads_sha256": key + c["case_id"], "sdk_packed_input": packed} for c in cases]
        manifest = {"stage": stage, "model": "general", "rows": rows, "panel_sha256": artifacts["panel"]["sha256"],
                    "nef_sha256": artifacts["nef"]["sha256"], "bie_sha256": artifacts["bie"]["sha256"]}
        artifacts[key] = save(key, json.dumps(manifest).encode())
    for case in cases:
        key = case["case_id"]
        decisions.append({"case_id": key, "passed": True, "within_tolerance": True,
                          "input_tensor_sha256": case["tensor_sha256"], "sdk_packed_input": packed,
                          "bie_heads_sha256": "bie_outputs" + key, "nef_heads_sha256": "nef_outputs" + key})
    for stage, state in card["engineering"].items():
        state.update(status="passed", evidence="test_" + stage)
    stages = ["native", *STAGE_BINDINGS]
    for stage in stages:
        body = {"model": "general", "passed": True, "completed": True, "status": "complete",
                "classification": {"support_status": "supported"},
                "gate_sha256": card["provenance"]["gate_lf_sha256"],
                "models": {"general": {"threshold": card["threshold"],
                                       "files_sha256": {"best.pt": artifacts["best.pt"]["sha256"]}}}}
        body.update({field: artifacts[key]["sha256"] for field, key in STAGE_BINDINGS.get(stage, {}).items()})
        if stage == "nef_simulator_parity":
            body.update(stage=stage, cases=63, rows=decisions)
        artifacts["test_" + stage] = save(stage, json.dumps(body).encode())
        if stage == "onnx_parity":
            artifacts["onnx_parity_receipt"] = artifacts["test_" + stage]
        if stage == "bie_fidelity":
            artifacts["bie_gate"] = artifacts["test_" + stage]
        if stage == "nef_compilation":
            artifacts["compile_receipt"] = artifacts["test_" + stage]
    assert qualified_nef(card, tmp_path) == tmp_path / "model.nef"
    for invalid in (
        dict(body, stage="nef_compilation"),
        dict(body, rows=decisions[:-1]),
        dict(body, rows=decisions[:-1] + decisions[:1]),
        dict(body, rows=[dict(decisions[0], passed=False), *decisions[1:]]),
        dict(body, rows=[dict(decisions[0], nef_heads_sha256="other"), *decisions[1:]]),
        dict(body, rows=[dict(decisions[0], sdk_packed_input={}), *decisions[1:]]),
    ):
        artifacts["test_nef_simulator_parity"] = save("invalid", json.dumps(invalid).encode())
        with pytest.raises(ValueError):
            qualified_nef(card, tmp_path)
    body.update(nef_sha256="different NEF")
    artifacts["test_nef_simulator_parity"] = save("mismatched", json.dumps(body).encode())
    with pytest.raises(ValueError, match="registered nef"):
        qualified_nef(card, tmp_path)


def test_catalogue_without_scientific_runtime_or_hardware():
    # -S excludes site packages, including numpy, torch, ONNX and vendor bindings.
    source = str(Path(registry.__file__).resolve().parents[2])
    cmd = [sys.executable, "-B", "-S", "-c", (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from radiation_edge_ai.hardware.pretrained_models import main; "
        "raise SystemExit(main(['--list']))"
    ), source]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    entries = json.loads(result.stdout)["models"]
    assert len(entries) == 6
    assert all(m["artifact_availability"] == {"checked": False} for m in entries)
    assert all(m["scientific"]["nasa_independent_biological_validation"] == "not_established" for m in entries)
    passed = [m for m in entries if m["engineering"]["physical_kl720"] == "passed"]
    assert [(m["model"], m["variant"]) for m in passed] == [("spotiflow/general", "full_int16_transport")]
    assert passed[0]["contract"]["input"]["shape"] == [1, 1, 128, 128]
    assert passed[0]["requirements"]["qualified_device_runtime"] == "Kneron PLUS 3.2.0"


@pytest.mark.parametrize("path", [
    "../outside", "nested/../name", "/absolute", "//host/share", r"C:\absolute", "c:relative",
    r"back\slash", "file:stream", "folder/CON.txt", "folder.", "x ", "x//file", "a/./b", "til*e",
])
def test_ambiguous_or_outside_artifact_location_is_rejected(tmp_path, path):
    with pytest.raises(ValueError, match="artifact path"):
        verified_artifact({"artifacts": {"x": {"path": path, "sha256": "0" * 64}}}, "x", tmp_path)


def test_link_outside_data_root_is_rejected(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "asset").write_bytes(b"outside")
    link = folder / "redirect"
    if os.name == "nt":
        # Directory junctions work without Windows symlink privileges.
        import _winapi

        _winapi.CreateJunction(str(outside), str(link))
    else:
        link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="inside"):
        verified_artifact({"artifacts": {"x": {"path": "redirect/asset", "sha256": "0" * 64}}}, "x", folder)


def test_duplicate_catalogue_keys_and_versions_fail_closed(tmp_path):
    path = tmp_path / "catalogue.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate"):
        registry.catalogue_summary(catalog=path)
    for version in (0, True, "1", 2):
        path.write_text(json.dumps({"schema_version": version, "families": {}, "models": {"example": {}}}))
        with pytest.raises(ValueError, match="version"):
            registry.catalogue_summary(catalog=path)


@pytest.mark.parametrize("problem,expected", [
    ("bad_sha", "SHA256"), ("conflicting_path", "Ambiguous"), ("unknown_evidence", "evidence"),
    ("imagined_stage", "status"), ("physical_only", "prerequisite"), ("release_override", "released weights"),
    ("scientific", "biological"), ("variant_alias", "Ambiguous"),
])
def test_registry_schema_rejects_invalid_claims(tmp_path, problem, expected):
    tree = json.loads(registry.CATALOG.read_bytes())
    card = tree["models"]["spotiflow/general"]
    stage = card["engineering"]
    choice = "separate_heads"
    artifacts = card["artifacts"]
    if problem == "bad_sha":
        artifacts["archive"]["sha256"] = "wrong"
    elif problem == "conflicting_path":
        card["variants"][choice]["artifacts"]["alias"] = dict(artifacts["best.pt"], sha256="0"*64)
    elif problem == "unknown_evidence":
        stage["native"]["evidence"] = "unknown"
    elif problem == "imagined_stage":
        stage["native"]["status"] = "certified"
    elif problem == "physical_only":
        stage["physical_kl720"] = {"status": "passed", "evidence": "native_receipt"}
    elif problem == "release_override":
        card["variants"][choice]["artifacts"]["best.pt"] = dict(artifacts["archive"])
    elif problem == "scientific":
        card["scientific"] = {}
    else:
        card["default_variant"] = choice
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(tree), encoding="utf8")
    with pytest.raises(ValueError, match=expected):
        registry.catalogue_summary(catalog=path)


def test_cli_distinguishes_claims_presence_and_integrity(tmp_path, capsys, monkeypatch):
    model = "spotiflow/general"
    args = [model, "--variant", "full_int16_transport"]
    card = model_record(model, variant="full_int16_transport")
    item = card["artifacts"]["nef"]
    source = tmp_path / item["path"]
    source.parent.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Missing"):
        verified_artifact(card, "nef", tmp_path)
    source.write_bytes(b"synthetic, changed NEF")
    with monkeypatch.context() as m:
        m.setattr(registry, "sha256_file", lambda _: pytest.fail("A presence check must not hash files"))
        assert registry.main([*args, "--data-root", str(tmp_path)]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["engineering"]["physical_kl720"]["status"] == "passed"
    assert info["artifact_availability"]["artifacts"]["nef"] == "present_unverified"
    assert registry.main([*args, "--verify", str(tmp_path)]) == 1
    states = json.loads(capsys.readouterr().out)["artifact_availability"]["artifacts"]
    assert states["nef"] == "checksum_mismatch"
    assert states["bie"] == "missing"
    tree = json.loads(registry.CATALOG.read_bytes())
    family = tree["families"]["spotiflow_2d"]
    local = tmp_path / family["artifacts"]["software_license"]["path"]
    local.parent.mkdir(parents=True)
    local.write_bytes(b"synthetic license fixture")
    family["artifacts"]["software_license"]["sha256"] = hashlib.sha256(local.read_bytes()).hexdigest()
    catalogue = tmp_path / "test.json"
    catalogue.write_text(json.dumps(tree), encoding="utf8")
    digest = registry.sha256_file
    hashed = []

    def count_access(path):
        hashed.append(path)
        return digest(path)

    monkeypatch.setattr(registry, "sha256_file", count_access)
    report = registry.catalogue_summary(tmp_path, verify=True, catalog=catalogue)
    assert all(m["artifact_availability"]["artifacts"]["software_license"] == "verified" for m in report["models"])
    assert hashed.count(local) == 1


def test_hash_bound_paired_device_discovery(tmp_path):
    device = {"kn_number": "0x1234", "product_id": "0x720"}
    system = {"kn_number": 0x1234, "firmware_version": "test"}
    path = tmp_path / "discovery.json"
    raw = json.dumps({"device": device, "system_info": system}).encode()
    path.write_bytes(raw)
    record = {"artifacts": {"device_discovery": {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest()}}}
    kn = registered_kn(record, tmp_path)
    assert kn == 0x1234
    check_device(device, system, "test", expected_kn=kn)
    for wrong_device, wrong_system, firmware in [
        (dict(device, product_id=0x520), system, "test"),
        (dict(device, kn_number=0x1235), system, "test"),
        (device, dict(system, kn_number=0x1235), "test"),
        (device, system, "other"),
    ]:
        with pytest.raises(ValueError):
            check_device(wrong_device, wrong_system, firmware, expected_kn=kn)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        registered_kn(record, tmp_path)
