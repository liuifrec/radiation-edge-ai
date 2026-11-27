import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from radiation_edge_ai.nasa_bps import neural16_nef as gate
from radiation_edge_ai.nasa_bps import neural16_transport as transport
from radiation_edge_ai.nasa_bps.multimodel_bie_comparison import V1_LIMITS
from radiation_edge_ai.nasa_bps.multimodel_ptq import load_json, save_json, sha256


def policy_and_receipts():
    docs = Path(__file__).resolve().parents[1] / "docs"
    p = load_json(docs / gate.regression.POLICY_NAME)
    values = dict(V1_LIMITS)
    membership = {k: p[k] for k in ("cohort_membership_sha256", "contrast_membership_sha256")}
    comparison = dict(membership, metrics=values, decisions={k: {"pass": True, "actual": v, "maximum": v}
                      for k, v in values.items()}, all_declared_limits_pass=True, eligible_for_nef_compilation=True,
                      policy=p, scope=p["scope"], status="COMPARISON_COMPLETE")
    native = dict(membership, status="BIE_SIMULATION_COMPLETE", evaluated_nuclei=2400,
                  physical_execution=False, image_id=gate.IMAGE_ID)
    audit = dict(membership, status="AUDIT_COMPLETE", eligible_for_nef_compilation=True, bags=24, contrasts=12,
                 metrics=values.copy(), gate_decisions=dict.fromkeys(values, True),
                 native_jobs=2400, verified_native_words=2400)
    return p, comparison, native, audit


def test_all_eight_and_independent_word_audit_required():
    policy, comparison, native, audit = policy_and_receipts()
    gate.verified_audit(policy, comparison, native, audit)
    for change in ({"verified_native_words": 2399}, {"eligible_for_nef_compilation": False},
                   {"metrics": {}}, {"contrast_membership_sha256": "wrong"}):
        with pytest.raises(ValueError):
            gate.verified_audit(policy, comparison, native, dict(audit, **change))
    different = copy.deepcopy(audit)
    different["gate_decisions"]["bag_paired_mae"] = False
    with pytest.raises(ValueError, match="disagrees"):
        gate.verified_audit(policy, comparison, native, different)
    comparison["decisions"]["bag_paired_mae"]["actual"] += 1e-3
    with pytest.raises(ValueError):
        gate.verified_audit(policy, comparison, native, audit)


def test_locked_policy_and_compiler_allocation(tmp_path):
    docs = Path(__file__).resolve().parents[1] / "docs"
    assert sha256(docs / gate.NEF_POLICY) == gate.NEF_POLICY_SHA
    p = load_json(docs / gate.NEF_POLICY)
    assert p["development_policy_sha256"] == gate.regression.POLICY_SHA
    assert p["max_abs_transport_vs_bie_foci_per_nucleus"] == transport.MAX_ABS
    path = Path(__file__).resolve().parents[1] / "scripts/windows/nasa_neural16_nef_worker.py"
    spec = importlib.util.spec_from_file_location("neural16_worker", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    fw = tmp_path / "fw_info.txt"
    fw.write_text("Total [1] models:\nid: [32783], version: [0x1]\n"
                  "size: input [0x4], output [0x4], buf [0x8], cmd [0x4], weight [0x8], fw_code [0x4]\n"
                  "addr: input [0x10], output [0x14], buf [0x18], cmd [0x20], weight [0x24], fw_code [0x2c]\n"
                  "dram_addr_end [0x30], total bin size: [0x10]\n")
    assert worker.fw_layout(fw)["address_span_bytes"] == 32
    fw.write_text(fw.read_text().replace("buf [0x18]", "buf [0x14]"))
    with pytest.raises(ValueError, match="memory"):
        worker.fw_layout(fw)


def test_csim_carrier_width_and_radix_are_verified(tmp_path):
    (tmp_path / "dma2seq_0.info").write_text("1\n1\n1\n1\n13\n%d\n" % transport.descriptor_bits())
    (tmp_path / "dma2seq_0.seq").write_text("3760")
    with pytest.raises(FileNotFoundError):
        transport.captured_scalar(tmp_path, 3760, 0.6950960159301758)
    file = tmp_path / "dma2seq_0.seq.16B"
    file.write_text("7520")
    r = transport.captured_scalar(tmp_path, 3760, 0.6950960159301758)
    assert r["passed"] and r["double_native_code"] and r["abs_error"] == 0
    for native_word in (-1, 1):
        file.write_text(str(2 * native_word))
        raw = np.float32(native_word) / np.float32(4096) / np.float32(transport.SCALE)
        bie = float(np.float32(format(float(raw), ".8f")))
        assert transport.captured_scalar(tmp_path, native_word, bie)["passed"]
    file.write_text("3760")
    assert not transport.captured_scalar(tmp_path, 3760, 0.6950960159301758)["passed"]
    file.write_text("32768")
    with pytest.raises(ValueError, match="transport scalar"):
        transport.captured_scalar(tmp_path, 3760, 0.6950960159301758)
    assert float(transport.decode_transport(-32768)) < -3


def test_three_channel_signed_packing_keeps_channel_order_padding_and_spatial_order():
    x = np.zeros(transport.SHAPE, np.float32)
    x[0, :, 4, 9] = np.array([-32768, -1, 32767]) / 32768
    x[0, :, 5, 0] = [1, -2, 0]
    packed = np.frombuffer(transport.input_buffer(x), np.uint8).reshape(256, 256, 2, 16)
    assert packed[4, 9, :, :3].tolist() == [[0, 127, 127], [128, 255, 127]]
    assert packed[5, 0, :, :3].tolist() == [[127, 0, 0], [127, 128, 0]]
    assert not np.any(packed[:, :, :, 3:])
    assert not np.any(packed[9, 4])


def test_compile_receipt_is_bound_to_checked_worker_and_log(tmp_path):
    (tmp_path / "source").mkdir()
    worker = tmp_path / "source/nasa_neural16_nef_worker.py"
    worker.write_text("original")
    log = tmp_path / "docker.log"
    log.write_text("ok")
    chain = {"audit_sha256": "A"}
    save_json(tmp_path / "eligibility.json", chain)
    eligibility_sha = sha256(tmp_path / "eligibility.json")
    stage = {"eligibility": chain, "eligibility_sha256": eligibility_sha, "stage": "compile",
             "physical_execution": False, "passed": True, "worker_sha256": sha256(worker)}
    save_json(tmp_path / "compile.json", stage)
    digest = sha256(tmp_path / "compile.json")
    save_json(tmp_path / "invocation.json", {"stage_receipt_sha256": digest, "log_sha256": sha256(log),
              "returncode": 0, "sources": {worker.name: sha256(worker)}, "eligibility_sha256": eligibility_sha})
    assert gate.completed_stage(tmp_path, "compile", digest, chain) == stage
    worker.write_text("changed")
    with pytest.raises(ValueError, match="source snapshot"):
        gate.completed_stage(tmp_path, "compile", digest, chain)
    worker.write_text("original")
    log.write_text("changed")
    with pytest.raises(ValueError, match="host invocation"):
        gate.completed_stage(tmp_path, "compile", digest, chain)


def test_usb_offline_receipt_never_opens_device(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts/windows"
    monkeypatch.syspath_prepend(str(scripts))
    import run_resnet_neural16_usb as usb_run

    nef = tmp_path / "model.nef"
    nef.write_bytes(b"isolated guard fixture")
    args = SimpleNamespace(data=tmp_path, out=tmp_path / "offline", mode="smoke", execute=False,
                           audit_sha256="A", compile_sha256="C", sim_sha256="S")
    monkeypatch.setitem(sys.modules, "kp", SimpleNamespace())
    monkeypatch.setattr(usb_run, "preflight", lambda a, kp: (nef, {}, list(range(48)), 1, "fixture"))
    usb_run.run(args)
    receipt = load_json(args.out / "receipt.json")
    assert receipt["passed"] and receipt["verified_offline_cases"] == 48 and not receipt["physical_execution"]
    with pytest.raises(ValueError, match="smoke receipt"):
        usb_run.passing_smoke(None, None, sha256(nef), "S", {}, None, 1, "fixture", 7520)


def test_offline_plus_scalar_retrieval_on_signed_boundary_carriers(monkeypatch):
    kp = pytest.importorskip("kp")
    scripts = Path(__file__).resolve().parents[1] / "scripts/windows"
    monkeypatch.syspath_prepend(str(scripts))
    from nasa_neural16_plus import csim_packet, replay_packet, scalar_head

    setup = {"outputs": [{"name": "burden", "shape": [1, 1, 1, 1], "raw_shape": [1], "format": 6,
                          "buffer": {"segment": "OUTPUT", "entry": 0, "len": 16}}]}
    head = scalar_head(setup)
    for i, word in enumerate((-32768, -1, 0, 1, 7520, 32767)):
        packet = csim_packet(head, word, i)
        header = {"product_id": 0x720, "inference_number": i, "crop_number": 0, "num_output_node": 1}
        assert replay_packet(kp, packet, header, head, word, i) == {
            "fixed_word": word, "decoded": float(transport.decode_transport(word))}
