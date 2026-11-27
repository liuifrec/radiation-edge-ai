"""Authenticate the selected ResNet18 simulator replay and optionally submit it to the approved KL720."""

import argparse
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from radiation_edge_ai import _kl720_worker as usb
from radiation_edge_ai.hardware import spotiflow_plus as abi
from radiation_edge_ai.hardware.spotiflow_transport import check_device
from radiation_edge_ai.nasa_bps import neural16_nef as gate
from radiation_edge_ai.nasa_bps import neural16_transport as transport
from radiation_edge_ai.nasa_bps.multimodel_ptq import calibration_path, checked_rows, load_json, require, save_json, sha256
from nasa_neural16_plus import csim_packet, replay_packet, scalar_head

HARDWARE_REF = "scratch/spotiflow_general_transport_usb_smoke_20261120T025814Z_87f21ac3/receipt.json"
HARDWARE_REF_SHA = "fce011c42d497c0725506a20b4b07944696785bdd6fa92a2df66286ee878a9e6"


def preflight(args, kp):
    run = args.campaign.resolve(strict=True)
    chain = gate.eligibility(Path(__file__).resolve().parents[2] / "docs", args.data, run, args.audit_sha256)
    compiled = gate.completed_stage(args.compile_dir, "compile", args.compile_sha256, chain)
    gate.compiled_files(args.compile_dir, compiled)
    stage = "development-reference" if args.mode == "development" else "simulate"
    sim = gate.completed_stage(args.sim_dir, stage, args.sim_sha256, chain)
    if args.mode == "development":
        require(args.training_sim_dir is not None, "The 48-witness pass is required")
        training = gate.completed_stage(args.training_sim_dir, "simulate", sim["training_simulation_sha256"], chain)
        require(training["compile_sha256"] == sim["compile_sha256"] and
                training["nef_sha256"] == sim["nef_sha256"] and len(training["rows"]) == 48
                and sim["development_replay_policy_sha256"] == gate.REPLAY_POLICY_SHA, "Training NEF decision changed")
    nef = args.compile_dir / "model.nef"
    flow = args.compile_dir / "kneron_flow/model_modelid_32783"
    require(compiled["nef_sha256"] == sim["nef_sha256"] == sha256(nef) and sim["compile_sha256"] == args.compile_sha256
            and compiled["model_id"] == 32783 and compiled["model_version"] == 1
            and compiled["cpu_nodes"] == [] and compiled["reference_memory"]["fits_reference_map"] is True
            and sha256(flow / "ioinfo.json") == compiled["compiled_io_sha256"]
            and sha256(flow / "setup.bin") == sim["typed_setup"]["setup_sha256"] == compiled["setup_bin_sha256"]
            and sha256(args.sim_dir / "typed_setup.tsv") == sim["typed_setup"]["typed_descriptors_sha256"]
            and sim["atol"] == transport.MAX_ABS and sim["rtol"] == 0, "Changed compiled model or NEF decision")
    setup = load_json(flow / "setup.txt")
    require(sha256(flow / "setup.txt") == compiled["setup_text_sha256"], "Changed compiler output descriptions")
    head = scalar_head(setup)
    prep = load_json(run / "preparation.json")
    files = prep["artifacts"]
    if args.mode == "development":
        root = Path(prep["historical_preparation"])
        cases = checked_rows(root / "inputs.csv")
        pairs = checked_rows(run / "vendor/bie_nuclei.csv")
    else:
        root = Path(files["calibration_manifest_sha256"]["path"]).parent
        cases = load_json(Path(files["training_panel_sha256"]["path"]))["rows"]
        pairs = checked_rows(Path(files["training_predictions_sha256"]["path"]))
    metadata = {r["sample_id"]: r for r in cases}
    refs = {r["sample_id"]: r for r in pairs}
    expected = len(cases)
    rows = sim["rows"]
    require(len(rows) == expected == (2400 if args.mode == "development" else 48)
            and {r["sample_id"] for r in rows} == metadata.keys() == refs.keys(), "Incomplete simulator panel")
    if args.mode != "development":
        require(rows[0]["sample_id"] == transport.SMOKE, "Wrong positive smoke case")
    verified = []
    for index, item in enumerate(rows):
        sid = item["sample_id"]
        member, ref = metadata[sid], refs[sid]
        case = args.sim_dir / sid
        require(case.resolve().parent == args.sim_dir.resolve() and item["passed"] is True
                and item["input_sha256"] == member["array_sha256"] == ref["array_sha256"], "Different panel input")
        for name, digest in item["files"].items():
            f = (case / name).resolve(strict=True)
            require(f.parent == case.resolve() and sha256(f) == digest, "Changed raw simulator output " + sid)
        if args.mode == "development":
            path = (root / member["array_file"]).resolve(strict=True)
            require(root.resolve() in path.parents and path.name.endswith(".npy"), "Unexpected input path")
            code = int(ref["bie_fixed_code"])
        else:
            path = calibration_path(root, member["array_file"])
            code = int(ref["fixed_word"])
        require(sha256(path) == member["array_sha256"], "Preprocessed input changed")
        measured = transport.captured_scalar(case, code, float(ref["bie_burden"]))
        require(all(item[k] == v for k, v in measured.items()) and measured["passed"], "CSIM scalar changed: " + sid)
        packed = transport.input_buffer(np.load(path, allow_pickle=False))
        require(len(packed) == item["sdk_input_bytes"] and hashlib.sha256(packed).hexdigest() == item["sdk_input_sha256"],
                "Host packing differs from qualified SDK bytes")
        if args.mode != "development":
            require((case / "sdk_packed.bin").read_bytes() == packed, "Staged packed bytes changed")
        word = measured["word"]
        artificial = csim_packet(head, word, index)
        synthetic_header = {"product_id": 0x720, "inference_number": index, "crop_number": 0, "num_output_node": 1}
        replay = replay_packet(kp, artificial, synthetic_header, head, word, index)
        require(replay["decoded"] == measured["decoded"], "PLUS offline float retrieval differs from CSIM")
        verified.append((path, item))
    reference_file = args.data / HARDWARE_REF
    require(sha256(reference_file) == HARDWARE_REF_SHA, "Changed physical reference identity")
    old = load_json(reference_file)
    firmware = old["system_info"]["firmware_version"]
    expected_kn = int(old["device"]["kn_number"], 0)
    require(kp.core.get_version() == old["plus_version"] == "3.2.0"
            and sha256(Path(kp.__file__).parent / "lib/libkplus.dll") == old["plus_dll_sha256"] == abi.DLL_SHA,
            "Unexpected PLUS native runtime")
    return nef, head, verified, expected_kn, firmware


def passing_smoke(path, digest, nef_sha, training_sim_sha, head, kp, expected_kn, firmware, smoke_word):
    require(path is not None and sha256(path) == digest, "A pinned physical smoke receipt is required")
    smoke = load_json(path)
    require(smoke["stage"] == "smoke" and smoke["passed"] is True and smoke["physical_execution"] is True
            and smoke["nef_sha256"] == nef_sha and smoke["sim_sha256"] == training_sim_sha
            and len(smoke["rows"]) == 1, "Physical smoke was incomplete or belongs to a different model")
    check_device(smoke["device"], smoke["system"], firmware, expected_kn=expected_kn)
    (item,) = smoke["rows"]
    require(item["sample_id"] == transport.SMOKE and item["passed"] is True
            and item["fixed_word"] == smoke_word, "Physical smoke did not reproduce the qualified positive witness")
    packet = path.parent / (transport.SMOKE + ".bin")
    require(sha256(packet) == item["raw_receive"]["allocation"]["sha256"], "Changed raw smoke output")
    replay = replay_packet(kp, packet.read_bytes(), item["raw_receive"]["header"], head, smoke_word, 0)
    require(replay["decoded"] == item["decoded"], "Smoke scalar differs from archived PLUS retrieval")


def loaded_descriptors(kp, model):
    (inp,), (out,) = model.input_nodes, model.output_nodes
    require(int(model.id) == 32783 and int(model.version) == 1 and usb.descriptor_shape(inp) == transport.SHAPE
            and usb.descriptor_shape(out) == (1, 1, 1, 1)
            and inp.data_layout == kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8BHL
            and out.data_layout == kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_8W1C16B,
            "Loaded tensor layout or model ID differs from NEF")
    for node, radix, scale, count in ((inp, 15, 1.0, (1, 3)), (out, 13, transport.SCALE, (1,))):
        types = node.quantization_parameters.v1.quantized_fixed_point_descriptor_list
        require(len(types) in count and all(t.radix == radix and t.scale.value == scale for t in types),
                "Loaded fixed-point descriptor differs from the authenticated transport")


def run(args):
    import kp

    require(args.data.resolve() in args.out.resolve().parents, "Use a fresh external record directory")
    nef, head, panel, kn, firmware = preflight(args, kp)
    rows = panel[:1] if args.mode == "smoke" else panel
    rec = {"stage": args.mode, "physical_execution": False, "passed": False, "rows": [],
           "created_utc": datetime.now(timezone.utc).isoformat(), "nef_sha256": sha256(nef),
           "audit_sha256": args.audit_sha256, "compile_sha256": args.compile_sha256, "sim_sha256": args.sim_sha256,
           "mode": "physical" if args.execute else "offline_plus_transport_preflight",
           "available_references": len(panel), "operator_requested_execute": args.execute,
           "scope": "known_development" if args.mode == "development" else "training_engineering"}
    args.out.mkdir(exist_ok=False, parents=True)
    group = None
    try:
        if not args.execute:
            rec.update(passed=True, verified_offline_cases=len(panel))
            return
        if args.mode != "smoke":
            if args.mode == "development":
                sim = load_json(args.sim_dir / "development-reference.json")
                training_sim_sha = sim["training_simulation_sha256"]
                positive = load_json(args.training_sim_dir / "simulate.json")["rows"][0]
            else:
                training_sim_sha = args.sim_sha256
                positive = panel[0][1]
            require(positive["sample_id"] == transport.SMOKE, "Wrong smoke case")
            passing_smoke(args.smoke_receipt, args.smoke_sha256, rec["nef_sha256"], training_sim_sha,
                          head, kp, kn, firmware, positive["word"])
            rec["physical_smoke_sha256"] = args.smoke_sha256
        dev = usb.select_kl720(usb.scan_descriptors(kp), args.port)
        rec["device"] = dev.get_member_variable_dict()
        check_device(rec["device"], expected_kn=kn)
        port = int(dev.usb_port_id)
        group = kp.core.connect_devices(usb_port_ids=[port])
        kp.core.set_timeout(device_group=group, milliseconds=10000)
        rec["system"] = kp.core.get_system_info(group, port).get_member_variable_dict()
        check_device(rec["device"], rec["system"], firmware, expected_kn=kn)
        start = time.perf_counter()
        (model,) = kp.core.load_model_from_file(group, str(nef)).models
        rec.update(load_seconds=time.perf_counter()-start, model_id=int(model.id), model_version=str(model.version))
        loaded_descriptors(kp, model)
        rec["physical_execution"] = True
        for index, (path, item) in enumerate(rows):
            start = time.perf_counter()
            require(sha256(path) == item["input_sha256"], "Sealed tensor changed after preflight")
            packed = transport.input_buffer(np.load(path, allow_pickle=False))
            require(hashlib.sha256(packed).hexdigest() == item["sdk_input_sha256"], "Input changed after preflight")
            request = kp.GenericDataInferenceDescriptor(model_id=model.id, inference_number=index,
                       input_node_data_list=[kp.GenericInputNodeData(buffer=packed)])
            sent = time.perf_counter()
            kp.inference.generic_data_inference_send(group, request)
            response = kp.inference.generic_data_inference_receive(group)
            received = time.perf_counter()
            file = args.out / (item["sample_id"] + ".bin")
            raw = usb.record_raw_receive(response, file, index)
            actual = replay_packet(kp, file.read_bytes(), raw["header"], head, item["word"], index)
            require(actual["decoded"] == item["decoded"], "Device float differs from the qualified CSIM")
            rec["rows"].append(dict(actual, passed=True, sample_id=item["sample_id"], input_sha256=item["input_sha256"],
                                    packed_sha256=item["sdk_input_sha256"], raw_receive=raw,
                                    send_receive_ms=(received-sent)*1000,
                                    host_to_host_ms=(time.perf_counter()-start)*1000))
        rec["passed"] = len(rec["rows"]) == len(rows)
    except Exception as exc:
        rec["error"] = repr(exc)
        raise
    finally:
        try:
            if group is not None:
                kp.core.disconnect_devices(group)
        finally:
            save_json(args.out / "receipt.json", rec)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["smoke", "training", "development"])
    for key in ("data", "campaign", "compile-dir", "sim-dir", "out"):
        p.add_argument("--" + key, required=True, type=Path)
    for key in ("audit-sha256", "compile-sha256", "sim-sha256"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--smoke-receipt", type=Path)
    p.add_argument("--smoke-sha256")
    p.add_argument("--training-sim-dir", type=Path)
    p.add_argument("--port", type=int)
    p.add_argument("--execute", action="store_true", help="Submit to USB only after separate operator approval")
    run(p.parse_args())
