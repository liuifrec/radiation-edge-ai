"""Replay all registered simulator-qualified Spotiflow panel cases on a resident physical KL720."""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from radiation_edge_ai import _kl720_worker as usb
from radiation_edge_ai.audit_utils import sha256_file
from radiation_edge_ai.hardware import pretrained_models, spotiflow, spotiflow_transport
from radiation_edge_ai.hardware.spotiflow import target_assignment
from radiation_edge_ai.hardware.spotiflow_replay_io import (
    JOINED, SEPARATE, align, checked_array, compare_points, load_sdk_heads,
    original_head_parity, save_heads, tensor_parity, verify_parts, write_json,
)


def require_simulator_binding(manifest_hash, gate, simulator, panel_hash, nef_hash, checkpoint, ids):
    if (gate.get("stage") != "nef_simulator_parity" or gate.get("passed") is not True
            or gate.get("output_manifest_sha256") != manifest_hash
            or any(r.get("model") != checkpoint or r.get("panel_sha256") != panel_hash
                   or r.get("nef_sha256") != nef_hash for r in (gate, simulator))
            or simulator.get("stage") != "nef_simulator" or len(ids) != 63 or len(set(ids)) != 63):
        raise ValueError("Physical replay requires a matching passed NEF simulator gate and all 63 cases")
    aligned = align(simulator["rows"], ids)
    gates = align(gate["rows"], ids)
    if not all(r["passed"] is True for r in gates.values()):
        raise ValueError("Each simulator case must pass before USB load")
    if any(aligned[k].get("sdk_packed_input") != gates[k].get("sdk_packed_input") for k in ids):
        raise ValueError("SDK input identities differ between simulator outputs and gate")
    return aligned


def load_sdk_input(folder, spec, name, required):
    if spec is None or spec == {"available": False}:
        if required:
            raise ValueError("Missing SDK DRAM input for the INT16 ingress candidate")
        return None
    if spec.get("available") is not True or spec.get("input_name") != name:
        raise ValueError("Unexpected SDK DRAM input identity")
    rel = Path(spec["path"])
    path = (folder / rel).resolve()
    if rel.is_absolute() or not path.is_relative_to(folder.resolve()):
        raise ValueError("SDK DRAM input path outside simulator output directory")
    data = path.read_bytes()
    if not data or len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError("SDK DRAM input checksum or size differs")
    return data


def packing_comparison(buffer, sdk_input):
    result = {"host_sha256": hashlib.sha256(buffer).hexdigest(), "host_bytes": len(buffer),
              "sdk_available": sdk_input is not None, "equal": None}
    if sdk_input is not None:
        result.update(sdk_sha256=hashlib.sha256(sdk_input).hexdigest(), sdk_bytes=len(sdk_input),
                      equal=buffer == sdk_input)
    return result


def preflight(args):
    record = pretrained_models.model_record(args.model, variant=args.variant)
    nef = pretrained_models.qualified_nef(record, args.data_root)
    if record["adapter"] != spotiflow.__name__ or version("spotiflow") != record["provenance"]["software_version"]:
        raise ValueError("Expected the registered Spotiflow runtime")
    def registered(name):
        return pretrained_models.verified_artifact(record, name, args.data_root)
    panel_path = registered("panel")
    transport = args.variant == spotiflow_transport.VARIANT
    sim_path = registered("transport_outputs" if transport else "nef_outputs")
    packing_dir = registered("nef_outputs").parent if transport else sim_path.parent
    gate_path = registered(record["engineering"]["nef_simulator_parity"]["evidence"])
    ip = json.loads(registered(record["engineering"]["kl720_ip"]["evidence"]).read_bytes())
    panel = json.loads(panel_path.read_bytes())
    sim = json.loads(sim_path.read_bytes())
    gate = json.loads(gate_path.read_bytes())
    rows = [r for r in panel["rows"] if r["role"] == "readout"]
    ids = [r["case_id"] for r in rows]
    simulated = (align(sim["rows"], ids) if transport else
                 require_simulator_binding(sha256_file(sim_path), gate, sim, sha256_file(panel_path),
                                           sha256_file(nef), record["checkpoint"], ids))
    contract = ip["optimized"]["outputs"]
    expected = {v["name"]: v["shape"] for v in contract}
    if len(expected) != len(contract) or expected not in (JOINED, SEPARATE):
        raise ValueError("Unexpected registered output interface")
    reference = {}
    signed_input = record.get("analysis_overrides", {}).get("model_in_bitwidth_mode") == "int16"
    graph_input, = ip["optimized"]["inputs"]
    for row in rows:
        key = row["case_id"]
        if not key or Path(key).name != key or "/" in key or "\\" in key:
            raise ValueError("Invalid case identifier")
        pred = simulated[key]
        arrays = checked_array(sim_path.parent, pred["raw_heads_path"], pred["raw_heads_sha256"])
        tensor_parity(arrays, arrays)
        if len(contract) == 2:
            if transport:
                pred = dict(pred, sdk_returned_tensors=pred["separate_tensors"])
            originals = verify_parts(sim_path.parent, pred, sorted(contract, key=lambda p: p["shape"][1]), arrays)
            if originals is None:
                raise ValueError("Missing original SDK returns for the separate simulator outputs")
        else:
            originals = load_sdk_heads(sim_path.parent, pred, contract, arrays)
        reference[key] = {"heads": arrays, "originals": originals, "sdk_input": load_sdk_input(
            packing_dir, pred.get("sdk_packed_input"), graph_input["name"], signed_input)}
        if transport:
            fixed = []
            for part, floating in originals:
                signed = checked_array(sim_path.parent, part["fixed_path"], part["fixed_sha256"])
                meta = part["dll_checks"]
                if (signed.shape != floating.shape or signed.dtype != np.int16 or not meta["exact_words"]
                        or not meta["exact_descriptor_float32"] or meta["words"] != signed.size
                        or hashlib.sha256(signed.tobytes()).hexdigest() != meta["fixed_sha256"]
                        or hashlib.sha256(floating.tobytes()).hexdigest() != meta["float_sha256"]):
                    raise ValueError("Unverified PLUS transport head " + part["graph_name"])
                fixed.append((part, signed))
            reference[key]["fixed"] = fixed
        raw = checked_array(panel_path.parent, row["raw_path"], row["raw_sha256"])
        tensor = checked_array(panel_path.parent, row["tensor_path"], row["tensor_sha256"])
        if not np.array_equal(spotiflow.prepare_input(raw), tensor):
            raise ValueError("Preprocessed panel tensor mismatch for " + key)
        if row["target_mask_path"]:
            mask = checked_array(panel_path.parent, row["target_mask_path"], row["target_mask_sha256"])
            if mask.shape != raw.shape or not np.any(mask == row["target_label"]):
                raise ValueError("Target mask differs from registered image")
    return record, nef, ip, panel_path, sim_path, gate_path, rows, reference


def physical_counts(decoded, mask, label):
    points = decoded["points"]
    target = None if mask is None else sum(target_assignment(mask, label, points, len(points))[1])
    return {"crop": len(points), "target": target}


def compare_case(reference, heads, expected, actual, ref_counts, counts):
    tensor = tensor_parity(reference["heads"], heads)
    originals = []
    channel = 0
    for record, raw in reference["originals"]:
        end = channel + raw.shape[1]
        originals.append(dict(graph_name=record["graph_name"], returned_index=record["returned_index"],
                              dtype=record["dtype"], sha256=record["sha256"],
                              **original_head_parity(heads[:, channel:end], raw)))
        channel = end
    if channel != heads.shape[1]:
        raise ValueError("Original simulator returns do not cover every physical output channel")
    points = compare_points(expected["points"], actual["points"])
    return {"tensor": tensor, "original_return_parity": originals, "points": points,
            "reference_counts": ref_counts, "physical_counts": counts,
            "passed": bool(tensor["within_tolerance"] and all(r["within_tolerance"] for r in originals)
                           and points["equal_count"]
                           and points["max_coordinate_abs_px"] <= .001 and ref_counts == counts)}


def resident_loop(kp, group, loaded, contract, panel_root, rows, reference, threshold, args, receipt):
    if len(loaded.input_nodes) != 1 or usb.descriptor_shape(loaded.input_nodes[0]) != (1, 1, 128, 128):
        raise ValueError("Unexpected physical input shape")
    out_shapes = [usb.descriptor_shape(n) for n in loaded.output_nodes]
    widths = [s["shape"][1] for s in sorted(contract, key=lambda n: n["shape"][1])]
    order, logical = usb.output_order(out_shapes, widths)
    if logical != (1, 4, 128, 128):
        raise ValueError("Unexpected physical output shape")
    receipt["output_descriptors"] = [n.get_member_variable_dict() for n in loaded.output_nodes]
    receipt["input_descriptor"] = loaded.input_nodes[0].get_member_variable_dict()
    receipt["joined_descriptor_indices"] = order
    receipt["packing_preflight"] = []
    if getattr(args, "variant", None) == spotiflow_transport.VARIANT:
        inp = loaded.input_nodes[0]
        quant = inp.quantization_parameters.v1.quantized_fixed_point_descriptor_list
        scale = quant[0].scale if len(quant) == 1 else None
        if (not hasattr(inp.tensor_shape_info, "v1") or inp.data_layout != kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8BHL
                or len(quant) != 1 or int(quant[0].radix) != 14 or getattr(scale, "value", scale) != 1.0):
            raise ValueError("Unexpected qualified V1 ingress descriptor")
        if any(node.data_layout != kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_8W1C16B for node in loaded.output_nodes):
            raise ValueError("Unexpected qualified physical transport layout")
    # PLUS exposes the V1 NEF input descriptor after load. Check the whole panel
    # first so an incompatibility on any case blocks every inference submission.
    for row in rows:
        key = row["case_id"]
        sdk_input = reference[key]["sdk_input"]
        if sdk_input is not None:
            original = checked_array(panel_root, row["tensor_path"], row["tensor_sha256"])
            packed = usb.pack_kl720_input(kp, loaded.input_nodes[0], original)[0]
            comparison = packing_comparison(packed, sdk_input)
            receipt["packing_preflight"].append(dict(comparison, case_id=key))
            if not comparison["equal"]:
                path = args.out / (key + ".preflight_host_input.bin")
                with path.open("xb") as handle:
                    handle.write(packed)
                receipt["packing_preflight"][-1]["host_input_path"] = path.name
                raise ValueError("Descriptor packing differs from SDK DRAM input before inference at " + key)
    for i, row in enumerate(rows):
        key = row["case_id"]
        start = time.perf_counter_ns()
        raw = checked_array(panel_root, row["raw_path"], row["raw_sha256"])
        mask = (checked_array(panel_root, row["target_mask_path"], row["target_mask_sha256"])
                if row["target_mask_path"] else None)
        tensor = spotiflow.prepare_input(raw)
        prepared = time.perf_counter_ns()
        buffer, pack = usb.pack_kl720_input(kp, loaded.input_nodes[0], tensor)
        packing = packing_comparison(buffer, reference[key]["sdk_input"])
        if packing["equal"] is False:
            receipt["packing_error"] = dict(packing, case_id=key)
            path = args.out / (key + ".live_host_input.bin")
            with path.open("xb") as handle:
                handle.write(buffer)
            receipt["packing_error"]["host_input_path"] = path.name
            raise ValueError("Live input packing differs from the SDK DRAM input at " + key)
        request = kp.GenericDataInferenceDescriptor(model_id=loaded.id, inference_number=i,
                      input_node_data_list=[kp.GenericInputNodeData(buffer=buffer)])
        packed = time.perf_counter_ns()
        kp.inference.generic_data_inference_send(device_group=group, generic_inference_input_descriptor=request)
        response = kp.inference.generic_data_inference_receive(device_group=group)
        received = time.perf_counter_ns()
        try:
            if int(response.header.num_output_node) != len(out_shapes) or int(response.header.inference_number) != i:
                raise RuntimeError("Unexpected physical result header")
            parts = []
            for idx, shape in enumerate(out_shapes):
                node = kp.inference.generic_inference_retrieve_float_node(
                    node_idx=idx, generic_raw_result=response, channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW)
                parts.append(usb.normalize_retrieved_output(usb.extract_array(node), shape).copy())
            heads = parts[order[0]] if len(order) == 1 else spotiflow.join_heads(*(parts[j] for j in order))
            actual = spotiflow.decode_heads(heads, raw.shape, threshold)
            counts = physical_counts(actual, mask, row["target_label"])
            done = time.perf_counter_ns()
        finally:
            raw_record = usb.record_raw_receive(response, args.out / (key + ".bin"), int(response.header.inference_number))
            receipt.setdefault("raw_receives", []).append({"case_id": key, "submitted_inference_number": i,
                                                           "submitted_model_id": int(loaded.id), **raw_record})
        fixed_parity = []
        for index, (part, expected_fixed) in enumerate(reference[key].get("fixed", [])):
            idx = order[index]
            fixed_node = kp.inference.generic_inference_retrieve_fixed_node(
                node_idx=idx, generic_raw_result=response, channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW)
            fixed = np.asarray(fixed_node.ndarray).copy()
            if fixed.dtype != np.int16 or fixed.shape != expected_fixed.shape:
                raise ValueError("Unexpected signed physical output shape or width for " + part["graph_name"])
            path = args.out / f"{key}_fixed_{idx}.npy"
            with path.open("xb") as f:
                np.save(f, fixed, allow_pickle=False)
            fixed_parity.append({"graph_name": part["graph_name"], "device_index": idx, "words": fixed.size,
                                 "equal": bool(np.array_equal(fixed, expected_fixed)), "path": path.name,
                                 "sha256": sha256_file(path), "reference_sha256": part["fixed_sha256"]})
        saved = save_heads(args.out, key, parts, contract)
        ref_heads = reference[key]["heads"]
        expected = spotiflow.decode_heads(ref_heads, raw.shape, threshold)
        result = compare_case(reference[key], heads, expected, actual,
                              physical_counts(expected, mask, row["target_label"]), counts)
        if fixed_parity:
            result["signed_transport_outputs"] = fixed_parity
            result["passed"] = result["passed"] and all(p["equal"] for p in fixed_parity)
        result.update(saved, inference_number=i, submitted_model_id=int(loaded.id), threshold=threshold, points_yx=actual["points"].tolist(),
                      source_id=row["source_id"], domain=row["domain"], input_hw=list(raw.shape), target_label=row["target_label"],
                      confidence=actual["confidence"].tolist(), raw_receive=raw_record, fixed_input=pack,
                      sdk_packing=packing, packed_input_sha256=packing["host_sha256"], raw_image_sha256=row["raw_sha256"],
                      target_mask_sha256=row["target_mask_sha256"], tensor_sha256=row["tensor_sha256"],
                      timings_ms={"read_verify_prepare": (prepared-start)/1e6, "pack": (packed-prepared)/1e6,
                                  "send_receive": (received-packed)/1e6, "decode_and_count": (done-received)/1e6,
                                  "end_to_end": (done-start)/1e6})
        receipt["rows"].append(result)


def run(args):
    # Spotiflow's fast maximum filter reads OMP_NUM_THREADS on each call.
    os.environ["OMP_NUM_THREADS"] = "2"
    record, nef, ip, panel, sim, gate, rows, reference = preflight(args)
    if args.preflight_only:
        return {"preflight": "passed", "case_count": len(rows), "device_access": False,
                "sdk_inputs_verified": sum(r["sdk_input"] is not None for r in reference.values())}
    if args.replay_simulator:
        if record["variant"] != spotiflow_transport.VARIANT:
            raise ValueError("Archived signed transport requires full_int16_transport")
        from radiation_edge_ai.hardware.spotiflow_replay_io import replay_transport

        return replay_transport(record, args.data_root, nef, panel, rows, reference, args.out)
    if not (args.smoke or args.panel) or record["variant"] != spotiflow_transport.VARIANT:
        raise ValueError("Choose an explicit physical mode on the qualified transport variant")
    if args.smoke:
        # Preflight still authenticates the complete qualification panel. Use
        # its registered first input for an independently repeatable USB check.
        rows = rows[:1]
    import torch
    torch.set_num_threads(2)
    sys.path.append(str(args.plus_site.resolve()))
    import kp
    kp_path = Path(kp.__file__).resolve()
    if not kp_path.is_relative_to(args.plus_site.resolve()) or kp.core.get_version() != "3.2.0":
        raise ValueError("Expected the pinned local PLUS 3.2.0 installation")
    if args.timeout_ms <= 0:
        raise ValueError("Expected a positive USB timeout")
    if not args.out.resolve().is_relative_to(args.data_root.resolve()):
        raise ValueError("Save physical replay evidence under the external data root")
    start = datetime.now(timezone.utc).isoformat()
    stage = "physical_kl720_smoke" if args.smoke else "physical_kl720_panel"
    receipt = {"stage": stage, "passed": False, "completed": False, "started_utc": start,
               "model": record["checkpoint"], "variant": record["variant"], "rows": [],
               "panel_sha256": sha256_file(panel), "nef_sha256": sha256_file(nef), "nef_path": str(nef),
               "original_onnx_sha256": ip["original_onnx_sha256"], "optimized_onnx_sha256": ip["optimized_onnx_sha256"],
               "checkpoint_sha256": record["artifacts"]["best.pt"]["sha256"], "bie_sha256": record["artifacts"]["bie"]["sha256"],
               "gate_sha256": record["provenance"]["gate_lf_sha256"], "nef_outputs_sha256": sha256_file(sim),
               "nef_simulator_gate_sha256": sha256_file(gate), "script_sha256": sha256_file(Path(__file__)),
               "host_adapter_sha256": sha256_file(Path(spotiflow.__file__)), "usb_worker_sha256": sha256_file(Path(usb.__file__)),
               "python": platform.python_version(), "numpy": np.__version__, "spotiflow": version("spotiflow"),
               "torch": torch.__version__, "torch_host_threads": torch.get_num_threads(), "omp_num_threads": 2,
               "plus_version": kp.core.get_version(), "kp_module": str(kp_path),
               "plus_dll_sha256": sha256_file(kp_path.parent / "lib/libkplus.dll")}
    transport = record["variant"] == spotiflow_transport.VARIANT
    if transport:
        from radiation_edge_ai.hardware.spotiflow_plus import descriptors
        from spotiflow_physical_smoke import authenticate_received_outputs, require_smoke

        expected_kn = spotiflow_transport.registered_kn(record, args.data_root)
        discovery = pretrained_models.verified_artifact(record, "device_discovery", args.data_root)
        registered_firmware = json.loads(discovery.read_bytes())["system_info"]["firmware_version"]
        raw_path = pretrained_models.verified_artifact(record, "transport_dll_receipt", args.data_root)
        raw_report = json.loads(raw_path.read_bytes())
        if raw_report["plus_dll_sha256"] != receipt["plus_dll_sha256"]:
            raise ValueError("Live PLUS DLL differs from qualified fixed and floating retrieval")
        receipt["transport_capture_sha256"] = record["artifacts"]["transport_capture"]["sha256"]
        receipt["transport_dll_receipt_sha256"] = sha256_file(raw_path)
        if not args.smoke:
            fields = ("model", "variant", "panel_sha256", "nef_sha256", "bie_sha256", "gate_sha256",
                      "nef_outputs_sha256", "nef_simulator_gate_sha256", "script_sha256", "host_adapter_sha256",
                      "usb_worker_sha256", "plus_version", "plus_dll_sha256", "transport_capture_sha256",
                      "transport_dll_receipt_sha256")
            bindings = {k: receipt[k] for k in fields}
            bindings["model_id"] = ip["model_id"]
            receipt["passed_smoke"] = require_smoke(
                args.smoke_receipt, args.smoke_sha256, args.data_root, bindings,
                rows[0], reference[rows[0]["case_id"]], json.loads(gate.read_bytes())["utc"], kp, descriptors(nef.parent.parent), expected_kn=expected_kn)
    args.out.mkdir(parents=True, exist_ok=False)
    group = None
    try:
        dev = usb.select_kl720(usb.scan_descriptors(kp), args.port)
        if transport:
            spotiflow_transport.check_device(dev.get_member_variable_dict(), expected_kn=expected_kn)
        port = int(dev.usb_port_id)
        group = kp.core.connect_devices(usb_port_ids=[port])
        kp.core.set_timeout(device_group=group, milliseconds=args.timeout_ms)
        receipt.update(device=dev.get_member_variable_dict(), system_info=kp.core.get_system_info(
            group, port).get_member_variable_dict())
        if transport:
            firmware = registered_firmware if args.smoke else receipt["passed_smoke"]["firmware_version"]
            spotiflow_transport.check_device(receipt["device"], receipt["system_info"], firmware, expected_kn=expected_kn)
        start = time.perf_counter()
        package = kp.core.load_model_from_file(device_group=group, file_path=str(nef))
        receipt["model_load_seconds"] = time.perf_counter() - start
        model, = package.models
        if int(model.id) != ip["model_id"]:
            raise ValueError("Loaded model ID differs from qualified artifact")
        receipt.update(model_id=int(model.id), model_version=str(model.version))
        resident_loop(kp, group, model, ip["optimized"]["outputs"], panel.parent, rows, reference,
                      record["threshold"], args, receipt)
        if transport and args.smoke:
            only, = receipt["rows"]
            receipt["smoke_raw_retrieval"] = authenticate_received_outputs(
                kp, args.out, only, descriptors(nef.parent.parent), reference[only["case_id"]], 0)
        receipt["timings_ms"] = {name: {
            "median": float(np.median([r["timings_ms"][name] for r in receipt["rows"]])),
            "p95": float(np.percentile([r["timings_ms"][name] for r in receipt["rows"]], 95))}
            for name in ("send_receive", "end_to_end")}
        receipt.update(completed=True, passed=all(row["passed"] for row in receipt["rows"]))
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if group is not None:
                kp.core.disconnect_devices(group)
        finally:
            receipt["ended_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(args.out / "receipt.json", receipt)
    return receipt


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plus-site", type=Path, default=Path.home() / "venvs/kneron720/Lib/site-packages")
    parser.add_argument("--port", type=int)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true", help="Authenticate inputs and gates offline")
    mode.add_argument("--replay-simulator", action="store_true", help="Decode registered archived simulator transport offline")
    mode.add_argument("--smoke", action="store_true", help="Replay the first qualified input on the physical USB device")
    mode.add_argument("--panel", action="store_true", help="Submit all cases after a matching physical smoke")
    parser.add_argument("--smoke-receipt", type=Path, help="Prior successful physical smoke receipt for the full replay")
    parser.add_argument("--smoke-sha256", help="SHA256 of the prior physical smoke receipt")
    return parser


if __name__ == "__main__":
    verdict = run(argument_parser().parse_args())
    print(json.dumps({k: verdict[k] for k in ("preflight", "device_access", "case_count", "sdk_inputs_verified",
                                              "completed", "passed") if k in verdict}))
    sys.exit(0 if verdict.get("passed", verdict.get("preflight") == "passed") else 2)
