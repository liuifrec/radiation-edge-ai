"""Compile the eligible frozen BIE or capture real KL720 CSIM carrier files in the pinned image."""

import argparse
import inspect
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from radiation_edge_ai.nasa_bps.multimodel_ptq import calibration_path, checked_rows, load_json, require, save_json, sha256
from radiation_edge_ai.nasa_bps.neural16_nef import NEF_POLICY_SHA, REPLAY_POLICY_SHA
from radiation_edge_ai.nasa_bps.neural16_transport import SCALE, SHAPE, SMOKE, captured_scalar, descriptor_bits, input_buffer

OUT = Path("/out")
BIE = Path("/model.bie")


def fw_layout(path):
    content = path.read_text(encoding="utf-8")
    require("Total [1] models" in content and "id: [32783], version: [0x1]" in content, "Invalid firmware model identity")
    sections = {}
    for key in ("size", "addr"):
        line, = [s for s in content.splitlines() if s.strip().startswith(key + ":")]
        sections[key] = {k: int(v, 16) for k, v in re.findall(r"(\w+) \[(0x[0-9a-fA-F]+)\]", line)}
    include = {"input", "output", "buf", "cmd", "weight", "fw_code"}
    require(set(sections["size"]) == set(sections["addr"]) == include, "Incomplete firmware allocations")
    spans = sorted((addr, addr + sections["size"][key]) for key, addr in sections["addr"].items())
    require(all(e > s for s, e in spans) and all(x[1] <= y[0] for x, y in zip(spans, spans[1:])),
            "Invalid firmware memory region")
    end = int(re.search(r"dram_addr_end \[(0x[\da-fA-F]+)\]", content).group(1), 16)
    require(spans[-1][1] <= end, "Model allocation exceeds compiler DRAM end")
    bin_bytes = int(re.search(r"total bin size: \[(0x[\da-fA-F]+)\]", content).group(1), 16)
    return {"segments_bytes": sections["size"], "addresses": sections["addr"], "dram_end": end,
            "model_binary_bytes": bin_bytes,
            "address_span_bytes": end - spans[0][0], "actual_device_load_verified": False}


def reference_memory(memory):
    # Identified PLUS 3.2.0 reference KL720 board support, read offline. The device
    # loader may have additional reservations that are observable only at runtime.
    project, platform = Path("/map/project.h"), Path("/map/membase.h")
    require(sha256(project) == "0d9bf507684fddf726fe6534451eea6c0013640b3bfc0e0b2333b58c35b61e32"
            and sha256(platform) == "2707717650e1a1c5d92fe7e655419e8dd29122748d9fa96e47b501706503c3bc",
            "Unrecognized PLUS reference board memory map")
    base, end = 0x80220000, 0x85900000
    limit = (end - base) & ~0xfffff
    ok = (min(memory["addresses"].values()) >= base and memory["dram_end"] <= end
          and memory["model_binary_bytes"] <= limit)
    return {"scope": "installed PLUS KL720 sn72096_9x9 reference firmware",
            "project_header_sha256": sha256(project), "platform_header_sha256": sha256(platform),
            "base": base, "heap_end": end, "model_binary_limit_bytes": limit,
            "bin_headroom_bytes": limit - memory["model_binary_bytes"],
            "layout_headroom_bytes": end - memory["dram_end"], "fits_reference_map": ok}


def compile_model(receipt):
    import ktc

    start = time.monotonic()
    flow = OUT / "kneron_flow"
    model = ktc.ModelConfig(32783, "0001", "720", bie_path=str(BIE))
    product = Path(ktc.compile([model], output_dir=str(flow), dedicated_output_buffer=True,
                               weight_compress=False, flatbuffer=True, debug=False, compiler_tiling="default"))
    receipt["compiler_wall_seconds"] = time.monotonic() - start
    require(product.is_file() and product.stat().st_size, "Vendor did not return a NEF")
    nef = OUT / "model.nef"
    shutil.copyfile(str(product), str(nef))
    receipt.update(nef_sha256=sha256(nef), nef_bytes=nef.stat().st_size,
                   vendor_api_sha256=sha256(inspect.getsourcefile(ktc.compile)))
    inspect_compilation(flow, receipt)


def inspect_compilation(flow, receipt):
    """Examine existing compiler outputs without invoking the compiler or writing a NEF."""
    cfg_dir = flow / "model_modelid_32783"
    cfg, setup = (load_json(cfg_dir / f) for f in ("model_config.json", "setup.txt"))
    require(cfg["model_id"] == 32783 and cfg["version"] == 1 and setup["header"]["target"] == "KL720"
            and setup["cpu_nodes"] == [], "Unexpected target, compiled identity or CPU fallback")
    for key, name in (("inputs", "input"), ("outputs", "burden")):
        node, = setup[key]
        require(node["name"] == name, "Unexpected compiled tensor")
    firmware = flow / "fw_info.txt"
    receipt.update(memory=fw_layout(firmware), cpu_nodes=setup["cpu_nodes"], model_id=cfg["model_id"],
                   model_version=cfg["version"], setup_segments=setup["header"]["segment_info"],
                   firmware_map_sha256=sha256(firmware), compiler_config_sha256=sha256(cfg_dir / "model_config.json"),
                   setup_text_sha256=sha256(cfg_dir / "setup.txt"),
                   compiled_io_sha256=sha256(cfg_dir / "ioinfo.json"), setup_bin_sha256=sha256(cfg_dir / "setup.bin"),
                   compiler_log_sha256=sha256(flow / "batch_compile.log"))
    receipt["reference_memory"] = reference_memory(receipt["memory"])
    require(receipt["reference_memory"]["fits_reference_map"], "Compiler allocation exceeds reference board model region")
    require(sha256(BIE) == receipt["eligibility"]["bie_sha256"], "BIE bytes changed during compilation")


def capture_setup(nef, folder, compile_out, work):
    compiled = compile_out / "kneron_flow/model_modelid_32783"
    setup_bin, = folder.glob("*setup.bin")
    raw = setup_bin.read_bytes()
    nef_bytes = nef.read_bytes()
    require(raw == (compiled / "setup.bin").read_bytes() and nef_bytes.count(raw) == 1,
            "Extracted setup differs from embedded NEF or compiler output")
    header = Path("/plus_headers/kneron_setup_reader.h")
    require(sha256(header) == "bcd2db443a8c6d87681f9dd044d0e96abd519df0c1337ee5f929bb21089283f0",
            "Use the registered PLUS 3.2.0 generated setup header")
    elf = work / "inspect"
    source = Path("/launch/nasa_neural16_setup.c")
    subprocess.run(["gcc", "-O0", "-std=c11", "-I/plus_headers", str(source), "-o", str(elf)], check=True)
    typed = subprocess.check_output([str(elf), str(setup_bin)]).decode("ascii")
    (OUT / "typed_setup.tsv").write_text(typed, encoding="ascii")
    inputs, outputs = [], []
    for line in typed.splitlines():
        role, node, layout, idx, bits, radix = line.split("\t")
        if role == "input":
            require(node == "input" and layout == "2" and bits == "3f800000" and radix == "15",
                    "Unexpected typed input packing contract")
            inputs.append(int(idx))
        else:
            require(role == "output" and node == "burden" and layout == "6"
                    and int(bits, 16) == descriptor_bits() and radix == "13", "Unexpected typed output")
            outputs.append(int(idx))
    require(inputs in ([0], [0, 1, 2]) and outputs == [0], "Unexpected typed quantizer count")
    return {"setup_offset": nef_bytes.find(raw), "setup_sha256": sha256(setup_bin),
            "plus_generated_header_sha256": sha256(header), "probe_sha256": sha256(source),
            "typed_descriptors_sha256": sha256(OUT / "typed_setup.tsv")}


def inputs(policy):
    selection_path, prediction_path = Path("/selection.json"), Path("/training.csv")
    require(sha256(selection_path) == policy["training_manifest_sha256"]
            and sha256(prediction_path) == policy["training_predictions_sha256"], "Training witnesses changed")
    selection = load_json(selection_path)["rows"]
    pairs = checked_rows(prediction_path)
    require(len(selection) == len(pairs) == len({r["sample_id"] for r in pairs}) == 48,
            "Incomplete training reference")
    items = []
    for meta, ref in zip(selection, pairs):
        sample = meta["sample_id"]
        require(re.fullmatch(r"BPSR1V2D_\d{5}", sample) and meta["fold_female_role"] == "train"
                and sample == ref["sample_id"] and meta["array_sha256"] == ref["array_sha256"],
                "Invalid training witness identity")
        case = calibration_path(Path("/export"), meta["array_file"])
        require(sha256(case) == meta["array_sha256"], "Training tensor changed")
        items.append((meta, ref, case))
    return sorted(items, key=lambda case: case[0]["sample_id"] != SMOKE)


def vendor_packed(converter, descriptor, tensor, local):
    paths, _, _ = converter({"input": [tensor]}, [descriptor], local / "converter")
    packed, = paths[0]
    resident = input_buffer(tensor)
    require(packed.read_bytes() == resident, "Three-channel host and SDK input packing differ")
    return packed


def development_inputs(chain):
    from radiation_edge_ai.nasa_bps.multimodel_bie_inference import IDENTITY

    root = Path("/dev_inputs")
    manifest = root / "inputs.csv"
    predictions = Path("/dev_predictions.csv")
    require(sha256(predictions) == chain["predictions_sha256"]
            and sha256(manifest) == chain["input_manifest_sha256"], "Different development references")
    rows, pairs = checked_rows(manifest), checked_rows(predictions)
    require(len(rows) == len(pairs) == len(set(r["sample_id"] for r in rows)) == 2400,
            "Incomplete known development input manifest")
    items = []
    for meta, ref in zip(rows, pairs):
        require(all(meta[k] == ref[k] for k in IDENTITY) and meta["array_file"] == ref["array_file"]
                and re.fullmatch(r"tensors/[0-9]{4}\.npy", meta["array_file"])
                and re.fullmatch(r"BPSR1V2D_\d{5}", meta["sample_id"]), "Changed development identity")
        case = root / meta["array_file"]
        require(case.resolve() == case and sha256(case) == meta["array_sha256"], "Changed development tensor")
        items.append((meta, dict(ref, fixed_word=ref["bie_fixed_code"]), case))
    return items


def simulate(receipt, policy, compiled_digest, training_sim_sha):
    import jinja2
    import ktc
    from sys_flow import csim_utils, flow_constants
    from sys_flow.compiler_v2 import unpack_nefs
    from sys_flow.inference import data_converter
    from radiation_edge_ai.nasa_bps import kneron_verify_r1_v2_nef_vs_bie as scalar_parser

    compile_out = Path("/compiled")
    cp = load_json(compile_out / "compile.json")
    require(sha256(compile_out / "compile.json") == compiled_digest and cp["passed"] is True
            and cp["eligibility"] == receipt["eligibility"], "Changed compile/eligibility chain")
    original = compile_out / "model.nef"
    require(sha256(original) == cp["nef_sha256"], "Compiled NEF changed")
    io = load_json(compile_out / "kneron_flow/model_modelid_32783/ioinfo.json")
    inp, = io["input"]
    out, = io["output"]
    require(inp["name"] == "input" and inp["shape"] == list(SHAPE)
            and inp["onnx_shape"] == inp["shape"] and inp["bitw"] == 16 and inp["data_format"] == "1W16C8BHL"
            and set(inp["scale"]) == {1.0} and set(inp["radix"]) == {15}
            and out["name"] == "burden" and out["bitw"] == 16 and out["data_format"] == "8W1C16B"
            and out["onnx_shape"] == [1] and out["shape"] == [1, 1, 1, 1]
            and out["radix"] == [13] and out["scale"] == [SCALE], "Unexpected compiled IO contract")
    receipt.update(compile_sha256=compiled_digest, nef_sha256=cp["nef_sha256"],
                   compiled_io=io, atol=policy["max_abs_transport_vs_bie_foci_per_nucleus"], rtol=0,
                   data_converter_code_sha256=sha256(inspect.getsourcefile(data_converter)),
                   csim_binary_sha256=sha256(flow_constants.BIN_SET["csim"][720]), rows=[])
    development = receipt["stage"] == "development-reference"
    if development:
        prior = load_json(Path("/training_sim.json"))
        require(sha256(Path("/replay.md")) == REPLAY_POLICY_SHA
                and sha256(Path("/training_sim.json")) == training_sim_sha
                and prior["stage"] == "simulate" and prior["passed"] is True
                and prior["nef_sha256"] == cp["nef_sha256"] and prior["compile_sha256"] == compiled_digest,
                "Invalid 48-case transport qualification")
        receipt.update(training_simulation_sha256=training_sim_sha, development_replay_policy_sha256=REPLAY_POLICY_SHA)
        panel = development_inputs(receipt["eligibility"])
    else:
        panel = inputs(policy)
        require(panel[0][0]["sample_id"] == SMOKE, "Missing declared positive witness")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        nef = work / "model.nef"
        shutil.copyfile(str(original), str(nef))
        scalar_parser._install_nef_scalar_output_parser_compatibility()
        models, _ = unpack_nefs(nef, 720)
        (mid, (folder, _)), = models.items()
        require(int(mid) == 32783, "Unexpected NEF model id")
        receipt["typed_setup"] = capture_setup(nef, folder, compile_out, work)
        tmpl = jinja2.Environment(loader=jinja2.FileSystemLoader(str(flow_constants.P_FLOW) + "/template")
                                  ).get_template("run_csim_720.ini")
        for meta, ref, case in panel:
            sid = meta["sample_id"]
            dest = OUT / sid
            dest.mkdir()
            tensor = np.load(case, allow_pickle=False)
            with tempfile.TemporaryDirectory(dir=tmp) as local:
                local = Path(local)
                packed_file = vendor_packed(data_converter, inp, tensor, local)
                if not development:
                    staged = dest / "sdk_packed.bin"
                    shutil.copyfile(str(packed_file), str(staged))
                    require(sha256(staged) == sha256(packed_file), "Staged SDK packing differs: " + sid)
                ini = local / "run_csim.ini"
                csim_utils.gen_csim_ini([packed_file], folder, 720, template=tmpl, fn_ini=ini)
                start = time.monotonic()
                try:
                    cmd, proc = csim_utils.run_csim({0: [local, ini]}, flow_constants.BIN_SET["csim"][720],
                                                    local / "invocation.sh", n_thread=1)
                    elapsed = time.monotonic() - start
                    for name in ("stdout", "stderr"):
                        output = getattr(proc, name)
                        if output is not None:
                            (dest / (name + ".log")).write_bytes(output if isinstance(output, bytes) else output.encode())
                    require(proc.returncode == 0, "CSIM exited " + str(proc.returncode))
                finally:
                    for name in ("dma2seq_0.info", "dma2seq_0.seq", "dma2seq_0.seq.16B", "run_csim.ini", "invocation.sh"):
                        if (local / name).is_file():
                            shutil.copyfile(str(local / name), str(dest / name))
                measurement = captured_scalar(dest, int(ref["fixed_word"]), float(ref["bie_burden"]))
                ordinary = dest / "dma2seq_0.seq"
                rec = dict(measurement, sample_id=sid, input_sha256=meta["array_sha256"],
                           sdk_input_sha256=sha256(packed_file), sdk_input_bytes=packed_file.stat().st_size,
                           ordinary_word=ordinary.read_text().strip() if ordinary.is_file() else None,
                           files={f.name: sha256(f) for f in dest.iterdir() if f.is_file()},
                           simulator_seconds=elapsed, command=cmd, returncode=proc.returncode)
                receipt["rows"].append(rec)
                save_json(dest / "case.json", rec)
                save_json(OUT / "progress.json", {"completed": len(receipt["rows"]), "latest": sid})
                print(sid, rec["word"], rec["decoded"], rec["bie"], rec["abs_error"], flush=True)
                require(rec["passed"], "Compiled transport scalar exceeded frozen NASA tolerance: " + sid)
                if not development and sid == SMOKE:
                    start = time.monotonic()
                    public = {"scope": "public simulator diagnostic",
                              "api_source_sha256": sha256(inspect.getsourcefile(ktc.kneron_inference))}
                    try:
                        values = ktc.kneron_inference([tensor.copy()], nef_file=str(nef), input_names=["input"],
                                                      platform=720, data_type="float")
                        value, = values
                        require(np.asarray(value).size == 1 and np.isfinite(value).all(), "Invalid public NEF result")
                        public.update(value=float(np.asarray(value).item()), shape=list(np.asarray(value).shape))
                    except (Exception, SystemExit) as exc:
                        public["error"] = repr(exc)
                    public["seconds"] = time.monotonic() - start
                    receipt["smoke_public_nef"] = public
        require(sha256(nef) == sha256(original) == cp["nef_sha256"], "NEF changed during qualification")
        receipt["scalar_parser_metadata_compatibility"] = dict(scalar_parser.NEF_SCALAR_PARSER_COMPAT)
    require(len(receipt["rows"]) == (2400 if development else 48) and all(r["passed"] for r in receipt["rows"]),
            "Full panel is incomplete")
    receipt["max_abs_error"] = max(r["abs_error"] for r in receipt["rows"])
    receipt["transport_equals_doubled_native"] = sum(r["double_native_code"] for r in receipt["rows"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("compile", "simulate", "development-reference"))
    parser.add_argument("--eligibility-sha256", required=True)
    parser.add_argument("--compile-sha256")
    parser.add_argument("--training-simulation-sha256")
    args = parser.parse_args()
    require(Path("/workspace/version.txt").read_text().strip() == "kneron/toolchain:v0.33.1", "Wrong SDK")
    require(sha256(Path("/policy.json")) == NEF_POLICY_SHA, "Unexpected simulator protocol")
    policy = load_json(Path("/policy.json"))
    chain = load_json(Path("/eligibility.json"))
    require(sha256(Path("/eligibility.json")) == args.eligibility_sha256 and chain["all_eight_pass"] is True
            and sha256(BIE) == chain["bie_sha256"] == policy["bie_sha256"] and chain["nef_policy_sha256"] == NEF_POLICY_SHA,
            "Ineligible or changed BIE")
    receipt = {"stage": args.stage, "passed": False, "eligibility": chain, "physical_execution": False,
               "worker_sha256": sha256(__file__), "eligibility_sha256": args.eligibility_sha256}
    start = time.monotonic()
    try:
        if args.stage == "compile":
            compile_model(receipt)
        else:
            simulate(receipt, policy, args.compile_sha256, args.training_simulation_sha256)
        receipt["passed"] = True
    except (Exception, SystemExit) as exc:
        receipt["error"] = repr(exc)
        raise
    finally:
        receipt["stage_wall_seconds"] = time.monotonic() - start
        save_json(OUT / (args.stage + ".json"), receipt)


if __name__ == "__main__":
    main()
