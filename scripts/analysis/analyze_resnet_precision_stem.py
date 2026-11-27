"""Measure the same FP32 suffix contribution on standard and full-INT16 native stem tensors."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

import numpy as np

from radiation_edge_ai.nasa_bps.feature_attribution import FeatureGraph, decode_layer, layout_scores
from radiation_edge_ai.nasa_bps.multimodel_ptq import checked_rows, load_json, require, sha256


STEM = {"stem_relu": "/features/features.2/Relu_output_0",
        "stem_pool": "/features/features.3/MaxPool_output_0"}
TAPS = dict(STEM, head="/head/Gemm_output_0", flatten="/Flatten_output_0", burden="burden")
MODEL = "3f654f267ae280f27d6aaf6715b32af4285c5de6f4eb13c844e02e70f2f69c9b"
GRAPH = "afb362866af2f61d51e7aae17ae11e262c4a551f790acf8ff8e7c104aa00573a"
SELECTION = "fecec721ddf8ad424ed9f5ac13594a0488d2fe6805aa0031aff11c18a602113a"
STANDARD = "f356dfd867cfdf63ce8dc6eb94ac72b6d6d32fd9f633156037c056dfa356011f"
READOUT = "e170017ede26e851a253eda87f33d609663b3641c81b89b3a2b8c3bb8eb73e82"
PILOT = "BPSR1V2D_01501"


def comparison(rows):
    out = {"n": len(rows)}
    for key in ("standard_increment", "full16_increment", "paired_change"):
        values = [r[key] for r in rows]
        out[key] = dict(mean=mean(values), mean_abs=mean(map(abs, values)),
                        min=min(values), max=max(values), negative=sum(x < 0 for x in values))
    return out


def main():
    import onnx
    import onnxruntime as ort

    p = argparse.ArgumentParser(description=__doc__)
    for key in ("selection", "bie", "capture", "inputs", "optimized", "standard", "readout", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    for path, expected in ((a.selection, SELECTION), (a.bie, MODEL), (a.optimized, GRAPH),
                           (a.standard, STANDARD), (a.readout / "paired.csv", READOUT)):
        require(sha256(path) == expected, "Unexpected scientific input: " + str(path))
    meta = a.readout / "metadata"
    metadata_hashes = {"SnrShapeInfo.json": "83cad30003764e4a5ab2ff8d14e5ceb47e6174a7b98a65bb541a30f807a9f6a5",
                       "radix.json": "86471161782c4de94d7d42cef8e6f98f3426d7e13de7e127d40fca2e60168c3d",
                       "ioinfo.json": "0e7ceb9e34396c640b6aeeec7e02bf9f42044aa5ded1e9543af6315418421e3c"}
    for name, expected in metadata_hashes.items():
        require(sha256(meta / name) == expected, "Changed descriptor: " + name)
    shapes = load_json(meta / "SnrShapeInfo.json")["dp_info"]
    radix = load_json(meta / "radix.json")
    terminal, = load_json(meta / "ioinfo.json")["output"]
    standard = {r["sample_id"]: r for r in checked_rows(a.standard)}
    scalar = {r["sample_id"]: r for r in checked_rows(a.readout / "paired.csv")}
    graph = FeatureGraph(a.optimized, TAPS)
    suffix = {label: graph.suffix(label) for label in STEM}
    pool_model = onnx.utils.Extractor(graph.source).extract_model([STEM["stem_relu"]], [STEM["stem_pool"]])
    require([n.op_type for n in pool_model.graph.node] == ["MaxPool"], "Unexpected local pooling graph")
    maxpool = ort.InferenceSession(pool_model.SerializeToString(), graph.options, providers=["CPUExecutionProvider"])
    archived = a.selection.parent
    require(sha256(archived / "analysis.json") == "d1112aebf7a4ff7f25e85c9c613145913c4883ce509cd71d8de26f91a712684f",
            "Changed standard BIE activation evidence")
    old_rows = {r["sample_id"]: r for r in load_json(archived / "analysis.json")["rows"]}
    rows = []
    members = load_json(a.selection)["rows"]
    signed_codes = []
    for r in members:
        sid = r["sample_id"]
        folder = a.capture / sid
        rec, log = load_json(folder / "receipt.json"), folder / "native.zip"
        require(rec["sample_id"] == sid and rec["bie_sha256"] == MODEL and rec["selection_sha256"] == SELECTION
                and rec["native_sha256"] == sha256(log) and rec["input_sha256"] == r["array_sha256"]
                == standard[sid]["input_sha256"] == scalar[sid]["array_sha256"], "Invalid native capture")
        tensor = a.inputs / r["array_file"].replace("\\", "/")
        require(sha256(tensor) == r["array_sha256"] and r["fold_female_role"] == "train", "Changed training tensor")
        fp = graph.run(np.load(tensor, allow_pickle=False))
        require(abs(fp["burden"].item() - float(standard[sid]["original_fp32"])) <= 1e-5, "FP32 replay failed")
        effects, errors, widths = {}, [], {}
        pool_step = None
        with ZipFile(log) as archive:
            for label, name in dict(STEM, burden="burden").items():
                info = shapes[name]
                node, = info["node_name"]
                native = decode_layer(archive.read(label + "_fx.bin"), archive.read(label + "_fl.txt"),
                                      archive.read(label + "_output_size.csv"), info, radix[node])
                widths[label] = native["bits"]
                errors.append(native["max_decode_abs"])
                if label == "burden":
                    word = native["codes"].item()
                    require(word == int(scalar[sid]["fixed_word"])
                            and native["native"].item() == rec["burden"] == float(scalar[sid]["bie_burden"]),
                            "Signed binary, simulator text and independent endpoint run disagree")
                    signed_codes.append(word)
                else:
                    layout_scores(native["decoded"], fp[label])
                    if label == "stem_pool":
                        pool_step = native["step"].astype(np.float64).reshape(1, -1, 1, 1)
                    run = suffix[label]
                    original = run.run(None, {name: fp[label]})[0].item()
                    require(abs(original - fp["head"].item()) <= 1e-5, "Original FP32 suffix did not replay")
                    effects[label] = run.run(None, {name: native["decoded"]})[0].item()
        standard_increment = float(standard[sid]["stem_pool"])
        new = effects["stem_pool"] - effects["stem_relu"]
        case = archived / sid
        require(sha256(case / "receipt.json") == old_rows[sid]["receipt_sha256"], "Changed standard capture receipt")
        old = load_json(case / "receipt.json")
        require(old["tensor_sha256"] == r["array_sha256"] and old["sample_id"] == sid
                and sha256(case / "activations.npz") == old["files"]["activations.npz"], "Changed standard stem")
        with np.load(case / "activations.npz", allow_pickle=False) as arrays:
            relu = arrays["stem_relu"].reshape(fp["stem_relu"].shape)
        held = suffix["stem_relu"].run(None, {STEM["stem_relu"]: relu})[0].item()
        floating = maxpool.run(None, {STEM["stem_relu"]: relu})[0]
        pool_reference = suffix["stem_pool"].run(None, {STEM["stem_pool"]: floating})[0].item()
        require(abs(held - pool_reference) <= 1e-5, "Pure FP32 pool changed the held suffix")
        grid = floating.astype(np.float64) / pool_step
        rounded = np.rint(grid)
        bound = 1 << (widths["stem_pool"] - 1)
        clipped = np.clip(rounded, -bound, bound - 1)
        fine_pool = (clipped * pool_step).astype(np.float32)
        result = suffix["stem_pool"].run(None, {STEM["stem_pool"]: fine_pool})[0].item()
        rows.append(dict(sample_id=sid, source_name=r["source_name"], panel="pilot" if sid == PILOT else "nonpilot",
                         input_sha256=r["array_sha256"], archive_sha256=rec["native_sha256"],
                         signed_native_word=word, bie=rec["burden"], standard_increment=standard_increment,
                         full16_increment=new, paired_change=new-standard_increment, **effects,
                         host_pool_only_increment=result-held, host_pool_clipped=int(np.count_nonzero(clipped != rounded)),
                         host_pool_exact_ties=int(np.count_nonzero(np.abs(grid - rounded) == .5)),
                         max_descriptor_decode_error=max(errors), **{k+"_bits": v for k,v in widths.items()}))
    nonpilot = [r for r in rows if r["panel"] == "nonpilot"]
    require(len(rows) == len(standard) == len(scalar) == 48 and len(nonpilot) == 47, "Incomplete comparison")
    a.out.mkdir(exist_ok=False)
    table = a.out / "paired_stem.csv"
    with table.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rec = dict(scope="descriptive_female_fold_training_only", model_sha256=MODEL, selection_sha256=SELECTION,
               standard_pairs_sha256=STANDARD, scalar_readout_sha256=READOUT, optimized_sha256=GRAPH,
               descriptor_hashes=metadata_hashes, code_sha256=sha256(__file__), paired_sha256=sha256(table),
               pilot=comparison([r for r in rows if r["panel"] == "pilot"]), nonpilot=comparison(nonpilot),
               by_source={s: comparison([r for r in nonpilot if r["source_name"] == s])
                          for s in sorted({r["source_name"] for r in nonpilot})},
               max_native_decode_error=max(r["max_descriptor_decode_error"] for r in rows),
               signed_internal_word_range=[min(signed_codes),max(signed_codes)],
               nonzero_signed_words=sum(x != 0 for x in signed_codes), transport_descriptor=terminal)
    rec["host_pool_only"] = {s: {"n": len(v), "mean_increment": mean(r["host_pool_only_increment"] for r in v),
                                 "mean_abs_increment": mean(abs(r["host_pool_only_increment"]) for r in v),
                                 "clipped": sum(r["host_pool_clipped"] for r in v),
                                 "exact_half_steps": sum(r["host_pool_exact_ties"] for r in v)}
                             for s,v in [("nonpilot",nonpilot)] + [
                                 (s,[r for r in nonpilot if r["source_name"] == s]) for s in rec["by_source"]]}
    (a.out / "receipt.json").write_text(json.dumps(rec, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
