"""Verify native ResNet word decoding and score the 24 common training inputs."""

import argparse
import csv
import ctypes
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from replay_resnet_common_training import (
    BIE, BIE_SHA, EXPORT, HELPER, IMAGE, MANIFEST, PANEL, PANEL_SHA, TAPS, check, digest,
    save, training_rows,
)


def vendor_metadata():
    out = Path("/out/metadata")
    out.mkdir()
    model = check(Path("/model.bie"), BIE_SHA)
    library = Path("/workspace/libs/dynasty/E2ESimulator/libdynasty.so")
    dump = ctypes.CDLL(str(library)).dump_radix_json
    dump.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
    dump.restype = None
    dump(720, os.fsencode(model), os.fsencode(out))
    save(out / "receipt.json", {"bie_sha256": BIE_SHA, "image": IMAGE, "library_sha256": digest(library),
                               "api": "libdynasty.so::dump_radix_json", "files": {
                                   k: digest(out / k) for k in ("radix.json", "SnrShapeInfo.json")}})


def launch_metadata(root, out):
    script = Path(__file__).resolve()
    mounts = [(script, "/score.py", True), (script.with_name("replay_resnet_common_training.py"),
              "/replay_resnet_common_training.py", True), (root/BIE, "/model.bie", True), (out, "/out", False)]
    command = ["docker", "run", "--rm", "--network", "none", "--pull", "never", "--cpus", "1", "--memory", "1g"]
    for src, target, ro in mounts:
        command += ["--mount", "type=bind,source={},target={}{}".format(src, target, ",readonly" if ro else "")]
    command += ["--entrypoint", "/workspace/miniconda/bin/python", IMAGE, "-B", "/score.py", "vendor"]
    save(out / "metadata_command.json", command)
    with (out / "metadata.log").open("xb") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def load_decoder(path):
    """Same implementation and commit as the MobileNet readout, including before branch integration."""
    expected = "f3570fb904a8c7f03b9e02c6cb835595d5ac8c2cad0a7c4e08eb120f6c01461b"
    if hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected:
        raise ValueError("Use the shared feature decoder from a3a299d")
    spec = importlib.util.spec_from_file_location("frozen_feature_attribution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_layer


def score(root, out, decoder, dest):
    from radiation_edge_ai.nasa_bps.resnet_training_features import (
        ANALYSIS, CAPTURE, FrozenGraph, checked_bytes, close, exact, write_csv,
    )

    start = time.perf_counter()
    decode = load_decoder(decoder)
    analysis = json.loads(checked_bytes(root / CAPTURE / "analysis.json", ANALYSIS))
    selection = json.loads((out / "selection.json").read_bytes())
    manifest = root / EXPORT / "calibration_manifest.csv"
    with check(manifest, MANIFEST).open(encoding="utf-8-sig") as stream:
        rows = training_rows(json.loads(check(root / PANEL / "selection.json", PANEL_SHA).read_bytes()),
                             list(csv.DictReader(stream)))
    if (selection["rows"] != rows or selection["bie_sha256"] != BIE_SHA or selection["workers"] != 2
            or selection["taps"] != TAPS or selection["common_panel_sha256"] != PANEL_SHA):
        raise ValueError("Common training panel or capture config differs")
    check(out / "common_capture.py", selection["source_sha256"])
    check(out / "shared_capture.py", HELPER)
    check(root / BIE, BIE_SHA)
    metadata = out / "metadata"
    meta = json.loads((metadata / "receipt.json").read_bytes())
    if meta["bie_sha256"] != BIE_SHA or meta["image"] != IMAGE:
        raise ValueError("Unexpected metadata export")
    shapes = json.loads(check(metadata / "SnrShapeInfo.json", meta["files"]["SnrShapeInfo.json"]).read_bytes())["dp_info"]
    radix = json.loads(check(metadata / "radix.json", meta["files"]["radix.json"]).read_bytes())
    graphs = [FrozenGraph(checked_bytes(root / path, sha)) for path, sha in
              (analysis["graphs"][k] for k in ("original", "optimized"))]
    exact(graphs[0].weights, graphs[1].weights, "effective head weights")
    exact(graphs[0].bias, graphs[1].bias, "effective head bias")
    archived = {r["sample_id"]: r for r in analysis["rows"]}
    paired, cross, tensors = [], [], []
    maximum = {k: 0.0 for k in TAPS}
    for row in rows:
        sid = row["sample_id"]
        folder = out / sid
        rec = json.loads((folder / "receipt.json").read_bytes())
        if (rec["sample_id"] != sid or rec["input_sha256"] != row["array_sha256"] or rec["bie_sha256"] != BIE_SHA
                or rec["selection_sha256"] != digest(out / "selection.json")):
            raise ValueError("Native sample receipt mismatch")
        x = np.load(check(root / EXPORT / row["array_file"].replace("\\", "/"), row["array_sha256"]), allow_pickle=False)
        fp, opt = [g.run(x) for g in graphs]
        for name in ("global_pool", "flatten", "head", "nonnegative", "burden"):
            exact(fp[name], opt[name], "original/optimized " + name)
        decoded, bins = {}, {}
        with ZipFile(check(folder / "native.zip", rec["native_sha256"])) as raw:
            if raw.testzip():
                raise ValueError("Native ZIP CRC failed")
            job, = csv.DictReader(io.StringIO(raw.read("jobs.tsv").decode()), delimiter="\t")
            if (job["Exitval"] != "0" or job["Signal"] != "0" or float(job["JobRuntime"]) != rec["job_seconds"]):
                raise ValueError("Native job status mismatch")
            for key, name in TAPS.items():
                shape = shapes[name]
                node, = shape["node_name"]
                bins[key] = raw.read(key + "_fx.bin")
                val = decode(bins[key], raw.read(key + "_fl.txt"), raw.read(key + "_output_size.csv"),
                             shape, radix[node])
                decoded[key] = val["decoded"]
                maximum[key] = max(maximum[key], val["max_decode_abs"])
        exact(decoded["input"], np.clip(np.rint(x * 128), -128, 127) / 128, "native input order")
        exact(decoded["pool"].ravel(), decoded["flatten"].ravel(), "native pool/flatten")
        close(rec["burden"], decoded["burden"].item(), "simulator returned burden", atol=1e-8)
        pool_pre = graphs[0].affine(decoded["pool"])
        original, bie = fp["burden"].item(), decoded["burden"].item()
        pooled = max(0, pool_pre)
        if sid in archived:
            old = archived[sid]
            old_case = root / CAPTURE / sid
            old_receipt = json.loads(check(old_case / "receipt.json", old["receipt_sha256"]).read_bytes())
            with ZipFile(check(old_case / "native.zip", old_receipt["files"]["native.zip"])) as old_zip:
                for key, prev_key in zip(TAPS, ("input", "global_pool", "flatten", "head", "nonnegative", "burden")):
                    prefix = old_receipt["taps"][prev_key]["text"][:-len("_fl.txt")]
                    if bins[key] != old_zip.read(prefix + "_fx.bin"):
                        raise ValueError("Replay differs from the historical native codes at " + key)
            close(pool_pre, old["heads"]["float_head_on_native_pool"], "old pooled affine", atol=1e-7)
            exact(original, old["predictions"]["original"], "old FP32")
            close(bie, old["predictions"]["bie"], "old BIE", atol=1e-8)
            cross.append(sid)
        paired.append(dict(sample_id=sid, source_name=row["source_name"], input_sha256=row["array_sha256"],
                           panel="confirmation", original_fp32=original, pooled_head_fp32=pooled, bie=bie,
                           actual_change=bie-original, upstream=pooled-original, downstream=bie-pooled,
                           original_pre=fp["head"].item(), pooled_pre=pool_pre, native_pre=decoded["head"].item(),
                           terminal_code=int.from_bytes(bins["burden"], "little", signed=True),
                           native_job_seconds=rec["job_seconds"], native_zip_sha256=rec["native_sha256"],
                           native_receipt_sha256=digest(folder / "receipt.json")))
        tensors.append((fp["global_pool"].ravel(), decoded["pool"].ravel()))
    if len(cross) != 4:
        raise ValueError("Expected four shared prior cases")
    dest.mkdir(exist_ok=False)
    write_csv(dest / "paired.csv", paired)
    with (dest / "feature_pairs.npz").open("xb") as stream:
        np.savez_compressed(stream, sample_id=np.asarray([r["sample_id"] for r in paired]),
                            original=np.stack([v[0] for v in tensors]), bie=np.stack([v[1] for v in tensors]),
                            head_weights=graphs[0].weights, head_bias=graphs[0].bias)
    shutil.copyfile(str(decoder), str(dest / "shared_decoder.py"))
    shutil.copyfile(str(Path(__file__)), str(dest / "analysis_source.py"))
    result = dict(schema="resnet_common_training_pool_v1", selection_sha256=digest(out / "selection.json"),
                  shared_mobilenet_selection_sha256=PANEL_SHA, graphs=analysis["graphs"],
                  historical_fixed_capture_analysis_sha256=ANALYSIS, vendor_metadata=meta,
                  capture_command_sha256=digest(out / "command.json"), capture_source_sha256=selection["source_sha256"],
                  helper_sha256=HELPER, successful_native_jobs=len(paired), original_capture_word_identical_cases=cross,
                  max_descriptor_decode_to_native_text=maximum, native_job_seconds_sum=sum(r["native_job_seconds"] for r in paired),
                  capture_wall_seconds=(json.loads((out/"execution.json").read_bytes())["elapsed_seconds"]
                                        if (out/"execution.json").exists() else None),
                  summary={source:{k: float(np.mean([r[k] for r in paired if source=="all" or r["source_name"]==source]))
                                   for k in ("actual_change","upstream","downstream")} for source in
                           ["all"] + sorted({r["source_name"] for r in paired})},
                  host_analysis_seconds=time.perf_counter()-start,
                  files={p.name:digest(p) for p in sorted(dest.iterdir())})
    save(dest / "receipt.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("metadata", "vendor", "score"))
    p.add_argument("--root", type=Path, default=Path("D:/radiation-edge-ai-data"))
    p.add_argument("--out", type=Path)
    p.add_argument("--decoder", type=Path, default=Path("src/radiation_edge_ai/nasa_bps/feature_attribution.py"))
    p.add_argument("--dest", type=Path)
    args = p.parse_args()
    if args.stage == "vendor":
        vendor_metadata()
    elif args.stage == "score":
        score(args.root, args.out, args.decoder, args.dest)
    else:
        launch_metadata(args.root, args.out)
