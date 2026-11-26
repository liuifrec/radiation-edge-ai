"""Evaluate prepared ResNet18 female-fold tensors on the KL720 Dynasty fixed simulator.

Run with the pinned toolchain Python 3.7. Each call to inference_dynasty_fx takes
a list of independent batch-one inputs. Its float results are decoded fixed-point
outputs. The integer codes, native dumps and per-process logs are also retained.
"""
import argparse
import csv
import hashlib
import inspect
import io
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__:
    from .multimodel_ptq import ARCH, FOLD, EXPECTED_SOURCE_SHA, EXPECTED_TRAIN_SHA, IMAGE_ID
    from .multimodel_ptq import FIXED_SIM_API, FIXED_SIM_SHA, V1_POLICY_SHA, checked_csv
    from .multimodel_ptq import load_json, require, save_json, sha256
else:
    from multimodel_ptq import ARCH, FOLD, EXPECTED_SOURCE_SHA, EXPECTED_TRAIN_SHA, IMAGE_ID
    from multimodel_ptq import FIXED_SIM_API, FIXED_SIM_SHA, V1_POLICY_SHA, checked_csv
    from multimodel_ptq import load_json, require, save_json, sha256


EXPECTED = {
    "architecture": ARCH, "fold": FOLD,
    "training_run_sha256": EXPECTED_TRAIN_SHA, "source_onnx_sha256": EXPECTED_SOURCE_SHA,
    "bie_sha256": "2428c60742e862f3bc3b6a6831284a3afa2b9e4db652fa760f331e3500bbea2f",
    "optimized_onnx_sha256": "afb362866af2f61d51e7aae17ae11e262c4a551f790acf8ff8e7c104aa00573a",
    "development_manifest_sha256": "23e21fdc8bf00a2814b64c56223cfd3d8e96eda71bdbff5d8ec74dae218cf5a9",
}
IDENTITY = ("fold", "sample_id", "nucleus_key", "sample_name", "source_name", "array_sha256")
FIELDS = IDENTITY + ("array_file", "chunk", "row_in_chunk", "bie_burden", "bie_fixed_code")
SHAPE = (1, 3, 256, 256)


def utc():
    return datetime.now(timezone.utc).isoformat()


def load_inputs(root, prep_sha, policy_sha):
    prep = root / "preparation.json"
    require(sha256(prep) == prep_sha, "Preparation receipt SHA256 mismatch")
    p = load_json(prep)
    require(all(p.get(k) == v for k, v in EXPECTED.items()), "Model or development cohort mismatch")
    require(p.get("status") == "PREPARATION_COMPLETE" and p.get("evaluated_nuclei") == 2400
            and p.get("error") is None and p.get("policy_sha256") == policy_sha
            and policy_sha == V1_POLICY_SHA, "Incomplete or mismatched preparation")
    raw = (root / "inputs.csv").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == p["input_manifest_sha256"], "Input manifest changed")
    rows = checked_csv(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    require(len(rows) == 2400 and all(r["fold"] == FOLD for r in rows), "Expected all female-fold nuclei")
    for key in ("sample_id", "nucleus_key", "array_file"):
        require(len({r[key] for r in rows if r[key]}) == 2400, "Missing or repeated " + key)
    bags = Counter((r["source_name"], r["sample_name"]) for r in rows)
    require(len(bags) == 24 and set(bags.values()) == {100}, "Incomplete held-out bags")
    require(Counter(source for source, _ in bags) == {"BALBCF1": 12, "C57BLF1": 12},
            "Unexpected source-held-out panel")
    return rows, p


def tensor(root, row):
    require(re.fullmatch(r"tensors/[0-9]{4}\.npy", row["array_file"]), "Invalid evaluation tensor path")
    path = (root / row["array_file"]).resolve()
    require(path == root.resolve() / row["array_file"], "Evaluation tensor escapes prepared directory")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == row["array_sha256"], "Tensor SHA256 mismatch")
    value = np.load(io.BytesIO(raw), allow_pickle=False)
    require(value.shape == SHAPE and value.dtype == np.float32 and value.flags.c_contiguous
            and np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1
            and not np.any(value[:, 2]), "Invalid normalized NCHW evaluation tensor")
    return value


def scalar_outputs(values, n, fixed=False):
    require(set(values) == {"burden"} and len(values["burden"]) == n, "Incomplete burden outputs")
    result = []
    for item in values["burden"]:
        a = np.asarray(item)
        require(a.shape == (1,) and np.isfinite(a).all(), "Invalid scalar burden")
        if fixed:
            require(float(a[0]).is_integer(), "Nonintegral fixed-point code")
            result.append(int(a[0]))
        else:
            result.append(float(a[0]))
    return result


def capture_chunk(work, archive):
    """Archive native stdout, stderr, commands, exit status and raw scalar dumps."""
    patterns = ("run_dynasty*.sh", "jobs.tsv", "console/**/*", "res_*/**/layer_output_burden_*")
    for pattern in patterns:
        for path in sorted(work.glob(pattern)):
            if path.is_file():
                archive.write(str(path), path.relative_to(work).as_posix())


def verify_native_dumps(work, decoded, raw):
    # np2txt enumerates inputs as in_000000.txt, in_000001.txt, etc.
    # Cross-check the SDK's return order against those actual per-image files.
    roots = list(work.glob("res_kdp720_*"))
    require(len(roots) == 1, "Missing or ambiguous native result directory")
    names = {"in_{:06d}.txt".format(i) for i in range(len(decoded))}
    require({d.name for d in roots[0].iterdir() if d.is_dir()} == names, "Incomplete native result IDs")
    for i, (value, code) in enumerate(zip(decoded, raw)):
        folder = roots[0] / "in_{:06d}.txt".format(i) / "mode_720_piano"
        saved_value = np.float32((folder / "layer_output_burden_fl.txt").read_text().strip())
        saved_code = int((folder / "layer_output_burden_fx.txt").read_text().strip())
        require(float(saved_value) == value and code == saved_code,
                "Native sample/output ordering mismatch at offset {}".format(i))
        console = work / "console" / str(i + 1)
        require((console / "stdout").is_file() and (console / "stderr").is_file(),
                "Missing native simulator console logs")


def native_batch(fx, model, model_io, arrays, work, jobs):
    nodes, _, shapes, io_info = model_io
    env = os.environ.get("PARALLEL")
    try:
        os.environ["PARALLEL"] = "--jobs {} --results {}/console/{{#}}/ --joblog {}/jobs.tsv".format(
            jobs, work, work)
        decoded, raw = fx(model, 720, {"input": arrays}, input_nodes=nodes, out_node_shape=shapes,
                          d_ioinfo=io_info, p_working=work)
    finally:
        if env is None:
            os.environ.pop("PARALLEL", None)
        else:
            os.environ["PARALLEL"] = env
    with (work / "jobs.tsv").open(encoding="utf-8", newline="") as stream:
        statuses = list(csv.DictReader(stream, delimiter="\t"))
    require(sorted(int(r["Seq"]) for r in statuses) == list(range(1, 1 + len(arrays))) and
            all(r["Exitval"] == "0" and r["Signal"] == "0" for r in statuses), "Incomplete vendor job log")
    decoded = scalar_outputs(decoded, len(arrays))
    raw = scalar_outputs(raw, len(arrays), fixed=True)
    verify_native_dumps(work, decoded, raw)
    return decoded, raw


def simulator_probe(ktc, fx, model, model_io, out, jobs):
    zero = np.zeros(SHAPE, np.float32)
    fitc, dapi = zero.copy(), zero.copy()
    fitc[:, 0, 64:188, 64:188] = 1
    dapi[:, 1, 32:190, 60:120] = 0.875
    arrays = [zero, fitc, dapi]
    started = time.monotonic()
    singles = []
    for arr in arrays:
        result = ktc.kneron_inference([arr], input_names=["input"], bie_file=str(model),
                                      platform=720, data_type="float")
        require(len(result) == 1, "Unexpected public simulator output count")
        singles.extend(scalar_outputs({"burden": result}, 1))
    single_seconds = time.monotonic() - started
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        started = time.monotonic()
        try:
            decoded, raw = native_batch(fx, model, model_io, arrays, work, jobs)
        finally:
            with zipfile.ZipFile(str(out / "synthetic_sdk_logs.zip"), "x", zipfile.ZIP_DEFLATED) as archive:
                capture_chunk(work, archive)
    require(decoded == singles and len(set(raw)) == len(raw),
            "Native multi-image fixed-point outputs disagree with the ordered single-image calls")
    return {"inputs": ["zero", "fitc_patch", "dapi_patch"], "public_burden": singles,
            "native_burden": decoded, "native_fixed_codes": raw, "single_seconds": single_seconds,
            "native_seconds_including_log_capture": time.monotonic() - started,
            "logs_sha256": sha256(out / "synthetic_sdk_logs.zip")}


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = args.out / "bie_receipt.json"
    report = dict(EXPECTED, status="STARTED", started_utc=utc(), error=None, evaluated_nuclei=0,
                  preparation_sha256=args.preparation_sha256, policy_sha256=args.policy_sha256,
                  image_id=IMAGE_ID, chunks=[], jobs=args.jobs, chunk_size=args.chunk_size,
                  worker_sha256=sha256(__file__), sdk_api=FIXED_SIM_API,
                  platform=720, calibrated=False, nef_compiled=False, physical_execution=False)
    save_json(receipt, report)
    try:
        import ktc
        from sys_flow.inference import get_model_io, inference_dynasty_fx

        rows, prep = load_inputs(args.inputs, args.preparation_sha256, args.policy_sha256)
        report.update(input_manifest_sha256=prep["input_manifest_sha256"], python_version=sys.version,
                      sdk_module_sha256=sha256(inspect.getfile(inference_dynasty_fx)),
                      public_api_sha256=sha256(inspect.getfile(ktc.kneron_inference)))
        require(report["sdk_module_sha256"] == FIXED_SIM_SHA, "Unexpected fixed-point SDK identity")
        require(sha256(args.bie) == EXPECTED["bie_sha256"], "Frozen BIE SHA256 mismatch")
        chunks = args.out / "chunks"
        chunks.mkdir()
        with tempfile.TemporaryDirectory() as storage:
            model = Path(storage) / "model.bie"
            shutil.copyfile(str(args.bie), str(model))
            require(sha256(model) == EXPECTED["bie_sha256"], "Simulator model copy changed")
            schema = get_model_io(model, 720)
            require(schema[0] == ["input"] and schema[1] == ["burden"] and
                    tuple(schema[2]["burden"]) == (1,), "Unexpected BIE tensor contract")
            report["synthetic_probe"] = simulator_probe(ktc, inference_dynasty_fx, model, schema,
                                                        args.out, args.jobs)
            report["status"] = "SIMULATING"
            save_json(receipt, report)
            predictions = args.out / "bie_nuclei.csv"
            with predictions.open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                for start in range(0, len(rows), args.chunk_size):
                    subset = rows[start:start + args.chunk_size]
                    arrays = [tensor(args.inputs, r) for r in subset]
                    name = "{:04d}_{:04d}".format(start, start + len(subset) - 1)
                    artifact = chunks / (name + ".zip")
                    print("Starting fixed-point simulation {}".format(name), flush=True)
                    began = time.monotonic()
                    with tempfile.TemporaryDirectory() as temp:
                        work = Path(temp)
                        with zipfile.ZipFile(str(artifact), "x", zipfile.ZIP_DEFLATED) as archive:
                            try:
                                decoded, raw = native_batch(inference_dynasty_fx, model, schema,
                                                            arrays, work, args.jobs)
                            finally:
                                capture_chunk(work, archive)
                            staged = io.StringIO(newline="")
                            chunk_csv = csv.DictWriter(staged, fieldnames=FIELDS)
                            chunk_csv.writeheader()
                            output_rows = []
                            for i, r in enumerate(subset):
                                pred = {k: r[k] for k in IDENTITY}
                                pred.update(array_file=r["array_file"], chunk=name, row_in_chunk=i,
                                            bie_burden=decoded[i], bie_fixed_code=raw[i])
                                output_rows.append(pred)
                            chunk_csv.writerows(output_rows)
                            archive.writestr("predictions.csv", staged.getvalue())
                    writer.writerows(output_rows)
                    stream.flush()
                    report["chunks"].append({"file": "chunks/" + artifact.name, "sha256": sha256(artifact),
                                             "count": len(subset), "seconds": time.monotonic() - began})
                    report["evaluated_nuclei"] += len(subset)
                    save_json(receipt, report)
                    print("Completed {}/{} nuclei".format(report["evaluated_nuclei"], len(rows)), flush=True)
            require(sha256(model) == EXPECTED["bie_sha256"], "Working BIE changed during simulation")
        require(sha256(args.bie) == EXPECTED["bie_sha256"] and
                sha256(args.inputs / "inputs.csv") == report["input_manifest_sha256"] and
                sha256(args.inputs / "preparation.json") == args.preparation_sha256,
                "Input evidence changed during simulation")
        report.update(status="BIE_SIMULATION_COMPLETE", predictions_sha256=sha256(predictions))
    except (Exception, SystemExit) as exc:
        report.update(status="FAILED", error="{}: {}".format(type(exc).__name__, exc))
        traceback.print_exc()
    finally:
        report["completed_utc"] = utc()
        save_json(receipt, report)
    return 0 if report["status"] == "BIE_SIMULATION_COMPLETE" else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "bie", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--policy-sha256", required=True)
    p.add_argument("--preparation-sha256", required=True)
    p.add_argument("--jobs", type=int, choices=range(1, 7), default=3)
    p.add_argument("--chunk-size", type=int, choices=range(1, 49), default=24)
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
