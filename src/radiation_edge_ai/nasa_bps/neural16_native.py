"""Run sealed development tensors on the pinned Dynasty BIE simulator (Python 3.7)."""

import argparse
import csv
import inspect
import io
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

from . import multimodel_bie_inference as sim
from . import multimodel_ptq as ptq
from . import neural16_regression as qualification
from . import training_precision
from . import training_precision_readout as precision_readout
from .deployment_cohort import require_cohort


def check_archive(archive, expected, descriptor, chunk):
    """Read back raw bytes, text, returned scalar and sealed identities in manifest order."""
    ptq.require(len(archive.namelist()) == len(set(archive.namelist())), "Ambiguous native archive")
    rows = ptq.checked_csv(io.StringIO(archive.read("predictions.csv").decode("utf8"), newline=""))
    ptq.require(len(rows) == len(expected), "Incomplete chunk predictions")
    roots = {n.split("/")[0] for n in archive.namelist() if n.startswith("res_kdp720_")}
    ptq.require(len(roots) == 1, "Ambiguous native dump root")
    root = roots.pop()
    for i, (row, intended) in enumerate(zip(rows, expected)):
        ptq.require(all(row[k] == intended[k] for k in sim.IDENTITY + ("array_file",))
                    and row["chunk"] == chunk and int(row["row_in_chunk"]) == i, "Prediction/input misalignment")
        prefix = "{}/in_{:06d}.txt/mode_720_piano/layer_output_burden_".format(root, i)
        precision_readout.check_native_word_bytes(archive.read(prefix + "fx.bin"),
                                                  archive.read(prefix + "fx.txt"),
                                                  dict(row, fixed_word=row["bie_fixed_code"]), descriptor)
        vendor_float = sim.np.float32(archive.read(prefix + "fl.txt").strip())
        ptq.require(float(vendor_float) == float(row["bie_burden"]), "Archived native float differs")
    return rows


def rows_csv(rows):
    data = io.StringIO(newline="")
    writer = csv.DictWriter(data, fieldnames=sim.FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return data.getvalue()


def witnesses(ktc, fx, model, schema, args, descriptor):
    samples = ptq.checked_rows(args.training / "paired.csv")
    ptq.require(ptq.sha256(args.training / "paired.csv") == args.policy_data["training_predictions_sha256"],
                "Training witness table changed")
    calibration_file = args.calibration / "calibration_manifest.csv"
    ptq.require(ptq.sha256(calibration_file) == args.policy_data["calibration_manifest_sha256"],
                "Calibration membership changed")
    calibration = {r["sample_id"]: r for r in ptq.checked_rows(calibration_file)}
    controls = [samples[0], next(r for r in samples if r["fixed_word"] != samples[0]["fixed_word"])]
    arrays, singles = [], []
    for row in controls:
        member = calibration[row["sample_id"]]
        ptq.require(member[ptq.FOLD] == "train" and member["array_sha256"] == row["array_sha256"],
                    "Training control mismatch")
        arr = ptq.read_calibration_array(ptq.calibration_path(args.calibration, member["array_file"]),
                                         row["array_sha256"])
        arrays.append(arr)
        result = ktc.kneron_inference([arr], input_names=["input"], bie_file=str(model), platform=720,
                                      data_type="float")
        value, = sim.scalar_outputs({"burden": result}, 1)
        ptq.require(value == float(row["bie_burden"]), "Training singleton mismatch")
        singles.append(value)
    order = [1, 0, 1]
    path = args.out / "training_order.zip"
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        with zipfile.ZipFile(str(path), "x", zipfile.ZIP_DEFLATED) as z:
            try:
                values, codes = sim.native_batch(fx, model, schema, [arrays[i] for i in order], work, args.jobs)
                ptq.require(values == [singles[i] for i in order]
                            and codes == [int(controls[i]["fixed_word"]) for i in order],
                            "Training interleaved native order differs from frozen singletons")
                for i, (value, code) in enumerate(zip(values, codes)):
                    folder, = list(work.glob("res_kdp720_*/in_{:06d}.txt/mode_720_piano".format(i)))
                    precision_readout.check_native_word_bytes((folder / "layer_output_burden_fx.bin").read_bytes(),
                                                              (folder / "layer_output_burden_fx.txt").read_bytes(),
                                                              {"bie_burden": value,"fixed_word": code}, descriptor)
            finally:
                sim.capture_chunk(work, z)
    return {"file": path.name, "sha256": ptq.sha256(path), "ordered_indices": order,
            "sample_ids": [r["sample_id"] for r in controls], "public_singletons": singles,
            "native_burdens": values, "native_codes": codes}


def evaluate(args, rows, model, schema, fx, descriptor, report):
    completed = {item["file"]: item for item in report["chunks"]}
    table = []
    for start in range(0, len(rows), args.chunk):
        subset = rows[start:start + args.chunk]
        chunk = "{:04d}_{:04d}".format(start, start + len(subset) - 1)
        relative = "chunks/" + chunk + ".zip"
        artifact = args.out / relative
        tick = time.monotonic()
        if relative in completed:
            ptq.require(ptq.sha256(artifact) == completed[relative]["sha256"], "Sealed native archive changed")
        else:
            ptq.require(not artifact.exists(), "Unsealed prior attempt requires evidence recovery: " + relative)
            arrays = [sim.tensor(args.inputs, r) for r in subset]
            with tempfile.TemporaryDirectory() as temp:
                work = Path(temp)
                with zipfile.ZipFile(str(artifact), "x", zipfile.ZIP_DEFLATED) as z:
                    try:
                        values, codes = sim.native_batch(fx, model, schema, arrays, work, args.jobs)
                    finally:
                        sim.capture_chunk(work, z)
                    staged = [dict({k: r[k] for k in sim.IDENTITY}, array_file=r["array_file"],
                                   chunk=chunk, row_in_chunk=i, bie_burden=v, bie_fixed_code=c)
                              for i, (r, v, c) in enumerate(zip(subset, values, codes))]
                    z.writestr("predictions.csv", rows_csv(staged))
        with zipfile.ZipFile(str(artifact)) as z:
            table.extend(check_archive(z, subset, descriptor, chunk))
        if relative not in completed:
            report["chunks"].append({"file": relative, "sha256": ptq.sha256(artifact),
                                     "count": len(subset), "seconds": time.monotonic() - tick})
            report["evaluated_nuclei"] = len(table)
            ptq.save_json(args.out / "bie_receipt.json", report)
            print("Native results sealed: {}/{}".format(len(table), len(rows)), flush=True)
    ptq.require(len(table) == len(rows) == 2400, "Incomplete fixed simulator panel")
    final = args.out / "bie_nuclei.csv"
    data = rows_csv(table).encode("utf8")
    if final.exists():
        ptq.require(final.read_bytes() == data, "Existing final table differs from native archives")
    else:
        with final.open("xb") as stream:
            stream.write(data)
    return ptq.sha256(final)


def run(args):
    import fcntl
    import ktc
    from sys_flow.inference import get_model_io, inference_dynasty_fx

    policy = qualification.verify_policy(args.policy)
    args.policy_data = policy
    raw_sdk = ptq.sha256(inspect.getfile(inference_dynasty_fx))
    ptq.require(raw_sdk == ptq.FIXED_SIM_SHA and ptq.sha256(args.bie) == policy["bie_sha256"]
                and ptq.sha256(args.binding) == args.binding_sha, "Unexpected BIE, binding or simulator")
    bind = ptq.load_json(args.binding)
    rows, previous = sim.load_inputs(args.inputs, policy["historical_fp32_preparation_sha256"],
                                     policy["historical_policy_sha256"])
    require_cohort(rows, policy)
    ptq.require(bind["policy_sha256"] == qualification.POLICY_SHA and bind["bie_sha256"] == qualification.BIE_SHA
                and bind["input_manifest_sha256"] == previous["input_manifest_sha256"], "Binding mismatch")
    args.out.mkdir(exist_ok=True)
    (args.out / "chunks").mkdir(exist_ok=True)
    receipt = args.out / "bie_receipt.json"
    with (args.out / "run.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected = dict(policy_sha256=qualification.POLICY_SHA, preparation_sha256=args.binding_sha,
                        input_manifest_sha256=previous["input_manifest_sha256"], sdk_api=ptq.FIXED_SIM_API,
                        sdk_module_sha256=raw_sdk, worker_sha256=ptq.sha256(__file__), platform=720,
                        calibrated=False, physical_execution=False, nef_compiled=False, image_id=ptq.IMAGE_ID,
                        chunk_size=args.chunk, jobs=args.jobs,
                        **{k: policy[k] for k in ("architecture", "fold", "training_run_sha256", "bie_sha256",
                           "source_onnx_sha256", "optimized_onnx_sha256", "development_manifest_sha256",
                           "cohort_membership_sha256", "contrast_membership_sha256", "historical_fp32_preparation_sha256")})
        if receipt.exists():
            report = ptq.load_json(receipt)
            ptq.require(all(report[k] == v for k, v in expected.items()) and report["status"] in
                        ("SIMULATING", "STARTED"), "Cannot resume a failed, finished or different native study")
        else:
            report = dict(expected, started_utc=sim.utc(), status="STARTED", chunks=[], evaluated_nuclei=0,
                          error=None, scope=policy["scope"], public_transport_words_observed=False)
        ptq.save_json(receipt, report)
        try:
            with tempfile.TemporaryDirectory() as temp:
                model = Path(temp) / "model.bie"
                shutil.copyfile(str(args.bie), str(model))
                ptq.require(ptq.sha256(model) == qualification.BIE_SHA, "Working model digest mismatch")
                schema = get_model_io(model, 720)
                ptq.require(schema[0] == ["input"] and schema[1] == ["burden"] and
                            tuple(schema[2]["burden"]) == (1,), "Unexpected BIE I/O")
                metadata = Path(temp) / "metadata"
                desc = training_precision.scalar_descriptor(model, metadata)
                qualification.verify_descriptor(desc)
                ptq.require(desc == bind["output_descriptor"], "Native descriptors differ from training witnesses")
                if "training_witnesses" not in report:
                    if not (args.out / "metadata").exists():
                        shutil.copytree(str(metadata), str(args.out / "metadata"))
                    report["output_descriptor"] = desc
                    report["training_witnesses"] = witnesses(ktc, inference_dynasty_fx, model, schema, args, desc)
                ptq.require(ptq.sha256(args.out / "training_order.zip") == report["training_witnesses"]["sha256"],
                            "Training witness archive changed")
                report["status"] = "SIMULATING"
                ptq.save_json(receipt, report)
                digest = evaluate(args, rows, model, schema, inference_dynasty_fx, desc, report)
                ptq.require(ptq.sha256(model) == ptq.sha256(args.bie) == qualification.BIE_SHA
                            and ptq.sha256(args.binding) == args.binding_sha
                            and ptq.sha256(args.inputs / "inputs.csv") == report["input_manifest_sha256"],
                            "Scientific inputs changed during development simulation")
            report.update(status="BIE_SIMULATION_COMPLETE", predictions_sha256=digest)
        except (Exception, SystemExit) as exc:
            report.update(status="FAILED", error=repr(exc))
            raise
        finally:
            report["last_utc"] = sim.utc()
            ptq.save_json(receipt, report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("policy", "binding", "bie", "training", "inputs", "calibration", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--binding-sha", required=True)
    p.add_argument("--jobs", type=int, choices=range(1, 7), default=6)
    p.add_argument("--chunk", type=int, default=24, choices=(24, 48))
    run(p.parse_args())

