"""Run a prespecified training-only descriptive MobileNet pilot on the pinned KL720 fixed simulator."""
import argparse
import csv
import inspect
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from statistics import mean

if __package__:
    from . import multimodel_ptq as ptq
    from . import multimodel_bie_inference as simulator
else:
    import multimodel_ptq as ptq
    import multimodel_bie_inference as simulator


SOURCES = {"BALBCM1", "BALBCM2", "C57BLM1", "C57BLM3"}


def panel(rows):
    """First manifest row in each of the first four lexically ordered bags per training source."""
    groups = defaultdict(dict)
    for row in rows:
        ptq.require(row["source_name"] in SOURCES and row[ptq.FOLD] == "train"
                    and row["calibration_fold"] == ptq.FOLD
                    and row["dataset_role"] == "development_train_ptq_calibration",
                    "Non-training pilot source or role")
        groups[row["source_name"]].setdefault(row["sample_name"], row)
    ptq.require(len(rows) == 192 and set(groups) == SOURCES and all(len(g) == 12 for g in groups.values()),
                "Incomplete training-only pilot population")
    return [groups[source][bag] for source in sorted(groups) for bag in sorted(groups[source])[:4]]


def write_table(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def characterize(args):
    import ktc
    import numpy as np
    from sys_flow.inference import get_model_io, inference_dynasty_fx

    out = args.out
    receipt = out / "training_pilot.json"
    ptq.require(not receipt.exists(), "Training pilot requires a fresh output")
    current = dict(status="STARTED", purpose="descriptive_training_source_quantization_characterization",
                   architecture="mobilenet_v2", fold=ptq.FOLD, image_id=ptq.IMAGE_ID,
                   simulator=ptq.FIXED_SIM_API, selected_by="first_manifest_row_in_first_four_bags_per_train_source",
                   physical_execution=False, qualification_bags=0, nef_eligible=False)
    ptq.save_json(receipt, current)
    try:
        host = ptq.load_json(args.ptq / "host_ptq_receipt.json")
        worker_path = args.ptq / "pilot/ptq_receipt.json"
        worker = ptq.load_json(worker_path)
        ptq.require(host["status"] == worker["status"] == "PTQ_GENERATED_UNVALIDATED"
                    and host["worker_report_sha256"] == ptq.sha256(worker_path)
                    and worker["model_id"] == ptq.MOBILENET_FEMALE.model_id
                    and worker["input"]["source_onnx"]["sha256"] == ptq.MOBILENET_FEMALE.source
                    and host["bie_sha256"] == worker["bie"]["sha256"], "Changed PTQ provenance")
        manifest = args.inputs / "calibration_manifest.csv"
        pairs = args.ptq / "fp32_optimized_parity/fp32_optimized_parity.csv"
        parity_receipt = pairs.with_suffix(".json")
        ptq.require(ptq.sha256(manifest) == worker["input"]["calibration_manifest"]["sha256"]
                    and ptq.sha256(parity_receipt) == worker["optimized_fp32_parity_receipt_sha256"]
                    and ptq.sha256(pairs) == ptq.load_json(parity_receipt)["predictions_sha256"],
                    "Changed training inputs or parity table")
        selected = panel(ptq.checked_rows(manifest))
        paired = {r["sample_id"]: r for r in ptq.checked_rows(pairs)}
        samples, arrays = [], []
        for row in selected:
            sample = {k: row[k] for k in ("sample_id", "source_name", "sample_name", "array_sha256")}
            predictions = paired[row["sample_id"]]
            ptq.require(all(sample[k] == predictions[k] for k in ("sample_id", "source_name", "sample_name")),
                        "Training parity/panel membership mismatch")
            samples.append(dict(sample, source_fp32=float(predictions["source_fp32"]),
                                optimized_fp32=float(predictions["optimized_fp32"])))
            arrays.append(ptq.read_calibration_array(ptq.calibration_path(args.inputs, row["array_file"]),
                                                     row["array_sha256"]))
        write_table(out / "training_inputs.csv", samples)
        current.update(host_ptq_sha256=ptq.sha256(args.ptq / "host_ptq_receipt.json"),
                       worker_ptq_sha256=ptq.sha256(worker_path), bie_sha256=host["bie_sha256"],
                       selection_sha256=ptq.sha256(out / "training_inputs.csv"),
                       sdk_module_sha256=ptq.sha256(inspect.getfile(inference_dynasty_fx)))
        ptq.require(current["sdk_module_sha256"] == ptq.FIXED_SIM_SHA, "Changed fixed simulator implementation")
        ptq.save_json(receipt, current)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.bie"
            shutil.copyfile(str(args.ptq / "pilot/quantized_model.bie"), str(model))
            ptq.require(ptq.sha256(model) == current["bie_sha256"], "Changed BIE")
            schema = get_model_io(model, 720)
            ptq.require(schema[0] == ["input"] and schema[1] == ["burden"] and tuple(schema[2]["burden"]) == (1,),
                        "Unexpected BIE input or output contract")
            work = Path(directory) / "training"
            work.mkdir()
            try:
                values, codes = simulator.native_batch(inference_dynasty_fx, model, schema, arrays, work, 3)
            finally:
                with zipfile.ZipFile(str(out / "native_logs.zip"), "x", zipfile.ZIP_DEFLATED) as archive:
                    simulator.capture_chunk(work, archive)
            for index in (0, len(samples) - 1):
                official = ktc.kneron_inference([arrays[index]], input_names=["input"],
                                                bie_file=str(model), platform=720, data_type="float")
                result = np.asarray(official[0])
                ptq.require(len(official) == 1 and result.shape == (1,) and float(result[0]) == values[index],
                            "Public fixed-point API disagrees with native batch")
        for row, value, code in zip(samples, values, codes):
            row.update(bie_burden=value, bie_fixed_code=code, bie_minus_source=value - row["source_fp32"])
        write_table(out / "training_predictions.csv", samples)
        def summary(subset):
            errors = [r["bie_minus_source"] for r in subset]
            return {"n": len(subset), "bias": mean(errors), "mae": mean(map(abs, errors)),
                    "max_abs": max(map(abs, errors))}
        current.update(status="TRAINING_CHARACTERIZATION_COMPLETE", summary=summary(samples),
                       per_source={s: summary([r for r in samples if r["source_name"] == s]) for s in sorted(SOURCES)},
                       native_logs_sha256=ptq.sha256(out / "native_logs.zip"),
                       predictions_sha256=ptq.sha256(out / "training_predictions.csv"))
    except Exception as exc:
        current.update(status="FAILED", error=str(exc))
        raise
    finally:
        ptq.save_json(receipt, current)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "ptq", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    characterize(parser.parse_args())
