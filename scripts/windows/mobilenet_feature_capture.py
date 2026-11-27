"""Freeze the training panel and capture named tensors from the unchanged MobileNetV2 BIE."""

import argparse
import csv
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


IMAGE = "sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041"
EXPORT = Path(
    "models/nasa_bps_53bp1/multimodel_v1_onnx_20261118T002659Z_47b5d92a/mobilenet_v2/fold_female_role"
)
RUN = Path("scratch/nasa_mobilenet_female_int8_20261126T234932Z_a7c472a7")
SCREEN = Path(
    "scratch/nasa_trained_graph_screen_20261118T020501Z_06255b35/vendor/mobilenet_v2_fold_female_role"
)
META = Path("scratch/nasa_mobilenet_training_terminal_decode_20261127T001431Z_8a4ad6fb/metadata")
FILES = {
    "manifest": (
        EXPORT / "calibration_manifest.csv",
        "79aef81e921a723187d73cef5151ff9b943ad486d7e83af19daf063a5b5a1489",
    ),
    "parity": (
        RUN / "fp32_optimized_parity/fp32_optimized_parity.csv",
        "208d8ca94df50694d19a75ba176297c32c842c805bdad167688b1d6056da29e7",
    ),
    "pilot": (
        RUN / "training_pilot_selection.json",
        "6705097bfb99f6038dd82aaf6acbe46c3c0ccd9ce20cc902d524058ac3ca6f2f",
    ),
    "pilot_predictions": (
        RUN / "training_pilot/training_predictions.csv",
        "4faef21b9ae90e3147c3e9fa00f9a3b57959a99a81c1e8f5425c6a8a58f988d7",
    ),
    "bie": (
        RUN / "pilot/quantized_model.bie",
        "2102c400d696bc385e23d2d9b978b779e4c95b82661272e6873ac1eb57607e26",
    ),
    "original": (
        EXPORT / "mobilenet_v2_fold_female_role_fp32_opset11.onnx",
        "2f18b2330ca3289ab422e1ff47b526b4bfa6a46fc4bd7354346609715f575e45",
    ),
    "optimized": (
        SCREEN / "optimized.onnx",
        "75824e1ed4f8520d90a942c1351926b922db9b4284b8b6d7cd5663428e73c074",
    ),
    "export": (
        EXPORT / "export.json",
        "40b89d4b349dfa4060e4fd62a31cf27647b6fb04c1bdef93b4f7b84165f32257",
    ),
    "shapes": (
        META / "SnrShapeInfo.json",
        "7491051f8633ce92ec0e4a7c62efa4610d2e0c86e84cb5e6eda588b9d2240c44",
    ),
    "radix": (
        META / "radix.json",
        "15d42a2a05c3e5ebcbc7c60c44444e3a949c14697d0737feb61ace8db4554dea",
    ),
}
TAPS = {
    "input": "input",
    "stem_conv": "/features/features.0/features.0.0/Conv_output_0",
    "stem_clip": "/features/features.0/features.0.2/Clip_output_0",
    "block1": "/features/features.1/conv/conv.1/Conv_output_0",
    "block2": "/features/features.2/conv/conv.2/Conv_output_0",
    "block13": "/features/features.13/Add_output_0",
    "block17": "/features/features.17/conv/conv.2/Conv_output_0",
    "spatial": "/features/features.18/features.18.2/Clip_output_0",
    "pool": "/pool/GlobalAveragePool_output_0",
    "flatten": "/Flatten_output_0",
    "head": "/head/Gemm_output_0",
    "relu": "/nonnegative/Relu_output_0",
    "burden": "burden",
}
SOURCES = ("BALBCM1", "BALBCM2", "C57BLM1", "C57BLM3")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def checked(path, expected):
    check(digest(path) == expected, "Changed file " + str(path))
    return path


def read_json(path):
    return json.loads(path.read_bytes())


def save_json(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def select_panel(manifest, predictions, pilot):
    rows = {r["sample_id"]: r for r in manifest}
    fp = {r["sample_id"]: float(r["source_fp32"]) for r in predictions}
    check(
        len(rows) == len(fp) == len(manifest) == len(predictions) == 192 and set(rows) == set(fp),
        "Expected the frozen calibration census",
    )
    check(len(set(pilot)) == len(pilot) == 16 and set(pilot) <= set(rows), "Invalid pilot")
    check(
        all(
            r["fold_female_role"] == "train"
            and r["calibration_fold"] == "fold_female_role"
            and r["cohort"] == "development"
            and r["dataset_role"] == "development_train_ptq_calibration"
            and r["source_name"] in SOURCES
            for r in manifest
        ),
        "Nontraining row",
    )
    check(all(np.isfinite(v) and v >= 0 for v in fp.values()), "Unexpected reference")
    labels = [(key, "pilot", "archived") for key in pilot]
    for source in SOURCES:
        ranked = sorted(
            (k for k in rows if rows[k]["source_name"] == source and k not in pilot),
            key=lambda k: (fp[k], k),
        )
        check(len(ranked) == 44, "Expected 44 remaining nuclei per source")
        first = ranked[0]
        labels.append((first, "confirmation", "zero_or_min"))
        positives = [k for k in ranked if k != first and fp[k] > 0]
        check(len(positives) >= 5, "Insufficient positive FP32 predictions")
        used = set()
        for q in (0, 0.25, 0.5, 0.75, 1):
            target = q * (len(positives) - 1)
            i = min(
                (i for i in range(len(positives)) if i not in used),
                key=lambda i: (abs(i - target), positives[i]),
            )
            used.add(i)
            labels.append((positives[i], "confirmation", "positive_q" + str(q)))
    return [dict(rows[k], panel=p, stratum=t, source_fp32=fp[k]) for k, p, t in labels]


def prepare(root, out):
    paths = {name: checked(root / path, sha) for name, (path, sha) in FILES.items()}
    export = read_json(paths["export"])
    checkpoint = Path(export["checkpoint"]["path"])
    checked(checkpoint, export["checkpoint"]["sha256"])
    rows = select_panel(
        table(paths["manifest"]), table(paths["parity"]), read_json(paths["pilot"])["sample_ids"]
    )
    for row in rows:
        checked(root / EXPORT / row["array_file"].replace("\\", "/"), row["array_sha256"])
    out.mkdir()
    save_json(
        out / "selection.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            architecture="mobilenet_v2",
            role="development_train_ptq_calibration",
            gated_for_training_only=True,
            image=IMAGE,
            inputs={k: h for k, (_, h) in FILES.items()},
            checkpoint=export["checkpoint"]["sha256"],
            definition="Six nonpilot nuclei per source: zero/min and five evenly spaced positive reference ranks; nearest distinct rank, sample ID tiebreak",
            capture_source_sha256=digest(__file__),
            rows=rows,
        ),
    )
    print(out, digest(out / "selection.json"), flush=True)


def capture_one(args):
    from sys_flow.inference import get_model_io, inference_dynasty_fx

    row, out, model, arrays, selection_sha = args
    case = out / row["sample_id"]
    case.mkdir()
    start = time.monotonic()
    x = np.load(
        str(checked(arrays / row["array_file"].replace("\\", "/"), row["array_sha256"])),
        allow_pickle=False,
    )
    check(
        x.shape == (1, 3, 256, 256) and x.dtype == np.float32 and np.isfinite(x).all(),
        "Unexpected input",
    )
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        bie = work / "model.bie"
        shutil.copyfile(str(model), str(bie))
        nodes, outputs, shapes, info = get_model_io(bie, 720)
        check(nodes == ["input"] and outputs == ["burden"], "Wrong model I/O")
        # Each process owns its native PARALLEL environment and temporary workspace.
        os.environ["PARALLEL"] = (
            "--jobs 1 --results {0}/console/{{#}}/ --joblog {0}/jobs.tsv".format(work)
        )
        result, _ = inference_dynasty_fx(
            bie,
            720,
            {"input": [x]},
            input_nodes=nodes,
            out_node_shape=shapes,
            d_ioinfo=info,
            p_working=work,
            dump_level=2,
        )
        jobs = work / "jobs.tsv"
        with jobs.open() as stream:
            (job,) = csv.DictReader(stream, delimiter="\t")
        check(job["Exitval"] == job["Signal"] == "0", "Native simulator error")
        (folder,) = work.glob("res_kdp720_*/in_000000.txt/mode_720_piano")
        with ZipFile(str(work / "native.zip"), "x", ZIP_DEFLATED, compresslevel=1) as archive:
            for label, name in TAPS.items():
                prefix = {"input": "layer_input_input", "burden": "layer_output_burden"}.get(
                    label, "layer_" + name.replace("/", "_")
                )
                for suffix in ("_fx.bin", "_output_size.csv", "_fl.txt"):
                    path = folder / (prefix + suffix)
                    archive.write(str(path), label + suffix)
            archive.write(str(jobs), "jobs.tsv")
            for path in (work / "console/1").iterdir():
                if path.is_file():
                    archive.write(str(path), "console/" + path.name)
        shutil.copyfile(str(work / "native.zip"), str(case / "native.zip"))
        save_json(
            case / "receipt.json",
            dict(
                input_sha256=row["array_sha256"],
                sample_id=row["sample_id"],
                selection_sha256=selection_sha,
                bie_sha256=digest(bie),
                native_sha256=digest(case / "native.zip"),
                burden=float(np.asarray(result["burden"][0]).item()),
                job_seconds=float(job["JobRuntime"]),
                elapsed_seconds=time.monotonic() - start,
                sdk_sha256=digest(inspect.getfile(inference_dynasty_fx)),
            ),
        )
    print(row["sample_id"], "captured", flush=True)


def worker(out, selection_sha):
    from concurrent.futures import ProcessPoolExecutor

    record = read_json(checked(out / "selection.json", selection_sha))
    checked(Path(__file__), record["capture_source_sha256"])
    rows = record["rows"]
    model = checked(Path("/model.bie"), FILES["bie"][1])
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(capture_one, [(r, out, model, Path("/export"), selection_sha) for r in rows]))


def launch(root, out):
    sha = digest(out / "selection.json")
    check(not (out / "command.json").exists(), "Use a fresh output directory")
    script = checked(
        Path(__file__).resolve(), read_json(out / "selection.json")["capture_source_sha256"]
    )
    snapshot = out / "capture_source.py"
    shutil.copyfile(str(script), str(snapshot))
    binds = [
        (root / EXPORT, "/export", ",readonly"),
        (root / FILES["bie"][0], "/model.bie", ",readonly"),
        (snapshot, "/capture.py", ",readonly"),
        (out, "/out", ""),
    ]
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--pull",
        "never",
        "--cpus",
        "4",
        "--memory",
        "5g",
    ]
    for source, target, mode in binds:
        command.extend(["--mount", "type=bind,source={},target={}{}".format(source, target, mode)])
    command.extend(
        [
            "--entrypoint",
            "/workspace/miniconda/bin/python",
            IMAGE,
            "-u",
            "-B",
            "/capture.py",
            "worker",
            "--out",
            "/out",
            "--selection-sha",
            sha,
        ]
    )
    save_json(out / "command.json", command)
    with (out / "docker.log").open("xb") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "capture", "worker"))
    parser.add_argument("--root", type=Path, default=Path("D:/radiation-edge-ai-data"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--selection-sha")
    args = parser.parse_args()
    if args.stage == "worker":
        worker(args.out, args.selection_sha)
    elif args.stage == "prepare":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
        prepare(
            args.root,
            args.out
            or args.root / "scratch" / ("mobilenet_feature_training_" + stamp + uuid4().hex[:8]),
        )
    else:
        launch(args.root, args.out)
