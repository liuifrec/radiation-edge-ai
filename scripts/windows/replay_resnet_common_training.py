"""Capture the unchanged ResNet standard BIE on the frozen MobileNet confirmation panel."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


PANEL = Path("scratch/mobilenet_feature_training_20261127T015528Z_795ec600")
PANEL_SHA = "534fffa2f104bf395952fe25ecc8708ca4b91bf032dc521b19298cf5fe35ebfe"
HELPER = "82dbd9ae23a8a3b54a79625d849f6355c39612623990d48ee5c96407d31df52f"
MANIFEST = "79aef81e921a723187d73cef5151ff9b943ad486d7e83af19daf063a5b5a1489"
EXPORT = Path("models/nasa_bps_53bp1/multimodel_v1_onnx_20261118T011220Z_f49930f7/resnet18/fold_female_role")
BIE = Path("scratch/nasa_resnet18_female_ptq_20261118T052424Z_61d42817/pilot/quantized_model.bie")
BIE_SHA = "2428c60742e862f3bc3b6a6831284a3afa2b9e4db652fa760f331e3500bbea2f"
IMAGE = "sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041"
TAPS = {"input": "input", "pool": "/pool/GlobalAveragePool_output_0", "flatten": "/Flatten_output_0",
        "head": "/head/Gemm_output_0", "relu": "/nonnegative/Relu_output_0", "burden": "burden"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Unexpected identity for {path.name}")
    return path


def save(path, data):
    with path.open("x", encoding="utf8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def training_rows(panel, calibration):
    selected = [r for r in panel["rows"] if r["panel"] == "confirmation"]
    by_id = {r["sample_id"]: r for r in calibration}
    if (len(selected) != 24 or len({r["sample_id"] for r in selected}) != 24
            or len({r["array_sha256"] for r in selected}) != 24 or len(by_id) != 192):
        raise ValueError("Unexpected training panel")
    sources = {}
    for row in selected:
        ref = by_id[row["sample_id"]]
        if (any(row[k] != v for k, v in ref.items()) or ref["fold_female_role"] != "train"
                or ref["dataset_role"] != "development_train_ptq_calibration" or ref["cohort"] != "development"
                or ref["calibration_fold"] != "fold_female_role"):
            raise ValueError("Mismatched training case")
        source = ref["source_name"]
        sources[source] = sources.get(source, 0) + 1
    if sources != {k: 6 for k in ("BALBCM1", "BALBCM2", "C57BLM1", "C57BLM3")}:
        raise ValueError("Expected six confirmation nuclei per training source")
    return selected


def capture_case(job):
    import shared_capture

    shared_capture.TAPS = TAPS
    shared_capture.capture_one(job)


def worker(out, selection_sha):
    check(Path("/shared_capture.py"), HELPER)
    rows = json.loads(check(out / "selection.json", selection_sha).read_bytes())["rows"]
    model = check(Path("/model.bie"), BIE_SHA)
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(capture_case, [(r, out, model, Path("/export"), selection_sha) for r in rows]))


def launch(root, out):
    panel = json.loads(check(root / PANEL / "selection.json", PANEL_SHA).read_bytes())
    helper = check(root / PANEL / "capture_source.py", HELPER)
    model = check(root / BIE, BIE_SHA)
    export = root / EXPORT
    with check(export / "calibration_manifest.csv", MANIFEST).open(encoding="utf-8-sig") as stream:
        rows = training_rows(panel, list(csv.DictReader(stream)))
    for row in rows:
        check(export / row["array_file"].replace("\\", "/"), row["array_sha256"])
    out.mkdir(exist_ok=False)
    script = Path(__file__).resolve()
    save(out / "selection.json", {"common_panel_sha256": PANEL_SHA, "training_manifest_sha256": MANIFEST,
                                 "bie_sha256": BIE_SHA, "helper_sha256": HELPER, "source_sha256": digest(script),
                                 "workers": 2, "taps": TAPS, "rows": rows})
    command = ["docker", "run", "--rm", "--network", "none", "--pull", "never", "--cpus", "2", "--memory", "2500m"]
    mounts = [(model, "/model.bie", True), (export / "calibration_nchw", "/export/calibration_nchw", True),
              (helper, "/shared_capture.py", True), (script, "/common.py", True), (out, "/out", False)]
    for src, dest, ro in mounts:
        command += ["--mount", "type=bind,source={},target={}{}".format(src, dest, ",readonly" if ro else "")]
    command += ["--entrypoint", "/workspace/miniconda/bin/python", IMAGE, "-u", "-B", "/common.py", "worker",
                "--out", "/out", "--selection-sha", digest(out / "selection.json")]
    shutil.copyfile(str(script), str(out / "common_capture.py"))
    shutil.copyfile(str(helper), str(out / "shared_capture.py"))
    save(out / "command.json", command)
    start = time.monotonic()
    with (out / "docker.log").open("xb") as log:
        job = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    save(out / "execution.json", {"returncode": job.returncode, "elapsed_seconds": time.monotonic() - start,
                                  "log_sha256": digest(out / "docker.log"), "command_sha256": digest(out / "command.json")})
    job.check_returncode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capture", "worker"))
    parser.add_argument("--root", type=Path, default=Path("D:/radiation-edge-ai-data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection-sha")
    args = parser.parse_args()
    if args.stage == "worker":
        worker(args.out, args.selection_sha)
    else:
        launch(args.root, args.out)
