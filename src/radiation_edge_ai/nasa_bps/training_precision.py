"""Training-only precision conversion for frozen, screened female-fold NASA networks.

Run as a standalone script with the pinned Python 3.7 vendor image. The caller
supplies a prospectively sealed recipe, input set and separate output mount.
"""

import argparse
import ctypes
import inspect
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

if __package__:
    from . import multimodel_ptq as ptq
else:
    import multimodel_ptq as ptq


SDK_SHA256 = "e37d836d52651a3bc897b432e916ad36b77f9acd6a8c3b72f651a5ff3acb566a"
POLICY_SHA256 = "9b50fefcb86e6b01df766c0506fc545b807ac2fc4f9303ebdf029d18c3744dd9"
RECIPES = {
    "datapath16": {"datapath_bitwidth_mode": "int16"},
    "neural16": dict.fromkeys(
        (
            "datapath_bitwidth_mode",
            "weight_bitwidth_mode",
            "model_in_bitwidth_mode",
            "model_out_bitwidth_mode",
        ),
        "int16",
    ),
}
SETTINGS = {
    "input": "model_in_bitwidth_mode",
    "output": "model_out_bitwidth_mode",
    "weight": "weight_bitwidth_mode",
    "datapath": "datapath_bitwidth_mode",
    "cpu": "cpu_node_bitwidth_mode",
}


def precision_rows(path):
    for table in re.findall(
        r"<table\b.*?</table>", path.read_text(encoding="utf8"), flags=re.DOTALL
    ):
        tree = ET.fromstring(table)
        headers = [h.text for h in tree.findall("./thead/tr/th")]
        keys = ("node", "type", "bw in", "bw out", "bw weight")
        if all(k in headers for k in keys):
            rows = [dict(zip(headers, [v.text for v in row])) for row in tree.findall("./tbody/tr")]
            ptq.require(rows and all(r["node"] for r in rows), "Empty layer precision table")
            return [{k: row[k] for k in keys} for row in rows]
    raise ValueError("Missing native per-node width report")


def verify_report(directory, bie, recipe):
    recipe_args = RECIPES[recipe]
    widths = {k: recipe_args.get(setting, "int8") for k, setting in SETTINGS.items()}
    report_path = directory / "model_fx_report.json"
    report = ptq.load_json(report_path)
    ptq.require(
        report["docker_version"] == "kneron/toolchain:v0.33.1"
        and all(report.get("kdp720/" + k + " bitwidth") == v for k, v in widths.items()),
        "Effective precision summary differs from request",
    )
    ptq.require(
        report["kdp720/bie"] == "input.kdp720.scaled.bie"
        and ptq.sha256(directory / report["kdp720/bie"]) == ptq.sha256(bie),
        "BIE differs from report",
    )
    html = directory / "model_fx_report.html"
    rows = precision_rows(html)
    ptq.require(
        all(r["type"] == "NPU" for r in rows) and report.get("kdp720/ip_eval/cpu_node") == "N/A",
        "Model requires CPU fallback in the vendor report",
    )
    return {
        "requested": recipe_args,
        "reported": widths,
        "report_sha256": ptq.sha256(report_path),
        "html_sha256": ptq.sha256(html),
        "per_node": rows,
        "actual_counts": {
            k: dict(Counter(r[k] for r in rows)) for k in ("bw in", "bw out", "bw weight")
        },
        "ip_estimates": {k: v for k, v in report.items() if k.startswith("kdp720/ip_eval/")},
    }


def quantize(args):
    import ktc

    case = ptq.trained_case(args)
    out = Path(args.out)
    record = out / "quantization.json"
    ptq.require(
        not record.exists() and not (out / "kneron_flow").exists(), "Fresh PTQ output required"
    )
    result = {
        "status": "STARTED",
        "architecture": case.architecture,
        "model_id": case.model_id,
        "platform": "720",
        "fold": ptq.FOLD,
        "recipe": args.recipe,
        "image_id": ptq.IMAGE_ID,
        "calibrated": False,
        "nef_compiled": False,
        "physical_execution": False,
        "worker_sha256": ptq.sha256(__file__),
    }
    start = time.monotonic()
    try:
        sdk = ktc.ModelConfig.analysis
        ptq.require(ptq.sha256(inspect.getfile(sdk)) == SDK_SHA256, "SDK implementation changed")
        signature = inspect.signature(sdk).parameters
        defaults = {k: signature[k].default for k in SETTINGS.values()}
        ptq.require(
            set(defaults.values()) == {"int8"} and signature["optimize"].default == 0,
            "Unexpected SDK width or optimization defaults",
        )
        result.update(sdk_sha256=SDK_SHA256, sdk_signature=str(inspect.signature(sdk)))
        policy = ptq.load_json(args.policy)
        policy_sha = ptq.sha256(args.policy)
        expected_sha = (
            POLICY_SHA256
            if case.architecture == "mobilenet_v2"
            else "9cecad4e39979589ef211382810a0515ef81cc8ab30c079eb50e88e5782916aa"
        )
        ptq.require(
            policy_sha == expected_sha
            and args.recipe in policy["recipes"]
            and policy["recipes"][args.recipe] == RECIPES[args.recipe]
            and policy["parity_sha256"] == ptq.sha256(args.parity_receipt),
            "Training precision protocol, recipe or parity mismatch",
        )
        result["policy_sha256"] = policy_sha
        ptq.verify_parity_receipt(args.parity_receipt, args)
        check = ptq.preflight(args)
        result.update(
            input=check["input_record"],
            parity_sha256=ptq.sha256(args.parity_receipt),
            checkpoint_sha256=ptq.load_json(args.export)["checkpoint"]["sha256"],
        )
        model = ktc.ModelConfig(
            case.model_id, "0001", "720", onnx_model=ptq.vendor_model(args.optimized)
        )
        result.update(status="ANALYSIS_STARTED", overrides=RECIPES[args.recipe])
        ptq.save_json(record, result)
        began = time.monotonic()
        produced = model.analysis(
            {"input": check["arrays"]},
            output_dir=str(out / "kneron_flow"),
            threads=2,
            **RECIPES[args.recipe],
        )
        result.update(ptq_seconds=time.monotonic() - began, calibrated=True)
        bie = Path(produced).resolve()
        ptq.require(
            bie.parent == (out / "kneron_flow").resolve()
            and bie.suffix == ".bie"
            and bie.is_file(),
            "Unexpected SDK output location",
        )
        result["bie"] = dict(ptq.identity(bie), path=bie.relative_to(out.resolve()).as_posix())
        result["precision"] = verify_report(out / "kneron_flow", bie, args.recipe)
        result.update(status="TRAINING_BIE_GENERATED")
    except (Exception, SystemExit) as exc:
        stage = result["status"]
        status = "BLOCKED" if stage == "ANALYSIS_STARTED" else "PREFLIGHT_FAILED"
        result.update(status=status, failed_stage=stage, error=repr(exc))
        raise
    finally:
        result["seconds"] = time.monotonic() - start
        ptq.save_json(record, result)
    return result


def scalar_descriptor(model, out):
    libs = Path("/workspace/libs/dynasty/E2ESimulator/libdynasty.so")
    out.mkdir(exist_ok=False)
    dump = ctypes.CDLL(str(libs)).dump_radix_json
    dump.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
    dump.restype = None
    dump(720, os.fsencode(model), os.fsencode(out))
    meta = ptq.load_json(out / "SnrShapeInfo.json")
    radix = ptq.load_json(out / "radix.json")
    info = meta["dp_info"]["burden"]
    ptq.require(
        meta["dp_out"] == ["burden"]
        and info["onnx_shape"] == [1]
        and info["hw_shape"] == [1, 1, 1, 1],
        "Expected one burden scalar",
    )
    (node,) = info["node_name"]
    descriptor = radix[node]
    ((scale,),) = descriptor["output_scale"]
    ((bits,),) = descriptor["output_datapath_radix"]
    (width,) = descriptor["output_datapath_bitwidth"]
    import math

    ptq.require(
        width in (8, 16) and type(bits) is int and math.isfinite(scale) and scale > 0,
        "Unrecognized effective output descriptor",
    )
    return {
        "node": node,
        "width": width,
        "radix": bits,
        "scale": scale,
        "libs_sha256": ptq.sha256(libs),
        "metadata": {p.name: ptq.sha256(p) for p in out.iterdir() if p.is_file()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "onnx",
        "screen",
        "optimized",
        "export",
        "calibration-root",
        "manifest",
        "parity-receipt",
        "out",
        "policy",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--architecture", choices=["resnet18", "mobilenet_v2"], required=True)
    parser.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    quantize(parser.parse_args())


if __name__ == "__main__":
    main()
