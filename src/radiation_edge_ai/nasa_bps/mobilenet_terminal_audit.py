"""Replay the MobileNet training-only scalar grid audit from frozen vendor descriptors."""
import argparse
import bisect
import csv
import math
import zipfile
from pathlib import Path
from statistics import mean

import numpy as np

from . import multimodel_ptq as ptq


PILOT_SHA = "3fbe0da7cfbfeb5bccfa66572364fc71d47d6903af677b24555e19b1b8491a33"
SELECTION_SHA = "6705097bfb99f6038dd82aaf6acbe46c3c0ccd9ce20cc902d524058ac3ca6f2f"
VENDOR_SHA = "89458722580c4a1e829c597331e87dda745a8d9dbdde994f635fca3673fd7bc9"
DESCRIPTORS = {
    "SnrShapeInfo.json": "7491051f8633ce92ec0e4a7c62efa4610d2e0c86e84cb5e6eda588b9d2240c44",
    "radix.json": "15d42a2a05c3e5ebcbc7c60c44444e3a949c14697d0737feb61ace8db4554dea",
}


def verified(path, expected):
    ptq.require(ptq.sha256(path) == expected, f"Changed {path.name}")
    return path


def signed_grid(scale, radix):
    ptq.require(math.isfinite(scale) and scale > 0 and type(radix) is int, "Invalid scalar descriptor")
    # The native FixUtils.h path uses float32 division by 2**radix, then by scale.
    return [float(v) for v in np.arange(-128, 128, dtype=np.float32) / np.float32(2**radix) / np.float32(scale)]


def compare_scalar(grid, original, code, native_text, returned):
    ptq.require(math.isfinite(original) and -128 <= code <= 127, "Invalid training scalar")
    ptq.require(float(np.float32(native_text)) == returned, "Native dump differs from API return")
    decoded = grid[code + 128]
    matches_text = format(decoded, ".8f") == native_text
    ptq.require(matches_text, "Decoded code differs from the archived eight-decimal text")
    low = max(0, bisect.bisect_right(grid, original) - 1)
    high = min(255, bisect.bisect_left(grid, original))
    return dict(original=original, returned=returned, code=code, decoded=decoded, native_float_text=native_text,
                exact_observed_decimal_text=matches_text, text_roundtrip_float32=float(np.float32(format(decoded, ".8f"))),
                lo_code=low - 128, hi_code=high - 128, lower=grid[low], upper=grid[high],
                outside_code_interval=not low <= code + 128 <= high,
                minimum_distance_outside_adjacent_grid=max(grid[low] - returned, returned - grid[high], 0))


def audit(run, descriptors, out):
    pilot = run / "training_pilot"
    receipt = ptq.load_json(verified(pilot / "training_pilot.json", PILOT_SHA))
    selection = ptq.load_json(verified(run / "training_pilot_selection.json", SELECTION_SHA))
    verified(run / "pilot/quantized_model.bie", receipt["bie_sha256"])
    report = ptq.load_json(verified(run / "kneron_flow/model_fx_report.json", VENDOR_SHA))
    info, radix = (ptq.load_json(verified(descriptors / name, sha)) for name, sha in DESCRIPTORS.items())
    ptq.require(info["dp_out"] == ["burden"] and info["dp_info"]["burden"]["onnx_shape"] == [1],
                "Expected one burden scalar")
    node, = info["dp_info"]["burden"]["node_name"]
    output = radix[node]
    ptq.require(report["kdp720/output bitwidth"] == "int8" and output["output_datapath_bitwidth"] == [8],
                "Expected signed INT8 output")
    (bits,), = output["output_datapath_radix"]
    (scale,), = output["output_scale"]
    grid = signed_grid(scale, bits)
    predictions = ptq.checked_rows(verified(pilot / "training_predictions.csv", receipt["predictions_sha256"]))
    ptq.require(len(predictions) == 16 and [p["sample_id"] for p in predictions] == selection["sample_ids"],
                "Expected the prespecified training panel")
    rows = []
    with zipfile.ZipFile(verified(pilot / "native_logs.zip", receipt["native_logs_sha256"])) as archive:
        root, = {p.split("/")[0] for p in archive.namelist() if p.startswith("res_kdp720_")}
        for i, pred in enumerate(predictions):
            prefix = f"{root}/in_{i:06}.txt/mode_720_piano/layer_output_burden_"
            code = int(archive.read(prefix + "fx.txt"))
            ptq.require(code == int(pred["bie_fixed_code"]), "Changed signed training code")
            scalar = compare_scalar(grid, float(pred["source_fp32"]), code,
                                    archive.read(prefix + "fl.txt").decode("ascii").strip(), float(pred["bie_burden"]))
            rows.append(dict(sample_id=pred["sample_id"], source_name=pred["source_name"], **scalar))
    out.mkdir(parents=True, exist_ok=False)
    pairs = out / "training_terminal_pairs.csv"
    with pairs.open("x", encoding="utf-8", newline="") as stream:
        table = csv.DictWriter(stream, fieldnames=list(rows[0]))
        table.writeheader()
        table.writerows(rows)
    changes = [r["returned"] - r["original"] for r in rows]
    possible_errors = [max(abs(r[k] - r["original"]) for k in ("lower", "upper")) for r in rows]
    ptq.save_json(out / "receipt.json", dict(
        stage="training_only_terminal_grid_audit", specimens=len(rows), pilot_sha256=PILOT_SHA,
        selection_sha256=SELECTION_SHA, descriptor_sha256=DESCRIPTORS, vendor_report_sha256=VENDOR_SHA,
        bie_sha256=receipt["bie_sha256"], source_sha256=ptq.sha256(__file__), rows_sha256=ptq.sha256(pairs),
        output_name="burden", terminal_node=node, radix=bits, output_scale=scale, step=1/(scale * 2**bits),
        decoded_range=[grid[0], grid[-1]], original_outside_range=sum(not grid[0] <= r["original"] <= grid[-1] for r in rows),
        returned_at_extreme_code=sum(r["code"] in (-128, 127) for r in rows),
        matching_native_dumps=len(rows), exact_fp32_decodes=sum(r["decoded"] == r["returned"] for r in rows),
        exact_after_observed_text_and_float32_parse=sum(r["text_roundtrip_float32"] == r["returned"] for r in rows),
        outside_code_interval=sum(r["outside_code_interval"] for r in rows), actual_bias=mean(changes),
        actual_mae=mean(map(abs, changes)), terminal_only_mae_ceiling=mean(possible_errors),
        terminal_only_max_abs_ceiling=max(possible_errors),
        terminal_only_bias_interval=[mean(r[k] - r["original"] for r in rows) for k in ("lower", "upper")],
        mean_minimum_distance_outside_adjacent_grid=mean(r["minimum_distance_outside_adjacent_grid"] for r in rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("ptq-run", "descriptors", "out"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    audit(args.ptq_run, args.descriptors, args.out)
