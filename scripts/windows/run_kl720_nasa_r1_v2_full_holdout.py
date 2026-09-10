"""Run the frozen NASA BPS R1 v2 holdout through the physical KL720.

This Windows/PLUS-runtime stage is deliberately outcome-blind. It verifies the
exact accepted NEF, NEF deployment freeze, successful one-sample physical smoke
freeze, and QC1 holdout manifest; reconstructs the frozen FITC+DAPI input tensor
for each of 2,058 nuclei; runs the physical KL720; and writes only physical
per-nucleus latent burden predictions plus timing/provenance.

It does NOT read the raw NASA phenotype table, the FP32 confirmation outputs, or
the BIE holdout outputs. Biological targets and the seven frozen gates are only
introduced after this complete physical prediction set is frozen.

Important Windows path convention: --image-root must point to the materialized
"images" directory that directly contains fitc/ and dapi/ subdirectories, e.g.
...\\r1_v2_final_holdout_blinded\\images.

Python 3.9 compatible for Kneron PLUS 3.2.0 on the user's Windows host.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

EXPECTED_NEF_SHA256 = "d04f02855a8a82ae7cb5ff48af91eee69c341142d1097469890385bd23ee097b"
EXPECTED_NEF_FREEZE_SHA256 = "f818b9d912017a1d8fdc4aafd4bf321b8bf5b84e8214e6ef0ba029b48b0b90bb"
EXPECTED_PHYSICAL_SMOKE_FREEZE_SHA256 = "224b754235ebcd22eb053ca1f279043f3b74e9e3b461ed1f8dad26beddfa679d"
EXPECTED_QC1_MANIFEST_SHA256 = "f89962f96bec05930c07149717a90bd682016a2778bbfae6b827425159aa0643"
EXPECTED_PLATFORM = 720
EXPECTED_MODEL_ID = 32770
EXPECTED_MODEL_VERSION = "8b29"
EXPECTED_USB_PORT = 81
EXPECTED_NUCLEI = 2058
EXPECTED_BAGS = 22
EXPECTED_SOURCE_NUCLEI = {"BALBCF2": 858, "C57BLF2": 600, "C57BLF3": 600}
EXPECTED_HOLDOUT_STATUS = "FINAL_BLINDED_DO_NOT_JOIN_PHENOTYPES"
EXPECTED_INPUT_SHAPE = (1, 3, 256, 256)
CANVAS = 256
P_LOW = 1.0
P_HIGH = 99.5
FORBIDDEN_MANIFEST_TOKENS = (
    "nfoci",
    "phenotype",
    "reference",
    "ground_truth",
    "prediction",
    "predicted",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError("Expected JSON object: {}".format(path))
    return obj


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError("CSV is empty: {}".format(path))
    return rows


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("Refusing to write empty CSV")
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def check_sha(path, expected, label):
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            "{} SHA256 mismatch: expected {}, got {}".format(label, expected, actual)
        )
    return actual


def norm_number(value):
    x = float(str(value).strip())
    if x.is_integer():
        return "{:.1f}".format(x)
    return format(x, "g")


def norm_hour(value):
    x = float(str(value).strip())
    return str(int(x)) if x.is_integer() else format(x, "g")


def validate_holdout_manifest(rows):
    required = {
        "sample_id", "nucleus_key", "source_name", "sample_name", "strain",
        "sex", "particle_type", "dose_Gy", "hr_post_exposure",
        "fitc_filename", "dapi_filename", "mask_filename", "holdout_status",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise RuntimeError("QC1 manifest missing columns: {}".format(missing))
    forbidden = sorted(
        column for column in rows[0]
        if any(token in column.lower() for token in FORBIDDEN_MANIFEST_TOKENS)
    )
    if forbidden:
        raise RuntimeError("QC1 manifest contains outcome/reference columns: {}".format(forbidden))
    if len(rows) != EXPECTED_NUCLEI:
        raise RuntimeError("Holdout nuclei={}; expected {}".format(len(rows), EXPECTED_NUCLEI))
    if len({row["sample_id"] for row in rows}) != EXPECTED_NUCLEI:
        raise RuntimeError("Duplicate sample_id in QC1 holdout manifest")
    if len({row["nucleus_key"] for row in rows}) != EXPECTED_NUCLEI:
        raise RuntimeError("Duplicate nucleus_key in QC1 holdout manifest")
    if {row["holdout_status"] for row in rows} != {EXPECTED_HOLDOUT_STATUS}:
        raise RuntimeError("Unexpected holdout_status")
    source_counts = Counter(row["source_name"] for row in rows)
    if dict(source_counts) != EXPECTED_SOURCE_NUCLEI:
        raise RuntimeError("Unexpected source counts: {}".format(dict(source_counts)))
    if {row["sex"] for row in rows} != {"Female"}:
        raise RuntimeError("Frozen all-female holdout structure changed")
    bags = defaultdict(int)
    for row in rows:
        bags[row["sample_name"]] += 1
    if len(bags) != EXPECTED_BAGS:
        raise RuntimeError("Holdout bags={}; expected {}".format(len(bags), EXPECTED_BAGS))
    return source_counts, bags


def robust_normalize(image):
    if image.ndim != 2:
        raise RuntimeError("Expected 2D microscopy crop, got {}".format(image.shape))
    x = image.astype(np.float32, copy=False)
    lo = float(np.percentile(x, P_LOW))
    hi = float(np.percentile(x, P_HIGH))
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise RuntimeError("Non-finite normalization percentile")
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    np.clip(x, 0.0, 1.0, out=x)
    return x


def load_tensor(row, image_root):
    fitc_path = image_root / "fitc" / row["fitc_filename"]
    dapi_path = image_root / "dapi" / row["dapi_filename"]
    for path in (fitc_path, dapi_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    fitc = cv2.imread(str(fitc_path), cv2.IMREAD_UNCHANGED)
    dapi = cv2.imread(str(dapi_path), cv2.IMREAD_UNCHANGED)
    if fitc is None or dapi is None:
        raise RuntimeError("Could not read FITC/DAPI for {}".format(row["sample_id"]))
    if fitc.ndim != 2 or dapi.ndim != 2 or fitc.shape != dapi.shape:
        raise RuntimeError(
            "Invalid FITC/DAPI geometry for {}: {} / {}".format(
                row["sample_id"], getattr(fitc, "shape", None), getattr(dapi, "shape", None)
            )
        )
    h, w = fitc.shape
    if h > CANVAS or w > CANVAS:
        raise RuntimeError(
            "Native crop exceeds frozen 256x256 canvas for {}: {}x{}; resize forbidden".format(
                row["sample_id"], h, w
            )
        )
    out = np.zeros(EXPECTED_INPUT_SHAPE, dtype=np.float32)
    y0 = (CANVAS - h) // 2
    x0 = (CANVAS - w) // 2
    out[0, 0, y0:y0+h, x0:x0+w] = robust_normalize(fitc)
    out[0, 1, y0:y0+h, x0:x0+w] = robust_normalize(dapi)
    if not np.all(out[:, 2, :, :] == 0.0):
        raise RuntimeError("Third channel is not identically zero")
    return np.ascontiguousarray(out, dtype=np.float32)


def extract_array(node_output):
    if isinstance(node_output, np.ndarray):
        return node_output
    for attr in ("ndarray", "array", "data"):
        if hasattr(node_output, attr):
            value = getattr(node_output, attr)
            if isinstance(value, np.ndarray):
                return value
    public = [name for name in dir(node_output) if not name.startswith("_")]
    raise RuntimeError(
        "Could not extract NumPy array from float-node result; type={!r}, attrs={}".format(
            type(node_output), public
        )
    )


def descriptor_shape(tensor_descriptor):
    info = tensor_descriptor.tensor_shape_info
    if hasattr(info, "v1") and hasattr(info.v1, "shape_npu"):
        return tuple(int(v) for v in info.v1.shape_npu)
    if hasattr(info, "v2") and hasattr(info.v2, "shape"):
        return tuple(int(v) for v in info.v2.shape)
    raise RuntimeError("Unsupported NEF tensor descriptor shape API")


def scan_descriptors(kp):
    scan = kp.core.scan_devices()
    descriptors = getattr(scan, "device_descriptor_list", None)
    if descriptors is None:
        try:
            descriptors = list(scan)
        except TypeError:
            descriptors = []
    return list(descriptors)


def kl720_quantization(input_node):
    params = input_node.quantization_parameters
    if not hasattr(params, "v1"):
        raise RuntimeError("KL720 input quantization parameters do not expose v1")
    descriptors = params.v1.quantized_fixed_point_descriptor_list
    if len(descriptors) != 1:
        raise RuntimeError("Expected one KL720 input fixed-point descriptor")
    item = descriptors[0]
    radix = int(item.radix)
    scale_obj = item.scale
    scale = float(scale_obj.value if hasattr(scale_obj, "value") else scale_obj)
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("Invalid KL720 input scale: {}".format(scale))
    return radix, scale


def pack_kl720_input(kp, input_node, onnx_data):
    shape = descriptor_shape(input_node)
    if shape != EXPECTED_INPUT_SHAPE or onnx_data.shape != EXPECTED_INPUT_SHAPE:
        raise RuntimeError("KL720/input shape mismatch: {} / {}".format(shape, onnx_data.shape))
    _, channels, height, width = shape
    radix, scale = kl720_quantization(input_node)
    factor = float(np.power(2.0, radix) * scale)
    hwc = np.asarray(onnx_data[0].transpose(1, 2, 0), dtype=np.float32)
    quantized = np.rint(hwc * factor)
    quantized = np.clip(quantized, -128, 127).astype(np.int8)

    layout = input_node.data_layout
    if layout == kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B:
        width_align_base, channel_align_base, layout_name = 4, 4, "4W4C8B"
    elif layout == kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8B:
        width_align_base, channel_align_base, layout_name = 1, 16, "1W16C8B"
    elif layout == kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8B:
        width_align_base, channel_align_base, layout_name = 16, 1, "16W1C8B"
    else:
        raise RuntimeError("Unsupported KL720 NPU input layout: {}".format(layout))

    width_aligned = width_align_base * int(math.ceil(width / float(width_align_base)))
    channel_blocks = int(math.ceil(channels / float(channel_align_base)))
    relayout = np.zeros(
        (channel_blocks, height, width_aligned, channel_align_base), dtype=np.int8
    )
    channel_offset = 0
    for block in range(channel_blocks):
        channel_end = min(channel_offset + channel_align_base, channels)
        count = channel_end - channel_offset
        values = quantized[:, :, channel_offset:channel_end]
        relayout[block, :height, :width, :count] = values
        channel_offset = channel_end

    meta = {
        "radix": radix,
        "scale": scale,
        "quantization_factor": factor,
        "layout": layout_name,
        "width_align_base": width_align_base,
        "channel_align_base": channel_align_base,
        "width_aligned": width_aligned,
        "channel_blocks": channel_blocks,
        "buffer_bytes": int(relayout.nbytes),
        "quantized_min": int(quantized.min()),
        "quantized_max": int(quantized.max()),
    }
    return relayout.tobytes(), meta


def normalize_scalar(array):
    arr = np.asarray(array, dtype=np.float32)
    if arr.size != 1 or not np.isfinite(arr).all():
        raise RuntimeError("Physical output is not one finite scalar: {}".format(arr.shape))
    return float(arr.reshape(-1)[0])


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nef", required=True)
    parser.add_argument("--nef-freeze", required=True)
    parser.add_argument("--physical-smoke-freeze", required=True)
    parser.add_argument("--holdout-manifest", required=True)
    parser.add_argument("--image-root", required=True, help="Directory containing fitc/ and dapi/ subdirectories (normally ...\\r1_v2_final_holdout_blinded\\images).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", type=int, default=EXPECTED_USB_PORT)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    args = parser.parse_args()

    nef_path = Path(args.nef).resolve()
    nef_freeze_path = Path(args.nef_freeze).resolve()
    smoke_freeze_path = Path(args.physical_smoke_freeze).resolve()
    manifest_path = Path(args.holdout_manifest).resolve()
    image_root = Path(args.image_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    nucleus_path = output_dir / "physical_holdout_nucleus_predictions.csv"
    summary_path = output_dir / "physical_holdout_inference_summary.json"
    tmp_nucleus_path = output_dir / "physical_holdout_nucleus_predictions.csv.tmp"

    if summary_path.exists() or nucleus_path.exists() or tmp_nucleus_path.exists():
        raise RuntimeError("Physical full-holdout output already exists/incomplete; refusing overwrite: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    nef_sha = check_sha(nef_path, EXPECTED_NEF_SHA256, "Frozen NEF")
    nef_freeze_sha = check_sha(nef_freeze_path, EXPECTED_NEF_FREEZE_SHA256, "NEF deployment freeze")
    smoke_freeze_sha = check_sha(smoke_freeze_path, EXPECTED_PHYSICAL_SMOKE_FREEZE_SHA256, "Physical smoke PASS freeze")
    manifest_sha = check_sha(manifest_path, EXPECTED_QC1_MANIFEST_SHA256, "QC1 holdout manifest")
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    nef_freeze = read_json(nef_freeze_path)
    smoke_freeze = read_json(smoke_freeze_path)
    if nef_freeze.get("status") != "FROZEN_KL720_NEF_AFTER_COMPILE_INTEGRITY_PASS_BEFORE_PHYSICAL_HARDWARE":
        raise RuntimeError("Unexpected NEF deployment freeze status")
    if nef_freeze.get("nef_sha256") != EXPECTED_NEF_SHA256:
        raise RuntimeError("NEF deployment freeze does not pin expected NEF")
    if nef_freeze.get("post_holdout_ptq_retuning_authorized") is not False:
        raise RuntimeError("NEF deployment freeze permits PTQ retuning")
    if smoke_freeze.get("status") != "FROZEN_PHYSICAL_KL720_ONE_SAMPLE_SMOKE_PASS_BEFORE_FULL_HOLDOUT":
        raise RuntimeError("Unexpected physical-smoke freeze status")
    if smoke_freeze.get("physical_bie_numerical_smoke_pass") is not True:
        raise RuntimeError("Physical smoke freeze is not PASS")
    if smoke_freeze.get("physical_output_byte_identical_to_bie_reference") is not True:
        raise RuntimeError("Physical smoke output was not byte-identical to frozen BIE reference")
    if smoke_freeze.get("post_holdout_ptq_retuning_authorized") is not False:
        raise RuntimeError("Physical smoke freeze permits PTQ retuning")

    rows = read_csv(manifest_path)
    source_counts, bag_counts = validate_holdout_manifest(rows)
    ordered_rows = sorted(rows, key=lambda row: row["sample_id"])

    print("Radiation Edge AI - NASA BPS R1 v2 full physical KL720 holdout inference")
    print("NEF SHA256: {}".format(nef_sha))
    print("NEF deployment freeze SHA256: {}".format(nef_freeze_sha))
    print("physical smoke PASS freeze SHA256: {}".format(smoke_freeze_sha))
    print("QC1 holdout manifest SHA256: {}".format(manifest_sha))
    print("holdout nuclei/bags: {}/{}".format(len(rows), len(bag_counts)))
    print("preprocessing: p1/p99.5 FITC+DAPI; no resize; center pad 256; zero third channel")
    print("MASK used as model input: NO")
    print("raw phenotype table read: NO")
    print("FP32/BIE holdout reference outputs read: NO")
    print("biological targets/gates evaluated in this stage: NO")
    print("")

    import kp

    descriptors = scan_descriptors(kp)
    descriptor = next((item for item in descriptors if int(getattr(item, "usb_port_id", -1)) == args.port), None)
    if descriptor is None:
        raise RuntimeError("KL720 USB port {} not found; detected={}".format(
            args.port, [int(getattr(d, "usb_port_id", -1)) for d in descriptors]
        ))
    product_id = int(getattr(descriptor, "product_id", -1))
    if product_id != EXPECTED_PLATFORM:
        raise RuntimeError("Port {} product_id={}; expected KL{}".format(args.port, product_id, EXPECTED_PLATFORM))
    print("[Device] port={} KL720 detected".format(args.port))

    try:
        device_group = kp.core.connect_devices(usb_port_ids=[args.port])
    except TypeError:
        device_group = kp.core.connect_devices([args.port])
    print("[Connect] PASS")
    try:
        kp.core.set_timeout(device_group=device_group, milliseconds=args.timeout_ms)
    except TypeError:
        kp.core.set_timeout(device_group, args.timeout_ms)

    t0 = time.perf_counter()
    model_desc = kp.core.load_model_from_file(device_group=device_group, file_path=str(nef_path))
    model_load_ms = (time.perf_counter() - t0) * 1000.0
    if len(model_desc.models) != 1:
        raise RuntimeError("Expected one model in NEF")
    model = model_desc.models[0]
    if int(model.id) != EXPECTED_MODEL_ID:
        raise RuntimeError("Physical NEF model_id {}; expected {}".format(model.id, EXPECTED_MODEL_ID))
    if len(model.input_nodes) != 1 or len(model.output_nodes) != 1:
        raise RuntimeError("Expected one input/output node")
    input_node = model.input_nodes[0]
    output_node = model.output_nodes[0]
    input_shape = descriptor_shape(input_node)
    output_shape = descriptor_shape(output_node)
    if input_shape != EXPECTED_INPUT_SHAPE:
        raise RuntimeError("NEF input shape mismatch: {}".format(input_shape))
    if int(np.prod(np.asarray(output_shape, dtype=np.int64))) != 1:
        raise RuntimeError("NEF output is not scalar: {}".format(output_shape))
    print("[Load NEF] PASS ({:.1f} ms), model_id={}, output_shape={}".format(model_load_ms, model.id, output_shape))

    physical_rows = []
    hardware_times = []
    pack_times = []
    started_all = time.perf_counter()
    static_pack = None

    for index, row in enumerate(ordered_rows, start=1):
        tensor = load_tensor(row, image_root)
        t_pack = time.perf_counter()
        npu_buffer, pack_meta = pack_kl720_input(kp, input_node, tensor)
        pack_ms = (time.perf_counter() - t_pack) * 1000.0
        static_now = {
            key: pack_meta[key]
            for key in (
                "radix", "scale", "quantization_factor", "layout",
                "width_align_base", "channel_align_base", "width_aligned",
                "channel_blocks", "buffer_bytes",
            )
        }
        if static_pack is None:
            static_pack = static_now
        elif static_now != static_pack:
            raise RuntimeError("KL720 input descriptor/packing changed during run")

        descriptor = kp.GenericDataInferenceDescriptor(
            model_id=model.id,
            inference_number=index - 1,
            input_node_data_list=[kp.GenericInputNodeData(buffer=npu_buffer)],
        )
        t_hw = time.perf_counter()
        kp.inference.generic_data_inference_send(
            device_group=device_group,
            generic_inference_input_descriptor=descriptor,
        )
        raw_result = kp.inference.generic_data_inference_receive(device_group=device_group)
        hw_ms = (time.perf_counter() - t_hw) * 1000.0
        if int(raw_result.header.num_output_node) != 1:
            raise RuntimeError("Expected one output node at {}".format(row["sample_id"]))
        float_obj = kp.inference.generic_inference_retrieve_float_node(
            node_idx=0,
            generic_raw_result=raw_result,
            channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW,
        )
        burden = normalize_scalar(extract_array(float_obj))

        physical_rows.append({
            "sample_id": row["sample_id"],
            "nucleus_key": row["nucleus_key"],
            "sample_name": row["sample_name"],
            "source_name": row["source_name"],
            "strain": row["strain"],
            "sex": row["sex"],
            "particle_type": row["particle_type"],
            "dose_Gy": norm_number(row["dose_Gy"]),
            "hr_post_exposure": norm_hour(row["hr_post_exposure"]),
            "physical_kl720_burden": "{:.10g}".format(burden),
            "host_pack_ms": "{:.6f}".format(pack_ms),
            "hardware_send_receive_ms": "{:.6f}".format(hw_ms),
            "quantized_min": pack_meta["quantized_min"],
            "quantized_max": pack_meta["quantized_max"],
            "label_semantics": "latent_continuous_burden_not_individually_supervised",
        })
        pack_times.append(pack_ms)
        hardware_times.append(hw_ms)

        if index % 50 == 0 or index == EXPECTED_NUCLEI:
            print("physical holdout inference: {}/{}".format(index, EXPECTED_NUCLEI))

    if len(physical_rows) != EXPECTED_NUCLEI:
        raise RuntimeError("Incomplete physical holdout inference")
    values = np.asarray([float(row["physical_kl720_burden"]) for row in physical_rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite physical burden values")

    write_csv(tmp_nucleus_path, physical_rows)
    os.replace(str(tmp_nucleus_path), str(nucleus_path))
    elapsed = time.perf_counter() - started_all

    summary = {
        "status": "COMPLETE_OUTCOME_BLIND_FULL_PHYSICAL_KL720_HOLDOUT_INFERENCE",
        "nef_sha256": nef_sha,
        "nef_deployment_freeze_sha256": nef_freeze_sha,
        "physical_smoke_pass_freeze_sha256": smoke_freeze_sha,
        "qc1_holdout_manifest_sha256": manifest_sha,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version_frozen": EXPECTED_MODEL_VERSION,
        "usb_port": int(args.port),
        "timeout_ms": int(args.timeout_ms),
        "nuclei": EXPECTED_NUCLEI,
        "bags": EXPECTED_BAGS,
        "source_nuclei": dict(source_counts),
        "nef_input_shape": list(input_shape),
        "nef_output_shape": list(output_shape),
        "preprocessing": "p1/p99.5 FITC+DAPI native crop normalization; no resize; center pad 256; zero third channel",
        "mask_used_as_model_input": False,
        "raw_phenotype_table_read": False,
        "fp32_or_bie_holdout_reference_outputs_read": False,
        "biological_targets_or_gate_results_read": False,
        "ptq_bie_or_nef_changed": False,
        "post_holdout_ptq_retuning_authorized": False,
        "output_semantics": "scalar latent continuous 53BP1 burden; not per-nucleus focus count",
        "kl720_input_descriptor": static_pack,
        "physical_burden_min": float(values.min()),
        "physical_burden_max": float(values.max()),
        "physical_burden_mean": float(values.mean()),
        "model_load_ms": model_load_ms,
        "mean_host_pack_ms": float(np.mean(pack_times)),
        "mean_hardware_send_receive_ms": float(np.mean(hardware_times)),
        "median_hardware_send_receive_ms": float(np.median(hardware_times)),
        "p95_hardware_send_receive_ms": percentile(hardware_times, 95),
        "elapsed_seconds": elapsed,
        "estimated_device_inferences_per_second_from_mean_send_receive": float(1000.0 / np.mean(hardware_times)),
        "physical_nucleus_predictions": str(nucleus_path),
        "physical_nucleus_predictions_sha256": sha256_file(nucleus_path),
        "physical_nucleus_predictions_size_bytes": nucleus_path.stat().st_size,
        "authorized_next_stage": "FREEZE_OUTCOME_BLIND_PHYSICAL_PREDICTIONS_THEN_EVALUATE_SEVEN_BIOLOGICAL_GATES",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("NASA BPS R1 V2 FULL PHYSICAL KL720 HOLDOUT INFERENCE COMPLETE: YES")
    print("nuclei/bags: {}/{}".format(EXPECTED_NUCLEI, EXPECTED_BAGS))
    print("physical burden range: [{:.8g}, {:.8g}]".format(values.min(), values.max()))
    print("mean hardware send+receive: {:.3f} ms".format(summary["mean_hardware_send_receive_ms"]))
    print("median hardware send+receive: {:.3f} ms".format(summary["median_hardware_send_receive_ms"]))
    print("p95 hardware send+receive: {:.3f} ms".format(summary["p95_hardware_send_receive_ms"]))
    print("prediction CSV: {}".format(nucleus_path))
    print("prediction CSV SHA256: {}".format(summary["physical_nucleus_predictions_sha256"]))
    print("summary: {}".format(summary_path))
    print("summary SHA256: {}".format(sha256_file(summary_path)))
    print("raw phenotype/reference outcomes read: NO")
    print("PTQ/BIE/NEF changed: NO")
    print("NEXT GATE: freeze these outcome-blind physical predictions before reading biological targets/gates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
