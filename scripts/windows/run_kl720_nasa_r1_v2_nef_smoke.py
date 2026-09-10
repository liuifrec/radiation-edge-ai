"""Run the frozen NASA BPS R1 v2 53BP1 regression NEF on a physical KL720.

This is the first one-sample physical-device smoke test for the accepted NASA
R1 v2 deployment artifact. It must be run with the known-good Kneron PLUS
Python runtime on Windows, not inside the Kneron toolchain container.

The script verifies the exact frozen NEF, NEF deployment freeze, and pre-hardware
BIE smoke-reference summary before touching the device. The input is the already
materialized float32 NCHW development tensor BPSR1V2D_00324. For KL720 Generic
Data inference, host-side fixed-point quantization and NPU re-layout are applied
using the quantization/layout metadata embedded in the loaded NEF descriptor.

No final holdout image or phenotype is read. No model, PTQ, BIE, or NEF artifact
is changed. The physical-vs-BIE max-absolute-error gate (1e-4) was frozen in the
pre-hardware reference stage before any physical output was observed.

Python 3.9 compatible for the user's Kneron PLUS 3.2.0 runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

EXPECTED_NEF_SHA256 = "d04f02855a8a82ae7cb5ff48af91eee69c341142d1097469890385bd23ee097b"
EXPECTED_NEF_FREEZE_SHA256 = "f818b9d912017a1d8fdc4aafd4bf321b8bf5b84e8214e6ef0ba029b48b0b90bb"
EXPECTED_SMOKE_REFERENCE_SUMMARY_SHA256 = "18d261568361e22f7acb704b9ab15022fe4feca27d9a76776631a8f5e202286d"
EXPECTED_SMOKE_INPUT_SHA256 = "6adba2ab37a0cb2224604f120d2f855688768858d581521d90fa70656a75e1c4"
EXPECTED_BIE_REFERENCE_SHA256 = "2b253970d07ea67761631670cc4ae4a1096cdb24bed2d2353a2d0ef41f95b66a"
EXPECTED_SAMPLE_ID = "BPSR1V2D_00324"
EXPECTED_PLATFORM = 720
EXPECTED_MODEL_ID = 32770
EXPECTED_MODEL_VERSION = "8b29"
EXPECTED_INPUT_SHAPE = (1, 3, 256, 256)
EXPECTED_BIE_REFERENCE_SCALAR = 2.498514175415039
PHYSICAL_BIE_MAX_ABS_GATE = 1.0e-4


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


def check_sha(path, expected, label):
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            "{} SHA256 mismatch: expected {}, got {}".format(label, expected, actual)
        )
    return actual


def extract_array(node_output):
    """Extract ndarray across PLUS Python return-shape variants."""
    if isinstance(node_output, np.ndarray):
        return node_output
    for attr in ("ndarray", "array", "data"):
        if hasattr(node_output, attr):
            value = getattr(node_output, attr)
            if isinstance(value, np.ndarray):
                return value
    public = [name for name in dir(node_output) if not name.startswith("_")]
    raise RuntimeError(
        "Could not extract NumPy array from float-node result; type={!r}, public attributes={}".format(
            type(node_output), public
        )
    )


def descriptor_shape(tensor_descriptor):
    """Return logical BCHW shape across KL720(v1) and newer(v2) descriptors."""
    info = tensor_descriptor.tensor_shape_info
    if hasattr(info, "v1") and hasattr(info.v1, "shape_npu"):
        return tuple(int(v) for v in info.v1.shape_npu)
    if hasattr(info, "v2") and hasattr(info.v2, "shape"):
        return tuple(int(v) for v in info.v2.shape)
    public = [name for name in dir(info) if not name.startswith("_")]
    raise RuntimeError(
        "NEF tensor descriptor exposes neither v1.shape_npu nor v2.shape; attributes={}".format(
            public
        )
    )


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
        raise RuntimeError(
            "Expected one KL720 input fixed-point descriptor, got {}".format(
                len(descriptors)
            )
        )
    item = descriptors[0]
    radix = int(item.radix)
    scale_obj = item.scale
    scale = float(scale_obj.value if hasattr(scale_obj, "value") else scale_obj)
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("Invalid KL720 input scale: {}".format(scale))
    return radix, scale


def pack_kl720_input(kp, input_node, onnx_data):
    """Quantize BCHW ONNX-space float input and pack for KL720 Generic Data."""
    shape = descriptor_shape(input_node)
    if shape != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(
            "KL720 NEF input shape {}; expected {}".format(shape, EXPECTED_INPUT_SHAPE)
        )
    if onnx_data.shape != shape:
        raise RuntimeError(
            "Input shape {}; NEF expects {}".format(onnx_data.shape, shape)
        )

    batch, channels, height, width = shape
    if batch != 1:
        raise RuntimeError("Only batch=1 supported, got {}".format(batch))

    radix, scale = kl720_quantization(input_node)
    factor = float(np.power(2.0, radix) * scale)
    hwc = np.asarray(onnx_data[0].transpose(1, 2, 0), dtype=np.float32)
    quantized = np.rint(hwc * factor)
    quantized = np.clip(quantized, -128, 127).astype(np.int8)

    layout = input_node.data_layout
    if layout == kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B:
        width_align_base = 4
        channel_align_base = 4
        layout_name = "4W4C8B"
    elif layout == kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8B:
        width_align_base = 1
        channel_align_base = 16
        layout_name = "1W16C8B"
    elif layout == kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8B:
        width_align_base = 16
        channel_align_base = 1
        layout_name = "16W1C8B"
    else:
        raise RuntimeError(
            "Unsupported KL720 NPU input layout for validated Generic Data path: {}".format(
                layout
            )
        )

    width_aligned = width_align_base * int(math.ceil(width / float(width_align_base)))
    channel_blocks = int(math.ceil(channels / float(channel_align_base)))
    relayout = np.zeros(
        (channel_blocks, height, width_aligned, channel_align_base), dtype=np.int8
    )

    channel_offset = 0
    for block in range(channel_blocks):
        channel_end = min(channel_offset + channel_align_base, channels)
        count = channel_end - channel_offset
        block_values = quantized[:, :, channel_offset:channel_end]
        if block_values.shape != (height, width, count):
            raise RuntimeError(
                "Unexpected quantized block shape {}; expected {}".format(
                    block_values.shape, (height, width, count)
                )
            )
        relayout[block, :height, :width, :count] = block_values
        channel_offset = channel_end

    metadata = {
        "radix": radix,
        "scale": scale,
        "quantization_factor": factor,
        "layout": layout_name,
        "width_align_base": width_align_base,
        "channel_align_base": channel_align_base,
        "width_aligned": width_aligned,
        "channel_blocks": channel_blocks,
        "quantized_min": int(quantized.min()),
        "quantized_max": int(quantized.max()),
        "buffer_bytes": int(relayout.nbytes),
    }
    return relayout.tobytes(), metadata


def normalize_scalar(array, label):
    arr = np.asarray(array, dtype=np.float32)
    if arr.size != 1:
        raise RuntimeError("{} expected one scalar element, got shape {}".format(label, arr.shape))
    if not np.isfinite(arr).all():
        raise RuntimeError("{} contains non-finite value".format(label))
    return np.ascontiguousarray(arr.reshape(1), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nef", required=True)
    parser.add_argument("--nef-freeze", required=True)
    parser.add_argument("--smoke-reference-summary", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--bie-reference", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", type=int, default=81)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    args = parser.parse_args()

    nef_path = Path(args.nef).resolve()
    nef_freeze_path = Path(args.nef_freeze).resolve()
    smoke_summary_path = Path(args.smoke_reference_summary).resolve()
    input_path = Path(args.input).resolve()
    bie_reference_path = Path(args.bie_reference).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_path = output_dir / "physical_output.npy"
    summary_path = output_dir / "physical_smoke_summary.json"

    if summary_path.exists():
        raise RuntimeError(
            "Physical smoke result already exists; refusing overwrite: {}".format(summary_path)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    nef_sha = check_sha(nef_path, EXPECTED_NEF_SHA256, "Frozen NEF")
    nef_freeze_sha = check_sha(
        nef_freeze_path, EXPECTED_NEF_FREEZE_SHA256, "NEF deployment freeze"
    )
    smoke_summary_sha = check_sha(
        smoke_summary_path,
        EXPECTED_SMOKE_REFERENCE_SUMMARY_SHA256,
        "Pre-hardware smoke reference summary",
    )
    input_sha = check_sha(input_path, EXPECTED_SMOKE_INPUT_SHA256, "Smoke input")
    bie_ref_sha = check_sha(
        bie_reference_path, EXPECTED_BIE_REFERENCE_SHA256, "Smoke BIE reference"
    )

    nef_freeze = read_json(nef_freeze_path)
    if nef_freeze.get("nef_sha256") != EXPECTED_NEF_SHA256:
        raise RuntimeError("NEF deployment freeze does not pin expected NEF")
    if int(nef_freeze.get("model_id", -1)) != EXPECTED_MODEL_ID:
        raise RuntimeError("NEF deployment freeze model_id mismatch")
    if str(nef_freeze.get("model_version", "")) != EXPECTED_MODEL_VERSION:
        raise RuntimeError("NEF deployment freeze model_version mismatch")
    if nef_freeze.get("compile_integrity_pass") is not True:
        raise RuntimeError("NEF deployment freeze compile-integrity is not PASS")
    if nef_freeze.get("post_holdout_ptq_retuning_authorized") is not False:
        raise RuntimeError("Unexpected post-holdout PTQ retuning authorization")

    smoke = read_json(smoke_summary_path)
    smoke_checks = {
        "status": "FROZEN_PRE_HARDWARE_ONE_SAMPLE_BIE_REFERENCE",
        "nef_deployment_freeze_sha256": EXPECTED_NEF_FREEZE_SHA256,
        "nef_sha256": EXPECTED_NEF_SHA256,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "selected_sample_id": EXPECTED_SAMPLE_ID,
        "smoke_input_sha256": EXPECTED_SMOKE_INPUT_SHA256,
        "bie_reference_sha256": EXPECTED_BIE_REFERENCE_SHA256,
        "physical_bie_max_abs_gate": PHYSICAL_BIE_MAX_ABS_GATE,
        "gate_frozen_before_hardware_result": True,
        "final_holdout_read": False,
        "ptq_bie_or_nef_changed": False,
        "authorized_next_stage": "PHYSICAL_KL720_ONE_SAMPLE_SMOKE_WITH_THIS_EXACT_REFERENCE",
    }
    for key, expected in smoke_checks.items():
        if smoke.get(key) != expected:
            raise RuntimeError(
                "Smoke reference invariant mismatch for {}: expected {!r}, got {!r}".format(
                    key, expected, smoke.get(key)
                )
            )

    array = np.load(str(input_path), allow_pickle=False)
    if array.shape != EXPECTED_INPUT_SHAPE or array.dtype != np.float32:
        raise RuntimeError(
            "Unexpected smoke input {} {}; expected {} float32".format(
                array.shape, array.dtype, EXPECTED_INPUT_SHAPE
            )
        )
    if not np.isfinite(array).all():
        raise RuntimeError("Smoke input contains non-finite values")
    array = np.ascontiguousarray(array, dtype=np.float32)

    bie_reference = np.load(str(bie_reference_path), allow_pickle=False)
    bie_reference = normalize_scalar(bie_reference, "BIE reference")
    if abs(float(bie_reference[0]) - EXPECTED_BIE_REFERENCE_SCALAR) > 1e-7:
        raise RuntimeError(
            "Unexpected BIE reference scalar: expected {:.10g}, got {:.10g}".format(
                EXPECTED_BIE_REFERENCE_SCALAR, float(bie_reference[0])
            )
        )

    import kp

    print("Radiation Edge AI - NASA BPS R1 v2 physical KL720 one-sample smoke")
    print("kp module: {}".format(Path(kp.__file__).resolve()))
    print("NEF SHA256: {}".format(nef_sha))
    print("NEF deployment freeze SHA256: {}".format(nef_freeze_sha))
    print("pre-hardware smoke summary SHA256: {}".format(smoke_summary_sha))
    print("sample: {}".format(EXPECTED_SAMPLE_ID))
    print("input SHA256: {}".format(input_sha))
    print("BIE reference SHA256: {}".format(bie_ref_sha))
    print("BIE reference scalar: {:.10g}".format(float(bie_reference[0])))
    print("physical-vs-BIE max-abs gate frozen before hardware: <= {:.1e}".format(PHYSICAL_BIE_MAX_ABS_GATE))
    print("final holdout read by this stage: NO")
    print("PTQ/BIE/NEF changed by this stage: NO")
    print("")

    descriptors = scan_descriptors(kp)
    matching = [
        d for d in descriptors if int(getattr(d, "usb_port_id", -1)) == args.port
    ]
    if not matching:
        visible = [getattr(d, "usb_port_id", None) for d in descriptors]
        raise RuntimeError(
            "KL720 USB port {} not found; visible ports={}".format(args.port, visible)
        )
    device_desc = matching[0]
    product_id = int(getattr(device_desc, "product_id", -1))
    if product_id != 0x720:
        raise RuntimeError(
            "USB port {} is not KL720: product_id={:#x}".format(args.port, product_id)
        )
    print("[Device] port={} KL720 detected (product_id=0x720)".format(args.port))
    firmware = getattr(device_desc, "firmware", None)
    if firmware is not None:
        print("  firmware: {}".format(firmware))

    try:
        device_group = kp.core.connect_devices(usb_port_ids=[args.port])
    except TypeError:
        device_group = kp.core.connect_devices([args.port])
    print("[Connect] PASS")

    try:
        kp.core.set_timeout(device_group=device_group, milliseconds=args.timeout_ms)
    except TypeError:
        kp.core.set_timeout(device_group, args.timeout_ms)
    print("[Timeout] {} ms".format(args.timeout_ms))

    t0 = time.perf_counter()
    model_nef_descriptor = kp.core.load_model_from_file(
        device_group=device_group, file_path=str(nef_path)
    )
    model_load_ms = (time.perf_counter() - t0) * 1000.0
    print("[Load NEF] PASS ({:.1f} ms)".format(model_load_ms))

    if len(model_nef_descriptor.models) != 1:
        raise RuntimeError(
            "Expected one model in NEF, got {}".format(len(model_nef_descriptor.models))
        )
    model = model_nef_descriptor.models[0]
    if int(model.id) != EXPECTED_MODEL_ID:
        raise RuntimeError(
            "Physical NEF descriptor model_id {}; expected {}".format(
                model.id, EXPECTED_MODEL_ID
            )
        )
    if len(model.input_nodes) != 1 or len(model.output_nodes) != 1:
        raise RuntimeError(
            "Expected one input and one output; got {}/{}".format(
                len(model.input_nodes), len(model.output_nodes)
            )
        )

    input_node = model.input_nodes[0]
    output_node = model.output_nodes[0]
    input_shape = descriptor_shape(input_node)
    output_shape = descriptor_shape(output_node)
    if input_shape != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(
            "Physical NEF input shape {}; expected {}".format(
                input_shape, EXPECTED_INPUT_SHAPE
            )
        )
    if int(np.prod(np.asarray(output_shape, dtype=np.int64))) != 1:
        raise RuntimeError(
            "Physical NEF output must contain one scalar element, got {}".format(
                output_shape
            )
        )

    version_value = getattr(model, "version", None)
    print("[NEF descriptor] model_id={}".format(model.id))
    if version_value is not None:
        print("  descriptor version: {}".format(version_value))
    print("  input shape: {}".format(input_shape))
    print("  input layout: {}".format(getattr(input_node, "data_layout", "<unknown>")))
    print("  output shape: {}".format(output_shape))

    t0 = time.perf_counter()
    npu_input_buffer, pack_meta = pack_kl720_input(kp, input_node, array)
    host_pack_ms = (time.perf_counter() - t0) * 1000.0
    print(
        "[Host KL720 quantize+relayout] PASS ({:.1f} ms, {} bytes)".format(
            host_pack_ms, len(npu_input_buffer)
        )
    )
    print(
        "  radix={} scale={:.9g} factor={:.9g} layout={} qrange=[{},{}]".format(
            pack_meta["radix"],
            pack_meta["scale"],
            pack_meta["quantization_factor"],
            pack_meta["layout"],
            pack_meta["quantized_min"],
            pack_meta["quantized_max"],
        )
    )

    generic_descriptor = kp.GenericDataInferenceDescriptor(
        model_id=model.id,
        inference_number=0,
        input_node_data_list=[kp.GenericInputNodeData(buffer=npu_input_buffer)],
    )

    t0 = time.perf_counter()
    kp.inference.generic_data_inference_send(
        device_group=device_group,
        generic_inference_input_descriptor=generic_descriptor,
    )
    raw_result = kp.inference.generic_data_inference_receive(device_group=device_group)
    hardware_ms = (time.perf_counter() - t0) * 1000.0
    print("[Physical inference] PASS ({:.1f} ms send+receive)".format(hardware_ms))

    n_outputs = int(raw_result.header.num_output_node)
    if n_outputs != 1:
        raise RuntimeError("Expected one output node, got {}".format(n_outputs))

    float_output_obj = kp.inference.generic_inference_retrieve_float_node(
        node_idx=0,
        generic_raw_result=raw_result,
        channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW,
    )
    physical = normalize_scalar(extract_array(float_output_obj), "Physical KL720 output")
    np.save(str(output_path), physical, allow_pickle=False)

    physical_scalar = float(physical[0])
    bie_scalar = float(bie_reference[0])
    abs_error = abs(physical_scalar - bie_scalar)
    passed = bool(abs_error <= PHYSICAL_BIE_MAX_ABS_GATE)

    print("[Retrieve float output] PASS")
    print("  physical scalar: {:.10g}".format(physical_scalar))
    print("")
    print("[Physical KL720 vs frozen BIE reference]")
    print("BIE scalar: {:.10g}".format(bie_scalar))
    print("physical scalar: {:.10g}".format(physical_scalar))
    print("absolute error: {:.10g}".format(abs_error))
    print("frozen max-abs gate: <= {:.1e}".format(PHYSICAL_BIE_MAX_ABS_GATE))
    print("PHYSICAL/BIE NUMERICAL SMOKE PASS: {}".format("YES" if passed else "NO"))

    summary = {
        "status": "COMPLETE_PHYSICAL_KL720_ONE_SAMPLE_NASA_R1_V2_SMOKE",
        "nef_sha256": nef_sha,
        "nef_deployment_freeze_sha256": nef_freeze_sha,
        "pre_hardware_smoke_reference_summary_sha256": smoke_summary_sha,
        "sample_id": EXPECTED_SAMPLE_ID,
        "input_sha256": input_sha,
        "bie_reference_sha256": bie_ref_sha,
        "platform": EXPECTED_PLATFORM,
        "model_id": int(model.id),
        "model_version_frozen": EXPECTED_MODEL_VERSION,
        "usb_port": int(args.port),
        "timeout_ms": int(args.timeout_ms),
        "nef_input_shape": list(input_shape),
        "nef_output_shape": list(output_shape),
        "model_load_ms": model_load_ms,
        "host_pack_ms": host_pack_ms,
        "hardware_send_receive_ms": hardware_ms,
        "kl720_pack": pack_meta,
        "bie_reference_scalar": bie_scalar,
        "physical_scalar": physical_scalar,
        "absolute_error": abs_error,
        "physical_bie_max_abs_gate": PHYSICAL_BIE_MAX_ABS_GATE,
        "gate_frozen_before_hardware_result": True,
        "physical_bie_numerical_smoke_pass": passed,
        "physical_output": str(output_path),
        "physical_output_sha256": sha256_file(output_path),
        "final_holdout_read": False,
        "ptq_bie_or_nef_changed": False,
        "post_holdout_ptq_retuning_authorized": False,
        "authorized_next_stage": (
            "FREEZE_PHYSICAL_SMOKE_THEN_RUN_FULL_HOLDOUT_HARDWARE_EQUIVALENCE"
            if passed
            else "BLOCKED_PHYSICAL_SMOKE_FAILURE_NO_RETUNING"
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("physical output: {}".format(output_path))
    print("physical output SHA256: {}".format(sha256_file(output_path)))
    print("summary: {}".format(summary_path))
    print("summary SHA256: {}".format(sha256_file(summary_path)))
    print("final holdout read: NO")
    print("PTQ/BIE/NEF changed: NO")
    if passed:
        print("NEXT GATE: freeze this physical smoke PASS, then run full frozen holdout on KL720.")
        return 0
    print("NEXT GATE: STOP; diagnose host/device runtime path without PTQ or model retuning.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
