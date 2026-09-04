"""Run one frozen DNAi 512x512 validation window on the physical KL720.

This script must be executed with the known-good Kneron PLUS Python runtime,
not the separate DNAi reference environment.

Important KL720 detail:
- KL520/KL630/KL720 NEF tensor descriptors use tensor_shape_info.v1.shape_npu.
- Their Generic Data path requires host-side fixed-point quantization and manual
  NPU re-layout (4W4C8B, 1W16C8B, or 16W1C8B).
- The newer convert_onnx_data_to_npu_data helper is a KL730/v2-descriptor path
  and must not be used for this KL720 model.

Input is the already-normalized float32 ONNX tensor used by the frozen BIE
validation. No DNAi image normalization is repeated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

EXPECTED_NEF_SHA256 = "4b3dfec9a61c99e186dd4b8482fa5b06e6a4958f325ed4a0db0546f1dcab2bfc"
EXPECTED_INPUT_SHAPE = (1, 3, 512, 512)
EXPECTED_OUTPUT_SHAPE = (1, 3, 512, 512)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    ref64 = reference.astype(np.float64, copy=False)
    cand64 = candidate.astype(np.float64, copy=False)
    diff = cand64 - ref64
    abs_diff = np.abs(diff)
    ref_class = reference.argmax(axis=1)
    cand_class = candidate.argmax(axis=1)
    disagreement = int(np.count_nonzero(ref_class != cand_class))
    total = int(ref_class.size)
    return {
        "max_abs_error": float(abs_diff.max()),
        "mean_abs_error": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "argmax_agreement": float(np.mean(ref_class == cand_class)),
        "disagreement_pixels": disagreement,
        "total_pixels": total,
    }


def extract_array(node_output) -> np.ndarray:
    """Extract ndarray across PLUS Python return-shape variants without transposing."""
    if isinstance(node_output, np.ndarray):
        return node_output

    for attr in ("ndarray", "array", "data"):
        if hasattr(node_output, attr):
            value = getattr(node_output, attr)
            if isinstance(value, np.ndarray):
                return value

    public = [name for name in dir(node_output) if not name.startswith("_")]
    raise RuntimeError(
        "Could not extract NumPy array from generic_inference_retrieve_float_node result; "
        f"type={type(node_output)!r}, public attributes={public}"
    )


def descriptor_shape(tensor_descriptor) -> tuple[int, ...]:
    """Return logical BCHW shape across KL720(v1) and newer(v2) descriptors."""
    info = tensor_descriptor.tensor_shape_info
    if hasattr(info, "v1") and hasattr(info.v1, "shape_npu"):
        return tuple(int(v) for v in info.v1.shape_npu)
    if hasattr(info, "v2") and hasattr(info.v2, "shape"):
        return tuple(int(v) for v in info.v2.shape)
    public = [name for name in dir(info) if not name.startswith("_")]
    raise RuntimeError(
        "NEF tensor descriptor exposes neither v1.shape_npu nor v2.shape; "
        f"tensor_shape_info attributes={public}"
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


def kl720_quantization(input_node) -> tuple[int, float]:
    params = input_node.quantization_parameters
    if not hasattr(params, "v1"):
        raise RuntimeError("KL720 input quantization parameters do not expose v1")
    descriptors = params.v1.quantized_fixed_point_descriptor_list
    if len(descriptors) != 1:
        raise RuntimeError(
            "Expected one KL720 input fixed-point descriptor, "
            f"got {len(descriptors)}"
        )
    item = descriptors[0]
    radix = int(item.radix)
    scale_obj = item.scale
    scale = float(scale_obj.value if hasattr(scale_obj, "value") else scale_obj)
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"Invalid KL720 input scale: {scale}")
    return radix, scale


def pack_kl720_input(kp, input_node, onnx_data: np.ndarray) -> tuple[bytes, dict[str, object]]:
    """Quantize normalized BCHW ONNX input and re-layout it for KL720 NPU input."""
    shape = descriptor_shape(input_node)
    if shape != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(f"KL720 NEF input shape {shape}; expected {EXPECTED_INPUT_SHAPE}")
    if onnx_data.shape != shape:
        raise RuntimeError(f"ONNX input shape {onnx_data.shape}; NEF expects {shape}")

    batch, channels, height, width = shape
    if batch != 1:
        raise RuntimeError(f"Only batch=1 is supported by this smoke test, got {batch}")

    radix, scale = kl720_quantization(input_node)
    quantization_factor = float(np.power(2.0, radix) * scale)

    # Frozen validation tensor is BCHW. The documented KL720 re-layout example
    # consumes HWC values before packing channel blocks, so transpose only after
    # preserving the exact already-normalized float values.
    hwc = np.asarray(onnx_data[0].transpose(1, 2, 0), dtype=np.float32)
    quantized = np.rint(hwc * quantization_factor)
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
            "Unsupported KL720 NPU input layout for this validated path: "
            f"{layout}"
        )

    width_aligned = width_align_base * math.ceil(width / float(width_align_base))
    channel_blocks = math.ceil(channels / float(channel_align_base))

    # KL630/KL720 documented dimension order: channel-block x H x aligned-W x
    # channels-within-block.
    relayout = np.zeros(
        (channel_blocks, height, width_aligned, channel_align_base),
        dtype=np.int8,
    )

    channel_offset = 0
    for block in range(channel_blocks):
        channel_end = min(channel_offset + channel_align_base, channels)
        count = channel_end - channel_offset
        relayout[block, :height, :width, :count] = hwc_quant = quantized[
            :, :, channel_offset:channel_end
        ]
        if hwc_quant.shape != (height, width, count):
            raise RuntimeError(
                f"Unexpected quantized block shape {hwc_quant.shape}; "
                f"expected {(height, width, count)}"
            )
        channel_offset = channel_end

    metadata = {
        "radix": radix,
        "scale": scale,
        "quantization_factor": quantization_factor,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nef", required=True)
    parser.add_argument("--input", required=True, help="Frozen normalized float32 NCHW .npy tensor")
    parser.add_argument("--bie-reference", default=None, help="Optional frozen v3 BIE output .npy")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", type=int, default=81)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    args = parser.parse_args()

    nef_path = Path(args.nef).resolve()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bie_reference_path = Path(args.bie_reference).resolve() if args.bie_reference else None

    for path in (nef_path, input_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if bie_reference_path is not None and not bie_reference_path.is_file():
        raise FileNotFoundError(bie_reference_path)

    observed_nef_sha = sha256_file(nef_path)
    if observed_nef_sha.lower() != EXPECTED_NEF_SHA256:
        raise RuntimeError(
            "Refusing to run unexpected NEF. "
            f"observed={observed_nef_sha}, expected={EXPECTED_NEF_SHA256}"
        )

    import kp

    array = np.load(input_path, allow_pickle=False)
    if array.shape != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(f"Input shape {array.shape}; expected {EXPECTED_INPUT_SHAPE}")
    if array.dtype != np.float32:
        raise RuntimeError(f"Input dtype {array.dtype}; expected float32")
    if not np.isfinite(array).all():
        raise RuntimeError("Input contains non-finite values")
    array = np.ascontiguousarray(array, dtype=np.float32)

    print("Radiation Edge AI - physical KL720 DNAi NEF smoke test")
    print(f"kp module: {Path(kp.__file__).resolve()}")
    print(f"NEF: {nef_path}")
    print(f"NEF SHA256: {observed_nef_sha}")
    print(f"Input: {input_path}")
    print(f"Input shape/dtype: {array.shape} / {array.dtype}")
    print(f"Input range: [{float(array.min()):.6f}, {float(array.max()):.6f}]")
    print("")

    descriptors = scan_descriptors(kp)
    matching = [d for d in descriptors if int(getattr(d, "usb_port_id", -1)) == args.port]
    if not matching:
        visible = [getattr(d, "usb_port_id", None) for d in descriptors]
        raise RuntimeError(f"KL720 USB port {args.port} not found; visible ports={visible}")
    device_desc = matching[0]
    product_id = int(getattr(device_desc, "product_id", -1))
    if product_id != 0x720:
        raise RuntimeError(
            f"USB port {args.port} is not KL720: product_id={product_id:#x}"
        )
    print(f"[Device] port={args.port} KL720 detected (product_id=0x720)")
    firmware = getattr(device_desc, "firmware", None)
    if firmware is not None:
        print(f"  firmware: {firmware}")

    try:
        device_group = kp.core.connect_devices(usb_port_ids=[args.port])
    except TypeError:
        device_group = kp.core.connect_devices([args.port])
    print("[Connect] PASS")

    try:
        kp.core.set_timeout(device_group=device_group, milliseconds=args.timeout_ms)
    except TypeError:
        kp.core.set_timeout(device_group, args.timeout_ms)
    print(f"[Timeout] {args.timeout_ms} ms")

    t0 = time.perf_counter()
    model_nef_descriptor = kp.core.load_model_from_file(
        device_group=device_group,
        file_path=str(nef_path),
    )
    model_load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[Load NEF] PASS ({model_load_ms:.1f} ms)")

    if len(model_nef_descriptor.models) != 1:
        raise RuntimeError(f"Expected one model in NEF, got {len(model_nef_descriptor.models)}")
    model = model_nef_descriptor.models[0]
    if len(model.input_nodes) != 1:
        raise RuntimeError(f"Expected one model input, got {len(model.input_nodes)}")
    input_node = model.input_nodes[0]
    input_shape = descriptor_shape(input_node)
    if input_shape != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(f"NEF input shape {input_shape}; expected {EXPECTED_INPUT_SHAPE}")

    output_shapes = [descriptor_shape(node) for node in model.output_nodes]
    print(f"[NEF descriptor] model_id={model.id}")
    print(f"  input shape (v1.shape_npu): {input_shape}")
    print(f"  input layout: {getattr(input_node, 'data_layout', '<unknown>')}")
    print(f"  output shapes: {output_shapes}")

    # The .npy already is ONNX-space normalized float32 input. KL720 then needs
    # host-side fixed-point quantization and v1 NPU re-layout.
    t0 = time.perf_counter()
    npu_input_buffer, pack_meta = pack_kl720_input(kp, input_node, array)
    host_pack_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"[Host KL720 quantize+relayout] PASS ({host_pack_ms:.1f} ms, "
        f"{len(npu_input_buffer)} bytes)"
    )
    print(
        "  "
        f"radix={pack_meta['radix']} scale={pack_meta['scale']:.9g} "
        f"factor={pack_meta['quantization_factor']:.9g} "
        f"layout={pack_meta['layout']} qrange=[{pack_meta['quantized_min']},"
        f"{pack_meta['quantized_max']}]"
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
    print(f"[Physical inference] PASS ({hardware_ms:.1f} ms send+receive)")

    n_outputs = int(raw_result.header.num_output_node)
    if n_outputs != 1:
        raise RuntimeError(f"Expected one output node, got {n_outputs}")

    float_output_obj = kp.inference.generic_inference_retrieve_float_node(
        node_idx=0,
        generic_raw_result=raw_result,
        channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW,
    )
    output = np.asarray(extract_array(float_output_obj), dtype=np.float32)

    if output.shape == EXPECTED_OUTPUT_SHAPE[1:]:
        output = output[None, ...]
    if output.shape != EXPECTED_OUTPUT_SHAPE:
        raise RuntimeError(
            f"Retrieved output shape {output.shape}; expected {EXPECTED_OUTPUT_SHAPE}. "
            "No transpose or silent re-layout is attempted."
        )
    if not np.isfinite(output).all():
        raise RuntimeError("Physical output contains non-finite values")
    output = np.ascontiguousarray(output, dtype=np.float32)

    output_path = output_dir / "physical_output.npy"
    np.save(output_path, output, allow_pickle=False)
    print(f"[Retrieve float output] PASS shape={output.shape}")
    print(f"  range: [{float(output.min()):.6f}, {float(output.max()):.6f}]")
    print(f"  saved: {output_path}")

    comparison = None
    if bie_reference_path is not None:
        reference = np.load(bie_reference_path, allow_pickle=False)
        if reference.shape != EXPECTED_OUTPUT_SHAPE or reference.dtype != np.float32:
            raise RuntimeError(
                f"BIE reference {reference.shape} {reference.dtype}; "
                f"expected {EXPECTED_OUTPUT_SHAPE} float32"
            )
        comparison = error_metrics(reference, output)
        print("")
        print("[Physical KL720 vs frozen v3 BIE]")
        print(f"max absolute error: {comparison['max_abs_error']:.8g}")
        print(f"mean absolute error: {comparison['mean_abs_error']:.8g}")
        print(f"RMSE: {comparison['rmse']:.8g}")
        print(f"argmax agreement: {comparison['argmax_agreement']:.8f}")
        print(
            f"disagreement pixels: {comparison['disagreement_pixels']}/"
            f"{comparison['total_pixels']}"
        )

    summary = {
        "nef": str(nef_path),
        "nef_sha256": observed_nef_sha,
        "input": str(input_path),
        "input_shape": list(array.shape),
        "input_dtype": str(array.dtype),
        "usb_port": args.port,
        "timeout_ms": args.timeout_ms,
        "model_id": int(model.id),
        "nef_input_shape": list(input_shape),
        "nef_output_shapes": [list(shape) for shape in output_shapes],
        "model_load_ms": model_load_ms,
        "host_pack_ms": host_pack_ms,
        "kl720_pack": pack_meta,
        "hardware_send_receive_ms": hardware_ms,
        "physical_output": str(output_path),
        "physical_output_shape": list(output.shape),
        "bie_reference": str(bie_reference_path) if bie_reference_path else None,
        "comparison": comparison,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("PHYSICAL KL720 INFERENCE COMPLETE: YES")
    if comparison is not None:
        class_pass = comparison["disagreement_pixels"] == 0
        print(f"PHYSICAL/BIE ARGMAX PARITY PASS: {'YES' if class_pass else 'NO'}")
    print(f"Summary: {summary_path}")
    print("")
    print("NEXT GATE: if physical-vs-BIE parity is satisfactory, run all 180 frozen windows on KL720.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
