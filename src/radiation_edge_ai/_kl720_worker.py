"""Isolated physical KL720 worker for Radiation Edge AI.

Run this only with the known-good Kneron PLUS Python environment. The parent
``radedge`` process never imports ``kp``.

The input contract is deliberately narrow: an already-preprocessed float32
batch-1 BCHW NumPy tensor and a frozen one-input KL720 NEF. Multiple
outputs require an explicit, unambiguous channel-joining contract.

No assay-specific preprocessing, model conversion, PTQ, endpoint reconstruction,
or biological acceptance evaluation occurs here.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def descriptor_shape(tensor_descriptor) -> tuple[int, ...]:
    info = tensor_descriptor.tensor_shape_info
    if hasattr(info, "v1") and hasattr(info.v1, "shape_npu"):
        return tuple(int(v) for v in info.v1.shape_npu)
    if hasattr(info, "v2") and hasattr(info.v2, "shape"):
        return tuple(int(v) for v in info.v2.shape)
    raise RuntimeError("NEF tensor descriptor exposes no supported logical shape")


def scan_descriptors(kp):
    scan = kp.core.scan_devices()
    descriptors = getattr(scan, "device_descriptor_list", None)
    if descriptors is None:
        try:
            descriptors = list(scan)
        except TypeError:
            descriptors = []
    return list(descriptors)


def select_kl720(descriptors, requested_port):
    devices = [
        item
        for item in descriptors
        if int(getattr(item, "product_id", -1)) == 0x720
    ]
    if requested_port is not None:
        matches = [
            item
            for item in devices
            if int(getattr(item, "usb_port_id", -1)) == requested_port
        ]
        if not matches:
            visible = [getattr(item, "usb_port_id", None) for item in devices]
            raise RuntimeError(
                f"Requested KL720 USB port {requested_port} not found; KL720 ports={visible}"
            )
        return matches[0]

    if len(devices) != 1:
        visible = [getattr(item, "usb_port_id", None) for item in devices]
        raise RuntimeError(
            "Automatic KL720 selection requires exactly one KL720; "
            f"found {len(devices)} at ports {visible}. Pass --port."
        )
    return devices[0]


def kl720_quantization(input_node) -> tuple[int, float]:
    params = input_node.quantization_parameters
    if not hasattr(params, "v1"):
        raise RuntimeError("KL720 input quantization parameters do not expose v1")
    descriptors = params.v1.quantized_fixed_point_descriptor_list
    if len(descriptors) != 1:
        raise RuntimeError(
            f"Expected one KL720 input fixed-point descriptor, got {len(descriptors)}"
        )
    item = descriptors[0]
    radix = int(item.radix)
    scale_obj = item.scale
    scale = float(scale_obj.value if hasattr(scale_obj, "value") else scale_obj)
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"Invalid KL720 input scale: {scale}")
    return radix, scale


def pack_kl720_input(kp, input_node, onnx_data: np.ndarray):
    """Match the previously validated KL720 Generic Data host packing path."""

    shape = descriptor_shape(input_node)
    if tuple(onnx_data.shape) != shape:
        raise RuntimeError(
            f"Input shape {tuple(onnx_data.shape)} does not match NEF input {shape}"
        )
    if len(shape) != 4 or shape[0] != 1:
        raise RuntimeError(f"v0.1c requires batch-1 BCHW input, got {shape}")

    _, channels, height, width = shape
    radix, scale = kl720_quantization(input_node)
    factor = float(np.power(2.0, radix) * scale)

    hwc = np.asarray(onnx_data[0].transpose(1, 2, 0), dtype=np.float32)
    layout = input_node.data_layout
    dtype = np.dtype(np.int8)
    if layout == kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B:
        width_align_base, channel_align_base, layout_name = 4, 4, "4W4C8B"
    elif layout == kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8B:
        width_align_base, channel_align_base, layout_name = 1, 16, "1W16C8B"
    elif layout == kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8B:
        width_align_base, channel_align_base, layout_name = 16, 1, "16W1C8B"
    elif layout == getattr(kp.ModelTensorDataLayout, "KP_MODEL_TENSOR_DATA_LAYOUT_8W1C16B", object()):
        # KL720 V1 standard signed fixed point, with eight little-endian words per 128-bit entry.
        if channels != 1 or not hasattr(input_node.tensor_shape_info, "v1"):
            raise RuntimeError("Expected a single-channel KL720 V1 input for this 8W1C16B adapter")
        width_align_base, channel_align_base, layout_name = 8, 1, "8W1C16B"
        dtype = np.dtype("<i2")
    elif layout == getattr(kp.ModelTensorDataLayout, "KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8BHL", object()):
        if channels != 1 or not hasattr(input_node.tensor_shape_info, "v1"):
            raise RuntimeError("Expected a single-channel KL720 V1 input for this 1W16C8BHL adapter")
        width_align_base, channel_align_base, layout_name = 1, 16, "1W16C8BHL"
        dtype = np.dtype("<i2")
    else:
        raise RuntimeError(
            f"Unsupported KL720 Generic Data input layout: {layout}"
        )

    limits = np.iinfo(dtype)
    quantized = np.clip(np.rint(hwc * factor), limits.min, limits.max).astype(dtype)
    width_aligned = width_align_base * math.ceil(
        width / float(width_align_base)
    )
    channel_blocks = math.ceil(channels / float(channel_align_base))
    if layout_name == "1W16C8BHL":
        # SDK FMT_RAW_NPU16B -> FMT_1W16C8BHL: each pixel has two 16-lane entries.
        # The low entry carries bits 7..1, the high entry bits 15..8.
        words = quantized[:, :, 0].view("<u2")
        relayout = np.zeros((height, width, 2, channel_align_base), dtype=np.uint8)
        relayout[:, :, 0, 0] = (words >> 1) & 0x7f
        relayout[:, :, 1, 0] = words >> 8
    else:
        relayout = np.zeros((channel_blocks, height, width_aligned, channel_align_base), dtype=dtype)
        for block in range(channel_blocks):
            channel_offset = block * channel_align_base
            count = min(channel_align_base, channels - channel_offset)
            relayout[block, :, :width, :count] = quantized[:, :, channel_offset:channel_offset + count]

    metadata = {
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
    return relayout.tobytes(), metadata


def extract_array(node_output) -> np.ndarray:
    if isinstance(node_output, np.ndarray):
        return node_output
    for attr in ("ndarray", "array", "data"):
        if hasattr(node_output, attr):
            value = getattr(node_output, attr)
            if isinstance(value, np.ndarray):
                return value
    public = [name for name in dir(node_output) if not name.startswith("_")]
    raise RuntimeError(
        "Could not extract NumPy array from KL720 float-node result; "
        f"type={type(node_output)!r}, attrs={public}"
    )


def normalize_retrieved_output(value: np.ndarray, logical_shape):
    output = np.asarray(value, dtype=np.float32)
    expected = tuple(int(v) for v in logical_shape)
    if not np.isfinite(output).all():
        raise RuntimeError("Physical KL720 output contains non-finite values")

    if output.size == 1:
        return np.ascontiguousarray(output.reshape(1), dtype=np.float32)
    if output.shape == expected:
        return np.ascontiguousarray(output, dtype=np.float32)
    if len(expected) > 1 and expected[0] == 1 and output.shape == expected[1:]:
        return np.ascontiguousarray(output[None, ...], dtype=np.float32)

    raise RuntimeError(
        f"Retrieved KL720 output shape {output.shape} does not match descriptor {expected}. "
        "No silent reshape or transpose is allowed."
    )


def record_raw_receive(result, path: Path, inference_number: int):
    """Preserve the PLUS 3.2 allocated receive buffer and identify its header."""
    header = result.header.get_member_variable_dict()
    if header.get("inference_number") is not None and int(header["inference_number"]) != inference_number:
        raise RuntimeError("Received a different KL720 inference number")
    raw = result.raw_result
    size = int(raw.buffer_size)
    address = raw._get_element_buffer()._LP_raw_out_buffer
    if not address or size <= 0:
        raise RuntimeError("Empty KL720 receive buffer")
    with path.open("xb") as handle:
        handle.write(ctypes.string_at(address, size))
    return {"header": header, "allocation": artifact_record(path)}


def output_order(shapes, channels=None):
    if channels is None:
        if len(shapes) != 1:
            raise RuntimeError("Multiple NEF outputs require an explicit output channel contract")
        return [0], shapes[0]
    if not shapes or any(len(s) != 4 or min(s) <= 0 or s[0] != 1 or s[2:] != shapes[0][2:] for s in shapes):
        raise RuntimeError("Joined outputs require matching batch-1 spatial dimensions")
    widths = [shape[1] for shape in shapes]
    if len(set(channels)) != len(channels) or sorted(widths) != sorted(channels):
        raise RuntimeError("Cannot unambiguously match NEF outputs to the requested channels")
    return [widths.index(c) for c in channels], (1, sum(channels), *shapes[0][2:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--expected-kn", type=lambda x: int(x, 0))
    parser.add_argument("--expected-firmware-version")
    parser.add_argument("--required-plus-version")
    parser.add_argument("--required-plus-dll-sha256")
    parser.add_argument("--raw-receive", type=Path, help="Optional fresh file for the complete PLUS receive allocation")
    parser.add_argument("--join-output-channels", nargs="+", type=int,
                        help="Dequantize each output using PLUS and concatenate in this channel-count order")
    parser.add_argument("--timeout-ms", type=int, default=10000)
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()

    if model_path.suffix.lower() != ".nef":
        raise RuntimeError("KL720 worker requires a .nef model")
    if input_path.suffix.lower() != ".npy":
        raise RuntimeError("KL720 worker requires a preprocessed .npy tensor")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if args.timeout_ms <= 0:
        raise RuntimeError("--timeout-ms must be positive")
    if output_path.exists() or manifest_path.exists() or (args.raw_receive and args.raw_receive.exists()):
        raise RuntimeError("Refusing to overwrite existing KL720 worker artifacts")

    array = np.load(str(input_path), allow_pickle=False)
    if not isinstance(array, np.ndarray):
        raise TypeError("KL720 input is not a NumPy ndarray")
    if array.dtype != np.float32:
        raise RuntimeError(
            f"KL720 input dtype {array.dtype}; expected float32"
        )
    if array.ndim != 4 or array.shape[0] != 1:
        raise RuntimeError(
            f"KL720 input must be batch-1 BCHW; got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise RuntimeError("KL720 input contains non-finite values")
    array = np.ascontiguousarray(array, dtype=np.float32)

    import kp

    descriptors = scan_descriptors(kp)
    device_desc = select_kl720(descriptors, args.port)
    if args.expected_kn is not None and int(str(device_desc.kn_number), 0) != args.expected_kn:
        raise RuntimeError("Selected USB descriptor has a different KN number")
    if args.required_plus_version and kp.core.get_version() != args.required_plus_version:
        raise RuntimeError("Unexpected PLUS runtime version")
    if args.required_plus_dll_sha256 and sha256_file(Path(kp.__file__).parent / "lib/libkplus.dll") != args.required_plus_dll_sha256:
        raise RuntimeError("Unexpected PLUS DLL identity")
    port = int(getattr(device_desc, "usb_port_id", -1))
    product_id = int(getattr(device_desc, "product_id", -1))
    if product_id != 0x720:
        raise RuntimeError("Selected device is not KL720")

    try:
        device_group = kp.core.connect_devices(usb_port_ids=[port])
    except TypeError:
        device_group = kp.core.connect_devices([port])

    try:
        kp.core.set_timeout(
            device_group=device_group,
            milliseconds=args.timeout_ms,
        )
    except TypeError:
        kp.core.set_timeout(device_group, args.timeout_ms)

    if args.expected_kn is not None or args.expected_firmware_version:
        system = kp.core.get_system_info(device_group, port).get_member_variable_dict()
        if ((args.expected_kn is not None and int(str(system["kn_number"]), 0) != args.expected_kn)
                or (args.expected_firmware_version and system["firmware_version"] != args.expected_firmware_version)):
            kp.core.disconnect_devices(device_group)
            raise RuntimeError("Connected firmware identity differs from the required physical deployment")
    started = time.perf_counter()
    model_descriptor = kp.core.load_model_from_file(
        device_group=device_group,
        file_path=str(model_path),
    )
    model_load_seconds = time.perf_counter() - started

    if len(model_descriptor.models) != 1:
        raise RuntimeError(
            f"v0.1c requires exactly one model in NEF; got {len(model_descriptor.models)}"
        )
    model = model_descriptor.models[0]
    if len(model.input_nodes) != 1:
        raise RuntimeError(f"Expected one input; got {len(model.input_nodes)}")

    input_node = model.input_nodes[0]
    input_shape = descriptor_shape(input_node)
    output_shapes = [descriptor_shape(node) for node in model.output_nodes]
    order, output_shape = output_order(output_shapes, args.join_output_channels)
    if tuple(array.shape) != input_shape:
        raise RuntimeError(
            f"Input shape {tuple(array.shape)} does not match loaded NEF {input_shape}"
        )

    started = time.perf_counter()
    npu_buffer, pack_meta = pack_kl720_input(kp, input_node, array)
    host_pack_seconds = time.perf_counter() - started

    descriptor = kp.GenericDataInferenceDescriptor(
        model_id=model.id,
        inference_number=0,
        input_node_data_list=[kp.GenericInputNodeData(buffer=npu_buffer)],
    )

    started = time.perf_counter()
    kp.inference.generic_data_inference_send(
        device_group=device_group,
        generic_inference_input_descriptor=descriptor,
    )
    raw_result = kp.inference.generic_data_inference_receive(
        device_group=device_group
    )
    inference_seconds = time.perf_counter() - started

    if int(raw_result.header.num_output_node) != len(order):
        raise RuntimeError(f"Physical output count {raw_result.header.num_output_node} differs from descriptor")

    parts = []
    for idx in order:
        float_obj = kp.inference.generic_inference_retrieve_float_node(
            node_idx=idx, generic_raw_result=raw_result,
            channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW,
        )
        parts.append(normalize_retrieved_output(extract_array(float_obj), output_shapes[idx]))
    output = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        np.save(handle, output, allow_pickle=False)

    output_record = artifact_record(output_path)
    output_record.update(
        {
            "dtype": str(output.dtype),
            "shape": [int(v) for v in output.shape],
        }
    )

    firmware = getattr(device_desc, "firmware", None)
    model_version = getattr(model, "version", None)
    kp_version = getattr(kp, "__version__", None)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "backend": "kl720",
        "input": {
            **artifact_record(input_path),
            "dtype": str(array.dtype),
            "shape": [int(v) for v in array.shape],
        },
        "model": artifact_record(model_path),
        "output": output_record,
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "kp_module": str(Path(kp.__file__).resolve()),
            "kp_version": str(kp_version) if kp_version is not None else None,
            "selected_usb_port": port,
            "product_id": product_id,
            "firmware": str(firmware) if firmware is not None else None,
            "model_id": int(model.id),
            "model_version": (
                str(model_version) if model_version is not None else None
            ),
            "nef_input_shape": [int(v) for v in input_shape],
            "nef_output_shape": [int(v) for v in output_shape],
            "timeout_ms": int(args.timeout_ms),
        },
        "timing": {
            "model_load_seconds": model_load_seconds,
            "host_pack_seconds": host_pack_seconds,
            "inference_send_receive_seconds": inference_seconds,
        },
        "kl720_input_pack": pack_meta,
        "scientific_interpretation": {
            "preprocessing_performed": False,
            "endpoint_reconstruction_performed": False,
            "biological_acceptance_evaluated": False,
        },
    }

    if args.raw_receive:
        args.raw_receive.parent.mkdir(parents=True, exist_ok=True)
        manifest["raw_receive"] = record_raw_receive(raw_result, args.raw_receive, 0)
        manifest["runtime"].update(
            plus_version=kp.core.get_version(),
            device=device_desc.get_member_variable_dict(),
            system_info=kp.core.get_system_info(device_group, port).get_member_variable_dict(),
        )
    if args.join_output_channels:
        manifest["runtime"].update(nef_output_node_shapes=output_shapes, joined_node_order=order)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
