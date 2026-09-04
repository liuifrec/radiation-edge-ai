"""Run one frozen DNAi 512x512 validation window on the physical KL720.

This script must be executed with the known-good Kneron PLUS Python runtime,
not the separate DNAi reference environment. It deliberately reuses the
installed PLUS ExampleHelper.convert_onnx_data_to_npu_data implementation so
host-side quantization and NPU re-layout exactly follow the user's installed
PLUS 3.2.0 examples.

Input is the already-normalized float32 ONNX tensor used by the frozen BIE
validation. No DNAi image normalization is repeated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

EXPECTED_NEF_SHA256 = "4b3dfec9a61c99e186dd4b8482fa5b06e6a4958f325ed4a0db0546f1dcab2bfc"
EXPECTED_INPUT_SHAPE = (1, 3, 512, 512)
EXPECTED_OUTPUT_SHAPE = (1, 3, 512, 512)
EXPECTED_KL720_PRODUCT_ID = 0x720


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
    info = tensor_descriptor.tensor_shape_info
    if hasattr(info, "v2") and hasattr(info.v2, "shape"):
        return tuple(int(v) for v in info.v2.shape)
    raise RuntimeError("NEF tensor descriptor does not expose v2.shape")


def scan_descriptors(kp):
    scan = kp.core.scan_devices()
    descriptors = getattr(scan, "device_descriptor_list", None)
    if descriptors is None:
        try:
            descriptors = list(scan)
        except TypeError:
            descriptors = []
    return list(descriptors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nef", required=True)
    parser.add_argument("--input", required=True, help="Frozen normalized float32 NCHW .npy tensor")
    parser.add_argument("--bie-reference", default=None, help="Optional frozen v3 BIE output .npy")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", type=int, default=81)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument(
        "--examples-root",
        default=r"C:\Users\yul03\kneron_plus\python\example",
        help="Installed Kneron PLUS Python example directory containing utils/ExampleHelper.py",
    )
    args = parser.parse_args()

    nef_path = Path(args.nef).resolve()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bie_reference_path = Path(args.bie_reference).resolve() if args.bie_reference else None
    examples_root = Path(args.examples_root).resolve()

    for path in (nef_path, input_path, examples_root):
        if not path.exists():
            raise FileNotFoundError(path)
    if bie_reference_path is not None and not bie_reference_path.is_file():
        raise FileNotFoundError(bie_reference_path)

    observed_nef_sha = sha256_file(nef_path)
    if observed_nef_sha.lower() != EXPECTED_NEF_SHA256:
        raise RuntimeError(
            "Refusing to run unexpected NEF. "
            f"observed={observed_nef_sha}, expected={EXPECTED_NEF_SHA256}"
        )

    # Import the exact helper shipped with the installed PLUS distribution.
    if str(examples_root) not in sys.path:
        sys.path.insert(0, str(examples_root))
    from utils.ExampleHelper import convert_onnx_data_to_npu_data  # type: ignore
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
    device_descriptor = matching[0]
    product_id = int(getattr(device_descriptor, "product_id", -1))
    if product_id != EXPECTED_KL720_PRODUCT_ID:
        raise RuntimeError(
            f"USB port {args.port} is not a KL720: product_id={product_id}, "
            f"expected={EXPECTED_KL720_PRODUCT_ID}"
        )
    firmware = getattr(device_descriptor, "firmware", getattr(device_descriptor, "firmware_version", "<unknown>"))
    print(f"[Device] port={args.port} KL720 detected (product_id=0x{product_id:x})")
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
    print(f"  input shape: {input_shape}")
    print(f"  input layout: {getattr(input_node, 'data_layout', '<unknown>')}")
    print(f"  output shapes: {output_shapes}")

    # The .npy already is ONNX-space normalized float32 input. The PLUS helper
    # performs only the required fixed-point quantization and NPU re-layout here.
    t0 = time.perf_counter()
    npu_input_buffer = convert_onnx_data_to_npu_data(
        tensor_descriptor=input_node,
        onnx_data=array,
    )
    host_pack_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[Host ONNX->NPU conversion] PASS ({host_pack_ms:.1f} ms, {len(npu_input_buffer)} bytes)")

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
        channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_DEFAULT,
    )
    output = np.asarray(extract_array(float_output_obj), dtype=np.float32)

    # Some PLUS releases omit a size-1 batch axis when returning a float node.
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
        "product_id": product_id,
        "timeout_ms": args.timeout_ms,
        "model_id": int(model.id),
        "nef_input_shape": list(input_shape),
        "nef_output_shapes": [list(shape) for shape in output_shapes],
        "model_load_ms": model_load_ms,
        "host_pack_ms": host_pack_ms,
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
