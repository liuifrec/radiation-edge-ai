"""Run the frozen 20-image DNAi validation panel on the physical KL720.

This is the full-hardware counterpart of ``run_kl720_dnai_nef_smoke.py``.
It reuses the already verified v3 NEF, loads the model once, then runs every
512x512 / 50%-overlap validation tensor from the frozen manifest through the
physical KL720 Generic Data path.

The input .npy tensors are already-normalized ONNX-space float32 BCHW arrays.
KL720 host-side fixed-point quantization and NPU re-layout are delegated to the
validated helper functions in ``run_kl720_dnai_nef_smoke.py``. Outputs are
saved as float32 BCHW logits for downstream stitching and biological-fidelity
analysis in the separate DNAi Python environment.

The runner is resumable by default: existing valid physical output arrays are
skipped. Use --overwrite to regenerate them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from run_kl720_dnai_nef_smoke import (
    EXPECTED_INPUT_SHAPE,
    EXPECTED_NEF_SHA256,
    EXPECTED_OUTPUT_SHAPE,
    descriptor_shape,
    extract_array,
    pack_kl720_input,
    scan_descriptors,
    sha256_file,
)


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if tuple(manifest.get("tile", ())) != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(
            f"Validation manifest tile {manifest.get('tile')} does not match {EXPECTED_INPUT_SHAPE}"
        )
    if float(manifest.get("overlap", -1)) != 0.5:
        raise RuntimeError(f"Expected frozen overlap=0.5, got {manifest.get('overlap')}")
    records = manifest.get("records", [])
    if not records:
        raise RuntimeError("Validation manifest contains no records")
    return manifest


def parse_sample_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    result = {token.strip() for token in value.split(",") if token.strip()}
    if not result:
        raise argparse.ArgumentTypeError("--sample-ids selected no samples")
    return result


def stem_for(record: dict) -> str:
    return f"{int(record['image_index']):02d}_{int(record['window_index']):02d}_{record['sample_id']}"


def valid_saved_output(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        array = np.load(path, allow_pickle=False)
    except Exception:
        return False
    return (
        array.shape == EXPECTED_OUTPUT_SHAPE
        and array.dtype == np.float32
        and bool(np.isfinite(array).all())
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nef", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", type=int, default=81)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument(
        "--sample-ids",
        default=None,
        help="Optional comma-separated sample IDs; otherwise run the complete frozen panel",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional window limit after filtering")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    nef_path = Path(args.nef).resolve()
    manifest_path = Path(args.validation_manifest).resolve()
    output_root = Path(args.output_dir).resolve()
    physical_dir = output_root / "physical_outputs"
    physical_dir.mkdir(parents=True, exist_ok=True)

    if not nef_path.is_file():
        raise FileNotFoundError(nef_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    observed_nef_sha = sha256_file(nef_path)
    if observed_nef_sha.lower() != EXPECTED_NEF_SHA256:
        raise RuntimeError(
            "Refusing to run unexpected NEF. "
            f"observed={observed_nef_sha}, expected={EXPECTED_NEF_SHA256}"
        )

    manifest = load_manifest(manifest_path)
    records = list(manifest["records"])
    requested_samples = parse_sample_ids(args.sample_ids)
    if requested_samples is not None:
        available = {str(r["sample_id"]) for r in records}
        missing = sorted(requested_samples - available)
        if missing:
            raise RuntimeError(f"Requested sample IDs not in manifest: {missing}")
        records = [r for r in records if str(r["sample_id"]) in requested_samples]
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        records = records[: args.limit]
    if not records:
        raise RuntimeError("No validation windows selected")

    validation_root = manifest_path.parent

    import kp

    print("Radiation Edge AI - physical KL720 DNAi frozen-panel run")
    print(f"kp module: {Path(kp.__file__).resolve()}")
    print(f"NEF: {nef_path}")
    print(f"NEF SHA256: {observed_nef_sha}")
    print(f"Manifest: {manifest_path}")
    print(f"Selected windows: {len(records)}")
    print(f"Output: {physical_dir}")
    print(f"Resume existing outputs: {'NO' if args.overwrite else 'YES'}")
    print("")

    descriptors = scan_descriptors(kp)
    matching = [d for d in descriptors if int(getattr(d, "usb_port_id", -1)) == args.port]
    if not matching:
        visible = [getattr(d, "usb_port_id", None) for d in descriptors]
        raise RuntimeError(f"KL720 USB port {args.port} not found; visible ports={visible}")
    device = matching[0]
    product_id = int(getattr(device, "product_id", -1))
    if product_id != 0x720:
        raise RuntimeError(
            f"USB port {args.port} is product_id=0x{product_id:x}, expected KL720 (0x720)"
        )
    print(f"[Device] port={args.port} KL720 detected (product_id=0x{product_id:x})")
    firmware = getattr(device, "firmware", getattr(device, "firmware_version", None))
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
    if len(model.output_nodes) != 1:
        raise RuntimeError(f"Expected one model output, got {len(model.output_nodes)}")
    input_node = model.input_nodes[0]
    if descriptor_shape(input_node) != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(
            f"NEF input shape {descriptor_shape(input_node)}; expected {EXPECTED_INPUT_SHAPE}"
        )
    output_shape = descriptor_shape(model.output_nodes[0])
    if output_shape != EXPECTED_OUTPUT_SHAPE:
        raise RuntimeError(f"NEF output shape {output_shape}; expected {EXPECTED_OUTPUT_SHAPE}")

    # Probe packing metadata once. Every tensor uses the same NEF input descriptor.
    first_tensor = validation_root / records[0]["tensor"]
    first_array = np.load(first_tensor, allow_pickle=False)
    if first_array.shape != EXPECTED_INPUT_SHAPE or first_array.dtype != np.float32:
        raise RuntimeError(f"Invalid first validation tensor {first_array.shape} {first_array.dtype}")
    _first_buffer, pack_meta = pack_kl720_input(kp, input_node, first_array)
    print("[KL720 input packing]")
    print(
        f"  radix={pack_meta['radix']} scale={pack_meta['scale']:.8g} "
        f"factor={pack_meta['quantization_factor']:.8g} layout={pack_meta['layout']}"
    )
    print(f"  packed bytes/window={pack_meta['buffer_bytes']}")
    print("")

    rows: list[dict] = []
    started_all = time.perf_counter()
    n_ran = 0
    n_skipped = 0

    for position, record in enumerate(records, start=1):
        stem = stem_for(record)
        input_path = validation_root / record["tensor"]
        output_path = physical_dir / f"{stem}.npy"

        if not args.overwrite and valid_saved_output(output_path):
            n_skipped += 1
            rows.append(
                {
                    "position": position,
                    "image_index": int(record["image_index"]),
                    "window_index": int(record["window_index"]),
                    "sample_id": record["sample_id"],
                    "key": record["key"],
                    "input_tensor": str(input_path),
                    "physical_output": str(output_path),
                    "host_pack_ms": None,
                    "hardware_send_receive_ms": None,
                    "status": "existing",
                }
            )
            print(f"[{position:03d}/{len(records):03d}] {record['sample_id']} w{record['window_index']} existing")
            continue

        array = np.load(input_path, allow_pickle=False)
        if array.shape != EXPECTED_INPUT_SHAPE or array.dtype != np.float32:
            raise RuntimeError(f"Invalid validation tensor {array.shape} {array.dtype}: {input_path}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"Non-finite validation tensor: {input_path}")
        array = np.ascontiguousarray(array, dtype=np.float32)

        t0 = time.perf_counter()
        npu_input_buffer, current_meta = pack_kl720_input(kp, input_node, array)
        host_pack_ms = (time.perf_counter() - t0) * 1000.0
        for key in ("radix", "scale", "quantization_factor", "layout", "buffer_bytes"):
            if current_meta[key] != pack_meta[key]:
                raise RuntimeError(f"KL720 packing metadata changed at {stem}: {key}")

        descriptor = kp.GenericDataInferenceDescriptor(
            model_id=model.id,
            inference_number=position - 1,
            input_node_data_list=[kp.GenericInputNodeData(buffer=npu_input_buffer)],
        )

        t0 = time.perf_counter()
        kp.inference.generic_data_inference_send(
            device_group=device_group,
            generic_inference_input_descriptor=descriptor,
        )
        raw_result = kp.inference.generic_data_inference_receive(device_group=device_group)
        hardware_ms = (time.perf_counter() - t0) * 1000.0

        if int(raw_result.header.num_output_node) != 1:
            raise RuntimeError(
                f"Expected one output node for {stem}, got {raw_result.header.num_output_node}"
            )
        float_obj = kp.inference.generic_inference_retrieve_float_node(
            node_idx=0,
            generic_raw_result=raw_result,
            channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_DEFAULT,
        )
        output = np.asarray(extract_array(float_obj), dtype=np.float32)
        if output.shape == EXPECTED_OUTPUT_SHAPE[1:]:
            output = output[None, ...]
        if output.shape != EXPECTED_OUTPUT_SHAPE:
            raise RuntimeError(f"Physical output shape {output.shape}; expected {EXPECTED_OUTPUT_SHAPE}: {stem}")
        if not np.isfinite(output).all():
            raise RuntimeError(f"Physical output contains non-finite values: {stem}")
        output = np.ascontiguousarray(output, dtype=np.float32)
        np.save(output_path, output, allow_pickle=False)
        n_ran += 1

        rows.append(
            {
                "position": position,
                "image_index": int(record["image_index"]),
                "window_index": int(record["window_index"]),
                "sample_id": record["sample_id"],
                "key": record["key"],
                "input_tensor": str(input_path),
                "physical_output": str(output_path),
                "host_pack_ms": host_pack_ms,
                "hardware_send_receive_ms": hardware_ms,
                "status": "ran",
            }
        )
        print(
            f"[{position:03d}/{len(records):03d}] {record['sample_id']} w{record['window_index']} "
            f"pack={host_pack_ms:.1f}ms hw={hardware_ms:.1f}ms"
        )

    elapsed = time.perf_counter() - started_all

    csv_path = output_root / "per_window.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    selected_stems = {stem_for(r) for r in records}
    missing_outputs = sorted(
        stem for stem in selected_stems if not valid_saved_output(physical_dir / f"{stem}.npy")
    )
    ran_rows = [r for r in rows if r["status"] == "ran"]
    mean_pack = (
        float(np.mean([float(r["host_pack_ms"]) for r in ran_rows])) if ran_rows else None
    )
    mean_hw = (
        float(np.mean([float(r["hardware_send_receive_ms"]) for r in ran_rows])) if ran_rows else None
    )
    p50_hw = (
        float(np.median([float(r["hardware_send_receive_ms"]) for r in ran_rows])) if ran_rows else None
    )

    unique_samples = []
    seen = set()
    for record in records:
        sample = str(record["sample_id"])
        if sample not in seen:
            unique_samples.append(sample)
            seen.add(sample)

    summary = {
        "nef": str(nef_path),
        "nef_sha256": observed_nef_sha,
        "validation_manifest": str(manifest_path),
        "usb_port": args.port,
        "model_id": int(model.id),
        "selected_samples": unique_samples,
        "selected_windows": len(records),
        "n_ran": n_ran,
        "n_existing_skipped": n_skipped,
        "n_valid_outputs": len(records) - len(missing_outputs),
        "missing_outputs": missing_outputs,
        "model_load_ms": model_load_ms,
        "mean_host_pack_ms": mean_pack,
        "mean_hardware_send_receive_ms": mean_hw,
        "median_hardware_send_receive_ms": p50_hw,
        "estimated_device_windows_per_second": (1000.0 / mean_hw) if mean_hw else None,
        "elapsed_seconds": elapsed,
        "packing": pack_meta,
        "physical_output_dir": str(physical_dir),
        "per_window_csv": str(csv_path),
        "complete": not missing_outputs,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("PHYSICAL KL720 FROZEN-PANEL RUN COMPLETE: " + ("YES" if not missing_outputs else "NO"))
    print(f"Selected windows: {len(records)}")
    print(f"Ran now: {n_ran}")
    print(f"Reused existing: {n_skipped}")
    print(f"Valid outputs: {len(records) - len(missing_outputs)}/{len(records)}")
    if mean_pack is not None:
        print(f"Mean host pack: {mean_pack:.2f} ms/window")
    if mean_hw is not None:
        print(f"Mean physical send+receive: {mean_hw:.2f} ms/window")
        print(f"Estimated physical throughput: {1000.0 / mean_hw:.3f} windows/s")
    print(f"Elapsed wall time: {elapsed:.1f} s")
    print(f"Summary: {summary_path}")
    if missing_outputs:
        print(f"Missing/invalid outputs: {len(missing_outputs)}")
        return 2
    print("")
    print("NEXT GATE: stitch physical outputs and evaluate full 20-image biological fidelity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
