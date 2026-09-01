"""Connect to a Kneron device and upload a NEF model as a smoke test.

Run this with the known-good Kneron PLUS Python interpreter, not necessarily the
project development virtual environment.  The command deliberately stops after
model upload and descriptor inspection; image preprocessing and inference are
kept out of this hardware gate so failures remain easy to localize.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

KL720_PRODUCT_ID = 0x720


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _format_metadata(metadata: Any) -> list[str]:
    if metadata is None:
        return []
    if isinstance(metadata, dict):
        return [f"  {key}: {value}" for key, value in sorted(metadata.items())]
    return [f"  {metadata}"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect to KL720 and verify that a NEF model can be uploaded."
    )
    parser.add_argument("nef", type=Path, help="Path to a KL720-compatible .nef file")
    parser.add_argument(
        "-p",
        "--port-id",
        type=int,
        default=None,
        help="USB port ID. If omitted, use the first detected KL720.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=5000,
        help="USB communication timeout in milliseconds (default: 5000).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    nef_path = args.nef.expanduser().resolve()

    print("Radiation Edge AI - KL720 NEF Model Probe")
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    print(f"NEF: {nef_path}")

    if not nef_path.is_file():
        print("NEF file: FAIL - file does not exist")
        return 2

    try:
        import kp
    except Exception as exc:
        print("kp import: FAIL")
        print(f"Reason: {exc}")
        return 3

    print("kp import: PASS")

    try:
        descriptors = kp.core.scan_devices()
    except Exception as exc:
        print("device scan: FAIL")
        print(f"Reason: {exc}")
        return 4

    devices = list(_safe_attr(descriptors, "device_descriptor_list", []) or [])
    kl720_devices = [
        device for device in devices if int(_safe_attr(device, "product_id", -1)) == KL720_PRODUCT_ID
    ]

    if not kl720_devices:
        print("KL720 detection: FAIL - no product_id 0x720 device found")
        return 5

    if args.port_id is None:
        selected = kl720_devices[0]
        port_id = int(_safe_attr(selected, "usb_port_id"))
    else:
        matches = [
            device
            for device in kl720_devices
            if int(_safe_attr(device, "usb_port_id", -1)) == args.port_id
        ]
        if not matches:
            available = ", ".join(
                str(_safe_attr(device, "usb_port_id", "?")) for device in kl720_devices
            )
            print(
                f"KL720 selection: FAIL - port {args.port_id} not found "
                f"(available KL720 ports: {available})"
            )
            return 6
        selected = matches[0]
        port_id = args.port_id

    print("KL720 detection: PASS")
    print(f"  usb_port_id: {port_id}")
    print(f"  product_id: {_safe_attr(selected, 'product_id')} (0x720)")
    firmware = _safe_attr(selected, "firmware", None)
    if firmware is not None:
        print(f"  firmware: {firmware}")

    try:
        print("[Connect Device]")
        device_group = kp.core.connect_devices(usb_port_ids=[port_id])
        print(" - Success")
    except Exception as exc:
        print(" - FAIL")
        print(f"Reason: {exc}")
        return 7

    try:
        print("[Set Device Timeout]")
        kp.core.set_timeout(device_group=device_group, milliseconds=args.timeout_ms)
        print(" - Success")
    except Exception as exc:
        print(" - FAIL")
        print(f"Reason: {exc}")
        return 8

    try:
        print("[Upload Model]")
        descriptor = kp.core.load_model_from_file(
            device_group=device_group,
            file_path=str(nef_path),
        )
        print(" - Success")
    except Exception as exc:
        print(" - FAIL")
        print(f"Reason: {exc}")
        return 9

    print("\n[NEF Descriptor]")
    target_chip = _safe_attr(descriptor, "target_chip", None)
    if target_chip is not None:
        print(f"target_chip: {target_chip}")
    crc = _safe_attr(descriptor, "crc", None)
    if crc is not None:
        print(f"crc: {crc}")

    metadata = _safe_attr(descriptor, "metadata", None)
    metadata_lines = _format_metadata(metadata)
    if metadata_lines:
        print("metadata:")
        for line in metadata_lines:
            print(line)

    models = list(_safe_attr(descriptor, "models", []) or [])
    print(f"models: {len(models)}")
    for index, model in enumerate(models):
        print(f"  model[{index}]")
        for field in ("id", "max_raw_out_size"):
            value = _safe_attr(model, field, None)
            if value is not None:
                print(f"    {field}: {value}")

        input_nodes = _safe_attr(model, "input_nodes", None)
        if input_nodes is not None:
            try:
                print(f"    input_nodes: {len(input_nodes)}")
            except TypeError:
                pass
        output_nodes = _safe_attr(model, "output_nodes", None)
        if output_nodes is not None:
            try:
                print(f"    output_nodes: {len(output_nodes)}")
            except TypeError:
                pass

    if not models:
        print("\nKL720 MODEL READY: NO - NEF descriptor contains no models")
        return 10

    print("\nKL720 MODEL READY: YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
