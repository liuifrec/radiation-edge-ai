"""Minimal KL720 hardware discovery check.

Run this with a Python interpreter that already has the Kneron PLUS Python
package (``kp``) installed. It intentionally avoids importing the rest of the
scientific stack so the known-good hardware environment can stay isolated.
"""

from __future__ import annotations

import sys


def _safe_attr(obj, name, default="unknown"):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def main() -> int:
    print("Radiation Edge AI - KL720 Hardware Check")
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")

    try:
        import kp
    except Exception as exc:
        print("kp import: FAIL")
        print(f"Reason: {exc}")
        return 2

    print("kp import: PASS")
    print(f"kp module: {_safe_attr(kp, '__file__')}")

    try:
        descriptors = kp.core.scan_devices()
    except Exception as exc:
        print("device scan: FAIL")
        print(f"Reason: {exc}")
        return 3

    count = int(_safe_attr(descriptors, "device_descriptor_number", 0))
    print(f"device scan: PASS ({count} device(s))")

    device_list = _safe_attr(descriptors, "device_descriptor_list", []) or []
    for index, device in enumerate(device_list):
        print(f"\nDevice {index}")
        for field in (
            "usb_port_id",
            "product_id",
            "kn_number",
            "firmware",
            "firmware_version",
            "connectable",
        ):
            value = _safe_attr(device, field, None)
            if value is not None:
                print(f"  {field}: {value}")

    if count < 1:
        print("\nKL720 READY: NO - no Kneron device detected")
        return 4

    print("\nKL720 READY: YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
