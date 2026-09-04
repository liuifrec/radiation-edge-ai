"""Inspect the known-good Windows Kneron PLUS runtime before physical DNAi inference.

This probe is intentionally read-mostly. It imports the installed ``kp`` package,
scans attached devices, prints signatures for the core/inference APIs that are
actually available in the user's pinned environment, and searches the local
Kneron example directory for generic-data/image inference examples.

It does not upload a model or run inference. The goal is to remove API-guessing
before wiring the verified DNAi NEF to the physical KL720.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import kp


def safe_signature(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "<signature unavailable>"


def iter_public_callables(module):
    for name in sorted(dir(module)):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if callable(obj):
            yield name, obj


def print_matching_callables(module, label: str, keywords: tuple[str, ...]) -> None:
    print(f"[{label}]")
    found = 0
    for name, obj in iter_public_callables(module):
        lower = name.lower()
        if any(key in lower for key in keywords):
            print(f"  {name}{safe_signature(obj)}")
            found += 1
    if not found:
        print("  <none>")
    print("")


def scan_devices() -> None:
    print("[Device scan]")
    try:
        scan = kp.core.scan_devices()
    except Exception as exc:  # runtime-specific exception classes vary
        print(f"  scan_devices: ERROR: {type(exc).__name__}: {exc}")
        print("")
        return

    descriptors = getattr(scan, "device_descriptor_list", None)
    if descriptors is None:
        try:
            descriptors = list(scan)
        except TypeError:
            descriptors = []

    print(f"  devices: {len(descriptors)}")
    for i, desc in enumerate(descriptors):
        print(f"  device[{i}]")
        for attr in ("usb_port_id", "product_id", "firmware", "firmware_version"):
            if hasattr(desc, attr):
                print(f"    {attr}: {getattr(desc, attr)}")
        # Deliberately do not print kn_number / serial identifiers.
    print("")


def search_examples(root: Path) -> None:
    print("[Local Kneron example search]")
    print(f"  root: {root}")
    if not root.is_dir():
        print("  example root not found")
        print("")
        return

    needles = (
        "generic_data_inference",
        "generic_image_inference",
        "load_model_from_file",
        "inference_send",
        "inference_receive",
    )
    hits: list[tuple[Path, int, str]] = []
    for path in sorted(root.rglob("*.py")):
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, start=1):
            lower = line.lower()
            if any(needle in lower for needle in needles):
                hits.append((path, lineno, line.strip()))

    if not hits:
        print("  no matching example lines found")
    else:
        for path, lineno, text in hits[:120]:
            try:
                shown = path.relative_to(root)
            except ValueError:
                shown = path
            print(f"  {shown}:{lineno}: {text}")
        if len(hits) > 120:
            print(f"  ... {len(hits) - 120} additional matching lines omitted")
    print("")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--examples-root",
        default=r"C:\Users\yul03\kneron_plus\python\example",
        help="Installed Kneron PLUS Python examples directory",
    )
    args = parser.parse_args()

    print("Radiation Edge AI - Windows KL720 runtime API probe")
    print(f"kp module: {Path(kp.__file__).resolve()}")
    print(f"kp version: {getattr(kp, '__version__', '<not exposed>')}")
    print("")

    scan_devices()

    print_matching_callables(
        kp.core,
        "kp.core relevant callables",
        ("scan", "connect", "model", "timeout", "device"),
    )
    print_matching_callables(
        kp.inference,
        "kp.inference relevant callables",
        ("generic", "data", "image", "send", "receive", "node", "output"),
    )

    for module_name in ("GenericDataInferenceDescriptor", "GenericRawResultHeader"):
        obj = getattr(kp, module_name, None)
        if obj is not None:
            print(f"[{module_name}]")
            print(f"  {obj}")
            print(f"  signature: {safe_signature(obj)}")
            print("")

    search_examples(Path(args.examples_root))

    print("KL720 RUNTIME API PROBE COMPLETE: YES")
    print("NEXT: use the discovered generic-data API/signatures to build a one-window physical NEF smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
