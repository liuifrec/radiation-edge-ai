"""Replay the three frozen DNAi fields on the physical KL720 and verify their reconstruction."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from uuid import uuid4

from verify_dnai_hardware_baseline import (
    FROZEN, MANIFEST, MASKS, MODEL_BASE, NEF, PHYSICAL_PROBABILITIES, SAMPLES, V3, WINDOWS,
    checked, digest, rows, verify,
)

REPO = Path(__file__).resolve().parents[2]
PANEL = Path(__file__).with_name("run_kl720_dnai_panel.py")
MODEL = MODEL_BASE.parent
ARCHIVE = Path("scratch/dnai_kl720_v3_replay_20261118T074048Z_3908b678")
ARCHIVE_RECEIPT = "a52c7aa80ef3bdf18a2c6ffd92febad15e67a914493b296878eb4b47414c399c"
NEF_FILE = V3 / "nef/unet_mobileone_s1_512x512_v3.nef"
MODEL_FILES = {
    MODEL / "onnx/unet_mobileone_s1_512x512_opset11.onnx":
        "bb8244208328d54d5b971b254abc1088f96691b286a607d28155fa606b158008",
    MODEL_BASE / "unet_mobileone_s1_512x512_opset11_optimized.onnx":
        "a901d1b309a9a0e5026febd5070252787e4b110a2a7d2828e16a19364d6094d0",
    V3 / "unet_mobileone_s1_512x512_int8.bie":
        "01f902e283b565b712cbecbf22c71f19ae6f9ba826d2c6be87e8976cd1bc9863",
    MODEL / "calibration_512_v2/manifest.json":
        "08fe74d646342c36709f44a116b10379c0c073e16ecffc67fd66fa35a1634a79",
    NEF_FILE: NEF,
}


def driver_inventory():
    """Capture Windows' signed-driver records for the KL720 product."""
    cmd = ("$ErrorActionPreference='Stop'; "
           "Get-CimInstance Win32_PnPSignedDriver -Filter \"DeviceID LIKE 'USB%VID_3231%PID_0720%'\" "
           "| Select-Object DeviceName,DeviceID,DriverVersion,DriverProviderName,InfName "
           "| ConvertTo-Json -Compress")
    try:
        p = subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd],
                           capture_output=True, text=True, errors="replace", timeout=30, check=True)
        return json.loads(p.stdout) if p.stdout.strip() else []
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)}


def preflight(root):
    for relative, checksum in {**MODEL_FILES, **FROZEN}.items():
        checked(root / relative, checksum)
    inventory = {(r["sample_id"], int(r["window_index"])): r for r in rows(root / WINDOWS)}
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    tensors = [r for r in manifest["records"] if r["sample_id"] in SAMPLES]
    keys = [(r["sample_id"], r["window_index"]) for r in tensors]
    if keys != [(s, i) for s in SAMPLES for i in range(9)]:
        raise ValueError("Unexpected field or window ordering in the baseline")
    for key, record in zip(keys, tensors):
        checked((root / MANIFEST).parent / record["tensor"], inventory[key]["input_sha256"])
        checked(root / V3 / "hardware_panel/physical_outputs" / Path(record["tensor"]).name,
                inventory[key]["physical_sha256"])
    masks = {(r["sample_id"], r["artifact"]): r["sha256"] for r in rows(root / MASKS)}
    for sample in SAMPLES:
        field = root / V3 / "hardware_biological_fidelity" / sample
        for name in ("floating512", "physical_kl720"):
            checked(field / f"{name}_segmentation_class_ids.png", masks[sample, name + "_segmentation"])
        checked(field / "physical_kl720_stitched_probabilities.npz", PHYSICAL_PROBABILITIES[sample])
    return {k: manifest[k] for k in ("tile", "full_image_shape", "overlap", "stride", "blend",
                                     "pixel_size_um", "clarity", "reverse_channel")}


def offline_receipt(root, include_archive):
    """Read registered inputs and references; never import PLUS or open a device."""
    result = {
        "schema": "dnai_kl720_v3_offline_verification_v1", "usb_accessed": False,
        "verifier_sha256": digest(Path(__file__)),
        "utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing": preflight(root), "verified_input_windows": 27,
        "verified_reference_windows": 27, "verified_reference_masks": 6,
        "verified_reference_probability_maps": 3,
        "verified_artifacts": {p.as_posix(): h for p, h in {**MODEL_FILES, **FROZEN}.items()},
    }
    if include_archive:
        archive = root / ARCHIVE
        receipt = json.loads(checked(archive / "hardware_receipt.json", ARCHIVE_RECEIPT).read_bytes())
        panel = archive / "panel"
        comparison = json.loads(checked(panel / "comparison/comparison.json", receipt["comparison_sha256"]).read_bytes())
        summary = receipt["hardware_summary"]
        checked(panel / "summary.json", comparison["run_summary_sha256"])
        table = checked(panel / "per_window.csv", summary["per_window_sha256"])
        captured = rows(table)
        expected = [r for r in json.loads((root / MANIFEST).read_bytes())["records"] if r["sample_id"] in SAMPLES]
        if receipt["status"] != "REPRODUCED" or comparison["passed"] is not True or len(captured) != len(expected):
            raise ValueError("Incomplete archived v3 physical replay")
        for row, record in zip(captured, expected):
            stem = Path(record["tensor"]).stem
            if (row["sample_id"], int(row["window_index"]), row["status"]) != (
                record["sample_id"], record["window_index"], "ran"
            ):
                raise ValueError("Archived v3 replay order differs from the verified manifest")
            checked(panel / "raw_receive_buffers" / (stem + ".bin"), row["raw_receive_sha256"])
            checked(panel / "physical_outputs" / (stem + ".npy"), row["output_sha256"])
            checked(root / V3 / "hardware_panel/physical_outputs" / (stem + ".npy"), row["output_sha256"])
        result["historical_replay"] = {
            "receipt_sha256": ARCHIVE_RECEIPT, "comparison_sha256": receipt["comparison_sha256"],
            "per_window_sha256": summary["per_window_sha256"], "model_nef_sha256": summary["nef_sha256"],
            "model_id": summary["model_id"], "model_version": summary["model_version"],
            "plus_version": summary["runtime"]["plus_version"],
            "firmware_version": summary["system_info"]["firmware_version"],
            "raw_receive_buffers_rehashed": len(captured), "byte_identical_reference_windows": len(captured),
            "archived_mean_send_receive_ms": summary["mean_hardware_send_receive_ms"],
            "archived_fields_vs_fp512": receipt["fields"],
        }
    return result


def run(root, plus_python, port):
    root = root.resolve(strict=True)
    plus_python = plus_python.resolve(strict=True)
    if root == REPO or REPO in root.parents:
        raise ValueError("Choose an external data root outside the repository")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    # Detect missing science packages before sending anything to the device.
    from radiation_edge_ai.dna_fiber import evaluate_kl720_hardware_biological_fidelity_512  # noqa: F401
    from radiation_edge_ai.dnai_window_origin import REPLAY_ATOL  # noqa: F401
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = root / "scratch" / f"dnai_kl720_v3_replay_{stamp}_{uuid4().hex[:8]}"
    out.mkdir(exist_ok=False)
    panel = out / "panel"
    receipt = out / "hardware_receipt.json"
    record = {
        "schema": "dnai_kl720_v3_hardware_receipt_v1",
        "started_utc": datetime.now(timezone.utc).isoformat(), "samples": SAMPLES,
        "upstream_dnai_commit": "fcf20c7d6eb385675ff7d07da4fdf471589ce0cf",
        "model_files_sha256": {p.as_posix(): h for p, h in MODEL_FILES.items()},
        "data_root": str(root), "science_python": sys.executable,
        "science_packages": {name: metadata.version(name) for name in
                             ("dnafiber", "numpy", "torch", "scipy", "scikit-image")},
        "scripts_sha256": {p.name: digest(p) for p in (
            Path(__file__), PANEL, PANEL.with_name("run_kl720_dnai_nef_smoke.py"),
            PANEL.with_name("verify_dnai_hardware_baseline.py"),
            REPO / "src/radiation_edge_ai/dna_fiber/evaluate_bie_biological_fidelity_512.py",
            REPO / "src/radiation_edge_ai/dna_fiber/evaluate_kl720_hardware_biological_fidelity_512.py")},
    }
    try:
        record["frozen_preprocessing"] = preflight(root)
        record["windows_driver_records"] = driver_inventory()
        command = [str(plus_python), "-B", "-u", str(PANEL), "--baseline-replay", "--nef", str(root / NEF_FILE),
                   "--validation-manifest", str(root / MANIFEST), "--output-dir", str(panel),
                   "--sample-ids", ",".join(SAMPLES)]
        if port is not None:
            command += ["--port", str(port)]
        record["device_command"] = command
        print(f"Running the historical NEF on the physical KL720. Evidence: {out}", flush=True)
        with (out / "hardware.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Physical runner exited with {result.returncode}. See {out / 'hardware.log'}")
        summary = json.loads((panel / "summary.json").read_text(encoding="utf-8"))
        record["hardware_summary"] = summary
        checked(panel / "per_window.csv", summary["per_window_sha256"])
        timings = rows(panel / "per_window.csv")
        for row in timings:
            checked(Path(row["raw_receive_path"]), row["raw_receive_sha256"])
            checked(Path(row["physical_output"]), row["output_sha256"])
        print(f"Received {summary['n_ran']} windows. Reconstructing and comparing the three fields.", flush=True)
        comparison = verify(root, panel)
        for timing, window in zip(timings, comparison["windows"]):
            if timing["input_sha256"] != window["input_sha256"]:
                raise ValueError("The input consumed by PLUS differs from the frozen input")
        record["comparison_path"] = str(panel / "comparison/comparison.json")
        record["comparison_sha256"] = digest(Path(record["comparison_path"]))
        record["fields"] = {f["sample_id"]: f["measured_metrics"] for f in comparison["fields"]}
        record["physical_logits_byte_identical_windows"] = sum(
            w["frozen_physical_sha256"] == w["new_physical_sha256"] for w in comparison["windows"])
        record["status"] = "REPRODUCED" if comparison["passed"] else "REPLAY_MISMATCH"
        return 0 if comparison["passed"] else 1
    except Exception as exc:
        record.update(status="EXECUTION_FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["completed_utc"] = datetime.now(timezone.utc).isoformat()
        receipt.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"{record['status']} | {receipt}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true", help="Read and hash model, inputs and references")
    mode.add_argument("--verify-archive", action="store_true", help="Also verify the historical 27-window replay")
    mode.add_argument("--execute", action="store_true", help="Run the three frozen fields on a physical KL720")
    parser.add_argument("--plus-python", type=Path, default=os.environ.get("RADEDGE_KL720_PYTHON"))
    parser.add_argument("--port", type=int, help="Select a specific KL720 if multiple are visible")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(REPO / "src"))
    if args.execute:
        if args.plus_python is None:
            parser.error("Supply --plus-python or set RADEDGE_KL720_PYTHON for physical execution")
        return run(args.data_root, args.plus_python, args.port)
    print(json.dumps(offline_receipt(args.data_root.resolve(strict=True), args.verify_archive),
                     indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
