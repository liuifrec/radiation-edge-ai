"""Run a registered fixed-shape Spotiflow graph and recover its subpixel spots on the host."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from io import BytesIO
from pathlib import Path

import numpy as np

from radiation_edge_ai import _kl720_worker
from radiation_edge_ai.audit_utils import sha256_file
from radiation_edge_ai.hardware import pretrained_models, spotiflow, spotiflow_transport
from radiation_edge_ai.hardware.spotiflow import target_assignment


def read_plane(path):
    data = path.read_bytes()
    if path.suffix.lower() == ".npy":
        array = np.load(BytesIO(data), allow_pickle=False)
    elif path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        array = tifffile.imread(BytesIO(data))
    else:
        raise ValueError("Expected a 2D .npy or TIFF image")
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Expected a finite 2D image")
    return array, hashlib.sha256(data).hexdigest()


def run(args):
    import yaml

    if (args.mask is None) != (args.target_label is None):
        raise ValueError("Supply both the instance mask and its positive target label")
    record = pretrained_models.model_record(args.model, variant=args.variant)
    if record["adapter"] != spotiflow.__name__ or version("spotiflow") != record["provenance"]["software_version"]:
        raise ValueError("Model or installed release is incompatible with this adapter")
    threshold_file = pretrained_models.verified_artifact(record, "thresholds.yaml", args.data_root)
    if yaml.safe_load(threshold_file.read_bytes())["prob_thresh_best"] != record["threshold"]:
        raise ValueError("Registered threshold differs from the released checkpoint")
    runtime = {
        "python": platform.python_version(), "numpy": np.__version__,
        "spotiflow": version("spotiflow"), "torch": version("torch"),
        "adapter_sha256": sha256_file(Path(spotiflow.__file__)),
        "launcher_sha256": sha256_file(Path(__file__)),
    }
    if args.backend == "kl720":
        if record["variant"] != spotiflow_transport.VARIANT:
            raise ValueError("Select the physically qualified full_int16_transport variant for USB")
        transport_panel = spotiflow_transport.qualified_physical_panel(record, args.data_root)
        model_path = pretrained_models.qualified_nef(record, args.data_root)
    elif args.backend == "pytorch":
        import torch
        from spotiflow.model import Spotiflow

        model_path = pretrained_models.verified_artifact(record, "best.pt", args.data_root)
        config_path = pretrained_models.verified_artifact(record, "config.yaml", args.data_root)
        if any(path != model_path.parent / name for path, name in
               ((model_path, "best.pt"), (threshold_file, "thresholds.yaml"), (config_path, "config.yaml"))):
            raise ValueError("Registered release files must form the exact upstream checkpoint folder")
        model = Spotiflow.from_folder(model_path.parent, which="best", map_location="cpu").eval()
        core = spotiflow.neural_core(model)
        runtime.update(provider="pytorch_cpu", config_sha256=sha256_file(config_path))
    else:
        import onnxruntime as ort
        model_path = pretrained_models.verified_artifact(record, "onnx", args.data_root)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        runtime.update(onnxruntime=ort.__version__, provider=session.get_providers(),
                       graph_optimization=str(opts.graph_optimization_level))
    if not args.out.resolve().is_relative_to(args.data_root.resolve()):
        raise ValueError("Save inference evidence under the external data root")
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.perf_counter()
    image, image_sha = read_plane(args.image)
    mask, mask_sha = read_plane(args.mask) if args.mask else (None, None)
    if mask is not None and (mask.shape != image.shape or mask.dtype.kind not in "iu" or args.target_label <= 0):
        raise ValueError("Expected a registered integer label mask at the same geometry")
    tensor = spotiflow.prepare_input(image)
    t1 = time.perf_counter()
    worker = None
    if args.backend == "onnx":
        names = record.get("outputs", ["spot_heads"])
        parts = session.run(names, {"input": tensor})
        heads = parts[0] if len(parts) == 1 else spotiflow.join_heads(*parts)
    elif args.backend == "pytorch":
        with torch.inference_mode():
            heads = core(torch.from_numpy(tensor)).numpy()
    else:
        input_file = args.out / "input.npy"
        output_file = args.out / "heads.npy"
        np.save(input_file, tensor, allow_pickle=False)
        manifest = args.out / "usb_receipt.json"
        command = [str(args.plus_python), "-B", str(Path(_kl720_worker.__file__).resolve()),
                   "--model", str(model_path), "--input", str(input_file), "--output", str(output_file),
                   "--manifest", str(manifest), "--raw-receive", str(args.out / "usb_receive.bin")]
        if args.port is not None:
            command.extend(("--port", str(args.port)))
        expected_kn = spotiflow_transport.registered_kn(record, args.data_root)
        command.extend(("--expected-kn", hex(expected_kn), "--required-plus-version", transport_panel["plus_version"],
                        "--required-plus-dll-sha256", transport_panel["plus_dll_sha256"],
                        "--expected-firmware-version", transport_panel["system_info"]["firmware_version"]))
        if len(record.get("outputs", ["spot_heads"])) == 2:
            command.extend(("--join-output-channels", "1", "3"))
        with (args.out / "usb.log").open("xb") as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
        worker = json.loads(manifest.read_bytes())
        if (worker["status"] != "complete" or worker["backend"] != "kl720"
                or worker["input"]["sha256"] != sha256_file(input_file)
                or worker["model"]["sha256"] != record["artifacts"]["nef"]["sha256"]
                or worker["output"]["sha256"] != sha256_file(output_file)):
            raise ValueError("USB receipt does not bind the requested model and output")
        heads = np.load(output_file, allow_pickle=False)
    t2 = time.perf_counter()
    decoded = spotiflow.decode_heads(heads, image.shape, record["threshold"])
    points = decoded["points"]
    target_count = None
    if mask is not None:
        _, keep, _ = target_assignment(mask, args.target_label, points, len(points))
        target_count = sum(keep)
    t3 = time.perf_counter()
    if args.backend != "kl720":
        np.save(args.out / "heads.npy", heads, allow_pickle=False)
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(), "backend": args.backend,
        "model_id": args.model, "variant": record["variant"], "runtime": runtime,
        "threshold": record["threshold"], "thresholds_sha256": record["artifacts"]["thresholds.yaml"]["sha256"],
        "artifact_sha256": sha256_file(model_path), "weights_sha256": record["artifacts"]["best.pt"]["sha256"],
        "image": str(args.image.resolve()), "image_sha256": image_sha, "image_hw": list(image.shape),
        "tensor_bytes_sha256": hashlib.sha256(tensor.tobytes()).hexdigest(),
        "mask_sha256": mask_sha, "target_label": args.target_label, "target_count": target_count,
        "crop_count": len(points), "points_yx": points.tolist(), "confidence": decoded["confidence"].tolist(),
        "heads_sha256": sha256_file(args.out / "heads.npy"), "engineering": record["engineering"],
        "scientific": record["scientific"], "usb_worker": worker, "timings_seconds": {
            "read_and_prepare": t1 - t0, "backend": t2 - t1, "decode_and_assign": t3 - t2,
            "input_to_count": t3 - t0,
            "scope": "CPU model already loaded. USB backend includes worker startup, NEF load, file exchange and inference.",
        },
    }
    dest = args.out / "receipt.json"
    dest.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(dest)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", help="Registered graph boundary for the chosen checkpoint")
    parser.add_argument("--backend", required=True, choices=("pytorch", "onnx", "kl720"))
    for key in ("image", "out", "data-root"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--target-label", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--plus-python", type=Path, default=Path.home() / "venvs/kneron720/Scripts/python.exe")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
