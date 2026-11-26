"""Read-only cross-assay catalogue for deployment selection."""

import argparse
import json
from pathlib import Path

from radiation_edge_ai.control import sha256_file
from radiation_edge_ai.hardware.pretrained_models import model_record


NASA = "nasa-53bp1-r1-v2"
DNAI = "dnai-fiber-v3"
SPOT = "spotiflow/general@full_int16_transport"
RUN_BACKENDS = {NASA: ("kl720",), DNAI: ("kl720",), SPOT: ("pytorch", "onnx", "kl720")}

NASA_ROOT = "models/nasa_bps_53bp1/r1_v2_deployment/kneron_nef/"
DNAI_ROOT = "models/dnai/unet_mobileone_s1/kneron/int8_512_v3_pct100/"
REFERENCES = {
    NASA: {
        "endpoint": "Latent 53BP1 burden per nucleus, aggregated by sample",
        "scope": "Frozen 2,058-nucleus physical replay",
        "scientific": "Passed seven original fidelity criteria for the documented cohort",
        "runner": "scripts/windows/run_nasa_kl720.py",
        "artifacts": {
            "nef": (NASA_ROOT + "r1_countnet_v2_final_kl720.nef",
                    "d04f02855a8a82ae7cb5ff48af91eee69c341142d1097469890385bd23ee097b"),
            "physical_smoke_freeze": (NASA_ROOT + "physical_smoke_hardware/physical_smoke_pass_freeze.json",
                                      "224b754235ebcd22eb053ca1f279043f3b74e9e3b461ed1f8dad26beddfa679d"),
        },
    },
    DNAI: {
        "endpoint": "Two-colour fiber segmentation, counts and tract morphology",
        "scope": "Frozen three-field batch, nine windows per field",
        "scientific": "FP512 agreement characterized on three fields; population fidelity remains open",
        "runner": "scripts/windows/run_dnai_hardware_baseline.py",
        "artifacts": {
            "nef": (DNAI_ROOT + "nef/unet_mobileone_s1_512x512_v3.nef",
                    "4b3dfec9a61c99e186dd4b8482fa5b06e6a4958f325ed4a0db0546f1dcab2bfc"),
            "physical_replay": ("scratch/dnai_kl720_v3_replay_20261118T074048Z_3908b678/hardware_receipt.json",
                                "a52c7aa80ef3bdf18a2c6ffd92febad15e67a914493b296878eb4b47414c399c"),
        },
    },
}
CANDIDATES = {
    "nasa-resnet18-female": "INT8 BIE failed two frozen bag-difference limits",
    "nasa-mobilenetv2": "Development candidate; physical qualification has not been established",
    "dnai-fiber-v4": "Physical characterization recovered one three-grader-supported fiber and lost two",
    "dnai-fiber-v5": "Failed the frozen directional instance-retention criterion",
    "perilipin2": "Full model blocked at KL720 compilation",
    "gamma-h2ax": "Training diagnostics; no released physical assay",
    "micronuclei": "Training diagnostics; no released physical assay",
}


def catalogue():
    """Return public engineering and assay scope independently of local availability."""
    rows = {key: {**value, "track": "physical_reference"} for key, value in REFERENCES.items()}
    for name in ("spotiflow/general", "spotiflow/synth_complex"):
        base = model_record(name)
        for variant in (base["variant"], *sorted(base.get("variants", {}))):
            record = model_record(name, variant=variant)
            key = name + "@" + variant
            stage = record["engineering"]["physical_kl720"]
            qualified = stage["status"] == "passed"
            if (key == SPOT) != qualified:
                raise ValueError("Review platform selection against the changed Spotiflow qualification")
            names = [n for n in ("onnx", "nef", stage.get("evidence")) if n in record["artifacts"]]
            rows[key] = {
                "track": "physical_reference" if qualified else "experimental",
                "endpoint": record["contract"]["endpoint"], "scope": record["contract"]["tiling"],
                "scientific": record["scientific"], "physical_stage": stage,
                "engineering": {k: v["status"] for k, v in record["engineering"].items()},
                "runner": "scripts/windows/run_registered_spotiflow.py" if qualified else None,
                "artifacts": {n: (record["artifacts"][n]["path"], record["artifacts"][n]["sha256"]) for n in names},
            }
    for key, reason in CANDIDATES.items():
        rows[key] = {"track": "experimental", "scope": reason, "artifacts": {}}
    return rows


def report(model=None, data_root=None, verify=False):
    rows = catalogue()
    if model is not None and model not in rows:
        raise ValueError(f"Unknown deployment model: {model}")
    if verify and data_root is None:
        raise ValueError("Hash verification requires a data root")
    result = []
    for key, record in rows.items():
        if model is not None and key != model:
            continue
        artifacts = {}
        for name, (relative, digest) in record["artifacts"].items():
            state = "unchecked"
            if data_root is not None:
                root = Path(data_root).resolve()
                path = (root / relative).resolve()
                if not path.is_relative_to(root):
                    raise ValueError(f"Artifact outside the data root: {relative}")
                if not path.is_file():
                    state = "missing"
                elif verify:
                    state = "verified" if sha256_file(path) == digest else "checksum_mismatch"
                else:
                    state = "present_unverified"
            artifacts[name] = {"path": relative, "sha256": digest, "state": state}
        result.append({**record, "id": key, "artifacts": artifacts,
                       "desktop_backends": list(RUN_BACKENDS.get(key, ()))})
    return {"schema": "deployment_catalogue_v1", "models": result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Select one model and graph boundary")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--verify", action="store_true", help="Rehash the listed external artifacts")
    args = parser.parse_args(argv)
    try:
        result = report(args.model, args.data_root, args.verify)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
    artifacts = [a for r in result["models"] for a in r["artifacts"].values()]
    return int(args.verify and (not artifacts or any(a["state"] != "verified" for a in artifacts)))


if __name__ == "__main__":
    raise SystemExit(main())
