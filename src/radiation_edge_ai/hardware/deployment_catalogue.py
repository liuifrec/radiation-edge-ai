"""Read-only cross-assay catalogue for deployment selection."""

import argparse
import json
from pathlib import Path

from radiation_edge_ai.control import sha256_file
from radiation_edge_ai.hardware.pretrained_models import model_record


NASA = "nasa-53bp1-r1-v2"
DNAI = "dnai-fiber-v3"
SPOT = "spotiflow/general@full_int16_transport"
MOBILENET = "nasa-mobilenetv2"
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
    MOBILENET: "Female-source development fold, two sources and 24 sample means",
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
    rows[MOBILENET].update(
        architecture="mobilenet_v2", fold="fold_female_role",
        endpoint="53BP1 burden aggregated by sample and matched exposed-minus-sham contrasts",
        scientific="Previously examined source-held-out development cohort",
        engineering={"training_fp32_parity": "passed", "standard_int8_ptq": "passed",
                     "bie_fidelity": "failed_mean_absolute_bag_shift", "engineering_gates_passed": 7,
                     "engineering_gates_total": 8, "nef_compilation": "blocked_by_bie_fidelity",
                     "physical_kl720": "not_run"},
        artifacts={
            "checkpoint": ("models/nasa_bps_53bp1/multimodel_v1_20261117T053459Z_b02f5836/"
                           "mobilenet_v2/fold_female_role/model.pt",
                           "bb00a25e2db05334e113852760414adafa22f95614e4c43f92b6607e690bd543"),
            "onnx": ("models/nasa_bps_53bp1/multimodel_v1_onnx_20261118T002659Z_47b5d92a/"
                     "mobilenet_v2/fold_female_role/mobilenet_v2_fold_female_role_fp32_opset11.onnx",
                     "2f18b2330ca3289ab422e1ff47b526b4bfa6a46fc4bd7354346609715f575e45"),
            "optimized_onnx": ("scratch/nasa_trained_graph_screen_20261118T020501Z_06255b35/"
                               "vendor/mobilenet_v2_fold_female_role/optimized.onnx",
                               "75824e1ed4f8520d90a942c1351926b922db9b4284b8b6d7cd5663428e73c074"),
            "bie": ("scratch/nasa_mobilenet_female_int8_20261126T234932Z_a7c472a7/pilot/quantized_model.bie",
                    "2102c400d696bc385e23d2d9b978b779e4c95b82661272e6873ac1eb57607e26"),
            "bie_comparison": ("scratch/nasa_mobilenet_female_bie_readout_20261127T000412Z_e601260b/"
                               "comparison/comparison.json",
                               "90e40ad338df96d13ab0b32e3cb8db936509ca7c2d1256cac74e893407a7b6cf"),
        },
    )
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
