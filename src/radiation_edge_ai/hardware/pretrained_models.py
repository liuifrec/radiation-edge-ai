"""Inspect pretrained model qualifications and verify the registered external artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path, PureWindowsPath

from radiation_edge_ai.control import load_json_object as read_json_object
from radiation_edge_ai.control import sha256_file

CATALOG = Path(__file__).with_suffix(".json")
BIE_PREREQUISITES = ("native", "onnx_parity", "optimized_fp32_parity", "kl720_ip",
                     "bie_generation", "bie_fidelity")
NEF_PREREQUISITES = (*BIE_PREREQUISITES, "nef_compilation", "nef_simulator_parity")

# Receipt fields must identify the artifacts for this exact graph boundary.
STAGE_BINDINGS = {
    "onnx_parity": {"graph_sha256": "onnx", "panel_sha256": "panel"},
    "optimized_fp32_parity": {
        "original_onnx_sha256": "onnx", "optimized_onnx_sha256": "optimized_onnx",
        "panel_sha256": "panel", "native_manifest_sha256": "onnx_parity_receipt",
    },
    "kl720_ip": {"original_onnx_sha256": "onnx", "optimized_onnx_sha256": "optimized_onnx"},
    "bie_generation": {
        "original_onnx_sha256": "onnx", "optimized_onnx_sha256": "optimized_onnx",
        "bie_sha256": "bie", "panel_sha256": "panel",
    },
    "bie_fidelity": {
        "bie_sha256": "bie", "panel_sha256": "panel", "candidate_manifest_sha256": "bie_outputs",
        "native_manifest_sha256": "onnx_parity_receipt",
    },
    "nef_compilation": {"bie_sha256": "bie", "nef_sha256": "nef", "bie_gate_sha256": "bie_gate"},
    "nef_simulator_parity": {
        "bie_sha256": "bie", "nef_sha256": "nef", "panel_sha256": "panel",
        "output_manifest_sha256": "nef_outputs", "compile_receipt_sha256": "compile_receipt",
        "bie_outputs_sha256": "bie_outputs",
    },
}


def _unique_keys(entries):
    result = {}
    for name, value in entries:
        if name in result:
            raise ValueError(f"Ambiguous duplicate catalogue key: {name}")
        result[name] = value
    return result


def _read_catalog(catalog):
    registry = json.loads(Path(catalog).read_bytes(), object_pairs_hook=_unique_keys)
    if (not isinstance(registry, dict) or type(registry.get("schema_version")) is not int
            or registry["schema_version"] != 1 or not isinstance(registry.get("families"), dict)
            or not isinstance(registry.get("models"), dict) or not registry["models"]):
        raise ValueError("Expected a version 1 pretrained model catalogue")
    locations = {}
    sections = [*registry["families"].values(), *registry["models"].values()]
    for model in registry["models"].values():
        if not isinstance(model, dict) or model.get("family") not in registry["families"]:
            raise ValueError("Unknown model family")
        variants = model.get("variants", {})
        if not isinstance(variants, dict) or model.get("default_variant") in variants:
            raise ValueError("Ambiguous model variant")
        sections.extend(variants.values())
    for section in sections:
        if not isinstance(section, dict) or not isinstance(section.get("artifacts"), dict):
            raise TypeError("Expected an artifact mapping in each catalogue section")
        for entry in section["artifacts"].values():
            location = _relative_artifact(entry).as_posix().casefold()
            identity = (entry["path"], entry["sha256"])
            if locations.setdefault(location, identity) != identity:
                raise ValueError(f"Ambiguous catalogue identity for {entry['path']}")
    return registry


def _relative_artifact(item):
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise TypeError("Expected a relative artifact path")
    name = item["path"]
    # Registry paths use portable components on both Windows and POSIX. Reject
    # drive-relative paths, alternate streams, reserved names and Windows aliases.
    parts = name.split("/")
    if (any(part in ("", ".", "..") or part.endswith((" ", ".")) or PureWindowsPath(part).is_reserved()
            for part in parts) or re.search(r'[<>:"|?*\\\x00-\x1f]', name)):
        raise ValueError(f"Invalid portable artifact path: {name!r}")
    if not isinstance(item.get("sha256"), str) or not re.fullmatch("[0-9a-f]{64}", item["sha256"]):
        raise ValueError(f"Invalid SHA256 for {name}")
    return Path(*parts)


def _validate_record(record):
    artifacts, stages = record["artifacts"], record["engineering"]
    locations = [_relative_artifact(spec).as_posix().casefold() for spec in artifacts.values()]
    if not isinstance(record.get("scientific"), dict) or not record["scientific"].get(
        "nasa_independent_biological_validation"
    ):
        raise ValueError("Missing independent biological qualification status")
    if len(set(locations)) != len(locations):
        raise ValueError("Ambiguous artifact location in model variant")
    if not isinstance(stages, dict) or set(stages) != {*NEF_PREREQUISITES, "physical_kl720"}:
        raise ValueError("Incomplete engineering stages in model variant")
    for stage, result in stages.items():
        if not isinstance(result, dict) or result.get("status") not in {"passed", "failed", "blocked", "not_run"}:
            raise ValueError(f"Unknown engineering status for {stage}")
        if (result["status"] in {"passed", "failed"} or "evidence" in result) and result.get("evidence") not in artifacts:
            raise ValueError(f"Missing registered evidence for {stage}")
    if (not isinstance(record["threshold"], (int, float)) or isinstance(record["threshold"], bool)
            or not math.isfinite(record["threshold"]) or not 0 < record["threshold"] < 1):
        raise ValueError("Invalid released spot threshold")
    if stages["physical_kl720"]["status"] == "passed" and any(
        stages[stage]["status"] != "passed" for stage in NEF_PREREQUISITES
    ):
        raise ValueError("Physical qualification requires every prerequisite")


def _record(registry, model_id, variant):
    model = registry["models"][model_id]
    family = registry["families"][model["family"]]
    shared = family["artifacts"].keys()
    if shared & model["artifacts"].keys():
        raise ValueError("Model artifacts may not replace shared provenance")
    default_variant = model.get("default_variant", "default")
    record = {**model, "id": model_id, "variant": variant or default_variant,
              "adapter": family["adapter"], "provenance": family["provenance"],
              "contract": family["contract"], "requirements": family["requirements"],
              "artifacts": {**family["artifacts"], **model["artifacts"]}}
    if variant and variant != default_variant:
        candidate = model["variants"][variant]
        if variant == "full_int16_transport":
            if model_id != "spotiflow/general" or candidate.get("inherits") != "full_int16":
                raise ValueError("Unexpected INT16 transport source")
            parent = model["variants"]["full_int16"]
            if set(candidate) != {"inherits", "artifacts", "engineering"} or (
                set(parent["artifacts"]) & set(candidate["artifacts"])
            ) or set(candidate["engineering"]) != {"nef_simulator_parity", "physical_kl720"}:
                raise ValueError("The transport variant may only add evidence for the existing binary")
            candidate = dict(parent, artifacts={**parent["artifacts"], **candidate["artifacts"]},
                             engineering={**parent["engineering"], **candidate["engineering"]})
        # Variant graph/quantization artifacts may never fall back to another
        # boundary's model. Keep only the release and shared provenance here.
        weights = {key: model["artifacts"][key] for key in family["release_artifact_names"]}
        if (shared | weights.keys()) & candidate["artifacts"].keys():
            raise ValueError("A variant may not replace shared provenance or released weights")
        record.update(engineering=candidate["engineering"], outputs=candidate["outputs"],
                      analysis_overrides=candidate.get("analysis_overrides", {}),
                      artifacts={**family["artifacts"], **weights, **candidate["artifacts"]})
    _validate_record(record)
    return record


def model_record(model_id, catalog=CATALOG, variant=None):
    return _record(_read_catalog(catalog), model_id, variant)


def _artifact_path(record, name, data_root):
    root = Path(data_root).resolve()
    relative = _relative_artifact(record["artifacts"][name])
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Expected a path inside the external data directory for {name}")
    return path


def verified_artifact(record, name, data_root):
    path = _artifact_path(record, name, data_root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing artifact {name}: {path}")
    if sha256_file(path) != record["artifacts"][name]["sha256"]:
        raise ValueError(f"Artifact checksum differs for {name}: {path}")
    return path


def _require_simulator_panel(record, data_root, gate):
    panel = read_json_object(verified_artifact(record, "panel", data_root))
    readout = [r for r in panel["rows"] if r["role"] == "readout"]
    inputs = {r["case_id"]: r["tensor_sha256"] for r in readout}
    if (len(readout) != 63 or len(inputs) != 63 or gate.get("cases") != 63
            or gate.get("stage") != "nef_simulator_parity"):
        raise ValueError("Missing complete 63-case Spotiflow simulator decision")
    measurements = {}
    for key, stage in (("nef_outputs", "nef_simulator"), ("bie_outputs", "bie_readout")):
        manifest = read_json_object(verified_artifact(record, key, data_root))
        if (manifest.get("stage") != stage or manifest.get("model") != record["checkpoint"]
                or manifest.get("bie_sha256") != gate["bie_sha256"]
                or manifest.get("panel_sha256") != gate["panel_sha256"]
                or (key == "nef_outputs" and manifest.get("nef_sha256") != gate["nef_sha256"])):
            raise ValueError(f"Wrong model, panel or stage in {key}")
        measurements[key] = manifest["rows"]
    measurements["decisions"] = gate["rows"]
    indexed = {}
    for key, rows in measurements.items():
        by_id = {r["case_id"]: r for r in rows}
        if len(rows) != len(inputs) or by_id.keys() != inputs.keys():
            raise ValueError(f"Missing, duplicate or different registered cases in {key}")
        indexed[key] = by_id
    for case, tensor_sha in inputs.items():
        decision = indexed["decisions"][case]
        bie = indexed["bie_outputs"][case]
        nef = indexed["nef_outputs"][case]
        packed = nef.get("sdk_packed_input") or {}
        if (decision.get("passed") is not True or decision.get("within_tolerance") is not True
                or decision.get("input_tensor_sha256") != tensor_sha or bie.get("input_tensor_sha256") != tensor_sha
                or decision.get("bie_heads_sha256") != bie["raw_heads_sha256"]
                or decision.get("nef_heads_sha256") != nef["raw_heads_sha256"]
                or packed.get("available") is not True or not packed.get("sha256")
                or decision.get("sdk_packed_input") != packed):
            raise ValueError(f"Simulator decision does not bind passing outputs for {case}")


def _qualified_artifact(record, data_root, stages, name):
    for stage in stages:
        result = record["engineering"][stage]
        if result["status"] != "passed" or not result.get("evidence"):
            raise ValueError(f"{record['id']} lacks a passing {stage} receipt")
    artifacts = record["artifacts"]
    for stage in stages:
        evidence = record["engineering"][stage]["evidence"]
        receipt = read_json_object(verified_artifact(record, evidence, data_root))
        if stage == "native":
            release = receipt.get("models", {}).get(record["checkpoint"], {})
            if (receipt.get("status") != "complete" or release.get("threshold") != record["threshold"]
                    or release.get("files_sha256", {}).get("best.pt") != artifacts["best.pt"]["sha256"]):
                raise ValueError("Native receipt does not bind this checkpoint")
            continue
        if receipt.get("model") != record["checkpoint"]:
            raise ValueError(f"Receipt belongs to a different checkpoint at {stage}")
        gate = receipt.get("gate_sha256", receipt.get("gates_sha256"))
        if gate != record["provenance"]["gate_lf_sha256"]:
            raise ValueError(f"Receipt references different engineering gates at {stage}")
        if (stage in ("onnx_parity", "optimized_fp32_parity", "bie_fidelity", "nef_compilation",
                      "nef_simulator_parity") and receipt.get("passed") is not True):
            raise ValueError(f"Receipt does not qualify {record['id']} for {stage}")
        if stage in ("bie_generation", "bie_fidelity", "nef_compilation", "nef_simulator_parity"):
            if receipt.get("analysis_overrides", {}) != record.get("analysis_overrides", {}):
                raise ValueError(f"Receipt references different quantization settings at {stage}")
            if "precision_ip_receipt" in artifacts and (
                receipt.get("precision_ip_sha256") != artifacts["precision_ip_receipt"]["sha256"]
            ):
                raise ValueError(f"Receipt does not bind precision-specific placement at {stage}")
        for field, artifact in STAGE_BINDINGS[stage].items():
            if receipt.get(field) != artifacts[artifact]["sha256"]:
                raise ValueError(f"{stage} receipt does not bind the registered {artifact}")
        if stage == "nef_simulator_parity":
            _require_simulator_panel(record, data_root, receipt)
        if stage == "kl720_ip" and (
            not receipt.get("completed") or receipt["classification"]["support_status"] != "supported"
        ):
            raise ValueError("Missing KL720 placement evidence")
    return verified_artifact(record, name, data_root)


def qualified_bie(record, data_root):
    """Resolve a numerically qualified fixed-point model for NEF compilation."""
    return _qualified_artifact(record, data_root, BIE_PREREQUISITES, "bie")


def qualified_nef(record, data_root):
    """Resolve a simulator-qualified NEF for nonpersistent physical inference."""
    if record["variant"] == "full_int16_transport":
        from radiation_edge_ai.hardware.spotiflow_transport import qualified_transport

        qualified_transport(record, data_root)
        return _qualified_artifact(record, data_root, NEF_PREREQUISITES[:-1], "nef")
    return _qualified_artifact(record, data_root, NEF_PREREQUISITES, "nef")


def _availability(record, data_root, verify, file_hashes):
    if data_root is None:
        return {"checked": False}
    states = {}
    for name, item in record["artifacts"].items():
        path = _artifact_path(record, name, data_root)
        if not path.is_file():
            states[name] = "missing"
        elif not verify:
            states[name] = "present_unverified"
        else:
            if path not in file_hashes:
                file_hashes[path] = sha256_file(path)
            states[name] = "verified" if file_hashes[path] == item["sha256"] else "checksum_mismatch"
    return {"checked": True, "artifacts": states}


def catalogue_summary(data_root=None, verify=False, catalog=CATALOG):
    """List every historical decision independently of local artifact availability."""
    if verify and data_root is None:
        raise ValueError("Artifact verification requires an external data root")
    registry = _read_catalog(catalog)
    records, file_hashes = [], {}
    for model_id, model in sorted(registry["models"].items()):
        for variant in [model["default_variant"], *sorted(model.get("variants", {}))]:
            rec = _record(registry, model_id, variant)
            records.append({
                "model": model_id, "variant": variant,
                "engineering": {stage: result["status"] for stage, result in rec["engineering"].items()},
                "scientific": rec["scientific"], "requirements": rec["requirements"],
                "contract": {**rec["contract"], "output_nodes": rec.get("outputs", ["spot_heads"])},
                "artifact_availability": _availability(rec, data_root, verify, file_hashes),
            })
    return {"schema_version": 1, "models": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?")
    parser.add_argument("--list", action="store_true", help="List every checkpoint and variant as JSON")
    parser.add_argument("--variant", help="Select a registered graph boundary for this checkpoint")
    availability = parser.add_mutually_exclusive_group()
    availability.add_argument("--verify", type=Path, metavar="DATA_ROOT", help="Rehash the registered local artifacts")
    availability.add_argument("--data-root", type=Path, help="Report which artifacts exist without hashing their contents")
    args = parser.parse_args(argv)
    root = args.verify or args.data_root
    if args.list:
        if args.model or args.variant:
            parser.error("--list cannot be combined with a single model selection")
        result = catalogue_summary(root, verify=args.verify is not None)
        states = [v for m in result["models"] for v in m["artifact_availability"].get("artifacts", {}).values()]
    elif args.model is None:
        if root is not None or args.variant:
            parser.error("Select a model ID or use --list to inspect the catalogue")
        print("\n".join(sorted(_read_catalog(CATALOG)["models"])))
        return 0
    else:
        try:
            result = model_record(args.model, variant=args.variant)
        except KeyError as exc:
            parser.error(f"Unknown model, variant or registry field: {exc}")
        result["artifact_availability"] = _availability(result, root, args.verify is not None, {})
        states = result["artifact_availability"].get("artifacts", {}).values()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return int(args.verify is not None and any(state != "verified" for state in states))


if __name__ == "__main__":
    raise SystemExit(main())
