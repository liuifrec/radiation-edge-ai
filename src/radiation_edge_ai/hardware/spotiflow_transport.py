"""Resolve the KL720 INT16 transport decision against the original registered engineering panel."""

from pathlib import Path

from radiation_edge_ai.control import load_json_object as read_json_object
from radiation_edge_ai.control import sha256_file
from radiation_edge_ai.hardware.pretrained_models import verified_artifact

VARIANT = "full_int16_transport"


def registered_kn(record, root):
    """Read the paired discovery identity without publishing the unit identifier."""
    discovery = read_json_object(verified_artifact(record, "device_discovery", root))
    kn = int(str(discovery["device"]["kn_number"]), 0)
    check_device(discovery["device"], discovery["system_info"], expected_kn=kn)
    return kn


def require_rows(rows, expected):
    indexed = {r["case_id"]: r for r in rows}
    if len(rows) != 63 or len(indexed) != 63 or indexed.keys() != expected:
        raise ValueError("Expected all 63 unique, matching transport cases")
    return indexed


def check_device(device, system=None, expected_firmware=None, *, expected_kn):
    def number(value):
        return int(str(value), 0)
    if number(device["product_id"]) != 0x720 or number(device["kn_number"]) != expected_kn:
        raise ValueError("Unexpected physical KL720 device identity")
    if system is not None and (number(system["kn_number"]) != expected_kn or not system.get("firmware_version")):
        raise ValueError("Connected firmware identity is incomplete or reports a different KN number")
    if expected_firmware is not None and (system is None or system["firmware_version"] != expected_firmware):
        raise ValueError("Physical panel firmware differs from its authenticated device smoke")


def qualified_physical_panel(record, root):
    stage = record["engineering"]["physical_kl720"]
    if stage.get("status") != "passed" or not stage.get("evidence"):
        raise ValueError("Use the guarded resident smoke and panel before arbitrary physical inputs")
    report = read_json_object(verified_artifact(record, stage["evidence"], root))
    smoke = read_json_object(verified_artifact(record, "physical_smoke", root))
    score = read_json_object(verified_artifact(record, "physical_native_score", root))
    panel = read_json_object(verified_artifact(record, "panel", root))
    expected = {r["case_id"] for r in panel["rows"] if r["role"] == "readout"}
    data = require_rows(report["rows"], expected)
    for field, name in (("nef_sha256", "nef"), ("panel_sha256", "panel"), ("nef_outputs_sha256", "transport_outputs"),
                        ("nef_simulator_gate_sha256", "transport_point_score")):
        if report[field] != record["artifacts"][name]["sha256"]:
            raise ValueError("Physical panel refers to another release or decision")
    kn = registered_kn(record, root)
    check_device(smoke["device"], smoke["system_info"], expected_kn=kn)
    check_device(report["device"], report["system_info"], smoke["system_info"]["firmware_version"], expected_kn=kn)
    if (smoke["stage"] != "physical_kl720_smoke" or smoke["passed"] is not True or smoke["completed"] is not True
            or report["passed_smoke"]["sha256"] != record["artifacts"]["physical_smoke"]["sha256"]
            or smoke["nef_sha256"] != report["nef_sha256"] or smoke["model_id"] != report["model_id"]
            or smoke["smoke_raw_retrieval"]["raw_plus_retrieval_verified"] is not True
            or score["stage"] != "native_vs_physical_kl720" or score["passed"] is not True
            or score["candidate_manifest_sha256"] != record["artifacts"][stage["evidence"]]["sha256"]
            or score["native_manifest_sha256"] != record["artifacts"]["onnx_parity_receipt"]["sha256"]
            or score["nef_output_manifest_sha256"] != record["artifacts"]["transport_outputs"]["sha256"]
            or score["nef_gate_sha256"] != record["artifacts"]["transport_point_score"]["sha256"]):
        raise ValueError("Registered physical smoke, signed transport and full-panel scores do not match")
    if (report["passed"] is not True or report["completed"] is not True or report["stage"] != "physical_kl720_panel"
            or report["variant"] != VARIANT or not report["passed_smoke"]["sha256"] or report["plus_version"] != "3.2.0"
            or not all(r["passed"] is True for r in data.values())):
        raise ValueError("The complete physical transport qualification is missing")
    return report


def qualified_transport(record, root):
    verdict = record["engineering"]["nef_simulator_parity"]
    if verdict.get("status") != "passed" or verdict.get("evidence") != "transport_point_score":
        raise ValueError("A passing 63-case INT16 transport decision is required before physical inference")
    def artifact(name):
        return read_json_object(verified_artifact(record, name, root))
    def sha(name):
        return record["artifacts"][name]["sha256"]

    panel, bie, old, failed = (artifact(k) for k in ("panel", "bie_outputs", "nef_outputs", "nef_sim_receipt"))
    packing, typed, embedded = (artifact(k) for k in ("input_packing_receipt", "typed_nef_setup", "embedded_nef_setup"))
    verified_artifact(record, "sdk_input_packing", root)
    if (packing["passed"] is not True or packing["sdk_sha256"] != sha("sdk_input_packing")
            or packing["panel_sha256"] != sha("panel") or packing["bie_sha256"] != sha("bie")
            or typed["nef_sha256"] != sha("nef") or embedded["nef_sha256"] != sha("nef")
            or typed["extraction_sha256"] != sha("embedded_nef_setup") or typed["setup_sha256"] != embedded["setup_sha256"]):
        raise ValueError("Different binary NEF descriptors or SDK input packing qualification")
    capture, outputs, raw, score = (artifact(k) for k in (
        "transport_capture", "transport_outputs", "transport_dll_receipt", "transport_point_score"))
    readout = [r for r in panel["rows"] if r["role"] == "readout"]
    expected_rows = {r["case_id"]: r for r in readout}
    if (len(readout) != 63 or len(expected_rows) != 63 or failed.get("passed") is not False
            or failed["output_manifest_sha256"] != sha("nef_outputs")):
        raise ValueError("Original failed simulator identity or frozen panel differs")
    limits_sha = record["provenance"]["gate_lf_sha256"]
    docs = (capture, outputs, raw, score)
    if any(d["nef_sha256"] != sha("nef") or d["bie_sha256"] != sha("bie") or
           d["panel_sha256"] != sha("panel") or d["gates_sha256"] != limits_sha for d in docs):
        raise ValueError("Transport evidence uses different models, panel or numerical gates")
    if (capture["stage"] != "nef_csim_int16_transport" or capture["original_manifest_sha256"] != sha("nef_outputs")
            or outputs["stage"] != "offline_plus_retrieval_of_csim_int16_transport"
            or raw["schema"] != outputs["stage"] + "_v1" or raw["physical_execution"] is not False
            or raw["capture_sha256"] != sha("transport_capture") or raw["outputs_sha256"] != sha("transport_outputs")
            or raw["original_failed_gate_sha256"] != sha("nef_sim_receipt") or raw["bie_outputs_sha256"] != sha("bie_outputs")
            or raw["original_csim_manifest_sha256"] != sha("nef_outputs") or raw["abs_tolerance"] != 0.000001
            or raw["rtol"] != 0 or raw["raw_tensor_passed"] is not True or raw["raw_tensor_passes"] != 63
            or score["source_manifest_sha256"] != sha("transport_outputs") or score["physical_execution"] is not False
            or score["native_manifest_sha256"] != sha("onnx_parity_receipt") or score["bie_manifest_sha256"] != sha("bie_outputs")
            or score["bie_gate_sha256"] != sha("bie_gate") or score["source_manifest_stage"] != outputs["stage"]
            or score["threshold"] != record["threshold"]):
        raise ValueError("Incomplete or mismatched transport decode and numerical decision")
    counts = score["bie_to_decoded"]
    criteria = score["native_comparison_using_original_bie_limits"]
    if (any(counts[k] != 63 for k in ("cases", "raw_tensor_passes", "point_count_passes", "coordinate_passes"))
            or counts["target_count_cases"] != 47 or counts["target_count_passes"] != 47
            or len(criteria) != 8 or criteria.keys() != artifact("bie_gate")["criteria"].keys()
            or not all(v is True for v in criteria.values())):
        raise ValueError("Transport evidence failed one or more frozen engineering criteria")
    br, archived, simulated, decoded, scores = (require_rows(rows, expected_rows.keys()) for rows in
        (bie["rows"], old["rows"], capture["rows"], outputs["rows"], score["bie_to_decoded_by_case"]))
    packed_cases = require_rows([r for r in packing["rows"] if r["role"] == "readout"], expected_rows.keys())
    folder = (Path(root).resolve() / record["artifacts"]["transport_capture"]["path"]).parent
    for case, row in expected_rows.items():
        src, data, numerical = simulated[case], decoded[case], scores[case]
        packed = archived[case]["sdk_packed_input"]
        signed = {w["graph_name"]: w["sha256"] for w in src["sdk_csim_words"]}
        if (br[case]["input_tensor_sha256"] != row["tensor_sha256"] or data["input_tensor_sha256"] != row["tensor_sha256"]
                or packed.get("available") is not True or packed != src["sdk_packed_input"] or packed != data["sdk_packed_input"]
                or packed_cases[case]["packed_equal"] is not True or packed_cases[case]["host_packed_sha256"] != packed["sha256"]
                or len(signed) != 2 or signed != data["source_word_sha256"] or data["within_tolerance"] is not True
                or not all(numerical[k] is True for k in ("point_counts_equal", "within_original_coordinate_tolerance"))
                or numerical["tensor"]["within_tolerance"] is not True or numerical["tensor"]["max_abs_error"] > 1e-6
                or (row["target_label"] is not None and numerical["target_counts_equal"] is not True)):
            raise ValueError("Transport case does not bind the frozen input and passing measurements: " + case)
        if case != Path(case).name or "/" in case or "\\" in case:
            raise ValueError("Invalid transport case ID")
        for i, w in enumerate(src["sdk_csim_words"]):
            name = f"dma2seq_{i}.seq.16B"
            path = (folder / w["path"]).resolve(strict=True)
            if (path.parent != folder / case or path.name != name or w["sdk_name"] != name
                    or path.stat().st_size != w["bytes"] or sha256_file(path) != w["sha256"]):
                raise ValueError("Raw signed CSIM transport differs for " + case)
    return decoded
