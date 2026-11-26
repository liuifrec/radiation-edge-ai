"""Runtime tensor, point and manifest checks from the qualified Spotiflow replay."""

from datetime import datetime, timezone
import json

import numpy as np

from radiation_edge_ai.audit_utils import sha256_file


JOINED = {"spot_heads": [1, 4, 128, 128]}


SEPARATE = {"/finest_output/block/block.0/Conv_output_0": [1, 1, 128, 128],
            "/flow/flow.6/Conv_output_0": [1, 3, 128, 128]}


def save_heads(out, key, arrays, tensors):
    import numpy as np
    contract = {t["name"]: t["shape"] for t in tensors}
    if contract not in (JOINED, SEPARATE) or not isinstance(arrays, (tuple, list)) or len(arrays) != len(contract):
        raise ValueError("Unknown output contract")
    # Each distinct output shape identifies a single checked graph tensor, including
    # SDK runs returning a list in a different order from ONNX graph.output.
    by_name, physical = {}, []
    for index, value in enumerate(arrays):
        a = np.asarray(value)
        names = [n for n, s in contract.items() if tuple(s) == a.shape]
        if len(names) != 1 or names[0] in by_name or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError("Invalid Spotiflow tensor " + key + " " + str(a.shape))
        name = names[0]
        by_name[name] = a
        if contract == SEPARATE:
            file = out / (key + ("_logit.npy" if a.shape[1] == 1 else "_flow.npy"))
            with file.open("xb") as f:
                np.save(f, a, allow_pickle=False)
            physical.append({"returned_index": index, "graph_name": name, "shape": list(a.shape),
                             "path": file.name, "sha256": sha256_file(file)})
    heads = (by_name["spot_heads"] if contract == JOINED else
             np.concatenate([by_name[n] for n in SEPARATE], axis=1))
    file = out / (key + ".npy")
    with file.open("xb") as f:
        np.save(f, heads, allow_pickle=False)
    row = {"case_id": key, "raw_heads_path": file.name, "raw_heads_sha256": sha256_file(file)}
    if physical:
        row["separate_tensors"] = physical
    return row


def write_json(path, record):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")


def compare_points(a, b):
    if a.shape != b.shape:
        return {"equal_count": False, "max_distance_px": None, "max_coordinate_abs_px": None}
    if np.array_equal(a, b):
        return {"equal_count": True, "max_distance_px": 0., "max_coordinate_abs_px": 0.}
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    distances = cdist(a, b)
    i, j = linear_sum_assignment(distances)
    return {"equal_count": True, "max_distance_px": float(distances[i, j].max()),
            "max_coordinate_abs_px": float(np.abs(a[i] - b[j]).max())}


def tensor_parity(bie, nef):
    import numpy as np
    if (bie.shape != (1, 4, 128, 128) or nef.shape != bie.shape or bie.dtype != np.float32
            or nef.dtype != np.float32 or not np.isfinite(bie).all() or not np.isfinite(nef).all()):
        raise ValueError("Expected a complete pair of finite, matching native-layout float32 output tensors")
    error = float(np.max(np.abs(nef.astype(np.float64)-bie)))
    return {"max_abs_error": error, "exact": bool(np.array_equal(bie, nef)), "within_tolerance": error <= 1e-6}


def original_head_parity(reference, returned):
    import numpy as np
    if (reference.shape != returned.shape or reference.dtype != np.float32
            or returned.dtype not in (np.float32, np.float64)
            or not np.isfinite(reference).all() or not np.isfinite(returned).all()):
        raise ValueError("Original NEF return is nonfinite or incompatible with its BIE output")
    # Widen both sides, retaining every original bit from the actual SDK output.
    error = float(np.max(np.abs(returned.astype(np.float64) - reference.astype(np.float64))))
    return {"max_abs_error": error, "exact": bool(np.array_equal(reference, returned)),
            "within_tolerance": error <= 1e-6}


def checked_array(base, name, digest):
    file = (base / name).resolve()
    if not file.is_relative_to(base.resolve()) or sha256_file(file) != digest:
        raise ValueError(f"Array identity mismatch for {name}")
    return np.load(file, allow_pickle=False)


def align(rows, ids):
    index = {r["case_id"]: r for r in rows}
    if len(index) != len(rows) or set(index) != set(ids):
        raise ValueError("Duplicate, missing or extraneous case in the fixed readout")
    return index


def verify_parts(base, row, expected, joined):
    parts = row["separate_tensors"]
    by_name = {part["graph_name"]: part for part in parts}
    if (len(parts) != len(expected) or set(by_name) != {h["name"] for h in expected}
            or sorted(p["returned_index"] for p in parts) != list(range(len(parts)))):
        raise ValueError("Missing or duplicate physical head")
    arrays = []
    for spec in expected:
        part = by_name[spec["name"]]
        arr = checked_array(base, part["path"], part["sha256"])
        if part["shape"] != spec["shape"] or list(arr.shape) != spec["shape"] or arr.dtype != np.float32:
            raise ValueError("Vendor tensor shape or type differs")
        arrays.append(arr)
    if (joined.dtype != np.float32 or not np.isfinite(joined).all()
            or not np.array_equal(np.concatenate(arrays, axis=1).view(np.uint32), joined.view(np.uint32))):
        raise ValueError("Host concatenation changed tensor values or channel order")
    if "sdk_returned_tensors" in row:
        return load_sdk_heads(base, row, expected, joined)
    return None


def load_sdk_heads(folder, row, contract, hosted):
    """Authenticate original simulator returns and their exact host float32 projections."""
    records = row.get("sdk_returned_tensors", [])
    by_name = {r["graph_name"]: r for r in records}
    if (not records or len(records) != len(contract) or len(by_name) != len(records)
            or set(by_name) != {c["name"] for c in contract}
            or sorted(r["returned_index"] for r in records) != list(range(len(records)))):
        raise ValueError("Missing or duplicate original SDK output identity")
    if len(contract) > 1:
        indices = {r["graph_name"]: r["returned_index"] for r in row["separate_tensors"]}
        if len(row["separate_tensors"]) != len(records) or indices != {name: r["returned_index"] for name, r in by_name.items()}:
            raise ValueError("Separate hosted and original SDK output indices differ")
    originals = []
    channel = 0
    if hosted.dtype != np.float32 or hosted.ndim != 4:
        raise ValueError("Invalid hosted NEF tensor")
    for spec in sorted(contract, key=lambda h: h["shape"][1]):
        record = by_name[spec["name"]]
        raw = checked_array(folder, record["path"], record["sha256"])
        if (list(raw.shape) != spec["shape"] or record["shape"] != spec["shape"]
                or str(raw.dtype) != record["dtype"] or raw.dtype not in (np.float32, np.float64)
                or not np.isfinite(raw).all()):
            raise ValueError("Original SDK output does not match the registered head shape or precision")
        end = channel + raw.shape[1]
        if not np.array_equal(raw.astype(np.float32).view(np.uint32), hosted[:, channel:end].view(np.uint32)):
            raise ValueError("Hosted output differs from the float32 projection of the original SDK output")
        originals.append((record, raw))
        channel = end
    if channel != hosted.shape[1]:
        raise ValueError("Original SDK tensors do not cover all hosted channels")
    return originals


def replay_transport(record, data_root, nef, panel, rows, reference, out):
    """Decode archived CSIM words against the registered PLUS returns and frozen point readout."""
    from radiation_edge_ai.hardware import pretrained_models, spotiflow, spotiflow_plus

    if not out.resolve().is_relative_to(data_root.resolve()):
        raise ValueError("Save replay evidence under the external data root")
    ids = [row["case_id"] for row in rows]
    capture = pretrained_models.verified_artifact(record, "transport_capture", data_root)
    point_file = pretrained_models.verified_artifact(record, "transport_point_score", data_root)
    sources = align(json.loads(capture.read_bytes())["rows"], ids)
    scores = align(json.loads(point_file.read_bytes())["comparisons"]["bie"]["rows"], ids)
    descriptors = spotiflow_plus.descriptors(nef.parent.parent)
    cases = []
    out.mkdir(parents=True, exist_ok=False)
    for row in rows:
        key = row["case_id"]
        source = sources[key]
        current = reference[key]
        entries = source["sdk_csim_words"]
        infos = source["descriptors"]
        if len(entries) != len(descriptors) or len(infos) != len(descriptors):
            raise ValueError("Incomplete transport head set")
        floats = []
        reads = []
        for head, entry, info, (part, expected_words) in zip(descriptors, entries, infos, current["fixed"]):
            if part["graph_name"] != head["name"] or entry["graph_name"] != head["name"]:
                raise ValueError("Transport head identity differs")
            files = []
            for artifact in (entry, info):
                path = (capture.parent / artifact["path"]).resolve()
                if path.parent != capture.parent.resolve() / key or sha256_file(path) != artifact["sha256"]:
                    raise ValueError("Archived transport input identity differs")
                files.append(path)
            words = spotiflow_plus.transport_words(head, *files)
            if not np.array_equal(words, expected_words):
                raise ValueError("Archived transport differs from PLUS signed retrieval")
            floats.append(words.astype(np.float32) / np.float32(head["scale"] * 2**head["radix"]))
            reads.append({"name": head["name"], "words": words.size, "word_file_sha256": entry["sha256"],
                          "info_sha256": info["sha256"]})
        joined = spotiflow.join_heads(*floats)
        identical = np.array_equal(joined.view(np.uint32), current["heads"].view(np.uint32))
        image = checked_array(panel.parent, row["raw_path"], row["raw_sha256"])
        points = spotiflow.decode_heads(joined, image.shape, record["threshold"])["points"]
        expected_points = np.asarray(scores[key]["nef_points"], dtype=np.float64).reshape(-1, 2)
        matching = compare_points(points, expected_points)
        target = None
        if row["target_mask_path"]:
            mask = checked_array(panel.parent, row["target_mask_path"], row["target_mask_sha256"])
            target = sum(spotiflow.target_assignment(mask, row["target_label"], points, len(points))[1])
        passed = (identical and matching["equal_count"] and matching["max_coordinate_abs_px"] <= .001
                  and target == scores[key]["candidate_target_count"])
        cases.append({"case_id": key, "tensor_sha256": row["tensor_sha256"], "heads": reads,
                      "float32_bitwise_equal": bool(identical), "point_comparison": matching,
                      "points_yx": points.tolist(), "target_count": target, "passed": bool(passed)})
    receipt = {"stage": "archived_nef_transport_replay", "utc": datetime.now(timezone.utc).isoformat(),
               "device_access": False, "physical_execution": False, "fresh_simulator_execution": False,
               "nef_sha256": sha256_file(nef), "panel_sha256": sha256_file(panel),
               "transport_capture_sha256": sha256_file(capture), "point_reference_sha256": sha256_file(point_file),
               "variant": record["variant"], "model": record["id"], "case_count": len(cases),
               "rows": cases, "completed": True, "passed": all(row["passed"] for row in cases)}
    write_json(out / "receipt.json", receipt)
    return receipt
