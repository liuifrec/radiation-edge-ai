"""Authenticate the female-fold ResNet18 development decision before compiling its frozen BIE."""

import argparse
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

from . import neural16_regression as regression
from .multimodel_nef import passing_metrics
from .multimodel_ptq import IMAGE_ID, load_json, require, save_json, sha256

NEF_POLICY = "NASA_RESNET18_NEURAL16_NEF_V1.json"
NEF_POLICY_SHA = "979d3899d651cae9b274ed40b5a4a958131e2f008f74c46c83c2ed1deae2ab38"
REPLAY_POLICY = "NASA_RESNET18_NEURAL16_DEVELOPMENT_REPLAY.md"
REPLAY_POLICY_SHA = "80b78d17bfebc6f495f0082b3253c89e6814265155e514542c4f9b8593b02c4f"
TABLES = {"nucleus_pairs.csv", "bag_pairs.csv", "matched_contrasts.csv"}


def verified_audit(policy, comparison, native, audit):
    require(comparison["status"] == "COMPARISON_COMPLETE" and audit["status"] == "AUDIT_COMPLETE"
            and native["status"] == "BIE_SIMULATION_COMPLETE", "Incomplete development readout")
    require(comparison["scope"] == policy["scope"] == "known_female_development_regression",
            "Unrecognized development scope")
    require(comparison["policy"] == policy and native["image_id"] == IMAGE_ID
            and native["evaluated_nuclei"] == 2400 and native["physical_execution"] is False,
            "Unrecognized model, engine or membership")
    require(audit["native_jobs"] == audit["verified_native_words"] == 2400
            and audit["bags"] == 24 and audit["contrasts"] == 12, "Incomplete independent audit")
    for key in ("cohort_membership_sha256", "contrast_membership_sha256"):
        require(comparison[key] == native[key] == audit[key] == policy[key], "Membership changed: " + key)
    require(audit["metrics"].keys() == policy["maximums"].keys()
            and audit["gate_decisions"].keys() == policy["maximums"].keys(), "Incomplete independent gates")
    failed = passing_metrics(comparison, policy["maximums"])
    for key, limit in policy["maximums"].items():
        independent = audit["metrics"][key]
        require(type(independent) in (int, float) and math.isfinite(independent)
                and abs(independent - comparison["metrics"][key]) <= 1e-12
                and audit["gate_decisions"][key] is (independent <= limit)
                and audit["gate_decisions"][key] is comparison["decisions"][key]["pass"],
                "Independent audit disagrees: " + key)
    require(not failed and audit["eligible_for_nef_compilation"] is True, "Development fidelity failed")


def eligibility(docs, data, run, audit_digest):
    from .neural16_comparison import completed_invocation, verify_simulation

    policy = regression.verify_policy(docs / regression.POLICY_NAME)
    require(sha256(docs / NEF_POLICY) == NEF_POLICY_SHA, "Changed prospective NEF policy")
    prep = load_json(run / "preparation.json")
    checked = regression.authenticate(docs, data)
    # Absolute paths depend on the checkout and external drive. Bind scientific
    # identities to verified file bytes and the remaining preparation fields.
    def portable(p):
        return {k: v for k, v in p.items() if k not in ("artifacts", "historical_preparation")}

    def identities(p):
        return {k: (v["sha256"], v["bytes"]) for k, v in p["artifacts"].items()}
    require(portable(prep) == portable(checked) and identities(prep) == identities(checked),
            "Candidate or sealed preparation changed")
    result_path = run / "comparison/comparison.json"
    audit_path = run / "independent_audit/verification.json"
    require(sha256(audit_path) == audit_digest, "Unexpected independent audit digest")
    result, audit = load_json(result_path), load_json(audit_path)
    folder = run / "vendor"
    native_file = folder / "bie_receipt.json"
    native = verify_simulation(run, policy)
    require(completed_invocation(run, native) == result["invocation"], "Changed development launch or source snapshot")
    require(regression.POLICY_SHA == audit["policy_sha256"] == native["policy_sha256"] == result["policy_sha256"]
            and audit["bie_sha256"] == native["bie_sha256"] == regression.BIE_SHA
            and audit["comparison_sha256"] == sha256(result_path)
            and sha256(native_file) == result["bie_receipt_sha256"] == audit["bie_receipt_sha256"]
            and sha256(folder / "bie_nuclei.csv") == native["predictions_sha256"]
            == result["bie_predictions_sha256"] == audit["predictions_sha256"]
            and native["preparation_sha256"] == sha256(run / "preparation.json")
            and prep["input_manifest_sha256"] == native["input_manifest_sha256"] == audit["input_manifest_sha256"],
            "Development receipt chain changed")
    regression.verify_descriptor(native["output_descriptor"])
    for name, expected in native["output_descriptor"]["metadata"].items():
        require(Path(name).name == name and sha256(folder / "metadata" / name) == expected,
                "Changed raw simulator metadata: " + name)
    require(audit["output_descriptor"] == native["output_descriptor"]
            and audit["training_witness_sha256"] == native["training_witnesses"]["sha256"]
            and audit["observed_transport_words"] == 0
            and result["preparation_sha256"] == native["preparation_sha256"]
            and result["bie_sha256"] == regression.BIE_SHA, "Native output domain or preparation changed")
    verified_audit(policy, result, native, audit)
    require(result["output_sha256"] == audit["output_sha256"] and set(result["output_sha256"]) == TABLES,
            "Independent table inventory changed")
    for name, digest in result["output_sha256"].items():
        require(sha256(result_path.parent / name) == digest, "Changed comparison table: " + name)
    chunks = native["chunks"]
    require(len(chunks) == len({r["file"] for r in chunks}) and chunks, "Invalid native chunk inventory")
    for item in chunks + [native["training_witnesses"]]:
        file = (folder / item["file"]).resolve(strict=True)
        require(folder.resolve() in file.parents and sha256(file) == item["sha256"], "Changed native archives")
    return {"scope": policy["scope"], "all_eight_pass": True, "audit_sha256": audit_digest,
            "comparison_sha256": sha256(result_path), "bie_receipt_sha256": sha256(native_file),
            "development_invocation_sha256": result["invocation"]["sha256"],
            "input_manifest_sha256": native["input_manifest_sha256"],
            "predictions_sha256": native["predictions_sha256"], "preparation_sha256": sha256(run / "preparation.json"),
            "policy_sha256": regression.POLICY_SHA, "bie_sha256": regression.BIE_SHA,
            "nef_policy_sha256": NEF_POLICY_SHA, "image_id": IMAGE_ID}


def completed_stage(folder, stage, digest, chain):
    require(not (folder / "invocation_failure.json").exists(), "Host recorded a failed invocation")
    return _stage_receipt(folder, stage, digest, chain, passed=True)


def _stage_receipt(folder, stage, digest, chain, passed):
    require(re.fullmatch("[0-9a-f]{64}", digest or ""), "Expected a SHA256 for the completed " + stage)
    file = folder / (stage + ".json")
    rec = load_json(file)
    invocation = load_json(folder / "invocation.json")
    require(sha256(file) == digest == invocation["stage_receipt_sha256"]
            and rec["eligibility"] == chain and rec["passed"] is passed
            and rec["stage"] == stage and rec["physical_execution"] is False
            and sha256(folder / "eligibility.json") == rec["eligibility_sha256"] == invocation["eligibility_sha256"]
            and load_json(folder / "eligibility.json") == chain and type(invocation["returncode"]) is int
            and (invocation["returncode"] == 0) is passed
            and sha256(folder / "docker.log") == invocation["log_sha256"], "Changed stage or host invocation")
    for name, expected in invocation["sources"].items():
        path = (folder / "source" / name).resolve()
        require((folder / "source").resolve() in path.parents and sha256(path) == expected,
                "Changed vendor execution source snapshot")
    require(invocation["sources"]["nasa_neural16_nef_worker.py"] == rec["worker_sha256"], "Worker mismatch")
    return rec


def compiler_fingerprints(folder):
    """Seal all binary and text inputs consumed by the post-compile inspector."""
    flow = folder / "kneron_flow"
    model = flow / "model_modelid_32783"
    result = {}
    for path, field in ((folder / "model.nef", "nef_sha256"), (flow / "fw_info.txt", "firmware_map_sha256"),
                        (flow / "batch_compile.log", "compiler_log_sha256"),
                        (model / "model_config.json", "compiler_config_sha256"),
                        (model / "setup.txt", "setup_text_sha256"), (model / "setup.bin", "setup_bin_sha256"),
                        (model / "ioinfo.json", "compiled_io_sha256")):
        require(path.resolve(strict=True) == folder.resolve() / path.relative_to(folder),
                "Compiled artifact escapes the preserved output: " + path.name)
        result[field] = sha256(path)
    return result


def compiled_files(folder, compiled):
    for field, digest in compiler_fingerprints(folder).items():
        require(digest == compiled[field], "Compiled artifact changed: " + field)


def inspection_evidence(folder, digest, chain):
    """Authenticate a failed inspection with a complete, pre-inspection seal."""
    rec = _stage_receipt(folder, "compile", digest, chain, passed=False)
    failure = load_json(folder / "invocation_failure.json")
    require(failure["stage"] == "compile" and failure["stage_receipt_sha256"] == digest
            and failure["returncode"] == load_json(folder / "invocation.json")["returncode"]
            and rec.get("compile_phase") == "postcompile_inspection"
            and rec.get("error") and rec["compiler_evidence_sha256"] == compiler_fingerprints(folder)
            and rec["nef_sha256"] == rec["compiler_evidence_sha256"]["nef_sha256"]
            and rec["nef_bytes"] == (folder / "model.nef").stat().st_size,
            "Failed compile has no authenticated post-compiler inspection inputs")
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["check", "compile", "reinspect", "simulate", "development-reference"])
    for name in ("data", "run"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--audit-sha256", required=True)
    p.add_argument("--compile-sha256", help="Expected original compile.json digest for simulation or reinspection")
    p.add_argument("--out", type=Path, help="New external evidence directory, required for execution")
    p.add_argument("--compiler-output", type=Path, help="Original compile directory, always mounted read-only")
    p.add_argument("--training-simulation", type=Path, help="Passing 48-case directory for development CSIM")
    p.add_argument("--training-simulation-sha256", help="Expected digest of its simulate.json")
    p.add_argument("--plus-headers", type=Path, default=Path.home() / "kneron_plus/src/include/local")
    args = p.parse_args(argv)
    repo = Path(__file__).resolve().parents[3]
    if args.stage == "check":
        print(json.dumps(eligibility(repo / "docs", args.data, args.run, args.audit_sha256), indent=2))
        return
    require(args.out is not None, "Provide fresh external output")
    require((args.data / "scratch").resolve() in args.out.resolve().parents,
            "Save compiler and simulator evidence in a fresh external scratch directory")
    for source in (args.compiler_output, args.training_simulation):
        if source is not None:
            require(source.resolve() != args.out.resolve() and source.resolve() not in args.out.resolve().parents,
                    "Output directory must be separate from archived stage evidence")
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    invocation = {"stage": args.stage, "phase": "authenticate", "physical_execution": False,
                  "audit_sha256": args.audit_sha256, "returncode": None}
    try:
        save_json(args.out / "invocation.json", invocation)
        chain = eligibility(repo / "docs", args.data, args.run, args.audit_sha256)
        invocation["phase"] = "prepare"
        execute(args, repo, chain, invocation)
        invocation["phase"] = "complete"
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        invocation["error"] = repr(exc)
        raise
    finally:
        invocation["host_wall_seconds"] = time.monotonic() - start
        try:
            for name, key in [("docker.log", "log_sha256"), (args.stage + ".json", "stage_receipt_sha256")]:
                if (args.out / name).is_file():
                    invocation[key] = sha256(args.out / name)
            save_json(args.out / "invocation.json", invocation)
        except (Exception, KeyboardInterrupt) as exc:
            invocation["evidence_error"] = repr(exc)
            raise
        finally:
            if "error" in invocation or "evidence_error" in invocation:
                save_json(args.out / "invocation_failure.json", invocation)


def execute(args, repo, chain, invocation):
    save_json(args.out / "eligibility.json", chain)
    invocation["eligibility_sha256"] = sha256(args.out / "eligibility.json")
    prepared = load_json(args.run / "preparation.json")["artifacts"]
    plus_fw = args.plus_headers.resolve().parents[2] / "firmware_development/KL720/firmware"
    mounts = [(repo / "docs" / NEF_POLICY, "/policy.json"),
              (plus_fw / "build/solution_kdp2_user_ex/sn72096_9x9/project.h", "/map/project.h"),
              (plus_fw / "platform/kl720/common/membase.h", "/map/membase.h"),
              (Path(prepared["bie_sha256"]["path"]), "/model.bie")]
    extra = []
    if args.stage == "reinspect":
        require(args.compiler_output is not None, "Supply the original failed compile directory")
        inspection_evidence(args.compiler_output, args.compile_sha256, chain)
        mounts.append((args.compiler_output, "/compiled"))
        extra = ["--compile-sha256", args.compile_sha256]
    if args.stage in ("simulate", "development-reference"):
        require(args.compiler_output and re.fullmatch("[0-9a-f]{64}", args.compile_sha256 or ""),
                "A hashed compile receipt is required")
        compiled = completed_stage(args.compiler_output, "compile", args.compile_sha256, chain)
        compiled_files(args.compiler_output, compiled)
        mounts.extend([(args.compiler_output, "/compiled"),
                       (Path(prepared["training_panel_sha256"]["path"]), "/selection.json"),
                       (Path(prepared["training_predictions_sha256"]["path"]), "/training.csv"),
                       (Path(prepared["calibration_manifest_sha256"]["path"]).parent, "/export"),
                       (args.plus_headers, "/plus_headers")])
        extra = ["--compile-sha256", args.compile_sha256]
    if args.stage == "development-reference":
        require(args.training_simulation is not None, "Supply the authenticated 48-case simulation")
        complete = completed_stage(args.training_simulation, "simulate", args.training_simulation_sha256, chain)
        require(complete["compile_sha256"] == args.compile_sha256 and complete["nef_sha256"] == compiled["nef_sha256"]
                and len(complete["rows"]) == 48 and all(row["passed"] for row in complete["rows"]),
                "Development reference requires a passing 48-case NEF qualification")
        require(sha256(repo / "docs" / REPLAY_POLICY) == REPLAY_POLICY_SHA, "Changed development replay scope")
        mounts.extend([(repo / "docs" / REPLAY_POLICY, "/replay.md"),
                       (Path(load_json(args.run / "preparation.json")["historical_preparation"]), "/dev_inputs"),
                       (args.run / "vendor/bie_nuclei.csv", "/dev_predictions.csv"),
                       (args.training_simulation / "simulate.json", "/training_sim.json")])
        extra += ["--training-simulation-sha256", args.training_simulation_sha256]
    snapshot = args.out / "source"
    # Preserve exactly the package modules and vendor entrypoints used by this process.
    package = repo / "src/radiation_edge_ai"
    names = ["__init__.py", "nasa_bps/__init__.py"]
    names += ["nasa_bps/" + name + ".py" for name in (
        "neural16_nef", "neural16_regression", "multimodel_bie_inference", "multimodel_nef",
        "multimodel_qualification", "multimodel_ptq", "deployment_cohort", "neural16_transport",
        "kneron_verify_r1_v2_nef_vs_bie")]
    files = [(package / name, "radiation_edge_ai/" + name) for name in names]
    files.extend((repo / "scripts/windows" / name, name) for name in
                 ("nasa_neural16_nef_worker.py", "nasa_neural16_setup.c"))
    source_hashes = {}
    for source, relative in files:
        copy = snapshot / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copy)
        source_hashes[relative] = sha256(copy)
    mounts.extend([(snapshot, "/source"), (snapshot, "/launch")])
    mounts.append((args.out / "eligibility.json", "/eligibility.json"))
    cmd = ["docker", "run", "--rm", "--network", "none", "--pull", "never", "--cpus", "4",
           "--memory", "6g", "-e", "PYTHONPATH=/source", "-e", "OMP_NUM_THREADS=1"]
    for local, guest in mounts:
        cmd += ["--mount", f"type=bind,source={local.resolve(strict=True)},target={guest},readonly"]
    cmd += ["--mount", f"type=bind,source={args.out.resolve()},target=/out",
            "--entrypoint", "/workspace/miniconda/bin/python", IMAGE_ID, "-B", "/launch/nasa_neural16_nef_worker.py",
            args.stage, "--eligibility-sha256", sha256(args.out / "eligibility.json")] + extra
    invocation.update(argv=cmd, sources=source_hashes, phase="launch")
    save_json(args.out / "invocation.json", invocation)
    with (args.out / "docker.log").open("xb") as log:
        start = time.monotonic()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    invocation.update(returncode=proc.returncode, seconds=time.monotonic()-start, phase="verify_worker")
    if proc.returncode:
        raise SystemExit(proc.returncode)
    verdict = load_json(args.out / (args.stage + ".json"))
    require(verdict["stage"] == args.stage and verdict["physical_execution"] is False
            and verdict["eligibility_sha256"] == invocation["eligibility_sha256"] and verdict["eligibility"] == chain
            and verdict["worker_sha256"] == source_hashes["nasa_neural16_nef_worker.py"], "Invalid worker receipt")
    if args.stage == "reinspect":
        require(verdict["passed"] is False and verdict["inspection_completed"] is True
                and verdict["source_failed_compile_sha256"] == args.compile_sha256, "Invalid diagnostic receipt")
    else:
        require(verdict["passed"] is True, "Worker recorded a failed qualification")


if __name__ == "__main__":
    main()
