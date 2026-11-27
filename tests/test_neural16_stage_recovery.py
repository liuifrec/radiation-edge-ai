from pathlib import Path
from types import SimpleNamespace

import pytest

from radiation_edge_ai.nasa_bps import neural16_nef as gate
from radiation_edge_ai.nasa_bps.multimodel_ptq import load_json, save_json, sha256


def failed_compiler(folder):
    chain = {"all_eight_pass": True, "bie_sha256": "synthetic"}
    paths = ["model.nef", "source/nasa_neural16_nef_worker.py", "docker.log", "kneron_flow/fw_info.txt",
             "kneron_flow/batch_compile.log"]
    paths += ["kneron_flow/model_modelid_32783/" + name for name in
              ("model_config.json", "setup.txt", "setup.bin", "ioinfo.json")]
    for relative in paths:
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    save_json(folder / "eligibility.json", chain)
    code = sha256(folder / "source/nasa_neural16_nef_worker.py")
    evidence = gate.compiler_fingerprints(folder)
    rec = {"stage": "compile", "compile_phase": "postcompile_inspection", "passed": False,
           "worker_sha256": code, "physical_execution": False, "compiler_evidence_sha256": evidence,
           "eligibility": chain, "eligibility_sha256": sha256(folder / "eligibility.json"),
           "error": "original inspection failure", "nef_bytes": (folder / "model.nef").stat().st_size,
           "nef_sha256": evidence["nef_sha256"]}
    save_json(folder / "compile.json", rec)
    digest = sha256(folder / "compile.json")
    invocation = {"stage": "compile", "returncode": 1, "stage_receipt_sha256": digest,
                  "eligibility_sha256": rec["eligibility_sha256"], "sources": {"nasa_neural16_nef_worker.py": code},
                  "log_sha256": sha256(folder / "docker.log")}
    save_json(folder / "invocation.json", invocation)
    save_json(folder / "invocation_failure.json", invocation)
    return chain, digest


def test_reinspection_is_read_only_and_never_qualifies(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/windows"))
    import nasa_neural16_nef_worker as worker

    folder = tmp_path / "original"
    chain, digest = failed_compiler(folder)
    before = {path: sha256(path) for path in folder.rglob("*") if path.is_file()}
    receipt = {"stage": "reinspect", "passed": False, "eligibility": chain}
    monkeypatch.setattr(worker, "compile_model", lambda *unused: pytest.fail("Called compiler"))
    def inspect(flow, result):
        assert flow == folder / "kneron_flow"
        result.update(gate.compiler_fingerprints(folder))
    monkeypatch.setattr(worker, "inspect_compilation", inspect)
    worker.reinspect(receipt, digest, folder)
    assert receipt["inspection_completed"] and receipt["diagnostics_only"] and receipt["passed"] is False
    assert receipt["source_failed_compile_sha256"] == digest and receipt["original_error"] == "original inspection failure"
    assert before == {path: sha256(path) for path in folder.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="failed invocation"):
        gate.completed_stage(folder, "compile", digest, chain)


@pytest.mark.parametrize("relative", ["model.nef", "kneron_flow/model_modelid_32783/setup.bin",
                                     "source/nasa_neural16_nef_worker.py", "docker.log", "compile.json"])
def test_reinspection_rejects_changes(tmp_path, relative):
    chain, digest = failed_compiler(tmp_path)
    with (tmp_path / relative).open("ab") as file:
        file.write(b"modified")
    with pytest.raises(ValueError):
        gate.inspection_evidence(tmp_path, digest, chain)


def test_reinspection_requires_an_authenticated_failure_and_original_seal(tmp_path):
    chain, digest = failed_compiler(tmp_path)
    with pytest.raises(ValueError):
        gate.inspection_evidence(tmp_path, digest, {"all_eight_pass": False})
    with pytest.raises(ValueError):
        gate.inspection_evidence(tmp_path, "0" * 64, chain)
    failed = load_json(tmp_path / "compile.json")
    failed.pop("compile_phase")
    save_json(tmp_path / "compile.json", failed)
    digest = sha256(tmp_path / "compile.json")
    for filename in ["invocation.json", "invocation_failure.json"]:
        inv = load_json(tmp_path / filename)
        save_json(tmp_path / filename, dict(inv, stage_receipt_sha256=digest))
    with pytest.raises(ValueError, match="no authenticated post-compiler"):
        gate.inspection_evidence(tmp_path, digest, chain)


def stage_args(data, out):
    return ["compile", "--data", str(data), "--run", str(data / "run"), "--audit-sha256", "a" * 64, "--out", str(out)]


def test_preflight_exception_is_recorded_once(tmp_path, monkeypatch):
    out = tmp_path / "scratch/attempt"
    def bad_audit(*args):
        raise ValueError("audit differs")
    monkeypatch.setattr(gate, "eligibility", bad_audit)
    with pytest.raises(ValueError, match="audit differs"):
        gate.main(stage_args(tmp_path, out))
    failure = load_json(out / "invocation_failure.json")
    assert failure["phase"] == "authenticate" and "audit differs" in failure["error"]
    assert failure["returncode"] is None and failure["host_wall_seconds"] >= 0
    assert failure == load_json(out / "invocation.json")
    digest = sha256(out / "invocation_failure.json")
    with pytest.raises(FileExistsError):
        gate.main(stage_args(tmp_path, out))
    assert sha256(out / "invocation_failure.json") == digest


def host_inputs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    bie = tmp_path / "fixture.bie"
    bie.write_bytes(b"fixture")
    save_json(run / "preparation.json", {"artifacts": {"bie_sha256": {"path": str(bie)}}})
    headers = tmp_path / "plus/src/include/local"
    for relative in ("build/solution_kdp2_user_ex/sn72096_9x9/project.h", "platform/kl720/common/membase.h"):
        path = tmp_path / "plus/firmware_development/KL720/firmware" / relative
        path.parent.mkdir(parents=True)
        path.write_text("fixture")
    return headers


@pytest.mark.parametrize("behavior,exc_type,code", [("missing_runtime", FileNotFoundError, None),
                                                  ("child_exit", SystemExit, 7), ("missing_verdict", FileNotFoundError, 0)])
def test_host_retains_vendor_failure_and_log(tmp_path, monkeypatch, behavior, exc_type, code):
    out = tmp_path / "scratch/attempt"
    headers = host_inputs(tmp_path)
    monkeypatch.setattr(gate, "eligibility", lambda *args: {"fixture": True})
    def launch(cmd, stdout, stderr, check):
        assert check is False
        stdout.write(b"launch fixture")
        if behavior == "missing_runtime":
            raise FileNotFoundError("missing docker fixture")
        if behavior == "child_exit":
            save_json(out / "compile.json", {"passed": False, "error": "vendor error"})
        return SimpleNamespace(returncode=code)
    monkeypatch.setattr(gate.subprocess, "run", launch)
    with pytest.raises(exc_type):
        gate.main(stage_args(tmp_path, out) + ["--plus-headers", str(headers)])
    failure = load_json(out / "invocation_failure.json")
    assert failure["returncode"] == code and failure["log_sha256"] == sha256(out / "docker.log")
    assert failure["physical_execution"] is False and failure["sources"]
    if behavior == "child_exit":
        assert failure["stage_receipt_sha256"] == sha256(out / "compile.json")


def test_reinspection_refuses_outputs_inside_original_before_creation(tmp_path):
    original = tmp_path / "scratch/original"
    out = original / "nested"
    original.mkdir(parents=True)
    with pytest.raises(ValueError, match="separate from archived"):
        gate.main(["reinspect"] + stage_args(tmp_path, out)[1:] + ["--compiler-output", str(original)])
    assert list(original.iterdir()) == []


@pytest.mark.parametrize("allow_pass", [False, True])
def test_reinspect_launch_only_mounts_original_readonly_and_requires_failed_status(tmp_path, monkeypatch, allow_pass):
    headers = host_inputs(tmp_path)
    original = tmp_path / "scratch/original"
    chain, digest = failed_compiler(original)
    monkeypatch.setattr(gate, "eligibility", lambda *args: chain)
    out = tmp_path / "scratch/diagnostic"
    def diagnostic(cmd, stdout, stderr, check):
        assert check is False
        assert f"type=bind,source={original.resolve()},target=/compiled,readonly" in cmd
        assert [arg for arg in cmd if arg.endswith(",target=/out")] == [f"type=bind,source={out.resolve()},target=/out"]
        assert cmd[cmd.index("/launch/nasa_neural16_nef_worker.py") + 1] == "reinspect"
        assert cmd[-2:] == ["--compile-sha256", digest]
        save_json(out / "reinspect.json", {"stage": "reinspect", "physical_execution": False,
                  "eligibility_sha256": sha256(out / "eligibility.json"), "eligibility": chain,
                  "worker_sha256": sha256(out / "source/nasa_neural16_nef_worker.py"),
                  "passed": allow_pass, "inspection_completed": True, "source_failed_compile_sha256": digest})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(gate.subprocess, "run", diagnostic)
    args = ["reinspect"] + stage_args(tmp_path, out)[1:] + ["--compiler-output", str(original),
            "--compile-sha256", digest, "--plus-headers", str(headers)]
    if allow_pass:
        with pytest.raises(ValueError, match="Invalid diagnostic receipt"):
            gate.main(args)
        assert (out / "invocation_failure.json").is_file()
    else:
        gate.main(args)
        assert load_json(out / "invocation.json")["phase"] == "complete"
        assert not (out / "compile.json").exists()
    assert sha256(original / "compile.json") == digest
