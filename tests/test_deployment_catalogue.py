import json
import sys
from pathlib import Path

import pytest

from radiation_edge_ai.desktop_runtime import Settings, build_job
from radiation_edge_ai.hardware import deployment_catalogue as directory


def test_references_and_candidates_keep_independent_decisions():
    records = {r["id"]: r for r in directory.report()["models"]}
    refs = {k for k, v in records.items() if v["track"] == "physical_reference"}
    assert refs == set(directory.RUN_BACKENDS)
    assert records[directory.SPOT]["physical_stage"]["cases_passed"] == 63
    assert records["spotiflow/synth_complex@joined_heads"]["engineering"]["onnx_parity"] == "failed"
    assert not records["dnai-fiber-v4"]["desktop_backends"]
    assert not records["dnai-fiber-v5"]["desktop_backends"]
    mobile = records[directory.MOBILENET]
    assert mobile["track"] == "experimental" and mobile["desktop_backends"] == []
    assert mobile["engineering"]["nef_compilation"] == "blocked_by_bie_fidelity"
    assert mobile["engineering"]["engineering_gates_passed"] == 7
    assert "bie_comparison" in mobile["artifacts"] and "nef" not in mobile["artifacts"]
    assert all(v["state"] == "unchecked" for m in records.values() for v in m["artifacts"].values())


def test_offline_missing_and_corrupt_return_failure_without_creating_assets(tmp_path, capsys):
    argv = ["--model", directory.DNAI, "--data-root", str(tmp_path), "--verify"]
    assert directory.main(argv) == 1
    result = json.loads(capsys.readouterr().out)["models"][0]
    assert all(v["state"] == "missing" for v in result["artifacts"].values())
    assert list(tmp_path.iterdir()) == []
    file = tmp_path / result["artifacts"]["nef"]["path"]
    file.parent.mkdir(parents=True)
    file.write_bytes(b"altered fixture")
    assert directory.main(argv) == 1
    assert json.loads(capsys.readouterr().out)["models"][0]["artifacts"]["nef"]["state"] == "checksum_mismatch"
    assert directory.main(["--model", "dnai-fiber-v5", "--data-root", str(tmp_path), "--verify"]) == 1


@pytest.mark.parametrize("model", [directory.NASA, directory.DNAI])
def test_desktop_frozen_batch_requires_kl720_and_creates_fresh_evidence(tmp_path, model):
    cfg = Settings(sys.executable, str(tmp_path), sys.executable, sys.executable, 42)
    repo = Path(__file__).resolve().parents[1]
    job = build_job(cfg, repo, "deployment-run", model=model, selected_backend="kl720")
    assert job.hardware_requested and job.backend == "kl720" and job.assay_id == model
    assert job.argv[-2:] == ("--port", "42")
    assert any(a.endswith(directory.REFERENCES[model]["runner"].split("/")[-1]) for a in job.argv)
    assert ("--execute" in job.argv) == (model == directory.DNAI)
    for model_id, backend in ((model, ""), (model, "onnx"), ("dnai-fiber-v4", "kl720")):
        with pytest.raises(ValueError, match="supported model and backend"):
            build_job(cfg, repo, "deployment-run", model=model_id, selected_backend=backend)
    with pytest.raises(ValueError, match="Clear the input"):
        build_job(cfg, repo, "deployment-run", "unused.csv", model=model, selected_backend="kl720")


def test_desktop_spotiflow_selects_exact_variant_and_requested_backend(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    image = tmp_path / "input.tiff"
    image.write_bytes(b"stand-in for command construction only")
    cfg = Settings(sys.executable, str(tmp_path), sys.executable, "", 42)
    job = build_job(cfg, repo, "deployment-run", str(image), str(tmp_path / "new"),
                    model=directory.SPOT, selected_backend="onnx")
    assert not job.hardware_requested and "--plus-python" not in job.argv and "--port" not in job.argv
    assert job.argv[job.argv.index("--variant") + 1] == "full_int16_transport"
    assert job.argv[job.argv.index("--backend") + 1] == "onnx"
    assert job.source_sha256 and not Path(job.output).exists()
    with pytest.raises(ValueError, match="PLUS Python"):
        build_job(cfg, repo, "deployment-run", str(image), str(tmp_path / "new"),
                  model=directory.SPOT, selected_backend="kl720")
    cfg = Settings(sys.executable, str(tmp_path), sys.executable, sys.executable, 42)
    hardware = build_job(cfg, repo, "deployment-run", str(image), str(tmp_path / "new"),
                         model=directory.SPOT, selected_backend="kl720")
    assert hardware.hardware_requested and "--plus-python" in hardware.argv
    assert hardware.argv[hardware.argv.index("--backend") + 1] == "kl720"
    assert hardware.argv[-2:] == ("--port", "42")
    verify = build_job(cfg, repo, "deployment-verify", model=directory.SPOT)
    assert not verify.hardware_requested and "--verify" in verify.argv
