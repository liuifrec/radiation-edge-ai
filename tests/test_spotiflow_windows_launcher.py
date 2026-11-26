"""Exercise the Windows argument routing with an inert Python stand-in."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows launcher")
@pytest.mark.parametrize("flags,expected", [
    (["-Preflight"], ["--preflight-only"]),
    (["-ReplaySimulator"], ["--replay-simulator"]),
    (["-Smoke"], ["--smoke"]),
    (["-Panel", "-SmokeReceipt", "prior.json", "-SmokeSha256", "0123"], ["--panel", "--smoke-receipt", "prior.json"]),
    (["-Image", "test input.npy"], ["--backend", "kl720", "--image", "test input.npy"]),
])
def test_launcher_selects_exactly_one_backend(tmp_path, flags, expected, monkeypatch):
    fixture = tmp_path / "stub.ps1"
    fixture.write_text("ConvertTo-Json -Compress -InputObject @($args) | Set-Content -LiteralPath $env:RADE_TEST_ARGS")
    argfile = tmp_path / "args.json"
    monkeypatch.setenv("RADE_TEST_ARGS", str(argfile))
    repo = Path(__file__).resolve().parents[1]
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    subprocess.run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(repo / "scripts/windows/run_spotiflow_kl720.ps1"), "-DataRoot", str(tmp_path),
                    "-SpotiflowPython", str(fixture), *flags], check=True)
    passed = json.loads(argfile.read_text(encoding="utf-8-sig"))
    start = passed.index(expected[0])
    assert passed[start:start + len(expected)] == expected
    assert passed[passed.index("--variant") + 1] == "full_int16_transport"
    mode = [w for w in passed if w in ("--preflight-only", "--replay-simulator", "--smoke", "--panel", "--backend")]
    assert len(mode) == 1 and not (tmp_path / "scratch").exists()
