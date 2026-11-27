"""Exercise the PowerShell handoff with an inert, local stand-in for PLUS."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell launcher")


@pytest.fixture
def invoke(tmp_path):
    stub = tmp_path / "capture.ps1"
    stub.write_text(
        "$dir = $args[[array]::IndexOf($args,'--out')+1]\n"
        "New-Item -ItemType Directory -Path $dir | Out-Null\n"
        "ConvertTo-Json -Compress -InputObject @($args) | Set-Content (Join-Path $dir args.json) -Encoding UTF8\n"
        "$name = if ($args -contains 'development-reference') {'development-reference.json'} else {'receipt.json'}\n"
        "Set-Content (Join-Path $dir $name) '{}' -Encoding UTF8\nexit 0\n",
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts/windows/run_resnet_neural16_kl720.ps1"
    shell = shutil.which("powershell.exe") or shutil.which("pwsh")

    def run(*flags):
        return subprocess.run([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
                               "-DataRoot", str(tmp_path), "-PlusPython", str(stub), *flags],
                              capture_output=True, text=True)
    return run


@pytest.mark.parametrize("mode", ["default", "Training", "DevelopmentReference", "Development"])
def test_routing_and_explicit_physical_execution(tmp_path, invoke, mode):
    opts = [] if mode == "default" else ["-Mode", mode]
    if mode == "Training":
        opts += ["-Execute", "-SmokeReceipt", "confirmed smoke.json", "-SmokeSha256", "a" * 64]
    if mode == "Development":
        opts += ["-ReferenceDir", "external reference", "-ReferenceSha256", "c" * 64]
    call = invoke(*opts)
    assert call.returncode == 0, call.stderr
    record, = tmp_path.rglob("args.json")
    arguments = json.loads(record.read_text(encoding="utf-8-sig"))
    assert arguments[arguments.index("--audit-sha256") + 1] == "4f9dd6f061586840636de0aa4292944f880bef6325e4f0dcc0f90f7aa571bf20"
    assert arguments[arguments.index("--compile-sha256") + 1] == "d068c2b2647edd6a75749bae2d34b7f50e241843a42ec5ed1ac62f92c8130f06"
    assert ("--execute" in arguments) is (mode == "Training")
    if mode == "DevelopmentReference":
        assert "development-reference" in arguments
        assert arguments[arguments.index("--training-simulation-sha256") + 1] == "1706ad42edcf401b055077534e6839e63597449f783a0422ab70ebce4d60d59b"
    elif mode == "Development":
        assert arguments[arguments.index("--sim-sha256") + 1] == "c" * 64
        assert arguments[arguments.index("--training-sim-dir") + 1].endswith("nef_sim")
    else:
        assert arguments[2] == ("smoke" if mode == "default" else "training")
        assert arguments[arguments.index("--sim-sha256") + 1] == "1706ad42edcf401b055077534e6839e63597449f783a0422ab70ebce4d60d59b"
    if mode == "Training":
        assert arguments[arguments.index("--smoke-receipt") + 1] == "confirmed smoke.json"
    elif mode == "default":
        assert invoke().returncode == 0 and len(list(tmp_path.rglob("args.json"))) == 2


@pytest.mark.parametrize("flags", [
    ["-Mode", "Training", "-Execute"],
    ["-Mode", "Development"],
    ["-Mode", "DevelopmentReference", "-Execute"],
    ["-Mode", "Smoke", "-SmokeReceipt", "unrelated.json", "-SmokeSha256", "a" * 64],
])
def test_rejects_incomplete_or_misrouted_evidence(tmp_path, invoke, flags):
    assert invoke(*flags).returncode != 0
    assert not list(tmp_path.rglob("args.json"))
