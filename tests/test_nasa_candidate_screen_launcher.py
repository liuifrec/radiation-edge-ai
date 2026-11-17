"""Exercise the PowerShell launcher with real child processes and synthetic receipts.

The fake processes deliberately do not import a vendor SDK or run a model. Native
Windows CI checks path quoting, process status, Docker arguments, and receipt
validation that cannot be established by inspecting PowerShell source text.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "windows" / "screen_nasa_kl720_models.ps1"
IMAGE_ID = "sha256:" + "a" * 64


FAKE_DRIVER = r'''
import hashlib
import json
import os
import sys
from pathlib import Path

mode, args = sys.argv[1], sys.argv[2:]
case = os.environ.get("RADEDGE_FAKE_CASE", "supported")
with open(os.environ["RADEDGE_FAKE_CAPTURE"], "a", encoding="utf-8") as f:
    f.write(json.dumps({"mode": mode, "args": args, "pythonpath": os.environ.get("PYTHONPATH")}) + "\n")

if mode == "shadow":
    print("A lower-priority Docker executable must not run", file=sys.stderr)
    sys.exit(91)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")

if mode == "export":
    if case == "export_failure":
        print("synthetic export failure", file=sys.stderr)
        sys.exit(9)
    root = Path(args[args.index("--out") + 1])
    root.mkdir()
    models = args[args.index("--models") + 1:]
    entries = []
    for name in models:
        folder = root / name
        folder.mkdir()
        graph = folder / "candidate.onnx"
        graph.write_bytes(("synthetic graph " + name).encode())
        manifest = folder / "candidate.json"
        write(manifest, {"architecture": name, "onnx_sha256": digest(graph)})
        entries.append({"architecture": name, "onnx_file": name + "/candidate.onnx",
                        "candidate_manifest": name + "/candidate.json", "onnx_sha256": digest(graph),
                        "candidate_manifest_sha256": digest(manifest)})
    if case == "export_hash_mismatch":
        entries[0]["onnx_sha256"] = "0" * 64
    write(root / "candidate_graphs.json", {"status": "COMPLETE", "models": entries})
    print("Synthetic exporter fixture complete")
    sys.exit(0)

if args[0] == "info":
    print("daemon unavailable" if case == "daemon_failure" else "linux")
    sys.exit(1 if case == "daemon_failure" else 0)
if args[:2] == ["image", "inspect"]:
    if case == "missing_image":
        print("synthetic image absent", file=sys.stderr)
        sys.exit(1)
    print(json.dumps([{"Id": "sha256:" + "a" * 64, "Os": "linux", "RepoDigests": ["synthetic@sha256:" + "b" * 64]}]))
    sys.exit(0)
assert args[0] == "run", args
if case == "missing_report":
    print("synthetic missing interpreter", file=sys.stderr)
    sys.exit(127)
mounts = {}
for index, arg in enumerate(args):
    if arg == "--mount":
        fields = dict(part.split("=", 1) for part in args[index + 1].split(",") if "=" in part)
        mounts[fields["target"]] = Path(fields["source"])
name = args[args.index("--out") + 1].split("/")[-1]
folder = mounts["/results"] / name
assert not folder.exists(), "The vendor output must be fresh"
folder.mkdir()
graph = mounts["/graphs"] / name / "candidate.onnx"
manifest = mounts["/graphs"] / name / "candidate.json"
status = "unsupported" if case == "unsupported" and name == "mobilenet_v2" else "supported"
if case == "unknown" and name == "mobilenet_v2":
    status = "unknown"
code = {"supported": 0, "unsupported": 2, "unknown": 3}[status]
report = {"architecture": name, "support_status": status, "exit_code": code,
          "source_onnx_sha256": digest(graph), "candidate_manifest_sha256": digest(manifest),
          "toolchain_image_id": args[args.index("--toolchain-image-id") + 1],
          "reasons": ["Synthetic launcher contract fixture; no hardware or vendor execution."]}
if case == "identity_mismatch":
    report["source_onnx_sha256"] = "f" * 64
if case == "status_mismatch":
    code = 2
if case == "unrecognized":
    report["support_status"] = "maybe"
write(folder / "screen.json", report)
print("Synthetic vendor fixture " + name + " " + status)
sys.exit(code)
'''


@pytest.fixture
def launcher_fixture(tmp_path):
    if sys.platform != "win32":
        pytest.skip("Native Windows PowerShell contract is exercised in Windows CI")
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    # Spaces exercise actual PowerShell/native argument quoting on Windows.
    checkout = tmp_path / "source checkout"
    scripts = checkout / "scripts" / "windows"
    scripts.mkdir(parents=True)
    launcher = scripts / LAUNCHER.name
    shutil.copyfile(LAUNCHER, launcher)
    vendor = checkout / "src" / "radiation_edge_ai" / "nasa_bps" / "kneron_candidate_screen.py"
    vendor.parent.mkdir(parents=True)
    vendor.write_text("# Synthetic launcher fixture, never executed.\n", encoding="utf-8")
    binaries = tmp_path / "fake programs"
    binaries.mkdir()
    driver = binaries / "driver.py"
    driver.write_text(FAKE_DRIVER, encoding="utf-8")
    for command, mode in (("scientific_python", "export"), ("docker", "docker")):
        (binaries / (command + ".cmd")).write_text(
            f'@echo off\r\n"{sys.executable}" "{driver}" {mode} %*\r\nexit /b %errorlevel%\r\n',
            encoding="utf-8",
        )
    # Guarantee duplicate Docker executable matches, even on a machine without
    # Docker installed. Only the first PATH match is the permitted test driver.
    shadow_binaries = tmp_path / "lower priority programs"
    shadow_binaries.mkdir()
    (shadow_binaries / "docker.cmd").write_text(
        f'@echo off\r\n"{sys.executable}" "{driver}" shadow %*\r\nexit /b %errorlevel%\r\n',
        encoding="utf-8",
    )
    capture = tmp_path / "calls.jsonl"
    output = tmp_path / "candidate results"
    # CI starts pytest from PowerShell 7. Python otherwise passes its PS7
    # module paths to Windows PowerShell 5.1, breaking built-in module
    # autoloading (for example Get-FileHash / Microsoft.PowerShell.Utility).
    # Let the native child construct its own default module paths, following:
    # https://learn.microsoft.com/powershell/module/microsoft.powershell.core/about/about_psmodulepath
    env = {key: value for key, value in os.environ.items() if key.upper() != "PSMODULEPATH"}
    env["PATH"] = os.pathsep.join((str(binaries), str(shadow_binaries), env["PATH"]))
    env["RADEDGE_FAKE_CAPTURE"] = str(capture)
    env["PYTHONPATH"] = str(tmp_path / "existing python path")
    return {
        "powershell": powershell,
        "launcher": launcher,
        "python": binaries / "scientific_python.cmd",
        "output": output,
        "capture": capture,
        "env": env,
        "checkout": checkout,
    }


def _run(fixture, case="supported", *extra):
    env = dict(fixture["env"], RADEDGE_FAKE_CASE=case)
    result = subprocess.run(
        [fixture["powershell"], "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(fixture["launcher"]), "-TrainingPython", str(fixture["python"]),
         "-OutputDir", str(fixture["output"]), *extra],
        capture_output=True, text=True, env=env, check=False, timeout=60,
    )
    receipt_file = fixture["output"] / "screen_receipt.json"
    receipt = json.loads(receipt_file.read_text(encoding="utf-8")) if receipt_file.exists() else None
    calls = []
    if fixture["capture"].exists():
        calls = [json.loads(line) for line in fixture["capture"].read_text(encoding="utf-8").splitlines()]
    return result, receipt, calls


def test_full_campaign_uses_pinned_read_only_inputs_and_fresh_containers(launcher_fixture):
    f = launcher_fixture
    result, receipt, calls = _run(f)
    assert result.returncode == 0, result.stdout + result.stderr
    assert receipt["status"] == "COMPLETE_SUPPORTED"
    assert receipt["physical_device_used"] is False
    assert receipt["accuracy_evaluated"] is False
    assert len(receipt["models"]) == 3
    assert all(row["support_status"] == "supported" for row in receipt["models"])
    exporter = calls[0]
    assert exporter["mode"] == "export"
    assert exporter["pythonpath"].split(os.pathsep)[0] == str(f["checkout"] / "src")
    assert not any(call["mode"] == "shadow" for call in calls)
    runs = [call["args"] for call in calls if call["mode"] == "docker" and call["args"][0] == "run"]
    assert len(runs) == 3
    for args in runs:
        assert args[args.index("--pull") + 1] == "never"
        assert args[args.index("--network") + 1] == "none"
        assert args[args.index("--entrypoint") + 2] == IMAGE_ID
        mounts = [args[index + 1] for index, arg in enumerate(args) if arg == "--mount"]
        assert f"type=bind,source={f['checkout']},target=/repo,readonly" in mounts
        assert f"type=bind,source={f['output'] / 'graphs'},target=/graphs,readonly" in mounts
        assert f"type=bind,source={f['output'] / 'vendor'},target=/results" in mounts
    for row in receipt["models"]:
        report = f["output"] / row["screen_report"]
        assert hashlib.sha256(report.read_bytes()).hexdigest() == row["screen_report_sha256"]


@pytest.mark.parametrize(
    "case,status,exit_code",
    [("unsupported", "COMPLETE_WITH_UNSUPPORTED", 2),
     ("unknown", "COMPLETE_WITH_UNKNOWN", 3),
     ("missing_report", "COMPLETE_WITH_UNKNOWN", 3),
     ("identity_mismatch", "COMPLETE_WITH_UNKNOWN", 3),
     ("status_mismatch", "COMPLETE_WITH_UNKNOWN", 3),
     ("unrecognized", "COMPLETE_WITH_UNKNOWN", 3)],
)
def test_uncertain_or_failed_vendor_evidence_cannot_pass(launcher_fixture, case, status, exit_code):
    result, receipt, calls = _run(launcher_fixture, case)
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert receipt["status"] == status
    assert any(row["support_status"] != "supported" for row in receipt["models"])
    assert len([call for call in calls if call["mode"] == "docker" and call["args"][0] == "run"]) == 3


@pytest.mark.parametrize("case", ["missing_image", "daemon_failure"])
def test_docker_preflight_failure_keeps_exports_and_records_unknown(launcher_fixture, case):
    result, receipt, calls = _run(launcher_fixture, case)
    assert result.returncode == 3, result.stdout + result.stderr
    assert receipt["status"] == "COMPLETE_WITH_UNKNOWN"
    assert all(row["support_status"] == "unknown" and not row["vendor_invoked"] for row in receipt["models"])
    assert not any(call["mode"] == "docker" and call["args"][0] == "run" for call in calls)
    assert (launcher_fixture["output"] / "graphs" / "candidate_graphs.json").exists()


def test_export_only_never_invokes_docker(launcher_fixture):
    result, receipt, calls = _run(launcher_fixture, "supported", "-ExportOnly")
    assert result.returncode == 0, result.stdout + result.stderr
    assert receipt["status"] == "EXPORTED_ONLY"
    assert all(row["support_status"] == "unknown" for row in receipt["models"])
    assert [call["mode"] for call in calls] == ["export"]


@pytest.mark.parametrize("case", ["export_failure", "export_hash_mismatch"])
def test_failed_export_prevents_docker_and_saves_failure_receipt(launcher_fixture, case):
    result, receipt, calls = _run(launcher_fixture, case)
    assert result.returncode == 1, result.stdout + result.stderr
    assert receipt["status"] == "FAILED"
    assert receipt["error"]
    assert [call["mode"] for call in calls] == ["export"]


def test_existing_output_is_refused_without_touching_receipt(launcher_fixture):
    f = launcher_fixture
    first, _, _ = _run(f)
    assert first.returncode == 0, first.stdout + first.stderr
    receipt_path = f["output"] / "screen_receipt.json"
    original = receipt_path.read_bytes()
    original_calls = f["capture"].read_bytes()
    second, _, _ = _run(f)
    assert second.returncode != 0
    assert receipt_path.read_bytes() == original
    assert f["capture"].read_bytes() == original_calls
