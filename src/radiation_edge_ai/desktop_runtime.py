"""Thin, offline desktop command boundary; assay science stays in existing CLI.

This module never imports kp, torch, ONNX or Docker. Physical inference is
explicit and delegated to the existing frozen-plan/assay adapters. Process
completion is not labelled biological validation or a new device benchmark.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

ASSAYS = {
    "nasa-53bp1-r1-v2": "NASA 53BP1: latent burden, not an individual focus count.",
    "dnai-fiber-v3": "DNA fibers: physical execution exists; biological fidelity unvalidated.",
}
ACTIONS = ("doctor", "device-scan", "run-plan", "run-assay", "verify", "export")
CLI_ENTRY = "from radiation_edge_ai.cli import main; raise SystemExit(main())"


@dataclass(frozen=True)
class Settings:
    host_python: str = sys.executable
    data_root: str = r"D:\radiation-edge-ai-data"
    scientific_python: str = ""
    kl720_python: str = ""
    kl720_port: Optional[int] = None
    kl720_timeout_ms: int = 10000

    def save(self, path: Path) -> None:
        """Explicit user action; atomically save paths/settings, never model data."""
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(asdict(self), handle, indent=2, allow_nan=False)
                handle.write("\n")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: Path) -> "Settings":
        return cls(**read_object(path))


def read_object(path: Path) -> dict:
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Choose a manifest/receipt, not a large data file.")
    def reject(value):
        raise ValueError("Nonfinite JSON value: " + value)
    obj = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject)
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object.")
    return obj


def file_path(value: str, label: str) -> Path:
    if not value.strip():
        raise ValueError("Choose " + label + ".")
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(label + " must be a file.")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def outside_repository(path: Path, repo: Path) -> None:
    if path == repo or repo in path.parents:
        raise ValueError("Keep outputs and local settings outside the Git worktree.")


@dataclass(frozen=True)
class Job:
    action: str
    argv: tuple[str, ...]
    source: Optional[str]
    source_sha256: Optional[str]
    backend: Optional[str]
    assay_id: Optional[str]
    hardware_requested: bool
    output: Optional[str]


def build_job(settings: Settings, repo: Path, action: str,
              source: str = "", output: str = "") -> Job:
    """Only approved CLI actions; never shell-evaluate user or manifest text."""
    repo = repo.expanduser().resolve()
    if action not in ACTIONS:
        raise ValueError("Unknown desktop action: " + action)
    host = file_path(settings.host_python, "host Python interpreter")
    root = Path(settings.data_root).expanduser().resolve()
    outside_repository(root, repo)
    if type(settings.kl720_timeout_ms) is not int or settings.kl720_timeout_ms <= 0:
        raise ValueError("Device timeout must be a positive integer.")
    port = settings.kl720_port
    if port is not None and (type(port) is not int or port < 0):
        raise ValueError("USB port must be blank or a non-negative integer.")
    prefix = [str(host), "-u", "-c", CLI_ENTRY]
    source_path = None
    obj = {}
    assay = backend = out = None
    physical = False
    if action not in ("doctor", "device-scan"):
        source_path = file_path(source, "JSON manifest or receipt")
        obj = read_object(source_path)
    if action == "doctor":
        command = prefix + ["doctor", "--json"]
    elif action == "device-scan":
        plus = file_path(settings.kl720_python, "existing Kneron PLUS Python")
        probe = repo / "src/radiation_edge_ai/hardware/kl720_check.py"
        if not probe.is_file():
            raise ValueError("Device probe is missing from this worktree.")
        command = [str(plus), "-u", str(probe)]
        physical = True  # USB discovery only; no model is loaded by this action.
    elif action == "verify":
        command = prefix + ["verify", str(source_path), "--json"]
    elif action == "export":
        command = prefix + ["assay-report", str(source_path)]
    else:
        if action == "run-plan":
            identity = obj.get("assay")
            if not isinstance(identity, dict) or not obj.get("plan_fingerprint_sha256"):
                raise ValueError("Choose a frozen run_plan.json, not an assay manifest.")
            assay = identity.get("assay_id")
            outside_repository(source_path, repo)
            command = prefix + ["run", str(source_path)]
        else:
            assay = obj.get("assay_id")
            command = prefix + ["assay-run", "--manifest", str(source_path), "--json"]
        backend = obj.get("backend")
        if assay not in ASSAYS:
            raise ValueError("No enabled desktop assay adapter: {!r}. "
                             "Perilipin2/PNKP are not enabled KL720 assays.".format(assay))
        if backend not in ("cpu-onnx", "kl720"):
            raise ValueError("Choose an explicit cpu-onnx or kl720 backend manifest.")
        if action == "run-assay" and assay == "nasa-53bp1-r1-v2" and backend == "kl720":
            raise ValueError("NASA batch-manifest dispatch is CPU-only on this baseline. "
                             "Use Run frozen plan for existing NASA KL720 plans; no fallback applied.")
        if action == "run-assay" or backend == "cpu-onnx":
            scientific = file_path(settings.scientific_python, "existing scientific/ONNX Python")
            command += ["--onnx-python", str(scientific)]
        if backend == "kl720":
            plus = file_path(settings.kl720_python, "existing Kneron PLUS Python")
            command += ["--kl720-python", str(plus), "--kl720-timeout-ms",
                        str(settings.kl720_timeout_ms)]
            if port is not None:
                command += ["--kl720-port", str(port)]
            physical = True
    if action in ("run-assay", "export"):
        if not output.strip():
            raise ValueError("Choose a NEW output folder.")
        destination = Path(output).expanduser().resolve()
        outside_repository(destination, repo)
        if destination.exists():
            raise ValueError("Output exists. Preserve it and choose a new folder.")
        if source_path and destination in source_path.parents:
            raise ValueError("Output cannot contain the selected source.")
        out = str(destination)
        command += ["--output-dir", out]
    return Job(action, tuple(command), str(source_path) if source_path else None,
               sha256(source_path) if source_path else None, backend, assay, physical, out)


def run_job(job: Job, settings: Settings, repo: Path,
            emit: Callable[[str], None] = lambda text: None) -> Path:
    """Run one explicit job; preserve byte logs and a separate launcher receipt.

    Readiness is not assumed from exit code zero. Existing CLI verification and
    scientific records remain authoritative. No automatic retry or CPU fallback.
    """
    repo = repo.expanduser().resolve()
    if job.source and sha256(Path(job.source)) != job.source_sha256:
        raise ValueError("Selected source changed since preview. Inspect it again.")
    root = Path(settings.data_root).expanduser().resolve()
    outside_repository(root, repo)
    jobs = root / "desktop_jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:12]
    folder = jobs / name
    folder.mkdir(exist_ok=False)
    request = {"schema": "radedge-desktop-job-v1", **asdict(job)}
    (folder / "request.json").write_text(
        json.dumps(request, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")  # Bind execution to this worktree.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8:backslashreplace"
    env["RADEDGE_DATA_ROOT"] = str(root)
    env["RADEDGE_MODEL_ROOT"] = str(root / "models")
    env["RADEDGE_CACHE_ROOT"] = str(root / "cache")
    if settings.kl720_python:
        env["RADEDGE_KL720_PYTHON"] = settings.kl720_python
    rc = None
    error = None
    start = datetime.now(timezone.utc).isoformat()
    try:
        with (folder / "console.raw").open("xb") as raw, \
                (folder / "console.txt").open("x", encoding="utf-8") as text:
            with subprocess.Popen(list(job.argv), cwd=str(repo), env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  shell=False, bufsize=0) as process:
                assert process.stdout is not None
                for line in process.stdout:
                    raw.write(line)
                    raw.flush()
                    decoded = line.decode("utf-8", errors="backslashreplace")
                    text.write(decoded)
                    text.flush()
                    emit(decoded)
                rc = process.wait()
    except Exception as exc:
        error = "{}: {}".format(type(exc).__name__, exc)
        emit(error + "\n")
    raw_path = folder / "console.raw"
    result = {
        "schema": "radedge-desktop-job-v1", "action": job.action,
        "status": "COMMAND_RETURNED_REVIEW_CORE_RECORDS" if rc == 0 and not error else "FAILED",
        "started_utc": start, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "exit_code": rc, "error": error,
        "source_sha256": job.source_sha256,
        "hardware_requested": job.hardware_requested,
        "physical_inference_verified_by_launcher": False,
        "biological_validation_performed_by_launcher": False,
        "raw_console_sha256": sha256(raw_path) if raw_path.exists() else None,
        "note": "Use the existing core run/transaction/package records as execution evidence.",
    }
    receipt = folder / "launcher_receipt.json"
    receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    emit("\nLauncher receipt: " + str(receipt) + "\n")
    return receipt
