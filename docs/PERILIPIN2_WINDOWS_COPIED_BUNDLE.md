# Run Track A after copying the reference bundle to Windows

This entry point uses the existing Track A numerical adapter and its exact
reference comparison. It does not change scientific thresholds, measurements,
saved masks, model weights, the desktop UI, or the KL720 SDK environment.

## Defaults matching the transferred folder

- Input: `D:\radiation-edge-ai-data\references\Perilipin2_TrackA_portable_20261111_v1`
- Default scientific environment: `D:\radiation-edge-ai-data\envs\perilipin2-tracka`
- Results: a NEW uniquely named directory under `D:\radiation-edge-ai-data\scratch`

A folder visible in Explorer does not prove transfer integrity. The command
verifies the bundle fingerprint and file hashes before creating an environment
or starting the analysis.

## November 11 setup correction: do not inherit an incompatible launcher Python

The copied input bundle passed verification with fingerprint
`80918290134fb014adcdd10f5ef6c4ca6875607f379dcba970a32f785f75cf2a`.
Setup then failed installing the unchanged pin `numpy==2.4.6`; pip listed
versions only through 2.0.2. No image measurements were run.

The earlier helper used `sys.executable -m venv`, so its new environment
inherited the launcher's Python version. That was a setup design defect:
the launcher/SDK interpreter may be older than the scientific requirements.
The version listing is consistent with Python 3.9, but is not a direct
version report. NumPy's official 2.4.6 notes specify Python 3.11-3.14:
https://numpy.org/doc/stable/release/2.4.6-notes.html
Python documents that venv uses the invoking Python installation:
https://docs.python.org/3.11/library/venv.html

The helper now accepts `--base-python`. Before creating a NEW environment,
it probes that executable with a bounded read-only subprocess and requires
64-bit CPython 3.11-3.14. Without an explicit base, it still checks the launcher
and stops early if incompatible; it never silently chooses a different runtime
or downloads Python. An existing worker is checked separately before package
imports. This version-range check is not proof that every recorded package
wheel is available on the configured index: pip still verifies that normally.

**Preserve the previous incomplete `perilipin2-tracka` folder.** The recovery
below selects a fresh `perilipin2-tracka-py311-v2` destination. Do not delete
old environments, downgrade NumPy, edit the reference pins or reinstall Docker.

## Update the desktop worktree, not the compiler worktree

The user's earlier DNAi workflow used
`D:\radiation-edge-ai-data\envs\dnai311\Scripts\python.exe`.
Use it only as the base for creating a separate venv; no packages are installed
into `dnai311`, `perilipin2-m3c`, or the Kneron PLUS runtime. The helper verifies
its actual interpreter version rather than trusting the folder name.

```powershell
$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\yul03\Projects\radiation-edge-ai-desktop'

git fetch origin feature/perilipin2-tracka-analysis-v1
if ($LASTEXITCODE -ne 0) { throw 'Fetch failed. Stop here.' }

git merge --ff-only FETCH_HEAD
if ($LASTEXITCODE -ne 0) { throw 'Update refused. Preserve local changes.' }

$HostPython = 'D:\radiation-edge-ai-data\envs\perilipin2-m3c\Scripts\python.exe'
$AnalysisBase = 'D:\radiation-edge-ai-data\envs\dnai311\Scripts\python.exe'
$AnalysisEnv = 'D:\radiation-edge-ai-data\envs\perilipin2-tracka-py311-v2'

& $HostPython -c 'import sys; print("Launcher Python:", sys.version); print(sys.executable)'
if ($LASTEXITCODE -ne 0) { throw 'Host Python check failed.' }

& $HostPython -X utf8 .\scripts\windows\accept_perilipin2_tracka.py `
    --base-python $AnalysisBase `
    --environment $AnalysisEnv `
    --create-environment
if ($LASTEXITCODE -ne 0) { throw 'Acceptance did not pass. Keep the terminal output and result folders.' }
```

If that previously used base executable no longer exists, the helper stops
before creating the new venv. `py -0p` lists installed Windows Python paths;
use an actual installed compatible executable for `--base-python`. Do not
substitute Docker's `/workspace/miniconda/bin/python`, and do not upgrade the
known-good PLUS environment to satisfy the measurement dependencies.

The switch explicitly authorizes initial creation of a separate analysis
venv and installation of the six package versions recorded in the verified
reference bundle. It does NOT authorize changing an existing environment.
Package downloads require access to the configured package index. Installation
uses wheel packages only; a missing compatible wheel is reported, not replaced
by an unrecorded version or a source build.

When the analysis environment already exists, the command checks its interpreter,
location, package versions, imports and dependency consistency. It never installs
or upgrades anything into that existing environment. `--base-python` is used
only for NEW environment creation; it cannot change an existing venv's Python.
Preserve any incomplete environment and inspect the diagnostic.

For read-only transfer verification, use:

```powershell
& $HostPython -X utf8 .\scripts\windows\accept_perilipin2_tracka.py --check-only
```

## What happens after setup

The adapter reads the real RGB images, reuses the saved nuclear/exclusion masks,
recomputes the brown-object measurements, tables, quantitative arrays and QC,
and compares against the original M2 reference. It then verifies the result
file manifest. No cached analysis is substituted for this newly generated run.
The original bundle is never overwritten; each run has a fresh output path.

Expected terminal milestones for new-environment setup:

```text
TRACKA_COPIED_BUNDLE_VERIFIED
TRACKA_ANALYSIS_BASE_VERIFIED
TRACKA_SCIENTIFIC_ENVIRONMENT_VERIFIED
TRACKA_REFERENCE_REPLAY_PASS
TRACKA_OUTPUT_INTEGRITY_PASS
TRACKA_WINDOWS_REFERENCE_ACCEPTANCE_PASS
```

The terminal prints the exact paths to `TRACKA_EXECUTION.json`,
`REFERENCE_COMPARISON.json` and the generated `index.html` viewer. A mismatch,
worker error or dependency failure stops this sequence. Preserve the output
or `.BUILDING` folder and share the diagnostic; do not loosen comparison rules.

This remains CPU measurement/replay with saved nuclei, NOT fresh segmentation,
a completed KL720 Perilipin2 deployment, or proof of biological detection accuracy.
The end-to-end real-image acceptance must run on the local Windows PC. GitHub
write access and passing CI do not provide access to its D: drive or USB device.

## Test scope

`tests/test_tracka_windows_acceptance.py` checks setup authorization, strict
package allowlisting, environment isolation, no installation into existing
environments, worker error propagation and read-only mode. The new tests cover
explicit base selection, Python 3.9 rejection before mkdir/pip, 32-bit and
out-of-range interpreter rejection, preservation of incomplete environments,
and protection of the selected base runtime. Tests mock package installation
and analysis: they do not download scientific packages or process private
research images. They run in the Linux offline suite and Windows CI.
Existing frozen-source-dependent numerical tests retain their separate scope.
