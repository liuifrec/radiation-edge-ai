# Run Track A after copying the reference bundle to Windows

This entry point uses the existing Track A numerical adapter and its exact
reference comparison. It does not change scientific thresholds, measurements,
saved masks, model weights, the desktop UI, or the KL720 SDK environment.

## Defaults matching the transferred folder

- Input: `D:\radiation-edge-ai-data\references\Perilipin2_TrackA_portable_20261111_v1`
- Dedicated scientific environment: `D:\radiation-edge-ai-data\envs\perilipin2-tracka`
- Results: a NEW uniquely named directory under `D:\radiation-edge-ai-data\scratch`

A folder visible in Explorer does not prove transfer integrity. The command
verifies the bundle fingerprint and file hashes before creating an environment
or starting the analysis.

## Update the desktop worktree, not the compiler worktree

```powershell
cd C:\Users\yul03\Projects\radiation-edge-ai-desktop

git fetch origin feature/perilipin2-tracka-analysis-v1
if ($LASTEXITCODE -ne 0) { throw 'Fetch failed. Stop here.' }

git merge --ff-only FETCH_HEAD
if ($LASTEXITCODE -ne 0) { throw 'Update refused. Preserve local changes.' }

$HostPython = 'D:\radiation-edge-ai-data\envs\perilipin2-m3c\Scripts\python.exe'
& $HostPython -X utf8 .\scripts\windows\accept_perilipin2_tracka.py --create-environment
if ($LASTEXITCODE -ne 0) { throw 'Acceptance did not pass. Keep the terminal output and result folders.' }
```

The switch explicitly authorizes the initial creation of a separate analysis
venv and installation of the six package versions recorded in the verified
reference bundle. It does NOT authorize changing an existing environment.
Package downloads require access to the configured package index. Installation
uses wheel packages only; a missing compatible wheel is reported, not replaced
by an unrecorded version or a source build.

When the analysis environment already exists, the command checks its location,
package versions, imports and dependency consistency. It never installs or
upgrades anything into that existing environment. If it is incomplete or differs
from the reference, preserve it and inspect the diagnostic rather than deleting
or silently repairing it. The host and Kneron PLUS environments are not modified.

For read-only transfer verification, omit setup/run and use:

```powershell
& $HostPython -X utf8 .\scripts\windows\accept_perilipin2_tracka.py --check-only
```

## What happens after setup

The adapter reads the real RGB images, reuses the saved nuclear/exclusion masks,
recomputes the brown-object measurements, tables, quantitative arrays and QC,
and compares against the original M2 reference. It then verifies the result
file manifest. No cached analysis is substituted for this newly generated run.
The original bundle is never overwritten; each run has a fresh output path.

Expected terminal milestones:

```text
TRACKA_COPIED_BUNDLE_VERIFIED
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
environments, worker error propagation and read-only mode. Tests mock package
installation and analysis: they do not download scientific packages or process
private research images. They run in the Linux offline suite and Windows CI.
Existing frozen-source-dependent numerical tests retain their separate scope.
