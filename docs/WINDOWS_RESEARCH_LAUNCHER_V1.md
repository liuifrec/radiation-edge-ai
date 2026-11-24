# Windows research launcher v1 (experimental)

## Purpose and exact scope

Provide one local desktop entry point over the **existing** radiation-edge-ai
CLI, without changing model inference, image preprocessing, endpoint algorithms,
artifact identity, hardware SDK, or the scientific claims of any assay.
This is an integration prototype for a prepared-data workflow, not a raw-image
analysis application, automatic installer, or new hardware validation result.
It is independent of experimental Perilipin2 PR #26 and starts from main
`e9d8a1a120e2fe626c9b155d7e1f094e6e5797d8`.

### Current evidence and supported routes

| Task | Desktop action | Existing underlying implementation |
| --- | --- | --- |
| Inspect paths/import availability | Environment check | `radedge doctor --json`; no USB scan |
| Discover KL720 through chosen PLUS Python | Scan KL720 | existing `hardware/kl720_check.py`; no model load |
| NASA or DNAi inference for a prepared frozen plan | Run frozen plan | `radedge run`; CPU or physical KL720 as explicitly recorded in the plan |
| DNAi prepared-field execution and reconstruction | Run assay manifest | `radedge assay-run`; preserves its scientific-runtime and PLUS-runtime separation |
| NASA batch-manifest execution | Run assay manifest | existing CPU-only NASA batch adapter; **not** silently upgraded to KL720 |
| Read-only core artifact integrity | Verify core record/package | `radedge verify --json`; core hash verification remains authoritative |
| Export a supported completed assay transaction | Export assay transaction | `radedge assay-report`; no inference |

The NASA KL720 route is a frozen run plan, not its CPU-only batch-manifest
adapter. Do not reinterpret NASA latent continuous 53BP1 burden as per-cell
focus counts or apply it to PNKP samples without a separate validation.
DNAi physical execution exists, but broad biological-fidelity validation is not
established by this launcher. See `UNIFIED_KL720_V0_11_RELEASE.md` and
`NASA_BPS_R1_V2_KL720_VALIDATION_RECORD.md` for the scope of existing evidence.

**Perilipin2 and PNKP are not enabled desktop KL720 assays.** Successful
single-operator ConvTranspose compiler tests do not demonstrate that the
full Perilipin2 graph compiles, that a NEF was built, that the device executes
it, or that microscopy endpoints are accurate. No compiler status is promoted
into the runtime assay registry by this work.

## Components

- `src/radiation_edge_ai/desktop.py`: local Tk interface, command preview,
  user-selected interpreter paths, explicit hardware confirmation, one active
  job per window, live logs, and local settings.
- `src/radiation_edge_ai/desktop_runtime.py`: approved CLI command construction,
  source-change checks, separate process execution, exact-byte console log,
  UTF-8-readable log, and separate launcher receipts.
- `scripts/windows/start_radedge_desktop.ps1`: binds the host interpreter to
  this worktree's source and launches the GUI. No pip installs, Docker startup,
  model downloads, firmware changes or hardware actions happen on opening it.
- `tests/test_desktop_runtime.py`: routing, unknown-assay rejection, explicit
  hardware selection, output preservation, JSON safety and real subprocess
  byte-logging tests. The repository CI also parses the PowerShell entrypoint
  and imports Tk on Windows; physical hardware remains outside CI.

## Start on the Alienware without disturbing working branches

Create a separate worktree; do not switch, reset or clean the current
Perilipin2 or DNAi worktree. The new worktree can remain detached for review.

```powershell
cd C:\Users\yul03\Projects\radiation-edge-ai

git fetch origin feature/windows-research-launcher-v1
if ($LASTEXITCODE -ne 0) { throw 'Fetch failed; stop here.' }

$Desktop = 'C:\Users\yul03\Projects\radiation-edge-ai-desktop'
if (Test-Path -LiteralPath $Desktop) { throw 'Desktop worktree already exists; do not overwrite it.' }
git worktree add --detach $Desktop FETCH_HEAD
if ($LASTEXITCODE -ne 0) { throw 'Worktree creation failed; stop here.' }

cd $Desktop
$Python = 'D:\radiation-edge-ai-data\envs\perilipin2-m3c\Scripts\python.exe'
.\scripts\windows\start_radedge_desktop.ps1 -Python $Python -DataRoot 'D:\radiation-edge-ai-data'
```

The host interpreter needs the existing core dependencies (NumPy) and Tk.
The user's existing Perilipin2 host environment is a possible host, **not**
the Kneron PLUS interpreter and **not** the DNAi scientific interpreter.
No package installation into the known-good PLUS environment is required.
If Tk is absent, the launcher fails on import; do not modify the SDK environment
to fix a host-GUI dependency. The ordinary CLI remains available.

## Initial acceptance sequence

1. Open the GUI and run **Environment check**. This reads local paths and
   reports dependencies. It does not test Docker or touch the KL720.
2. Select the **existing known-good Kneron PLUS Python** using Browse (or
   `RADEDGE_KL720_PYTHON` inherited from the shell), save settings explicitly,
   then choose **Scan KL720** and confirm. Use the actual interpreter from
   the successful DNAi/NASA runs; the GUI deliberately does not guess it.
3. Choose a prepared, frozen `run_plan.json` or DNAi assay manifest. For a
   DNAi field manifest, also select its existing scientific runtime using
   Scientific/ONNX Python. Preview the command and verify the intended
   backend before confirming execution. Close other KL720 applications first.
4. Inspect the core result, verify it, and export a supported completed
   transaction to a fresh directory. Existing result reuse is delegated to
   the original core implementation and must not be reported as a new
   physical inference run.

A successful first GUI launch is not an end-to-end acceptance test. Final
acceptance needs one **new**, provenance-verified physical KL720 execution
through the GUI on the Alienware and a matched check of the existing
assay endpoint/report. A cached result only tests reuse, not new inference.
Use new scientific run plans/assay output folders through the normal core
procedures; do not delete or edit frozen records to force recomputation.

## Storage and safety

Local settings, when the user clicks Save, go to:
`D:\radiation-edge-ai-data\config\desktop.json`.
Each explicit action has a new UUID-suffixed folder under
`D:\radiation-edge-ai-data\desktop_jobs`, containing:

- `request.json`: argv preview, selected assay/backend and selected-file SHA256;
- `console.raw`: exact subprocess bytes;
- `console.txt`: readable UTF-8 with invalid bytes escaped;
- `launcher_receipt.json`: exit status and log hash.

A zero subprocess status is called `COMMAND_RETURNED_REVIEW_CORE_RECORDS`,
not biological validation. Existing core run/field/transaction records remain
the authoritative inference and measurement evidence. The launcher does not
produce new NEFs, quantify speedups, or claim that the hardware was used merely
because a job was selected. Physical access is explicit; no GPU/CPU fallback
is substituted for a KL720 request.

All job logs are local and may contain file paths or experimental metadata.
They are **not automatically uploaded**. Do not publish them without review.
The new UI never sends images or weights to GitHub or an external service.
One window runs one job at a time; there is no cross-application USB lock.
The GUI does not forcibly terminate an active scientific/SDK process: keep
the window open while it runs. Existing per-request KL720 timeouts remain
in force, but whole-job cancellation and watchdogs are future work.

## Validation status

Local tests before submission: routing and real byte-logging tests passed
(16 tests, 5 subtests; one full-core CLI parsing test was skipped because the
isolated local source copy did not include the full repository). The GUI was
opened, visually inspected and closed in a Linux virtual display; opening it
created no data/settings directory. This is not a Windows user acceptance test.
The full-repository CI checks the real CLI parser plus the new tests. Consult
the PR checks for actual current results, not this document for an assumed pass.

Not yet validated: Alienware GUI interaction, new physical KL720 execution
through this UI, a clean laboratory-PC install, arbitrary raw-image ingestion,
unified NASA KL720 batch manifests, Perilipin2 device deployment, and complete
PNKP/Perilipin2 dataset-to-endpoint integration. The branch stays draft until
those applicable acceptance checks are completed.

## Engineering direction after this milestone

Keep the shared shell (configuration, registry, device runtime, provenance,
logs, verification/export) separate from assay science. Compiler experiments
belong to the developer/model-build path, not the routine laboratory run path.
Ordinary lab users should ultimately use prebuilt, versioned model bundles;
they should not have to run Kneron Docker or manually edit ONNX graphs.
