# Perilipin2 M3C — Windows / Alienware KL720 engineering integration

**Status:** experimental source integration. The M3B FP32 prototype completed training with eight reviewed calibration patches and ONNX numerical parity on Ubuntu, but **has no reviewed held-out nuclear accuracy score**, no KL720 toolchain evaluation, no BIE, no NEF, no physical KL720 inference and no Windows runtime verification yet. Do **not** include Perilipin2 in the repository's validated assay registry until its own biological endpoint gates are met.

## Architecture and portable deployment contract

```text
Local Windows data directory (D:, not Git)
  M3B parity-verified ONNX + ONNX_PARITY.json
          |
          v
PowerShell GUI-ready controller -> standard-library Python intake checks
          |
          v
Kneron toolchain Docker (KL720 graph optimizer + evaluator)
          |
          v
ONNX Runtime CPU original-vs-optimized numeric parity
          |
          v
[Future gated steps] RGB calibration tensors -> INT8 BIE -> BIE parity ->
  conditional NEF compilation -> physical KL720 inference (known-good PLUS Python)
          |
          v
[Future] CPU foreground/boundary -> nuclei instance reconstruction ->
  frozen Perilipin2 Track A M2 brown object/counting/reporting
```

Only **neural model inference** is intended to occur on KL720. Image decoding, RGB normalization, tissue/exclusion masks, watershed instance reconstruction, DAB stain separation, brown-object classification, quantitative tables and HTML dashboard remain on CPU. No scientific endpoint validation is inferred from ONNX/Docker success.

The integration intentionally uses `src/radiation_edge_ai/perilipin2/` without modifying the frozen NASA or DNAi adapters or the active `feature/v0.15-dnai-int16out-preflight` branch.

## 1. Move the TWO frozen model files, once

Use a secure lab-approved local transfer (Windows file share/USB) to copy these two already-generated files from Ubuntu:

```text
/home/liuyuchen/DNA_fiber_assay/analysis/Perilipin2_EdgeAI_M3B_ONNX_v1/
    nuclear_foreground_boundary_512_opset11.onnx
    ONNX_PARITY.json
```

On the Alienware, store them at:

```text
D:\radiation-edge-ai-data\models\perilipin2\m3b_poc_v1\
```

**Do not retrain or run evaluation on the Ubuntu workstation.** The model export is already complete; Windows becomes the engineering execution host. The source model SHA256 must agree with the unchanged `ONNX_PARITY.json`. The two image fields, M1 nuclei, M2 analysis results, training masks, and the model ONNX itself must **never** be uploaded to the public repository.

## 2. Windows prerequisites — one-time

In **PowerShell on the Alienware**, with Docker Desktop already running and its Kneron image already locally available:

```powershell
cd C:\Users\yul03\Projects\radiation-edge-ai
.\scripts\windows\configure_storage.ps1
.\scripts\windows\probe_kneron_toolchain.ps1
```

Install host *engineering* requirements into a dedicated Python environment with ONNX, ONNX Runtime and NumPy, not the pinned Kneron PLUS hardware environment. For example, from an already-created Python environment:

```powershell
python -m pip install -e ".[perilipin2]"
```

Do not reinstall, upgrade, or import `kp` in this environment; physical inference remains isolated to the separately tested PLUS interpreter. If your Docker image uses a nonstandard tag, pass its exact **already installed** reference via `-Image`. No automatic image pulls occur.

## 3. Run the first Windows-only gates

```powershell
cd C:\Users\yul03\Projects\radiation-edge-ai

# Read-only environment and local Docker image inventory.
.\scripts\windows\run_perilipin2_m3c.ps1 -Stage Doctor

# Fingerprint the exact ONNX and parity receipt, copy to a new D: work directory.
.\scripts\windows\run_perilipin2_m3c.ps1 -Stage Import `
  -SourceFolder 'D:\radiation-edge-ai-data\models\perilipin2\m3b_poc_v1'

# Existing Kneron Docker image: optimize + evaluate ONLY; no PTQ/BIE/NEF.
.\scripts\windows\run_perilipin2_m3c.ps1 -Stage Evaluate `
  -Image 'kneron/toolchain:latest'

# CPU numerical check between original and Kneron-optimized ONNX.
.\scripts\windows\run_perilipin2_m3c.ps1 -Stage Parity
```

Results are under `D:\radiation-edge-ai-data\scratch\perilipin2_m3c_poc_v1`. The workflow is **new-only**: never delete results to rerun an earlier stage; use a new `-RunName` such as `perilipin2_m3c_trial2`. The Docker image is inspected for its actual image ID and invoked by that immutable local ID with `--pull never`; any difference between model bytes and the frozen manifest stops processing. The evaluator uses the **validated Kneron Toolchain v0.33.1 interpreter** at `/workspace/miniconda/bin/python` (Python 3.7.15, observed ONNX 1.7.0), rather than `/usr/bin/python3`. `Doctor` probes that interpreter for ONNX, KTC and **either** the modern `kneronnxopt.optimize` or the legacy `ktc.onnx_optimizer.onnx2onnx_flow` interface, as in the already validated NASA/KL720 pipeline. The `Evaluate` stage repeats that probe before starting model processing; it prints progress live while saving the same lines to `WINDOWS_DOCKER_CONSOLE.txt`. Do not install additional ONNX packages into system Python in Docker. The Docker helper is copied into the work directory and hash bound at intake.

**November 10 Windows CP932 / compiler-front-end diagnostic:** A real KL720
evaluation reached `kdp720/compiler frontend`, printed
`Failure for model "input/input"`, then the *Windows host* incorrectly
decoded UTF-8/binary Docker output as Japanese `cp932` and crashed with
`UnicodeDecodeError`. The runner now streams raw binary bytes without the
Windows text decoder. It preserves both `WINDOWS_DOCKER_CONSOLE.raw` (exact
bytes; never published) and `WINDOWS_DOCKER_CONSOLE.txt` (UTF-8 with
undecodable bytes escaped as `\\xNN`), and always writes
`WINDOWS_DOCKER_EXECUTION.json` after Docker returns. An explicit
`Failure for model` is a blocked compiler gate, **even when the vendor
process exits zero or writes an evaluation receipt**. The Docker helper
attempts to harvest fresh `batch_compile.log`, `backtrace.log` and model
reports while its transient container is still alive, including when vendor
`evaluate()` throws; it then writes `M3C_KNERON_FAILURE.json`.
The raw failure log and reports stay under D: and must not be committed.

**Confirmed on 2026-11-10:** The Kneron v0.33.1 vendor table showed
`kdp720/compiler frontend Err: 134` while its `ktc.ModelConfig.evaluate()`
returned a success exit code. The first diagnostic runner already blocked the
outer Windows gate, but still left an internally misleading
`M3C_KNERON_EVAL_RECEIPT.json`. New runs now recognize nonzero `Err: N` table
codes before writing any completion receipt; they write
`M3C_KNERON_FAILURE.json` instead. The existing failed run and its
`backtrace.log`/`model_fx_report.json` should be inspected **without
rerunning or deleting it**. This error code alone does not identify whether
the cause is an unsupported graph structure, compiler bug or resource abort.

This **does not fix Kneron compiler compatibility**. It only unmasks the real
vendor error so the next action can be based on evidence. If the compiler
rejects the graph, do not run Parity, PTQ, BIE, NEF or hardware inference.
Use a brand-new `-RunName` when retrying after a runner update because
`M3C_HOST_READY.json` hash-freezes the copied runner.

**After a failed evaluation:** preserve the failed work directory and its `WINDOWS_DOCKER_EXECUTION.json` / `WINDOWS_DOCKER_CONSOLE.txt`; its runner was copied and hash-frozen at import. After changing the repository runner, first fetch the branch and create an **entirely new** `-RunName` using the original source folder (`-Stage Import`), then `-Stage Evaluate`. Never edit the copied runner or delete the older result to force reuse. For example:

```powershell
git fetch origin feature/perilipin2-windows-m3c-poc
git merge --ff-only FETCH_HEAD  # update only if the worktree has no diverging commits
$Python = 'D:\radiation-edge-ai-data\envs\perilipin2-m3c\Scripts\python.exe'
$RunName = 'perilipin2_m3c_miniconda_v3'
.\scripts\windows\run_perilipin2_m3c.ps1 -Stage Doctor -Python $Python
.\scripts\windows\run_perilipin2_m3c.ps1 -Stage Import -RunName $RunName -Python $Python -SourceFolder 'D:\radiation-edge-ai-data\models\perilipin2\m3b_poc_v1'
.\scripts\windows\run_perilipin2_m3c.ps1 -Stage Evaluate -RunName $RunName -Python $Python
```

For device discovery, separate from Docker/model graph evaluation:

```powershell
.\scripts\windows\run_perilipin2_m3c.ps1 -Stage HardwareProbe `
  -Kl720Python 'C:\Users\yul03\venvs\kneron720\Scripts\python.exe'
```

Adjust that interpreter to the **actual** validated Kneron PLUS Python on your machine. Hardware probing scans devices only; it does not load a Perilipin2 NEF.

## Doctor that appears to hang

The Doctor is not physical KL720 inference: it checks Docker, the installed image,
and a Kneron Python import/optimizer interface. Earlier M3C versions silently
captured the Docker import output and had no subprocess timeout. On a slow or blocked
vendor import this appeared as an indefinitely frozen PowerShell prompt.

The current Doctor prints `[Doctor] Checking Docker daemon...`,
`[Doctor] Inspecting local Kneron image...`, and
`[Doctor] Importing Kneron toolchain and checking optimizer...`.
The first two Docker metadata subprocesses have 20-second limits; the last
Kneron Docker import probe has a 90-second limit. On that timeout only the
uniquely named temporary `radedge-perilipin2-probe-*` container is targeted
for cleanup; the original ONNX, existing runs, DNAi and NASA assets are untouched.
The JSON reports `docker_toolchain_python_ready: false` and
`M3C_TOOLCHAIN_PROBE_TIMEOUT`. Do **not** advance to Evaluate when that
field is false. The actual model optimizer may still take longer after the
preflight passes; the model evaluation has no engineering performance promise.

A Doctor retry only requires fetching the updated worktree; no M3B model re-import.
If a previous Evaluate attempt was made, its imported runner is separately
hash-frozen and must be preserved. Re-import under a new `-RunName` after
updating the repository runner; never edit the frozen runner in place.

## 4. Inspection and evidence gates

- `M3C_HOST_READY.json`: parity result SHA256, exact 1×3×512×512 input and 1×2×512×512 logits output, graph operators and runner identity.
- `WINDOWS_DOCKER_EXECUTION.json` and `WINDOWS_DOCKER_CONSOLE.txt`: actual Docker image ID, exit code and full console log.
- `M3C_KNERON_EVAL_RECEIPT.json`, `model_fx_report.html` and `ioinfo.csv` when available: vendor graph result, reported CPU partitions and unsupported operators. **No report = no NPU-only claim**; all Kneron results require manual inspection.
- `M3C_OPTIMIZED_FP32_PARITY.json`: numeric differences on deterministic synthetic normalized input; does not substitute for real-image or biological evaluation.

The Kneron evaluator may reject M3B's skip-connected nearest-neighbor decoder. In that case preserve the exact toolchain error and work directory; do not rush to BIE or compile. A model that requires CPU neural operators is **not** NPU-only.

## 5. What is deliberately not finished yet

Passing the Windows M3C check will **not** make a Perilipin2 application deployable. We still need a locked calibration set (without validation leakage), Kneron fixed-point BIE/float parity, NEF smoke on the physical KL720, CPU instance reconstruction and *exact M2 downstream endpoint* comparisons with properly reviewed masks. The prior two field images are not biological replicates. The 78.7-second M2 CPU reference replay excluded nuclear segmentation and is **not** a GPU/NPU performance benchmark.

Future GUI integration should call this same gate-oriented Python controller and read the JSON receipts; do not duplicate model science in the GUI. A local folder picker, device status, image preview, run button and QC/export would be a thin front end over this validated pipeline. Do not promise one-click installation until a clean Windows lab-member PC has been tested.