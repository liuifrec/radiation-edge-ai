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

Results are under `D:\radiation-edge-ai-data\scratch\perilipin2_m3c_poc_v1`. The workflow is **new-only**: never delete results to rerun an earlier stage; use a new `-RunName` such as `perilipin2_m3c_trial2`. The Docker image is inspected for its actual image ID and invoked by that immutable local ID with `--pull never`; any difference between model bytes and the frozen manifest stops processing. The evaluator uses the **validated Kneron Toolchain v0.33.1 interpreter** at `/workspace/miniconda/bin/python` (Python 3.7.15, observed ONNX 1.7.0), rather than `/usr/bin/python3`. `Doctor` probes that interpreter with `import onnx, ktc, kneronnxopt` before an evaluation attempt. Do not install additional ONNX packages into system Python in Docker. The Docker helper is copied into the work directory and hash bound at intake.

**After a failed evaluation:** preserve the failed work directory and its `WINDOWS_DOCKER_EXECUTION.json` / `WINDOWS_DOCKER_CONSOLE.txt`; its runner was copied and hash-frozen at import. After changing the repository runner, first fetch the branch and create an **entirely new** `-RunName` using the original source folder (`-Stage Import`), then `-Stage Evaluate`. Never edit the copied runner or delete the older result to force reuse. For example:

```powershell
git fetch origin feature/perilipin2-windows-m3c-poc
git switch feature/perilipin2-windows-m3c-poc
git pull --ff-only
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

## 4. Inspection and evidence gates

- `M3C_HOST_READY.json`: parity result SHA256, exact 1×3×512×512 input and 1×2×512×512 logits output, graph operators and runner identity.
- `WINDOWS_DOCKER_EXECUTION.json` and `WINDOWS_DOCKER_CONSOLE.txt`: actual Docker image ID, exit code and full console log.
- `M3C_KNERON_EVAL_RECEIPT.json`, `model_fx_report.html` and `ioinfo.csv` when available: vendor graph result, reported CPU partitions and unsupported operators. **No report = no NPU-only claim**; all Kneron results require manual inspection.
- `M3C_OPTIMIZED_FP32_PARITY.json`: numeric differences on deterministic synthetic normalized input; does not substitute for real-image or biological evaluation.

The Kneron evaluator may reject M3B's skip-connected nearest-neighbor decoder. In that case preserve the exact toolchain error and work directory; do not rush to BIE or compile. A model that requires CPU neural operators is **not** NPU-only.

## 5. What is deliberately not finished yet

Passing the Windows M3C check will **not** make a Perilipin2 application deployable. We still need a locked calibration set (without validation leakage), Kneron fixed-point BIE/float parity, NEF smoke on the physical KL720, CPU instance reconstruction and *exact M2 downstream endpoint* comparisons with properly reviewed masks. The prior two field images are not biological replicates. The 78.7-second M2 CPU reference replay excluded nuclear segmentation and is **not** a GPU/NPU performance benchmark.

Future GUI integration should call this same gate-oriented Python controller and read the JSON receipts; do not duplicate model science in the GUI. A local folder picker, device status, image preview, run button and QC/export would be a thin front end over this validated pipeline. Do not promise one-click installation until a clean Windows lab-member PC has been tested.