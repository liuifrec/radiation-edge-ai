# NASA: run the KL720, then expand the model collection

The first command below runs the **existing frozen NASA R1 v2 model on the
physical KL720**. Candidate-graph screening and the shared training/export
workflow then support a fair comparison of the compact CNN, MobileNetV2
and ResNet18.

These are separate completion states: the frozen baseline has historical
physical evidence; the additional architectures are candidates until their
NASA accuracy, conversion and physical execution are measured.

## 1. Run the frozen NASA model on the Windows dongle

Open PowerShell in a checkout containing this change and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\run_nasa_kl720.ps1
```

This uses the existing Kneron PLUS Python at
`$env:USERPROFILE\venvs\kneron720\Scripts\python.exe`, or the interpreter named
by `RADEDGE_KL720_PYTHON`. It auto-selects the device when exactly one KL720 is
connected. It uses the existing assets under `D:\radiation-edge-ai-data`.
It does not install or upgrade packages in the PLUS environment.

Useful explicit options:

```powershell
# Check frozen assets, image decoding and device discovery without inference.
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\run_nasa_kl720.ps1 -PreflightOnly

# Select an existing PLUS interpreter, data/model root, or a particular device.
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\run_nasa_kl720.ps1 `
  -Python 'C:\Users\yul03\venvs\kneron720\Scripts\python.exe' `
  -Root 'D:\radiation-edge-ai-data' `
  -Port 81
```

Port 81 is only an example from the historical machine. Automatic discovery
is the default; if multiple KL720 devices are visible, choose one explicitly.
`RADEDGE_DATA_ROOT` and `RADEDGE_MODEL_ROOT`, when set, override the corresponding
data and model subdirectories. The Python entry point also accepts
`--data-root`, `--model-root`, and `--image-root` for relocated files.

### What the run does

1. Verify the exact frozen NEF, deployment freeze, physical smoke freeze and
   2,058-nucleus manifest by SHA256.
2. Hash and decode the 4,116 FITC/DAPI input files. Preserve the historical
   native-scale normalization and 256 x 256 center padding.
3. Connect to the KL720, upload the model once and perform **2,058 new hardware
   inferences**. Print progress every 50 nuclei. Each inference has batch size
   one; 256 is the image canvas dimension.
4. Verify the complete prediction identities and worker result, then confirm
   that the image bytes did not change during the run.
5. If the historical physical prediction CSV exists, verify its frozen hash
   and compare the fresh scalar outputs by nucleus identity using a fixed
   maximum absolute difference of `1e-4`.

Every inference run that passes preflight creates a new directory:

```text
D:\radiation-edge-ai-data\scratch\nasa_kl720_<UTC time>_<unique suffix>\
```

An explicit `-OutputDir` must not already exist. A cached prediction file is
never substituted for a hardware invocation. Hardware errors stop the run;
there is no CPU inference fallback.

### Read the result

| File | Meaning |
| --- | --- |
| `replay_receipt.json` | Run status, verified inference count, device port, completed upload count, hashes, timing and numerical repeatability |
| `input_files.json` | Measured SHA256 and size of every current input TIFF |
| `hardware_run.log` | Live worker output and any runtime error |
| `hardware/physical_holdout_nucleus_predictions.csv` | Fresh physical burden predictions and per-nucleus timing |
| `hardware/physical_holdout_inference_summary.json` | Original worker result, preserved with its historical schema |

Successful console output ends with:

```text
FRESH KL720 REPLAY COMPLETE: 2058 real inferences; 1 model upload.
Historical numerical repeatability: PASS
Receipt: <new directory>\replay_receipt.json
```

`COMPLETE` means the full hardware replay finished and numerical repeatability
passed. `COMPLETE_UNCOMPARED` means hardware finished but the historical CSV
was absent; `numerical_repeatability_verified` is false. A mismatch produces
`NUMERICAL_MISMATCH` and a nonzero exit code. An interrupted or failed run
cannot produce a successful receipt.

Device LEDs are a useful visual cue, but their behavior is not the verification
criterion. The receipt and fresh hardware results establish what completed.
This session cannot operate a USB device attached to another computer.

### Expected existing assets

All paths in this table are relative to `D:\radiation-edge-ai-data` unless the
data/model root overrides are set.

| Asset | Relative path |
| --- | --- |
| Frozen NEF | `models/nasa_bps_53bp1/r1_v2_deployment/kneron_nef/r1_countnet_v2_final_kl720.nef` |
| Deployment freeze | `models/nasa_bps_53bp1/r1_v2_deployment/kneron_nef/nef_deployment_candidate_freeze.json` |
| Physical smoke freeze | `models/nasa_bps_53bp1/r1_v2_deployment/kneron_nef/physical_smoke_hardware/physical_smoke_pass_freeze.json` |
| Historical predictions | `models/nasa_bps_53bp1/r1_v2_deployment/kneron_nef/physical_holdout_hardware/physical_holdout_nucleus_predictions.csv` |
| Manifest | `data/nasa_bps_microscopy/metadata/r1_v2_freeze/r1_v2_final_holdout_manifest_blinded_qc1.csv` |
| Input image root | `data/nasa_bps_microscopy/r1_v2_final_holdout_blinded/images` |

The final `images` component is required. The directory contains `fitc` and
`dapi` subdirectories, with filenames taken directly from the frozen manifest.

This is a **fresh replay of a previously evaluated historical panel**. Its
numerical repeatability test is not a new biological validation. Do not run
the old freeze/unblind procedures on this output. The unchanged worker retains
historical field names, while the new receipt states the current replay scope.
New input-file hashes establish this run's provenance; they do not establish
equality with historical TIFF bytes when those receipts are unavailable.

## 2. Set up the separate scientific environment

Training and ONNX export belong in a scientific Python environment, separate
from the working Python 3.9 Kneron PLUS environment. The new `nasa` extra pins
the PyTorch/torchvision and ONNX dependencies exercised by the software tests.

An existing **64-bit Python 3.9** installation can create the separate NASA
environment. Python 3.11 is also supported, but is not required. The NASA extra
selects ONNX Runtime 1.19.2 on Python 3.9 and 1.20.1 on Python 3.10 or later:
[1.19.2 provides a Windows Python 3.9 wheel](https://pypi.org/project/onnxruntime/1.19.2/),
whereas [1.20.1 does not](https://pypi.org/project/onnxruntime/1.20.1/).
The other model/export pins remain identical. Export manifests record the
actual dependency versions, and the same numerical parity requirements apply.

From the repository root, use this block when the Windows Python launcher
lists Python 3.9. The outer block stops the entire setup at the first failure:

```powershell
$TrainPython = 'D:\radiation-edge-ai-data\envs\nasa39\Scripts\python.exe'
& {
    $ErrorActionPreference = 'Stop'
    if (-not (Test-Path -LiteralPath $TrainPython)) {
        py -3.9 -m venv 'D:\radiation-edge-ai-data\envs\nasa39'
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the NASA Python 3.9 environment.' }
    }
    & $TrainPython -c "import sys, struct; assert sys.prefix != sys.base_prefix; assert struct.calcsize('P') == 8; print(sys.version)"
    if ($LASTEXITCODE -ne 0) { throw 'A separate 64-bit scientific environment is required.' }
    & $TrainPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'NASA pip update failed.' }
    & $TrainPython -m pip install -e '.[nasa]'
    if ($LASTEXITCODE -ne 0) { throw 'NASA scientific dependency installation failed.' }
    & $TrainPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'NASA dependency validation failed.' }
    & $TrainPython -m radiation_edge_ai.nasa_bps.multimodel list
    if ($LASTEXITCODE -ne 0) { throw 'NASA model discovery failed.' }
}
```

Run these commands from the repository root. A compatible existing scientific
environment can also be used with its explicit interpreter path. For an
installed Python 3.11, use `py -3.11` and a separate `nasa311` directory instead.
Upgrading pip happens only inside the selected scientific environment so an
older bundled pip can install the editable package. The `nasa` extra does not
install Kneron PLUS.
GPU training additionally requires a CUDA-enabled PyTorch installation
compatible with the machine; verify `torch.cuda.is_available()` before choosing
`--device cuda`. The CPU path is suitable for software checks, and full training
may take substantially longer there.

If an earlier setup stopped at `Python 3.11 not found`, the subsequent missing
`python.exe` errors are consequences of the environment never being created.
Keep the existing checkout and update it to the Python 3.9 compatibility fix
before installing `.[nasa]`; the earlier unconditional ONNX Runtime 1.20.1 pin
cannot install on Python 3.9. An earlier physical KL720 replay is independent
of scientific-environment setup.

The installed console command is `radedge-nasa`. Listing the three model
contracts does not import PyTorch, ONNX, OpenCV or the vendor SDK. Preflight
needs OpenCV to inspect the actual TIFFs; run `radedge-nasa preflight` before
training. Candidate-graph screening in the next section needs no NASA data.

## 3. Screen the complete candidate graphs before full training

From the repository root, using the separate scientific environment above:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\screen_nasa_kl720_models.ps1 `
  -TrainingPython $TrainPython
```

This exports the complete compact CNN, MobileNetV2 and ResNet18, including
their actual scalar regression heads, at the NASA input shape
`[1,3,256,256]`. The deterministic, randomly initialized graphs use opset 11.
Each graph must pass the ONNX checker and PyTorch-to-ONNX Runtime parity on
synthetic zero, ramp and random inputs with a zero third channel. These
checks do not read NASA images or labels and do not measure accuracy.

An untrained network can emit the same scalar for every probe. The manifest
records the output span and flags constant outputs as a weak numerical
diagnostic; graph structure and head preservation are checked separately.
The trained-model export later checks parity on the selected real calibration
inputs.

The launcher then runs one fresh Docker container per candidate. It uses
the project's recorded Kneron Toolchain v0.33.1 image:

```text
kneron/toolchain@sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041
```

The image must already exist locally and Docker Desktop's Linux engine must
be running. The launcher inspects the image and executes its immutable local
image ID, with networking disabled, read-only repository/graph mounts and a
separate writable results directory. It does not install packages, pull an
image, change the PLUS environment, or connect to the USB dongle.
`-ToolchainImage` can select a different existing image; its identity is
recorded and its results belong to that toolchain configuration.

The scientific interpreter defaults to `RADEDGE_NASA_PYTHON` when set.
Otherwise it selects an existing `envs\nasa311\Scripts\python.exe` or
`envs\nasa39\Scripts\python.exe` under `D:\radiation-edge-ai-data`, in that
order. An explicit `-TrainingPython` selects the requested interpreter.
`-Root` changes the default data root. Every launch uses a fresh output
directory; an explicit `-OutputDir` must not already exist.

For an export-only check without Docker:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\screen_nasa_kl720_models.ps1 `
  -TrainingPython $TrainPython -ExportOnly
```

The same export stage is available directly on Linux or Windows:

```text
python -m radiation_edge_ai.nasa_bps.candidate_graphs --out <new-directory>
```

### Screening results

| File | Meaning |
| --- | --- |
| `graphs/candidate_graphs.json` | Export completion, candidate graph identities and per-model manifest hashes |
| `graphs/<architecture>/candidate.json` | Input/output contract, random seed, source/dependency identities, operators and numerical export checks |
| `graphs/<architecture>/candidate.onnx` | Complete untrained model graph; never a trained deployment artifact |
| `vendor/<architecture>/screen.json` | Model-specific Kneron assessment with original/optimized graph identities, toolchain information and evidence |
| `screen_receipt.json` | Host summary, per-model statuses, process exits and artifact hashes |

| Status | Interpretation and next action |
| --- | --- |
| `supported` | Fresh positive vendor evaluation with recognized CPU-partition evidence; eligible for the next development step under this configuration |
| `unsupported` | Explicit vendor rejection or neural-network computation assigned to CPU; inspect the saved evidence and revise or retain as a labeled hybrid candidate |
| `unknown` | Missing dependencies, failed invocation, missing/stale/unrecognized evidence or inconsistent identities; diagnose the saved error before advancing |

Named shape operations are distinguished from neural computation using the
actual optimized graph. An unrecognized CPU partition remains unresolved;
the absence of an error is never sufficient to report support. Unchanged
artifacts from earlier evaluations are excluded from positive evidence.
The [official Kneron workflow](https://doc.kneron.com/docs/toolchain/manual_1_overview/)
separates floating-point IP evaluation, quantization and final compilation.
Its evaluator reports estimated NPU performance; these values are not
physical-device timing.

The launcher exits zero when all requested candidates are supported (or the
requested export-only stage completes), 2 for a completed screen containing
an unsupported candidate, and 3 when any result is unknown. Missing Docker
or an unavailable image preserves the completed exports and records unknown
vendor results. `EXPORTED_ONLY` means vendor screening was not requested.
Always inspect the per-model result, not only the aggregate exit code.

This screen does not train models, quantize to INT8, produce a deployable NEF,
or establish biological accuracy. The exact trained graph must be checked
again. Retain all attempted candidates and their reasons in the comparison;
a failed or unknown candidate does not prevent investigation of the others.

## 4. Train the first three architectures on identical source folds

```powershell
$Run = 'D:\radiation-edge-ai-data\models\nasa_bps_53bp1\multimodel_v1'
& $TrainPython -m radiation_edge_ai.nasa_bps.multimodel train `
  --models r1countnet_v1 mobilenet_v2 resnet18 `
  --out $Run `
  --epochs 60 `
  --train-nuclei-per-bag 16 `
  --eval-chunk-size 16 `
  --device cuda
if ($LASTEXITCODE -ne 0) { throw 'NASA training stopped; inspect the run output.' }
```

Use a new `--out` path for each experiment. The default runs all three existing
source-held-out folds. For a short pipeline check, select one model and fold
and set `--epochs 1 --device cpu`; that run is a software experiment, not a
comparable accuracy result. Do not pool runs with different declared budgets.

The shared recipe is fixed before candidate evaluation:

| Contract | Implementation |
| --- | --- |
| Data | Frozen repaired development membership: 7,200 nuclei, 72 sample bags, six biological sources |
| Splits | Three existing source-grouped folds, each with four training and two held-out sources; 48 training and 24 test bags |
| Input | FITC, DAPI, zeros; native pixel scale; independent p1/p99.5 normalization; center-pad to 256 |
| Targets | NASA sample/plate-well `avg_nfoci`; no individual-nucleus pseudo-labels |
| Training | Deterministic 16-nucleus subset per bag per epoch, shared across architectures; matched sham/exposed aggregate and contrast losses |
| Budget | Random initialization, AdamW, learning rate `3e-4`, weight decay `1e-4`, cosine schedule, 60 fixed epochs by default |
| Selection | Final scheduled epoch; no checkpoint selection using held-out fold results |
| BatchNorm | Fixed training-source-only recalibration after the final epoch; learned parameters unchanged |
| Evaluation | Every one of the 100 nuclei per test bag, processed in memory-bounded chunks, then arithmetic-mean aggregation |
| PTQ preparation | Four nuclei from each training bag by default: 192 calibration tensors per fold; no held-out source contributes |

This is a new common training recipe. It does not reproduce the historical
100-nucleus-per-step training run. The compact architecture is retrained from
scratch within each fold so its comparison with the new backbones is valid.
The frozen all-development checkpoint is never used as an out-of-fold baseline.

The final BatchNorm calibration addresses the previously documented
[running-state mismatch](NASA_BPS_R1_V2_BN_REPAIR_POLICY.md). It is declared
for every new candidate in advance, uses only its fold's training images,
and preserves a separate checkpoint from before recalibration.

Preflight verifies frozen manifest/reference identities, grouped roles and
input geometry, then records current image hashes. Historical geometry repair
did not refresh every old download receipt, so this is explicitly a newly
measured snapshot. If `--inventory` is supplied, missing or mismatched receipts
are fatal. Export rechecks the recorded calibration TIFF hashes.

Each architecture/fold directory contains the checkpoint identity, contract,
training history, per-nucleus and per-bag predictions, matched contrasts and
source-specific metrics. Each architecture also has pooled out-of-fold
predictions and metrics, with a flag identifying whether all three folds
completed. The run-level `comparison.csv` brings the architecture metrics
together in one table. No result is labeled accuracy qualified by this command.

## 5. Export a trained fold and prepare conversion inputs

```powershell
& $TrainPython -m radiation_edge_ai.nasa_bps.multimodel export `
  --run-dir "$Run\mobilenet_v2\fold_female_role" `
  --out "$Run\exports\mobilenet_v2_fold_female"
if ($LASTEXITCODE -ne 0) { throw 'NASA export failed; inspect export.json if present.' }
```

Export verifies the checkpoint identity, model/preprocessing implementation,
contract, calibration membership and raw image hashes. It writes a fixed
batch-one `[1,3,256,256]` ONNX graph at opset 11, runs the ONNX checker, and
compares PyTorch with ONNX Runtime on every selected calibration input.

The export directory contains:

- the model-specific FP32 ONNX;
- `calibration_nchw/` with float32 NumPy input tensors;
- `calibration_manifest.csv` with per-array and source-image hashes;
- `onnx_parity.csv` and `export.json` with the parity result and model identity.

This completes the reusable FP32/export path. **INT8 quantization, full-model
Kneron compilation, and physical execution of MobileNetV2/ResNet18 remain
separate required steps.** The frozen-model replay launcher intentionally
accepts only the already validated baseline NEF. It cannot substitute a new
NEF and inherit the old model's accuracy result.

## Progress and next acceptance decision

The implementation provides one real-hardware baseline replay, complete
candidate-graph export and vendor-screen adapters, and an extensible
three-architecture training/export workflow. Software tests exercise the
actual models and ONNX path on synthetic inputs. Vendor parser/process tests
use controlled fixtures and do not count as Kneron acceptance. These checks
establish implementation behavior, not accuracy on public NASA data or
successful execution of the new architectures on a KL720.

Next, collect fresh Kneron screens on the toolchain host, record the declared
accuracy/degradation requirements, and run the NASA development experiments
under this common recipe. Qualifying
additional architectures still requires both biological reference performance
and paired FP32-to-physical results. Continue expanding the queue after these
first candidates; three is not a project limit.
