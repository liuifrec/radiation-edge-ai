# v0.15 DNAi — experimental INT16 model-output-only preflight

## Scientific motivation and status

Frozen v3 KL720 outputs are unchanged. For nine previously selected raw 512x512
windows, CPU FP512 and Kneron floating outputs had zero argmax disagreements,
while the legacy floating-to-v3 BIE pair had 1463 and BIE-to-physical had zero.
The production provenance of historical Kneron simulator cache files is
incomplete; this is descriptive localization, not proof of a quantizer defect.
Across three paired development samples, bias-optimized v4 recovered one
3/4-human-supported FP512 fiber but lost two others with 3/4 support.
Do not deploy v4 as a biological fidelity repair.

**Status:** first v0.15 candidate code prepared, but no new PTQ, BIE, NEF,
hardware run or biological validation has been performed.

## Exactly one experimentally changed parameter

Test `ktc.ModelConfig.analysis(..., model_out_bitwidth_mode='int16')` on KL720.
The pinned local `kneron/toolchain:v0.33.1` exposes that exact parameter.
Preserve default INT8 datapath, default INT8 weights, original
`quantize_mode='default'`, original ONNX model, input preprocessing and
the original selected v3 PTQ recipe:

- `datapath_range_method=percentage`, `percentage=1.0`, `percentage_16b=1.0`
- `percentile=0.001`, `outlier_factor=1.0`, `optimize=0`
- Frozen KL720 model ID 32769, model version 8b28
- 192 existing non-validation calibration tensors from unique source-image hashes

Do not mix in `mixbw`, bias optimization, new calibration, modified match
thresholds, or reconstruction/postprocessing changes.

## Frozen toolchain identity

- Image ID: `sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041`
- Docker image tag at inspection: `kneron/toolchain:v0.33.1`
- Python: `/workspace/miniconda/bin/python` (NOT `/usr/bin/python3`)
- Installed ONNX: `1.7.0`
- `ModelConfig.analysis` explicitly declares `model_out_bitwidth_mode='int8'`
  and documents `int16` as supported for platform 720.

## Fail-closed implementation

The original `kneron_quantize_int8_512.py` remains backward compatible.
When no `--model-out-bitwidth-mode` option is supplied it calls `km.analysis`
with exactly the original seven explicit kwargs, in the same order,
and writes the original `ptq_config` keys. An INT16 candidate requires
explicit flags, and is prohibited from reusing or being nested under any
v1/v2/v3/v4 candidate directory. Its directory basename must be exactly
`int8_512_v5_int16out` and **must not already exist**.

Before any model conversion, the candidate checks exact frozen optimized
ONNX SHA256 `a901d1b309a9a0e5026febd5070252787e4b110a2a7d2828e16a19364d6094d0`,
independently supplied 64-hex calibration manifest SHA256, the original
v3 PTQ recipe, 192 unique sources/tensors, explicit validation exclusion,
input shapes/dtypes/finiteness, and each input tensor's current SHA256.
On actual generation it preserves those hashes and the declared image ID
in a new `v015_candidate_provenance` section of its own summary.
The image ID is declared by the caller; run via verified image ID rather
than a mutable `latest` tag. The historic v3 compile script must NOT
be reused to compile this new experimental model.

### Next permitted action: preflight only

First find the exact model and manifest paths recorded in the accepted v3
summary (historical paths may be inside Docker rather than Windows):

```powershell
$V3 = 'D:\radiation-edge-ai-data\models\dnai\unet_mobileone_s1\kneron\int8_512_v3_pct100'
$M = Get-Content (Join-Path $V3 'int8_analysis_summary.json') -Raw | ConvertFrom-Json
[pscustomobject]@{
    RecordedONNX = $M.onnx
    RecordedONNXSHA = $M.onnx_sha256
    RecordedCalibrationManifest = $M.calibration_manifest
    CalibrationCount = $M.calibration_count
    PTQ = ($M.ptq_config | ConvertTo-Json -Compress)
} | Format-List
```

Resolve the **actual host** ONNX and calibration manifest files, verify
their hashes independently and map their exact relative paths under D:.
Do not substitute a guessed path or run against a different ONNX.

Then use the already-installed immutable image and the correct Conda Python.
The following is a **template**, with placeholders that must be resolved:

```powershell
$ImageID = 'sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041'
$RepoRoot = 'C:\Users\yul03\Projects\radiation-edge-ai'
$DataRoot = 'D:\radiation-edge-ai-data'
$Onnx = '/radedge_data/<VERIFIED_OPTIMIZED_ONNX_RELATIVE_PATH>'
$Manifest = '/radedge_data/<VERIFIED_CALIBRATION_MANIFEST_RELATIVE_PATH>'
$ManifestSHA = '<HOST_VERIFIED_64_HEX_MANIFEST_SHA256>'
$Output = '/radedge_data/models/dnai/unet_mobileone_s1/kneron/int8_512_v5_int16out'

docker run --rm --pull never --network none `
    -v "${RepoRoot}:/radedge_repo:ro" `
    -v "${DataRoot}:/radedge_data:ro" `
    --entrypoint /workspace/miniconda/bin/python $ImageID -B `
    /radedge_repo/src/radiation_edge_ai/dna_fiber/kneron_quantize_int8_512.py `
    --onnx $Onnx --calibration-manifest $Manifest --output-dir $Output `
    --percentage 1.0 --percentage-16b 1.0 --optimize 0 `
    --model-out-bitwidth-mode int16 `
    --candidate-toolchain-image-id $ImageID `
    --candidate-manifest-sha256 $ManifestSHA --candidate-preflight-only
if ($LASTEXITCODE -ne 0) { throw 'STOP: v0.15 preflight failed' }
```

Preflight passes only if it prints
`V015_PREFLIGHT_PASS_NO_ANALYSIS_NO_OUTPUT_DIRECTORY`. It does not call
`ktc.ModelConfig` or `km.analysis`, create an output directory or BIE,
or write to model data directories. Both mounted host folders are read-only.

## Acceptance before a physical deployment

Only after preflight and a separately frozen development/confirmatory
dataset split may the output bitwidth candidate be generated to a **new**
directory. Contemporaneously bind each input/model/output with SHA256.
Compare same-sample floating-to-BIE margins, 3-class argmax, retained
human-supported fiber identities, bbox IoU>=0.5, tract-length errors,
red/green ratios and all BIE-only detections. Require an independent
untouched confirmatory cohort before claiming biological equivalence.
The historical 20-image PTQ characterization panel is not such a cohort.

## Actual Windows source preflight and isolated write-enabled candidate

On 2026-10-29, the investigator ran the **actual pinned Docker read-only
preflight**, which returned
`V015_PREFLIGHT_PASS_NO_ANALYSIS_NO_OUTPUT_DIRECTORY` (exit code 0).
The optimized ONNX hash matched
`a901d1b309a9a0e5026febd5070252787e4b110a2a7d2828e16a19364d6094d0`,
and the calibration manifest hash matched
`08fe74d646342c36709f44a116b10379c0c073e16ecffc67fd66fa35a1634a79`.
All 192 current .npy inputs were loadable float32 `(1,3,512,512)`, finite,
with global min/max `[-2.117904,2.428571]`; the manifest declares no
validation-panel calibration. The host confirmed no new candidate output
directory was created. The preflight creates present-day tensor SHA256
fingerprints; it **does not** establish byte equality to historical
per-tensor fingerprints because the old manifest does not contain them.

**Permission boundary:** once the scientific evaluation split is fixed,
write candidate outputs to a **new D: scratch** directory only. Mount
all existing model/calibration data **read-only**, and mount the unique
experiment scratch directory as the **only writable host source**.
This differs deliberately from an in-place model-directory candidate
to protect all historical v1-v4 sources and the selected v3 NEF.

The command below generates the first experimental BIE. **It is not a
read-only preflight**: Kneron's `analysis()` runs and writes results
in a new scratch folder. Do not run it more than once on the same
output directory; preserve any incomplete output for failure forensics.
The sole modeled PTQ parameter difference remains output INT16.

```powershell
Set-Location 'C:\Users\yul03\Projects\radiation-edge-ai'
$ErrorActionPreference = 'Stop'
git switch feature/v0.15-dnai-int16out-preflight
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Git update failed' }

$ImageID = 'sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041'
$RepoRoot = (Get-Location).Path
$DataRoot = 'D:\radiation-edge-ai-data'
$SourceRoot = Join-Path $DataRoot 'models\dnai\unet_mobileone_s1'
$OnnxHost = Join-Path $SourceRoot 'kneron\unet_mobileone_s1_512x512_opset11_optimized.onnx'
$ManifestHost = Join-Path $SourceRoot 'calibration_512_v2\manifest.json'
$ExpectedOnnx = 'a901d1b309a9a0e5026febd5070252787e4b110a2a7d2828e16a19364d6094d0'
$ExpectedManifest = '08fe74d646342c36709f44a116b10379c0c073e16ecffc67fd66fa35a1634a79'
if ((Get-FileHash $OnnxHost -Algorithm SHA256).Hash -ine $ExpectedOnnx) {
    throw 'Frozen model hash mismatch'
}
if ((Get-FileHash $ManifestHost -Algorithm SHA256).Hash -ine $ExpectedManifest) {
    throw 'Frozen manifest hash mismatch'
}
$Dfree = (Get-PSDrive -Name D).Free / 1GB
Write-Host ('D: free space: {0:N1} GiB' -f $Dfree)
if ($Dfree -lt 20) { throw 'Insufficient D: scratch margin (under 20 GiB)' }

$RunRoot = Join-Path $DataRoot ('scratch\dnai_v015_int16out_' +
    (Get-Date -Format 'yyyyMMdd_HHmmss'))
if (Test-Path -LiteralPath $RunRoot) { throw 'Scratch run already exists' }
New-Item -ItemType Directory -Path $RunRoot | Out-Null
$CandidateHost = Join-Path $RunRoot 'int8_512_v5_int16out'
if (Test-Path -LiteralPath $CandidateHost) { throw 'Candidate already exists' }

docker run --rm --pull never --network none `
    --mount "type=bind,source=$RepoRoot,target=/radedge_repo,readonly" `
    --mount "type=bind,source=$DataRoot,target=/workspace/data,readonly" `
    --mount "type=bind,source=$RunRoot,target=/v015_work" `
    --entrypoint /workspace/miniconda/bin/python `
    $ImageID -B `
    /radedge_repo/src/radiation_edge_ai/dna_fiber/kneron_quantize_int8_512.py `
    --onnx /workspace/data/models/dnai/unet_mobileone_s1/kneron/unet_mobileone_s1_512x512_opset11_optimized.onnx `
    --calibration-manifest /workspace/data/models/dnai/unet_mobileone_s1/calibration_512_v2/manifest.json `
    --output-dir /v015_work/int8_512_v5_int16out `
    --percentage 1.0 --percentage-16b 1.0 --optimize 0 `
    --model-out-bitwidth-mode int16 `
    --candidate-toolchain-image-id $ImageID `
    --candidate-manifest-sha256 $ExpectedManifest

if ($LASTEXITCODE -ne 0) {
    throw ('Candidate PTQ failed; preserve scratch: ' + $RunRoot)
}
$SummaryPath = Join-Path $CandidateHost 'int8_analysis_summary.json'
$BiePath = Join-Path $CandidateHost 'unet_mobileone_s1_512x512_int16out.bie'
if (-not (Test-Path $SummaryPath) -or -not (Test-Path $BiePath)) {
    throw 'Candidate output summary/BIE missing'
}
$S = Get-Content $SummaryPath -Raw | ConvertFrom-Json
$ActualBieSHA = (Get-FileHash $BiePath -Algorithm SHA256).Hash
if ($ActualBieSHA -ine $S.bie_sha256 -or
    $S.ptq_config.model_out_bitwidth_mode -ne 'int16' -or
    $S.v015_candidate_provenance.calibration_tensor_count_verified -ne 192) {
    throw 'Candidate BIE / PTQ / calibration provenance verification failed'
}
Write-Host ('EXPERIMENTAL BIE verified: ' + $BiePath)
Write-Host ('SHA256: ' + $ActualBieSHA)
Write-Host ('Scratch: ' + $RunRoot)
```

The standard KTC `km.analysis` API may write temporary files under
`/data1/kneron_flow` inside the ephemeral container as in the original
toolchain pipeline. The candidate wrapper copies the final BIE and fresh
FX/analysis reports to the writable scratch output directory. Candidate
v0.15 now rejects stale SDK reports and names the output file
`unet_mobileone_s1_512x512_int16out.bie`, to avoid ambiguity with the
historical `int8.bie` artifact. Toolchain runtime errors are not proof
of candidate biological failure; preserve scratch and report exact logs.

**After generation:** do not overwrite v3, compile NEF or run physical KL720
until candidate BIE output is tested with contemporaneous input/output
hashes and frozen object-level criteria. Neither the historic 20-image
panel nor already used PTQ test images are an untouched confirmatory set.

