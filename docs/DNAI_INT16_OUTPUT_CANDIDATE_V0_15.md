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
