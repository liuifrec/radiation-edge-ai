# Real-model Perilipin2 KL720 compiler isolation

**Status: experimental compiler diagnosis, NOT NEF or physical-device inference.**

The pinned Kneron v0.33.1 compiler accepts the isolated 1-, 24- and
96-channel grouped ConvTranspose synthetic models, including C=96, H=32.
However the entire FP32-parity-proven 91-node Perilipin2 nuclear model still
fails inside ConvTransposeNode::InferPadding / InvalidProgramInput.
These observations do not imply any KL720 on-device Perilipin2 performance.

This branch provides dependency-closed prefix tests from the REAL frozen ONNX
model, retaining original learned weights and original input contract
[1,3,512,512]. No retraining, graph re-optimization or new upsampling rewrite.

| Test | Cut-point tensor shape | Question |
| --- | --- | --- |
| encoder_before_first_upsample | [1,96,32,32] | Can the real encoder compile without any ConvTranspose? |
| upsample_1 | [1,96,64,64] | Does the first ConvTranspose become incompatible when attached to encoder? |
| upsample_2 | [1,72,128,128] | Does the second decoder stage introduce incompatibility? |
| upsample_3 | [1,48,256,256] | Does the third stage introduce incompatibility? |
| upsample_4 | [1,24,512,512] | Does the complete decoder upsampling path compile? |

For each probe, ONNX Runtime compares two deterministic intermediate tensor
outputs between the complete original graph and its dependency-closed prefix.
It requires exact FP32 values, identical shapes, and no nonfinite values.
This is a numerical graph-survival test, not real-image biological validation.

The compiler test executes ONE staged model at a time, with the exact pinned
Docker image SHA256
2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041,
--pull never and --network none. It does not optimize the model again.
Original ONNX, stagewise candidate models, learned weights, images and masks
MUST remain under private D: folders, never be pushed to Git or uploaded
to public vendor issues.

## Windows: fetch the separate KL720 compiler branch

Use the Perilipin2 compiler worktree, NOT the desktop GUI worktree:

    $ErrorActionPreference = "Stop"
    Set-Location "C:\Users\yul03\Projects\radiation-edge-ai-perilipin2"
    git fetch origin feature/perilipin2-kl720-stagewise-isolation-v1
    if ($LASTEXITCODE -ne 0) { throw "Fetch failed" }
    git merge --ff-only FETCH_HEAD
    if ($LASTEXITCODE -ne 0) { throw "Preserve local changes" }

Identify the PREVIOUSLY generated parity-proven deconvolution model folder
without rerunning any image or modifying the 91-node graph:

    $Proven = Get-ChildItem "D:\radiation-edge-ai-data\scratch" -Filter M3C_RESIZE_PROOF.json -Recurse -File |
        Where-Object {
            try {
                $r = Get-Content $_.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
                $r.variants.'depthwise-deconv'.parity_pass -eq $true
            } catch { $false }
        }
    $Proven | Select-Object -ExpandProperty DirectoryName

Choose the folder that contains BOTH M3C_RESIZE_PROOF.json and
nuclear_candidate.kneron.opt.depthwise-deconv.onnx. If several are listed,
select deliberately; do not blindly pick the newest one.

## Generate five parity-locked probes (no Docker)

    $Python = "D:\radiation-edge-ai-data\envs\perilipin2-m3c\Scripts\python.exe"
    $ProofDir = "<paste actual depthwise-deconv proof directory>"
    $Stagewise = "D:\radiation-edge-ai-data\scratch\perilipin2_kl720_stagewise_20261112_v1"
    & $Python -X utf8 .\scripts\windows\perilipin2_kl720_stagewise_proof.py --proof-dir $ProofDir --out $Stagewise
    if ($LASTEXITCODE -ne 0) { throw "Stagewise FP32 proof rejected" }
    Get-Content "$Stagewise\STAGEWISE_PROOF.json" -Raw -Encoding UTF8

Expected milestone: STAGEWISE_FP32_PROOF_PASS. If parity fails, STOP.
Each output is source-locked and saved in a NEW D: scratch directory.

## First vendor compiler experiment: the encoder WITHOUT upsampling

    $First = "D:\radiation-edge-ai-data\scratch\perilipin2_kl720_stagewise_encoder_20261112_v1"
    & $Python -X utf8 .\scripts\windows\perilipin2_kl720_stagewise_vendor_gate.py --proof-dir $Stagewise --stage encoder_before_first_upsample --out $First
    if (Test-Path "$First\STAGEWISE_VENDOR_EXECUTION.json") {
        Get-Content "$First\STAGEWISE_VENDOR_EXECUTION.json" -Raw -Encoding UTF8
    }
    if (Test-Path "$First\backtrace.log") {
        Get-Content "$First\backtrace.log" -Tail 35
    }

A vendor rejection is informative; do NOT delete the failed run.
VENDOR_RETURNED_REQUIRES_REPORT_REVIEW is not an NEF or device proof.
The compiler stdout/NPU/CPU allocation and backtrace must also be inspected.

Only if encoder compiles cleanly, run a SECOND new vendor folder using
--stage upsample_1. Continue with upsample_2, upsample_3, upsample_4 only
until the first rejection, with one experiment at a time. If encoder fails,
the problem is upstream of the grouped deconvolution and further decoder-only
experiments should stop. If all five prefix models compile and the full model
still fails, next isolate the final output/segmentation head.

## The path to real KL720 Perilipin2

Compiler acceptance is necessary but insufficient. Only after a faithful full
graph passes would we start explicit INT8 calibration with locked representative
images, BIE reproducibility, NEF build, physical USB KL720 execution, and
paired numerical/object-endpoint comparison to the CPU model and independent
microscopy annotations. None of these steps is claimed completed here.

Already physical-device tested DNAi/NASA paths remain separate and unchanged.
