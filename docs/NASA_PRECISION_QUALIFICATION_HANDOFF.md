# Prospective development-regression handoff

A standard INT8 failure remains attached to its own binary. Each new declaration
has its own BIE identity. The historical female panel is a **known development
regression**. The endpoint is the equally weighted sample mean of continuous
burden in foci/nucleus. The screening evidence consists of training nuclei from
the original calibration material. Source-independent confirmation requires a
separately enrolled cohort.

## One readout per declared candidate

1. Commit the candidate declaration and record the SHA256 of its file before
   any development BIE call. Verify the declared BIE, checkpoint, source and
   optimized ONNX, original calibration manifest, training precision report
   and selection/readout receipts. Use the native BIE internal output precision
   and compare the received signed words and decoded values on the original
   training witnesses. Keep its transport descriptor in a separate field.
2. Verify the historical FP32 preparation by its checked digest. Use a fresh
   run directory and rebind the original sealed input manifest, file hashes
   and FP32 predictions to **only the new BIE**. Archive the exact candidate
   declaration, evaluator, SDK and all command arguments. The original
   standard-INT8 launchers pin their specific historical policies. A new
   runner must explicitly admit the newly committed policy digest, scope and
   candidate and reproduce those same input and identity checks.
3. Execute the pinned Dynasty fixed simulator on the 2,400 already prepared
   tensors in order. Use the two original held-out female sources, 100 nuclei
   per bag, original FITC/DAPI/zero channel order and identical normalization
   and geometry. Check the manifest, tensor bytes, sample IDs, nuclear IDs,
   signs and output order, including singleton controls. Archive all native
   return codes, output integers, floats and invocation timings.
4. Feed the pairs into the existing
   `radiation_edge_ai.nasa_bps.multimodel_bie_comparison.score_rows` with the
   original source and bag membership, reference and frozen FP32 tables and
   same prepared tensor rows. Use original FP32 as comparator. Recompute all
   24 sample means and 12 exposed-minus-sham deltas. For each unrounded value,
   evaluate every limit in the declaration, including strict 4 h positivity,
   strong-effect signs and complete identities. Check all eight independent
   gate decisions and hash the per-nucleus, bag and contrast tables. An
   invalid or failing result closes this binary's qualification attempt.
5. A fully authenticated development pass can make this specific BIE eligible
   for a separate NEF compilation. Recheck the full receipt chain before
   compiling. A subsequent NEF simulator study must establish input packing
   and output transport fidelity. USB execution requires separate approval.
   Record biological validation independently. Leave the six historical
   development and three reserved source IDs out of future independent
   enrollment. [Training protocol and prospective cohort design](NASA_PRECISION_TRAINING_PROTOCOL_V1.md)

## Reference replay commands

The following are the physical reference entry points for a later approved
hardware session. The current campaign verified the three reference NEF hashes
offline. Use the configured PLUS and scientific interpreters and fresh external
output directories. The NASA R1 command uses its historical reference cohort.

```powershell
.\scripts\windows\run_nasa_kl720.ps1 -Root $env:RADEDGE_DATA_ROOT
& $env:RADEDGE_DNAI_PYTHON -B .\scripts\windows\run_dnai_hardware_baseline.py --data-root $env:RADEDGE_DATA_ROOT --execute
$smoke = Join-Path $env:RADEDGE_DATA_ROOT ('scratch/spotiflow_smoke_' + [guid]::NewGuid())
.\scripts\windows\run_spotiflow_kl720.ps1 -Smoke -Out $smoke
if ($LASTEXITCODE -ne 0) { throw 'Smoke failed' }
$receipt = Join-Path $smoke 'receipt.json'
.\scripts\windows\run_spotiflow_kl720.ps1 -Panel -SmokeReceipt $receipt -SmokeSha256 (Get-FileHash $receipt).Hash.ToLowerInvariant()
```

The explicit CPU, simulator and NPU routes and original replay requirements
are documented in the [platform deployment guide](PLATFORM_DEPLOYMENT_SELECTION.md).
