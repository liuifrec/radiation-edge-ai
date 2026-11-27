# MobileNetV2 training-only precision experiment

**Neither candidate advances.** The original standard-BIE mean absolute error on the 24 training confirmation nuclei was 0.237050 foci/nucleus. The datapath-only configuration reduced it to 0.112621 and the full neural configuration to 0.078457. Both increased mean absolute error on C57BLM3, violating the prospective source nonregression rule.

The prospective protocol in `NASA_MOBILENET_TRAINING_PRECISION_V1.json` was frozen at `921cafc`. Both conversions preserve the released female-fold training checkpoint, original optimized ONNX, model ID 32781 and ordered 192 calibration tensors. The four training sources each contributed 48 tensors. The 24-case training confirmation panel spans original FP32 predictions from zero through the positive range, with six nuclei per source. The 16-case previous training pilot is reported separately. Neither panel provides independent biological-source validation.

## Precision settings

| Recipe | Model input | Datapath | Weights | Model output |
| --- | --- | --- | --- | --- |
| Original standard | INT8 | INT8 | INT8 | INT8 |
| `datapath16` | INT8 | INT16 | INT8 | INT8 |
| `neural16` | INT16 | INT16 | INT16 | INT16 |

These are global SDK requests. The per-layer hardware widths may differ even in the standard BIE. The analysis receipt saves all rows of the SDK's per-node HTML report along with the effective output descriptor from the generated BIE.

The installed SDK is Kneron 0.33.1. In its `/workspace/ktc/toolchain.py`, the KL720 branch of `ModelConfig.analysis` forwards input, output, weight and datapath modes to `gen_fx_model_v1`. It omits the `quan_config` and `flops_ratio` arguments present in the common signature. That interface supplies no verified method to assign a width to one named MobileNet block, GlobalAveragePool or regression-head layer. The first experiment therefore targets internal datapaths across the graph. The second uses the four global settings previously verified for Spotiflow on its own graph. Both use the ordinary percentage range method with `optimize=0` and two calibration threads.

The prespecified trigger was met for training confirmation nucleus `BPSR1V2D_01229`. The original FP32 affine head output was −0.9943286. Applying that head to the original BIE's pooled features yielded −1.6462805. Both are below the output ReLU floor. The native terminal word was zero. The host verified the trigger receipt, input identity, original BIE burden and captured scalar identity before each corrective conversion. This observation establishes a training example of upstream feature drift.

Direct inspection also covered the lower-level `sys_flow.run.gen_fx_model`. The function serializes its optional constraint dictionary to `user_config.json`. The bundled post-sigmoid example specifies fine-shift ranges, output scale and radix. Kneron's PTQ parameter table describes node-specific scale, radix and range constraints. Neither the installed example nor the documented contract establishes how to assign bitwidth to a named layer. The two global experiments therefore exhaust this protocol. The SDK inspection receipt has SHA256 `ec77f0e2860c2f43fb3729ca18df588dd8e1371ca298502c9e23605809c76304`. [Vendor PTQ parameter table](https://doc.kneron.com/docs/toolchain/quantization/1.2_Flow_and_Steps/)

Across the 123 SDK node rows, the original BIE reported 66 eight-bit, 21 fifteen-bit and 36 sixteen-bit outputs. The datapath-only recipe reported 3, 84 and 36. The full neural request reported 0, 87 and 36, with 48 eight-bit, 73 fifteen-bit and one sixteen-bit entry in the weight column, plus one inapplicable input entry. These counts include bookkeeping and non-weighted operators.

| Actual output width at a named tensor | Standard | Datapath INT16 | Neural INT16 |
| --- | ---: | ---: | ---: |
| Stem convolution | 16 | 16 | 16 |
| Stem Clip | 8 | 15 | 15 |
| Global average pool and flatten | 8 | 15 | 15 |
| Affine head before ReLU | 16 | 16 | 16 |
| Terminal Gather | 8 | 8 | 15 |

The datapath-only Gather uses an effective signed eight-bit word at radix 5 on a `16W1C8B` transport. The full neural candidate's effective signed fifteen-bit word uses radix 12. Its external `8W1C16B` descriptor uses radix 13. The scale is 1.0953782796859741 for both models. The float check uses the native word's radix. The test suite exercises both boundaries of signed 15-bit arithmetic, binary byte order, the independent transport radix and mismatched descriptor rejection. The external descriptor is metadata from the BIE. The campaign measured simulator values.

| Resource or artifact | Datapath INT16 | Neural INT16 |
| --- | ---: | ---: |
| PTQ wall time, seconds | 388.93 | 392.35 |
| BIE size, bytes | 14,140,184 | 14,156,723 |
| Vendor IP estimated ITC, ms | 5.13637 | 11.3783 |
| Vendor reported RV / WV, MB | 4.56602 / 3.96495 | 6.50855 / 3.96495 |
| BIE SHA256 | `585a7f78b9b8aade552aa4da55f88e376edc8729c155cc91e738e556cecb993c` | `85abdb0793594b7840e1ef379885b5804e925879fb96291e2302f1ffe6c00716` |

The original standard BIE's estimated ITC was 1.8529 ms. Both generated candidate reports list NPU placement for all 123 rows. The IP estimates describe the vendor's model of KL720 execution. Benchmarks below use the host's actual software simulator.

## Paired training readout

| Population or measurement | Standard INT8 | Datapath INT16 | Neural INT16 |
| --- | ---: | ---: | ---: |
| 24 confirmation nuclei, mean signed error | −0.193437 | −0.055547 | −0.077241 |
| 24 confirmation nuclei, mean absolute error | 0.237050 | 0.112621 | 0.078457 |
| 24 confirmation nuclei, maximum absolute error | 0.981690 | 0.405920 | 0.209625 |
| BALBCM1, mean absolute error | 0.348451 | 0.166918 | 0.086936 |
| BALBCM2, mean absolute error | 0.229857 | 0.097329 | 0.101787 |
| C57BLM1, mean absolute error | 0.315526 | 0.117207 | 0.064689 |
| C57BLM3, mean absolute error | **0.054366** | **0.069030** | **0.060415** |
| 16 historical pilot nuclei, mean absolute error | 0.137175 | 0.056407 | 0.038857 |
| Simulator readout including SDK witnesses and archival work, seconds | 147.90 | 260.33 | 358.65 |

Each confirmation source contributed six nuclei. On C57BLM3, full neural precision shifted the signed mean from +0.003478 to −0.060415 and datapath-only precision shifted it to −0.048825. The prespecified source check applies regardless of the magnitude of a source's baseline error. The pilot did not enter selection. Across 12 sampled Docker telemetry readings during each conversion, the largest observed memory usage was 891.9 MiB for datapath-only and 891.3 MiB for full neural precision. Those samples cover part of the conversion, and the software simulator runs overlapped.

The finalized scorer authenticated all 120 native binary words across the original and two generated BIEs, including archive CRCs, exact binary lengths, signed little-endian decoding, native text, returned float, nucleus identity and FP32 prediction pairing. Each eight-input chunk had one independently retrieved public single-image result. The selection receipt contains the paths and hashes of the three readouts, the tests for each candidate and a null selection. Both BIEs and all native outputs remain external.

| External record under `scratch/nasa_mobilenet_precision_pair_20261127T020000Z_c53693e1` | SHA256 |
| --- | --- |
| `selection.json` | `53f9e06626d9cd22728b81668bc2a7e13f6e33c96896907e302a67af9da4286b` |
| `baseline/evaluation/readout.json` | `3ce0cd572deafdd192f1de869f91a2335f0cb81c61ce94fcad9dc480a8ffb8d0` |
| `datapath16/quantization.json` | `97fc2e78d20ce3d28d98ff4bb65b763489b8c4e25dc13c3893080e9df39eab16` |
| `datapath16/effective_word_replay/evaluation/readout.json` | `7a7f69a4f4a5998d11d07a7575b0788839243b538803a988192e22ffdac58c31` |
| `neural16/quantization.json` | `6d605b693665da81e829e72e8d9b22b1cd4063e6e66117f977a8e1c50c6712c2` |
| `neural16/effective_word_replay/evaluation/readout.json` | `66596423d2ebe457f79011e4a477ca4a09d3b519aec7056d8fa2f654e1c3408c` |

The initial host processes ended after conversion. The two simulator runs resumed from verified BIEs using `--replay-effective-word`, which writes separate readouts. This continuation ran each candidate once. No development bag, final holdout, NEF compilation or physical USB run was used. A subsequent QAT study would need its own weights, source-separated training protocol and prospective engineering criteria. The original standard INT8 decision and external BIE remain the reference for this MobileNet fold.

## Reproduction

Use the checkout containing the prospective policy, the training-only attribution capture, and the archived original MobileNet PTQ directory. Run the following from the checkout on Windows. The launcher verifies the original tissue-source split, checkpoint, graph, training receipts, every normalized calibration input and model identities. Each recipe writes to a fresh child of external scratch. The policy is required in both the host and simulated readout.

```powershell
$data = 'D:\radiation-edge-ai-data'
$python = Join-Path $data 'envs/nasa39/Scripts/python.exe'
$env:PYTHONPATH = (Resolve-Path src).Path
$dest = Join-Path $data ('scratch/nasa_training_precision_' + [guid]::NewGuid())
$trigger = Join-Path $data 'scratch/mobilenet_feature_training_20261127T015528Z_795ec600/confirmation_implementation_check/receipt.json'
foreach ($stage in 'baseline','datapath16','neural16','score') {
    & $python -B scripts/windows/run_nasa_training_precision.py $stage --data-root $data --out $dest --trigger $trigger
    if ($LASTEXITCODE -ne 0) { throw "Inspect the preserved $stage receipt before continuing." }
}
```

All fixed-point outputs come from the pinned vendor simulator. The native run verifies per-input signed and floating dumps, job status, and ordered results. The public single-image API independently checks the final input of every chunk. The raw signed word for every nucleus is decoded from that BIE's measured radix, width and scale. The receipt retains native logs, descriptor files, measured durations, per-nucleus output and paired error. Only the 24 confirmation rows enter the training selection rule. The previous NASA bag and radiation-contrast decisions remain in their own campaign receipts.
