# MobileNetV2 female-fold KL720 qualification

The candidate is the original trained `mobilenet_v2/fold_female_role` checkpoint,
model ID 32781. The conversion used Kneron 0.33.1 on the screened optimized graph.
All model binaries, TIFFs, tensor files, simulator dumps and per-nucleus predictions
are held in external project storage. The immutable
[protocol](NASA_MOBILENET_V2_FEMALE_INT8_V1.md) and
[candidate binding](NASA_MOBILENET_V2_FEMALE_INT8_V1.json) identify the files by digest.

## Input provenance and training characterization

| Item | SHA256 |
| --- | --- |
| Trained checkpoint | `bb00a25e2db05334e113852760414adafa22f95614e4c43f92b6607e690bd543` |
| Original FP32 ONNX | `2f18b2330ca3289ab422e1ff47b526b4bfa6a46fc4bd7354346609715f575e45` |
| Optimized FP32 ONNX | `75824e1ed4f8520d90a942c1351926b922db9b4284b8b6d7cd5663428e73c074` |
| Selected BIE, 14,066,993 bytes | `2102c400d696bc385e23d2d9b978b779e4c95b82661272e6873ac1eb57607e26` |
| Prospective policy | `1f69f8f866ca55bcd551889008de135dd14b2fd502caf0d925561903f5574d37` |

Both FP32 graphs ran on all 192 training calibration tensors. Their largest absolute
prediction difference was zero. The calibration used four nuclei per training bag
from 48 bags and four sources. Comparing IDs and hashes across calibration and
evaluation found zero shared sample IDs, nuclei, bags, biological sources or
serialized input tensors. The vendor's precision report confirmed INT8 input,
output, weights, datapath and CPU precision. The PTQ worker took 653.1 seconds.

The training-only pilot compared four prespecified nuclei from each source.
Values below are per-nucleus prediction shifts in foci/nucleus. The biological
reference applies at the sample mean. The pilot used the original training cohort
and had no role in the eight bag and contrast decisions.

| Training source | Nuclei | BIE minus FP32 mean | Mean absolute shift | Maximum absolute shift |
| --- | ---: | ---: | ---: | ---: |
| BALBCM1 | 4 | −0.10322 | 0.15524 | 0.42813 |
| BALBCM2 | 4 | −0.30211 | 0.36217 | 0.91098 |
| C57BLM1 | 4 | 0.00000 | 0.00000 | 0.00000 |
| C57BLM3 | 4 | +0.03129 | 0.03129 | 0.12517 |

The compiled BIE's scalar descriptor has radix 5 and scale 1.09537828, giving
a signed-grid spacing of 0.02852896. All 16 original predictions fell in range.
Seven observed output codes lay outside both neighboring codes that bracketed
their respective FP32 predictions. Any rounding rule that selected either
neighbor would have produced mean absolute error at most 0.009805 on these
inputs. The observed error was 0.137175. For the largest error, FP32 was
0.939508, bracketed by codes 32 and 33, and the simulator returned code 1.
All 16 native scalar words reproduced their saved text and decoded API values.
This localizes part of the error upstream of the final scalar encoding. The
observations do not distinguish the feature extractor, pooling and regression
head. The training-only audit receipt SHA256 is
`5a4b6edb5c7a47b524920c91b17bddf0394599948bbc555cca1558fa1b3badbe`.
The offline verifier reproduced the same paired CSV, SHA256
`1ab3a19ce15c799f9190d9c8d43810797765f43db5a47f1949bf8d01ad7cf6a7`.
Run `python -m radiation_edge_ai.nasa_bps.mobilenet_terminal_audit --help`
for the three required external paths.

The protocol commit was `b42badec9f4d8e4a41cd8c6b11f5a3cff32f7454`.
Commit `3e1b2fcfe535f343815d4ebe8cd48640b27148bb` then sealed the BIE and
prepared FP32 inputs before the development simulation. Only the female fold was
selected. Male-1 remains excluded on its original ONNX parity result.

## Reproduction on the configured Windows host

Run from this campaign's checkout. Set the external root to the directory
containing `scratch`, `models`, `data` and `envs`. The first two invocations
generate and characterize a new training-only binary in a fresh directory. A
new conversion carries its own receipt and cannot replace the sealed BIE.

```powershell
$root = 'D:\radiation-edge-ai-data'
$py = Join-Path $root 'envs/nasa39/Scripts/python.exe'
$env:PYTHONPATH = (Resolve-Path src).Path
$new = Join-Path $root ('scratch/nasa_mobilenet_ptq_' + [guid]::NewGuid())
& $py -B scripts/windows/run_mobilenet_training_ptq.py quantize --data-root $root --out $new
if ($LASTEXITCODE -ne 0) { throw 'PTQ failed' }
& $py -B scripts/windows/run_mobilenet_training_ptq.py characterize --data-root $root --out $new
```

The following separate command replays the originally sealed candidate using
its hash-bound prepared tensors. It writes a fresh simulator log, raw archive
for each chunk, paired predictions, scored tables and `evaluation_receipt.json`.
The original paired readout is a development regression panel.

```powershell
$prepared = Join-Path $root 'scratch/nasa_mobilenet_female_bie_evaluation_20261126T234928Z_df112c4c/preparation'
& $py -B scripts/windows/run_mobilenet_female_bie_evaluation.py --data-root $root --prepared $prepared --jobs 5
```
