# MobileNetV2 female-fold standard INT8 training attribution

On 24 prespecified nonpilot training nuclei, the BIE's mean signed endpoint change was −0.193437 foci/nucleus and its MAE against the original FP32 graph was 0.237050. Feeding the decoded native pooled features through the frozen FP32 head accounted for −0.163574 of that mean change. The remaining −0.029863 occurred downstream. These are nuclear comparisons on training calibration material selected across FP32 prediction ranks. They do not estimate the error of a random bag or establish biological validation.

## Panel and identity

Each of four sources contributed six additional nuclei after excluding the archived 16. Four of the new inputs had a zero FP32 output. The other 20 sampled five positive FP32 ranks for each source. The prior pilot reproduced 16/16 archived scalar words and predictions. Its signed change and MAE were −0.093510 and 0.137175, respectively. The largest pilot discrepancy was 0.939508 FP32 → 0.073316 FP32 head on the BIE pool → 0.028529 BIE.

| Source, six additional nuclei each | Mean BIE − FP32 | Mean absolute change | Mean change through pooled features | Remaining mean change |
| --- | ---: | ---: | ---: | ---: |
| BALBCM1 | −0.348451 | 0.348451 | −0.321684 | −0.026767 |
| BALBCM2 | −0.187671 | 0.229857 | −0.162382 | −0.025289 |
| C57BLM1 | −0.241104 | 0.315526 | −0.207066 | −0.034038 |
| C57BLM3 | +0.003478 | 0.054366 | +0.036836 | −0.033357 |

The four zero-reference nuclei all returned zero. Across the five positive FP32 ranks, mean signed output changes were −0.035181, −0.392706, −0.629256, +0.055840 and −0.159319 in increasing rank order. This order is source relative. The largest absolute shift was 0.981690 on a BALBCM1 nucleus with FP32 output 2.094320. The FP32 head on its native pool produced 1.165505 and the BIE produced 1.112629. The 12 nuclei above the panel median input RMS had a mean shift of −0.246110 versus −0.140764 in the lower half. The receipt also gives individual radiation/time/dose strata and their denominators.

## Localization and mechanism

All 520 native taps across 40 nuclei came directly from the unchanged BIE at simulator dump level 2. Native shape and radix metadata, signed binary words, per-channel scales and native float dumps were checked before substitution. FP32 output declarations exposed matched tensors in each graph. Each tapped graph reproduced its own untapped output exactly, and every extracted FP32 suffix reproduced its full-graph control preactivation within 0.00001. The feature archives include paired original and optimized FP32 tensors, per-channel errors and weight-projected contributions.

| Boundary | Mean cumulative preactivation change through the FP32 suffix |
| --- | ---: |
| Signed Q0.7 input | −0.024770 |
| Stem convolution | −0.060794 |
| Stem Clip | −0.101835 |
| First inverted residual block | −0.370209 |
| Block 2 | −0.418417 |
| Block 13 | −0.467096 |
| Block 17 | −0.519812 |
| Last spatial feature map | −0.518467 |

The earliest sampled boundary with a negative mean in every source was the end of the first inverted residual block. The increment from the stem Clip to that block accounted for −0.268373 on average. The sign of the cumulative stem effect varied by source. Substitutions localize a difference between observed boundaries. They cannot distinguish constituent operations within the block or establish a vendor arithmetic defect.

Executing the original ONNX pool-to-Gemm suffix on the original FP32 pool exactly reproduced every control head. On the native pooled vectors, its maximum difference from the float64 accumulation with the same frozen parameters was below 0.00000087. Of 1,280 pooled channels, indices 245, 369 and 779 had the largest absolute mean weight-projected contributions on the 24 additional nuclei (−0.006437, −0.004931 and −0.004659 preactivation units). The external summary retains every channel.

| Additive contribution | Mean signed preactivation decomposition | Mean rectified endpoint decomposition |
| --- | ---: | ---: |
| Original → optimized FP32 | 0.000000 | 0.000000 |
| Input packing/rounding | −0.024770 | +0.001763 |
| Extractor after input rounding | −0.493698 | −0.164954 |
| Global pooling and flatten | −0.000793 | −0.000381 |
| Quantized affine head | −0.049186 | −0.028117 |
| Change in the zero floor | +0.376756 | Included at each stage |
| Encoding of the rectified scalar | −0.001746 | −0.001746 |
| Final Gather | 0.000000 | 0.000000 |

The signed affine-head residual was negative on all 24 nuclei (−0.052146 to −0.047040). Pre-pool storage has 15 effective signed bits, pooling and flatten have 8 bits, Gemm has 16 bits, and post-ReLU and Gather have 8 bits. The 24 pre-pool maps occupied the highest code 246 times in 1,966,080 values. No pooled vector reached either signed limit, and no optimized FP32 pooled value fell outside its assigned native channel range. At the first block 460 of 6,291,456 native values reached either endpoint. Occupancy documents the opportunity for clipping. It does not quantify clipping's causal share of the shift. Contributions use ordered substitutions, so the allocation of effects depends on boundary order.

All model weights remained frozen. Wider feature arithmetic is supported as a bounded technical experiment. The standard INT8 failure and its NEF block remain in force. A training-only reduction in these nuclear errors does not authorize compilation or establish independent biological performance.

## Reproduce

The external capture directory is `scratch/mobilenet_feature_training_20261127T015528Z_795ec600` under the data root. The machine receipts retain absolute paths where needed for replay. Repository files contain the program and aggregate results.

| Artifact | SHA256 |
| --- | --- |
| Frozen checkpoint | `bb00a25e2db05334e113852760414adafa22f95614e4c43f92b6607e690bd543` |
| Original ONNX | `2f18b2330ca3289ab422e1ff47b526b4bfa6a46fc4bd7354346609715f575e45` |
| Optimized ONNX | `75824e1ed4f8520d90a942c1351926b922db9b4284b8b6d7cd5663428e73c074` |
| Standard INT8 BIE | `2102c400d696bc385e23d2d9b978b779e4c95b82661272e6873ac1eb57607e26` |
| Training panel selection | `534fffa2f104bf395952fe25ecc8708ca4b91bf032dc521b19298cf5fe35ebfe` |
| Complete attribution receipt | `1b05af34fdfc8d847d82b7cbfa2f6d4c893b91852a3a62295d7787fd0508ad9f` |
| Per-nucleus authoritative capture links and measurements | `64cedfd5dc2016f1c586f56f7bc18089d9a86e4779c5718bde30424065414885` |
| Original ONNX pooled-head response pairs, including input SHA256 and pilot indicator | `d823a75b530f92e467575435e5f4431eac8d3e5ee012ec57357bff38e53333a2` |
| Channel, source, boundary and direct-head summary receipt | `5ef684c77cc7b638dc6c97b9f456aabc50881749d2e0637cc820583244f3af86` |

The 40 single-nucleus simulator jobs with layer dumping took 1,565.016 summed job-seconds (39.125 s mean, two workers). This is simulation plus diagnostic output time. The completed FP32 and BIE archive analysis took 73.0 s with ONNX Runtime 1.19.2, ONNX 1.17.0 and NumPy 2.0.2. An interrupted host pass left 37 partial per-case results. The complete host pass independently reread all 40 immutable native captures into `attribution_verified`.

The following commands run from the diagnostic worktree with the NASA scientific Python environment and pinned simulator image available. Use fresh external output directories for each run. The selection and analysis commands print the digests used in the subsequent command.

```powershell
$env:PYTHONPATH = "src"
python scripts/windows/mobilenet_feature_capture.py prepare --out $Capture
python scripts/windows/mobilenet_feature_capture.py capture --out $Capture
python scripts/windows/analyze_mobilenet_features.py --capture $Capture --selection-sha $SelectionSHA --out $Analysis
python scripts/windows/summarize_mobilenet_features.py --attribution $Analysis --receipt-sha $AnalysisSHA --out $Summary --figure
```

The final command writes channelwise values, signed-code occupancy, stage increments, direct original ONNX head replays, and optional PNG/PDF figures. The full and independent source-level biological evaluation requirements remain separate from this training diagnostic.
