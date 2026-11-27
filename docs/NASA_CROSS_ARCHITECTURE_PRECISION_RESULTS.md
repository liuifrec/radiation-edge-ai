# NASA female-fold precision campaign

**ResNet18 full-neural INT16 warrants a separate prospective development
regression. Neither MobileNetV2 candidate passed its prespecified training
screen.** All new model measurements used the four training sources. The
standard MobileNetV2 and ResNet18 BIEs retain their original 7/8 and 6/8
decisions. All three new BIEs remain experimental and ineligible for NEF
compilation. The Windows physical references remain R1 v2, DNA-fiber v3 and
Spotiflow general in its qualified full-INT16 transport configuration.

## What the two regressors share

Both original INT8 BIEs reduced the mean prediction on the **same 24 nuclei**.
Six come from each of BALBCM1, BALBCM2, C57BLM1 and C57BLM3, selected by
MobileNetV2 FP32 range before the new fixed-point comparisons. Each substitution
uses that architecture's original FP32 regression head and rectification applied
to its correctly decoded native BIE pooled vector. The remaining component is
the difference from that result to the native BIE endpoint. The table gives
mean signed changes in foci/nucleus, with BIE minus original FP32 as the sign
convention. NASA supplies sample means, so these per-nucleus figures describe
numerical fidelity.

| Standard BIE | Extractor and pooling | Head and output | Total shift | Total MAE | FP32 zeros → BIE zeros |
| --- | ---: | ---: | ---: | ---: | ---: |
| MobileNetV2 | −0.163574 | −0.029863 | −0.193437 | 0.237050 | 4 → 10 |
| ResNet18 | −0.109150 | −0.006120 | −0.115270 | 0.115270 | 7 → 8 |

The older per-architecture diagnostic panels shared only four nonpilot nuclei.
The additional ResNet trace on all 24 matched the prior captures on those four.
Each head substitution is a measured counterfactual with a frozen FP32 suffix.
Similar historical bag errors alone would not identify a shared layer defect.

* **MobileNetV2.** On the 24 nonpilot cases, the cumulative mean preactivation
  shift reached −0.10184 at the stem clip and −0.37021 after the first inverted
  residual block. This was the earliest sampled block with a negative mean in
  all four sources. Global pooling added −0.000793 before rectification and
  fixed-head arithmetic added −0.04919. Six originally positive predictions
  reached zero. All 16 separate pilot outputs reproduced. All 520 native
  tensors decoded within 7.46 × 10⁻⁹ of their simulator float dumps.
  [Feature attribution](NASA_MOBILENET_FEATURE_ATTRIBUTION_RESULTS.md)
* **ResNet18.** On its separate 47 nonpilot traces, the stem convolution shifted
  the mean head preactivation upward in each source. The following MaxPool
  boundary added −0.06905 on average. Standard native ReLU and MaxPool used
  15→8 effective bits and 16→8 storage bits. Of 1,535,274 in-range non-tie
  positions distinguishing floor from nearest, every observed word agreed
  with nearest. The rest of the extractor contributed further loss. This
  evidence localizes sensitivity at a boundary without establishing an
  arithmetic error in the vendor implementation. [ResNet trace PR](https://github.com/liuifrec/radiation-edge-ai/pull/45)

## What changed at ResNet18 MaxPool

The full-neural INT16 BIE removed the measured downward contribution at the
stem MaxPool on **47 of 47 nonpilot training nuclei**. For each nucleus we
decoded its actual native ReLU output and following MaxPool output, ran the
corresponding unchanged FP32 suffix from each cut, and subtracted the two
head preactivations. The standard BIE gave a negative difference on every
nucleus. The full-neural BIE gave exactly zero on every nucleus. The separate
pilot moved from −0.022113 to zero.

| Training source | Nonpilot nuclei | Standard MaxPool contribution | Full-neural MaxPool contribution |
| --- | ---: | ---: | ---: |
| BALBCM1 | 11 | −0.065766 | 0 |
| BALBCM2 | 12 | −0.071770 | 0 |
| C57BLM1 | 12 | −0.066935 | 0 |
| C57BLM3 | 12 | −0.071447 | 0 |
| All | 47 | −0.069048 | 0 |

The units are foci/nucleus at the FP32 head before rectification. Both
full-neural stem tensors used 15 effective bits with matching per-channel
grids. Across the 48 independent captures, signed binary decoding differed
from the vendor float dumps by at most 7.46 × 10⁻⁹. Final full-BIE prediction
MAE was still 0.010918 foci/nucleus.

One predeclared host experiment isolated the output grid. We held the
standard BIE's ReLU activations fixed, applied the original MaxPool, rounded
its output to the measured full-neural grid and replayed the FP32 suffix.
The mean contribution was −0.00001193 and the mean absolute contribution
was 0.00001493. There were zero clipped codes or exact rounding ties. The
pinned KL720 toolchain exposes no verified per-layer width control for this
boundary. The host experiment supports investigating a finer output grid if
such a control becomes available. Native mixed-precision performance and
end-to-end fidelity remain unmeasured. The paired table SHA256 is
`8cfe4591e0b8b36f5167b75e471ca38ffdf6da2f83137e5458bbacecf8c8ab6c`.
[Prespecified boundary probe](NASA_RESNET18_MAXPOOL_PROBE.md)

## Actual candidate conversions and costs

Each conversion used the frozen weights and all 192 original training-only
NCHW tensors from 48 bags, four nuclei per bag. The budgets below are specific
to the same pinned Kneron 0.33.1 SDK. The BIE byte count is the simulator model
file size. ITC is the vendor IP evaluation's latency estimate. CPU simulation,
compilation and physical inference are different stages.

| Architecture and requested precision | Nonpilot cases | Training bias | Training MAE | BIE bytes | IP ITC, ms | Training decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MobileNetV2 standard INT8 | 24 | −0.193437 | 0.237050 | 14,066,993 | 1.8529 | Historical failed binary |
| MobileNetV2 INT16 datapath, INT8 I/O and weights | 24 | −0.055547 | 0.112621 | 14,140,184 | 5.13637 | Failed source MAE |
| MobileNetV2 full-neural INT16 | 24 | −0.077241 | 0.078457 | 14,156,723 | 11.3783 | Failed source MAE |
| ResNet18 standard INT8 | 47 | −0.065922 | 0.065922 | 47,300,679 | 9.94965 | Historical failed binary |
| ResNet18 full-neural INT16 | 47 | −0.010918 | 0.010918 | 47,334,682 | 66.7988 | Descriptive comparator with prospective protocol |

All five vendor reports identify no CPU neural fallback. The three new PTQ
analysis calls took 388.93 s (MobileNetV2 datapath), 392.35 s (MobileNetV2
full) and 246.36 s (ResNet18 full). The respective native training readouts
took 260.33, 358.65 and 534.79 s, including the SDK checks and collection of
native evidence. Wall times reflect concurrent host workloads. The global
requests produce heterogeneous graphs. The MobileNetV2 full report has 87
output rows at 15 bits and 36 at 16 bits, and 48 weight rows at 8 bits.
The ResNet18 full report has 61 output rows at 15 bits and 18 at 16 bits.
The actual vendor report is registered alongside each BIE. NEF compilation,
device memory fit and USB latency have no measured result for the new candidates.

**Source non-regression determined the MobileNetV2 outcome.** On C57BLM3,
standard MAE was 0.054366. Datapath-only gave 0.069030 and full-neural INT16
gave 0.060415. Both variants lowered overall MAE and absolute mean bias,
but the prospectively declared screen required all four source MAEs to hold
or improve. The source panel has six cases. Both failures stand. [Precision screen PR](https://github.com/liuifrec/radiation-edge-ai/pull/44)

## ResNet18 scalar representation

The initial verification stopped before endpoint scoring. The same
`3f654f267ae280f27d6aaf6715b32af4285c5de6f4eb13c844e02e70f2f69c9b` BIE was
used for the corrected run. Its scalar is named `burden`, produced by
`/Gather`, with ONNX shape [1] and logical hardware shape [1,1,1,1].

| Contract | Effective or transport width | Radix | Scale | Representation |
| --- | ---: | ---: | ---: | --- |
| Native Dynasty node dump | Signed 15 | 12 | 1.3206359148025513 | Little-endian signed 16-bit storage holding the effective code |
| BIE public I/O descriptor | Signed 16 | 13 | 1.3206359148025513 | `8W1C16B` transport layout |

For the historical pilot, the native signed word was 3,760. Evaluating
`float32(word) / 2**12 / float32(scale)` gives 0.6950960, matching the native
output. The same value would occupy transport code 7,520 on the declared
radix-13 grid. This second code is a metadata prediction. Only the native
words were observed in this campaign. The simulator readout paired integer
and float dumps on all 48 inputs and repeated one case from each batch with
the public singleton API. Signed boundary tests exercise negative values
that a rectified burden cannot supply.

An independent byte audit matched the singleton native scalar files to the
batched native words on all 48 inputs. The internal rule reconstructed all
48 floats, including all 20 nonzeros. Applying radix 13 directly to the
same internal words halved all 20 nonzero outputs and induced a maximum error
of 1.044123 foci/nucleus. The 28 zeros alone cannot distinguish these
representations. The original failed and corrected readout receipt SHA256s are
`7cb0bdc69ad5f457e336a3d4761fe752ff65b79faa240cf03c8afc879e8cf35a` and
`3290161922587ed8f3a3adc931df6fc5b10d48339ec371a7de604e93cefc2a8f`.

## Evidence and reproduction

Use a checkout containing PRs #39–46 or the relevant stacked branches. Set
`RADEDGE_DATA_ROOT` to the external collection and `PYTHONPATH` to the
checkout's `src`. Archived inputs, model binaries and native outputs remain
outside the repository. The machine-readable policies and catalogue identify
the exact BIEs and prospective limits. Precision-screen and per-architecture
PRs give the corresponding native capture and readout commands.

* `python -m radiation_edge_ai.nasa_bps.cross_arch_attribution --help` lists
  the two frozen selections, original calibration manifest, three paired
  CSVs and their expected SHA256 options. Include `--matched-resnet` and
  its digest to reproduce the 24-case comparison. Use
  `python scripts/analysis/plot_nasa_training_attribution.py --comparison <run> --matched --out <new-figure-dir>`
  for the matched PDF and PNG.
* The full-precision ResNet18 study used the `neural16` recipe of
  `training_precision.py` with the ResNet18-specific policy, screened optimized
  graph and SHA-bound training parity. The shared
  `training_precision_readout.py` then consumed the generated BIE, frozen
  48-case selection, training calibration directory and the same policy.
  The original failed-width readout and corrected `resnet_readout_v2` are separate.
* For the native MaxPool trace, mount the BIE as `/model.bie`, the original
  selection as `/panel.json`, its calibration directory as `/export` and
  a fresh external location as `/out`. Mount the pinned capture helper
  `scripts/windows/mobilenet_feature_capture.py` beside
  `scripts/analysis/capture_resnet_precision_stem.py`. Execute the latter with
  `--out /out/full16_stem` in the same pinned image. The host command
  `python scripts/analysis/analyze_resnet_precision_stem.py --help` enumerates
  its hash-checked inputs for the paired suffix and localized grid analysis.
* Check catalogue identities offline with
  `python -m radiation_edge_ai.hardware.deployment_catalogue --model nasa-resnet18-female@neural16 --data-root <external-root> --verify`.
  The MobileNetV2 keys are `nasa-mobilenetv2@datapath16` and
  `nasa-mobilenetv2@neural16`. Their training records have no physical backend.

[Prospective ResNet18 regression policy](NASA_RESNET18_NEURAL16_REGRESSION_V1.json) ·
[One-time development-regression protocol and reference replay commands](NASA_PRECISION_QUALIFICATION_HANDOFF.md) ·
[Training design and independent-source enrollment](NASA_PRECISION_TRAINING_PROTOCOL_V1.md)
