# NASA BPS R1 v2 KL720 validation record

Status: **terminal physical-KL720 biological deployment-equivalence PASS, frozen**

This document records the completed end-to-end deployment validation for the NASA BPS OSD-366 53BP1 R1 v2 model. The scientific question was whether the fixed compact FP32 model could be converted through ONNX, Kneron optimization, development-only INT8 PTQ, BIE, NEF compilation, and physical KL720 execution while preserving the same predeclared radiation-response conclusions on the untouched final holdout.

The terminal result is:

> **Physical KL720 biological deployment-equivalence PASS under the original seven predeclared biological gates.**

This is a biological-equivalence claim, not a claim of strict numerical equivalence between FP32, BIE, and physical-device outputs.

## Biological endpoint and interpretation

The NASA reference endpoint is sample/plate-well aggregate `avg_nfoci`. It is not a per-nucleus label.

The network output for each selected nucleus is therefore interpreted as a **latent continuous 53BP1 burden score**, not an individually supervised focus count. Per-nucleus scores are averaged within each frozen sample/plate-well bag and compared with the aggregate NASA reference. Biological replication/grouping remains at Source Name / sample level; the 2,058 nuclei are not independent biological replicates.

Frozen neural input:

- float32 NCHW `[1, 3, 256, 256]` before KL720 quantization/relayout;
- FITC / 53BP1 channel with p1/p99.5 per-crop normalization;
- DAPI channel with the same normalization rule;
- third channel identically zero;
- native crop scale, no resize;
- center-pad to 256 x 256;
- MASK used only for QC, never as model input.

## Development and untouched final holdout

R1 v2 development used 7,200 nuclei in 72 bags from six Source Names. Final evaluation used three source-held-out female Source Names that were excluded from R1 v1/v2 development:

- `BALBCF2`: 858 nuclei, 10 bags, 5 matched contrasts;
- `C57BLF2`: 600 nuclei, complete X-ray branch, 3 matched contrasts;
- `C57BLF3`: 600 nuclei, complete Fe branch, 3 matched contrasts.

Final total: **2,058 nuclei, 22 bags, 11 branch-specific exposed-minus-sham contrasts**.

Because all three held-out Source Names are female, this validation is a deployment stress test and does not support a sex-effect claim. Public strain mapping used here supports BALB/cByJ and C57BL/6J; the result must not be presented as 15-strain generalization.

## Confirmatory FP32 reference

Authoritative BatchNorm-recalibrated FP32 checkpoint SHA256:

`2b07c67d6e50e7a36e26e0052feebc826b518e6efae863ba312ba9761a4eb3d5`

QC1 final-holdout manifest SHA256:

`f89962f96bec05930c07149717a90bd682016a2778bbfae6b827425159aa0643`

FP32 deployment-reference freeze SHA256:

`abb58026d4f798554ef7e44346a989499242fcbd89d8c6629a1c4209d6c16ce3`

Observed FP32 final-holdout result:

| Metric | FP32 |
| --- | ---: |
| Bag MAE | 0.5843 |
| Bag RMSE | 0.8839 |
| Bag Pearson | 0.9715 |
| Bag Spearman | 0.8616 |
| Delta MAE | 0.9851 |
| Delta Pearson | 0.9885 |
| Delta Spearman | 0.9909 |
| Overall direction | 10/11 |
| 4 h direction | 4/4 |
| 24+48 h direction | 6/7 |
| BALBCF2 direction | 4/5 |
| C57BLF2 direction | 3/3 |
| C57BLF3 direction | 3/3 |
| Peak-time recovery | 3/3 |

All seven original biological gates passed.

### Confirmatory-evaluation protocol deviation

The first explicit FP32 unblinding execution opened and hash-verified the raw phenotype table and then stopped on the first bag because the frozen evaluator used an overly strict strain normalizer (`BALB/cByJ` versus raw `BALBC`). No target value had yet been dereferenced and no confirmatory metric had been computed. The alias rule had already been established and codified before R1 v2 development.

The original evaluator was preserved. Confirmation was completed with a separate recovery evaluator using only the pre-existing alias policy. The final confirmatory result must therefore continue to be described as:

**confirmatory final-holdout evaluation with a documented non-outcome metadata-normalization implementation recovery**.

No model, holdout membership, endpoint, preprocessing, threshold, or biological gate changed.

## Deployment conversion chain

### FP32 -> ONNX

Fixed-shape opset-11 ONNX SHA256:

`a65718a8f07d5bcd8707bae78d8f730d56047db21e575fa55ba5396adc49a07e`

ONNX deployment freeze SHA256:

`e31c64050244ea93cbb3ea57433e0f281218bbbd352ee66d058e98bb3e36fe06`

ONNX Runtime parity passed all predeclared numerical gates and reproduced the FP32 biological direction/peak signature.

### Kneron floating optimization / IP support

Optimized ONNX SHA256:

`c3a6aa5ed80280b0286f3b1aebf5f77f44dc60f569cc42fe2ab00ba7edde820c`

Kneron floating optimization/IP report SHA256:

`6c0355e8d0d6eb8b2345429a9f130b0a5512d9cc1c17695eaee4c0f341f6f008`

Toolchain image:

`kneron/toolchain@sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041`

Toolchain version: `kneron/toolchain:v0.33.1`.

KL720 graph/IP evaluation succeeded with no meaningful CPU computation nodes. The toolchain IP-evaluation FPS estimate is not a physical-device throughput measurement.

### Development-only INT8 PTQ

Calibration used exactly 288 development nuclei = 4 deterministic nuclei from each of 72 development bags. The final holdout was not used to choose calibration samples, calibration ranges, PTQ settings, optimizer settings, compiler settings, or thresholds.

Calibration manifest SHA256:

`78c9ed83e1992907264dfeec9ef6e2a434868bd3f299672f610a42a0018b5d4e`

Calibration freeze SHA256:

`f017f8a27563611e49fa55147920da2860b38fbb3f64a4c7db2995a36752a9df`

Frozen PTQ settings included percentage range calibration (`0.999`, `0.999999`) and `optimize=0`.

BIE SHA256:

`c52b8a78c595347667bad7a34950af92adbe734179b913646d06513a88fc1dd8`

PTQ summary SHA256:

`98f0e096c059b8f7312bd9cb14c4733d5901183c0ffd712fd996f248db43fde1`

Pre-holdout BIE deployment freeze SHA256:

`63192365e77aa269d5ec792f2b184f8ada6f1603c451da47b1d1718ceb3dcc57`

### BIE holdout biological deployment-equivalence

BIE result summary SHA256:

`c15776c6820a4aa28fff8e18c16b576164ce3ed2cad61ba2057efdef79800ca4`

BIE accepted-result freeze SHA256:

`e901d4be044c5ea5507b3b60643043eb8aa0af5643a0b7fba2f9bdd7c588bb0a`

Observed BIE result:

| Metric | BIE INT8 |
| --- | ---: |
| Bag MAE | 0.7922 |
| Bag RMSE | 0.8893 |
| Bag Pearson | 0.9501 |
| Bag Spearman | 0.5729 |
| Delta MAE | 0.9405 |
| Delta Pearson | 0.9634 |
| Delta Spearman | 0.8182 |
| Overall direction | 10/11 |
| 4 h direction | 4/4 |
| 24+48 h direction | 6/7 |
| BALBCF2 direction | 4/5 |
| C57BLF2 direction | 3/3 |
| C57BLF3 direction | 3/3 |
| Peak-time recovery | 3/3 |

All seven original biological deployment gates passed and the exact FP32 direction/peak signature was reproduced. Numerical fidelity degraded relative to FP32, especially bag Spearman, so the accepted conclusion is biological deployment-equivalence rather than numerical equivalence.

## NEF compilation and compile-integrity

The accepted BIE was compiled directly to NEF without re-quantization.

Compiled NEF SHA256:

`d04f02855a8a82ae7cb5ff48af91eee69c341142d1097469890385bd23ee097b`

Compile summary SHA256:

`813213f224da7ec9a9621a27c92b45ab87f8909eb0c7e203403f5a8c6e64c2ea`

Compiler-verified model identity:

- platform: KL720;
- model ID: `32770`;
- model version: `0x8b29`;
- command nodes: 10 NPU / 0 CPU.

The Toolchain v0.33.1 NEF simulator had a scalar-output metadata parser bug because `raw_shape=[1]` was combined with hard-coded `ch_dim=1`. A verifier-only temporary-JSON compatibility shim represented the scalar as `[1,1]` for parser metadata only; the persistent NEF was never modified.

NEF-vs-BIE compile-integrity max absolute error:

`7.152557373e-07` with frozen gate `<=1e-4`: **PASS**.

Compile-integrity summary SHA256:

`90aaac45fb84c61aaf2959dfbcf0fffd3bbb9b8f1c2afb1084596e96c3833037`

NEF deployment-candidate freeze SHA256:

`f818b9d912017a1d8fdc4aafd4bf321b8bf5b84e8214e6ef0ba029b48b0b90bb`

## Physical KL720 validation

The project hardware baseline uses Kneron PLUS 3.2.0-compatible Python tooling on Windows 11. The frozen NASA NEF was then tested on the physical KL720 through the known-good generic data-inference path.

### One-sample physical smoke

Deterministic development sample: `BPSR1V2D_00324`.

Frozen BIE scalar: `2.498514175415039`.

Physical KL720 scalar: `2.498514175415039`.

Absolute error: `0` under a pre-hardware frozen max-abs gate of `1e-4`.

Physical output was byte-identical to the frozen BIE reference.

Physical smoke reference summary SHA256:

`18d261568361e22f7acb704b9ab15022fe4feca27d9a76776631a8f5e202286d`

Physical smoke summary SHA256:

`ec72104a5d0ed958b8dc84c4e7ae289cf1286b845e9ccd082255b1d3150be958`

Physical smoke PASS freeze SHA256:

`224b754235ebcd22eb053ca1f279043f3b74e9e3b461ed1f8dad26beddfa679d`

### Outcome-blind full physical holdout

All 2,058 frozen holdout nuclei were then run through the physical KL720 **before physical predictions were joined to biological targets or reference outputs**.

Physical prediction CSV SHA256:

`91ea895a16cd97067af63b5c5c4434e972629bfda4c9582ad456a79e007b6d46`

Physical inference summary SHA256:

`7147ba4de4ba44fa64ed6b414e990970cd08ac83ea5e4b993ed38e9ad0a93aba`

Outcome-blind physical prediction freeze SHA256:

`e4e7d9150e2e33479e2ec8894e583f0d09a3d890a6a97fbd356a1a0fb78d2b30`

Observed device send+receive timing over 2,058 nuclei:

- mean: `3.676 ms`;
- median: `3.636 ms`;
- p95: `4.269 ms`.

These are descriptive hardware send+receive measurements for this workflow, not a complete energy/throughput benchmark of the full microscopy pipeline.

A first full-holdout invocation stopped before the first nucleus inference because the host `--image-root` argument pointed one directory above the materialized `images/` directory. The corrected invocation used the same frozen model, NEF, manifest, preprocessing, and runner and completed all 2,058 nuclei. No partial physical prediction artifact or biological outcome was produced by the failed invocation.

### Terminal physical biological deployment-equivalence

Only after the full physical prediction set had been frozen outcome-blind was it joined to the immutable FP32 biological reference.

Final physical biological-equivalence summary SHA256:

`8d236ac3cee264975d0bb3de2397f15063cb213070956f45ce174baa71d69786`

Final physical biological-equivalence freeze SHA256:

`0259ad745215dfb567a2edbd1593fb7dc789de6a4366dbcdfeb7234901b844bb`

Observed terminal physical result:

| Metric | Physical KL720 |
| --- | ---: |
| Bag MAE | 0.7922 |
| Bag RMSE | 0.8893 |
| Bag Pearson | 0.9501 |
| Bag Spearman | 0.5729 |
| Delta MAE | 0.9405 |
| Delta Pearson | 0.9635 |
| Delta Spearman | 0.8182 |
| Overall direction | 10/11 |
| 4 h direction | 4/4 |
| 24+48 h direction | 6/7 |
| BALBCF2 direction | 4/5 |
| C57BLF2 direction | 3/3 |
| C57BLF3 direction | 3/3 |
| Peak-time recovery | 3/3 |

All seven original biological gates passed. The exact FP32 direction/peak signature was reproduced.

Physical versus accepted BIE, descriptive only:

- nucleus exact equality: no;
- nucleus maximum absolute error: `0.03903937345`;
- bag maximum absolute error: `0.0007807853612`.

No post-hoc physical-vs-BIE numerical acceptance gate was introduced. These numerical differences do not alter the frozen biological decision.

## Frozen biological gates

The terminal physical deployment had to satisfy the same predeclared biological criteria used for the accepted deployment path:

1. bag Spearman >= 0.50;
2. matched-delta Spearman >= 0.60;
3. overall direction agreement >= 9/11;
4. 4 h direction agreement = 4/4;
5. 24+48 h direction agreement >= 5/7;
6. peak-time recovery = 3/3 for the three complete source x radiation-branch series;
7. per-source direction minima: BALBCF2 >=4/5, C57BLF2 >=2/3, C57BLF3 >=2/3.

Terminal physical result: **7/7 PASS**.

## What may be claimed

Supported wording:

> The fixed INT8 KL720 deployment preserved the predeclared radiation-effect direction and repair-kinetic conclusions of the frozen FP32 reference on the source-held-out NASA BPS OSD-366 validation set.

Also supported:

- physical KL720 execution was demonstrated for all 2,058 frozen holdout nuclei;
- the full physical prediction set was frozen before biological evaluation;
- all seven predeclared biological deployment-equivalence criteria passed;
- the physical device reproduced the complete FP32 direction/peak signature;
- no final-holdout outcome was used for PTQ/model/threshold tuning.

## What must not be claimed

Do not describe this result as:

- strict FP32/BIE/hardware numerical equivalence;
- per-nucleus 53BP1 focus counting accuracy;
- a sex-effect analysis;
- validation across 15 mouse strains;
- proof that 2,058 nuclei are 2,058 independent biological replicates;
- a complete system-level throughput or energy benchmark.

The terminal result also carries forward the documented non-outcome metadata-normalization recovery from the FP32 confirmation stage.

## Final deployment identity

```text
FP32 checkpoint
2b07c67d6e50e7a36e26e0052feebc826b518e6efae863ba312ba9761a4eb3d5
        |
ONNX
a65718a8f07d5bcd8707bae78d8f730d56047db21e575fa55ba5396adc49a07e
        |
optimized ONNX
c3a6aa5ed80280b0286f3b1aebf5f77f44dc60f569cc42fe2ab00ba7edde820c
        |
INT8 BIE
c52b8a78c595347667bad7a34950af92adbe734179b913646d06513a88fc1dd8
        |
compiled NEF
d04f02855a8a82ae7cb5ff48af91eee69c341142d1097469890385bd23ee097b
        |
physical KL720 predictions
91ea895a16cd97067af63b5c5c4434e972629bfda4c9582ad456a79e007b6d46
        |
terminal physical PASS freeze
0259ad745215dfb567a2edbd1593fb7dc789de6a4366dbcdfeb7234901b844bb
```

No model, PTQ, BIE, NEF, biological threshold, or gate change is authorized from this holdout result.