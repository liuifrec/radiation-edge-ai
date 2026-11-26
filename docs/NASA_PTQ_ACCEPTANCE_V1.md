# NASA 53BP1 ResNet18 PTQ acceptance, version 1

Declared 2026-11-18 05:59:47 UTC, before held-out BIE inference. This policy governs the
trained ResNet18 female-source development fold and the BIE with SHA256
`2428c60742e862f3bc3b6a6831284a3afa2b9e4db652fa760f331e3500bbea2f`.
The companion JSON binds this document, the artifacts and the numerical limits.
Both policy files use UTF-8 and LF line endings.

These **provisional engineering margins** support a decision about eligibility
for KL720 NEF compilation. Statistical and biological equivalence would require
separate justification and validation.

## Measurements

The cohort comprises BALBCF1 and C57BLF1 from the frozen development manifest.
A bag is one sample/plate-well containing exactly 100 nuclei. A model bag
prediction is the unweighted arithmetic mean of its 100 scalar burden predictions.
Only the bag has a NASA `avg_nfoci` reference, expressed in foci per nucleus.
Matched radiation effects use this same scale. Each delta is the exposed bag
mean minus its zero-dose sham mean for the same source, radiation type and time.
The panel includes Fe (0.82 Gy) and X-ray (1.0 Gy) at 4, 24 and 48 hours.
These definitions follow the [three-model campaign](NASA_THREE_MODEL_CAMPAIGN_V1.md).

Run the original trained FP32 ONNX, its vendor-optimized FP32 ONNX and the
fixed-point BIE on identical per-nucleus tensors. Use the original ONNX as the
FP32 comparator for every decision below. Preserve the optimized ONNX comparisons
and the original training predictions as diagnostic evidence. Reuse the frozen
FITC/DAPI normalization, zero third channel, native geometry and centered
256 × 256 padding. The 192 PTQ calibration tensors belong to the training sources.

## Decision

Use the unrounded values. Each of the following eight criteria must pass.
Absolute errors and changes in MAE have units of foci per nucleus. Compute
bag statistics with equal weight per bag and delta statistics with equal
weight per matched contrast.

| Criterion | Inclusive acceptance limit |
| --- | --- |
| 1. Mean absolute difference between BIE and original FP32 bag means | ≤ 0.05 |
| 2. Maximum absolute difference between BIE and original FP32 bag means | ≤ 0.15 |
| 3. Mean absolute difference between BIE and original FP32 matched deltas | ≤ 0.10 |
| 4. Maximum absolute difference between BIE and original FP32 matched deltas | ≤ 0.20 |
| 5. Bag MAE against the frozen NASA reference, BIE MAE minus original FP32 MAE | ≤ 0.05 |
| 6. New sign reversals among effects with absolute original FP32 delta ≥ 0.20 | 0 |
| 7. Preservation of positive four-hour effects | All four original FP32 deltas and corresponding BIE deltas strictly positive |
| 8. Completeness, identity and finiteness | All 2,400 distinct nuclei matched across backends, all 24 complete bags, and all 12 matched contrasts with finite predictions, references and calculated effects |

A reversal means strictly opposite signs between the original FP32 delta and
the BIE delta. Report exact zeros separately. Zero does not reverse a nonzero
sign under criterion 6 and fails strict positivity under criterion 7.
Report signed bias as BIE minus original FP32. Report results by source,
radiation type and time point, including the sample and contrast identities
of any failures. A missing or invalid panel prevents a passing decision.

Passing all eight criteria makes this exact BIE eligible for NEF compilation.
Preserve any failure with the original limits. Physical USB execution requires
separate approval. This procedure uses only the source-held-out development
fold for evaluation and reserves it from calibration and parameter tuning.
