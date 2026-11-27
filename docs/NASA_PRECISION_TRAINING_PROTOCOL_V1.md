# Female-fold precision campaign, version 1

Frozen on 2026-11-27 UTC before the new precision conversions. The scientific
endpoint remains mean continuous 53BP1 burden per sample, in foci/nucleus.
The immediate decision is whether a frozen BIE warrants a single prospective
engineering regression. All diagnostic measurements in this campaign use the
four original training sources and the 192 original calibration tensors.
The per-nucleus FP32 prediction is a fidelity reference. NASA supplies labels
at the sample level.

## Architecture comparison

Use the original standard INT8 BIEs and the unmodified, released female-fold
checkpoints and ONNX weights. For each case let F be the original rectified
FP32 burden, P the rectified original floating-point head applied to the
dequantized native post-pool feature vector, and B the native BIE output.
Measure P−F, B−P and B−F per nucleus. Average signed components telescope.
Report the absolute errors of each component separately since they can cancel.
Report pilot and nonpilot results separately, and compare exact common
nonpilot inputs by sample ID, source and serialized tensor hash. Also report
the separate panels and zero versus positive FP32 outputs within architecture.
Inference along a substituted FP32 suffix estimates the output sensitivity
to the observed boundary values. Local causes inside a fused stage remain
unresolved by that substitution.

MobileNetV2 uses its 16 previous pilot cases and 24 additional cases chosen on
original FP32 range within source. Its sealed panel SHA256 is
`534fffa2f104bf395952fe25ecc8708ca4b91bf032dc521b19298cf5fe35ebfe`.
ResNet18 uses its original minimum-rank nucleus per training bag, 48 cases,
selection SHA256 `fecec721ddf8ad424ed9f5ac13594a0488d2fe6805aa0031aff11c18a602113a`.
Report its historical pilot `BPSR1V2D_01501` separately from the other 47.
All nine source identities reserved or previously used for NASA development
and confirmation stay on the exclusion list for a future independent study.

## Bounded conversions

| Graph | Kneron KL720 request | Purpose |
| --- | --- | --- |
| MobileNetV2 | 16-bit datapath, 8-bit ingress, egress and weights | Test internal activation precision after verifying an upstream feature shift on a nonpilot case |
| MobileNetV2 | 16-bit datapath, weights, ingress and egress | Test the full neural precision comparator on the same panel |
| ResNet18 | 16-bit datapath, weights, ingress and egress | Descriptive cross-architecture comparator on its 48 archived training inputs |

Use the pinned 0.33.1 image and identical calibration membership in each case.
The ResNet18 comparator follows the existing evidence of early stem loss on
four sources. Use the vendor default CPU precision and bias settings, save
the actual precision report, and check requested fields and endpoint decoding.
No per-layer setting is counted as applied without the vendor's confirmation.
A proven technical wrapper failure can be corrected on the same predetermined
recipe. Preserve that failure and all attempts. Do not substitute another recipe.

Only the two MobileNetV2 variants enter selection. Each must strictly reduce
both confirmation MAE and absolute signed mean error versus the standard BIE
and must keep every training source's MAE at or below its standard result.
Among passing variants take the one with the lowest confirmation MAE, then
the smallest BIE file, then datapath-only. Confirm native execution, input and
output identity and completeness before making the decision. The ResNet18
comparator is descriptive, with the same three comparisons reported separately.
Neither architecture can advance to NEF from a training result.

## Next qualification

Freeze one candidate-specific policy for the selected binary before a fresh
development readout. Bind its actual BIE, training selection, effective
precision report, unchanged source/optimized graphs, checkpoint, calibration,
prepared inputs, scorer and all of the reference identities from the appropriate
original policy. Require the same eight unrounded engineering limits and
cardinality. In order, the numerical limits remain 0.05 and 0.15 for bag
shifts, 0.10 and 0.20 for matched radiation-contrast shifts, and 0.05 for
increase in bag MAE. Require zero strong sign reversals (absolute FP32 delta
at least 0.20), strict positivity of all four paired four-hour effects and
finite, correct pairing of all 2,400 nuclei, 24 bags and 12 contrasts. Use
the same sham, dose, time and sample aggregation rules as the original policy.

Call this a **known female development regression**. Perform one readout of
the exact selected binary. Any failure blocks compilation. Recheck the exact
sealed hashes and all eight decisions before compiling. Follow a passing
compile with descriptor, input-packing and numerical simulator parity prior
to seeking separate physical approval. The historical INT8 policies keep their
original candidate digests and decisions.

The metadata-only source inventory for the available BPS subset has six
development sources and three historical reserved sources, with no remaining
registered source for independent confirmation. Inventory SHA256
`573992e0aa03b586b655e4504f58d4d203ef05535b4434060dafedc631a4ed56`.
For a new collection, record independent biological source IDs and acquisition
identities, strain/sex, sample and plate-well crosswalks, physical pixel size,
channel identities and reference measurement protocol before enrollment.
Exclude all historical source IDs and aliases, including re-imaging of the
same source. Aim for complete sham/exposed pairs at 4, 24 and 48 hours for
both radiation branches. Plan source recruitment and uncertainty with an
assay scientist before observing outcomes. Freeze the membership, endpoint,
reference assay, missing-data rules, biological margins and sample size
against a stated precision or power objective. Blind the per-sample reference
until the single candidate's predictions are fixed. Aggregate within samples
and treat biological sources as the independent units. Engineering margins
alone do not establish biological equivalence.

