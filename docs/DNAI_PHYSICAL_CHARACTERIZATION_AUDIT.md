# DNAi physical KL720 deployment-characterization audit

Status: **completed read-only evidence and sensitivity audit, 2026-09-16**.

The frozen 20-image / 180-window DNA-fiber panel is a
**development/deployment characterization panel**, not an untouched
confirmatory holdout. It had already contributed to overlap and PTQ candidate
decisions.

No neural inference, KL720 hardware execution, PTQ, model selection, or
historical result rewriting was performed during the post-hoc audits described
here.

Evidence identities:

- DNAi upstream commit: fcf20c7d6eb385675ff7d07da4fdf471589ce0cf
- optimized 512 ONNX SHA256:
  a901d1b309a9a0e5026febd5070252787e4b110a2a7d2828e16a19364d6094d0
- physical v3 NEF SHA256:
  4b3dfec9a61c99e186dd4b8482fa5b06e6a4958f325ed4a0db0546f1dcab2bfc
- completion-state audit SHA256: f5ae998ad72bb781d1e90c1d467d8d99538d9aa1b6ca39cce012ab0989db03b4
- matching-sensitivity audit SHA256: c9210d1983c2c159134fb90ea380a6db11a1c400689387f84ab016b5a2c26766
- denominator-aware audit SHA256: 523eb031f30d535141b8b274a45ea6bde1a44df7cb02d6ce53571e35e607fd6e
- preserved legacy biological-characterization summary SHA256:
  14ae38abe4b8fdc2a19639c42d45345916ffaab855557a0ede0354d3d7968ab2

## Completion state

The physical deployment dataset is complete:

- physical windows: 180/180
- images: 20
- windows per image: 9
- frozen tiling: 512 x 512
- overlap: 50%
- blending: Gaussian

The pre-existing full hardware biological-characterization result was found and
preserved rather than rerun.

Recorded physical timing:

- mean host packing: 5.693 ms/window
- mean send + receive: 112.842 ms/window
- median send + receive: 112.884 ms/window
- estimated throughput: 8.862 windows/s
- 180-window panel wall time: 42.713 s

These are not end-to-end raw-image workflow timings.

## Historical physical-versus-FP512 characterization

Per-window:

- mean argmax agreement: 0.9987726
- minimum argmax agreement: 0.9934540
- disagreement pixels: 57,915 / 47,185,920
- mean raw-logit RMSE: 0.8993

After nine-window stitching, all 20 images:

- mean pixel argmax agreement: 0.9989050
- mean foreground Dice: 0.8070254
- minimum foreground Dice: 0.0
- exact fiber-count fraction: 0.45
- mean per-image fiber-match F1: 0.8172639
- mean matched-fiber ratio MAE: 0.212281
- mean matched-fiber total-length MAE: 0.986135 um

The very high pixel-label agreement therefore does not imply identical
post-processed biological objects.

## Human comparison

All-20 mean comparison:

Foreground Dice:

- strict FP32 1024 baseline: 0.396297
- floating FP512: 0.395840
- physical KL720: 0.353821

Detection F1:

- strict FP32 1024 baseline: 0.530319
- floating FP512: 0.542865
- physical KL720: 0.480620

Human annotations are descriptive evidence rather than infallible ground truth.

## Matching sensitivity

Historical primary matching used greedy one-to-one bbox IoU matching at the
frozen threshold.

A separately labelled maximum-cardinality / total-IoU sensitivity analysis
found exactly the same number of object matches on every image.

All 20:

- floating FP512 fibers: 169
- physical KL720 fibers: 172
- matched fibers: 144
- floating unmatched: 25
- physical unmatched: 28
- pooled object F1: 0.844575

The maximum-cardinality sensitivity matcher produced the same 144 matches and
the same pooled F1.

Therefore the observed object-level discrepancy is not explained by the
historical greedy matching algorithm.

## Reference-defined nonempty population

The historical nonempty definition could be influenced by baseline,
floating-reference, physical, or human detections.

A post-hoc sensitivity population was therefore defined only by independent
reference information:

> strict FP32 1024 baseline contains at least one valid fiber OR at least one
> of H1-H4 contains at least one valid fiber.

This defines 19/20 images as reference-nonempty.

The sole reference-defined empty image is:

- 13__tile_14

Within the 19 reference-defined nonempty images:

- floating FP512 fibers: 169
- physical KL720 fibers: 169
- matched fibers: 144
- floating unmatched: 25
- physical unmatched: 25
- pooled object F1: 0.852071
- mean per-image F1: 0.860278

Thus even identical aggregate fiber counts can conceal changes in object
identity.

## Human-supported object disagreements

Among 25 FP512 fibers unmatched by the physical output:

- support from 0/4 graders: 13
- support from 1/4 graders: 2
- support from 2/4 graders: 4
- support from 3/4 graders: 5
- support from 4/4 graders: 1

Therefore 6/25 unmatched floating-reference fibers had support from at least
3/4 graders.

Among 28 physical-only fibers:

- support from 0/4 graders: 22
- support from 1/4 graders: 1
- support from 2/4 graders: 2
- support from 3/4 graders: 2
- support from 4/4 graders: 1

Therefore 3/28 physical-only fibers had support from at least 3/4 graders.

## Denominator-aware length sensitivity

Among all 74 floating-reference fibers supported by at least 3/4 human graders,
physical KL720 failed to match 6:

- overall: 6/74 = 8.1%
- <10 um: 0/1 = 0%
- 10-20 um: 1/27 = 3.7%
- 20-30 um: 4/20 = 20.0%
- >=30 um: 1/26 = 3.8%

This post-hoc pattern is non-monotonic. It does not support a general claim
that increasingly long fibers are increasingly vulnerable.

The elevated 20-30 um mismatch rate is descriptive and hypothesis-generating
only; the panel was not designed or powered for a confirmatory subgroup test.

## Interpretation

The DNA-fiber deployment should **not** be described as a NASA-style
biological-fidelity PASS.

A defensible summary is:

> Physical KL720 retained very high pixel-class agreement with the floating
> FP512 reference, while biologically relevant object reconstruction was not
> fully preserved. The discrepancy remained under maximum-cardinality matching
> and included several floating-reference fibers independently supported by
> multiple human graders.

This is useful evidence that numerical or pixel-level similarity alone is not
a sufficient surrogate for biological endpoint fidelity.

The 20-image panel must not now be used to select another PTQ variant,
overlap policy, or favorable object matcher.

## Provenance caveat

The historical floating-reference cache accepted cached arrays based primarily
on shape, dtype, and finiteness. It did not cryptographically bind every cached
output to input, model, preprocessing configuration, and runtime/provider
identity.

The new provenance-cache helper is intended for future runs. It does not
retroactively authenticate legacy caches.

A future independent confirmation should preserve:

- original acquisition/experiment grouping;
- input-image and preprocessing identities;
- calibration-source overlap checks;
- model and upstream pretrained ancestry where available;
- annotation provenance;
- channel semantics and pixel size;
- runtime/provider identity;
- content-addressed cached outputs;
- predeclared biological endpoint tolerances.
