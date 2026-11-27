# ResNet18 female-fold full-neural INT16 development regression

**The selected BIE passed all eight engineering limits on the known female
development panel.** One Dynasty invocation evaluated the 2,400 previously
prepared nuclei. Original FP32 and both BIEs were compared on the same
tensors, 24 sample means and 12 matched exposed-minus-sham contrasts.
The selected BIE SHA256 is
`3f654f267ae280f27d6aaf6715b32af4285c5de6f4eb13c844e02e70f2f69c9b`.
Its recorded development policy SHA256 is
`94a7ba514a32676c32b30f745faab5c037b729de89e3c8d29d5e854ffe656cc8`.
The original INT8 BIE retains its 6/8 result.

## Frozen gate decisions

Absolute differences and MAE changes are in foci/nucleus. Numerical values below
are rounded for display. Both independent scoring passes used unrounded values.

| Decision | Required | Standard INT8 | Full-neural INT16 |
| --- | ---: | ---: | ---: |
| Mean absolute bag shift | ≤ 0.05 | 0.085603, fail | 0.013386, pass |
| Maximum absolute bag shift | ≤ 0.15 | 0.168735, fail | 0.025160, pass |
| Mean absolute matched-delta shift | ≤ 0.10 | 0.050084, pass | 0.007054, pass |
| Maximum absolute matched-delta shift | ≤ 0.20 | 0.088205, pass | 0.012602, pass |
| Increase in bag MAE against NASA | ≤ 0.05 | 0.014408, pass | 0.001981, pass |
| New sign reversals for strong FP32 effects | 0 | 0, pass | 0, pass |
| Positive four-hour effects | 4/4 | 4/4, pass | 4/4, pass |
| Correctly matched and finite nuclei / bags / contrasts | 2,400 / 24 / 12 | Complete, pass | Complete, pass |

## Results by biological source

Each source contributed 12 bags and six contrasts. Every bag mean and every
matched delta remained below original FP32 with both binaries. The reduction
in mean absolute bag shift was 84.4% across the panel. The original and
optimized FP32 bag means were identical.

| Source | INT8 bag MAE | INT16 bag MAE | INT8 matched-delta MAE | INT16 matched-delta MAE |
| --- | ---: | ---: | ---: | ---: |
| BALBCF1 | 0.087790 | 0.013837 | 0.055448 | 0.007693 |
| C57BLF1 | 0.083416 | 0.012935 | 0.044720 | 0.006416 |

The full-neural four-hour deltas for Fe and X-ray were 0.804241 and 0.920780
for BALBCF1, and 0.874697 and 0.643186 for C57BLF1. All 12 directions
agreed with the original model. Both FP32 and the full-neural BIE agreed with
the NASA reference direction on 11 of 12 contrasts. On the 24 sample references
the absolute MAE was 0.695999 for FP32 and 0.697980 for full-neural INT16.
These references apply to sample means. The evaluation did not enroll an
independent biological source. [Independent-source study design](NASA_RESNET18_INDEPENDENT_SOURCE_DESIGN.md)

## Evidence and timing

The external campaign directory is
`scratch/nasa_resnet_neural16_dev_20261127_v1` under the data root.
The independent audit opened all 100 archives, checked all 2,400 successful
native job records, and verified the raw little-endian signed words against the
vendor floats. The effective width was 15 bits at radix 12 and scale
1.3206359148025513. There were 1,226 nonzero outputs and three BALBCF1 nuclei
reached the maximum native code of 16,383. A second offline audit reproduced
every decision and paired-table digest. The development BIE readout produced
zero public transport observations.

| External evidence | SHA256 |
| --- | --- |
| Authenticated preparation | `463d448f3b3af66f9ce99965fee4fa731e4148294940af767f7197c96500820f` |
| Native simulation receipt | `053f3351f0c43df35e19abd8088f7f56f459895bc05f1f1294b575ebe78e43e1` |
| Ordered predictions and native words | `5e53b7fc8ac90a29000fab3d367a58787722e45ea1d316dfe963023c48d7bbda` |
| Official comparison | `b621663167bd60272b84e2e5a279251b28b4e79d3a28d621713768d671179736` |
| Independent audit | `4f9dd6f061586840636de0aa4292944f880bef6325e4f0dcc0f90f7aa571bf20` |
| Independent audit replay | `5ef0217c61b075c1c16936535f46a6f1518f9a8934dc1c25444de55c900cccd4` |
| Paired historical INT8 comparison report | `4dbde1c00aa0428bc2bbedf60a33b2c86a307cc8972211099143d14d9f8bcd1f` |

The pinned 0.33.1 Docker invocation took **7,484.55 s** wall time. Six concurrent
native jobs processed each 24-input batch. The median per-job wall time was
16.53 s. The earlier vendor IP estimate was 66.7988 ms per inference.
The rehashed full-neural BIE occupies 47,334,682 bytes. The historical standard
INT8 BIE occupies 47,304,479 bytes, correcting the standard file size printed
in the earlier cross-architecture report. Its recorded SHA256 is unchanged.

The paired PDF, PNG and source-level tables are in `paired_report`. Their
report binds the original INT8 comparison and the new independent audit.
[Windows commands and offline verification](NASA_RESNET18_NEURAL16_DEVELOPMENT_RUN.md)

## KL720 compilation

The compile guard reopened the signed archive inventory, paired tables, host
invocation, model bindings and independent eight-gate decision. Kneron 0.33.1
compiled model 32783, version 1, for KL720 with 82 NPU nodes and zero CPU
nodes. The resulting NEF is 51,392,996 bytes, SHA256
`7f1843a62bb5244048f4df977051b6795f7a39320c6be890cc0fdd9a7f92d4ba`.
The compilation receipt SHA256 is
`d068c2b2647edd6a75749bae2d34b7f50e241843a42ec5ed1ac62f92c8130f06`.

The compiler allocated 55,062,636 bytes of address space, including
51,364,864 bytes of weights, a 2,097,152-byte input, a 16-byte aligned
output and 1,572,864 bytes of working buffer. Its high-water address leaves
36,032,404 bytes in the installed PLUS reference firmware's model region.
This establishes feasibility against that map. Physical load and USB
latency require a device measurement. The compiler call took 31.00 s and
the enclosing Docker invocation took 44.55 s.
