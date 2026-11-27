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

## Compiled simulator qualification

All **48/48** prospectively selected training witnesses passed the frozen
0.0001 foci/nucleus absolute tolerance with zero relative tolerance. The
maximum observed absolute difference was **0.0**. These witnesses test the
compiled interface. The development BIE regression above used its separate
2,400-nucleus panel.

The typed reader verified the setup bytes embedded in this exact NEF. The
input is signed 16-bit `1W16C8BHL`, shape `[1,3,256,256]`, radix 15, scale 1.
Host-packed input matched all 100,663,296 SDK bytes. The scalar output
`burden` has logical shape `[1]`, accelerator shape `[1,1,1,1]`, signed
16-bit `8W1C16B` layout, radix 13 and binary32 scale bits `0x3fa90a99`.
The independent audit reversed the input packing, checked every archived file
hash and decoded both scalar carriers for all 48 cases.

Each ordinary `.seq` output equaled the original signed native BIE word.
Each observed `.seq.16B` transport word equaled twice that native word. There
were 20 nonzero cases and 28 zero cases. On the predeclared positive input,
the BIE emitted 3760 at radix 12 and the NEF emitted 7520 at radix 13.
Both decode to 0.6950960159301758 foci/nucleus. The public simulator float
helper returned 0.34754798066991965 on that same input. The qualification
used the actual signed carrier with the artifact's typed descriptor, as
specified before the readout.

In the pinned image, the public `kneron_inference` path calls
`sys_flow.inference.inference_csim_v2`. Its `sys_flow.csim_utils.txt2np`
reads the ordinary `dma2seq_0.seq`, and `sys_flow.dynasty_v3.np_fx2fl`
applies the NEF's radix 13 to that ordinary word. This explains the
half-sized public value on the positive witness. The vendor's KL720
device-comparison path, `sys_flow.test_case.btm_csim_nef`, selects
`dma2seq_0.seq.16B` for INT16 model output. Unpacking this same NEF through
the public SDK produced an `ioinfo.json` with SHA256
`7c39bb6062527be4a5e39f8c97492900e8878054a108c165e18adc150c4d56b4`
and scale 1.320636. Dividing ordinary word 3760 by `8192 × 1.320636`
exactly reproduces the public diagnostic. The signed
transport decoder reads the original binary32 scale. The installed module SHA256s are
`608563b37efd4b4ae888cff8edabadcaa5071f4c3e39edd9f7908820871a261a`
for `csim_utils`, `387cc2e90068cdc75434eb4171f62edf718d5ec1ba8b96e9fcf44f04e78efb76`
for `dynasty_v3` and `f6301fd758824c335fe437906ea1e87ea121731bcfa5f18975e18c2da80ec8fc`
for `test_case`.

| External evidence | SHA256 |
| --- | --- |
| NEF simulator qualification | `1706ad42edcf401b055077534e6839e63597449f783a0422ab70ebce4d60d59b` |
| Independent NEF archive and input audit | `e10031a6ab672516d16028c6c9ebf620cf6485d80501b5f46660792e0533db31` |
| Independent SDK input-packing preflight | `8330315c4a06ac33bd79d6c71eb78bb072efecd2f65b392c52b9658e07f03b5c` |
| Offline PLUS smoke command | `e07910e608f51522dcad21edc262bc78ff1f52ee85df23ab48b3064ff514b1f4` |
| Offline PLUS training-panel command | `725d77106a7d04bfb3a16fc9b2210f873e9b8bc12a7d1c448ce53976c8589147` |

The complete NEF simulator invocation took **10,128.59 s** wall time. The 48
direct simulator calls averaged **204.05 s** and had a median of 201.89 s and
a range of 200.09–215.00 s. The separately measured public-helper diagnostic
took 208.30 s. The simulator invocation includes that diagnostic and setup.

The Windows smoke and panel launchers both passed offline preflight using
PLUS 3.2.0. Each checked all 48 measured carriers and their SDK-packed inputs.
The next hardware step requires separate operator approval for a one-input
smoke on KL720 `0xAA06244C`, firmware `2.2.0-build.615`, with exact word
7520 as its acceptance condition. Device load, USB latency and physical
panel results remain unmeasured. A separately approved 48-witness physical
replay requires a passing physical smoke receipt. The optional 2,400-nucleus
physical replay also requires generating a complete development NEF
simulator reference. [Guarded Windows commands](NASA_RESNET18_NEURAL16_WINDOWS.md)
