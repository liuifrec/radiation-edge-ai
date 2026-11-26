# NASA 53BP1 MobileNetV2 female-fold KL720 protocol, version 1

Declared 2026-11-26 UTC (2026-11-27 in Japan), before the development-panel BIE readout.
These provisional engineering margins govern eligibility for NEF compilation. Statistical
equivalence and biological accuracy on independent sources require separate evidence.

## Candidate and inputs

Use the existing `mobilenet_v2/fold_female_role` checkpoint, model ID 32781, its already
screened graph, and the pinned KL720 image. The sole candidate for this readout is the
standard signed INT8 PTQ result from all 192 frozen training-only tensors. Record a
descriptive fixed-point pilot on the first calibration member of the first four bags by
lexical bag name per training source. Check input order, finite scalar retrieval and
the vendor dumps. Use the first technically successful standard conversion. There is
no selection on pilot accuracy or choice among precision variants for this panel.
A technical failure before readout requires a separately recorded decision before any
secondary fold. Male-1 is excluded. After readout, failure ends this candidate.

| Frozen input | SHA256 |
| --- | --- |
| Model checkpoint | `bb00a25e2db05334e113852760414adafa22f95614e4c43f92b6607e690bd543` |
| Training run | `861e93ef0a128fdab8f81cb2d992a3fce8833aa92fdc09494c9008ebad4fe5b5` |
| Original ONNX | `2f18b2330ca3289ab422e1ff47b526b4bfa6a46fc4bd7354346609715f575e45` |
| Screened optimized ONNX | `75824e1ed4f8520d90a942c1351926b922db9b4284b8b6d7cd5663428e73c074` |
| Export / graph-screen receipt | `40b89d4b349dfa4060e4fd62a31cf27647b6fb04c1bdef93b4f7b84165f32257` / `9bab4fa38ef118a2da7cfbaa9f7fa710eb75f02c132c0e3d87be945c53dad983` |
| Training calibration manifest | `79aef81e921a723187d73cef5151ff9b943ad486d7e83af19daf063a5b5a1489` |
| Development manifest | `23e21fdc8bf00a2814b64c56223cfd3d8e96eda71bdbff5d8ec74dae218cf5a9` |
| Frozen nucleus / bag predictions | `13fe8bec71a7077f5c98e4548d1cf845550140837e55972b5e99dac8ea98379f` / `374355b631164f1c8a0ae7e587dfb7fc41c099c924861d8f5611c5fe5b12ca7f` |
| Frozen NASA sample reference | `fcbf1339805e02a7ca9cec36b476438b0512b222bd70ac2f90ed75c238a0b01f` |
| Nucleus-to-bag membership | `b0b1b89000bccb73225c809eede7ded6b11736b881cf1fadc71630c39bdd5033` |
| Matched-contrast membership | `19c5d8ed467dfa979f99fd5728c7c951f6bb3927b83872b653f8b1bb36c1f82f` |

Verify the checkpoint and training membership, the 192 calibration hashes and FP32
original-versus-optimized parity before PTQ. Bind the generated BIE, training PTQ
receipts, prepared FP32 panel and this document by SHA256 in the prospective candidate
JSON before fixed-point simulation of any held-out nucleus. Keep the original logs,
per-input hashes, decoded predictions, fixed codes and run times outside Git.

## Evaluation and decision

The development panel consists of BALBCF1 and C57BLF1. Each of 24 plate-well samples
contains 100 distinct nuclei. Evaluate identical normalized, centered 256 × 256
FITC/DAPI/zero tensors with original FP32 ONNX, optimized FP32 ONNX and BIE. Follow
the original export's channel order and packing. Use arithmetic means within each
bag and give each bag equal weight. Use NASA `avg_nfoci` as the bag-level reference.
Each of 12 contrasts subtracts the sham mean from the exposed mean in the same
source, particle and time stratum (Fe 0.82 Gy or X-ray 1 Gy at 4, 24 or 48 hours).
The two canonical membership hashes are calculated without reading predictions.

| Decision, applied to unrounded measurements | Inclusive limit |
| --- | --- |
| Mean absolute difference between BIE and original FP32 bag means | ≤ 0.05 foci/nucleus |
| Maximum absolute difference between BIE and original FP32 bag means | ≤ 0.15 foci/nucleus |
| Mean absolute difference between BIE and original FP32 matched contrasts | ≤ 0.10 foci/nucleus |
| Maximum absolute difference between BIE and original FP32 matched contrasts | ≤ 0.20 foci/nucleus |
| Increase in bag MAE against NASA, BIE MAE minus original FP32 MAE | ≤ 0.05 foci/nucleus |
| Opposite signs for contrasts with absolute original FP32 delta ≥ 0.20 foci/nucleus | 0 reversals |
| Positive four-hour effects | All four original FP32 and corresponding BIE deltas strictly positive |
| Completeness, identity and finiteness | 2,400 paired nuclei, 24 complete bags and 12 complete contrasts, all finite |

Report exact zeros separately. A zero has no opposite sign and fails strict positivity.
Report BIE minus original FP32 bias and break down errors by biological source, radiation
type and time. The scorer records all eight decisions independently. An invalid panel
cannot advance. Compile only the exact BIE passing every criterion. Before proposing
USB execution, authenticate the compiled model, verify simulator descriptors and
decoding against the BIE, and qualify input packing. Physical execution requires
separate approval. This engineering panel is previously examined development material.
