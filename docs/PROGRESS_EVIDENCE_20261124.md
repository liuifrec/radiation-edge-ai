# Progress evidence ledger — 2026-11-24

This ledger consolidates repository records and investigator-supplied Windows
reports received through 2026-11-20. The [project status](PROJECT_STATUS.md)
uses the entry IDs below. Artifact names are locators from those reports;
listing a name or reported commit does not make its content available here.

**Review boundary:** public source at
`561fed0b3572d19392f2471e0b274840ba468fcd` was inspected on 2026-11-24.
Newer Windows receipts, predictions, weights, BIEs and NEFs were not retrieved,
rehashed or executed during this consolidation. All later measurements below
are attributed to the investigator's reports. No scientific acceptance margin,
frozen artifact or historical decision is changed by this document.

## E01: NASA R1 v2

**Sources:** repository [validation record](NASA_BPS_R1_V2_KL720_VALIDATION_RECORD.md),
[post-release audit](NASA_R1_V2_POSTRELEASE_AUDIT.md), and investigator console
for the 2026-11-17 replay, also summarized in [PR #31](https://github.com/liuifrec/radiation-edge-ai/pull/31).

The historical physical validation covered 2,058 nuclei, 22 sample/well bags
and 11 matched radiation contrasts from three held-out sources. All seven
original biological-fidelity criteria passed. Recorded physical bag
MAE/RMSE/Pearson/Spearman were 0.7922/0.8893/0.9501/0.5729; direction agreement
was 10/11 and peak recovery 3/3. The endpoint is aggregate latent 53BP1 burden.

The post-release audit found one literal FP32-to-physical sign change:
BALBCF2 X-ray at 24 h, approximately −0.00955 to +0.11925, against a zero NASA
reference delta. The same 10 contrasts agreed with the reference and the same
one did not. The seven-gate decision is unchanged. Literal FP32 sign identity
and formal statistical equivalence are not supported descriptions.

The fresh replay reported 2,058 physical inferences, one model upload and
historical numerical repeatability PASS. Host-observed send/receive was mean
3.930 ms, median 4.001 ms and p95 4.408 ms. The historical validation's mean
3.676 ms is a separate run, not a controlled speed comparison.

Reported replay locator: `nasa_kl720_20261117T012356Z_c82c4017/replay_receipt.json`,
with `physical_holdout_nucleus_predictions.csv` and
`physical_holdout_inference_summary.json` in its `hardware` directory.
The replay does not create a new biological accuracy result.

## E02: Three-model NASA campaign

**Source:** investigator reports dated 2026-11-17 and 2026-11-18; the original
[campaign declaration](NASA_THREE_MODEL_CAMPAIGN_V1.md) defines the comparison.

Full candidate graph screening reported support for R1CountNetV1, MobileNetV2
and ResNet18 with no CPU neural computation. This was feasibility evidence,
followed by GPU training on an RTX 2070 Super using PyTorch 2.6.0+cu124.
The campaign completed three architectures × three grouped folds × 60 epochs
in 36 minutes 32.6 seconds. Nine fold checkpoints are not nine architectures.
The common development cohort contains 7,200 nuclei in 72 bags from six
sources; each fold uses four training sources and two held-out sources.

| Architecture | Reported pooled bag MAE | Bag RMSE | Bag Spearman |
| --- | ---: | ---: | ---: |
| R1CountNetV1 | 0.551129 | 0.905577 | 0.655318 |
| MobileNetV2 | 0.533489 | 0.875759 | 0.643754 |
| ResNet18 | 0.602701 | 0.961460 | 0.564676 |

Errors are in foci/nucleus at the aggregate bag endpoint. These are development
results on the common recipe, not an independent comparison with the historical
holdout. The one-epoch GPU engineering check was excluded from rankings.

Campaign locators: `multimodel_v1_20261117T053459Z_b02f5836/benchmark.json`
and `comparison.csv`. The recipe uses 16 training nuclei per bag, 100 for
evaluation, training-only BatchNorm recalibration, fixed final-epoch selection
and no pretrained backbone or historical checkpoint.

The later deployment reports identify five eligible MobileNetV2/ResNet18 fold
exports. **MobileNetV2 male-1 remains excluded:** only 191 of 192 original
export comparisons passed the strict tolerance, before quantization. The
trained-graph vendor screen reported support for the eligible exports;
`nasa_trained_graph_screen_20261118T020501Z_06255b35/screen_receipt.json`
records that stage. IP evaluation is not PTQ, NEF or USB execution.

## E03: ResNet18 PTQ and training diagnostics

**Sources:** investigator reports dated 2026-11-18 and 2026-11-19. The reported
local documents include `NASA_MULTIMODEL_DEPLOYMENT.md`,
`NASA_PTQ_ACCEPTANCE_V1.md`, `NASA_MULTISOURCE_TRAINING_BIE_DIAGNOSIS.md`
and `NASA_RESNET18_OPT1_TRAINING_EXPERIMENT.md`; their source is not imported here.

The first PTQ pilot stopped at an invalid/duplicate calibration-array path.
The later Windows report identified the backslash serialization problem,
repaired the reader and future exports, and preserved the frozen manifest.
Original and optimized FP32 predictions then matched exactly on 192 training
calibration nuclei from 48 bags and four sources. A 47,304,479-byte BIE was
generated and reportedly rehashed against both execution receipts.

The paired development readout covered 2,400 nuclei, 24 bags and 12 contrasts.
All predictions were finite and matched by identity. Original and optimized
FP32 agreed; every BIE bag mean shifted downward.

| Frozen engineering criterion | Limit | Reported result | Decision |
| --- | ---: | ---: | --- |
| Mean absolute bag difference | ≤ 0.05 | 0.085603 | Fail |
| Maximum absolute bag difference | ≤ 0.15 | 0.168735 | Fail |
| Mean absolute matched-delta difference | ≤ 0.10 | 0.050084 | Pass |
| Maximum absolute matched-delta difference | ≤ 0.20 | 0.088205 | Pass |
| Increase in bag MAE against NASA reference | ≤ 0.05 | 0.014408 | Pass |
| New sign reversals where absolute FP32 delta ≥ 0.20 | 0 | 0 of 12 | Pass |
| Positive four-hour responses | 4 of 4 | 4 of 4 | Pass |
| Finite correctly matched outputs | 24 bags / 12 contrasts | 24 / 12 | Pass |

Difference units are foci/nucleus. The largest bag difference was BALBCF1,
1 Gy X-ray, four hours: FP32 1.488885 versus BIE 1.320150. The candidate failed
two of eight criteria and did not advance to qualified NEF compilation.
The reported acceptance declaration was committed before readout at `ae0f085`.

Readout locator: `nasa_resnet18_female_bie_readout_20261118T060725Z_438c78b7/evaluation_receipt.json`.
Successful PTQ locator: `nasa_resnet18_female_ptq_20261118T052424Z_61d42817/host_ptq_receipt.json`.

Training-only follow-up measured 17 native fixed-point tensors on 48 nuclei
from four sources. All 21 positive FP32 predictions decreased and 27 stayed
zero; mean terminal bias was −0.06651. The extractor's mean contribution was
−0.15023, versus −0.00230 for global pooling and −0.00916 for the head.
These contributions precede the terminal output floor and do not sum directly
to the reported mean terminal bias.

SDK `optimize=1` bias adjustment worsened terminal MAE from 0.06651 to 0.07588
and extractor contribution magnitude from 0.15023 to 0.17260. Seventeen nuclei
worsened, 31 tied and none improved. A later trace reported an additional
downward shift at the first residual block in 48/48 nuclei, averaging −0.0442.
The bias-adjusted candidate did not advance. The development readout and
historical holdout were not used for that training-only correction experiment.

## E04: DNAi reference and candidates

**Sources:** investigator physical/diagnostic reports dated 2026-11-18 through
2026-11-20; existing [DNAi characterization](DNAI_PHYSICAL_CHARACTERIZATION_AUDIT.md)
and [v0.15 experimental protocol](DNAI_INT16_OUTPUT_CANDIDATE_V0_15.md).

DNAi v3 repeated all 27 frozen physical windows byte-for-byte and reproduced
the stitched arrays, masks and field comparison metrics. The reference source
was reported at `54672d0`; a later replay followed the Spotiflow run.

| Field | FP512 / v3 fiber counts | Matched fibers | Instance F1 | Foreground Dice |
| --- | --- | ---: | ---: | ---: |
| `1__tile_4` | 14 / 12 | 11 | 0.8462 | 0.8653 |
| `2__tile_9` | 11 / 13 | 9 | 0.7500 | 0.7588 |
| `16__tile_5` | 11 / 10 | 10 | 0.9524 | 0.8449 |

The floating FP512 workflow is the comparison reference; these numbers are not
a complete human-accuracy study. Replay locator:
`dnai_kl720_v3_replay_20261120T030156Z_3cd52afe/hardware_receipt.json`.

V4 also ran physically. Its 27 windows matched archived BIE outputs; the
isolated characterization NEF passed the NEF–BIE gate with maximum logit error
2.86e-6 and no class disagreements. Relative to v3, v4 recovered one reference
fiber supported by three graders and lost two. Its field F1 values were
0.880, 0.818 and 0.900. Improved aggregate F1 on some fields did not erase those
losses. V3 remains the physical reference.

V4 locator: `dnai_v4_physical_20261118T081928Z_c1a59327/hardware_receipt.json`;
the reported comparison source commit was `4a9bc57`.

V5 lost a three-grader-supported fiber on `2__tile_9`; its surviving valid
fragment had bounding-box IoU 0.278, below the frozen 0.5 match threshold.
Other recovered fibers did not satisfy the criterion prohibiting that loss.
Training-only comparisons showed source-dependent gains and losses. A later
candidate combining precision/bias options lost annotated fiber 4 on `V0/1135`,
reducing instance F1 from 0.667 to 0.400. V4 and v5 remain experimental.

## E05: Pretrained native screen

**Source:** investigator's 2026-11-19 native pretrained-model report, with local
document `NASA_PRETRAINED_53BP1_SCREEN.md` and source commit `5b43b85`.

Both released Spotiflow checkpoints ran on the same 1,200 NASA training nuclei
from four sources, 12 wells and six matched contrasts. The historical holdout
remained unused. The reported paired results were:

| Native model | Bag MAE | Radiation-delta MAE | Correct directions |
| --- | ---: | ---: | ---: |
| Spotiflow `general` | 11.724 | 7.255 | 2/6 |
| Spotiflow `synth_complex` | 0.776 | 1.322 | 4/6 |
| ResNet18 FP32 | 0.615 | 0.976 | 4/6 |
| MobileNetV2 FP32 | 0.465 | 0.725 | 4/6 |
| R1CountNet v1 FP32 | 0.572 | 0.988 | 5/6 |
| Deployed R1 v2, original FP32 | 0.560 | 1.005 | 6/6 |

Errors are foci/nucleus at the bag endpoint. This panel uses training sources;
it is separate from the grouped out-of-fold campaign and cannot be substituted
for it. Aggregate NASA references do not establish individual spot accuracy.
In a separate public field with 78 annotated centers, `general` matched 55 of
78 with 90 calls (F1 0.655); `synth_complex` matched none with five calls.

The project owner subsequently prioritized retaining useful pretrained-model
implementations after engineering qualification, even when NASA accuracy does
not qualify. The native scientific results remain unchanged by that decision.

## E06: Spotiflow conversion and physical qualification

**Sources:** investigator reports spanning 2026-11-19 and 2026-11-20, culminating
in the reported local integration commit `fa5047f`. The report names
`SPOTIFLOW_KL720_GENERAL_TRANSPORT_READOUT.md` and `PRETRAINED_KL720_LIBRARY.md`;
those implementation documents are not part of the inspected public base.

The engineering study used 83 calibration inputs and 63 separate readout
inputs, including microscopy and synthetic spot images. Forty-seven readout
inputs had target-nucleus count measurements. These inputs were reused for
declared candidate comparisons; they are a fixed engineering panel.

| `general` stage / configuration | Reported result | Disposition |
| --- | --- | --- |
| Native / original and optimized FP32 | All 63 inputs passed | Eligible for PTQ |
| First INT8 BIE | Point F1 0.826 against native | Failed 0.95 minimum |
| Separate-output INT8 BIE | Point F1 0.867; target-count limits failed | Did not advance |
| INT16 terminal outputs | Point F1 0.949521 | Below unchanged 0.950000 minimum |
| Higher-precision inputs, neural computations, weights and outputs | All eight engineering criteria passed; F1 0.988889; target-count MAE 0.06383 | Advanced to NEF |
| Initial public NEF-simulator float readout | Both output heads failed 1e-6 tolerance on every input despite identical signed words | Failed receipt retained; representation investigated |
| Qualified INT16-stream readout | Worst BIE difference 7.45e-9 per head | Within original 1e-6 limit |
| Physical KL720 readout | All 63 inputs matched the qualified simulator | Physical engineering qualification reported |

The higher-precision candidate is fixed-point inference, not FP32 execution.
The vendor report identified higher precision in all 34 convolutions while
four resize operations still emitted eight-bit values.

The NEF discrepancy was attributed to the selected output stream: the pinned
public simulator used `dma2seq_N.seq`, while the INT16 hardware checker used
`dma2seq_N.seq.16B`. Decoding the latter through PLUS 3.2.0 with the original
NEF descriptors met the original tolerance. The BIE, NEF, thresholds and first
failed readout were preserved.

Physical results reported exact agreement in all 4,128,768 signed output words
and decoded floats, all 1,168 detected points and coordinates, and all 47 target
counts. Against native Spotiflow, 1,157 of 1,172 spots matched, with 11 extras
and 15 misses: F1 0.988889. Target-count MAE was 0.06383 spots/nucleus.
These are deployment-agreement metrics, not manual-reference accuracy.

| Physical timing scope | Median | p95 |
| --- | ---: | ---: |
| USB send through receive | 57.54 ms | 57.79 ms |
| Host input preparation through final spot count | 67.18 ms | 68.79 ms |

Reported environment: KL720, firmware 2.2.0-build.615, PLUS 3.2.0.
Hardware locator: `spotiflow_general_transport_usb_panel_20261120T025850Z_5dce80fd/receipt.json`.
Native comparison locator: `spotiflow_general_transport_usb_panel_native_score_20261120_v1/receipt.json`.

`synth_complex` remains blocked: six of 63 original ONNX comparisons exceeded
the raw-output tolerance despite preserving all 705 spots. Explicit BatchNorm
and substitution of the native final projection did not resolve those cases.
The physical success of `general` does not qualify `synth_complex`.

## E07: Other microscopy workstreams

**Source:** investigator reports dated 2026-11-19 and 2026-11-20. These are
development diagnostics, with local reports awaiting source integration.

| Workstream | Reported measurement | Current decision |
| --- | --- | --- |
| Perilipin2 | Across 29 probes, adding `/Concat_7` to a passing 71-operator prefix reproduced the full-graph DMA-address failure. Splitting the subsequent projection reduced work buffer from 25.2 MB to 19.6 MB but still failed | Complete deployment remains blocked |
| γH2AX | A marker-specific detector matched 11/51 points among 40 calls on one section; a 50-section volume run produced 30,008 peaks and matched 29/270 annotated foci | Readers work; detector accuracy remains inadequate for advancement |
| Micronuclei | No matches among 54 annotated objects and 93 candidates; 82 candidates overlapped main nuclei. Miss tracing assigned 37 to intensity threshold, 3 to no positive response, 11 to area filtering and 3 to overlap matching | Preserve diagnostics; no qualified detector |

## E08: Local freeze and source integration

**Source:** investigator's 2026-11-20 freeze and bundle-verification consoles.

- Reported integration branch: `codex/spotiflow-nef-output-representation`.
- Frozen commit: `fa5047fd20c104de98f62ad39641a17fee1489dc`.
- Local annotated tag: `poc-kl720-multimodel-2026-11-20`.
- Snapshot directory name: `kl720_multimodel_poc_20261120_fa5047f_20261120T043929Z`.
- Snapshot manifest: `freeze_manifest.json`.
- Git archive: `radiation-edge-ai-local.bundle`; Git reported it valid and
  recording complete history, with 98 refs.
- Snapshot copied 12 evidence files and hashed 12 Spotiflow binary files.
  The actual binaries still require an independent backup; a successful Git
  bundle verification does not verify or restore external artifacts.

The local integration report also records 579 passes and 19 skips in the NASA
environment, 77 offline Spotiflow passes and a passed opt-in physical USB test
in their respective environments. These overlap and are not summed as distinct tests.

On 2026-11-24, the public repository still pointed to `561fed0`; neither the
frozen commit nor its PoC tag was retrievable there. The source inspection and
fresh 350-pass/25-Windows-skip check apply to that public base only. The
[integration plan](POC_INTEGRATION_PLAN.md) keeps importing the actual Windows
implementation and verifying external artifacts as explicit remaining work.
