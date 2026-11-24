# NASA public-data multi-model NPU plan

**Benchmark design declared 2026-11-17; progress links updated 2026-11-24.**
This document retains the NASA benchmark design and its original implementation
checkpoint. The [current project status](PROJECT_STATUS.md) is the authoritative
progress summary; the [evidence ledger](PROGRESS_EVIDENCE_20261124.md) records the
later investigator-reported GPU campaign, conversion failures and physical PoC.
The scientific task, folds and frozen acceptance criteria are unchanged.

The [project charter](PROJECT_CHARTER.md) also retains faithful implementations
of released models as a reusable engineering collection, even when they are
unsuitable for NASA 53BP1 quantification. Deployment qualification and
task-specific accuracy are separate statuses. The immediate software milestone
is the [PoC source integration](POC_INTEGRATION_PLAN.md); the next NASA research
milestone remains an additional accurately evaluated physical architecture.

## Goal

Build as broad a collection as practical of different models that achieve
high accuracy on public NASA microscopy data and execute locally on the
physical KL720. Expand model breadth through one reusable benchmark and
deployment workflow. Biological endpoint fidelity remains part of accuracy
assessment; it is not a reason to limit development to the original model.

The scientific framing is **edge AI for space radiation biology**. The
[space-biology narrative](SPACE_RADIATION_BIOLOGY_NARRATIVE.md) provides the
primary-source context and evidence plan. The current platform is a ground
demonstrator whose next result is an additional accurately evaluated NASA
architecture executing on physical KL720 hardware.

## Public data and first task

The initial common dataset is the
[NASA BPS Microscopy Benchmark](https://registry.opendata.aws/bps_microscopy/),
with the existing OSD-366 metadata/reference crosswalk (DOI
[10.26030/v8w4-rg83](https://doi.org/10.26030/v8w4-rg83)). The NASA-managed
benchmark provides public microscopy data without an AWS account. Record
source versions, access dates and selected metadata/image hashes because the
collection can change. Reference access information was checked on 2026-11-17.

Start with the already implemented **sample-level 53BP1 burden regression**:

- supervision: NASA sample/plate-well `avg_nfoci`;
- prediction: a continuous latent score for each selected nucleus;
- endpoint: arithmetic mean within the corresponding sample bag;
- initial input: float32 `[1,3,256,256]`, normalized FITC and DAPI plus a zero
  channel, native-scale crops and center padding;
- shared nucleus selection, bag membership, preprocessing and scorer.

Do not replicate aggregate labels onto nuclei as independent focus-count
ground truth. Direct focus counting or segmentation needs appropriate labels.
Additional NASA tasks or datasets can broaden the collection after the common
workflow is established, with separate task-specific comparisons.

The current input/endpoint definitions are documented in the
[R1 v2 contrast policy](NASA_BPS_R1_V2_CONTRAST_POLICY.md) and
[physical validation record](NASA_BPS_R1_V2_KL720_VALIDATION_RECORD.md).

## Architecture queue at the 2026-11-17 checkpoint

This table preserves the starting queue. Training and screening subsequently
reported from Windows are summarized in [current status](PROJECT_STATUS.md);
the pending labels below describe the original checkpoint.

| Candidate | Purpose | Evidence at this planning checkpoint |
| --- | --- | --- |
| Frozen NASA R1 v2 / `R1CountNetV1` | Known working compact CNN and deployment anchor | Physical NASA validation recorded; retain original weights and interpretation |
| `R1CountNetV1` retraining candidate | Fold-matched compact CNN comparator | Full untrained 256 x 256 graph reported supported by the real KL720 toolchain; new grouped training pending |
| MobileNetV2 with scalar regression head | First depthwise-separable backbone comparison | Full untrained 256 x 256 graph reported supported; NASA accuracy, trained-weight conversion and physical execution pending |
| ResNet18 with scalar regression head | First residual backbone comparison | Full untrained 256 x 256 graph reported supported; NASA accuracy, trained-weight conversion and physical execution pending |
| ResNet50 with scalar regression head | Higher-capacity residual comparison | Vendor model listed; exact geometry, resources and NASA graph still require checking |
| Further compatible families / capacity variants | Grow breadth and explore accuracy/cost tradeoffs | Admit through full-graph feasibility and the shared benchmark |

Kneron's [official Model Zoo](https://github.com/kneron/Model_Zoo), checked on
2026-11-17, lists MobileNetV2, ResNet18 and ResNet50 with KL720 examples at
224 x 224. Our initial NASA contract is 256 x 256 with a different output head
and channel semantics. Those changes require actual full-model compilation
and numerical checks. Published ImageNet accuracy or vendor FPS are not NASA
accuracy or measurements from this project.

Record separately:

- **architecture family** (e.g. residual or depthwise-separable network);
- **architecture variant** (specific depth, width and head);
- **trained model identity** (weights, initialization and training recipe);
- **deployment variant** (precision, quantization and compiler configuration).

Report family count and qualified model count, including attempted/failed
models in the denominator. Different quantization settings or random seeds
do not inflate the architecture count. The initial queue is a starting point,
not a permanent cap.

## Common evaluation protocol

### Data separation

Reuse the existing metadata resolution and integrity checks, then freeze the
new benchmark's development membership and grouped folds before training.
Keep all related samples from a biological source together; retain any higher
acquisition/experiment grouping needed to prevent leakage. Select calibration
data only from the permitted training/development partition, with fixed input
selection across comparable candidates.

Calibration is fold-specific: use only that fold's training sources and exclude
the current held-out fold even if its labels are not read. After model selection, final-fit
calibration may use all permitted development sources, still excluding any
independent confirmation cohort.

The former final holdout (2,058 nuclei / 22 bags / three Source Names) has
already been examined. For new architecture work it is a **locked historical
comparison panel**, not an untouched model-selection cohort. Do not optimize
architectures, quantization or thresholds using that panel. Use grouped
development evaluation for selection. Reserve genuinely unused sources or
acquisitions for independent confirmation if available; more nuclei from an
already used source do not establish independence. If no independent material
is available, report developmental benchmarking and its limitation explicitly.

### Fair architecture comparisons

Use the same primary task, input contract, bag-level objective, scorer and
declared training/search budget. Report pretrained versus random initialization,
seeds, optimization schedule and training resources. Different input geometry,
losses or supervision belong in named ablations so a backbone comparison
remains interpretable.

Retain the frozen R1 v2 checkpoint as a historical deployment anchor. For the
new grouped architecture comparison, retrain the R1CountNetV1 architecture in
each of the same folds as the new backbones. The frozen checkpoint was trained
on the original development sources and cannot supply out-of-fold predictions
for those sources. Report historical-anchor results separately from the new
fold-matched baseline.

The implemented [model factory](../src/radiation_edge_ai/nasa_bps/model_factory.py)
and [shared training/export entry point](../src/radiation_edge_ai/nasa_bps/multimodel.py)
reuse appropriate preprocessing, bag aggregation and scoring functions while
preserving the historical frozen scripts and hashes. A new model profile
must retain its own identity and acceptance evidence.

### Accuracy means both reference performance and retained performance

| Evaluation | Required reporting |
| --- | --- |
| FP32 and physical predictions versus NASA reference | Bag MAE/RMSE, agreement/bias, source-specific errors, descriptive Pearson/Spearman |
| Radiation-response measurements | Matched-contrast error, direction and time-course behavior where supported by the selected cohort |
| Physical versus the same model's FP32 output | Paired prediction/endpoint differences and deployment-induced degradation |
| Uncertainty | Group-aware uncertainty and per-source results; nuclei are not independent biological replicates |

Define numeric accuracy requirements and permitted deployment degradation
before evaluating the new candidates. Justify them from intended use, reference
variation and development evidence. No universal percentage such as "95%
accuracy" is assigned to continuous regression. The old seven acceptance
criteria remain valid for their historical cohort; several are cohort-specific
counts and are not automatically the new benchmark's definition of high
accuracy. Keep descriptive rankings available even for candidates that do not
meet the qualification criteria.

## Deployment workflow and model states

1. **Candidate:** instantiate the complete NASA-adapted graph and record its
   task/input/output contract.
2. **Feasibility screened:** run ONNX and full-graph vendor checks early,
   before expensive training. Isolated operator support is insufficient.
3. **FP32 evaluated:** train using the shared protocol and report reference
   accuracy on grouped development evaluation.
4. **Converted:** verify trained FP32/ONNX parity, quantize using the frozen
   calibration selection, evaluate BIE and compile a distinct NEF.
5. **Physically executed:** run the exact model on the KL720, record actual
   model/input/output identities and compare with the appropriate simulator
   and FP32 outputs.
6. **Accuracy qualified:** meet the declared reference-accuracy and deployment
   requirements with the evaluation stage named (development or independent
   confirmation). Only these models enter the accuracy-qualified collection.

The precise trained graph is rechecked even if an untrained architecture passed
the early feasibility screen. Blocked and inaccurate candidates remain visible
with a specific reason and next decision. Continue other candidates while a
difficult model is investigated.

CPU image handling, metadata processing and endpoint aggregation are explicit
parts of the local application. If neural-network compute operators execute
on the CPU, including activations or pooling, label that deployment as hybrid
and keep it distinct from full-network NPU execution. Harmless shape bookkeeping
is recorded separately from network computation. Training may use a GPU; local inference requires no CUDA or
cloud service. Compiler estimates do not count as physical timing.

## Runtime and benchmark deliverables

Extend the existing NASA application path to support physical KL720 batches
and explicit selection among versioned model profiles. The present NASA
high-level batch adapter is CPU-only; historical physical validation used a
dedicated runner. Keep hardware/scientific interpreters isolated and amortize
model loading across inputs where the implementation permits it.

Each model's result should identify:

- task, architecture, weights, preprocessing, dataset/split and runtime;
- FP32, BIE and physical accuracy with complete evaluation denominators;
- model size, measured host/device memory where available, conversion status;
- cold-start/model-load time, steady device time and complete input-to-report
  time, including CPU work;
- actual inference versus cache reuse;
- failures and scientific interpretation limits.

Use one comparison table and reproducible report format across models. Energy
can be added when it can be measured reliably; it is not a prerequisite for
the first additional accurate model. Publish code and redistributable benchmark
metadata; keep existing dataset/model-binary storage rules and upstream terms.

## Ordered next work

1. Recover and review the actual frozen Windows source, then integrate the
   qualified runtime, model catalogue and conversion fixes according to the
   [integration plan](POC_INTEGRATION_PLAN.md). A report is not a source import.
2. Preserve the completed three-architecture/nine-fold GPU campaign as
   investigator-reported development evidence. Import and verify its existing
   reports before repeating work; do not treat the historical checkpoint as
   an out-of-fold comparator or pool different training budgets.
3. Select one eligible additional NASA model and declare a bounded conversion
   attempt before its new readout. MobileNetV2 is a candidate; its male-1
   export remains excluded. Keep the ResNet18 failed BIE and original limits
   unchanged. Record the selected fold, checkpoint, calibration and criteria.
4. Complete the selected model's PTQ, BIE, NEF and physical assessment. A
   single-fold success is partial development evidence; complete the intended
   folds before claiming pooled physical out-of-fold accuracy.
5. Finish the general NASA physical batch adapter and measure complete local
   workflow cost. Inventory unused sources/acquisitions for independent
   confirmation, then expand the architecture queue through the shared path.

The [campaign declaration](NASA_THREE_MODEL_CAMPAIGN_V1.md) retains the original
eight development advancement criteria. These are not universal independent
biological qualification criteria. The reused historical panel remains locked.

DNAi v3 and Spotiflow `general` are part of the reported reusable PoC, with
their own biological limits. Other assay and GUI investigations remain
secondary; completing every side branch is not a NASA prerequisite.

## Historical implementation checkpoint — 2026-11-17

The following records what was available at that checkpoint. Later reported
progress and present source availability are consolidated in the
[evidence ledger](PROGRESS_EVIDENCE_20261124.md).

The frozen NASA R1 v2 model is the established
physical reference. Its seven historical gates passed, with a latent burden
endpoint, source-limited validation and documented metadata-normalization
recovery. See the [post-release audit](NASA_R1_V2_POSTRELEASE_AUDIT.md) for the
current wording of the result.

On 2026-11-17, the user's Windows console recorded a fresh 2,058-nucleus,
22-bag replay with one NEF upload, numerical repeatability PASS and mean
host-observed send/receive time of 3.930 ms. The
[PR #31 evidence record](https://github.com/liuifrec/radiation-edge-ai/pull/31)
identifies the source as a user-provided console transcript; the result is
repeatability evidence for the frozen model, separate from its original
biological validation.

The first implementation now provides a fresh physical-baseline replay launcher,
an architecture factory for R1CountNetV1/MobileNetV2/ResNet18, shared source-grouped
training, fixed-shape ONNX export with training-source-only calibration
inputs and parity checks, and early full-graph screening adapters. The latter
use deterministic untrained weights and synthetic inputs, with fresh vendor
evidence required for a supported result. The common new recipe uses deterministic 16-nucleus
training minibags, all-nucleus evaluation and a fixed fold-training-only
BatchNorm recalibration before deployment-state evaluation. It is not a
reproduction of the original training recipe.

On 2026-11-17, the user's Windows console recorded `COMPLETE_SUPPORTED` for all
three complete 256 x 256 candidate graphs using the pinned Kneron image. Each
reported fresh target-720 IP evaluation with no CPU neural computation. The
[campaign evidence record](NASA_THREE_MODEL_CAMPAIGN_V1.md#entry-evidence)
preserves the graph hashes and local receipt path, and identifies this as
user-supplied console evidence rather than an independent artifact review.

See the [NASA local NPU quickstart](NASA_LOCAL_NPU_QUICKSTART.md) for runnable
commands and exact result states. Numerical software checks use synthetic
images. No new NASA accuracy result or new architecture's physical KL720
execution is claimed. Trained-weight PTQ/NEF conversion, independent
qualification criteria, new model results and unified high-level NASA model
selection remain open.
