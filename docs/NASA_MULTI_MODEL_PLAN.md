# NASA public-data multi-model NPU plan

**Priority decision: 2026-11-17.** This is the current development plan for
Radiation Edge AI. It supersedes the original equal-priority three-assay
roadmap and the former NASA pilot sequence.

## Goal

Build as broad a collection as practical of different models that achieve
high accuracy on public NASA microscopy data and execute locally on the
physical KL720. Expand model breadth through one reusable benchmark and
deployment workflow. Biological endpoint fidelity remains part of accuracy
assessment; it is not a reason to limit development to the original model.

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

## Architecture queue and honest model counts

| Candidate | Purpose | Evidence at this planning checkpoint |
| --- | --- | --- |
| Frozen NASA R1 v2 / `R1CountNetV1` | Known working compact CNN and deployment anchor | Physical NASA validation recorded; retain original weights and interpretation |
| MobileNetV2 with scalar regression head | First depthwise-separable backbone comparison | Kneron KL720 example exists; NASA adaptation and execution pending |
| ResNet18 with scalar regression head | First residual backbone comparison | Kneron KL720 example exists; NASA adaptation and execution pending |
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

Refactor a new benchmark entry point around a model factory; existing R1 v2
training instantiates `R1CountNetV1` directly. Reuse appropriate preprocessing,
bag aggregation and scoring functions while preserving historical frozen
scripts and hashes. A new model profile must not overwrite the old model's
identity or imply that its acceptance transfers automatically.

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

1. Freeze the common benchmark specification, grouped evaluation roles and
   accuracy criteria; inventory available unused NASA sources.
2. Implement a reusable architecture factory and training/export adapter for
   the existing compact CNN, MobileNetV2 and ResNet18. Preserve the historical
   baseline and perform complete-graph feasibility checks early.
3. Train and evaluate the new architectures under the common protocol; carry
   promising models through conversion and physical KL720 evaluation.
4. Complete NASA batch/model-profile integration and generate the common
   model comparison report, then expand the compatible architecture queue.

DNAi, Perilipin2, general GUI expansion and new assays are secondary unless
they directly unblock this NASA workflow or the owner reprioritizes them.
Existing side-branch results remain useful evidence, with no requirement to
finish every branch before starting the NASA benchmark.

## Current evidence versus planned capability

At this planning checkpoint, the frozen NASA R1 v2 model is the established
physical reference. Its seven historical gates passed, with a latent burden
endpoint, source-limited validation and documented metadata-normalization
recovery. See the [post-release audit](NASA_R1_V2_POSTRELEASE_AUDIT.md) for the
current wording of the result.

The multi-architecture benchmark, new model results and unified NASA KL720
batch path are **planned**. This priority update does not claim that any
additional NASA architecture has been trained, compiled or run on hardware.
