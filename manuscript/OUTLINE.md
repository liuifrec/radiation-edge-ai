# Manuscript Outline

**Scope updated 2026-11-17:** NASA public-data benchmarking of multiple accurate
models for local NPU inference. This outline follows the
[current project plan](../docs/NASA_MULTI_MODEL_PLAN.md). It describes planned
analyses, not completed multi-model results. Venue selection follows the final
evidence package.

## Working title

**Benchmarking accurate local NPU inference across neural-network architectures
for NASA radiation microscopy**

## Central research question

Which model architectures can execute locally on an edge NPU while achieving
high accuracy on public NASA microscopy data, and how do architecture and
quantization affect biological measurements and complete workflow performance?

## Intended contribution

- A reproducible NASA public-data benchmark with explicit task and grouping.
- A growing collection of physically executed, accuracy-assessed models.
- A common comparison of reference accuracy, deployment-induced error,
  architecture breadth, compatibility and computational cost.
- An offline inference workflow with versioned models and auditable outputs.

The frozen NASA R1 v2 result supplies a baseline. The new study must establish
its additional model results. DNAi and Perilipin2 are secondary context; a
successful three-assay comparison is not a prerequisite for this manuscript.

## Abstract structure

1. Problem: accurate microscopy inference should be available on local PCs
   with limited compute and no cloud dependency.
2. Objective: compare multiple architectures on a public NASA task and measure
   their actual NPU execution and accuracy.
3. Methods: grouped evaluation, shared training/scoring, FP32 and converted
   models, physical KL720 predictions and complete local timing.
4. Results: report only completed model counts, accuracy, deployment gaps and
   performance; include blocked and unsuccessful candidates.
5. Impact: a reproducible basis for selecting local NPU models for NASA
   microscopy and extending the collection.

## Results plan

### 1. NASA benchmark, task definitions and source-aware evaluation

- Public BPS/OSD-366 data provenance and metadata crosswalk.
- Initial task: sample-level 53BP1 burden regression using aggregate `avg_nfoci`.
- Native-scale 256 x 256 FITC/DAPI input and sample-bag aggregation.
- Grouped development roles and availability of independent confirmation data.
- Explicit separation from the previously examined historical final holdout.

### 2. Accuracy across multiple FP32 architectures

- R1CountNetV1 and additional backbones retrained in the same grouped folds;
  the frozen R1 v2 checkpoint remains a separate historical anchor.
- Common task, preprocessing, loss and declared training/search budget.
- Architecture families, capacity variants and pretrained initialization.
- Bag MAE/RMSE, agreement, source-specific errors and radiation contrasts.
- Group-aware uncertainty and failure analysis.

### 3. Conversion and physical local NPU inference

- Complete-graph feasibility, export parity, calibration, BIE and NEF stages.
- Physical execution of the exact trained models on KL720.
- Model-by-model compatibility and failure records.
- Explicit CPU preprocessing/postprocessing and any hybrid network computation.

### 4. Accuracy retained after deployment

- Physical predictions versus NASA reference labels.
- Paired FP32-to-NPU changes in predictions and biological endpoints.
- Qualification against requirements defined before candidate evaluation.
- Distinguish numerical agreement, biological decision criteria and formal
  statistical equivalence; use the latter only if actually tested.

### 5. Accuracy, breadth and practical inference cost

- Model/family counts with attempted-model denominators.
- Accuracy versus measured latency, model size and memory where available.
- Cold start, steady device execution and complete input-to-report timing.
- NASA physical batch inference and explicit model selection.
- Demonstrate the local workflow without a CUDA or cloud inference dependency.

## Discussion

- Which architectures provide useful accuracy and local deployment tradeoffs?
- When does quantization alter the sample endpoint despite close predictions?
- How do compiler restrictions and training choices constrain model breadth?
- How can a shared workflow make additional models reproducible?
- Limits: aggregate supervision, biological-source coverage, available labels,
  historical-panel exposure and hardware/toolchain specificity.
- Additional NASA tasks/datasets and other NPU targets as future extensions.

## Methods and reporting requirements

- Public data versions, selected images, task labels and source grouping.
- Training/calibration/evaluation separation and model-selection policy.
- Model definitions, initialization, seeds, losses and training/search budgets.
- Preprocessing, ONNX parity and pinned conversion/runtime identities.
- Physical prediction collection and sample-level aggregation.
- Accuracy requirements, effect metrics, uncertainty and timing definitions.
- Model/profile manifests, result verification and reproducibility commands.

## Current evidence boundary

The historical R1 v2 physical deployment passed its original seven biological
criteria. The [post-release audit](../docs/NASA_R1_V2_POSTRELEASE_AUDIT.md)
governs interpretation: latent per-nucleus burden, sample-level comparison,
preserved reference-agreement pattern, limited source coverage, and the
documented metadata-normalization recovery. This is not a direct focus-count
accuracy result or automatic validation of additional models.

## Data and code availability

Release code, model definitions, redistributable manifests, configuration,
benchmark tables and reproducibility instructions. Identify public NASA
sources and upstream model terms. Model binaries and microscopy images remain
outside Git under the existing project storage rules.
