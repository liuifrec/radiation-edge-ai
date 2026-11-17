# Project Charter

## Primary direction — effective 2026-11-17

Build an expanding collection of high-accuracy models using publicly available
NASA microscopy data, with as many different model architectures as practical
executing locally on the physical Kneron KL720 NPU.

This owner-directed priority replaces the original three-assay development
order. The [NASA multi-model plan](NASA_MULTI_MODEL_PLAN.md) is the current
operational plan. Historical validation records retain their original scope.

## Objectives

1. **Model breadth:** qualify multiple distinct architecture families and
   useful capacity variants on a shared NASA benchmark.
2. **High accuracy:** measure performance against NASA reference labels and
   the additional error introduced by quantization and physical deployment.
3. **Local NPU inference:** provide actual KL720 execution on an ordinary host
   PC, with explicit CPU preprocessing/postprocessing and no CUDA or cloud
   requirement for the deployed inference workflow.
4. **Reproducible comparison:** use common data, task definitions, grouped
   splits, scoring and declared training/search budgets.
5. **Practical deployment:** produce model-specific reports covering accuracy,
   compatibility, latency, memory and complete local workflow behavior.

The project optimizes model breadth subject to measured accuracy and physical
deployment requirements. A renamed checkpoint or precision setting does not
create a new architecture. There is no arbitrary ceiling on the collection.

## Initial common task and baseline

Use NASA BPS / OSD-366 sample-level 53BP1 burden regression first. Preserve the
existing FITC/DAPI/zero-channel, native-scale 256 x 256 input and bag aggregation
as the initial comparison contract. NASA `avg_nfoci` is a sample/plate-well
reference; individual network outputs are latent continuous burden scores.

The frozen R1 v2 deployment based on `R1CountNetV1` is the first reference
model. Its original seven biological criteria passed on the recorded physical
holdout, with the limitations in the
[post-release audit](NASA_R1_V2_POSTRELEASE_AUDIT.md). That result does not
automatically qualify future architectures or define their accuracy thresholds.
For a fair new comparison, retrain the R1CountNetV1 architecture on the same
grouped folds as the other candidates. The frozen checkpoint is a historical
deployment anchor, not an out-of-fold model for sources it already trained on.

Start expansion with MobileNetV2 and ResNet18, then consider ResNet50 and other
compatible families. Check the complete NASA-adapted graph early. Vendor
examples are feasibility evidence, not NASA accuracy or adapted-model support.
Training and distillation may use GPU resources.

## Accuracy and evaluation principles

Measure two separate quantities for every candidate:

- **Reference accuracy:** prediction error and agreement against NASA labels,
  including source-specific performance and radiation-response endpoints.
- **Deployment fidelity:** paired changes from the same model's FP32 output
  to the physical NPU output, including downstream measurements.

Use continuous-error metrics for continuous endpoints. Define acceptance
requirements before candidate evaluation using the benchmark's scientific
purpose and reference variability. Correlation or classifier-style percentage
accuracy alone cannot establish regression accuracy.

Architecture selection uses grouped development evaluation. Keep related
samples from the same biological source together. Each fold's quantization
calibration uses only that fold's training sources, excluding its evaluation
fold even when labels are not read. The previously examined final holdout is
a historical comparison panel for new work, not newly
untouched selection data. Independent confirmation requires unused sources or
acquisitions; otherwise label results as developmental benchmarking.

## Engineering responsibilities

- **Benchmark/model development:** data manifests, architecture factory,
  common training/scoring, export parity, calibration and compilation.
- **Deployment runtime:** explicit model selection, physical KL720 execution,
  consistent preprocessing/aggregation and structured output.
- **Evidence/reporting:** model and dataset identity, measured accuracy,
  deployment state, execution timing and reproducible result packages.

Keep the application host, scientific runtime, Kneron PLUS runtime and compiler
container isolated. Extend the shared NASA workflow instead of creating a new
standalone runner for every architecture. Preserve the frozen R1 v2 evidence
and evaluators; introduce new benchmark profiles for new models.

## Priority boundaries

| Work | Priority |
| --- | --- |
| NASA common benchmark and additional model architectures | Primary |
| NASA physical batch inference, model profiles and reporting | Primary enabling work |
| Accuracy recovery for promising NASA candidates | Primary within the declared development protocol |
| DNAi and Perilipin2 | Secondary; retain existing work and evidence |
| Broad desktop expansion | Only when it directly supports the NASA model workflow |
| Micronucleus, PNKP and other assay expansion | Deferred unless explicitly reprioritized |

Existing secondary PRs can remain open without becoming prerequisites for
NASA progress. A difficult candidate should receive a documented compatibility
decision so other models can continue through the common pipeline.

## Near-term deliverable

A common NASA benchmark that includes the frozen reference and the first
MobileNetV2/ResNet18 candidates, with explicit per-model states and comparable
accuracy reports. Qualify each model through FP32 evaluation, conversion,
physical execution and the declared accuracy criteria. Expand the collection
once this path is reusable.

## Publication direction

Develop a reproducible public-data benchmark and model collection for accurate
local NPU analysis of NASA radiation microscopy. Report architecture breadth,
absolute accuracy, quantization effects, physical compatibility and local
workflow cost. Manuscript claims follow completed results; no three-assay
success requirement or publication venue is predetermined.
