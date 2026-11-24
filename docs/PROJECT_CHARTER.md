# Project Charter

**Direction consolidated 2026-11-24.**

The [project status](PROJECT_STATUS.md) is the current progress entry point.
The [evidence ledger](PROGRESS_EVIDENCE_20261124.md) distinguishes inspected
repository content from investigator-reported Windows results. The
[PoC integration plan](POC_INTEGRATION_PLAN.md) defines the next deliverable.
Historical protocols and their acceptance criteria retain their original scope.

## Primary objective

Build an expanding collection of accurate biological-image models that execute
locally on the physical Kneron KL720 NPU. Public NASA microscopy remains the
primary quantitative benchmark. Preserve useful implementations of released
pretrained models, including models whose NASA results do not meet the
task-specific requirements.

The collection supports prospective radiation-biology research, including
potential RERF applications. Current evidence comes from the recorded public
datasets and engineering panels. It does not establish performance on RERF
research material.

The project is a ground-based research demonstrator. Analysis near data
acquisition provides the longer-term space-biology motivation; the
[space narrative](SPACE_RADIATION_BIOLOGY_NARRATIVE.md) records that context.
Mission integration and environmental qualification remain future work.

## Two independent qualification decisions

Every model has both a deployment status and a task-specific accuracy status.

| Decision | Required interpretation |
| --- | --- |
| Deployment fidelity | Identify the exact weights and converted artifacts; distinguish ONNX parity, vendor placement, BIE comparison, NEF integrity and physical execution |
| Reference accuracy | Name the assay, endpoint, dataset, reference type, evaluation stage, metrics and criteria; agreement with a native model is not independent biological truth |

A physically qualified model can remain in the engineering collection while
its NASA accuracy is unsuitable or unresolved. Entry into the NASA
accuracy-qualified subset requires the applicable NASA reference-performance
and deployment criteria. A model's suitability for another assay requires
that assay's own assessment.

This distinction retains the reported Spotiflow implementation without
overriding its unfavorable NASA training-screen result. It also preserves
DNAi v3 as the physical reference while v4 and v5 remain experimental. Detailed
dispositions and evidence provenance belong in the current status and ledger.

Count architecture families, architecture variants, trained checkpoints and
deployment precision variants separately. Additional folds or precision
settings do not create new architecture families. There is no arbitrary ceiling
on useful models in the collection.

## NASA benchmark and scientific controls

The initial task remains NASA BPS / OSD-366 sample-level 53BP1 burden regression.
The existing contract uses native-scale FITC/DAPI crops, a zero third channel,
center padding to 256 x 256 and arithmetic-mean bag aggregation. NASA
`avg_nfoci` is a sample/plate-well reference. Individual regression outputs
remain latent continuous burden scores, not individually supervised counts.

The historical R1 v2 checkpoint is the deployment anchor. It passed its
original seven biological-fidelity criteria within the documented cohort.
The [post-release audit](NASA_R1_V2_POSTRELEASE_AUDIT.md) governs interpretation:
reference-agreement patterns were preserved, but literal FP32 predicted-sign
identity and formal statistical equivalence are not established.

The [three-model campaign](NASA_THREE_MODEL_CAMPAIGN_V1.md) defines the shared
R1CountNetV1/MobileNetV2/ResNet18 development comparison. The
[NASA plan](NASA_MULTI_MODEL_PLAN.md) retains the benchmark and deployment
contracts. Its dated implementation checkpoints are read alongside current
status, rather than used to erase subsequently reported results.

Maintain source-grouped evaluation, training-only calibration and declared
training/search budgets. Keep the previously examined final holdout as a
historical comparison panel for new work. Independent confirmation requires
unused sources or acquisitions; additional nuclei from an already used source
do not establish biological independence.

Use continuous-error metrics, source-specific results and group-aware
uncertainty for continuous endpoints. Define advancement and deployment
requirements before the relevant readout. Preserve failed candidates and
their original criteria. Reused engineering panels remain useful regression
tests without becoming new independent biological validation.

## Shared platform design

Separate four identities: scientific task, trained model, deployment profile
and evaluation record. Share model discovery, artifact verification, device
execution and reporting infrastructure. Keep preprocessing, aggregation,
object reconstruction and biological QC specific to each assay.

Training, adaptation and distillation may use GPUs. Deployed inference should
run locally without CUDA or cloud inference. Record CPU preprocessing and
postprocessing explicitly; CPU neural computation is a separate hybrid
deployment status. Record the actual precision profile, including mixed
fixed-point configurations, rather than describing every deployment as INT8.

Keep the application, scientific environment, Kneron PLUS runtime and compiler
container isolated. Use resident device sessions to amortize model loading
across a batch. Preserve the frozen historical implementations and introduce
versioned profiles for new models and benchmarks.

## Ordered priorities

| Priority | Deliverable | Completion condition |
| --- | --- | --- |
| 1 | Consolidate the reported NASA R1 v2, DNAi v3 and Spotiflow `general` PoC | Recover the actual source, integrate the qualified components, provide one catalogue and replay route, and bind fresh integration checks to the frozen artifacts |
| 2 | Qualify another NASA architecture | Carry an eligible trained candidate through its declared conversion, physical and reference-accuracy assessments; name partial-fold results explicitly |
| 3 | Expand the pretrained-model collection | Select useful endpoints and released weights, assess native behavior on appropriate public data, then measure complete-graph deployment fidelity |
| 4 | Advance selected assay investigations | Run a bounded experiment with a specific hypothesis and stopping decision; retain other tracks and their evidence in the backlog |

An eligible MobileNetV2 fold is a reasonable next NASA deployment candidate.
Its male-1 export remains excluded while its strict parity failure is unresolved.
The failed ResNet18 BIE and corrective experiments remain recorded failures;
their criteria are not relaxed by this charter.

DNAi improvements, Perilipin2 compiler work, gamma-H2AX and micronucleus
development remain useful tracks. Their completion is not a prerequisite for
releasing the existing PoC. General GUI expansion follows the needs of the
shared model workflow.

## Research and release outcome

The intended contribution connects model portability, retained biological
measurements and measured local workflow cost. Report cold start, device
exchange and end-to-end timing with their workload definitions. Add memory
and energy claims only when they have been measured.

Publish source and redistributable metadata with the evidence stage named.
External images, model binaries and upstream materials retain their storage
and licensing requirements. A source release must explain how the corresponding
artifacts are obtained or located. Preserve failures and distinguish reported
results from independently inspected evidence. Manuscript claims follow the
completed evidence package; no venue or multi-assay success requirement is
predetermined.
