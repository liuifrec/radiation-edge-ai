# Edge AI for space radiation biology

**Positioning and progress consolidation — 2026-11-24**

## Project statement

Radiation Edge AI develops accurate local NPU inference for space radiation
biology using public NASA microscopy. Its scientific contribution is an
expanding, reproducible comparison of model architectures: how well they recover
NASA biological reference measurements, how much deployment changes those
measurements, and which models actually execute on physical edge hardware.

The current platform is a ground-based research demonstrator. The longer-term
motivation is biological analysis near data acquisition, including future
systems whose computing and communication resources constrain remote analysis.
The [current status](PROJECT_STATUS.md) and [integration plan](POC_INTEGRATION_PLAN.md)
define immediate work. The [NASA multi-model plan](NASA_MULTI_MODEL_PLAN.md)
retains the common benchmark design. This document supplies the space context
and the distinction between reported progress and reproducible source.

**Working scientific title:**

**Edge AI for space radiation biology: benchmarking local NPU inference on
public NASA microscopy**

**Central question:** Which locally deployable architectures accurately recover
sample-level radiation-response measurements, and what additional error and
execution cost does NPU deployment introduce?

## Why this is a space-biology problem

The biological connection is direct. NASA's [OSD-366 study](https://osdr.nasa.gov/bio/repo/data/studies/OSD-366)
describes ex vivo mouse fibroblast experiments with simulated deep-space
radiation and 53BP1-positive foci as a surrogate DNA-damage measurement.
These are ground experiments. The selected
[NASA BPS microscopy benchmark](https://registry.opendata.aws/bps_microscopy/)
provides nucleus images associated with Fe-particle or X-ray exposure.
The project's selected sources and conditions are narrower than the complete
parent study.

The computing motivation is also specific. NASA's
[HPSC description](https://www.nasa.gov/game-changing-development-projects/high-performance-spaceflight-computing-hpsc/)
links communications delay to onboard autonomy. JPL's
[OWLS science-autonomy work](https://www.jpl.nasa.gov/go/owls/onboard-science-autonomy/)
shows how instrument data, including microscopy, can be summarized and
prioritized under downlink constraints. NASA's
[Spectrum instrument](https://science.nasa.gov/mission/spectrum/) establishes
automated fluorescence imaging as part of biological research on the ISS.

Our inference from that context is that accurate local interpretation of
biological images is a useful capability to investigate. These examples
motivate the research question; they do not establish integration between
this project and any NASA instrument or mission. All linked context sources
were checked on 2026-11-17.

## Reusable project description

Radiation Edge AI investigates how diverse neural-network architectures can
preserve radiation-biology measurements when inference runs on local edge
hardware. Using public NASA microscopy and aggregate sample references, the
project connects source-grouped model evaluation with reproducible conversion,
physical NPU execution and traceable biological outputs. A frozen compact-CNN
baseline already has recorded KL720 validation, and the common NASA workflow
supports compact CNN, MobileNetV2 and ResNet18 candidates. Later Windows reports
describe physical DNA-fiber and released Spotiflow deployments. Their engineering
qualification is recorded separately from assay accuracy; Spotiflow's native
NASA result remains poor. The next software milestone is a reproducible release
of the working pipelines, followed by another physically qualified NASA
architecture. This ground-based platform supports prospective RERF research
applications and investigation of local analysis under computing and
communication constraints. Neither institutional assay validation nor mission
integration follows from the current demonstration.

The current data and supervision definitions are recorded in the
[multi-model plan](NASA_MULTI_MODEL_PLAN.md) and the
[R1 v2 input/contrast policy](NASA_BPS_R1_V2_CONTRAST_POLICY.md).
Each nucleus receives a latent continuous burden score; the sample mean is
compared with NASA `avg_nfoci`. The current endpoint is a research measurement,
with the interpretation limits in the [post-release audit](NASA_R1_V2_POSTRELEASE_AUDIT.md).

## Evidence at the consolidation checkpoint

The [evidence ledger](PROGRESS_EVIDENCE_20261124.md) attributes the newer numbers
to investigator-supplied Windows reports. Their raw receipts and the frozen
`fa5047f` source were not retrieved in this consolidation. The documented result
and the availability of its implementation are separate facts.

| Evidence | What it establishes within its stated source | What remains |
| --- | --- | --- |
| Frozen NASA R1 v2 physical validation | One compact model passed its original seven biological criteria within the documented cohort | New architectures require their own evaluation |
| Fresh KL720 replay on 2026-11-17 | User-reported completion of 2,058 nuclei, 22 bags and one model upload; historical numerical repeatability PASS | An independent file audit would require the receipt and result files |
| NASA GPU campaign | Reported completion of three architectures and nine fold models, with shared development metrics | Additional NASA architecture's full physical accuracy qualification; ResNet18 female BIE failed two criteria |
| DNAi v3 | Reported 27-window physical replay and exact reconstructed-field repeatability | Broader biological validation; v4/v5 remain experimental |
| Spotiflow `general` | Reported 63-case physical qualification, native-agreement F1 0.988889 and median input-preparation-to-count time 67.18 ms | Source integration and assay-specific accuracy; the NASA counting result is not qualified |
| Published software base `561fed0` | Five successful CI jobs; separate Linux audit with 350 passes and 25 Windows-only skips | New Windows source is not covered by that base revision's CI |

Source records: [current interpretation of historical validation](NASA_R1_V2_POSTRELEASE_AUDIT.md),
[PR #31 and user-provided replay evidence](https://github.com/liuifrec/radiation-edge-ai/pull/31),
[current base CI](https://github.com/liuifrec/radiation-edge-ai/actions/runs/44691562702),
and the [consolidated reports](PROGRESS_EVIDENCE_20261124.md).

The fresh replay's `PASS` uses a maximum absolute difference limit of `1e-4`
against the historical physical predictions. The console does not disclose
the observed maximum error. The 3.930 ms replay timing and the historical
3.676 ms timing belong to different runs and are not a controlled speed
comparison. Neither run measures pure NPU kernel time or the whole microscopy
workflow. The fresh replay confirms repeatability of the established model;
it does not create another architecture or a new blinded biological result.

The current record contains **one established NASA physical model**, a reported
three-architecture/nine-fold NASA development campaign, and reported physical
DNAi and Spotiflow pipelines. Those three physical pipelines serve different
tasks; they are not three NASA accuracy-qualified models. Precision settings,
folds and repeated replays do not add architecture families.

## Next milestones: integrate the PoC and qualify another NASA model

The immediate software aim is to recover the existing qualified Windows source
and deliver the [shared PoC release](POC_INTEGRATION_PLAN.md). The NASA research
aim remains a second architecture with recorded reference performance and real
KL720 execution, accompanied by the common comparison and its failures.

| Step | Concrete deliverable | Completion rule |
| --- | --- | --- |
| 1. Integrate qualified source | Reviewed import of the frozen runtime, model registry and known conversion fixes | Actual source available, existing artifact identities preserved, scoped software and physical checks |
| 2. Reconcile the completed NASA campaign | Import the existing grouped comparison and exact trained-graph/export records | Preserve failures and attribution; do not repeat training just to fill a documentation gap |
| 3. Declare the next conversion attempt | One eligible checkpoint/fold, training-only calibration, applicable degradation requirements and a bounded experiment budget | Written before the new readout; preserve failed BIEs and existing margins |
| 4. Convert a promising new model | Trained-graph parity, training-source-only PTQ, BIE and a distinct compiled NEF | Model-specific hashes and fresh conversion evidence; screening alone is insufficient |
| 5. Execute and assess the new model | Physical predictions against both NASA references and the same model's FP32 outputs | Named evaluation stage and declared criteria; complete all folds before claiming pooled physical OOF performance |
| 6. Measure practical local cost | Model load, device exchange and complete input-to-report timing; model size and host memory | Fixed host and workload, with timing scope and failed cases recorded |

A first physical run on one frozen development fold is a useful engineering
milestone. It is labeled partial development evidence until the intended
evaluation is complete. The frozen historical holdout remains a comparison
panel; it does not become unused selection data again.

### Early graph screening implementation

The following describes implemented screening capabilities, not an instruction
to repeat the reported completed campaign. Consult [current status](PROJECT_STATUS.md)
before choosing a new run.

The [candidate exporter](../src/radiation_edge_ai/nasa_bps/candidate_graphs.py)
now exports all three complete randomly initialized models using the NASA
input shape and scalar head. Synthetic PyTorch/ONNX Runtime parity checks
verify export behavior. The [vendor adapter](../src/radiation_edge_ai/nasa_bps/kneron_candidate_screen.py)
collects a fresh whole-graph assessment, records graph/toolchain identities
and distinguishes positive support, explicit rejection and unknown evidence.
CPU neural computation and recognized shape bookkeeping have separate records.
The [Windows launcher and result guide](NASA_LOCAL_NPU_QUICKSTART.md#3-screen-the-complete-candidate-graphs-before-full-training)
document how to reproduce this stage or screen a new candidate.

The existing [trained-fold exporter](../src/radiation_edge_ai/nasa_bps/multimodel.py)
still requires a completed, BatchNorm-audited training run. New fold-specific
exports default to 192 calibration inputs from 48 training bags. The historical
NASA vendor scripts remain tied to the original R1 v2 identities and its
288-input panel. Later Windows reports describe trained-model PTQ work; the
reusable path and its fixes await source integration here. Every exact trained
graph still requires its own screening and deployment evidence.

The new screen requires positive, fresh evidence. It does not inherit the
[historical optimizer/evaluator](../src/radiation_edge_ai/nasa_bps/kneron_optimize_evaluate_r1_v2.py)
heuristic that inferred support from the absence of failure strings or could
select an older log when no fresh artifact existed. Unknown evidence never
advances a model. Software tests of this behavior are separate from actual
Kneron acceptance and physical execution.

This step directly supports architecture breadth. It does not require a new
assay, a GUI, a different accelerator, or spacecraft integration work.

## Manuscript and demonstration structure

| Planned figure | Scientific point | Required evidence |
| --- | --- | --- |
| 1. Biological task and local workflow | Explain the radiation experiment, input channels, source groups and sample-level endpoint | Public-source provenance and the actual preprocessing/aggregation contract |
| 2. Accuracy across architectures | Identify which models recover the NASA reference measurements | Matched development folds, sample errors, source-specific results and radiation contrasts |
| 3. Deployment and retained accuracy | Show which complete graphs deploy and how physical inference changes the biological result | Model-stage matrix including failures; paired FP32/BIE/physical results |
| 4. Practical local computation | Show the accuracy/cost choices available to an instrument-side application | Measured workflow timing, model size and memory; add energy only when measured |

The [manuscript outline](../manuscript/OUTLINE.md) uses this order: biological
problem, comparative accuracy, physical deployment, then local resource cost.
The device is part of the experiment rather than the sole scientific outcome.

After additional models are qualified, a focused space-oriented extension
could evaluate image retention and endpoint summaries under a declared
simulated communications budget. Its result must quantify retained scientific
information as well as bytes, and preserve access to raw data during validation.
This is a future experiment, not a present downlink-saving claim.

## Scope of the current space claim

The present claim is **local inference for space-radiation-biology research,
demonstrated on the ground with public NASA data and physical NPU hardware**.
The project has no demonstrated mission integration, flight qualification or
NASA endorsement. NASA's HPSC discussion identifies fault tolerance,
radiation effects and environmental reliability as separate engineering
requirements; these are outside the current KL720 validation.

The primary work remains broadening the collection of accurately evaluated
NASA models. Additional datasets and capacity variants follow that reusable
path, with the evaluation task and evidence stage named for every model.
