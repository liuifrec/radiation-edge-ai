# Edge AI for space radiation biology

**Positioning and next evidence milestone — 2026-11-17**

## Project statement

Radiation Edge AI develops accurate local NPU inference for space radiation
biology using public NASA microscopy. Its scientific contribution is an
expanding, reproducible comparison of model architectures: how well they recover
NASA biological reference measurements, how much deployment changes those
measurements, and which models actually execute on physical edge hardware.

The current platform is a ground-based research demonstrator. The longer-term
motivation is biological analysis near data acquisition, including future
systems whose computing and communication resources constrain remote analysis.
The [NASA multi-model plan](NASA_MULTI_MODEL_PLAN.md) remains the operational
plan; this document supplies the space context and evidence-based narrative.

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
baseline already has recorded KL720 validation, and a common workflow now
supports compact CNN, MobileNetV2 and ResNet18 candidates. The next result is
a comparative model collection with measured reference accuracy and paired
FP32-to-physical differences. This ground-based benchmark provides a practical
foundation for investigating autonomous biological analysis under computing
and communication constraints relevant to future space research.

The current data and supervision definitions are recorded in the
[multi-model plan](NASA_MULTI_MODEL_PLAN.md) and the
[R1 v2 input/contrast policy](NASA_BPS_R1_V2_CONTRAST_POLICY.md).
Each nucleus receives a latent continuous burden score; the sample mean is
compared with NASA `avg_nfoci`. The current endpoint is a research measurement,
with the interpretation limits in the [post-release audit](NASA_R1_V2_POSTRELEASE_AUDIT.md).

## Evidence available now

| Evidence | What it establishes | What remains to measure |
| --- | --- | --- |
| Frozen NASA R1 v2 physical validation | One compact model passed its original seven biological criteria within the documented cohort | New architectures require their own evaluation |
| Fresh KL720 replay on 2026-11-17 | User-reported completion of 2,058 nuclei, 22 bags and one model upload; historical numerical repeatability PASS | An independent file audit would require the receipt and result files |
| Shared three-architecture implementation | Factory, grouped training, training-only BatchNorm recalibration, all-nucleus evaluation and ONNX parity export | Actual NASA development results for the new common recipe |
| Early graph screening adapters | Complete untrained candidate export and fresh vendor evidence collection with explicit support status | Actual Kneron results from the toolchain host, followed by checks of the trained graphs |
| Software validation at PR #31 | 250 local tests and four passing CI jobs, including Windows and CPU/ONNX checks | Software tests use synthetic inputs and do not establish new biological accuracy |
| Baseline host-observed device timing | Fresh replay mean 3.930 ms, median 4.001 ms, p95 4.408 ms per nucleus for send/receive | Complete workflow latency, memory and measured energy |

Source records: [historical physical validation](NASA_BPS_R1_V2_KL720_VALIDATION_RECORD.md),
[PR #31 and user-provided replay evidence](https://github.com/liuifrec/radiation-edge-ai/pull/31),
and [the passing PR checks](https://github.com/liuifrec/radiation-edge-ai/actions/runs/43994526339).

The fresh replay's `PASS` uses a maximum absolute difference limit of `1e-4`
against the historical physical predictions. The console does not disclose
the observed maximum error. The 3.930 ms replay timing and the historical
3.676 ms timing belong to different runs and are not a controlled speed
comparison. Neither run measures pure NPU kernel time or the whole microscopy
workflow. The fresh replay confirms repeatability of the established model;
it does not create another architecture or a new blinded biological result.

The current count is **one established NASA physical model** and **three
implemented architecture definitions** for the new comparison. A completed
three-architecture, three-fold campaign will produce nine trained fold models,
representing three architectures.

## Next milestone: the first additional accurate physical model

The immediate aim is a second architecture with recorded NASA reference
performance and real KL720 execution, accompanied by a common comparison
covering the first three architectures and any failures.

| Step | Concrete deliverable | Completion rule |
| --- | --- | --- |
| 1. Declare advancement criteria | Versioned requirements for sample-reference error, source-specific behavior and deployment-induced degradation | Written before reviewing new candidate results; distinguish development qualification from independent confirmation |
| 2. Screen the three complete graphs | One fresh toolchain screening record per NASA-adapted architecture, with ONNX identity and operator/partition evidence | Explicit supported, unsupported or unknown status; unrecognized output stays unknown |
| 3. Run the common NASA benchmark | Three architectures evaluated on the same three source-held-out folds | Full out-of-fold predictions and one comparison table; report training budgets and initialization |
| 4. Convert a promising new model | Trained-graph parity, training-source-only PTQ, BIE and a distinct compiled NEF | Model-specific hashes and fresh conversion evidence; screening alone is insufficient |
| 5. Execute and assess the new model | Physical predictions against both NASA references and the same model's FP32 outputs | Named evaluation stage and declared criteria; complete all folds before claiming pooled physical OOF performance |
| 6. Measure practical local cost | Model load, device exchange and complete input-to-report timing; model size and host memory | Fixed host and workload, with timing scope and failed cases recorded |

A first physical run on one frozen development fold is a useful engineering
milestone. It is labeled partial development evidence until the intended
evaluation is complete. The frozen historical holdout remains a comparison
panel; it does not become unused selection data again.

### Early graph screening implementation

The [candidate exporter](../src/radiation_edge_ai/nasa_bps/candidate_graphs.py)
now exports all three complete randomly initialized models using the NASA
input shape and scalar head. Synthetic PyTorch/ONNX Runtime parity checks
verify export behavior. The [vendor adapter](../src/radiation_edge_ai/nasa_bps/kneron_candidate_screen.py)
collects a fresh whole-graph assessment, records graph/toolchain identities
and distinguishes positive support, explicit rejection and unknown evidence.
CPU neural computation and recognized shape bookkeeping have separate records.
The [Windows launcher and result guide](NASA_LOCAL_NPU_QUICKSTART.md#3-screen-the-complete-candidate-graphs-before-full-training)
make this the next executable host step.

The existing [trained-fold exporter](../src/radiation_edge_ai/nasa_bps/multimodel.py)
still requires a completed, BatchNorm-audited training run. New fold-specific
exports default to 192 calibration inputs from 48 training bags. The historical
NASA vendor scripts remain tied to the original R1 v2 identities and its
288-input panel. Reusable trained-model PTQ, compilation and physical profiles
are subsequent work, and the eventual trained graph is screened again.

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
