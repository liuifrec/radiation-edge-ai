# NASA three-model development campaign v1

Declared on 2026-11-17, after architecture feasibility and before reviewing
new NASA training results. This implements the
[NASA multi-model priority](NASA_MULTI_MODEL_PLAN.md). The purpose is to compare
three architecture families fairly, then carry promising additional models
through quantization, compilation and physical KL720 assessment.

## Entry evidence

The investigator supplied a Windows console transcript from
`nasa_candidate_screen_20261117T042747Z_aa7f370e`, using repository commit
`cebbf364198308e898f3cdeaed9e1b6ea2dd2358`. It records:

- SDK preflight PASS with `/workspace/miniconda/bin/python`, Python 3.7.15,
  and `ktc.onnx_optimizer.onnx2onnx_flow`;
- immutable toolchain image
  `sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041`;
- `COMPLETE_SUPPORTED`, with all three candidates reporting
  `fresh_target720_ip_evaluation_with_no_cpu_neural_compute`.

| Randomly initialized full graph | ONNX SHA256 |
| --- | --- |
| `r1countnet_v1` | `fc54f464aff11e14d61aa1efa7a18c331412e6ac6c070a734133653d01bd0049` |
| `mobilenet_v2` | `daca9f0ff96859c0fbef9b4c466e2c412d402cc808685ff164f2b3a8a0123ad9` |
| `resnet18` | `bed1f1f0584fbc30140afadcd6dc9babb887e2d06bec5b5fd3ee260927db55a4` |

The reported receipt is
`D:\radiation-edge-ai-data\scratch\nasa_candidate_screen_20261117T042747Z_aa7f370e\screen_receipt.json`.
This evidence record is based on the supplied console transcript; the receipt
and raw vendor files have not been independently retrieved in this session.
Preserve that directory for later artifact review.

These are three architecture-feasibility results. They do not establish NASA
accuracy, trained-weight quantization, a deployable NEF, or physical execution
of the new models. Recheck each exact trained graph during conversion.

## Fixed comparison

| Item | Campaign definition |
| --- | --- |
| Architectures | R1CountNetV1, MobileNetV2, ResNet18, each with its implemented scalar burden head |
| Initialization | Random for every architecture/fold; no pretrained or frozen historical checkpoint loaded |
| Dataset | The existing repaired 7,200-nucleus / 72-bag / six-source development cohort |
| Membership | Manifest SHA256 `23e21fdc8bf00a2814b64c56223cfd3d8e96eda71bdbff5d8ec74dae218cf5a9` |
| Folds | `fold_female_role`, `fold_male_1_role`, `fold_male_2_role`; four training sources and two held-out sources per fold |
| Replication | Three architectures x three folds = nine trained fold models; this is not nine architecture families |
| Seeds | 720, 1720, 2720 for the respective folds, shared across architectures |
| Inputs | Native-scale FITC/DAPI, independent p1/p99.5 normalization, zero third channel, center padding to 256 x 256 |
| Supervision | NASA sample/plate-well `avg_nfoci`; arithmetic-mean bag predictions and matched sham/exposed contrast objective |
| Sampling | Deterministic 16-nucleus minibag per training bag per epoch; the same sampling rule across architectures |
| Optimization | AdamW, learning rate `3e-4`, weight decay `1e-4`, cosine schedule, 60 epochs |
| Checkpoint choice | Final scheduled epoch; no held-out early stopping or fold-specific checkpoint selection |
| BatchNorm | Mandatory training-source-only recalibration after training: one nucleus from each of 48 training bags per batch, 100 rank batches; learned parameters unchanged |
| Evaluation | All 100 nuclei in each held-out bag, aggregated once; complete OOF evaluation covers 72 bags and six sources |
| PTQ input selection | Four nuclei per training bag, 192 tensors per fold; no held-out source enters calibration |

The [shared implementation](../src/radiation_edge_ai/nasa_bps/multimodel.py)
records actual software, code, inputs, recipe and checkpoint identities.
Training verifies the frozen metadata and current image geometry/hashes before
running. The image snapshot is newly measured; it does not silently inherit
incomplete historical download receipts.

Retain all attempts, including failures. Do not change epochs, seeds,
architecture settings or sampling after looking at partial accuracy results.
If a resource limitation requires a recipe change, record the reason and
declare one common replacement before comparing results. Do not pool runs
with different budgets as the same campaign.

## Development advancement criteria

For this same six-source cohort, explicitly retain the eight numerical
development criteria from the
[historical contrast policy](NASA_BPS_R1_V2_CONTRAST_POLICY.md#predeclared-development-cv-gate).
Their role here is to decide which models merit deployment assessment. The
old instruction to open a blinded final holdout is not reused: that panel has
already been evaluated.

| Complete OOF metric | Advancement threshold |
| --- | --- |
| Pooled bag Spearman | >= 0.50 |
| Weakest fold bag Spearman | >= 0.25 |
| Pooled bag MAE | <= 0.6654, the historical R1 v1 development benchmark |
| Matched radiation-delta Spearman | >= 0.70 |
| Matched-effect direction agreement | >= 30/36 |
| Direction agreement at 4 h | 12/12 |
| Direction agreement at 24 h and 48 h combined | >= 17/24 |
| Peak-time recovery | >= 10/12 source x branch series |

Assess these only after all three folds are complete, with exactly 72 OOF bags,
36 matched contrasts and 12 complete three-time-point series. Require finite
metrics in every fold; an undefined correlation is unresolved, not a pass.
Do not rescale count thresholds to a smoke run or a partial fold set.

Retain the historical metric definitions for comparability and report predicted
or reference peak ties separately. The historical peak helper resolves a tie
by row order; such a tie alone does not demonstrate recovered temporal biology.
Report the actual errors, per-source and per-fold results, direction by time
point and uncertainty alongside the threshold decisions. Nuclei are not
independent biological replicates.

These thresholds are development advancement heuristics, not a universal
definition of high accuracy or independent biological confirmation. The
current training CLI writes descriptive metrics and labels every result as a
development candidate; it does not automatically apply this eight-part gate.
The complete gate report remains a required subsequent assessment.

## Engineering check before the full campaign

Run one epoch of `r1countnet_v1` on `fold_female_role`, using the same 16-nucleus
training minibags, followed by trained ONNX export. Use a separate output
directory explicitly named as a smoke run. This verifies real-data preflight,
training, BatchNorm recalibration, checkpoint reload, evaluation plumbing and
PyTorch/ONNX Runtime parity. Its output is excluded from campaign rankings and
advancement decisions. It is not used to tune the recipe.

The [Windows quickstart](NASA_LOCAL_NPU_QUICKSTART.md#4-train-the-first-three-architectures-on-identical-source-folds)
uses the installed scientific Python on CPU for this bounded first run and
reports GPU/driver availability without changing packages. GPU training
configuration is chosen from that measured information before starting nine
60-epoch jobs. A CPU smoke does not establish GPU memory sufficiency for the
larger backbones. Training currently has no resume operation and saves its
trained checkpoint only after the last scheduled epoch.

## Advancement after FP32 evaluation

For each architecture that advances, export and recheck its exact trained
graph, quantize using training-source-only inputs, compare BIE predictions,
compile its own NEF and execute that NEF on the physical KL720. Preserve FP32,
simulator and device predictions by identity so endpoint changes can be
measured. Declare acceptable deployment degradation before inspecting those
paired results; no new numerical tolerance is implied by this document.

Use the former 2,058-nucleus panel only as a locked historical comparison, not
for architecture, checkpoint or quantization selection. Independent
confirmation requires genuinely unused biological sources/acquisitions and a
separate frozen evaluation. Until that material and its criteria are defined,
describe the new model results as development benchmarking. Continue to report
the frozen baseline's existing physical evidence separately.
