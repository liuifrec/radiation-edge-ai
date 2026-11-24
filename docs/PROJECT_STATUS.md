# Project status

**Consolidated 2026-11-24 (Japan time).** This is the current project status
index. The [evidence ledger](PROGRESS_EVIDENCE_20261124.md) records the source,
measurement scope and limitations behind each entry. Historical protocols and
failed results retain their original identities and decisions.

## Direction and evidence boundary

The primary scientific objective is multiple accurate models for **public NASA
microscopy**, executing locally on the **physical KL720**. A complementary
engineering collection retains useful released models that can run faithfully
on the device, even when they are not accurate for the NASA task. These
capabilities support a ground-based research demonstrator and prospective RERF
applications; they do not establish RERF assay validation or flight readiness.

Two evidence sources are deliberately separate:

- **Repository inspection:** source and CI at base commit
  `561fed0b3572d19392f2471e0b274840ba468fcd` were inspected on 2026-11-24.
- **Investigator reports:** the later Windows training, conversion, device and
  freeze results were supplied by the investigator. Their raw local artifacts
  and reported integration commit `fa5047f` were not available in this checkout
  or on GitHub when this consolidation was prepared. They were not rehashed or
  rerun here.

This documentation consolidates those reports. It does not import the missing
Windows implementation or turn reported measurements into an independent audit.

## Model and assay status

Deployment fidelity and biological accuracy are independent qualifications.
Agreement with a native model identifies retained behavior; accuracy against
manual annotations or NASA references identifies task performance.

| Model / track | Deployment evidence | Task-specific accuracy and disposition | Evidence |
| --- | --- | --- | --- |
| NASA R1 v2 / R1CountNetV1 | Historical physical validation is recorded in the repository; a fresh 2,058-nucleus replay was reported | Original seven biological-fidelity criteria passed within the documented cohort; retain as the NASA reference | [E01](PROGRESS_EVIDENCE_20261124.md#e01-nasa-r1-v2) |
| DNAi v3 | Repeated 27-window physical replays and downstream reproduction reported | Retain as the physical DNA-fiber reference; three-field characterization is not broad human-ground-truth validation | [E04](PROGRESS_EVIDENCE_20261124.md#e04-dnai-reference-and-candidates) |
| Spotiflow `general` | All 63 engineering inputs reportedly qualified on physical KL720 using the higher-precision deployment | Keep in the hardware-qualified collection; native NASA bag MAE remained 11.724 foci/nucleus, so NASA accuracy is not qualified | [E05](PROGRESS_EVIDENCE_20261124.md#e05-pretrained-native-screen), [E06](PROGRESS_EVIDENCE_20261124.md#e06-spotiflow-conversion-and-physical-qualification) |
| New NASA MobileNetV2 | GPU training completed; eligible trained exports and vendor graph support reported; male-1 export excluded | Promising development performance; no completed new physical NASA qualification | [E02](PROGRESS_EVIDENCE_20261124.md#e02-three-model-nasa-campaign) |
| New NASA ResNet18 | Trained graphs supported; female-fold BIE generated and evaluated | Two of eight frozen deployment criteria failed; no qualified NEF or physical rollout; bias adjustment did not advance | [E03](PROGRESS_EVIDENCE_20261124.md#e03-resnet18-ptq-and-training-diagnostics) |
| DNAi v4 / v5 | V4 physical characterization reported; v5 conversion and diagnostic results retained | Grader-supported losses remain; both experimental, with v3 retained | [E04](PROGRESS_EVIDENCE_20261124.md#e04-dnai-reference-and-candidates) |
| Spotiflow `synth_complex` | Six of 63 original ONNX cases exceeded the frozen raw-output tolerance | Conversion blocked; preserved spot counts do not override the failed raw-output gate | [E06](PROGRESS_EVIDENCE_20261124.md#e06-spotiflow-conversion-and-physical-qualification) |
| Perilipin2 | Full-model compiler DMA failure persists after isolated probes and a split projection | Experimental; no complete KL720 deployment | [E07](PROGRESS_EVIDENCE_20261124.md#e07-other-microscopy-workstreams) |
| γH2AX / micronuclei | Annotated-data readers and diagnostic measurements reported | Current detector results do not establish useful assay accuracy | [E07](PROGRESS_EVIDENCE_20261124.md#e07-other-microscopy-workstreams) |

The three named reference pipelines are NASA burden regression, DNAi fiber
segmentation and Spotiflow point detection. Experimental physical variants such
as DNAi v4 remain visible separately. Folds, seeds and precision variants do not
increase the architecture-family count. Only the historical NASA R1 v2 has
completed the NASA-specific physical and original biological-gate path.

## What is runnable from this repository

The published base provides the NASA R1 v2 replay, a shared NASA three-model
training/export workflow, early graph screening, and existing DNAi execution
and audit tools. Start with the [NASA quickstart](NASA_LOCAL_NPU_QUICKSTART.md)
and the [historical unified release guide](UNIFIED_KL720_V0_11_RELEASE.md).
Both require the documented external artifacts and runtime environments.

The newer commands `run_spotiflow_kl720.ps1 -Smoke` and
`run_dnai_hardware_baseline.py` were reported from the local Windows branch.
Those launchers and the newer pretrained-model registry are **not supplied by
this documentation consolidation**. Their names identify integration targets,
not commands a new clone can currently execute.

The inspected base still has these implementation gaps:

- The unified NASA batch adapter supports `cpu-onnx`; historical physical NASA
  inference uses a dedicated runner that loads its model once.
- The generic KL720 worker packs INT8 inputs and requires a single output.
  The reported Spotiflow path needs the qualified higher-precision packing and
  two-output decoding implementation.
- The trained exporter still serializes calibration paths with native path
  separators. The reported Windows-to-Linux portability fix remains to import.
- Architecture construction, assay registration and qualified deployment
  identities are not yet joined into the reported local model catalogue.

The [integration plan](POC_INTEGRATION_PLAN.md) describes how to bring in the
actual local source and evidence without reconstructing it from console text.

## Software and freeze status

At the inspected base, [GitHub CI](https://github.com/liuifrec/radiation-edge-ai/actions/runs/44691562702)
passed all five jobs, including native Windows launcher checks. A fresh Linux
audit ran **350 tests successfully**, with **25 Windows-specific tests skipped**.
These are software checks, not vendor or physical biological validation.
The investigator separately reported 579 passes and 19 skips in the NASA
environment, 77 offline Spotiflow passes and a passed opt-in USB test for the local integration;
overlapping suites must not be added together.

The investigator froze commit `fa5047fd20c104de98f62ad39641a17fee1489dc` under
local tag `poc-kl720-multimodel-2026-11-20`. Git reported its bundle valid with
complete history. The snapshot copied 12 evidence files and hashed 12 Spotiflow
binary files; a hash inventory does not itself back up the binaries. Independent
artifact backup and a restoration check remain unconfirmed here. See
[E08](PROGRESS_EVIDENCE_20261124.md#e08-local-freeze-and-source-integration).

## Ordered next milestones

1. **Integrate and release the working collection.** Import the attributable
   Windows source, reconcile it with current main, expose the three reference
   pipelines through a consistent catalogue, and run their existing regression
   panels. Keep task-specific result payloads and separate accuracy statuses.
2. **Complete an additional NASA architecture.** Choose one eligible checkpoint
   and a bounded conversion experiment before reading its paired results.
   MobileNetV2 is a reasonable candidate from the reported development results;
   male-1 remains excluded. A single-fold success is partial development evidence.
3. **Expand the pretrained queue deliberately.** Assess endpoint relevance,
   native accuracy, complete-graph compatibility and conversion cost. Hardware
   qualification can justify retaining an implementation without NASA promotion.
4. **Keep secondary investigations bounded.** ResNet18 corrections, DNAi v4/v5,
   Perilipin2 and the newer microscopy detectors retain their failures and a
   specific next decision; they do not all block release of working models.

## Interpretation rules

- NASA `avg_nfoci` is a sample/well mean; per-nucleus regression outputs remain
  latent burden scores, not individually supervised focus counts.
- The historical holdout is already examined. New model selection uses grouped
  development data; broader confirmation needs genuinely unused sources or
  acquisitions. Nuclei are not independent biological replicates.
- Reused engineering panels remain useful regression panels. Repeatedly
  comparing candidates on them does not create independent assay validation.
- Spotiflow's physical point F1 of 0.988889 is agreement with native Spotiflow.
  DNAi correspondence to FP512 is not a substitute for human-reference accuracy.
- The [NASA post-release audit](NASA_R1_V2_POSTRELEASE_AUDIT.md) governs the
  historical claim: the biological-reference agreement/failure pattern was
  preserved, but literal FP32 predicted-sign identity was not.
- CPU preprocessing and postprocessing are explicit. Report neural CPU
  partitions separately, and distinguish device exchange from complete local
  workflow timing. Higher-precision fixed-point inference is not FP32 inference.
