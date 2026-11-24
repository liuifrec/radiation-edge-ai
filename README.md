# Radiation Edge AI

**Local NPU inference for radiation-biology microscopy, with public NASA data
as the primary quantitative benchmark.**

We develop and compare models that execute on the physical Kneron KL720.
The initial task uses NASA's [ground-based OSD-366 radiation study](https://osdr.nasa.gov/bio/repo/data/studies/OSD-366)
and [public BPS microscopy benchmark](https://registry.opendata.aws/bps_microscopy/).
The collection also retains useful released models for other microscopy tasks,
with separate records for deployment fidelity and assay-specific accuracy.
Training may use a GPU; the deployed inference workflow is intended to run
locally without CUDA or cloud inference.

## Start here

| Document | Purpose |
| --- | --- |
| [Current project status](docs/PROJECT_STATUS.md) | Authoritative model/workstream status and immediate priorities |
| [Progress evidence through 2026-11-24](docs/PROGRESS_EVIDENCE_20261124.md) | Reported measurements, source identities, failed candidates and evidence limits |
| [PoC integration plan](docs/POC_INTEGRATION_PLAN.md) | Bring the existing Windows implementation into a reproducible shared release |
| [Project charter](docs/PROJECT_CHARTER.md) | NASA priority, reusable model collection and prospective research scope |
| [NASA local NPU quickstart](docs/NASA_LOCAL_NPU_QUICKSTART.md) | Commands implemented here for baseline replay, screening and training/export |

## Current checkpoint — 2026-11-24

The project has a published historical NASA deployment and investigator-reported
physical results for DNAi and Spotiflow. Together these describe three model
pipelines spanning burden regression, fiber segmentation and point detection.
They have different biological qualification levels.

**Source availability matters:** this update consolidates the progress record.
The later Windows source was reported frozen at `fa5047f` under local tag
`poc-kl720-multimodel-2026-11-20`. That commit and tag were unavailable in the
public repository and this checkout at the 2026-11-24 audit. The new Spotiflow
runner, pretrained-model registry and related Windows fixes have therefore
**not been imported by this documentation update**. Raw Windows receipts and
model binaries were not independently reopened or rehashed during consolidation.
The [evidence ledger](docs/PROGRESS_EVIDENCE_20261124.md) distinguishes these
reports from repository-inspected code and historical published records.

| Model / workstream | Deployment evidence | Task-specific accuracy and disposition |
| --- | --- | --- |
| NASA R1 v2 / R1CountNetV1 | Historical physical deployment; 2,058-nucleus fresh replay reported | Passed the original seven NASA biological-fidelity criteria within the documented scope; retain as NASA reference |
| DNAi v3 | Repeated 27-window physical replay reported, with identical reconstructed results | Retain as physical DNA-fiber reference; broader assay validation remains incomplete |
| Spotiflow `general` | All 63 engineering inputs reportedly qualified on physical KL720 using the higher-precision deployment | Retain in the reusable deployment collection; poor native NASA quantification remains, so it is not NASA accuracy-qualified |
| New NASA MobileNetV2 | GPU training and eligible trained-graph screening reported | Additional physical NASA qualification pending; male-1 export remains excluded |
| New NASA ResNet18 | Trained graph supported and female-fold BIE generated | BIE failed two frozen degradation criteria; no qualified NEF/device rollout |
| DNAi v4 / v5 and Spotiflow `synth_complex` | Physical characterization or conversion investigations reported | Experimental; recorded biological or export failures remain open |
| Perilipin2, γH2AX and micronuclei | Compiler, reader and detector investigations reported | Exploratory; no new accurate deployed assay established |

The detailed [status table](docs/PROJECT_STATUS.md) and
[evidence ledger](docs/PROGRESS_EVIDENCE_20261124.md) retain numerical results and
the reasons each candidate advances or remains blocked. Fold checkpoints and
precision variants do not increase the architecture-family count.

## Development priorities

1. **Consolidate the working PoC.** Integrate the actual qualified Windows
   source for NASA R1 v2, DNAi v3 and Spotiflow `general`, with one discoverable
   catalogue, explicit model identities and consistent result records.
2. **Qualify an additional NASA architecture.** Carry an eligible candidate
   through trained ONNX parity, training-only calibration, BIE, NEF and paired
   physical accuracy assessment. A single-fold success is partial development
   evidence until the intended evaluation is complete.
3. **Expand useful model breadth.** Evaluate released models on appropriate
   public references, retain faithful KL720 implementations, and record
   task-specific suitability separately. A model can be hardware-qualified
   while unsuitable for NASA 53BP1 quantification.

The [integration plan](docs/POC_INTEGRATION_PLAN.md) defines completion criteria.
The [NASA multi-model plan](docs/NASA_MULTI_MODEL_PLAN.md) retains the benchmark
design; the [three-model campaign](docs/NASA_THREE_MODEL_CAMPAIGN_V1.md) retains
its declared folds, training recipe and advancement criteria. Existing failures
and frozen acceptance limits are preserved.

## Run the available NASA baseline

On the Windows host with the KL720 attached and the existing frozen assets
available, run from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\run_nasa_kl720.ps1
```

The launcher uses the existing PLUS environment, verifies the frozen assets,
loads the NEF once and performs 2,058 fresh inferences. Each run produces a new
receipt with input hashes, hardware timing and a comparison with the historical
physical predictions when that reference is available. See the
[quickstart](docs/NASA_LOCAL_NPU_QUICKSTART.md) for setup, preflight and result
interpretation. The data and model binaries are external assets; a fresh clone
alone is insufficient to run this replay.

Candidate architecture screening is also implemented:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\screen_nasa_kl720_models.ps1
```

The shared training/export entry point is `radedge-nasa`. The lightweight
application entry point is `radedge`; use `--help` to inspect available commands.
The generic NASA application batch path is currently CPU-only; historical
physical replay uses the dedicated runner above. The later Spotiflow and DNAi
one-command launchers reported from Windows require source import before they
can be advertised as commands available from this repository.

## Accuracy and interpretation

Every model needs two distinct assessments:

- **Reference accuracy:** performance against the relevant biological labels
  or measurements, with the dataset and evaluation stage named.
- **Deployment fidelity:** changes from that model's native or FP32 outputs
  to its converted and physical-device outputs.

Spotiflow's reported physical point F1 of `0.988889` is agreement with native
Spotiflow detections. It is not F1 against independent NASA annotations. Its
reported native NASA bag MAE of `11.724` foci per nucleus remains part of the
record. Keeping the implementation does not promote it to the NASA
accuracy-qualified subset. See the [paired reports](docs/PROGRESS_EVIDENCE_20261124.md).

NASA `avg_nfoci` is an aggregate sample/plate-well label. R1 per-nucleus outputs
are latent burden scores, not individually supervised focus counts. Grouped
development evaluation is used for model selection; the historical final
holdout has already been examined and remains a locked comparison panel for
new work. Reused engineering panels provide regression evidence. Independent
biological confirmation requires unused sources or acquisitions and an
appropriate reference. See the [biological-fidelity framework](docs/BIOLOGICAL_FIDELITY.md).

### Historical NASA R1 v2 result

The recorded final physical evaluation used **2,058 nuclei, 22 sample bags
and 11 matched radiation contrasts**. Physical predictions were generated and
frozen outcome-blind before biological reference outputs were read.

| Measurement | Recorded physical result |
| --- | --- |
| Bag MAE / RMSE | `0.7922 / 0.8893` |
| Bag Pearson / Spearman | `0.9501 / 0.5729` |
| Matched-delta MAE / Pearson / Spearman | `0.9405 / 0.9635 / 0.8182` |
| Direction agreement with NASA reference | `10/11`; `4/4` at four hours |
| Peak-time recovery | `3/3` |
| Original biological-fidelity criteria | Seven of seven passed |

The [post-release audit](docs/NASA_R1_V2_POSTRELEASE_AUDIT.md) governs current
interpretation of the [historical validation record](docs/NASA_BPS_R1_V2_KL720_VALIDATION_RECORD.md).
The per-contrast agreement/failure pattern against the NASA reference was
preserved; **literal FP32 predicted-sign identity was not**. One raw sign
changed on a zero-reference contrast, without changing the seven-gate result.
This is not formal statistical equivalence, a sex-effect analysis, or broad
15-strain generalization.

The historical mean send/receive time was `3.676 ms` per nucleus. The separately
reported 2026-11-17 replay mean was `3.930 ms`. These are different runs, with
device-exchange timing rather than complete microscopy-workflow latency.

## Runtime and reproducibility

The application separates CPU image preparation and measurement reconstruction
from neural inference on the KL720. Precision is deployment-specific: the
historical NASA model uses INT8, while the reported Spotiflow deployment uses
higher-precision fixed-point settings and retains some eight-bit operations.
Neither CPU preprocessing nor a higher-precision setting implies neural CPU
fallback or floating-point execution on the NPU.

The source separates assay adapters, model training, hardware workers and
result verification. Large datasets, model binaries and Kneron build artifacts
remain outside Git. The [integration plan](docs/POC_INTEGRATION_PLAN.md) covers
artifact recovery, the missing shared runtime capabilities and a reproducible
release. A verified Git bundle preserves Git history; artifact hashes alone
do not back up the corresponding weights, BIEs or NEFs.

At the inspected base `561fed0`, [GitHub CI](https://github.com/liuifrec/radiation-edge-ai/actions/runs/44691562702)
passed all five jobs. An additional 2026-11-24 Linux source audit passed 350
tests with 25 Windows-only skips. These checks concern that code revision;
they do not rerun the Windows PoC or establish biological accuracy. Later
Windows test counts remain attributed reports in the evidence ledger.

## Research direction

The project is a ground-based research demonstrator. Public NASA microscopy
supports the space-radiation-biology benchmark; the wider model collection
supports prospective RERF research applications. Current evidence does not
establish RERF assay validation, mission integration or flight qualification.
The [research narrative](docs/SPACE_RADIATION_BIOLOGY_NARRATIVE.md) and
[manuscript outline](manuscript/OUTLINE.md) separate completed evidence from
planned comparisons.

## License

Source code is licensed under BSD-3-Clause (see [LICENSE](LICENSE)). Third-party
datasets, model weights and upstream code retain their original licenses and
terms. A successful local conversion does not by itself authorize redistribution
of those assets.
