# Radiation Edge AI

**NASA public-data models for accurate, local NPU inference.**

## Current priority — 2026-11-17

Build an expanding collection of different neural-network architectures that
achieve high accuracy on **public NASA microscopy data** and execute locally
on the **physical Kneron KL720 NPU**. Model breadth, measured accuracy and usable
local inference are the primary development goals.

The first common benchmark is NASA BPS / OSD-366 sample-level 53BP1 burden
estimation. The existing validated R1 v2 model is the reference anchor; new
architectures will share a task, data split, preprocessing and evaluator.
Accuracy against NASA reference measurements and accuracy retained after
quantization are both evaluated. Training may use a GPU; the deployed workflow
should run locally without CUDA or cloud inference.

**Authoritative current plan:** [NASA multi-model plan](docs/NASA_MULTI_MODEL_PLAN.md).
This priority supersedes the original three-assay development order.
DNAi and Perilipin2 retain their existing code and evidence as secondary work.
Their unresolved experiments do not block NASA model expansion. Micronucleus
and other assay expansion are deferred unless explicitly reprioritized.

### First architecture queue

| Model | Role | NASA / physical KL720 status |
| --- | --- | --- |
| Existing R1CountNetV1 / NASA R1 v2 | Frozen compact-CNN reference | Historical physical validation completed within its documented scope |
| MobileNetV2 | First additional backbone | Model factory and shared training/export implemented; NASA accuracy and physical execution pending |
| ResNet18 | First additional residual backbone | Model factory and shared training/export implemented; NASA accuracy and physical execution pending |
| ResNet50 and additional compatible families | Continued breadth and capacity comparisons | Candidate queue, subject to full-graph feasibility and accuracy |

Kneron's [official Model Zoo](https://github.com/kneron/Model_Zoo) lists KL720
examples for the named MobileNet/ResNet models. This motivates the queue; it
does not validate our NASA-adapted graphs, input geometry or task accuracy.
There is no fixed ceiling on the number of models. Count architecture families,
architecture variants and deployment variants separately, and retain failed
candidates with their reasons.

### Run the NASA baseline now

From the repository root on the Windows host with the KL720 attached:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\run_nasa_kl720.ps1
```

The launcher uses the existing PLUS environment, discovers the connected KL720,
loads the frozen NEF once and performs 2,058 fresh NASA inferences. Each run
gets a new results folder and a receipt with input hashes, hardware timing and
a numerical comparison against the historical physical predictions when present.
See the [NASA local NPU quickstart](docs/NASA_LOCAL_NPU_QUICKSTART.md) for exact
paths, preflight options, result interpretation and the new training/export commands.

### Next milestone

Retain the frozen deployment reference and establish one reusable comparison
workflow for R1CountNetV1, MobileNetV2 and ResNet18 trained on the same grouped
folds: FP32 evaluation, ONNX parity,
quantization, compilation, physical inference and a common accuracy/latency
report. The architecture factory, grouped training adapter and ONNX parity
export are implemented for the first three models. NASA training results,
full-graph conversion and physical qualification of the new backbones are the
next evidence needed. NASA KL720 batch integration supports this milestone;
general GUI expansion and unrelated compiler investigations are secondary.

## Accuracy and scientific interpretation

Models are assessed against public reference measurements as well as their
own floating-point outputs. Faithful deployment of an inaccurate model is
insufficient. For the initial regression task, report sample-level MAE/RMSE,
agreement, source-specific errors and radiation-contrast preservation;
correlation is supplementary. Define accuracy requirements and acceptable
deployment degradation before evaluating candidates.

NASA `avg_nfoci` is an aggregate sample/plate-well label. Per-nucleus predictions
remain latent burden scores, not individually supervised focus counts. The
historical final holdout has already been examined and is a locked historical
comparison panel for new work. Architecture selection uses grouped development
evaluation; independent confirmation requires genuinely unused sources or
acquisitions. See the [benchmark plan](docs/NASA_MULTI_MODEL_PLAN.md) and
[biological-fidelity framework](docs/BIOLOGICAL_FIDELITY.md).

Only a model with recorded physical KL720 execution and the applicable
accuracy result belongs in the accuracy-qualified collection. Compiler
acceptance, simulator output and device execution have separate statuses.
CPU preprocessing and postprocessing remain explicit; CPU execution of
neural-network compute operators is reported separately as hybrid inference.

## Deployment concept

```text
microscopy image
       |
CPU preprocessing / tiling
       |
KL720 INT8 inference
       |
CPU post-processing
       |
quantitative biological endpoint
       |
results.csv + annotations + QC
```

The final inference pipeline should run on an ordinary PC without a CUDA installation. GPU resources may be used for training, distillation, or generation of reference outputs.

## Repository layout

```text
radiation-edge-ai/
|-- .github/workflows/ci.yml
|-- docs/
|-- manuscript/OUTLINE.md
|-- scripts/windows/
|-- src/radiation_edge_ai/
|   |-- assay_runtime.py
|   |-- execution.py
|   |-- dna_fiber/
|   |-- nasa_bps/
|   `-- hardware/
`-- tests/
```

Large microscopy datasets, proprietary materials, model binaries, and KL720
build artifacts are not committed to Git. Versioned upstream sources and
frozen checksum evidence are documented where redistribution is permitted.

## Original roadmap (historical; superseded by the NASA multi-model priority)

- **v0.1 — Reference baselines:** reproduce/freeze reference outputs for DNA fiber, gamma-H2AX/53BP1, and micronucleus public datasets.
- **v0.2 — Compact students:** establish KL720-compatible lightweight FP32 models.
- **v0.3 — Edge deployment:** ONNX/quantization/NEF conversion and real KL720 inference for all three tracks.
- **v0.4 — Biological fidelity:** endpoint-level agreement, equivalence, and compression-fidelity analyses.
- **v0.5 — Reproducible manuscript:** regenerate publication figures/tables from versioned benchmark outputs.

## Validated NASA BPS 53BP1 milestone

The NASA BPS OSD-366 R1 v2 track has completed end-to-end deployment validation from the frozen FP32 model through ONNX, development-only INT8 PTQ, BIE, NEF, and the physical KL720.

The final source-held-out physical validation used **2,058 nuclei -> 22 sample/plate-well bags -> 11 matched radiation contrasts**. Physical predictions were generated and frozen outcome-blind before biological reference outputs were read.

Observed physical KL720 result:

- bag MAE / RMSE / Pearson / Spearman: `0.7922 / 0.8893 / 0.9501 / 0.5729`;
- matched-delta MAE / Pearson / Spearman: `0.9405 / 0.9635 / 0.8182`;
- direction agreement: `10/11` overall, `4/4` at 4 h, `6/7` at 24+48 h;
- source direction: BALBCF2 `4/5`, C57BLF2 `3/3`, C57BLF3 `3/3`;
- peak-time recovery: `3/3`;
- **all seven original predeclared biological-fidelity acceptance criteria passed**;
- the physical deployment preserved the per-contrast agreement/failure pattern against the NASA aggregate reference and recovered all three reference peak times.

Mean physical KL720 send+receive time over the 2,058-nucleus run was `3.676 ms` per nucleus (median `3.636 ms`, p95 `4.269 ms`), reported as descriptive device timing rather than a complete system/energy benchmark.

The scientific claim is **passing the original predeclared biological-fidelity acceptance criteria**, not strict numerical or formal statistical equivalence. Per-nucleus output remains a latent continuous 53BP1 burden score rather than an individually supervised focus count. The held-out sources are all female, so this is not a sex-effect analysis, and it must not be presented as broad 15-strain generalization.

Full provenance, hashes, protocol-deviation documentation, claim boundaries, and the terminal freeze are recorded in `docs/NASA_BPS_R1_V2_KL720_VALIDATION_RECORD.md`.

## Publication strategy

The current manuscript direction is a **NASA public-data benchmark and model
collection for accurate local NPU inference**. Compare multiple architectures
on the same task, then report reference accuracy, deployment degradation,
physical compatibility and complete local workflow performance.

Biological endpoint fidelity is part of accuracy assessment. A three-assay
success narrative is no longer a prerequisite. The
[manuscript outline](manuscript/OUTLINE.md) follows this scope; venue selection
depends on the completed benchmark and validation evidence.

## Current implementation and historical release evidence

The reference NASA model has completed physical validation. A fresh baseline
replay launcher and shared three-architecture training/ONNX export workflow
are implemented. New architecture accuracy, conversion and hardware evidence,
the model-selection interface and the unified NASA KL720 batch adapter remain
pending. Main also includes later DNAi audit utilities. The v0.11 milestone
below is a historical release record; its test count is not a current-suite count.

- **NASA BPS 53BP1**: frozen physical KL720 holdout passed the original
  predeclared biological-fidelity gates (within the documented scope).
- **DNA-fiber**: physical nine-window KL720 field inference, stitching, fiber
  reconstruction, backend-aware `df2`/`dt2` provenance, and relocatable `drp1`
  reporting have been verified. On one frozen field, foreground Dice versus
  floating FP512 was `0.8653`; valid-fiber measurement correspondence was
  limited. **This is not DNAi biological-fidelity validation.**
- **Software**: public schema-aware verification, export integrity and tamper
  detection, and GitHub Actions CI for Python 3.9 and 3.11 (131 tests per
  Python version) have passed.
- **Future scope**: micronucleus and broader independent DNA-fiber validation
  are separate follow-up milestones.

See [v0.11 evidence and claim boundaries](docs/UNIFIED_KL720_V0_11_RELEASE.md).

## License

Source code is licensed under BSD-3-Clause (see `LICENSE`). Third-party datasets, model weights, and upstream code remain subject to their original licenses and terms.
