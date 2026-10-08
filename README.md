# Radiation Edge AI

**Provenance-aware offline edge AI for quantitative radiation microscopy.** Biological fidelity must be evaluated per assay rather than inferred from deployment or pixel-level accuracy.

`radiation-edge-ai` develops and evaluates compact neural-network inference pipelines for the Kneron KL720 NPU, with the central question:

> Can aggressively compressed and INT8-quantized edge models preserve biologically meaningful radiation-response measurements well enough that the resulting scientific conclusions remain equivalent to conventional reference analysis?

The project is intentionally evaluated at three levels:

1. **Model level** — FP32 teacher -> compact FP32 student -> INT8 edge model.
2. **Image level** — segmentation/detection accuracy and error structure.
3. **Biological level** — preservation of the quantitative endpoint and scientific conclusion.

A modest loss in Dice, IoU, F1, or detection sensitivity may be acceptable if the relevant biological measurement remains equivalent. Conversely, a visually accurate model is unacceptable if compression systematically biases dose-response, repair kinetics, micronucleus frequency, or DNA-replication phenotypes.

## Tier I: Radiation Edge Phenotyping

The long-term portfolio has three complementary assay targets; the current offline demonstrator implements NASA 53BP1 and DNA-fiber:

| Track | Biological axis | Edge-AI task | Primary biological outputs |
| --- | --- | --- | --- |
| **gamma-H2AX / 53BP1** | DNA damage and repair | nuclei/foci detection or segmentation | foci per cell, dose-response, repair kinetics |
| **Micronucleus** | chromosomal damage | nuclei/cell/micronucleus detection or segmentation | micronucleus frequency, cell-level scoring, later biodosimetry |
| **DNA fiber** | replication stress | fiber segmentation | tract length, fork-related ratios, condition/genotype effect sizes |

The shared comparison is:

```text
manual/public reference
        vs
original FP32 teacher
        vs
compact FP32 student
        vs
INT8 KL720 deployment
```

## Core principle: biological fidelity

The primary question is not whether a model survives quantization pixel-for-pixel. It is whether a biologist would reach the **same quantitative conclusion** using the edge deployment.

Planned analyses include conventional computer-vision metrics together with endpoint-specific agreement and equivalence statistics, for example:

- Dice / IoU / precision / recall / F1;
- Bland-Altman agreement;
- ICC or other reproducibility measures where appropriate;
- effect-size preservation;
- equivalence testing with pre-specified margins;
- dose-response parameter preservation;
- latency, model size, RAM, throughput, and hardware requirements;
- energy per image where measurement is practical.

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

## Original roadmap (historical; not the v0.11 release status)

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

The project is designed first as a **general biological-methods contribution**, with biological fidelity under model compression as the central methodological advance rather than a hardware demonstration.

Publication venue will be chosen after the final validation scope is frozen. A public-data/software-only package is being developed as a reproducible methods/software contribution; sufficiently strong independent radiation-biology validation could support reassessment for a field-specific methods venue.

## Status: verified v0.11 offline two-assay demonstrator

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
