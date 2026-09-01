# Radiation Edge AI

**Biologically faithful INT8 edge AI for quantitative radiation microscopy.**

`radiation-edge-ai` develops and evaluates compact neural-network inference pipelines for the Kneron KL720 NPU, with the central question:

> Can aggressively compressed and INT8-quantized edge models preserve biologically meaningful radiation-response measurements well enough that the resulting scientific conclusions remain equivalent to conventional reference analysis?

The project is intentionally evaluated at three levels:

1. **Model level** — FP32 teacher -> compact FP32 student -> INT8 edge model.
2. **Image level** — segmentation/detection accuracy and error structure.
3. **Biological level** — preservation of the quantitative endpoint and scientific conclusion.

A modest loss in Dice, IoU, F1, or detection sensitivity may be acceptable if the relevant biological measurement remains equivalent. Conversely, a visually accurate model is unacceptable if compression systematically biases dose-response, repair kinetics, micronucleus frequency, or DNA-replication phenotypes.

## Tier I: Radiation Edge Phenotyping

The initial proof-of-concept portfolio contains three complementary assays:

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
|-- docs/
|   |-- PROJECT_CHARTER.md
|   |-- BIOLOGICAL_FIDELITY.md
|   `-- DATA_SOURCES.md
|-- src/radiation_edge_ai/
|   |-- core/
|   |-- gamma_foci/
|   |-- micronucleus/
|   `-- dna_fiber/
|-- scripts/
|-- benchmarks/
|-- tests/
`-- manuscript/
    |-- OUTLINE.md
    `-- FIGURE_PLAN.md
```

Large microscopy datasets, proprietary material, model binaries, and KL720 build artifacts are **not** committed to Git. Public datasets should be obtained through versioned manifests/download instructions with source provenance and checksums where practical.

## Milestones

- **v0.1 — Reference baselines:** reproduce/freeze reference outputs for DNA fiber, gamma-H2AX/53BP1, and micronucleus public datasets.
- **v0.2 — Compact students:** establish KL720-compatible lightweight FP32 models.
- **v0.3 — Edge deployment:** ONNX/quantization/NEF conversion and real KL720 inference for all three tracks.
- **v0.4 — Biological fidelity:** endpoint-level agreement, equivalence, and compression-fidelity analyses.
- **v0.5 — Reproducible manuscript:** regenerate publication figures/tables from versioned benchmark outputs.

## Publication strategy

The project is designed first as a **general biological-methods contribution**, with biological fidelity under model compression as the central methodological advance rather than a hardware demonstration.

- **Primary target:** Cell Reports Methods
- **Fallback target:** Radiation Research

## Status

Early research scaffold. The KL720 platform itself has already been validated separately for generic image inference; assay-specific reference baselines and KL720 models are now being developed in this repository.

## License

A software license will be pinned before the first distributable release. Third-party datasets, model weights, and upstream code remain subject to their original licenses and terms.
