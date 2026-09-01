# Manuscript Outline

Working target: **Cell Reports Methods**  
Fallback: **Radiation Research**

## Working title

**Biological-fidelity-preserving edge AI for quantitative radiation microscopy**

Alternative title:

**Preserving quantitative biological phenotypes under INT8 edge inference across radiation-relevant microscopy assays**

## Central claim

Aggressive model compression and INT8 deployment can be evaluated more meaningfully by preservation of downstream biological measurements than by image-level accuracy alone. A shared validation framework can identify assay-specific compression limits across distinct quantitative microscopy tasks.

## Abstract structure

1. Problem: modern microscopy AI often assumes GPU-class inference.
2. Gap: quantization studies emphasize vision metrics rather than biological endpoint preservation.
3. Method: compare R0/R1/R2/R3 across three complementary assays.
4. Results: quantify image-level degradation, biological endpoint fidelity, and edge deployment performance.
5. Impact: establish a reproducible framework for biologically trustworthy edge microscopy.

## Introduction

- AI has improved quantitative microscopy but deployment remains hardware- and software-heavy.
- Edge NPUs offer inexpensive local inference, but INT8 compression can alter measurements in subtle ways.
- Conventional metrics do not establish whether biological conclusions are preserved.
- Radiation-relevant assays provide diverse, quantitative test cases.
- Study objective: define and test a biological-fidelity framework across gamma-H2AX/53BP1, micronucleus, and DNA-fiber analysis.

## Results

### 1. A common edge-AI framework for quantitative microscopy

- reference hierarchy R0-R3;
- common preprocessing/inference/post-processing architecture;
- KL720 deployment path;
- shared benchmark design.

### 2. DNA-fiber edge segmentation preserves replication measurements

- DNAi reference baseline;
- compact student;
- FP32 vs INT8 vs KL720;
- tract-length/ratio agreement;
- failure analysis for difficult fibers.

### 3. Edge gamma-H2AX/53BP1 inference preserves radiation-response measurements

- public reference baseline;
- nuclei/foci detection/segmentation;
- foci-per-cell agreement;
- dose-response and/or repair-related endpoint preservation.

### 4. Distilled micronucleus inference preserves chromosomal-damage scoring

- high-capacity teacher/reference;
- compact student;
- nucleus/MN segmentation or detection;
- MN frequency and cell-level agreement.

### 5. Image accuracy and biological fidelity diverge under progressive compression

- cross-assay compression frontier;
- model size/compute vs Dice/F1 vs endpoint error;
- assay-specific biological operating points.

### 6. Real KL720 deployment enables GPU-free quantitative inference

- latency/throughput;
- model size/RAM;
- CPU-only comparison;
- reproducibility and deployment constraints;
- energy if measured robustly.

## Discussion

- Biological fidelity is distinct from computer-vision fidelity.
- Acceptable compression is assay-dependent.
- Edge inference can reduce deployment barriers without assuming that INT8 is harmless.
- Teacher architectures need not be hardware-compatible if compact students preserve endpoints.
- Limitations: public datasets, domain shift, assay-specific margins, hardware-specific compiler constraints.
- Future validation with institution-specific/historical radiation-biology datasets.
- Potential selective-escalation architecture for uncertain cases.

## Methods

- public datasets and provenance;
- reference implementations;
- student architectures;
- training/distillation;
- ONNX export;
- calibration/INT8 conversion;
- KL720 compilation/runtime;
- assay-specific post-processing;
- computer-vision metrics;
- biological endpoint definitions;
- equivalence/agreement statistics;
- hardware benchmarking;
- reproducibility.

## Data and code availability

All redistributable source code, manifests, configuration, and publication-safe derived benchmark tables will be released in this repository. Upstream datasets/models will be referenced by canonical DOI/accession/release rather than duplicated unless redistribution is explicitly appropriate.
