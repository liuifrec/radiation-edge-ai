# Biological Fidelity Framework

## Purpose

This project distinguishes **image-level accuracy** from **biological measurement fidelity**.

The central validation question is:

> Would a biologist reach the same quantitative conclusion using KL720 INT8 inference as using the accepted reference workflow?

## Evaluation layers

### 1. Model / image layer

Use conventional metrics appropriate to the task:

- precision;
- recall;
- F1;
- Dice;
- IoU;
- object-level detection sensitivity;
- false-positive and false-negative rates;
- calibration/confidence where available.

These metrics diagnose model behavior but are not the final success criterion.

### 2. Biological endpoint layer

Evaluate the assay-specific quantity directly.

#### gamma-H2AX / 53BP1

Examples:
- foci per cell;
- foci-positive fraction;
- colocalized foci per cell;
- dose-response slope/curvature;
- difference between post-irradiation time points;
- genotype/cell-type effect size.

#### Micronucleus

Examples:
- micronuclei per scored cell;
- micronucleus-positive cell fraction;
- agreement in accepted/rejected cells;
- group effect size;
- dose reconstruction if a radiation calibration series is available.

#### DNA fiber

Examples:
- CldU tract length;
- IdU tract length;
- total tract length;
- IdU/CldU ratio;
- distributional shift between experimental conditions;
- preservation of genotype/treatment effect sizes.

### 3. Deployment layer

Record at minimum:

- model size;
- input dimensions;
- MACs/parameter count where available;
- inference latency;
- throughput;
- CPU/RAM burden;
- conversion/quantization configuration;
- KL720 firmware/SDK/compiler versions;
- host CPU/OS;
- energy per image if reliable measurement is available.

## Reference levels

Use explicit labels in benchmark outputs:

- **R0:** manual/public ground truth;
- **R1:** original FP32 teacher/reference workflow;
- **R2:** compact FP32 student;
- **R3:** INT8 KL720 student.

Where practical, retain raw paired measurements so that agreement can be analyzed without relying only on aggregate metrics.

## Statistical strategy

The final strategy will be endpoint-specific, but candidate analyses include:

- paired error and relative error distributions;
- Bland-Altman plots and limits of agreement;
- ICC/reliability coefficients where appropriate;
- Pearson/Spearman correlation as descriptive measures only, not proof of agreement;
- bootstrap confidence intervals;
- effect-size preservation;
- regression-parameter comparison for dose-response/kinetics;
- equivalence testing with pre-specified margins.

## Equivalence margins

Do **not** select margins after viewing the final results.

For each endpoint, a margin should be justified before confirmatory analysis using one or more of:

- manual scorer variability;
- published assay reproducibility;
- biologically meaningful effect size;
- known technical variation;
- intended use (screening vs quantitative measurement vs biodosimetry).

A provisional engineering margin may be used during development, but manuscript-level equivalence claims require a documented scientific rationale.

## Compression frontier

For each assay, evaluate progressively smaller/cheaper models where feasible:

```text
teacher FP32
   -> student FP32
   -> student INT8
   -> smaller INT8 variants
```

Plot at least three axes:

1. computational cost/model size;
2. image-level accuracy;
3. biological endpoint error.

The goal is to identify the **biological operating point**, not simply the maximum Dice/F1 model.

## Failure analysis

Track failures by biological context rather than only as generic false positives/negatives. Examples:

- low-dose vs high-dose fields;
- dense vs sparse foci;
- small vs large micronuclei;
- crossing/short DNA fibers;
- low-signal or high-background images.

A deployment that performs well on average but fails systematically in one biologically important regime is not considered trustworthy.

## Selective escalation (advanced)

A later extension may allow uncertain edge cases to escalate to a larger model or human scorer:

```text
KL720 inference
    |
confidence / QC
    |-- high confidence -> accept locally
    `-- ambiguous -> teacher/GPU/human
```

This will be evaluated only after the standalone edge models are established.
