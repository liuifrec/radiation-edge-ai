# Project Charter

## Mission

Develop biologically trustworthy edge-AI methods for radiation-response microscopy by converting GPU-trained or otherwise high-capacity reference models into compact KL720-deployable inference systems and determining whether model compression and INT8 quantization preserve the quantitative biological conclusions.

## Tier I scope

Tier I deliberately contains three complementary radiation-relevant phenotyping tracks:

1. **gamma-H2AX / 53BP1** — DNA-damage burden and repair.
2. **Micronucleus** — chromosomal damage and genome instability.
3. **DNA fiber** — replication stress and fork-related phenotypes.

The project does not treat these as independent software demonstrations. All three are used to test one shared hypothesis:

> Biological endpoints can remain quantitatively trustworthy even when edge-oriented model compression causes some degradation in conventional computer-vision metrics, but the acceptable compression limit must be established empirically for each assay.

## Scientific questions

### Q1. Model compression
How far can each assay model be compressed while maintaining adequate image-level performance?

### Q2. Quantization
Does INT8 deployment introduce systematic bias that is not obvious from aggregate Dice/F1 metrics?

### Q3. Biological fidelity
Are endpoint estimates from edge inference statistically equivalent to the accepted reference workflow?

### Q4. Generality
Do the same validation principles hold across object detection, segmentation, and single-molecule image analysis?

### Q5. Deployment value
Can the resulting pipelines provide useful throughput on ordinary laboratory PCs without dedicated GPUs?

## Reference hierarchy

Every track should preserve a four-level comparison wherever possible:

```text
R0  manual/public ground truth
R1  original FP32 teacher/reference workflow
R2  compact FP32 student
R3  INT8 KL720 deployment
```

R0 and R1 are not assumed to be identical. Teacher-vs-human disagreement is part of the error model where public annotations permit it.

## Tier-I biological endpoints

### gamma-H2AX / 53BP1

Primary candidates:
- foci per nucleus/cell;
- fraction of foci-positive cells;
- gamma-H2AX/53BP1 colocalization where supported;
- dose-response parameters;
- post-irradiation repair-related change.

### Micronucleus

Primary candidates:
- micronuclei per scored cell;
- micronucleus-positive cell fraction;
- binucleated-cell scoring where labels support it;
- later extension to calibration/dose reconstruction if dose-labelled radiation microscopy is available.

### DNA fiber

Primary candidates:
- CldU tract length;
- IdU tract length;
- total tract length;
- IdU/CldU or other protocol-appropriate tract ratios;
- preservation of condition/genotype effect sizes.

## Engineering architecture

```text
reference data
    |
FP32 teacher/reference
    |
compact student
    |
ONNX
    |
calibration / INT8 quantization
    |
KL720 NEF
    |
edge inference
    |
CPU post-processing
    |
biological measurement
```

Where an upstream model cannot compile cleanly for KL720, it should be treated as a teacher rather than forcing unsupported operators onto the NPU.

## Validation philosophy

Model accuracy is necessary but not sufficient.

A deployment may be considered biologically acceptable even if some image-level metrics decrease, provided that:

- endpoint agreement remains within a pre-specified scientifically defensible margin;
- group ordering and principal effects are preserved;
- no meaningful condition-specific bias is introduced;
- uncertainty and failure cases are identifiable.

Conversely, excellent pixel/object metrics do not validate deployment if biological endpoints are biased.

## Initial development order

1. Freeze all three public reference baselines.
2. Convert DNA-fiber segmentation first because a working GPU DNAi reference pipeline already exists.
3. Build the gamma-H2AX/53BP1 edge proof of concept as the first radiation-biology showcase.
4. Distill micronucleus inference from a high-capacity public teacher into a compact CNN-compatible student.
5. Perform shared biological-fidelity and hardware benchmarking.
6. Package a reproducible demo suitable for internal RERF presentation and later RP development.

## Publication positioning

The first manuscript should be written as a general biological-methods paper rather than a chip-porting report.

**Primary target:** Cell Reports Methods  
**Fallback:** Radiation Research

The methodological contribution should remain meaningful even if future hardware differs from KL720.
