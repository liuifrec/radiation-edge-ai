# NASA BPS R1 v2 deployment-equivalence policy

Status: **predeclared before R1 v2 ONNX / INT8 / BIE / NEF / physical-KL720 deployment testing**

This document freezes the deployment-equivalence logic for the authoritative NASA BPS R1 v2 53BP1 model after successful FP32 final-holdout confirmation.

## Authoritative FP32 biological reference

Checkpoint SHA256:

`2b07c67d6e50e7a36e26e0052feebc826b518e6efae863ba312ba9761a4eb3d5`

QC1 repaired holdout manifest SHA256:

`f89962f96bec05930c07149717a90bd682016a2778bbfae6b827425159aa0643`

Final-holdout structure:

- 2,058 nuclei;
- 22 sample/plate-well bags;
- 11 branch-specific exposed-minus-sham contrasts;
- BALBCF2: 5 contrasts;
- C57BLF2: 3 contrasts;
- C57BLF3: 3 contrasts.

The one-time confirmation completed with the documented metadata-alias implementation recovery. No holdout outcome was used for training, model selection, BatchNorm repair, threshold selection, or any later tuning.

Observed frozen FP32 confirmation summary:

- bag MAE 0.5843;
- bag RMSE 0.8839;
- bag Pearson 0.9715;
- bag Spearman 0.8616;
- delta MAE 0.9851;
- delta Pearson 0.9885;
- delta Spearman 0.9909;
- overall direction agreement 10/11;
- 4 h direction agreement 4/4;
- 24 h + 48 h direction agreement 6/7;
- source direction agreement BALBCF2 4/5, C57BLF2 3/3, C57BLF3 3/3;
- peak-time recovery 3/3;
- all seven predeclared FP32 biological-fidelity criteria passed.

These rounded values are descriptive. The exact deployment reference is the completed confirmation JSON/CSV output set, frozen by SHA256 before ONNX export.

## Accelerator boundary

The deployable neural graph is exactly:

`float32 NCHW [1, 3, 256, 256] -> scalar continuous latent 53BP1 burden`

The model input channels remain:

1. FITC / 53BP1 channel after the frozen per-nucleus p1/p99.5 normalization;
2. DAPI channel after the same frozen normalization rule;
3. all-zero channel.

Native crop scale is preserved. Images are center-padded to 256 x 256. No resize is permitted.

The following remain **outside** ONNX / KL720 and are host preprocessing or aggregation:

- TIFF loading;
- target-crop identity and hard-QC logic;
- p1/p99.5 FITC/DAPI normalization;
- center padding;
- construction of the zero third channel;
- bag averaging over selected nuclei;
- branch-specific sham matching;
- biological metrics and seven-gate evaluation.

MASK remains QC-only and is never model input.

## Stage A - FP32 ONNX parity

The authoritative checkpoint is exported as a fixed-shape ONNX graph using opset 11, matching the repository's established KL720 deployment convention.

ONNX Runtime must be compared against both:

1. a fresh CPU PyTorch load of the authoritative checkpoint; and
2. the frozen confirmatory GPU-FP32 per-nucleus outputs.

Numerical gates are predeclared as:

### ONNX Runtime versus CPU PyTorch

- maximum absolute per-nucleus error <= 1e-4;
- mean absolute per-nucleus error <= 1e-5;
- RMSE <= 1e-5.

### ONNX Runtime versus frozen confirmatory GPU FP32

- maximum absolute per-nucleus error <= 1e-3;
- mean absolute per-nucleus error <= 1e-4;
- RMSE <= 1e-4.

These numerical tolerances detect export/preprocessing mistakes while allowing normal FP32 kernel-order differences.

The ONNX biological gate is stronger than numerical similarity alone. On the same 22 bags / 11 contrasts, ONNX must:

- pass all seven original biological-fidelity criteria;
- reproduce the FP32 direction-count signature exactly: overall 10/11, 4 h 4/4, 24+48 h 6/7, BALBCF2 4/5, C57BLF2 3/3, C57BLF3 3/3;
- reproduce peak-time recovery 3/3.

ONNX failure at either numerical or biological parity blocks quantization.

## Stage B - INT8 calibration and Kneron optimization

Calibration data must come **only from the 7,200-nucleus R1 v2 development set**. The 2,058-nucleus final holdout must not be used to choose calibration images, ranges, quantization options, optimizer settings, thresholds, or compiler settings.

Any calibration subset, if a subset is required by the Kneron toolchain, must be selected by a phenotype-blind deterministic rule frozen before inspecting INT8 holdout outcomes. The selection rule and exact calibration manifest SHA256 must be recorded.

No retraining, fine-tuning, BatchNorm recalibration, knowledge distillation, threshold tuning, or outcome-guided quantization is permitted.

## Stage C - BIE / INT8 biological equivalence

After INT8 configuration is frozen from development data, the same 2,058 holdout nuclei may be used strictly as a deployment-equivalence test.

INT8/BIE does not need to be numerically identical to FP32. It must nevertheless preserve the scientific conclusions. Mandatory biological gates are:

- all seven original final-holdout criteria pass;
- overall direction agreement is at least 9/11;
- 4 h direction agreement remains 4/4;
- 24+48 h direction agreement remains at least 5/7;
- source minima remain BALBCF2 >=4/5, C57BLF2 >=2/3, C57BLF3 >=2/3;
- peak-time recovery remains 3/3.

The exact INT8-vs-FP32 numerical error distribution, bag residuals, correlations, and any changed individual contrast directions are mandatory descriptive outputs even if all gates pass.

## Stage D - NEF and physical KL720

The compiled NEF must be tied by SHA256 to the frozen INT8/BIE artifact.

Physical KL720 inference must use the same external preprocessing and the same 2,058 nuclei. Device outputs are aggregated into the same 22 bags and 11 contrasts.

A successful physical-device result requires:

- all seven original biological criteria pass;
- the same mandatory direction and peak-time minima used for BIE pass;
- no host-side outcome-dependent correction of KL720 outputs;
- the deployed model, NEF, PLUS/runtime versions, firmware version, and device identity are recorded.

The intended final claim is therefore biological rather than merely numerical:

> INT8 KL720 inference preserved the radiation-effect direction and repair-kinetic conclusions of the frozen FP32 reference.

A numerical match alone is insufficient for this claim.

## Failure rule

At every deployment stage, settings may be debugged for implementation correctness using development data and toolchain diagnostics. Once a stage is evaluated on the final holdout, the holdout result must not be used to tune settings and rerun until it passes.

If ONNX export itself is implementation-broken, a corrected exporter may be created only if weights, preprocessing, inputs, biological rules, and thresholds remain unchanged and the correction is documented before re-evaluation.
