# NASA BPS R1 v2 INT8 calibration freeze

Status: **predeclared before any R1 v2 INT8/BIE holdout evaluation**

This document freezes the post-training quantization calibration selection rule for the NASA BPS R1 v2 53BP1 deployment candidate.

## Inputs

Authoritative repaired development manifest SHA256:

`23e21fdc8bf00a2814b64c56223cfd3d8e96eda71bdbff5d8ec74dae218cf5a9`

Frozen ONNX deployment candidate SHA256:

`a65718a8f07d5bcd8707bae78d8f730d56047db21e575fa55ba5396adc49a07e`

The development manifest contains exactly:

- 7,200 nuclei;
- 72 sample/plate-well bags;
- 100 nuclei per bag;
- 6 development Source Names;
- all frozen Fe and X-ray sham/exposed/time conditions used during R1 v2 development.

The final 2,058-nucleus holdout is forbidden for calibration-image selection, calibration ranges, PTQ settings, optimizer tuning, compiler tuning, or threshold tuning.

## Frozen selection rule

The calibration panel contains exactly **288 development nuclei**: **4 nuclei from each of the 72 development bags**.

Selection is deterministic and phenotype-blind:

1. group the repaired development manifest by exact `sample_name`;
2. require exactly 100 nuclei in every bag;
3. for every nucleus compute SHA256 over the ASCII/UTF-8 string
   `7203|<sample_id>|<nucleus_key>`;
4. sort nuclei within each bag by that digest, then by `sample_id` as a deterministic tie-breaker;
5. select the first four nuclei from every bag;
6. do not inspect NASA phenotype values, model predictions, FITC intensity, DAPI intensity, segmentation masks, residuals, or any holdout result during selection.

Because every development bag contributes the same number of nuclei, the calibration panel is exactly balanced across the 72 source x branch x dose x time sample/plate-well bags represented by the frozen development design.

No alternative calibration subset may be tried after viewing final-holdout INT8/BIE performance.

## Frozen preprocessing

Each selected development nucleus is converted to one float32 tensor using the unchanged R1 preprocessing implementation:

- load native FITC and DAPI TIFF crops;
- per-channel p1/p99.5 robust normalization;
- clip to [0, 1];
- no resize;
- center-pad at native scale to 256 x 256;
- third channel identically zero;
- save NCHW shape `[1, 3, 256, 256]` as `.npy`.

MASK remains QC-only and is not an INT8 calibration input.

## Audit requirements

The calibration materializer must record:

- repaired development manifest SHA256;
- frozen ONNX SHA256 and ONNX-freeze SHA256;
- seed `7203`;
- exactly 72 bags and 288 selected nuclei;
- exactly 4 nuclei per bag;
- exact source/branch/dose/time counts;
- each selected `sample_id`, `nucleus_key`, deterministic selection digest, tensor path, tensor SHA256, tensor shape/dtype/min/max/mean/std;
- global tensor value range;
- explicit statements that final-holdout images and outcomes were not read for calibration selection.

Calibration tensors are generated artifacts and are not committed to Git.

## PTQ rule

The first Kneron PTQ attempt must use the toolchain's standard percentage-range settings already used by this project unless an implementation error prevents analysis:

- `datapath_range_method = percentage`;
- `percentage = 0.999`;
- `percentage_16b = 0.999999`;
- `optimize = 0`;
- platform `720`.

Any implementation-only correction must be documented. The final holdout must never be used to choose among PTQ settings.

## Next gates

The deployment sequence remains:

`frozen ONNX -> KL720 floating-point optimize/IP evaluation -> development-only PTQ calibration -> BIE -> final-holdout BIE biological equivalence -> NEF -> physical KL720`.

Only after the INT8/BIE configuration is frozen from development data may the 2,058-nucleus final holdout be evaluated again for deployment equivalence under the already-predeclared biological gates.
