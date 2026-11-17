# NASA BPS 53BP1 plan

**Updated direction: 2026-11-17.** The active work is the
[NASA public-data multi-model NPU plan](NASA_MULTI_MODEL_PLAN.md): expand the
number and variety of accurate models that execute locally on physical KL720,
using a common NASA benchmark.

## Starting point

The original metadata reconnaissance, pilot development and single-model
deployment have already produced a frozen NASA BPS / OSD-366 R1 v2 physical
validation. The next milestone is a reusable multi-architecture comparison,
starting with the established compact CNN, MobileNetV2 and ResNet18.

The current common task is sample-level 53BP1 burden regression. Inputs are
native-scale FITC/DAPI crops normalized and center-padded to **256 x 256**, with
a zero third channel. Sample/plate-well `avg_nfoci` is aggregate supervision;
individual nucleus scores are latent burden estimates. Earlier proposals for
512 x 512 input or directly supervised focus maps are not the implemented
NASA contract.

## Public sources

- [NASA BPS Microscopy Benchmark](https://registry.opendata.aws/bps_microscopy/)
- Public data prefix: `s3://nasa-bps-training-data/Microscopy/`
- FITC metadata: `Microscopy/train/meta.csv`
- DAPI/MASK metadata: `Microscopy/DAPI_MASK_images/meta_DAPI_MASK.csv`
- OSD-366 / GLDS-366: [10.26030/v8w4-rg83](https://doi.org/10.26030/v8w4-rg83)

Retain access dates, source identities, resolved grouping and checksum records.
The overall study's range of conditions does not imply that the project has
validated every condition or mouse strain.

## Current operational references

- [Multi-model plan and next steps](NASA_MULTI_MODEL_PLAN.md)
- [Project charter and priorities](PROJECT_CHARTER.md)
- [R1 v2 input and contrast definitions](NASA_BPS_R1_V2_CONTRAST_POLICY.md)
- [Historical physical validation](NASA_BPS_R1_V2_KL720_VALIDATION_RECORD.md)
- [Post-release claim correction and limits](NASA_R1_V2_POSTRELEASE_AUDIT.md)

The former pilot sequence is retained in Git history. Its initial-stage tasks
are not a request to restart completed work. New architecture selection uses
grouped development evaluation; the already examined historical final holdout
is not newly untouched confirmation data.
