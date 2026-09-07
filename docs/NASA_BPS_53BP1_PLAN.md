# NASA BPS 53BP1 Space-Radiation Edge-AI Plan

## Objective

Develop a radiation-biology-native edge-AI case using the NASA Biological and Physical Sciences (BPS) Microscopy Benchmark. The central question is not simply whether a compact model can run on KL720, but whether physical INT8 edge inference preserves biologically meaningful conclusions about radiation-induced 53BP1 foci.

Working biological-fidelity question:

> Would a radiation biologist reach the same conclusion about dose, radiation quality/LET, repair kinetics, and biological background using the KL720 workflow as using the floating-point reference workflow?

## Why this dataset

The NASA BPS Microscopy Benchmark contains fluorescence microscopy images of individual mouse fibroblast nuclei with 53BP1-positive radiation-induced foci after X-ray or particle irradiation. NASA documents the images as maximum-intensity projections derived from 9-layer microscopy Z-stacks.

The associated OSDR study OSD-366 / GLDS-366 analyzes ex-vivo primary skin fibroblasts from 15 mouse strains exposed to X rays and high-LET 40Ar / 56Fe particles. This gives the project direct space-radiation relevance while retaining familiar radiation-biology endpoints.

## Canonical public sources

- AWS Open Data Registry: `https://registry.opendata.aws/bps_microscopy/`
- Public S3 prefix: `s3://nasa-bps-training-data/Microscopy/`
- FITC/53BP1 metadata: `Microscopy/train/meta.csv`
- DAPI/MASK metadata: `Microscopy/DAPI_MASK_images/meta_DAPI_MASK.csv`
- NASA OSDR study: OSD-366 / GLDS-366
- OSDR DOI: `10.26030/v8w4-rg83`
- Penninckx et al., Radiation Research 2019, DOI `10.1667/RR15338.1`

The AWS registry states that there are no restrictions on use of the benchmark dataset. Because the benchmark is updated when new fluorescence microscopy data become available, publication-grade work should pin access date and SHA256 checksums of metadata and selected image files rather than assuming a timeless release.

## Stage 1 — metadata-only reconnaissance

Do not bulk-download images first.

Run the project-owned metadata inventory to:

1. download only `meta.csv` and `meta_DAPI_MASK.csv`;
2. record metadata SHA256 values and access time;
3. inventory image counts across dose, particle type, and hours post-exposure;
4. inspect whether plate, mouse, strain, well, or sample identifiers are already present;
5. determine whether an OSD-366/file-name crosswalk is needed before defining train/test splits.

The utility intentionally avoids classifying sham samples from `particle_type`; `dose_Gy` is the safer primary sham/irradiated discriminator because particle labels may also be populated for 0-Gy rows.

## Stage 2 — freeze a small radiation-native pilot

The first pilot should be biologically informative, not merely convenient for machine learning. Prefer conditions that create orthogonal contrasts:

- sham versus irradiated;
- low-LET X ray versus higher-LET particle exposure;
- early versus late post-exposure time;
- multiple doses where sample counts support it;
- multiple biological backgrounds once strain/mouse identity is traceable.

Freeze the pilot only after resolving grouping variables so the split does not leak images from the same plate, mouse, or biological background across training and validation/test sets.

## Stage 3 — choose the reference task

Avoid making irradiated-versus-sham image classification the main scientific endpoint. It can be a smoke test, but the publication case should quantify 53BP1 biology.

Preferred endpoints:

- 53BP1 foci per nucleus;
- focus size / repair-domain area;
- residual foci at later time points;
- dose-response behavior;
- repair kinetics;
- radiation-quality / LET dependence;
- preservation of biological-background ranking.

A compact dense-prediction or heatmap model is preferred over a generic detector if small-foci localization is more stable that way.

## Stage 4 — KL720 deployment

Reuse the deployment discipline established by the DNA-fiber case:

1. compact FP32 model;
2. ONNX export and parity check;
3. Kneron optimization;
4. INT8 PTQ with held-out biological validation;
5. BIE validation;
6. NEF compilation;
7. physical KL720 inference;
8. phenotype-level comparison against the reference workflow.

The current verified 512x512 KL720 deployment envelope is the default starting point. Do not reopen larger-tile compiler tuning unless the biology requires it.

## Stage 5 — biological-fidelity gate

Report conventional CV metrics, but do not use them alone to claim success.

The decisive analysis should compare reference versus KL720 conclusions for:

- direction and magnitude of dose effects;
- early-to-late repair trends;
- low- versus high-LET contrasts;
- strain/background ordering where available;
- effect sizes and uncertainty, not only mean pixel accuracy.

The strongest result is not necessarily zero numerical drift. It is preservation of the biologically relevant radiation-effects conclusion under constrained edge inference.

## Space-biology deployment rationale

A future spacecraft or remote radiation-biology platform may not be able to assume continuous high-bandwidth connectivity or a dedicated GPU. Edge inference can therefore support:

- local continuous analysis during communication gaps;
- reduced downlink by transmitting quantitative summaries instead of every raw frame;
- event-triggered retention of high-resolution images;
- lower-power near-instrument inference;
- reproducible autonomous monitoring of radiation-induced biological change.

This is the long-term framing: **autonomous radiation biology at the edge**.
