# NASA BPS 53BP1 Pilot v1 — Frozen Design

Pilot v1 was frozen after resolving the public BPS microscopy benchmark to OSD-366 biological metadata.

## Frozen manifest

- 2,160 nuclei
- 6 complete OSD Source Names
- 12 radiation condition cells
- 30 nuclei per Source Name x condition
- each selected nucleus requires a FITC/53BP1 image plus paired DAPI and MASK files
- manifest SHA256: `41ff2e4d64577f17c976dea5bbc17221fe63be9e2e7e2291309b520027fbe725`

## Biological sources

- BALBCF1 — BALB/cByJ, Female
- BALBCM1 — BALB/cByJ, Male
- BALBCM2 — BALB/cByJ, Male
- C57BLF1 — C57BL/6J, Female
- C57BLM1 — C57BL/6J, Male
- C57BLM3 — C57BL/6J, Male

These six sources each cover all 17 original benchmark condition cells. Pilot v1 intentionally uses only these complete sources.

## Pilot radiation panel

Fe:
- 0 Gy at 4, 24, 48 h
- 0.82 Gy at 4, 24, 48 h

X-ray:
- 0 Gy at 4, 24, 48 h
- 1.0 Gy at 4, 24, 48 h

Low-dose X-ray 0.1 Gy and Fe 0.3 Gy are reserved for expansion after the first source-held-out proof of concept.

## Leakage-safe validation folds

- `fold_female`: test BALBCF1 + C57BLF1
- `fold_male_1`: test BALBCM1 + C57BLM1
- `fold_male_2`: test BALBCM2 + C57BLM3

Each fold holds out whole OSD Source Names. Random nucleus-level train/test splitting is prohibited.

## Data retrieval and QC

Use:

- `src/radiation_edge_ai/nasa_bps/download_pilot_v1.py` to download only the 2,160 frozen FITC/DAPI/MASK triplets from the public NASA BPS S3 resource, with local SHA256 inventory and resumable transfer.
- `src/radiation_edge_ai/nasa_bps/inspect_pilot_v1_images.py` to inspect TIFF shape, dtype, triplet alignment, nucleus-mask area, and descriptive intensity distributions before choosing a 53BP1 reference endpoint/model.

Raw public TIFF files and local download inventories remain under `RADEDGE_DATA_ROOT` and are not committed.

## Scientific gate

The next decision is not a classifier architecture choice. First determine from the downloaded pilot:

1. actual image geometry and bit depth;
2. DAPI/MASK/FITC alignment;
3. mask semantics and area distribution;
4. whether the public benchmark contains sufficient information for a direct foci-level reference target, or whether a separate validated foci teacher/annotation source is required;
5. only then define the compact FP32 -> INT8 KL720 model and biological-fidelity endpoints.

The eventual deployment claim must be based on preservation of radiation-response conclusions (dose, radiation quality/LET, repair time and biological-source effects), not merely image-classification accuracy.
