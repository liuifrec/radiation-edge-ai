# NASA BPS 53BP1 pilot notes

## Metadata inventory checkpoint — 2026-09-07

The metadata-only inventory completed successfully without downloading microscopy images.

### FITC / 53BP1

- rows: 77,177
- SHA256: `0951d7a3c9a28683be99f09e2e9019aefbdee96c02af74c2777dc6a5d9d03247`
- columns: `filename`, `dose_Gy`, `particle_type`, `hr_post_exposure`
- dose values: 0, 0.1, 0.3, 0.82, 1 Gy
- particle values: Fe, X-ray
- post-exposure times: 4, 24, 48 h
- observed condition cells: 17
- explicit plate/mouse/strain/sample identifier columns: none

### DAPI / MASK

- rows: 171,504
- SHA256: `f51159a10abf4882a57dc1ccb58d542533bde657c7c3e89345a3bafe07ba473c`
- same four exposed metadata columns as FITC/53BP1
- same observed dose, particle and time value sets
- observed condition cells: 17
- explicit plate/mouse/strain/sample identifier columns: none

## Immediate implication

Do **not** freeze a random image-level train/test split yet. The underlying OSD-366 study contains multiple animals and genetic backgrounds, but the benchmark CSVs do not expose those grouping variables directly. We must first determine whether grouping information is encoded in filenames or recoverable from OSD-366 metadata.

A random cell/nucleus split could produce severe biological leakage because many nuclei from the same plate, animal, strain or acquisition batch may appear in both training and validation.

## Next gate

Run `src/radiation_edge_ai/nasa_bps/inspect_filename_groups.py` to report:

- all 17 condition counts;
- representative filenames per condition;
- filename token cardinalities;
- candidate prefix grouping structure;
- FITC vs DAPI/MASK filename crosswalk statistics.

Only after a defensible independent grouping key is established should the radiation-native pilot and held-out validation split be frozen.
