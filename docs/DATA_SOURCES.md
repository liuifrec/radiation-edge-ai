# Public Data and Model Sources

This file records the canonical public sources used for Tier-I reference baselines. Exact releases, checksums, licenses, and train/validation/test manifests should be pinned before benchmark results are treated as publication-grade.

## gamma-H2AX / 53BP1

Planned starting points:

- **DeepFoci** — public gamma-H2AX/53BP1/DAPI microscopy with manual annotations and dose/time information.
- **FociRad** — public gamma-H2AX radiation-biodosimetry workflow/data suitable for an early detector-oriented proof of concept.

Required before use:

- exact dataset DOI/release;
- license/terms;
- image dimensionality and channel mapping;
- published split definitions;
- dose and post-irradiation metadata;
- annotation format;
- local checksum manifest.

## Micronucleus

Planned starting point:

- **mnDINO** public model/code and associated annotated microscopy data.

The high-capacity transformer-based reference should be treated as a teacher/reference. A compact CNN-compatible student will be developed for KL720 rather than assuming direct transformer deployment.

Required before use:

- exact repository/model release;
- BioImage Archive accession/release;
- annotation classes and filtering rules;
- published train/validation/test split;
- license/terms;
- checksum manifest.

## DNA fiber

Planned starting point:

- **DNAi** public training/test images, masks, model/code, and post-processing workflow.

A working GPU DNAi analysis pipeline already exists outside this repository and will serve as an important R1 reference. This repository should preserve only publication-safe code/configuration needed to reproduce the comparison; private/local biological images must not be committed.

Required before use:

- exact DNAi release/DOI;
- architecture and checkpoint used as R1;
- test-set manifest;
- channel conventions;
- pixel/physical scaling conventions where required;
- post-processing version;
- license/terms;
- checksum manifest.

## Data policy

Do not commit:

- raw public datasets that can be fetched from their canonical archive;
- unpublished RERF data;
- identifiable or sensitive institutional material;
- large model binaries unless redistribution is explicitly permitted and technically justified;
- proprietary Kneron compiler/runtime files.

Instead commit:

- download instructions/scripts;
- source URLs/DOIs/accessions;
- release identifiers;
- license notes;
- checksums;
- deterministic split/sample manifests;
- preprocessing parameters;
- derived publication-safe benchmark tables.
