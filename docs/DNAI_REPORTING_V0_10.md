# DNAi portable reporting v0.10

## Scope

v0.10 adds portable result packaging for the already-runnable
`dnai-fiber-v3` assay.

The enabled path is:

verified DNAi measurement transaction
-> generic `radedge assay-report`
-> path-independent portable result package
-> generic package verification

Packaging performs no inference, image preprocessing, stitching,
segmentation, fiber reconstruction, human-annotation lookup,
FP1024-reference comparison, biological-fidelity evaluation, or KL720 access.

The source `dnai_fiber_measurement_transaction` remains the provenance
authority.

## Package contents

The portable package contains five files:

- `package_manifest.json`
- `result.json`
- `provenance.json`
- `valid_fibers.csv`
- `report.md`

It intentionally excludes:

- model binaries
- deployment-window tensors
- raw microscopy
- stitched probability arrays
- segmentation images
- human annotations
- FP1024 reference outputs
- historical biological-validation artifacts

## First real package

The first real v0.10 package was exported from frozen v0.9 transaction:

- transaction ID: `dt1-368578251dbcffe2`
- field ID: `df1-72c2102e46e5a4af`
- package ID: `drp1-c30e7b91a3f0250d`
- windows: 9
- reconstructed fibers: 18
- valid fibers: 14
- mean valid tract ratio: 1.0139419446401647
- median valid tract ratio: 1.0802218824979772
- mean valid fiber length: 16.36714469581888 um
- total valid fiber length: 229.1400257414643 um

The package verified through the generic multi-assay verifier with all package
identity, file-integrity, derivation, scientific-scope, count, and
fiber-table-binding checks passing.

## Exact export fidelity

The portable export was compared directly with the frozen source transaction
and field record.

Exact agreement was confirmed for:

- source transaction ID and SHA256
- field ID
- sample ID
- image index
- counts
- measurements
- scientific scope
- `valid_fibers.csv` SHA256
- `valid_fibers.csv` byte size

This is export/presentation fidelity. It is not a new biological-validation
result.

## Portability

A copied package was verified successfully from a different directory.

A separate audit confirmed that none of the five package files contained the
original absolute paths for:

- source transaction
- source field record
- worker source

Thus package verification does not depend on the original application working
directory or upstream artifact paths.

## Tamper detection

A disposable copy of the portable package was first verified clean.

The copied `valid_fibers.csv` was then modified without changing the package
manifest. Verification detected the modification at the package-file integrity
layer and returned:

- `files_ok: false`
- `ok: false`
- CLI exit code 3

The original package remained fully verified afterward.

## Cross-platform source provenance amendment

While preparing v0.10, the frozen v0.9 transaction initially failed
verification on Windows because Git had materialized `_field_worker.py` with
CRLF line endings while the transaction had bound the original LF bytes.

Read-only audit showed:

- application manifest: exact stored SHA256 and size
- field record: exact stored SHA256 and size
- worker source: physical working-tree bytes differed
- worker source after CRLF-to-LF canonicalization: exact stored SHA256 and size

v0.10 therefore canonicalizes only the worker source text to LF for provenance
creation and verification.

Scientific artifacts remain strict byte-verified.

The original frozen field, transaction, measurements, and inference outputs
were not regenerated or modified.

The verifier exposes this explicitly:

- `worker_source_exact_bytes_ok: false`
- `worker_source_lf_canonical_ok: true`
- `worker_source_lf_normalization_used: true`

This is a cross-platform source-text provenance compatibility amendment, not a
change to the scientific result.

## Multi-assay control plane

The runtime registry now exposes:

- DNAi manifest execution: enabled
- DNAi result packaging: enabled
- package adapter: `dnai-fiber-measurement-package-v1`

NASA result-package behavior remains supported.

CLI regression tests cover both NASA-shaped and DNAi-shaped portable-package
verification output.

## Explicit exclusions

v0.10 does not add or claim:

- raw-microscopy ingestion through the public DNAi application front door
- new DNAi inference results
- new biological validation
- new human-comparison analysis
- FP1024-reference evaluation
- physical KL720 DNAi execution
- KL720 deployment equivalence
- retuning of tiling, PTQ, segmentation, or object reconstruction

The historical DNAi deployment characterization remains separate.

## Interpretation

v0.10 completes the current DNAi offline application loop:

`assay-run -> verified DNAi transaction -> assay-report -> portable verified result package`

The milestone demonstrates deterministic, portable, tamper-detecting export of
an already-produced floating-FP512 DNA-fiber measurement.

It is a reporting/provenance milestone, not a new biological-validation or
hardware-equivalence result.
