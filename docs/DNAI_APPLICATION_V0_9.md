# DNAi application adapter v0.9

## Scope

v0.9 makes `dnai-fiber-v3` the second assay executable through the generic
offline Radiation Edge AI / EDGE-RAD application control plane.

The enabled path is intentionally narrow:

frozen preprocessed 512 x 512 deployment windows
-> floating CPU ONNX inference
-> softmax
-> Gaussian overlap stitching
-> segmentation
-> DNA-fiber object reconstruction
-> tract measurements
-> verified field record
-> provenance-bound DNAi measurement transaction

The frozen deployment policy remains:

- processed field size: 1024 x 1024
- tile size: 512 x 512
- overlap: 50%
- stride: 256
- windows per field: 9
- Gaussian stitching sigma scale: 0.125
- pixel size: 0.26 um
- model: frozen optimized DNAi MobileOne-S1 FP512 ONNX

## High-level application result

The first generic `radedge assay-run` execution on the already characterized
field `1__tile_4` produced:

- field execution: `executed`
- field ID: `df1-72c2102e46e5a4af`
- transaction ID: `dt1-368578251dbcffe2`
- windows: 9
- reconstructed fibers: 18
- valid fibers: 14
- mean valid tract ratio: 1.0139419446401647
- median valid tract ratio: 1.0802218824979772
- mean valid fiber length: 16.36714469581888 um
- total valid fiber length: 229.1400257414643 um
- verification: PASS

A second identical invocation reported `field execution: reused`, demonstrating
deterministic reuse of the verified field instead of repeating inference.

## Scientific parity check

The field generated through the generic multi-assay front door was compared
against the previously frozen manual FP512 application reproduction.

All of the following were exact:

- field identity
- counts
- measurements
- scientific scope
- all 9 window-output SHA256 values
- stitched-probability artifact SHA256
- segmentation artifact SHA256
- valid-fibers CSV SHA256

The field ID remained `df1-72c2102e46e5a4af`.

This establishes application/orchestration parity for this frozen field. It
does not constitute a new biological validation.

## Provenance controls

The DNAi field verifier checks the frozen input/model/deployment identity,
window artifacts, derived artifacts, scientific scope, counts, and
measurements.

The DNAi measurement transaction additionally binds the exact field-record
bytes, application manifest, worker source, input identity, and measurement
summary.

A disposable negative control modified the copied field-record bytes after a
clean rebound. Transaction verification then returned:

- `artifacts_ok: false`
- `ok: false`
- CLI exit code 3

Thus the transaction detects modification of the exact field record it binds.

## Software gates

Before release freeze:

- changed-file Ruff: PASS
- focused v0.9 tests: 13 passed
- full repository tests: 95 passed
- `git diff --check`: PASS apart from informational Windows LF/CRLF notices
- DNAi result packaging fail-closed: PASS, CLI exit code 2

## Explicit exclusions

v0.9 does not claim or enable:

- raw-microscopy preprocessing through the public application front door
- use of human annotations during application execution
- FP1024-reference comparison during application execution
- new biological-fidelity validation
- KL720 DNAi application execution
- DNAi result packaging

The historical DNAi deployment characterization remains separate. In
particular, high pixel-class agreement between floating FP512 and physical
KL720 is not treated as proof of fiber-object fidelity.

## Interpretation

The v0.9 result demonstrates that the generic EDGE-RAD control plane can
execute a second assay-specific scientific pipeline without changing the
frozen scientific result for the characterized field.

It is application and orchestration evidence, not a new biological validation
or a new KL720-equivalence result.
