# Multi-assay application abstraction v0.8

## Purpose

v0.8 introduces an application-level assay runtime registry above the assay-specific scientific pipelines.

The abstraction selects an approved assay adapter. It does not attempt to make scientifically different assays share inference, reconstruction, measurement, or QC logic.

## Architecture

radedge
  -> assay runtime registry
     -> nasa-53bp1-r1-v2
        -> existing NASA batch/measurement application path
     -> dnai-fiber-v3
        -> registered scientific contract
        -> execution disabled until the DNAi tile/stitch/object pipeline is integrated

## Registered assays

### nasa-53bp1-r1-v2

- application adapter: nasa-batch-measurement-v1
- package adapter: nasa-sample-aggregate-package-v1
- manifest execution: enabled
- result packaging: enabled
- prediction semantics: per-nucleus latent continuous 53BP1 burden
- endpoint: sample-level arithmetic mean latent 53BP1 burden

### dnai-fiber-v3

- required application adapter: dnai-tile-stitch-object-v1
- required package adapter: dnai-fiber-measurement-package-v1
- manifest execution: disabled in v0.8
- result packaging: disabled in v0.8
- endpoint: post-processed DNA-fiber objects and tract measurements

DNAi fails closed rather than being routed through the NASA measurement logic.

Its scientific boundary remains explicit: pixel agreement is not a surrogate for fiber-object fidelity, and object reconstruction plus downstream tract measurements require assay-specific QC.

## NASA compatibility evidence

- frozen transaction: t1-387be2e6fe7b3a0c
- transaction SHA256: a3360d92c93ee05351dd3c6a022b88bde38bd8bbaef41a6089526594209574c6
- frozen package: rp1-6c2271046fac4549
- package-manifest SHA256: 6a29dc1e3657cb2d0216f993ba890694122d82945cc19e98379f8db83b9efc67
- sample: BALBCF2_P244_C7_BALBC_FEMALE_1
- nuclei: 3
- mean latent burden: 0.5816842913627625

The generic assay dispatcher reproduced the existing NASA application result while an intentionally nonexistent ONNX worker executable was supplied. The frozen transaction was reused without modification, demonstrating that no new inference was required.

The frozen v0.7 package also passed verification through the generic assay registry using the NASA package adapter and remained byte-identical.

## DNAi fail-closed evidence

The generic application recognizes dnai-fiber-v3 as a registered assay but returns an explicit disabled-adapter error rather than applying NASA-specific science.

Required future application adapter:
dnai-tile-stitch-object-v1

Required future result-package adapter:
dnai-fiber-measurement-package-v1

## Tests

- runtime registry tests: 9/9 PASS
- existing NASA application/package tests: 33/33 PASS
- full suite: 91/91 PASS
- changed-file Ruff: PASS
- git diff check: PASS

## Evidence freeze

External v0.8 evidence freeze SHA256:
20ecb6dd184b0688d56ec3f4cf4ead0367e2305be7345b5540d51ca6849eea64

Superseded preliminary v0.8 freeze SHA256:
1cec91af9e894d5df1e1d83ef6fb8227b3d501462cd12e0e2d7058fc4e276f01

The preliminary freeze was created before the final changed-file Ruff gate and is retained as immutable historical evidence. It is superseded by the final freeze above after three auto-fixable lint issues were repaired and the complete test and compatibility gates were rerun.

Prior NASA v0.7 freeze SHA256:
d33cd5aaccab27ec48792d162c55391d3cfd961de880d2e34641127549e25a02

## Interpretation

v0.8 demonstrates a multi-assay application boundary that preserves assay-specific scientific logic.

NASA remains fully enabled through its previously validated application path. DNAi is discoverable and scientifically described but intentionally disabled until its distinct tiling, segmentation, stitching, object reconstruction, tract measurement, and QC workflow is integrated.

This is architectural compatibility evidence. It is not a new biological-validation, deployment-equivalence, or DNAi-performance result.
