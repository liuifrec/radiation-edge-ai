# NASA portable assay-result package v0.7

## Purpose

v0.7 adds a portable, path-independent assay-result export derived from a fully verified v0.5 batch-measurement transaction.

The result package is a presentation and exchange artifact. It is not a replacement for the source transaction record, which remains the provenance authority.

No neural inference, prediction transformation, endpoint reconstruction, biological-reference lookup, biological-acceptance evaluation, or KL720 hardware access is performed by the export layer.

## Package layout

rp1-6c2271046fac4549/
  package_manifest.json
  result.json
  provenance.json
  sample_aggregates.csv
  report.md

The package contains no model, input tensors, raw microscopy, biological reference, ONNX file, NEF file, PyTorch checkpoint, or other upstream binary artifact.

## Real NASA export

Package ID:
rp1-6c2271046fac4549

Source transaction ID:
t1-387be2e6fe7b3a0c

Source transaction SHA256:
a3360d92c93ee05351dd3c6a022b88bde38bd8bbaef41a6089526594209574c6

Package-manifest SHA256:
6a29dc1e3657cb2d0216f993ba890694122d82945cc19e98379f8db83b9efc67

Exported artifact SHA256 values:

- result.json: 65eb40ec386d7ff26c06dfbbe28b67eb62c63fffe6c2086d5f4335e2bb041c47
- provenance.json: 6f59ad9f9245da079da5d9aa836645ae743b9988ac7ce2b2b5c68530a3b21e94
- sample_aggregates.csv: 7832c1a9916743d706741791975537c8124a909951f0a28c965e24b0e3f19370
- report.md: af69f56283cf6d09e9aa30cb030ff46bf798bca5013a149f16dbcdd8f851e270

## Scientific endpoint

- sample: BALBCF2_P244_C7_BALBC_FEMALE_1
- radiation branch: Fe
- dose: 0.0 Gy
- timepoint: 4 h
- nuclei: 3
- mean latent 53BP1 burden: 0.5816842913627625

The exported endpoint is copied from the already verified transaction-linked aggregate artifact. v0.7 does not recalculate it.

## Verification

- source transaction verification: PASS
- package fingerprint: PASS
- package identity: PASS
- exported file hashes: PASS
- deterministic derivation: PASS
- scientific scope: PASS
- counts: PASS
- local-path leakage check: PASS
- binary/model/image artifact exclusion: PASS
- package idempotence: PASS

Synthetic tests additionally verify that a copied package remains valid after relocation and that modification of an exported file is detected.

## Scientific scope

- prediction semantics: per-nucleus latent continuous 53BP1 burden
- endpoint: sample-level arithmetic mean latent 53BP1 burden
- per-nucleus focus-count interpretation: false
- biological reference read: false
- biological acceptance evaluated: false
- KL720 hardware access performed: false

## Evidence freeze

External v0.7 evidence freeze SHA256:
d33cd5aaccab27ec48792d162c55391d3cfd961de880d2e34641127549e25a02

v0.6 external evidence freeze SHA256:
0caa9d2385d82adf78e702d7131f9a406235e36ed4ba1a366caca4714ed4c342

The external package, deployment artifacts, model assets, input data, transaction artifacts, and audit evidence remain outside Git.

## Interpretation

v0.7 demonstrates that a verified assay result can be exported into a compact portable package suitable for offline inspection, exchange, and collaborator-facing review while preserving explicit provenance bindings and scientific-scope warnings.

This is result-presentation and provenance-packaging evidence, not a new biological-validation, holdout-performance, or statistical-equivalence result.

The previously frozen full NASA physical KL720 seven-gate validation remains the authority for biological deployment performance.
