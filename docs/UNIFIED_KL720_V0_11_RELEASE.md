# Unified physical KL720 integration: v0.11 release evidence

**Evidence status: 2026-10-08.** This document records an offline software
integration milestone, not a new biological validation study or an end-to-end
hardware performance/energy benchmark. Large datasets, frozen binary artifacts,
and device SDK dependencies remain outside Git.

## Verified application boundary

The unified control plane supports two distinct NASA 53BP1 and DNA-fiber
assay adapters, retaining separate scientific reconstruction implementations.
For DNA-fiber, the supported physical route is:

```text
frozen preprocessed 512x512 deployment windows
  -> physical KL720 NEF inference (nine overlapping windows)
  -> stitched probabilities and assay-specific fiber reconstruction
  -> df2 physical field record -> dt2 measurement transaction
  -> drp1 portable result package -> independent package verification
```

The existing CPU-ONNX DNAi route remains `df1 -> dt1 -> drp1`. The `drp1`
portable-package format is unchanged; the exporter additionally records
source transaction schema and execution backend for physical KL720 sources.
The legacy v1 package derivation/verification is preserved.

## Frozen physical DNAi field

- Assay: `dnai-fiber-v3`; sample: `1__tile_4`; image index: `0`.
- KL720 field: `df2-d3955d46bdebf2c1`.
- KL720 transaction: `dt2-a8168c59e8a3e316`.
- Nine 512x512 model windows; 30 reconstructed foreground objects, 12 valid
  DNA fibers in the physical route.
- Physical valid-fiber mean ratio: `1.0339907040121177`; median ratio:
  `1.1058453569689006`; total valid fiber length: `186.21969101489583 um`.
- The source application performed physical KL720 inference, Gaussian
  stitching, and DNA-fiber object reconstruction. It did **not** perform raw
  microscopy preprocessing, human-annotation lookup, FP1024 lookup, or
  biological-fidelity evaluation.

The DNAi model NEF SHA256 is
`4b3dfec9a61c99e186dd4b8482fa5b06e6a4958f325ed4a0db0546f1dcab2bfc`.
The v0.11 physical-field verification code was frozen at
`c6247bd42944fb4d539850f80f2a84cc6de76327` before subsequent reporting
and CI changes.

## Deployment-difference audit (single field)

The frozen floating-FP512 field had 18 reconstructed objects and 14 valid
fibers. Comparison of its 1024x1024 stitched segmentation with the physical
KL720 reconstruction yielded:

| Comparison | Observed value |
| --- | ---: |
| Stitched-probability argmax agreement | 0.9992494583 |
| Final segmentation pixel accuracy | 0.9985227585 |
| Foreground Dice | 0.8653002775 |
| Foreground IoU | 0.7625808487 |
| Changed final segmentation pixels | 1,549 |
| FP512 / KL720 8-connected foreground components | 18 / 30 |
| KL720 components without FP512 component overlap | 16 |
| FP512 components without KL720 component overlap | 5 |
| Exact valid-fiber measurement signatures | 1 / 12 KL720 valid fibers |

These 8-connected components are **segmentation proxies**, not independently
verified DNA-fiber instance identities. An exact measurement signature is not
spatial proof that an object is identical between runtimes. High pixel accuracy
is dominated by background; foreground and instance differences are meaningful.
The physical mean/median valid-fiber ratios differed from floating FP512 by
approximately +2.0%/+2.4%, but the field is not evidence of object-level or
biological endpoint equivalence. No post-hoc tuning was performed against this
comparison.

## Verified physical export and tamper detection

- Portable result package: `drp1-d3c2bb8da0635736`.
- Frozen package manifest SHA256:
  `0f002061aaf85165d2c302018936b4f43c97a96d1776f7cb1f9b99eadcbea7f4`.
- Full physical-field freeze SHA256:
  `d8283d93332446f491962a82a06a045c4f41b9f59ca1f7571e5e35e582e6f499`.
- FP512/KL720 discrepancy audit SHA256:
  `793d9c809382115b1530d5d5dc1f15ede4bbe69d21a46a727017afdd56e4148c`.
- Physical source run manifest SHA256:
  `824d234f0143bcaad5f752e24d0612357c494d2feb32cbc913d11711017bbfc7`.

The package has `package_manifest.json`, `result.json`, `provenance.json`,
`valid_fibers.csv`, and `report.md`. Source transaction, source field, and CSV
hash bindings were checked at export. Public verification passed for both the
original package and a relocated copy. Modifying `valid_fibers.csv` in a
**disposable** copy made `files_ok=false` and `ok=false`; the immutable original
manifest and fiber CSV retained their original hashes.

Export **does not** rerun inference or validate biological agreement. A
standalone portable package can be verified without original absolute source
paths; regenerating the assay field requires separately obtained frozen input,
model, software environment, and relevant vendor runtime.

## Timing scope

The observed physical nine-window inference values sum to approximately
`1,034.4831 ms` (mean `114.9426 ms/window`); a separately recorded
postprocessing stage took approximately `9,026.8 ms`. These are measured
**phases**, not a fully synchronized end-to-end wall-clock result or an
energy/power benchmark. No claims about device energy reduction, system
throughput, or equivalence to GPU inference are made from these figures.

## CI and software verification

- Physical reporting bridge commit: `4983b54`.
- CI workflow commit: `7af193934f41d2f090206ac5e8f53a8b59f62ef5`.
- GitHub Actions run: https://github.com/liuifrec/radiation-edge-ai/actions/runs/37741150926
- Python 3.9 job: **131 passed**; Python 3.11 job: **131 passed**.
- Both jobs passed `pip check`, Python compilation, and Ruff
  `--select E4,E7,E9,F` across Python files changed from `main`.
- Hardware inference, model compilation, raw image processing, and access to
  private or external frozen source artifacts are **not** CI operations.
- The repository-wide default Ruff rule set has historical lint debt; the
  targeted changed-file gate is not a claim that every historical module
  satisfies all configured or optional lint rules.

## Scientific and release claim boundary

NASA BPS OSD-366 R1 v2 separately passed its original predeclared biological
fidelity criteria with a frozen physical holdout, as documented in
`docs/NASA_BPS_R1_V2_KL720_VALIDATION_RECORD.md`. This is not a claim of strict
numerical equivalence, broad cross-strain generalization, or supervised
per-nucleus focus counts. DNAi v0.11 instead establishes verified physical
execution, downstream reconstruction, provenance, export, and a descriptive
single-field FP512/KL720 discrepancy audit; **DNAi biological fidelity remains
unevaluated in this integration milestone**.

Micronucleus is future work and is not a prerequisite for the present
**two-assay offline methods-software demonstrator**. The package metadata
version in `pyproject.toml` is an independent Python distribution version;
`v0.11` here names an application/integration milestone rather than a published
PyPI release.
