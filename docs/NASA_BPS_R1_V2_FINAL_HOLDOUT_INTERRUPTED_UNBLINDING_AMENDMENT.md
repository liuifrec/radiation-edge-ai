# NASA BPS R1 v2 interrupted final-holdout unblinding amendment

Status: **documented protocol deviation after unblinding started, before confirmatory metrics were computed**

## Incident

The frozen one-time evaluator passed its phenotype-blind preflight with the exact repaired FP32 checkpoint, QC1 holdout manifest, 2,058/2,058 hard-QC pass, and finite 2,058-nucleus -> 22-bag inference.

The explicit `--unblind` execution then:

1. reverified the same checkpoint and manifest identities;
2. reran 2,058/2,058 hard image QC;
3. reran phenotype-blind inference;
4. wrote `FINAL_HOLDOUT_UNBLINDING_STARTED.json`;
5. opened and hash-verified the frozen raw NASA phenotype table;
6. parsed the raw table;
7. failed on the first holdout bag during metadata validation, before a target value was dereferenced and before any bag/delta/gate metric was computed.

Observed exception:

`Phenotype metadata mismatch for BALBCF2_P242_C7_BALBC_FEMALE_1: strain='balb/cbyj' != Strain='balbc'`

No confirmatory prediction/reference CSV, confirmation summary, or confirmation-complete marker was produced by the failed execution.

## Diagnosis

This is an evaluator implementation omission, not a newly discovered biological inconsistency.

Before R1 v2 training, holdout image materialization, or final-holdout evaluation, repository commit

`6e8791061563d221915fda9567be37bbd8e29264` — `Validate OSD-366 reference metadata aliases`

had already established from the full OSD-366 ISA-to-LSDS-111 metadata join that the raw phenotype table uses deterministic shortened controlled-vocabulary aliases, including:

- `BALB/cByJ -> BALBC`
- `C57BL/6J -> C57`
- ISA/BPS `Fe` corresponding to raw `Fe 600 MeV/n`
- X-ray naming variants

The subsequent reference-freeze commit

`3141f1e168ffe100d177fbc59b8e8c2772853b71` — `Freeze NASA BPS pilot v1 aggregate reference with branch-aware sham semantics`

codified those mappings and also used punctuation-insensitive plate/well normalization and branch-aware radiation normalization.

The frozen final-holdout evaluator accidentally reverted these metadata checks to stricter generic normalizers. The holdout failure therefore reproduces a metadata vocabulary discrepancy that was identified, validated, and resolved before R1 v2 development.

## Scientific-integrity classification

The original frozen evaluator is **not modified**.

Because the raw phenotype table had already been opened when the failure occurred, the final result can no longer be described as a perfectly pristine uninterrupted one-time execution. Any completed evaluation must be labeled:

**confirmatory final-holdout evaluation with a documented non-outcome metadata-normalization implementation recovery**.

This recovery is considered scientifically interpretable only because:

- the alias rules predate R1 v2 training and final-holdout evaluation;
- the model checkpoint is unchanged;
- BatchNorm state is unchanged;
- the QC1 holdout membership is unchanged;
- image preprocessing and model predictions are unchanged;
- the primary endpoint remains NASA sample/plate-well aggregate `avg_nfoci`;
- all seven predeclared pass/fail criteria are unchanged;
- no threshold, resampling, model selection, or tuning is permitted;
- the recovery first validates all 22 Sample Name/metadata joins using only the pre-existing alias policy, then dereferences `avg_nfoci`;
- a failure of any remaining metadata check stops the recovery rather than expanding the alias policy.

## Frozen original identities

Original evaluator runtime/file SHA256:

`0bb389434ef3d9b64ff5d5bd29761765a6a53ec905194ec88cefaef6c698a1ca`

Authoritative repaired FP32 checkpoint SHA256:

`2b07c67d6e50e7a36e26e0052feebc826b518e6efae863ba312ba9761a4eb3d5`

QC1 repaired holdout manifest SHA256:

`f89962f96bec05930c07149717a90bd682016a2778bbfae6b827425159aa0643`

Frozen raw phenotype SHA256:

`d28dc5d9c53221c66a90074a92783fd4708ea83db0fc1a9c767e47500e643d6a`

## Recovery evaluator freeze

Recovery evaluator:

`src/radiation_edge_ai/nasa_bps/resume_r1_v2_final_holdout_after_metadata_alias_bug.py`

Introduced at commit:

`3faaa32621576041cb8b2cd497bc1377f084bb25`

Frozen recovery evaluator Git blob:

`c7b0c4ad830c59a11d675506b665295960900fd0`

The recovery evaluator pins the original evaluator runtime SHA and all three scientific input hashes. It requires the existing interrupted-unblinding marker and refuses to run if a complete result already exists.

Default mode is **recovery audit only**. It does not reopen the raw phenotype table. It rechecks the original evaluator source identity, checkpoint identity, QC1 manifest identity, 2,058-image hard QC, and frozen inference. Its runtime source SHA256 must be recorded before the explicit recovery resume.

Only the unchanged recovery evaluator may then be rerun with `--resume-confirmation`.

## Recovery semantics

On resume, the recovery evaluator must:

1. require the original `FINAL_HOLDOUT_UNBLINDING_STARTED.json` marker;
2. require the exact original evaluator runtime SHA256;
3. require the exact repaired FP32 and QC1 holdout hashes;
4. require no prior confirmation-complete/recovery-complete outputs;
5. rerun frozen hard QC and inference;
6. verify the raw phenotype SHA256;
7. join all 22 holdout bags by exact Sample Name;
8. validate source, sex, dose and time with the original frozen semantics;
9. validate strain with only `BALB/cByJ -> BALBC` and `C57BL/6J -> C57`;
10. validate radiation with the already-established branch-aware Fe/X-ray aliases;
11. validate plate/well with the already-established punctuation-insensitive normalizer;
12. stop if any metadata discrepancy remains;
13. only after all 22 metadata joins pass, dereference `avg_nfoci` and `num_nuc`;
14. compute the unchanged 22-bag / 11-contrast metrics and seven predeclared gates;
15. write the same scientific outputs plus explicit protocol-deviation provenance;
16. permanently block another recovery after completion.

## Interpretation rule

If all seven gates pass, the model may proceed to ONNX/INT8/KL720 equivalence testing, but all reporting must carry the documented metadata-normalization recovery note.

If any gate fails, R1 v2 fails the confirmatory biological-fidelity evaluation. The holdout must not be reused as an untouched tuning set.
