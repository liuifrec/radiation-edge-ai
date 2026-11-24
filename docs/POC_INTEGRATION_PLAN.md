# KL720 PoC integration plan

**Plan recorded 2026-11-24.** The [current status](PROJECT_STATUS.md) and
[evidence ledger](PROGRESS_EVIDENCE_20261124.md) supply the model dispositions
and sources. This document defines source integration and release work; it
does not change scientific gates or claim a new hardware run.

## What can be consolidated now

The progress reports can be incorporated into repository documentation with
their provenance and limitations. This makes the successful deployments,
failed candidates and next decisions discoverable in one place.

The actual later Windows implementation is not present in the source available
for this consolidation. The inspected public base is
`561fed0b3572d19392f2471e0b274840ba468fcd`. The reported Windows milestone is
`fa5047fd20c104de98f62ad39641a17fee1489dc` on
`codex/spotiflow-nef-output-representation`, with the local annotated tag
`poc-kl720-multimodel-2026-11-20`. Its commit was absent from the available
checkout and the GitHub commit lookup on 2026-11-24.

Documentation consolidation therefore does not import the missing registry,
Spotiflow worker, PTQ readers or physical replay launchers. The code must be
recovered and inspected before those features can be integrated. References to
the Windows implementation in the ledger describe investigator reports.

## 1. Recover the source and establish ancestry

Use either the already reported verified `radiation-edge-ai-local.bundle` or a
single explicitly selected review branch pushed from the Windows repository.
The freeze-kit ZIP contains the freezing instructions; it is not the resulting
Git bundle or a copy of the model artifacts.

Verify the supplied bundle in a Git repository, inspect its refs, and confirm
that the milestone commit and tag target match the recorded identities. Import
the required history into an isolated checkout. Preserve the original tag and
commit; a later integration release receives a new identity.

Fetch the actual current `main` and start a new integration branch from it.
Inspect the common ancestor, commit history, changed files and prior merges.
Reconcile overlapping fixes deliberately. Do not blanket-push bundle refs or
blindly replay every experimental commit. Keep the recovered original history
available for provenance even when the integration requires a focused patch.

## 2. Integrate the qualified implementation

| Component | Integration requirement |
| --- | --- |
| Model catalogue | Bring over the existing local registry; distinguish task, weights, deployment and evaluation identities, with independent deployment/accuracy statuses |
| Shared KL720 transport | Reuse the tested packing and decoding implementation, including the supported higher-precision layouts and multiple outputs |
| Spotiflow `general` | Preserve released weights, the qualified BIE/NEF, native comparison and the corrected `.seq.16B` simulator interpretation; retain the original failed receipt |
| DNAi v3 | Retain the original physical reference and replay behavior; v4/v5 remain experimental |
| NASA R1 v2 | Preserve its original artifacts, post-release interpretation and one-upload batch behavior |
| NASA trained-model conversion | Include the portable calibration-path export fix and compatible reading of frozen Windows manifests without rewriting their bytes |
| NASA application batch path | Add a resident KL720 batch adapter using the established physical runner behavior; avoid one process/model upload per nucleus |
| Result reporting | Share provenance, timing and status fields while retaining assay-specific scores, points, masks and fiber measurements |

Public `main` currently has a CPU-only high-level NASA batch path and a generic
KL720 worker with INT8 input packing and one output. These are integration gaps,
not evidence that the investigator's later physical runs failed.

Review source and committed history for the intended release content and
third-party terms. Keep model binaries and microscopy data in their designated
external storage. Reference artifact identities and acquisition instructions;
personal host paths are not required to communicate the results.

## 3. Verify the integration at the appropriate layer

First run the offline and scientific software checks for the changed contracts.
Exercise model discovery without importing the device SDK, manifest and artifact
verification, Windows/Linux calibration paths, multi-output decoding and the
qualified input layouts. Reuse existing tests for the recovered implementation.

On the configured Windows host, replay the relevant frozen panels after the
shared runtime is integrated. Verify the same model and input identities before
comparing results. Use the existing acceptance criteria and preserve fresh
receipts for NASA R1 v2, DNAi v3 and Spotiflow `general`. No physical check is
assumed to have occurred as part of this documentation change.

Report actual neural backend, host preprocessing/postprocessing, model-load
time, device exchange and end-to-end timing separately. Keep simulator/native
agreement distinct from biological reference accuracy. A passed code suite
cannot replace a device or scientific result.

## 4. Make the release reproducible and recoverable

Test installation and documented artifact resolution in a clean environment.
Provide a discoverable model list and a consistent replay route, with clear
messages for unavailable assets or runtime dependencies. Confirm that the
deployed inference route does not require a CUDA installation or cloud service.

Verify an independent copy of the actual weights, BIEs, NEFs, replay inputs and
key receipts needed to reconstruct the milestone. The reported Git bundle
protects source history; hashing external artifacts in a manifest does not
copy them. Check hashes after copying and perform a bounded restore/replay
check from the backup location without altering the frozen originals.

Prepare a focused review of the integrated source, evidence crosswalk and
reproduction instructions. The integration release records its own commit,
artifact identities, software checks and physical replay receipts, while
linking back to the preserved November 20 milestone.

## 5. Resume model expansion with a bounded experiment

After the working PoC is reproducible, carry one eligible additional NASA model
through the complete deployment assessment. Select the checkpoint, fold,
training-only calibration, applicable criteria and experiment budget before
opening the new candidate comparison. MobileNetV2 is a reasonable candidate;
the male-1 export remains excluded until its original parity issue is resolved.

A single-fold success is partial development evidence. Complete the intended
fold coverage before reporting pooled physical out-of-fold performance. Keep
the historical holdout locked against new model selection and use genuinely
unused sources or acquisitions for independent confirmation.

The remaining Perilipin2, GUI and assay PRs are reviewed according to their
dependencies and current need. Their stacked branches and experimental results
do not all need to merge for the three-pipeline PoC to become usable.
