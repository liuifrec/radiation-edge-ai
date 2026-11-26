# KL720 PoC integration plan

**Plan recorded 2026-11-24; source recovery updated 2026-11-26.** The [current status](PROJECT_STATUS.md) and
[evidence ledger](PROGRESS_EVIDENCE_20261124.md) supply the model dispositions
and sources. This document defines source integration and release work; it
does not change scientific gates or claim a new hardware run.

## What can be consolidated now

The progress reports can be incorporated into repository documentation with
their provenance and limitations. This makes the successful deployments,
failed candidates and next decisions discoverable in one place.

The later Windows implementation was recovered from the investigator's verified
Git bundle on November 26. The initial published integration base was
`4881cd914c50da03a30ca35b5be15a9c76db6a18`. The frozen Windows milestone is
`fa5047fd20c104de98f62ad39641a17fee1489dc` on
`codex/spotiflow-nef-output-representation`, with the local annotated tag
`poc-kl720-multimodel-2026-11-20`. Both recovered targets were verified; see the
[source recovery record](WINDOWS_POC_SOURCE_RECOVERY_20261126.md).

PR #35 merged at `0b0393c1adff5edc40c8e09c43a6ad0a73a77194`. It covers portable
calibration filenames for **new** exports and three trained Windows ONNX export
regressions. The six CI jobs on its head passed. [PR #35](https://github.com/liuifrec/radiation-edge-ai/pull/35)

### Recovered source and dependencies

The November 26 comparison used that merge and the frozen tag above. Their
common ancestor is `21042ddb3b2a7cf2a4d5d6aa8bff30289eaa5f36`. The trees
differ at 236 paths. Importing the entire older tree would remove newer desktop
and Perilipin2 code. The bundle's SHA256, 98 refs, tag object and peeled commit
were rechecked before extracting individual components.

| Source in the November 20 tree | Dependency and integration boundary |
| --- | --- |
| `hardware/pretrained_models.json`, `pretrained_models.py`, `spotiflow_transport.py` | The catalogue supplies checkpoint/variant identities and separates engineering and scientific decisions. The read-only CLI and desktop listing work without the SDK. Authenticating a NEF also needs upstream gates, the correct panel and the external receipt files. |
| `hardware/spotiflow.py`, `pretrained_infer.py`, `_kl720_worker.py`, resident and smoke launchers | Execution depends on the catalogue, original host spot decoder, per-head signed transport and exact input packing. Mocked USB verifies software contracts. Physical verification requires an explicit hardware run. |
| `nasa_bps/multimodel_ptq.py`, BIE inference/preparation/comparison modules and V1 protocols | Trained export is upstream of PTQ. Calibration is training-only; the numerical decision also requires the fixed-point engine, matched identities, all paired development predictions and the frozen margins. NASA R1 v2 keeps its dedicated physical runner. |
| DNAi replay, simulator probes, decoder isolation and activation diagnostics | Frozen controls and failed experiments have distinct roles. V3 is the physical fiber reference. Exploratory v4/v5, NASA bias probes and isolated Perilipin2 operators cannot authorize a new model. |

The capability matrix records the routes exposed in this checkout and the
evidence required to use each. [Model and assay status](PROJECT_STATUS.md#model-and-assay-status)

## 1. Source recovery completed; integrate reviewed changes

The supplied `radiation-edge-ai-local.bundle` passed Git verification with
complete history and 98 advertised refs. Its frozen branch and annotated tag
were imported into separate recovery refs and inspected in an isolated checkout.
The bundle, original tag object and milestone commit retain their identities.

Continue each integration from current `main`, inspecting ancestry, changed
files and prior merges. Reconcile overlapping fixes deliberately. Do not
blanket-push bundle refs or replay every experimental commit. Keep the recovered
history available for provenance and attribute each imported change. A later
integration release receives a new identity.

## 2. Integrate the qualified implementation

| Component | Integration requirement |
| --- | --- |
| Model catalogue | Bring over the existing local registry; distinguish task, weights, deployment and evaluation identities, with independent deployment/accuracy statuses |
| Shared KL720 transport | Reuse the tested packing and decoding implementation, including the supported higher-precision layouts and multiple outputs |
| Spotiflow `general` | Preserve released weights, the qualified BIE/NEF, native comparison and the corrected `.seq.16B` simulator interpretation; retain the original failed receipt |
| DNAi v3 | Retain the original physical reference and replay behavior; v4/v5 remain experimental |
| NASA R1 v2 | Preserve its original artifacts, post-release interpretation and one-upload batch behavior |
| NASA trained-model conversion | Portable paths for new exports are imported; next integrate compatible reading of frozen Windows manifests without rewriting their bytes |
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

The Perilipin2, GUI and Track A PRs are now merged. Their experimental model
dispositions remain unchanged. The next integration focuses on the recovered
registry, qualified transport and reference replay commands.
