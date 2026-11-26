# Windows PoC source recovery

The investigator supplied the frozen Git bundle on **26 November 2026**.
The source can now be inspected directly and integrated into current main.
The [November 24 evidence ledger](PROGRESS_EVIDENCE_20261124.md) remains the
record of the earlier investigator reports; source recovery does not rerun
their hardware or biological measurements.

## Verified identities

| Item | Verified value |
| --- | --- |
| Bundle | `radiation-edge-ai-local.bundle` |
| Bundle size | 1,844,901 bytes |
| Bundle SHA256 | `dcf04b218fe5b63e0e97c5b1a66741df76bccc99458ff6c405c3b61fdba97e73` |
| Git verification | Valid bundle; complete history; 98 advertised refs |
| Frozen branch | `codex/spotiflow-nef-output-representation` |
| Frozen commit | `fa5047fd20c104de98f62ad39641a17fee1489dc` |
| Annotated tag | `poc-kl720-multimodel-2026-11-20` |
| Tag object | `d32f4c93c4ac51258db0634b4af2cd446eed08e5` |
| Peeled tag target | `fa5047fd20c104de98f62ad39641a17fee1489dc` |
| Integration base | `4881cd914c50da03a30ca35b5be15a9c76db6a18` |

The frozen branch and tag were fetched into separate recovery refs and checked
out independently. The bundle and frozen objects were preserved. Other bundle
branch names were not copied over the current repository's branches.

The integration base includes merged PRs #26 and #27. Its six CI jobs passed,
and its tree matches the previously tested combined checkout (484 passes,
34 skips). [Main CI](https://github.com/liuifrec/radiation-edge-ai/actions/runs/45304236354)

## First imported fix: portable NASA calibration filenames

The original Windows fix is in source commit
`2b057f8ed6e3d0487d85524c09f66be1ccc02d1e` and is retained in the frozen
milestone. This first integration imports only its exporter path change and
matching assertion from `tests/test_nasa_multimodel.py`.

`multimodel.export` now writes calibration filenames using
`relative_path.as_posix()`. A Windows export therefore records
`calibration_nchw/0000.npy`, which can also be resolved by the Linux toolchain.
Previously, `str(relative_path)` used Windows backslashes on that host.

The existing synthetic train/evaluate/export test checks exact manifest paths,
real saved arrays, strict PyTorch/ONNX parity and input identity checks for all
three architectures. CI now also runs that test inside the native Windows
Python 3.9 scientific environment and requires all three cases to execute with
no skips. Linux alone would not exercise the original Windows separator failure.
The focused local NASA multimodel suite passed all 14 tests; both imported
Python files match the frozen milestone byte-for-byte. Ruff, workflow structure
and local documentation-link checks also passed.

This changes new export manifests. Existing frozen manifests and their hashes
are not rewritten. The frozen implementation's backward-compatible reader
in `nasa_bps/multimodel_ptq.py` remains a separate integration item. No training
recipe, model weights, inference calculation or acceptance limit is changed.

## Remaining integration

The recovered source contains the pretrained registry, Spotiflow transport,
physical replay launchers and associated tests described in the Windows
reports. These components are not included in this first small change. Review
their dependencies and qualification checks together before enabling their
commands in main; see the [integration plan](POC_INTEGRATION_PLAN.md).

The external model binaries, image panels and raw execution receipts have not
been rehashed or replayed here. Their identities in recovered source are useful
provenance, but they do not constitute a fresh artifact or physical validation.
