# NASA BPS R1 v2 post-release evidence audit

Status: **completed read-only post-release audit, 2026-09-16**.

This record supplements rather than replaces the frozen NASA BPS R1 v2
validation record. No model weights, ONNX/BIE/NEF artifacts, predictions,
acceptance thresholds, biological gates, or RC1 historical result files were
modified during this audit.

Repository base audited:

- branch: audit/evidence-and-offline-runtime
- base commit: 39ce05f7fb7b81d8111bab79e52fc4683b1eab44

Audit evidence:

- post-release audit JSON SHA256: 8c715cb646158090b36ed67889e9e4c378e9a49fe291d1f8a21b5605ae049d6a

## Frozen deployment result

The physical KL720 deployment continues to **pass the original seven
predeclared biological-fidelity acceptance criteria**.

The audit does not convert those criteria into a formal statistical
equivalence test and does not justify a claim of strict numerical
equivalence.

## Per-contrast direction audit

The historical field described as an exact FP32 direction signature compared
aggregate agreement counts rather than literal element-by-element FP32
predicted signs.

The post-release audit joined all 11 FP32, BIE, and physical contrasts using
source, radiation branch, time point, sham sample identity, and exposed sample
identity.

Results:

- contrasts audited: 11
- FP32 versus physical literal predicted-sign identity: false
- FP32 versus BIE literal predicted-sign identity: false
- BIE versus physical predicted-sign identity: true for all 11 contrasts
- number of FP32-to-physical raw sign changes: 1
- original seven-gate decision changed: no

The single raw-sign change is:

- source: BALBCF2
- branch: X-ray
- time point: 24 h
- sham: BALBCF2_P282_A7_BALBC_FEMALE_1
- exposed: BALBCF2_P284_A7_BALBC_FEMALE_1
- NASA reference delta: 0.0
- FP32 predicted delta: -0.00954929819758632
- BIE predicted delta: +0.11925168720461565
- physical KL720 predicted delta: +0.11925168692347854

Because the NASA reference delta for this contrast is exactly zero, FP32, BIE,
and physical KL720 all fail direction agreement for this same contrast despite
the FP32-to-INT8 raw sign change.

Therefore the **per-contrast biological-reference agreement/failure pattern is
preserved**: the same 10 contrasts agree with the reference and the same
single zero-delta contrast does not.

Do not describe this as literal FP32 predicted-sign identity.

## Peak-series audit

Three complete 4/24/48 h source-by-radiation series were audited. The
historical 3/3 peak-recovery result remains unchanged.

## Quantization diagnostic

The maximum absolute physical-versus-BIE nucleus-level difference remains
approximately 0.03903937.

However, exact persisted output quantization radix/scale metadata were not
located unambiguously. Therefore the earlier suggestion that this difference
represented exactly one output quantization step remains a diagnostic
hypothesis only.

No output-code occupancy, clipping frequency, or difference-in-quantization-step
histogram is inferred from rounded console metadata.

## RC packaging coverage

All expected direct artifact-rehash keys were present in the RC manifest.

The RC manifest also records additional frozen identities, including the ONNX,
optimized ONNX, calibration manifest, calibration freeze, and Kneron toolchain
identity. Those additional identities were recorded separately rather than
directly rehashed by the artifact SHA map.

RC1 should therefore be described as a strong provenance index, not a
self-contained portable evidence archive.

## Claim boundary

Preferred wording:

> Physical KL720 passed the original seven predeclared biological-fidelity
> acceptance criteria and preserved the per-contrast agreement/failure pattern
> against the NASA aggregate biological reference.

Avoid:

- strict numerical equivalence;
- formal statistical equivalence;
- literal FP32 predicted-sign identity;
- interpretation of the approximately 0.039 difference as a proven output
  quantization step.

The final-holdout evaluation should continue to be described as a
**confirmatory final-holdout evaluation with a documented non-outcome
metadata-normalization implementation recovery**.

Additional frozen interpretation limits remain unchanged:

- per-nucleus model output is a latent continuous 53BP1 burden score, not an
  individually supervised focus count;
- biological reference avg_nfoci is an aggregate sample/plate-well endpoint;
- the all-female final holdout is a stress-test cohort, not a sex-effect study;
- the available public crosswalk does not support a 15-strain generalization
  claim;
- reported KL720 timing is device send-and-receive timing, not end-to-end
  microscopy workflow latency.
