# NASA BPS R1 v2 final-candidate BatchNorm repair policy

Status: **frozen before final-holdout image or phenotype access**

This document records a development-only deployment-state repair for the NASA BPS R1 v2 all-development FP32 candidate. It is an amendment to `NASA_BPS_R1_V2_FINAL_HOLDOUT_POLICY.md` and does not change the model architecture, learned weights, training data, training objective, or final-holdout confirmation gate.

## Trigger for this amendment

The frozen all-development candidate checkpoint was created before any final-holdout access:

`6ee614030df15a44c616fa0229c7c23399ba59b2bc017eed103c7b21531d5f58`

Training reached a low epoch-60 aggregate loss, but post-training development inference in `model.eval()` mode was unexpectedly poor:

- development-fit bag MAE: 2.9700;
- bag Spearman: 0.4940;
- matched-delta Spearman: 0.7586.

Because the compact `R1CountNetV1` contains BatchNorm layers, a development-only diagnostic was run on the exact frozen checkpoint without modifying it and without reading any final-holdout data.

Observed diagnostic behavior:

- frozen eval-mode: bag MAE 2.9700, bag Spearman 0.4940, delta Spearman 0.7586, direction agreement 0.833;
- disposable train-mode/batch-stat inference: bag MAE 0.1614, bag Spearman 0.9117, delta Spearman 0.9393, direction agreement 0.944;
- disposable deterministic BatchNorm-recalibrated eval-mode: bag MAE 0.5995, bag Spearman 0.8097, delta Spearman 0.8275, direction agreement 0.944.

This pattern supports a BatchNorm running-statistics state mismatch rather than failure of the learned convolutional/linear parameters.

## Allowed repair

The repair is restricted to **BatchNorm running state only**.

The exact original checkpoint is loaded, and all learned parameters are preserved bit-for-bit:

- convolution weights/biases;
- BatchNorm affine parameters (`weight`, `bias`);
- final linear-head parameters;
- every other trainable tensor.

Only the following non-trainable BatchNorm state entries may change:

- `running_mean`;
- `running_var`;
- `num_batches_tracked`.

No gradient updates, optimizer steps, epoch selection, loss reweighting, threshold tuning, architecture changes, or phenotype-dependent image selection are allowed.

## Frozen BatchNorm recalibration procedure

Recalibration uses only the already frozen six-source R1 v2 development set:

- repaired development manifest SHA256: `23e21fdc8bf00a2814b64c56223cfd3d8e96eda71bdbff5d8ec74dae218cf5a9`;
- 72 development bags;
- 100 nuclei per bag;
- 7,200 nuclei total;
- FITC + DAPI + zero-channel input;
- native scale, no resizing, center-pad to 256 x 256;
- the same p1/p99.5 normalization as R1 v2;
- **no augmentation** during recalibration.

To avoid condition-homogeneous BatchNorm updates, nuclei are deterministically interleaved by within-bag rank across all 72 sample bags:

1. sort bags by exact Sample Name;
2. take nucleus rank 0 from all 72 bags, then rank 1 from all 72 bags, and so on through rank 99;
3. process this 7,200-nucleus sequence in fixed batches of 100;
4. reset BatchNorm running statistics before recalibration;
5. set BatchNorm momentum to `None` so PyTorch uses a cumulative moving average across recalibration batches;
6. run forward passes under `torch.no_grad()` with the model in training mode only to update BatchNorm running state;
7. restore the original BatchNorm momentum settings;
8. switch the repaired model to `eval()` for all subsequent inference.

No final-holdout manifest, images, or phenotypes may be read during this procedure.

## Serialization and provenance

The repaired checkpoint must be written to a new file; the original checkpoint must never be overwritten.

The repaired checkpoint must record at least:

- parent checkpoint SHA256;
- repaired development-manifest SHA256;
- the deterministic BatchNorm recalibration rule;
- calibration nuclei = 7,200;
- mixed batch size = 100;
- number of BatchNorm layers;
- statement that learned parameters were unchanged;
- statement that final holdout was not accessed.

The new repaired-checkpoint SHA256 becomes the only FP32 candidate identity eligible for final-holdout evaluation and later ONNX/INT8/KL720 conversion, provided the repair audit below passes.

## Repair audit

Immediately after serialization, the repaired checkpoint must be reloaded and evaluated in `model.eval()` mode on the six-source development set.

This audit is not a new model-selection gate. It verifies that the serialized checkpoint reproduces the already observed disposable deterministic BN-recalibrated behavior.

Required checks:

1. the original parent checkpoint SHA256 exactly matches the frozen value above;
2. all non-BatchNorm-running-state tensors are bit-identical to the parent checkpoint;
3. at least one BatchNorm running-state tensor changes, confirming that a state repair was actually applied;
4. the reloaded repaired checkpoint reproduces the in-memory repaired predictions/metrics to numerical tolerance;
5. the repair remains development-only and records `final_holdout_read = false`.

If any check fails, final-holdout access remains closed.

## Final-holdout governance after repair

Only after the repaired checkpoint is successfully serialized, hashed, and audited may its SHA256 be pinned into the phenotype-blind holdout materialization/QC code.

The remainder of `NASA_BPS_R1_V2_FINAL_HOLDOUT_POLICY.md` is unchanged:

1. materialize exactly the frozen final-holdout images;
2. perform phenotype-blind hard QC and geometry-only repair if required;
3. freeze the QC-passed holdout manifest SHA256;
4. freeze the one-time evaluator with exact repaired-checkpoint and holdout-manifest hashes;
5. only then join NASA `avg_nfoci` and run the one-time confirmatory biological-fidelity test.

The final-holdout confirmation criteria themselves are **not changed** by this BatchNorm repair.

## Interpretation guardrail

This repair addresses deployment-state calibration of BatchNorm only. It must not be described as retraining on the holdout, post-holdout tuning, or evidence of KL720 equivalence. The biological claim remains contingent on the untouched final-holdout test and then on separate FP32-to-INT8/device biological-equivalence validation.
