# NASA BPS R1 v2 final-holdout policy

Status: **frozen before final-holdout image or phenotype access**

This document defines the confirmatory final-holdout procedure for the NASA BPS 53BP1 R1 v2 model. The six-source development-CV gate has passed all eight predeclared criteria, but the three final-holdout Source Names remain unopened for image/phenotype evaluation.

## Development gate result that authorizes this freeze

R1 v2 development used six Source Names, 72 bags, and 7,200 nuclei under the already frozen contrast-aware policy. The three source-held-out folds were treated as development CV rather than untouched final validation.

Observed six-source out-of-fold development performance:

- bag MAE: 0.5768;
- bag RMSE: 0.7783;
- bag Pearson: 0.8740;
- bag Spearman: 0.6528;
- matched radiation-delta Spearman: 0.7228;
- matched-effect direction agreement: 33/36 = 0.9167;
- 4 h direction agreement: 12/12;
- combined 24 h + 48 h direction agreement: 21/24;
- NASA 4 h peak-time recovery: 12/12 source x branch series;
- weakest fold bag Spearman: 0.6436.

All eight development criteria in `NASA_BPS_R1_V2_CONTRAST_POLICY.md` passed. This authorizes a one-time final-holdout evaluation sequence, but does not itself establish biological equivalence.

## Frozen data identities

Final repaired six-source development manifest SHA256:

`23e21fdc8bf00a2814b64c56223cfd3d8e96eda71bdbff5d8ec74dae218cf5a9`

Frozen 72-sample development aggregate-reference CSV SHA256:

`fcbf1339805e02a7ca9cec36b476438b0512b222bd70ac2f90ed75c238a0b01f`

Frozen NASA raw phenotype table SHA256:

`d28dc5d9c53221c66a90074a92783fd4708ea83db0fc1a9c767e47500e643d6a`

Blinded final-holdout manifest SHA256:

`40f015f7f8a8acb71289c1fe91218253e0b518c7e98f920511a5792dd5b6862d`

The final holdout Source Names are:

- `BALBCF2`;
- `C57BLF2`;
- `C57BLF3`.

All three are female sources and none was used for R1 v1 or R1 v2 development.

Known metadata-only coverage, frozen before phenotype access:

- BALBCF2: 10/12 conditions, 5/6 matched contrasts, 858 selected nuclei;
- C57BLF2: 6/12 conditions, complete X-ray branch, 3 matched contrasts, 600 selected nuclei;
- C57BLF3: 6/12 conditions, complete Fe branch, 3 matched contrasts, 600 selected nuclei;
- total: 2,058 selected nuclei, 22 available sample/condition bags, 11 complete matched sham/exposed contrasts.

One or more available cells contain fewer than 100 selected nuclei because fewer than 100 eligible nuclei exist. The frozen holdout bag sizes are used as-is; no phenotype-dependent resampling is allowed.

## Step 1 - train one final FP32 candidate on all development sources

Before any final-holdout images or phenotypes are accessed, train one final R1 v2 candidate using all six development Source Names.

The final candidate keeps the frozen R1 v2 model and input policy exactly:

- 215,833-parameter `R1CountNetV1` backbone;
- FITC + DAPI + zero channel;
- native pixel scale;
- no resize;
- center-pad to 256 x 256;
- p1/p99.5 per-image normalization;
- geometry-only rotations/flips;
- MASK excluded from model input.

The final candidate training objective is unchanged:

`L_total = L_abs + 1.0 * L_delta`

with SmoothL1 beta = 0.5.

All six development sources contribute all 36 matched sham/exposed pairs once per epoch. The training schedule remains:

- AdamW;
- 60 epochs;
- learning rate 3e-4;
- weight decay 1e-4;
- cosine annealing to zero;
- seed 720;
- no early stopping;
- epoch-60 checkpoint is the final candidate.

The final checkpoint SHA256 must be recorded before any holdout image access. No ensemble of development-CV fold models is used for the confirmatory final holdout; the confirmatory model is the single all-development final candidate.

## Step 2 - open holdout images only, still without phenotype outcomes

After the final candidate checkpoint is frozen, the blinded holdout image set may be materialized exactly from the frozen holdout manifest.

At this stage:

- phenotype rows for the holdout Sample Names must still not be joined or inspected;
- FITC/DAPI/MASK images may be downloaded only for the 2,058 frozen holdout nuclei;
- the same hard QC as development is applied: file decode, channel shape agreement, uint16 expectations, target-label presence, center-label agreement, no target-border touch, finite normalization range, native geometry <=256 x 256.

MASK remains QC-only and is not passed to the model.

If a selected holdout nucleus fails **only** because native geometry exceeds 256 x 256, a deterministic replacement is allowed before phenotype unblinding, using the same Source Name x condition, the already frozen holdout ranking seed, and the first previously unselected candidate that passes the identical hard QC. Any replacement must be phenotype-blind, audited, and produce a new frozen holdout-QC manifest SHA256.

Any non-geometry hard failure, inability to form a required matched contrast, or phenotype-dependent replacement is a protocol deviation and stops confirmatory evaluation until documented.

## Step 3 - freeze the one-time evaluator and unblind phenotype outcomes once

After the final checkpoint SHA256 and the QC-passed holdout manifest SHA256 are known, freeze the final evaluation script with those exact hashes.

Only then may the evaluator join the holdout Sample Names to the already frozen NASA raw phenotype table by exact Sample Name and read the corresponding `avg_nfoci` values.

The model must not be retrained, recalibrated, thresholded, or otherwise modified after holdout outcomes are opened.

## Holdout evaluation units

Primary machine-learning unit:

- available sample/plate-well bag (22 expected bags before any geometry-only replacement).

Primary biological contrasts:

- available matched exposed-minus-sham differences within the same Source Name, radiation branch, and time point (11 expected contrasts).

Biological grouping remains Source Name. The 2,058 nuclei are not independent biological replicates.

The expected complete matched-contrast structure is:

- BALBCF2: Fe 4/24/48 h and X-ray 4/24 h = 5 contrasts;
- C57BLF2: X-ray 4/24/48 h = 3 contrasts;
- C57BLF3: Fe 4/24/48 h = 3 contrasts.

Thus the holdout contains:

- four 4 h matched contrasts;
- four 24 h matched contrasts;
- three 48 h matched contrasts;
- three complete 4/24/48 source x branch series for peak-time assessment.

## Predeclared final-holdout confirmation gate

R1 v2 is authorized to advance to ONNX/INT8/KL720 deployment only if **all** of the following confirmatory criteria are met on the one-time final holdout:

1. available-bag Spearman correlation with NASA `avg_nfoci` >= 0.50;
2. matched radiation-delta Spearman >= 0.60 across the 11 available contrasts;
3. overall matched-effect direction agreement >= 9/11;
4. 4 h matched-effect direction agreement = 4/4;
5. combined 24 h + 48 h direction agreement >= 5/7;
6. 4 h peak-time recovery = 3/3 among the three complete 4/24/48 source x branch series;
7. no holdout Source Name has matched-effect direction agreement below 2/3 (therefore BALBCF2 must be at least 4/5 and each C57BL source at least 2/3).

Bag MAE/RMSE/Pearson and per-source residuals are mandatory descriptive outputs but are not additional pass/fail criteria because the holdout has unequal available bag sizes, including a rare cell with fewer than 100 nuclei, and because the primary manuscript claim is preservation of radiation-effect biology rather than absolute regression calibration alone.

Passing the final-holdout gate means the FP32 R1 v2 candidate is sufficiently biologically faithful to become the fixed deployment reference. It does **not** yet establish KL720 equivalence.

Failing any criterion means the holdout result is reported as a failed confirmatory evaluation. The same holdout cannot be reused as an untouched tuning set. Any later R1 v3 development must be explicitly labeled post-holdout and requires a new validation strategy.

## Mandatory final-holdout outputs

The final evaluator must write at least:

- bag-level predictions and NASA targets;
- per-nucleus latent burden predictions, clearly labeled as not individually supervised;
- 11 matched radiation deltas;
- bag MAE/RMSE/Pearson/Spearman;
- delta MAE/Pearson/Spearman;
- direction agreement overall, by time point, by branch, and by Source Name;
- complete-series peak-time recovery;
- the seven predeclared gate criteria with PASS/FAIL;
- exact development-manifest, holdout-manifest, phenotype-table, and final-checkpoint SHA256 values;
- an explicit statement that no holdout outcome was used for tuning.

## After a successful final holdout

If all confirmatory criteria pass, the exact frozen FP32 final checkpoint becomes the reference for:

FP32 -> ONNX parity -> Kneron optimization/calibration -> INT8/BIE -> NEF -> physical KL720 inference.

The same final holdout bags and biological endpoints may then be reused **only for deployment-equivalence comparison** between the already frozen FP32 reference and its derived ONNX/INT8/device forms. They must not be used to tune the FP32 model.

The deployment claim is accepted only if quantized/device inference preserves the FP32-held-out biological conclusions within a separately frozen equivalence policy.

## Interpretation guardrails

- NASA `avg_nfoci` is sample/plate-well aggregate supervision, not per-nucleus ground truth.
- Per-nucleus model outputs remain continuous burden scores, not validated discrete foci counts.
- Zero-Gy Fe/X-ray labels are experimental branch labels; branch-specific shams are not pooled casually.
- The public subset contains only two strains; no broad unseen-strain generalization claim is allowed.
- All three final-holdout sources are female, making this a particularly useful check of the R1-v1 female-generalization weakness, but not a general sex-effect study.
- No final-holdout criterion may be changed after phenotype outcomes are opened.
