# NASA BPS R1 v2 contrast-aware training policy

Status: **frozen before final-holdout access**

This document defines the second learnable reference model for the NASA BPS 53BP1 case after R1 v1 was diagnosed and the expanded development image set passed QC.

The final holdout remains blinded. Nothing in this policy requires reading final-holdout images or phenotype outcomes.

## Motivation from R1 v1

R1 v1 demonstrated that aggregate-label learning can recover real radiation biology, but it was not sufficiently robust for deployment:

- OOF bag Pearson: 0.5997;
- OOF bag Spearman: 0.3419;
- matched radiation-effect direction agreement: 26/36 = 0.722;
- matched-effect Spearman: 0.6219;
- 4 h direction agreement: 12/12;
- peak-time recovery: 10/12 source x branch series;
- the female-held-out fold was the main generalization failure.

R1 v2 is therefore intended to improve biological fidelity without introducing spatial pseudo-labels or a more complex model unless necessary.

## Frozen development cohort

R1 v2 development uses the same six OSD Source Names and the same 12 matched Fe/X-ray condition cells as R1 v1:

- BALBCF1
- BALBCM1
- BALBCM2
- C57BLF1
- C57BLM1
- C57BLM3

Each Source Name x condition bag contains 100 deterministic nucleus crops, for:

- 72 sample/plate-well bags;
- 7,200 development nuclei;
- all 2,160 R1-v1 nuclei preserved;
- two rare >256-pixel geometry outliers replaced deterministically before training;
- final repaired development manifest SHA256:
  `23e21fdc8bf00a2814b64c56223cfd3d8e96eda71bdbff5d8ec74dae218cf5a9`.

Development QC1 passed 7,200/7,200 hard checks at native scale.

## Final holdout governance

The three Source Names never used for R1 v1 or R1 v2 development are:

- BALBCF2;
- C57BLF2;
- C57BLF3.

The blinded holdout manifest was frozen from metadata only with SHA256:

`40f015f7f8a8acb71289c1fe91218253e0b518c7e98f920511a5792dd5b6862d`

Its known metadata-only coverage is complementary rather than complete per source:

- BALBCF2: 10/12 conditions, 5/6 matched contrasts;
- C57BLF2: complete X-ray branch, 3/6 matched contrasts;
- C57BLF3: complete Fe branch, 3/6 matched contrasts.

R1 v2 development code must not read this manifest, holdout images, or holdout phenotype values. Holdout access is allowed only after the development-CV gate below is evaluated and the final evaluation procedure is frozen.

## Biological reference hierarchy

The primary supervision remains NASA `avg_nfoci` at sample/plate-well level.

NASA `avg_nfoci` is an aggregate over hundreds of nuclei and is not an individual-nucleus label. The 100 selected nucleus crops are an estimator of the matched sample-level phenotype.

Per-nucleus model outputs remain continuous expected 53BP1 burden values calibrated only through aggregate supervision. They must not be reported as validated discrete focus counts.

## Input policy

R1 v2 preserves the R1-v1 deployment input policy exactly:

1. FITC/53BP1 + DAPI only.
2. NASA MASK is QC metadata only and is not a deployable model input.
3. Native pixel scale is preserved.
4. No resizing is allowed.
5. Native crops are center-padded to 256 x 256.
6. FITC and DAPI are independently normalized on the native crop by p1/p99.5 and clipped to [0, 1].
7. Model tensor channels are FITC, DAPI, zero.
8. Padding is zero after normalization.
9. Training augmentation is restricted to 90-degree rotations and horizontal/vertical flips.

No blur, intensity jitter, elastic deformation, synthetic foci, random resizing, or channel mixing is allowed.

## Model policy

R1 v2 keeps the implemented R1-v1 backbone unchanged:

- six Conv2d -> BatchNorm2d -> ReLU stages;
- channels 24, 32, 48, 64, 96, 128;
- global/adaptive average pooling;
- linear scalar head;
- ReLU non-negativity transform;
- 215,833 trainable parameters.

This intentionally changes the supervision and bag sampling before changing architecture.

Historical note: the original R1-v1 policy text mentioned Softplus, but the actual frozen R1-v1 implementation used `nn.ReLU()` for the non-negativity transform. R1 v2 follows the implementation so architecture is held constant.

## Contrast-aware objective

For each training Source Name, branch and time point, a matched sham/exposed pair contains two 100-nucleus bags.

For bag `b`, let:

`y_hat_b = mean_i(z_i)`

where `z_i >= 0` is the model's continuous nucleus burden output.

For a matched pair with sham bag `s` and exposed bag `e`:

`L_abs = 0.5 * [SmoothL1(y_hat_s, y_s) + SmoothL1(y_hat_e, y_e)]`

`L_delta = SmoothL1((y_hat_e - y_hat_s), (y_e - y_s))`

with SmoothL1 beta = 0.5.

The frozen R1 v2 objective is:

`L_total = L_abs + 1.0 * L_delta`

Every epoch contains all matched training pairs exactly once in randomized order. Because each of the 12 condition bags belongs to exactly one branch/time matched pair, every training bag contributes once per epoch to the absolute component and every matched radiation contrast contributes once per epoch to the contrast component.

The contrast target uses the signed NASA difference, including negative late-time differences where present. No direction label is manually imposed.

## Development-CV split policy

The original three Source-Name-held-out folds are retained, but they are now explicitly **development CV**, because their R1-v1 outcomes have already been examined:

- `fold_female`: test BALBCF1 + C57BLF1;
- `fold_male_1`: test BALBCM1 + C57BLM1;
- `fold_male_2`: test BALBCM2 + C57BLM3.

For each fold:

- 4 Source Names / 48 bags / 24 matched contrasts are used for training;
- 2 Source Names / 24 bags / 12 matched contrasts are used for development-CV evaluation;
- no held-out Source Name enters training for that fold.

These folds may be used to decide whether R1 v2 is good enough to advance, but they are no longer described as untouched final validation.

## Frozen training schedule

To isolate the effect of larger bags and contrast-aware supervision, R1 v2 keeps the R1-v1 optimizer schedule:

- optimizer: AdamW;
- epochs: 60;
- learning rate: 3e-4;
- weight decay: 1e-4;
- cosine annealing to zero over 60 epochs;
- base random seed: 720;
- fold seeds: 720, 1720, 2720;
- no development-test early stopping;
- no fold-specific epoch selection;
- no final-holdout tuning.

The model trained at epoch 60 is the evaluated model for each fold.

## Predeclared development-CV gate

R1 v2 may proceed to a one-time final-holdout evaluation only if **all** of the following are met on six-source OOF development predictions:

1. overall bag Spearman >= 0.50;
2. weakest fold bag Spearman >= 0.25;
3. overall bag MAE <= 0.6654 (non-worse than R1 v1);
4. matched radiation-delta Spearman >= 0.70;
5. overall matched-effect direction agreement >= 30/36 = 0.8333;
6. 4 h matched-effect direction agreement = 12/12;
7. combined 24 h + 48 h direction agreement >= 17/24;
8. NASA 4 h peak-time recovery >= 10/12 source x branch series.

Failure of any criterion means the final holdout remains closed. Further development may use only the six development Source Names until a new model policy is frozen.

Passing this gate does **not** itself establish biological equivalence. It only authorizes freezing a final-holdout evaluation script and then opening the holdout once.

## Primary reporting

For six-source development CV, report:

- bag MAE/RMSE/Pearson/Spearman;
- metrics for each held-out-source fold;
- matched sham-to-exposed delta MAE/Pearson/Spearman;
- matched-effect direction agreement overall and by time point;
- 4/24/48 h repair-effect ordering and peak-time recovery;
- source-level results without treating nuclei as biological replicates.

## Interpretation guardrails

- Biological n is Source Name/sample structure, not 7,200 independent biological replicates.
- Fe-branch and X-ray-branch sham controls remain branch-specific.
- `particle_type` at 0 Gy is an experimental branch label.
- MASK is never required for deployed inference.
- The final public BPS subset contains only two strains; no broad unseen-strain claim is allowed.
- Per-nucleus outputs are latent continuous burden estimates, not validated integer counts.
- No INT8/KL720 equivalence claim is allowed until the chosen FP32 model passes biological validation and the same endpoints are compared after quantization and on the physical device.
