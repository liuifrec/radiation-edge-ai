# NASA BPS R1 weak-supervision policy

Status: **frozen for pilot-v1 baseline training**

This document defines the first learnable reference model (R1) for the NASA BPS 53BP1 case after the sample-level phenotype hierarchy was resolved and frozen.

## Biological reference hierarchy

The public microscopy pilot contains 2,160 individual nucleus crops arranged as 72 sample/plate-well bags:

- 6 OSD Source Names;
- 12 radiation-condition cells per source;
- 30 nucleus crops per source x condition bag.

NASA `LSDS-111_immunostaining_Raw_pheno_V3.csv` provides one aggregate phenotype row per matched sample/plate-well. For the frozen pilot, all 72 Sample Names match uniquely and all 2,160 nucleus crops map to one of those 72 aggregate rows.

The primary NASA target is:

- `avg_nfoci`: mean 53BP1 focus count across the full NASA sample population.

Secondary reference fields are retained for evaluation only, especially:

- `avg_foci_no_outl`;
- `std_nfoci`;
- `num_nuc`;
- `avg_fitc_bg`;
- `spot_fitc_sum`.

The NASA aggregate reference was computed from 249-795 nuclei per matched sample in pilot-v1 (median 694.5), whereas the frozen training bag contains 30 nucleus images. Therefore **the NASA sample mean must never be copied onto an individual nucleus as if it were a per-nucleus ground-truth count**.

## R1 v1 learning formulation

R1 v1 is a multiple-instance / learning-from-aggregate-labels regression baseline.

For nucleus image `i` in sample bag `b`, the compact CNN outputs a non-negative continuous burden estimate:

`z_i >= 0`.

The model prediction for the bag is the arithmetic mean of its 30 nucleus predictions:

`y_hat_b = mean_i(z_i)`.

The supervised loss is applied **only at bag level** against NASA `avg_nfoci`:

`L_b = SmoothL1(y_hat_b, avg_nfoci_b)`.

Every sample bag receives equal weight. `num_nuc` is retained as reference metadata but is not used to replicate or up-weight individual nuclei.

The per-nucleus output is interpreted as a **continuous expected 53BP1 burden calibrated through sample-level mean focus count**. It is not claimed to be a validated integer focus count until an independent per-focus/per-nucleus annotation set is introduced.

## Input policy

Frozen pilot-v1 R1 input policy:

1. FITC/53BP1 and DAPI channels only.
2. The supplied NASA instance mask is **not a model input**.
3. Native microscopy pixel scale is preserved.
4. No geometric resizing is allowed.
5. Each native crop is center-padded to 256 x 256.
6. Each channel is robustly normalized independently on the native, unpadded crop using p1 and p99.5, clipped to [0, 1].
7. For the first KL720-compatible baseline, the network receives a 3-channel tensor packed as:
   - channel 0: normalized FITC;
   - channel 1: normalized DAPI;
   - channel 2: zeros.
8. Padding is zero-valued after normalization.

The third zero channel is a deployment-format convenience, not a biological signal.

The adaptive per-image normalization is intentional. Pilot QC demonstrated strong acquisition/background intensity shifts, including a large Fe-branch 0-Gy 48-h brightness effect. R1 v1 is therefore designed to learn punctate/morphological signal rather than raw absolute fluorescence intensity.

## Augmentation policy

Only geometry-preserving augmentations that do not alter native scale are allowed in R1 v1:

- rotations by 0/90/180/270 degrees;
- horizontal flip;
- vertical flip.

No random resizing, elastic deformation, blur, synthetic spot insertion, intensity jitter, or channel mixing is used in the frozen baseline.

## Compact model policy

R1 v1 uses a deliberately simple KL720-friendly CNN composed only of common deployment operators:

- Conv2d;
- BatchNorm2d;
- ReLU;
- adaptive/global average pooling;
- linear output head;
- Softplus non-negativity transform.

The first baseline is a scalar burden regressor, not yet a spatial focus-map model. This is deliberate: the available NASA supervision is sample-level aggregate `avg_nfoci`, and forcing spatial pseudo-labels before an aggregate baseline is validated would introduce an unnecessary assumption.

A weakly supervised density/focus-map model becomes R1 v2 only if R1 v1 demonstrates source-held-out biological signal and if spatial interpretability is needed for the final assay endpoint.

## Split policy

The already frozen three source-held-out outer folds remain authoritative:

- `fold_female`: test BALBCF1 + C57BLF1;
- `fold_male_1`: test BALBCM1 + C57BLM1;
- `fold_male_2`: test BALBCM2 + C57BLM3.

Each test fold contains one BALB/cByJ and one C57BL/6J source. Nuclei from a held-out Source Name never appear in training for that fold.

R1 v1 uses a fixed training schedule and no test-set early stopping or test-driven hyperparameter search. Any later tuning must be performed without using the outer held-out sources for model selection.

## Primary evaluation level

The primary machine-learning evaluation unit is the **sample bag**, not the nucleus.

Report for source-held-out out-of-fold predictions:

- bag-level MAE against NASA `avg_nfoci`;
- bag-level RMSE;
- Pearson correlation;
- Spearman correlation;
- secondary agreement with `avg_foci_no_outl`.

Nucleus-level predictions may be inspected for distributional plausibility but are not treated as individually labeled observations.

## Biological-fidelity evaluation

For each held-out Source Name, radiation branch and time point, compare the model-derived bag mean between matched sham and irradiated samples:

- Fe branch: 0.82 Gy - 0 Gy at 4, 24 and 48 h;
- X-ray branch: 1.0 Gy - 0 Gy at 4, 24 and 48 h.

The R1 baseline should preserve, at minimum:

- the direction of matched irradiated-vs-sham effects;
- the strong early 4-h response;
- the decline in damage burden across the repair interval;
- branch-specific effects without pooling Fe-branch and X-ray-branch sham controls.

Because the frozen public BPS subset contains only two strains, strain comparisons are descriptive/background effects and are **not** evidence of broad unseen-strain generalization.

## Interpretation guardrails

- `particle_type` at 0 Gy is an experimental branch label, not a physical irradiation exposure.
- Fe-branch and X-ray-branch 0-Gy samples are not automatically pooled.
- NASA `avg_nfoci` is a sample/plate-well aggregate phenotype.
- The 30 frozen nucleus crops are a deterministic subset used to estimate the sample-level phenotype; they are not 30 biological replicates.
- The biological grouping unit remains OSD Source Name.
- R1 v1 produces a continuous burden estimate, not a validated discrete per-nucleus foci count.
- Raw absolute FITC intensity is not the primary endpoint because pilot QC demonstrated strong batch/acquisition effects.
- KL720 equivalence is not claimed until the FP32 reference, INT8 model and physical device reproduce the same held-out biological conclusions.

## Frozen source hashes

Pilot manifest SHA256:

`41ff2e4d64577f17c976dea5bbc17221fe63be9e2e7e2291309b520027fbe725`

NASA Raw phenotype table SHA256:

`d28dc5d9c53221c66a90074a92783fd4708ea83db0fc1a9c767e47500e643d6a`
