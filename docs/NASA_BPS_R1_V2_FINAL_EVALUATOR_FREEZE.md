# NASA BPS R1 v2 one-time final-holdout evaluator freeze

Status: **frozen after phenotype-blind holdout QC/geometry repair and before phenotype unblinding**

This document freezes the exact inputs, evaluator source identity, confirmatory endpoints, and decision rule for the one-time NASA BPS R1 v2 FP32 final-holdout evaluation.

## Phenotype-blind QC result

The original 2,058-nucleus holdout underwent the same hard image QC as development while NASA phenotype outcomes remained blinded.

Observed initial QC:

- 2,057/2,058 nuclei passed all hard checks;
- exactly one geometry-only failure;
- zero non-geometry hard failures;
- failed frozen slot: `BPSR1V2H_01219`;
- failed nucleus: `P280_73668439105-E5_003_019`;
- native geometry: `277 x 228`, exceeding the frozen `256 x 256` native-scale canvas.

The predeclared geometry-only repair rule was then applied using the frozen holdout ranking seed. The first previously unselected candidate in the same Source Name x Sample Name x radiation condition passed the identical hard QC:

`P280_73668439105-E5_003_019 -> P280_73668439105-E5_002_024`

Candidate attempts required: `1`.

The repaired manifest then passed identical hard QC for `2058/2058` nuclei. NASA phenotype/reference values were not read during QC or repair.

## Exact frozen input identities

Authoritative BN-repaired FP32 checkpoint SHA256:

`2b07c67d6e50e7a36e26e0052feebc826b518e6efae863ba312ba9761a4eb3d5`

QC1 repaired final-holdout manifest SHA256:

`f89962f96bec05930c07149717a90bd682016a2778bbfae6b827425159aa0643`

Frozen raw NASA phenotype-table SHA256:

`d28dc5d9c53221c66a90074a92783fd4708ea83db0fc1a9c767e47500e643d6a`

The raw phenotype table remains unopened for holdout outcomes at the time of this freeze.

## Frozen evaluator source

Evaluator:

`src/radiation_edge_ai/nasa_bps/evaluate_r1_v2_final_holdout_once.py`

Evaluator source was introduced at commit:

`fb697cc434f8ec2cf5b8ad6cd2f433b341a6ce65`

Frozen evaluator Git blob identity:

`0e39b789505f92aea1d51b26311a86bc0a0f8c69`

The evaluator computes and prints its own ordinary file SHA256 at runtime. The phenotype-blind preflight must be run first; its reported evaluator-source SHA256 is to be recorded in the final audit before the explicit `--unblind` execution. No evaluator source changes are permitted between that preflight and unblinding.

## Frozen holdout structure

- nuclei: 2,058;
- sample/plate-well bags: 22;
- biological sources: `BALBCF2`, `C57BLF2`, `C57BLF3`;
- matched exposed-minus-sham contrasts: 11;
- BALBCF2: 5 contrasts;
- C57BLF2: 3 contrasts;
- C57BLF3: 3 contrasts;
- 4 h contrasts: 4;
- 24 h + 48 h contrasts: 7;
- complete 4/24/48 Source Name x branch series: 3.

All three holdout sources are female. This is a stress test of the previously weak female-held-out generalization, not a sex-effect study.

## Frozen evaluation semantics

Primary NASA reference endpoint:

`avg_nfoci`

This is a sample/plate-well aggregate over many nuclei, not an individual-nucleus label.

The model's per-nucleus scalar remains a continuous latent 53BP1 burden score. It must not be described as a validated discrete focus count. Bag prediction is the arithmetic mean of the frozen selected nuclei for that sample/plate-well bag.

Biological replication is defined by Source Name/sample structure. The 2,058 nuclei are not treated as independent biological replicates.

Fe and X-ray branch-specific 0 Gy controls remain distinct.

## Frozen seven-part confirmation gate

The exact FP32 candidate may advance to ONNX/INT8/KL720 deployment-equivalence testing only if **all** criteria pass:

1. available-bag Spearman with NASA `avg_nfoci` >= 0.50;
2. matched radiation-delta Spearman >= 0.60 across the 11 contrasts;
3. overall matched-effect direction agreement >= 9/11;
4. 4 h matched-effect direction agreement = 4/4;
5. combined 24 h + 48 h direction agreement >= 5/7;
6. peak-time recovery = 3/3 across the three complete 4/24/48 Source Name x branch series, evaluated with the same peak-time logic used in development;
7. Source Name direction minima: BALBCF2 >= 4/5, C57BLF2 >= 2/3, C57BLF3 >= 2/3.

Bag MAE, RMSE, Pearson; delta MAE and Pearson; per-source residuals; direction summaries; and complete-series peak-time details are mandatory descriptive outputs but do not add new pass/fail thresholds.

## One-time execution rule

Default evaluator mode is phenotype-blind preflight. It may verify the exact checkpoint and QC1 manifest, rerun image QC, load the model, and perform phenotype-blind inference, but it must not hash/read/join the raw phenotype table or write prediction values.

Only after preflight confirms the frozen evaluator source and inputs may the same unchanged evaluator be run with `--unblind`.

The explicit unblinding run must:

1. finish all model/image checks and phenotype-blind inference before opening the phenotype table;
2. verify the raw phenotype-table SHA256;
3. join each of the 22 Sample Names to exactly one raw NASA phenotype row;
4. verify metadata consistency before accepting `avg_nfoci`;
5. compute the predeclared bag, matched-delta, direction, source, and peak-time outputs;
6. write all seven PASS/FAIL checks;
7. record that no holdout outcome was used for tuning.

No retraining, BatchNorm recalibration, threshold changes, resampling, replacement, or model selection is permitted after unblinding.

A failure of any criterion is a failed confirmatory evaluation. The same holdout must not then be reused as an untouched tuning set.

A complete pass authorizes only the next deployment stage:

`frozen FP32 -> ONNX parity -> INT8 calibration/quantization -> BIE -> NEF -> physical KL720 -> biological-equivalence comparison`

It does not itself establish KL720 equivalence.
