# DNAi MobileOne-S1 INT8 PTQ v3 plan

## Motivation

INT8 v2 improved biological fidelity over v1 on the frozen first full-image gate, but remained below the acceptance target:

| metric | v1 | v2 |
|---|---:|---:|
| mean window argmax agreement | 0.99912177 | 0.99928453 |
| window disagreement pixels | 2072 / 2359296 | 1688 / 2359296 |
| mean softmax probability RMSE | 0.020018169 | 0.014796005 |
| stitched FP512 vs BIE512 argmax agreement | 0.99896145 | 0.99920845 |
| stitched foreground Dice | 0.82091452 | 0.86237652 |
| valid fibers (FP512 -> BIE) | 14 -> 10 | 14 -> 11 |
| fiber-match F1 | 0.7500 | 0.8000 |

V2 therefore shows that calibration representativeness matters, but calibration size/composition alone did not restore biological equivalence.

## V3 hypothesis

Kneron PTQ uses `datapath_range_method="percentage"` with `percentage=0.999` by default, excluding the outer 0.1% of activation values from datapath range estimation. DNA-fiber segmentation is sparse: thin, bright red/green tracts occupy a small fraction of predominantly dark microscopy fields. Biologically decisive tract activations may therefore be disproportionately represented in the clipped tail.

V3 changes one PTQ variable only:

- calibration manifest: **same frozen v2 192-image calibration set**
- architecture/weights/ONNX: unchanged
- datapath bitwidth: unchanged INT8 default
- weight bitwidth: unchanged INT8 default
- range method: `percentage`
- `percentage`: **1.0** instead of 0.999
- `percentage_16b`: **1.0** to satisfy Kneron's percentage constraint
- `optimize`: 0

This isolates whether activation-tail clipping contributes materially to loss of thin-fiber biological structure.

## Evaluation policy

Do not change the frozen validation panel or deployment tiling policy.

1. Generate v3 BIE in a separate `int8_512_v3_pct100` directory.
2. Run the same first 1024x1024 image as nine 512x512 windows at 50% overlap.
3. Stitch with the same Gaussian policy.
4. Compare v3 against the same Kneron floating-point reference and DNAi/human endpoints.
5. Do not compile NEF unless biological fidelity is acceptable.

If v3 does not materially improve over v2, next experiments should use Kneron's documented PTQ precision controls (MMSE/outlier tuning and/or bias adjustment) rather than simply increasing calibration-set size again.
