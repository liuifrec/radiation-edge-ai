# DNAi Reference Baseline

This project uses DNAi as the first Tier-I biological reference workflow for DNA-fiber segmentation and downstream tract measurement.

## Frozen upstream revision

- Upstream repository: `ClementPla/DNAi`
- Frozen commit: `fcf20c7d6eb385675ff7d07da4fdf471589ce0cf`
- Local external source location used during initial Windows development: `D:\radiation-edge-ai-data\external\DNAi`

The external checkout is not vendored into this repository. Reproduction should use the exact upstream commit above.

## Environment constraint

The frozen DNAi revision declares:

```text
requires-python = ">=3.10"
```

The KL720 hardware runtime remains a separate Python 3.9 environment. DNAi should therefore use its own environment, preferably stored on the external data drive, rather than modifying the known-good Kneron runtime.

## Reference segmentation path

At the frozen revision, DNAi exposes multiple segmentation backbones, including compact CNN options such as:

- UNet + MobileOne S0
- UNet + MobileOne S1
- UNet + MobileOne S2
- UNet + MobileOne S3
- UNet + ResNet18
- UNet + ResNet34

The public notebook demonstrates `UNET_MOBILEONE_S1` as a single-model inference option. This is the first architecture to evaluate for direct ONNX/KL720 conversion before attempting knowledge distillation.

The model predicts three semantic classes:

```text
0 = background
1 = red tract
2 = green tract
```

## Frozen preprocessing/inference facts

The reference inference implementation currently uses:

- ImageNet normalization: mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`;
- rescaling to the training resolution of `0.26 um/pixel` when the input pixel size differs;
- `1024 x 1024` sliding-window inference;
- `25%` overlap;
- Gaussian overlap blending;
- softmax probabilities over three classes;
- final semantic mask from `argmax`;
- a small `3 x 3` morphological dilation in `probas_to_segmentation`.

These operations must be frozen for the first reference experiment. We should not change preprocessing, model architecture, tile strategy, and post-processing simultaneously.

## Reference hierarchy

The first DNA-fiber experiment will preserve the following levels:

```text
R0  public/manual annotation where available
R1  original DNAi FP32 reference workflow
R2  compact FP32 candidate / ONNX-equivalent model
R3  KL720 INT8 deployment
```

The immediate goal is to create a small, deterministic R1 reference subset and save:

- source image identifiers;
- input pixel-size assumptions;
- model revision;
- raw class probabilities or logits where practical;
- semantic masks;
- downstream fiber reconstruction outputs;
- tract-level measurements;
- runtime metadata.

## Storage policy

Large datasets, Hugging Face caches, checkpoints, ONNX files, NEF files, and generated masks belong under the external storage roots (`RADEDGE_*`) and must not be committed to Git.
