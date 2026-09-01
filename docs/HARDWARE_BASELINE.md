# KL720 Hardware Baseline

Baseline established on 2026-09-01 using Kneron PLUS 3.2.0-compatible Python tooling on Windows 11.

## Device discovery

The project-owned hardware check successfully detected one Kneron device with:

- product ID: `0x720` (KL720)
- firmware mode: `KDP2 Comp/F`
- device connection: PASS
- model upload: PASS

Machine-specific USB port IDs are intentionally not treated as portable configuration.

## Known-good Python runtime

A dedicated Python 3.9.13 environment with the Kneron `kp` package is used for hardware-facing commands. The normal project development environment remains separate so scientific dependencies can evolve without destabilizing the known-good KL720 runtime.

## NEF model upload smoke test

A KL720 YOLO NEF from the installed Kneron examples was uploaded successfully through:

```text
scan_devices -> connect_devices -> set_timeout -> load_model_from_file
```

The returned descriptor reported:

- target chip: `KP_MODEL_TARGET_CHIP_KL720`
- 2 models in the NEF
- model IDs: 221 and 225
- model 0: 1 input node, 1 output node
- model 1: 1 input node, 3 output nodes

This establishes a project-owned baseline independent of the original Kneron demo scripts.

## Semantic-segmentation smoke test

The installed Kneron STDC-Seg model-zoo example was run successfully on the physical KL720 using:

```text
KL720KnModelZooGenericDataInferenceMMSegSTDC.py
```

Observed result:

- device connection: PASS
- model upload: PASS
- image input: PASS
- inference: PASS
- output-node retrieval: PASS
- dense per-pixel segmentation array returned
- colorized segmentation image generated as `output_city_scape_640x428.jpg`

This confirms that the KL720 can execute dense semantic-segmentation inference, not only classification/detection workloads. The STDC example therefore provides the closest existing model-zoo deployment pattern for the planned DNA-fiber segmentation backend.

## Platform qualification status

```text
device discovery             PASS
KL720 NEF loading            PASS
object-detection workflow    PASS
semantic-segmentation flow   PASS
```

The generic KL720 qualification phase is complete.

## First biological conversion

The next milestone is DNA fiber. The intended architecture is:

```text
DNAi reference segmentation
        -> compact KL720-compatible student
        -> ONNX verification
        -> INT8 / NEF conversion
        -> KL720 inference
        -> original/compatible CPU fiber reconstruction
        -> biological endpoint agreement
```

The first scientific task is to freeze a small public DNAi reference subset and identify the exact segmentation model, preprocessing, tiling, and post-processing used by the reference workflow before changing any of them.
