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

## Next hardware gate

Before adapting a biological model, validate semantic-segmentation inference on KL720 using the installed STDC-Seg example. This is the closest existing Kneron model-zoo path to the DNA-fiber segmentation workload.

The biological conversion sequence is then:

```text
DNAi reference segmentation
        -> compact KL720-compatible student
        -> ONNX verification
        -> INT8 / NEF conversion
        -> KL720 inference
        -> original/compatible CPU fiber reconstruction
        -> biological endpoint agreement
```
