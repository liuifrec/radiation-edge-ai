# DNA-fiber v3 KL720 replay

DNAi UNet–MobileOne S1 v3 is the physical reference for three frozen DNA-fiber fields. The original replay at `54672d0` reproduced all 27 stored output tensors byte-for-byte. It used Kneron PLUS 3.2.0, resident KL720 firmware `2.2.0-build.615`, model ID 32769 and model version `0x8b28`. Its archived mean send/receive time was 113.93 ms per window. The biological differences from FP512 remain measurable.

## Windows commands

Run in the repository root. Set `RADEDGE_DATA_ROOT` to the external project data tree, `RADEDGE_DNAI_PYTHON` to the DNAi scientific interpreter and `RADEDGE_KL720_PYTHON` to the separate Windows interpreter with Kneron PLUS. A copy of the data tree can be used at another location. The archive verification resolves its members under that copy.

```powershell
& $env:RADEDGE_DNAI_PYTHON -B .\scripts\windows\run_dnai_hardware_baseline.py --data-root $env:RADEDGE_DATA_ROOT --verify-archive
```

This command reads the model chain, nine inputs and nine reference outputs per field, both segmentation masks and the stitched physical probabilities. It then authenticates the original successful v3 hardware receipt, CSV, summary and comparison, rehashes all 27 raw receive allocations and compares every archived logits file hash with the original physical panel. It writes JSON to standard output. Use `--preflight-only` to check the inputs and references when the archived run is unavailable. Both modes run with NumPy and require no SDK or attached device.

For an explicitly requested physical batch, use the following command on the configured hardware host.

```powershell
& $env:RADEDGE_DNAI_PYTHON -B .\scripts\windows\run_dnai_hardware_baseline.py --data-root $env:RADEDGE_DATA_ROOT --execute
```

Use `--port <port>` to select a KL720 when several are visible. The launcher verifies the artifacts and imports the scientific reconstruction stack before USB inference. The worker checks the resident firmware and PLUS version, uploads the registered NEF once and runs three fields in manifest order. It creates an exclusive external `scratch/dnai_kl720_v3_replay_*` run directory with raw receive allocations, float logits, per-window timings, reconstructed fields, instance measurements, hashes and a hardware receipt. The original panel runner also accepts a user-selected subset or the full prepared panel. The controlled baseline launcher always requires all three fields. No compiler or Docker environment is needed for replay.

## Model and preprocessing

| Artifact | SHA256 |
| --- | --- |
| Original FP512 ONNX | `bb8244208328d54d5b971b254abc1088f96691b286a607d28155fa606b158008` |
| Optimized FP512 ONNX | `a901d1b309a9a0e5026febd5070252787e4b110a2a7d2828e16a19364d6094d0` |
| Training-only calibration manifest | `08fe74d646342c36709f44a116b10379c0c073e16ecffc67fd66fa35a1634a79` |
| INT8 BIE | `01f902e283b565b712cbecbf22c71f19ae6f9ba826d2c6be87e8976cd1bc9863` |
| KL720 NEF | `4b3dfec9a61c99e186dd4b8482fa5b06e6a4958f325ed4a0db0546f1dcab2bfc` |
| Historical successful replay receipt | `a52c7aa80ef3bdf18a2c6ffd92febad15e67a914493b296878eb4b47414c399c` |

DNAi source commit `fcf20c7d6eb385675ff7d07da4fdf471589ce0cf` supplied the segmentation and fiber reconstruction. Original training-source calibration used 192 windows. Replay reads the frozen ImageNet-normalized float32 BCHW tensors with original channel order. Each 1024 × 1024 field contributes nine 512 × 512 windows at stride 256, with a scale of 0.26 µm/pixel. The KL720 packs signed INT8 inputs with radix 5, scale 1, and layout 4W4C8B. The host stitches softmax probabilities with Gaussian weighting at sigma scale 0.125, segments the field and measures valid fibers with the original DNAi package. It requires identical logits arrays, physical masks and historical FP512 comparison metrics, and a maximum absolute stitched-probability difference of `5e-7`.

The reconstruction imports the repository’s original FP512 fidelity modules, NumPy, OpenCV, PyTorch, MONAI, ONNX Runtime and DNAi’s `dnafiber` package. The hardware interpreter needs PLUS and NumPy. Device selection uses discovered product, connectability and an optional operator-supplied port.

| Frozen field | FP512 fibers | Physical v3 fibers | Matched fibers | Instance F1 | Foreground Dice |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1__tile_4` | 14 | 12 | 11 | 0.846154 | 0.865300 |
| `2__tile_9` | 11 | 13 | 9 | 0.750000 | 0.758757 |
| `16__tile_5` | 11 | 10 | 10 | 0.952381 | 0.844947 |

The v4 characterization recovered one three-grader-supported FP512 fiber and lost two. V5 lost a grader-supported fiber on `2__tile_9`. V3 remains the reference. These comparisons describe the frozen fields and establish no population-level biological equivalence.
