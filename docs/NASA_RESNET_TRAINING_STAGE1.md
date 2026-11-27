# ResNet18 first-block bias by training source

The existing female-fold ResNet18 BIE failed two development bag-shift limits.
This follow-up audits archived training activations. It asks whether the extra
downward shift under Kneron's `optimize=1` persists in each training source
after excluding the original diagnostic nucleus. Stop after this source-stratified
test. A downward increment in every source gives no basis to advance that
setting. This audit has no held-out readout or additional quantization run.

For each nucleus, run the analysis on the archived predictions made by the same
FP32 network suffix after substituting the measured activation at each boundary.
Use pre-ReLU head units (foci/nucleus) so that zero flooring does not mask the
change. For each BIE separately, subtract its stem-pool suffix prediction from
its first residual-block suffix prediction. The paired setting difference is
the increment under `optimize=1` minus the increment under standard INT8.

| Training source | Nuclei | Standard stem shift vs FP32 | Standard first-block increment | Paired setting difference | Negative paired differences |
| --- | ---: | ---: | ---: | ---: | ---: |
| BALBCM1 | 11 | −0.04438 | +0.00663 | −0.04352 | 11/11 |
| BALBCM2 | 12 | −0.04260 | +0.00917 | −0.04479 | 12/12 |
| C57BLM1 | 12 | −0.03540 | +0.00340 | −0.04578 | 12/12 |
| C57BLM3 | 12 | −0.03694 | +0.00883 | −0.04395 | 12/12 |

The standard BIE already shifted the head downward by the stem-pool boundary
in every source mean. Its first block partly offset this on average. Enabling
the bias adjustment introduced an additional downward first-block increment
in all 47 remaining nuclei. The boundary measurements cannot distinguish
clipping from rounding or identify a specific fused operation. Further
investigation should first determine which pre-pool stem tensors the SDK
can expose on the unchanged training graph. The failed ResNet18 BIE and its
frozen criteria retain their original disposition.

## Reproduce

With the package installed, set `RADEDGE_DATA_ROOT` to the external collection.
The command verifies the three original files and each member's training role.

```powershell
$root = $env:RADEDGE_DATA_ROOT
$out = Join-Path $root ("scratch/resnet_stage1_sources_" + [guid]::NewGuid() + ".json")
python -m radiation_edge_ai.nasa_bps.resnet_stage1_source_audit `
  --selection "$root/scratch/nasa_multisource_bie_20261118T232421Z_3a8998bc/selection.json" `
  --trace "$root/scratch/nasa_stage1_blocks_analyze_20261119T063934Z_c3ef1b66/receipt.json" `
  --calibration "$root/models/nasa_bps_53bp1/multimodel_v1_onnx_20261118T011220Z_f49930f7/resnet18/fold_female_role/calibration_manifest.csv" `
  --out $out
```

The November 27 Windows receipt had SHA256
`326452bc1ab2fd6d3cec295c4d91256a0ff5ac278af16f3324340a6b9254a85a`.
All 48 starting rows belonged to different training bags and four biological
sources. Exclusion of the original pilot left 47. Nuclei from the same source
remain related observations.
