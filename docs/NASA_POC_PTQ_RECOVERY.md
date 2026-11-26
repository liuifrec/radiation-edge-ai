# NASA trained-model PTQ and fixed-point replay

The frozen ResNet18 female-source BIE remains ineligible for NEF compilation.
The November 26 recovery reproduced the original six passing and two failing
engineering decisions from 2,400 archived simulator predictions. The three
paired result tables were byte-identical to the November 18 readout.

The recovered workers and synthetic fixtures originate at Windows milestone
`fa5047fd20c104de98f62ad39641a17fee1489dc`. PR #35 added portable paths for
new exports and tested trained ONNX export on native Windows. This recovery adds
a reader for existing exports, a trained-graph screening mode, a training-only
PTQ worker, fixed-point inference, and an offline scorer. All SDK and model
artifacts are external. The [NASA R1 v2 USB replay](NASA_LOCAL_NPU_QUICKSTART.md)
uses its own NEF and evidence. The recovered ResNet18 workers stop at the BIE
decision and do not dispatch to USB.

## What the records verify

| Stage | Evidence checked in the recovery |
| --- | --- |
| Trained PyTorch to ONNX | Verified the archived export, graph and 192-member parity table for each of five trained screens. They cover ResNet18 on all three folds and MobileNetV2 on the female and male-2 folds. |
| PTQ inputs | Verified all 192 original backslash-delimited array names and SHA256s, four nuclei from each of 48 bags from BALBCM1, BALBCM2, C57BLM1 and C57BLM3. The manifest retained SHA256 `79aef81e921a723187d73cef5151ff9b943ad486d7e83af19daf063a5b5a1489`. |
| Original versus optimized ResNet18 FP32 | Executed both archived ONNX graphs in ONNX Runtime on the 192 training tensors. Every prediction matched exactly. The paired prediction CSV SHA256 was `83e212b5a7da98aecab908c3e803dfd8ba045aaa14a7d0173c48838f1b21147b`. |
| INT8 BIE | Verified binary SHA256 `2428c60742e862f3bc3b6a6831284a3afa2b9e4db652fa760f331e3500bbea2f`. The historical fixed-point simulator evaluated 24 source-held-out development bags and 12 matched radiation contrasts. The November 26 offline run consumed the archived prediction CSV. |
| NEF and USB | The frozen bag mean absolute difference was 0.085602881 foci/nucleus (limit 0.05), and the maximum was 0.168734688 (limit 0.15). These results block compilation of this BIE. |

The matched-delta mean and maximum absolute differences were 0.050084051 and
0.088205092 foci/nucleus. Bag MAE against the NASA reference increased by
0.014407648. All strong-effect directions and the four-hour direction were
preserved, and all pairs were complete and finite. The
[prospective policy](NASA_PTQ_ACCEPTANCE_V1.md) defines all eight engineering
limits. Biological validation and historical final-holdout analysis are
outside this replay.

## Re-score the archived simulator predictions

With the repository installed, set `RADEDGE_DATA_ROOT` to the local copy of
the Windows evidence tree. From the checkout, run in PowerShell:

```powershell
$root = (Resolve-Path -LiteralPath $env:RADEDGE_DATA_ROOT -ErrorAction Stop).Path
$scratch = Join-Path $root 'scratch'
$prep = Join-Path $scratch 'nasa_resnet18_female_eval_20261118T060157Z_5ffa44b2/preparation'
$run = Join-Path $scratch 'nasa_resnet18_female_bie_readout_20261118T060725Z_438c78b7'
$fp32 = Join-Path $root 'models/nasa_bps_53bp1/multimodel_v1_20261117T053459Z_b02f5836/resnet18/fold_female_role'
$metadata = Join-Path $root 'data/nasa_bps_microscopy/metadata'
$out = Join-Path $scratch ('nasa_bie_rescore_' + [guid]::NewGuid().ToString('N'))
$options = @(
  '--policy', 'docs/NASA_PTQ_ACCEPTANCE_V1.json',
  '--policy-sha256', '20a685eba6502e5cb10d4506f39f8b155d6b29548b2df10915fe9dd5cd470db9',
  '--acceptance-document', 'docs/NASA_PTQ_ACCEPTANCE_V1.md',
  '--prepared-inputs', (Join-Path $prep 'inputs.csv'),
  '--preparation-receipt', (Join-Path $prep 'preparation.json'),
  '--development-manifest', (Join-Path $metadata 'r1_v2_freeze/r1_v2_development_manifest_100_qc1.csv'),
  '--nasa-reference', (Join-Path $metadata 'pilot_v1/reference_phenotypes_v2/pilot_v1_sample_reference_frozen.csv'),
  '--fp32-nuclei', (Join-Path $fp32 'nucleus_predictions.csv'),
  '--fp32-bags', (Join-Path $fp32 'bag_predictions.csv'),
  '--bie-receipt', (Join-Path $run 'vendor/bie_receipt.json'),
  '--bie-nuclei', (Join-Path $run 'vendor/bie_nuclei.csv'),
  '--out', $out
)
python -m radiation_edge_ai.nasa_bps.multimodel_bie_comparison @options
Get-Content -LiteralPath (Join-Path $out 'comparison.json')
```

For this archive, exit code 1 with `status=COMPARISON_COMPLETE` and
`eligible_for_nef_compilation=false` reproduces the original decision. Check
`output_sha256` against the historical comparison receipt (its SHA256 is
`4f8efacd1f5bd3223deb08319f37d72ea2604029f0ddc29880d52483cef75146`).
The new receipt records its own creation time and scorer hash. Missing or changed
sources fail verification. Each output directory is single-use.

## Worker boundaries

* `kneron_candidate_screen.py --trained-export` checks an allowlisted
  historical graph and export identity, exact parity-table digest and paired
  training outputs before dispatching to the vendor evaluator. The existing
  untrained `--candidate-manifest` mode remains available.
* `multimodel_ptq.py parity` verifies the exact archived ResNet18 screen and
  export, then runs original and optimized graphs on the training inputs. A
  fresh screen has its own receipt identity and cannot replace the archived
  prerequisite. Legacy and portable paths resolve against the calibration
  export directory. Case, filename, directory confinement, unique membership,
  file contents, shape, channels, dtype, range and training role are checked.
* `multimodel_ptq.py quantize` rechecks the complete optimized FP32 parity CSV
  and its inputs before calling `ktc.ModelConfig.analysis`. A returned BIE
  has status `PTQ_GENERATED_UNVALIDATED`. Use the pinned Linux toolchain
  image `sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041`
  with read-only input mounts and a fresh writable `/data1` work directory.
* `prepare_multimodel_bie_evaluation` verifies the artifact chain, materializes
  identical FP32 and BIE tensors, and checks that the 192 calibration nuclei
  are disjoint from the 2,400 development readout nuclei. The standalone
  `multimodel_bie_inference.py` worker requires its sibling
  `multimodel_ptq.py` and the pinned SDK. It verifies the fixed-point API
  identity, integer codes, decoded burdens, raw vendor dumps, process exit
  statuses and a public-API probe. It emits the CSV and receipt for scoring.

The recovery exercised the host FP32 path and historical receipt replay. Fresh
vendor optimization, quantization and BIE inference require the external SDK.
Future candidate selection belongs on training-source inputs. Reusing this
development readout for selection would compromise an independent assessment.
