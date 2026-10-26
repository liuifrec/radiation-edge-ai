# v0.14 DNAi saved four-stage diagnostic (read-only)

## Status and intended question

**Exploratory numerical preflight for one previously characterized field, `1__tile_4`.**
Do differences in saved 512x512 logits appear in the optimized CPU FP512 output,
the legacy Kneron floating simulator output, the legacy fixed BIE output, or
the physical KL720 output? The new script reads all four **existing** output
sets and executes **no inference, compilation, quantization, calibration,
reconstruction, or parameter tuning**.

It is independent of the v0.13 scientific worker and therefore can be
reviewed separately. The original v0.13 Gaussian replay has already
established that *both* stitched fields reproduce exactly from the nine
saved windows, and this script does not alter that result.

**Input evidence:** earlier tests verified all 9 normalized input `.npy` SHA256
hashes and tile coordinates against the FP512 field. Current script repeats
those checks against **both** authoritative FP512 and physical KL720 field
records. It rechecks saved per-window output bytes against both field records.

**Model provenance:** checks actual v3 BIE and NEF binary SHA256 against fixed
selected identities, Kneron BIE/NEF compile summary, recorded ONNX model
identity, and PTQ recipe. It does **not** execute the ONNX model; ONNX SHA256
is currently validated against *production metadata*, not freshly computed
from the ONNX model path in this command.

**Legacy-cache caveat:** the original simulator runner overwrote
`bie_validation/per_window.csv` and `bie_validation/summary.json` during
subsequent subset runs; both now describe `2__tile_9`, while the earlier
`1__tile_4` nine-window floating and BIE `.npy` caches remain. Those earlier
cache files **do not carry contemporaneous per-output hash bindings to model
and input**. The script computes their current hashes only. It is not possible
to restore missing production evidence by comparing filenames or current
hashes. Consequently **all comparisons involving these legacy files are
labelled `descriptive_only_legacy_simulator_outputs`**, even if the current BIE
bytes equal those of the independently verified physical KL720 outputs.

## Predeclared, non-tuned outputs

For each 512x512 saved window, compare four stages:

1. Source-bound CPU FP512 optimized ONNX output.
2. Legacy Kneron floating ONNX simulator output.
3. Legacy selected v3 fixed-point BIE simulator output.
4. Source-bound physical KL720 v3 NEF output.

For each adjacent pair and for CPU-to-physical endpoints, report:

- whole-window **class-centered** logit RMSE (common additive logit offsets
  removed, since they do not affect softmax);
- float64-intermediate/float32-output softmax probability MAE;
- number of per-window argmax classification differences;
- probability MAE and signed class-minus-background margin differences on a
  **fixed target subset**: positions where the saved CPU FP512 and physical
  KL720 argmax classes disagree. A foreground target class for each pixel is
  fixed as the CPU class if nonbackground, otherwise the physical class;
- algebraic stagewise telescoping residual for that signed margin, required
  to be at most 1e-12, with both BIE and physical arrays identical by value.

The endpoint-disagreement subset is **defined by these same observations**;
it is not an unbiased independent test population, nor is it equivalent to a
reconstructed DNA-fiber instance. Its purpose is to avoid a trivial
background-dominated average while examining the pixel differences relevant
to already-frozen endpoints. **Do not treat adjacent-stage errors as causal
percentages**, since signed shifts can cancel or reinforce.

The first and last source-bound field outputs must match their recorded
SHA256. The BIE and physical KL720 `.npy` bytes must be exactly equal on all
nine windows, as observed in the preceding verified fingerprint gate; an
unexpected mismatch **aborts** and demands investigation. Any source/model
SHA256 mismatch, missing tensor, wrong shape/dtype/finiteness, or
input-index/coordinate mismatch also aborts. The program never writes source
artifacts or output report files: its JSON goes to stdout only.

## Windows invocation

From the repository root in PowerShell, activate the existing `.venv` if
needed, use the dedicated D: `TEMP`/`TMP` scratch if disk C: remains nearly
full, and run:

```powershell
Set-Location 'C:\Users\yul03\Projects\radiation-edge-ai'
$Kneron = 'D:\radiation-edge-ai-data\models\dnai\unet_mobileone_s1\kneron'
$V3 = Join-Path $Kneron 'int8_512_v3_pct100'
$Manifest = Join-Path $Kneron 'int8_512\validation_512_overlap50\validation_windows_manifest.json'
$FP512 = 'D:\radiation-edge-ai-data\cache\radedge\dnai-assay-run-v0.9-final\fields\df1-72c2102e46e5a4af\field_result.json'
$KL720 = 'D:\radiation-edge-ai-data\cache\radedge\dnai-assay-run-v0.11-kl720-1__tile_4-20261006\execution\fields\df2-d3955d46bdebf2c1\field_result.json'

$Text = python -B scripts/windows/audit_dnai_saved_stagewise_v014.py `
    --fp512 $FP512 `
    --kl720 $KL720 `
    --v3-root $V3 `
    --input-manifest $Manifest
if ($LASTEXITCODE -ne 0) { throw 'STOP: saved stagewise preflight failed' }
$Stage = ($Text -join "`n") | ConvertFrom-Json

$Stage.status
$Stage.model_lineage | Format-List
$Stage.summary | Format-List

$Stage.windows | ForEach-Object {
    [pscustomobject]@{
        Window = $_.window_index
        ChangedCpuKlPixels = $_.cpu_physical_argmax_disagreement_pixels
        CpuToFloatProbMAE = $_.stage_pairs.cpu_to_kneron_floating.probability_mae_all
        FloatToBieProbMAE = $_.stage_pairs.kneron_floating_to_bie.probability_mae_all
        BieToKlProbMAE = $_.stage_pairs.bie_to_physical.probability_mae_all
        CpuToFloatClassChanged = $_.stage_pairs.cpu_to_kneron_floating.argmax_changed_pixels_all
        FloatToBieClassChanged = $_.stage_pairs.kneron_floating_to_bie.argmax_changed_pixels_all
        BieToKlClassChanged = $_.stage_pairs.bie_to_physical.argmax_changed_pixels_all
    }
} | Format-Table -AutoSize | Out-String -Width 240
```

All 9-window source hashes/arrays and quantitative results remain in the
session's `$Stage` object. No original files are modified or regenerated.

## Interpreting results

- If BIE→physical is identically zero, the **saved** compiled/hardware stage is
  not the source of the observed discrepancy for these nine windows. It does
  not independently prove how the old BIE cache was generated.
- If CPU→Kneron floating is numerically negligible but floating→BIE is
  substantial, the results are *consistent with* a fixed-point stage
  accounting for most of the change in these cached outputs. This is
  **not** a causal quantization test because legacy producer-side evidence is
  not contemporaneously bound.
- If CPU→Kneron floating has material numerical differences, investigate
  graph/runtime equivalence before discussing quantization.
- Do not tune the BIE recipe, select a different PTQ candidate based on
  these data, or claim new radiobiological endpoint fidelity.
- `1__tile_4` and the 20-image panel are **development characterization data**,
  not a fresh independent holdout. Any future biological-equivalence claim
  requires a separately frozen independent cohort and predeclared endpoints.

For project status see [issue #20](https://github.com/liuifrec/radiation-edge-ai/issues/20).

## First read-only Windows execution checkpoint (2026-10-26)

The investigator executed the v0.14 script on Windows against the original
frozen `1__tile_4` development field and its nine saved 512×512 input windows,
after merging integration PR #21 and the v0.14 script PR #22 into `main`.
The script completed with `status=descriptive_only_legacy_simulator_outputs`;
all source/hash/comparability gates passed without inference or frozen-file writes.

The source-bound FP512 and physical KL720 output sets remain intact and the
selected v3 BIE-to-physical NEF artifact lineage was independently checked
in the previous preflight. The current nine BIE simulator `.npy` files are
**byte-identical to the physical KL720 logits in every window**.

### Four-stage comparison

| Adjacent/endpoint stages | Mean whole-window probability MAE | Raw argmax disagreements across 9 windows | Mean class-centered logit RMSE |
| --- | ---: | ---: | ---: |
| Optimized CPU FP512 → Kneron floating simulator | 6.83379561118426e-09 | **0** | 1.46638934560703e-06 |
| Kneron floating simulator → selected v3 fixed BIE | 0.0034825329531602634 | **1463** | 0.7705391500775934 |
| Selected v3 fixed BIE → physical KL720 | **0** | **0** | **0** |
| Optimized CPU FP512 → physical KL720 | 0.003482533898520016 | **1463** | 0.7705391421362062 |

The 1463 disagreements are counted **separately across nine raw 512×512
window argmax masks** (overlapping windows may count an image pixel more
than once). They are not the 1549 changed pixels previously counted in
the final 1024×1024 segmentation PNGs after Gaussian blending and dilation.
Do not conflate the raw and postprocessed denominators.

| Window index | CPU → floating probability MAE | Floating → BIE probability MAE | Float → BIE raw argmax-changed pixels | BIE → KL720 probability MAE |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 6.03430067950643e-09 | 0.003144089982948664 | 196 | 0.0 |
| 1 | 1.21258627583387e-08 | 0.006389644218619511 | 44 | 0.0 |
| 2 | 1.81880967324114e-08 | 0.010544469304107152 | 41 | 0.0 |
| 3 | 2.7070895617746e-09 | 0.0015880191735591824 | 211 | 0.0 |
| 4 | 3.58386404812154e-09 | 0.0015267036939834912 | 188 | 0.0 |
| 5 | 6.00049787635726e-09 | 0.0027670460709835927 | 106 | 0.0 |
| 6 | 7.63818906272344e-09 | 0.0025779861023931807 | **531** | 0.0 |
| 7 | 1.45603656909188e-09 | 0.0012184101066770993 | 101 | 0.0 |
| 8 | 3.77022321233304e-09 | 0.0015864279251704971 | 45 | 0.0 |

All nine CPU→floating raw argmax-changed counts were **0**, whereas all
nine floating→BIE changed counts equal their corresponding CPU→physical
changed counts; all nine BIE→KL720 changed counts were **0**.

### What this does—and does not—establish

- In the *saved outputs for this previously selected field*, the
  observed **discrete pre-stitch class differences emerge at the floating
  simulator → fixed-point BIE simulator comparison**, not at the CPU →
  floating simulator or BIE → physical KL720 comparisons. This strongly
  localizes the apparent numerical discrepancy to the fixed-point/model
  conversion stage.
- The Kneron floating simulator is nearly equivalent to CPU FP512 on this
  selected field: 0 changed class assignments, mean softmax MAE ~6.83e-9.
  Kneron BIE matches the physical NPU saved arrays exactly, but that does
  not demonstrate that the earlier BIE cache was contemporaneously bound
  to the same model/input/output provenance record.
- There is **no basis to isolate a particular quantization operator or
  PTQ range calibration parameter as the cause**, nor to infer hardware
  equivalence on other inputs. The selected BIE was the product of
  conversion/quantization; this descriptive comparison does not separate
  those internal operations.
- **Window 6** has the largest number of changed raw class labels (531),
  and earlier v0.13 analysis located FP512-only fibers #15/#17 within
  single-window-covered parts of that same window. This shared location
  is hypothesis-generating, **not** an analysis of v0.14 simulator
  class margins at the two particular fiber supports.
- **Window 2** has the largest whole-window float→BIE probability MAE
  (0.0105444693) but only 41 changed class labels, whereas window 6
  has a smaller MAE (0.0025779861) but 531 changed class labels.
  Mean probability shifts can therefore understate local categorical or
  object-reconstruction disruption.
- There is **no new independent biological holdout**, biological fidelity
  acceptance, radiation-response result, PTQ tuning, or inference benchmark
  in this checkpoint. The development 20-image panel should not be used
  retroactively to select an apparently favorable quantization candidate.

### Decision / remaining acceptance boundaries

**PASS: v0.14 read-only four-stage numerical diagnosis** of this frozen
development field. Legacy cached simulator outputs have no contemporaneous
producer-side per-output hash binding, so reported relationships involving
them remain `descriptive_only_legacy_simulator_outputs` even when they are
currently byte-identical to authenticated physical outputs.

Before a mechanistic/biological-equivalence claim: obtain newly selected
independent input data, freeze model and input/output SHA256s before execution,
prespecify instance-level endpoints and tolerances, and compare any fixed
PTQ intervention without using the development field to choose parameters.
