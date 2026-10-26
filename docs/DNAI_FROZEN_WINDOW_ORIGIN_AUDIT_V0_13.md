# Frozen-window origin audit (v0.13 development, stacked on PR #18)

## One question, no new inference

Determine whether the already-characterized FP512/KL720 red/green probability
disagreements can be observed **in the saved individual 512x512 model outputs
before Gaussian stitching**, and measure exactly how the common frozen
stitching weights combine their signed class-vs-background margin differences.

This audit is a **read-only diagnostic on the previously characterized
development field**, not a new independent biological validation or a claim
that a particular model component caused the discrepancy.

**Important:** Both v0.9 FP512 and v0.11 physical KL720 field records contain
nine source-bound raw float32 `.npy` logit artifacts. Each is linked to one of
the **same nine frozen normalized inputs**, with original SHA256, size and
window position. Their saved stitched probability `.npz` files are also
source-bound. The code does not load/run the ONNX model, touch the KL720,
regenerate inference, calibrate, or alter segmentation/matching.

## What the command does

1. Run both complete frozen field verifiers and exact input-identity gate
   (source artifacts, run records, tensor hashes, field JSON).
2. Re-run **only existing DNAi scientific reconstruction on the two
   saved segmentation PNGs** and reproduce the historical valid-fiber CSVs.
   Match with the existing deterministic greedy bbox-IoU **0.5** threshold
   and retain maximum-cardinality sensitivity, without changing either.
3. Load all 18 source-bound raw logits (9 FP512 + 9 KL720), check each SHA256,
   byte size, shape, dtype and finiteness. Softmax with the frozen float64
   intermediate / float32 output semantics.
4. Use the original MONAI `compute_importance_map`, `gaussian` mode,
   `sigma_scale=0.125`, 512x512, CPU float32. Multiply float32 probabilities by float32 weights **before**
   adding into a float64 accumulator in **ascending frozen window order**, normalize
   and cast to float32, exactly as the historical pipeline did.
5. **Fail closed** if the replay disagrees with either saved stitched
   probability `.npz` by more than the *predeclared numeric tolerance*
   **5e-7 maximum absolute probability error**. This tolerance is a
   numerical replay safeguard, not a biological acceptance threshold.
6. On every previously reconstructed valid fiber (matched and unmatched),
   use the fiber's **own red/green support pixels** to measure all covering
   saved windows. For each class, compare the signed
   `p(class)-p(background)` margin between KL720 and FP512 **before**
   stitching. Record the coverage count (1/2/4 windows), each window's
   support count, mean signed delta, fraction negative, and its mean
   normalized Gaussian-weight contribution.
7. Require that the weighted per-window deltas reconcile to the actual
   frozen stitched margin difference on every sampled source pixel
   (predeclared maximum absolute residual **2e-6**). The algebra is:

   `Δmargin_stitched(p) = Σw weight_w(p) * Δmargin_window_w(p) / Σw weight_w(p)`

   where weights are identical across runtimes. This is an identity, not a
   counterfactual or causal attribution.
8. Report distance to **internal 256/512/768-pixel window edges** separately
   from minimum reconstructed-bbox distance to the **outer 1024-pixel
   image boundary**. The 32-pixel proximity indicator is descriptive only;
   it is not fitted to this field.
9. Reverify both full source records *and bound artifacts* at the end.
   Return a JSON report to stdout only.

Differences in single-coverage pixels must already be present in the
corresponding source window's saved probabilities. In overlap regions,
weighted combination can increase or decrease the final signed difference
according to the combination of the existing window deltas; it cannot create
a nonzero FP512-vs-KL720 difference out of zero differences in every source
window. This fact alone **does not** establish quantization, stitching,
padding, NPU inference, or border handling as the cause.

## Windows test I/O incident (2026-10-16)

The first local `.venv` full-suite run produced **184 passed, 7 failed**.
All seven failures occurred when NumPy attempted to **write test-only
`.npy` fixtures into the user's C: TEMP/pytest directory**, with
`OSError: ... requested and 0 written`. This is strong evidence of
insufficient C: temporary free space, a filesystem limit, or a temp-path
write restriction, **not evidence of scientific algorithm failure**.
It does not count as completed local acceptance. Linux GitHub Actions
passed **191 tests** on Python 3.9 and 3.11.

Before retrying any scientific work, check free space on both C: and D:.
Do **not** remove or overwrite frozen evidence directories. If D: has
comfortable free capacity, redirect pytest-only scratch to a dedicated,
explicitly disposable directory on D: in the **current PowerShell session**:

```powershell
Get-PSDrive C,D | Select-Object Name,@{Name='FreeGiB';Expression={[math]::Round($_.Free/1GB,2)}}
$Scratch = 'D:\radiation-edge-ai-data\scratch\pytest-temp-v013'
New-Item -ItemType Directory -Path $Scratch -Force | Out-Null
$env:TEMP = $Scratch
$env:TMP = $Scratch
# Only this dedicated pytest base is automatically cleaned between runs.
$TestBase = Join-Path $Scratch 'pytest-basetemp-disposable'
python -B -m pytest -q tests --basetemp $TestBase
if ($LASTEXITCODE -ne 0) { throw 'Offline regression must pass before acceptance' }
```

These settings affect only new processes launched from this PowerShell
session. Leave all model files, field records, frozen .npy tensors, and
saved .npz probabilities untouched. If disk free space is low on D:
as well, stop and resolve storage before retrying.

The **initial** pasted console capture ended after the `audit-dnai-window-origin`
command was issued. Its output was not included in that first capture.
The later console capture **does** verify successful execution and both exact
replays. See the follow-up checkpoint below; retain this historical incident
as a record of how the temporary-storage problem was diagnosed.

## Follow-up Windows acceptance checkpoint (2026-10-16)

The investigator confirmed that **C: has only 0.14 GiB free** whereas **D:
has 223.84 GiB free**. This strongly supports insufficient temporary storage
as the cause of seven earlier NumPy fixture-write failures; it is not a
scientific algorithm regression. The active PowerShell session was changed
to use a dedicated `D:\radiation-edge-ai-data\scratch\pytest-temp-v013`
scratch path for `TEMP`, `TMP` and `pytest --basetemp`. With no code
modification, the complete local regression **passed 191 tests in 19.26 s**.

### Frozen-window replay: confirmed success

The already-launched `radedge audit-dnai-window-origin` process exited
successfully, and its full JSON parsed as `$Window`. The printed results
confirm exact bitwise replay of the two saved 1024×1024 stitched arrays
from all 18 bound cached 512×512 logits (nine per runtime):

| Replay gate | FP512 | Physical KL720 |
| --- | ---: | ---: |
| Max absolute stitched-probability error | 0.0 | 0.0 |
| Bitwise identical to saved NPZ | true | true |
| Shared numerical acceptance threshold | 5e-7 | 5e-7 |

Signed class-margin residual limit remains predeclared at 2e-6; the
worker returned `both_replays_passed=True` and would fail closed on a
class-margin decomposition mismatch. The individual per-class residual
values have **not yet been printed** in the supplied console output.

Historical primary matching remained **14 FP512 / 12 KL720 valid**,
**11 matched**, **3 FP512 unmatched / 1 KL720 unmatched**,
precision 0.9166666666666666, recall 0.7857142857142857,
F1 0.8461538461538461. No threshold, reconstruction, or scientific
conclusion changed.

### Saved prediction differences before stitching

| Window index | FP512–KL720 pre-stitch probability MAE | Argmax agreement |
| ---: | ---: | ---: |
| 0 | 0.003144088102389963 | 0.9992523193359375 |
| 1 | 0.0063896423762793775 | 0.9998321533203125 |
| 2 | 0.01054447550657732 | 0.9998435974121094 |
| 3 | 0.0015880196696566882 | 0.9991950988769531 |
| 4 | 0.0015267053433933961 | 0.9992828369140625 |
| 5 | 0.0027670467492700093 | 0.9995956420898438 |
| 6 | 0.0025779898114368636 | 0.9979743957519531 |
| 7 | 0.0012184101969679574 | 0.9996147155761719 |
| 8 | 0.0015864273307085637 | 0.9998283386230469 |

Window 2 has the greatest *average* probability difference, while
window 6 has the lowest overall raw-argmax agreement. Both precede
stitching. This demonstrates that the two sets of frozen predictions
are not identical *before* applying the common Gaussian weighting.
Whole-window aggregates **cannot** establish which saved windows
explain FP512 fiber IDs 1, 15, 17 or KL720 ID 19; this requires
the per-source-fiber class coverage/contribution inspection below.

**Status:** local testing PASS; source-bound full replay PASS;
historical matching PASS; **fiber-specific internal-window proximity
and per-window contribution interpretation still pending**.

Source JSON and artifact verification is performed inside the worker
at the start and end; the investigator's independent Windows-side
SHA256 comparisons against the report have not yet been shown.
Do not merge PR #19 before confirming the remaining specific evidence.

All results are descriptive. Neither an image-boundary association
nor deterministic Gaussian margin decomposition establishes the
cause of the original FP512/KL720 inference differences.


## Final fiber-level origin checkpoint (2026-10-16)

The investigator inspected the **existing in-memory $Window JSON report**
from the completed frozen-window replay; **no new inference or source
mutation** was performed. Both source JSON hashes were independently
rechecked with PowerShell against the report: **FP512_SHA_OK=True** and
**KL720_SHA_OK=True**. The full-window replay still gives **zero absolute
difference for both stitched fields**, exact bitwise identity, and
unchanged historical greedy bbox-IoU 0.5 matching
(14 FP512 / 12 KL720 valid, 11 matched, F1 0.8461538461538461).
Local regression remains **191 passed** with pytest scratch on D:.

### Class support, fixed overlap and internal-window proximity

Class **1 = red**, class **2 = green**. All counts below are on each
**unmatched reconstructed source-fiber's own class pixels**, not whole
bounding boxes, and not independent biological annotations. Window
coverage is from the immutable nine-window layout. `Near32` is the
number of these pixels within 32 px of an **internal** window edge
(x or y at 256, 512 or 768); that is separate from the **outer image
border distance** and is not evidence of a defective seam.

| Source fiber | Class | Class pixels | 1-window | 2-window | 4-window | Near32 | Pixels with opposing pre-stitch signs | KL720−FP512 final margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP512 #1 | 1 | 29 | 0 | 29 | 0 | 0 | 0 | -1.1332696307930243 |
| FP512 #1 | 2 | 7 | 0 | 7 | 0 | 0 | 0 | -1.286643013507793 |
| FP512 #15 | 1 | 39 | 39 | 0 | 0 | 0 | 0 | -0.39372232610073227 |
| FP512 #15 | 2 | 16 | 16 | 0 | 0 | 8 | 0 | -1.2437842316867318 |
| FP512 #17 | 1 | 22 | 22 | 0 | 0 | 4 | 0 | -1.4313522491985085 |
| FP512 #17 | 2 | 22 | 22 | 0 | 0 | 0 | 0 | -0.6369702007558028 |
| KL720 #19 | 1 | 5 | 0 | 5 | 0 | 0 | 1 | +0.8764627280645072 |
| KL720 #19 | 2 | 3 | 0 | 3 | 0 | 0 | 1 | +0.8450821148386846 |

**Important distinction:** "single-window" and "not near an internal
edge" are not equivalent categories. FP512 #15 has 8/16 green source
pixels near an internal edge **yet all 16 are covered by only one
window**; FP512 #17 has 4/22 red pixels near an internal edge with
only one covering window. KL720 #19 is covered by two windows but has
no source-class pixels within 32 px of any internal edge. A geometric
seam-proximity statistic alone does not identify a stitching mechanism.

### Independent saved-window class-margin contributions

`RawDelta` denotes the **mean per-window** KL720-minus-FP512
difference in `p(class)-p(background)` over the source-class pixels
covered by that saved window. `Contribution` is the Gaussian-weighted
mean contribution **over the entire source-class support**. Contributions
from all covering windows sum to the saved final margin delta
within floating point rounding.

| Source fiber | Class | Window | Source class pixels sampled | RawDelta | Contribution |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP512 #1 | 1 | 0 | 29 | -1.6827180903226313 | -0.8419627269670368 |
| FP512 #1 | 1 | 3 | 29 | -0.5826703591216035 | -0.29130689126485787 |
| FP512 #1 | 2 | 0 | 7 | -1.5698295245279692 | -0.7849147622639846 |
| FP512 #1 | 2 | 3 | 7 | -1.0034564464357183 | -0.5017282232178591 |
| FP512 #15 | 1 | 6 | 39 | -0.39372233173642784 | -0.39372233173642784 |
| FP512 #15 | 2 | 6 | 16 | -1.2437842318904586 | -1.2437842318904586 |
| FP512 #17 | 1 | 6 | 22 | -1.4313522502144969 | -1.4313522502144969 |
| FP512 #17 | 2 | 6 | 22 | -0.6369701929665594 | -0.6369701929665594 |
| KL720 #19 | 1 | 5 | 5 | +0.9774356555426493 | +0.876028047051167 |
| KL720 #19 | 1 | 8 | 5 | +0.004189684242010117 | +0.00043467400159148846 |
| KL720 #19 | 2 | 5 | 3 | +0.9425962370975564 | +0.8448031695428989 |
| KL720 #19 | 2 | 8 | 3 | +0.002688887412659824 | +0.00027896838222429707 |

Interpretation for **this development field only**:

- **FP512 #1** has two source windows, 0 and 3. Both provide
  *negative* pre-stitch class-minus-background shifts for both colors.
  All 36 class-source pixels have two-window coverage, no opposing
  sign, and none lie within 32 px of an internal window edge.
  The final negative class-margin shifts are weighted combinations of
  pre-existing negative per-window shifts, not contradictory seams.
- **FP512 #15** is wholly in the single-covered part of saved window
  **6**, including the green class with a -1.2437842318904586 raw
  delta. **No overlapping-window blend occurs at its 55 measured
  source-class pixels**. Earlier reconstruction had selective green
  disappearance.
- **FP512 #17** is also wholly in the single-covered part of saved
  window **6**; red is -1.4313522502144969 before stitching.
  The earlier raw segmentation had zero red on the 22 red source
  pixels and dilation yielded one same-class pixel.
- **KL720 #19** is covered by windows 5 and 8, but window 5
  contributes ~99.95% of the final red-margin shift and ~99.97%
  of the final green-margin shift. One source pixel per color has
  opposite-signed per-window deltas, despite both window-averaged
  deltas being positive. Its local margins are overwhelmingly
  attributable *algebraically* to the saved window 5 prediction.

### Acceptance decision

**PASS: frozen-output window-origin diagnosis for the previously
characterized nine-window development field.** The core inference
difference is *observable upstream of the common Gaussian stitch*,
and the common stitched implementation is exactly replayable for both
runtimes. For FP512 #15 and #17 in particular, a multiwindow blending
effect **cannot be necessary** to produce the measured source-class
margin shifts because every class-source pixel is covered by only
one saved window.

The worker enforces the **unchanged** margin decomposition residual
limit of 2e-6 per source-class support; it completed successfully.
The console output supplied here did not display the numeric maximum
residuals (the wide PowerShell table truncated that column), so
no residual values should be invented or reported beyond the
programmatic fixed-threshold pass.

This does **not** establish which upstream mechanism caused the saved
logit differences (quantization, model conversion, execution precision,
hardware behavior, etc.) or establish a biological fiber ground truth.
Whole-window metrics (window 2 largest probability MAE; window 6
lowest argmax agreement) cannot by themselves identify an individual
fiber mechanism. No inference was rerun, no matching threshold
was changed, and no historical biological-fidelity conclusion
was revised.

**Next decision gate:** Review PR #18 first and this stacked PR #19
second. If approved, freeze this observational investigation and
design one separate controlled model-conversion/precision comparison
with explicit ground truth and preserved input/output provenance.
That intervention is **not** part of this accepted read-only audit.


## Windows PowerShell: actual paired fields

From the active project `.venv` in
`C:\Users\yul03\Projects\radiation-edge-ai`:

```powershell
Set-Location 'C:\Users\yul03\Projects\radiation-edge-ai'
git status --short
git fetch origin
git switch feature/v0.13-dnai-window-origin
git pull --ff-only origin feature/v0.13-dnai-window-origin

$SciencePy = 'D:\radiation-edge-ai-data\envs\dnai311\Scripts\python.exe'
$FP512 = 'D:\radiation-edge-ai-data\cache\radedge\dnai-assay-run-v0.9-final\fields\df1-72c2102e46e5a4af\field_result.json'
$KL720 = 'D:\radiation-edge-ai-data\cache\radedge\dnai-assay-run-v0.11-kl720-1__tile_4-20261006\execution\fields\df2-d3955d46bdebf2c1\field_result.json'

foreach ($p in @($FP512,$KL720)) {
    if (!(Test-Path -LiteralPath $p -PathType Leaf)) { throw "Missing: $p" }
}
$BeforeFP = (Get-FileHash -LiteralPath $FP512 -Algorithm SHA256).Hash
$BeforeKL = (Get-FileHash -LiteralPath $KL720 -Algorithm SHA256).Hash

# Offline tests use .venv, not the scientific interpreter.
python -B -m pytest -q tests

# The command reads frozen outputs, never runs a model/hardware.
$Text = radedge audit-dnai-window-origin `
    --fp512 $FP512 `
    --kl720 $KL720 `
    --scientific-python $SciencePy `
    --json
if ($LASTEXITCODE -ne 0) { throw "Frozen-window audit failed" }
$Window = ($Text -join "`n") | ConvertFrom-Json

if ((Get-FileHash -LiteralPath $FP512 -Algorithm SHA256).Hash -ne $BeforeFP) {
    throw "FP512 field record changed"
}
if ((Get-FileHash -LiteralPath $KL720 -Algorithm SHA256).Hash -ne $BeforeKL) {
    throw "KL720 field record changed"
}

$Window.frozen_replay | Format-List
$Window.historical_instance_matching.primary.summary | Format-List
$Window.internal_window_edges_px | Format-List
$Window.window_pair_metrics |
    Select-Object window_index,x,y,pre_stitch_probability_mae,pre_stitch_argmax_agreement |
    Format-Table -AutoSize

$UnmatchedClasses = foreach ($side in @('fp512_source','kl720_source')) {
    foreach ($fiber in $Window.per_fiber.$side) {
        if ($fiber.historical_primary_matched) { continue }
        foreach ($class in $fiber.per_class) {
            [pscustomobject]@{
                Side = $side
                FiberID = $fiber.local_fiber_id
                ClassID = $class.class_id
                ClassPixels = $class.source_class_support_pixels
                Cover1 = $class.n_support_pixels_covered_by_1_window
                Cover2 = $class.n_support_pixels_covered_by_2_windows
                Cover4 = $class.n_support_pixels_covered_by_4_windows
                NearInternalEdge = $class.n_support_pixels_within_32px_of_internal_window_edge
                SignConflicts = $class.n_multiwindow_pixels_with_opposite_delta_signs
                StitchedMarginDelta = $class.saved_stitched_margin_delta_mean
                MaxResidual = $class.maximum_absolute_margin_decomposition_residual
            }
        }
    }
}
$UnmatchedClasses | Format-Table -AutoSize | Out-String -Width 250

# Show per-window contributions specifically for the four unmatched IDs.
foreach ($side in @('fp512_source','kl720_source')) {
    foreach ($fiber in $Window.per_fiber.$side) {
        if ($fiber.historical_primary_matched) { continue }
        Write-Host "$side local fiber $($fiber.local_fiber_id)"
        foreach ($class in $fiber.per_class) {
            Write-Host "  class $($class.class_id)"
            $class.per_window |
                Select-Object window_index,n_class_support_pixels_in_window,mean_pre_stitch_margin_delta,mean_weighted_contribution_over_all_class_support_pixels |
                Format-Table -AutoSize
        }
    }
}
```

No files need to be generated in the frozen evidence directories.
The complete report is retained in the in-memory `$Window` object.

## Predeclared stop/interpretation criteria

- **STOP without scientific conclusion:** source record or any window hash
  fails; nine-window input identity mismatches; cached tensor contract
  differs; either stitched replay exceeds fixed 5e-7 tolerance; class-margin
  decomposition exceeds fixed 2e-6 tolerance; or historical matching changes.
  Review implementation/numeric-version provenance first; do not relax
  tolerances until the reason is explained.
- **If differences exist in single-coverage pixels:** they are observable
  in *the saved pre-stitch model outputs*, independent of how multiple
  windows are weighted.
- **If window contributions have opposing signs at shared pixels:**
  the Gaussian aggregation can modulate the final discrepancy, but this
  describes numerical combination of *already differing predictions*.
- **If every individual window agrees at a pixel:** identical fixed
  stitching weights mathematically imply equal stitched probabilities;
  a contradiction means a replay/source-contract bug, not biological
  evidence.
- **Do not conclude:** model quantization caused this, image proximity
  proves a stitching defect, missing segmentation proves missing biological
  fiber, or one development field establishes radiobiological equivalence.

## Release boundary

This is a stacked **v0.13 experimental diagnostic**, based on unmerged
PR #18, **not** a replacement for v0.11/v0.12 field validation or the
previous frozen analysis. Its new interpretation must be accepted only
after the user's read-only Windows execution and source-hash confirmation.
