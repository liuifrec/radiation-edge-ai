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
nine source-bound raw float32 \`.npy\` logit artifacts. Each is linked to one of
the **same nine frozen normalized inputs**, with original SHA256, size and
window position. Their saved stitched probability \`.npz\` files are also
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
4. Use the original MONAI \`compute_importance_map\`, \`gaussian\` mode,
   \`sigma_scale=0.125\`, 512x512, CPU float32. Accumulate weighted
   probabilities in float64 in **ascending frozen window order**, normalize
   and cast to float32, exactly as the historical pipeline did.
5. **Fail closed** if the replay disagrees with either saved stitched
   probability \`.npz\` by more than the *predeclared numeric tolerance*
   **5e-7 maximum absolute probability error**. This tolerance is a
   numerical replay safeguard, not a biological acceptance threshold.
6. On every previously reconstructed valid fiber (matched and unmatched),
   use the fiber's **own red/green support pixels** to measure all covering
   saved windows. For each class, compare the signed
   \`p(class)-p(background)\` margin between KL720 and FP512 **before**
   stitching. Record the coverage count (1/2/4 windows), each window's
   support count, mean signed delta, fraction negative, and its mean
   normalized Gaussian-weight contribution.
7. Require that the weighted per-window deltas reconcile to the actual
   frozen stitched margin difference on every sampled source pixel
   (predeclared maximum absolute residual **2e-6**). The algebra is:

   \`Δmargin_stitched(p) = Σw weight_w(p) * Δmargin_window_w(p) / Σw weight_w(p)\`

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

## Windows PowerShell: actual paired fields

From the active project \`.venv\` in
\`C:\Users\yul03\Projects\radiation-edge-ai\`:

\`\`\`powershell
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
\`\`\`

No files need to be generated in the frozen evidence directories.
The complete report is retained in the in-memory \`$Window\` object.

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
