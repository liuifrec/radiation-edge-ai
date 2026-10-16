# Read-only DNAi failure-mode audit (v0.12 additive development)

## Purpose and scientific boundary

`radedge audit-dnai-failure-modes` extends the source-verified FP512 versus
physical KL720 *single-field* diagnostics. It does **not** change v0.11/v0.12
field artifacts, inference, reconstruction, matching rules, scientific thresholds,
validation decisions, or the historical 20-image characterization.

The upstream DNAi source is frozen at `fcf20c7d6eb385675ff7d07da4fdf471589ce0cf`.
Its `probas_to_segmentation` applies **uint8 3-class argmax followed by 3x3
grayscale max dilation**. Dilation allows a neighboring class 2 to replace class
1; independent binary per-color dilation is **not equivalent**. The new audit
requires exact equality with both saved bound segmentation PNGs before reporting.

## Windows PowerShell

```powershell
Set-Location 'C:\Users\yul03\Projects\radiation-edge-ai'
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
git switch feature/v0.12-dnai-failure-modes
git pull --ff-only origin feature/v0.12-dnai-failure-modes

$SciencePy = 'D:\radiation-edge-ai-data\envs\dnai311\Scripts\python.exe'

# Tests run in the active development .venv, not in isolated dnai311.
# The scientific DNAi interpreter does not need pytest.
python -B -m pytest -q tests/test_dnai_failure_modes.py
python -B -m pytest -q tests

# Supply both existing ABSOLUTE frozen field-record paths explicitly.
# Do not rely on unset $FP512/$KL720 session variables.
$FP512 = '<existing verified df1 field_result.json absolute path>'
$KL720 = '<existing verified df2 field_result.json absolute path>'
if (!(Test-Path -LiteralPath $FP512 -PathType Leaf)) {
    throw "Missing FP512 field record: $FP512"
}
if (!(Test-Path -LiteralPath $KL720 -PathType Leaf)) {
    throw "Missing KL720 field record: $KL720"
}

# The active .venv radedge CLI launches the worker with $SciencePy.
# This performs no neural inference, writes no source artifacts.
radedge audit-dnai-failure-modes `
    --fp512 $FP512 --kl720 $KL720 --scientific-python $SciencePy --json
```

The CLI launches an explicit scientific-Python worker because the frozen
`dnafiber` runtime, `cv2`, and `pandas` are excluded from the lightweight
control-plane install. Source field records and artifacts are fully verified
before reconstruction, and verified again after metrics are computed.
The saved valid-fiber CSVs must reproduce under the installed frozen DNAi
version. JSON output goes to stdout, not to evidence directories. Optional
shell redirection can be used **outside** any frozen source directory.

## Fixed diagnostic definitions

* **Support:** The reconstructed fiber's nonzero class-1 (red) and class-2
  (green) pixels in its own localized `fiber.data` mask; not the whole bbox.
  Source support is fixed independently for each direction.
* **Class disappearance:** For each class present on source support, count
  retained class pixels under opposite-runtime raw argmax and saved dilated
  segmentation. Distinguish zero on support from zero anywhere in a **4-pixel
  padded ROI**. A local zero is not a global biological absence.
* **Foreground retention:** Fraction of source-colored support pixels that
  remain non-background in the opposite runtime's raw and dilated masks.
  This can differ from class retention when red changes to green or vice versa.
* **Probability margins:** On the same source-fiber class pixels,
  `p(class)-p(background)` and `p(class)-max(p(other classes))`;
  report mean/median/fraction nonpositive for both runtimes and signed mean
  delta KL720 minus FP512. Raw argmax margins are **not** calculated on
  final dilated-class assignments.
* **Morphological effects:** Raw versus final same-class retention and
  pixel-count delta on each fixed source support. Class-2 priority follows
  the exact upstream grayscale dilation operation.
* **Fragmentation:** 8-connected raw/final binary foreground and per-class
  *segmentation component proxies* in the padded ROI. Counts include the
  number of components intersecting source support. These are **not**
  reconstructed-fiber split/merge counts.
* **Boundary association:** Minimum bbox-edge distance to the 1024x1024
  image boundary, plus a fixed descriptive **<=32-pixel** grouping.
  Denominators are separately reported for primary matched/unmatched
  reconstructed valid fibers on each source side.
* **Correspondence:** Reuses the same historical primary greedy
  reconstructed-fiber bbox IoU threshold **0.5**, local IDs and
  maximum-cardinality sensitivity. There is no reassignment or threshold
  selection based on probabilities, boundary location, or segmentation.
* **Sampling:** All available valid reconstructed fibers in both directions;
  no exclusion by ROI position or apparent artifact severity.

Every output is observational and **not** a causal diagnosis of quantization,
stitching, border padding, numerical precision, hardware execution, or fiber
identifiability. No human-annotation support, radiation effect, population
generalization or biological fidelity is assessed.

## Checkpoint retained verbatim (not recomputed here)

For the previously verified `1__tile_4` nine-window field:
FP512 18 reconstructed/14 valid, KL720 30 reconstructed/12 valid,
11 primary matched valid fibers, 3 unmatched FP512 and 1 unmatched KL720,
matching F1 0.8461538462, matched length MAE 0.3187430143 um,
and KL720-minus-FP512 total valid-fiber-length delta -42.9203347266 um.
Exploratory v2 observations: FP512 unmatched 1 both color signals disappear;
15 green disappears; 17 severe red loss; KL720 unmatched 19 is a small
two-color reconstruction. Three of three unmatched FP512 fibers were within
32 pixels of the image boundary, versus zero of eleven matched. These are
**earlier observations**, not yet a completed acceptance result for this
permanent command. The raw-margin v2 conclusion was descriptive only.

A real-data acceptance must check both exact morphology replays; confirm
reconstructed counts and match pairs/length decomposition against v0.12;
inspect per-row local IDs including 1, 15, 17, and 19; check per-class margin
signs/denominators and boundary grouping; verify source SHA256 unchanged; and
record any discrepancy without modifying the frozen results.


## Permanent-command local acceptance checkpoint (2026-10-16)

The investigator ran the **new permanent read-only command** on the two
existing verified frozen `1__tile_4` fields, using
`D:\radiation-edge-ai-data\envs\dnai311\Scripts\python.exe` as
the scientific worker and the active project `.venv` for offline tests.
This is *not* a new inference run or an independent holdout.

**Confirmed by the shared PowerShell console output:**

- `python -B -m pytest -q tests` in `.venv`: **180 passed**.
  The earlier `dnai311` pytest invocation failed because that isolated
  scientific environment does not include pytest; this was a setup issue,
  not a test or audit failure.
- Both source records existed. The investigator captured SHA256 before the
  audit and checked them again afterward; **neither JSON SHA256 changed**.
- The command exited successfully, and the returned JSON parsed successfully.
- `morphology_reproduced_exactly`: **FP512 true; KL720 true**.
- Historical primary matching: **14 FP512 valid, 12 KL720 valid,
  11 matched, 3 FP512 unmatched, 1 KL720 unmatched**, precision
  0.9166666666666666, recall 0.7857142857142857,
  F1 0.8461538461538461. Historical IoU threshold remains 0.5.
- FP512 image-edge grouping (fixed **<=32-pixel** descriptive cutoff):
  **3/3 unmatched** versus **0/11 matched** near the 1024x1024 image edge.
- Unmatched FP512 source classes: **6 class-presence tests** (2 classes
  per unmatched valid fiber); **3** had zero KL720 same-class pixels on
  source support and **3** had zero same-class pixels in the padded ROI.
- The following raw/final foreground retention is measured only on each
  source fiber's own reconstructed colored pixel support:

| Source-side unmatched local fiber ID | Source bbox edge distance (px) | Opposite-runtime raw foreground retained | Opposite-runtime final foreground retained |
| --- | ---: | ---: | ---: |
| FP512 1 | 0 | 0.0% | 0.0% |
| FP512 15 | 22 | 47.27272727% | 56.36363636% |
| FP512 17 | 0 | 25.0% | 43.18181818% |
| KL720 19 | 92 | 0.0% | 0.0% |

The KL720-specific unmatched ID 19 is **not** close to the image edge
under the same 32-pixel definition. Its zero opposite-runtime foreground
retention is relative to *its own KL720 support coordinates*, not a claim
that no corresponding physical or biological structure exists anywhere in
the FP512 field. Dilation increased surviving FP512-source support foreground
retention for IDs 15 and 17, without restoring primary reconstructed-fiber
correspondence at the unchanged IoU threshold.

**Important reporting distinction:** The console default object formatting
truncated the detailed KL720 group, KL720 per-class disappearance totals,
individual color-specific metrics and probability-margin tables. Thus this
entry records the **confirmed core reproducibility and displayed
descriptive results**, not independent manual review of every per-class
probability margin. The earlier corrected exploratory v2 margin diagnostic
passed separately. The historical matched length/ratio results are in the
preceding v0.12 correspondence record and were not printed from the new
audit in this acceptance console capture.

To inspect the remaining fields without rerunning the audit, reuse the
in-memory `$Audit` object from the successful session:

```powershell
$Audit.boundary_association.PSObject.Properties | ForEach-Object {
    [pscustomobject]@{ SideGroup = $_.Name; N = $_.Value.n;
        Near32px = $_.Value.n_within_32px;
        DistancesPx = ($_.Value.distances_px -join ',') }
} | Format-Table -AutoSize

$Audit.unmatched_class_disappearance.PSObject.Properties | ForEach-Object {
    [pscustomobject]@{ Side = $_.Name;
        Tested = $_.Value.unmatched_source_classes_tested;
        ZeroOnSupport = $_.Value.n_zero_target_class_on_support;
        ZeroInROI = $_.Value.n_zero_target_class_in_padded_roi }
} | Format-Table -AutoSize

foreach ($side in @('fp512_source','kl720_source')) {
    $Audit.per_fiber.$side | Where-Object { -not $_.primary_matched } |
        ForEach-Object {
            $fiber = $_
            foreach ($class in $fiber.per_class) {
                [pscustomobject]@{
                    Side = $side; FiberID = $fiber.local_fiber_id;
                    ClassID = $class.class_id;
                    SupportPixels = $class.source_fiber_support_pixels;
                    TargetRawClassPixels = $class.target_raw_same_class_on_support;
                    TargetFinalClassPixels = $class.target_final_same_class_on_support;
                    FinalZeroInROI = $class.target_final_class_absent_in_padded_roi;
                    DeltaMarginVsBg = $class.probability_margins.against_background.delta_kl720_minus_fp512_mean
                }
            }
        }
} | Format-Table -AutoSize

$Audit.historical_instance_matching.primary.length_decomposition_um
$Audit.historical_instance_matching.primary.matched_length_mae_um
$Audit.source_fp512.sha256
$Audit.source_kl720.sha256
```

Do not convert the 3/3 versus 0/11 edge association into a claim of
overlapping-window, stitching, image padding, or quantization causation.
There is no independent per-window ablation, causal intervention or
multi-image biological-fidelity analysis here. The frozen validations
and pre-existing scientific conclusions remain unchanged.

## Synthetic tests

`tests/test_dnai_failure_modes.py` exercises exact class-priority dilation,
morphology mismatch failure, disappearance of one class, retained alternate
color, raw/after-dilation differences, margins, support-aware fragmentation,
bbox boundary distances, input-contract failures, provenance/no-write
behavior and scientific-worker contract. CI runs on Python 3.9 and 3.11,
with no KL720 device or DNAi model needed.

Success means **diagnostic code and source-contract reproducibility**, not
scientific equivalence. For independent radiation-biology validation, use a
separate preregistered multi-field human-supported experiment.
