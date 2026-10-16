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
$FP512 = '<existing verified df1 field_result.json absolute path>'
$KL720 = '<existing verified df2 field_result.json absolute path>'

# First rerun synthetic tests without accessing frozen science data.
& $SciencePy -B -m pytest -q tests/test_dnai_failure_modes.py
& $SciencePy -B -m pytest -q tests
# One read-only real-data acceptance run, not inference.
& $SciencePy -B -m radiation_edge_ai.cli audit-dnai-failure-modes `
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
