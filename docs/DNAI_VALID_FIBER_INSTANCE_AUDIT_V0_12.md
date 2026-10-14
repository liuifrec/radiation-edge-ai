# Verified DNAi reconstructed-fiber instance audit (v0.12 development)

`radedge compare-dnai-fibers` adds a separate, **read-only, source-verified**
comparison of actual DNAi *valid reconstructed fibers* in matched frozen FP512
(`df1`) and physical KL720 (`df2`) fields. It is complementary to the existing
`compare-dnai-fields` pixel/connected-component proxy audit, not a replacement.

## Command (Windows PowerShell)

```powershell
$SciencePy = 'D:\radiation-edge-ai-data\envs\dnai311\Scripts\python.exe'
radedge compare-dnai-fibers --fp512 '<df1-field_result.json>' `
    --kl720 '<df2-field_result.json>' --scientific-python $SciencePy --json
```

The scientific Python must already contain the frozen upstream `dnafiber`
version plus `cv2`, `numpy` and `pandas`. It is intentionally **not** added
to the control-plane's minimal runtime dependencies. The command launches a
separate worker using that interpreter and the current repository's `src`
module path; no source data or output files are written. Do not use an
untrusted interpreter path. The worker has a 180-second timeout.

## Verification before comparison

The scientific worker re-verifies both complete source-field records and all
bound artifacts with the existing v0.11 verifiers. It requires the two records
to have **identical** nine deployment inputs, preprocessing policy, upstream
commit, source validation manifest, image identity and image index. It checks
the actual frozen `dnafiber` software version; the saved segmentation image and
fiber CSV are read only from hash-checked bytes. The worker regenerates
`refine_segmentation(segmentation).valid_copy().to_df(...)`, checks both total
and valid fiber counts, compares the entire output table with its frozen CSV
(tolerance 1e-10), and checks reconstructed total lengths against the original
field summaries. It checks source JSON SHA256 before and after analysis.

A failure **aborts the comparison**. This does not run ONNX/KL720 inference,
retune the model, change the segmentation, or create a replacement scientific
record. JSON emitted to stdout is path-bearing ephemeral inspection.

## Comparison contract

- The primary matcher is the pre-existing, deterministic greedy one-to-one
  *reconstructed-fiber bounding-box* IoU matcher at the historical threshold
  of **0.5**. Ties use ascending reconstructed-row indexes.
- The pre-existing maximum-cardinality, then maximum-total-IoU matcher is
  reported **separately as sensitivity**, including whether match *counts*
  and individual index pairs are identical. No matching threshold is optimized.
- CSV `Fiber ID` values are *local* IDs **not** cross-runtime correspondences.
  Both local IDs and zero-based reconstructed row indexes are shown for every
  spatial match and every unmatched reconstructed valid fiber.
- Length and ratio errors are calculated **only among matched valid fibers**.
  Non-finite ratios are excluded from ratio MAE with an explicit denominator.
- The total valid-fiber-length difference is decomposed without remainder:

  `KL720 minus FP512 = matched length deltas + unmatched KL720 length - unmatched FP512 length`.

- An unmatched valid reconstruction is not proven biologically absent. Fiber
  splits, merges, invalid/rejected reconstructions, movement of inferred
  bounding boxes, and overlap below threshold can contribute. No split/merge
  *classification*, new annotations, radiation effect or biological fidelity
  assessment is performed.

## Initial real frozen single-field preflight (2026-10-14)

Input field IDs: `df1-72c2102e46e5a4af` and `df2-d3955d46bdebf2c1`.
Both field records passed full source verification. Scientific DNAi version
was identical (`1.1.dev10+gfcf20c7d6.d20260901`), and regenerating the
frozen valid-fiber CSV tables succeeded without modifying source evidence.

| Descriptive single-field result | Value |
| --- | ---: |
| All DNAi reconstructions | 18 FP512 / 30 KL720 |
| Valid reconstructed fibers | 14 FP512 / 12 KL720 |
| Valid-fiber bounding-box IoU matches (greedy, threshold 0.5) | 11 |
| Unmatched valid reconstructions | 3 FP512 / 1 KL720 |
| Precision / recall / F1 using FP512 as reference | 0.916667 / 0.785714 / 0.846154 |
| Maximum-cardinality sensitivity match count | 11 |
| Matched valid-fiber length MAE | 0.3187430143 um |
| Matched valid-fiber ratio MAE | 0.0696598154 |
| Matched-fiber net KL720 minus FP512 length delta | +1.2107821049 um |
| FP512 unmatched valid-fiber length | 45.9511168314 um |
| KL720 unmatched valid-fiber length | 1.82 um |
| Net KL720 minus FP512 total valid length | -42.9203347266 um |

**Interpretation:** In this *one previously characterized development field*,
most of the lower KL720 total valid-fiber length results from the changed set
of matched/valid reconstructed fibers, not a net shortening among the 11
spatial matches. The preflight has not established equality of the two sets
of index pairs selected by the primary and sensitivity matcher; the permanent
command will report that separately. This cannot support a population-level
biological-fidelity PASS, independence claim or measured irradiation effect.
The historical 20-image development characterization remains separately
reported in `docs/DNAI_PHYSICAL_CHARACTERIZATION_AUDIT.md`.

## Permanent command real-data acceptance (2026-10-14)

The new compare-dnai-fibers command passed its real frozen-field
acceptance gate using the previously verified FP512 and physical
KL720 records.

- Valid reconstructed fibers: 14 FP512, 12 KL720.
- Historical greedy bbox-IoU threshold: 0.5.
- Matched reconstructed valid fibers: 11.
- Unmatched reconstructed valid fibers: 3 FP512, 1 KL720.
- Matching F1: 0.8461538461538461.
- Matched length MAE: 0.3187430143001842 um.
- Matched ratio MAE: 0.06965981538479663.
- KL720 minus FP512 total valid length: -42.920334726568484 um.
- The maximum-cardinality sensitivity produced identical match pairs.
- Both frozen valid-fiber CSV tables were reproduced.
- The length decomposition reconciled exactly within numerical tolerance.
- Original field-record SHA256 values remained unchanged.

This confirms reproducibility of a single previously characterized
development field, not independent biological validation. Matches
represent spatial correspondence between reconstructed DNAi objects,
not independently established biological fiber identities.
