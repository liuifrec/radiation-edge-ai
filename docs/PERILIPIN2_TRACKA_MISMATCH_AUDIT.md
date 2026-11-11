# Track A Windows reference mismatch: read-only diagnosis

## Evidence received on 2026-11-11

The Windows real-image run completed brown-object measurement but retained
`TRACKA_REFERENCE_MISMATCH` at exact comparison. Its uploaded comparison reports:

- All fourteen decoded TIFF arrays match (both fields, all three object-label
  rules, DAB/H optical-density arrays, tissue and analysis-valid masks).
- T01: all six field-summary rows match.
- T03: all 803 nuclear-audit rows match.
- T04: all 24 patch-review-plan rows match.
- T05: both provenance rows match, allowing six expected absolute-path relocations.
- T02 has 62,420 rows in both versions, with 73 reported differing table cells.
  These are catalog rows across three rule sets, not independent biological
  replicates. Differing cells are not necessarily distinct objects.
- The report includes only the first forty differing cells. Every displayed
  difference is in `axis_ratio` or `major_axis_px`, but the uploaded report
  does NOT contain the reference/actual values, delta magnitudes, or a complete
  listing of the other 33 cells. Do not claim all 73 differences are rounding.
- The six recorded scientific package versions match the reference versions.
  Saved M1 masks were reused; there was no new nuclear segmentation or KL720 run.

The result localizes the next investigation to derived object measurements.
It does not establish the numerical cause or permit the old receipt to be
rewritten as a pass. The analysis thresholds and exact comparator are unchanged.

## Added diagnostic, not a new measurement implementation

`scripts/windows/audit_perilipin2_tracka.py` accepts the verified reference
bundle and an existing completed result folder. It checks input and result
file manifests, run/bundle/source/configuration identity, and reproduces the
original exact comparison from the saved files. It never calls the detector,
loads a new neural model, installs packages, or invokes Docker/KL720.

It reads every T02 row by the existing `(image_uid, rule, object_id)` key and
writes ALL differing cells into `ALL_OBJECT_CELL_DIFFERENCES.tsv`, with:

- Original reference and Windows value strings; no rounding is applied.
- Absolute decimal difference and symmetric relative difference.
- Diagnostic binary64 ULP distance (floating-point representable steps).
  A zero ULP distance does not erase a difference between decimal strings.
- For the two morphology columns, the recorded gate and whether the stored
  values differ under the original strict `>` predicate.

`TRACKA_MISMATCH_AUDIT.json` summarizes per-column maxima, affected row count,
exact status/reason agreement, non-morphology differences, and threshold
crossings. Comparisons use the stored exported values; they do not reconstruct
unrecorded intermediate floating-point state. ULP/delta statistics are purely
descriptive, not new acceptance tolerances.

The original result, input bundle, measurements, thresholds and receipts remain
untouched. Diagnostics go to a new uniquely named sibling directory unless an
explicit new `--out` is selected. Existing output directories are refused.

## Run on the Alienware

No new environment setup or real-image measurement rerun is required. Use the
already verified Track A scientific environment, not the older launcher/SDK
interpreter:

```powershell
& {
    $ErrorActionPreference = 'Stop'
    Set-Location 'C:\Users\yul03\Projects\radiation-edge-ai-desktop'
    git fetch origin feature/perilipin2-tracka-analysis-v1
    if ($LASTEXITCODE -ne 0) { throw 'Fetch failed.' }
    git merge --ff-only FETCH_HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Update refused; preserve local changes.' }

    $Python = 'D:\radiation-edge-ai-data\envs\perilipin2-tracka-py311-v2\Scripts\python.exe'
    $Bundle = 'D:\radiation-edge-ai-data\references\Perilipin2_TrackA_portable_20261111_v1'
    $Result = 'D:\radiation-edge-ai-data\scratch\Perilipin2_TrackA_windows_20261111T054214Z_3f7dbcdb'

    & $Python -X utf8 -u .\scripts\windows\audit_perilipin2_tracka.py `
        --bundle $Bundle --run $Result
    if ($LASTEXITCODE -ne 0) { throw 'Diagnostic audit stopped; preserve the output.' }
}
```

Expected diagnostic status for the currently known result:
`AUDIT_COMPLETE_EXACT_MISMATCH_REMAINS`.
This means the audit completed, NOT that the original acceptance gate passed.
The terminal prints per-column magnitudes, classification agreement, threshold
crossings and the exact report paths. Those outputs are the next evidence to
review. Share the audit summary/TSV privately; no image or model upload is needed.

## Test scope and next decision

Seventeen local diagnostic tests passed using synthetic table fixtures and
mocked bundle/array verification. They cover all-difference enumeration beyond
forty entries, decimal versus binary64 differences, strict-gate crossings,
status changes with unchanged totals, missing/nonfinite values, duplicate or
changed row keys, immutable prior receipts and failed-input guards. These tests
are also included in Windows CI and the full Linux suite. They do not validate
the magnitude/cause of the actual 73 discrepancies.

Only after reviewing all actual values can we decide whether a separately
versioned numerical-portability criterion is justified. Any such future policy
must retain this exact mismatch record and require unchanged object identities,
classification decisions and endpoints; it must not be chosen to hide an
unexplained discrepancy or change biology to obtain a pass.
