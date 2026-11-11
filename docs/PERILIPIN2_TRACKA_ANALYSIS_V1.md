# Perilipin2 Track A analysis integration v1

## Delivered scope

This adapter **executes the recovered Track A M2 measurements from RGB pixels**.
It does not just display old tables. It reuses the saved M1 nuclear-instance
labels and annotation-exclusion masks, then regenerates the field/object/nuclear
tables, all three exploratory rule sets, quantitative arrays, colored QC and
the original offline review viewer. It compares the new scientific values with
the frozen M2 reference, with no numeric tolerance.

This is the first application-integration acceptance step: the two original
X500/X40 fields (#13, 0 Gy; #27, 3.8 Gy). It is **not** an unrestricted batch
release, new nuclear segmentation, whole-cell identification, lipid-droplet
ground truth, biological replication, or a KL720 deployment. The Perilipin2
compiler branch and its synthetic models are not used by this analysis.

The module is `radiation_edge_ai.perilipin2_tracka`. The desktop has an explicit
`Perilipin2 Track A reference (CPU, saved nuclei)` action. That action uses the
selected Scientific/ONNX Python, not the Kneron PLUS interpreter. Existing
DNAi/NASA adapters and their scientific algorithms are unchanged.

## Recovered reference, not invented replacement thresholds

The exact source package was recovered from the existing microscopy handoff:

- `EdgeAI_Microscopy_Handoff_20261030.zip` SHA256:
  `caa052667320eff5784933844dcc25e5877f0e0d9922037953b2a0a5eb58afde`
- Nested `Perilipin2_TrackA_M2.zip` SHA256:
  `68d3a682d63eb6497b0105ac0f10199bf3a050487fbb0a1bbc93f63fea13fdaf`
- Original `run_tracka.py` SHA256:
  `b92659b7784233695ebc708da34e975b1cc39195c64d45076a76b585aa96dd9c`
- Original `viewer.html` SHA256:
  `a86a75f15acfd7ea5bacaa2d6a91f2cad2c490fc7d7be51621f1151461a4ca6e`

The adapter loads the byte-verified original source as an external dependency.
It calls the original `process_field` function; it does not replace stain
separation, tissue eligibility, candidate rules, connected components, nuclear
audit, contours or patch selection. The configuration is copied from the actual
M2 `RUN_CONFIG.json`, not selected to favor a treatment direction. The original
source archive and all scientific inputs remain outside Git.

The transport and report-assembly code is new. It uses explicit UTF-8, portable
asset references and a new dataset ID. Five tables are compared by their row
keys and exact numeric/text values; only the three absolute-path columns in
T05 may relocate. Fourteen TIFF arrays are compared by decoded shape, dtype
and values. HTML bytes, output paths, new run IDs and TIFF container metadata
are not scientific equivalence criteria.

## Measurement contract

Main endpoint: **provisional compact brown objects per M1 nuclear proposal**.
Nuclei are counted independently when their centroids fall in eligible tissue.
The detector searches eligible tissue, not guessed cell territories. Nuclear
labels still inform nuclear-overlap review, so changing the nuclear backend
later can affect both numerator classification and denominator.

The original M2 categories and secondary endpoints remain available: rings,
complex structures, nuclear overlaps, dark artifacts and edge objects; tissue
area, brown-support area fraction, object density per 100,000 tissue pixels,
and H-supported nuclear-denominator sensitivity. Rings are review objects,
not automatically false positives. A zero denominator is not quantifiable,
not a zero ratio. Physical units require recorded calibration; magnification
and JPEG DPI are not substituted for micrometers per pixel.

Preserving these rules does not validate their real-image accuracy. The low-
staining image and ambiguous structures still require independent inspection.
No statistical significance or biological N=3 is inferred from multiple fields.

## One-time source export where the original files reside

The actual X500 image pair, M1 masks and full M2 outputs are on the user's
workstation. The recovered source and old replay receipts alone do not supply
those pixels. `prepare` verifies all original M1/M2 byte manifests and exports
a portable folder without rerunning inference or modifying originals.

Use this module from the integration worktree or the supplied source ZIP:

```bash
python3 src/radiation_edge_ai/perilipin2_tracka.py prepare \
  --handoff "$HOME/Downloads/EdgeAI_Microscopy_Handoff_20261030.zip" \
  --m1 /home/liuyuchen/DNA_fiber_assay/analysis/Perilipin2_DAB_M1 \
  --m2 /home/liuyuchen/DNA_fiber_assay/analysis/Perilipin2_TrackA_M2_results \
  --images /home/liuyuchen/DNA_fiber_assay/ImmuoF \
  --out /home/liuyuchen/DNA_fiber_assay/analysis/Perilipin2_TrackA_portable_20261111_v1
```

This operation uses only the Python standard library. It copies only the
selected images, masks, reference tables/arrays and pinned source needed by
this two-field acceptance test, rather than the entire microscopy collection.
It records the reference dependency versions in `reference_requirements.txt`.
An existing output or `.BUILDING` folder is never overwritten.

Copy the **entire resulting folder** using the institution's usual local-file
transfer to, for example:
`D:\radiation-edge-ai-data\references\Perilipin2_TrackA_portable_20261111_v1`.
The folder contains research images and masks; do not commit it to Git or
attach it to a public issue. This is a one-time data export, not a request to
restart GPU development on the workstation.

## Windows: use a separate scientific environment

Do not install analysis packages into the known-good Kneron PLUS environment.
A separately created Python 3.11 environment can use the exact package
versions recorded with the original M2 outputs. Verify the bundle before
installing from its requirements file:

```powershell
# Run from the integration worktree; Python 3.11 must already be installed.
$Bundle = 'D:\radiation-edge-ai-data\references\Perilipin2_TrackA_portable_20261111_v1'
$EnvDir = 'D:\radiation-edge-ai-data\envs\perilipin2-tracka'
if (Test-Path $EnvDir) { throw 'Environment exists: inspect it rather than overwrite it.' }
py -3.11 -m venv $EnvDir
if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
$TrackAPython = Join-Path $EnvDir 'Scripts\python.exe'
& $TrackAPython -X utf8 .\src\radiation_edge_ai\perilipin2_tracka.py verify-bundle $Bundle
if ($LASTEXITCODE -ne 0) { throw 'Input bundle verification failed.' }
& $TrackAPython -m pip install -r "$Bundle\reference_requirements.txt"
if ($LASTEXITCODE -ne 0) { throw 'Recorded analysis dependencies are not ready.' }
```

Dependency installation requires package availability/network access; the
analysis and its embedded viewer run locally without network access. A
version-match does not by itself prove cross-platform numerical equality;
the reference comparison is still mandatory.

### Desktop execution

Use branch `feature/perilipin2-tracka-analysis-v1`, stacked on the draft desktop
launcher branch. Keep the Perilipin2 compiler worktree unchanged. Select:

- Action: `Perilipin2 Track A reference (CPU, saved nuclei)`.
- Scientific/ONNX Python: the dedicated Track A interpreter above.
- Input JSON: the bundle's `TRACKA_INPUT_BUNDLE.json`.
- Output: a NEW directory under the data drive, outside the input bundle.

Preview and run. No PLUS interpreter, Docker or attached accelerator is
required for this action. The ordinary KL720 actions remain separate.
The launcher status is only command completion; inspect `TRACKA_EXECUTION.json`
and `REFERENCE_COMPARISON.json` for the analysis outcome.

### Equivalent command-line execution

```powershell
$Result = 'D:\radiation-edge-ai-data\scratch\Perilipin2_TrackA_windows_reference_v1'
.\scripts\windows\run_perilipin2_tracka.ps1 -Stage Analyze `
  -Python $TrackAPython -Bundle $Bundle -Out $Result
Get-Content "$Result\TRACKA_EXECUTION.json" -Encoding UTF8
Get-Content "$Result\REFERENCE_COMPARISON.json" -Encoding UTF8
```

Expected success status: `TRACKA_REFERENCE_REPLAY_PASS`. A measurement
mismatch produces `TRACKA_REFERENCE_MISMATCH` and exit code 2; measurements
and difference diagnostics are retained. Do not relax thresholds/tolerances
to conceal a platform or source mismatch. An exception leaves its fresh
`.BUILDING` directory and failure receipt for investigation.

## What is newly produced

- `tables/T01_field_measurements.tsv`: independent counts, normalized values,
  tissue/brown areas and all three recorded exploratory rules.
- `tables/T02_brown_object_catalog.tsv`: object coordinates, morphology,
  stain/color evidence and explicit review categories.
- `tables/T03_independent_nuclear_audit.tsv`: the saved nuclear proposals,
  tissue inclusion and support/boundary audit.
- `tables/T04_random_patch_review_plan.tsv`: original deterministic patch roles.
- `tables/T05_input_provenance.tsv`: exact source/mask identities and new paths.
- `per_field/*/QC_overlay.png`, original-image views and quantitative TIFFs.
- `index.html`: the unchanged, embedded-data M2 viewer, generated anew.
- `TRACKA_EXECUTION.json`, `REFERENCE_COMPARISON.json`, `FILE_MANIFEST.tsv`.

Share a reviewed whole results folder, not just its HTML. The input bundle
is separate. Original image pixels are preserved; there are no generative
redrawings or new cell-boundary estimates.

## Independent object review and scoring

The original viewer supports exhaustive annotation of selected tissue patches.
Export its review JSON explicitly. A patch with no points is a negative only
when it is marked exhaustively reviewed. Candidate-only labels cannot estimate
recall. The scoring operation uses the frozen M2 one-to-one matching method;
it does not modify measurements or tune the working rule.

```powershell
& $TrackAPython -X utf8 .\src\radiation_edge_ai\perilipin2_tracka.py score `
  --bundle $Bundle --run $Result --review 'D:\path\review.json' `
  --radius 6 --out 'D:\radiation-edge-ai-data\scratch\TrackA_review_eval_v1'
```

The example radius is the original M2 default in source pixels, not a validated
biological acceptance margin. Prespecify the review/matching protocol before
scoring and keep calibration versus held-out review roles separate.

## Verification performed and still required

Local tests used the actual recovered source and entirely synthetic images:
**17 adapter tests and 5 subtests passed**, including end-to-end relocation,
5/5 table agreement, 14/14 decoded-array agreement, counts on known synthetic
dots/ring/line, a blank field, NQ denominator handling, changed-mask refusal,
result tampering and incomplete-review rejection. The recovered source's
own **23 tests passed** separately. All source inputs remained byte-identical.
A generated colored QC PNG was visually inspected. This is not a new real-
image, Windows hardware or biological accuracy result.

The normal public CI has no external frozen source ZIP. The nine numerical
integration tests explicitly skip when `RADEDGE_TRACKA_HANDOFF` is absent;
stdlib guards and desktop dispatch tests still run. To execute the full local
suite, set that variable to the verified handoff ZIP and install the scientific
libraries. CI test counts must not be described as real microscopy validation.

Remaining acceptance: copy the real source bundle, rerun the two fields on
Windows, inspect exact-comparison results and actual QC, review representative
positive/negative/difficult patches, then expand the input contract to new
fields. New nuclear segmentation, validated KL720 nuclear masks, multi-sample
statistics and broader biological claims remain separate work.
