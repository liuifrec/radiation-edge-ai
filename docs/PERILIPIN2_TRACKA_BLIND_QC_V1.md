# Perilipin2 Track A: independent blinded patch QC v1

## Purpose and status

This workflow moves from **numerical portability** to **manual scientific validation** without changing the frozen M2 detector, its thresholds, its reference images or saved M1 nuclear masks. It operates on the existing **two-field X500/X40 Windows reference** and the fixed random tissue-grid patch plan recorded by M2.

The earlier real-image run retained `TRACKA_REFERENCE_MISMATCH` because 73 cells of derived morphology descriptors were not decimal-identical. The separately verified `PORTABILITY_INVARIANTS_PASS_EXACT_MISMATCH_RETAINED` result found 14/14 arrays exact, four other tables exact, unchanged catalog identities/status/reasons, and zero recorded threshold crossings. **Both original receipts remain unchanged.** This new QC work tests object definitions and false detections; it is **not** KL720 deployment.

## Exactly which images are reviewed

The frozen M2 patch plan selected **12 non-overlapping tissue-grid patches per field**, each 224 × 224 source pixels with an 8-pixel unscored context margin. There are 24 patches total: 6 `calibration` and 6 `held_out_validation` from each field. Selection used the original deterministic image-dependent random seed and tissue eligibility, **not candidate density or treatment labels**.

The generator verifies the original bundle, run, all fourteen image arrays, four non-catalog tables, and the catalog's object/decision invariants. It then crops **original RGB pixels without recoloring, generating or modifying features**. No prediction outline, treatment dose, patch role, sample ID or image ID is included in the **reviewer-only** folder.

- `blind/annotate.html`: offline, original-pixel review tool; no internet or model required.
- `blind/crops/R001.png` through `R024.png`: anonymous random-order source crops.
- `private/REVIEW_MAP.json`: original patch IDs, coordinates, selection roles and field labels. **Do not give this folder to an independent reviewer.**
- `QC_PACK_MANIFEST.json`: checksums, run identity and generation metadata. Research pixels and maps must remain on laboratory storage, never GitHub.

These measures conceal metadata in the reviewer UI, but **are not proof of formal experimental blinding** if a reviewer already knows or recognizes the source images. The preparation scientist can only achieve independent blinding by distributing the `blind/` subfolder alone.

## Prepare locally on the Alienware

The desktop worktree is a separate worktree and should remain independent of the compiler/Perilipin2 ONNX worktree. No Docker/PLUS Python/device access, new Python environments, dependency installs or new image measurement are needed.

Open Windows PowerShell:

```powershell
$ErrorActionPreference = "Stop"
Set-Location "C:\Users\yul03\Projects\radiation-edge-ai-desktop"

git fetch origin feature/perilipin2-tracka-blind-qc-v1
if ($LASTEXITCODE -ne 0) { throw "Fetch failed." }

git merge --ff-only FETCH_HEAD
if ($LASTEXITCODE -ne 0) { throw "Non-fast-forward update refused; preserve local changes." }

$Pack = "D:\radiation-edge-ai-data\scratch\Perilipin2_TrackA_blind_QC_v1"

.\scripts\windows\run_perilipin2_tracka_qc.ps1 -Stage Prepare -Out $Pack

if ($LASTEXITCODE -ne 0) { throw "QC packet preparation stopped." }
```

The command should finish with `TRACKA_BLIND_QC_PACK_READY` and print an `annotate.html` location. The PowerShell wrapper uses the existing, previously verified `D:\radiation-edge-ai-data\envs\perilipin2-tracka-py311-v2\Scripts\python.exe` and the frozen bundle/run paths by default. If your paths differ, pass `-Python`, `-Bundle` or `-Run` explicitly.

**If a packet already exists, choose a different NEW `-Out` directory.** No existing packet or receipt is overwritten.

For an integrity-only check before review:

```powershell
.\scripts\windows\run_perilipin2_tracka_qc.ps1 -Stage VerifyPack -Pack $Pack
```

## Review instructions

Open `$Pack\blind\annotate.html` in your regular browser. Use only the anonymous patch images during independent annotation. Each click toggles a cyan cross at a proposed **discrete compact brown DAB-positive object's center**. The yellow inner rectangle defines what is scored; the outer 8 pixels provide context only.

Do not force diffuse staining, obvious merged structures, rings, or debris to be compact objects. Document disagreements, ambiguity and any uncertain structures in Notes. A patch with zero marks is a verified **negative** only after the reviewer deliberately checks **“This patch is exhaustively reviewed.”** Every added/removed point resets that checkbox. Incomplete patches never contribute to recall.

The browser deliberately has **no autosave**. Export the anonymous JSON repeatedly and keep it somewhere private. To continue a session, use **Import previous review**. The reviewer JSON stores anonymous crop-relative coordinates and a pack fingerprint; no treatment labels, original image IDs or scoring roles are required in it.

The current two fields are **not biological replicates**. A single reviewer is also not independent expert adjudication. If possible, request a second blinded review of ambiguous cases and reconcile disagreement before accepting final reference points.

## Convert reviewed points and score using the frozen M2 evaluator

After at least one patch is explicitly marked exhaustive, provide the exported anonymous JSON using `-Review`. The scorer verifies the pack and the original source identities again, checks exact positions and margin bounds, converts crop-local points into original image coordinates using the **private** map, then calls the **unchanged frozen M2 one-to-one matching evaluator**. The evaluator compares all three recorded rules separately; no rule is silently selected or tuned.

```powershell
$Pack = "D:\radiation-edge-ai-data\scratch\Perilipin2_TrackA_blind_QC_v1"
$Review = "D:\path\to\TrackA_blind_review_<fingerprint>.json"
$ScoreOut = "D:\radiation-edge-ai-data\scratch\Perilipin2_TrackA_QC_score_v1"

.\scripts\windows\run_perilipin2_tracka_qc.ps1 -Stage Score -Pack $Pack -Review $Review -Out $ScoreOut -Radius 6.0
```

The **6-pixel** radius is the original M2 example/default, not automatically a validated biological match distance. Keep it prespecified, record it, and do not optimize on held-out annotations. For another radius, make an explicitly versioned fresh evaluation.

The scoring folder contains `QC_REVIEW_SCORING.json` plus the frozen original scorer's `T01_patch_detection_evaluation.tsv`, `T02_role_rule_evaluation.tsv`, and `review_provenance.json`. The original measurement run, reference bundle and QC pack are never modified. If fewer than all 24 patches are completed, the receipt says `REVIEW_PARTIAL_SCORING_NOT_BIOLOGICAL_VALIDATION`, not `PASS`. Even complete annotations do not by themselves validate hepatocyte/lipid-droplet biological endpoints or nuclear segmentation.

## Scientific acceptance decisions after scoring

Evaluate **false-positive and false-negative patterns**, particularly on low-signal/blank patches and ring/complex/artifact regions. Interpret the frozen `calibration` versus `held_out_validation` roles separately, quantify uncertainty, and document the object definition and review disagreements. Do not pool random patches as independent biological samples.

Only after that should the team decide whether the existing provisional compact-object detector meets prespecified accuracy requirements. Nuclear-count accuracy needs its own review; current M1 nuclei are reused, not validated here. KL720 Perilipin2 compilation/device inference remains a distinct, incomplete milestone.

## Test boundary

Repository tests use a synthetic frozen patch plan and simulated image crops to exercise identity guards, anonymous-only HTML, role balance, field coordinate conversion, explicit empty-patch completion, rejection of out-of-margin/duplicate points, checksum checks and the frozen-scorer dispatch contract. **These CI tests contain no microscopy pixels and do not measure detection accuracy.** Windows PowerShell syntax is checked in CI; actual review and scoring require the local real-image packet.
