# Independent-source evaluation of the ResNet18 53BP1 endpoint

**Study status — design prepared, enrollment pending.** The selected female-fold
BIE has SHA256 `3f654f267ae280f27d6aaf6715b32af4285c5de6f4eb13c844e02e70f2f69c9b`.
The target quantity is mean 53BP1 burden per nucleus in a biological sample.
NASA supplies sample-level reference means. The engineering regression on the
two previously examined female sources evaluates reproducibility of the FP32
endpoint through quantization. This design asks how accurately the locked
predictor measures the same endpoint on previously unseen biological sources.

## Register eligible material before scoring

Recruit newly identified biological sources from the intended female population.
Define the target strains, acquisition settings and intended deployment
population with the assay lead before enrollment. Register the original
source identifiers and aliases in a private crosswalk and reject overlap
with the six development and three historical reserved sources on the existing
metadata-only exclusion list. Track re-imaged wells, technical replicates and
sister specimens under their original biological source. The project inventory
contains no eligible registered sources. The historical final holdout stays
sealed. [Inventory and eligibility scope](NASA_PRECISION_TRAINING_PROTOCOL_V1.md)

Plan complete matched sham and exposed samples within source, radiation branch
and time point. For comparability with the original study the strata are Fe
0.82 Gy and X-ray 1.0 Gy, each at 4, 24 and 48 hours, with their corresponding
zero-dose controls. Record plate/well identity, acquisition batch, physical
pixel size, channel identities and preprocessing version. Balance acquisition
batches across study conditions. Document replicate wells and any shared
controls so the analysis preserves their dependence.

## Seal the reference and model independently

1. Assign analysis IDs and freeze the source-to-sample crosswalk, eligible
   fields, nucleus inclusion and sampling rule, image-quality exclusions,
   required pairing and missing-data handling before inference. For comparison
   to the engineering panel, use the established sampling target of 100 nuclei
   per sample when feasible. Predetermine how undersized samples contribute.
   Record exclusions and attrition by source and condition.
2. Designate an independent reference team. Record the protocol for producing
   sample `avg_nfoci` values, its units, grader blinding, repeatability and
   adjudication method. Blind graders to predictions and withhold the reference
   table from the inference team until the predictions and run receipt are sealed.
3. Freeze one binary and execution backend, its verified input pipeline, per-file
   hashes, registration timestamp and analysis code. Specify the BIE or physical
   measurement as the primary implementation before scoring. Give each additional
   backend a separate identity and fidelity analysis. Record any acquisition
   deviation or backend failure against the enrolled sample.
4. Seal one prediction per included nucleus and one arithmetic mean per sample.
   Link prediction and reference by immutable sample ID. A mismatch remains
   visible in the enrollment accounting. Freeze the pair table before unblinding
   source-level summaries.

## Specify the analysis and required precision

Use the biological source as the independent unit. For each sample compare
predicted and reference means in foci/nucleus. For each prespecified pair,
compute exposed minus sham on each backend and against the reference. Average
replicated wells within the prespecified source and condition before combining
sources with equal weight. The primary estimand is the mean across sources of
within-source sample MAE. Report its uncertainty, sample signed bias, paired
contrast errors, and signed four-hour responses by radiation branch. Show per-source
values, missing pairs and the numbers of sources, samples and nuclei.

Before enrollment, have the assay lead sign off a scientifically relevant
maximum error, the desired confidence interval precision, a variance estimate
from independent reference-repeatability or planning data, and the planned
number of biological sources. For a precision-based design, use the source-level
dispersion of the primary summary and calculate the number of independent
sources needed for the agreed interval half-width. Allow for attrition and
check adequacy within the target strain strata. Predeclare a source-level
interval method and resample whole sources together with all their paired
conditions if using a clustered bootstrap. Limited source counts warrant
reporting the observed source results and uncertainty explicitly.

The eight engineering limits retain their existing scope. Biological
qualification requires a prospective decision rule, adequate source enrollment,
a blinded reference and a completed independent readout. A failed independent
study stays attached to this model and enrollment. Any subsequent revision
requires a new candidate and a new independent cohort.
