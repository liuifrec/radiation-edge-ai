# NASA BPS R1 v2 physical KL720 holdout inference — outcome-blind result

The exact frozen NASA R1 v2 NEF was run on the physical KL720 over the complete QC1 final holdout before any physical-device predictions were joined to biological targets or reference outputs.

## Frozen upstream identities

- NEF SHA256: `d04f02855a8a82ae7cb5ff48af91eee69c341142d1097469890385bd23ee097b`
- NEF deployment freeze SHA256: `f818b9d912017a1d8fdc4aafd4bf321b8bf5b84e8214e6ef0ba029b48b0b90bb`
- physical one-sample smoke PASS freeze SHA256: `224b754235ebcd22eb053ca1f279043f3b74e9e3b461ed1f8dad26beddfa679d`
- QC1 final-holdout manifest SHA256: `f89962f96bec05930c07149717a90bd682016a2778bbfae6b827425159aa0643`

## Physical inference result

- nuclei / bags: 2058 / 22
- physical latent-burden range: `[0, 4.9579887]`
- mean hardware send+receive: `3.676 ms`
- median hardware send+receive: `3.636 ms`
- p95 hardware send+receive: `4.269 ms`
- physical prediction CSV SHA256: `91ea895a16cd97067af63b5c5c4434e972629bfda4c9582ad456a79e007b6d46`
- physical inference summary SHA256: `7147ba4de4ba44fa64ed6b414e990970cd08ac83ea5e4b993ed38e9ad0a93aba`

The physical stage read neither the raw phenotype table nor the frozen FP32/BIE holdout reference outputs, and it did not evaluate any biological target or gate. PTQ/BIE/NEF were unchanged, and post-holdout PTQ retuning remains unauthorized.

A first attempted physical full-holdout invocation stopped before nucleus inference because the host `--image-root` argument pointed one directory above the materialized `images/` directory. The corrected invocation used the unchanged frozen runner/model/data and completed all 2058 nuclei. This was a host path invocation error only; it did not expose outcomes or generate a partial prediction artifact.

The complete physical prediction set must be frozen before biological evaluation. `freeze_r1_v2_physical_holdout_predictions.py` pins the prediction CSV and summary hashes and audits prediction identities/metadata against the QC1 manifest without reading phenotype/reference outcomes.
