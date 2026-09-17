# NASA endpoint reconstruction v0.2

## Scope

v0.2 adds deterministic, content-addressed reconstruction of the NASA BPS R1 v2 sample-level endpoint from frozen per-nucleus latent 53BP1 burden predictions.

Per-nucleus output remains a latent continuous burden score, not an individually supervised 53BP1 focus count.

The reconstructed endpoint is the arithmetic mean latent burden across nuclei belonging to one biological sample.

Endpoint reconstruction is distinct from biological-fidelity evaluation.

## Frozen physical KL720 reconstruction

Source physical prediction SHA256:
91ea895a16cd97067af63b5c5c4434e972629bfda4c9582ad456a79e007b6d46

Application aggregation ID:
a1-1cd7571ca6413d47

2058 frozen physical-KL720 nucleus predictions were reconstructed into 22 sample-level endpoints.

Generated endpoint_report.json SHA256:
2d4fd8923fc375cd3480030b58ab62d9e9f10ed0b6927c9cd85b574f5f5b80f3

Generated sample_aggregates.csv SHA256:
bb222942f55fad326f15faaa65612260d659357affebf8bedb92e5e13c87e116

Historical physical sample endpoint SHA256:
61cfd195b22f9bdb427697cac75e25fe1d0d94e1b4a9beeb886eb6ad9eb8ca25

Reconstruction result:
- sample identities: 22/22
- nucleus-count mismatches: 0
- metadata mismatches: 0
- exact floating-point matches: 22/22
- maximum absolute difference: 0.0
- mean absolute difference: 0.0

## Evidence freeze

External evidence freeze SHA256:
ec2e3bc7795ac07470ca99fe23d5ef6cf9f0fe9379729d3e645bad1933addc34

The external freeze remains outside Git with the biological data and deployment artifacts.

## Scientific interpretation

This demonstrates application-layer endpoint reconstruction reproducibility.

It is not a new biological-fidelity evaluation, new holdout analysis, or new claim of statistical equivalence.

The previously frozen physical KL720 biological validation remains authoritative for biological deployment performance.
