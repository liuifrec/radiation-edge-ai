# NASA assay-run operator interface v0.6

## Purpose

v0.6 adds a user-facing manifest-to-measurement entry point over the existing verified control-plane layers.

The operator supplies one batch manifest and invokes assay-run. The application creates or reuses the content-addressed batch plan, executes or reuses the v0.5 transaction, verifies the complete provenance chain, and presents the resulting assay endpoint.

v0.6 introduces no new inference implementation, prediction transformation, endpoint calculation, biological-reference lookup, biological acceptance logic, or durable provenance type.

The v0.5 transaction record remains the provenance authority.

## Application path

batch_manifest.json
-> radedge assay-run
-> content-addressed batch plan
-> existing v0.5 transaction
-> verified transaction_record.json
-> human-readable or JSON scientific summary

## Real NASA three-nucleus operator smoke

Semantic batch ID:
b1-dfe95904d1991502

Transaction ID:
t1-387be2e6fe7b3a0c

Prediction projection ID:
p1-0b7b117fb35ecba6

Measurement ID:
m1-06a26289fd977ec2

Aggregation ID:
a1-811ca4d3d38350d1

Batch-plan SHA256:
7151a16459e0440827e6dd8fac477d7180af6d900d2efb7869b49593cd9cd2cd

Transaction-record SHA256:
a3360d92c93ee05351dd3c6a022b88bde38bd8bbaef41a6089526594209574c6

Counts:

- batch items: 3
- prediction rows: 3
- nuclei: 3
- sample endpoints: 1

Endpoint:

- sample: BALBCF2_P244_C7_BALBC_FEMALE_1
- nuclei: 3
- mean latent 53BP1 burden: 0.5816842913627625
- absolute delta versus the previously frozen expected arithmetic mean: 0.0

The semantic batch ID, measurement ID, aggregation ID, and scientific endpoint were preserved.

The prediction-report and transaction IDs differ from the earlier v0.5 smoke because the v0.6 operator run materialized a new provenance chain from the manifest. This does not represent a change in scientific endpoint.

## Verification

- operator endpoint gate: PASS
- scientific-boundary gate: PASS
- underlying transaction fingerprint: PASS
- transaction record fingerprint: PASS
- identity bindings: PASS
- bound artifacts: PASS
- component stages: PASS
- cross-artifact bindings: PASS
- real assay-run idempotence: PASS

## Scientific scope

- prediction semantics: per-nucleus latent continuous 53BP1 burden
- endpoint: sample-level arithmetic mean latent 53BP1 burden
- per-nucleus focus-count interpretation: false
- biological reference read: false
- biological acceptance evaluated: false
- KL720 hardware access performed: false

## Evidence freeze

External v0.6 evidence freeze SHA256:
0caa9d2385d82adf78e702d7131f9a406235e36ed4ba1a366caca4714ed4c342

v0.5 external evidence freeze SHA256:
85e9731a56942bf9fa153012eb9b276e7e00d2c6c4a80aed485c466269a7a37d

External deployment, model, data, transaction, and audit artifacts remain outside Git.

## Interpretation

v0.6 demonstrates a usable offline application entry point in which an operator can provide an assay manifest and obtain a verified scientific endpoint through one command.

This is application-interface and orchestration evidence, not a new biological-validation, holdout-performance, or statistical-equivalence result.

The previously frozen full NASA physical KL720 seven-gate validation remains the authority for biological deployment performance.
