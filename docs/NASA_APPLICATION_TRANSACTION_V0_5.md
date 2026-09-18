# NASA application transaction v0.5

## Purpose

v0.5 adds one high-level offline command that composes the previously validated NASA batch-inference, prediction-projection, measurement-planning, and endpoint-reconstruction stages.

It does not introduce a second inference implementation, a second endpoint implementation, biological-reference lookup, biological acceptance logic, or KL720 execution.

## Application path

verified batch_plan.json
-> batch-measure
-> existing batch execution
-> existing prediction-table projection
-> existing measurement planning
-> existing endpoint reconstruction
-> transaction_record.json

The transaction record binds all component artifacts and verifies the complete cross-stage provenance chain.

## Real NASA three-nucleus transaction

Transaction ID:
t1-d77ff003432e2e51

Transaction record SHA256:
9e61b5b0bc34bedb2263d5389f3414154abea16bc8f5ea656f840675ee28a3c6

Component identities:

- batch: b1-dfe95904d1991502
- prediction projection: p1-85a3cfdf2e6e83bf
- measurement: m1-06a26289fd977ec2
- endpoint aggregation: a1-811ca4d3d38350d1

Counts:

- batch items: 3
- prediction rows: 3
- nuclei: 3
- sample endpoints: 1

Transaction verification passed for:

- transaction fingerprint
- record fingerprint
- identity bindings
- all bound artifacts
- component-stage verification
- cross-artifact bindings
- transaction-level idempotence

## Scientific scope

- prediction semantics: per-nucleus latent continuous 53BP1 burden
- endpoint: sample-level arithmetic mean latent 53BP1 burden
- per-nucleus focus-count interpretation: false
- biological reference read: false
- biological acceptance evaluated: false
- KL720 hardware access performed: false

The v0.5 transaction is orchestration only. Scientific inference and endpoint reconstruction remain delegated to the previously validated lower-level components.

## Relationship to v0.4

v0.4 established the explicit multi-stage path from verified batch inference through deterministic prediction-table projection into the existing v0.3 measurement layer.

v0.5 wraps that same path in a single content-addressed, verifiable application transaction.

The lower-level scientific and deployment interpretation is unchanged.

v0.4 external evidence freeze SHA256:
555336ea5b6339f883e4560aed8c6698bac30a2c9b3db374f3e4a0a6599f98c5

## v0.5 evidence freeze

External v0.5 evidence freeze SHA256:
85e9731a56942bf9fa153012eb9b276e7e00d2c6c4a80aed485c466269a7a37d

The external freeze and all deployment/data artifacts remain outside Git.

## Interpretation

This result demonstrates that the offline application can execute the complete real-data NASA batch-to-measurement workflow through one deterministic high-level control-plane operation while retaining verifiable provenance across all lower-level artifacts.

It is an application-orchestration smoke test, not a new biological-validation, holdout-performance, or statistical-equivalence result.

The previously frozen full NASA physical KL720 seven-gate validation remains the authority for biological deployment performance.
