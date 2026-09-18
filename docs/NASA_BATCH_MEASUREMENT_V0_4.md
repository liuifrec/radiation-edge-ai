# NASA batch measurement orchestration v0.4

## Scope

v0.4 extends the offline application from single-run execution to deterministic multi-input CPU-ONNX batch execution and prediction-table projection.

The batch executor composes the existing content-addressed run-plan and run-record primitives rather than introducing a second inference implementation.

The resulting prediction table is compatible with the existing v0.3 NASA measurement layer.

No KL720 hardware access, biological-reference lookup, or biological acceptance evaluation is performed by the v0.4 transaction.

## Application path

preprocessed nuclei
-> batch_plan.json
-> ordinary run_plan.json artifacts
-> ordinary verified run_record.json artifacts
-> batch_record.json
-> prediction_table.csv
-> existing v0.3 measurement_plan.json
-> endpoint reconstruction
-> measurement_record.json

## Real NASA three-nucleus transaction

Batch ID:
b1-dfe95904d1991502

Batch plan SHA256:
3772d8342804b6230d5babacfafeb5540e0fa1013d728c1b353c4597cb496ad5

Batch record SHA256:
b9ad7c813c5dee6b6ee69f75b6a0bc7f1615016f8bb9b10c1d8bb3a18a418358

Frozen FP32 ONNX SHA256:
a65718a8f07d5bcd8707bae78d8f730d56047db21e575fa55ba5396adc49a07e

The first nucleus reproduced the previously frozen single-run CPU-ONNX result exactly at stored float precision:

- BPSR1V2H_00001: 0.6421411037445068
- absolute delta versus frozen single-run result: 0.0

The three batch predictions were:

- BPSR1V2H_00001: 0.6421411037445068
- BPSR1V2H_00002: 0.5384736061096191
- BPSR1V2H_00003: 0.5644381642341614

All three child run records and their cross-artifact bindings verified successfully.

## Prediction-table projection

Prediction ID:
p1-85a3cfdf2e6e83bf

The deterministic projection produced three endpoint-compatible per-nucleus rows with burden column cpu_onnx_burden.

Source batch-record verification, prediction-table verification, derivation verification, and projection idempotence all passed.

## Measurement reconstruction

Measurement ID:
m1-06a26289fd977ec2

Measurement plan SHA256:
351f7e61e2146dc946f306e3224fd7b369cb1d31c66f03d5afb451ad65b3ce9d

Aggregation ID:
a1-811ca4d3d38350d1

Endpoint report SHA256:
6006153e402cc550d05522d5444c88123caf2ada5b4c7bc50856d07dd1d9cc14

Sample aggregate SHA256:
7832c1a9916743d706741791975537c8124a909951f0a28c965e24b0e3f19370

Measurement record SHA256:
49fa2fd17ce4bbbaa4a85dc8455aa4634bc3cfab04f7fa0d2988a98cfca278ab

The three nuclei belonged to one frozen sample:

- sample: BALBCF2_P244_C7_BALBC_FEMALE_1
- nuclei: 3
- sample-level mean latent 53BP1 burden: 0.5816842913627625
- absolute delta versus the independently calculated arithmetic mean: 0.0

Measurement-plan verification, endpoint-report verification, aggregate verification, cross-artifact binding, planning idempotence, and execution idempotence all passed.

## Scientific scope

- neural inference performed: true
- prediction table created: true
- endpoint reconstruction performed: true
- per-nucleus focus-count interpretation: false
- biological reference read: false
- biological acceptance evaluated: false
- KL720 hardware access performed: false

The per-nucleus outputs remain latent continuous 53BP1 burden scores, not individually supervised focus counts.

## Evidence freeze

External v0.4 evidence freeze SHA256:
555336ea5b6339f883e4560aed8c6698bac30a2c9b3db374f3e4a0a6599f98c5

The external freeze and deployment/data artifacts remain outside Git.

## Interpretation

This v0.4 result demonstrates a real-data, multi-input application transaction from frozen preprocessed nuclei through CPU-ONNX inference, provenance-bound prediction-table construction, and assay endpoint reconstruction.

It is an application-orchestration smoke test, not a new biological-validation, holdout-performance, or statistical-equivalence result.

The previously frozen full NASA physical KL720 seven-gate validation remains the authority for biological deployment performance.
