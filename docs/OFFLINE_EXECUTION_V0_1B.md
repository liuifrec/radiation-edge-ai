# Offline execution v0.1b

This increment adds the first real execution path behind the deterministic
control plane introduced in v0.1.

## Execution contract

`radedge run RUN_PLAN.json` supports the `cpu-onnx` backend through an isolated
Python interpreter containing ONNX Runtime. The main project environment does
not import ONNX Runtime.

For v0.1b, the model input is deliberately a preprocessed float32 NumPy tensor
(`.npy`). Raw microscopy-image preprocessing is not part of this increment.
The worker validates the ONNX input dtype and shape before inference, forces
`CPUExecutionProvider`, saves the single raw model output as `raw_output.npy`,
and records runtime and timing metadata.

A successful run directory contains:

- `run_plan.json` — frozen v0.1 scientific/execution identity;
- `raw_output.npy` — raw ONNX tensor output;
- `onnx_worker_manifest.json` — isolated runtime/model-I/O execution evidence;
- `run_record.json` — top-level immutable provenance record.

`radedge verify` accepts either a run plan or a completed run record. For a run
record it verifies the run-plan hash/identity, source artifacts (unless
`--no-artifacts` is used), raw output hash/dtype/shape, worker-manifest hash,
and the worker manifest's input/model/output identities.

## Scientific boundary

This increment establishes reproducible model execution only.

- NASA R1 v2 raw output is a latent per-nucleus burden value, not an
  individually supervised 53BP1 focus count and not yet a bag-level biological
  endpoint.
- DNAi raw output is a segmentation-logit tensor, not reconstructed DNA-fiber
  objects or tract measurements.

No biological-fidelity acceptance decision is made by `radedge run` v0.1b.

## Runtime isolation

Set `RADEDGE_ONNX_PYTHON` to a Python interpreter with NumPy and ONNX Runtime,
or pass `--onnx-python` explicitly. This keeps the repository `.venv` and the
known-good Kneron environment independent.
