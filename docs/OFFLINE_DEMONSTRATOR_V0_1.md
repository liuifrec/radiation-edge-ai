# Offline demonstrator v0.1 control plane

This sprint adds a deliberately small, deterministic control plane before any
new inference adapter is introduced.

## Scope

Commands:

- `radedge doctor` - read-only runtime, storage, optional-backend, and asset probes.
- `radedge assays` - frozen scientific assay specifications.
- `radedge plan` - validate metadata/backend selection and create a
  content-addressed immutable run plan.
- `radedge verify` - verify the plan fingerprint and optionally rehash all
  referenced artifacts.

The initial registry contains:

- `nasa-53bp1-r1-v2`
- `dnai-fiber-v3`

## Non-goals for this commit

This commit performs no neural inference, KL720 device access, PTQ/model
selection, image reconstruction, biological gate re-evaluation, or automatic
instrument control.

The known-good Kneron Python environment remains isolated. `doctor` only checks
whether a candidate interpreter exists; it does not import `kp` or scan USB.

## Content-addressed planning

`radedge plan` hashes the input file, model artifact, metadata JSON, and optional
configuration JSON. The run ID is derived from scientific identity, backend
selection, content hashes/sizes, embedded metadata, and configuration.
Absolute paths and creation time are not part of the identity.

The generated `run_plan.json` explicitly records that execution has not occurred.

## Next increment

After this layer is stable, add execution adapters behind the same plan:

1. CPU ONNX Runtime.
2. KL720 through an isolated subprocess using the known-good Kneron environment.
3. Endpoint reconstruction and QC.
4. A completed run record whose outputs are independently hash-verifiable.

NASA and DNA-fiber adapters must preserve their documented scientific
interpretation boundaries rather than sharing a generic accuracy criterion.
