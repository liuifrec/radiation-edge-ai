# Offline KL720 execution v0.1c

`radedge run` supports the approved `kl720` backend by launching the known-good
Kneron PLUS Python interpreter as a subprocess. The normal repository
environment does not import `kp`.

The v0.1c transport contract is deliberately narrow:

- input: already-preprocessed float32 batch-1 BCHW `.npy`;
- model: frozen `.nef`;
- runtime: isolated Kneron PLUS Python;
- device path: KL720 Generic Data inference;
- output: float32 `raw_output.npy`;
- provenance: `kl720_worker_manifest.json` and `run_record.json`.

The worker follows the previously validated KL720 host path used for NASA R1 v2
and DNAi: descriptor-derived fixed-point quantization followed by KL720 v1 NPU
re-layout for 4W4C8B, 1W16C8B, or 16W1C8B.

v0.1c does not preprocess raw microscopy, convert/compile models, perform PTQ,
reconstruct assay-specific endpoints, or evaluate biological-fidelity criteria.

`--kl720-port` may pin a USB port. If omitted, exactly one visible KL720 is
required. `--kl720-timeout-ms` defaults to 10000 ms.

The interpreter is resolved in this order:

1. `--kl720-python`;
2. `RADEDGE_KL720_PYTHON`;
3. the conventional per-user `venvs/kneron720` path.

NASA per-nucleus output remains a latent continuous 53BP1 burden score, not an
individually supervised focus count. DNAi raw logits remain insufficient by
themselves for fiber-object biological fidelity.
