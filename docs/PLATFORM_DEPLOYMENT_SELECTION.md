# Select a model and backend

The experimental `nasa-mobilenetv2` entry identifies the female-fold checkpoint,
two FP32 graphs, standard INT8 BIE and its completed comparison. Seven of eight
fidelity gates passed. Mean absolute bag shift was 0.084169 foci/nucleus
against a limit of 0.05, so the launcher offers no USB backend for this model.
Use `radedge-deployments --model nasa-mobilenetv2 --data-root $Root --verify`
to verify the original five artifact hashes. [Campaign PR](https://github.com/liuifrec/radiation-edge-ai/pull/41)

The desktop exposes three reference pipelines. Choose **List all deployment references and candidates** to see engineering status and assay scope. **Verify selected reference artifacts (offline)** checks the listed file hashes under the external data root. These actions do not open USB. The existing pretrained registry still exposes per-stage Spotiflow qualification, licenses, schemas and evidence.

| Model | Selectable backend | Input and output |
| --- | --- | --- |
| NASA 53BP1 R1 v2 | KL720 | Replays the 2,058 nuclei in the frozen historical evaluation set and records sample-level burden comparison. The seven original biological-fidelity criteria apply to that cohort. |
| DNAi fiber v3 | KL720 | Runs nine windows from each of three frozen fields, then measures per-field fiber counts, morphology and FP512 fidelity. |
| Spotiflow general, `full_int16_transport` | PyTorch, ONNX Runtime CPU, KL720 | Processes one finite 2D grayscale TIFF or NPY. Each side must be 113–128 pixels. Writes subpixel spots, head outputs and a backend-specific receipt. The 63-case physical qualification applies to this configuration. NASA biological accuracy remains unqualified. |

Set the data root and interpreters in the desktop. Select **Run selected reference model**, choose the model, explicitly select its backend, then use **Preview** to inspect the command. For NASA and DNAi leave the input and output blank. Their runners generate fresh evidence directories. For Spotiflow select the crop, enter a new directory under the external data root and use the Spotiflow scientific interpreter. KL720 runs require the selected PLUS interpreter and device confirmation. The existing plan and assay-manifest actions remain available for their supported CPU or device workflows.

## Offline catalogue from a source checkout

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m radiation_edge_ai.hardware.deployment_catalogue --data-root $env:RADEDGE_DATA_ROOT
python -m radiation_edge_ai.hardware.deployment_catalogue --model dnai-fiber-v3 --data-root $env:RADEDGE_DATA_ROOT --verify
```

The installed entry point is `radedge-deployments`. Presence and SHA256 verification are distinct fields. The catalogue checks the listed model binaries and historical evidence files. Full model-specific preflight also checks the runtime, inputs and supporting records required by that launcher. Current development records include the failed NASA ResNet18 bag shifts, MobileNetV2 development, the DNAi v4 and v5 grader-supported losses, the Spotiflow graph variants, and the remaining microscopy candidates.

## Batch boundaries

The NASA and DNAi reference commands preload one NEF for the whole frozen batch. Each window or nucleus retains its source identity, request order, output and timing. Failure is reported by the chosen backend. Spotiflow’s desktop action accepts one crop. Its separate engineering-panel command supports a fixed 63-case hardware batch, following a matching physical smoke. Multi-crop acquisitions with new user images need their own explicit input manifest, identity checks, one model upload per worker and per-image output and timing checks before that mode can be enabled. [NASA replay](NASA_LOCAL_NPU_QUICKSTART.md) · [DNAi replay](DNAI_V3_KL720_REPLAY.md) · [Spotiflow contracts](SPOTIFLOW_RUNTIME_RECOVERY.md)
