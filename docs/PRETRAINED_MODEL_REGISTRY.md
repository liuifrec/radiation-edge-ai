# Inspect pretrained microscopy models

The catalogue recovers both released Spotiflow 2D checkpoints and six graph or transport variants from the Windows milestone `fa5047fd20c104de98f62ad39641a17fee1489dc`. Each variant carries its own engineering decisions, artifact paths and SHA256 identities. The checkpoint carries its original-domain evidence, NASA training-source measurements, independent biological qualification and approved uses.

## Read-only commands

After installing the package, run the following commands in PowerShell. Set `$DataRoot` to your existing external artifact root for the last two commands.

```powershell
radedge-models --list
radedge-models spotiflow/general --variant full_int16_transport
radedge-models --list --data-root $DataRoot
radedge-models --list --verify $DataRoot
```

`python -m radiation_edge_ai.hardware.pretrained_models` accepts the same arguments. With no arguments, it prints the two model IDs. With `--list`, it returns JSON with the endpoint, tensor contract, required runtime, nine engineering stages and scientific status for each variant. A single-model query also returns release links, hashes, detailed decisions and evidence names.

Without a data root, availability is unchecked. `--data-root` distinguishes `missing` from `present_unverified` using file presence alone. `--verify` reads the files, reports `verified`, `missing` or `checksum_mismatch` for each reference and exits with status 1 if any verification fails. Shared files are hashed once per listing. The commands use the Python standard library and never initiate a download, simulation or USB connection.

## Engineering and scientific decisions

| Checkpoint and variant | Historical engineering decision |
| --- | --- |
| general, joined_heads | BIE produced. Numerical fidelity failed. |
| general, separate_heads | BIE produced. Numerical fidelity failed. |
| general, separate_heads_int16 | BIE produced. Numerical fidelity failed. |
| general, full_int16 | BIE passed all eight fidelity criteria. Compilation passed. Public simulator float retrieval failed the original BIE-to-NEF numerical tolerance. |
| general, full_int16_transport | Same BIE and NEF as full_int16. Decoding signed simulator transport with PLUS passed the frozen gates. Archived physical smoke and all 63 physical comparisons passed. |
| synth_complex, joined_heads | KL720 placement completed. Six readout inputs failed original ONNX parity. Optimized parity also failed. PTQ was withheld. |

Native NASA training-source bag MAE was **11.7242 foci/nucleus** for general and **0.7758** for synth_complex. The checkpoints reproduced two and four of six radiation directions, respectively. Both have exploratory-use status and independent NASA biological validation remains unestablished. The catalogue covers one two-dimensional crop padded to a fixed 128 × 128 network input. Larger-image tiling, overlap reconciliation and cross-tile deduplication require separate evidence.

The recorded engineering baseline uses Spotiflow 0.6.5, weight release 0.6.0, Kneron toolchain 0.33.1, PLUS 3.2.0 and KL720 firmware 2.2.0-build.615. The catalogue records software and model license identities. The released weights, third-party software and scientific records reside in external storage under their original distribution conditions.

## Recheck of the recovered archive

On 2026-11-26 the verification command matched all **102 distinct external files**, covering **212 artifact references**. General joined, separate, separate INT16, full INT16 and full INT16 transport accounted for 32, 29, 28, 44 and 52 references, respectively. Synth_complex accounted for 27. The archive had zero missing files and zero checksum mismatches.

The offline `qualified_bie`, `qualified_nef` and `qualified_physical_panel` checks also authenticated the transport variant's prerequisite decisions, packed inputs, 126 signed simulator head files and archived 63-input device receipt. This recheck read existing evidence without executing a new device inference. The public simulator failure remains a separate decision for the same binary. The gate document retains its original LF SHA256 `21335fb5b86e0c391c23cd827f32b0fd089474f80b482e3c0d80d477b7c4beb8`. [Frozen engineering gates](SPOTIFLOW_KL720_ENGINEERING_GATES_V1.json)

DNA-fiber v3 and NASA R1 v2 retain their dedicated routes and existing model qualifications. Their device runners are outside the Spotiflow catalogue. [Platform status and assay routes](PROJECT_STATUS.md)
