# Spotiflow single-crop inference and qualified replay

The recovered adapter uses the released Spotiflow 0.6.5 host implementation and its two 2D checkpoints. The frozen source is `fa5047fd20c104de98f62ad39641a17fee1489dc`. The catalogue records the hashes, original licenses and separate engineering and scientific decisions. General’s `full_int16_transport` variant is the physical replay reference. Its original simulator API decoding failure remains in the catalogue. The six original ONNX-parity failures for `synth_complex` block its hardware path. Both checkpoints are experimental for the NASA 53BP1 endpoint.

## Runtime contract

Each inference accepts one finite grayscale 2D TIFF or NPY. Each dimension must be 113–128 pixels. Upstream percentile normalization runs on the original field, followed by centered reflection padding to 128 × 128 and float32 batch-1 NCHW packing. The runner rejects larger fields and other padding buckets. This single-crop contract has no overlapping tile joins or cross-tile spot deduplication.

The released neural core returns one finest-scale heatmap logit and three raw flow channels. The separate-head ONNX and physical paths identify the 1- and 3-channel tensors before joining them. The host applies sigmoid to the heatmap and channel normalization to the flow, center-crops back to the input field, detects local maxima at the released threshold and refines them with subpixel flow. YX coordinates remain relative to the original image. Out-of-bounds refined points and their associated confidences are removed together. With an integer instance mask and positive target label, each point is assigned to its nearest pixel centre with half-pixel ties going to the higher index. Multiple points mapping to one pixel retain their individual identities.

The qualified physical ingress is V1 `1W16C8BHL`, radix 14 and scale 1. The two V1 `8W1C16B` signed output heads use independent descriptor scales and radices. The offline replay checks the vendor-selected `dma2seq_N.seq.16B` words and companion descriptors against the compiled NEF and archived PLUS 3.2.0 retrieval. It compares regenerated YX points and nucleus counts against the frozen point readout without a vendor installation or USB device.

## Host and offline execution

Install the optional runtime with `python -m pip install ".[spotiflow]"` or use the original Spotiflow Python environment. The Python package and the externally staged, registered checkpoints are both required. Use `radedge-models --help` to inspect the available records. Set `RADEDGE_DATA_ROOT` to the directory containing the original registered artifacts. The examples run from a repository checkout. Supply `$Image` as the path to an original-size crop.

```powershell
radedge-pretrained-infer --model spotiflow/general --variant full_int16_transport --backend pytorch --image $Image --out (Join-Path $env:RADEDGE_DATA_ROOT ('scratch/native_' + [guid]::NewGuid())) --data-root $env:RADEDGE_DATA_ROOT
.\scripts\windows\run_spotiflow_kl720.ps1 -Preflight
.\scripts\windows\run_spotiflow_kl720.ps1 -ReplaySimulator
```

Select `--backend onnx` for CPU ONNX Runtime. For the native `spotiflow/synth_complex` checkpoint omit `--variant`. Its ONNX backend provides an experimental comparison path. Add `--mask $Mask --target-label $Label` to bind a count to an instance. The single-image runs save the joined head tensor, point coordinates, image and artifact identities, and backend-labelled timings. Preflight authenticates the original 63 registered cases without writing results. Archive replay writes a new receipt with signed transport identities, bitwise float parity, per-case points and target counts. It does not run a fresh simulator.

## Physical replay

The PowerShell launcher accepts `-SpotiflowPython` and `-PlusEnvironment` to locate existing host and Kneron environments. Physical replay requires the registered device and firmware, Kneron PLUS 3.2.0 with the registered DLL, and the full INT16 NEF verified against the original receipts. The model is loaded into volatile device state. The launcher has no firmware maintenance operation.

```powershell
$run = '.\scripts\windows\run_spotiflow_kl720.ps1'
$smoke = Join-Path $env:RADEDGE_DATA_ROOT ('scratch/smoke_' + [guid]::NewGuid())
& $run -Smoke -Out $smoke
if ($LASTEXITCODE -ne 0) { throw 'Spotiflow smoke failed' }
$receipt = Join-Path $smoke 'receipt.json'
& $run -Panel -SmokeReceipt $receipt -SmokeSha256 (Get-FileHash $receipt -Algorithm SHA256).Hash.ToLowerInvariant()
```

Smoke rechecks the first registered case, its packed bytes, raw receive buffer and signed and floating PLUS retrievals. The full panel requires a successful smoke using the same launcher, NEF, evidence and runtime hashes. It authenticates that raw smoke packet before USB model loading. The panel checks all available SDK DRAM input bytes before submitting the first case. It saves full receive allocations, signed and floating heads, points, per-case times, and receipts under a fresh external directory. Provide the smoke and panel receipt digests, `completed` and `passed` flags, per-case parity and latency summaries when reporting an execution. Keep raw files and device identifiers in controlled external storage.

The separate `-Image $Image` mode opts into single-image physical inference after verifying the existing physical panel and device identity. A new input receives its own inference receipt. Scientific interpretation still requires an assay-specific evaluation. The Tkinter desktop exposes read-only pretrained model discovery. Use the explicit launcher above for Spotiflow acquisition.
