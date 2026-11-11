# Kneron KL720 compiler compatibility — sanitized engineering report

**Status:** Reproducible rejection in full Perilipin2 experimental graph;
standalone synthetic minimal reproducer **not yet vendor-tested**.
**Date:** 2026-11-10. **Severity:** Blocks experimental KL720 model
deployment, not the CPU/FP32 biological-analysis reference.
**Scope:** Experimental Perilipin2 M3C branch, not production DNAi/NASA.

## Product and environment

- Accelerator target: **KL720**.
- Vendor image: `kneron/toolchain:v0.33.1` as reported by `evaluate()`.
- Exact pinned local Docker image identity:
  `sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041`.
- Vendor interpreter: `/workspace/miniconda/bin/python` in that image.
- Windows host invokes Docker Linux toolchain with `--pull never`,
  `--network none`, mounted isolated scratch folder, and raw byte-level
  logging. No Docker reset, external image fetch or persistent system
  modification required.
- ONNX input contract: **[1, 3, 512, 512]**; output contract:
  **[1, 2, 512, 512]** (foreground and boundary logits), ONNX opset 11.
- Original frozen model SHA256:
  `8c256cd69249e01d6066f6dd3a377e1ef0c57e6c4c1c3799ef8aeeb0e204b797`.
  **The model file, learned weights, sample microscopy images and masks
  must not be committed or attached to public issues.**

## Validated numerical invariants

Four decoder stages upsample their feature maps by a factor of exactly
2 in height and width:

| Input NCHW | Target NCHW |
| --- | --- |
| [1, 96, 32, 32] | [1, 96, 64, 64] |
| [1, 72, 64, 64] | [1, 72, 128, 128] |
| [1, 48, 128, 128] | [1, 48, 256, 256] |
| [1, 24, 256, 256] | [1, 24, 512, 512] |

Original ONNX uses nearest/asymmetric/floor `Resize` with
shape-derived `sizes`. The Kneron-optimized graph contained 123 nodes.
A parity-gated, deletion-only graph cleanup removed 32 unreachable
shape-processing nodes (20 Constant, 4 Pad, 4 Shape, 4 Slice); 91 live
nodes remained. Both fixed integer-scale and fixed-size variants passed
FP32 ONNX Runtime parity on **two deterministic synthetic inputs**
with zero output difference and zero binary foreground/boundary
decision mismatches. This is not real-image biological validation.

A further non-trained transformation changed all four nearest-2x
`Resize` ops into `ConvTranspose` nodes using:
- `group = C`, with C = 96, 72, 48 or 24;
- `kernel_shape = [2, 2]`; `strides = [2, 2]`;
- `pads = [0, 0, 0, 0]`;
- fixed float32 all-one kernels, tensor shape `[C, 1, 2, 2]`.

The 91-node transformed graph has
**19 Conv, 4 ConvTranspose, 18 Relu, 4 Concat, 46 Constant** nodes
and no `Resize` nodes. Learned weights were not modified.
The two-input FP32 parity gate still returned maximum absolute error 0.0
and zero segmentation-decision disagreements. This validates the
rewrite on those inputs, **not compiler acceptance or endpoint accuracy**.

## Observed sequence — all vendor attempts rejected

| Trial | Compiler status | Backtrace location |
| --- | --- | --- |
| Original Kneron-optimized ONNX | `Err: 134` | `ReplaceResizeWithKneronResize -> ResizeNode::GetVScale -> __assert_fail` |
| Fixed `sizes`, unpruned | `Err: 139` | `BatchSizeInterpreter::InterpretAllInputBatch` |
| Fixed integer `scales`, unpruned | `Err: 139` | Same batch interpretation path |
| Fixed `sizes`, dead branches pruned | `Err: 134` | `BatchSizeInterpreter::CheckBatch -> __assert_fail` |
| Four depthwise transposed convolutions | **`compiler invalid input`** | `PreprocessVisitor::Visit(ConvTransposeNode*) -> ConvTransposeNode::InferPadding() -> InvalidProgramInput` |

**Current terminal failure:**

```text
Failure for model "input/input" when running "kdp720/compiler"
...
compiler invalid input
...
PreprocessVisitor::Visit(ConvTransposeNode*)
ConvTransposeNode::InferPadding()
InvalidProgramInput::InvalidProgramInput(...)
```

This is a vendor **input-validation exception** in the
`ConvTranspose` preprocessing path. Do not mislabel it as the
original `ResizeNode::GetVScale` crash, or infer that a particular
`group` or `pads` value is unsupported without an isolating test.

The model flow report (`model_fx_report.json`, 172 bytes) contained
only the Docker version and paths to itself and its HTML counterpart;
`ioinfo.csv` was absent. The Kneron CLI supplied no failing node
name, failing attribute value, or explanatory parameter constraint.
The vendor `evaluate()` API sometimes returned metadata or a
zero process status despite reporting a compiler failure in the
console. The experimental host wrapper now recognizes explicit
rejection text, nonzero `Err: N` tables, and fresh backtraces, and
reports `VENDOR_REJECTED` fail-closed.

**Not performed:** PTQ, BIE, NEF, physical KL720 inference, device
latency or accuracy benchmarking. This report does not support
any claim of a functioning Perilipin2 KL720 application.

## Minimal synthetic candidate (publicly shareable after evaluation)

A deliberately separate script
`scripts/windows/perilipin2_kl720_synthetic_repro.py`
generates standalone, **fully synthetic** ONNX `ConvTranspose`
graphs with only a fixed all-one weight tensor and one input/output
per graph. No model data are imported. It produces a SHA256 manifest
and ONNX Runtime numerical parity check against nearest-2x.

Synthetic cases (all initially **UNTESTED by Kneron**):

- `control_c1_h8`: C=1, H=W=8, group=1.
- `group_c24_h8`: C=24, H=W=8, group=24.
- `group_c96_h8`: C=96, H=W=8, group=96.
- `stage_c96_h32`: C=96, H=W=32, group=96; spatial/channel
  shape matches the first Perilipin2 decoder stage.

On a Windows host with an isolated ONNX + ONNX Runtime environment,
generate them into a **new D: scratch folder**, outside the Git repo:

```powershell
cd C:\Users\yul03\Projects\radiation-edge-ai-perilipin2
$Python = 'D:\radiation-edge-ai-data\envs\perilipin2-m3c\Scripts\python.exe'
$Out = 'D:\radiation-edge-ai-data\scratch\perilipin2_kl720_synthetic_repro_20261110_v1'
& $Python .\scripts\windows\perilipin2_kl720_synthetic_repro.py --out $Out
Get-Content "$Out\SYNTHETIC_REPRO_MANIFEST.json"
```

These generated ONNX files are a **proposed isolation set**, not yet a
confirmed reproducer. Run one isolated vendor evaluate per case
under the same pinned Docker image before claiming a minimal
reproduction. Never include the original 2 MB model or private
measurements in a public bug report.

## November 11 result: synthetic cases generated and host-validated

Four synthetic-only standalone models were produced on Windows under
`D:\radiation-edge-ai-data\scratch\perilipin2_kl720_synthetic_repro_20261111_v1`.
Each returned ONNX Runtime maximum absolute error 0.0 against 2x nearest
replication. The generator reported **no vendor compilation**:

| Case | SHA256 |
| --- | --- |
| `control_c1_h8` | `717ac4bfe568bbdb00438ac752b35683d62b86b5a739a4c9a5822c844a415bea` |
| `group_c24_h8` | `bfae5b1c30528c7a50e63a696297dee1d8ba643a2352d60094dfb78f8de629f0` |
| `group_c96_h8` | `37b3fa71296a0b95a785de27f15bbe5c01fd95afb200b64e32162aa95d79a9a0` |
| `stage_c96_h32` | `185abfa06e4932e89bea311446156dcd463fe8c09948f60e29c00f046494348d` |

The next test targets **only** the simplest `control_c1_h8` graph, using
the pinned v0.33.1 Kneron image. An independent Python script checks
the manifest, synthetic ONNX hash, exact one-node ConvTranspose contract,
image identity, and a fresh output folder before invoking Docker.
The container re-verifies the staged candidate and runs only
`ktc.ModelConfig(..., onnx_model=model).evaluate()` (no graph
re-optimization). It saves byte-exact console logs and a fail-closed
vendor report. No microscopy data or original model are loaded.

```powershell
cd C:\Users\yul03\Projects\radiation-edge-ai-perilipin2
git fetch origin feature/perilipin2-windows-m3c-poc
git merge --ff-only FETCH_HEAD
$Python = 'D:\radiation-edge-ai-data\envs\perilipin2-m3c\Scripts\python.exe'
$Synthetic = 'D:\radiation-edge-ai-data\scratch\perilipin2_kl720_synthetic_repro_20261111_v1'
$Vendor = 'D:\radiation-edge-ai-data\scratch\perilipin2_kl720_vendor_control_c1_h8_20261111_v1'
& $Python .\scripts\windows\perilipin2_kl720_synthetic_vendor_gate.py --manifest-dir $Synthetic --case control_c1_h8 --out $Vendor
Get-Content "$Vendor\SYNTHETIC_VENDOR_EXECUTION.json"
if (Test-Path "$Vendor\SYNTHETIC_VENDOR_REPORT.json") { Get-Content "$Vendor\SYNTHETIC_VENDOR_REPORT.json" }
if (Test-Path "$Vendor\backtrace.log") { Get-Content "$Vendor\backtrace.log" }
```

Do **not** run the other three variants until this control's compiler
response is reviewed. `VENDOR_RETURNED_REQUIRES_REPORT_REVIEW` is
not a successful NEF compilation or an NPU-only result.

## November 11 control result: 1-channel ConvTranspose frontend accepted

The smallest standalone **synthetic** trial, `control_c1_h8`, was
submitted to the pinned KL720 Toolchain v0.33.1 image without re-optimization.
The candidate SHA256 matched the generator receipt:
`717ac4bfe568bbdb00438ac752b35683d62b86b5a739a4c9a5822c844a415bea`.

The toolchain console reported:

```text
Success for model "input/input" when running "general/Success"
kdp720/compiler frontend: checkmark
kdp720/compiler_cfg: checkmark
kdp720/compiler: checkmark
kdp720/cpu_node: N/A
SYNTHETIC_KL720_INNER_STATUS: VENDOR_RETURNED_REQUIRES_REPORT_REVIEW
SYNTHETIC_KL720_HOST_STATUS: VENDOR_RETURNED_REQUIRES_REPORT_REVIEW
```

The vendor's node report labeled the zero-insertion, convolution and
output-reformat operations as `NPU`. Its estimated
`kdp720/ip_eval/ITC(ms)` was `0.00591125 ms` with estimated
`169169 FPS`. **These figures are vendor analytic estimates for a
tiny synthetic model, not measured on-device performance.**
No quantization experiment, NEF, or physical KL720 inference was performed
by this controlled gate.

**Interpretation:** `ConvTranspose` with 2x2 kernel, stride 2,
zero pads, and `group=1` is accepted as an isolated one-channel
case by this compiler. This rules out an unconditional prohibition
of all such `ConvTranspose` nodes. It does not establish support
for `group=24`, `group=96`, the full 91-node decoder, or hardware
inference. `VENDOR_RETURNED_REQUIRES_REPORT_REVIEW` remains an
appropriately conservative status.

Next: compare the already-generated `group_c24_h8` model under the
same pinned image, in its own **new** output directory. Do not modify
or rerun the frozen `control_c1_h8` folder. Record the vendor flow table,
CPU/NPU report, any backtrace and both gate receipts. If the grouped
case fails, isolate group-count effects from channel-count effects
with a separate synthetic control rather than changing the trained
Perilipin2 model.

```powershell
cd C:\Users\yul03\Projects\radiation-edge-ai-perilipin2
git fetch origin feature/perilipin2-windows-m3c-poc
git merge --ff-only FETCH_HEAD
$Python = 'D:\radiation-edge-ai-data\envs\perilipin2-m3c\Scripts\python.exe'
$Synthetic = 'D:\radiation-edge-ai-data\scratch\perilipin2_kl720_synthetic_repro_20261111_v1'
$Vendor = 'D:\radiation-edge-ai-data\scratch\perilipin2_kl720_vendor_group_c24_h8_20261111_v1'
& $Python .\scripts\windows\perilipin2_kl720_synthetic_vendor_gate.py --manifest-dir $Synthetic --case group_c24_h8 --out $Vendor
Get-Content "$Vendor\SYNTHETIC_VENDOR_EXECUTION.json"
if (Test-Path "$Vendor\SYNTHETIC_VENDOR_REPORT.json") { Get-Content "$Vendor\SYNTHETIC_VENDOR_REPORT.json" }
if (Test-Path "$Vendor\backtrace.log") { Get-Content "$Vendor\backtrace.log" }
```

## Questions for the Kneron engineering team

1. For KL720 Toolchain v0.33.1, what are the exact supported
   `ConvTranspose` `group`, `kernel_shape`, `pads`,
   `auto_pad`, `output_padding` and `strides` combinations?
   Is `group = C` valid when `C in {24, 48, 72, 96}`?
2. What condition triggers
   `ConvTransposeNode::InferPadding()` to raise
   `InvalidProgramInput`? Could `evaluate()` report the
   node name and offending attribute values?
3. Does the compiler provide an officially supported graph
   representation of nearest-neighbor 2x upsampling for U-Net-like
   skip-connected decoders on KL720? If not NPU-only, is a
   **documented** CPU/NPU partition solution available?
4. Could `evaluate()` propagate frontend failure through a
   nonzero exit status and a machine-readable error object? A
   metadata-only JSON/HTML report prevents timely diagnosis.
5. Can Kneron supply a vendor-supported minimal upsampling/
   ConvTranspose test graph for v0.33.1? We can compare that with
   the fully synthetic one-operator candidates.

## Reproducibility and sharing policy

The engineer-facing material may include this report, test harness
source, **synthetic** ONNX models after validation, redacted vendor
console text and backtrace. Retain complete original model, image,
mask and timing evidence only in the user's protected D: work folders.

Do **not** claim the minimal synthetic model reproduces an issue
until a separate Kneron run confirms that outcome. Do not merge
experimental PR #26 or treat compiler frontend failure as proof
of algorithmic or biological failure. A follow-up methodological
paper should report these negative compatibility results and
limitations transparently while focusing on independently validated
DNA fiber/other RERF EdgeAI applications.
