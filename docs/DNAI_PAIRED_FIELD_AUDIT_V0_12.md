# Paired DNAi FP512/KL720 field audit (v0.12 development)

## Purpose and strict boundary

`radedge compare-dnai-fields` is a **read-only, source-verified diagnostic**.
It compares one frozen `df1` floating-FP512 field and one frozen `df2`
physical-KL720 field without rerunning inference, touching model artifacts,
reconstructing fiber objects, or reading human annotations. The two source
field JSON records and their artifacts must independently pass the existing
public full-artifact verification gate.

## Example

```powershell
radedge compare-dnai-fields --fp512 '<path-to-df1-field_result.json>' --kl720 '<path-to-df2-field_result.json>' --json
```

Requirements: Python >=3.9, NumPy and Pillow (install with
`pip install -e '.[analysis]'`). The ordinary control plane remains usable
without Pillow; image decoding occurs only when this command is invoked.

## Comparability preflight

The command **fails closed** unless both input records describe exactly the
same `sample_id`, image key/index, DNAi upstream commit, validation-manifest
SHA256, deployment policy, and nine `(window_index, x, y, tensor SHA256, byte
size)` inputs. Model SHA256 and backend are intentionally allowed to differ.
A matching field name alone is not adequate evidence of a matched input.
Source records are SHA256-checked before and after comparison; segmentation
PNGs and stitched-probability NPZs are rechecked against each field's bound
byte hashes when read.

## Descriptive outputs

- Stitched-probability argmax agreement and mean absolute probability change
  between two 3-class float32 arrays (3 x 1024 x 1024).
- Class confusion matrix, pixel agreement, changed pixels, foreground Dice,
  foreground IoU and class-specific Dice/IoU based on the final saved
  8-bit segmentation PNGs (1024 x 1024). Background-dominated accuracy must
  not be interpreted as fiber fidelity.
- **8-connected nonzero-class component proxy** counts and greedy one-to-one
  *mask* IoU matches at a fixed 0.5 threshold. They are **NOT** DNA-fiber
  identities and may merge or split fiber objects. The threshold is diagnostic
  and must not be tuned to make results favorable.
- Valid-fiber counts and reported field-level mean/median ratio and total
  length, including unpaired KL720-minus-FP512 differences. No per-fiber
  correspondence or statistical equivalence test is implied.

The JSON report is printed to stdout and is path-bearing **ephemeral
inspection**, not a new signed/frozen research manifest. The command makes
no output-directory writes. Exit code 0 means a successful *comparison*, not
biological agreement; invalid or unverified source evidence fails closed.

## Existing scientific evidence

The historical 20-image/180-window characterization is documented in
`docs/DNAI_PHYSICAL_CHARACTERIZATION_AUDIT.md`. It already includes actual
DNA-fiber-object bbox matching and human-annotation sensitivity work, but the
panel contributed to development decisions and has a documented historical
cache-provenance caveat. This new v0.12 one-field comparison must **not** be
presented as an independent holdout or as a replacement for that work.

The frozen v0.11 field, physical transaction, discrepancy audit, portable
package and Git tag remain untouched. For a future biological-fidelity claim,
a separately designed independent, adequately replicated study is required.

## First verified frozen-field reproduction (2026-10-13)

This is a read-only reproduction of the existing single-field deployment
discrepancy, not an independent biological holdout.

- FP512: df1-72c2102e46e5a4af; field-record SHA256
  4d94b7ef92072ceb0767403a8214bb64de3a087e81725e08e77083eaa0fc7f37
- KL720: df2-d3955d46bdebf2c1; field-record SHA256
  f5ce55beb241e8885bb3c5b6ea2ee2327cd345dbe1e35b95308a4a4f520e77cf
- Shared nine-window input digest:
  42a4ba3a2ef01c352f21487e46b1b82efbd91d61bc1d61e441f1a6f98b3f4015
- Probability argmax agreement: 0.9992494583129883.
- Final segmentation pixel accuracy: 0.9985227584838867.
- Changed final segmentation pixels: 1,549.
- Foreground Dice: 0.865300277454578.
- Foreground IoU: 0.7625808487143082.
- Nonzero-class 8-connected components: 18 FP512 and 30 KL720.
- Greedy mask-IoU >= 0.5 component-proxy matches: 11.
- Valid fibers: 14 FP512 and 12 KL720.
- Total valid-fiber length: 229.1400257414643 um FP512 versus
  186.21969101489583 um KL720, a difference of -42.920334726568484 um
  (-18.73% relative to FP512).

Component proxies are not independently identified DNA fibers.
The length comparison refers to different sets of valid reconstructed fibers;
it does not establish the measurement bias of spatially matched fibers.
No inference, model tuning, biological-equivalence testing, or source-artifact
modification was performed.
