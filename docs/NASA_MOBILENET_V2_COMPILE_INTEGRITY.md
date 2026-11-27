# MobileNetV2 scalar compile and physical handoff

The first command checks the prospective BIE policy, the selected BIE, the actual
simulator prediction table and the three scored output tables. It reads files
only. Run from the campaign worktree with the package installed or `PYTHONPATH=src`.
Set `$Readout` to the new readout directory and `$Pilot` to the frozen training PTQ run.

```powershell
$comparison = Join-Path $Readout 'comparison/comparison.json'
$check = @(
  '--policy', 'docs/NASA_MOBILENET_V2_FEMALE_INT8_V1.json',
  '--comparison', $comparison,
  '--comparison-sha256', (Get-FileHash $comparison).Hash.ToLowerInvariant(),
  '--simulation', (Join-Path $Readout 'vendor/bie_receipt.json'),
  '--predictions', (Join-Path $Readout 'vendor/bie_nuclei.csv'),
  '--bie', (Join-Path $Pilot 'pilot/quantized_model.bie')
)
python -m radiation_edge_ai.nasa_bps.multimodel_nef @check
```

Check the comparison digest against the campaign report before compilation.
Exit code 2 records a completed readout that failed one or more limits. A passing
readout permits running the same module with `--compile-out` in a new writable
scratch directory inside the pinned toolchain image. Mount the original BIE,
policy, comparisons and simulator outputs read-only. Use model ID 32781 and version
`0001`. Save the compiler log and hash the NEF before numerical testing.

## Prospective transport check

1. Read the compiled input/output names, axis order, effective width, signedness,
   fixed scales and radices from the NEF and the installed PLUS tensor descriptors.
   Bind them and both executable hashes in the simulation receipt. A scalar
   `burden` output must contain exactly one element.
2. Begin with the prespecified 16 training inputs. Compare the two official
   simulation paths. Authenticate packaged input bytes, raw signed scalar codes
   and decoded values. Require identical codes and a maximum absolute decoded
   difference ≤ 0.0001 foci/nucleus on each case, the existing NASA scalar
   compile-integrity tolerance. Preserve all original outputs for failed cases.
3. Before a physical panel, verify the frozen development inputs and compile-level
   simulation against the accepted BIE. Reuse the declared cohort, runtime output
   representation and eight engineering criteria for quantitative response. Use a distinct
   receipt for each backend.
4. Physical access requires approval for this model and NEF digest. Use a smoke
   case first, then the same hashes and successful smoke receipt for a single-upload
   panel. Record device and firmware identity privately, runtime version, packing,
   signed and decoded outputs, USB exchange time, host processing and wall time.
   A failed BIE decision ends the campaign before a NEF or USB command is eligible.

The existing R1 physical launcher authenticates R1's NEF and cohort. MobileNetV2
requires its own authenticated simulator evidence and device route. The source
for the scalar tolerance and shape-specific parser handling is the
[R1 verifier](../src/radiation_edge_ai/nasa_bps/kneron_verify_r1_v2_nef_vs_bie.py).
