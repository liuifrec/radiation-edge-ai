# Windows replay of the selected ResNet18 NEF

Run these commands from the campaign checkout on the Windows host with the
original external collection at `D:\radiation-edge-ai-data`. The launcher uses
the PLUS interpreter at `%USERPROFILE%\venvs\kneron720\Scripts\python.exe`.
It generates a fresh external campaign directory and prints its receipt path
and SHA256. Use `-Out` to name a fresh directory explicitly.

The default command verifies all 48 simulator references, preprocessing,
SDK-packed inputs and PLUS 3.2.0 scalar retrieval offline.

```powershell
.\scripts\windows\run_resnet_neural16_kl720.ps1
```

After separate USB approval, run the positive training witness on KL720
`0xAA06244C` with firmware `2.2.0-build.615`.

```powershell
.\scripts\windows\run_resnet_neural16_kl720.ps1 -Mode Smoke -Execute
```

The launcher checks the embedded model ID 32783, version 1 and the exact
NEF SHA256 `7f1843a62bb5244048f4df977051b6795f7a39320c6be890cc0fdd9a7f92d4ba`.
It verifies the three-channel 256 × 256 input buffer against SDK packing.
The output is one signed 16-bit transport word at radix 13 and scale
1.3206359148025513. The physical smoke must reproduce the positive witness's
actual CSIM word `7520` and its PLUS-decoded value of
0.6950960159301758 foci/nucleus exactly. The receipt records the
raw response identity, input hash, device identity and timings.

For the approved 48-witness replay, supply the passing physical smoke receipt
path and SHA256 printed by the smoke command.

```powershell
.\scripts\windows\run_resnet_neural16_kl720.ps1 -Mode Training -Execute -SmokeReceipt $smokeReceipt -SmokeSha256 $smokeSha
```

## Optional known development replay

Generate a separate reference if the 2,400-nucleus development transport replay
is needed. This command runs the NEF simulator on the frozen inputs and reuses
the original candidate BIE predictions. It requires the pinned Docker toolchain
and the installed PLUS headers. Set `$referenceDir` to a fresh external output
directory and retain the printed SHA256 of `development-reference.json`.

```powershell
.\scripts\windows\run_resnet_neural16_kl720.ps1 -Mode DevelopmentReference -Out $referenceDir
.\scripts\windows\run_resnet_neural16_kl720.ps1 -Mode Development -ReferenceDir $referenceDir -ReferenceSha256 $referenceSha
```

A separately approved physical development replay uses the hashes of both
that reference and the passing physical smoke. It checks each device word
against the observed CSIM carrier for the same input.

```powershell
.\scripts\windows\run_resnet_neural16_kl720.ps1 -Mode Development -Execute -ReferenceDir $referenceDir -ReferenceSha256 $referenceSha -SmokeReceipt $smokeReceipt -SmokeSha256 $smokeSha
```

| Qualification receipt pinned by the launcher | SHA256 |
| --- | --- |
| Independent development audit | `4f9dd6f061586840636de0aa4292944f880bef6325e4f0dcc0f90f7aa571bf20` |
| NEF compilation | `d068c2b2647edd6a75749bae2d34b7f50e241843a42ec5ed1ac62f92c8130f06` |
| 48-case NEF simulation | `1706ad42edcf401b055077534e6839e63597449f783a0422ab70ebce4d60d59b` |

[Development and BIE results](NASA_RESNET18_NEURAL16_DEVELOPMENT_RESULTS.md) ·
[Prospective simulator gates](NASA_RESNET18_NEURAL16_NEF_V1.json) ·
[Development transport replay protocol](NASA_RESNET18_NEURAL16_DEVELOPMENT_REPLAY.md)
