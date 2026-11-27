# ResNet18 training feature fidelity

In the standard female-fold INT8 BIE, the first measured boundary to lower
the pre-ReLU regression head consistently across all four training sources was
the stem MaxPool. The preceding Conv raised the head in each source mean, and
the small ReLU increment varied in sign. Later residual stages added further
downward error. These conclusions use the four training sources shared by the
192 calibration tensors. The 48 selected nuclei include one historical pilot
and one nucleus from each training bag.

| Source (original pilot excluded) | Nuclei | Conv increment | ReLU increment | Stem MaxPool increment | Remaining extractor | Final burden change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BALBCM1 | 11 | +0.01724 | −0.00003 | −0.06577 | −0.10396 | −0.09464 |
| BALBCM2 | 12 | +0.02388 | +0.00002 | −0.07177 | −0.10977 | −0.05901 |
| C57BLM1 | 12 | +0.02659 | −0.00080 | −0.06694 | −0.11459 | −0.04072 |
| C57BLM3 | 12 | +0.02874 | −0.00009 | −0.07145 | −0.09800 | −0.07171 |
| All four | 47 | +0.02426 | −0.00023 | −0.06905 | −0.10664 | −0.06592 |
| Historical pilot alone | 1 | −0.00837 | 0.00000 | −0.02211 | −0.05278 | −0.09415 |

Each stage increment is the change in foci/nucleus when its native activation
replaces the adjacent previous native activation in an otherwise FP32 suffix.
The Conv increment starts at the FP32 network on the native input grid.
The remaining extractor ends at the FP32 global spatial mean and affine head
applied to the final native residual map. Other contributions to the final
burden include input rounding, global pooling, head arithmetic and zero
flooring. The per-nucleus CSV provides the complete signed decomposition.
The output of each FP32 stem suffix agreed exactly with the unmodified graph
preactivation when fed its FP32 activation across 48 inputs.

At the final nonnegative output, applying the frozen FP32 head and ReLU to
the BIE pooled vector moved burden by a mean of −0.06298, compared with the
observed −0.06592. The residual downstream shift averaged −0.00294.
All 27 original zeros stayed zero. All 20 positive outputs decreased.
The mean losses in the four positive training-prediction quartiles were
0.12392, 0.15087, 0.13391 and 0.21635. The per-source, radiation/time/dose,
prediction and input-magnitude means and MAEs are in `groups.csv`.

The final residual map had mean activation 0.81895 FP32 and 0.80246 BIE.
Its zero fraction rose from 43.76% to 46.38%. The global-pool zero fraction
rose from 0.87% to 2.58%. Channel 451 had the largest absolute mean
contribution to the head shift (mean pooled-feature error −0.12824 times
head weight +0.02726, contributing −0.00350 foci/nucleus). The channel and
distribution tables preserve positive and negative differences across all
512 channels.

## What the native words show

The Conv and ReLU stem words occupied 16 bits each. MaxPool stored eight.
At the final residual map, global pool, affine head, output ReLU and burden
the widths were 16, 8, 16, 8 and 8 bits respectively.

Of 49,283,072 stem Conv values in the 47-nucleus panel, 15,881 reached the
signed lower endpoint. All 15,881 had zero or negative value after the native
ReLU; none reached the upper endpoint. The native global pool used neither
signed endpoint, and every FP32 pooled feature fell within its channel's
observed signed range (24,064 values).

Applying the isolated FP32 MaxPool to the observed native ReLU gave a
reference for the 8-bit native stem pool codes. Of 12,320,768 positions,
12,320,660 fell within the signed 8-bit range. In range and away from
integer and half-integer boundaries, floor and nearest rounding disagreed
on 1,535,274 positions. Every one matched nearest. Across **all** positions,
including ties, the counts matching nearest-even and floor were 11,717,971
and 8,703,233 respectively. The panel rejects a systematic round-down account
of the stem shift on the prespecified discriminating positions. These data
support testing greater precision across the stem pool boundary. The later
residual-extractor contribution remains material.

## Native output provenance

The replay authenticated the original and optimized ONNX, BIE and every
selected input and native capture against the hash-bound archival analysis.
The original ONNX stores the affine weights transposed relative to the
optimized graph. Their effective weight vector and bias were identical.
The BIE capture used Kneron's pinned native simulator at dump level 2 on the
unchanged binary. The archived capture checked every returned burden against
the decoded native dump. This run loaded those fixed-point captures locally,
reran both FP32 graphs, and verified each terminal signed word against public
BIE output scale 1.3206359148025513 and radix 5. The global pooled-channel
spacing uses paired native words and vendor-decoded floats across the
original 48 training cases. It is an observed spacing, distinct from the
public terminal descriptor.

## Reproduce

Install the NASA ONNX dependencies and set `RADEDGE_DATA_ROOT` to the
existing external collection. The output directory must be new.

```powershell
$root = $env:RADEDGE_DATA_ROOT
$out = Join-Path $root ("scratch/resnet_training_features_" + [guid]::NewGuid())
python -m radiation_edge_ai.nasa_bps.resnet_training_features --root $root --out $out
```

The run writes paired scalar results, stratified summaries, channelwise
errors, original/BIE paired pooled and spatial features, and a receipt with
their hashes. The input hashes in `paired.csv` are SHA256 of the frozen NPY
files. Its `confirmation` label means the 47 previously measured training
nuclei after the pilot was excluded. The standard BIE retains its original
failed qualification.

The two archived replay directories beneath the external scratch root are
`nasa_resnet_feature_fidelity_20261127_split_stem_v1` and
`nasa_resnet_feature_fidelity_20261127_pool_codes_v1`. Their receipt SHA256s
are `d0107bebdb0432c4e1e9b99360a453e54af20882ec010e96f0fe68e53fbd5fdc`
and `b0addc686d1ab8151955040853612a2afcdafd8ef32da092f1f573d91ba7aa3d`.
The full second replay took 113.07 seconds on the host. The archived native
simulator jobs totaled 2,020.78 seconds. This experiment uses the frozen
standard BIE and does not change its compilation eligibility.

## Same-input comparison with MobileNetV2

An additional 24 native singleton jobs ran the unchanged ResNet18 BIE on the
same six nonpilot nuclei per source as the MobileNetV2 confirmation panel.
Selection used the frozen MobileNetV2 FP32 prediction ranks. Seven of these
inputs had zero ResNet18 FP32 burden. The two architectures can now be
compared on precisely the same input hashes.

| Training source | Nuclei | Observed BIE minus FP32 | Upstream through global pool | Downstream head and output |
| --- | ---: | ---: | ---: | ---: |
| BALBCM1 | 6 | −0.10699 | −0.10094 | −0.00605 |
| BALBCM2 | 6 | −0.12189 | −0.11161 | −0.01027 |
| C57BLM1 | 6 | −0.12448 | −0.12051 | −0.00397 |
| C57BLM3 | 6 | −0.10772 | −0.10353 | −0.00419 |
| Overall | 24 | −0.11527 | −0.10915 | −0.00612 |

All values are signed changes in nonnegative burden (foci/nucleus).
Upstream is the FP32 affine head and ReLU applied to the decoded BIE pool
minus the original prediction. Downstream is the BIE minus this substituted
prediction. For all four previously captured cases, all six input/activation
word arrays reproduced byte for byte.

The pinned vendor API `libdynasty.so::dump_radix_json` exported per-channel
scales, radices and signed widths. The shared MobileNet decoder agreed with
native text within 7.46 × 10⁻⁹ for pool/flatten and exactly for input, head,
output ReLU and final burden. The native jobs all exited successfully and
took 682.519 seconds summed across workers. The interrupted host launcher
did not save a total capture wall time. The FP32/word audit took 71.90 seconds.
The readout receipt and paired CSV have respective SHA256s
`d0c7a965c04473a101b4a1ecaa1a6a844277f94cd942f4dcd079ad1dc51f2b48`
and `a5569ea52f41a46a5b7d702f6e86f2c4d3e196603f5cc4690f73e774128d58ef`.

The common-panel input selection and the simulator capture helper are bound
to their hashes from the archived MobileNet feature experiment. Use the
shared decoder from commit `a3a299d`. With both scientific dependencies
installed and the pinned Docker image available locally, run:

```powershell
$root = $env:RADEDGE_DATA_ROOT
$out = Join-Path $root ("scratch/resnet_common24_" + [guid]::NewGuid())
python scripts/windows/replay_resnet_common_training.py capture --root $root --out $out
python scripts/windows/score_resnet_common_training.py metadata --root $root --out $out
python scripts/windows/score_resnet_common_training.py score --root $root --out $out --dest "$out/readout"
```

When the shared decoder is in a separate checkout, pass its
`src/radiation_edge_ai/nasa_bps/feature_attribution.py` path via `--decoder`.
