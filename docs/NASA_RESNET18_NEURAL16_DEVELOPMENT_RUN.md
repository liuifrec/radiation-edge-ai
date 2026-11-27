# Reproduce the ResNet18 neural16 development regression

The primary comparison uses the original FP32 ONNX predictions and the selected
female-fold BIE on identical tensors. The original INT8 comparison provides the
paired historical control. This cohort was examined previously. Inputs, model
binaries, individual predictions, native logs and figures belong in external
storage. [Candidate policy](NASA_RESNET18_NEURAL16_REGRESSION_V1.json) ·
[Qualification handoff](NASA_PRECISION_QUALIFICATION_HANDOFF.md)

## Windows environment

Run these commands from the integrated campaign checkout. Set `$python` to the
scientific interpreter that provides the project dependencies. Figure generation
also requires Matplotlib. The native worker uses the Python inside the pinned
Kneron image. The local run used the following paths.

```powershell
$data = 'D:\radiation-edge-ai-data'
$python = Join-Path $data 'envs\nasa39\Scripts\python.exe'
$plotPython = Join-Path $data 'scratch\spotiflow_pretrained_models_20261119_native_v1\venv\Scripts\python.exe'
$env:PYTHONPATH = (Resolve-Path .\src).Path
$run = Join-Path $data 'scratch\nasa_resnet_neural16_dev_20261127_v1'
$env:MPLBACKEND = 'Agg'
$env:MPLCONFIGDIR = Join-Path $run 'figure_cache'
$prepared = Join-Path $data 'scratch\nasa_resnet18_female_eval_20261118T060157Z_5ffa44b2\preparation'
$standard = Join-Path $data 'scratch\nasa_resnet18_female_bie_readout_20261118T060725Z_438c78b7\comparison'
$policy = (Resolve-Path .\docs\NASA_RESNET18_NEURAL16_REGRESSION_V1.json).Path
```

## Recorded one-time simulation

The original launch command below identifies the procedure used for this
campaign. The launcher refuses an existing readout. Its `--resume` option is
for a technically interrupted invocation of the same authenticated study and
verifies completed chunks before continuing.

```powershell
& $python -B .\scripts\windows\run_resnet_neural16_regression.py --data-root $data --out $run --jobs 6
```

The launcher authenticates the policy at commit
`a6c3da20273a9c43b5a47bc7501724f3981dc206`, its training evidence, both
graphs, the checkpoint and the development preparation. It checks each tensor's
SHA256, layout, normalization bounds, zero channel and membership. It binds the
frozen BIE as a read-only Docker mount. Two archived training singletons and an
interleaved native batch establish sample order. The development readout archives
each job's exit status, commands, console output and terminal signed word and
float in manifest order. The working BIE and source inputs are rechecked at exit.

## Score and independently audit existing output

With a completed native run, the following commands create the official
comparison, independent audit and historical-control tables. The output
directories are created once. Keep the first receipts with the binary.

```powershell
& $python -B -m radiation_edge_ai.nasa_bps.neural16_comparison --run $run --prepared $prepared --data-root $data --policy $policy
& $python -B -m radiation_edge_ai.nasa_bps.neural16_audit --run $run --prepared $prepared --data-root $data --policy $policy
& $plotPython -B .\scripts\analysis\report_resnet_precision_regression.py --run $run --historical $standard --out (Join-Path $run 'paired_report')
```

To verify again from archived native jobs, use a fresh audit destination. This
command rehashes inputs and results and reconstructs the eight decisions without
executing the model or changing the recorded qualification.

```powershell
$replay = Join-Path $run ('audit_replay_' + [guid]::NewGuid())
& $python -B -m radiation_edge_ai.nasa_bps.neural16_audit --run $run --prepared $prepared --data-root $data --policy $policy --out $replay
```

The independent audit interprets the native little-endian signed storage word
at radix 12 and verifies the vendor's float serialization. It checks GNU
Parallel job records and nucleus identities, reconstructs sample means and
exposed-minus-sham contrasts, and checks every gate against its original
unrounded limit. Public transport has a separate 16-bit radix-13 descriptor.
Only a compiled-artifact simulator or subsequent authenticated physical readout
can provide direct evidence of that interface.

NEF compilation requires a passing official decision and its independent audit.
The NEF engineering study uses the original 48 training witnesses, including
the positive smoke case, to test the compiled scalar transport. Its panel,
absolute tolerance and physical handoff rules were fixed before the development
results. [NEF integrity policy](NASA_RESNET18_NEURAL16_NEF_V1.json) ·
[Independent biological study design](NASA_RESNET18_INDEPENDENT_SOURCE_DESIGN.md)
