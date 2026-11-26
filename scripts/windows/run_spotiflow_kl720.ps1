# Run from a source checkout with the recorded Spotiflow and PLUS installations.
[CmdletBinding(DefaultParameterSetName='Image')]
param(
    [Parameter(Mandatory,ParameterSetName='Image')][string]$Image,
    [Parameter(ParameterSetName='Image')][string]$Mask,
    [Parameter(ParameterSetName='Image')][int]$TargetLabel,
    [Parameter(Mandatory,ParameterSetName='Preflight')][switch]$Preflight,
    [Parameter(Mandatory,ParameterSetName='ReplaySimulator')][switch]$ReplaySimulator,
    [Parameter(Mandatory,ParameterSetName='Smoke')][switch]$Smoke,
    [Parameter(Mandatory,ParameterSetName='Panel')][switch]$Panel,
    [Parameter(Mandatory,ParameterSetName='Panel')][string]$SmokeReceipt,
    [Parameter(Mandatory,ParameterSetName='Panel')][string]$SmokeSha256,
    [string]$Out,
    [string]$DataRoot = $env:RADEDGE_DATA_ROOT,
    [string]$SpotiflowPython,
    [string]$PlusEnvironment = "$env:USERPROFILE\venvs\kneron720",
    [int]$Port
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $DataRoot) { throw 'Supply -DataRoot or set RADEDGE_DATA_ROOT.' }
if (-not $SpotiflowPython) {
    $SpotiflowPython = Join-Path $DataRoot 'scratch\spotiflow_pretrained_models_20261119_native_v1\venv\Scripts\python.exe'
}
if (-not $Out) {
    $Out = Join-Path $DataRoot ('scratch\spotiflow_kl720_' + [guid]::NewGuid().ToString('N'))
}
$arguments = @('--model', 'spotiflow/general', '--variant', 'full_int16_transport', '--data-root', $DataRoot, '--out', $Out)
if ($PSBoundParameters.ContainsKey('Port')) { $arguments += @('--port', $Port) }
if ($Smoke -or $Panel -or $Preflight -or $ReplaySimulator) {
    $runner = Join-Path $PSScriptRoot 'run_spotiflow_physical_panel.py'
    $arguments += @('--plus-site', (Join-Path $PlusEnvironment 'Lib\site-packages'))
    if ($Preflight) { $arguments += '--preflight-only' }
    if ($ReplaySimulator) { $arguments += '--replay-simulator' }
    if ($Smoke) { $arguments += '--smoke' }
    if ($Panel) { $arguments += @('--panel', '--smoke-receipt', $SmokeReceipt, '--smoke-sha256', $SmokeSha256) }
} else {
    if ([bool]$Mask -ne $PSBoundParameters.ContainsKey('TargetLabel')) {
        throw 'Supply both -Mask and -TargetLabel for a nucleus-specific count.'
    }
    $runner = Join-Path $PSScriptRoot 'run_registered_spotiflow.py'
    $arguments += @('--backend', 'kl720', '--image', $Image, '--plus-python', (Join-Path $PlusEnvironment 'Scripts\python.exe'))
    if ($Mask) { $arguments += @('--mask', $Mask, '--target-label', $TargetLabel) }
}
$previousPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repoRoot 'src'
    & $SpotiflowPython -B $runner @arguments
    $result = $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $previousPath
}
if ($result -eq 0 -and -not $Preflight) { Write-Output (Join-Path $Out 'receipt.json') }
exit $result
