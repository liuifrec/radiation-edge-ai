[CmdletBinding()]
param(
    [ValidateSet('Smoke', 'Training', 'DevelopmentReference', 'Development')]
    [string]$Mode = 'Smoke',
    [switch]$Execute,
    [string]$SmokeReceipt,
    [ValidatePattern('^[0-9a-f]{64}$')][string]$SmokeSha256,
    [string]$ReferenceDir,
    [ValidatePattern('^[0-9a-f]{64}$')][string]$ReferenceSha256,
    [string]$DataRoot = 'D:\radiation-edge-ai-data',
    [string]$PlusPython = (Join-Path $env:USERPROFILE 'venvs\kneron720\Scripts\python.exe'),
    [string]$Out
)
$ErrorActionPreference = 'Stop'
$campaign = Join-Path $DataRoot 'scratch\nasa_resnet_neural16_dev_20261127_v1'
$compileDir = Join-Path $campaign 'nef_compile'
$trainingDir = Join-Path $campaign 'nef_sim'
$auditSha = '4f9dd6f061586840636de0aa4292944f880bef6325e4f0dcc0f90f7aa571bf20'
$compileSha = 'd068c2b2647edd6a75749bae2d34b7f50e241843a42ec5ed1ac62f92c8130f06'
$trainingSha = '1706ad42edcf401b055077534e6839e63597449f783a0422ab70ebce4d60d59b'

$physicalPanel = $Execute -and ($Mode -eq 'Training' -or $Mode -eq 'Development')
if ($physicalPanel -and (-not $SmokeReceipt -or -not $SmokeSha256)) {
    throw 'Physical panel execution requires -SmokeReceipt and -SmokeSha256 from a passing device smoke.'
}
if (-not $physicalPanel -and ($SmokeReceipt -or $SmokeSha256)) {
    throw 'Supply smoke evidence only for an explicitly requested physical panel.'
}
if ($Mode -eq 'Development' -and (-not $ReferenceDir -or -not $ReferenceSha256)) {
    throw 'Development replay requires -ReferenceDir and -ReferenceSha256 from the completed development simulator run.'
}
if ($Mode -ne 'Development' -and ($ReferenceDir -or $ReferenceSha256)) {
    throw 'Supply development reference evidence only in Development mode.'
}
if ($Mode -eq 'DevelopmentReference' -and $Execute) {
    throw 'DevelopmentReference runs the NEF simulator. Omit -Execute.'
}
if (-not $Out) {
    $Out = Join-Path $campaign ('replay_' + $Mode.ToLowerInvariant() + '_' + [guid]::NewGuid().ToString('N'))
}

if ($Mode -eq 'DevelopmentReference') {
    $arguments = @('-B', '-m', 'radiation_edge_ai.nasa_bps.neural16_nef', 'development-reference',
        '--data', $DataRoot, '--run', $campaign, '--audit-sha256', $auditSha, '--compiler-output', $compileDir,
        '--compile-sha256', $compileSha, '--training-simulation', $trainingDir,
        '--training-simulation-sha256', $trainingSha, '--out', $Out)
    $receiptName = 'development-reference.json'
} else {
    $arguments = @('-B', (Join-Path $PSScriptRoot 'run_resnet_neural16_usb.py'), $Mode.ToLowerInvariant(),
        '--data', $DataRoot, '--campaign', $campaign, '--compile-dir', $compileDir, '--audit-sha256', $auditSha,
        '--compile-sha256', $compileSha, '--out', $Out)
    if ($Mode -eq 'Development') {
        $arguments += @('--sim-dir', $ReferenceDir, '--sim-sha256', $ReferenceSha256, '--training-sim-dir', $trainingDir)
    } else {
        $arguments += @('--sim-dir', $trainingDir, '--sim-sha256', $trainingSha)
    }
    if ($physicalPanel) { $arguments += @('--smoke-receipt', $SmokeReceipt, '--smoke-sha256', $SmokeSha256) }
    if ($Execute) { $arguments += '--execute' }
    $receiptName = 'receipt.json'
}

$previousPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = (Resolve-Path (Join-Path $PSScriptRoot '..\..\src')).Path
    Write-Host "Output directory: $Out"
    & $PlusPython @arguments
    $code = $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $previousPath
}
if ($code -eq 0) {
    $receipt = Join-Path $Out $receiptName
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = $algorithm.ComputeHash([IO.File]::ReadAllBytes($receipt))
        $sha = [BitConverter]::ToString($bytes).Replace('-', '').ToLowerInvariant()
    } finally {
        $algorithm.Dispose()
    }
    Write-Host "Receipt: $receipt"
    Write-Host "SHA256: $sha"
}
exit $code
