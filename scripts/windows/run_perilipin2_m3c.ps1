<#
.SYNOPSIS
    Run experimental Perilipin2 M3C gates on Windows/Alienware.
.DESCRIPTION
    One host interface for a future GUI. Docker is used ONLY for Kneron graph
    evaluation. Physical KL720 probing uses the existing isolated PLUS Python.
    This script NEVER installs Docker, downloads an image, runs PTQ, or compiles NEF.
.EXAMPLE
    .\scripts\windows\run_perilipin2_m3c.ps1 -Stage Doctor
.EXAMPLE
    .\scripts\windows\run_perilipin2_m3c.ps1 -Stage Import -SourceFolder 'D:\radiation-edge-ai-data\models\perilipin2\m3b_poc_v1'
#>
[CmdletBinding()]
param(
    [ValidateSet('Doctor', 'Import', 'Evaluate', 'Parity', 'HardwareProbe')]
    [string]$Stage = 'Doctor',
    [string]$DataRoot = 'D:\radiation-edge-ai-data',
    [string]$RunName = 'perilipin2_m3c_poc_v1',
    [string]$SourceFolder = '',
    [string]$Image = 'kneron/toolchain:latest',
    [string]$Python = 'python',
    [string]$Kl720Python = ''
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Runner = Join-Path $RepoRoot 'src\radiation_edge_ai\perilipin2\windows_cli.py'
if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
    throw "Repository Perilipin2 runner missing: $Runner"
}
if ($RunName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$') {
    throw 'RunName must be a safe, simple directory name (1-80 characters).'
}
$WorkRoot = Join-Path (Join-Path $DataRoot 'scratch') $RunName
Write-Host 'Radiation Edge AI / Perilipin2 experimental M3C'
Write-Host "Stage: $Stage"
Write-Host "Work:  $WorkRoot"
Write-Host 'No reference images, masks, ONNX, weights or NEFs will be committed to Git.'

function Invoke-CheckedPython {
    param([string[]]$ScriptArguments)
    & $Python $Runner @ScriptArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Perilipin2 $Stage failed with Python exit code $LASTEXITCODE. Keep the work folder for investigation."
    }
}

switch ($Stage) {
    'Doctor' {
        Invoke-CheckedPython -ScriptArguments @('doctor', '--image', $Image)
        Write-Host 'To scan the plugged-in KL720 separately, use -Stage HardwareProbe.'
    }
    'Import' {
        if (-not $SourceFolder) {
            throw 'Pass -SourceFolder pointing to the copied M3B_ONNX_v1 folder on this Windows PC.'
        }
        $Onnx = Join-Path $SourceFolder 'nuclear_foreground_boundary_512_opset11.onnx'
        $Parity = Join-Path $SourceFolder 'ONNX_PARITY.json'
        foreach ($p in @($Onnx, $Parity)) {
            if (-not (Test-Path -LiteralPath $p -PathType Leaf)) {
                throw "M3B model asset missing: $p"
            }
        }
        Invoke-CheckedPython -ScriptArguments @(
            'import', '--onnx', $Onnx, '--parity', $Parity, '--out', $WorkRoot
        )
    }
    'Evaluate' {
        if (-not (Test-Path -LiteralPath (Join-Path $WorkRoot 'M3C_HOST_READY.json'))) {
            throw 'Import must PASS before Kneron graph evaluation.'
        }
        Invoke-CheckedPython -ScriptArguments @('evaluate', '--work', $WorkRoot, '--image', $Image)
    }
    'Parity' {
        if (-not (Test-Path -LiteralPath (Join-Path $WorkRoot 'M3C_KNERON_EVAL_RECEIPT.json'))) {
            throw 'Kneron evaluation receipt missing. Run -Stage Evaluate first.'
        }
        Invoke-CheckedPython -ScriptArguments @('parity', '--work', $WorkRoot)
    }
    'HardwareProbe' {
        if (-not $Kl720Python) { $Kl720Python = $env:RADEDGE_KL720_PYTHON }
        if (-not $Kl720Python -or -not (Test-Path -LiteralPath $Kl720Python -PathType Leaf)) {
            throw 'Provide -Kl720Python with the known-good Kneron PLUS Python executable, or set RADEDGE_KL720_PYTHON.'
        }
        $Probe = Join-Path $RepoRoot 'src\radiation_edge_ai\hardware\kl720_check.py'
        & $Kl720Python $Probe
        if ($LASTEXITCODE -ne 0) { throw 'Kneron PLUS device probe failed.' }
        Write-Host 'Hardware discovered only. Perilipin2 NEF is not compiled and no inference was run.'
    }
}