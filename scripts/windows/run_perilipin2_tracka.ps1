<#
.SYNOPSIS
Recompute Perilipin2 Track A M2 measurements with frozen saved nuclear masks.
No CUDA, Docker, KL720, retraining or automatic package installation.
#>
[CmdletBinding()]
param(
    [ValidateSet('Analyze', 'VerifyBundle', 'VerifyResults')]
    [string]$Stage = 'Analyze',
    [Parameter(Mandatory=$true)] [string]$Python,
    [string]$Bundle = '',
    [string]$Out = ''
)
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Runner = Join-Path $RepoRoot 'src\radiation_edge_ai\perilipin2_tracka.py'
if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) { throw 'Track A adapter is missing.' }
Write-Host 'Perilipin2 Track A: actual RGB measurements, frozen M1 nuclei, CPU only'
Write-Host 'No new nuclei, whole-cell segmentation or KL720 inference is performed.'
switch ($Stage) {
    'Analyze' {
        if (-not $Bundle -or -not $Out) { throw 'Analyze requires -Bundle and a NEW -Out folder.' }
        & $Python -X utf8 -u $Runner run --bundle $Bundle --out $Out
    }
    'VerifyBundle' {
        if (-not $Bundle) { throw 'VerifyBundle requires -Bundle.' }
        & $Python -X utf8 -u $Runner verify-bundle $Bundle
    }
    'VerifyResults' {
        if (-not $Out) { throw 'VerifyResults requires -Out.' }
        & $Python -X utf8 -u $Runner verify $Out
    }
}
if ($LASTEXITCODE -ne 0) {
    throw 'Track A did not pass. Preserve the output/BUILDING folder and its diagnostic receipt.'
}
