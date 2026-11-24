<#
.SYNOPSIS
Prepare or score a blinded Perilipin2 Track A QC review without rerunning images.
Only the dedicated Track A CPU scientific environment is used. No KL720/Docker.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("Prepare","VerifyPack","Score")]
    [string]$Stage,
    [string]$Python = "D:\radiation-edge-ai-data\envs\perilipin2-tracka-py311-v2\Scripts\python.exe",
    [string]$Bundle = "D:\radiation-edge-ai-data\references\Perilipin2_TrackA_portable_20261111_v1",
    [string]$Run = "D:\radiation-edge-ai-data\scratch\Perilipin2_TrackA_windows_20261111T054214Z_3f7dbcdb",
    [string]$Pack = "",
    [string]$Review = "",
    [string]$Out = "",
    [double]$Radius = 6.0
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Scientific Python missing: $Python"
}
$OldPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $Root 'src'
    if ($Stage -eq 'Prepare') {
        if (-not $Out) { throw 'Prepare needs a NEW -Out directory.' }
        & $Python -X utf8 -u -m radiation_edge_ai.perilipin2_qc prepare --bundle $Bundle --run $Run --out $Out
    } elseif ($Stage -eq 'VerifyPack') {
        if (-not $Pack) { throw 'VerifyPack needs -Pack.' }
        & $Python -X utf8 -u -m radiation_edge_ai.perilipin2_qc verify-pack --pack $Pack
    } else {
        if (-not $Pack -or -not $Review -or -not $Out) {
            throw 'Score needs -Pack, -Review, and a NEW -Out directory.'
        }
        if ($Radius -le 0 -or [double]::IsInfinity($Radius) -or [double]::IsNaN($Radius)) {
            throw 'Use a positive finite, prespecified match radius.'
        }
        & $Python -X utf8 -u -m radiation_edge_ai.perilipin2_qc score --bundle $Bundle --run $Run --pack $Pack --review $Review --out $Out --radius $Radius
    }
    if ($LASTEXITCODE -ne 0) {
        throw 'Track A QC stage stopped. Preserve the source and any .BUILDING diagnostics.'
    }
} finally {
    $env:PYTHONPATH = $OldPath
}
