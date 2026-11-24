<#
.SYNOPSIS
Launch the experimental local GUI over the existing radedge workflows.
No Docker, CUDA, package installation or model downloads are performed.
#>
[CmdletBinding()]
param(
    [string]$Python = 'python',
    [string]$DataRoot = 'D:\radiation-edge-ai-data'
)
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$PreviousPythonPath = $env:PYTHONPATH
$PreviousIOEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONPATH = Join-Path $RepoRoot 'src'
    $env:PYTHONIOENCODING = 'utf-8:backslashreplace'
    & $Python -m radiation_edge_ai.desktop --data-root $DataRoot
    if ($LASTEXITCODE -ne 0) {
        throw 'Desktop launcher failed. Keep console output; do not change frozen models or SDK.'
    }
} finally {
    $env:PYTHONPATH = $PreviousPythonPath
    $env:PYTHONIOENCODING = $PreviousIOEncoding
}
