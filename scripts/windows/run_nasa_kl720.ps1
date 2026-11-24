<# Run the NASA baseline on the connected KL720 using the existing PLUS runtime. #>
[CmdletBinding()]
param(
    [string]$Root = 'D:\radiation-edge-ai-data',
    [string]$Python = $env:RADEDGE_KL720_PYTHON,
    [string]$OutputDir,
    [int]$Port = -1,
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $env:USERPROFILE 'venvs\kneron720\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Kneron PLUS Python not found at '$Python'. Pass -Python '<existing PLUS python.exe>'."
}

$Runner = Join-Path $PSScriptRoot 'run_nasa_kl720.py'
$RunnerArgs = @('-X', 'utf8', '-u', $Runner, '--root', $Root)
if ($Port -ge 0) { $RunnerArgs += @('--port', [string]$Port) }
if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
    $RunnerArgs += @('--output-dir', $OutputDir)
}
if ($PreflightOnly) { $RunnerArgs += '--preflight-only' }

Write-Host 'Using your existing Kneron PLUS environment:' $Python
& $Python @RunnerArgs
if ($LASTEXITCODE -ne 0) {
    throw "NASA KL720 run exited with code $LASTEXITCODE. Read the message above and hardware_run.log if created."
}
