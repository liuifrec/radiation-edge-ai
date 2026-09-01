param(
    [string]$Root = "D:\radiation-edge-ai-data"
)

$ErrorActionPreference = "Stop"

$paths = @{
    RADEDGE_DATA_ROOT  = Join-Path $Root "data"
    RADEDGE_MODEL_ROOT = Join-Path $Root "models"
    RADEDGE_CACHE_ROOT = Join-Path $Root "cache"
    HF_HOME            = Join-Path $Root "cache\huggingface"
    TORCH_HOME         = Join-Path $Root "cache\torch"
    PIP_CACHE_DIR      = Join-Path $Root "cache\pip"
}

$tmp = Join-Path $Root "cache\tmp"
$external = Join-Path $Root "external"
$envs = Join-Path $Root "envs"

foreach ($path in ($paths.Values + @($tmp, $external, $envs))) {
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}

foreach ($entry in $paths.GetEnumerator()) {
    Set-Item -Path "Env:$($entry.Key)" -Value $entry.Value
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "User")
}

# Keep large temporary build files off C: in this shell only. We deliberately
# do not change the user's global TEMP/TMP settings.
$env:TEMP = $tmp
$env:TMP = $tmp

Write-Host "Radiation Edge AI storage configured"
Write-Host "Root: $Root"
Write-Host ""
foreach ($key in $paths.Keys | Sort-Object) {
    Write-Host ("{0,-20} {1}" -f $key, $paths[$key])
}
Write-Host ("{0,-20} {1}" -f "TEMP (session)", $env:TEMP)
Write-Host ("{0,-20} {1}" -f "TMP (session)", $env:TMP)
Write-Host ("{0,-20} {1}" -f "External source", $external)
Write-Host ("{0,-20} {1}" -f "Large envs", $envs)
