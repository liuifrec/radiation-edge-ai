param(
    [string]$Root = "D:\radiation-edge-ai-data",
    [switch]$SkipExtract
)

$ErrorActionPreference = "Stop"

$datasetRoot = Join-Path $Root "data\dnai_public_v2"
$downloadRoot = Join-Path $Root "data\downloads"
$zipPath = Join-Path $downloadRoot "DNAI_Data_V2.zip"
$extractRoot = Join-Path $datasetRoot "extracted"
$metadataPath = Join-Path $datasetRoot "SOURCE.txt"

$url = "https://zenodo.org/records/18868353/files/DNAI_Data_V2.zip?download=1"
$expectedMd5 = "52f205d5d6c81f2f0a9fb016d5fa534f"

New-Item -ItemType Directory -Force -Path $datasetRoot, $downloadRoot | Out-Null

function Test-ArchiveHash {
    if (-not (Test-Path $zipPath)) {
        return $false
    }
    Write-Host "[Verify existing archive]"
    $actual = (Get-FileHash -Path $zipPath -Algorithm MD5).Hash.ToLowerInvariant()
    Write-Host "MD5: $actual"
    return $actual -eq $expectedMd5
}

if (-not (Test-ArchiveHash)) {
    Write-Host "[Download DNAi public v2 dataset]"
    Write-Host "Destination: $zipPath"
    Write-Host "Expected download size: approximately 2.1 GB"
    Write-Host "Using curl resume mode; rerunning this script resumes a partial download."

    & curl.exe -L --fail --retry 5 --retry-delay 5 --continue-at - --output $zipPath $url
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset download failed with exit code $LASTEXITCODE"
    }

    Write-Host "[Verify downloaded archive]"
    $actual = (Get-FileHash -Path $zipPath -Algorithm MD5).Hash.ToLowerInvariant()
    Write-Host "Expected MD5: $expectedMd5"
    Write-Host "Actual MD5:   $actual"
    if ($actual -ne $expectedMd5) {
        throw "DNAi dataset checksum mismatch. Do not use this archive."
    }
} else {
    Write-Host "Archive already present and checksum-valid; download skipped."
}

@"
DNAi public dataset v2
Zenodo record: 18868353
DOI: 10.5281/zenodo.18868353
Archive: DNAI_Data_V2.zip
Expected MD5: $expectedMd5
Local archive: $zipPath
Retrieved/verified: $(Get-Date -Format o)
Upstream DNAi code baseline: fcf20c7d6eb385675ff7d07da4fdf471589ce0cf
"@ | Set-Content -Encoding UTF8 $metadataPath

if ($SkipExtract) {
    Write-Host "Extraction skipped by request."
    exit 0
}

if (Test-Path $extractRoot) {
    $items = Get-ChildItem -Path $extractRoot -Force -ErrorAction SilentlyContinue
    if ($items.Count -gt 0) {
        Write-Host "Extraction directory is already non-empty; leaving it unchanged: $extractRoot"
    } else {
        Write-Host "[Extract archive]"
        Expand-Archive -Path $zipPath -DestinationPath $extractRoot -Force
    }
} else {
    New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null
    Write-Host "[Extract archive]"
    Expand-Archive -Path $zipPath -DestinationPath $extractRoot -Force
}

Write-Host ""
Write-Host "DNAi public v2 dataset ready"
Write-Host "Archive:   $zipPath"
Write-Host "Extracted: $extractRoot"
Write-Host ""
Write-Host "Top-level extracted contents:"
Get-ChildItem -Path $extractRoot | Select-Object Name, Mode, Length
