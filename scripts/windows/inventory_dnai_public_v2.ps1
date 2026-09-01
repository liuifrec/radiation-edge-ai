param(
    [string]$Root = "D:\radiation-edge-ai-data"
)

$ErrorActionPreference = "Stop"

$extractRoot = Join-Path $Root "data\dnai_public_v2\extracted"
if (-not (Test-Path $extractRoot)) {
    throw "DNAi extracted dataset not found at $extractRoot"
}

Write-Host "DNAi public v2 inventory"
Write-Host "Root: $extractRoot"
Write-Host ""

Write-Host "[Directory tree: first 3 levels]"
# Windows PowerShell 5.1 runs on .NET Framework, which does not provide
# System.IO.Path.GetRelativePath(). Since every item below is discovered under
# $extractRoot, a prefix substring is sufficient and works on both PS 5.1 and 7+.
Get-ChildItem -Path $extractRoot -Directory -Recurse |
    ForEach-Object {
        $relative = $_.FullName.Substring($extractRoot.Length)
        while ($relative.StartsWith("\") -or $relative.StartsWith("/")) {
            $relative = $relative.Substring(1)
        }
        $normalized = $relative.Replace("\", "/")
        $parts = @($normalized.Split("/") | Where-Object { $_.Length -gt 0 })
        $depth = $parts.Count
        if ($depth -le 3) {
            [PSCustomObject]@{
                Depth = $depth
                Path = $relative
            }
        }
    } |
    Sort-Object Depth, Path |
    Format-Table -AutoSize

Write-Host ""
Write-Host "[File-extension counts]"
Get-ChildItem -Path $extractRoot -File -Recurse |
    Group-Object Extension |
    Sort-Object Count -Descending |
    Select-Object Count, Name |
    Format-Table -AutoSize

Write-Host ""
Write-Host "[Candidate test / annotation / grader paths]"
$candidates = Get-ChildItem -Path $extractRoot -Recurse -Force |
    Where-Object {
        $_.FullName -match '(?i)test|annot|mask|grader|grade|ground.?truth|validation|val'
    } |
    Select-Object -First 120 FullName, PSIsContainer, Length

if ($candidates) {
    $candidates | Format-Table -AutoSize
} else {
    Write-Host "No paths matched the candidate keywords."
}

Write-Host ""
Write-Host "[First 40 image-like files]"
Get-ChildItem -Path $extractRoot -File -Recurse |
    Where-Object { $_.Extension -match '(?i)^\.(tif|tiff|png|jpg|jpeg|czi|dv)$' } |
    Select-Object -First 40 FullName, Length |
    Format-Table -AutoSize

Write-Host ""
Write-Host "[Dataset totals]"
$allFiles = @(Get-ChildItem -Path $extractRoot -File -Recurse)
$imageFiles = @($allFiles | Where-Object { $_.Extension -match '(?i)^\.(tif|tiff|png|jpg|jpeg|czi|dv)$' })
$totalBytes = ($allFiles | Measure-Object Length -Sum).Sum
if ($null -eq $totalBytes) {
    $totalBytes = 0
}

Write-Host ("Files total:      {0}" -f $allFiles.Count)
Write-Host ("Image-like files: {0}" -f $imageFiles.Count)
Write-Host ("Extracted size:    {0:N2} GiB" -f ($totalBytes / 1GB))
