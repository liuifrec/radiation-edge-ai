$ErrorActionPreference = "Stop"

$ExpectedDockerDataRoot = "D:\radiation-edge-ai-data\docker-desktop-wsl"

Write-Host "Radiation Edge AI - Kneron Toolchain Host Probe"
Write-Host ""

Write-Host "[Drive space]"
Get-PSDrive -PSProvider FileSystem |
    Where-Object { $_.Name -in @("C", "D") } |
    Select-Object Name,
        @{Name="UsedGB";Expression={[math]::Round($_.Used / 1GB, 1)}},
        @{Name="FreeGB";Expression={[math]::Round($_.Free / 1GB, 1)}},
        Root |
    Format-Table -AutoSize

Write-Host "[Docker command]"
$docker = Get-Command docker -ErrorAction SilentlyContinue
if ($null -eq $docker) {
    Write-Host "docker: NOT FOUND"
    Write-Host ""
    Write-Host "KNERON TOOLCHAIN HOST READY: NO"
    Write-Host "Reason: Docker CLI is not installed or not on PATH."
    exit 2
}
Write-Host ("docker: {0}" -f $docker.Source)

Write-Host ""
Write-Host "[Docker client]"
$oldPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
docker version --format "Client={{.Client.Version}}" 2>$null
$clientExit = $LASTEXITCODE
$ErrorActionPreference = $oldPreference
if ($clientExit -ne 0) {
    docker --version
}

Write-Host ""
Write-Host "[Docker daemon]"
$ErrorActionPreference = "Continue"
docker info --format "Server={{.ServerVersion}}`nDockerRootDir={{.DockerRootDir}}`nDriver={{.Driver}}`nOSType={{.OSType}}`nArchitecture={{.Architecture}}" 2>$null
$daemonExit = $LASTEXITCODE
$ErrorActionPreference = $oldPreference
if ($daemonExit -ne 0) {
    Write-Host "Docker daemon: NOT REACHABLE"
    Write-Host "Start Docker Desktop, wait until it reports that the engine is running, then rerun this probe."
    Write-Host ""
    Write-Host "KNERON TOOLCHAIN HOST READY: NO"
    exit 3
}

Write-Host ""
Write-Host "[Host Docker storage]"
if (Test-Path $ExpectedDockerDataRoot) {
    Write-Host "Expected D: Docker data root: PRESENT"
    Write-Host $ExpectedDockerDataRoot
    $vhdx = Get-ChildItem -Path $ExpectedDockerDataRoot -Recurse -File -Filter *.vhdx -ErrorAction SilentlyContinue
    if ($vhdx) {
        $vhdx |
            Select-Object FullName,
                @{Name="SizeGB";Expression={[math]::Round($_.Length / 1GB, 2)}} |
            Format-Table -AutoSize
    } else {
        Write-Host "No VHDX located under the expected root yet; Docker Desktop may be using its newer managed storage layout."
    }
} else {
    Write-Host "Expected D: Docker data root: NOT FOUND"
    Write-Host "Expected: $ExpectedDockerDataRoot"
    Write-Host "Check Docker Desktop > Settings > Resources/Advanced before pulling large images."
}

Write-Host ""
Write-Host "[Existing Kneron toolchain image]"
# `docker image inspect` emits an expected error when an image does not exist.
# Querying the image list avoids PowerShell converting that stderr into a
# terminating RemoteException under ErrorActionPreference=Stop.
$imageId = docker images --filter "reference=kneron/toolchain:latest" --format "{{.ID}}"
if (-not [string]::IsNullOrWhiteSpace($imageId)) {
    $imageSize = docker image inspect kneron/toolchain:latest --format "{{.Size}}"
    $sizeGiB = [math]::Round(([double]$imageSize / 1GB), 2)
    $repoDigest = docker image inspect kneron/toolchain:latest --format "{{join .RepoDigests ","}}"
    Write-Host "kneron/toolchain:latest: PRESENT"
    Write-Host ("Image ID: {0}" -f $imageId.Trim())
    Write-Host ("Repo digest: {0}" -f $repoDigest)
    Write-Host ("Uncompressed image size: {0} GiB" -f $sizeGiB)
} else {
    Write-Host "kneron/toolchain:latest: NOT PRESENT"
    Write-Host "No image was pulled by this probe."
}

Write-Host ""
Write-Host "[Docker disk usage]"
docker system df

Write-Host ""
Write-Host "[WSL distributions]"
$wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
if ($null -ne $wsl) {
    wsl.exe -l -v
} else {
    Write-Host "wsl.exe: not found"
}

Write-Host ""
Write-Host "KNERON TOOLCHAIN HOST READY: YES"
Write-Host "NOTE: This probe does not pull the Kneron toolchain image."
