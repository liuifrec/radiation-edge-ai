$ErrorActionPreference = "Stop"

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
docker version --format "Client={{.Client.Version}}" 2>$null
$clientExit = $LASTEXITCODE
if ($clientExit -ne 0) {
    docker --version
}

Write-Host ""
Write-Host "[Docker daemon]"
docker info --format "Server={{.ServerVersion}}`nDockerRootDir={{.DockerRootDir}}`nDriver={{.Driver}}`nOSType={{.OSType}}`nArchitecture={{.Architecture}}" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker daemon: NOT REACHABLE"
    Write-Host "Start Docker Desktop, wait until it reports that the engine is running, then rerun this probe."
    Write-Host ""
    Write-Host "KNERON TOOLCHAIN HOST READY: NO"
    exit 3
}

Write-Host ""
Write-Host "[Existing Kneron toolchain image]"
$imageId = docker image inspect kneron/toolchain:latest --format "{{.Id}}" 2>$null
if ($LASTEXITCODE -eq 0) {
    $imageSize = docker image inspect kneron/toolchain:latest --format "{{.Size}}"
    $sizeGiB = [math]::Round(([double]$imageSize / 1GB), 2)
    Write-Host "kneron/toolchain:latest: PRESENT"
    Write-Host ("Image ID: {0}" -f $imageId)
    Write-Host ("Uncompressed image size: {0} GiB" -f $sizeGiB)
} else {
    Write-Host "kneron/toolchain:latest: NOT PRESENT"
    Write-Host "No image was pulled by this probe."
}

Write-Host ""
Write-Host "[Docker disk usage]"
docker system df 2>$null

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
Write-Host "NOTE: This probe does not pull the approximately multi-GB Kneron image."
Write-Host "      Confirm Docker storage is safe before pulling, especially with limited C: free space."
