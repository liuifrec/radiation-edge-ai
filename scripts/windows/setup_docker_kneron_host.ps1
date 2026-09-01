param(
    [switch]$Install
)

$ErrorActionPreference = "Stop"

$Root = "D:\radiation-edge-ai-data"
$ToolsRoot = Join-Path $Root "tools"
$DockerInstallRoot = Join-Path $ToolsRoot "DockerDesktop"
$DockerDataRoot = Join-Path $Root "docker-desktop-wsl"
$InstallerRoot = Join-Path $ToolsRoot "docker-installer"
$Installer = Join-Path $InstallerRoot "Docker Desktop Installer.exe"
$InstallerUrl = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"

function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

Write-Host "Radiation Edge AI - Docker/Kneron Host Setup"
Write-Host ""
Write-Host "Heavy Docker data root: $DockerDataRoot"
Write-Host "Docker program root:    $DockerInstallRoot"
Write-Host ""

Write-Host "[Disk space]"
Get-PSDrive -PSProvider FileSystem |
    Where-Object { $_.Name -in @("C", "D") } |
    Select-Object Name,
        @{Name="UsedGB";Expression={[math]::Round($_.Used / 1GB, 1)}},
        @{Name="FreeGB";Expression={[math]::Round($_.Free / 1GB, 1)}},
        Root |
    Format-Table -AutoSize

Write-Host "[WSL]"
$wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
if ($null -eq $wsl) {
    Write-Host "wsl.exe: NOT FOUND"
    Write-Host "Run in an Administrator PowerShell:"
    Write-Host "  wsl --install --no-distribution"
    Write-Host "Then reboot Windows and rerun this script."
    exit 2
}

wsl.exe --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "WSL is present but its modern version command failed."
    Write-Host "Run in an Administrator PowerShell: wsl --update"
    exit 3
}

Write-Host ""
Write-Host "[WSL distributions]"
wsl.exe -l -v

$docker = Get-Command docker -ErrorAction SilentlyContinue
if ($null -ne $docker) {
    Write-Host ""
    Write-Host "Docker CLI already exists: $($docker.Source)"
    Write-Host "Start Docker Desktop if necessary, then run:"
    Write-Host "  .\scripts\windows\probe_kneron_toolchain.ps1"
    exit 0
}

if (-not $Install) {
    Write-Host ""
    Write-Host "Docker Desktop is not installed."
    Write-Host "This script has made NO system changes."
    Write-Host ""
    Write-Host "To install Docker Desktop with its heavy WSL data on D:,"
    Write-Host "open PowerShell AS ADMINISTRATOR and run:"
    Write-Host ""
    Write-Host "  cd C:\Users\yul03\Projects\radiation-edge-ai"
    Write-Host "  .\scripts\windows\setup_docker_kneron_host.ps1 -Install"
    Write-Host ""
    exit 10
}

if (-not (Test-Admin)) {
    throw "-Install requires an Administrator PowerShell. Reopen PowerShell as Administrator and rerun."
}

Write-Host ""
Write-Host "[Update WSL]"
wsl.exe --update
if ($LASTEXITCODE -ne 0) {
    throw "wsl --update failed with exit code $LASTEXITCODE"
}

New-Item -ItemType Directory -Force $InstallerRoot | Out-Null
New-Item -ItemType Directory -Force $DockerDataRoot | Out-Null
New-Item -ItemType Directory -Force $DockerInstallRoot | Out-Null

if (-not (Test-Path $Installer)) {
    Write-Host ""
    Write-Host "[Download Docker Desktop installer]"
    Write-Host $InstallerUrl
    Invoke-WebRequest -Uri $InstallerUrl -OutFile $Installer
} else {
    Write-Host ""
    Write-Host "[Docker Desktop installer]"
    Write-Host "Using existing installer: $Installer"
}

Write-Host ""
Write-Host "[Install Docker Desktop]"
Write-Host "Backend: WSL 2"
Write-Host "WSL data root: $DockerDataRoot"
Write-Host "Installation root: $DockerInstallRoot"

$arguments = @(
    "install",
    "--accept-license",
    "--backend=wsl-2",
    "--wsl-default-data-root=$DockerDataRoot",
    "--installation-dir=$DockerInstallRoot"
)

$process = Start-Process -FilePath $Installer -ArgumentList $arguments -Wait -PassThru
if ($process.ExitCode -ne 0) {
    throw "Docker Desktop installer failed with exit code $($process.ExitCode)"
}

Write-Host ""
Write-Host "DOCKER DESKTOP INSTALL COMPLETE: YES"
Write-Host ""
Write-Host "Next:"
Write-Host "  1. Start Docker Desktop from the Start menu."
Write-Host "  2. Wait until Docker Engine reports Running."
Write-Host "  3. If Windows requests a reboot, reboot first."
Write-Host "  4. In a normal PowerShell run:"
Write-Host "       cd C:\Users\yul03\Projects\radiation-edge-ai"
Write-Host "       .\scripts\windows\probe_kneron_toolchain.ps1"
Write-Host ""
Write-Host "Do NOT pull kneron/toolchain until the probe confirms Docker is using the intended storage location."
