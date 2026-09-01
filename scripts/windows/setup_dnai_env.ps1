param(
    [string]$Root = "D:\radiation-edge-ai-data",
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"

function Assert-LastExitCode([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

$toolsRoot = Join-Path $Root "tools"
$uvInstallDir = Join-Path $toolsRoot "uv"
$uvCacheDir = Join-Path $Root "cache\uv"
$uvPythonDir = Join-Path $Root "pythons"
$envDir = Join-Path $Root "envs\dnai311"
$dnaiSource = Join-Path $Root "external\DNAi"

$dirs = @($toolsRoot, $uvInstallDir, $uvCacheDir, $uvPythonDir, (Split-Path $envDir -Parent))
foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

if (-not (Test-Path $dnaiSource)) {
    throw "DNAi source tree not found at $dnaiSource"
}

# Keep uv's own downloads, managed Python, and environments off the system drive.
$env:UV_INSTALL_DIR = $uvInstallDir
$env:UV_CACHE_DIR = $uvCacheDir
$env:UV_PYTHON_INSTALL_DIR = $uvPythonDir

[Environment]::SetEnvironmentVariable("UV_CACHE_DIR", $uvCacheDir, "User")
[Environment]::SetEnvironmentVariable("UV_PYTHON_INSTALL_DIR", $uvPythonDir, "User")

$uv = Join-Path $uvInstallDir "uv.exe"
if (-not (Test-Path $uv)) {
    Write-Host "[Install uv]"
    Write-Host "Installing official Astral uv binary into $uvInstallDir"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}

if (-not (Test-Path $uv)) {
    throw "uv installation did not create $uv"
}

Write-Host "[uv]"
& $uv --version
Assert-LastExitCode "uv version check"

Write-Host "[Install managed Python $PythonVersion]"
& $uv python install $PythonVersion --install-dir $uvPythonDir
Assert-LastExitCode "managed Python installation"

Write-Host "[Create DNAi environment]"
if (Test-Path $envDir) {
    Write-Host "Environment already exists at $envDir; reusing it."
} else {
    & $uv venv $envDir --python $PythonVersion
    Assert-LastExitCode "DNAi virtual environment creation"
}

$python = Join-Path $envDir "Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "DNAi environment Python not found at $python"
}

Write-Host "[Install PyTorch CUDA 12.8 baseline]"
& $uv pip install --python $python `
    "torch==2.7.0+cu128" `
    "torchvision==0.22.0+cu128" `
    --index-url https://download.pytorch.org/whl/cu128
Assert-LastExitCode "PyTorch CUDA installation"

# albumentations 2.0.8 -> albucore 0.0.24 requires stringzilla>=3.10.4.
# stringzilla 5.1.2 currently lacks a CPython 3.11 Windows x86-64 wheel,
# which makes installers fall back to a local C++ build. 5.0.1 has a
# compatible cp311-win_amd64 wheel, so pin it for this reproducible Windows
# environment until the upstream wheel gap is resolved.
#
# Explicitly pin the CUDA PyTorch pair in the same transaction. Otherwise the
# general PyPI dependency resolver may replace the CUDA wheel with a newer CPU
# build while satisfying Lightning/torchmetrics dependencies.
Write-Host "[Install pinned DNAi source and declared dependencies]"
Write-Host "Windows compatibility pin: stringzilla==5.0.1"
Write-Host "CUDA pins: torch==2.7.0+cu128, torchvision==0.22.0+cu128"
& $uv pip install --python $python `
    -e $dnaiSource `
    "stringzilla==5.0.1" `
    "torch==2.7.0+cu128" `
    "torchvision==0.22.0+cu128" `
    --extra-index-url https://download.pytorch.org/whl/cu128
Assert-LastExitCode "DNAi dependency installation"

Write-Host "[Dependency check]"
& $uv pip check --python $python
Assert-LastExitCode "dependency check"

Write-Host "[Verify CUDA]"
& $python -c "import torch; print('torch:', torch.__version__); print('torch CUDA runtime:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2) if torch.cuda.is_available() else 0); assert torch.cuda.is_available(), 'CUDA PyTorch is required for the DNAi reference environment'"
Assert-LastExitCode "CUDA verification"

Write-Host "[Verify DNAi import]"
& $python -c "import dnafiber; from dnafiber.model.models_zoo import Models; print('dnafiber import: PASS'); print('reference candidate:', Models.UNET_MOBILEONE_S1.value)"
Assert-LastExitCode "DNAi import verification"

Write-Host ""
Write-Host "DNAi environment ready"
Write-Host "Python: $python"
Write-Host "Source: $dnaiSource"
Write-Host "Activate with:"
Write-Host "  & '$envDir\Scripts\Activate.ps1'"
