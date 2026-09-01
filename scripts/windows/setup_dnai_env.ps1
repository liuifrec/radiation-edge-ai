param(
    [string]$Root = "D:\radiation-edge-ai-data",
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"

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

Write-Host "[Install managed Python $PythonVersion]"
& $uv python install $PythonVersion --install-dir $uvPythonDir

Write-Host "[Create DNAi environment]"
if (Test-Path $envDir) {
    Write-Host "Environment already exists at $envDir; reusing it."
} else {
    & $uv venv $envDir --python $PythonVersion
}

$python = Join-Path $envDir "Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "DNAi environment Python not found at $python"
}

Write-Host "[Install PyTorch CUDA 12.8 baseline]"
& $uv pip install --python $python `
    "torch==2.7.0" `
    "torchvision==0.22.0" `
    --index-url https://download.pytorch.org/whl/cu128

Write-Host "[Install pinned DNAi source and declared dependencies]"
& $uv pip install --python $python -e $dnaiSource

Write-Host "[Verify CUDA]"
& $python -c "import torch; print('torch:', torch.__version__); print('torch CUDA runtime:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2) if torch.cuda.is_available() else 0)"

Write-Host "[Verify DNAi import]"
& $python -c "import dnafiber; from dnafiber.model.models_zoo import Models; print('dnafiber import: PASS'); print('reference candidate:', Models.UNET_MOBILEONE_S1.value)"

Write-Host ""
Write-Host "DNAi environment ready"
Write-Host "Python: $python"
Write-Host "Source: $dnaiSource"
Write-Host "Activate with:"
Write-Host "  & '$envDir\Scripts\Activate.ps1'"
