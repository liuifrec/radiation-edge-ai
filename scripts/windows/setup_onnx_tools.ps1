param(
    [string]$Python = "D:\radiation-edge-ai-data\envs\dnai311\Scripts\python.exe",
    [string]$Uv = "D:\radiation-edge-ai-data\tools\uv\uv.exe"
)

$ErrorActionPreference = "Stop"

function Assert-LastExitCode([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $Python)) {
    throw "DNAi Python not found: $Python"
}
if (-not (Test-Path $Uv)) {
    throw "uv not found: $Uv"
}

Write-Host "[Install ONNX verification tools]"
& $Uv pip install --python $Python onnx onnxruntime
Assert-LastExitCode "ONNX tool installation"

Write-Host "[Dependency check]"
& $Uv pip check --python $Python
Assert-LastExitCode "Dependency check"

Write-Host "[Verify imports]"
& $Python -c "import onnx, onnxruntime as ort; print('onnx:', onnx.__version__); print('onnxruntime:', ort.__version__); print('ORT providers:', ort.get_available_providers())"
Assert-LastExitCode "ONNX import verification"

Write-Host ""
Write-Host "ONNX TOOLS READY: YES"
Write-Host "Python: $Python"
