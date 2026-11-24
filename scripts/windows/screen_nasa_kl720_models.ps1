<# Export complete NASA candidate graphs and screen them in an existing Kneron image. #>
[CmdletBinding()]
param(
    [string]$Root = 'D:\radiation-edge-ai-data',
    [Alias('Python')][string]$TrainingPython = $env:RADEDGE_NASA_PYTHON,
    [string]$ToolchainImage = 'kneron/toolchain@sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041',
    [string]$ContainerPython = '/workspace/miniconda/bin/python',
    [string]$OutputDir,
    [ValidateSet('r1countnet_v1', 'mobilenet_v2', 'resnet18')]
    [string[]]$Models = @('r1countnet_v1', 'mobilenet_v2', 'resnet18'),
    [ValidateRange(0, 2147483647)][int]$Seed = 720,
    [switch]$ExportOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Write-JsonFile {
    param([string]$Path, [object]$Value)
    $Encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 30), $Encoding)
}

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Invoke-NativeCaptured {
    param([string]$FilePath, [string[]]$ArgumentList, [string]$LogPath)
    $PreviousPreference = $ErrorActionPreference
    try {
        # Native stderr is evidence to retain; the actual process exit code decides success.
        $ErrorActionPreference = 'Continue'
        $Lines = @(& $FilePath @ArgumentList 2>&1 | ForEach-Object {
            $Line = $_.ToString()
            if (-not [string]::IsNullOrWhiteSpace($LogPath)) {
                Write-Host $Line
                Add-Content -LiteralPath $LogPath -Value $Line -Encoding UTF8
            }
            $Line
        })
        $NativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousPreference
    }
    return [pscustomobject]@{ ExitCode = $NativeExitCode; Text = ($Lines -join "`n") }
}

function Resolve-GraphFile {
    param([string]$GraphsRoot, [string]$RelativePath)
    if ([string]::IsNullOrWhiteSpace($RelativePath) -or
        [System.IO.Path]::IsPathRooted($RelativePath) -or $RelativePath.Contains('\')) {
        throw "Candidate manifest must use a relative slash-separated path: '$RelativePath'."
    }
    $Parts = @($RelativePath.Split('/'))
    if (@($Parts | Where-Object { $_ -in @('', '.', '..') }).Count -gt 0) {
        throw "Unsafe candidate manifest path: '$RelativePath'."
    }
    $Candidate = [System.IO.Path]::GetFullPath((Join-Path $GraphsRoot $RelativePath))
    $Prefix = $GraphsRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    if (-not $Candidate.StartsWith($Prefix, [System.StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
        throw "Candidate manifest file is missing or outside the graph directory: '$RelativePath'."
    }
    return $Candidate
}

function Invoke-CandidateCampaign {
    $RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
    $SourceRoot = Join-Path $RepoRoot 'src'
    $VendorScript = Join-Path $SourceRoot 'radiation_edge_ai\nasa_bps\kneron_candidate_screen.py'
    if ([string]::IsNullOrWhiteSpace($TrainingPython)) {
        $ScientificCandidates = @(
            (Join-Path $Root 'envs\nasa311\Scripts\python.exe'),
            (Join-Path $Root 'envs\nasa39\Scripts\python.exe')
        )
        $TrainingPython = $ScientificCandidates |
            Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
            Select-Object -First 1
        if ([string]::IsNullOrWhiteSpace($TrainingPython)) {
            throw "No NASA scientific environment was found under '$Root\envs'. Python 3.9 is supported in a separate nasa39 environment; follow docs/NASA_LOCAL_NPU_QUICKSTART.md or pass -TrainingPython '<existing scientific python.exe>'."
        }
    }
    if (-not (Test-Path -LiteralPath $TrainingPython -PathType Leaf)) {
        throw "Scientific Python not found at '$TrainingPython'. Set up the separate NASA environment in docs/NASA_LOCAL_NPU_QUICKSTART.md or pass -TrainingPython '<existing scientific python.exe>'."
    }
    if (@($Models).Count -eq 0 -or @($Models | Select-Object -Unique).Count -ne @($Models).Count) {
        throw 'Select at least one architecture, with no duplicates.'
    }
    if ([string]::IsNullOrWhiteSpace($OutputDir)) {
        $Stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
        $Suffix = [Guid]::NewGuid().ToString('N').Substring(0, 8)
        $OutputDir = Join-Path $Root ("scratch\nasa_candidate_screen_{0}_{1}" -f $Stamp, $Suffix)
    }
    $OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Output directory already exists; choose a fresh path: $OutputDir"
    }
    # Docker --mount uses comma-separated fields. Do not reinterpret a host path as mount options.
    if ($RepoRoot.Contains(',') -or $OutputDir.Contains(',')) {
        throw 'Repository and output paths must not contain commas for Docker bind mounts.'
    }
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
    $GraphsRoot = Join-Path $OutputDir 'graphs'
    $VendorRoot = Join-Path $OutputDir 'vendor'
    $ReceiptPath = Join-Path $OutputDir 'screen_receipt.json'
    $Receipt = [ordered]@{
        schema_version = 1
        purpose = 'architecture_feasibility_only'
        status = 'STARTED'
        created_utc = [DateTime]::UtcNow.ToString('o')
        completed_utc = $null
        output_directory = $OutputDir
        training_python = [System.IO.Path]::GetFullPath($TrainingPython)
        launcher_sha256 = Get-Sha256 $PSCommandPath
        vendor_script_sha256 = $null
        seed = $Seed
        requested_models = @($Models)
        export_only = [bool]$ExportOnly
        graph_manifest_sha256 = $null
        toolchain_image_requested = $ToolchainImage
        toolchain_image_id = $null
        toolchain_repo_digests = @()
        container_python = $ContainerPython
        selected_container_python = $null
        runtime_preflight_status = 'NOT_RUN'
        runtime_probe_exit_code = $null
        runtime_probe = $null
        runtime_probe_sha256 = $null
        runtime_log = $null
        runtime_log_sha256 = $null
        trained = $false
        nasa_images_read = $false
        physical_device_used = $false
        accuracy_evaluated = $false
        ptq_performed = $false
        nef_compiled = $false
        models = @()
        error = $null
    }
    Write-JsonFile $ReceiptPath $Receipt
    $PreviousPythonPath = $env:PYTHONPATH
    try {
        Write-Host 'NASA candidate architecture screen: full 256 x 256 graphs with scalar regression heads.'
        Write-Host 'Scientific Python:' $TrainingPython
        Write-Host 'Results:' $OutputDir
        $env:PYTHONPATH = $SourceRoot
        if (-not [string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
            $env:PYTHONPATH += [System.IO.Path]::PathSeparator + $PreviousPythonPath
        }
        $ExportArgs = @('-X', 'utf8', '-u', '-m', 'radiation_edge_ai.nasa_bps.candidate_graphs',
            '--out', $GraphsRoot, '--seed', [string]$Seed, '--models') + @($Models)
        $ExportResult = Invoke-NativeCaptured $TrainingPython $ExportArgs (Join-Path $OutputDir 'export.log')
        if ($ExportResult.ExitCode -ne 0) {
            throw "Candidate ONNX export exited with code $($ExportResult.ExitCode). Inspect export.log."
        }
        $GraphManifestPath = Join-Path $GraphsRoot 'candidate_graphs.json'
        $GraphManifest = Get-Content -LiteralPath $GraphManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($GraphManifest.status -ne 'COMPLETE') {
            throw 'Candidate graph manifest does not record a complete export.'
        }
        $Entries = @($GraphManifest.models)
        $ActualModels = @($Entries | ForEach-Object { $_.architecture })
        if ($Entries.Count -ne @($Models).Count -or
            @($ActualModels | Select-Object -Unique).Count -ne $Entries.Count -or
            @($ActualModels | Where-Object { $_ -notin $Models }).Count -gt 0) {
            throw 'Candidate graph manifest does not match the requested architecture set.'
        }
        $Receipt.graph_manifest_sha256 = Get-Sha256 $GraphManifestPath
        foreach ($Entry in $Entries) {
            $OnnxPath = Resolve-GraphFile $GraphsRoot $Entry.onnx_file
            $ManifestPath = Resolve-GraphFile $GraphsRoot $Entry.candidate_manifest
            if ((Get-Sha256 $OnnxPath) -ne $Entry.onnx_sha256 -or
                (Get-Sha256 $ManifestPath) -ne $Entry.candidate_manifest_sha256) {
                throw "Export hash mismatch for $($Entry.architecture)."
            }
            $Receipt.models += [ordered]@{
                architecture = $Entry.architecture
                onnx_file = $Entry.onnx_file
                onnx_sha256 = $Entry.onnx_sha256
                candidate_manifest = $Entry.candidate_manifest
                candidate_manifest_sha256 = $Entry.candidate_manifest_sha256
                support_status = 'unknown'
                vendor_invoked = $false
                docker_exit_code = $null
                screen_report = $null
                screen_report_sha256 = $null
                reasons = @('Vendor evaluation has not run.')
            }
        }
        Write-JsonFile $ReceiptPath $Receipt
        if ($ExportOnly) {
            foreach ($Result in $Receipt.models) {
                $Result.reasons = @('ExportOnly requested; Kneron compatibility has not been evaluated.')
            }
            $Receipt.status = 'EXPORTED_ONLY'
            Write-Host 'Full candidate ONNX export and synthetic parity complete. Vendor compatibility: UNKNOWN.'
            return 0
        }

        # Application lookup can return multiple PATH matches. Invoke only the
        # first resolved executable, exactly as ordinary command lookup would.
        $Docker = Get-Command docker -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        $DockerProblem = $null
        if ($null -eq $Docker) {
            $DockerProblem = 'Docker CLI was not found. Start the existing Docker Desktop installation and rerun with a fresh output directory.'
        }
        else {
            $Info = Invoke-NativeCaptured $Docker.Source @('info', '--format', '{{.OSType}}')
            if ($Info.ExitCode -ne 0 -or $Info.Text.Trim() -ne 'linux') {
                $DockerProblem = 'A Linux Docker engine is unavailable. Start Docker Desktop in Linux-container mode and rerun with a fresh output directory. ' + $Info.Text
            }
            else {
                $Inspection = Invoke-NativeCaptured $Docker.Source @('image', 'inspect', $ToolchainImage)
                if ($Inspection.ExitCode -ne 0) {
                    $DockerProblem = "The requested Kneron image is not available locally: $ToolchainImage. Pass -ToolchainImage for an existing image; this launcher does not pull images. " + $Inspection.Text
                }
                else {
                    try {
                        $Image = @($Inspection.Text | ConvertFrom-Json)[0]
                        if ($Image.Id -notmatch '^sha256:[0-9a-f]{64}$' -or $Image.Os -ne 'linux') {
                            throw 'Image inspection did not return one immutable Linux image identity.'
                        }
                        $Receipt.toolchain_image_id = $Image.Id
                        $Receipt.toolchain_repo_digests = @($Image.RepoDigests)
                        Write-JsonFile (Join-Path $OutputDir 'toolchain_image.json') $Image
                    }
                    catch {
                        $DockerProblem = 'Could not verify the local toolchain image: ' + $_.Exception.Message
                    }
                }
            }
        }
        if ($null -ne $DockerProblem) {
            foreach ($Result in $Receipt.models) { $Result.reasons = @($DockerProblem) }
            $Receipt.status = 'COMPLETE_WITH_UNKNOWN'
            Write-Host $DockerProblem
            return 3
        }
        if (-not (Test-Path -LiteralPath $VendorScript -PathType Leaf)) {
            throw "Vendor screening adapter is missing: $VendorScript"
        }
        $Receipt.vendor_script_sha256 = Get-Sha256 $VendorScript

        # The pinned image's validated SDK lives in its Miniconda environment;
        # an interactive-shell alias or system Python is not a runtime contract.
        # Probe the exact image and interpreter once before any model attempt.
        $RuntimeRoot = Join-Path $OutputDir 'runtime'
        New-Item -ItemType Directory -Path $RuntimeRoot | Out-Null
        $RuntimeProbePath = Join-Path $RuntimeRoot 'runtime_probe.json'
        $RuntimeLogPath = Join-Path $OutputDir 'runtime.log'
        New-Item -ItemType File -Path $RuntimeLogPath | Out-Null
        $Receipt.runtime_log = 'runtime.log'
        $RuntimeArgs = @('run', '--rm', '--pull', 'never', '--network', 'none',
            '--mount', "type=bind,source=$RepoRoot,target=/repo,readonly",
            '--mount', "type=bind,source=$RuntimeRoot,target=/runtime",
            '--workdir', '/runtime', '--entrypoint', $ContainerPython,
            $Receipt.toolchain_image_id, '-B',
            '/repo/src/radiation_edge_ai/nasa_bps/kneron_candidate_screen.py',
            '--probe-runtime', '--out', '/runtime/runtime_probe.json',
            '--toolchain-image-id', $Receipt.toolchain_image_id)
        Write-Host '[Kneron runtime preflight]' $ContainerPython
        $RuntimeResult = Invoke-NativeCaptured $Docker.Source $RuntimeArgs $RuntimeLogPath
        $Receipt.runtime_probe_exit_code = $RuntimeResult.ExitCode
        if (Test-Path -LiteralPath $RuntimeLogPath -PathType Leaf) {
            $Receipt.runtime_log_sha256 = Get-Sha256 $RuntimeLogPath
        }
        try {
            if (-not (Test-Path -LiteralPath $RuntimeProbePath -PathType Leaf)) {
                throw "Interpreter produced no runtime receipt (exit $($RuntimeResult.ExitCode))."
            }
            $Receipt.runtime_probe = 'runtime/runtime_probe.json'
            $Receipt.runtime_probe_sha256 = Get-Sha256 $RuntimeProbePath
            $RuntimeProbe = Get-Content -LiteralPath $RuntimeProbePath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($RuntimeProbe.schema_version -ne 1 -or
                $RuntimeProbe.purpose -ne 'toolchain_runtime_preflight' -or
                $RuntimeProbe.toolchain_image_id -ne $Receipt.toolchain_image_id) {
                throw 'Runtime receipt identity does not match this toolchain preflight.'
            }
            if ($RuntimeProbe.status -ne 'PASS' -or $RuntimeResult.ExitCode -ne 0 -or
                $RuntimeProbe.exit_code -ne 0) {
                throw "SDK import/API check failed (exit $($RuntimeResult.ExitCode)): $($RuntimeProbe.error)"
            }
            $ResolvedContainerPython = [string]$RuntimeProbe.python_executable
            if (-not $ResolvedContainerPython.StartsWith('/') -or
                $ResolvedContainerPython -match '[\r\n\x00]' -or
                ($ContainerPython.StartsWith('/') -and $ResolvedContainerPython -cne $ContainerPython)) {
                throw 'Runtime receipt does not identify the requested container interpreter.'
            }
            if ([version]$RuntimeProbe.python_version -lt [version]'3.7' -or
                $RuntimeProbe.optimizer_backend -notin @('kneronnxopt.optimize', 'ktc.onnx_optimizer.onnx2onnx_flow')) {
                throw 'Runtime receipt does not identify a compatible Python and optimizer API.'
            }
            $Receipt.selected_container_python = $ResolvedContainerPython
            $Receipt.runtime_preflight_status = 'PASS'
            Write-Host '[Kneron runtime preflight] PASS:' $ResolvedContainerPython
        }
        catch {
            $Receipt.runtime_preflight_status = 'FAILED'
            $RuntimeProblem = "Kneron runtime preflight failed: $($_.Exception.Message) See '$RuntimeLogPath'. Model evaluation did not start."
            foreach ($Result in $Receipt.models) { $Result.reasons = @($RuntimeProblem) }
            $Receipt.status = 'COMPLETE_WITH_UNKNOWN'
            Write-Host $RuntimeProblem
            return 3
        }
        Write-JsonFile $ReceiptPath $Receipt
        New-Item -ItemType Directory -Path $VendorRoot | Out-Null
        Write-Host 'Existing immutable toolchain image:' $Receipt.toolchain_image_id
        foreach ($Result in $Receipt.models) {
            $Architecture = $Result.architecture
            $ContainerOutput = '/results/' + $Architecture
            $DockerArgs = @('run', '--rm', '--pull', 'never', '--network', 'none',
                '--mount', "type=bind,source=$RepoRoot,target=/repo,readonly",
                '--mount', "type=bind,source=$GraphsRoot,target=/graphs,readonly",
                '--mount', "type=bind,source=$VendorRoot,target=/results",
                '--workdir', '/results', '--entrypoint', $ResolvedContainerPython,
                $Receipt.toolchain_image_id, '-B',
                '/repo/src/radiation_edge_ai/nasa_bps/kneron_candidate_screen.py',
                '--input', ('/graphs/' + $Result.onnx_file),
                '--candidate-manifest', ('/graphs/' + $Result.candidate_manifest),
                '--out', $ContainerOutput, '--platform', '720',
                '--model-id', '32770', '--model-version', '0001',
                '--toolchain-image-id', $Receipt.toolchain_image_id)
            Write-Host ("[Kneron architecture screen] {0}" -f $Architecture)
            $Result.vendor_invoked = $true
            $Worker = Invoke-NativeCaptured $Docker.Source $DockerArgs (Join-Path $OutputDir ("{0}_docker.log" -f $Architecture))
            $Result.docker_exit_code = $Worker.ExitCode
            $ReportPath = Join-Path (Join-Path $VendorRoot $Architecture) 'screen.json'
            try {
                if (-not (Test-Path -LiteralPath $ReportPath -PathType Leaf)) {
                    throw "Container produced no screen.json (exit $($Worker.ExitCode)). Check ${Architecture}_docker.log and the image's Python/dependencies; use -ContainerPython if needed."
                }
                $Result.screen_report = "vendor/$Architecture/screen.json"
                $Result.screen_report_sha256 = Get-Sha256 $ReportPath
                $Screen = Get-Content -LiteralPath $ReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
                $ExpectedExit = @{ supported = 0; unsupported = 2; unknown = 3 }
                if ($Screen.support_status -notin @('supported', 'unsupported', 'unknown')) {
                    throw 'Vendor screen returned an unrecognized support state.'
                }
                if ($Screen.architecture -ne $Architecture -or
                    $Screen.source_onnx_sha256 -ne $Result.onnx_sha256 -or
                    $Screen.candidate_manifest_sha256 -ne $Result.candidate_manifest_sha256 -or
                    $Screen.toolchain_image_id -ne $Receipt.toolchain_image_id -or
                    $Screen.python_executable -cne $ResolvedContainerPython) {
                    throw 'Vendor screen identity does not match this graph, manifest, architecture, toolchain and interpreter.'
                }
                if ($Worker.ExitCode -ne $ExpectedExit[$Screen.support_status] -or
                    $Screen.exit_code -ne $ExpectedExit[$Screen.support_status]) {
                    throw 'Vendor screen status and process exit code disagree.'
                }
                $Result.support_status = $Screen.support_status
                $Result.reasons = @($Screen.reasons)
            }
            catch {
                $Result.support_status = 'unknown'
                $Result.reasons = @($_.Exception.Message)
            }
            Write-Host ("{0}: {1}" -f $Architecture, $Result.support_status.ToUpperInvariant())
            Write-JsonFile $ReceiptPath $Receipt
        }
        # A completed receipt also binds the immutable input artifacts after all workers exit.
        if ((Get-Sha256 $RuntimeProbePath) -ne $Receipt.runtime_probe_sha256 -or
            ($null -ne $Receipt.runtime_log_sha256 -and
                (Get-Sha256 $RuntimeLogPath) -ne $Receipt.runtime_log_sha256)) {
            throw 'Toolchain runtime evidence changed during vendor screening.'
        }
        if ((Get-Sha256 $GraphManifestPath) -ne $Receipt.graph_manifest_sha256) {
            throw 'Candidate graph manifest changed during vendor screening.'
        }
        foreach ($Result in $Receipt.models) {
            if ((Get-Sha256 (Resolve-GraphFile $GraphsRoot $Result.onnx_file)) -ne $Result.onnx_sha256 -or
                (Get-Sha256 (Resolve-GraphFile $GraphsRoot $Result.candidate_manifest)) -ne $Result.candidate_manifest_sha256) {
                throw "Candidate input changed during vendor screening: $($Result.architecture)."
            }
            if ($null -ne $Result.screen_report -and
                (Get-Sha256 (Join-Path $OutputDir $Result.screen_report)) -ne $Result.screen_report_sha256) {
                throw "Vendor report changed after it was read: $($Result.architecture)."
            }
        }
        if (@($Receipt.models | Where-Object { $_.support_status -eq 'unknown' }).Count -gt 0) {
            $Receipt.status = 'COMPLETE_WITH_UNKNOWN'
            return 3
        }
        if (@($Receipt.models | Where-Object { $_.support_status -eq 'unsupported' }).Count -gt 0) {
            $Receipt.status = 'COMPLETE_WITH_UNSUPPORTED'
            return 2
        }
        $Receipt.status = 'COMPLETE_SUPPORTED'
        return 0
    }
    catch {
        $Receipt.status = 'FAILED'
        $Receipt.error = $_.Exception.Message
        Write-Host ('NASA candidate screen failed: ' + $Receipt.error)
        return 1
    }
    finally {
        $env:PYTHONPATH = $PreviousPythonPath
        $Receipt.completed_utc = [DateTime]::UtcNow.ToString('o')
        Write-JsonFile $ReceiptPath $Receipt
        Write-Host ''
        Write-Host ('Candidate screen status: ' + $Receipt.status)
        foreach ($Result in $Receipt.models) {
            Write-Host ("  {0}: {1}" -f $Result.architecture, $Result.support_status)
        }
        Write-Host 'This is architecture feasibility only; NASA accuracy, PTQ/NEF and physical execution remain unevaluated.'
        Write-Host 'Receipt:' $ReceiptPath
    }
}

exit (Invoke-CandidateCampaign)
