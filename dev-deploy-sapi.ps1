#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Installs LightBlue TTS SAPI as a SAPI 5 voice on Windows.
.DESCRIPTION
    Copies the built DLL, ONNX models, ONNX Runtime (DirectML/CUDA/OpenVINO),
    and Phonikud model to a permanent location and registers the COM server.

    Requires admin privileges. Run from an elevated terminal, or use:

    powershell -ExecutionPolicy Bypass -Command "Start-Process powershell -Verb RunAs -Wait -ArgumentList '-ExecutionPolicy Bypass -File ""dev-deploy-sapi.ps1""'"

    To verify deployment:
    ls "$env:ProgramFiles\LightBlue TTS SAPI\*.dll"
#>

param(
    [string]$BuildProfile = "release",
    [string]$InstallDir   = "$env:ProgramFiles\LightBlue TTS SAPI",
    [switch]$PauseAtEnd
)

$ErrorActionPreference = "Stop"

# Resolve paths relative to this script's location
$RepoRoot   = Split-Path -Parent $MyInvocation.MyCommand.Path
$TargetDir  = Join-Path $RepoRoot "target\$BuildProfile"

$DllName    = "lightblue_sapi.dll"
$DllSource  = Join-Path $TargetDir $DllName

# Validate that the DLL exists
if (-not (Test-Path $DllSource)) {
    Write-Error "DLL not found at $DllSource. Did you build in $BuildProfile mode? Run: cargo build --release"
    exit 1
}

Write-Host "Installing LightBlue TTS SAPI to: $InstallDir" -ForegroundColor Cyan

# Create install directory
if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

# 1. Copy the main DLL
Write-Host "  Copying $DllName..."
Copy-Item -Path $DllSource -Destination (Join-Path $InstallDir $DllName) -Force

# 2. Copy onnxruntime.dll and provider/runtime DLLs (CUDA or OpenVINO if present)
$OrtDlls = @(
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
    # DirectML
    "DirectML.dll",
    # CUDA
    "onnxruntime_providers_cuda.dll",
    "onnxruntime_providers_tensorrt.dll",
    # OpenVINO
    "onnxruntime_providers_openvino.dll",
    "openvino.dll",
    "openvino_c.dll",
    "openvino_intel_cpu_plugin.dll",
    "openvino_intel_gpu_plugin.dll",
    "openvino_intel_npu_plugin.dll",
    "openvino_onnx_frontend.dll",
    "openvino_ir_frontend.dll",
    "openvino_auto_plugin.dll",
    "tbb12.dll"
)
foreach ($dll in $OrtDlls) {
    $src = Join-Path $TargetDir $dll
    if (Test-Path $src) {
        Write-Host "  Copying $dll..."
        Copy-Item -Path $src -Destination (Join-Path $InstallDir $dll) -Force
    }
}

# 3. Copy entire models directory (ONNX models, phonikud, tokenizer, style, tts.json)
$ModelsSource = Join-Path $RepoRoot "models"
if (Test-Path $ModelsSource) {
    Write-Host "  Copying models (ONNX pipeline + phonikud + tokenizer + style)..."
    $ModelsDest = Join-Path $InstallDir "models"
    if (Test-Path $ModelsDest) {
        Remove-Item -Path $ModelsDest -Recurse -Force
    }
    Copy-Item -Path $ModelsSource -Destination $ModelsDest -Recurse -Force
} else {
    Write-Error "Models directory not found at $ModelsSource. Download models first."
    exit 1
}

# 5. Register the COM server
Write-Host "  Registering COM server..."
$DllPath = Join-Path $InstallDir $DllName
$regResult = Start-Process -FilePath "regsvr32" -ArgumentList "/s `"$DllPath`"" -Wait -PassThru
if ($regResult.ExitCode -ne 0) {
    Write-Error "regsvr32 failed with exit code $($regResult.ExitCode)"
    exit 1
}

Write-Host ""
Write-Host "LightBlue TTS SAPI installed successfully!" -ForegroundColor Green
Write-Host "The voice should now appear in Windows Speech settings and any SAPI 5 application."

if ($PauseAtEnd) {
    Write-Host ""
    Read-Host "Deployment finished. Press Enter to close"
}
