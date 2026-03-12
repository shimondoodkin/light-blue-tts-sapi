$ErrorActionPreference = "Stop"

param(
    [string]$BuildProfile = "release",
    [string]$InstallDir = "$env:ProgramFiles\LightBlue TTS SAPI"
)

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = Join-Path $repo "target\$BuildProfile"

$dllName = "lightblue_sapi.dll"
$dllSource = Join-Path $target $dllName
if (-not (Test-Path $dllSource)) {
    throw "DLL not found at $dllSource. Build first."
}

New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null

Copy-Item $dllSource (Join-Path $InstallDir $dllName) -Force

$ortDlls = @(
    "onnxruntime.dll", "onnxruntime_providers_shared.dll",
    "DirectML.dll",
    "onnxruntime_providers_cuda.dll", "onnxruntime_providers_tensorrt.dll",
    "onnxruntime_providers_openvino.dll",
    "openvino.dll", "openvino_c.dll",
    "openvino_intel_cpu_plugin.dll", "openvino_intel_gpu_plugin.dll", "openvino_intel_npu_plugin.dll",
    "openvino_onnx_frontend.dll", "openvino_ir_frontend.dll", "openvino_auto_plugin.dll",
    "tbb12.dll"
)
foreach ($dll in $ortDlls) {
    $src = Join-Path $target $dll
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $InstallDir $dll) -Force
    }
}

$modelsSrc = Join-Path $repo "models"
if (-not (Test-Path $modelsSrc)) {
    throw "Models directory not found at $modelsSrc"
}

$modelsDst = Join-Path $InstallDir "models"
if (Test-Path $modelsDst) {
    Remove-Item $modelsDst -Recurse -Force
}
Copy-Item $modelsSrc $modelsDst -Recurse -Force

Write-Host "Copied build output to $InstallDir"
Write-Host "COM registration was not run."
