$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceDir = Join-Path $repoRoot "target\release"
$installDir = "C:\Program Files\LightBlue TTS SAPI"
$reportPath = Join-Path $repoRoot "deploy-current-release-report.txt"

Copy-Item (Join-Path $sourceDir "lightblue_sapi.dll") (Join-Path $installDir "lightblue_sapi.dll") -Force
Copy-Item (Join-Path $sourceDir "onnxruntime.dll") (Join-Path $installDir "onnxruntime.dll") -Force

$optionalDlls = @(
    "onnxruntime_providers_shared.dll",
    "DirectML.dll",
    "onnxruntime_providers_cuda.dll",
    "onnxruntime_providers_tensorrt.dll",
    "onnxruntime_providers_openvino.dll"
)

foreach ($dll in $optionalDlls) {
    $src = Join-Path $sourceDir $dll
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $installDir $dll) -Force
    }
}

regsvr32 /s (Join-Path $installDir "lightblue_sapi.dll")

$sourceHash = (Get-FileHash (Join-Path $sourceDir "lightblue_sapi.dll") -Algorithm SHA256).Hash
$installedHash = (Get-FileHash (Join-Path $installDir "lightblue_sapi.dll") -Algorithm SHA256).Hash
$sourceItem = Get-Item (Join-Path $sourceDir "lightblue_sapi.dll")
$installedItem = Get-Item (Join-Path $installDir "lightblue_sapi.dll")

@(
    "source=$($sourceItem.FullName)"
    "source_len=$($sourceItem.Length)"
    "source_hash=$sourceHash"
    "installed=$($installedItem.FullName)"
    "installed_len=$($installedItem.Length)"
    "installed_hash=$installedHash"
) | Set-Content $reportPath

Write-Host ""
Read-Host "Deployment finished. Press Enter to close"
