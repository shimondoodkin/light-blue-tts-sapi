#Requires -Version 5.1
<#
.SYNOPSIS
    Downloads or updates LightBlue TTS models from HuggingFace.
.DESCRIPTION
    Fetches ONNX models, phonikud model, tokenizer, and style configs.
    Can be re-run to update models to the latest version.
.PARAMETER InstallDir
    Installation directory (default: Program Files\LightBlue TTS)
.PARAMETER Force
    Re-download all files even if they already exist
#>

param(
    [string]$InstallDir = "$env:ProgramFiles\LightBlue TTS",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$ModelsDir = Join-Path $InstallDir "models"
$OnnxDir   = Join-Path $ModelsDir "onnx"
$DictDir   = Join-Path $InstallDir "dictionaries"

# HuggingFace base URLs
$HF_TTS      = "https://huggingface.co/notmax123/LightBlue/resolve/main"
$HF_PHONIKUD = "https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main"
$ORT_VERSION  = "1.22.0"
$ORT_URL      = "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/onnxruntime-win-x64-$ORT_VERSION.zip"

# Files to download: (url, local_path)
$Files = @(
    # TTS ONNX models
    @{ Url = "$HF_TTS/onnx/backbone_keys.onnx";     Path = "$OnnxDir\backbone_keys.onnx" }
    @{ Url = "$HF_TTS/onnx/text_encoder.onnx";       Path = "$OnnxDir\text_encoder.onnx" }
    @{ Url = "$HF_TTS/onnx/reference_encoder.onnx";   Path = "$OnnxDir\reference_encoder.onnx" }
    @{ Url = "$HF_TTS/onnx/vocoder.onnx";             Path = "$OnnxDir\vocoder.onnx" }
    @{ Url = "$HF_TTS/onnx/length_pred.onnx";         Path = "$OnnxDir\length_pred.onnx" }
    @{ Url = "$HF_TTS/onnx/length_pred_style.onnx";   Path = "$OnnxDir\length_pred_style.onnx" }
    @{ Url = "$HF_TTS/onnx/stats.npz";                Path = "$OnnxDir\stats.npz" }
    @{ Url = "$HF_TTS/onnx/uncond.npz";               Path = "$OnnxDir\uncond.npz" }
    # Voice styles
    @{ Url = "$HF_TTS/style.json";                    Path = "$ModelsDir\style.json" }
    @{ Url = "$HF_TTS/style_female.json";             Path = "$ModelsDir\style_female.json" }
    # Phonikud
    @{ Url = "$HF_PHONIKUD/phonikud.onnx";            Path = "$ModelsDir\phonikud.onnx" }
    @{ Url = "$HF_PHONIKUD/tokenizer.json";            Path = "$ModelsDir\tokenizer.json" }
)

Write-Host "=== LightBlue TTS Model Downloader ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Install directory: $InstallDir"
Write-Host ""

# Create directories
foreach ($dir in @($ModelsDir, $OnnxDir, $DictDir)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Download ONNX Runtime if missing
$OrtDll = Join-Path $InstallDir "onnxruntime.dll"
if ($Force -or -not (Test-Path $OrtDll)) {
    Write-Host "Downloading ONNX Runtime v$ORT_VERSION..." -ForegroundColor Yellow
    $zipPath = Join-Path $env:TEMP "onnxruntime.zip"
    $extractDir = Join-Path $env:TEMP "onnxruntime"
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $wc = New-Object System.Net.WebClient
        $wc.DownloadFile($ORT_URL, $zipPath)
        if (Test-Path $extractDir) { Remove-Item $extractDir -Recurse -Force }
        Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force
        $dllSource = Get-ChildItem -Path $extractDir -Recurse -Filter "onnxruntime.dll" | Select-Object -First 1
        Copy-Item $dllSource.FullName $OrtDll -Force
        Write-Host "  onnxruntime.dll OK" -ForegroundColor Green
    } finally {
        Remove-Item $zipPath -ErrorAction SilentlyContinue
        Remove-Item $extractDir -Recurse -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "onnxruntime.dll already exists, skipping (use -Force to re-download)"
}

# Download model files
$total = $Files.Count
$current = 0
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

foreach ($file in $Files) {
    $current++
    $name = Split-Path $file.Path -Leaf

    if (-not $Force -and (Test-Path $file.Path)) {
        Write-Host "  [$current/$total] $name - exists, skipping"
        continue
    }

    Write-Host "  [$current/$total] Downloading $name..." -NoNewline
    try {
        $wc = New-Object System.Net.WebClient
        $wc.DownloadFile($file.Url, $file.Path)
        $size = [math]::Round((Get-Item $file.Path).Length / 1MB, 1)
        Write-Host " ${size}MB OK" -ForegroundColor Green
    } catch {
        Write-Host " FAILED: $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Model download complete!" -ForegroundColor Green
Write-Host ""

# Verify all files exist
$missing = @()
foreach ($file in $Files) {
    if (-not (Test-Path $file.Path)) {
        $missing += (Split-Path $file.Path -Leaf)
    }
}
if (-not (Test-Path $OrtDll)) { $missing += "onnxruntime.dll" }

if ($missing.Count -gt 0) {
    Write-Host "WARNING: Missing files:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
} else {
    Write-Host "All files verified." -ForegroundColor Green
}
