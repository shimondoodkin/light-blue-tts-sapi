#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Deploys models from the repo's models/ folder to the installed app directory.
.DESCRIPTION
    Copies all model files (ONNX, voices, tokenizer, etc.) to the installation folder.
    Useful during development to update models without re-running the full installer.
#>

param(
    [string]$InstallDir = "$env:ProgramFiles\LightBlue TTS SAPI"
)

$ErrorActionPreference = "Stop"

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = Split-Path -Parent $ScriptDir
$ModelsSource = Join-Path $RepoRoot "models"

if (-not (Test-Path $ModelsSource)) {
    Write-Error "Models directory not found at $ModelsSource. Download models first with: python scripts/download_models.py"
    exit 1
}

if (-not (Test-Path $InstallDir)) {
    Write-Error "Install directory not found at $InstallDir. Install the app first."
    exit 1
}

Write-Host "Deploying models to: $InstallDir\models" -ForegroundColor Cyan

$ModelsDest = Join-Path $InstallDir "models"
if (Test-Path $ModelsDest) {
    Remove-Item -Path $ModelsDest -Recurse -Force
}
Copy-Item -Path $ModelsSource -Destination $ModelsDest -Recurse -Force

Write-Host ""
Write-Host "Models deployed successfully!" -ForegroundColor Green
