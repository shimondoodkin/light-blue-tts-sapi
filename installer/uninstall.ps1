#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Uninstalls LightBlue TTS SAPI 5 voice from Windows.
.DESCRIPTION
    Unregisters the COM server and removes the installation directory.
#>

param(
    [string]$InstallDir = "$env:ProgramFiles\LightBlue TTS"
)

$ErrorActionPreference = "Stop"

$DllName = "lightblue_sapi.dll"
$DllPath = Join-Path $InstallDir $DllName

Write-Host "Uninstalling LightBlue TTS..." -ForegroundColor Cyan

# 1. Unregister the COM server
if (Test-Path $DllPath) {
    Write-Host "  Unregistering COM server..."
    $regResult = Start-Process -FilePath "regsvr32" -ArgumentList "/u /s `"$DllPath`"" -Wait -PassThru
    if ($regResult.ExitCode -ne 0) {
        Write-Warning "regsvr32 /u returned exit code $($regResult.ExitCode) -- continuing anyway."
    }
} else {
    Write-Warning "DLL not found at $DllPath -- skipping unregistration."
}

# 2. Remove the installation directory
if (Test-Path $InstallDir) {
    Write-Host "  Removing installation directory: $InstallDir"
    Remove-Item -Path $InstallDir -Recurse -Force
} else {
    Write-Host "  Installation directory not found -- nothing to remove."
}

Write-Host ""
Write-Host "LightBlue TTS uninstalled successfully." -ForegroundColor Green
