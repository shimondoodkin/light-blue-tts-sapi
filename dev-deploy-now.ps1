$ErrorActionPreference = "Stop"
$src = "C:\Users\user\Documents\projects\light-blue-tts-windows\lightblue-sapi\target\debug"
$dst = "C:\Program Files\LightBlue TTS SAPI"
foreach ($f in @("lightblue_sapi.dll", "onnxruntime.dll", "onnxruntime_providers_shared.dll", "DirectML.dll")) {
    $s = Join-Path $src $f
    $d = Join-Path $dst $f
    if (Test-Path $s) {
        Copy-Item $s $d -Force
        Write-Host "Copied $f"
    }
}
