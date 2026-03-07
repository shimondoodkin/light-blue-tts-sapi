$dest = "C:\Program Files\LightBlue TTS"
$src = "C:\Users\user\Documents\projects\light-blue-tts-windows\lightblue-sapi\target\release"

regsvr32 /u /s "$dest\lightblue_sapi.dll"
Copy-Item "$src\lightblue_sapi.dll" "$dest\lightblue_sapi.dll" -Force
Copy-Item "$src\onnxruntime.dll" "$dest\onnxruntime.dll" -Force
Copy-Item "$src\lightblue-tts.exe" "$dest\lightblue-tts.exe" -Force
regsvr32 /s "$dest\lightblue_sapi.dll"

Write-Host "Deployed and registered."
reg query "HKLM\SOFTWARE\Microsoft\Speech\Voices\Tokens" 2>&1 | Select-String "LightBlue"
reg query "HKLM\SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens" 2>&1 | Select-String "LightBlue"
