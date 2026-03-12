; LightBlue Hebrew TTS SAPI - NSIS Installer (App Only)
; No models included — install models separately with LightBlue-TTS-SAPI-Models.exe

!include "MUI2.nsh"

; --- General ---
Name "LightBlue Hebrew TTS SAPI"
OutFile "..\target\installer\LightBlue-TTS-SAPI-Setup.exe"
InstallDir "$PROGRAMFILES64\LightBlue TTS SAPI"
RequestExecutionLevel admin
Unicode true

; --- Icon ---
!define MUI_ICON "icon.ico"
!define MUI_UNICON "icon.ico"

; --- UI ---
!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

; --- Install ---
Section "Install"
    SetOutPath "$INSTDIR"

    ; Icon
    File "icon.ico"

    ; Main DLL, CLI tools
    File "..\target\release\lightblue_sapi.dll"
    File "..\target\release\lightblue-tts.exe"
    ; ONNX Runtime DirectML — skip if already present (don't overwrite a GPU/OpenVINO version)
    SetOverwrite off
    File "..\target\release\onnxruntime.dll"
    File "..\target\release\onnxruntime_providers_shared.dll"
    File "..\target\release\DirectML.dll"
    SetOverwrite on

    ; Create directory structure for models
    CreateDirectory "$INSTDIR\models"
    CreateDirectory "$INSTDIR\models\voices"
    CreateDirectory "$INSTDIR\dictionaries"

    ; Register COM DLL
    ExecWait 'regsvr32 /s "$INSTDIR\lightblue_sapi.dll"'

    ; Create uninstaller
    WriteUninstaller "$INSTDIR\Uninstall.exe"

    ; Add/Remove Programs entry
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LightBlueTTSSAPI" \
        "DisplayName" "LightBlue Hebrew TTS SAPI"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LightBlueTTSSAPI" \
        "UninstallString" '"$INSTDIR\Uninstall.exe"'
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LightBlueTTSSAPI" \
        "Publisher" "LightBlue"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LightBlueTTSSAPI" \
        "DisplayIcon" "$INSTDIR\icon.ico"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LightBlueTTSSAPI" \
        "DisplayVersion" "1.0.0"
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LightBlueTTSSAPI" \
        "NoModify" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LightBlueTTSSAPI" \
        "NoRepair" 1
SectionEnd

; --- Uninstall ---
Section "Uninstall"
    ; Unregister COM DLL
    ExecWait 'regsvr32 /u /s "$INSTDIR\lightblue_sapi.dll"'

    ; Remove files
    RMDir /r "$INSTDIR\models"
    RMDir /r "$INSTDIR\dictionaries"
    Delete "$INSTDIR\icon.ico"
    Delete "$INSTDIR\lightblue_sapi.dll"
    Delete "$INSTDIR\lightblue-tts.exe"
    ; ORT core
    Delete "$INSTDIR\onnxruntime.dll"
    Delete "$INSTDIR\onnxruntime_providers_shared.dll"
    ; DirectML
    Delete "$INSTDIR\DirectML.dll"
    ; CUDA/TensorRT
    Delete "$INSTDIR\onnxruntime_providers_cuda.dll"
    Delete "$INSTDIR\onnxruntime_providers_tensorrt.dll"
    ; OpenVINO
    Delete "$INSTDIR\onnxruntime_providers_openvino.dll"
    Delete "$INSTDIR\openvino.dll"
    Delete "$INSTDIR\openvino_c.dll"
    Delete "$INSTDIR\openvino_intel_cpu_plugin.dll"
    Delete "$INSTDIR\openvino_intel_gpu_plugin.dll"
    Delete "$INSTDIR\openvino_intel_npu_plugin.dll"
    Delete "$INSTDIR\openvino_onnx_frontend.dll"
    Delete "$INSTDIR\openvino_ir_frontend.dll"
    Delete "$INSTDIR\openvino_auto_plugin.dll"
    Delete "$INSTDIR\tbb12.dll"
    ; Uninstaller
    Delete "$INSTDIR\Uninstall.exe"
    RMDir "$INSTDIR"

    ; Remove Add/Remove Programs entry
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LightBlueTTSSAPI"
SectionEnd
