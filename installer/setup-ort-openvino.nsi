; LightBlue Hebrew TTS SAPI - ONNX Runtime OpenVINO Installer
; Extracts Intel OpenVINO-enabled ORT DLLs to the app folder
; (overwrites the CPU/DirectML onnxruntime.dll from the app installer)

!include "MUI2.nsh"

; --- General ---
Name "LightBlue Hebrew TTS SAPI - ONNX Runtime OpenVINO"
OutFile "..\target\installer\LightBlue-TTS-SAPI-ORT-OpenVINO.exe"
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

    ; ONNX Runtime OpenVINO DLLs (overwrites CPU version)
    File "onnxruntime.dll"
    File "onnxruntime_providers_openvino.dll"
    File "onnxruntime_providers_shared.dll"

    ; OpenVINO runtime
    File "openvino.dll"
    File "openvino_c.dll"
    File "openvino_intel_cpu_plugin.dll"
    File "openvino_intel_gpu_plugin.dll"
    File "openvino_intel_npu_plugin.dll"
    File "openvino_onnx_frontend.dll"
    File "openvino_ir_frontend.dll"
    File "openvino_auto_plugin.dll"

    ; Threading (TBB)
    File "tbb12.dll"

    ; Overwrite uninstaller so it covers all installed components
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
SectionEnd

; --- Uninstall ---
Section "Uninstall"
    ; Unregister COM DLL
    ExecWait 'regsvr32 /u /s "$INSTDIR\lightblue_sapi.dll"'

    ; Remove known files
    Delete "$INSTDIR\icon.ico"
    Delete "$INSTDIR\lightblue_sapi.dll"
    Delete "$INSTDIR\lightblue-tts.exe"
    Delete "$INSTDIR\log.txt"
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
    ; Uninstaller itself
    Delete "$INSTDIR\Uninstall.exe"

    ; Remove entire install directory (models, dictionaries, any leftover files)
    RMDir /r "$INSTDIR"

    ; Remove Add/Remove Programs entry
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LightBlueTTSSAPI"
SectionEnd
