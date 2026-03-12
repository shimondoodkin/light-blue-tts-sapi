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
SectionEnd
