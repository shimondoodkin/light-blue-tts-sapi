; LightBlue Hebrew TTS - ONNX Runtime GPU Installer
; Extracts CUDA/TensorRT-enabled ORT DLLs to the app folder
; (overwrites the CPU/DirectML onnxruntime.dll from the app installer)

!include "MUI2.nsh"

; --- General ---
Name "LightBlue Hebrew TTS - ONNX Runtime GPU"
OutFile "..\target\installer\LightBlue-TTS-ORT-GPU.exe"
InstallDir "$PROGRAMFILES64\LightBlue TTS"
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

    ; ONNX Runtime GPU DLLs (overwrites CPU version)
    File "onnxruntime.dll"
    File "onnxruntime_providers_shared.dll"
    File "onnxruntime_providers_cuda.dll"
    File "onnxruntime_providers_tensorrt.dll"
SectionEnd
