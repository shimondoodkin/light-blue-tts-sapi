; LightBlue Hebrew TTS SAPI - Models Installer
; Extracts pre-optimized ONNX models to the app folder

!include "MUI2.nsh"

; --- General ---
Name "LightBlue Hebrew TTS SAPI - Models"
OutFile "..\target\installer\LightBlue-TTS-SAPI-Models.exe"
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
    SetOutPath "$INSTDIR\models"

    ; ONNX models
    File "..\models\backbone_keys.onnx"
    File "..\models\text_encoder.onnx"
    File "..\models\reference_encoder.onnx"
    File "..\models\vocoder.onnx"
    File "..\models\length_pred.onnx"
    File "..\models\length_pred_style.onnx"
    File "..\models\stats.npz"
    File "..\models\uncond.npz"

    ; Phonikud
    File "..\models\phonikud.onnx"
    File "..\models\tokenizer.json"

    ; Voice styles
    SetOutPath "$INSTDIR\models\voices"
    File "..\models\voices\male1.json"
    File "..\models\voices\female1.json"
SectionEnd
