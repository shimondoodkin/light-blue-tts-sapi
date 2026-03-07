; LightBlue Hebrew TTS - Models Installer
; Extracts pre-optimized ONNX models to the app folder

!include "MUI2.nsh"

; --- General ---
Name "LightBlue Hebrew TTS - Models"
OutFile "..\target\installer\LightBlue-TTS-Models.exe"
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
    ; TTS config
    SetOutPath "$INSTDIR\models"
    File "..\models\tts.json"
    File "..\models\phonikud.onnx"
    File "..\models\tokenizer.json"

    ; Voice styles
    SetOutPath "$INSTDIR\models\voices"
    File "..\models\voices\male1.json"
    File "..\models\voices\female1.json"

    ; ONNX models
    SetOutPath "$INSTDIR\models\onnx"
    File "..\models\onnx\backbone_keys.onnx"
    File "..\models\onnx\text_encoder.onnx"
    File "..\models\onnx\reference_encoder.onnx"
    File "..\models\onnx\vocoder.onnx"
    File "..\models\onnx\length_pred.onnx"
    File "..\models\onnx\length_pred_style.onnx"
    File "..\models\onnx\stats.npz"
    File "..\models\onnx\uncond.npz"
SectionEnd
