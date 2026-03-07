; LightBlue Hebrew TTS - NSIS Installer (App Only)
; No models included — install models separately with LightBlue-TTS-Models.exe

!include "MUI2.nsh"

; --- General ---
Name "LightBlue Hebrew TTS"
OutFile "..\target\installer\LightBlue-TTS-Setup.exe"
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
    File "..\target\release\lightblue-download.exe"

    ; ONNX Runtime (WinML/DirectML — CPU fallback)
    File "..\target\release\onnxruntime.dll"

    ; Model download script
    File "..\scripts\download_models.py"

    ; Management scripts
    File "install.ps1"
    File "uninstall.ps1"

    ; Create directory structure for models
    CreateDirectory "$INSTDIR\models\onnx"
    CreateDirectory "$INSTDIR\models\voices"
    CreateDirectory "$INSTDIR\dictionaries"

    ; Register COM DLL
    ExecWait 'regsvr32 /s "$INSTDIR\lightblue_sapi.dll"'

    ; Create uninstaller
    WriteUninstaller "$INSTDIR\Uninstall.exe"

    ; Start menu shortcuts
    CreateDirectory "$SMPROGRAMS\LightBlue TTS"
    CreateShortcut "$SMPROGRAMS\LightBlue TTS\Download Models.lnk" \
        "$INSTDIR\lightblue-download.exe" \
        '--dir "$INSTDIR"' \
        "$INSTDIR\icon.ico"
    CreateShortcut "$SMPROGRAMS\LightBlue TTS\Uninstall.lnk" "$INSTDIR\Uninstall.exe"

    ; Add/Remove Programs entry
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LightBlueTTS" \
        "DisplayName" "LightBlue Hebrew TTS"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LightBlueTTS" \
        "UninstallString" '"$INSTDIR\Uninstall.exe"'
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LightBlueTTS" \
        "Publisher" "LightBlue"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LightBlueTTS" \
        "DisplayIcon" "$INSTDIR\icon.ico"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LightBlueTTS" \
        "DisplayVersion" "1.0.0"
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LightBlueTTS" \
        "NoModify" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LightBlueTTS" \
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
    Delete "$INSTDIR\onnxruntime.dll"
    Delete "$INSTDIR\onnxruntime_providers_shared.dll"
    Delete "$INSTDIR\onnxruntime_providers_cuda.dll"
    Delete "$INSTDIR\onnxruntime_providers_tensorrt.dll"
    Delete "$INSTDIR\lightblue-tts.exe"
    Delete "$INSTDIR\lightblue-download.exe"
    Delete "$INSTDIR\download_models.py"
    Delete "$INSTDIR\install.ps1"
    Delete "$INSTDIR\uninstall.ps1"
    Delete "$INSTDIR\Uninstall.exe"
    RMDir "$INSTDIR"

    ; Remove Start Menu shortcuts
    RMDir /r "$SMPROGRAMS\LightBlue TTS"

    ; Remove Add/Remove Programs entry
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\LightBlueTTS"
SectionEnd
