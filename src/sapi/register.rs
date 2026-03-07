//! DllRegisterServer / DllUnregisterServer helpers.
//!
//! Registers two SAPI 5 voices (male and female) that share the same COM
//! CLSID / DLL but use different style JSON files selected via a custom
//! registry value on each voice token.

use windows::Win32::Foundation::{E_FAIL, S_OK};
use windows::Win32::System::Registry::{
    RegCloseKey, RegCreateKeyExW, RegDeleteTreeW, RegSetValueExW, HKEY, HKEY_LOCAL_MACHINE,
    KEY_WRITE, REG_OPTION_NON_VOLATILE, REG_SZ,
};
use windows_core::{HRESULT, PCWSTR};

use super::ENGINE_CLSID_STR;

// ---------------------------------------------------------------------------
// Voice definitions
// ---------------------------------------------------------------------------

struct VoiceDef {
    token_name: &'static str,
    display_name: &'static str,
    gender: &'static str,
    style_file: &'static str,
    steps: u32,
    device: &'static str,
}

const VOICES: &[VoiceDef] = &[
    // Male voices — CPU variants
    VoiceDef { token_name: "LightBlueHebrewMale_4CPU",   display_name: "LightBlueTTS Shaul 4 CPU - Hebrew",   gender: "Male",   style_file: "voices/male1.json",   steps: 4,   device: "CPU" },
    VoiceDef { token_name: "LightBlueHebrewMale_8CPU",   display_name: "LightBlueTTS Shaul 8 CPU - Hebrew",   gender: "Male",   style_file: "voices/male1.json",   steps: 8,   device: "CPU" },
    VoiceDef { token_name: "LightBlueHebrewMale_32CPU",  display_name: "LightBlueTTS Shaul 32 CPU - Hebrew",  gender: "Male",   style_file: "voices/male1.json",   steps: 32,  device: "CPU" },
    VoiceDef { token_name: "LightBlueHebrewMale_64CPU",  display_name: "LightBlueTTS Shaul 64 CPU - Hebrew",  gender: "Male",   style_file: "voices/male1.json",   steps: 64,  device: "CPU" },
    // Male voices — GPU variants
    VoiceDef { token_name: "LightBlueHebrewMale_8GPU",   display_name: "LightBlueTTS Shaul 8 GPU - Hebrew",   gender: "Male",   style_file: "voices/male1.json",   steps: 8,   device: "GPU" },
    VoiceDef { token_name: "LightBlueHebrewMale_32GPU",  display_name: "LightBlueTTS Shaul 32 GPU - Hebrew",  gender: "Male",   style_file: "voices/male1.json",   steps: 32,  device: "GPU" },
    VoiceDef { token_name: "LightBlueHebrewMale_64GPU",  display_name: "LightBlueTTS Shaul 64 GPU - Hebrew",  gender: "Male",   style_file: "voices/male1.json",   steps: 64,  device: "GPU" },
    // Female voices — CPU variants
    VoiceDef { token_name: "LightBlueHebrewFemale_4CPU",   display_name: "LightBlueTTS Rotem 4 CPU - Hebrew",   gender: "Female", style_file: "voices/female1.json", steps: 4,   device: "CPU" },
    VoiceDef { token_name: "LightBlueHebrewFemale_8CPU",   display_name: "LightBlueTTS Rotem 8 CPU - Hebrew",   gender: "Female", style_file: "voices/female1.json", steps: 8,   device: "CPU" },
    VoiceDef { token_name: "LightBlueHebrewFemale_32CPU",  display_name: "LightBlueTTS Rotem 32 CPU - Hebrew",  gender: "Female", style_file: "voices/female1.json", steps: 32,  device: "CPU" },
    VoiceDef { token_name: "LightBlueHebrewFemale_64CPU",  display_name: "LightBlueTTS Rotem 64 CPU - Hebrew",  gender: "Female", style_file: "voices/female1.json", steps: 64,  device: "CPU" },
    // Female voices — GPU variants
    VoiceDef { token_name: "LightBlueHebrewFemale_8GPU",   display_name: "LightBlueTTS Rotem 8 GPU - Hebrew",   gender: "Female", style_file: "voices/female1.json", steps: 8,   device: "GPU" },
    VoiceDef { token_name: "LightBlueHebrewFemale_32GPU",  display_name: "LightBlueTTS Rotem 32 GPU - Hebrew",  gender: "Female", style_file: "voices/female1.json", steps: 32,  device: "GPU" },
    VoiceDef { token_name: "LightBlueHebrewFemale_64GPU",  display_name: "LightBlueTTS Rotem 64 GPU - Hebrew",  gender: "Female", style_file: "voices/female1.json", steps: 64,  device: "GPU" },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a wide (UTF-16) null-terminated string from a Rust &str.
fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0u16)).collect()
}

unsafe fn create_key(parent: HKEY, subkey: &str) -> Result<HKEY, HRESULT> {
    let subkey_w = wide(subkey);
    let mut hkey = HKEY::default();
    let status = RegCreateKeyExW(
        parent,
        PCWSTR(subkey_w.as_ptr()),
        0,
        PCWSTR::null(),
        REG_OPTION_NON_VOLATILE,
        KEY_WRITE,
        None,
        &mut hkey,
        None,
    );
    if status.is_err() {
        log::error!("RegCreateKeyExW failed for {subkey}: {status:?}");
        Err(E_FAIL)
    } else {
        Ok(hkey)
    }
}

unsafe fn set_string_value(hkey: HKEY, name: Option<&str>, value: &str) -> Result<(), HRESULT> {
    let value_w = wide(value);
    let name_w;
    let name_ptr = match name {
        Some(n) => {
            name_w = wide(n);
            PCWSTR(name_w.as_ptr())
        }
        None => PCWSTR::null(), // default value
    };
    let byte_len = value_w.len() * 2; // including null terminator
    let status = RegSetValueExW(
        hkey,
        name_ptr,
        0,
        REG_SZ,
        Some(std::slice::from_raw_parts(
            value_w.as_ptr() as *const u8,
            byte_len,
        )),
    );
    if status.is_err() {
        Err(E_FAIL)
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Public: register
// ---------------------------------------------------------------------------

/// Write all registry entries required for SAPI to see this voice engine.
///
/// Registers the InprocServer32 for the shared CLSID, then creates a
/// separate voice token for each voice (male, female).
///
/// # Safety
/// Calls Win32 registry APIs.
pub unsafe fn register_server(dll_path: &str) -> HRESULT {
    let clsid_braced = format!("{{{}}}", ENGINE_CLSID_STR);

    // --- InprocServer32 (shared by all voices) ---
    let inproc_subkey = format!(
        "SOFTWARE\\Classes\\CLSID\\{}\\InprocServer32",
        clsid_braced
    );
    match (|| -> Result<(), HRESULT> {
        let hkey = create_key(HKEY_LOCAL_MACHINE, &inproc_subkey)?;
        set_string_value(hkey, None, dll_path)?;
        set_string_value(hkey, Some("ThreadingModel"), "Both")?;
        let _ = RegCloseKey(hkey);
        Ok(())
    })() {
        Ok(()) => {}
        Err(hr) => return hr,
    }

    // --- Register each voice in both classic SAPI and OneCore locations ---
    let speech_roots = [
        "SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens",
        "SOFTWARE\\Microsoft\\Speech_OneCore\\Voices\\Tokens",
    ];

    for voice in VOICES {
        for root in &speech_roots {
            let token_subkey = format!("{}\\{}", root, voice.token_name);
            match (|| -> Result<(), HRESULT> {
                let hkey = create_key(HKEY_LOCAL_MACHINE, &token_subkey)?;
                set_string_value(hkey, None, voice.display_name)?;
                set_string_value(hkey, Some("CLSID"), &clsid_braced)?;
                set_string_value(hkey, Some("40D"), voice.display_name)?;
                set_string_value(hkey, Some("StyleFile"), voice.style_file)?;
                set_string_value(hkey, Some("Steps"), &voice.steps.to_string())?;
                set_string_value(hkey, Some("Device"), voice.device)?;
                let _ = RegCloseKey(hkey);
                Ok(())
            })() {
                Ok(()) => {}
                Err(hr) => {
                    log::warn!("Failed to register voice in {root}: 0x{hr:08X?}");
                    // Don't fail entirely if OneCore registration fails
                }
            }

            let attr_subkey = format!("{}\\{}\\Attributes", root, voice.token_name);
            match (|| -> Result<(), HRESULT> {
                let hkey = create_key(HKEY_LOCAL_MACHINE, &attr_subkey)?;
                set_string_value(hkey, Some("Language"), "40D")?;
                set_string_value(hkey, Some("Gender"), voice.gender)?;
                set_string_value(hkey, Some("Name"), voice.display_name)?;
                set_string_value(hkey, Some("Vendor"), "LightBlue")?;
                let _ = RegCloseKey(hkey);
                Ok(())
            })() {
                Ok(()) => {}
                Err(hr) => {
                    log::warn!("Failed to register attributes in {root}: 0x{hr:08X?}");
                }
            }
        }

        log::info!("Registered voice: {}", voice.display_name);
    }

    log::info!("DllRegisterServer succeeded");
    S_OK
}

// ---------------------------------------------------------------------------
// Public: unregister
// ---------------------------------------------------------------------------

/// Remove all registry entries written by `register_server`.
///
/// # Safety
/// Calls Win32 registry APIs.
pub unsafe fn unregister_server() -> HRESULT {
    let clsid_braced = format!("{{{}}}", ENGINE_CLSID_STR);

    // Delete the CLSID subtree.
    let clsid_subkey = wide(&format!("SOFTWARE\\Classes\\CLSID\\{}", clsid_braced));
    let _ = RegDeleteTreeW(HKEY_LOCAL_MACHINE, PCWSTR(clsid_subkey.as_ptr()));

    // Delete all voice token subtrees from both classic and OneCore locations.
    let speech_roots = [
        "SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens",
        "SOFTWARE\\Microsoft\\Speech_OneCore\\Voices\\Tokens",
    ];
    for voice in VOICES {
        for root in &speech_roots {
            let token_subkey = wide(&format!("{}\\{}", root, voice.token_name));
            let _ = RegDeleteTreeW(HKEY_LOCAL_MACHINE, PCWSTR(token_subkey.as_ptr()));
        }
    }

    // Clean up legacy token names from previous versions
    for old_name in &[
        "LightBlueHebrew", "LightBlueHebrewMale", "LightBlueHebrewFemale",
        "LightBlueHebrewMale_100CPU", "LightBlueHebrewMale_100GPU",
        "LightBlueHebrewFemale_100CPU", "LightBlueHebrewFemale_100GPU",
    ] {
        for root in &speech_roots {
            let old_token = wide(&format!("{}\\{}", root, old_name));
            let _ = RegDeleteTreeW(HKEY_LOCAL_MACHINE, PCWSTR(old_token.as_ptr()));
        }
    }

    log::info!("DllUnregisterServer succeeded");
    S_OK
}
