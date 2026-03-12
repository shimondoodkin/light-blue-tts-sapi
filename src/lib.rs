//! LightBlue TTS -- Hebrew neural text-to-speech as a Windows SAPI 5 voice engine.
//!
//! This crate is compiled as a `cdylib` and exposes the four standard COM DLL
//! entry points (`DllGetClassObject`, `DllCanUnloadNow`, `DllRegisterServer`,
//! `DllUnregisterServer`) so that `regsvr32` can register it as a SAPI voice.

pub mod bridge;
pub mod phonemize;
pub mod sapi;
pub mod tts;

use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Once};

static INIT: Once = Once::new();

/// Initialize logging and register a lazy synthesizer wrapper.
///
/// This is safe to call multiple times; only the first call has any effect.
/// It is invoked automatically from `DllGetClassObject`, but it avoids heavy
/// model loading on the COM activation path so host applications do not stall
/// while enumerating or instantiating the voice.
pub fn init() {
    INIT.call_once(|| {
        // If lightblue.log exists next to the DLL, log into it (debug level).
        // Otherwise fall back to env_logger (stderr, respects RUST_LOG).
        let mut file_logging = false;
        if let Some(dll_dir) = get_dll_dir() {
            let log_path = PathBuf::from(&dll_dir).join("log.txt");
            if log_path.exists() {
                if let Ok(file) = std::fs::OpenOptions::new()
                    .append(true)
                    .open(&log_path)
                {
                    file_logging = env_logger::Builder::new()
                        .filter_level(log::LevelFilter::Debug)
                        .target(env_logger::Target::Pipe(Box::new(file)))
                        .format(|buf, record| {
                            writeln!(buf, "[{} {}] {}", record.level(), record.target(), record.args())
                        })
                        .try_init()
                        .is_ok();
                }
            }
        }
        if !file_logging {
            let _ = env_logger::try_init();
        }

        log::info!("LightBlue TTS engine initializing");

        // Register a lazy wrapper so COM activation stays fast.
        match create_lazy_synthesizer() {
            Ok(synth) => {
                sapi::set_global_synthesizer(synth.clone());

                // Warm the pipeline in the background without blocking the host.
                std::thread::spawn(move || {
                    if let Err(e) = synth.warm_up() {
                        log::warn!("Background warmup failed: {e}");
                    }
                });
                log::info!("LightBlue TTS lazy pipeline registered");
            }
            Err(e) => log::error!("Failed to prepare lazy TTS pipeline (will use stub): {e}"),
        }
    });
}

/// Determine the directory where our DLL lives, then create a lazy wrapper
/// that can build the real synthesizer on first use.
fn create_lazy_synthesizer() -> Result<Arc<bridge::LazyLightBlueSynthesizer>, Box<dyn std::error::Error>> {
    let dll_dir = get_dll_dir()
        .ok_or("Could not determine DLL directory")?;
    let dll_dir = PathBuf::from(&dll_dir);

    log::info!("DLL directory: {}", dll_dir.display());

    // Point ORT to our bundled onnxruntime.dll to avoid picking up wrong versions
    let ort_dll = dll_dir.join("onnxruntime.dll");
    if ort_dll.exists() {
        std::env::set_var("ORT_DYLIB_PATH", &ort_dll);
        log::info!("Set ORT_DYLIB_PATH to {}", ort_dll.display());
    }

    // Add the DLL directory to the DLL search path so that dependencies of
    // onnxruntime.dll (e.g. onnxruntime_providers_shared.dll, DirectML.dll,
    // OpenVINO DLLs) can be found when loaded via LoadLibraryW.
    // SetDllDirectoryW is needed because SAPI hosts (svchost.exe etc.) won't
    // have our install dir in their search path, and modifying PATH env var
    // alone does NOT affect LoadLibraryW on modern Windows.
    {
        use windows::core::HSTRING;
        use windows::Win32::System::LibraryLoader::{SetDllDirectoryW, AddDllDirectory};
        let dir_hstr = HSTRING::from(dll_dir.as_os_str());
        unsafe {
            // SetDllDirectoryW adds to the DLL search order
            if SetDllDirectoryW(&dir_hstr).is_ok() {
                log::info!("SetDllDirectoryW to {}", dll_dir.display());
            }
            // AddDllDirectory is also used by LOAD_LIBRARY_SEARCH_USER_DIRS
            let _ = AddDllDirectory(&dir_hstr);
        }
    }
    // Also add to PATH for any subprocess or SearchPath usage
    if let Ok(current_path) = std::env::var("PATH") {
        let dll_dir_str = dll_dir.to_string_lossy();
        if !current_path.split(';').any(|d| d == dll_dir_str.as_ref()) {
            std::env::set_var("PATH", format!("{};{}", dll_dir.display(), current_path));
        }
    }

    // Model paths — all relative to the DLL's install directory
    let models_dir = dll_dir.join("models");
    let phonikud_model = models_dir.join("phonikud.onnx");
    let phonikud_tokenizer = models_dir.join("tokenizer.json");

    let style_json = models_dir.join("voices/male1.json");
    let style_json_path = if style_json.exists() {
        Some(style_json.to_string_lossy().into_owned())
    } else {
        None
    };

    // Build TTSConfig
    let mut tts_config = tts::TTSConfig::new(&models_dir);
    tts_config.default_style_json = style_json_path.as_ref().map(PathBuf::from);

    let synth = bridge::LazyLightBlueSynthesizer::new(
        tts_config,
        &phonikud_model.to_string_lossy(),
        &phonikud_tokenizer.to_string_lossy(),
        style_json_path,
    );

    Ok(Arc::new(synth))
}

/// Get the parent directory of the currently loaded DLL.
fn get_dll_dir() -> Option<String> {
    // Safety: uses Win32 APIs to locate our own DLL.
    unsafe {
        use windows::Win32::Foundation::HMODULE;
        use windows::Win32::System::LibraryLoader::{
            GetModuleFileNameW, GetModuleHandleExW,
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        };

        let mut hmod: HMODULE = HMODULE::default();
        let flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
            | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;

        let ok = GetModuleHandleExW(
            flags,
            windows_core::PCWSTR(get_dll_dir as *const u16),
            &mut hmod,
        );
        if ok.is_err() {
            return None;
        }

        let mut buf = vec![0u16; 1024];
        let len = GetModuleFileNameW(hmod, &mut buf);
        if len == 0 {
            return None;
        }
        let path = String::from_utf16_lossy(&buf[..len as usize]);
        // Return the parent directory
        PathBuf::from(&path)
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
    }
}

// ---------------------------------------------------------------------------
// COM DLL entry points -- delegate to the sapi module
// ---------------------------------------------------------------------------

use windows_core::HRESULT;

/// COM entry point: returns a class factory for the requested CLSID.
///
/// # Safety
/// Called by the COM runtime. The caller must supply valid pointers.
#[no_mangle]
pub unsafe extern "system" fn DllGetClassObject(
    rclsid: *const windows_core::GUID,
    riid: *const windows_core::GUID,
    ppv: *mut *mut core::ffi::c_void,
) -> HRESULT {
    init();
    sapi::dll_get_class_object(rclsid, riid, ppv)
}

/// COM entry point: returns `S_OK` if the DLL can be unloaded, `S_FALSE` otherwise.
///
/// # Safety
/// Called by the COM runtime.
#[no_mangle]
pub unsafe extern "system" fn DllCanUnloadNow() -> HRESULT {
    sapi::dll_can_unload_now()
}

/// COM entry point: registers this DLL as a SAPI voice in the Windows registry.
///
/// # Safety
/// Called by `regsvr32`. Requires administrator privileges.
#[no_mangle]
pub unsafe extern "system" fn DllRegisterServer() -> HRESULT {
    sapi::dll_register_server()
}

/// COM entry point: removes the SAPI voice registration from the Windows registry.
///
/// # Safety
/// Called by `regsvr32 /u`. Requires administrator privileges.
#[no_mangle]
pub unsafe extern "system" fn DllUnregisterServer() -> HRESULT {
    sapi::dll_unregister_server()
}
