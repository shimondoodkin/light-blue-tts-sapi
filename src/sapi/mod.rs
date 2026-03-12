//! SAPI 5 COM module — COM interface definitions, class factory, engine, and
//! registry helpers.
//!
//! # Layout
//! - `engine.rs`   — `TtsEngine` struct + `TtsSynthesizer` trait
//! - `factory.rs`  — `ClassFactory` implementing `IClassFactory`
//! - `register.rs` — registry helpers for `DllRegisterServer`
//!
//! The four DLL entry points live in `lib.rs` and delegate to the public
//! functions exported from this module.

pub mod engine;
pub mod factory;
pub mod register;

// Re-exports for convenience.
pub use engine::{
    BoxErr,
    SynthesisMetadata,
    SynthesisOutput,
    TextSpan,
    TtsEngine,
    TtsSynthesizer,
    WordTiming,
};
pub use factory::{set_global_synthesizer, ClassFactory};

use windows::Win32::Foundation::{E_FAIL, E_NOINTERFACE, S_FALSE};
use windows_core::{Interface, IUnknown, IUnknown_Vtbl, GUID, HRESULT};

// ---------------------------------------------------------------------------
// CLSID for our TTS engine: {A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
// ---------------------------------------------------------------------------

pub const ENGINE_CLSID: GUID = GUID::from_values(
    0xA1B2C3D4,
    0xE5F6,
    0x7890,
    [0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x78, 0x90],
);

pub const ENGINE_CLSID_STR: &str = "A1B2C3D4-E5F6-7890-ABCD-EF1234567890";

// Keep the old constant around as an alias so existing code that references
// it continues to compile.
pub const CLSID_LIGHTBLUE_TTS: GUID = ENGINE_CLSID;

// ---------------------------------------------------------------------------
// Manually-defined COM interfaces
// ---------------------------------------------------------------------------
// The `windows` crate's Win32_Media_Speech feature does not expose
// ISpTTSEngine / ISpTTSEngineSite / ISpObjectWithToken in its generated
// bindings.  We define them here using the `#[interface]` attribute so the
// `#[implement]` macro can generate vtables for them.

// ---- SPVTEXTFRAG (simplified) ----

/// Minimal representation of the SPVTEXTFRAG linked-list node that SAPI
/// passes to `ISpTTSEngine::Speak`.  We only care about the text pointer,
/// length, and the `pNext` link.
#[repr(C)]
pub struct SPVTEXTFRAG {
    pub pNext: *const SPVTEXTFRAG,
    pub State: SPVSTATE,
    pub pTextStart: *const u16,
    pub ulTextLen: u32,
    pub ulTextSrcOffset: u32,
}

/// SPVSTATE matching the SAPI 5 SDK layout.
/// We don't interpret the fields but the struct size must be exact so that
/// SPVTEXTFRAG field offsets are correct.
#[repr(C)]
#[allow(non_snake_case)]
pub struct SPVSTATE {
    pub eAction: u32,          // SPVACTIONS enum
    pub LangID: u16,           // LANGID
    pub wReserved: u16,
    pub EmphAdj: i32,          // long
    pub RateAdj: i32,          // long
    pub Volume: u32,           // ULONG
    pub PitchMiddle: i32,      // SPVPITCH.MiddleAdj (long)
    pub PitchRange: i32,       // SPVPITCH.RangeAdj  (long)
    pub SilenceMSecs: u32,     // ULONG
    pub pPhoneIds: *const u16, // SPPHONEID* (WCHAR*)
    pub ePartOfSpeech: u32,    // SPPARTOFSPEECH enum
    _pad: u32,                 // alignment padding before Context pointers
    pub pCategory: *const u16, // SPVCONTEXT.pCategory (LPCWSTR)
    pub pBefore: *const u16,   // SPVCONTEXT.pBefore   (LPCWSTR)
    pub pAfter: *const u16,    // SPVCONTEXT.pAfter    (LPCWSTR)
}

/// Action flags returned by `ISpTTSEngineSite::GetActions`.
pub const SPVES_ABORT: u32 = 0x1; // SPVESACTIONS::SPVES_ABORT
pub const SPEI_WORD_BOUNDARY: u16 = 5;
pub const SPET_LPARAM_IS_UNDEFINED: u16 = 0;
pub const SPFEI_FLAGCHECK: u64 = (1u64 << 30) | (1u64 << 33);
pub const SPFEI_WORD_BOUNDARY: u64 = (1u64 << SPEI_WORD_BOUNDARY) | SPFEI_FLAGCHECK;

/// Minimal SPEVENT layout used with ISpEventSink::AddEvents.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SPEVENT {
    pub eEventId: u16,
    pub elParamType: u16,
    pub ulStreamNum: u32,
    pub ullAudioStreamOffset: u64,
    pub wParam: usize,
    pub lParam: isize,
}

// ---- ISpTTSEngineSite ----

/// ISpTTSEngineSite  {9880499B-CCE9-11D2-B503-00C04F797396}
///
/// Inheritance chain: ISpTTSEngineSite -> ISpEventSink -> IUnknown.
/// We flatten the vtable here because we only consume (not implement) this
/// interface.  The method order must match the real COM vtable exactly:
///   IUnknown (3 slots) | ISpEventSink (2 slots) | ISpTTSEngineSite (6 slots)
#[windows_core::interface("9880499B-CCE9-11D2-B503-00C04F797396")]
pub unsafe trait ISpTTSEngineSite: IUnknown {
    // -- ISpEventSink methods --
    unsafe fn AddEvents(&self, p_event_array: *const SPEVENT, ul_count: u32) -> HRESULT;
    unsafe fn GetEventInterest(&self, pull_event_interest: *mut u64) -> HRESULT;
    // -- ISpTTSEngineSite methods --
    unsafe fn GetActions(&self) -> u32;
    unsafe fn Write(
        &self,
        p_buff: *const core::ffi::c_void,
        cb: u32,
        pcb_written: *mut u32,
    ) -> HRESULT;
    unsafe fn GetRate(&self, prate: *mut i32) -> HRESULT;
    unsafe fn GetVolume(&self, pvolume: *mut u16) -> HRESULT;
    unsafe fn GetSkipInfo(
        &self,
        pe_type: *mut u32,
        pl_num_items: *mut i32,
    ) -> HRESULT;
    unsafe fn CompleteSkip(&self, ul_num_skipped: i32) -> HRESULT;
}

// ---- ISpTTSEngine ----

/// ISpTTSEngine  {A74D7C8E-4CC5-4F2F-A6EB-804DEE18500E}
/// IMPORTANT: Method order MUST match the SAPI SDK vtable — Speak first (slot 3),
/// GetOutputFormat second (slot 4).
#[windows_core::interface("A74D7C8E-4CC5-4F2F-A6EB-804DEE18500E")]
pub unsafe trait ISpTTSEngine: IUnknown {
    unsafe fn Speak(
        &self,
        dw_speak_flags: u32,
        rguid_format_id: *const GUID,
        p_wave_format_ex: *const u8,
        p_text_frag_list: *const SPVTEXTFRAG,
        p_output_site: *mut core::ffi::c_void,
    ) -> HRESULT;

    unsafe fn GetOutputFormat(
        &self,
        p_target_fmt_id: *const GUID,
        p_target_wave_format_ex: *const u8,
        p_output_format_id: *mut GUID,
        pp_comem_output_wave_format_ex: *mut *mut u8,
    ) -> HRESULT;
}

// ---- ISpObjectWithToken ----

/// ISpObjectWithToken  {5B559F40-E952-11D2-BB91-00C04F8EE6C0}
#[windows_core::interface("5B559F40-E952-11D2-BB91-00C04F8EE6C0")]
pub unsafe trait ISpObjectWithToken: IUnknown {
    unsafe fn SetObjectToken(
        &self,
        ptoken: *mut core::ffi::c_void,
    ) -> HRESULT;
    unsafe fn GetObjectToken(
        &self,
        pptoken: *mut *mut core::ffi::c_void,
    ) -> HRESULT;
}

// ---------------------------------------------------------------------------
// Public functions called from the DLL entry points in lib.rs
// ---------------------------------------------------------------------------

/// Implementation of `DllGetClassObject`.
///
/// # Safety
/// Caller must pass valid COM pointers.
pub unsafe fn dll_get_class_object(
    rclsid: *const GUID,
    riid: *const GUID,
    ppv: *mut *mut core::ffi::c_void,
) -> HRESULT {
    if ppv.is_null() {
        return E_NOINTERFACE;
    }
    *ppv = std::ptr::null_mut();

    if rclsid.is_null() || riid.is_null() {
        return E_NOINTERFACE;
    }

    let clsid = &*rclsid;
    if *clsid != ENGINE_CLSID {
        log::warn!("DllGetClassObject: unknown CLSID {clsid:?}");
        return E_NOINTERFACE;
    }

    let factory = ClassFactory;
    let unknown: IUnknown = factory.into();
    (Interface::vtable(&unknown).QueryInterface)(
        Interface::as_raw(&unknown),
        riid,
        ppv,
    )
}

/// Implementation of `DllCanUnloadNow`.
pub fn dll_can_unload_now() -> HRESULT {
    S_FALSE
}

/// Implementation of `DllRegisterServer`.
///
/// # Safety
/// Calls Win32 registry APIs and `GetModuleHandleExW` / `GetModuleFileNameW`.
pub unsafe fn dll_register_server() -> HRESULT {
    let dll_path = match get_dll_path() {
        Some(p) => p,
        None => {
            log::error!("DllRegisterServer: could not determine DLL path");
            return E_FAIL;
        }
    };

    register::register_server(&dll_path)
}

/// Implementation of `DllUnregisterServer`.
///
/// # Safety
/// Calls Win32 registry APIs.
pub unsafe fn dll_unregister_server() -> HRESULT {
    register::unregister_server()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Retrieve the full filesystem path of the currently loaded DLL.
unsafe fn get_dll_path() -> Option<String> {
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
        windows_core::PCWSTR(get_dll_path as *const u16),
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
    Some(String::from_utf16_lossy(&buf[..len as usize]))
}
