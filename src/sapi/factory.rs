//! COM ClassFactory for TtsEngine.
//!
//! SAPI (via CoCreateInstance / DllGetClassObject) asks for an IClassFactory
//! and then calls `CreateInstance` to obtain our TtsEngine.

use std::sync::Arc;

use windows::Win32::Foundation::{CLASS_E_NOAGGREGATION, E_POINTER};
use windows::Win32::System::Com::{IClassFactory, IClassFactory_Impl};
use windows_core::{IUnknown, Interface, GUID};

use super::engine::{TtsEngine, TtsSynthesizer};

// ---------------------------------------------------------------------------
// Global synthesizer that will be injected before COM activation.
// ---------------------------------------------------------------------------

static SYNTHESIZER: std::sync::OnceLock<Arc<dyn TtsSynthesizer>> = std::sync::OnceLock::new();

/// Call this once during DLL initialisation (before SAPI creates instances)
/// to provide the real TTS pipeline.
pub fn set_global_synthesizer(synth: Arc<dyn TtsSynthesizer>) {
    let _ = SYNTHESIZER.set(synth);
}

/// Retrieve the global synthesizer, if one has been set.
pub fn get_global_synthesizer() -> Option<Arc<dyn TtsSynthesizer>> {
    SYNTHESIZER.get().cloned()
}

// ---------------------------------------------------------------------------
// ClassFactory COM object
// ---------------------------------------------------------------------------

#[windows_core::implement(IClassFactory)]
pub struct ClassFactory;

impl IClassFactory_Impl for ClassFactory_Impl {
    fn CreateInstance(
        &self,
        punkouter: Option<&IUnknown>,
        riid: *const GUID,
        ppvobject: *mut *mut core::ffi::c_void,
    ) -> windows_core::Result<()> {
        unsafe {
            if ppvobject.is_null() {
                return Err(E_POINTER.into());
            }
            *ppvobject = std::ptr::null_mut();

            // Aggregation is not supported.
            if punkouter.is_some() {
                return Err(CLASS_E_NOAGGREGATION.into());
            }

            // Build the engine, injecting the synthesizer if available.
            log::info!("ClassFactory::CreateInstance called — creating new TtsEngine");
            let engine = match get_global_synthesizer() {
                Some(synth) => TtsEngine::new(synth),
                None => {
                    log::warn!("No global synthesizer set — using stub (silence)");
                    TtsEngine::new_stub()
                }
            };

            // The `into()` conversion gives us an IUnknown-backed COM pointer
            // with ref-count == 1. QueryInterface for the requested IID.
            let unknown: IUnknown = engine.into();
            let hr = (Interface::vtable(&unknown).QueryInterface)(
                Interface::as_raw(&unknown),
                riid,
                ppvobject,
            );
            hr.ok()
        }
    }

    fn LockServer(&self, _flock: windows::Win32::Foundation::BOOL) -> windows_core::Result<()> {
        // We report that we can never unload (see DllCanUnloadNow), so
        // LockServer is effectively a no-op.
        Ok(())
    }
}
