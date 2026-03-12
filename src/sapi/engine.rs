//! TtsEngine — implements ISpTTSEngine and ISpObjectWithToken for SAPI 5.
//!
//! The engine receives text fragments from SAPI, passes them through a
//! `TtsSynthesizer` trait object, converts the resulting f32 samples to
//! 16-bit PCM, and writes them back to SAPI via `ISpTTSEngineSite::Write`.

use std::sync::{Arc, Mutex};

use windows::Win32::Foundation::{E_FAIL, E_POINTER, S_OK};
use windows_core::{GUID, HRESULT, IUnknown_Vtbl, Interface};

use super::{ISpTTSEngine, ISpTTSEngine_Impl, ISpTTSEngineSite, ISpObjectWithToken, ISpObjectWithToken_Impl};
use super::{
    SPEVENT, SPEI_WORD_BOUNDARY, SPET_LPARAM_IS_UNDEFINED, SPFEI_WORD_BOUNDARY, SPVTEXTFRAG,
    SPVES_ABORT,
};

// ---------------------------------------------------------------------------
// TtsSynthesizer trait — the bridge to the actual neural TTS pipeline
// ---------------------------------------------------------------------------

pub type BoxErr = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Clone, Copy)]
pub struct TextSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone)]
pub struct WordTiming {
    pub text: String,
    pub normalized_span: TextSpan,
    pub original_span: Option<TextSpan>,
    pub start_sec: f32,
    pub end_sec: f32,
}

#[derive(Debug, Clone)]
pub struct SynthesisMetadata {
    pub original_text: String,
    pub normalized_text: String,
    pub combined_ipa: String,
    pub words: Vec<WordTiming>,
}

#[derive(Debug, Clone)]
pub struct SynthesisOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub metadata: SynthesisMetadata,
}

/// Trait that the neural TTS pipeline must implement.
///
/// `synthesize` receives a plain-text string, an optional style JSON path
/// (to select male/female voice), optional step count, and a GPU flag,
/// and returns f32 audio samples + sample rate.
pub trait TtsSynthesizer: Send + Sync {
    fn synthesize(
        &self,
        text: &str,
        style_json: Option<&str>,
        steps: Option<u32>,
        use_gpu: bool,
    ) -> Result<SynthesisOutput, BoxErr>;

    fn synthesize_stream(
        &self,
        text: &str,
        style_json: Option<&str>,
        steps: Option<u32>,
        use_gpu: bool,
        sink: &mut dyn FnMut(&[f32]) -> Result<(), BoxErr>,
    ) -> Result<SynthesisMetadata, BoxErr> {
        let output = self.synthesize(text, style_json, steps, use_gpu)?;
        sink(&output.samples)?;
        Ok(output.metadata)
    }
}

// ---------------------------------------------------------------------------
// A placeholder / stub synthesizer used when none has been set.
// ---------------------------------------------------------------------------

struct StubSynthesizer;

impl TtsSynthesizer for StubSynthesizer {
    fn synthesize(
        &self,
        text: &str,
        _style_json: Option<&str>,
        _steps: Option<u32>,
        _use_gpu: bool,
    ) -> Result<SynthesisOutput, BoxErr> {
        // 0.5 seconds of silence at 44100 Hz
        Ok(SynthesisOutput {
            samples: vec![0.0f32; 22050],
            sample_rate: 44100,
            metadata: SynthesisMetadata {
                original_text: text.to_string(),
                normalized_text: text.to_string(),
                combined_ipa: String::new(),
                words: Vec::new(),
            },
        })
    }
}

// ---------------------------------------------------------------------------
// TtsEngine COM object
// ---------------------------------------------------------------------------

/// The SAPI 5 TTS engine COM object.
///
/// Marked with `#[implement]` so the windows crate generates the COM
/// pointers and reference-counting.  We list both COM interfaces that SAPI
/// queries for.
#[windows_core::implement(ISpTTSEngine, ISpObjectWithToken)]
pub struct TtsEngine {
    synthesizer: Arc<dyn TtsSynthesizer>,
    /// Full path to the style JSON for this voice instance (set from token registry).
    style_json_path: Mutex<Option<String>>,
    /// Diffusion steps for this voice (read from registry).
    steps: Mutex<Option<u32>>,
    /// Whether to use GPU for this voice (read from registry).
    use_gpu: Mutex<bool>,
    /// COM token pointer from SetObjectToken; kept with AddRef/Release semantics.
    object_token: Mutex<*mut core::ffi::c_void>,
}

impl TtsEngine {
    /// Create a new `TtsEngine` backed by the given synthesizer.
    pub fn new(synthesizer: Arc<dyn TtsSynthesizer>) -> Self {
        Self {
            synthesizer,
            style_json_path: Mutex::new(None),
            steps: Mutex::new(None),
            use_gpu: Mutex::new(false),
            object_token: Mutex::new(std::ptr::null_mut()),
        }
    }

    /// Create a `TtsEngine` with the stub (silent) synthesizer.
    pub fn new_stub() -> Self {
        Self::new(Arc::new(StubSynthesizer))
    }
}

impl Drop for TtsEngine {
    fn drop(&mut self) {
        log::info!("TtsEngine dropped (engine {:p})", self as *const _);
        if let Ok(mut slot) = self.object_token.lock() {
            unsafe {
                if !(*slot).is_null() {
                    let token = *slot;
                    let token_vtbl = *(token as *mut *mut IUnknown_Vtbl);
                    ((*token_vtbl).Release)(token);
                    *slot = std::ptr::null_mut();
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ISpObjectWithToken implementation
// ---------------------------------------------------------------------------

impl ISpObjectWithToken_Impl for TtsEngine_Impl {
    unsafe fn SetObjectToken(
        &self,
        ptoken: *mut core::ffi::c_void,
    ) -> HRESULT {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.set_object_token_inner(ptoken)
        }));
        match result {
            Ok(hr) => hr,
            Err(_) => {
                log::error!("SetObjectToken panicked");
                E_FAIL
            }
        }
    }

    unsafe fn GetObjectToken(
        &self,
        pptoken: *mut *mut core::ffi::c_void,
    ) -> HRESULT {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.get_object_token_inner(pptoken)
        }));
        match result {
            Ok(hr) => hr,
            Err(_) => {
                log::error!("GetObjectToken panicked");
                E_FAIL
            }
        }
    }
}

impl TtsEngine_Impl {
    unsafe fn set_object_token_inner(
        &self,
        ptoken: *mut core::ffi::c_void,
    ) -> HRESULT {
        if ptoken.is_null() {
            log::error!("SetObjectToken received null token");
            return E_POINTER;
        }

        // Hold our own reference to the token for GetObjectToken.
        let token_vtbl = *(ptoken as *mut *mut IUnknown_Vtbl);
        ((*token_vtbl).AddRef)(ptoken);

        let mut slot = self.object_token.lock().unwrap();
        if !(*slot).is_null() {
            let old_vtbl = *(*slot as *mut *mut IUnknown_Vtbl);
            ((*old_vtbl).Release)(*slot);
        }
        *slot = ptoken;

        log::debug!("SetObjectToken stored token {:?}", ptoken);

        // Read the token's registry ID to determine which voice variant this is.
        // Call ISpObjectToken::GetId via raw vtable (slot 16).
        if let Some(token_name) = get_token_name_from_com(ptoken) {
            log::info!("SetObjectToken: detected token '{token_name}'");
            let settings = read_token_settings(&token_name);

            if let Some(ref path) = settings.style_path {
                log::info!("  StyleFile: {path}");
                *self.style_json_path.lock().unwrap() = Some(path.clone());
            }
            if let Some(steps) = settings.steps {
                log::info!("  Steps: {steps}");
                *self.steps.lock().unwrap() = Some(steps);
            }
            log::info!("  Device: {}", if settings.use_gpu { "GPU" } else { "CPU" });
            *self.use_gpu.lock().unwrap() = settings.use_gpu;
        }

        S_OK
    }

    unsafe fn get_object_token_inner(
        &self,
        pptoken: *mut *mut core::ffi::c_void,
    ) -> HRESULT {
        if pptoken.is_null() {
            return E_POINTER;
        }

        let slot = self.object_token.lock().unwrap();
        if (*slot).is_null() {
            *pptoken = std::ptr::null_mut();
            log::error!("GetObjectToken requested before SetObjectToken");
            return E_FAIL;
        }

        let token = *slot;
        let token_vtbl = *(token as *mut *mut IUnknown_Vtbl);
        ((*token_vtbl).AddRef)(token);
        *pptoken = token;
        log::debug!("GetObjectToken returning token {:?}", token);
        S_OK
    }
}

// ---------------------------------------------------------------------------
// ISpTTSEngine implementation
// ---------------------------------------------------------------------------

/// Output audio parameters.
const SAMPLE_RATE: u32 = 44100;
const BITS_PER_SAMPLE: u16 = 16;
const NUM_CHANNELS: u16 = 1;
const BLOCK_ALIGN: u16 = NUM_CHANNELS * (BITS_PER_SAMPLE / 8);
const AVG_BYTES_PER_SEC: u32 = SAMPLE_RATE * BLOCK_ALIGN as u32;

/// SPDFID_WaveFormatEx  {C31ADBAE-527F-4FF5-A230-F62BB61FF70C}
const SPDFID_WAVE_FORMAT_EX: GUID = GUID::from_values(
    0xC31ADBAE,
    0x527F,
    0x4FF5,
    [0xA2, 0x30, 0xF6, 0x2B, 0xB6, 0x1F, 0xF7, 0x0C],
);

/// WAVE_FORMAT_PCM
const WAVE_FORMAT_PCM: u16 = 1;

/// Size of each PCM chunk written to the output site (in bytes).
/// ~100 ms of audio at 44100 Hz mono 16-bit = 8820 bytes.
const CHUNK_BYTES: usize = 8820;

/// WAVEFORMATEX structure laid out exactly as the C struct.
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct WaveFormatEx {
    w_format_tag: u16,
    n_channels: u16,
    n_samples_per_sec: u32,
    n_avg_bytes_per_sec: u32,
    n_block_align: u16,
    w_bits_per_sample: u16,
    cb_size: u16,
}

impl ISpTTSEngine_Impl for TtsEngine_Impl {
    unsafe fn Speak(
        &self,
        _dw_speak_flags: u32,
        _rguid_format_id: *const GUID,
        _p_wave_format_ex: *const u8,
        p_text_frag_list: *const SPVTEXTFRAG,
        p_output_site: *mut core::ffi::c_void,
    ) -> HRESULT {
        // Catch any panic to prevent aborting the host process.
        // COM callbacks are `extern "system"` — unwinding across them is UB and
        // triggers "panic in a function that cannot unwind" which calls abort().
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.speak_inner(p_text_frag_list, p_output_site)
        }));
        match result {
            Ok(hr) => hr,
            Err(panic) => {
                let msg = if let Some(s) = panic.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = panic.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "unknown panic".to_string()
                };
                log::error!("Speak panicked: {msg}");
                E_FAIL
            }
        }
    }

    unsafe fn GetOutputFormat(
        &self,
        _p_target_fmt_id: *const GUID,
        _p_target_wave_format_ex: *const u8,
        p_output_format_id: *mut GUID,
        pp_comem_output_wave_format_ex: *mut *mut u8,
    ) -> HRESULT {
        // Catch any panic to prevent aborting the host process.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.get_output_format_inner(
                _p_target_fmt_id,
                _p_target_wave_format_ex,
                p_output_format_id,
                pp_comem_output_wave_format_ex,
            )
        }));
        match result {
            Ok(hr) => hr,
            Err(panic) => {
                let msg = if let Some(s) = panic.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = panic.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "unknown panic".to_string()
                };
                log::error!("GetOutputFormat panicked: {msg}");
                E_FAIL
            }
        }
    }
}

impl TtsEngine_Impl {
    unsafe fn speak_inner(
        &self,
        p_text_frag_list: *const SPVTEXTFRAG,
        p_output_site: *mut core::ffi::c_void,
    ) -> HRESULT {
        let speak_start = std::time::Instant::now();
        log::info!("TtsEngine::Speak called (engine {:p})", self as *const _);

        if p_output_site.is_null() {
            log::error!("Speak: p_output_site is null");
            return E_POINTER;
        }

        let site = match ISpTTSEngineSite::from_raw_borrowed(&p_output_site) {
            Some(s) => s,
            None => {
                log::error!("Speak: failed to borrow ISpTTSEngineSite");
                return E_POINTER;
            }
        };

        let text = match collect_text(p_text_frag_list) {
            Some(t) => t,
            None => {
                log::debug!("Speak: empty text fragment list");
                return S_OK;
            }
        };
        log::info!("TtsEngine::Speak text=\"{}\"", text);

        let style_path = self.style_json_path.lock().unwrap().clone();
        let steps = *self.steps.lock().unwrap();
        let use_gpu = *self.use_gpu.lock().unwrap();
        let wants_word_boundary = site_wants_word_boundary(site);
        let synth_start = std::time::Instant::now();
        let (total_samples, metadata) = if wants_word_boundary {
            let output = match self
                .synthesizer
                .synthesize(&text, style_path.as_deref(), steps, use_gpu)
            {
                Ok(output) => output,
                Err(e) => {
                    log::error!("Speak: synthesis failed: {e}");
                    return E_FAIL;
                }
            };
            emit_word_boundary_events(site, &output.metadata);
            let mut aborted = false;
            if let Err(hr) = write_samples_to_site(site, &output.samples, &mut aborted) {
                log::error!("Speak: site.Write failed: {hr:?}");
                return hr;
            }
            if aborted {
                log::debug!("Speak aborted by caller");
                return S_OK;
            }
            (output.samples.len(), output.metadata)
        } else {
            let mut total_samples = 0usize;
            let mut aborted = false;
            let metadata = match self.synthesizer.synthesize_stream(
                &text,
                style_path.as_deref(),
                steps,
                use_gpu,
                &mut |samples_f32| {
                    total_samples += samples_f32.len();
                    write_samples_to_site(site, samples_f32, &mut aborted)
                        .map_err(|hr| format!("site.Write failed: {hr:?}").into())
                },
            ) {
                Ok(metadata) => metadata,
                Err(e) => {
                    if aborted {
                        log::debug!("Speak aborted by caller");
                        return S_OK;
                    }
                    log::error!("Speak: synthesis failed: {e}");
                    return E_FAIL;
                }
            };
            (total_samples, metadata)
        };
        let synth_elapsed = synth_start.elapsed();
        log::info!("Speak: synthesis took {synth_elapsed:?} ({} samples)", total_samples);
        log::info!(
            "Speak: normalized_len={} ipa_len={} words={}",
            metadata.normalized_text.chars().count(),
            metadata.combined_ipa.chars().count(),
            metadata.words.len()
        );

        log::info!("Speak: total elapsed {:?}", speak_start.elapsed());
        S_OK
    }

    unsafe fn get_output_format_inner(
        &self,
        _p_target_fmt_id: *const GUID,
        _p_target_wave_format_ex: *const u8,
        p_output_format_id: *mut GUID,
        pp_comem_output_wave_format_ex: *mut *mut u8,
    ) -> HRESULT {
        log::debug!("TtsEngine::GetOutputFormat called");

        log::debug!("  p_target_fmt_id={:?}", _p_target_fmt_id);
        if !_p_target_fmt_id.is_null() {
            log::debug!("  *p_target_fmt_id={:?}", *_p_target_fmt_id);
        }
        log::debug!("  p_target_wave_format_ex={:?}", _p_target_wave_format_ex);
        if !_p_target_wave_format_ex.is_null() {
            let target_wfx_bytes = std::slice::from_raw_parts(_p_target_wave_format_ex, 18);
            log::debug!("  target waveformat bytes: {:02x?}", target_wfx_bytes);
        }
        log::debug!("  p_output_format_id={:?}", p_output_format_id);
        log::debug!("  pp_comem_output_wave_format_ex={:?}", pp_comem_output_wave_format_ex);

        if p_output_format_id.is_null() || pp_comem_output_wave_format_ex.is_null() {
            log::debug!("  null pointer, returning E_POINTER");
            return E_POINTER;
        }

        log::debug!("  writing format GUID...");
        // Tell SAPI we produce SPDFID_WaveFormatEx.
        *p_output_format_id = SPDFID_WAVE_FORMAT_EX;
        log::debug!("  format GUID written");

        // Allocate a WAVEFORMATEX with CoTaskMemAlloc (SAPI will free it).
        // If SAPI suggested a format, echo it back; otherwise use our default.
        let size = std::mem::size_of::<WaveFormatEx>(); // 18 bytes
        log::debug!("  CoTaskMemAlloc({size})...");
        let ptr = windows::Win32::System::Com::CoTaskMemAlloc(size) as *mut u8;
        if ptr.is_null() {
            log::error!("  CoTaskMemAlloc returned null");
            return E_FAIL;
        }
        log::debug!("  allocated at {ptr:?}");

        // Always report our actual output format (44100 Hz 16-bit mono).
        // Do NOT echo the target format — SAPI may suggest a different rate
        // (e.g. 22050 for file streams) which causes pitch/speed mismatch.
        let wfx = WaveFormatEx {
            w_format_tag: WAVE_FORMAT_PCM,
            n_channels: NUM_CHANNELS,
            n_samples_per_sec: SAMPLE_RATE,
            n_avg_bytes_per_sec: AVG_BYTES_PER_SEC,
            n_block_align: BLOCK_ALIGN,
            w_bits_per_sample: BITS_PER_SAMPLE,
            cb_size: 0,
        };
        std::ptr::copy_nonoverlapping(
            &wfx as *const WaveFormatEx as *const u8,
            ptr,
            size,
        );

        *pp_comem_output_wave_format_ex = ptr;
        log::debug!("  GetOutputFormat returning S_OK");

        S_OK
    }
}

unsafe fn write_samples_to_site(
    site: &ISpTTSEngineSite,
    samples_f32: &[f32],
    aborted: &mut bool,
) -> Result<(), HRESULT> {
    let mut pcm = Vec::with_capacity(samples_f32.len() * 2);
    for &s in samples_f32 {
        let clamped = s.clamp(-1.0, 1.0);
        let val = (clamped * i16::MAX as f32) as i16;
        pcm.extend_from_slice(&val.to_le_bytes());
    }

    for chunk in pcm.chunks(CHUNK_BYTES) {
        let actions = site.GetActions();
        if (actions & SPVES_ABORT) != 0 {
            *aborted = true;
            break;
        }

        let mut written = 0u32;
        let hr = site.Write(chunk.as_ptr() as *const _, chunk.len() as u32, &mut written);
        if hr.is_err() {
            return Err(hr);
        }
        if written != chunk.len() as u32 {
            log::warn!(
                "Speak: site.Write wrote {} of {} bytes",
                written,
                chunk.len()
            );
        }
    }

    Ok(())
}

unsafe fn emit_word_boundary_events(site: &ISpTTSEngineSite, metadata: &SynthesisMetadata) {
    if !site_wants_word_boundary(site) {
        return;
    }

    let mut events = Vec::with_capacity(metadata.words.len());
    for word in &metadata.words {
        let Some(original_span) = word.original_span else {
            continue;
        };
        let (char_pos, word_len, adjusted) = utf16_span_metrics(
            &metadata.original_text,
            original_span.start,
            original_span.end,
        );
        if adjusted {
            log::debug!(
                "Speak: adjusted invalid span {:?} for original_text len={}",
                original_span,
                metadata.original_text.len()
            );
        }
        if word_len == 0 {
            continue;
        }

        let offset_bytes = ((word.start_sec.max(0.0) * AVG_BYTES_PER_SEC as f32).round() as u64)
            .min(u64::MAX);
        log::info!(
            "Speak: word event text={:?} stream_offset={} char_pos={} len={} original_span={:?}",
            word.text,
            offset_bytes,
            char_pos,
            word_len,
            original_span,
        );
        events.push(SPEVENT {
            eEventId: SPEI_WORD_BOUNDARY,
            elParamType: SPET_LPARAM_IS_UNDEFINED,
            ulStreamNum: 1,
            ullAudioStreamOffset: offset_bytes,
            // SpVoice automation surfaces OnWord(CharacterPosition, Length)
            // from lParam and wParam respectively for SPEI_WORD_BOUNDARY.
            wParam: word_len,
            lParam: char_pos as isize,
        });
    }

    if events.is_empty() {
        return;
    }

    let hr = site.AddEvents(events.as_ptr(), events.len() as u32);
    if hr.is_err() {
        log::warn!("Speak: AddEvents for word boundaries failed: {hr:?}");
    }
}

unsafe fn site_wants_word_boundary(site: &ISpTTSEngineSite) -> bool {
    let mut interest = 0u64;
    let hr = site.GetEventInterest(&mut interest);
    hr.is_ok() && (interest & SPFEI_WORD_BOUNDARY) == SPFEI_WORD_BOUNDARY
}

fn utf16_len(text: &str) -> usize {
    text.encode_utf16().count()
}

fn utf16_span_metrics(text: &str, start: usize, end: usize) -> (usize, usize, bool) {
    let start_b = clamp_to_char_boundary(text, start);
    let end_b = clamp_to_char_boundary(text, end.max(start_b));
    let adjusted = start_b != start || end_b != end;
    let char_pos = utf16_len(&text[..start_b]);
    let word_len = utf16_len(&text[start_b..end_b]);
    (char_pos, word_len, adjusted)
}

fn clamp_to_char_boundary(text: &str, idx: usize) -> usize {
    let mut i = idx.min(text.len());
    while i > 0 && !text.is_char_boundary(i) {
        i -= 1;
    }
    i
}

// ---------------------------------------------------------------------------
// Helper: walk SPVTEXTFRAG linked list
// ---------------------------------------------------------------------------

unsafe fn collect_text(mut frag: *const SPVTEXTFRAG) -> Option<String> {
    let mut buf = String::new();
    let mut count = 0usize;
    while !frag.is_null() {
        if count > 256 {
            log::warn!("collect_text: too many fragments, aborting traversal");
            break;
        }
        let f = &*frag;

        if !f.pTextStart.is_null() && f.ulTextLen > 0 {
            let slice = std::slice::from_raw_parts(f.pTextStart, f.ulTextLen as usize);
            if let Ok(s) = String::from_utf16(slice) {
                if !buf.is_empty() {
                    buf.push(' ');
                }
                buf.push_str(&s);
            }
        }
        frag = f.pNext;
        count += 1;
    }
    if buf.is_empty() {
        None
    } else {
        Some(buf)
    }
}

// ---------------------------------------------------------------------------
// Helper: read StyleFile from a SAPI voice token
// ---------------------------------------------------------------------------

/// Extract the token name from a COM ISpObjectToken pointer by calling GetId.
///
/// ISpObjectToken::GetId returns a wide string like:
///   `HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\LightBlueHebrewMale_8CPU`
///
/// We extract the last path component as the token name.
unsafe fn get_token_name_from_com(ptoken: *mut core::ffi::c_void) -> Option<String> {
    // ISpObjectToken vtable layout:
    //   IUnknown (3 methods) + ISpDataKey (12 methods) + SetId(15) + GetId(16)
    // GetId signature: fn(this, ppszCoMemTokenId: *mut *mut u16) -> HRESULT
    type GetIdFn = unsafe extern "system" fn(*mut core::ffi::c_void, *mut *mut u16) -> HRESULT;

    let vtbl_ptr = *(ptoken as *const *const usize);
    let get_id: GetIdFn = std::mem::transmute(*vtbl_ptr.add(16));

    let mut id_ptr: *mut u16 = std::ptr::null_mut();
    let hr = get_id(ptoken, &mut id_ptr);
    if hr.is_err() || id_ptr.is_null() {
        log::warn!("GetId failed: {hr:?}");
        return None;
    }

    // Read the wide string
    let mut len = 0;
    while *id_ptr.add(len) != 0 {
        len += 1;
    }
    let id_str = String::from_utf16_lossy(std::slice::from_raw_parts(id_ptr, len));

    // Free with CoTaskMemFree
    windows::Win32::System::Com::CoTaskMemFree(Some(id_ptr as *const _));

    log::debug!("Token ID: {id_str}");

    // Extract last component: "...\Tokens\LightBlueHebrewMale_8CPU" -> "LightBlueHebrewMale_8CPU"
    id_str.rsplit('\\').next().map(|s| s.to_string())
}

/// Voice settings read from the SAPI token registry.
struct VoiceTokenSettings {
    style_path: Option<String>,
    steps: Option<u32>,
    use_gpu: bool,
}

/// Read a REG_SZ value from an open registry key.
unsafe fn reg_read_sz(hkey: windows::Win32::System::Registry::HKEY, name: &str) -> Option<String> {
    use windows::Win32::System::Registry::{RegQueryValueExW, REG_SZ};

    let value_name: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    let mut buf = vec![0u8; 1024];
    let mut buf_len = buf.len() as u32;
    let mut reg_type = REG_SZ;
    let status = RegQueryValueExW(
        hkey,
        windows_core::PCWSTR(value_name.as_ptr()),
        None,
        Some(&mut reg_type),
        Some(buf.as_mut_ptr()),
        Some(&mut buf_len),
    );
    if status.is_err() || buf_len == 0 {
        return None;
    }
    let wide_slice = std::slice::from_raw_parts(buf.as_ptr() as *const u16, (buf_len as usize) / 2);
    let end = wide_slice.iter().position(|&c| c == 0).unwrap_or(wide_slice.len());
    let val = String::from_utf16_lossy(&wide_slice[..end]);
    if val.is_empty() { None } else { Some(val) }
}

/// Read StyleFile, Steps, and Device from the SAPI voice token registry.
unsafe fn read_token_settings(token_name: &str) -> VoiceTokenSettings {
    use windows::Win32::System::Registry::{
        RegOpenKeyExW, RegCloseKey, HKEY, HKEY_LOCAL_MACHINE, KEY_READ,
    };

    let mut settings = VoiceTokenSettings {
        style_path: None,
        steps: None,
        use_gpu: false,
    };

    let subkeys = [
        format!("SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\{}", token_name),
        format!("SOFTWARE\\Microsoft\\Speech_OneCore\\Voices\\Tokens\\{}", token_name),
    ];

    for subkey in &subkeys {
        let subkey_w: Vec<u16> = subkey.encode_utf16().chain(std::iter::once(0)).collect();
        let mut hkey = HKEY::default();
        let status = RegOpenKeyExW(
            HKEY_LOCAL_MACHINE,
            windows_core::PCWSTR(subkey_w.as_ptr()),
            0,
            KEY_READ,
            &mut hkey,
        );
        if status.is_err() {
            continue;
        }

        // StyleFile
        if settings.style_path.is_none() {
            if let Some(style_file) = reg_read_sz(hkey, "StyleFile") {
                if let Some(dll_dir) = crate::get_dll_dir() {
                    let models_dir = std::path::PathBuf::from(&dll_dir).join("models");
                    let full_path = models_dir.join(&style_file);
                    if full_path.exists() {
                        settings.style_path = Some(full_path.to_string_lossy().into_owned());
                    } else {
                        log::warn!("Style file not found: {}", full_path.display());
                    }
                }
            }
        }

        // Steps
        if settings.steps.is_none() {
            if let Some(steps_str) = reg_read_sz(hkey, "Steps") {
                settings.steps = steps_str.parse::<u32>().ok();
            }
        }

        // Device
        if let Some(device) = reg_read_sz(hkey, "Device") {
            settings.use_gpu = device.eq_ignore_ascii_case("GPU");
        }

        let _ = RegCloseKey(hkey);

        // If we found settings, no need to check OneCore
        if settings.style_path.is_some() {
            break;
        }
    }

    settings
}
