//! Main TTS inference pipeline — port of `hebrew_inference_helper.py`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Once;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use super::config::{TTSConfig, LATENT_DIM, CHUNK_COMPRESS_FACTOR, NORMALIZER_SCALE, SAMPLE_RATE, HOP_LENGTH};
use super::npz::load_npz;
use super::style::StyleJson;

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

// ---------------------------------------------------------------------------
// ORT initialisation (once per process)
// ---------------------------------------------------------------------------

static ORT_INIT: Once = Once::new();
static mut ORT_INIT_ERROR: Option<String> = None;

/// Find CUDA toolkit bin directory and add it to PATH.
/// The CUDA toolkit installs to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\`
/// but this may not be on PATH, causing ORT CUDA EP to silently fail.
fn add_cuda_to_dll_search_path() {
    // Check if cudart64_12.dll is already on PATH
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in path_var.split(';') {
            if std::path::Path::new(dir).join("cudart64_12.dll").exists() {
                log::info!("CUDA runtime already on PATH: {}", dir);
                return;
            }
        }
    }

    let cuda_base = std::path::Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
    if !cuda_base.exists() {
        return;
    }

    // Find the newest CUDA 12.x version directory
    let mut best_dir: Option<std::path::PathBuf> = None;
    if let Ok(versions) = std::fs::read_dir(cuda_base) {
        for ver in versions.flatten() {
            let name = ver.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("v12.") {
                let bin_dir = ver.path().join("bin");
                if bin_dir.join("cudart64_12.dll").exists() {
                    best_dir = Some(bin_dir);
                }
            }
        }
    }

    if let Some(dir) = best_dir {
        log::info!("Adding CUDA toolkit bin to PATH: {}", dir.display());
        if let Ok(current_path) = std::env::var("PATH") {
            std::env::set_var("PATH", format!("{};{}", dir.display(), current_path));
        }
    }
}

/// Find cuDNN 9 DLL directory and add it to PATH.
/// This is needed because the NVIDIA cuDNN exe installer puts DLLs in
/// `C:\Program Files\NVIDIA\CUDNN\v9.x\bin\12.x\` which is not on PATH.
fn add_cudnn_to_dll_search_path() {
    let cudnn_base = std::path::Path::new(r"C:\Program Files\NVIDIA\CUDNN");
    if !cudnn_base.exists() {
        return;
    }

    // Find the newest cuDNN version dir containing cudnn64_9.dll
    let mut best_dir: Option<std::path::PathBuf> = None;
    if let Ok(versions) = std::fs::read_dir(cudnn_base) {
        for ver in versions.flatten() {
            if let Some(dir) = find_cudnn_dll_dir(&ver.path()) {
                best_dir = Some(dir);
            }
        }
    }

    if let Some(dir) = best_dir {
        log::info!("Adding cuDNN DLL path to PATH: {}", dir.display());
        // Prepend to PATH so dependent DLLs can be found
        if let Ok(current_path) = std::env::var("PATH") {
            std::env::set_var("PATH", format!("{};{}", dir.display(), current_path));
        }
    }
}

/// Recursively find a directory containing cudnn64_9.dll under the given path.
fn find_cudnn_dll_dir(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.file_name().and_then(|f| f.to_str()) == Some("cudnn64_9.dll") {
                return Some(dir.to_path_buf());
            }
            if path.is_dir() {
                if let Some(found) = find_cudnn_dll_dir(&path) {
                    return Some(found);
                }
            }
        }
    }
    None
}

/// Find OpenVINO runtime DLLs and add to PATH.
/// Checks common install locations and the exe's own directory.
fn add_openvino_to_dll_search_path() {
    // Check if openvino.dll is already findable
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in path_var.split(';') {
            if std::path::Path::new(dir).join("openvino.dll").exists() {
                log::info!("OpenVINO already on PATH: {}", dir);
                return;
            }
        }
    }

    // Check exe directory (for deployed installs)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if dir.join("openvino.dll").exists() {
                log::info!("OpenVINO found next to exe: {}", dir.display());
                if let Ok(current_path) = std::env::var("PATH") {
                    std::env::set_var("PATH", format!("{};{}", dir.display(), current_path));
                }
                return;
            }
        }
    }

    // Check common OpenVINO install locations
    let search_bases = [
        std::path::PathBuf::from(r"C:\Program Files (x86)\Intel\openvino"),
        std::path::PathBuf::from(r"C:\Program Files\Intel\openvino"),
        std::path::PathBuf::from(r"C:\Intel\openvino"),
    ];
    for base in &search_bases {
        // Try exact path and versioned paths (openvino_2025.3.0, etc.)
        let candidates: Vec<std::path::PathBuf> = std::iter::once(base.clone())
            .chain(
                base.parent()
                    .and_then(|p| std::fs::read_dir(p).ok())
                    .into_iter()
                    .flatten()
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.file_name().and_then(|f| f.to_str()).map_or(false, |n| n.starts_with("openvino")))
            )
            .collect();

        for candidate in candidates {
            let bin_dir = candidate.join("runtime").join("bin").join("intel64").join("Release");
            if bin_dir.join("openvino.dll").exists() {
                log::info!("Adding OpenVINO DLL path to PATH: {}", bin_dir.display());
                if let Ok(current_path) = std::env::var("PATH") {
                    std::env::set_var("PATH", format!("{};{}", bin_dir.display(), current_path));
                }
                return;
            }
        }
    }
}

/// Pre-load onnxruntime.dll using LoadLibraryExW with LOAD_WITH_ALTERED_SEARCH_PATH.
///
/// This ensures that all implicit dependencies (vcruntime140.dll, msvcp140.dll, etc.)
/// are resolved from the DLL's own directory first, rather than from whatever is on
/// PATH. This prevents conflicts when the host process (e.g. Python with PyQt5) has
/// older/incompatible versions of these runtime DLLs on its search path.
///
/// The DLL remains loaded (reference count increases). When `ort` later calls
/// `LoadLibraryExW(path, NULL, 0)`, Windows returns the already-loaded module.
fn preload_ort_dll(dll_path: &std::path::Path) {
    if !dll_path.is_absolute() || !dll_path.exists() {
        return;
    }

    use windows::core::HSTRING;
    use windows::Win32::System::LibraryLoader::LoadLibraryExW;
    use windows::Win32::System::LibraryLoader::LOAD_LIBRARY_FLAGS;

    // First, force-load the correct VC++ runtime DLLs from system32.
    // Some host processes (e.g. Python with PyQt5) load OLDER/bundled versions
    // of msvcp140.dll from their own directories. If those are already loaded,
    // onnxruntime.dll's DllMain will fail (error 1114) because of version mismatch.
    // Loading from system32 explicitly ensures we have the system version.
    const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x00000800;
    let sys_flags = LOAD_LIBRARY_FLAGS(LOAD_LIBRARY_SEARCH_SYSTEM32);
    for rt_dll in ["msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll"] {
        let h_rt = HSTRING::from(rt_dll);
        let _ = unsafe { LoadLibraryExW(&h_rt, None, sys_flags) };
    }

    // Now load onnxruntime.dll with LOAD_WITH_ALTERED_SEARCH_PATH so its
    // dependencies are resolved from ITS directory first.
    const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x00000008;
    let h = HSTRING::from(dll_path.as_os_str());
    let flags = LOAD_LIBRARY_FLAGS(LOAD_WITH_ALTERED_SEARCH_PATH);
    let result = unsafe { LoadLibraryExW(&h, None, flags) };
    match result {
        Ok(handle) => {
            log::info!(
                "Pre-loaded onnxruntime.dll with LOAD_WITH_ALTERED_SEARCH_PATH (handle={:?})",
                handle
            );
            // Intentionally leak the handle — we want the DLL to stay loaded.
        }
        Err(e) => {
            let win_err = unsafe { windows::Win32::Foundation::GetLastError() };
            log::error!(
                "Failed to pre-load onnxruntime.dll: {} (Win32 error {})",
                e, win_err.0
            );
        }
    }
}

/// Log diagnostic info about the ORT DLL environment (file existence, PATH).
fn diagnose_dll_load(dll_path: &std::path::Path) {
    let dll_str = dll_path.to_string_lossy();

    // Check the file exists and its size
    match std::fs::metadata(dll_path) {
        Ok(meta) => log::info!("DLL diagnostic: {} size={} bytes", dll_str, meta.len()),
        Err(e) => {
            log::error!("DLL diagnostic: cannot stat {}: {e}", dll_str);
            return;
        }
    }

    // Log PATH for debugging
    if let Ok(path_var) = std::env::var("PATH") {
        let dirs: Vec<&str> = path_var.split(';').take(20).collect();
        log::info!("DLL diagnostic: PATH (first 20 dirs): {:?}", dirs);
    }

    // Check companion DLLs exist (don't load them — just file checks)
    if let Some(parent) = dll_path.parent() {
        for dep in ["onnxruntime_providers_shared.dll", "DirectML.dll"] {
            let dep_path = parent.join(dep);
            if dep_path.exists() {
                log::info!("DLL diagnostic: {} found", dep);
            } else {
                log::info!("DLL diagnostic: {} not present", dep);
            }
        }
    }
}

fn init_ort() -> Result<(), BoxErr> {
    init_ort_with_dir(None)
}

fn init_ort_with_dir(dll_dir: Option<&Path>) -> Result<(), BoxErr> {
    ORT_INIT.call_once(|| {
        // Add CUDA toolkit and cuDNN DLL directories to PATH so ORT CUDA EP can find them.
        add_cuda_to_dll_search_path();
        add_cudnn_to_dll_search_path();
        // Add OpenVINO runtime DLLs to search path.
        add_openvino_to_dll_search_path();

        // Prefer loading from our own DLL directory to avoid picking up
        // a wrong version (e.g. Python's onnxruntime).
        // Search order: explicit dir, exe dir, then fallback to system.
        let dll_name = "onnxruntime.dll";
        let candidates: Vec<std::path::PathBuf> = [
            dll_dir.map(|d| d.join(dll_name)),
            std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join(dll_name))),
        ]
        .into_iter()
        .flatten()
        .filter(|p| p.exists())
        .collect();

        let lib_path = candidates.into_iter().next()
            .unwrap_or_else(|| std::path::PathBuf::from("onnxruntime"));
        log::info!("Loading ORT from: {}", lib_path.display());

        // Log diagnostic info (file existence, PATH, etc.)
        diagnose_dll_load(&lib_path);

        // Pre-load onnxruntime.dll with LOAD_WITH_ALTERED_SEARCH_PATH so its
        // dependencies are resolved from ITS directory, not from wherever the
        // host process has on PATH (e.g. PyQt5/Qt5/bin ships older vcruntime).
        preload_ort_dll(&lib_path);

        // Wrap ort::init_from in catch_unwind because the ort crate internally
        // panics (via expect()) when LoadLibraryExW fails for onnxruntime.dll.
        // Without this, the panic would poison the ORT_INIT Once and cause
        // cascading panics in all subsequent calls, ultimately crashing the
        // host application (e.g. via "panic in a function that cannot unwind"
        // in extern "system" COM callbacks).
        let init_result = std::panic::catch_unwind(|| {
            ort::init_from(lib_path.to_string_lossy().as_ref())
        });

        match init_result {
            Ok(Ok(builder)) => {
                let _ = builder.with_name("lightblue").commit();
                let available = query_ort_available_providers();
                log::info!("ORT available providers: {:?}", available);
            }
            Ok(Err(e)) => unsafe {
                let msg = format!("ORT init failed: {}", e);
                log::error!("{msg}");
                ORT_INIT_ERROR = Some(msg);
            },
            Err(panic_payload) => unsafe {
                let msg = if let Some(s) = panic_payload.downcast_ref::<String>() {
                    format!("ORT init panicked: {s}")
                } else if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    format!("ORT init panicked: {s}")
                } else {
                    "ORT init panicked (unknown payload)".to_string()
                };
                log::error!("{msg}");
                ORT_INIT_ERROR = Some(msg);
            },
        }
    });
    unsafe {
        if let Some(ref err) = ORT_INIT_ERROR {
            return Err(err.clone().into());
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Phoneme vocabulary
// ---------------------------------------------------------------------------

/// Build the CHAR_TO_ID map.
///
/// vocab_phonemes sorted by Unicode codepoint, then CHAR_TO_ID = {c: i+1}.
/// Unknown chars map to 0.
fn build_char_to_id() -> HashMap<char, i64> {
    // The 36 vocab phonemes sorted by Unicode codepoint:
    let sorted: &[char] = &[
        ' ', '!', '"', '\'', ',', '-', '.', ':', '?',
        'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k',
        'l', 'm', 'n', 'o', 'p', 's', 't', 'u', 'v',
        'w', 'z',
        '\u{0261}', // ɡ
        '\u{0281}', // ʁ
        '\u{0283}', // ʃ
        '\u{0292}', // ʒ
        '\u{0294}', // ʔ
        '\u{02C8}', // ˈ
        '\u{03C7}', // χ
    ];
    let mut map = HashMap::new();
    for (i, &c) in sorted.iter().enumerate() {
        map.insert(c, (i + 1) as i64);
    }
    map
}

/// Normalize text before phoneme-to-ID conversion.
fn normalize_text(text: &str) -> String {
    text.replace('g', "\u{0261}") // g -> ɡ
        .replace('r', "\u{0281}") // r -> ʁ
}

/// Convert an IPA phoneme string to a vector of integer IDs.
fn text_to_indices(text: &str, char_to_id: &HashMap<char, i64>) -> Vec<i64> {
    let norm = normalize_text(text);
    norm.chars()
        .map(|c| *char_to_id.get(&c).unwrap_or(&0))
        .collect()
}

// ---------------------------------------------------------------------------
// Text chunking
// ---------------------------------------------------------------------------

/// Split text at sentence boundaries, ensuring each chunk is at most
/// `max_len` **characters** (not bytes).
fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
    if text.chars().count() <= max_len {
        return vec![text.to_string()];
    }

    let sentence_ends: &[char] = &['.', '!', '?'];
    let mut chunks = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        let char_count = remaining.chars().count();
        if char_count <= max_len {
            chunks.push(remaining.to_string());
            break;
        }

        // Find the byte offset of the max_len-th character
        let byte_limit = remaining
            .char_indices()
            .nth(max_len)
            .map(|(idx, _)| idx)
            .unwrap_or(remaining.len());

        let search_region = &remaining[..byte_limit];

        // Try to find a sentence boundary within the region
        let split_pos = search_region
            .rfind(sentence_ends)
            .map(|p| p + remaining[p..].chars().next().unwrap().len_utf8())
            .unwrap_or_else(|| {
                // Fall back to space
                search_region
                    .rfind(' ')
                    .map(|p| p + 1)
                    .unwrap_or(byte_limit)
            });

        let (chunk, rest) = remaining.split_at(split_pos);
        chunks.push(chunk.trim().to_string());
        remaining = rest.trim_start();
    }

    chunks.retain(|c| !c.is_empty());
    chunks
}

// ---------------------------------------------------------------------------
// ndarray tensor helpers
// ---------------------------------------------------------------------------

/// Create an ort Tensor from flat data + shape.
fn make_f32_tensor(data: &[f32], shape: &[usize]) -> Result<Tensor<f32>, BoxErr> {
    Ok(Tensor::from_array((shape.to_vec(), data.to_vec()))?)
}

fn make_i64_tensor(data: &[i64], shape: &[usize]) -> Result<Tensor<i64>, BoxErr> {
    Ok(Tensor::from_array((shape.to_vec(), data.to_vec()))?)
}

// ---------------------------------------------------------------------------
// HebrewTTS
// ---------------------------------------------------------------------------

/// The main TTS engine.
pub struct HebrewTTS {
    // ONNX sessions
    text_encoder: Session,
    backbone: Session,
    vocoder: Session,
    length_pred_style: Session,

    // Stats for normalization — stored as flat [CC] vectors
    stat_mean: Vec<f32>,
    stat_std: Vec<f32>,

    // Unconditional embeddings for CFG
    u_text: Vec<f32>,
    u_text_shape: Vec<usize>,
    u_ref: Vec<f32>,
    u_ref_shape: Vec<usize>,
    u_keys: Vec<f32>,
    u_keys_shape: Vec<usize>,
    cond_keys: Vec<f32>,
    cond_keys_shape: Vec<usize>,

    // Default style (optional)
    default_style: Option<StyleJson>,

    // Model config
    latent_dim: usize,
    chunk_compress_factor: usize,
    normalizer_scale: f32,
    sample_rate: u32,
    hop_length: usize,

    // Runtime config
    cfg_scale: f32,
    steps: u32,
    text_chunk_len: usize,
    silence_sec: f32,
    fade_duration: f32,

    // Vocab
    char_to_id: HashMap<char, i64>,
}

impl HebrewTTS {
    /// Create a new TTS engine from the given configuration.
    pub fn new(config: TTSConfig) -> Result<Self, BoxErr> {
        // Derive the DLL directory (parent of models → install dir)
        let dll_dir = config.onnx_dir.parent();
        init_ort_with_dir(dll_dir)?;

        // Model constants
        let latent_dim = LATENT_DIM;
        let chunk_compress_factor = CHUNK_COMPRESS_FACTOR;
        let normalizer_scale = NORMALIZER_SCALE;
        let sample_rate = SAMPLE_RATE;
        let hop_length = HOP_LENGTH;

        let threads = config.threads;
        let onnx_dir = &config.onnx_dir;

        // Load ONNX sessions in parallel
        let t_all = std::time::Instant::now();
        let onnx_dir_owned = onnx_dir.to_path_buf();
        let (text_encoder, backbone, vocoder, length_pred_style) = {
            let dir = onnx_dir_owned.clone();
            let t = threads;
            std::thread::scope(|s| {
                let h1 = s.spawn(move || {
                    let t0 = std::time::Instant::now();
                    let r = load_session_cpu(&dir.join("text_encoder.onnx"), t);
                    log::info!("  text_encoder loaded in {:?}", t0.elapsed());
                    r
                });
                let dir = onnx_dir_owned.clone();
                let h2 = s.spawn(move || {
                    let t0 = std::time::Instant::now();
                    let r = load_session(&dir.join("backbone_keys.onnx"), t);
                    log::info!("  backbone loaded in {:?}", t0.elapsed());
                    r
                });
                let dir = onnx_dir_owned.clone();
                let h3 = s.spawn(move || {
                    let t0 = std::time::Instant::now();
                    let r = load_session(&dir.join("vocoder.onnx"), t);
                    log::info!("  vocoder loaded in {:?}", t0.elapsed());
                    r
                });
                let dir = onnx_dir_owned.clone();
                let h4 = s.spawn(move || {
                    let t0 = std::time::Instant::now();
                    let r = load_session_cpu(&dir.join("length_pred_style.onnx"), t);
                    log::info!("  length_pred_style loaded in {:?}", t0.elapsed());
                    r
                });
                let join_result = |r: std::thread::Result<Result<Session, BoxErr>>| -> Result<Session, BoxErr> {
                    match r {
                        Ok(inner) => inner,
                        Err(panic) => {
                            let msg = if let Some(s) = panic.downcast_ref::<String>() {
                                format!("ONNX session loading panicked: {s}")
                            } else if let Some(s) = panic.downcast_ref::<&str>() {
                                format!("ONNX session loading panicked: {s}")
                            } else {
                                "ONNX session loading panicked".to_string()
                            };
                            log::error!("{msg}");
                            Err(msg.into())
                        }
                    }
                };
                (
                    join_result(h1.join()),
                    join_result(h2.join()),
                    join_result(h3.join()),
                    join_result(h4.join()),
                )
            })
        };
        let text_encoder = text_encoder?;
        let backbone = backbone?;
        let vocoder = vocoder?;
        let length_pred_style = length_pred_style?;
        log::info!("All ONNX sessions loaded in {:?}", t_all.elapsed());

        // Load stats
        let stats = load_npz(&onnx_dir.join("stats.npz"))?;
        let stat_mean_arr = stats.get("mean").ok_or("stats.npz missing 'mean'")?;
        let stat_std_arr = stats.get("std").ok_or("stats.npz missing 'std'")?;
        // Override normalizer_scale from stats.npz if present (matches Python behavior)
        let normalizer_scale = if let Some(ns_arr) = stats.get("normalizer_scale") {
            let ns = ns_arr.data[0];
            log::info!("Using normalizer_scale={ns} from stats.npz");
            ns
        } else {
            normalizer_scale
        };

        // Load unconditional embeddings
        let uncond = load_npz(&onnx_dir.join("uncond.npz"))?;
        let u_text_arr = uncond.get("u_text").ok_or("uncond.npz missing 'u_text'")?;
        let u_ref_arr = uncond.get("u_ref").ok_or("uncond.npz missing 'u_ref'")?;
        let u_keys_arr = uncond.get("u_keys").ok_or("uncond.npz missing 'u_keys'")?;
        let cond_keys_arr = uncond
            .get("cond_keys")
            .ok_or("uncond.npz missing 'cond_keys'")?;

        // Load default style
        let default_style = match &config.default_style_json {
            Some(p) => Some(StyleJson::load(p)?),
            None => None,
        };

        Ok(Self {
            text_encoder,
            backbone,
            vocoder,
            length_pred_style,
            stat_mean: stat_mean_arr.data.clone(),
            stat_std: stat_std_arr.data.clone(),
            u_text: u_text_arr.data.clone(),
            u_text_shape: u_text_arr.shape.clone(),
            u_ref: u_ref_arr.data.clone(),
            u_ref_shape: u_ref_arr.shape.clone(),
            u_keys: u_keys_arr.data.clone(),
            u_keys_shape: u_keys_arr.shape.clone(),
            cond_keys: cond_keys_arr.data.clone(),
            cond_keys_shape: cond_keys_arr.shape.clone(),
            default_style,
            latent_dim,
            chunk_compress_factor,
            normalizer_scale,
            sample_rate,
            hop_length,
            cfg_scale: config.cfg_scale,
            steps: config.steps,
            text_chunk_len: config.text_chunk_len,
            silence_sec: config.silence_sec,
            fade_duration: config.fade_duration,
            char_to_id: build_char_to_id(),
        })
    }

    /// Get the audio sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Run full inference: text -> f32 PCM audio.
    ///
    /// `steps_override` — if `Some(n)`, use `n` diffusion steps instead of the
    /// config default.
    pub fn infer(&mut self, text: &str, style_json_path: Option<&str>, steps_override: Option<u32>) -> Result<Vec<f32>, BoxErr> {
        let style = match style_json_path {
            Some(p) => StyleJson::load(Path::new(p))?,
            None => self
                .default_style
                .clone()
                .ok_or("No style provided and no default style loaded")?,
        };

        let effective_steps = steps_override.unwrap_or(self.steps);

        let chunks = chunk_text(text, self.text_chunk_len);
        let silence_samples = (self.silence_sec * self.sample_rate as f32) as usize;
        let silence = vec![0.0f32; silence_samples];

        let mut all_audio = Vec::new();

        for (i, chunk) in chunks.iter().enumerate() {
            let audio = self.infer_chunk(chunk, &style, effective_steps)?;
            all_audio.extend_from_slice(&audio);
            if i + 1 < chunks.len() {
                all_audio.extend_from_slice(&silence);
            }
        }

        // Apply fade in/out
        apply_fade(&mut all_audio, self.fade_duration, self.sample_rate);

        Ok(all_audio)
    }

    /// Stream inference: returns audio chunks via a callback.
    /// Each chunk is a `Vec<f32>` of PCM samples.
    ///
    /// `do_fade_in` / `do_fade_out` control whether the overall first/last
    /// audio chunks get a fade applied.  When the caller is streaming multiple
    /// independent IPA segments (e.g. sentences) it should set these so that
    /// only the very first segment fades in and the very last fades out.
    pub fn infer_stream(
        &mut self,
        text: &str,
        style_json_path: Option<&str>,
        steps_override: Option<u32>,
        do_fade_in: bool,
        do_fade_out: bool,
        mut callback: impl FnMut(Vec<f32>) -> Result<(), BoxErr>,
    ) -> Result<(), BoxErr> {
        let style = match style_json_path {
            Some(p) => StyleJson::load(Path::new(p))?,
            None => self
                .default_style
                .clone()
                .ok_or("No style provided and no default style loaded")?,
        };
        let effective_steps = steps_override.unwrap_or(self.steps);

        let chunks = chunk_text(text, self.text_chunk_len);
        let silence_samples = (self.silence_sec * self.sample_rate as f32) as usize;
        let silence = vec![0.0f32; silence_samples];

        for (i, chunk) in chunks.iter().enumerate() {
            let mut audio = self.infer_chunk(chunk, &style, effective_steps)?;

            if do_fade_in && i == 0 {
                apply_fade_in(&mut audio, self.fade_duration, self.sample_rate);
            }
            if do_fade_out && i + 1 == chunks.len() {
                apply_fade_out(&mut audio, self.fade_duration, self.sample_rate);
            }

            callback(audio)?;
            if i + 1 < chunks.len() {
                callback(silence.clone())?;
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internal: single-chunk inference
    // -----------------------------------------------------------------------

    fn infer_chunk(&mut self, text: &str, style: &StyleJson, steps: u32) -> Result<Vec<f32>, BoxErr> {
        let chunk_start = std::time::Instant::now();

        // 1. Text to IDs
        let text_ids = text_to_indices(text, &self.char_to_id);
        let t_text = text_ids.len();

        // 2. Extract style tensors
        let style_ttl = &style.style_ttl;
        let style_keys_st = &style.style_keys;
        let style_dp = &style.style_dp;

        // 3. Predict duration
        let t0 = std::time::Instant::now();
        let duration = self.predict_duration(&text_ids, t_text, &style_dp.data, &style_dp.dims)?;
        log::info!("  length_pred: {:?} (predicted duration={duration:.1})", t0.elapsed());

        // Clamp duration (ceil + 3 extra frames so speech finishes before end trim)
        let max_dur = (t_text * 3 + 20).min(600).min(800);
        let duration_clamped = (duration.ceil() as usize + 3).max(10).min(max_dur);
        let t_lat = duration_clamped;

        // 4. Encode text
        let t0 = std::time::Instant::now();
        let text_emb = self.encode_text(
            &text_ids,
            t_text,
            &style_ttl.data,
            &style_ttl.dims,
        )?;
        log::info!("  text_encoder: {:?}", t0.elapsed());

        // 5. Flow matching diffusion loop
        let cc = self.latent_dim * self.chunk_compress_factor;
        let t0 = std::time::Instant::now();
        let z_pred = self.flow_matching(
            &text_emb,
            t_text,
            t_lat,
            cc,
            &style_ttl.data,
            &style_ttl.dims,
            steps,
        )?;
        let passes_per_step = if (self.cfg_scale - 1.0).abs() > 1e-6 { 2 } else { 1 };
        log::info!("  flow_matching ({} steps x {} backbone): {:?}", steps, passes_per_step, t0.elapsed());

        // 6. Denormalize
        let z_pred_unnorm = self.denormalize(&z_pred, cc, t_lat);

        // 7. Reshape for vocoder
        let z_dec_in = self.reshape_for_vocoder(&z_pred_unnorm, t_lat);

        // 8. Run vocoder
        let t_dec = t_lat * self.chunk_compress_factor;
        let t0 = std::time::Instant::now();
        let wav = self.run_vocoder(&z_dec_in, t_dec)?;
        log::info!("  vocoder: {:?}", t0.elapsed());
        log::info!("  infer_chunk total: {:?} (t_text={}, t_lat={})", chunk_start.elapsed(), t_text, t_lat);

        // 9. Trim: remove hop_length*ccf from each end (vocoder boundary artifacts).
        let trim = self.hop_length * self.chunk_compress_factor;
        let start = trim.min(wav.len());
        let end = wav.len().saturating_sub(trim);
        if start >= end {
            return Ok(Vec::new());
        }

        Ok(wav[start..end].to_vec())
    }

    /// `z_ref_norm = ((z_ref - mean) / std) * normalizer_scale`
    #[allow(dead_code)]
    fn normalize_z_ref(&self, z_ref: &[f32]) -> Vec<f32> {
        let cc = self.stat_mean.len();
        let t_ref = z_ref.len() / cc;
        let mut out = vec![0.0f32; z_ref.len()];
        for c in 0..cc {
            let mean = self.stat_mean[c];
            let std_val = self.stat_std[c];
            for t in 0..t_ref {
                let idx = c * t_ref + t;
                out[idx] = ((z_ref[idx] - mean) / std_val) * self.normalizer_scale;
            }
        }
        out
    }

    /// `z_pred_unnorm = (z_pred / normalizer_scale) * std + mean`
    fn denormalize(&self, z_pred: &[f32], cc: usize, t_lat: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; z_pred.len()];
        for c in 0..cc {
            let mean = self.stat_mean[c];
            let std_val = self.stat_std[c];
            for t in 0..t_lat {
                let idx = c * t_lat + t;
                out[idx] = (z_pred[idx] / self.normalizer_scale) * std_val + mean;
            }
        }
        out
    }

    /// Reshape `[1, CC, T_lat]` -> `[1, latent_dim, T_lat * chunk_compress_factor]`
    /// via intermediate `[1, latent_dim, chunk_compress_factor, T_lat]`
    /// transposed to `[1, latent_dim, T_lat, chunk_compress_factor]`.
    fn reshape_for_vocoder(&self, data: &[f32], t_lat: usize) -> Vec<f32> {
        let ld = self.latent_dim;
        let ccf = self.chunk_compress_factor;
        let t_out = t_lat * ccf;
        let mut out = vec![0.0f32; ld * t_out];

        // data layout: [ld*ccf, T_lat] row-major.
        // Interpret as [ld, ccf, T_lat], then transpose last two dims -> [ld, T_lat, ccf],
        // then flatten -> [ld, T_lat*ccf].
        for d in 0..ld {
            for cf in 0..ccf {
                for t in 0..t_lat {
                    let src_idx = (d * ccf + cf) * t_lat + t;
                    let dst_idx = d * t_out + t * ccf + cf;
                    out[dst_idx] = data[src_idx];
                }
            }
        }
        out
    }

    fn predict_duration(
        &mut self,
        text_ids: &[i64],
        t_text: usize,
        style_dp_data: &[f32],
        style_dp_dims: &[usize],
    ) -> Result<f32, BoxErr> {
        let text_ids_tensor = make_i64_tensor(text_ids, &[1, t_text])?;
        let text_mask_data = vec![1.0f32; t_text];
        let text_mask_tensor = make_f32_tensor(&text_mask_data, &[1, 1, t_text])?;
        let style_dp_tensor = make_f32_tensor(style_dp_data, style_dp_dims)?;

        let outputs = self.length_pred_style.run(ort::inputs![
            "text_ids" => text_ids_tensor,
            "style_dp" => style_dp_tensor,
            "text_mask" => text_mask_tensor
        ])?;

        let output_val = &outputs["duration"];
        let (_shape, data) = output_val.try_extract_tensor::<f32>()?;
        let duration = data[0];
        Ok(duration)
    }

    fn encode_text(
        &mut self,
        text_ids: &[i64],
        t_text: usize,
        style_ttl_data: &[f32],
        style_ttl_dims: &[usize],
    ) -> Result<Vec<f32>, BoxErr> {
        let text_ids_tensor = make_i64_tensor(text_ids, &[1, t_text])?;
        let text_mask_data = vec![1.0f32; t_text];
        let text_mask_tensor = make_f32_tensor(&text_mask_data, &[1, 1, t_text])?;
        let style_ttl_tensor = make_f32_tensor(style_ttl_data, style_ttl_dims)?;

        let outputs = self.text_encoder.run(ort::inputs![
            "text_ids" => text_ids_tensor,
            "text_mask" => text_mask_tensor,
            "style_ttl" => style_ttl_tensor
        ])?;

        let (_shape, data) = outputs["text_emb"].try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }

    #[allow(clippy::too_many_arguments)]
    fn flow_matching(
        &mut self,
        text_emb: &[f32],
        t_text: usize,
        t_lat: usize,
        cc: usize,
        style_ref_data: &[f32],
        style_ref_dims: &[usize],
        steps: u32,
    ) -> Result<Vec<f32>, BoxErr> {
        let cfg_scale = self.cfg_scale;

        // Initialize noisy latent with seeded random Gaussian noise (seed=42)
        let mut x = randn_seeded(cc * t_lat, 42);

        let text_mask_data = vec![1.0f32; t_text];
        let latent_mask_data = vec![1.0f32; t_lat];

        // Unconditional text dimension
        let u_t_text = *self.u_text_shape.last().unwrap_or(&t_text);
        let u_text_mask_data = vec![1.0f32; u_t_text];

        // Clone data to avoid borrow conflicts with &mut self
        let u_text = self.u_text.clone();
        let u_ref = self.u_ref.clone();
        let u_ref_shape = self.u_ref_shape.clone();
        let u_keys = self.u_keys.clone();
        let u_keys_shape = self.u_keys_shape.clone();
        // Conditional pass uses cond_keys from uncond.npz (NOT style_keys from style JSON)
        let cond_keys = self.cond_keys.clone();
        let cond_keys_shape = self.cond_keys_shape.clone();

        let use_cfg = (cfg_scale - 1.0).abs() > 1e-6;

        for i in 0..steps {
            let t_val = vec![i as f32];
            let total_t = vec![steps as f32];

            // --- Conditional pass ---
            let den_cond = self.run_backbone(
                &x,
                cc,
                t_lat,
                text_emb,
                t_text,
                style_ref_data,
                style_ref_dims,
                &latent_mask_data,
                &text_mask_data,
                &t_val,
                &total_t,
                &cond_keys,
                &cond_keys_shape,
            )?;

            if use_cfg {
                // --- Unconditional pass (only needed for classifier-free guidance) ---
                let den_uncond = self.run_backbone(
                    &x,
                    cc,
                    t_lat,
                    &u_text,
                    u_t_text,
                    &u_ref,
                    &u_ref_shape,
                    &latent_mask_data,
                    &u_text_mask_data,
                    &t_val,
                    &total_t,
                    &u_keys,
                    &u_keys_shape,
                )?;

                // x = den_uncond + cfg_scale * (den_cond - den_uncond)
                for j in 0..x.len() {
                    x[j] = den_uncond[j] + cfg_scale * (den_cond[j] - den_uncond[j]);
                }
            } else {
                // cfg_scale == 1.0: skip unconditional pass entirely
                x = den_cond;
            }
        }

        Ok(x)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_backbone(
        &mut self,
        noisy_latent: &[f32],
        cc: usize,
        t_lat: usize,
        text_emb: &[f32],
        t_text: usize,
        style_ref: &[f32],
        style_ref_dims: &[usize],
        latent_mask: &[f32],
        text_mask: &[f32],
        current_step: &[f32],
        total_step: &[f32],
        style_keys: &[f32],
        style_keys_dims: &[usize],
    ) -> Result<Vec<f32>, BoxErr> {
        // text_emb is [1, D, T_text]; compute D from data length
        let d_text = text_emb.len() / t_text;

        let noisy_tensor = make_f32_tensor(noisy_latent, &[1, cc, t_lat])?;
        let text_emb_tensor = make_f32_tensor(text_emb, &[1, d_text, t_text])?;
        let style_ref_tensor = make_f32_tensor(style_ref, style_ref_dims)?;
        let latent_mask_tensor = make_f32_tensor(latent_mask, &[1, 1, t_lat])?;
        let text_mask_tensor = make_f32_tensor(text_mask, &[1, 1, t_text])?;
        let step_tensor = make_f32_tensor(current_step, &[1])?;
        let total_tensor = make_f32_tensor(total_step, &[1])?;
        let style_keys_tensor = make_f32_tensor(style_keys, style_keys_dims)?;

        let outputs = self.backbone.run(ort::inputs![
            "noisy_latent" => noisy_tensor,
            "text_emb" => text_emb_tensor,
            "style_ttl" => style_ref_tensor,
            "style_keys" => style_keys_tensor,
            "latent_mask" => latent_mask_tensor,
            "text_mask" => text_mask_tensor,
            "current_step" => step_tensor,
            "total_step" => total_tensor
        ])?;

        let (_shape, data) = outputs["denoised_latent"].try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }

    fn run_vocoder(&mut self, latent: &[f32], t_dec: usize) -> Result<Vec<f32>, BoxErr> {
        let ld = self.latent_dim;
        let latent_tensor = make_f32_tensor(latent, &[1, ld, t_dec])?;

        let outputs = self.vocoder.run(ort::inputs![
            "latent" => latent_tensor
        ])?;

        let (_shape, data) = outputs["waveform"].try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_session(path: &Path, threads: usize) -> Result<Session, BoxErr> {
    load_session_with_device(path, threads, "auto")
}

fn load_session_cpu(path: &Path, threads: usize) -> Result<Session, BoxErr> {
    load_session_with_device(path, threads, "cpu")
}

/// Query the ORT runtime for which execution providers are actually available
/// in the loaded onnxruntime.dll. This calls the C API directly.
fn query_ort_available_providers() -> Vec<String> {
    // Wrap in catch_unwind because ort::api() panics if ORT failed to load.
    let result = std::panic::catch_unwind(|| {
        query_ort_available_providers_inner()
    });
    match result {
        Ok(providers) => providers,
        Err(_) => {
            log::error!("query_ort_available_providers panicked");
            vec![]
        }
    }
}

fn query_ort_available_providers_inner() -> Vec<String> {
    unsafe {
        let api = ort::api();
        let mut providers: *mut *mut std::ffi::c_char = std::ptr::null_mut();
        let mut num_providers: i32 = 0;
        let status = (api.GetAvailableProviders)(&mut providers, &mut num_providers);
        if !status.0.is_null() {
            (api.ReleaseStatus)(status.0);
            return vec![];
        }
        let mut result = Vec::new();
        for i in 0..num_providers {
            let cstr = std::ffi::CStr::from_ptr(*providers.offset(i as isize));
            if let Ok(s) = cstr.to_str() {
                result.push(s.to_string());
            }
        }
        let _ = (api.ReleaseAvailableProviders)(providers, num_providers);
        result
    }
}

/// Detect available providers by checking for EP DLLs next to the exe.
fn detect_available_providers() -> Vec<&'static str> {
    let mut providers = Vec::new();
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));

    let check_dll = |name: &str| -> bool {
        // Check exe dir and PATH
        if let Some(ref dir) = exe_dir {
            if dir.join(name).exists() {
                return true;
            }
        }
        if let Ok(path_var) = std::env::var("PATH") {
            for dir in path_var.split(';') {
                if std::path::Path::new(dir).join(name).exists() {
                    return true;
                }
            }
        }
        false
    };

    if check_dll("onnxruntime_providers_tensorrt.dll") {
        providers.push("TensorRT");
    }
    if check_dll("onnxruntime_providers_cuda.dll") {
        providers.push("CUDA");
    }
    if check_dll("onnxruntime_providers_openvino.dll") {
        providers.push("OpenVINO");
    }
    // DirectML is built into the WinML onnxruntime.dll, no separate DLL needed
    // but it's unreliable on low-end GPUs, so only add if explicitly requested
    if std::env::var("LIGHTBLUE_USE_DIRECTML").is_ok() {
        providers.push("DirectML");
    }

    providers
}

fn make_session_builder(threads: usize) -> Result<ort::session::builder::SessionBuilder, ort::Error> {
    let mut b = Session::builder()?;
    b = b.with_optimization_level(GraphOptimizationLevel::Level3)?;
    b = b.with_intra_threads(threads)?;
    if std::env::var("ORT_VERBOSE").is_ok() {
        b = b.with_log_level(ort::logging::LogLevel::Verbose)?;
    }
    // Disable ORT's heuristic that moves small ops to CPU.
    // Set ORT_NO_CPU_FALLBACK=1 to force all nodes onto the accelerator EP.
    if std::env::var("ORT_NO_CPU_FALLBACK").is_ok() {
        unsafe {
            let api = ort::api();
            let key = std::ffi::CString::new("session.disable_cpu_ep_fallback").unwrap();
            let val = std::ffi::CString::new("1").unwrap();
            let status = (api.AddSessionConfigEntry)(
                ort::AsPointer::ptr(&b) as *mut _,
                key.as_ptr(),
                val.as_ptr(),
            );
            if !status.0.is_null() {
                log::warn!("Failed to set disable_cpu_ep_fallback");
                (api.ReleaseStatus)(status.0);
            }
        }
    }
    Ok(b)
}

fn load_session_with_device(path: &Path, threads: usize, device: &str) -> Result<Session, BoxErr> {
    let fname = path.file_name().unwrap_or_default().to_string_lossy().to_string();

    let force_cpu = device == "cpu" || std::env::var("LIGHTBLUE_FORCE_CPU").is_ok();

    if !force_cpu {
        let available = detect_available_providers();
        log::info!("Available EPs for {fname}: {:?}", available);

        for ep_name in &available {
            let result = match *ep_name {
                "TensorRT" => {
                    let cache_dir = path.parent().unwrap_or(path).to_string_lossy().to_string();
                    make_session_builder(threads)
                        .and_then(|b| b.with_execution_providers([
                            ort::execution_providers::TensorRTExecutionProvider::default()
                                .with_fp16(true)
                                .with_engine_cache(true)
                                .with_engine_cache_path(&cache_dir)
                                .build()
                        ]))
                        .and_then(|b| b.commit_from_file(path))
                }
                "CUDA" => make_session_builder(threads)
                    .and_then(|b| b.with_execution_providers([ort::execution_providers::CUDAExecutionProvider::default().build()]))
                    .and_then(|b| b.commit_from_file(path)),
                "OpenVINO" => make_session_builder(threads)
                    .and_then(|b| b.with_execution_providers([ort::execution_providers::OpenVINO::default().build()]))
                    .and_then(|b| b.commit_from_file(path)),
                "DirectML" => make_session_builder(threads)
                    .and_then(|b| b.with_execution_providers([ort::execution_providers::DirectMLExecutionProvider::default().build()]))
                    .and_then(|b| b.commit_from_file(path)),
                _ => continue,
            };

            match result {
                Ok(session) => {
                    log::info!("Loaded {fname} with {ep_name}");
                    return Ok(session);
                }
                Err(e) => {
                    log::info!("{ep_name} failed for {fname}: {e}");
                }
            }
        }
    }

    let session = make_session_builder(threads)?.commit_from_file(path)?;
    log::info!("Loaded {fname} with CPU");
    Ok(session)
}

/// Apply symmetric linear fade in and fade out.
fn apply_fade(audio: &mut [f32], fade_duration: f32, sample_rate: u32) {
    apply_fade_in(audio, fade_duration, sample_rate);
    apply_fade_out(audio, fade_duration, sample_rate);
}

fn apply_fade_in(audio: &mut [f32], fade_duration: f32, sample_rate: u32) {
    let fade_samples = (fade_duration * sample_rate as f32) as usize;
    let fade_len = fade_samples.min(audio.len());
    for i in 0..fade_len {
        audio[i] *= i as f32 / fade_len as f32;
    }
}

fn apply_fade_out(audio: &mut [f32], fade_duration: f32, sample_rate: u32) {
    let fade_samples = (fade_duration * sample_rate as f32) as usize;
    let len = audio.len();
    let fade_len = fade_samples.min(len);
    for i in 0..fade_len {
        audio[len - 1 - i] *= i as f32 / fade_len as f32;
    }
}

/// Generate `n` samples from a standard normal distribution using a simple
/// seeded xoshiro128+ PRNG with Box-Muller transform (matches numpy's
/// RandomState(seed).randn behaviour closely enough for flow-matching init).
fn randn_seeded(n: usize, seed: u64) -> Vec<f32> {
    // Simple splitmix64 to seed xoshiro state
    let mut state = seed;
    let mut next_sm = || -> u64 {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };
    let mut s = [next_sm(), next_sm(), next_sm(), next_sm()];

    let mut next_u64 = || -> u64 {
        let result = (s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    };

    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        // Box-Muller transform
        let u1 = (next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let u2 = (next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let u1 = if u1 < 1e-12 { 1e-12 } else { u1 };
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((r * theta.cos()) as f32);
        if out.len() < n {
            out.push((r * theta.sin()) as f32);
        }
    }
    out
}
