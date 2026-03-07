use std::path::PathBuf;

// Model constants (from LightBlue model architecture)
pub const LATENT_DIM: usize = 24;
pub const CHUNK_COMPRESS_FACTOR: usize = 6;
pub const NORMALIZER_SCALE: f32 = 1.0;
pub const SAMPLE_RATE: u32 = 44100;
pub const HOP_LENGTH: usize = 512;

/// Runtime configuration for the TTS engine.
#[derive(Debug, Clone)]
pub struct TTSConfig {
    /// Directory containing ONNX model files, stats.npz, uncond.npz.
    pub onnx_dir: PathBuf,
    /// Default style JSON path (optional).
    pub default_style_json: Option<PathBuf>,
    /// Classifier-free guidance scale.
    pub cfg_scale: f32,
    /// Number of flow-matching diffusion steps.
    pub steps: u32,
    /// Maximum characters per text chunk.
    pub text_chunk_len: usize,
    /// Seconds of silence inserted between chunks.
    pub silence_sec: f32,
    /// Fade duration in seconds for fade-in/out.
    pub fade_duration: f32,
    /// Number of intra-op threads for ONNX Runtime.
    pub threads: usize,
}

impl TTSConfig {
    /// Create a config with sensible defaults; only `onnx_dir` is required.
    pub fn new(onnx_dir: impl Into<PathBuf>) -> Self {
        Self {
            onnx_dir: onnx_dir.into(),
            default_style_json: None,
            cfg_scale: 3.0,
            steps: 8,
            text_chunk_len: 150,
            silence_sec: 0.15,
            fade_duration: 0.01,
            threads: 4,
        }
    }
}
