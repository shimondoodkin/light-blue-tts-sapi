use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Top-level config loaded from `tts.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct TTSConfigJson {
    #[serde(default)]
    pub ttl: TtlConfig,
    #[serde(default)]
    pub ae: AeConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtlConfig {
    #[serde(default = "default_latent_dim")]
    pub latent_dim: usize,
    #[serde(default = "default_chunk_compress_factor")]
    pub chunk_compress_factor: usize,
    #[serde(default)]
    pub normalizer: NormalizerConfig,
}

impl Default for TtlConfig {
    fn default() -> Self {
        Self {
            latent_dim: default_latent_dim(),
            chunk_compress_factor: default_chunk_compress_factor(),
            normalizer: NormalizerConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct NormalizerConfig {
    #[serde(default = "default_scale")]
    pub scale: f32,
}

impl Default for NormalizerConfig {
    fn default() -> Self {
        Self {
            scale: default_scale(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AeConfig {
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default)]
    pub encoder: AeEncoderConfig,
}

impl Default for AeConfig {
    fn default() -> Self {
        Self {
            sample_rate: default_sample_rate(),
            encoder: AeEncoderConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AeEncoderConfig {
    #[serde(default)]
    pub spec_processor: SpecProcessorConfig,
}

impl Default for AeEncoderConfig {
    fn default() -> Self {
        Self {
            spec_processor: SpecProcessorConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpecProcessorConfig {
    #[serde(default = "default_hop_length")]
    pub hop_length: usize,
}

impl Default for SpecProcessorConfig {
    fn default() -> Self {
        Self {
            hop_length: default_hop_length(),
        }
    }
}

fn default_latent_dim() -> usize { 24 }
fn default_chunk_compress_factor() -> usize { 6 }
fn default_scale() -> f32 { 1.0 }
fn default_sample_rate() -> u32 { 44100 }
fn default_hop_length() -> usize { 512 }

/// Runtime configuration for the TTS engine.
#[derive(Debug, Clone)]
pub struct TTSConfig {
    /// Directory containing ONNX model files, stats.npz, uncond.npz.
    pub onnx_dir: PathBuf,
    /// Path to tts.json config.
    pub config_json_path: PathBuf,
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
    /// Create a config with sensible defaults; only `onnx_dir` and `config_json_path` are required.
    pub fn new(onnx_dir: impl Into<PathBuf>, config_json_path: impl Into<PathBuf>) -> Self {
        Self {
            onnx_dir: onnx_dir.into(),
            config_json_path: config_json_path.into(),
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

/// Load the JSON model config from disk.
pub(crate) fn load_model_config(path: &Path) -> Result<TTSConfigJson, Box<dyn std::error::Error + Send + Sync>> {
    let data = std::fs::read_to_string(path)?;
    let cfg: TTSConfigJson = serde_json::from_str(&data)?;
    Ok(cfg)
}
