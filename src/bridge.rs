//! Bridge between the SAPI engine and the HebrewTTS pipeline.
//!
//! `LightBlueSynthesizer` implements `TtsSynthesizer` by:
//! 1. Optionally adding nikud (diacritics) via phonikud-rs
//! 2. Phonemizing the diacritized Hebrew text to IPA
//! 3. Running neural TTS inference via `HebrewTTS`

use std::sync::{Mutex, OnceLock};

use crate::expander;
use crate::phonemize;
use crate::sapi::TtsSynthesizer;
use crate::tts::{HebrewTTS, TTSConfig};

/// Wrapper that wires phonikud + phonemize + HebrewTTS into a single
/// `TtsSynthesizer` implementation for SAPI.
pub struct LightBlueSynthesizer {
    tts: Mutex<HebrewTTS>,
    phonikud: Mutex<phonikud_rs::Phonikud>,
    style_json_path: Option<String>,
}

impl LightBlueSynthesizer {
    /// Create a new synthesizer.
    ///
    /// * `config` — TTS model configuration (ONNX dirs, etc.)
    /// * `phonikud_model` — path to the phonikud ONNX model
    /// * `phonikud_tokenizer` — path to the DictaBERT `tokenizer.json`
    /// * `style_json_path` — optional default voice style JSON
    pub fn new(
        config: TTSConfig,
        phonikud_model: &str,
        phonikud_tokenizer: &str,
        style_json_path: Option<String>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let t_all = std::time::Instant::now();
        let pm = phonikud_model.to_string();
        let pt = phonikud_tokenizer.to_string();
        let (tts_result, phonikud_result) = std::thread::scope(|s| {
            let h_tts = s.spawn(move || {
                let t0 = std::time::Instant::now();
                let r = HebrewTTS::new(config);
                log::info!("TTS models loaded in {:?}", t0.elapsed());
                r
            });
            let h_pk = s.spawn(move || {
                let t0 = std::time::Instant::now();
                let r = phonikud_rs::Phonikud::new(&pm, &pt);
                log::info!("Phonikud loaded in {:?}", t0.elapsed());
                r
            });
            (h_tts.join().unwrap(), h_pk.join().unwrap())
        });
        let tts = tts_result.map_err(|e| -> Box<dyn std::error::Error> { e })?;
        let phonikud = phonikud_result?;
        log::info!("All models loaded in {:?}", t_all.elapsed());
        Ok(Self {
            tts: Mutex::new(tts),
            phonikud: Mutex::new(phonikud),
            style_json_path,
        })
    }
}

/// Lazy wrapper that defers heavyweight model construction until first use.
pub struct LazyLightBlueSynthesizer {
    config: TTSConfig,
    phonikud_model: String,
    phonikud_tokenizer: String,
    style_json_path: Option<String>,
    inner: OnceLock<Result<LightBlueSynthesizer, String>>,
}

impl LazyLightBlueSynthesizer {
    pub fn new(
        config: TTSConfig,
        phonikud_model: &str,
        phonikud_tokenizer: &str,
        style_json_path: Option<String>,
    ) -> Self {
        Self {
            config,
            phonikud_model: phonikud_model.to_string(),
            phonikud_tokenizer: phonikud_tokenizer.to_string(),
            style_json_path,
            inner: OnceLock::new(),
        }
    }

    fn get_or_init(&self) -> Result<&LightBlueSynthesizer, Box<dyn std::error::Error>> {
        self.inner
            .get_or_init(|| {
                log::info!("Loading LightBlue TTS models on demand");
                LightBlueSynthesizer::new(
                    self.config.clone(),
                    &self.phonikud_model,
                    &self.phonikud_tokenizer,
                    self.style_json_path.clone(),
                )
                .map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| e.clone().into())
    }

    pub fn warm_up(&self) -> Result<(), Box<dyn std::error::Error>> {
        let _ = self.get_or_init()?;
        Ok(())
    }
}

/// Returns `true` if the character is a Hebrew letter/diacritic (U+0590..U+05FF).
fn is_hebrew_char(c: char) -> bool {
    ('\u{0590}'..='\u{05FF}').contains(&c)
}

/// Split text into segments of Hebrew vs non-Hebrew.
/// Each segment is `(is_hebrew, text)`.
fn split_by_language(text: &str) -> Vec<(bool, String)> {
    let mut segments: Vec<(bool, String)> = Vec::new();

    for ch in text.chars() {
        if ch.is_whitespace() || ch.is_ascii_punctuation() {
            // Attach whitespace/punctuation to the current segment, or start a new one
            if let Some(last) = segments.last_mut() {
                last.1.push(ch);
            } else {
                segments.push((false, ch.to_string()));
            }
        } else {
            let hebrew = is_hebrew_char(ch);
            if let Some(last) = segments.last_mut() {
                if last.0 == hebrew {
                    last.1.push(ch);
                } else {
                    segments.push((hebrew, ch.to_string()));
                }
            } else {
                segments.push((hebrew, ch.to_string()));
            }
        }
    }

    segments
}

impl TtsSynthesizer for LightBlueSynthesizer {
    fn synthesize(&self, text: &str, style_json: Option<&str>, steps: Option<u32>, _use_gpu: bool) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
        let effective_style = style_json.or(self.style_json_path.as_deref());

        // Normalize (expand numbers to words, etc.) then split by language
        let normalized = expander::expand_text(text);
        log::debug!("Normalized: \"{normalized}\"");
        let segments = split_by_language(&normalized);
        let mut all_ipa = String::new();

        for (is_hebrew, segment_text) in &segments {
            let trimmed = segment_text.trim();
            if trimmed.is_empty() {
                continue;
            }

            if *is_hebrew {
                let t0 = std::time::Instant::now();
                let diacritized = {
                    let mut pk = self.phonikud.lock().unwrap();
                    pk.add_diacritics(trimmed)?
                };
                log::info!("phonikud diacritization: {:?}", t0.elapsed());
                log::debug!("Diacritized: {diacritized}");

                let ipa = phonemize::phonemize(&diacritized);
                log::debug!("Phonemized: {ipa}");

                if !all_ipa.is_empty() && !all_ipa.ends_with(' ') {
                    all_ipa.push(' ');
                }
                all_ipa.push_str(&ipa);
            } else {
                let ipa = phonemize::english::phonemize_english(trimmed);
                log::debug!("English phonemized: \"{trimmed}\" -> \"{ipa}\"");

                if !ipa.trim().is_empty() {
                    if !all_ipa.is_empty() && !all_ipa.ends_with(' ') {
                        all_ipa.push(' ');
                    }
                    all_ipa.push_str(&ipa);
                }
            }
        }

        if all_ipa.trim().is_empty() {
            log::warn!("No phonemes produced for: \"{text}\"");
            let tts = self.tts.lock().unwrap();
            let sr = tts.sample_rate();
            let silence_len = (sr as f32 * 0.5) as usize;
            return Ok((vec![0.0f32; silence_len], sr));
        }

        log::info!("Combined IPA: \"{all_ipa}\"");

        let mut tts = self.tts.lock().unwrap();
        let samples = tts
            .infer(&all_ipa, effective_style, steps)
            .map_err(|e| -> Box<dyn std::error::Error> { e })?;
        let sr = tts.sample_rate();
        Ok((samples, sr))
    }
}

impl TtsSynthesizer for LazyLightBlueSynthesizer {
    fn synthesize(&self, text: &str, style_json: Option<&str>, steps: Option<u32>, use_gpu: bool) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
        self.get_or_init()?.synthesize(text, style_json, steps, use_gpu)
    }
}
