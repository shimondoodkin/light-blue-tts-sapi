//! Bridge between the SAPI engine and the HebrewTTS pipeline.
//!
//! `LightBlueSynthesizer` implements `TtsSynthesizer` by:
//! 1. Expanding/normalizing input text
//! 2. Preserving a best-effort map from original text spans to normalized words
//! 3. Diacritizing + phonemizing normalized words to estimate timings
//! 4. Running neural TTS inference via `HebrewTTS`

use std::sync::{Mutex, OnceLock};

use crate::phonemize;
use crate::sapi::{BoxErr, SynthesisMetadata, SynthesisOutput, TextSpan, TtsSynthesizer, WordTiming};
use crate::tts::{HebrewTTS, TTSConfig};
use phonikud_rs::expander;

/// Wrapper that wires phonikud + phonemize + HebrewTTS into a single
/// `TtsSynthesizer` implementation for SAPI.
pub struct LightBlueSynthesizer {
    tts: Mutex<HebrewTTS>,
    phonikud: Mutex<phonikud_rs::Phonikud>,
    style_json_path: Option<String>,
}

#[derive(Debug, Clone)]
struct WordToken {
    text: String,
    span: TextSpan,
}

#[derive(Debug, Clone)]
struct PreparedWord {
    text: String,
    normalized_span: TextSpan,
    original_span: Option<TextSpan>,
    ipa: String,
}

#[derive(Debug, Clone)]
struct PreparedSynthesis {
    normalized_text: String,
    combined_ipa: String,
    words: Vec<PreparedWord>,
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
            let tts_join = h_tts.join().map_err(|e| {
                let msg = if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "TTS model loading panicked".to_string()
                };
                log::error!("TTS thread panicked: {msg}");
                msg
            });
            let pk_join = h_pk.join().map_err(|e| {
                let msg = if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "Phonikud loading panicked".to_string()
                };
                log::error!("Phonikud thread panicked: {msg}");
                msg
            });
            (tts_join, pk_join)
        });
        let tts_inner = tts_result.map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        let tts = tts_inner.map_err(|e| -> Box<dyn std::error::Error> { e })?;
        let phonikud_inner = phonikud_result.map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        let phonikud = phonikud_inner?;
        log::info!("All models loaded in {:?}", t_all.elapsed());
        Ok(Self {
            tts: Mutex::new(tts),
            phonikud: Mutex::new(phonikud),
            style_json_path,
        })
    }

    /// Convert a text chunk to IPA, handling Hebrew/English language splitting.
    ///
    /// Hebrew segments are diacritized via phonikud then phonemized;
    /// English segments go through the English phonemizer.
    fn segment_to_ipa(&self, text: &str) -> Result<String, BoxErr> {
        let segments = split_by_language(text);
        let mut ipa = String::new();

        for segment in &segments {
            let trimmed = segment.trim();
            if trimmed.is_empty() {
                continue;
            }

            let segment_ipa = if trimmed.chars().any(is_hebrew_char) {
                let t0 = std::time::Instant::now();
                let diacritized = {
                    let mut pk = self.phonikud.lock().unwrap();
                    pk.add_diacritics(trimmed)?
                };
                log::info!("phonikud diacritization: {:?}", t0.elapsed());
                log::debug!("Diacritized: {diacritized}");
                phonemize::phonemize(&diacritized)
            } else {
                phonemize::english::phonemize_english(trimmed)
            };

            if !segment_ipa.trim().is_empty() {
                if !ipa.is_empty() && !ipa.ends_with(' ') {
                    ipa.push(' ');
                }
                ipa.push_str(&segment_ipa);
            }
        }

        Ok(ipa)
    }

    fn prepare_synthesis(&self, original_text: &str) -> Result<PreparedSynthesis, BoxErr> {
        let expanded = expander::expand_text_with_spans(original_text);
        let normalized_text = expanded.text.clone();
        log::debug!("Normalized: \"{normalized_text}\"");

        // Chunk at paragraph level for phonikud (keeps input within BERT token limit)
        let paragraphs = split_paragraphs(&normalized_text);
        let mut all_ipa = String::new();

        for para in &paragraphs {
            let ipa = self.segment_to_ipa(para)?;
            if !ipa.trim().is_empty() {
                if !all_ipa.is_empty() && !all_ipa.ends_with(' ') {
                    all_ipa.push(' ');
                }
                all_ipa.push_str(&ipa);
            }
        }

        let words = self.prepare_words(&expanded)?;
        log::info!("Combined IPA: \"{all_ipa}\"");

        Ok(PreparedSynthesis {
            normalized_text,
            combined_ipa: all_ipa,
            words,
        })
    }

    fn prepare_words(&self, expanded: &expander::ExpandedText) -> Result<Vec<PreparedWord>, BoxErr> {
        let mut words = Vec::new();

        for token in &expanded.tokens {
            let original_span = TextSpan {
                start: token.original_span.start,
                end: token.original_span.end,
            };

            for subword in tokenize_words(&token.expanded_text) {
                let normalized_span = TextSpan {
                    start: token.expanded_span.start + subword.span.start,
                    end: token.expanded_span.start + subword.span.end,
                };
                let ipa = self.word_to_ipa(&subword.text)?;
                words.push(PreparedWord {
                    text: subword.text,
                    normalized_span,
                    original_span: Some(original_span),
                    ipa,
                });
            }
        }

        Ok(words)
    }

    fn word_to_ipa(&self, word: &str) -> Result<String, BoxErr> {
        if word.chars().any(is_hebrew_char) {
            let diacritized = {
                let mut pk = self.phonikud.lock().unwrap();
                pk.add_diacritics(word)?
            };
            Ok(phonemize::phonemize(&diacritized))
        } else {
            Ok(phonemize::english::phonemize_english(word))
        }
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

/// Split text into paragraphs.
///
/// Splits on blank lines (`\n\n` or `\r\n\r\n`). If no blank lines are found
/// the whole text is returned as a single paragraph.
fn split_paragraphs(text: &str) -> Vec<&str> {
    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .flat_map(|p| p.split("\r\n\r\n"))
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();

    if paragraphs.is_empty() {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }
        return vec![trimmed];
    }

    paragraphs
}

/// Split an IPA string into sentences at sentence-ending punctuation.
///
/// Cuts after `.`, `!`, or `?` (keeping the punctuation with the sentence).
/// Trailing whitespace-only fragments are dropped.
fn split_ipa_sentences(ipa: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in ipa.chars() {
        current.push(ch);
        if ch == '.' || ch == '!' || ch == '?' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    // Remainder after the last sentence-ending punctuation
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

/// Split text into segments of Hebrew vs non-Hebrew.
fn split_by_language(text: &str) -> Vec<String> {
    let mut segments: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_hebrew: Option<bool> = None;

    for ch in text.chars() {
        if ch.is_whitespace() || ch.is_ascii_punctuation() {
            current.push(ch);
            continue;
        }

        let hebrew = is_hebrew_char(ch);
        match current_hebrew {
            Some(flag) if flag == hebrew => current.push(ch),
            Some(_) => {
                if !current.is_empty() {
                    segments.push(std::mem::take(&mut current));
                }
                current.push(ch);
                current_hebrew = Some(hebrew);
            }
            None => {
                current.push(ch);
                current_hebrew = Some(hebrew);
            }
        }
    }

    if !current.is_empty() {
        segments.push(current);
    }

    segments
}

fn tokenize_words(text: &str) -> Vec<WordToken> {
    let mut out = Vec::new();
    let mut current_start: Option<usize> = None;

    for (idx, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(start) = current_start.take() {
                out.push(WordToken {
                    text: text[start..idx].to_string(),
                    span: TextSpan { start, end: idx },
                });
            }
        } else if current_start.is_none() {
            current_start = Some(idx);
        }
    }

    if let Some(start) = current_start {
        out.push(WordToken {
            text: text[start..].to_string(),
            span: TextSpan {
                start,
                end: text.len(),
            },
        });
    }

    out
}

fn build_metadata(
    original_text: &str,
    prepared: &PreparedSynthesis,
    sample_count: usize,
    sample_rate: u32,
) -> SynthesisMetadata {
    let total_duration_sec = if sample_rate == 0 {
        0.0
    } else {
        sample_count as f32 / sample_rate as f32
    };
    let words = estimate_word_timings(&prepared.words, total_duration_sec);

    SynthesisMetadata {
        original_text: original_text.to_string(),
        normalized_text: prepared.normalized_text.clone(),
        combined_ipa: prepared.combined_ipa.clone(),
        words,
    }
}

fn estimate_word_timings(words: &[PreparedWord], total_duration_sec: f32) -> Vec<WordTiming> {
    if words.is_empty() || total_duration_sec <= 0.0 {
        return words
            .iter()
            .map(|word| WordTiming {
                text: word.text.clone(),
                normalized_span: word.normalized_span,
                original_span: word.original_span,
                start_sec: 0.0,
                end_sec: 0.0,
            })
            .collect();
    }

    let mut units = Vec::with_capacity(words.len());
    let mut total_units = 0.0f32;
    for word in words {
        let speech_weight = word_weight(word);
        let pause_after = pause_weight(&word.text);
        units.push((speech_weight, pause_after));
        total_units += speech_weight + pause_after;
    }

    if total_units <= 0.0 {
        total_units = words.len() as f32;
        units = vec![(1.0, 0.0); words.len()];
    }

    let sec_per_unit = total_duration_sec / total_units;
    let mut cursor = 0.0f32;
    let mut timings = Vec::with_capacity(words.len());

    for (word, (speech_weight, pause_after)) in words.iter().zip(units.into_iter()) {
        let start_sec = cursor;
        let end_sec = (cursor + speech_weight * sec_per_unit).min(total_duration_sec);
        timings.push(WordTiming {
            text: word.text.clone(),
            normalized_span: word.normalized_span,
            original_span: word.original_span,
            start_sec,
            end_sec,
        });
        cursor = end_sec + pause_after * sec_per_unit;
    }

    if let Some(last) = timings.last_mut() {
        last.end_sec = total_duration_sec.max(last.end_sec);
    }

    timings
}

fn word_weight(word: &PreparedWord) -> f32 {
    let ipa_units = word
        .ipa
        .chars()
        .filter(|c| !c.is_whitespace() && !c.is_ascii_punctuation())
        .count() as f32;
    let text_units = word
        .text
        .chars()
        .filter(|c| !c.is_whitespace())
        .count() as f32;
    ipa_units.max(text_units).max(1.0)
}

fn pause_weight(word: &str) -> f32 {
    if word.ends_with('.') || word.ends_with('!') || word.ends_with('?') {
        1.2
    } else if word.ends_with(',') || word.ends_with(';') || word.ends_with(':') {
        0.6
    } else {
        0.2
    }
}

/// Group prepared words into sentence groups.
///
/// A new sentence starts after a word whose text ends with sentence-ending
/// punctuation (`.`, `!`, `?`).  This aligns with `split_ipa_sentences`
/// which splits on the same characters.
fn group_words_by_sentence(words: &[PreparedWord]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = vec![Vec::new()];

    for (i, word) in words.iter().enumerate() {
        groups.last_mut().unwrap().push(i);
        let ends_sentence = word.text.ends_with('.')
            || word.text.ends_with('!')
            || word.text.ends_with('?');
        if ends_sentence && i + 1 < words.len() {
            groups.push(Vec::new());
        }
    }

    groups.retain(|g| !g.is_empty());
    groups
}

/// Estimate word timings within a sentence and offset by cumulative time.
fn estimate_sentence_word_timings(
    words: &[&PreparedWord],
    sentence_duration_sec: f32,
    time_offset_sec: f32,
) -> Vec<WordTiming> {
    if words.is_empty() {
        return Vec::new();
    }

    let mut units = Vec::with_capacity(words.len());
    let mut total_units = 0.0f32;
    for &word in words {
        let speech = word_weight(word);
        let pause = pause_weight(&word.text);
        units.push((speech, pause));
        total_units += speech + pause;
    }

    if total_units <= 0.0 {
        total_units = words.len() as f32;
        units = vec![(1.0, 0.0); words.len()];
    }

    let sec_per_unit = sentence_duration_sec / total_units;
    let mut cursor = 0.0f32;
    let mut timings = Vec::with_capacity(words.len());

    for (&word, (speech_weight, pause_after)) in words.iter().zip(units.into_iter()) {
        let start_sec = time_offset_sec + cursor;
        let end_sec = time_offset_sec
            + (cursor + speech_weight * sec_per_unit).min(sentence_duration_sec);
        timings.push(WordTiming {
            text: word.text.clone(),
            normalized_span: word.normalized_span,
            original_span: word.original_span,
            start_sec,
            end_sec,
        });
        cursor = (cursor + speech_weight * sec_per_unit + pause_after * sec_per_unit)
            .min(sentence_duration_sec);
    }

    if let Some(last) = timings.last_mut() {
        last.end_sec = (time_offset_sec + sentence_duration_sec).max(last.end_sec);
    }

    timings
}

impl TtsSynthesizer for LightBlueSynthesizer {
    fn synthesize(
        &self,
        text: &str,
        style_json: Option<&str>,
        steps: Option<u32>,
        _use_gpu: bool,
    ) -> Result<SynthesisOutput, BoxErr> {
        let effective_style = style_json.or(self.style_json_path.as_deref());
        let prepared = self.prepare_synthesis(text)?;

        if prepared.combined_ipa.trim().is_empty() {
            log::warn!("No phonemes produced for: \"{text}\"");
            let tts = self.tts.lock().unwrap();
            let sr = tts.sample_rate();
            let silence_len = (sr as f32 * 0.5) as usize;
            let samples = vec![0.0f32; silence_len];
            return Ok(SynthesisOutput {
                metadata: build_metadata(text, &prepared, samples.len(), sr),
                samples,
                sample_rate: sr,
            });
        }

        let mut tts = self.tts.lock().unwrap();
        let samples = tts.infer(&prepared.combined_ipa, effective_style, steps)?;
        let sr = tts.sample_rate();
        let metadata = build_metadata(text, &prepared, samples.len(), sr);
        Ok(SynthesisOutput {
            samples,
            sample_rate: sr,
            metadata,
        })
    }

    fn synthesize_stream(
        &self,
        text: &str,
        style_json: Option<&str>,
        steps: Option<u32>,
        _use_gpu: bool,
        sink: &mut dyn FnMut(&[f32]) -> Result<(), BoxErr>,
    ) -> Result<SynthesisMetadata, BoxErr> {
        let effective_style = style_json.or(self.style_json_path.as_deref());
        let expanded = expander::expand_text_with_spans(text);
        let normalized_text = expanded.text.clone();
        log::debug!("Normalized: \"{normalized_text}\"");

        // Split into paragraphs, then phonikud each, then split IPA into
        // sentences and stream each sentence through TTS immediately.
        let paragraphs = split_paragraphs(&normalized_text);
        let mut all_ipa = String::new();
        let mut total_samples = 0usize;

        // Collect all sentence IPA strings first so we know the total count
        // (needed to decide fade-in/fade-out on the first/last sentence).
        let mut sentence_ipas: Vec<String> = Vec::new();

        for para in &paragraphs {
            let para_ipa = self.segment_to_ipa(para)?;
            if para_ipa.trim().is_empty() {
                continue;
            }
            if !all_ipa.is_empty() && !all_ipa.ends_with(' ') {
                all_ipa.push(' ');
            }
            all_ipa.push_str(&para_ipa);

            let sentences = split_ipa_sentences(&para_ipa);
            for s in sentences {
                if !s.trim().is_empty() {
                    sentence_ipas.push(s);
                }
            }
        }

        if sentence_ipas.is_empty() {
            log::warn!("No phonemes produced for: \"{text}\"");
            let tts = self.tts.lock().unwrap();
            let sr = tts.sample_rate();
            let silence_len = (sr as f32 * 0.5) as usize;
            let samples = vec![0.0f32; silence_len];
            sink(&samples)?;
            return Ok(SynthesisMetadata {
                original_text: text.to_string(),
                normalized_text,
                combined_ipa: all_ipa,
                words: Vec::new(),
            });
        }

        let num_sentences = sentence_ipas.len();
        log::info!(
            "Streaming {} sentence(s) from {} paragraph(s)",
            num_sentences,
            paragraphs.len()
        );

        let mut tts = self.tts.lock().unwrap();

        for (i, sentence_ipa) in sentence_ipas.iter().enumerate() {
            let is_first = i == 0;
            let is_last = i + 1 == num_sentences;
            log::info!(
                "  sentence {}/{}: IPA=\"{}\"",
                i + 1,
                num_sentences,
                sentence_ipa
            );

            tts.infer_stream(sentence_ipa, effective_style, steps, is_first, is_last, |chunk| {
                total_samples += chunk.len();
                sink(&chunk)
            })?;

            // Inter-sentence silence
            if !is_last {
                let sr = tts.sample_rate();
                let pause_samples = (sr as f32 * 0.25) as usize;
                let silence = vec![0.0f32; pause_samples];
                total_samples += pause_samples;
                sink(&silence)?;
            }
        }

        log::info!("Streamed {} total samples", total_samples);

        Ok(SynthesisMetadata {
            original_text: text.to_string(),
            normalized_text,
            combined_ipa: all_ipa,
            words: Vec::new(),
        })
    }

    fn synthesize_sentences(
        &self,
        text: &str,
        style_json: Option<&str>,
        steps: Option<u32>,
        _use_gpu: bool,
        on_sentence: &mut dyn FnMut(&str, &[WordTiming], &[f32]) -> Result<(), BoxErr>,
    ) -> Result<SynthesisMetadata, BoxErr> {
        let effective_style = style_json.or(self.style_json_path.as_deref());
        let expanded = expander::expand_text_with_spans(text);
        let normalized_text = expanded.text.clone();
        log::debug!("Normalized: \"{normalized_text}\"");

        // Prepare words for timing estimation
        let all_words = self.prepare_words(&expanded)?;
        let word_groups = group_words_by_sentence(&all_words);

        // Split into paragraphs, phonikud each, split IPA into sentences
        let paragraphs = split_paragraphs(&normalized_text);
        let mut sentence_ipas: Vec<String> = Vec::new();
        let mut all_ipa = String::new();

        for para in &paragraphs {
            let para_ipa = self.segment_to_ipa(para)?;
            if para_ipa.trim().is_empty() {
                continue;
            }
            if !all_ipa.is_empty() && !all_ipa.ends_with(' ') {
                all_ipa.push(' ');
            }
            all_ipa.push_str(&para_ipa);

            for s in split_ipa_sentences(&para_ipa) {
                if !s.trim().is_empty() {
                    sentence_ipas.push(s);
                }
            }
        }

        if sentence_ipas.is_empty() {
            log::warn!("No phonemes produced for: \"{text}\"");
            let tts = self.tts.lock().unwrap();
            let sr = tts.sample_rate();
            let silence_len = (sr as f32 * 0.5) as usize;
            let silence = vec![0.0f32; silence_len];
            on_sentence(text, &[], &silence)?;
            return Ok(SynthesisMetadata {
                original_text: text.to_string(),
                normalized_text,
                combined_ipa: all_ipa,
                words: Vec::new(),
            });
        }

        let num_sentences = sentence_ipas.len();
        let num_word_groups = word_groups.len();
        let groups_aligned = num_sentences == num_word_groups;
        if !groups_aligned {
            log::warn!(
                "Sentence/word-group mismatch: {} IPA sentences vs {} word groups; \
                 falling back to flat word distribution",
                num_sentences,
                num_word_groups,
            );
        }

        log::info!(
            "synthesize_sentences: {} sentence(s), {} word group(s), aligned={}",
            num_sentences,
            num_word_groups,
            groups_aligned,
        );

        let mut tts = self.tts.lock().unwrap();
        let sr = tts.sample_rate();
        let mut cumulative_sec = 0.0f32;
        let mut total_samples = 0usize;

        for (i, sentence_ipa) in sentence_ipas.iter().enumerate() {
            let is_first = i == 0;
            let is_last = i + 1 == num_sentences;
            log::info!(
                "  sentence {}/{}: IPA=\"{}\"",
                i + 1,
                num_sentences,
                sentence_ipa
            );

            // Buffer audio for this sentence so we know exact duration
            let mut sentence_audio = Vec::new();
            tts.infer_stream(
                sentence_ipa,
                effective_style,
                steps,
                is_first,
                is_last,
                |chunk| {
                    sentence_audio.extend(chunk);
                    Ok(())
                },
            )?;

            let sentence_duration = sentence_audio.len() as f32 / sr as f32;

            // Compute word timings for this sentence
            let word_timings = if groups_aligned {
                let indices = &word_groups[i];
                let words: Vec<&PreparedWord> =
                    indices.iter().map(|&idx| &all_words[idx]).collect();
                estimate_sentence_word_timings(&words, sentence_duration, cumulative_sec)
            } else if i == 0 {
                // Fallback: put all words in the first sentence
                let words: Vec<&PreparedWord> = all_words.iter().collect();
                let total_dur: f32 = sentence_ipas.len() as f32 * sentence_duration;
                estimate_sentence_word_timings(&words, total_dur, 0.0)
            } else {
                Vec::new()
            };

            total_samples += sentence_audio.len();
            on_sentence(text, &word_timings, &sentence_audio)?;
            cumulative_sec += sentence_duration;

            // Insert inter-sentence silence (natural pause that also masks
            // processing time for the next sentence).
            if !is_last {
                let pause_samples = (sr as f32 * 0.25) as usize; // 250 ms
                let silence = vec![0.0f32; pause_samples];
                total_samples += pause_samples;
                on_sentence(text, &[], &silence)?;
                cumulative_sec += pause_samples as f32 / sr as f32;
            }
        }

        log::info!("synthesize_sentences: {} total samples", total_samples);

        Ok(SynthesisMetadata {
            original_text: text.to_string(),
            normalized_text,
            combined_ipa: all_ipa,
            words: Vec::new(),
        })
    }
}

impl TtsSynthesizer for LazyLightBlueSynthesizer {
    fn synthesize(
        &self,
        text: &str,
        style_json: Option<&str>,
        steps: Option<u32>,
        use_gpu: bool,
    ) -> Result<SynthesisOutput, BoxErr> {
        let synth = self
            .get_or_init()
            .map_err(|e| -> BoxErr { e.to_string().into() })?;
        synth.synthesize(text, style_json, steps, use_gpu)
    }

    fn synthesize_stream(
        &self,
        text: &str,
        style_json: Option<&str>,
        steps: Option<u32>,
        use_gpu: bool,
        sink: &mut dyn FnMut(&[f32]) -> Result<(), BoxErr>,
    ) -> Result<SynthesisMetadata, BoxErr> {
        let synth = self
            .get_or_init()
            .map_err(|e| -> BoxErr { e.to_string().into() })?;
        synth.synthesize_stream(text, style_json, steps, use_gpu, sink)
    }

    fn synthesize_sentences(
        &self,
        text: &str,
        style_json: Option<&str>,
        steps: Option<u32>,
        use_gpu: bool,
        on_sentence: &mut dyn FnMut(&str, &[WordTiming], &[f32]) -> Result<(), BoxErr>,
    ) -> Result<SynthesisMetadata, BoxErr> {
        let synth = self
            .get_or_init()
            .map_err(|e| -> BoxErr { e.to_string().into() })?;
        synth.synthesize_sentences(text, style_json, steps, use_gpu, on_sentence)
    }
}
