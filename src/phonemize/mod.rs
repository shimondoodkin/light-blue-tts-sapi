//! Hebrew phonemization module.
//!
//! Converts diacritized Hebrew text (with nikud) to IPA phonemes.
//! Port of the Python `phonikud` library.

pub mod english;
pub mod hebrew;
pub mod letter;
pub mod lexicon;
pub mod syllables;
pub mod utils;

use lexicon::PUNCTUATION;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Which phoneme schema to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Schema {
    /// Modern Israeli Hebrew (replaces x->chi, r->ʁ, g->ɡ)
    Modern,
    /// Academic / traditional transliteration
    Academic,
}

impl Default for Schema {
    fn default() -> Self {
        Schema::Modern
    }
}

/// Options controlling phonemization behaviour.
#[derive(Debug, Clone)]
pub struct PhonemizeOptions {
    /// Keep punctuation characters in the output (default: true).
    pub preserve_punctuation: bool,
    /// Keep stress marks in the output (default: true).
    pub preserve_stress: bool,
    /// Apply TTS post-normalization: strip trailing glottal-stop/h, ij->i (default: true).
    pub use_post_normalize: bool,
    /// Predict stress (milra) when no hatama is present (default: true).
    pub predict_stress: bool,
    /// Predict vocal shva for word-initial letters (default: true).
    pub predict_vocal_shva: bool,
    /// Phoneme schema (default: Modern).
    pub schema: Schema,
}

impl Default for PhonemizeOptions {
    fn default() -> Self {
        PhonemizeOptions {
            preserve_punctuation: true,
            preserve_stress: true,
            use_post_normalize: true,
            predict_stress: true,
            predict_vocal_shva: true,
            schema: Schema::Modern,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Phonemize Hebrew text with default options.
pub fn phonemize(text: &str) -> String {
    phonemize_with_options(text, &PhonemizeOptions::default())
}

/// Phonemize Hebrew text with custom options.
pub fn phonemize_with_options(text: &str, opts: &PhonemizeOptions) -> String {
    let mut result = String::new();

    for token in tokenize(text) {
        match token {
            Token::Punct(ch) => {
                if opts.preserve_punctuation {
                    result.push(ch);
                }
            }
            Token::Space => {
                result.push(' ');
            }
            Token::Word(w) => {
                let phonemes = phonemize_single_word(&w, opts);
                result.push_str(&phonemes);
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

enum Token {
    Word(String),
    Punct(char),
    Space,
}

/// Simple tokenizer: split text into words, punctuation, and spaces.
fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut current_word = String::new();

    for ch in text.chars() {
        if ch == ' ' {
            if !current_word.is_empty() {
                tokens.push(Token::Word(current_word.clone()));
                current_word.clear();
            }
            tokens.push(Token::Space);
        } else if PUNCTUATION.contains(&ch) && ch != ' ' {
            if !current_word.is_empty() {
                tokens.push(Token::Word(current_word.clone()));
                current_word.clear();
            }
            tokens.push(Token::Punct(ch));
        } else {
            current_word.push(ch);
        }
    }

    if !current_word.is_empty() {
        tokens.push(Token::Word(current_word));
    }

    tokens
}

/// Phonemize a single word.
fn phonemize_single_word(word: &str, opts: &PhonemizeOptions) -> String {
    let mut letters = utils::get_letters(word);
    if letters.is_empty() {
        return String::new();
    }

    // Pre-processing
    utils::sort_hatama(&mut letters);

    if opts.predict_vocal_shva {
        utils::mark_vocal_shva(&mut letters);
    }

    if opts.predict_stress {
        utils::add_milra_hatama(&mut letters);
    }

    // Core FST
    let phoneme_parts = hebrew::phonemize_word(&letters);
    let mut phonemes: String = phoneme_parts.join("");

    // Apply modern schema
    if opts.schema == Schema::Modern {
        for (&from, &to) in lexicon::MODERN_SCHEMA.iter() {
            phonemes = phonemes.replace(from, to);
        }
    }

    // Post-processing
    if opts.use_post_normalize {
        phonemes = utils::post_normalize(&phonemes);
    }

    if !opts.preserve_stress {
        phonemes = phonemes.replace(lexicon::STRESS_PHONEME, "");
    }

    phonemes
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shalom() {
        // שָׁלוֹם with nikud
        let input = "\u{05E9}\u{05C1}\u{05B8}\u{05DC}\u{05D5}\u{05B9}\u{05DD}";
        let result = phonemize(input);
        assert!(
            !result.is_empty(),
            "Phonemization of shalom should not be empty"
        );
        assert!(
            result.contains('a') || result.contains('o'),
            "Phonemization of shalom should contain vowels: got '{}'",
            result
        );
    }

    #[test]
    fn test_empty() {
        assert_eq!(phonemize(""), "");
    }

    #[test]
    fn test_punctuation_preserved() {
        let opts = PhonemizeOptions {
            preserve_punctuation: true,
            ..Default::default()
        };
        let result = phonemize_with_options(".", &opts);
        assert_eq!(result, ".");
    }
}
