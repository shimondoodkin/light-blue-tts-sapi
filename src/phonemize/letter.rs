//! The `Letter` struct – one Hebrew base character plus its diacritics.
//! Port of Python `phonikud/variants.py`.

use super::lexicon::{HATAMA_DIACRITIC, PREFIX_DIACRITIC};
use std::fmt;

/// A single Hebrew letter with its diacritics.
#[derive(Debug, Clone)]
pub struct Letter {
    /// The base consonant character (NFD-normalized).
    pub char_: String,
    /// All diacritics attached to this letter (including hatama / prefix markers).
    pub all_diac: String,
    /// Diacritics minus hatama and prefix marker – used for phoneme lookup.
    pub diac: String,
}

impl Letter {
    pub fn new(ch: &str, diac: &str) -> Self {
        // `ch` and `diac` are expected to already be NFD-normalized by the caller.
        let all_diac = diac.to_string();
        let filtered: String = all_diac
            .chars()
            .filter(|&c| c != HATAMA_DIACRITIC && c != PREFIX_DIACRITIC)
            .collect();
        Letter {
            char_: ch.to_string(),
            all_diac,
            diac: filtered,
        }
    }

    /// Reconstruct the original text for this letter.
    pub fn to_text(&self) -> String {
        format!("{}{}", self.char_, self.all_diac)
    }
}

impl fmt::Display for Letter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.char_, self.all_diac)
    }
}
