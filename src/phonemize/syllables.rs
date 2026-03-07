//! Hebrew syllable splitting for stress assignment.
//! Port of Python `phonikud/syllables.py`.

use regex::Regex;
use std::sync::LazyLock;

use super::lexicon::STRESS_PHONEME;

static RE_VOWEL: LazyLock<Regex> = LazyLock::new(|| Regex::new("[aeiou]").unwrap());

/// Split a phoneme string into syllables.
///
/// A syllable boundary is placed before a consonant that precedes a vowel,
/// unless we are at the very start.
///
/// Returns a vector of syllable strings.
pub fn split_syllables(phonemes: &str) -> Vec<String> {
    if phonemes.is_empty() {
        return vec![];
    }

    let chars: Vec<char> = phonemes.chars().collect();
    let len = chars.len();
    let mut syllables: Vec<String> = Vec::new();
    let mut current = String::new();

    let mut i = 0;
    while i < len {
        let c = chars[i];

        // Check if this is a consonant followed by a vowel => new syllable boundary
        // (but not if current syllable is empty, i.e., start of word)
        if !current.is_empty() && is_consonant(c) && has_upcoming_vowel(&chars, i) {
            // Also make sure the current syllable contains a vowel
            // (don't split before we have a vowel in the current syllable)
            if RE_VOWEL.is_match(&current) {
                syllables.push(current);
                current = String::new();
            }
        }

        current.push(c);
        i += 1;
    }

    if !current.is_empty() {
        syllables.push(current);
    }

    syllables
}

/// Check if character at position `i` is followed by a vowel (at i or later).
fn has_upcoming_vowel(chars: &[char], start: usize) -> bool {
    for &c in &chars[start..] {
        if is_vowel(c) {
            return true;
        }
    }
    false
}

fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
}

fn is_consonant(c: char) -> bool {
    // A phoneme character that isn't a vowel and isn't the stress mark
    c.is_alphabetic() && !is_vowel(c) && !STRESS_PHONEME.contains(c)
}

/// Count the number of syllables (= number of vowel nuclei).
pub fn count_syllables(phonemes: &str) -> usize {
    split_syllables(phonemes).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_basic() {
        // "shalom" -> sha-lom
        let syls = split_syllables("shalom");
        assert!(syls.len() >= 2, "Expected at least 2 syllables for 'shalom': {:?}", syls);
    }
}
