//! Utility functions for Hebrew phonemization.
//! Port of relevant parts of Python `phonikud/utils.py`.

use regex::Regex;
use std::sync::LazyLock;
use unicode_normalization::UnicodeNormalization;

use super::letter::Letter;
use super::lexicon::*;

// ---------------------------------------------------------------------------
// Regex patterns (compiled once)
// ---------------------------------------------------------------------------

/// Matches a Hebrew letter followed by optional combining marks / geresh / prefix marker.
static RE_LETTER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\p{L})([\p{M}'|]*)").unwrap());

static RE_VOWEL: LazyLock<Regex> = LazyLock::new(|| Regex::new("[aeiou]").unwrap());

// ---------------------------------------------------------------------------
// normalize / post-processing
// ---------------------------------------------------------------------------

/// NFD-normalize, sort combining marks, apply DEDUPLICATE map.
pub fn normalize(s: &str) -> String {
    // Step 1: NFD decomposition
    let nfd: String = s.nfd().collect();
    // Step 2: Apply deduplication map (Hebrew geresh -> ', maqaf -> -)
    let deduped: String = nfd
        .chars()
        .map(|c| *DEDUPLICATE.get(&c).unwrap_or(&c))
        .collect();
    // Step 3: Sort combining marks within each cluster.
    // We walk the string and whenever we hit a run of combining characters
    // (Unicode category M) we sort them by code-point so that dagesh < shin-dot
    // < nikud etc. in a deterministic order.
    sort_combining_marks(&deduped)
}

/// Sort runs of combining marks by code-point value.
fn sort_combining_marks(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut marks: Vec<char> = Vec::new();

    for ch in s.chars() {
        if is_combining(ch) {
            marks.push(ch);
        } else {
            if !marks.is_empty() {
                marks.sort();
                out.extend(marks.drain(..));
            }
            out.push(ch);
        }
    }
    if !marks.is_empty() {
        marks.sort();
        out.extend(marks.drain(..));
    }
    out
}

fn is_combining(c: char) -> bool {
    // Unicode Combining Diacritical Marks or Hebrew points/accents
    let cp = c as u32;
    (0x0300..=0x036F).contains(&cp)       // Combining Diacritical Marks
        || (0x0591..=0x05BD).contains(&cp) // Hebrew accents + points
        || (0x05BF..=0x05C7).contains(&cp) // More Hebrew points
}

/// Post-normalize phoneme string for TTS consumption:
/// - Remove trailing glottal stop (ʔ)
/// - Remove trailing h
/// - Collapse trailing "ij" to "i"
pub fn post_normalize(phonemes: &str) -> String {
    let mut s = phonemes.to_string();
    // Remove trailing ʔ
    while s.ends_with('\u{0294}') {
        s.pop();
    }
    // Remove trailing h
    while s.ends_with('h') {
        s.pop();
    }
    // Trailing ij -> i
    if s.ends_with("ij") {
        s.truncate(s.len() - 1); // remove the 'j'
    }
    s
}

/// Keep only valid phoneme characters.
pub fn post_clean(phonemes: &str) -> String {
    phonemes
        .chars()
        .filter(|c| SET_PHONEMES.contains(c))
        .collect()
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse a word (already NFD-normalized) into a list of `Letter` structs.
pub fn get_letters(word: &str) -> Vec<Letter> {
    let normalized = normalize(word);
    RE_LETTER
        .captures_iter(&normalized)
        .map(|cap| {
            let ch = cap.get(1).unwrap().as_str();
            let diac = cap.get(2).map_or("", |m| m.as_str());
            Letter::new(ch, diac)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// sort_stress
// ---------------------------------------------------------------------------

/// Move the stress mark (ˈ) to just before the first vowel in the syllable.
///
/// Python equivalent:
/// ```python
/// def sort_stress(syllable):
///     text = "".join(syllable)
///     if "ˈ" not in text or not re.search("[aeiou]", text): return syllable
///     syllable = [p.replace("ˈ", "") for p in syllable]
///     for i, p in enumerate(syllable):
///         syllable[i], n = re.subn(r"([aeiou])", r"ˈ\1", p, 1)
///         if n: break
///     return syllable
/// ```
pub fn sort_stress(syllable: &[String]) -> Vec<String> {
    let text: String = syllable.iter().map(|s| s.as_str()).collect();
    if !text.contains(STRESS_PHONEME) || !RE_VOWEL.is_match(&text) {
        return syllable.to_vec();
    }
    // Remove all stress marks
    let mut parts: Vec<String> = syllable
        .iter()
        .map(|p| p.replace(STRESS_PHONEME, ""))
        .collect();
    // Re-insert stress mark before the first vowel
    let re_insert = Regex::new("([aeiou])").unwrap();
    for part in parts.iter_mut() {
        if RE_VOWEL.is_match(part) {
            let replacement = format!("{}$1", STRESS_PHONEME);
            *part = re_insert.replace(part.as_str(), replacement.as_str()).to_string();
            break;
        }
    }
    parts
}

// ---------------------------------------------------------------------------
// mark_vocal_shva  (simplified: first-letter heuristic)
// ---------------------------------------------------------------------------

/// If the first letter of the word has a plain shva, mark it as vocal shva.
/// This is the simplified prediction used for TTS.
pub fn mark_vocal_shva(letters: &mut [Letter]) {
    if letters.is_empty() {
        return;
    }
    let first = &letters[0];
    // If the first letter has a shva and no other vowel nikud, mark it vocal
    if first.diac.contains(NIKUD_SHVA) {
        // Check it doesn't already have a real vowel
        let has_vowel = first.diac.chars().any(|c| {
            NIKUD_PHONEMES.contains_key(&c) && c != NIKUD_SHVA && c != VOCAL_SHVA_DIACRITIC
        });
        if !has_vowel {
            // Replace shva with vocal shva marker in all_diac and diac
            letters[0].all_diac = letters[0]
                .all_diac
                .replace(NIKUD_SHVA, &VOCAL_SHVA_DIACRITIC.to_string());
            letters[0].diac = letters[0]
                .diac
                .replace(NIKUD_SHVA, &VOCAL_SHVA_DIACRITIC.to_string());
        }
    }
}

// ---------------------------------------------------------------------------
// sort_hatama
// ---------------------------------------------------------------------------

/// Move hatama (stress mark) from a nikud-haser letter to the next letter.
pub fn sort_hatama(letters: &mut Vec<Letter>) {
    let len = letters.len();
    if len < 2 {
        return;
    }
    let mut i = 0;
    while i < len - 1 {
        if letters[i].all_diac.contains(NIKUD_HASER_DIACRITIC)
            && letters[i].all_diac.contains(HATAMA_DIACRITIC)
        {
            // Remove hatama from current letter
            letters[i].all_diac = letters[i]
                .all_diac
                .replace(HATAMA_DIACRITIC, "");
            // Add hatama to next letter
            letters[i + 1].all_diac.push(HATAMA_DIACRITIC);
        }
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// add_milra_hatama  (default stress on last syllable)
// ---------------------------------------------------------------------------

/// If no letter carries a hatama, add one to the last letter that has a vowel nikud.
pub fn add_milra_hatama(letters: &mut [Letter]) {
    // Check if any letter already has hatama
    let has_stress = letters
        .iter()
        .any(|l| l.all_diac.contains(HATAMA_DIACRITIC));
    if has_stress {
        return;
    }
    // Find the last letter with a vowel
    for letter in letters.iter_mut().rev() {
        // Does this letter produce a vowel phoneme?
        let produces_vowel = letter.diac.chars().any(|c| {
            if let Some(ph) = NIKUD_PHONEMES.get(&c) {
                !ph.is_empty() && *ph != STRESS_PHONEME
            } else {
                false
            }
        });

        if produces_vowel {
            letter.all_diac.push(HATAMA_DIACRITIC);
            break;
        }
    }
}
