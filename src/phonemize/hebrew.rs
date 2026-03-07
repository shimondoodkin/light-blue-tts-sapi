//! Core FST for Hebrew-to-IPA phonemization.
//! Port of Python `phonikud/hebrew.py`.

use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

use super::letter::Letter;
use super::lexicon::*;
use super::utils::sort_stress;

// ---------------------------------------------------------------------------
// Short aliases matching the Python code
// ---------------------------------------------------------------------------
static _D: LazyLock<String> = LazyLock::new(|| NIKUD_DAGESH.to_string());
static _SH: LazyLock<String> = LazyLock::new(|| NIKUD_SHVA.to_string());
static _HO: LazyLock<String> = LazyLock::new(|| NIKUD_HOLAM.to_string());
static _HI: LazyLock<String> = LazyLock::new(|| NIKUD_HIRIK.to_string());
static _KA: LazyLock<String> = LazyLock::new(|| NIKUD_KAMATZ.to_string());
static _PA: LazyLock<String> = LazyLock::new(|| NIKUD_PATAH.to_string());
static _TS: LazyLock<String> = LazyLock::new(|| NIKUD_TSERE.to_string());
static _SE: LazyLock<String> = LazyLock::new(|| NIKUD_SEGOL.to_string());
static _KU: LazyLock<String> = LazyLock::new(|| NIKUD_KUBUTS.to_string());
static _VS: LazyLock<String> = LazyLock::new(|| NIKUD_VOCAL_SHVA.to_string());
static _SI: LazyLock<String> = LazyLock::new(|| NIKUD_SIN.to_string());
static _HK: LazyLock<String> = LazyLock::new(|| NIKUD_HATAF_KAMATZ.to_string());

static PAT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(NIKUD_PATAH_LIKE_PATTERN).unwrap());

/// Patah gnuva: final patah under guttural letters
static GNUVA: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("\u{05D7}", "ax"); // ח
    m.insert("\u{05D4}", "ah"); // ה
    m.insert("\u{05E2}", "a");  // ע
    m
});

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Filter out empty strings and strings containing non-phoneme characters.
fn clean(out: &[String]) -> Vec<String> {
    out.iter()
        .filter(|p| {
            !p.is_empty() && p.chars().all(|c| SET_PHONEMES.contains(&c))
        })
        .cloned()
        .collect()
}

/// Get vowel phonemes from all diacritics on a letter.
fn vowels(cur: &Letter) -> Vec<String> {
    cur.all_diac
        .chars()
        .filter_map(|n| {
            NIKUD_PHONEMES.get(&n).map(|s| s.to_string())
        })
        .collect()
}

/// Get stress mark if this letter carries hatama.
fn stress(cur: &Letter) -> Vec<String> {
    if cur.all_diac.contains(HATAMA_DIACRITIC) {
        vec![STRESS_PHONEME.to_string()]
    } else {
        vec![]
    }
}

/// Build output: consonant phoneme + vowel phonemes, cleaned and stress-sorted.
/// Returns (phonemes, skip_count).
fn out(cur: &Letter, con: &str, vow: Option<Vec<String>>, skip: usize) -> (Vec<String>, usize) {
    let mut result: Vec<String> = Vec::new();
    if !con.is_empty() {
        result.push(con.to_string());
    }
    match vow {
        Some(v) => result.extend(v),
        None => result.extend(vowels(cur)),
    }
    (clean(&sort_stress(&result)), skip)
}

/// Determine vowel sound for vav based on its diacritics.
fn vav_vowel(d: &str) -> Option<&'static str> {
    if PAT_RE.is_match(d) {
        return Some("va");
    }
    if d.contains(_TS.as_str()) || d.contains(_SE.as_str()) || d.contains(_VS.as_str()) {
        return Some("ve");
    }
    if d.contains(_HO.as_str()) {
        return Some("o");
    }
    if d.contains(_KU.as_str()) || d.contains(_D.as_str()) {
        return Some("u");
    }
    if d.contains(_HI.as_str()) {
        return Some("vi");
    }
    None
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Phonemize a single word given as a list of `Letter` structs.
pub fn phonemize_word(letters: &[Letter]) -> Vec<String> {
    let mut phonemes: Vec<String> = Vec::new();
    let mut i = 0;
    let len = letters.len();

    while i < len {
        let prev = if i > 0 { Some(&letters[i - 1]) } else { None };
        let nxt = if i + 1 < len {
            Some(&letters[i + 1])
        } else {
            None
        };
        let (p, skip) = letter(&letters[i], prev, nxt);
        phonemes.extend(p);
        i += skip + 1;
    }
    phonemes
}

// ---------------------------------------------------------------------------
// FST transitions
// ---------------------------------------------------------------------------

fn letter(cur: &Letter, prev: Option<&Letter>, nxt: Option<&Letter>) -> (Vec<String>, usize) {
    let d = &cur.diac;
    let ch = cur.char_.as_str();
    let s = stress(cur);

    // Nikud haser – skip this letter entirely
    if cur.all_diac.contains(NIKUD_HASER_DIACRITIC) {
        return (vec![], 0);
    }

    // Geresh
    if d.contains('\'') {
        if let Some(gph) = GERESH_PHONEMES.get(ch) {
            let vow = if ch == "\u{05EA}" {
                // ת with geresh: no extra vowel
                Some(vec![])
            } else {
                None
            };
            return out(cur, gph, vow, 0);
        }
    }

    // Dagesh with special consonant mapping (beged kefet)
    if d.contains(_D.as_str()) {
        let key = format!("{}{}", ch, *_D);
        if let Some(ph) = LETTERS_PHONEMES.get(key.as_str()) {
            return out(cur, ph, None, 0);
        }
    }

    // Vav
    if ch == "\u{05D5}" && !cur.all_diac.contains(NIKUD_HASER_DIACRITIC) {
        return vav(cur, prev, nxt);
    }

    // Shin
    if ch == "\u{05E9}" {
        return shin(cur, prev, nxt);
    }

    // Patah gnuva (final patah under guttural)
    if nxt.is_none() && d.contains(_PA.as_str()) {
        if let Some(gph) = GNUVA.get(ch) {
            return out(cur, gph, Some(s), 0);
        }
    }

    // Kamatz before hataf-kamatz on next letter -> "o"
    if d.contains(_KA.as_str()) {
        if let Some(next) = nxt {
            if next.diac.contains(_HK.as_str()) {
                let mut vow = vec!["o".to_string()];
                vow.extend(s);
                let con = LETTERS_PHONEMES.get(ch).copied().unwrap_or("");
                return out(cur, con, Some(vow), 0);
            }
        }
    }

    // Silent alef (no diacritics, between other letters, next is not vav)
    if ch == "\u{05D0}" && d.is_empty() && prev.is_some() {
        if let Some(next) = nxt {
            if next.char_ != "\u{05D5}" {
                return out(cur, "", None, 0);
            }
        }
    }

    // Silent yod
    if ch == "\u{05D9}" && d.is_empty() && prev.is_some() {
        if let Some(next) = nxt {
            let prev_u = prev.unwrap();
            let prev_char_diac = format!("{}{}", prev_u.char_, prev_u.diac);
            // Not after אֵ and not before vav-with-diacritics (unless vav has shva)
            if prev_char_diac != "\u{05D0}\u{05B5}" {
                let next_is_vav_with_diac = next.char_ == "\u{05D5}"
                    && !next.diac.is_empty()
                    && !next.diac.contains(_SH.as_str());
                if !next_is_vav_with_diac {
                    return out(cur, "", None, 0);
                }
            }
        }
    }

    // Default: look up consonant
    let con = LETTERS_PHONEMES.get(ch).copied().unwrap_or("");
    out(cur, con, None, 0)
}

// ---------------------------------------------------------------------------
// Shin handler
// ---------------------------------------------------------------------------

fn shin(cur: &Letter, prev: Option<&Letter>, nxt: Option<&Letter>) -> (Vec<String>, usize) {
    // Sin dot present
    if cur.diac.contains(_SI.as_str()) {
        // Special case: sin + next shin without diacritics + patah-like on current
        if let Some(next) = nxt {
            if next.char_ == "\u{05E9}" && next.diac.is_empty() && PAT_RE.is_match(&cur.diac) {
                return out(cur, "sa", Some(stress(cur)), 1);
            }
        }
        return out(cur, "s", None, 0);
    }

    // No diacritics on current shin, but previous had sin dot
    if cur.diac.is_empty() {
        if let Some(p) = prev {
            if p.diac.contains(_SI.as_str()) {
                return out(cur, "s", None, 0);
            }
        }
    }

    // Default shin -> ʃ
    let con = LETTERS_PHONEMES.get("\u{05E9}").copied().unwrap_or("");
    out(cur, con, None, 0)
}

// ---------------------------------------------------------------------------
// Vav handler
// ---------------------------------------------------------------------------

fn vav(cur: &Letter, prev: Option<&Letter>, nxt: Option<&Letter>) -> (Vec<String>, usize) {
    let d = &cur.diac;
    let s = stress(cur);

    // After shva: vav + holam -> "vo"
    if let Some(p) = prev {
        if p.diac.contains(_SH.as_str()) && d.contains(_HO.as_str()) {
            return out(cur, "vo", Some(s), 0);
        }
    }

    // Double vav
    if let Some(next) = nxt {
        if next.char_ == "\u{05D5}" {
            let dd = format!("{}{}", d, next.diac);
            if dd.contains(_HO.as_str()) {
                return out(cur, "vo", Some(s), 1);
            }
            if *d == next.diac {
                return out(cur, "vu", Some(s), 1);
            }
            if let Some(v) = vav_vowel(d) {
                return out(cur, v, Some(s), 0);
            }
            if d.contains(_SH.as_str()) && next.diac.is_empty() {
                return out(cur, "v", Some(s), 0);
            }
            return out(cur, "", Some(s), 0);
        }
    }

    // Single vav with vowel
    if let Some(v) = vav_vowel(d) {
        return out(cur, v, Some(s), 0);
    }

    // Word-initial vav with shva
    if d.contains(_SH.as_str()) && prev.is_none() {
        return out(cur, "ve", Some(s), 0);
    }

    // Vav with no diacritics before another letter -> silent
    if nxt.is_some() && d.is_empty() {
        return out(cur, "", Some(s), 0);
    }

    // Default: consonantal "v"
    out(cur, "v", Some(s), 0)
}
