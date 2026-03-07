//! Constants and mappings for Hebrew phonemization.
//! Port of Python `phonikud/lexicon.py`.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

// Non-standard diacritics
/// Meteg – marks vocal shva
pub const VOCAL_SHVA_DIACRITIC: char = '\u{05BD}';
/// Ole – marks stress (hatama)
pub const HATAMA_DIACRITIC: char = '\u{05AB}';
/// Prefix separator
pub const PREFIX_DIACRITIC: char = '|';
/// Masora circle – marks nikud haser
pub const NIKUD_HASER_DIACRITIC: char = '\u{05AF}';
/// English geresh
pub const EN_GERESH: char = '\'';
/// IPA stress mark
pub const STRESS_PHONEME: &str = "\u{02C8}"; // ˈ

// ---------------------------------------------------------------------------
// Nikud code-points (named constants)
// ---------------------------------------------------------------------------
pub const NIKUD_PATAH: char = '\u{05B7}';
pub const NIKUD_KAMATZ: char = '\u{05B8}';
pub const NIKUD_HIRIK: char = '\u{05B4}';
pub const NIKUD_SEGOL: char = '\u{05B6}';
pub const NIKUD_TSERE: char = '\u{05B5}';
pub const NIKUD_HOLAM: char = '\u{05B9}';
pub const NIKUD_KUBUTS: char = '\u{05BB}';
pub const NIKUD_SHVA: char = '\u{05B0}';
pub const NIKUD_VOCAL_SHVA: char = '\u{05BD}'; // same as VOCAL_SHVA_DIACRITIC
pub const NIKUD_HATAF_KAMATZ: char = '\u{05B3}';
pub const NIKUD_DAGESH: char = '\u{05BC}';
pub const NIKUD_SIN: char = '\u{05C2}';
pub const NIKUD_HATAMA: char = '\u{05AB}'; // same as HATAMA_DIACRITIC
pub const NIKUD_VAV_HOLAM: char = '\u{05BA}';

// ---------------------------------------------------------------------------
// Static lookup tables
// ---------------------------------------------------------------------------

pub static MODERN_SCHEMA: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("x", "\u{03C7}"); // χ
    m.insert("r", "\u{0281}"); // ʁ
    m.insert("g", "\u{0261}"); // ɡ
    m
});

pub static GERESH_PHONEMES: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("\u{05D2}", "d\u{0292}"); // ג -> dʒ
    m.insert("\u{05D6}", "\u{0292}");   // ז -> ʒ
    m.insert("\u{05EA}", "ta");          // ת -> ta
    m.insert("\u{05E6}", "t\u{0283}");  // צ -> tʃ
    m.insert("\u{05E5}", "t\u{0283}");  // ץ -> tʃ
    m
});

pub static LETTERS_PHONEMES: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("\u{05D0}", "\u{0294}"); // א -> ʔ
    m.insert("\u{05D1}", "v");         // ב
    m.insert("\u{05D2}", "g");         // ג
    m.insert("\u{05D3}", "d");         // ד
    m.insert("\u{05D4}", "h");         // ה
    m.insert("\u{05D5}", "v");         // ו
    m.insert("\u{05D6}", "z");         // ז
    m.insert("\u{05D7}", "x");         // ח
    m.insert("\u{05D8}", "t");         // ט
    m.insert("\u{05D9}", "j");         // י
    m.insert("\u{05DA}", "x");         // ך
    m.insert("\u{05DB}", "x");         // כ
    m.insert("\u{05DC}", "l");         // ל
    m.insert("\u{05DD}", "m");         // ם
    m.insert("\u{05DE}", "m");         // מ
    m.insert("\u{05DF}", "n");         // ן
    m.insert("\u{05E0}", "n");         // נ
    m.insert("\u{05E1}", "s");         // ס
    m.insert("\u{05E2}", "\u{0294}"); // ע -> ʔ
    m.insert("\u{05E4}", "f");         // פ
    m.insert("\u{05E3}", "f");         // ף
    m.insert("\u{05E5}", "ts");        // ץ
    m.insert("\u{05E6}", "ts");        // צ
    m.insert("\u{05E7}", "k");         // ק
    m.insert("\u{05E8}", "r");         // ר
    m.insert("\u{05E9}", "\u{0283}"); // ש -> ʃ
    m.insert("\u{05EA}", "t");         // ת
    // Beged Kefet (consonant + dagesh)
    m.insert("\u{05D1}\u{05BC}", "b"); // בּ
    m.insert("\u{05DB}\u{05BC}", "k"); // כּ
    m.insert("\u{05E4}\u{05BC}", "p"); // פּ
    // Shin / Sin dots
    m.insert("\u{05E9}\u{05C1}", "\u{0283}"); // שׁ -> ʃ
    m.insert("\u{05E9}\u{05C2}", "s");         // שׂ -> s
    // Geresh
    m.insert("'", "");
    m
});

pub static NIKUD_PHONEMES: LazyLock<HashMap<char, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert('\u{05B4}', "i");  // Hiriq
    m.insert('\u{05B1}', "e");  // Hataf segol
    m.insert('\u{05B5}', "e");  // Tsere
    m.insert('\u{05B6}', "e");  // Segol
    m.insert('\u{05B2}', "a");  // Hataf Patah
    m.insert('\u{05B7}', "a");  // Patah
    m.insert('\u{05C7}', "o");  // Kamatz katan
    m.insert('\u{05B9}', "o");  // Holam
    m.insert('\u{05BA}', "o");  // Holam haser for vav
    m.insert('\u{05BB}', "u");  // Qubuts
    m.insert('\u{05B3}', "o");  // Hataf qamats
    m.insert('\u{05B8}', "a");  // Kamatz
    m.insert(HATAMA_DIACRITIC, STRESS_PHONEME); // ˈ
    m.insert(VOCAL_SHVA_DIACRITIC, "e"); // vocal shva
    m
});

pub static PUNCTUATION: LazyLock<HashSet<char>> = LazyLock::new(|| {
    let mut s = HashSet::new();
    for c in ['.', ',', '!', '?', ' '] {
        s.insert(c);
    }
    s
});

pub static DEDUPLICATE: LazyLock<HashMap<char, char>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert('\u{05F3}', '\''); // Hebrew geresh -> ASCII apostrophe
    m.insert('\u{05BE}', '-');  // maqaf -> hyphen
    m
});

/// The set of characters that are valid in phoneme output.
pub static SET_PHONEMES: LazyLock<HashSet<char>> = LazyLock::new(|| {
    let mut s = HashSet::new();
    // Collect all characters from NIKUD_PHONEMES values + LETTERS_PHONEMES values + stress
    for v in NIKUD_PHONEMES.values() {
        for c in v.chars() {
            s.insert(c);
        }
    }
    for v in LETTERS_PHONEMES.values() {
        for c in v.chars() {
            s.insert(c);
        }
    }
    for v in GERESH_PHONEMES.values() {
        for c in v.chars() {
            s.insert(c);
        }
    }
    for v in MODERN_SCHEMA.values() {
        for c in v.chars() {
            s.insert(c);
        }
    }
    for c in STRESS_PHONEME.chars() {
        s.insert(c);
    }
    // Basic vowels just in case
    for c in "aeiou".chars() {
        s.insert(c);
    }
    s
});

/// Regex pattern matching Patah or Kamatz: [\u05B7\u05B8]
pub const NIKUD_PATAH_LIKE_PATTERN: &str = "[\u{05B7}\u{05B8}]";
