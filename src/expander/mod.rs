//! Text expander — expand dates, numbers, and times into Hebrew words with nikud.
//!
//! Mirrors the structure of phonikud's Python expander:
//! <https://github.com/thewh1teagle/phonikud/tree/main/phonikud/expander>

pub mod dates;
pub mod dictionary;
pub mod hebrew_chars;
pub mod numbers;
pub mod punctuation;
pub mod time;

/// Normalize unicode characters: replace newlines with spaces, normalize
/// various dashes, quotes, and whitespace to ASCII equivalents.
fn normalize_unicode(text: &str) -> String {
    let mut s = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            // Newlines / vertical whitespace → space
            '\n' | '\r' | '\x0B' | '\x0C' => s.push(' '),
            // Non-breaking space, thin space, hair space, em space, en space, etc.
            '\u{00A0}' | '\u{2002}' | '\u{2003}' | '\u{2004}' | '\u{2005}'
            | '\u{2006}' | '\u{2007}' | '\u{2008}' | '\u{2009}' | '\u{200A}'
            | '\u{202F}' | '\u{205F}' | '\u{3000}' => s.push(' '),
            // Zero-width spaces — drop them
            '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}' => {}
            // Em dash, en dash → hyphen
            '\u{2013}' | '\u{2014}' => s.push('-'),
            // Smart quotes → ASCII quotes
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => s.push('\''),
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => s.push('"'),
            // Ellipsis → three dots
            '\u{2026}' => s.push_str("..."),
            // Bullet → hyphen
            '\u{2022}' => s.push('-'),
            // Everything else passes through
            _ => s.push(c),
        }
    }
    // Collapse multiple spaces into one
    let mut result = String::with_capacity(s.len());
    let mut prev_space = false;
    for c in s.chars() {
        if c == ' ' {
            if !prev_space {
                result.push(' ');
            }
            prev_space = true;
        } else {
            prev_space = false;
            result.push(c);
        }
    }
    result.trim().to_string()
}

/// Expand dates, times, and numbers in `text` to Hebrew words (with nikud).
/// This should run before phonemization.
pub fn expand_text(text: &str) -> String {
    let mut result = normalize_unicode(text);
    result = punctuation::expand_punctuation(&result);
    result = hebrew_chars::expand_geresh(&result);
    result = dates::expand_dates(&result);
    result = time::expand_times(&result);
    result = numbers::expand_numbers(&result);
    result
}
