//! Date expansion — convert date patterns to Hebrew words with nikud.
//!
//! Supports YYYY-MM-DD, DD.MM.YYYY, DD/MM/YYYY formats.
//! Month names include nikud from the phonikud expander.

use regex::Regex;
use std::sync::LazyLock;

use super::numbers::num_to_word;

static DATE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(\d{1,4})[./\-](\d{1,2})[./\-](\d{1,4})\b").unwrap()
});

const MONTHS: [&str; 13] = [
    "",
    "יָ֫נוּאָר",       // 1
    "פֶ֫בְרוּאָר",     // 2
    "מֵ֫רְץ",          // 3
    "אֵפְרִיל",        // 4
    "מַאי",             // 5
    "י֫וּנִי",         // 6
    "י֫וּלִי",         // 7
    "א֫וֹגֻסְט",      // 8
    "סֶפְּטֶ֫מְבֶּר",    // 9
    "אוֹקְט֫וֹבֶּר",     // 10
    "נוֹבֶ֫מְבֶּר",      // 11
    "דֶּצֶ֫מְבֶּר",      // 12
];

pub fn expand_dates(text: &str) -> String {
    DATE_RE.replace_all(text, |caps: &regex::Captures| {
        let a: u32 = caps[1].parse().unwrap_or(0);
        let b: u32 = caps[2].parse().unwrap_or(0);
        let c: u32 = caps[3].parse().unwrap_or(0);

        // YYYY-MM-DD (year > 31)
        if a > 31 && (1..=12).contains(&b) && (1..=31).contains(&c) {
            let day = num_to_word(c as i64);
            let month = MONTHS[b as usize];
            let year = num_to_word(a as i64);
            return format!("{} בֵּ{} {}", day, month, year);
        }
        // DD-MM-YYYY (c > 31)
        if c > 31 && (1..=12).contains(&b) && (1..=31).contains(&a) {
            let day = num_to_word(a as i64);
            let month = MONTHS[b as usize];
            let year = num_to_word(c as i64);
            return format!("{} בֵּ{} {}", day, month, year);
        }

        caps[0].to_string()
    }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ymd() {
        let r = expand_dates("2024-01-15");
        assert!(r.contains("יָ֫נוּאָר"), "got: {r}");
    }

    #[test]
    fn test_dmy() {
        let r = expand_dates("15/01/2024");
        assert!(r.contains("יָ֫נוּאָר"), "got: {r}");
    }

    #[test]
    fn test_not_date() {
        assert_eq!(expand_dates("hello"), "hello");
    }
}
