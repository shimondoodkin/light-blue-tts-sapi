//! Time expansion — convert HH:MM patterns to Hebrew words with nikud.
//!
//! Follows the phonikud expander's time_to_word.py approach.

use regex::Regex;
use std::sync::LazyLock;

use super::numbers::num_to_word;

static TIME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(\d{1,2}):(\d{2})").unwrap()
});

const HOURS: [&str; 13] = [
    "אֶ֫פֶס",
    "אַחַת",
    "שְׁתַּ֫יִם",
    "שָׁלוֹשׁ",
    "אַ֫רְבַּע",
    "חָמֵשׁ",
    "שֵׁשׁ",
    "שֶׁ֫בַע",
    "שְׁמ֫וֹנֶה",
    "תֵּשַׁע",
    "עֶ֫שֶׂר",
    "אַחַת עֶשְׂרֵה",
    "שְׁתֵּ֫ים עֶשְׂרֵה",
];

pub fn expand_times(text: &str) -> String {
    TIME_RE.replace_all(text, |caps: &regex::Captures| {
        let h: u32 = caps[1].parse().unwrap_or(0);
        let m: u32 = caps[2].parse().unwrap_or(0);

        if h > 23 || m > 59 {
            return caps[0].to_string();
        }

        let h12 = if h == 0 { 12 } else if h > 12 { h - 12 } else { h };
        let hour_word = HOURS[h12 as usize];

        if m == 0 {
            return hour_word.to_string();
        }

        let min_word = num_to_word(m as i64);
        format!("{} וֵ{} דַּקּוֹת", hour_word, min_word)
    }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_with_minutes() {
        let r = expand_times("14:30");
        assert!(r.contains("שְׁתַּ֫יִם") && r.contains("דַּקּוֹת"), "got: {r}");
    }

    #[test]
    fn test_time_on_hour() {
        let r = expand_times("8:00");
        assert!(r.contains("שְׁמ֫וֹנֶה"), "got: {r}");
        assert!(!r.contains("דַּקּוֹת"), "got: {r}");
    }

    #[test]
    fn test_midnight() {
        let r = expand_times("0:00");
        // 0 -> 12 -> שתים עשרה
        assert!(r.contains("שְׁתֵּ֫ים"), "got: {r}");
    }
}
