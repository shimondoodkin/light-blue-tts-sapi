//! Handle Hebrew geresh (׳/') and gershayim (״/") conventions.
//!
//! - Single letter + geresh (׳ or ') alone → expand to Hebrew letter name
//! - Letter + geresh before more letters (foreign sound) → strip geresh
//! - Two letters with gershayim (״ or ") between → expand both to letter names (gematria)


/// Hebrew letter names with nikud for phonikud.
fn letter_name(c: char) -> Option<&'static str> {
    match c {
        'א' => Some("אָלֶף"),
        'ב' => Some("בֵּית"),
        'ג' => Some("גִּימֶל"),
        'ד' => Some("דָּלֶת"),
        'ה' => Some("הֵא"),
        'ו' => Some("וָאו"),
        'ז' => Some("זַ֫יִן"),
        'ח' => Some("חֵית"),
        'ט' => Some("טֵית"),
        'י' => Some("יוֹד"),
        'כ' | 'ך' => Some("כָּף"),
        'ל' => Some("לָ֫מֶד"),
        'מ' | 'ם' => Some("מֵם"),
        'נ' | 'ן' => Some("נוּן"),
        'ס' => Some("סָ֫מֶך"),
        'ע' => Some("עַ֫יִן"),
        'פ' | 'ף' => Some("פֵּא"),
        'צ' | 'ץ' => Some("צָדִי"),
        'ק' => Some("קוֹף"),
        'ר' => Some("רֵישׁ"),
        'ש' => Some("שִׁין"),
        'ת' => Some("תָּו"),
        _ => None,
    }
}

fn is_hebrew_letter(c: char) -> bool {
    ('\u{05D0}'..='\u{05EA}').contains(&c) || c == 'ך' || c == 'ם' || c == 'ן' || c == 'ף' || c == 'ץ'
}

fn is_geresh(c: char) -> bool {
    c == '\u{05F3}' || c == '\'' || c == '\u{2019}' // ׳ or ' or '
}

fn is_gershayim(c: char) -> bool {
    c == '\u{05F4}' || c == '"' || c == '\u{201D}' // ״ or " or "
}

// (regex not needed — we handle it char-by-char below)

/// Process geresh and gershayim in text.
pub fn expand_geresh(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Check for two-letter gershayim: letter + ״ + letter (e.g., ט״ו)
        if i + 2 < len
            && is_hebrew_letter(chars[i])
            && is_gershayim(chars[i + 1])
            && is_hebrew_letter(chars[i + 2])
        {
            // Check it's not part of a longer word
            let before_ok = i == 0 || !is_hebrew_letter(chars[i - 1]);
            let after_ok = i + 3 >= len || !is_hebrew_letter(chars[i + 3]);

            if before_ok && after_ok {
                if let (Some(name1), Some(name2)) = (letter_name(chars[i]), letter_name(chars[i + 2])) {
                    result.push_str(name1);
                    result.push(' ');
                    result.push_str(name2);
                    i += 3;
                    continue;
                }
            }
        }

        // Check for single letter + geresh
        if i + 1 < len && is_hebrew_letter(chars[i]) && is_geresh(chars[i + 1]) {
            let before_ok = i == 0 || !is_hebrew_letter(chars[i - 1]);
            let after_ok = i + 2 >= len || !is_hebrew_letter(chars[i + 2]);

            if before_ok && after_ok {
                // Standalone letter + geresh → letter name
                if let Some(name) = letter_name(chars[i]) {
                    result.push_str(name);
                    i += 2;
                    continue;
                }
            } else if !after_ok {
                // Letter + geresh + more letters (foreign sound like ג׳ירפה) → keep geresh
                // The phonemizer needs it to produce correct foreign phonemes
                // (e.g., ג׳ → dʒ, צ׳ → tʃ, ז׳ → ʒ)
                result.push(chars[i]);
                result.push('\''); // normalize to ASCII apostrophe
                i += 2;
                continue;
            }
        }

        result.push(chars[i]);
        i += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foreign_geresh_kept() {
        // ת׳יל → ת'יל (keep geresh for phonemizer)
        let r = expand_geresh("ת׳יל");
        assert_eq!(r, "ת'יל");
    }

    #[test]
    fn test_standalone_letter_geresh() {
        // א׳ → letter name
        let r = expand_geresh("א׳");
        assert!(r.contains("אָלֶף"), "got: {r}");
    }

    #[test]
    fn test_standalone_letter_geresh_in_sentence() {
        let r = expand_geresh("סעיף א׳ בחוק");
        assert!(r.contains("אָלֶף"), "got: {r}");
    }

    #[test]
    fn test_gershayim_two_letters() {
        // ט״ו → tet vav
        let r = expand_geresh("ט״ו");
        assert!(r.contains("טֵית") && r.contains("וָאו"), "got: {r}");
    }

    #[test]
    fn test_gershayim_in_sentence() {
        let r = expand_geresh("ט״ו בשבט");
        assert!(r.contains("טֵית") && r.contains("וָאו"), "got: {r}");
    }

    #[test]
    fn test_no_change_normal_text() {
        let r = expand_geresh("שלום עולם");
        assert_eq!(r, "שלום עולם");
    }

    #[test]
    fn test_ascii_quote_geresh() {
        // ת'יל with ASCII quote — keep geresh for phonemizer
        let r = expand_geresh("ת'יל");
        assert_eq!(r, "ת'יל");
    }
}
