//! Number expansion — convert digits to Hebrew words with nikud.
//!
//! Handles integers, decimals, negative numbers, and Hebrew prefix letters
//! (ב,ו,ה,ל,מ,כ,ש) attached to numbers.
//!
//! Number words include nikud from the phonikud number_names dictionary so
//! that phonikud can skip diacritization and phonemize directly.

/// Expand numbers in text word-by-word (like phonikud's approach).
pub fn expand_numbers(text: &str) -> String {
    let mut words = Vec::new();
    for word in text.split(' ') {
        words.push(try_expand_number(word));
    }
    words.join(" ")
}

/// Try to expand a single word as a number. Returns the original word if not a number.
fn try_expand_number(word: &str) -> String {
    if word.is_empty() {
        return word.to_string();
    }

    let mut prefix = String::new();
    let mut num_part = word;

    // Strip Hebrew prefix letters (ב,ו,ה,ל,מ,כ,ש) followed by optional hyphen.
    // Supports multi-letter prefixes like מכ (from-about), שב (that-in), לכ (to-about).
    let chars: Vec<char> = word.chars().collect();
    let prefix_chars = "בוהלמכש";
    let mut prefix_len = 0;
    for &ch in &chars {
        if prefix_chars.contains(ch) {
            prefix_len += 1;
        } else {
            break;
        }
    }
    if prefix_len > 0 && prefix_len < chars.len() {
        let after_prefix = &chars[prefix_len..];
        if after_prefix[0] == '-' && after_prefix.len() > 1 && after_prefix[1].is_ascii_digit() {
            prefix = chars[..prefix_len].iter().collect();
            let byte_offset: usize = chars[..prefix_len].iter().map(|c| c.len_utf8()).sum::<usize>() + 1; // +1 for hyphen
            num_part = &word[byte_offset..];
        } else if after_prefix[0].is_ascii_digit() {
            prefix = chars[..prefix_len].iter().collect();
            let byte_offset: usize = chars[..prefix_len].iter().map(|c| c.len_utf8()).sum();
            num_part = &word[byte_offset..];
        }
    }

    // Strip trailing punctuation before decimal parsing, so `1.` is treated
    // as the integer 1 plus sentence punctuation rather than a malformed decimal.
    let trimmed_len = num_part
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_ascii_digit())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .unwrap_or(0);
    if trimmed_len == 0 {
        return word.to_string();
    }
    let suffix = num_part[trimmed_len..].to_string();
    num_part = &num_part[..trimmed_len];

    // Check if what remains is a number (possibly negative or decimal)
    let (is_neg, abs_part) = if let Some(stripped) = num_part.strip_prefix('-') {
        (true, stripped)
    } else {
        (false, num_part)
    };

    // Must start with a digit
    if abs_part.is_empty() || !abs_part.chars().next().unwrap().is_ascii_digit() {
        return word.to_string();
    }

    // Decimal
    if abs_part.contains('.') || abs_part.contains(',') {
        let parts: Vec<&str> = abs_part.split(&['.', ','][..]).collect();
        if parts.len() == 2 {
            if let (Ok(int_val), Ok(dec_val)) = (parts[0].parse::<i64>(), parts[1].parse::<i64>()) {
                let neg = if is_neg { "מִ֫ינּוּס " } else { "" };
                return format!("{}{}{} נְֽקֻדָּה {}{}", prefix, neg, num_to_word(int_val), num_to_word(dec_val), suffix);
            }
        }
        return word.to_string();
    }

    // Integer
    if let Ok(n) = abs_part.parse::<i64>() {
        let neg = if is_neg { "מִ֫ינּוּס " } else { "" };
        format!("{}{}{}{}", prefix, neg, num_to_word(n), suffix)
    } else {
        word.to_string()
    }
}

/// Convert a number to Hebrew words with nikud.
pub fn num_to_word(n: i64) -> String {
    let n = n.unsigned_abs();
    if n == 0 {
        return "אֶ֫פֶס".to_string();
    }
    if n > 999_999_999_999 {
        return n.to_string().chars().map(digit_to_word).collect::<Vec<_>>().join(" ");
    }

    let mut parts = Vec::new();

    let billions = n / 1_000_000_000;
    let millions = (n % 1_000_000_000) / 1_000_000;
    let thousands = (n % 1_000_000) / 1000;
    let remainder = n % 1000;

    if billions > 0 {
        if billions == 1 {
            parts.push("מִילְיַארְד".to_string());
        } else {
            parts.push(format!("{} מִילְיַארְד", hundreds_to_word(billions)));
        }
    }

    if millions > 0 {
        if millions == 1 {
            parts.push("מִילְיוֹן".to_string());
        } else {
            parts.push(format!("{} מִילְיוֹן", hundreds_to_word(millions)));
        }
    }

    if thousands > 0 {
        parts.push(thousands_to_word(thousands));
    }

    if remainder > 0 {
        parts.push(hundreds_to_word(remainder));
    }

    parts.join(" ")
}

fn hundreds_to_word(n: u64) -> String {
    let mut parts = Vec::new();
    let h = n / 100;
    let tu = n % 100;

    if h > 0 {
        parts.push(match h {
            1 => "מֵ֫אָה",
            2 => "מָאתַ֫יִם",
            3 => "שְׁלוֹשׁמֵאוֹת",
            4 => "אַרְבַּעמֵאוֹת",
            5 => "חָמֵשׁמֵאוֹת",
            6 => "שֵׁשׁמֵאוֹת",
            7 => "שֶׁ֫בַעמֵאוֹת",
            8 => "שְׁמ֫וֹנֶהמֵאוֹת",
            9 => "תֵּשַׁעמֵאוֹת",
            _ => unreachable!(),
        }.to_string());
    }

    if tu > 0 {
        parts.push(tens_units_to_word(tu));
    }

    parts.join(" ")
}

fn thousands_to_word(n: u64) -> String {
    match n {
        1  => "אֶ֫לֶף".to_string(),
        2  => "אַלְפַּאיִם".to_string(),
        3  => "שְׁלֹ֫שֶׁתאֲלָפִים".to_string(),
        4  => "אַרְבַּ֫עַתאֲלָפִים".to_string(),
        5  => "חֲמֵ֫שֶׁתאֲלָפִים".to_string(),
        6  => "שֵׁ֫שֶׁתאֲלָפִים".to_string(),
        7  => "שִׁבְעַ֫תאֲלָפִים".to_string(),
        8  => "שְׁמוֹנַ֫תאֲלָפִים".to_string(),
        9  => "תִּשְׁעַ֫תאֲלָפִים".to_string(),
        10 => "עֲשֶׂ֫רֶתאֲלָפִים".to_string(),
        _  => format!("{} אֶ֫לֶף", hundreds_to_word(n)),
    }
}

fn tens_units_to_word(n: u64) -> String {
    match n {
        1  => "אַחַת",
        2  => "שְׁתַּ֫יִם",
        3  => "שָׁלוֹשׁ",
        4  => "אַ֫רְבַּע",
        5  => "חָמֵשׁ",
        6  => "שֵׁשׁ",
        7  => "שֶׁ֫בַע",
        8  => "שְׁמ֫וֹנֶה",
        9  => "תֵּשַׁע",
        10 => "עֶ֫שֶׂר",
        11 => "אַחַתעֶשְׂרֵה",
        12 => "שְׁתֵּ֫ימעֶשְׂרֵה",
        13 => "שְׁלוֹשׁעֶשְׂרֵה",
        14 => "אַרְבַּععֶשְׂרֵה",
        15 => "חָמֵשׁעֶשְׂרֵה",
        16 => "שֵׁשׁעֶשְׂרֵה",
        17 => "שֶׁ֫בַעעֶשְׂרֵה",
        18 => "שְׁמ֫וֹנֶהעֶשְׂרֵה",
        19 => "תֵּשַׁעעֶשְׂרֵה",
        20 => "עֶשְׂרִ֫ים",
        30 => "שְׁלוֹשִׁים",
        40 => "אַרְבָּעִים",
        50 => "חֲמִשִּׁים",
        60 => "שִׁשִּׁים",
        70 => "שִׁבְעִים",
        80 => "שְׁמוֹנִים",
        90 => "תִּשְׁעִים",
        _ => {
            let tens = (n / 10) * 10;
            let units = n % 10;
            return format!("{} וֵ{}", tens_units_to_word(tens), tens_units_to_word(units));
        }
    }.to_string()
}

fn digit_to_word(c: char) -> &'static str {
    match c {
        '0' => "אֶ֫פֶס",
        '1' => "אַחַת",
        '2' => "שְׁתַּ֫יִם",
        '3' => "שָׁלוֹשׁ",
        '4' => "אַ֫רְבַּע",
        '5' => "חָמֵשׁ",
        '6' => "שֵׁשׁ",
        '7' => "שֶׁ֫בַע",
        '8' => "שְׁמ֫וֹנֶה",
        '9' => "תֵּשַׁע",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_number_formats() {
        for n in [15, 22, 92, 300, 512, 1992, 3000, 5000, 10000, 20000] {
            println!("{n} => {}", num_to_word(n));
        }
    }

    #[test]
    fn test_zero() { assert!(num_to_word(0).contains("אֶ֫פֶס")); }

    #[test]
    fn test_single() { assert!(num_to_word(3).contains("שָׁלוֹשׁ")); }

    #[test]
    fn test_teens() {
        let r = num_to_word(15);
        assert!(r.contains("חָמֵשׁ") && r.contains("עֶשְׂרֵה"), "got: {r}");
    }

    #[test]
    fn test_thousands() { assert!(num_to_word(2000).contains("אַלְפַּאיִם")); }

    #[test]
    fn test_expand_in_text() {
        let r = expand_numbers("יש לי 3 חתולים");
        assert!(r.contains("שָׁלוֹשׁ") && r.contains("חתולים"), "got: {r}");
    }

    #[test]
    fn test_decimal() {
        assert!(expand_numbers("3.5").contains("נְֽקֻדָּה"));
    }

    #[test]
    fn test_negative() {
        let r = expand_numbers("-5");
        assert!(r.contains("מִ֫ינּוּס") && r.contains("חָמֵשׁ"), "got: {r}");
    }

    #[test]
    fn test_prefix() {
        let r = expand_numbers("ו15");
        assert!(r.contains("ו") && r.contains("חָמֵשׁ"), "got: {r}");
    }

    #[test]
    fn test_hyphen_prefix_not_negative() {
        // ו-15 should be "ו" + "15" (not minus 15)
        let r = expand_numbers("ו-15");
        assert!(!r.contains("מִ֫ינּוּס"), "should not be negative, got: {r}");
        assert!(r.contains("חָמֵשׁ"), "got: {r}");
    }

    #[test]
    fn test_multi_letter_prefix() {
        // מכ-20 = "from about 20"
        let r = expand_numbers("מכ-20");
        assert!(r.contains("עֶשְׂרִ֫ים"), "should contain 'esrim' (20), got: {r}");
        assert!(r.starts_with("מכ"), "should keep prefix, got: {r}");
    }

    #[test]
    fn test_prefix_number_with_sentence_punctuation() {
        let r = expand_numbers("מ-0 ל-1.");
        assert!(r.contains("אֶ֫פֶס"), "got: {r}");
        assert!(r.contains("אַחַת."), "got: {r}");
    }
}
