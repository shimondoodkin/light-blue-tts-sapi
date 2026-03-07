//! Expand punctuation marks (brackets, quotes) into spoken Hebrew words.

/// Replace brackets and quotes with Hebrew spoken equivalents.
pub fn expand_punctuation(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let c = chars[i];
        match c {
            '[' => result.push_str(" בסוגריים מרובעים "),
            ']' => result.push_str(" סוגר סוגריים מרובעים "),
            '(' => result.push_str(" בסוגריים "),
            ')' => result.push_str(" סוגר סוגריים "),
            '{' => result.push_str(" בסוגריים מסולסלים "),
            '}' => result.push_str(" סוגר סוגריים מסולסלים "),
            '"' => {
                // Determine open vs close: if followed by non-space, it's opening
                if i + 1 < len && !chars[i + 1].is_whitespace() {
                    result.push_str(" בגרשיים ");
                } else {
                    result.push_str(" סוגר גרשיים ");
                }
            }
            _ => result.push(c),
        }
        i += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_brackets() {
        let result = expand_punctuation("שלום [עולם] טוב");
        assert!(result.contains("בסוגריים מרובעים"));
        assert!(result.contains("סוגר סוגריים מרובעים"));
    }

    #[test]
    fn test_parens() {
        let result = expand_punctuation("(שלום)");
        assert!(result.contains("בסוגריים"));
        assert!(result.contains("סוגר סוגריים"));
    }

    #[test]
    fn test_quotes() {
        let result = expand_punctuation("הוא אמר \"שלום\" לכולם");
        assert!(result.contains("בגרשיים"));
        assert!(result.contains("סוגר גרשיים"));
    }

    #[test]
    fn test_no_punctuation() {
        assert_eq!(expand_punctuation("שלום עולם"), "שלום עולם");
    }
}
