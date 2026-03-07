//! Simple rule-based English grapheme-to-phoneme converter.
//!
//! Produces IPA phonemes from the Hebrew TTS model's vocabulary:
//!   a e i o u b v d h z χ t j k l m n s f p w ʔ ɡ ʁ ʃ ʒ ˈ
//!
//! This is a *rough* approximation — English phonemes that don't exist in
//! the Hebrew model are mapped to their closest equivalents.

/// Convert English text to approximate IPA using only the Hebrew model's phoneme set.
pub fn phonemize_english(text: &str) -> String {
    let mut result = String::new();
    let words: Vec<&str> = text.split_whitespace().collect();

    for (i, word) in words.iter().enumerate() {
        if i > 0 {
            result.push(' ');
        }
        let lower = word.to_lowercase();
        let clean: String = lower.chars().filter(|c| c.is_ascii_alphabetic()).collect();
        // Preserve trailing punctuation
        let trailing_punct: String = lower.chars().rev().take_while(|c| !c.is_ascii_alphabetic()).collect::<String>().chars().rev().collect();

        if clean.is_empty() {
            result.push_str(&trailing_punct);
            continue;
        }

        let phonemes = word_to_phonemes(&clean);
        result.push_str(&phonemes);
        result.push_str(&trailing_punct);
    }

    result
}

/// Convert a single lowercase English word to approximate IPA.
fn word_to_phonemes(word: &str) -> String {
    // Check the small dictionary of common words first
    if let Some(ipa) = lookup_common(word) {
        return ipa.to_string();
    }

    // Fall back to rule-based grapheme-to-phoneme
    rules_g2p(word)
}

/// Small dictionary of common English words with hand-tuned IPA
/// (using only Hebrew model phonemes).
fn lookup_common(word: &str) -> Option<&'static str> {
    Some(match word {
        // Articles / pronouns / prepositions
        "a" => "e",
        "an" => "en",
        "the" => "de",
        "i" => "aj",
        "you" => "ju",
        "he" => "hi",
        "she" => "ʃi",
        "it" => "it",
        "we" => "vi",
        "they" => "dej",
        "me" => "mi",
        "him" => "him",
        "her" => "heʁ",
        "us" => "as",
        "them" => "dem",
        "my" => "maj",
        "your" => "joʁ",
        "his" => "hiz",
        "our" => "auʁ",
        "their" => "deʁ",
        "this" => "dis",
        "that" => "dat",
        "these" => "diz",
        "those" => "douz",
        "is" => "iz",
        "am" => "am",
        "are" => "aʁ",
        "was" => "voz",
        "were" => "veʁ",
        "be" => "bi",
        "been" => "bin",
        "being" => "biinɡ",
        "have" => "hav",
        "has" => "haz",
        "had" => "had",
        "do" => "du",
        "does" => "daz",
        "did" => "did",
        "will" => "vil",
        "would" => "vud",
        "could" => "kud",
        "should" => "ʃud",
        "can" => "kan",
        "may" => "mej",
        "might" => "majt",
        "must" => "mast",
        "shall" => "ʃal",
        "not" => "not",
        "no" => "no",
        "yes" => "jes",
        "and" => "and",
        "or" => "oʁ",
        "but" => "bat",
        "if" => "if",
        "of" => "ov",
        "in" => "in",
        "on" => "on",
        "at" => "at",
        "to" => "tu",
        "for" => "foʁ",
        "with" => "vid",
        "from" => "fʁom",
        "by" => "baj",
        "up" => "ap",
        "out" => "aut",
        "about" => "ebaut",
        "into" => "intu",
        "over" => "ouveʁ",
        "after" => "afteʁ",
        "before" => "bifoʁ",
        "between" => "bitvin",
        "under" => "andeʁ",

        // Common words
        "hello" => "helo",
        "hi" => "haj",
        "hey" => "hej",
        "good" => "ɡud",
        "bad" => "bad",
        "great" => "ɡʁejt",
        "well" => "vel",
        "here" => "hiʁ",
        "there" => "deʁ",
        "where" => "veʁ",
        "when" => "ven",
        "what" => "vot",
        "who" => "hu",
        "how" => "hau",
        "why" => "vaj",
        "all" => "ol",
        "each" => "itʃ",
        "every" => "evʁi",
        "both" => "bot",
        "few" => "fju",
        "more" => "moʁ",
        "most" => "moust",
        "other" => "adeʁ",
        "some" => "sam",
        "such" => "satʃ",
        "than" => "dan",
        "too" => "tu",
        "very" => "veʁi",
        "just" => "dʒast",
        "also" => "olso",
        "now" => "nau",
        "then" => "den",
        "so" => "so",
        "like" => "lajk",
        "time" => "tajm",
        "day" => "dej",
        "way" => "vej",
        "world" => "veʁld",
        "life" => "lajf",
        "man" => "man",
        "woman" => "vuman",
        "child" => "tʃajld",
        "children" => "tʃildʁen",
        "people" => "pipel",
        "name" => "nejm",
        "home" => "houm",
        "house" => "haus",
        "water" => "voteʁ",
        "food" => "fud",
        "work" => "veʁk",
        "money" => "mani",
        "year" => "jiʁ",
        "years" => "jiʁz",
        "new" => "nju",
        "old" => "ould",
        "big" => "biɡ",
        "small" => "smol",
        "long" => "lonɡ",
        "first" => "feʁst",
        "last" => "last",
        "next" => "nekst",
        "right" => "ʁajt",
        "left" => "left",
        "high" => "haj",
        "low" => "lou",
        "come" => "kam",
        "go" => "ɡo",
        "get" => "ɡet",
        "make" => "mejk",
        "know" => "no",
        "think" => "tink",
        "take" => "tejk",
        "see" => "si",
        "look" => "luk",
        "want" => "vont",
        "give" => "ɡiv",
        "use" => "juz",
        "find" => "fajnd",
        "tell" => "tel",
        "ask" => "ask",
        "say" => "sej",
        "said" => "sed",
        "try" => "tʁaj",
        "need" => "nid",
        "feel" => "fil",
        "leave" => "liv",
        "call" => "kol",
        "keep" => "kip",
        "let" => "let",
        "put" => "put",
        "show" => "ʃou",
        "turn" => "teʁn",
        "start" => "staʁt",
        "stop" => "stop",
        "open" => "oupen",
        "close" => "klouz",
        "run" => "ʁan",
        "play" => "plej",
        "move" => "muv",
        "live" => "liv",
        "love" => "lav",
        "read" => "ʁid",
        "write" => "ʁajt",
        "speak" => "spik",
        "talk" => "tok",
        "help" => "help",
        "thank" => "tank",
        "thanks" => "tanks",
        "please" => "pliz",
        "sorry" => "soʁi",
        "ok" => "okej",
        "okay" => "okej",
        "one" => "van",
        "two" => "tu",
        "three" => "tʁi",
        "four" => "foʁ",
        "five" => "fajv",
        "six" => "siks",
        "seven" => "seven",
        "eight" => "ejt",
        "nine" => "najn",
        "ten" => "ten",
        "hundred" => "handʁed",
        "thousand" => "tauzend",
        "test" => "test",
        "testing" => "testinɡ",
        "text" => "tekst",
        "voice" => "vojs",
        "speech" => "spitʃ",
        "sound" => "saund",
        "english" => "inɡliʃ",
        "hebrew" => "hibʁu",
        _ => return None,
    })
}

/// Rule-based grapheme-to-phoneme for unknown English words.
/// Processes digraphs first, then single letters.
fn rules_g2p(word: &str) -> String {
    let chars: Vec<char> = word.chars().collect();
    let len = chars.len();
    let mut result = String::new();
    let mut i = 0;

    while i < len {
        // Try trigraphs
        if i + 2 < len {
            let tri: String = chars[i..i + 3].iter().collect();
            if let Some(ph) = match_trigraph(&tri) {
                result.push_str(ph);
                i += 3;
                continue;
            }
        }

        // Try digraphs
        if i + 1 < len {
            let di: String = chars[i..i + 2].iter().collect();
            if let Some(ph) = match_digraph(&di, i, len) {
                result.push_str(ph);
                i += 2;
                continue;
            }
        }

        // Single letter
        let ch = chars[i];
        let is_end = i == len - 1;
        let next = if i + 1 < len { Some(chars[i + 1]) } else { None };
        result.push_str(single_letter(ch, is_end, next));
        i += 1;
    }

    result
}

fn match_trigraph(tri: &str) -> Option<&'static str> {
    Some(match tri {
        "tch" => "tʃ",
        "ght" => "t",       // light, right — the vowel before handles the sound
        "igh" => "aj",
        "tion" => "ʃen",
        "sion" => "ʃen",
        "ous" => "as",
        "ing" => "inɡ",
        "ble" => "bel",
        "tle" => "tel",
        "ple" => "pel",
        _ => return None,
    })
}

fn match_digraph(di: &str, _pos: usize, _len: usize) -> Option<&'static str> {
    Some(match di {
        // Consonant digraphs
        "th" => "d",
        "sh" => "ʃ",
        "ch" => "tʃ",
        "ph" => "f",
        "wh" => "v",
        "ck" => "k",
        "ng" => "nɡ",
        "nk" => "nk",
        "gh" => "",        // silent in most positions (caught by igh/ght trigraphs)
        "wr" => "ʁ",
        "kn" => "n",
        "gn" => "n",
        "qu" => "kv",

        // Vowel digraphs
        "ee" => "i",
        "ea" => "i",
        "oo" => "u",
        "ou" => "au",
        "ow" => "au",
        "ai" => "ej",
        "ay" => "ej",
        "oi" => "oj",
        "oy" => "oj",
        "ie" => "i",
        "ei" => "ej",
        "ey" => "ej",
        "au" => "o",
        "aw" => "o",

        // Consonant + silent e patterns handled by single_letter
        _ => return None,
    })
}

fn single_letter(ch: char, is_end: bool, next: Option<char>) -> &'static str {
    match ch {
        'a' => {
            if next == Some('e') { "ej" }
            else { "a" }
        }
        'b' => "b",
        'c' => {
            match next {
                Some('e') | Some('i') | Some('y') => "s",
                _ => "k",
            }
        }
        'd' => "d",
        'e' => {
            if is_end { "" }  // silent final e
            else { "e" }
        }
        'f' => "f",
        'g' => {
            match next {
                Some('e') | Some('i') | Some('y') => "dʒ",
                _ => "ɡ",
            }
        }
        'h' => "h",
        'i' => {
            // Check for long i (before consonant + e at end)
            "i"
        }
        'j' => "dʒ",
        'k' => "k",
        'l' => "l",
        'm' => "m",
        'n' => "n",
        'o' => "o",
        'p' => "p",
        'q' => "k",
        'r' => "ʁ",
        's' => {
            if is_end { "z" }
            else { "s" }
        }
        't' => "t",
        'u' => "a",  // cup, but, etc. — most common unstressed English u
        'v' => "v",
        'w' => "v",
        'x' => "ks",
        'y' => {
            if is_end { "i" }
            else { "j" }
        }
        'z' => "z",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hello_world() {
        // "hello" and "world" are in the dictionary
        let result = phonemize_english("hello world");
        assert_eq!(result, "helo veʁld");
    }

    #[test]
    fn test_common_words() {
        assert_eq!(phonemize_english("the"), "de");
        assert_eq!(phonemize_english("is"), "iz");
        assert_eq!(phonemize_english("good"), "ɡud");
    }

    #[test]
    fn test_unknown_word_rules() {
        // "cat" not in dictionary, should use rules: k-a-t
        let result = phonemize_english("cat");
        assert_eq!(result, "kat");
    }

    #[test]
    fn test_digraphs() {
        // "sheep" not in dictionary: sh->ʃ, ee->i, p->p
        let result = phonemize_english("sheep");
        assert_eq!(result, "ʃip");
    }

    #[test]
    fn test_preserves_punctuation() {
        let result = phonemize_english("hello!");
        assert_eq!(result, "helo!");
    }

    #[test]
    fn test_mixed_case() {
        let result = phonemize_english("Hello World");
        assert_eq!(result, "helo veʁld");
    }
}
