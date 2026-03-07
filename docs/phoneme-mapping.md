# Phoneme Mapping — Hebrew TTS Model Vocabulary

The LightBlue Hebrew TTS model uses a fixed set of IPA phonemes. All input text
must be converted to this phoneme set before inference.

## Available Phonemes

After `normalize_text` maps `g→ɡ` and `r→ʁ`:

| Category    | Phonemes                                           |
|-------------|-----------------------------------------------------|
| Consonants  | b, d, f, h, j, k, l, m, n, p, s, t, v, w, z, ɡ, ʁ, ʃ, ʒ, ʔ, χ |
| Vowels      | a, e, i, o, u                                       |
| Stress      | ˈ                                                   |
| Punctuation | space, `.` `,` `!` `?` `'` `"` `-` `:`             |

Total: 36 symbols (IDs 1–36, 0 = PAD/unknown).

## English Phoneme Approximations

English phonemes not in the Hebrew model are mapped to the closest available:

| English IPA | Example   | Mapped to | Notes                          |
|-------------|-----------|-----------|--------------------------------|
| θ           | **th**ink | t         | voiceless dental → alveolar    |
| ð           | **th**e   | d         | voiced dental → alveolar       |
| ŋ           | si**ng**  | n         | velar nasal → alveolar nasal   |
| ɹ           | **r**ed   | ʁ         | handled by `normalize_text`    |
| æ           | c**a**t   | a         | near-open front → open central |
| ɛ           | b**e**d   | e         | open-mid front → close-mid     |
| ɪ           | s**i**t   | i         | near-close front → close front |
| ʊ           | b**oo**k  | u         | near-close back → close back   |
| ɔ           | c**au**ght| o         | open-mid back → close-mid back |
| ə           | **a**bout | e         | schwa → close-mid front        |
| ʌ           | c**u**p   | a         | open-mid back → open central   |
| tʃ          | **ch**urch| tʃ        | both phonemes available!       |
| dʒ          | **j**udge | dʒ        | both phonemes available!       |

## Pipeline

```
Hebrew text  →  phonikud (diacritics)  →  Hebrew phonemizer (nikud→IPA)  →  TTS model
English text →  English phonemizer (grapheme→IPA approximation)          →  TTS model
Mixed text   →  split by language  →  process each segment  →  concatenate audio
```

The English phonemizer (`src/phonemize/english.rs`) uses:
1. A dictionary of ~200 common English words with hand-tuned IPA
2. A rule-based grapheme-to-phoneme fallback for unknown words

Since the TTS model was trained on Hebrew speech, English output will have a
strong Hebrew accent. This is expected and acceptable for a Hebrew-first engine.
