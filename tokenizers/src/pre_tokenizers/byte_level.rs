use ahash::{AHashMap, AHashSet};
use std::sync::LazyLock;

use crate::tokenizer::pattern::Pattern;
use crate::utils::SysRegex;
use serde::{Deserialize, Serialize};

use crate::tokenizer::{
    Decoder, Encoding, PostProcessor, PreTokenizedString, PreTokenizer, Result,
    SplitDelimiterBehavior,
};
use crate::utils::macro_rules_attribute;

/// Converts bytes to unicode characters.
/// See https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
pub(crate) fn bytes_char() -> AHashMap<u8, char> {
    let mut bs: Vec<u8> = vec![];
    bs.extend(b'!'..=b'~');
    bs.extend(b'\xA1'..=b'\xAC');
    bs.extend(b'\xAE'..=b'\xFF');

    let mut cs: Vec<u32> = bs.iter().map(|i| *i as u32).collect();
    let mut n = 0;

    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(u32::pow(2, 8) + n);
            n += 1;
        }
    }

    // Safety: cs contains all values from bs (between 0 and 255),
    // and some values of value 2⁸ + n, where n is between 0 and 255. This is between 255 and 512.
    // Both ranges are valid UTF-32 values (which is fully saturated until 0xD000)
    bs.into_iter()
        .zip(cs)
        .map(|(f, t)| (f, unsafe { std::char::from_u32_unchecked(t) }))
        .collect()
}

/// Regex that matches exactly one token.
/// See https://github.com/openai/gpt-2/blob/master/src/encoder.py#L98
static RE: LazyLock<SysRegex> = LazyLock::new(|| {
    SysRegex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        .unwrap()
});
static BYTES_CHAR: LazyLock<AHashMap<u8, char>> = LazyLock::new(bytes_char);
static CHAR_BYTES: LazyLock<AHashMap<char, u8>> =
    LazyLock::new(|| bytes_char().into_iter().map(|(c, b)| (b, c)).collect());

// Only the fixed, unanchored GPT-2 expression above is specialized. Unknown
// ASCII/Unicode boundaries use its original first match, including all lookahead.
struct ByteLevelPattern<'a>(&'a SysRegex);

#[derive(PartialEq)]
enum AsciiClass {
    Whitespace,
    Letter,
    Number,
    Other,
}

fn ascii_class(byte: u8) -> Option<AsciiClass> {
    match byte {
        b'\t'..=b'\r' | b' ' => Some(AsciiClass::Whitespace),
        b'A'..=b'Z' | b'a'..=b'z' => Some(AsciiClass::Letter),
        b'0'..=b'9' => Some(AsciiClass::Number),
        0..=0x7f => Some(AsciiClass::Other),
        _ => None,
    }
}

fn ascii_prefix_len(bytes: &[u8]) -> Option<usize> {
    let first = *bytes.first()?;
    if first == b'\'' {
        const ENDINGS: [&[u8]; 7] = [b"'s", b"'t", b"'re", b"'ve", b"'m", b"'ll", b"'d"];
        for ending in ENDINGS {
            if bytes.starts_with(ending) {
                return Some(ending.len());
            }
        }
    }
    let mut start = 0;
    if first == b' '
        && bytes
            .get(1)
            .and_then(|byte| ascii_class(*byte))
            .is_some_and(|kind| kind != AsciiClass::Whitespace)
    {
        start = 1;
    }
    let kind = ascii_class(bytes[start])?;
    let mut end = start + 1;
    while let Some(byte) = bytes.get(end) {
        let following = ascii_class(*byte)?;
        if following != kind {
            break;
        }
        end += 1;
    }
    // Before a nonspace, the first whitespace alternative backs up one byte.
    if kind == AsciiClass::Whitespace && end < bytes.len() && end > start + 1 {
        end -= 1;
    }
    Some(end)
}

impl Pattern for ByteLevelPattern<'_> {
    fn find_matches(&self, inside: &str) -> Result<Vec<(crate::Offsets, bool)>> {
        // fancy-regex stops at its backtracking limit on long valid strings.
        // Preserve its original whole-input matching, including that behavior.
        if !cfg!(feature = "onig") || inside.is_empty() {
            return self.0.find_matches(inside);
        }
        let mut matches = Vec::with_capacity(inside.len());
        let mut start = 0;
        while start < inside.len() {
            let remaining = &inside[start..];
            let length = match ascii_prefix_len(remaining.as_bytes()) {
                Some(length) => length,
                None => match self.0.find_iter(remaining).next() {
                    Some((0, end)) if end > 0 && remaining.is_char_boundary(end) => end,
                    _ => return self.0.find_matches(inside),
                },
            };
            matches.push(((start, start + length), true));
            start += length;
        }
        Ok(matches)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
/// Provides all the necessary steps to handle the BPE tokenization at the byte-level. Takes care
/// of all the required processing steps to transform a UTF-8 string as needed before and after the
/// BPE model does its job.
#[macro_rules_attribute(impl_serde_type!)]
#[non_exhaustive]
pub struct ByteLevel {
    /// Whether to add a leading space to the first word. This allows to treat the leading word
    /// just as any other word.
    pub add_prefix_space: bool,
    /// Whether the post processing step should trim offsets to avoid including whitespaces.
    pub trim_offsets: bool,

    /// Whether to use the standard GPT2 regex for whitespace splitting
    /// Set it to False if you want to use your own splitting.
    #[serde(default = "default_true")]
    pub use_regex: bool,
}

fn default_true() -> bool {
    true
}

impl Default for ByteLevel {
    fn default() -> Self {
        Self {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        }
    }
}

impl ByteLevel {
    pub fn new(add_prefix_space: bool, trim_offsets: bool, use_regex: bool) -> Self {
        Self {
            add_prefix_space,
            trim_offsets,
            use_regex,
        }
    }

    pub fn alphabet() -> AHashSet<char> {
        BYTES_CHAR.values().copied().collect()
    }

    #[must_use]
    pub fn add_prefix_space(mut self, v: bool) -> Self {
        self.add_prefix_space = v;
        self
    }

    #[must_use]
    pub fn trim_offsets(mut self, v: bool) -> Self {
        self.trim_offsets = v;
        self
    }

    #[must_use]
    pub fn use_regex(mut self, v: bool) -> Self {
        self.use_regex = v;
        self
    }
}

/// As a `PreTokenizer`, `ByteLevel` is in charge of transforming all the unicode characters into
/// their byte-level counterpart. It also splits the input according to the configured regex.
// TODO: Give the ability to modify this regex
impl PreTokenizer for ByteLevel {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        let re_ref: &SysRegex = &RE;
        pretokenized.split(|_, mut normalized| {
            if self.add_prefix_space && !normalized.get().starts_with(' ') {
                normalized.prepend(" ");
            }
            if self.use_regex {
                normalized.split(ByteLevelPattern(re_ref), SplitDelimiterBehavior::Isolated)
            } else {
                Ok(vec![normalized])
            }
        })?;
        pretokenized.normalize(|normalized| {
            let s = normalized.get();
            let mut transformations: Vec<(char, isize)> = Vec::with_capacity(s.len());
            for (i, cur_char) in s.char_indices() {
                let size = cur_char.len_utf8();
                transformations.extend(
                    s.as_bytes()[i..i + size]
                        .iter()
                        .enumerate()
                        .map(|(i, b)| (BYTES_CHAR[b], isize::from(i > 0))),
                );
            }
            normalized.transform(transformations, 0);
            Ok(())
        })
    }
}

/// As a `Decoder`, `ByteLevel` is in charge of converting any byte-level characters to their
/// unicode counterpart, before merging everything back into a single String.
/// This decoder will consume the tokens and merge them in one step to alleviate
/// the fact that single token decoded might be a byte not representable as
/// as String.
impl Decoder for ByteLevel {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let toks = tokens
            .into_iter()
            .flat_map(|t| {
                t.chars()
                    .try_fold(vec![], |mut acc, c| {
                        CHAR_BYTES.get(&c).map(|b| {
                            acc.push(*b);
                            acc
                        })
                    })
                    .unwrap_or_else(|| t.as_bytes().to_vec())
            })
            .collect::<Vec<u8>>();
        Ok(vec![String::from_utf8_lossy(&toks).to_string()])
    }
}

/// As a `PostProcessor`, `ByteLevel` is in charge of trimming the offsets if necessary.
impl PostProcessor for ByteLevel {
    fn added_tokens(&self, _is_pair: bool) -> usize {
        0
    }

    fn process_encodings(
        &self,
        mut encodings: Vec<Encoding>,
        _add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        if self.trim_offsets {
            for encoding in encodings.iter_mut() {
                process_offsets(encoding, self.add_prefix_space);
                encoding
                    .get_overflowing_mut()
                    .iter_mut()
                    .for_each(|encoding| process_offsets(encoding, self.add_prefix_space));
            }
        }
        for (i, encoding) in encodings.iter_mut().enumerate() {
            encoding.set_sequence_id(i);
        }
        Ok(encodings)
        //<dyn PostProcessor>::default_process(encodings, add_special_tokens)
    }
}

pub fn process_offsets(encoding: &mut Encoding, add_prefix_space: bool) {
    encoding.process_tokens_with_offsets_mut(|(i, (token, offsets))| {
        let mut leading_spaces = token
            .chars()
            .take_while(|c| *c == BYTES_CHAR[&b' '] || c.is_whitespace())
            .count();
        let trailing_spaces = token
            .chars()
            .rev()
            .take_while(|c| *c == BYTES_CHAR[&b' '] || c.is_whitespace())
            .count();

        if leading_spaces > 0 || trailing_spaces > 0 {
            if leading_spaces > 0 {
                // If user uses `is_pretokenized=True` we might have
                // offsets that might begin at the start of the string but are
                // NOT the first token.
                let is_first = i == 0 || offsets.0 == 0;
                if is_first && add_prefix_space && leading_spaces == 1 {
                    // If we are processing the first pair of offsets, with `add_prefix_space`,
                    // then we shouldn't remove anything we added. If there are more than one
                    // leading spaces though, it means we didn't add them, and they should be
                    // removed.
                    leading_spaces = 0;
                }
                offsets.0 = std::cmp::min(offsets.0 + leading_spaces, offsets.1);
            }
            if trailing_spaces > 0 && offsets.1 >= trailing_spaces {
                offsets.1 = std::cmp::max(offsets.1 - trailing_spaces, offsets.0);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{
        Decoder, Encoding, OffsetReferential, OffsetType, PostProcessor, PreTokenizedString,
        PreTokenizer,
    };
    use std::iter::FromIterator;

    fn check_hybrid_pattern(text: &str) {
        let original: &SysRegex = &RE;
        assert_eq!(
            ByteLevelPattern(original).find_matches(text).unwrap(),
            original.find_matches(text).unwrap(),
            "length={}, prefix={:?}",
            text.len(),
            text.chars().take(16).collect::<String>()
        );
    }

    #[test]
    fn hybrid_ascii_pattern_matches_original() {
        for first in 0u8..=127 {
            check_hybrid_pattern(&String::from_utf8(vec![first]).unwrap());
            for second in 0u8..=127 {
                check_hybrid_pattern(&String::from_utf8(vec![first, second]).unwrap());
            }
        }
        let alphabet = b"ar0' \t\x0b!";
        for length in 0..=6 {
            for mut ordinal in 0..alphabet.len().pow(length) {
                let mut text = String::with_capacity(length as usize);
                for _ in 0..length {
                    text.push(alphabet[ordinal % alphabet.len()] as char);
                    ordinal /= alphabet.len();
                }
                check_hybrid_pattern(&text);
            }
        }
    }

    #[test]
    fn hybrid_unicode_boundaries_match_original() {
        let parts = [
            "", "a", "0", "!", "'", "'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", " ", "\t",
            "\n", "\r", "\x0b", "\x0c", "é", "中", "²", "Ⅷ", "\u{301}", "🙂", "\u{85}", "\u{a0}",
            "\u{2003}", "\u{2028}", "\u{200d}", "\0",
        ];
        for head in parts {
            for middle in parts {
                for tail in parts {
                    check_hybrid_pattern(&format!("{head}{middle}{tail}"));
                }
            }
        }
    }

    #[test]
    fn hybrid_long_runs_match_original() {
        for repeated in ["a", "0", "!", " ", "\t", "\x0b", "é", "🙂"] {
            for length in [4096, 1_100_000] {
                let run = repeated.repeat(length);
                for end in ["", "a", " ", "\n", "é", "\u{301}", "🙂"] {
                    check_hybrid_pattern(&format!("{run}{end}"));
                }
            }
        }
    }

    #[test]
    fn bytelevel_preserves_long_regex_backend_behavior() {
        let text = "a".repeat(1_100_000) + "!";
        let original: &SysRegex = &RE;
        for prefix in [false, true] {
            let mut actual: PreTokenizedString = text.as_str().into();
            ByteLevel::default()
                .add_prefix_space(prefix)
                .pre_tokenize(&mut actual)
                .unwrap();
            let mut expected: PreTokenizedString = text.as_str().into();
            expected
                .split(|_, mut normalized| {
                    if prefix && !normalized.get().starts_with(' ') {
                        normalized.prepend(" ");
                    }
                    normalized.split(original, SplitDelimiterBehavior::Isolated)
                })
                .unwrap();
            ByteLevel::default()
                .add_prefix_space(false)
                .use_regex(false)
                .pre_tokenize(&mut expected)
                .unwrap();
            for referential in [OffsetReferential::Original, OffsetReferential::Normalized] {
                for offsets in [OffsetType::Byte, OffsetType::Char] {
                    let observed: Vec<_> = actual
                        .get_splits(referential, offsets)
                        .into_iter()
                        .map(|(text, offsets, _)| (text, offsets))
                        .collect();
                    let wanted: Vec<_> = expected
                        .get_splits(referential, offsets)
                        .into_iter()
                        .map(|(text, offsets, _)| (text, offsets))
                        .collect();
                    assert!(
                        observed == wanted,
                        "prefix={prefix}, actual_splits={}, expected_splits={}",
                        observed.len(),
                        wanted.len()
                    );
                }
            }
        }
    }

    #[test]
    fn pre_tokenization() {
        let bytelevel = ByteLevel::default().add_prefix_space(false);
        let mut pretokenized: PreTokenizedString = "Hello my friend, how is your day going?".into();
        bytelevel.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hello", (0, 5)),
                ("Ġmy", (5, 8)),
                ("Ġfriend", (8, 15)),
                (",", (15, 16)),
                ("Ġhow", (16, 20)),
                ("Ġis", (20, 23)),
                ("Ġyour", (23, 28)),
                ("Ġday", (28, 32)),
                ("Ġgoing", (32, 38)),
                ("?", (38, 39))
            ]
        );
    }

    #[test]
    fn pre_tokenization_no_regex() {
        let bytelevel = ByteLevel::default().use_regex(false);
        let mut pretokenized: PreTokenizedString = "Hello my friend, how is your day going?".into();
        bytelevel.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("ĠHelloĠmyĠfriend,ĠhowĠisĠyourĠdayĠgoing?", (0, 39))]
        );
    }

    #[test]
    fn decoding() {
        let bytelevel = ByteLevel::default().add_prefix_space(false);
        assert_eq!(
            bytelevel
                .decode_chain(
                    vec![
                        "Hello", "Ġmy", "Ġfriend", ",", "Ġhow", "Ġis", "Ġyour", "Ġday", "Ġgoing",
                        "?"
                    ]
                    .into_iter()
                    .map(|s| s.into())
                    .collect::<Vec<String>>()
                )
                .unwrap(),
            vec!["Hello my friend, how is your day going?"]
        );
    }

    #[test]
    fn add_prefix_space() {
        let bytelevel = ByteLevel::default().add_prefix_space(true);
        for s in &[
            " Hello my friend, how is your day going?",
            "Hello my friend, how is your day going?",
        ] {
            let mut pretokenized = PreTokenizedString::from(*s);
            bytelevel.pre_tokenize(&mut pretokenized).unwrap();
            assert_eq!(
                pretokenized
                    .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                    .into_iter()
                    .map(|(s, o, _)| (s, o))
                    .collect::<Vec<_>>(),
                vec![
                    ("ĠHello", (0, 7)),
                    ("Ġmy", (7, 11)),
                    ("Ġfriend", (11, 19)),
                    (",", (19, 20)),
                    ("Ġhow", (20, 25)),
                    ("Ġis", (25, 29)),
                    ("Ġyour", (29, 35)),
                    ("Ġday", (35, 40)),
                    ("Ġgoing", (40, 47)),
                    ("?", (47, 48))
                ]
            );
        }
    }

    #[test]
    fn decode_works_on_separated_tokens() {
        let samples = vec![
            "A Nuskhuri abbreviation of იესუ ქრისტე ( iesu kriste ) \" Jesus Christ \"",
            "An equal number have descenders , like p or q in English \
                 : გ , დ , ე , ვ , კ , ლ , ჟ , ტ , უ , ფ , ღ , ყ , ც",
        ];

        let bytelevel = ByteLevel::default().add_prefix_space(false);
        for sample in samples {
            let mut pretokenized = PreTokenizedString::from(sample);
            bytelevel.pre_tokenize(&mut pretokenized).unwrap();
            let separated_tokens = pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .iter()
                .flat_map(|(s, _, _)| s.split("").map(|t| t.into()))
                .collect::<Vec<_>>();
            assert_eq!(
                sample,
                bytelevel.decode_chain(separated_tokens).unwrap().join("")
            );
        }
    }

    #[test]
    fn handling_of_newlines() {
        let mut pretokenized = PreTokenizedString::from("Hello there\nHello there");
        let bytelevel = ByteLevel::default().add_prefix_space(false);
        bytelevel.pre_tokenize(&mut pretokenized).unwrap();

        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hello", (0, 5)),
                ("Ġthere", (5, 11)),
                ("Ċ", (11, 12)),
                ("Hello", (12, 17)),
                ("Ġthere", (17, 23))
            ]
        );
    }

    #[test]
    fn handling_of_multiple_whitespaces() {
        let mut pretokenized = PreTokenizedString::from("Hello there       dear");
        let bytelevel = ByteLevel::default().add_prefix_space(false);
        bytelevel.pre_tokenize(&mut pretokenized).unwrap();

        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hello", (0, 5)),
                ("Ġthere", (5, 11)),
                ("ĠĠĠĠĠĠ", (11, 17)),
                ("Ġdear", (17, 22))
            ]
        );
    }

    #[test]
    fn offsets_when_char_split_up() {
        let input = "i⭢j";
        let mut pretokenized = PreTokenizedString::from(input);
        let bytelevel = ByteLevel::default().add_prefix_space(false);
        bytelevel.pre_tokenize(&mut pretokenized).unwrap();

        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("i", (0, 1)), ("âŃ¢", (1, 4)), ("j", (4, 5))]
        );
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("i", (0, 1)), ("âŃ¢", (1, 7)), ("j", (7, 8))]
        );
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(_, o, _)| &input[o.0..o.1])
                .collect::<Vec<_>>(),
            vec!["i", "⭢", "j"]
        );
    }

    #[test]
    fn processor_trims_offsets_pre_tokenized() {
        // If user uses `is_pretokenized=True` we might have
        // offsets that might begin at the start of the string but are
        // NOT the first token.
        let mut encoding = Encoding::new(
            vec![0; 5],
            vec![],
            vec!["Ġl".into(), "ove".into(), "Ġl".into(), "ove".into()],
            vec![],
            vec![(0, 1), (1, 4), (0, 1), (1, 4)],
            vec![],
            vec![],
            vec![],
            AHashMap::new(),
        );
        process_offsets(&mut encoding, true);
        assert_eq!(
            encoding,
            Encoding::new(
                vec![0; 5],
                vec![],
                vec!["Ġl".into(), "ove".into(), "Ġl".into(), "ove".into()],
                vec![],
                vec![(0, 1), (1, 4), (0, 1), (1, 4)],
                vec![],
                vec![],
                vec![],
                AHashMap::new(),
            )
        );
    }

    #[test]
    fn processor_trims_offsets() {
        let start = Encoding::new(
            vec![0; 5],
            vec![],
            vec![
                "Ġ".into(),
                "ĠĠĠĠHelloĠĠ".into(),
                "ĠĠHello".into(),
                "HelloĠĠ".into(),
                "ĠĠĠĠ".into(),
            ],
            vec![],
            vec![(0, 1), (0, 11), (11, 18), (18, 25), (25, 29)],
            vec![],
            vec![],
            vec![],
            AHashMap::new(),
        );
        let expected = Encoding::new(
            vec![0; 5],
            vec![0; 5],
            vec![
                "Ġ".into(),
                "ĠĠĠĠHelloĠĠ".into(),
                "ĠĠHello".into(),
                "HelloĠĠ".into(),
                "ĠĠĠĠ".into(),
            ],
            vec![],
            vec![(0, 0), (4, 9), (13, 18), (18, 23), (29, 29)],
            vec![],
            vec![],
            vec![],
            AHashMap::from_iter(vec![(0, 0..5)]),
        );

        let bytelevel = ByteLevel::default().trim_offsets(true);
        assert_eq!(
            expected,
            bytelevel.process(start.clone(), None, false).unwrap()
        );

        let pair_expected = Encoding::new(
            vec![0; 10],
            vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            vec![
                "Ġ".into(),
                "ĠĠĠĠHelloĠĠ".into(),
                "ĠĠHello".into(),
                "HelloĠĠ".into(),
                "ĠĠĠĠ".into(),
                "Ġ".into(),
                "ĠĠĠĠHelloĠĠ".into(),
                "ĠĠHello".into(),
                "HelloĠĠ".into(),
                "ĠĠĠĠ".into(),
            ],
            vec![],
            vec![
                (0, 0),
                (4, 9),
                (13, 18),
                (18, 23),
                (29, 29),
                (0, 0),
                (4, 9),
                (13, 18),
                (18, 23),
                (29, 29),
            ],
            vec![],
            vec![],
            vec![],
            AHashMap::from_iter(vec![(0, 0..5), (1, 5..10)]),
        );
        assert_eq!(
            pair_expected,
            bytelevel
                .process(start.clone(), Some(start), false)
                .unwrap()
        );
    }

    #[test]
    fn decode_unknown_characters() {
        let byte_level = ByteLevel::default();
        assert_eq!(
            byte_level
                .decode_chain(vec![
                    "Hello".into(),
                    "Ġthere".into(),
                    "Ġdear".into(),
                    "Ġfriend!".into(),
                    "Ġ".into(),
                    "[PA D]".into()
                ])
                .unwrap(),
            vec!["Hello there dear friend! [PA D]"]
        );
    }

    #[test]
    fn deserialization() {
        // Before use_regex
        let byte_level: ByteLevel = serde_json::from_str(
            r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false}"#,
        )
        .unwrap();
        assert!(byte_level.use_regex);

        // Loading works, new future BC test.
        let byte_level: ByteLevel = serde_json::from_str(
            r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true}"#,
        )
        .unwrap();
        assert!(byte_level.use_regex);

        let byte_level: ByteLevel = serde_json::from_str(
            r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": false}"#,
        )
        .unwrap();
        assert!(!byte_level.use_regex);
    }
}
