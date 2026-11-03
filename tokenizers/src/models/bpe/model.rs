use super::{super::OrderedVocabIter, trainer::BpeTrainer, Error, Pair, Word};
use crate::tokenizer::{Model, Result, Token};
use crate::utils::cache::{Cache, DEFAULT_CACHE_CAPACITY, MAX_LENGTH};
use crate::utils::iter::ResultShunt;
use ahash::AHashMap;
use serde_json::Value;
use std::borrow::Cow;

use std::collections::HashMap;
use std::{
    fs::File,
    io::prelude::*,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

pub type Vocab = AHashMap<String, u32>;
type VocabR = AHashMap<u32, String>;
pub type MergeMap = AHashMap<Pair, (u32, u32)>;
pub type Merges = Vec<(String, String)>;

struct Config {
    files: Option<(String, String)>,
    vocab: Vocab,
    merges: Merges,
    cache_capacity: usize,
    dropout: Option<f32>,
    unk_token: Option<String>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    fuse_unk: bool,
    byte_fallback: bool,
    ignore_merges: bool,
}

/// A `BpeBuilder` can be used to create a `BPE` model with a custom configuration.
pub struct BpeBuilder {
    config: Config,
}

impl Default for BpeBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                files: None,
                vocab: AHashMap::new(),
                merges: vec![],
                cache_capacity: DEFAULT_CACHE_CAPACITY,
                dropout: None,
                unk_token: None,
                continuing_subword_prefix: None,
                end_of_word_suffix: None,
                fuse_unk: false,
                byte_fallback: false,
                ignore_merges: false,
            },
        }
    }
}

impl BpeBuilder {
    /// Constructs a new `BpeBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input files.
    #[must_use]
    pub fn files(mut self, vocab: String, merges: String) -> Self {
        self.config.files = Some((vocab, merges));
        self
    }

    /// Set the vocab (token -> ID) and merges mappings.
    #[must_use]
    pub fn vocab_and_merges<V: Into<AHashMap<String, u32>>>(
        mut self,
        vocab: V,
        merges: Merges,
    ) -> Self {
        self.config.vocab = vocab.into();
        self.config.merges = merges;
        self
    }

    /// Set the cache's capacity. Set to 0 if you want to disable caching.
    #[must_use]
    pub fn cache_capacity(mut self, capacity: usize) -> Self {
        self.config.cache_capacity = capacity;
        self
    }

    /// Use [dropout](https://arxiv.org/abs/1910.13267) with the model.
    #[must_use]
    pub fn dropout(mut self, dropout: f32) -> Self {
        self.config.dropout = Some(dropout);
        self
    }

    /// Set the `UNK` token for the vocab.
    #[must_use]
    pub fn unk_token(mut self, unk_token: String) -> Self {
        self.config.unk_token = Some(unk_token);
        self
    }

    /// Set the `continuing_subword_prefix` option.
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.config.continuing_subword_prefix = Some(prefix);
        self
    }

    /// Set the `end_of_word_suffix` option.
    #[must_use]
    pub fn end_of_word_suffix(mut self, prefix: String) -> Self {
        self.config.end_of_word_suffix = Some(prefix);
        self
    }

    /// Set the `fuse_unk` option.
    #[must_use]
    pub fn fuse_unk(mut self, fuse_unk: bool) -> Self {
        self.config.fuse_unk = fuse_unk;
        self
    }

    /// Set the `byte_fallback` option.
    #[must_use]
    pub fn byte_fallback(mut self, byte_fallback: bool) -> Self {
        self.config.byte_fallback = byte_fallback;
        self
    }
    /// Set the `ignore_merges` option.
    #[must_use]
    pub fn ignore_merges(mut self, ignore_merges: bool) -> Self {
        self.config.ignore_merges = ignore_merges;
        self
    }

    /// Returns a `BPE` model that uses the `BpeBuilder`'s configuration.
    pub fn build(mut self) -> Result<BPE> {
        // Validate dropout.
        if let Some(p) = self.config.dropout {
            if !(0.0..=1.0).contains(&p) {
                return Err(Error::InvalidDropout.into());
            }
        }

        // Read files if necessary
        if let Some((vocab, merges)) = self.config.files {
            let (v, m) = BPE::read_file(&vocab, &merges)?;
            self.config.vocab = v;
            self.config.merges = m;
        }

        let vocab_r = self
            .config
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();
        let cache = match self.config.cache_capacity {
            0 => None,
            capacity => Some(Cache::new(capacity)),
        };

        let vocab = self.config.vocab;
        let prefix_len = if let Some(prefix) = &self.config.continuing_subword_prefix {
            prefix.len()
        } else {
            0
        };
        let merge_map: MergeMap = self
            .config
            .merges
            .into_iter()
            .enumerate()
            .map(|(i, (a, b))| -> Result<(Pair, (u32, u32))> {
                let a_id = vocab
                    .get(&a)
                    .ok_or_else(|| Error::MergeTokenOutOfVocabulary(a.to_owned()))?;
                let b_id = vocab
                    .get(&b)
                    .ok_or_else(|| Error::MergeTokenOutOfVocabulary(b.to_owned()))?;
                let new_token = format!("{}{}", a, &b[prefix_len..]);
                let new_id = vocab
                    .get(&new_token)
                    .ok_or(Error::MergeTokenOutOfVocabulary(new_token))?;
                Ok(((*a_id, *b_id), (i as u32, *new_id)))
            })
            .collect::<Result<MergeMap>>()?;

        // merges.insert(pair, (rank as u32, *new_id));

        Ok(BPE {
            vocab,
            vocab_r,
            merges: merge_map,
            cache,
            dropout: self.config.dropout,
            unk_token: self.config.unk_token,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            end_of_word_suffix: self.config.end_of_word_suffix,
            fuse_unk: self.config.fuse_unk,
            byte_fallback: self.config.byte_fallback,
            ignore_merges: self.config.ignore_merges,
        })
    }
}

/// A [Byte Pair Encoding](https://www.aclweb.org/anthology/P16-1162/) model.
#[derive(PartialEq)]
pub struct BPE {
    /// The vocabulary assigns a number to each token.
    pub(crate) vocab: Vocab,
    /// Reversed vocabulary, to rebuild sentences.
    pub(crate) vocab_r: VocabR,
    /// Contains the mapping between Pairs and their (rank, new_id).
    pub(crate) merges: MergeMap,
    /// Contains the cache for optimizing the encoding step.
    cache: Option<Cache<String, Word>>,
    /// Dropout probability for merges. 0.0 = no dropout is the default. At 1.0, tokenization will
    /// perform no merges, so the result will just be characters.
    pub dropout: Option<f32>,
    /// The unknown token to be used when we encounter an unknown char
    pub unk_token: Option<String>,
    /// An optional prefix to use on any subword that exist only behind another one
    pub continuing_subword_prefix: Option<String>,
    /// An optional suffix to characterize and end-of-word subword
    pub end_of_word_suffix: Option<String>,
    /// Do multiple unk tokens get fused
    pub fuse_unk: bool,
    /// Byte fallback from sentence pieces, instead of UNK, uses `"<0x00>"`
    /// for each byte in the unk token
    pub byte_fallback: bool,
    /// Whether or not to direct output words if they are part of the vocab.
    pub ignore_merges: bool,
}

impl std::fmt::Debug for BPE {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("BPE")
            .field("dropout", &self.dropout)
            .field("unk_token", &self.unk_token)
            .field("continuing_subword_prefix", &self.continuing_subword_prefix)
            .field("end_of_word_suffix", &self.end_of_word_suffix)
            .field("fuse_unk", &self.fuse_unk)
            .field("byte_fallback", &self.byte_fallback)
            .field("vocab", &self.vocab.len())
            .field("merges", &self.merges.len())
            .field("ignore_merges", &self.ignore_merges)
            .finish()
    }
}

impl Default for BPE {
    fn default() -> Self {
        Self::builder().build().unwrap()
    }
}

impl Clone for BPE {
    // `Clone` can't be derive because it's not implemented for `Cache`.
    // To keep things simple when we clone, the new BPE will start with a fresh cache.
    fn clone(&self) -> Self {
        let fresh_cache = self.cache.as_ref().map(|cache| cache.fresh());
        Self {
            vocab: self.vocab.clone(),
            vocab_r: self.vocab_r.clone(),
            merges: self.merges.clone(),
            cache: fresh_cache,
            dropout: self.dropout,
            unk_token: self.unk_token.clone(),
            continuing_subword_prefix: self.continuing_subword_prefix.clone(),
            end_of_word_suffix: self.end_of_word_suffix.clone(),
            fuse_unk: self.fuse_unk,
            byte_fallback: self.byte_fallback,
            ignore_merges: self.ignore_merges,
        }
    }
}

/// Converts the merges strings (for example from `merges.txt` file) with the format
/// "{pair_a} {pair_b}" into the format expected by the BPE struct
pub(crate) fn convert_merges_to_hashmap<I: Iterator<Item = String>>(
    iter: I,
    _vocab: &Vocab,
) -> Result<Merges> {
    let mut merges = vec![];

    let lines = iter.filter(|l| !l.starts_with("#version"));
    for (rank, line) in lines.enumerate() {
        let parts = line.split(' ').collect::<Vec<_>>();
        if parts.len() != 2 {
            return Err(Error::BadMerges(rank + 1).into());
        }

        merges.push((parts[0].to_string(), parts[1].to_string()));
    }

    Ok(merges)
}

impl BPE {
    /// Initialize a `BpeBuilder`.
    pub fn builder() -> BpeBuilder {
        BpeBuilder::new()
    }

    /// Create a new BPE model with the given vocab and merges.
    pub fn new(vocab: Vocab, merges: Merges) -> Self {
        Self::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap()
    }

    /// Initialize a BpeBuilder model from vocab and merges files
    pub fn from_file(vocab: &str, merges: &str) -> BpeBuilder {
        Self::builder().files(vocab.to_owned(), merges.to_owned())
    }

    /// Read the given files to extract the vocab and merges
    pub fn read_file(vocab: &str, merges: &str) -> Result<(Vocab, Merges)> {
        // Read vocab.json
        let vocab_file = File::open(vocab)?;
        let mut vocab_file = BufReader::new(vocab_file);

        let mut buffer = String::new();
        vocab_file.read_to_string(&mut buffer)?;
        let json: Value = serde_json::from_str(&buffer)?;
        let mut vocab = AHashMap::new();
        match json {
            Value::Object(m) => {
                for (token, id) in m {
                    if let Value::Number(id) = id {
                        let id = id.as_u64().ok_or(Error::BadVocabulary)? as u32;
                        vocab.insert(token, id);
                    }
                }
            }
            _ => return Err(Box::new(Error::BadVocabulary)),
        };

        // Read merges file
        let merge_file = File::open(merges)?;
        let merge_file = BufReader::new(merge_file);
        let merges = ResultShunt::process(merge_file.lines(), |iter| {
            convert_merges_to_hashmap(iter, &vocab)
        })??;

        Ok((vocab, merges))
    }

    /// Reset the cache.
    pub fn clear_cache(&self) {
        if let Some(ref cache) = self.cache {
            cache.clear()
        }
    }

    /// Resize the cache
    pub fn resize_cache(&mut self, capacity: usize) {
        if let Some(ref mut cache) = self.cache {
            cache.resize(capacity);
        }
    }

    pub fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone().into_iter().collect()
    }

    pub fn get_unk_token(&self) -> &Option<String> {
        &self.unk_token
    }

    pub fn get_continuing_subword_prefix(&self) -> &Option<String> {
        &self.continuing_subword_prefix
    }

    fn merge_word(&self, w: &str) -> Result<Word> {
        self.merge_word_impl(w, false)
    }

    fn merge_word_impl(&self, w: &str, proof: bool) -> Result<Word> {
        let mut indices = w.char_indices().map(|(idx, _)| idx).peekable();
        let mut word = Word::with_capacity(w.len());
        let mut unk: Option<(u32, usize)> = None;
        while let Some(i) = indices.next() {
            let end = indices.peek();
            let is_first = i == 0;
            let is_last = end.is_none();

            let mut s = if let Some(e) = end {
                Cow::Borrowed(&w[i..*e])
            } else {
                Cow::Borrowed(&w[i..])
            };
            let byte_len = s.len();

            // Add the `continuing_subword_prefix` if relevant
            if !is_first {
                if let Some(prefix) = self
                    .continuing_subword_prefix
                    .as_ref()
                    .filter(|s| !s.is_empty())
                {
                    s = format!("{prefix}{s}").into()
                }
            }
            // Add the `end_of_word_suffix` if relevant
            if is_last {
                if let Some(suffix) = self.end_of_word_suffix.as_ref().filter(|s| !s.is_empty()) {
                    s = format!("{s}{suffix}").into()
                }
            }

            if let Some(id) = self.vocab.get(s.as_ref()) {
                if let Some((unk_id, unk_len)) = unk {
                    word.add(unk_id, unk_len);
                    unk = None;
                }
                word.add(*id, byte_len);
            } else {
                if self.byte_fallback {
                    let tokens: Option<Vec<_>> = s
                        .bytes()
                        .map(|b| -> Option<&u32> {
                            let code = format!("<{b:#04X}>");

                            self.vocab.get(&code)
                        })
                        .collect();
                    if let Some(tokens) = tokens {
                        for t in tokens {
                            word.add(*t, 1);
                        }
                        continue;
                    }
                }
                if let Some(unk_token) = &self.unk_token {
                    unk = match (unk, self.fuse_unk) {
                        (Some((unk_id, unk_len)), true) => {
                            // Fuse unk
                            Some((unk_id, unk_len + byte_len))
                        }
                        (Some((unk_id, unk_len)), false) => {
                            // Do not fuse unk, add the previous one
                            word.add(unk_id, unk_len);
                            Some((
                                *self.vocab.get(unk_token).ok_or_else(|| {
                                    Error::UnkTokenOutOfVocabulary(unk_token.to_owned())
                                })?,
                                byte_len,
                            ))
                        }
                        _ => Some((
                            *self.vocab.get(unk_token).ok_or_else(|| {
                                Error::UnkTokenOutOfVocabulary(unk_token.to_owned())
                            })?,
                            byte_len,
                        )),
                    };
                }
            }
        }
        if let Some((unk_id, unk_len)) = unk {
            word.add(unk_id, unk_len);
        }

        if proof && self.dropout.is_none() {
            word.merge_all_for_proof(&self.merges);
        } else {
            word.merge_all(&self.merges, self.dropout);
        }

        Ok(word)
    }

    fn word_to_tokens<'a>(&'a self, word: &'a Word) -> impl Iterator<Item = Token> + 'a {
        word.get_chars_iter()
            .zip(word.get_offsets_iter())
            .map(move |(id, offsets)| Token::new(id, self.vocab_r[&id].clone(), offsets))
    }

    /// Return the sole ID, if any, with the same deterministic BPE/cache operations.
    /// This avoids materializing token strings and offsets for model proofs.
    pub fn tokenize_single_id(&self, sequence: &str) -> Result<Option<u32>> {
        fn single_id(word: &Word) -> Option<u32> {
            let mut ids = word.get_chars_iter();
            let first = ids.next()?;
            if ids.next().is_none() {
                Some(first)
            } else {
                None
            }
        }
        if sequence.is_empty() {
            return Ok(None);
        }
        if self.dropout.is_some_and(|value| value != 0.0) {
            let tokens = self.tokenize(sequence)?;
            return Ok(if tokens.len() == 1 {
                Some(tokens[0].id)
            } else {
                None
            });
        }
        if self.ignore_merges {
            if let Some(id) = self.vocab.get(sequence) {
                return Ok(Some(*id));
            }
        }
        if let Some(ref hit) = self.cache.as_ref().and_then(|c| c.get(sequence)) {
            return Ok(single_id(hit));
        }
        let word = self.merge_word_impl(sequence, true)?;
        let result = single_id(&word);
        if let Some(ref cache) = self.cache {
            if sequence.len() < MAX_LENGTH {
                cache.set(sequence.to_owned(), word);
            }
        }
        Ok(result)
    }

    /// Prove the serialized byte-alphabet merge family directly from this
    /// immutable producer. The caller retains its live-vocabulary/configuration
    /// admission. Word order, logical work limits and BPE/cache operations match
    /// the serialized Python admission path.
    pub fn proved_byte_ranks(
        &self,
        max_entries: usize,
        max_word_chars: usize,
        max_total_chars: usize,
        mut remaining: usize,
    ) -> Result<Option<Vec<(Vec<u8>, u32)>>> {
        // Serialization is ordered by ID; repeated names retain the largest ID.
        let mut vocab = AHashMap::<&str, u32>::with_capacity(self.vocab_r.len());
        for (&id, word) in &self.vocab_r {
            vocab
                .entry(word.as_str())
                .and_modify(|old| *old = (*old).max(id))
                .or_insert(id);
        }
        if vocab.len() > max_entries
            || self
                .merges
                .len()
                .checked_add(256)
                .is_none_or(|n| n > max_entries)
        {
            return Ok(None);
        }
        let mut alphabet = Vec::with_capacity(256);
        let mut decoder = [0u8; 324];
        let mut hidden = 256u32;
        for byte in 0u32..256 {
            let visible = (33..127).contains(&byte)
                || (161..173).contains(&byte)
                || (174..256).contains(&byte);
            let point = if visible {
                byte
            } else {
                let point = hidden;
                hidden += 1;
                point
            };
            decoder[point as usize] = byte as u8;
            if visible {
                alphabet.push(char::from_u32(point).unwrap());
            }
        }
        alphabet.extend((256..hidden).map(|point| char::from_u32(point).unwrap()));
        let mut ordered = Vec::with_capacity(self.merges.len() + 256);
        let mut words = AHashMap::<&str, u32>::with_capacity(self.merges.len() + 256);
        for character in alphabet {
            let mut buffer = [0u8; 4];
            let Some((&word, &id)) =
                vocab.get_key_value(character.encode_utf8(&mut buffer) as &str)
            else {
                return Ok(None);
            };
            if id == u32::MAX {
                return Ok(None);
            }
            words.insert(word, id);
            ordered.push((word, id));
        }
        let mut merges = self.merges.iter().collect::<Vec<_>>();
        merges.sort_unstable_by_key(|(_, (rank, _))| *rank);
        let mut total = 256usize;
        let mut previous = 0u32;
        let mut combined = String::new();
        for (&(left_id, right_id), _) in merges {
            let left = self.vocab_r[&left_id].as_str();
            let right = self.vocab_r[&right_id].as_str();
            let Some(size) = left.chars().count().checked_add(right.chars().count()) else {
                return Ok(None);
            };
            let Some(next) = total.checked_add(size) else {
                return Ok(None);
            };
            total = next;
            if size > max_word_chars || total > max_total_chars {
                return Ok(None);
            }
            let (Some(&left_rank), Some(&right_rank)) = (words.get(left), words.get(right)) else {
                return Ok(None);
            };
            combined.clear();
            combined.push_str(left);
            combined.push_str(right);
            let Some((&word, &id)) = vocab.get_key_value(combined.as_str()) else {
                return Ok(None);
            };
            if id == u32::MAX
                || words.contains_key(word)
                || id <= previous.max(left_rank).max(right_rank)
            {
                return Ok(None);
            }
            words.insert(word, id);
            ordered.push((word, id));
            previous = id;
        }
        for &(word, expected) in &ordered {
            let length = word.chars().count();
            let Some(cost) = length.checked_mul(length.saturating_sub(1)).map(|n| n / 2) else {
                return Ok(None);
            };
            let Some(next) = remaining.checked_sub(cost) else {
                return Ok(None);
            };
            remaining = next;
            if self.tokenize_single_id(word)? != Some(expected) {
                return Ok(None);
            }
        }
        Ok(Some(
            ordered
                .into_iter()
                .map(|(word, id)| (word.chars().map(|ch| decoder[ch as usize]).collect(), id))
                .collect(),
        ))
    }

    fn tokenize_with_cache(&self, sequence: &str) -> Result<Vec<Token>> {
        if self.ignore_merges {
            if let Some(id) = self.vocab.get(sequence) {
                return Ok(vec![Token::new(
                    *id,
                    sequence.to_string(),
                    (0, sequence.len()),
                )]);
            }
        }
        if let Some(ref hit) = self.cache.as_ref().and_then(|c| c.get(sequence)) {
            return Ok(self.word_to_tokens(hit).collect());
        }
        let word = self.merge_word(sequence)?;
        let ret = self.word_to_tokens(&word).collect();
        if let Some(ref cache) = self.cache {
            if sequence.len() < MAX_LENGTH {
                cache.set(sequence.to_owned(), word);
            }
        }
        Ok(ret)
    }
}

impl Model for BPE {
    type Trainer = BpeTrainer;

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone().into_iter().collect()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        if sequence.is_empty() {
            return Ok(vec![]);
        }

        if self.dropout.is_none() || self.dropout == Some(0.0) {
            self.tokenize_with_cache(sequence)
        } else {
            let word = self.merge_word(sequence)?;
            Ok(self.word_to_tokens(&word).collect())
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        let vocab_file_name = match name {
            Some(name) => format!("{name}-vocab.json"),
            None => "vocab.json".to_string(),
        };

        // Write vocab.json
        let vocab_path: PathBuf = [folder, Path::new(vocab_file_name.as_str())]
            .iter()
            .collect();
        let mut vocab_file = File::create(&vocab_path)?;
        let order_vocab_iter = OrderedVocabIter::new(&self.vocab_r);
        let serialized = serde_json::to_string(&order_vocab_iter)?;
        vocab_file.write_all(serialized.as_bytes())?;

        // Write merges.txt
        let merges_file_name = match name {
            Some(name) => format!("{name}-merges.txt"),
            None => "merges.txt".to_string(),
        };

        let merges_path: PathBuf = [folder, Path::new(merges_file_name.as_str())]
            .iter()
            .collect();
        let mut merges_file = File::create(&merges_path)?;
        let mut merges: Vec<(&Pair, &u32)> = self
            .merges
            .iter()
            .map(|(pair, (rank, _))| (pair, rank))
            .collect();
        merges.sort_unstable_by_key(|k| *k.1);
        merges_file.write_all(b"#version: 0.2\n")?;
        merges_file.write_all(
            &merges
                .into_iter()
                .flat_map(|(pair, _)| {
                    format!("{} {}\n", self.vocab_r[&pair.0], self.vocab_r[&pair.1]).into_bytes()
                })
                .collect::<Vec<_>>()[..],
        )?;

        Ok(vec![vocab_path, merges_path])
    }

    fn get_trainer(&self) -> BpeTrainer {
        BpeTrainer::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn check_single_ids(model: BPE, inputs: &[&str]) {
        let other = model.clone();
        for _ in 0..2 {
            for input in inputs {
                let tokens = model.tokenize(input).unwrap();
                let expected = if tokens.len() == 1 {
                    Some(tokens[0].id)
                } else {
                    None
                };
                assert_eq!(other.tokenize_single_id(input).unwrap(), expected);
            }
        }
    }

    #[test]
    fn single_id_preserves_empty_multiple_and_cached_results() {
        let vocab: Vocab = [("a".into(), 0), ("b".into(), 1), ("ab".into(), 2)]
            .iter()
            .cloned()
            .collect();
        let model = BPE::builder()
            .vocab_and_merges(vocab, vec![("a".into(), "b".into())])
            .build()
            .unwrap();
        check_single_ids(model, &["", "a", "ab", "aba", "unknown", "baba"]);
    }

    #[test]
    fn single_id_preserves_ignore_merges_and_zero_dropout() {
        for ignore in [false, true] {
            let vocab: Vocab = [("a".into(), 0), ("b".into(), 1), ("ab".into(), 2)]
                .iter()
                .cloned()
                .collect();
            let model = BPE::builder()
                .vocab_and_merges(vocab, vec![])
                .ignore_merges(ignore)
                .dropout(0.0)
                .build()
                .unwrap();
            check_single_ids(model, &["", "a", "ab", "aba", "unknown"]);
        }
    }

    #[test]
    fn single_id_preserves_unknown_fusion() {
        for fuse in [false, true] {
            let vocab: Vocab = [("a".into(), 0), ("<unk>".into(), 1)]
                .iter()
                .cloned()
                .collect();
            let model = BPE::builder()
                .vocab_and_merges(vocab, vec![])
                .unk_token("<unk>".into())
                .fuse_unk(fuse)
                .build()
                .unwrap();
            check_single_ids(model, &["", "a", "z", "zz", "az", "🙂"]);
        }
    }

    #[test]
    fn single_id_preserves_missing_unknown_error() {
        let model = BPE::builder().unk_token("<unk>".into()).build().unwrap();
        assert_eq!(
            model.tokenize("z").unwrap_err().to_string(),
            model.tokenize_single_id("z").unwrap_err().to_string()
        );
    }

    #[test]
    fn test_ordered_vocab_iter() {
        let vocab_r: VocabR = [
            (0, "a".into()),
            (1, "b".into()),
            (2, "c".into()),
            (3, "ab".into()),
        ]
        .iter()
        .cloned()
        .collect();
        let order_vocab_iter = OrderedVocabIter::new(&vocab_r);
        let serialized = serde_json::to_string(&order_vocab_iter).unwrap();
        assert_eq!(serialized, "{\"a\":0,\"b\":1,\"c\":2,\"ab\":3}");
    }

    #[test]
    fn test_unk_not_fused() {
        let vocab: Vocab = [("<unk>".into(), 0), ("a".into(), 1), ("b".into(), 2)]
            .iter()
            .cloned()
            .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![])
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        let tokens = bpe.tokenize("c").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 1)),]);

        let tokens = bpe.tokenize("cc").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(0u32, "<unk>".into(), (0, 1)),
                Token::new(0u32, "<unk>".into(), (1, 2)),
            ]
        );

        let tokens = bpe.tokenize("accb").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(1u32, "a".into(), (0, 1)),
                Token::new(0u32, "<unk>".into(), (1, 2)),
                Token::new(0u32, "<unk>".into(), (2, 3)),
                Token::new(2u32, "b".into(), (3, 4)),
            ]
        );
    }
    #[test]
    fn test_unk_get_fused() {
        let vocab: Vocab = [("<unk>".into(), 0), ("a".into(), 1), ("b".into(), 2)]
            .iter()
            .cloned()
            .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![])
            .unk_token("<unk>".to_string())
            .fuse_unk(true)
            .build()
            .unwrap();
        let tokens = bpe.tokenize("c").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 1)),]);

        let tokens = bpe.tokenize("cc").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 2)),]);

        let tokens = bpe.tokenize("accb").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(1u32, "a".into(), (0, 1)),
                Token::new(0u32, "<unk>".into(), (1, 3)),
                Token::new(2u32, "b".into(), (3, 4)),
            ]
        );
    }

    #[test]
    // Test tokenization. With dropout set to 0 tokenization is deterministic,
    // so we know exactly what the result should be.
    //
    // To test this, we'll build a simple model to tokenize the word 'unrelated'.
    fn test_tokenize_with_and_without_dropout() {
        let vocab: Vocab = [
            ("u".into(), 0),
            ("n".into(), 1),
            ("r".into(), 2),
            ("e".into(), 3),
            ("l".into(), 4),
            ("a".into(), 5),
            ("t".into(), 6),
            ("d".into(), 7),
            ("re".into(), 8),
            ("at".into(), 9),
            ("ed".into(), 10),
            ("un".into(), 11),
            ("ated".into(), 12),
            ("rel".into(), 13),
            ("related".into(), 14),
            ("unrelated".into(), 15),
        ]
        .iter()
        .cloned()
        .collect();
        let merges: Merges = vec![
            ("r".to_string(), "e".to_string()),
            ("a".to_string(), "t".to_string()),
            ("e".to_string(), "d".to_string()),
            ("u".to_string(), "n".to_string()),
            ("at".to_string(), "ed".to_string()),
            ("re".to_string(), "l".to_string()),
            ("rel".to_string(), "ated".to_string()),
            ("un".to_string(), "related".to_string()),
        ];
        let mut bpe = BPE::new(vocab, merges);

        // With no dropout:
        let tokens = bpe.tokenize("unrelated").unwrap();
        assert_eq!(tokens, vec![Token::new(15u32, "unrelated".into(), (0, 9))]);

        // With dropout = 0.0 (equivalent to dropout == none)
        bpe.dropout = Some(0.0);
        let tokens = bpe.tokenize("unrelated").unwrap();
        assert_eq!(tokens, vec![Token::new(15u32, "unrelated".into(), (0, 9))]);

        // Now set dropout to 1.0. Result should be no merges performed.
        bpe.dropout = Some(1.0);
        let tokens = bpe.tokenize("unrelated").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(0u32, "u".into(), (0, 1)),
                Token::new(1u32, "n".into(), (1, 2)),
                Token::new(2u32, "r".into(), (2, 3)),
                Token::new(3u32, "e".into(), (3, 4)),
                Token::new(4u32, "l".into(), (4, 5)),
                Token::new(5u32, "a".into(), (5, 6)),
                Token::new(6u32, "t".into(), (6, 7)),
                Token::new(3u32, "e".into(), (7, 8)),
                Token::new(7u32, "d".into(), (8, 9)),
            ]
        );

        // Now try with dropout between 0 and 1.
        bpe.dropout = Some(0.5);
        let tokens = bpe.tokenize("unrelated").unwrap();
        assert!(!tokens.is_empty() && tokens.len() <= 9);
    }

    #[test]
    // Ensure `BPE::from_file` works as expected.
    fn test_bpe_from_file() {
        // Set up vocab file.
        let mut vocab_file = NamedTempFile::new().unwrap();
        vocab_file
            .write_all(b"{\"a\": 0, \"b\": 1, \"c\": 2, \"ab\": 3}")
            .unwrap();

        // Set up merges file.
        let mut merges_file = NamedTempFile::new().unwrap();
        merges_file.write_all(b"#version: 0.2\na b").unwrap();

        // Make sure we can instantiate a BPE model from the files.
        let builder = BPE::from_file(
            vocab_file.path().to_str().unwrap(),
            merges_file.path().to_str().unwrap(),
        );
        let bpe = builder.build().unwrap();

        // Check merges.
        assert_eq!(bpe.merges.get(&(0, 1)).unwrap(), &(0u32, 3u32));

        // Check vocab.
        assert_eq!(bpe.vocab.get("a").unwrap(), &0u32);
        assert_eq!(bpe.vocab.get("b").unwrap(), &1u32);
        assert_eq!(bpe.vocab.get("c").unwrap(), &2u32);
        assert_eq!(bpe.vocab.get("ab").unwrap(), &3u32);
    }

    #[test]
    // Ensure BPEBuilder with dropout = 0.0 doesn't error
    fn test_bpe_with_dropout_0() {
        let bpe = BPE::builder().dropout(0.0).build().unwrap();
        assert_eq!(bpe.dropout, Some(0.0));
    }

    #[test]
    // Ensure `BPE::from_file` works as expected.
    fn test_bpe_with_continuing_subword_prefix() {
        let vocab: Vocab = vec![
            ("a".to_string(), 0),
            ("##b".to_string(), 1),
            ("##c".to_string(), 2),
            ("ab".to_string(), 3),
            ("abc".to_string(), 4),
        ]
        .into_iter()
        .collect();

        let merges = vec![
            ("a".to_string(), "##b".to_string()),
            ("ab".to_string(), "##c".to_string()),
        ];

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .unk_token("[UNK]".to_string())
            .continuing_subword_prefix("##".to_string())
            .build()
            .unwrap();

        let res = bpe.tokenize("ab");
        assert_eq!(
            res.unwrap(),
            vec![Token {
                id: 3,
                value: "ab".to_string(),
                offsets: (0, 2)
            }]
        );
        let res = bpe.tokenize("abc");
        assert_eq!(
            res.unwrap(),
            vec![Token {
                id: 4,
                value: "abc".to_string(),
                offsets: (0, 3)
            }]
        );
    }

    #[test]
    // Ensure `MergeTokenOutOfVocabulary` error is returned when it should be.
    fn test_bpe_from_file_merge_token_oov() {
        // Set up vocab file.
        let mut vocab_file = NamedTempFile::new().unwrap();
        vocab_file
            .write_all(b"{\"a\": 0, \"b\": 1, \"c\": 2, \"ab\": 3}")
            .unwrap();

        // Set up merges file.
        let mut merges_file = NamedTempFile::new().unwrap();
        merges_file.write_all(b"#version: 0.2\na b\na d").unwrap();

        // Ensure the result of BPE::from_file is a MergeTokenOutOfVocabulary error.
        match BPE::from_file(
            vocab_file.path().to_str().unwrap(),
            merges_file.path().to_str().unwrap(),
        )
        .build()
        {
            Ok(_) => unreachable!(),
            Err(err) => match err.downcast_ref::<Error>() {
                Some(Error::MergeTokenOutOfVocabulary(token)) => {
                    assert_eq!(*token, String::from("d"))
                }
                _ => unreachable!(),
            },
        }
    }

    #[test]
    // Ensure `BadMerges` error is returned when there is an invalid line in the
    // merges.txt file.
    fn test_bpe_from_file_bad_merges() {
        // Set up vocab file.
        let mut vocab_file = NamedTempFile::new().unwrap();
        vocab_file
            .write_all("{\"a\": 0, \"b\": 1, \"c\": 2, \"ab\": 3}".as_bytes())
            .unwrap();

        // Set up merges file with a bad line.
        let mut merges_file = NamedTempFile::new().unwrap();
        merges_file.write_all(b"#version: 0.2\na b\nc").unwrap();

        // Ensure the result of BPE::from_file is a BadMerges error.
        match BPE::from_file(
            vocab_file.path().to_str().unwrap(),
            merges_file.path().to_str().unwrap(),
        )
        .build()
        {
            Ok(_) => unreachable!(),
            Err(err) => match err.downcast_ref::<Error>() {
                Some(Error::BadMerges(line)) => assert_eq!(*line, 2),
                _ => unreachable!(),
            },
        }
    }

    #[test]
    fn test_bpe_byte_fallback() {
        // 0x61 == 'a' in bytes
        let vocab: Vocab = [("<unk>".into(), 0), ("<0x61>".into(), 1)]
            .iter()
            .cloned()
            .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![])
            .unk_token("<unk>".to_string())
            .byte_fallback(true)
            .build()
            .unwrap();
        let tokens = bpe.tokenize("c").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 1)),]);

        let tokens = bpe.tokenize("a").unwrap();
        assert_eq!(tokens, vec![Token::new(1u32, "<0x61>".into(), (0, 1)),]);
    }

    #[test]
    fn test_bpe_byte_fallback_newline() {
        // 0x0A == '\n' in bytes
        let vocab: Vocab = [("<unk>".into(), 0), ("<0x0A>".into(), 1)]
            .iter()
            .cloned()
            .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![])
            .unk_token("<unk>".to_string())
            .byte_fallback(true)
            .build()
            .unwrap();
        let tokens = bpe.tokenize("\n").unwrap();
        assert_eq!(tokens, vec![Token::new(1u32, "<0x0A>".into(), (0, 1)),]);
    }

    #[test]
    fn test_ignore_merges() {
        // 0x0A == '\n' in bytes
        let vocab: Vocab = [
            (".:.:".into(), 0),
            ("Ġbelirtilen".into(), 1),
            (".".into(), 2),
            (":".into(), 3),
            ("bel".into(), 4),
            ("irtilen".into(), 5),
            ("Ġ".into(), 6),
            (".:".into(), 7),
            ("belirtilen".into(), 8),
            (".:.".into(), 9),
            ("be".into(), 10),
            ("l".into(), 11),
            ("ir".into(), 12),
            ("ti".into(), 13),
            ("en".into(), 14),
            ("irtil".into(), 15),
            ("irti".into(), 16),
            ("i".into(), 17),
            ("r".into(), 18),
            ("t".into(), 19),
            ("b".into(), 20),
            ("e".into(), 21),
            ("n".into(), 22),
        ]
        .iter()
        .cloned()
        .collect();
        let mut bpe = BpeBuilder::default()
            .vocab_and_merges(
                vocab,
                vec![
                    (".".into(), ":".into()),
                    ("b".into(), "e".into()),
                    ("be".into(), "l".into()),
                    ("i".into(), "r".into()),
                    ("t".into(), "i".into()),
                    ("ir".into(), "ti".into()),
                    ("e".into(), "n".into()),
                    ("irti".into(), "l".into()),
                ],
            )
            .ignore_merges(true)
            .build()
            .unwrap();
        let tokens = bpe.tokenize(".:.:").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, ".:.:".into(), (0, 4))]);

        let tokens = bpe.tokenize("Ġbelirtilen").unwrap();
        assert_eq!(
            tokens,
            vec![Token::new(1u32, "Ġbelirtilen".into(), (0, 12))]
        );

        bpe.ignore_merges = false;

        let tokens = bpe.tokenize(".:.:").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(7u32, ".:".into(), (0, 2)),
                Token::new(7u32, ".:".into(), (2, 4))
            ]
        );

        let tokens = bpe.tokenize("Ġbelirtilen").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token {
                    id: 6,
                    value: "Ġ".into(),
                    offsets: (0, 2)
                },
                Token {
                    id: 4,
                    value: "bel".into(),
                    offsets: (2, 5)
                },
                Token {
                    id: 15,
                    value: "irtil".into(),
                    offsets: (5, 10)
                },
                Token {
                    id: 14,
                    value: "en".into(),
                    offsets: (10, 12)
                }
            ]
        )
    }
}

#[cfg(test)]
mod native_byte_tables_tests {
    use super::*;

    fn model(merges: Merges) -> BPE {
        let mut hidden = 256u32;
        let mut vocab = Vocab::new();
        for byte in 0u32..256 {
            let point = if (33..127).contains(&byte)
                || (161..173).contains(&byte)
                || (174..256).contains(&byte)
            {
                byte
            } else {
                let p = hidden;
                hidden += 1;
                p
            };
            vocab.insert(char::from_u32(point).unwrap().to_string(), byte + 100);
        }
        for (index, (a, b)) in merges.iter().enumerate() {
            vocab.insert(format!("{a}{b}"), 1000 + index as u32);
        }
        BPE::new(vocab, merges)
    }

    #[test]
    fn byte_table_proof_preserves_boundaries_and_bytes() {
        let bpe = model(vec![("a".into(), "b".into())]);
        let rows = bpe.proved_byte_ranks(257, 2, 258, 1).unwrap().unwrap();
        assert_eq!(rows.len(), 257);
        let ranks = rows.into_iter().collect::<HashMap<_, _>>();
        for byte in 0u32..256 {
            assert_eq!(ranks[&vec![byte as u8]], byte + 100);
        }
        assert_eq!(ranks[&b"ab".to_vec()], 1000);
        assert!(bpe.proved_byte_ranks(256, 2, 258, 1).unwrap().is_none());
        assert!(bpe.proved_byte_ranks(257, 1, 258, 1).unwrap().is_none());
        assert!(bpe.proved_byte_ranks(257, 2, 257, 1).unwrap().is_none());
        assert!(bpe.proved_byte_ranks(257, 2, 258, 0).unwrap().is_none());
    }

    #[test]
    fn byte_table_proof_checks_actual_reduction() {
        let bpe = model(vec![
            ("a".into(), "b".into()),
            ("b".into(), "c".into()),
            ("a".into(), "bc".into()),
        ]);
        assert!(bpe
            .proved_byte_ranks(1000, 4096, 1 << 20, 1 << 20)
            .unwrap()
            .is_none());
    }

    #[test]
    fn byte_table_proof_counts_unicode_characters() {
        let bpe = model(vec![("ñ".into(), "b".into())]);
        let rows = bpe.proved_byte_ranks(257, 2, 258, 1).unwrap().unwrap();
        assert_eq!(rows.last(), Some(&(vec![0xf1, b'b'], 1000)));
    }

    #[test]
    fn empty_affixes_preserve_tokens_offsets_and_proof_limits() {
        let plain = model(vec![("a".into(), "b".into())]);
        let mut empty = plain.clone();
        empty.continuing_subword_prefix = Some(String::new());
        empty.end_of_word_suffix = Some(String::new());
        for input in ["", "a", "ab", "aba", "é", "中", "a中b", "🦀"] {
            assert_eq!(
                plain.tokenize(input).unwrap(),
                empty.tokenize(input).unwrap()
            );
            assert_eq!(
                plain.tokenize_single_id(input).unwrap(),
                empty.tokenize_single_id(input).unwrap()
            );
        }
        for limits in [(257, 2, 258, 1), (256, 2, 258, 1), (257, 2, 258, 0)] {
            assert_eq!(
                plain
                    .proved_byte_ranks(limits.0, limits.1, limits.2, limits.3)
                    .unwrap(),
                empty
                    .proved_byte_ranks(limits.0, limits.1, limits.2, limits.3)
                    .unwrap()
            );
        }
        assert_eq!(empty.continuing_subword_prefix.as_deref(), Some(""));
        assert_eq!(empty.end_of_word_suffix.as_deref(), Some(""));
    }

    #[test]
    fn empty_affixes_preserve_unknown_fusion_fallback_and_errors() {
        for byte_fallback in [false, true] {
            for fuse_unk in [false, true] {
                for valid_unk in [false, true] {
                    let mut vocab: Vocab =
                        vec![("a".into(), 0), ("<0xC3>".into(), 1), ("<0xA9>".into(), 2)]
                            .into_iter()
                            .collect();
                    if valid_unk {
                        vocab.insert("[UNK]".into(), 3);
                    }
                    let plain = BPE::builder()
                        .vocab_and_merges(vocab, vec![])
                        .unk_token("[UNK]".into())
                        .fuse_unk(fuse_unk)
                        .byte_fallback(byte_fallback)
                        .build()
                        .unwrap();
                    let mut empty = plain.clone();
                    empty.continuing_subword_prefix = Some(String::new());
                    empty.end_of_word_suffix = Some(String::new());
                    for input in ["", "a", "é", "é中", "a中🦀a"] {
                        match (plain.tokenize(input), empty.tokenize(input)) {
                            (Ok(left), Ok(right)) => assert_eq!(left, right),
                            (Err(left), Err(right)) => {
                                assert_eq!(left.to_string(), right.to_string())
                            }
                            other => panic!("empty affixes changed tokenization: {other:?}"),
                        }
                    }
                }
            }
        }
    }
}
