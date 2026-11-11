"""Vocabulary cardinality remains the union of model and added spellings."""

import pytest
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece


@pytest.mark.parametrize("kind", ["wordlevel", "bpe", "wordpiece", "unigram"])
def test_vocab_size_matches_current_union_after_mutations(kind):
    vocab = {"[UNK]": 0, "hello": 1, "world": 2}
    constructors = {
        "wordlevel": lambda: WordLevel(vocab, unk_token="[UNK]"),
        "bpe": lambda: BPE(vocab, [], unk_token="[UNK]"),
        "wordpiece": lambda: WordPiece(vocab, unk_token="[UNK]"),
        "unigram": lambda: Unigram([(t, -float(i)) for t, i in vocab.items()], 0),
    }
    tokenizer = Tokenizer(constructors[kind]())

    def check(current):
        assert current.get_vocab_size(True) == len(current.get_vocab(True))
        assert current.get_vocab_size(False) == len(current.get_vocab(False))

    check(tokenizer)
    for tokens in (
        ["hello"],
        [AddedToken("new", normalized=False), "world", "new"],
        [AddedToken("hello", special=True), AddedToken("other", special=True)],
    ):
        tokenizer.add_tokens(tokens)
        check(tokenizer)
        check(Tokenizer.from_str(tokenizer.to_str()))
    # Public model replacement must not leave a cached count behind.
    tokenizer.model = WordLevel({"[UNK]": 0, "new": 7, "other": 9}, unk_token="[UNK]")
    check(tokenizer)


def test_wordlevel_duplicate_ids_still_count_distinct_spellings():
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "same-a": 2, "same-b": 2}))
    tokenizer.add_tokens(["same-a", "outside"])
    assert tokenizer.get_vocab_size(True) == len(tokenizer.get_vocab(True)) == 4
    assert tokenizer.get_vocab_size(False) == len(tokenizer.get_vocab(False)) == 3


def test_duplicate_unigram_spellings_keep_union_fallback():
    tokenizer = Tokenizer(Unigram([("[UNK]", 0.0), ("same", -1.0), ("same", -2.0)], 0))
    tokenizer.add_tokens(["same", "outside"])
    assert tokenizer.get_vocab_size(True) == len(tokenizer.get_vocab(True)) == 3
