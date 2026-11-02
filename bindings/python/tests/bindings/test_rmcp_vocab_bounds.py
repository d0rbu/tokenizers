from __future__ import annotations

import itertools

import pytest
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece

from tokenizers import AddedToken, Tokenizer


def original_bound(t, entries, chars, total):
    used = set()
    count = 0
    for word, ordinal in t.get_vocab(with_added_tokens=True).items():
        if (
            type(word) is not str
            or len(word) > chars
            or type(ordinal) is not int
            or not (0 <= ordinal < 2**32 - 1)
            or ordinal >= entries
            or ordinal in used
        ):
            return False
        used.add(ordinal)
        count += len(word.encode("utf-8"))
        if count > total:
            return False
    return True


@pytest.mark.parametrize(
    "vocab",
    [{}, {"a": 0}, {"a": 0, "b": 1}, {"a": 0, "b": 0}, {"é": 0, "😀": 1, "á": 2}, {"longword": 7}, {"a": 2**32 - 1}],
)
def test_producer_bounds_match_original(vocab):
    t = Tokenizer(BPE(vocab, []))
    for entries, chars, total in itertools.product([0, 1, 2, 8, 2**32 - 1], [0, 1, 2, 4096], [0, 1, 3, 16, 2**20]):
        assert t._bounded_model_vocab(entries, chars, total) == original_bound(t, entries, chars, total)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BPE({"a": 0, "b": 1}, []),
        lambda: WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]"),
        lambda: WordPiece({"[UNK]": 0, "a": 1}, unk_token="[UNK]"),
        lambda: Unigram([("<unk>", 0.0), ("a", -1.0), ("b", -2.0)], 0),
    ],
)
def test_size_matches_overlay_after_added_token_changes(factory):
    t = Tokenizer(factory())
    for added in [[], ["a"], ["unseen"], [AddedToken("another", special=True)], ["b", "a", "unseen"], ["é", "😀"]]:
        t.add_tokens(added)
        for include in [False, True]:
            assert t.get_vocab_size(include) == len(t.get_vocab(include))
        for limits in [(100, 4096, 1024), (2, 1, 2), (0, 0, 0)]:
            assert t._bounded_model_vocab(*limits) == original_bound(t, *limits)


def test_unigram_duplicate_spelling_keeps_original_cardinality():
    t = Tokenizer(Unigram([("<unk>", 0.0), ("a", -1.0), ("a", -2.0)], 0))
    assert t.get_vocab_size(False) == 3
    assert t.get_vocab_size(True) == 2
    t.add_tokens(["a", "new"])
    assert t.get_vocab_size(False) == 3
    assert t.get_vocab_size(True) == len(t.get_vocab(True)) == 3
