from __future__ import annotations

import random
import struct
from typing import Any

import pytest
from tokenizers.models import BPE


def scalar(model: BPE, words: dict[str, Any], remaining: int) -> bool:
    for word, expected in words.items():
        remaining -= len(word) * (len(word) - 1) // 2
        if remaining < 0:
            return False
        tokens = model.tokenize(word)
        if len(tokens) != 1 or tokens[0].id != expected:
            return False
    return True


def model() -> BPE:
    return BPE({"a": 0, "b": 1, "ab": 2, "aba": 3, "ba": 4}, [("a", "b"), ("ab", "a")])


@pytest.mark.parametrize("quota", [0, 1, 2, 3, 4, 5, 9, 64_000_000])
@pytest.mark.parametrize(
    "words",
    [
        {},
        {"a": 0},
        {"a": 0, "ab": 2, "aba": 3},
        {"ab": 1},
        {"ba": 4},
        {"": 0},
        {"unknown": 0},
        {"a": True},
        {"b": True},
        {"a": -1},
        {"a": 2**80},
        {"aba": 3, "ab": 2, "a": 0},
    ],
)
def test_producer_matches_scalar_prefix_and_quota(words: dict[str, Any], quota: int) -> None:
    assert model()._prove_model_words(words, quota) == scalar(model(), words, quota)


def test_unicode_quota_counts_characters() -> None:
    for quota in range(5):

        def make() -> BPE:
            return BPE({"é": 0, "中": 1, "é中": 2}, [("é", "中")])

        words = {"é": 0, "中": 1, "é中": 2}
        assert make()._prove_model_words(words, quota) == scalar(make(), words, quota)


def test_random_words_and_mismatches() -> None:
    rng = random.Random(519007)
    for _ in range(300):
        words = {"".join(rng.choices("ab", k=rng.randrange(0, 14))): rng.randrange(-1, 6) for _ in range(9)}
        quota = rng.randrange(0, 100)
        assert model()._prove_model_words(words, quota) == scalar(model(), words, quota)


def test_expected_id_comparisons_are_ordered_and_short_circuit() -> None:
    class Expected:
        def __init__(self, name: str, value: int, seen: list[str]) -> None:
            self.name, self.value, self.seen = name, value, seen

        def __eq__(self, actual: object) -> bool:
            self.seen.append(self.name)
            return actual == self.value

    results = []
    for fast in [False, True]:
        seen: list[str] = []
        words = {word: Expected(word, value, seen) for word, value in [("a", 0), ("ab", 99), ("aba", 3)]}
        m = model()
        result = m._prove_model_words(words, 100) if fast else scalar(m, words, 100)
        results.append((result, seen))
    assert results == [(False, ["a", "ab"]), (False, ["a", "ab"])]


def test_comparison_runs_after_model_read_guard_is_released() -> None:
    # Setting a model property takes the write lock; this must not deadlock.
    m = model()

    class Expected:
        def __eq__(self, actual: object) -> bool:
            m.dropout = 0.0
            return actual == 0

    assert m._prove_model_words({"a": Expected()}, 0)


@pytest.mark.parametrize("word", [chr(0xD800), chr(0xDFFF), "a" + chr(0xD800)])
def test_surrogate_failure_and_quota_order(word: str) -> None:
    failures = []
    for fast in [False, True]:
        with pytest.raises(UnicodeEncodeError) as exc:
            m = model()
            if fast:
                m._prove_model_words({word: 0}, 100)
            else:
                scalar(m, {word: 0}, 100)
        failures.append((exc.value.encoding, exc.value.object, exc.value.start, exc.value.end, exc.value.reason))
    assert failures[0] == failures[1]
    assert not model()._prove_model_words({"a\ud800": 0}, 0)


def packed_model() -> tuple[BPE, bytes]:
    visible = [*range(33, 127), *range(161, 173), *range(174, 256)]
    hidden = [byte for byte in range(256) if byte not in visible]
    symbols = {byte: chr(byte) for byte in visible}
    symbols.update({byte: chr(256 + index) for index, byte in enumerate(hidden)})
    vocab = {symbols[byte]: byte for byte in range(256)}
    vocab.update({"ab": 256, "aba": 257, "special": 258})
    m = BPE(vocab, [("a", "b"), ("ab", "a")])
    rows = [(byte, bytes([byte])) for byte in visible + hidden]
    rows += [(256, b"ab"), (257, b"aba")]
    return m, b"".join(struct.pack("<II", ordinal, len(word)) + word for ordinal, word in rows)


def test_packed_table_bytes_and_added_membership() -> None:
    m, expected = packed_model()
    entries = [{"content": "ab", "id": 256}, {"content": "special", "id": 258}, {"content": "new", "id": 999}]
    assert m._native_packed_tables(259, 3, 261, 4, entries) == (
        expected,
        {"ab": 256, "special": 258},
        {256, 258},
        {256},
    )


@pytest.mark.parametrize(
    "limits,accepted",
    [
        ((259, 3, 261, 4), True),
        ((258, 3, 261, 4), False),
        ((259, 2, 261, 4), False),
        ((259, 3, 260, 4), False),
        ((259, 3, 261, 3), False),
    ],
)
def test_packed_table_limits(limits: tuple[int, int, int, int], accepted: bool) -> None:
    m, expected = packed_model()
    result = m._native_packed_tables(*limits, [])
    assert (result is not None) == accepted
    if result is not None:
        assert result == (expected, {}, set(), set())
