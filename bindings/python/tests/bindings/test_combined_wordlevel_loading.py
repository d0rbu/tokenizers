"""Model loading and vocabulary cardinality agree after real mutations."""

import json

import pytest
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel


@pytest.mark.parametrize("tag_first", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_loaded_wordlevel_union_tracks_added_tokens_and_model_replacement(tag_first, overlap):
    model = {"vocab": {"<unk>": 0, "alpha": 7, "beta": 9}, "unk_token": "<unk>"}
    model = {"type": "WordLevel", **model} if tag_first else {**model, "type": "WordLevel"}
    current = Tokenizer.from_str(json.dumps({"model": model}))
    current.add_tokens([AddedToken("alpha" if overlap else "gamma", normalized=False), "delta"])
    for tokenizer in (current, Tokenizer.from_str(current.to_str())):
        before = tokenizer.get_vocab(True)
        assert tokenizer.get_vocab_size(True) == len(before)
        assert tokenizer.encode("alpha").ids == [7]
        tokenizer.model = WordLevel({"<unk>": 0, "gamma": 2, "delta": 5}, unk_token="<unk>")
        after = tokenizer.get_vocab(True)
        assert tokenizer.get_vocab_size(True) == len(after)
        assert tokenizer.get_vocab_size(False) == 3
        assert tokenizer.encode("gamma").ids == [after["gamma"]]
