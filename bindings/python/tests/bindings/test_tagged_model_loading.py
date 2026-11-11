"""Tagged and legacy model JSON retain public tokenizer behavior."""

import json

import pytest
from tokenizers import Tokenizer

MODELS = [
    {"type": "BPE", "vocab": {"<unk>": 0, "a": 1, "b": 2, "ab": 3}, "merges": [["a", "b"]], "unk_token": "<unk>"},
    {
        "type": "WordPiece",
        "vocab": {"<unk>": 0, "a": 1, "b": 2},
        "unk_token": "<unk>",
        "continuing_subword_prefix": "##",
        "max_input_chars_per_word": 100,
    },
    {"type": "WordLevel", "vocab": {"<unk>": 0, "a": 1, "b": 2}, "unk_token": "<unk>"},
    {"type": "Unigram", "vocab": [["<unk>", -2.0], ["a", -1.0], ["b", 0.0]], "unk_id": 0},
]


@pytest.mark.parametrize("model", MODELS, ids=lambda model: model["type"])
@pytest.mark.parametrize("tag_position", ["first", "last", "legacy"])
def test_model_tag_position_preserves_encoding(model, tag_position):
    fields = dict(model)
    kind = fields.pop("type")
    if tag_position == "first":
        fields = {"type": kind, **fields}
    elif tag_position == "last":
        fields["type"] = kind
    serialized = json.dumps({"model": fields, "added_tokens": []})
    tokenizer = Tokenizer.from_str(serialized)
    tokenizer.add_special_tokens(["<extra>"])
    restored = Tokenizer.from_str(tokenizer.to_str())
    assert restored.get_vocab() == tokenizer.get_vocab()
    assert restored.encode("ab").ids == tokenizer.encode("ab").ids
    assert restored.decode(restored.encode("ab").ids) == tokenizer.decode(tokenizer.encode("ab").ids)


def test_duplicate_model_field_keeps_last_value_compatibility():
    text = '{"model":{"type":"WordLevel","vocab":false,"vocab":{"<unk>":0,"a":1},"unk_token":"<unk>"}}'
    tokenizer = Tokenizer.from_str(text)
    assert tokenizer.get_vocab() == {"<unk>": 0, "a": 1}
    assert tokenizer.encode("a").ids == [1]


def test_invalid_tagged_merges_keep_original_error():
    model = {**MODELS[0], "merges": ["a b c"]}
    with pytest.raises(Exception, match="Merges text file invalid at line 1"):
        Tokenizer.from_str(json.dumps({"model": model}))


@pytest.mark.parametrize("model", MODELS, ids=lambda model: model["type"])
def test_duplicate_type_keeps_original_rejection(model):
    fields = dict(model)
    kind = fields.pop("type")
    body = json.dumps(fields)[:-1] + ',"type":' + json.dumps(kind) + ',"type":' + json.dumps(kind) + "}"
    with pytest.raises(Exception, match="data did not match any variant of untagged enum ModelUntagged"):
        Tokenizer.from_str('{"model":' + body + "}")


def test_duplicate_bpe_null_option_keeps_last_value():
    body = json.dumps(MODELS[0])[:-1] + ',"unk_token":null}'
    tokenizer = Tokenizer.from_str('{"model":' + body + "}")
    assert json.loads(tokenizer.to_str())["model"]["unk_token"] is None
