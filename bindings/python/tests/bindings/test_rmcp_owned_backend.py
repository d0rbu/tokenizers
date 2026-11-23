import json

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors


def test_owned_backend_copy_keeps_encoding_and_isolates_settings():
    original = Tokenizer(models.BPE({"a": 0, "b": 1, "ab": 2}, [("a", "b")]))
    original.normalizer = normalizers.Lowercase()
    original.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    original.post_processor = processors.TemplateProcessing(single="$A")
    original.decoder = decoders.ByteLevel()

    clone = original._copy_owned_backend()
    assert clone is not None
    assert json.loads(clone.to_str()) == json.loads(original.to_str())
    assert clone.encode("AB").ids == original.encode("AB").ids == [2]
    assert clone.decode([2]) == original.decode([2])

    clone.enable_padding(length=32)
    clone.enable_truncation(max_length=32)
    clone.pre_tokenizer.add_prefix_space = True
    clone.add_special_tokens(["private"])
    clone.model.dropout = 0.5

    assert original.padding is None
    assert original.truncation is None
    assert original.pre_tokenizer.add_prefix_space is False
    assert original.token_to_id("private") is None
    assert original.model.dropout is None
    assert clone._copy_owned_backend() is None


def test_owned_backend_copy_declines_custom_and_non_bpe_backends():
    class CustomPreTokenizer:
        def pre_tokenize(self, _sequence):
            pass

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(CustomPreTokenizer())
    assert tokenizer._copy_owned_backend() is None
    assert Tokenizer(models.WordLevel())._copy_owned_backend() is None
