import json
import os

import pytest

from tokenizer import Tokenizer, gpt2_bytes_to_unicode

END_OF_TEXT = "<|endoftext|>"


@pytest.fixture
def tiny_tokenizer(tmp_path) -> Tokenizer:
    """A tokenizer folder with the 256 byte tokens, one special token and three merges, so " the" is one token."""
    to_unicode = gpt2_bytes_to_unicode()
    space = to_unicode[ord(" ")]
    merges = [("t", "h"), ("th", "e"), (space, "the")]
    vocab = {END_OF_TEXT: 0}
    for byte in range(256):
        vocab[to_unicode[byte]] = len(vocab)
    for left, right in merges:
        vocab[left + right] = len(vocab)

    with open(os.path.join(tmp_path, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(os.path.join(tmp_path, "merges.txt"), "w", encoding="utf-8") as f:
        f.writelines(f"{left} {right}\n" for left, right in merges)
    with open(os.path.join(tmp_path, "tokenizer_config.json"), "w") as f:
        json.dump({"vocab_size": len(vocab), "special_tokens": [END_OF_TEXT]}, f)
    return Tokenizer.from_dir(tmp_path)


def test_decoding_the_encoded_text_gives_the_text_back(tiny_tokenizer):
    text = "The quick brown fox: 42 jumps!\n\tOver a lazy dog — über, 日本語, 🚀."

    assert tiny_tokenizer.decode(tiny_tokenizer.encode(text)) == text


def test_a_special_token_in_the_text_is_one_id_and_survives_the_round_trip(tiny_tokenizer):
    text = f"first story{END_OF_TEXT}second story"

    ids = tiny_tokenizer.encode(text)

    assert ids.count(tiny_tokenizer.special_token_id(END_OF_TEXT)) == 1
    assert tiny_tokenizer.decode(ids) == text


def test_merges_apply_in_rank_order_so_a_merged_word_is_one_token(tiny_tokenizer):
    assert len(tiny_tokenizer.encode(" the")) == 1
    assert len(tiny_tokenizer.encode("the the")) == 2
    assert len(tiny_tokenizer.encode(" tha")) == 3  # "th" merges, "e" never follows, the space stays apart
