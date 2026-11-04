"""The trained BPE tokenizer, loaded from its folder (tokenizer_config.json, vocab.json, merges.txt): the encode and
decode part of the cs336 assignment's tokenizer, without the training code."""
import json
import os
from functools import cache
from itertools import pairwise

import regex as re

CONFIG_FILE = "tokenizer_config.json"
VOCAB_FILE = "vocab.json"
MERGES_FILE = "merges.txt"

# GPT-2's pretoken pattern: contractions, words, numbers, punctuation runs, whitespace.
PAT = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def read_tokenizer_config(tokenizer_dir: str | os.PathLike) -> dict:
    with open(os.path.join(tokenizer_dir, CONFIG_FILE)) as f:
        return json.load(f)


@cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """Every byte to a printable character, as in the GPT-2 code: printable bytes keep their character, the other 68 are
    shifted past 256. vocab.json and merges.txt store tokens in this form."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


def split_text_to_chunks(text: bytes, sorted_special_tokens: list[bytes]) -> list[bytes]:
    """Splits the text at the special tokens and keeps the special tokens as chunks of their own. The special tokens must
    be sorted from longest to shortest."""
    if not sorted_special_tokens:
        return [text]
    split_pattern = re.compile(b'(' + b'|'.join(re.escape(x) for x in sorted_special_tokens) + b')')
    return [c for c in split_pattern.split(text) if c]


def pretokenize_text(text: bytes) -> list[bytes]:
    return PAT.findall(text)


def merge_pretoken(pair: tuple[bytes, bytes], pretoken: tuple[bytes, ...]) -> tuple[bytes, ...]:
    new_pretoken: tuple[bytes, ...] = ()
    i: int = 0
    pretoken_len: int = len(pretoken)
    while i < pretoken_len:
        if i < pretoken_len - 1 and (pretoken[i], pretoken[i + 1]) == pair:
            new_pretoken += (pretoken[i] + pretoken[i + 1],)
            i += 2
        else:
            new_pretoken += (pretoken[i],)
            i += 1
    return new_pretoken


def get_byte_pairs(pretoken: tuple[bytes, ...]) -> set[tuple[bytes, bytes]]:
    return set(pairwise(pretoken))


class Tokenizer:
    vocab: dict[int, bytes]
    id_vocab: dict[bytes, int]
    merges_rank: dict[tuple[bytes, bytes], int]
    special_tokens: list[bytes]  # longest first

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        self.vocab = vocab
        self.id_vocab = {token: token_id for token_id, token in vocab.items()}
        self.merges_rank = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = sorted((x.encode("utf-8") for x in special_tokens), key=len, reverse=True)
        self._encoded_pretokens: dict[bytes, list[int]] = {}  # the same word encodes the same way every time

    @classmethod
    def from_dir(cls, tokenizer_dir: str | os.PathLike) -> "Tokenizer":
        special_tokens: list[str] = read_tokenizer_config(tokenizer_dir)["special_tokens"]
        byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        def to_bytes(token: str) -> bytes:
            return bytes([byte_decoder[character] for character in token])

        with open(os.path.join(tokenizer_dir, VOCAB_FILE)) as f:
            vocab = {token_id: to_bytes(token) for token, token_id in json.load(f).items()}
        merges = []
        with open(os.path.join(tokenizer_dir, MERGES_FILE)) as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) == 2:
                    merges.append((to_bytes(parts[0]), to_bytes(parts[1])))
        # A special token missing from the vocabulary gets the next id.
        for special_token in special_tokens:
            if special_token.encode("utf-8") not in set(vocab.values()):
                vocab[len(vocab)] = special_token.encode("utf-8")
        return cls(vocab, merges, special_tokens)

    def special_token_id(self, token: str) -> int:
        return self.id_vocab[token.encode("utf-8")]

    def special_token_ids(self) -> set[int]:
        return {self.id_vocab[token] for token in self.special_tokens}

    def encode(self, text: str) -> list[int]:
        encoded_ids: list[int] = []
        for chunk in split_text_to_chunks(text.encode("utf-8"), self.special_tokens):
            if chunk in self.special_tokens:
                encoded_ids.append(self.id_vocab[chunk])
            else:
                for pretoken in pretokenize_text(chunk):
                    encoded_ids += self._encode_pretoken(pretoken)
        return encoded_ids

    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        if pretoken not in self._encoded_pretokens:
            tuple_bytes: tuple[bytes, ...] = tuple(pretoken[i:i + 1] for i in range(len(pretoken)))
            while True:
                candidates = [pair for pair in get_byte_pairs(tuple_bytes) if pair in self.merges_rank]
                if not candidates:
                    break
                tuple_bytes = merge_pretoken(min(candidates, key=self.merges_rank.get), tuple_bytes)
            self._encoded_pretokens[pretoken] = [self.id_vocab[x] for x in tuple_bytes]
        return self._encoded_pretokens[pretoken]

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")
