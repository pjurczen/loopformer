"""The trained BPE tokenizer's folder: tokenizer_config.json, vocab.json and merges.txt."""
import json
import os

CONFIG_FILE = "tokenizer_config.json"


def read_tokenizer_config(tokenizer_dir: str | os.PathLike) -> dict:
    with open(os.path.join(tokenizer_dir, CONFIG_FILE)) as f:
        return json.load(f)
