"""One trained tiny model for the tests that need a trained run: 200 steps on a single sample, so every batch is that
sample and the model memorizes it."""
import json
import os

import numpy as np
import pytest
from support import CONTEXT_LENGTH, LOOP_MAX, TINY_CONFIG, VOCAB_SIZE

from train import parse_args, training

OVERFIT_STEPS = 200
OVERFIT_RUN_NAME = "overfit"


def one_sample_dataset(path, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, VOCAB_SIZE, size=CONTEXT_LENGTH + 1, dtype=np.uint16)
    np.save(path, ids)
    return ids


def tokenizer_dir_for(vocab_size: int, path) -> str:
    os.makedirs(path)
    with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
        json.dump({"vocab_size": vocab_size, "special_tokens": ["<|endoftext|>"]}, f)
    return str(path)


@pytest.fixture(scope="session")
def overfit_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("overfit")
    sample = one_sample_dataset(root / "train.npy")
    runs_dir = root / "runs"
    args = parse_args([
        "--train_data", str(root / "train.npy"), "--val_data", str(root / "train.npy"),
        "--tokenizer_dir", tokenizer_dir_for(VOCAB_SIZE, root / "tokenizer"),
        "--model_context_length", str(CONTEXT_LENGTH), "--loop_blocks", str(TINY_CONFIG.loop_blocks), "--loop_max", str(LOOP_MAX),
        "--model_d_model", str(TINY_CONFIG.d_model), "--model_num_heads", str(TINY_CONFIG.num_heads), "--model_d_ff", str(TINY_CONFIG.d_ff),
        "--optimizer_muon", "--optimizer_lr_max", "3e-3", "--optimizer_lr_min", "3e-4", "--optimizer_warmup_iters", "10",
        "--batch_size", "2", "--iter", str(OVERFIT_STEPS), "--val_batches", "1", "--val_interval", "100", "--log_interval", "10",
        "--checkpoints_iter", str(OVERFIT_STEPS), "--runs_dir", str(runs_dir), "--run_name", OVERFIT_RUN_NAME, "--device", "cpu",
    ])

    training(args)

    return {"sample": sample, "run_dir": runs_dir / OVERFIT_RUN_NAME, "args": args, "steps": OVERFIT_STEPS}
