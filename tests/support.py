"""Creation and action helpers shared by the tests: a tiny model that trains on the CPU in seconds."""
import json
import os
from dataclasses import replace

import numpy as np
import torch

from model import LoopFormerLM, ModelConfig
from run import METRICS_FILE
from trajectory import long_trajectory

VOCAB_SIZE = 100
CONTEXT_LENGTH = 16
LOOP_MAX = 2
TINY_CONFIG = ModelConfig(vocab_size=VOCAB_SIZE, context_length=CONTEXT_LENGTH, loop_blocks=2, loop_max=LOOP_MAX, d_model=64, num_heads=4, d_ff=128)
VARIANTS = {  # name -> (conditioning, loop_lora_rank)
    "adaln": ("adaln", 0),
    "gate": ("gate", 0),
    "additive": ("additive", 0),
    "additive-lora": ("additive", 4),
}
OVERFIT_VARIANT = "adaln"  # the variant of the session-wide trained run in conftest.py
OVERFIT_STEPS = 200


def overfit_args(root, run_name: str, variant: str = OVERFIT_VARIANT) -> list[str]:
    """train.py arguments for 200 steps on the one-sample dataset at root/train.npy."""
    conditioning, loop_lora_rank = VARIANTS[variant]
    return [
        "--train_data", str(root / "train.npy"), "--val_data", str(root / "train.npy"),
        "--tokenizer_dir", tokenizer_dir_for(VOCAB_SIZE, root / "tokenizer"),
        "--model_context_length", str(CONTEXT_LENGTH), "--loop_blocks", str(TINY_CONFIG.loop_blocks), "--loop_max", str(LOOP_MAX),
        "--model_d_model", str(TINY_CONFIG.d_model), "--model_num_heads", str(TINY_CONFIG.num_heads), "--model_d_ff", str(TINY_CONFIG.d_ff),
        "--conditioning", conditioning, "--loop_lora_rank", str(loop_lora_rank),
        "--optimizer_muon", "--optimizer_lr_max", "3e-3", "--optimizer_lr_min", "3e-4", "--optimizer_warmup_iters", "10",
        "--batch_size", "2", "--iter", str(OVERFIT_STEPS), "--val_batches", "1", "--val_interval", "100", "--log_interval", "10",
        "--checkpoints_iter", str(OVERFIT_STEPS), "--runs_dir", str(root / "runs"), "--run_name", run_name, "--device", "cpu",
    ]


def tiny_model(seed: int = 0, variant: str = "adaln") -> LoopFormerLM:
    conditioning, loop_lora_rank = VARIANTS[variant]
    torch.manual_seed(seed)
    return LoopFormerLM(replace(TINY_CONFIG, conditioning=conditioning, loop_lora_rank=loop_lora_rank))


def random_tokens(seed: int = 0, batch_size: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB_SIZE, (batch_size, CONTEXT_LENGTH), generator=generator)


def one_sample_dataset(path, seed: int = 0) -> np.ndarray:
    """A dataset of exactly one training sample, so every batch is that same sample."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, VOCAB_SIZE, size=CONTEXT_LENGTH + 1, dtype=np.uint16)
    np.save(path, ids)
    return ids


def random_dataset(path, windows: int, seed: int = 0) -> np.ndarray:
    """A dataset of `windows` non-overlapping context windows of random tokens."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, VOCAB_SIZE, size=windows * CONTEXT_LENGTH + 1, dtype=np.uint16)
    np.save(path, ids)
    return ids


def tokenizer_dir_for(vocab_size: int, path) -> str:
    os.makedirs(path, exist_ok=True)  # several runs may share one root
    with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
        json.dump({"vocab_size": vocab_size, "special_tokens": ["<|endoftext|>"]}, f)
    return str(path)


def read_metrics(run_dir) -> list[dict]:
    with open(os.path.join(run_dir, METRICS_FILE)) as f:
        return [json.loads(line) for line in f]


def next_token_loss(model: LoopFormerLM, tokens: torch.Tensor, schedule: list[float]) -> torch.Tensor:
    logits, _ = model(tokens[:, :-1], schedule)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())


def after_training_steps(model: LoopFormerLM, tokens: torch.Tensor, steps: int = 3) -> LoopFormerLM:
    """A model whose gates have left zero: a few plain SGD steps on the long trajectory."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad()
        next_token_loss(model, tokens, long_trajectory(LOOP_MAX)).backward()
        optimizer.step()
    optimizer.zero_grad()
    return model
