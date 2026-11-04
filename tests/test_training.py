import json
import os

import numpy as np
import pytest
import torch
from support import CONTEXT_LENGTH, LOOP_MAX, TINY_CONFIG, VOCAB_SIZE

from checkpoint import latest_checkpoint_path, load_model
from model import LoopFormerLM
from train import parse_args, training
from trajectory import evaluation_schedule

STEPS = 200
RUN_NAME = "overfit"


def one_sample_dataset(path, seed: int = 0) -> np.ndarray:
    """A dataset of exactly one training sample, so every batch is that same sample."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, VOCAB_SIZE, size=CONTEXT_LENGTH + 1, dtype=np.uint16)
    np.save(path, ids)
    return ids


def tokenizer_dir_for(vocab_size: int, path) -> str:
    os.makedirs(path)
    with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
        json.dump({"vocab_size": vocab_size, "special_tokens": ["<|endoftext|>"]}, f)
    return str(path)


def read_metrics(run_dir) -> list[dict]:
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
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
        "--batch_size", "2", "--iter", str(STEPS), "--val_batches", "1", "--val_interval", "100", "--log_interval", "10",
        "--checkpoints_iter", str(STEPS), "--runs_dir", str(runs_dir), "--run_name", RUN_NAME, "--device", "cpu",
    ])

    training(args)

    return {"sample": sample, "run_dir": runs_dir / RUN_NAME, "args": args}


def loss_on_the_sample(run_dir, sample: np.ndarray, loops: int) -> float:
    model = LoopFormerLM(TINY_CONFIG)
    load_model(latest_checkpoint_path(run_dir / "checkpoints"), model)
    inputs = torch.tensor(sample[:-1].astype(np.int64))[None]
    targets = torch.tensor(sample[1:].astype(np.int64))
    with torch.no_grad():
        logits, _ = model(inputs, evaluation_schedule(LOOP_MAX, loops))
    return torch.nn.functional.cross_entropy(logits[0], targets).item()


def test_training_on_one_sample_drives_the_full_route_loss_to_zero(overfit_run):
    loss = loss_on_the_sample(overfit_run["run_dir"], overfit_run["sample"], loops=LOOP_MAX)

    assert loss < 0.05


def test_training_on_one_sample_makes_one_loop_predict_it_too(overfit_run):
    loss = loss_on_the_sample(overfit_run["run_dir"], overfit_run["sample"], loops=1)

    assert loss < 1.0


def test_a_run_records_its_model_shape_tokenizer_and_code_version(overfit_run):
    with open(overfit_run["run_dir"] / "config.json") as f:
        config = json.load(f)

    assert config["tokenizer_dir"] == overfit_run["args"].tokenizer_dir
    assert config["loop_blocks"] == TINY_CONFIG.loop_blocks
    assert config["loop_max"] == LOOP_MAX
    assert config["model_vocab_size"] == VOCAB_SIZE
    assert "git_commit" in config


def test_a_run_logs_both_trajectories_losses_and_the_tokens_seen(overfit_run):
    train_rows = [row for row in read_metrics(overfit_run["run_dir"]) if "train_loss" in row]

    last = train_rows[-1]
    assert last["step"] == STEPS
    assert last["tokens"] == STEPS * 2 * CONTEXT_LENGTH
    assert {"short_loss", "consistency_loss", "lr", "grad_norm"} <= set(last)


def test_a_run_ends_with_a_validation_on_the_full_route(overfit_run):
    val_rows = [row for row in read_metrics(overfit_run["run_dir"]) if "val_loss" in row]

    assert val_rows[-1]["step"] == STEPS
    assert val_rows[-1]["val_loss"] < 0.05
