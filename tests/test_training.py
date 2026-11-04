import numpy as np
import pytest
import torch
from support import (
    CONTEXT_LENGTH,
    LOOP_MAX,
    OVERFIT_VARIANT,
    TINY_CONFIG,
    VARIANTS,
    VOCAB_SIZE,
    one_sample_dataset,
    overfit_args,
    read_metrics,
)

from run import load_run_config, load_trained_model
from train import parse_args, training
from trajectory import evaluation_schedule


def loss_on_the_sample(run_dir, sample: np.ndarray, loops: int) -> float:
    model, _ = load_trained_model(run_dir, device="cpu")
    inputs = torch.tensor(sample[:-1].astype(np.int64))[None]
    targets = torch.tensor(sample[1:].astype(np.int64))
    with torch.no_grad():
        logits, _ = model(inputs, evaluation_schedule(LOOP_MAX, loops))
    return torch.nn.functional.cross_entropy(logits[0], targets).item()


@pytest.mark.parametrize("variant", [v for v in VARIANTS if v != OVERFIT_VARIANT])  # the overfit_run fixture covers that one
def test_each_conditioning_variant_trains_on_one_sample_to_near_zero_loss(tmp_path, variant):
    sample = one_sample_dataset(tmp_path / "train.npy")
    args = parse_args(overfit_args(tmp_path, variant, variant))

    training(args)

    assert loss_on_the_sample(tmp_path / "runs" / variant, sample, loops=LOOP_MAX) < 0.05


def test_training_on_one_sample_drives_the_full_route_loss_to_zero(overfit_run):
    loss = loss_on_the_sample(overfit_run["run_dir"], overfit_run["sample"], loops=LOOP_MAX)

    assert loss < 0.05


def test_training_on_one_sample_makes_one_loop_predict_it_too(overfit_run):
    loss = loss_on_the_sample(overfit_run["run_dir"], overfit_run["sample"], loops=1)

    assert loss < 1.0


def test_a_run_records_its_model_shape_tokenizer_and_code_version(overfit_run):
    config = load_run_config(overfit_run["run_dir"])

    assert config.tokenizer_dir == overfit_run["args"].tokenizer_dir
    assert config.loop_blocks == TINY_CONFIG.loop_blocks
    assert config.loop_max == LOOP_MAX
    assert config.model_vocab_size == VOCAB_SIZE
    assert config.git_commit is not None


def test_a_run_logs_both_trajectories_losses_and_the_tokens_seen(overfit_run):
    train_rows = [row for row in read_metrics(overfit_run["run_dir"]) if "train_loss" in row]

    last = train_rows[-1]
    assert last["step"] == overfit_run["steps"]
    assert last["tokens"] == overfit_run["steps"] * 2 * CONTEXT_LENGTH
    assert {"short_loss", "consistency_loss", "lr", "grad_norm"} <= set(last)


def test_a_run_ends_with_a_validation_on_the_full_route(overfit_run):
    val_rows = [row for row in read_metrics(overfit_run["run_dir"]) if "val_loss" in row]

    assert val_rows[-1]["step"] == overfit_run["steps"]
    assert val_rows[-1]["val_loss"] < 0.05
