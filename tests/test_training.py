import numpy as np
import pytest
import torch
from support import (
    CONTEXT_LENGTH,
    LOOP_MAX,
    TINY_CONFIG,
    VOCAB_SIZE,
    one_sample_dataset,
    overfit_args,
    random_dataset,
    read_metrics,
)

from data import get_nonoverlapping_indices
from evaluate import validation_loss
from run import load_run_config, load_trained_model
from train import BUDGET_WARMUP_STEPS, parse_args, training
from trajectory import evaluation_schedule, long_trajectory


def loss_on_the_sample(run_dir, sample: np.ndarray, loops: int) -> float:
    model, _ = load_trained_model(run_dir, device="cpu")
    inputs = torch.tensor(sample[:-1].astype(np.int64))[None]
    targets = torch.tensor(sample[1:].astype(np.int64))
    with torch.no_grad():
        logits, _ = model(inputs, evaluation_schedule(LOOP_MAX, loops))
    return torch.nn.functional.cross_entropy(logits[0], targets).item()


def test_training_without_compilation_memorizes_the_sample_too(tmp_path):
    sample = one_sample_dataset(tmp_path / "train.npy")
    args = parse_args([*overfit_args(tmp_path, "eager"), "--no_compile"])

    training(args)

    assert loss_on_the_sample(tmp_path / "runs" / "eager", sample, loops=LOOP_MAX) < 0.05


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


def test_a_run_reports_its_final_losses_and_step_count_in_the_summary(overfit_run):
    summary = overfit_run["summary"]
    val_rows = [row for row in read_metrics(overfit_run["run_dir"]) if "val_loss" in row]
    one_loop_loss = loss_on_the_sample(overfit_run["run_dir"], overfit_run["sample"], loops=1)

    assert summary["val_loss"] == val_rows[-1]["val_loss"]
    assert summary["val_loss_1loop"] == pytest.approx(one_loop_loss, abs=1e-4)
    assert summary["steps"] == overfit_run["steps"]


def test_a_budgeted_run_counts_training_time_only_after_the_warm_up_steps(tmp_path):
    one_sample_dataset(tmp_path / "train.npy")
    budget_seconds = 1.2
    args = parse_args([*overfit_args(tmp_path, "budget"), "--iter", "100000", "--time_budget_minutes", str(budget_seconds / 60),
                       "--checkpoints_iter", "0"])

    summary = training(args)

    last_train_row = [row for row in read_metrics(tmp_path / "runs" / "budget") if "lr" in row][-1]
    assert summary["steps"] > BUDGET_WARMUP_STEPS
    assert budget_seconds <= summary["training_seconds"] < budget_seconds + 1
    assert last_train_row["lr"] == pytest.approx(args.optimizer_lr_min)


def test_the_seed_fixes_the_training_loss(tmp_path):
    random_dataset(tmp_path / "train.npy", windows=50)

    def loss_after_two_steps(run_name: str, seed: int) -> float:
        args = parse_args([*overfit_args(tmp_path, run_name), "--iter", "2", "--optimizer_warmup_iters", "1", "--checkpoints_iter", "0",
                           "--seed", str(seed)])
        training(args)
        return [row for row in read_metrics(tmp_path / "runs" / run_name) if "train_loss" in row][-1]["train_loss"]

    assert loss_after_two_steps("seed-1-again", 1) == loss_after_two_steps("seed-1", 1)
    assert loss_after_two_steps("seed-2", 2) != loss_after_two_steps("seed-1", 1)


def test_the_final_validation_can_be_capped_to_the_first_batches(tmp_path):
    one_sample_dataset(tmp_path / "train.npy")
    val = random_dataset(tmp_path / "val.npy", windows=5)
    steps = 20
    args = parse_args([*overfit_args(tmp_path, "capped"), "--val_data", str(tmp_path / "val.npy"), "--iter", str(steps),
                       "--checkpoints_iter", str(steps), "--final_val_batches", "1"])

    summary = training(args)

    model, _ = load_trained_model(tmp_path / "runs" / "capped", device="cpu")
    first_batch = get_nonoverlapping_indices(val, CONTEXT_LENGTH, args.batch_size)
    expected = validation_loss(model, val, first_batch, args.batch_size, CONTEXT_LENGTH, "cpu", long_trajectory(LOOP_MAX))
    assert summary["final_val_tokens"] == args.batch_size * CONTEXT_LENGTH
    assert summary["val_loss"] == pytest.approx(expected, abs=1e-4)


def test_checkpoints_iter_zero_saves_no_checkpoint(tmp_path):
    one_sample_dataset(tmp_path / "train.npy")
    args = parse_args([*overfit_args(tmp_path, "no-checkpoints"), "--iter", "20", "--checkpoints_iter", "0"])

    summary = training(args)

    assert not list((tmp_path / "runs" / "no-checkpoints" / "checkpoints").glob("*.pt"))
    assert summary["val_loss"] is not None


def test_a_diverging_run_stops_with_a_failure_exit_code(tmp_path, monkeypatch, capsys):
    one_sample_dataset(tmp_path / "train.npy")
    nan = torch.tensor(float("nan"))
    monkeypatch.setattr("train.training_step", lambda *args, **kwargs: {"train_loss": nan, "short_loss": nan, "consistency_loss": nan})
    args = parse_args([*overfit_args(tmp_path, "diverged"), "--iter", "20"])

    with pytest.raises(SystemExit) as exit_info:
        training(args)

    assert exit_info.value.code == 1
    assert "FAIL" in capsys.readouterr().out
