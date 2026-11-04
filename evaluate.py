"""The loss of a model on a dataset along a trajectory, and a finished run's loss at every loop count from 1 to L.

    uv run evaluate.py --run_dir runs/<run_name> --device cuda

For a budget of M loops it uses the one schedule evaluation_schedule gives (the grid units split as evenly as possible,
larger steps first). The result goes to <run_dir>/budget_eval.json: one row per M with the schedule, the loss and the
time per batch.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import numpy.typing as npt
import torch

from data import cross_entropy, get_batch, get_nonoverlapping_indices
from model import LoopFormerLM
from run import BUDGET_EVAL_FILE, RunConfig, load_trained_model
from trajectory import evaluation_schedule


def validation_loss(model: LoopFormerLM, dataset: npt.NDArray, indices: npt.NDArray, batch_size: int, context_length: int,
                    device: str, schedule: list[float], bf16: bool = False, full_dataset_eval: bool = False) -> float:
    """The mean cross-entropy per token over the windows at `indices`, run along `schedule`. With full_dataset_eval the
    tail of the dataset that does not fill a window counts too."""
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16):
        val_loss_sum: float = 0
        token_count: int = 0
        for idx in range(0, len(indices), batch_size):
            batch = indices[idx: (idx + batch_size)]
            val_inputs, val_targets = get_batch(dataset=dataset, indices=batch, context_length=context_length, device=device, dtype=torch.long)
            val_prediction, _ = model(val_inputs, schedule)
            val_loss_sum += cross_entropy(val_prediction.float(), val_targets).item() * (len(batch) * context_length)
            token_count += len(batch) * context_length

        if full_dataset_eval:
            tail_length: int = (len(dataset) - 1) % context_length
            if tail_length > 0:
                start = len(dataset) - 1 - tail_length
                val_inputs, val_targets = get_batch(dataset=dataset, indices=np.array([start]), context_length=tail_length, device=device,
                                                    dtype=torch.long)
                val_prediction, _ = model(val_inputs, schedule)
                val_loss_sum += cross_entropy(val_prediction.float(), val_targets).item() * tail_length
                token_count += tail_length

        return val_loss_sum / token_count


def evaluate_budgets(run_dir: str | os.PathLike, device: str, checkpoint_path: str | None = None, bf16: bool = False) -> list[dict]:
    model, run = load_trained_model(run_dir, device, checkpoint_path)
    model.compile_for_device(device)
    dataset = np.load(run.val_data, mmap_mode="r")
    indices = get_nonoverlapping_indices(dataset, run.model_context_length)

    rows = [measure_budget(model, run, dataset, indices, loops, device, bf16) for loops in range(1, run.loop_max + 1)]

    write_budget_eval(run_dir, rows)
    return rows


def measure_budget(model: LoopFormerLM, run: RunConfig, dataset: npt.NDArray, indices: npt.NDArray, loops: int, device: str, bf16: bool) -> dict:
    schedule = evaluation_schedule(run.loop_max, loops)
    t0 = time.perf_counter()
    loss = validation_loss(model, dataset, indices, run.batch_size, run.model_context_length, device, schedule, bf16=bf16, full_dataset_eval=True)
    seconds = time.perf_counter() - t0
    print(f"{loops}/{run.loop_max} loops, schedule {schedule}: validation loss {loss:.4f} ({seconds:.0f} s)")
    return {"loops": loops, "schedule": schedule, "loss": loss, "seconds_per_batch": seconds / math.ceil(len(indices) / run.batch_size)}


def write_budget_eval(run_dir: str | os.PathLike, rows: list[dict]) -> None:
    with open(os.path.join(run_dir, BUDGET_EVAL_FILE), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument("--run_dir", type=str, required=True)
    arg_parser.add_argument("--checkpoint", type=str, default=None, help="a checkpoint file; default: the run's newest")
    arg_parser.add_argument("--device", type=str, default="mps")
    arg_parser.add_argument("--bf16", action="store_true")
    args = arg_parser.parse_args()

    evaluate_budgets(args.run_dir, args.device, args.checkpoint, args.bf16)
