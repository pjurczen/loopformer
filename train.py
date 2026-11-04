"""Trains a LoopFormer on a tokenized dataset. Adapted from the cs336 assignment's training loop.

Each step runs the long trajectory (L loops of 1/L) and one sampled shorter trajectory, as in the paper:
loss = CE_long + shortcut_weight * CE_short + consistency_weight * MSE(stopgrad(h_long), h_short).
The gradient is computed in two backward passes, one per trajectory, so the long trajectory's activations are freed
before the short one runs. The gradient of a sum is the sum of the gradients, so this equals the paper's single backward
at about half the peak memory.
"""
import argparse
import json
import os
import random
import subprocess
import time
from datetime import datetime

import numpy as np
import numpy.typing as npt
import torch
import wandb

from checkpoint import load_checkpoint, save_checkpoint
from data import cross_entropy, get_batch, get_nonoverlapping_indices, get_random_indices
from evaluate import validation_loss
from model import LoopFormerLM
from optim import AdamWOptimizer, MuonOptimizer, cosine_lr_schedule, cosine_lr_schedule_by_time, gradient_clipping
from run import CHECKPOINTS_DIR, CONFIG_FILE, METRICS_FILE, LossConfig, RunConfig
from tokenizer import read_tokenizer_config
from trajectory import MIN_LOOP_MAX, long_trajectory, sample_short_trajectory


def training(args: argparse.Namespace):
    iteration: int = 1
    run = RunConfig.model_validate(vars(args))

    run_dir = os.path.join(args.runs_dir, args.run_name)
    checkpoints_dir = os.path.join(run_dir, CHECKPOINTS_DIR)
    os.makedirs(checkpoints_dir, exist_ok=True)
    with open(os.path.join(run_dir, CONFIG_FILE), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Run directory: {run_dir}")

    # Live dashboard, off unless --wandb_project is given. Needs WANDB_API_KEY or a prior `wandb login`.
    wandb_run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args)) if args.wandb_project else None

    model = LoopFormerLM(run.architecture(), device=args.device, dtype=torch.float32)
    model.compile_for_device(args.device)
    loss_config = run.loss()

    # With --optimizer_muon, the 2-D matrices inside the blocks go to Muon; everything else stays on AdamW.
    muon_params = model.muon_parameters() if args.optimizer_muon else []
    muon_ids = {id(p) for p in muon_params}
    adamw: torch.optim.Optimizer = AdamWOptimizer(
        params=[p for p in model.parameters() if id(p) not in muon_ids],
        lr=args.optimizer_lr_min,
        weight_decay=args.optimizer_weight_decay,
        betas=(args.optimizer_beta1, args.optimizer_beta2),
        eps=args.optimizer_eps,
    )
    muon: torch.optim.Optimizer | None = MuonOptimizer(
        params=muon_params,
        lr=args.optimizer_lr_min,
        weight_decay=args.optimizer_weight_decay,
    ) if args.optimizer_muon else None
    optimizers: list[torch.optim.Optimizer] = [adamw] if muon is None else [adamw, muon]

    elapsed_time = None
    if args.checkpoint_src:
        print("Loading checkpoint...")
        iteration, elapsed_time = load_checkpoint(args.checkpoint_src, model, adamw, muon)
        print(f"Resuming from iteration {iteration}/{args.iter}")
        iteration += 1

    train_dataset = np.load(file=args.train_data, mmap_mode='r')
    val_dataset = np.load(file=args.val_data, mmap_mode='r') if args.val_data else None

    iter_val_indices: npt.NDArray = get_nonoverlapping_indices(val_dataset, args.model_context_length,
                                                               args.val_batches * args.batch_size) if args.val_data else None
    final_val_indices: npt.NDArray = get_nonoverlapping_indices(val_dataset, args.model_context_length) if args.val_data else None
    full_route = long_trajectory(args.loop_max)

    with open(os.path.join(run_dir, METRICS_FILE), "a") as metric_file:

        t0 = time.perf_counter()
        if elapsed_time is not None:
            t0 -= elapsed_time

        # With a time budget, training ends when it's used up (minus the reserve for the final validation and checkpoint),
        # and the cosine follows the elapsed time instead of the step count. --iter is then only an upper limit.
        budget_seconds = (args.time_budget_minutes - args.time_reserve_minutes) * 60 if args.time_budget_minutes is not None else None

        for it in range(iteration, args.iter + 1):
            elapsed = time.perf_counter() - t0  # no synchronize: the CPU runs only a few steps ahead of the GPU
            last_step = it == args.iter or (budget_seconds is not None and elapsed >= budget_seconds)
            log_now = (it <= args.optimizer_warmup_iters and it % 10 == 0) or it % args.log_interval == 0 or last_step

            indices = get_random_indices(dataset=train_dataset, batch_size=args.batch_size, context_length=args.model_context_length)
            inputs, targets = get_batch(dataset=train_dataset, indices=indices, context_length=args.model_context_length, device=args.device,
                                        dtype=torch.long)
            losses = training_step(model, inputs, targets, loss_config, device=args.device, bf16=args.bf16)

            l2_norm: torch.Tensor = gradient_clipping(parameters=model.parameters(), max_l2_norm=args.gradient_max_l2_norm, eps=args.gradient_eps)

            if log_now:
                print(f"{it}/{args.iter} Training loss: {losses['train_loss'].item()}")
            if it % (10 * args.log_interval) == 0 or last_step:
                print(f"{it}/{args.iter} L2-norm: {l2_norm.item()}")

            if budget_seconds is not None:
                lr: float = cosine_lr_schedule_by_time(
                    t=it,
                    elapsed_seconds=elapsed,
                    budget_seconds=budget_seconds,
                    lr_max=args.optimizer_lr_max,
                    lr_min=args.optimizer_lr_min,
                    warmup_iters=args.optimizer_warmup_iters,
                )
            else:
                lr: float = cosine_lr_schedule(
                    t=it,
                    lr_max=args.optimizer_lr_max,
                    lr_min=args.optimizer_lr_min,
                    warmup_iters=args.optimizer_warmup_iters,
                    cosine_cycle_iters=args.optimizer_cosine_cycle_iters,
                )
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                optimizer.zero_grad()

            if log_now:
                synchronize(args.device)
                t1 = time.perf_counter()
                print(f"Elapsed time: {(t1 - t0):.0f} s")
                record_metrics(metric_file, wandb_run, {
                    "step": it, "time": round(t1 - t0, 2), "tokens": it * args.batch_size * args.model_context_length,
                    **{name: value.item() for name, value in losses.items()},
                    "lr": lr, "grad_norm": l2_norm.item(), "peak_memory_gb": peak_memory_gb(args.device),
                })

            if val_dataset is not None and (it % args.val_interval == 0 or last_step):
                val_indices = final_val_indices if last_step else iter_val_indices
                val_loss: float = validation_loss(model, val_dataset, val_indices, args.batch_size, args.model_context_length, args.device,
                                                  full_route, bf16=args.bf16, full_dataset_eval=last_step)
                t2 = time.perf_counter()
                record_metrics(metric_file, wandb_run, {"step": it, "time": round(t2 - t0, 2), "val_loss": val_loss})
                print(f"{it}/{args.iter} Validation loss: {val_loss}")

            if it % args.checkpoints_iter == 0 or last_step:
                t3 = time.perf_counter()
                save_checkpoint(model, adamw, it, os.path.join(checkpoints_dir, f"ckpt_{it}.pt"), t3 - t0, muon)
                print(f"{it}/{args.iter} Saving checkpoint...")

            if last_step:
                break

    if wandb_run is not None:
        wandb_run.finish()


def training_step(model: LoopFormerLM, inputs: torch.Tensor, targets: torch.Tensor, loss: LossConfig, device: str,
                  bf16: bool) -> dict[str, torch.Tensor]:
    """Accumulates the gradient of the paper's loss into .grad, in two backward passes. Returns the detached loss terms."""
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16):
        logits, hidden = model(inputs, long_trajectory(loss.loop_max))
        # No float32 copy of the logits: it costs batch × context × vocab × 4 bytes. Autocast runs exp, log and sum in float32.
        train_loss = cross_entropy(logits, targets)
    train_loss.backward()
    hidden_long = hidden.detach()  # the consistency target; stopgrad, so this term pulls only the short trajectory

    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16):
        short_logits, short_hidden = model(inputs, sample_short_trajectory(loss.loop_max, random))
        short_loss = cross_entropy(short_logits, targets)
        consistency_loss = torch.nn.functional.mse_loss(short_hidden.float(), hidden_long.float())
    (loss.shortcut_weight * short_loss + loss.consistency_weight * consistency_loss).backward()

    return {"train_loss": train_loss.detach(), "short_loss": short_loss.detach(), "consistency_loss": consistency_loss.detach()}


def record_metrics(metric_file, wandb_run, metrics: dict):
    # The same metrics go to metrics.jsonl and to wandb; wandb takes the step separately and skips missing values.
    metric_file.write(json.dumps(metrics) + "\n")
    metric_file.flush()
    if wandb_run is not None:
        wandb_run.log({key: value for key, value in metrics.items() if key != "step" and value is not None}, step=metrics["step"])


def synchronize(device: str):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def peak_memory_gb(device: str) -> float | None:
    # Most GPU memory held by tensors so far in this run (CUDA only).
    if device.startswith("cuda"):
        return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return None


def git_commit() -> str | None:
    # Inside a training image there is no Git; the cloud runner passes the version in as $GIT_COMMIT.
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], check=False).returncode != 0
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if dirty else commit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument("--optimizer_lr_min", type=float, default=1e-4)
    arg_parser.add_argument("--optimizer_lr_max", type=float, default=1e-3)
    arg_parser.add_argument("--optimizer_warmup_iters", type=int, default=None)
    arg_parser.add_argument("--optimizer_cosine_cycle_iters", type=int, default=None)
    arg_parser.add_argument("--optimizer_weight_decay", type=float, default=0.01)
    arg_parser.add_argument("--optimizer_beta1", type=float, default=0.9)
    arg_parser.add_argument("--optimizer_beta2", type=float, default=0.999)
    arg_parser.add_argument("--optimizer_eps", type=float, default=1e-8)
    arg_parser.add_argument("--optimizer_muon", action="store_true", help="Muon for the 2D matrices inside the blocks, AdamW for the rest")
    arg_parser.add_argument("--loop_blocks", type=int, required=True, help="k: the number of unique transformer blocks")
    arg_parser.add_argument("--loop_max", type=int, required=True, help="L: the number of loops of the long trajectory")
    arg_parser.add_argument("--shortcut_weight", type=float, default=0.1, help="weight of the short trajectory's cross-entropy")
    arg_parser.add_argument("--consistency_weight", type=float, default=0.1, help="weight of the MSE between the short and long hidden states")
    arg_parser.add_argument("--model_context_length", type=int, default=512)
    arg_parser.add_argument("--model_d_model", type=int, default=768)
    arg_parser.add_argument("--model_d_ff", type=int, default=2048)
    arg_parser.add_argument("--model_num_heads", type=int, default=12)
    arg_parser.add_argument("--model_eps", type=float, default=1e-5)
    arg_parser.add_argument("--model_rope_theta", type=float, default=10000)
    arg_parser.add_argument("--gradient_max_l2_norm", type=float, default=1.0)
    arg_parser.add_argument("--gradient_eps", type=float, default=1e-6)
    arg_parser.add_argument("--iter", type=int, default=100)
    arg_parser.add_argument("--device", type=str, default="mps")
    arg_parser.add_argument("--batch_size", type=int, default=32)
    arg_parser.add_argument("--tokenizer_dir", type=str, required=True, help="its tokenizer_config.json gives the vocabulary size")
    arg_parser.add_argument("--train_data", type=str, required=True)
    arg_parser.add_argument("--val_data", type=str)
    arg_parser.add_argument("--val_batches", type=int, default=32)
    arg_parser.add_argument("--val_interval", type=int, default=200)
    arg_parser.add_argument("--log_interval", type=int, default=100)
    arg_parser.add_argument("--runs_dir", type=str, default="runs")
    arg_parser.add_argument("--run_name", type=str, default=None)
    arg_parser.add_argument("--checkpoints_iter", type=int, default=100)
    arg_parser.add_argument("--checkpoint_src", type=str, default=None)
    arg_parser.add_argument("--wandb_project", type=str, default=None)
    arg_parser.add_argument("--bf16", action="store_true")
    arg_parser.add_argument("--time_budget_minutes", type=float, default=None,
                            help="train for this long instead of --iter steps (--iter becomes an upper limit); needs --optimizer_warmup_iters")
    arg_parser.add_argument("--time_reserve_minutes", type=float, default=0.0,
                            help="part of the time budget kept free after training, for the final validation pass and checkpoint")
    args = arg_parser.parse_args(argv)
    args.git_commit = git_commit()  # recorded in config.json and the wandb config

    if args.loop_max < MIN_LOOP_MAX:
        arg_parser.error(f"--loop_max must be at least {MIN_LOOP_MAX}")
    if args.time_budget_minutes is not None and args.optimizer_warmup_iters is None:
        arg_parser.error("--time_budget_minutes needs an explicit --optimizer_warmup_iters (the step count isn't known in advance)")
    if args.optimizer_warmup_iters is None:
        args.optimizer_warmup_iters = args.iter // 50
    if args.optimizer_cosine_cycle_iters is None:
        args.optimizer_cosine_cycle_iters = args.iter
    if args.run_name is None:
        args.run_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    args.model_vocab_size = read_tokenizer_config(args.tokenizer_dir)["vocab_size"]

    assert args.optimizer_lr_min <= args.optimizer_lr_max
    assert args.optimizer_warmup_iters < args.optimizer_cosine_cycle_iters <= args.iter
    return args


if __name__ == "__main__":
    training(parse_args())
