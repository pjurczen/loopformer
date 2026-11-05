"""Trains a LoopFormer on a tokenized dataset. Adapted from the cs336 assignment's training loop.

Each step runs the long trajectory (L loops of 1/L) and one sampled shorter trajectory, as in the paper:
loss = CE_long + shortcut_weight * CE_short + consistency_weight * MSE(stopgrad(h_long), h_short).
The gradient is computed in two backward passes, one per trajectory, so the long trajectory's activations are freed
before the short one runs. The gradient of a sum is the sum of the gradients, so this equals the paper's single backward
at about half the peak memory.
"""
import argparse
import json
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import numpy.typing as npt
import torch
import wandb

from checkpoint import load_checkpoint, save_checkpoint
from data import cross_entropy, get_batch, get_nonoverlapping_indices, get_random_indices, seed_batches
from evaluate import validation_loss
from model import COMPILE_SCOPES, CONDITIONINGS, LoopFormerLM
from optim import AdamWOptimizer, MuonOptimizer, cosine_lr_schedule, cosine_lr_schedule_by_time, gradient_clipping
from profiling import dump_and_summarize, start_recording
from run import CHECKPOINTS_DIR, CONFIG_FILE, METRICS_FILE, LossConfig, RunConfig
from tokenizer import read_tokenizer_config
from trajectory import MIN_LOOP_MAX, evaluation_schedule, long_trajectory, sample_short_trajectory
from version import git_commit

BUDGET_WARMUP_STEPS = 10  # the first steps of a run compile the graphs; a time budget counts the training time after them
DIVERGED_LOSS = 100.0  # a training loss above this (or not a number) ends the run with exit code 1


def training(args: argparse.Namespace) -> dict:
    """Trains the model the arguments describe and returns the run's summary: the final losses, steps, tokens, speed
    and memory, as printed at the end in one `key: value` line each."""
    iteration: int = 1
    run = RunConfig.model_validate(vars(args))
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)  # the short trajectories
        seed_batches(args.seed)

    run_dir = os.path.join(args.runs_dir, args.run_name)
    checkpoints_dir = os.path.join(run_dir, CHECKPOINTS_DIR)
    os.makedirs(checkpoints_dir, exist_ok=True)
    with open(os.path.join(run_dir, CONFIG_FILE), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Run directory: {run_dir}")

    # Live dashboard, off unless --wandb_project is given. Needs WANDB_API_KEY or a prior `wandb login`.
    wandb_run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args)) if args.wandb_project else None

    model = LoopFormerLM(run.architecture(), device=args.device, dtype=torch.float32)
    model.compile_for_device(args.device, scope=args.compile)
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
    # The final validation covers the whole validation set, or with --final_val_batches its first windows only.
    final_val_limit = args.final_val_batches * args.batch_size if args.final_val_batches else None
    final_val_indices: npt.NDArray = get_nonoverlapping_indices(val_dataset, args.model_context_length, final_val_limit) if args.val_data else None
    final_val_full = final_val_limit is None
    final_val_tokens = validation_tokens(val_dataset, final_val_indices, args.model_context_length, final_val_full) if args.val_data else None
    full_route = long_trajectory(args.loop_max)
    one_loop_route = evaluation_schedule(args.loop_max, 1)
    tokens_per_step = args.batch_size * args.model_context_length
    val_loss = val_loss_1loop = None

    with open(os.path.join(run_dir, METRICS_FILE), "a") as metric_file:

        t0 = time.perf_counter()
        # A time budget counts training time only: the step times after the first BUDGET_WARMUP_STEPS steps, which
        # compile the graphs. Training ends when it is used up (minus the reserve for the final validation and
        # checkpoint), the cosine follows that clock instead of the step count, and --iter is only an upper limit.
        # A resumed run continues the clock its checkpoint saved.
        training_seconds = elapsed_time or 0.0
        budget_seconds = (args.time_budget_minutes - args.time_reserve_minutes) * 60 if args.time_budget_minutes is not None else None

        for it in range(iteration, args.iter + 1):
            last_step = it == args.iter or (budget_seconds is not None and training_seconds >= budget_seconds)
            log_now = (it <= args.optimizer_warmup_iters and it % 10 == 0) or it % args.log_interval == 0 or last_step
            step_start = time.perf_counter()

            indices = get_random_indices(dataset=train_dataset, batch_size=args.batch_size, context_length=args.model_context_length)
            inputs, targets = get_batch(dataset=train_dataset, indices=indices, context_length=args.model_context_length, device=args.device,
                                        dtype=torch.long)
            snapshot_now = args.memory_snapshot_step == it and args.device.startswith("cuda")
            if snapshot_now:
                start_recording()
            losses = training_step(model, inputs, targets, loss_config, device=args.device, bf16=args.bf16)
            if snapshot_now:
                dump_and_summarize(run_dir)

            train_loss = losses["train_loss"].item()
            if not math.isfinite(train_loss) or train_loss > DIVERGED_LOSS:
                print(f"FAIL: training loss {train_loss} at step {it}", flush=True)
                if wandb_run is not None:
                    wandb_run.finish()
                raise SystemExit(1)

            l2_norm: torch.Tensor = gradient_clipping(parameters=model.parameters(), max_l2_norm=args.gradient_max_l2_norm, eps=args.gradient_eps)

            if log_now:
                print(f"{it}/{args.iter} Training loss: {train_loss}")
            if it % (10 * args.log_interval) == 0 or last_step:
                print(f"{it}/{args.iter} L2-norm: {l2_norm.item()}")

            if budget_seconds is not None:
                lr: float = cosine_lr_schedule_by_time(
                    t=it,
                    elapsed_seconds=training_seconds,
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

            synchronize(args.device)  # the step's kernels have finished: its time is the GPU's, not the CPU's
            if it - iteration >= BUDGET_WARMUP_STEPS:
                training_seconds += time.perf_counter() - step_start

            if log_now:
                t1 = time.perf_counter()
                print(f"Elapsed time: {(t1 - t0):.0f} s")
                record_metrics(metric_file, wandb_run, {
                    "step": it, "time": round(t1 - t0, 2), "training_time": round(training_seconds, 2), "tokens": it * tokens_per_step,
                    **{name: value.item() for name, value in losses.items()},
                    "lr": lr, "grad_norm": l2_norm.item(), "peak_memory_gb": peak_memory_gb(args.device),
                })

            if val_dataset is not None and (it % args.val_interval == 0 or last_step):
                val_indices = final_val_indices if last_step else iter_val_indices
                full_dataset_eval = last_step and final_val_full
                val_loss = validation_loss(model, val_dataset, val_indices, args.batch_size, args.model_context_length, args.device,
                                           full_route, bf16=args.bf16, full_dataset_eval=full_dataset_eval)
                row = {"step": it, "time": round(time.perf_counter() - t0, 2), "val_loss": val_loss}
                if last_step:  # the elastic-depth number: the same windows along the one-loop route
                    val_loss_1loop = validation_loss(model, val_dataset, val_indices, args.batch_size, args.model_context_length, args.device,
                                                     one_loop_route, bf16=args.bf16, full_dataset_eval=full_dataset_eval)
                    row["val_loss_1loop"] = val_loss_1loop
                record_metrics(metric_file, wandb_run, row)
                print(f"{it}/{args.iter} Validation loss: {val_loss}")

            if args.checkpoints_iter and (it % args.checkpoints_iter == 0 or last_step):
                save_checkpoint(model, adamw, it, os.path.join(checkpoints_dir, f"ckpt_{it}.pt"), training_seconds, muon)
                print(f"{it}/{args.iter} Saving checkpoint...")

            if last_step:
                break

    print("torch.compile times by phase (seconds, totals over the run):")
    print(torch._dynamo.utils.compile_times(repr="str"))
    if wandb_run is not None:
        wandb_run.finish()

    budgeted_steps = max(it - iteration + 1 - BUDGET_WARMUP_STEPS, 0)
    summary = {
        "val_loss": val_loss,
        "val_loss_1loop": val_loss_1loop,
        "final_val_tokens": final_val_tokens,
        "training_seconds": round(training_seconds, 1),
        "steps": it,
        "tokens_M": round(it * tokens_per_step / 1e6, 2),
        "tok_per_s": round(budgeted_steps * tokens_per_step / training_seconds) if training_seconds > 0 else None,
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "peak_memory_gb": peak_memory_gb(args.device),
    }
    print("---")  # grep-able: one `key: value` line each
    for key, value in summary.items():
        print(f"{key + ':':20}{value}")
    return summary


def validation_tokens(dataset: npt.NDArray, indices: npt.NDArray, context_length: int, full_dataset_eval: bool) -> int:
    """The tokens validation_loss scores for these windows; with full_dataset_eval the dataset's tail counts too."""
    tail = (len(dataset) - 1) % context_length if full_dataset_eval else 0
    return len(indices) * context_length + tail


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


_mps_peak = {"bytes": 0}  # MPS keeps no peak counter, so the largest allocation seen at the calls below stands in for it


def peak_memory_gb(device: str) -> float | None:
    """The most GPU memory held so far in this run, in GB: CUDA's peak counter, or on MPS the largest driver allocation
    seen when this was called (every log step and the end of the run)."""
    if device.startswith("cuda"):
        return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if device == "mps":
        _mps_peak["bytes"] = max(_mps_peak["bytes"], torch.mps.driver_allocated_memory())
        return round(_mps_peak["bytes"] / 1e9, 2)
    return None


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
    arg_parser.add_argument("--conditioning", choices=CONDITIONINGS, default="adaln",
                            help="how a block learns its loop position: adaln (scales + gates), gate, or an additive embedding")
    arg_parser.add_argument("--loop_lora_rank", type=int, default=0, help="rank of per-loop low-rank deltas on the shared matrices; 0 turns them off")
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
    arg_parser.add_argument("--checkpoints_iter", type=int, default=100, help="save a checkpoint every this many steps and at the end; 0 saves none")
    arg_parser.add_argument("--seed", type=int, default=None, help="seeds the weights, the batches and the short trajectories; default: random")
    arg_parser.add_argument("--final_val_batches", type=int, default=None,
                            help="the final validation scores this many batches (the validation set's first windows); default: the whole set")
    arg_parser.add_argument("--checkpoint_src", type=str, default=None)
    arg_parser.add_argument("--wandb_project", type=str, default=None)
    arg_parser.add_argument("--bf16", action="store_true")
    arg_parser.add_argument("--compile", choices=COMPILE_SCOPES, default="whole",
                            help="torch.compile the whole forward pass unrolled (one graph per loop count; the plain model's memory), "
                                 "each block on its own (9 GB more at batch 64), the whole pass with a while_loop (a third of the memory, "
                                 "15%% slower), or none: eager, for debugging and as the reference for a compiled run")
    arg_parser.add_argument("--memory_snapshot_step", type=int, default=None,
                            help="record every allocation of this training step and write memory_snapshot.pickle into the run folder (CUDA only)")
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
