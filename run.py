"""A run folder: the settings train.py writes to config.json, their translations, and the trained model it holds.

    runs/<run_name>/config.json        the settings (RunConfig)
    runs/<run_name>/metrics.jsonl      one JSON row per log step and per validation
    runs/<run_name>/checkpoints/       ckpt_<step>.pt
    runs/<run_name>/budget_eval.json   the per-loop evaluation, once evaluate.py ran
"""
import json
import os
from dataclasses import dataclass

import torch
from pydantic import BaseModel

from checkpoint import load_model
from model import LoopFormerLM, ModelConfig

CONFIG_FILE = "config.json"
METRICS_FILE = "metrics.jsonl"
CHECKPOINTS_DIR = "checkpoints"
BUDGET_EVAL_FILE = "budget_eval.json"


@dataclass(frozen=True)
class LossConfig:
    loop_max: int  # L: the loops of the long trajectory
    shortcut_weight: float  # weight of the short trajectory's cross-entropy
    consistency_weight: float  # weight of the MSE between the short and the long hidden states


class RunConfig(BaseModel):
    optimizer_lr_min: float
    optimizer_lr_max: float
    optimizer_warmup_iters: int
    optimizer_cosine_cycle_iters: int
    optimizer_weight_decay: float
    optimizer_beta1: float
    optimizer_beta2: float
    optimizer_eps: float
    optimizer_muon: bool
    loop_blocks: int
    loop_max: int
    conditioning: str = "adaln"  # older runs' config.json predates this field; they used adaln
    loop_lora_rank: int = 0  # older runs' config.json predates this field; they had no LoRA
    shortcut_weight: float
    consistency_weight: float
    model_vocab_size: int
    model_context_length: int
    model_d_model: int
    model_d_ff: int
    model_num_heads: int
    model_eps: float
    model_rope_theta: float
    gradient_max_l2_norm: float
    gradient_eps: float
    iter: int
    device: str
    batch_size: int
    tokenizer_dir: str
    train_data: str
    val_data: str | None
    val_batches: int
    val_interval: int
    log_interval: int
    runs_dir: str
    run_name: str
    checkpoints_iter: int
    checkpoint_src: str | None
    wandb_project: str | None
    bf16: bool
    compile: str = "blocks"  # older runs' config.json predates this field; they compiled per block
    memory_snapshot_step: int | None = None
    git_commit: str | None  # code version the run was trained with; "-dirty" means uncommitted changes
    time_budget_minutes: float | None  # with a budget, training ended on time rather than at iter
    time_reserve_minutes: float

    def architecture(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.model_vocab_size,
            context_length=self.model_context_length,
            loop_blocks=self.loop_blocks,
            loop_max=self.loop_max,
            d_model=self.model_d_model,
            num_heads=self.model_num_heads,
            d_ff=self.model_d_ff,
            eps=self.model_eps,
            rope_theta=self.model_rope_theta,
            conditioning=self.conditioning,
            loop_lora_rank=self.loop_lora_rank,
        )

    def loss(self) -> LossConfig:
        return LossConfig(loop_max=self.loop_max, shortcut_weight=self.shortcut_weight, consistency_weight=self.consistency_weight)


def load_run_config(run_dir: str | os.PathLike) -> RunConfig:
    with open(os.path.join(run_dir, CONFIG_FILE)) as f:
        return RunConfig.model_validate(json.load(f))


def latest_checkpoint_path(run_dir: str | os.PathLike) -> str:
    checkpoints_dir = os.path.join(run_dir, CHECKPOINTS_DIR)
    checkpoints = [os.path.join(checkpoints_dir, name) for name in os.listdir(checkpoints_dir) if name.endswith(".pt")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {checkpoints_dir}")
    return max(checkpoints, key=os.path.getmtime)


def load_trained_model(run_dir: str | os.PathLike, device: str, checkpoint_path: str | None = None) -> tuple[LoopFormerLM, RunConfig]:
    """The model of a finished run, built from its config.json and loaded from its newest checkpoint (or the given one),
    in eval mode, with the run's settings."""
    run = load_run_config(run_dir)
    model = LoopFormerLM(run.architecture(), device=device, dtype=torch.float32)
    load_model(checkpoint_path or latest_checkpoint_path(run_dir), model)
    model.eval()
    return model, run
