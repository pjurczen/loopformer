"""A run's settings, as train.py records them in config.json, and the translations other scripts need from them."""
import json
import os
from dataclasses import dataclass

from pydantic import BaseModel

from model import ModelConfig


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
    git_commit: str | None  # code version the run was trained with; "-dirty" means uncommitted changes
    time_budget_minutes: float | None  # with a budget, training ended on time rather than at iter
    time_reserve_minutes: float

    def architecture(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.model_vocab_size,
            context_length=self.model_context_length,
            loop_blocks=self.loop_blocks,
            d_model=self.model_d_model,
            num_heads=self.model_num_heads,
            d_ff=self.model_d_ff,
            eps=self.model_eps,
            rope_theta=self.model_rope_theta,
        )

    def loss(self) -> LossConfig:
        return LossConfig(loop_max=self.loop_max, shortcut_weight=self.shortcut_weight, consistency_weight=self.consistency_weight)


def load_run_config(run_dir: str | os.PathLike) -> RunConfig:
    with open(os.path.join(run_dir, "config.json")) as f:
        return RunConfig.model_validate(json.load(f))
