"""Checkpoint files: the model's weights, the optimizer states and the step count. Every checkpoint here is saved from
the plain model (never from a torch.compile wrapper), so the keys need no conversion on load."""
import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes],
        elapsed_time: float | None = None,
        muon_optimizer: torch.optim.Optimizer | None = None,
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    if elapsed_time is not None:
        checkpoint["elapsed_time"] = elapsed_time
    if muon_optimizer is not None:
        checkpoint["muon_optimizer"] = muon_optimizer.state_dict()
    torch.save(checkpoint, out)


def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        muon_optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, float | None]:
    """Restores the model and the optimizers for a resumed run. Returns the step count and the elapsed time."""
    checkpoint = _load_into_model(src, model)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if muon_optimizer is not None:
        if "muon_optimizer" not in checkpoint:
            raise ValueError("Resuming with Muon, but the checkpoint was trained without it")
        muon_optimizer.load_state_dict(checkpoint["muon_optimizer"])
    return checkpoint["iteration"], checkpoint.get("elapsed_time", None)


def load_model(src: str | os.PathLike, model: torch.nn.Module) -> None:
    """The weights only, for evaluation and decoding."""
    _load_into_model(src, model)


def _load_into_model(src, model: torch.nn.Module) -> dict:
    # Load onto the model's device, so a checkpoint saved on CUDA can be used on MPS or the CPU.
    checkpoint: dict = torch.load(src, map_location=next(model.parameters()).device)
    model.load_state_dict(checkpoint["model"])
    return checkpoint
