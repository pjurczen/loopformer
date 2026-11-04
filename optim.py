"""Optimizers, learning-rate schedules and gradient clipping: copies of the cs336 assignment versions."""
import math
from collections.abc import Callable, Iterable

import torch


class AdamWOptimizer(torch.optim.Optimizer):

    def __init__(self, params: Iterable[torch.nn.Parameter], lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None) -> Callable | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state["m"] if "m" in state else torch.zeros_like(p)
                v = state["v"] if "v" in state else torch.zeros_like(p)
                grad = p.grad.data
                lr_t = lr * (math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t))  # bias-corrected step size for iteration t
                p.data -= lr * weight_decay * p.data  # decoupled weight decay
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad ** 2
                p.data -= lr_t * m / (v.sqrt() + eps)
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


def newton_schulz(G: torch.Tensor, a: float = 3.4445, b: float = -4.7750, c: float = 2.0315) -> torch.Tensor:
    """Five Newton-Schulz iterations in bf16: an approximate orthogonalization of G."""
    eps = 1e-7
    X = G.bfloat16()
    X = X / (X.pow(2).sum().sqrt() + eps)
    if G.shape[0] > G.shape[1]:
        X = X.T
    for _ in range(5):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X


class MuonOptimizer(torch.optim.Optimizer):
    """Muon for 2-D matrices: Nesterov momentum, orthogonalized update, scaled to match AdamW's update size."""

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3, weight_decay: float = 0.01, momentum: float = 0.95) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "momentum": momentum}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None) -> Callable | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state["m"] if "m" in state else torch.zeros_like(p)
                grad = p.grad.data
                m = momentum * m + grad
                u = grad + momentum * m
                o = newton_schulz(u)
                n = lr * 0.2 * math.sqrt(max(p.shape))
                p.data -= lr * weight_decay * p.data
                p.data -= n * o
                state["m"] = m
        return loss


def cosine_lr_schedule(t: int, lr_max: float, lr_min: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if t < warmup_iters:
        return t / warmup_iters * lr_max
    if warmup_iters <= t <= cosine_cycle_iters:
        return lr_min + 1 / 2 * (1 + math.cos((t - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (lr_max - lr_min)
    return lr_min


def cosine_lr_schedule_by_time(t: int, elapsed_seconds: float, budget_seconds: float, lr_max: float, lr_min: float, warmup_iters: int) -> float:
    # Like cosine_lr_schedule, but the cosine follows the share of the time budget used instead of the step count.
    # Warmup stays in steps; it's short, so the cosine starting at time 0 instead of after warmup barely matters.
    if t < warmup_iters:
        return t / warmup_iters * lr_max
    if elapsed_seconds <= budget_seconds:
        return lr_min + 1 / 2 * (1 + math.cos(elapsed_seconds / budget_seconds * math.pi)) * (lr_max - lr_min)
    return lr_min


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> torch.Tensor:
    """Scales the gradients in place to a total L2 norm of at most max_l2_norm. Returns the norm before clipping, as a
    tensor on the device, so logging decides when to synchronize."""
    params = [p for p in parameters if p.grad is not None]
    l2_norm = torch.stack([p.grad.pow(2).sum() for p in params]).sum().sqrt()
    scaling_factor = (max_l2_norm / (l2_norm + eps)).clamp(max=1.0)
    for param in params:
        param.grad = param.grad.mul_(scaling_factor)
    return l2_norm
