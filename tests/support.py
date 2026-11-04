"""Creation and action helpers shared by the tests: a tiny model that trains on the CPU in seconds."""
import torch

from model import LoopFormerLM, ModelConfig
from trajectory import long_trajectory

VOCAB_SIZE = 100
CONTEXT_LENGTH = 16
LOOP_MAX = 2
TINY_CONFIG = ModelConfig(vocab_size=VOCAB_SIZE, context_length=CONTEXT_LENGTH, loop_blocks=2, d_model=64, num_heads=4, d_ff=128)


def tiny_model(seed: int = 0) -> LoopFormerLM:
    torch.manual_seed(seed)
    return LoopFormerLM(TINY_CONFIG)


def random_tokens(seed: int = 0, batch_size: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB_SIZE, (batch_size, CONTEXT_LENGTH), generator=generator)


def next_token_loss(model: LoopFormerLM, tokens: torch.Tensor, schedule: list[float]) -> torch.Tensor:
    logits, _ = model(tokens[:, :-1], schedule)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())


def after_training_steps(model: LoopFormerLM, tokens: torch.Tensor, steps: int = 3) -> LoopFormerLM:
    """A model whose gates have left zero: a few plain SGD steps on the long trajectory."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad()
        next_token_loss(model, tokens, long_trajectory(LOOP_MAX)).backward()
        optimizer.step()
    optimizer.zero_grad()
    return model
