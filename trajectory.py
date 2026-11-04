"""Routes through the loops.

A trajectory is a list of step sizes that sum to 1. Each step is a whole number of grid units of 1/loop_max, so every
(t, step) pair the model sees lies on the grid it trains on. loop_max is L in the paper, the number of loops of the
long trajectory.
"""
import random


def long_trajectory(loop_max: int) -> list[float]:
    return [1 / loop_max] * loop_max


def sample_short_trajectory(loop_max: int, rng: random.Random = random) -> list[float]:
    """The reference code's sampler: a length from 1 to loop_max - 1, then step by step any size that leaves at least one
    unit for each remaining step. The last step takes what is left."""
    if loop_max < 2:
        raise ValueError("loop_max must be at least 2")
    length = rng.randint(1, loop_max - 1)
    units = []
    remaining = loop_max
    for steps_left in range(length, 1, -1):
        step = rng.randint(1, remaining - (steps_left - 1))
        units.append(step)
        remaining -= step
    units.append(remaining)
    return [u / loop_max for u in units]


def evaluation_schedule(loop_max: int, loops: int) -> list[float]:
    """The one schedule used to evaluate a budget of `loops`: the grid units split as evenly as possible, larger steps
    first, as the paper found coarse-early schedules work best."""
    if not 1 <= loops <= loop_max:
        raise ValueError(f"loops must be between 1 and {loop_max}, got {loops}")
    base, extra = divmod(loop_max, loops)
    units = [base + 1] * extra + [base] * (loops - extra)
    return [u / loop_max for u in units]
