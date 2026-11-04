import random

import pytest

from trajectory import evaluation_schedule, long_trajectory, sample_short_trajectory

SAMPLES = 500


def on_grid(step: float, loop_max: int) -> bool:
    units = step * loop_max
    return abs(units - round(units)) < 1e-9 and round(units) >= 1


@pytest.mark.parametrize("loop_max", [2, 4, 6])
def test_sampled_short_trajectories_are_shorter_than_the_long_one_and_stay_on_the_grid(loop_max):
    rng = random.Random(0)

    trajectories = [sample_short_trajectory(loop_max, rng) for _ in range(SAMPLES)]

    for trajectory in trajectories:
        assert 1 <= len(trajectory) <= loop_max - 1
        assert all(on_grid(step, loop_max) for step in trajectory)
        assert sum(trajectory) == pytest.approx(1.0)


def test_sampling_reaches_every_short_length():
    loop_max = 4
    rng = random.Random(0)

    lengths = {len(sample_short_trajectory(loop_max, rng)) for _ in range(SAMPLES)}

    assert lengths == {1, 2, 3}


def test_the_long_trajectory_has_one_uniform_step_per_loop():
    assert long_trajectory(4) == [0.25, 0.25, 0.25, 0.25]


def test_the_evaluation_schedule_puts_the_larger_steps_first():
    assert evaluation_schedule(loop_max=4, loops=3) == [0.5, 0.25, 0.25]


@pytest.mark.parametrize("loop_max, loops", [(4, 1), (4, 2), (4, 4), (6, 4), (6, 5)])
def test_the_evaluation_schedule_has_the_requested_number_of_steps_on_the_grid(loop_max, loops):
    schedule = evaluation_schedule(loop_max, loops)

    assert len(schedule) == loops
    assert all(on_grid(step, loop_max) for step in schedule)
    assert sum(schedule) == pytest.approx(1.0)
    assert schedule == sorted(schedule, reverse=True)
