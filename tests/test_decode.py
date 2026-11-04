from functools import partial

import pytest
from support import LOOP_MAX

from decode import generate, sample_next_token
from run import load_trained_model
from trajectory import evaluation_schedule

PROMPT_LENGTH = 4
GREEDY = partial(sample_next_token, temperature=0.0)


@pytest.mark.parametrize("loops", [LOOP_MAX, 1])
def test_greedy_decoding_from_the_start_of_the_memorized_sample_reproduces_the_rest(overfit_run, loops):
    model, run = load_trained_model(overfit_run["run_dir"], device="cpu")
    sample = overfit_run["sample"].tolist()
    prompt, rest = sample[:PROMPT_LENGTH], sample[PROMPT_LENGTH:]

    generated = list(generate(model, prompt, evaluation_schedule(LOOP_MAX, loops), run.model_context_length, max_tokens=len(rest), sample=GREEDY))

    assert generated == rest


def test_generation_stops_at_a_stop_token(overfit_run):
    model, run = load_trained_model(overfit_run["run_dir"], device="cpu")
    sample = overfit_run["sample"].tolist()
    prompt, rest = sample[:PROMPT_LENGTH], sample[PROMPT_LENGTH:]
    stop_token = rest[3]

    generated = list(generate(model, prompt, evaluation_schedule(LOOP_MAX, LOOP_MAX), run.model_context_length, max_tokens=len(rest), sample=GREEDY,
                              stop_ids={stop_token}))

    assert generated == rest[:4]
