"""Generates text from a finished run's checkpoint, at a chosen loop count or along an explicit schedule.

    uv run decode.py --run_dir runs/<run_name> --loops 2 --temperature 0.8 --top_p 0.9
    uv run decode.py --run_dir runs/<run_name> --schedule 0.5,0.25,0.25

Type a prompt at the > and the model continues it, token by token, until a special token or --max_tokens. The context
carries over between prompts. Adapted from the cs336 assignment's decoding script.
"""
import argparse
from collections.abc import Callable, Iterator
from functools import partial

import torch

from model import LoopFormerLM, softmax
from run import load_trained_model
from tokenizer import Tokenizer
from trajectory import evaluation_schedule

Sampler = Callable[[torch.Tensor], int]  # the next token id for a vector of logits


def generate(model: LoopFormerLM, prompt_ids: list[int], schedule: list[float], context_length: int, max_tokens: int, sample: Sampler,
             stop_ids: frozenset[int] | set[int] = frozenset()) -> Iterator[int]:
    """Yields up to max_tokens continuation tokens, the last one a stop token if one is reached. Only the last
    context_length tokens feed the model."""
    context = list(prompt_ids)
    device = next(model.parameters()).device
    for _ in range(max_tokens):
        model_input = torch.tensor(context[-context_length:], device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(model_input, schedule)
        next_id = sample(logits[0, -1])
        context.append(next_id)
        yield next_id
        if next_id in stop_ids:
            return


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float | None = None, top_k: int | None = None) -> int:
    """Temperature 0 is greedy. top_k keeps the k most likely tokens, top_p the smallest set whose probability reaches p."""
    if temperature == 0:
        return torch.argmax(logits).item()
    probabilities = softmax(logits / temperature, 0)
    values, indices = torch.sort(probabilities, descending=True)

    if top_k:
        values[top_k:] = 0

    if top_p:
        cumulative_probabilities = torch.cumsum(values, dim=0)
        values[cumulative_probabilities - values >= top_p] = 0

    index = torch.multinomial(values, num_samples=1)
    return indices[index].item()


def parse_schedule(text: str) -> list[float]:
    schedule = [float(step) for step in text.split(",")]
    if abs(sum(schedule) - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError(f"the steps must sum to 1, got {sum(schedule)}")
    return schedule


def decoding(args: argparse.Namespace):
    model, run = load_trained_model(args.run_dir, args.device, args.checkpoint)
    tokenizer = Tokenizer.from_dir(run.tokenizer_dir)
    schedule = args.schedule if args.schedule else evaluation_schedule(run.loop_max, args.loops or run.loop_max)
    sample = partial(sample_next_token, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    stop_ids = tokenizer.special_token_ids()
    print(f"Loops: {len(schedule)}, schedule {schedule}")

    context: list[int] = []
    while True:
        try:
            prompt = input("> ")
        except (KeyboardInterrupt, EOFError):
            break
        if len(prompt) == 0:
            continue
        context += tokenizer.encode(prompt)
        for token_id in generate(model, context, schedule, run.model_context_length, args.max_tokens, sample, stop_ids):
            context.append(token_id)
            if token_id not in stop_ids:
                print(tokenizer.decode([token_id]), end="", flush=True)
        print()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument("--run_dir", type=str, required=True)
    arg_parser.add_argument("--checkpoint", type=str, default=None, help="a checkpoint file; default: the run's newest")
    arg_parser.add_argument("--loops", type=int, default=None, help="the loop count M at inference; default: the run's loop_max")
    arg_parser.add_argument("--schedule", type=parse_schedule, default=None, help="explicit step sizes, e.g. 0.5,0.25,0.25; overrides --loops")
    arg_parser.add_argument("--temperature", type=float, default=1.0)
    arg_parser.add_argument("--top_p", type=float)
    arg_parser.add_argument("--top_k", type=int)
    arg_parser.add_argument("--max_tokens", type=int, default=1000)
    arg_parser.add_argument("--device", type=str, default="mps")
    args = arg_parser.parse_args()

    decoding(args)
