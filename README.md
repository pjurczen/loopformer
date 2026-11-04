# LoopFormer on OpenWebText

An implementation of the LoopFormer ([Jeddi, Ciccone, Taati, 2026](https://arxiv.org/abs/2602.11451)): a decoder-only transformer with k unique blocks applied L times, each loop conditioned on its position in the journey, and trained so that fewer loops still give a useful answer. It is built from the components of my [CS336 assignment 1](../README.md) (the same attention with QK-norm, RoPE, SwiGLU, tied embeddings, Muon and AdamW), so its results compare directly with the plain transformer trained there. A curiosity project, not part of the assignment.

## What is different from a plain transformer

- **Looping.** k blocks are shared and applied L times. A trajectory is a list of step sizes that sum to 1, each a multiple of 1/L: the long route is L steps of 1/L, and a shorter route takes bigger steps.
- **Conditioning.** Each loop gets a vector c computed from the time t reached so far and its own step size, through fixed sine/cosine features and a small MLP each. Inside a block, c sets the scale after each RMSNorm, (1 + γ), and gates each residual branch, α. The modulator starts at zero, so a fresh block passes its input through unchanged.
- **Training.** Every step runs the long route and one sampled shorter route: loss = CE_long + 0.1 · CE_short + 0.1 · MSE(stopgrad(h_long), h_short). The two routes run backward one after the other, which gives the same gradient as the paper's single backward at about half the peak memory.
- **Elastic depth.** The trained model runs at any loop count M ≤ L. `evaluate.py` measures the loss at every M, `decode.py` generates text at a chosen M.

## Layout

| File | Job |
|---|---|
| `model.py` | the copied components, the conditioning, the LoopFormer block, `LoopFormerLM` |
| `trajectory.py` | the long route, the sampler for short routes, the evaluation schedule for a budget |
| `train.py` | the training loop: two-route loss, wall-clock budget, validation, checkpoints, metrics, wandb |
| `evaluate.py` | the validation loss along a schedule, and a run's loss at every loop count (`budget_eval.json`) |
| `decode.py` | text generation from a run, at a loop count or along an explicit schedule |
| `run.py` | a run folder: its settings (`config.json`), newest checkpoint and trained model |
| `optim.py`, `data.py`, `checkpoint.py`, `tokenizer.py` | copies of the assignment's optimizers, schedules, batches, loss, checkpoint files and BPE encode/decode |
| `modal_app.py`, `configs/` | the cloud runner and the flags of the experiment's runs |

## How to run

Everything runs from inside this folder, with its own environment:

```sh
uv run pytest                                           # behaviour tests on a tiny model, a few seconds on the CPU
uv run train.py --help                                  # a local run: --device mps or cpu, see configs/ for the flags
uv run evaluate.py --run_dir runs/<run_name> --device mps
uv run decode.py --run_dir runs/<run_name> --loops 2 --temperature 0.8 --top_p 0.9
```

The experiment's runs go to Modal (`modal_app.py`): `configs/smoke.sh` checks the whole path in 3 minutes, `configs/a100-20min.sh <k> <L>` trains one shape for 15 minutes on an A100-80GB, `configs/b200-45min.sh <k> <L>` is the 45-minute B200 run. Each run ends with the per-loop evaluation in the same container, and its folder lands in the volume `cs336-data` under `runs/`:

```sh
uv run modal volume get cs336-data runs/<run_name> runs/
```

The training data and the tokenizer are those of the assignment, `data/owt-32k/*.npy` and `tokenizers/owt-32k/`, on the volume.

## Results

Pending: see the design and plan in `../docs/playbooks/`.
