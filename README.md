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

All runs use OpenWebText with the 32K-vocabulary tokenizer, context 512, d_model 768, 12 heads, d_ff 2048, Muon for the block matrices and AdamW for the rest, bf16, warmup 120 steps, weight decay 0.1. The loss is the cross-entropy in nats per token on the whole validation set (66M tokens), measured at the full route of L loops unless a loop count is given.

### Phase 1: equal time, 15 minutes on one A100-80GB

Batch 64, learning rate 4e-3 decaying to 4e-4 over the 15 minutes: the settings of the plain 12-layer model with QK-norm, which is the reference. Unique weights exclude the embeddings; a LoopFormer block has 7.1M weights plus a 2.4M modulator, and the two conditioning MLPs add 1.6M.

| Model | Unique weights | Steps | Tokens seen | ms per step | Peak memory | Loss |
|---|---|---|---|---|---|---|
| plain 12L/768 + QK-norm (reference) | 85M | 2,784 | 91M | 302 | 37 GB | **3.693** |
| LoopFormer 6×2 | 58M | 2,103 | 69M | 422 | 46 GB | 3.831 |
| LoopFormer 4×3 | 39M | 2,136 | 70M | 416 | 46 GB | 3.854 |
| LoopFormer 3×4 | 30M | 2,130 | 70M | 417 | 46 GB | 3.879 |
| LoopFormer 2×6 | 21M | 2,191 | 72M | 408 | 46 GB | 3.910 |

- **Equal time:** every looped shape trails the plain 12-layer model, by 0.14 (6×2) to 0.22 (2×6). The more unique blocks, the better: looping a small stack more often does not make up for its missing weights. This matches the paper, where looped models trail non-looped ones on perplexity and the gap is explained by parameters and memorization.
- **Part of the gap is speed.** A LoopFormer step runs the long route and a sampled shorter one, about 1.4× the cost of a plain step, so in 15 minutes it sees 70M tokens where the plain model sees 91M.
- **Equal parameters:** 6×2 (58M) against the plain 8L/768 (57M, 3.669; trained on a 40 GB A100 at learning rate 2.8e-3 without QK-norm, 130M tokens in its 15 minutes): the looped model is 0.16 worse. It also saw half the tokens, so this is an equal-time comparison at equal parameters, not an equal-token one.
- **Stability:** the gradient norm spiked to 18 in the first steps of 3×4 and 2×6 (6 and 9 for 4×3 and 6×2; 2.4 for the plain model). All four runs recovered and trained normally after warmup.

### Elastic depth: one trained model at every loop count

The loss of each phase-1 model when run with M loops, on the schedule with the larger steps first (for example (½, ¼, ¼) for M = 3 of L = 4):

| Loops M | 2×6 | 3×4 | 4×3 | 6×2 |
|---|---|---|---|---|
| 1 | 4.564 | 4.110 | 4.010 | 3.876 |
| 2 | 4.073 | 3.928 | 3.879 | 3.831 |
| 3 | 3.961 | 3.893 | 3.854 | |
| 4 | 3.934 | 3.879 | | |
| 5 | 3.920 | | | |
| 6 | 3.910 | | | |

The loss falls with every added loop, by far the most from one loop to two. One loop fewer than L costs 0.01–0.03, half the loops cost 0.05–0.16, and a single loop costs 0.05 (6×2) to 0.65 (2×6). So a trained LoopFormer does give a usable model at a reduced budget, and the price of each skipped loop is small near the full route.

The per-loop evaluation of one run takes 10 to 12 minutes on the A100: the full validation set once per M, with M loops each.

### Phase 2: 45 minutes on a B200

The best phase-1 shape, 6×2, for the leaderboard budget. The plain model's batch 256 does not fit: phase 1 peaked at 46 GB at batch 64, which scales to about the B200's whole 180 GB. So batch 192 with the learning rate √-scaled from 4e-3 at batch 64, 6.9e-3 decaying to 6.9e-4, warmup 500 steps, and a reserve of 2 minutes for the final validation.

| Model | Unique weights | Steps | Tokens seen | ms per step | Peak memory | Loss |
|---|---|---|---|---|---|---|
| plain 12L/768 + QK-norm, batch 256, lr 8e-3 | 85M | 9,233 | 1.21B | 275 | 145 GB | **3.191** |
| LoopFormer 6×2, batch 192, lr 6.9e-3 | 58M | 7,586 | 746M | 339 | 137 GB | 3.346 |

At one loop the same model gives 3.414, at two loops 3.346.

- **The gap is 0.155, the same as on the A100** (0.14 at 15 minutes): more training did not close it. The looped model also saw 62% of the plain model's tokens, from the smaller batch and the two-route step.
- **At equal wall-clock** (step 1,000, about 340 s into each run) the looped model was at 3.947 against 3.776, with 500 of its steps in warmup against 120.
- **Stability:** a first attempt with the phase-1 warmup of 120 steps spiked right after warmup (loss 4.98 → 7.33 at step 300) and recovered by step 1,000; it was stopped and restarted with warmup 500. The restart showed short gradient bursts (norm up to 1.5) as the rate climbed past 6e-3, each contained by clipping at 1.0, and nothing after warmup. The plain model at 8e-3 sat at norms of 0.07–0.14 throughout. The shared block receives gradient from every loop, so the looped model is closer to its stability edge at the same rate, which matches the paper's remark that looped models need a long warmup.
