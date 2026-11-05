# LoopFormer on OpenWebText

An independent reimplementation of the LoopFormer ([Jeddi, Ciccone, Taati, 2026](https://arxiv.org/abs/2602.11451)): a decoder-only transformer with k unique blocks applied L times, each loop conditioned on its position in the journey, and trained so that fewer loops still give a useful answer. It is written from the paper, not from the authors' code, and built from the components of my [CS336 assignment 1](https://github.com/pjurczen/cs336-assignment1-basics) (the same attention with QK-norm, RoPE, SwiGLU, tied embeddings, Muon and AdamW), so its results compare directly with the plain transformer trained there. A curiosity project, not part of the assignment.

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

Everything runs from the repository root, in the environment `uv` builds from `uv.lock`:

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

The training data and the tokenizers are not in the repository. They come from the assignment's BPE pipeline: `data/<name>/{train,valid}_ids.npy` (token ids as uint16) and `tokenizers/<name>/{vocab.json,merges.txt,tokenizer_config.json}`, for `owt-32k` (OpenWebText, 32K vocabulary) and `tinystories-10k`. Link or copy them into `data/` and `tokenizers/` (both ignored by git); the Modal runs read them from the volume.

### Autoresearch on the Mac

`autoresearch.md` is the program for an agent that tests the ideas of `ideas.md` on its own, after [Karpathy's autoresearch](https://github.com/karpathy/autoresearch): one change, one 10-minute run on the Mac's GPU, one number, keep or `git reset`, all night. `configs/mps-10min.sh <run_name>` is the fixed experiment it runs (TinyStories, 4 blocks × 2 loops at d_model 384, inductor-compiled on MPS at ~12K tokens/s), and `CLAUDE.md` hands the agent the program. Start a fresh Claude Code session in the repository root and say: "Read autoresearch.md and set up a new experiment run; after setup, start the loop."

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

### Ablations: lighter ways to condition the shared block

The LoopFormer block peaked at 46 GB at batch 64 against 37 GB for a plain block, which kept the B200 run from the plain model's batch 256. The adaLN modulation (the (1 + γ) scale after each norm and the gate α on each branch) looked like the cause, so three lighter conditionings ran on the 3×4 shape with the phase-1 flags, one change each against the phase-1 run (`--conditioning gate`, `--conditioning additive`, `--conditioning additive --loop_lora_rank 32`):

- **gate:** the plain block (norms with their learned weight) plus a zero-start gate α per branch; no scales.
- **additive:** the plain block; the (t, Δt) vector is added to the hidden state at each loop start, as Universal Transformers add their step embedding. Its conditioning MLPs start at zero.
- **additive + LoRA 32:** additive, plus per-loop low-rank deltas (rank 32) on the five shared matrices, as in Relaxed Recursive Transformers: 5M extra weights for 3×4, trained with AdamW.

| Variant | Loss at 4 loops | Loss at 1 / 2 / 3 loops | Steps | Peak memory | ms per step | Max gradient norm in warmup |
|---|---|---|---|---|---|---|
| adaLN, scales + gates (phase 1) | 3.879 | 4.110 / 3.928 / 3.893 | 2,130 | 46.0 GB | 417 | 18.1 |
| gate | 3.881 | 4.120 / 3.929 / 3.895 | 2,148 | 45.9 GB | 413 | 8.4 |
| additive | 3.902 | 4.133 / 3.968 / 3.913 | 2,041 | 45.3 GB | 438 | 3.7 |
| additive + LoRA 32 | 3.901 | 4.138 / 3.988 / 3.923 | 1,951 | 35.8 GB | 456 | 2.7 |

- **The scales do no work.** Gates alone match adaLN within noise at every loop count, and halve the gradient bursts of early warmup. The paper's full modulation is not needed for its result here.
- **Dropping the modulation costs 0.023**, mostly at two loops (3.968 against 3.928). A multiplicative per-loop signal is worth that much, in line with TMLT's argument; an additive one is not enough.
- **A few per-loop weights add nothing at this budget.** LoRA 32 lands on top of additive (3.901) while costing 9% per step.
- **None of them is lighter, which is the surprise.** The design of this ablation assumed the modulation was the 9 GB between a LoopFormer block and a plain one; that assumption was never measured, and it was wrong. Removing the whole modulation moved the peak from 46.0 to 45.3 GB. And the LoRA variant, which adds tensors, peaked at 35.8 GB.

### Where the memory goes, measured

`train.py --memory_snapshot_step N` records every allocation of one training step (`torch.cuda.memory._record_memory_history`) and writes `memory_snapshot.pickle` into the run folder; `profiling.py` replays the trace to the peak. For 3×4 at batch 64 (step 150, 45.3 GB live at the peak):

| GB | What | Where |
|---|---|---|
| 14.6 | the attention probabilities after softmax, kept for backward: 805 MB × 12 block applications in float32, plus a bf16 copy each | `softmax` |
| 6.0 | the raw q·k score matrices (bf16, 403 MB × 12) and the attention outputs | `scaled_dot_product_attention` |
| 10.5 | the loss: the bf16 logits, their max-subtracted copy, and a float32 cast and a float32 `exp` over 64 × 512 × 32,000 (4.2 GB each) | `cross_entropy` |
| 9.8 | the inputs the linear layers keep: qkv outputs, SwiGLU inner activations, output projections | `Linear.forward` |
| 1.6 | the SwiGLU gate product | `SwiGLU.forward` |
| 1.2 | the adaLN multiplies | `AdaLNBlock.forward` |
| 0.6 | norms, embeddings, the rest | |

The modulation is 1.2 GB. The stored attention matrices are 20 GB, 45% of the peak, and half of that is float32 because autocast runs the hand-written softmax's `exp` and `sum` in float32. The loss is another 10 GB; the "no float32 copy of the logits" in the training step avoided one copy, and autocast makes two more inside `exp`.

The same snapshot on the plain 12-layer model (same batch, same step; peak 36.1 GB) explains the gap. Every row matches within a gigabyte except one: the float32 softmax output. The plain model, compiled as one graph, keeps a single one (0.8 GB) and recomputes the others during backward from the bf16 scores; the LoopFormer, compiled block by block, keeps all twelve (9.7 GB). The 9 GB between the two models is a compile-granularity effect, not the architecture.

### Compile scope: the same model, three ways to compile it

`train.py --compile` picks how `torch.compile` sees the model. The schedule enters the compiled forward as a tensor, so its values are graph inputs rather than constants (with a Python list, every distinct schedule would be a new graph, which is why the reference repository trains without compilation at all). Same 3×4 model, batch 64, A100-80GB, memory snapshot at step 150:

| `--compile` | What is compiled | Peak memory | ms per step | Graphs built |
|---|---|---|---|---|
| `blocks` (all runs above) | each block on its own, called once per loop | 46.0 GB | 417 | one per block |
| `whole` | the whole forward with the loop unrolled | **37.2 GB** | ~380 | one per loop count (4), plus 2 for validation shapes |
| `loop` | the whole forward with the loop as a `torch.while_loop`, the loop count a tensor | **16.7 GB** | 488 | 1, plus 2 for validation shapes |
| plain 12-layer, for reference | the whole model | 37.1 GB | 302 | 1 |

- **`whole` lands on the plain model's number**, and its snapshot shows the plain model's breakdown: the float32 softmax outputs are gone, recomputed in backward. It is also no slower than `blocks`. The compile cost is about 80 s for the first graph and 15 s for each shorter loop count.
- **`loop` cuts the peak to a third.** `while_loop`'s backward keeps only the carried state of each iteration (0.5 GB) and recomputes the loop body, which is activation checkpointing per loop for free. The price is the recomputed forward, about 15% per step. One graph serves every loop count. Per-loop LoRA is not supported in this scope, since it sets the loop position on the modules, which a `while_loop` body may not do.
- For the B200 this means batch 256 fits with `whole` (about 145 GB, like the plain model) and batch 512 would fit with `loop`.

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
