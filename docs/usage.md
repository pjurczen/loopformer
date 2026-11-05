# Running LoopFormer

## Layout

| File                                                   | Job                                                                                                     |
|--------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `model.py`                                             | the copied components, the conditioning, the LoopFormer block, `LoopFormerLM`                           |
| `trajectory.py`                                        | the long route, the sampler for short routes, the evaluation schedule for a budget                      |
| `train.py`                                             | the training loop: two-route loss, wall-clock budget, validation, checkpoints, metrics, wandb           |
| `evaluate.py`                                          | the validation loss along a schedule, and a run's loss at every loop count (`budget_eval.json`)         |
| `decode.py`                                            | text generation from a run, at a loop count or along an explicit schedule                               |
| `run.py`                                               | a run folder: its settings (`config.json`), newest checkpoint and trained model                         |
| `optim.py`, `data.py`, `checkpoint.py`, `tokenizer.py` | copies of the assignment's optimizers, schedules, batches, loss, checkpoint files and BPE encode/decode |
| `modal_app.py`, `configs/`                             | the cloud runner and the flags of the experiment's runs                                                 |

## How to run

Everything runs from the repository root, in the environment `uv` builds from `uv.lock`:

```sh
uv run pytest                                           # behaviour tests on a tiny model, a few seconds on the CPU
uv run train.py --help                                  # a local run: --device mps or cpu, see configs/ for the flags
uv run evaluate.py --run_dir runs/<run_name> --device mps
uv run decode.py --run_dir runs/<run_name> --loops 2 --temperature 0.8 --top_p 0.9
```

The experiment's runs go to Modal (`modal_app.py`): `configs/smoke.sh` checks the whole path in 3 minutes, `configs/a100-20min.sh <k> <L>` trains one shape for
15 minutes on an A100-80GB, `configs/b200-45min.sh <k> <L>` is the 45-minute B200 run. Each run ends with the per-loop evaluation in the same container, and its
folder lands in the volume `cs336-data` under `runs/`:

```sh
uv run modal volume get cs336-data runs/<run_name> runs/
```


## Data and tokenizers

Any text corpus and any BPE tokenizer work, as long as they arrive in the two shapes the code reads, both in folders git ignores:

- `data/<name>/train_ids.npy` and `valid_ids.npy`: one flat array of token ids each (`uint16`, or `uint32` for vocabularies above 65,536), read with
  `np.memmap`; a training batch is random windows of it, a validation pass walks it in non-overlapping windows.
- `tokenizers/<name>/`: `tokenizer_config.json` (`{"vocab_size": …, "special_tokens": […]}`; training reads only the vocabulary size) plus `vocab.json` and
  `merges.txt` in the GPT-2 byte-to-unicode format, which `decode.py` needs to turn ids back into text.

The corpora and tokenizers of the results, and the commands that produced them, are in the README.

## The flags that matter

`uv run train.py --help` lists them all; these shape an experiment:

- `--loop_blocks k --loop_max L`: k unique blocks, L loops on the long route.
- `--conditioning adaln|gate|additive`: how a block learns its loop position: scales and gates (the paper), gates only, or the (t, Δt) vector added to the
  hidden state at each loop start. `--loop_lora_rank r` adds per-loop low-rank deltas on the shared matrices (0 = off).
- `--shortcut_weight`, `--consistency_weight`: the two 0.1 of the loss.
- `--compile blocks|whole|loop|none`: how `torch.compile` sees the model (see the compile-scope comparison in [results.md](results.md)); `whole` is the
  default, `none` runs eagerly.
- `--time_budget_minutes M --time_reserve_minutes R --optimizer_warmup_iters W`: train for M − R minutes of training time (the first 10 steps, which compile
  the graphs, are not counted) with the cosine following that clock, and keep R minutes for the final validation. Without a budget, `--iter` steps and a
  step-based cosine.
- `--seed`: the weights, the batches and the short trajectories (MPS stays nondeterministic even so).
- `--final_val_batches N`: the final validation scores the first N batches instead of the whole set. `--checkpoints_iter 0`: no checkpoints.
- `--memory_snapshot_step N`: record every allocation of step N into the run folder (CUDA); `uv run profiling.py <pickle>` summarizes it.
- `--bf16`, `--optimizer_muon`, `--wandb_project`: precision, the optimizer split, the live dashboard.

A run writes `runs/<run_name>/config.json`, `metrics.jsonl` and `checkpoints/`, and ends with a summary block (`val_loss:`, `val_loss_1loop:`, `steps:`,
`tok_per_s:`, `peak_memory_gb:`, …) that the autoresearch loop greps.
