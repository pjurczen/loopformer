# LoopFormer autoresearch on the Mac — overnight setup

## Context

The LoopFormer trails the plain model at equal time; `loopformer/ideas.md` holds 14 ranked ideas, five of them
cheap loop-quality changes (6 prelude–loop–coda, 8 per-token gate with the hidden state, 1 shared first-loop K/V,
7 hyper-connections once per loop, 9 stability / learning rate). Testing them one A100 run at a time is slow and costs
money. Karpathy's autoresearch (`~/repo/autoresearch`) shows the alternative: an agent edits the code, trains for a
fixed wall-clock budget, reads one number, keeps or `git reset`s, repeats all night. Its asset is the protocol in
`program.md` (~100 lines), not its CUDA-only nanochat code; the macOS fork (miolini) only swaps flash-attention for
SDPA and turns compile and autocast off on MPS; the MLX fork (trevin-creator) is a rewrite in Apple's MLX, useless
for our PyTorch model, but its `rigor.py` records the one lesson both forks learned on Apple Silicon: re-running the
*same* code moves the metric by ~0.03 even with the seed pinned (nondeterministic GPU reductions), so a keep rule
that accepts any run below the running best produces an optimistic curve that regresses on re-evaluation. Our keep
rule below confirms every would-be keep with a second seed. We keep our model, data loader, optimizer and loss in
nats (fixed tokenizer, so bits-per-byte buys nothing) and copy the protocol.

Measured today on this Mac (M2 Pro, 16 GB, torch 2.11, fp32, context 256, batch 32, full two-route LoopFormer step):
4 blocks × 2 loops at d_model 384 / 6 heads / d_ff 1024 (12.6M params) runs at **12.4K tok/s with
`torch.compile(backend="inductor")` on MPS** (6.5 GB), 2.1× eager; `aot_eager` (what `compile_for_device` picks off
CUDA today) gives nothing; bf16 is slower on MPS; batch 64 at d512 swaps. A 10-minute run sees ~7M TinyStories
tokens; the user's plain TinyStories model was at val 2.11 after 8M tokens, so loop variants can separate there, but
only above the (unmeasured) seed noise: the baseline runs twice first.

Decisions taken with the user (2026-11-05): 10 minutes training per experiment; the agent may change the architecture
(`model.py`, `trajectory.py`, `training_step` in `train.py`) and the optimizer variables (lr, warmup, weight decay) in
the config script, nothing else; a `loopformer/CLAUDE.md` scopes the course policy out of the side project. Dataset:
TinyStories 10K-vocab (local; Karpathy's own advice for small machines). Shape, batch, context, budget and the metric
are fixed.

Workspace: branch `loopformer`, folder `loopformer/`; commands run from inside it as `env -u VIRTUAL_ENV uv run ...`.
Tests: `env -u VIRTUAL_ENV uv run pytest` (55 pass, CPU). Root `.gitignore` ignores `data/`, `runs/` and `*.log`, not
`tokenizers/` — so the config uses `../data/...` and `../tokenizers/...` paths (as the earlier MPS smoke runs did),
no symlinks. `/opt/homebrew/bin/timeout` and `/usr/bin/caffeinate` exist; 24 GB disk free.

## Changes

### 1. `loopformer/model.py` — inductor on MPS, scope `none`
- `compile_for_device` (`model.py:375-399`): `self._scope = scope; if scope == "none": return` *before* `set_logs`
  and `cache_size_limit`; `backend = "inductor" if device.startswith(("cuda", "mps")) else "aot_eager"` (CPU = tests).
  `COMPILE_SCOPES = ("blocks", "whole", "loop", "none")`. `forward` already falls back to `_forward_unrolled` /
  `self.blocks` when nothing is compiled. `evaluate.py:54` then also uses inductor on MPS (fine).
- `train.py --compile` help text gains `none`.
- The model must be built with `device="mps"` (the attention mask is created on the constructor's device) — `train.py`
  already does.

### 2. `loopformer/train.py` (+ `run.py`) — the autoresearch-ready run
- **Training clock** (`BUDGET_WARMUP_STEPS = 10`, autoresearch's `if step > 10`): `training_seconds` accumulates
  per-step time (with `synchronize(args.device)` at the step end — needed for a true step time; ~0 cost at 600 ms
  steps) for steps `it - iteration >= 10`. `last_step` (`train.py:95`) and `cosine_lr_schedule_by_time`
  (`train.py:116`, `optim.py:96`; warmup stays step-based) use `training_seconds`; during the 10 excluded steps the
  cosine sits at `lr_max` after warmup, harmless. Metrics keep `time` (wall since start) and add `training_time`.
  Checkpoints save `training_seconds` instead of `t3 - t0` (`train.py:158`) and resume restores it (`train.py:86`).
  Config comments in `a100-20min.sh`/`b200-45min.sh`: the budget now excludes compile, so a future run trains ~1.5 min
  longer than the README's phase-1 runs; `time_reserve_minutes` now only covers the final validation.
- **`--seed N`** (default `None` = as today): `torch.manual_seed`, `random.seed` (short-trajectory sampler) and the
  batch sampler (`data.py get_random_indices`: a module-level generator that `train.py` reseeds). Same seed → the
  eager-vs-inductor check compares near-identical runs; different seeds → the noise floor is a seed effect, not luck.
- **`--final_val_batches N`** (default `None` = full set, today's behaviour): `final_val_indices =
  get_nonoverlapping_indices(val, ctx, N * batch_size)` and `full_dataset_eval = last_step and args.final_val_batches
  is None` — the tail window has a different context length and would both skew a 64-batch mean and trigger a
  recompile. `RunConfig` gets `final_val_batches: int | None = None` and `seed: int | None = None` (pydantic ignores
  unknown keys but a declared field without default breaks old `config.json`s).
- **Summary**: `training()` builds and returns a dict and prints it as a grep-able block: `---`, `val_loss` (full
  route), `val_loss_1loop` (`evaluation_schedule(L, 1)` = `[1.0]`, same indices; one extra compile at the end),
  `final_val_tokens`, `training_seconds`, `steps`, `tokens_M`, `tok_per_s` (`(steps - 10) × batch × ctx /
  training_seconds`, guarded for 0), `params_M`, `peak_memory_gb`. `conftest.py` keeps the returned dict in the
  fixture so tests assert on it, not on captured output.
- **Fail fast**: right after `training_step` (`train.py:104`, before gradient clipping, which would raise on empty
  grads): `train_loss = losses["train_loss"].item()`; if `not math.isfinite(train_loss) or train_loss > 100`: print
  `FAIL: training loss … at step …`, finish wandb, `raise SystemExit(1)`.
- **`--checkpoints_iter 0`** = never: `if args.checkpoints_iter and (it % args.checkpoints_iter == 0 or last_step)`
  (the `or last_step` arm stays for the Modal configs, which rely on the last-step checkpoint for `evaluate.py`).
- **`peak_memory_gb` on MPS**: running max of `torch.mps.driver_allocated_memory() / 1e9` sampled at log steps and
  after the final validation (MPS has no peak API).

### 3. `loopformer/configs/mps-10min.sh` — the fixed experiment (standalone; does not source `common.sh`)
Usage from `loopformer/`: `configs/mps-10min.sh <run_name> [extra train.py flags]`. Top of file, the variables the
agent may edit: `LR_MAX=3e-3 LR_MIN=3e-4 WARMUP=100 WEIGHT_DECAY=0.1 BETA2=0.999` (one lr serves Muon and AdamW; the
OWT LoopFormer runs used Muon 4e-3/4e-4 at batch 64; ~900 steps expected, so warmup 100). Then
`export PYTHONUNBUFFERED=1 TORCHINDUCTOR_COMPILE_THREADS=1` (unbuffered `run.log`; inductor's default 10 compile
workers each import torch on a 16 GB machine) and
`exec timeout 1500 env -u VIRTUAL_ENV uv run train.py --device mps --train_data ../data/tinystories-10k/train_ids.npy
--val_data ../data/tinystories-10k/valid_ids.npy --tokenizer_dir ../tokenizers/tinystories-10k --loop_blocks 4
--loop_max 2 --model_d_model 384 --model_num_heads 6 --model_d_ff 1024 --model_context_length 256 --batch_size 32
--optimizer_muon --compile whole --time_budget_minutes 10 --time_reserve_minutes 0 --final_val_batches 64
--val_interval 1000000 --iter 1000000 --checkpoints_iter 0 --log_interval 50 --runs_dir runs/autoresearch
--run_name "$1" ${@:2}` (extra flags last, so they override). No wandb, no bf16. `timeout 1500` is the hard stop.

### 4. `loopformer/autoresearch.md` — the program (adapted from Karpathy's `program.md`)
- **Setup**: `git checkout -b autoresearch/<tag>` from `loopformer`; check `../data/tinystories-10k/*.npy` and the
  tokenizer exist; `results.tsv` header `commit	val_loss	val_loss_1loop	memory_gb	tok_per_s	status	description`
  (untracked, listed in a new `loopformer/.gitignore`); run `caffeinate -i` in a second terminal; run the **baseline
  twice** with `--seed 1` and `--seed 2`: the noise floor is `s = |difference|`, threshold `max(s, 0.01)`; confirm
  with the user, then loop without asking again.
- **In scope**: `model.py`, `trajectory.py`, `training_step` in `train.py`, the variables at the top of
  `configs/mps-10min.sh`. **Fixed**: `data.py`, `evaluate.py`, `tokenizer.py`, `optim.py`, `checkpoint.py`, the
  shape, batch, context, dataset, budget, `--final_val_batches`, the metric. No new dependencies.
- **Loop**: edit → `env -u VIRTUAL_ENV uv run pytest tests/test_model.py -q -x` (seconds; catches shape bugs before a
  12-minute run) → commit (only `loopformer/` paths by name; never `git add -A` at the root, never `git clean`) →
  `configs/mps-10min.sh <commit7> > run.log 2>&1` **as a background Bash call (`run_in_background`), then wait for
  its completion notification** — the foreground Bash limit is 10 minutes and a run takes ~12 → `grep -E
  "^(val_loss|val_loss_1loop|peak_memory_gb|tok_per_s):" run.log` → log the row → decide: `val_loss ≥ best −
  threshold` → `discard` (`git reset --hard <previous commit>`); `val_loss < best − threshold` → **candidate**: run it
  once more with another `--seed`, log that row too, **keep** only if the mean of the two beats `best` by the
  threshold and the change is not ugly (`best` is then that mean), otherwise `noise` and reset. A keep costs two runs,
  a discard one. Every ~8 experiments re-run the current best with a new seed to refresh `s` and `best`. Flag a
  `val_loss_1loop` regression larger than the threshold in the description. Empty grep =
  crash: `tail -n 50 run.log`, fix if trivial, else log `crash`. **Discard regardless of loss** when `peak_memory_gb
  > 8` or `tok_per_s` is 30% under the baseline (swapping). A run past 25 minutes is dead (`timeout` kills it).
- **Ideas, in order**: an lr check (3e-3 vs 6e-3 vs 1.5e-3), then ideas.md 6, 8, 1, 7, 9, then combinations of the
  keeps, then the agent's own; each `results.tsv` description names the idea. Simplicity criterion as in Karpathy's
  text. Caveat: an idea that changes `loop_max` or adds trajectory lengths compiles new graphs inside the budget
  (the warm-up excludes only the first 10 steps). Never stop; the program is re-injected via `CLAUDE.md` after a
  compaction; then re-read `results.tsv`, `git log --oneline loopformer..HEAD` and the last `run.log`.
- `README.md`: one paragraph "Autoresearch" pointing to `autoresearch.md` and `configs/mps-10min.sh`.

### 5. `loopformer/CLAUDE.md`
Names the conflict explicitly: the repository root `CLAUDE.md` is the CS336 course policy for the assignment code
(`cs336_basics/`, `tests/`); `loopformer/` is the owner's side project, not coursework; here the agent writes the
Python, runs `env -u VIRTUAL_ENV uv run ...`, commits, and runs the autonomous loop. Ends with `@autoresearch.md` so
the program is loaded at session start and after every compaction (the same import syntax as `~/.claude/CLAUDE.md`).
The session must start with cwd = `loopformer/`.

### 6. Housekeeping (first thing after approval; the user asked for it)
Commit the user's `.claude/settings.json` change (enables the playbooks plugin) on the branch as a `chore`:
`version.py` otherwise records every run as `-dirty`, and the loop's first `git reset --hard` would discard it.

## Tests (behaviour level, `loopformer/tests/`)
- `--compile none` memorizes the one-sample dataset (add to the `["whole", "loop"]` parametrization,
  `test_training.py:30`). MPS-only, skipped elsewhere: eager vs `compile_for_device("mps", "whole")` logits and one
  gradient agree (`assert_close`, atol 1e-3) on the tiny model.
- A budgeted run (`--time_budget_minutes 0.02`, `--iter 100000`, `--checkpoints_iter 0`) completes more than 10 steps,
  reports `1.2 ≤ training_seconds < 2.2`, and its last logged `lr` is `lr_min` (the cosine ran on the training clock).
- Two runs with the same `--seed` log the same first training loss; different seeds differ.
- `--final_val_batches 1` on a 5-window validation set: `val_loss` equals `validation_loss` over the first window pair
  and differs from the full-set value; `final_val_tokens == batch × ctx`.
- `summary["val_loss_1loop"]` equals the one-loop loss `loss_on_the_sample` computes; `summary["steps"] == 200`.
- `--checkpoints_iter 0` leaves `checkpoints/` empty while `val_loss` is still reported.
- A diverging run exits with code 1: monkeypatch `train.training_step` to return NaN losses, `pytest.raises(SystemExit)`,
  `FAIL` in the output.
- Existing tests keep passing: `overfit_args` saves via `--checkpoints_iter 200 == --iter`, and the two full-route
  final-validation tests rely on `--final_val_batches` defaulting to the full set.

## Milestones (one commit each)
1. `feat(loopformer): compile with inductor on MPS, add the none scope` — `model.py`, help text, tests.
2. `feat(loopformer): make a run autoresearch-ready` — training clock, `--seed`, `--final_val_batches`, summary,
   fail-fast, `--checkpoints_iter 0`, MPS memory; `run.py`, `data.py` generator, `conftest.py`; tests; config
   comments.
3. `feat(loopformer): the overnight autoresearch program` — `configs/mps-10min.sh`, `autoresearch.md`, `CLAUDE.md`,
   `.gitignore`, README paragraph; plus the `chore` commit of `.claude/settings.json`.

## Verification (the user runs these; ~20 minutes)
1. `cd loopformer && env -u VIRTUAL_ENV uv run pytest -q` — all green (the MPS test runs here).
2. Inductor correctness on MPS, same seed, 150 steps each:
   `configs/mps-10min.sh check-none --compile none --iter 150 --time_budget_minutes 100 --seed 1 > none.log 2>&1` and
   the same with `--compile whole` into `whole.log`; `grep -E "^val_loss:" none.log whole.log` agree within the
   same-seed noise (~0.03 on Apple Silicon per the MLX fork; a gap of 0.1 or more means the compiled graph computes
   something else) and the `train_loss` rows in the two `metrics.jsonl` track each other.
3. One short real run through the tool path the agent will use (checks the `rtk` Bash hook keeps the redirection):
   `configs/mps-10min.sh smoke --time_budget_minutes 1 > run.log 2>&1; grep -E "^(val_loss|training_seconds|
   tok_per_s|peak_memory_gb):" run.log` — summary present, `tok_per_s` ≈ 12K, memory ≈ 6.5 GB.
4. Start the night: close memory-heavy apps; `caffeinate -i` in a second terminal; a **fresh** Claude Code session
   started in `loopformer/` (auto mode is the user's default), prompt: "Read autoresearch.md and set up a new
   experiment run; after setup, start the loop." Two baselines ≈ 25 minutes, then ~5 experiments an hour (~35 by
   morning).

## Out of scope tonight
Running the ideas on the A100 (the Mac "keeps" are candidates for that), the B200 rerun, Karpathy's nanochat tricks
(value embeddings, polar-express Muon, warmdown), bits-per-byte.
