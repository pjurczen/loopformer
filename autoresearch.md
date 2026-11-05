# autoresearch: the LoopFormer ideas, tested overnight on the Mac

An experiment in letting the agent do the research itself, after Karpathy's autoresearch (github.com/karpathy/autoresearch).
You change the code, train for a fixed 10 minutes on the Mac's GPU, read one number, keep the change if it is better,
reset if it is not, and repeat all night. The user is asleep; you are on your own until they come back.

The question: which of the ideas in `ideas.md` make the LoopFormer learn more in a fixed time? The reference numbers
are in `README.md`.

## Setup

Work with the user through these steps, then ask once for the go-ahead.

1. **Branch.** Propose a tag from today's date (for example `nov5`). The branch `autoresearch/<tag>` must not exist
   yet. `git checkout -b autoresearch/<tag>` from `main`, with a clean working tree.
2. **Read the in-scope files**, they are small: `autoresearch.md` (this), `ideas.md` (the ranked ideas, with sources),
   `README.md` (what was measured so far), `model.py`, `trajectory.py`, `train.py`, `configs/mps-10min.sh`.
3. **Check the data**: `data/tinystories-10k/train_ids.npy`, `data/tinystories-10k/valid_ids.npy` and
   `tokenizers/tinystories-10k/tokenizer_config.json` exist (links or copies, see README.md). If not, stop and tell
   the user.
4. **Check the machine**: the Mac has 16 GB shared by everything. Ask the user to close memory-heavy applications and to
   run `caffeinate -i` in a second terminal so the Mac does not sleep.
5. **results.tsv**: create it in the repository root with the header row only (tab-separated). It is untracked
   (`.gitignore`).
6. **The baseline, twice**: run the unchanged code with `--seed 1` and with `--seed 2` (two experiments, see "One
   experiment"). Their `val_loss` difference `s` is the seed noise; the decision threshold is `max(s, 0.01)`; `best` is
   their mean. Log both rows with status `keep` and descriptions `baseline seed 1` / `baseline seed 2`.
7. **Confirm** the plan with the user and go. From here on, do not ask again.

## What you may change

- `model.py`, `trajectory.py`, and `training_step` in `train.py`: the architecture, the conditioning, the routes, the
  loss terms. A new option gets a flag with a default equal to today's behaviour, and the flag goes into
  `EXPERIMENT_FLAGS` in `configs/mps-10min.sh`.
- The optimizer variables at the top of `configs/mps-10min.sh` (`LR_MAX`, `LR_MIN`, `WARMUP`, `WEIGHT_DECAY`, `BETA2`).

## What stays fixed

- `data.py`, `evaluate.py`, `tokenizer.py`, `optim.py`, `checkpoint.py`, `run.py`, the tests' support code.
- Every other flag in `configs/mps-10min.sh`: the dataset, context 256, batch 32, d_model 384, 6 heads, d_ff 1024,
  4 blocks × 2 loops as the starting shape, the 10-minute budget, the final validation on the first 64 batches.
  The time budget is the one equalizer: a change may add or remove parameters and compute, the budget decides whether
  it pays. `params_M` and `tok_per_s` go into the log so the trade shows.
- The metric: `val_loss`, the cross-entropy in nats along the full route (2 loops) over the validation set's first 64
  batches. `val_loss_1loop` is the same windows along the one-loop route (the elastic-depth number); log it, do not
  optimize it.
- No new dependencies.

## One experiment

1. Edit the code for one idea. One change per experiment, so the result has one cause.
2. Fast check before spending 12 minutes: `uv run pytest tests/test_model.py -q -x` (about 20 s;
   the Apple-GPU test in it compiles the tiny model with inductor, which catches most shape and codegen mistakes).
   Fix failures first. A test that no longer fits a changed behaviour may be changed with the behaviour, never deleted
   to make a run go.
3. Commit, naming only the files you changed: `git add model.py configs/mps-10min.sh && git commit -m "experiment:
   <what and why>"`. The run name is the commit: `commit=$(git rev-parse --short HEAD)`.
4. **Run**: `configs/mps-10min.sh "$commit" > run.log 2>&1`. A run takes about 12 minutes (20 s compile, 10 min
   training, ~1 min of validation), longer than the Bash tool's foreground limit, so **start it with the Bash tool's
   `run_in_background` option and wait for its completion notification**. Do not poll `run.log`, do not start a
   second run meanwhile, do not use the GPU for anything else while it runs. `timeout` inside the script kills a run
   after 25 minutes; a killed run is a `crash`.
5. **Read**: `grep -E "^(val_loss|val_loss_1loop|peak_memory_gb|tok_per_s|params_M|steps):" run.log`. Never read the
   whole log into the context. An empty grep means the run crashed: `grep -n "FAIL\|Error\|error" run.log | tail -5`
   and `tail -n 40 run.log`. A trivial cause (a typo, a shape) you fix and re-run under a new commit; a broken idea you
   log as `crash` and move on. `FAIL: training loss …` is a divergence: the same, with the learning rate as the first
   suspect.
6. **Log** one tab-separated row in `results.tsv` (commas break descriptions; tabs do not):
   `commit	val_loss	val_loss_1loop	memory_gb	tok_per_s	status	description`
   — the 7-character commit, the two losses with 4 decimals (0.0000 for a crash), `peak_memory_gb`, `tok_per_s`, the
   status, a short description that names the idea (`idea 6: prelude and coda blocks around 2x2`).
7. **Decide**, against `best` and the threshold:
   - `val_loss >= best - threshold` → status `discard`, `git reset --hard <previous commit>`.
   - `val_loss < best - threshold` → status `candidate`: run the same commit once more with another seed (append
     `--seed <n>` to the command; the config's default is unseeded), log that row too. If the mean of the two beats
     `best` by the threshold, and the change is not ugly, the second row is `keep`, `best` becomes that mean and the
     branch stays on the commit. Otherwise the second row is `noise` and you reset.
   - Regardless of the loss: `peak_memory_gb` above 8 or `tok_per_s` more than 30% under the baseline's means the
     machine was swapping; `discard` and say so in the description.
8. Every 8 experiments re-run the current best with a new seed, log it as `recheck`, and refresh `s`, the threshold
   and `best` from its two or three samples. The running minimum of noisy numbers drifts down on its own; this keeps
   the comparison honest.

**Simplicity criterion** (Karpathy's): all else equal, simpler is better. A gain of one threshold that adds twenty
ugly lines is not worth it; deleting code at equal loss is a win. Weigh the code against the gain before a `keep`.

## The ideas, in order

Start with the learning rate, because every later comparison inherits it: `LR_MAX` 6e-3 and 1.5e-3 against the
baseline's 3e-3 (`LR_MIN` a tenth of `LR_MAX`). Then the ideas of `ideas.md`, in this order: **6** prelude–loop–coda
(unique first and last block around a looped middle), **8** gate per token with the hidden state as input, **1**
reuse of the first loop's keys and values in later loops, **7** hyper-connections once per loop, **9** stability
(residual scaling, bounded injection) and the learning rate it allows. Then combinations of the keeps, then your own
ideas; `ideas.md` also lists what the literature says does not help, do not spend the night on those. Note that an
idea that changes `loop_max` or adds trajectory lengths compiles new graphs inside the 10 minutes (the warm-up only
excludes the first 10 steps) and loses a little time to that.

Within one idea, prefer the simplest version first and the version the source paper used second. An idea that costs
step time must win on loss at equal time, not at equal steps.

## Never stop

Once the loop has begun, do not pause to ask whether to continue. The user is asleep and expects results in the
morning. Out of ideas? Re-read `ideas.md` and the near-misses in `results.tsv`, combine, try a smaller or a larger
version, read the source paper's section named in `ideas.md`. The loop runs until the user interrupts it.

After a context compaction this file comes back through `CLAUDE.md`; then re-read `results.tsv`,
`git log --oneline main..HEAD` and the summary of the last `run.log` before the next experiment. The state of the
research is in those three places, never only in your memory.

## Git hygiene

Only the files you changed, named one by one. Never `git add -A` or `git add .`, never `git clean` (it would delete
`results.tsv`, `run.log` and the data links), never force-push, never touch `main`. `git reset --hard` resets the
whole worktree: that is intended for a discard.

## In the morning

Leave `results.tsv` and the branch as they are. The user reads the table, re-runs the keeps on the A100 if they look
real, and decides what goes into `README.md` and `ideas.md`.
