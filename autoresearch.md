# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `nov5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main, with a clean working tree.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context; `docs/results.md` — what was measured so far.
   - `ideas.md` — the ranked ideas to try, with their sources.
   - `model.py`, `trajectory.py`, `train.py` — the files you modify: model architecture, trajectories, training step.
   - `configs/mps-10min.sh` — the fixed experiment: the shape, the budget, the evaluation. You modify only its optimizer variables.
   - `data.py`, `evaluate.py`, `tokenizer.py`, `optim.py`, `checkpoint.py`, `run.py` — fixed. Do not modify.
4. **Verify data exists**: Check that `data/tinystories-10k/train_ids.npy`, `data/tinystories-10k/valid_ids.npy` and `tokenizers/tinystories-10k/tokenizer_config.json` exist. If not, tell the human (they are links or copies, see README.md).
5. **Verify the machine**: The Mac has 16 GB shared by everything. Ask the human to close memory-heavy applications and to run `caffeinate -i` in a second terminal so the Mac does not sleep.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first two runs.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on the Mac's GPU. The training script runs for a **fixed time budget of 10 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `configs/mps-10min.sh <run_name>`.

**What you CAN do:**
- Modify `model.py`, `trajectory.py` and `training_step` in `train.py` — the architecture, the conditioning, the routes, the loss terms. Change the code directly: a kept change is the new code, a discarded one is reset away. Do not add flags or options to switch an experiment on and off.
- Modify the optimizer variables at the top of `configs/mps-10min.sh` (`LR_MAX`, `LR_MIN`, `WARMUP`, `WEIGHT_DECAY`, `BETA2`).
- Append to `ideas.md` the ideas you source overnight, with their arXiv ids (see "Sourcing new ideas").

**What you CANNOT do:**
- Modify `data.py`, `evaluate.py`, `tokenizer.py`, `optim.py`, `checkpoint.py`, `run.py` or the tests' support code. They are read-only.
- Modify the other flags in `configs/mps-10min.sh`: the dataset, context 256, batch 32, d_model 384, 6 heads, d_ff 1024, 4 blocks × 2 loops as the starting shape, the 10-minute budget, the final validation on the first 64 batches.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `validation_loss` function in `evaluate.py` is the ground truth metric.

**The goal is simple: get the lowest val_loss.** Since the time budget is fixed, you don't need to worry about training time — it's always 10 minutes. A change may add or remove parameters and compute; the budget decides whether it pays. The only constraint is that the code runs without crashing and finishes within the time budget. `val_loss_1loop` (the same windows along the one-loop route) is logged, not optimized.

**Memory** is a soft constraint. The GPU shares the Mac's 16 GB with everything else. Some increase is acceptable for meaningful val_loss gains, but `peak_memory_gb` above 8, or `tok_per_s` more than 30% under the baseline's, means the machine was swapping: discard such a run regardless of its loss and say so in the description.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.01 val_loss improvement that adds 20 lines of hacky code? Probably not worth it. A 0.01 val_loss improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first runs**: Your very first two runs should always be to establish the baseline, so you will run the training script as is, once with `--seed 1` and once with `--seed 2`. Their val_loss difference `s` is the seed noise: the decision threshold is `max(s, 0.01)`, and `best` is their mean. Log both with status `keep`.

**The ideas, in order**: Start with the learning rate, because every later comparison inherits it: `LR_MAX` 6e-3 and 1.5e-3 against the baseline's 3e-3 (`LR_MIN` a tenth of `LR_MAX`). Then the ideas of `ideas.md` in this order: **6** prelude–loop–coda, **8** gate per token with the hidden state as input, **1** reuse of the first loop's keys and values, **7** hyper-connections once per loop, **9** stability and the learning rate it allows. Then combinations of the keeps, then your own ideas. `ideas.md` also lists what the literature says does not help; do not spend the night on those. Within one idea, prefer the simplest version first and the version the source paper used second. An idea that changes `loop_max` or adds trajectory lengths compiles new graphs inside the 10 minutes and loses a little time to that.

**Sourcing new ideas**: `ideas.md` is the start, not the limit. While a run trains (reading costs no GPU time), search for the next ideas: arXiv through the arxiv plugin (`search_papers` for looped, recurrent-depth, weight-tied or universal transformers and for the mechanism behind your best keep so far; `get_abstract` to screen; `list_paper_latex_sections` and `get_paper_latex_section` to read the method and the results, not the whole paper) and the web (`WebSearch`). Take an idea only if this experiment can see its effect: it must act on the training loss (not on inference cost), and fit the fixed shape, context, batch and budget. Prefer mechanisms that are not about capacity (routing, experts and halting are the known scale-dependent ones; `ideas.md` lists what the literature says does not help), and papers with results at a few hundred million parameters or less over billion-scale-only ones. Append each idea you take to `ideas.md` under a heading `## Sourced during autoresearch`, with the arXiv id, the section you read and the number it reports, and name the arXiv id in the results.tsv description, so the morning review can trace every experiment to its source.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_loss:           3.5192043259739876
val_loss_1loop:     3.534491416066885
final_val_tokens:   524288
training_seconds:   600.3
steps:              930
tokens_M:           7.62
tok_per_s:          12310
params_M:           13.8
peak_memory_gb:     5.43
```

Note that the script is configured to always stop after 10 minutes of training. You can extract the key metrics from the log file:

```
grep -E "^(val_loss|val_loss_1loop|peak_memory_gb|tok_per_s|params_M|steps):" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 7 columns:

```
commit	val_loss	val_loss_1loop	memory_gb	tok_per_s	status	description
```

1. git commit hash (short, 7 chars)
2. val_loss achieved (e.g. 3.1234) — use 0.0000 for crashes
3. val_loss_1loop (same format)
4. peak memory in GB (`peak_memory_gb`, e.g. 5.4) — use 0.0 for crashes
5. tok_per_s (e.g. 12310) — use 0 for crashes
6. status: `keep`, `discard`, `candidate`, `noise`, `recheck`, or `crash`
7. short text description of what this experiment tried, naming the idea

Example:

```
commit	val_loss	val_loss_1loop	memory_gb	tok_per_s	status	description
a1b2c3d	3.1200	3.1400	5.4	12310	keep	baseline seed 1
b2c3d4e	3.1120	3.1330	5.4	12290	keep	baseline seed 2
c3d4e5f	3.0900	3.1100	5.5	12100	candidate	idea 6: prelude and coda blocks around 2x2
c3d4e5f	3.0950	3.1150	5.5	12080	keep	idea 6: prelude and coda blocks around 2x2 (seed 3, confirms)
d4e5f6g	3.1300	3.1500	5.6	11700	discard	idea 8: gate from the hidden state per token
e5f6g7h	0.0000	0.0000	0.0	0	crash	idea 7: hyper-connections (shape mismatch in the residual)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/nov5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune the code with an experimental idea by directly hacking the code.
3. Fast check: `uv run pytest tests/test_model.py -q -x` (about 20 s; the Apple-GPU test compiles the tiny model and catches most shape and codegen mistakes). Fix failures first. A test may change with a changed behaviour, never be deleted to make a run go.
4. git commit, naming only the files you changed (e.g. `git add model.py configs/mps-10min.sh && git commit -m "experiment: <what and why>"`). Never `git add -A` or `git add .`. The run name is the commit: `commit=$(git rev-parse --short HEAD)`.
5. Run the experiment: `configs/mps-10min.sh "$commit" > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context). A run takes about 12 minutes, longer than the Bash tool's foreground limit, so start it with the Bash tool's `run_in_background` option and wait for its completion notification. Do not poll `run.log`, do not start a second run meanwhile, do not use the GPU for anything else while it runs.
6. Read out the results: `grep -E "^(val_loss|val_loss_1loop|peak_memory_gb|tok_per_s|params_M|steps):" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. `FAIL: training loss …` is a divergence: the learning rate is the first suspect. If you can't get things to work after more than a few attempts, give up.
8. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. If val_loss improved by more than the threshold (`val_loss < best - threshold`), the run is a `candidate`: run the same commit once more with another seed (append `--seed <n>` to the command) and log that row too. If the mean of the two beats `best` by the threshold, you "advance" the branch, keeping the git commit: the second row is `keep` and `best` becomes that mean. Otherwise the second row is `noise` and you git reset.
10. If val_loss is equal or worse (within the threshold or above), you git reset back to where you started (`git reset --hard <previous commit>`; it resets the whole worktree, which is intended)
11. Every 8 experiments, re-run the current best with a new seed, log it as `recheck`, and refresh `s`, the threshold and `best` from its samples. The running minimum of noisy numbers drifts down on its own; this keeps the comparison honest.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

This loop is the whole workflow. Do not run the playbooks skills (brainstorming, writing-plans, executing-plans, finishing-branch) or write design or plan documents for an experiment: one change, one commit, one run.

**Timeout**: Each experiment should take ~12 minutes total (20 s of compilation, 10 minutes of training, about 1 minute of validation). `timeout` inside the script kills a run after 25 minutes; a killed run is a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**Git**: never `git clean` (it would delete `results.tsv`, `run.log` and the data links), never force-push, never touch `main`.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — search arXiv and the web as described under "Sourcing new ideas", re-read `ideas.md` and the near-misses in `results.tsv`, read the source paper's section named in `ideas.md`, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period. After a context compaction this file comes back through `CLAUDE.md`; then re-read `results.tsv`, `git log --oneline main..HEAD` and the summary of the last `run.log` before the next experiment — the state of the research is in those three places, never only in your memory.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~12 minutes then you can run approx 5/hour, for a total of about 35–40 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
