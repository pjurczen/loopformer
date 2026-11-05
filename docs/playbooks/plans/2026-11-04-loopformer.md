# LoopFormer experiment — implementation plan

Design: [2026-11-03-loopformer.md](../designs/2026-11-03-loopformer.md)

## Goal

Build the self-contained `loopformer/` project, train looped models on OpenWebText on Modal, and report equal-time, elastic-depth and parameter-efficiency results against the existing runs.

## Decomposition

- `model.py` — the copied components (`Linear`, `Embedding`, `RMSNorm`, RoPE, attention with QK-norm, SwiGLU), the (t, Δt) conditioning, the LoopFormer block and `LoopFormerLM`. Interface: `LoopFormerLM(config)`, `forward(tokens, schedule) -> logits, hidden`. Depends on torch, einops, jaxtyping.
- `trajectory.py` — the long route for L, the sampler for short routes, the evaluation schedule for M. Pure functions on lists of floats. No torch.
- `optim.py` — copied AdamW, Muon, the two cosine schedules and gradient clipping. Depends on torch.
- `data.py` — copied batch functions and cross-entropy. Depends on numpy, torch, einops.
- `checkpoint.py` — copied save and load with the Muon state, plus the newest-checkpoint lookup. Depends on torch.
- `config.py` — the run config model, a copy of `RunConfig` with the loop fields. Depends on pydantic.
- `tokenizer.py` — the copied encode and decode part of the BPE tokenizer, loaded from a tokenizer folder. Depends on regex.
- `train.py` — the training loop with the two-backward step, time budget, validation, metrics, checkpoints and wandb. Depends on all of the above except `tokenizer.py`.
- `evaluate.py` — the loss on the whole validation set at every loop count, written to `budget_eval.json`. Depends on `model.py`, `trajectory.py`, `data.py`, `checkpoint.py`, `config.py`.
- `decode.py` — text generation from a checkpoint at a chosen loop count or schedule. Depends on `model.py`, `trajectory.py`, `checkpoint.py`, `config.py`, `tokenizer.py`.
- `modal_app.py` — the cloud runner: image, volume, secret, `--gpu`, one function that runs `train.py` then `evaluate.py`. Depends on modal.
- `configs/` — one shell file per run shape that submits the run through `modal_app.py`.
- `tests/` — the six behavior tests of the design, on a tiny model.
- `README.md` — what it is, how to run it, the results.

## Behaviours to verify

1. Given one fixed batch and a tiny model, when training runs 200 steps, then the long trajectory's loss is below 0.05 and the loss at M = 1 is low too.
2. Given the overfit model, when `decode.py` continues the first tokens of that batch greedily, then it returns the rest exactly, at M = L and at M = 1.
3. Given any M, when one token at position j changes, then the logits before j do not change.
4. Given a fresh model, when one backward runs, then the attention and SwiGLU weights have zero gradient and the modulators, the embeddings and the final norm have non-zero gradient. After a few optimizer steps every parameter has a non-zero gradient.
5. Given a trained model, when it is saved and loaded into a fresh model, then the logits are identical for every M.
6. Given L, when many short routes are sampled, then each has 1 to L − 1 steps on the grid with a sum of 1. For each M the evaluation schedule has M steps on the grid, larger first.
7. Given a short local run on the Mac, when it ends, then the run folder holds `config.json` with the tokenizer path, k, L and the git commit, `metrics.jsonl` with `train_loss`, `short_loss`, `consistency_loss`, `tokens` and `val_loss`, and a checkpoint. The run stops at the budget minus the reserve and ends with a full validation.
8. Given a run folder, when `evaluate.py` runs, then `budget_eval.json` has one row per M from 1 to L with the loss and the time per batch.
9. Given a Modal submission, when the run ends, then `runs/<run_name>/` on the volume holds the files of behaviour 7 and `budget_eval.json`, and the per-loop evaluation ran in the same container after training.
10. Given a 3×4 run at batch 64 on the A100-80GB, when it ends, then `peak_memory_gb` is near one trajectory, about 38 GB, not two.

## Milestones

1. The model and its tests: the project skeleton (`pyproject.toml`, `uv.lock`, pytest), `model.py`, `trajectory.py`, `checkpoint.py`.
   Done when: `uv run pytest` inside `loopformer/` is green.
   Verified by: behaviours 3, 4, 5, 6.
2. The training loop: `optim.py`, `data.py`, `config.py`, `train.py`, the overfit test.
   Done when: a 2-minute run on the Mac with a tiny config writes the run folder of behaviour 7.
   Verified by: behaviours 1, 7; the test suite.
3. Evaluation and generation: `evaluate.py`, `tokenizer.py`, `decode.py`, the decode test.
   Done when: both scripts run on the milestone-2 checkpoint, and `decode.py` answers a prompt at M = L and at M = 1.
   Verified by: behaviours 2, 8; the test suite.
4. The cloud runner and the smoke run: `modal_app.py`, `configs/`, `README.md`.
   Done when: a 3-minute A100-80GB smoke run of the 3×4 shape ends with its files on the volume.
   Verified by: behaviour 9.
5. Phase 1: the four A100-80GB runs (3×4, 4×3, 6×2, 2×6), 20 minutes each with a 5-minute reserve, their `budget_eval.json` fetched, the results table and the per-loop curves in `README.md`.
   Done when: `README.md` compares each shape with 3.693 (equal time), shows loss against M, and compares 6×2 with 3.669 (parameters).
   Verified by: behaviours 9, 10; a run that diverges is run again with warmup 500 and learning rate 2e-3.
6. Phase 2: one B200 run of 45 minutes with the best shape at batch 256, learning rate 8e-3, floor 8e-4, reserve 2 minutes, its result in `README.md` against 3.191.
   Done when: the B200 result and its per-loop curve are in `README.md`.
   Verified by: behaviour 9; `peak_memory_gb` below 180 GB.

## Assumptions

- The Modal volume `cs336-data` holds `data/owt-32k/` and `tokenizers/owt-32k/`.
- The parent `.gitignore` ignores `runs/`, `data/` and `.venv` at any depth, so `loopformer/runs/` and `loopformer/.venv/` stay out of git.
- The parent `pyproject.toml` declares no uv workspace, so `loopformer/pyproject.toml` is an independent project.
- The copied files need einops, jaxtyping (type hints in the attention code), numpy, pydantic, regex and torch. `pyproject.toml` lists them, plus wandb, and modal and pytest as dev dependencies. torch is pinned `~=2.11.0`, the PyPI build with CUDA 13 on Linux.
- Modal's `uv_sync` builds the image from the project in the current directory, so Modal commands run from inside `loopformer/`.
- The equal-time reference is `runs/a100/owt-a100-80gb-12l-qknorm-lr-4e-3` (3.693), the parameter reference `runs/a100/owt-a100-ctx512-8l-d768-20min-wd-0.1` (3.669), the B200 reference 3.191.
- The per-loop evaluation runs the full validation set once per M. For L = 6 that is 3.5 full passes of a 12-layer model, about 15 minutes on the A100-80GB. The 75-minute hard stop covers the smoke run, the phase-1 runs and the B200 run.
- The optimizer split rule of `cs336_basics/train.py` selects 2-D parameters by name prefix, the block container. The modulators sit inside the blocks and go to Muon. The conditioning MLPs sit outside and go to AdamW, as the design says.

## Open questions

- None.
