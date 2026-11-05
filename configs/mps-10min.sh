#!/usr/bin/env bash
# The overnight autoresearch experiment on the Mac's GPU (see autoresearch.md): TinyStories, 4 blocks x 2 loops at
# d_model 384, 10 minutes of training time, then the validation set's first 64 batches along the full route and the
# one-loop route. The agent edits only the two blocks marked below; the shape, batch, context, dataset, budget and
# evaluation are fixed, so every run is comparable with every other.
#   configs/mps-10min.sh <run_name> [extra train.py flags]      (from loopformer/; extra flags come last and override)
# The run ends with train.py's summary: grep -E "^(val_loss|val_loss_1loop|peak_memory_gb|tok_per_s):" run.log
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- optimizer: the agent may change these ----------------------------------------------------------------------
LR_MAX=3e-3        # one learning rate serves Muon (the block matrices) and AdamW (the rest)
LR_MIN=3e-4
WARMUP=100         # steps; a 10-minute run makes ~900
WEIGHT_DECAY=0.1
BETA2=0.999
# ---- experiment flags: the agent adds here the flags of the model options it introduces -------------------------
EXPERIMENT_FLAGS=""
# -----------------------------------------------------------------------------------------------------------------

run_name="${1:?usage: configs/mps-10min.sh <run_name> [extra train.py flags]}"
shift

export PYTHONUNBUFFERED=1              # run.log fills as the run goes
export TORCHINDUCTOR_COMPILE_THREADS=1 # inductor's default 10 compile workers each import torch; the Mac has 16 GB
timeout_cmd=$(command -v timeout || true) # coreutils' timeout (brew install coreutils): the hard stop at 25 minutes

# shellcheck disable=SC2086  # EXPERIMENT_FLAGS is a list of flags
exec ${timeout_cmd:+"$timeout_cmd" 1500} env -u VIRTUAL_ENV uv run train.py \
  --device mps --compile whole \
  --train_data ../data/tinystories-10k/train_ids.npy --val_data ../data/tinystories-10k/valid_ids.npy \
  --tokenizer_dir ../tokenizers/tinystories-10k \
  --loop_blocks 4 --loop_max 2 --model_d_model 384 --model_num_heads 6 --model_d_ff 1024 --model_context_length 256 \
  --batch_size 32 --optimizer_muon \
  --optimizer_lr_max "$LR_MAX" --optimizer_lr_min "$LR_MIN" --optimizer_warmup_iters "$WARMUP" \
  --optimizer_weight_decay "$WEIGHT_DECAY" --optimizer_beta2 "$BETA2" \
  --time_budget_minutes 10 --time_reserve_minutes 0 --iter 1000000 \
  --val_interval 1000000 --val_batches 16 --final_val_batches 64 --checkpoints_iter 0 --log_interval 50 \
  --runs_dir runs/autoresearch --run_name "$run_name" $EXPERIMENT_FLAGS "$@"
