#!/usr/bin/env bash
# Sourced by the phase scripts: the shape from the command line, and the settings every run has in common.
#   source configs/common.sh <loop_blocks> <loop_max> [extra train.py flags]
#   submit <gpu> <run_name> <phase flags>
# The common settings are those of the plain 12L/768 QK-norm runs the results compare with: d_model 768, 12 heads,
# d_ff 2048, context 512, Muon, bf16, warmup 120, weight decay 0.1, beta2 0.999. Extra flags come last, so they
# override the phase's.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

loop_blocks="$1"
loop_max="$2"
shift 2
extra_args="$*"

common_args="--device cuda --wandb_project cs336 \
 --train_data data/owt-32k/train_ids.npy --val_data data/owt-32k/valid_ids.npy --tokenizer_dir tokenizers/owt-32k \
 --loop_blocks $loop_blocks --loop_max $loop_max \
 --model_context_length 512 --model_d_model 768 --model_num_heads 12 --model_d_ff 2048 \
 --bf16 --optimizer_warmup_iters 120 --optimizer_weight_decay 0.1 --optimizer_beta2 0.999 \
 --iter 1000000 --val_interval 1000 --val_batches 16 --log_interval 100 --checkpoints_iter 1000000"

submit() {
  local gpu="$1" run_name="$2"
  shift 2
  GPU="$gpu" uv run modal run --detach modal_app.py::main --dataset owt-32k --run-name "$run_name" --args "$common_args $* $extra_args"
}
