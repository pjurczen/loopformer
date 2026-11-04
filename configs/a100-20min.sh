#!/usr/bin/env bash
# Phase 1: one shape on an A100-80GB for 15 minutes of training (20 with the reserve for the final validation), at the
# learning rate of the equal-time reference, the plain 12L/768 QK-norm run at 4e-3 (validation loss 3.693).
#   configs/a100-20min.sh <loop_blocks> <loop_max> [extra train.py flags]
set -euo pipefail
source "$(dirname "$0")/common.sh" "$@"

submit A100-80GB "${RUN_NAME:-owt-a100-loop-${loop_blocks}x${loop_max}-20min}" \
  --optimizer_lr_max 4e-3 --optimizer_lr_min 4e-4 --batch_size 64 --time_budget_minutes 20 --time_reserve_minutes 5
