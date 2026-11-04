#!/usr/bin/env bash
# Phase 2: one shape on a B200 for 45 minutes, the leaderboard budget, with the plain model's batch 256 and learning
# rate 8e-3 (validation loss 3.191 for 12L/768 + QK-norm). The reserve of 2 minutes covers the final validation.
#   configs/b200-45min.sh <loop_blocks> <loop_max> [extra train.py flags]
set -euo pipefail
source "$(dirname "$0")/common.sh" "$@"

submit B200 "${RUN_NAME:-owt-b200-loop-${loop_blocks}x${loop_max}-45min}" \
  --optimizer_lr_max 8e-3 --optimizer_lr_min 8e-4 --batch_size 256 --time_budget_minutes 45 --time_reserve_minutes 2
