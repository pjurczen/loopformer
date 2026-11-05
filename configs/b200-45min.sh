#!/usr/bin/env bash
# Phase 2: one shape on a B200 for 45 minutes, the leaderboard budget, against the plain 12L/768 + QK-norm model's
# 3.191 (batch 256, learning rate 8e-3). Batch 192 rather than 256: phase 1 peaked at 46 GB at batch 64, so 256 would
# need about the B200's whole 180 GB. The learning rate is sqrt-scaled from 4e-3 at batch 64. The reserve of 2 minutes
# covers the final validation. Since 2026-11-05 the budget counts training time after the 10 warm-up steps (the compile
# no longer eats into it), so a run trains ~1.5 minutes longer than the README's phase-2 run did.
#   configs/b200-45min.sh <loop_blocks> <loop_max> [extra train.py flags]
set -euo pipefail
source "$(dirname "$0")/common.sh" "$@"

submit B200 "${RUN_NAME:-owt-b200-loop-${loop_blocks}x${loop_max}-45min}" \
  --optimizer_lr_max 6.9e-3 --optimizer_lr_min 6.9e-4 --batch_size 192 --time_budget_minutes 45 --time_reserve_minutes 2
