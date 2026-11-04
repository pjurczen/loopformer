#!/usr/bin/env bash
# A 3-minute 3x4 run on an A100-80GB: checks the whole path (image, data, training, checkpoint, per-loop evaluation,
# upload) for under a dollar.
set -euo pipefail

RUN_NAME=owt-a100-loop-3x4-smoke exec "$(dirname "$0")/a100-20min.sh" 3 4 --time_budget_minutes 3 --time_reserve_minutes 1 --val_interval 300 "$@"
