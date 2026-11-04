import json

import pytest
from support import LOOP_MAX, read_metrics

from evaluate import evaluate_budgets
from run import BUDGET_EVAL_FILE


def test_evaluation_reports_the_loss_at_every_budget_up_to_the_full_route(overfit_run):
    rows = evaluate_budgets(overfit_run["run_dir"], device="cpu")

    assert [row["loops"] for row in rows] == list(range(1, LOOP_MAX + 1))
    assert all(row["seconds_per_batch"] > 0 for row in rows)


def test_the_full_route_evaluation_matches_the_run_s_final_validation(overfit_run):
    final_val_loss = [row for row in read_metrics(overfit_run["run_dir"]) if "val_loss" in row][-1]["val_loss"]

    rows = evaluate_budgets(overfit_run["run_dir"], device="cpu")

    assert rows[-1]["loss"] == pytest.approx(final_val_loss, abs=1e-4)


def test_the_evaluation_is_written_next_to_the_run(overfit_run):
    rows = evaluate_budgets(overfit_run["run_dir"], device="cpu")

    with open(overfit_run["run_dir"] / BUDGET_EVAL_FILE) as f:
        assert json.load(f) == rows
