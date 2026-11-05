"""One trained tiny model for the tests that need a trained run: 200 steps on a single sample, so every batch is that
sample and the model memorizes it."""
import pytest
from support import OVERFIT_STEPS, one_sample_dataset, overfit_args

from train import parse_args, training

OVERFIT_RUN_NAME = "overfit"


@pytest.fixture(scope="session")
def overfit_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("overfit")
    sample = one_sample_dataset(root / "train.npy")
    args = parse_args(overfit_args(root, OVERFIT_RUN_NAME))

    summary = training(args)

    return {"sample": sample, "run_dir": root / "runs" / OVERFIT_RUN_NAME, "args": args, "steps": OVERFIT_STEPS, "summary": summary}
