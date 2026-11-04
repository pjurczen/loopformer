"""Runs train.py and then evaluate.py on a Modal GPU. Run from inside loopformer/:

    GPU=B200 uv run modal run --detach modal_app.py::main --dataset owt-32k --run-name <run_name> --args "<train.py flags>"
    GPU=A100-80GB uv run modal run --detach modal_app.py::reevaluate --run-name <run_name>
    uv run modal run modal_app.py::check     # builds the image and prints the torch build, without a GPU

The GPU type comes from the GPU environment variable (A100-80GB, B200, B200+ for B200-or-B300 at the B200 price) and
is fixed before the app is built, so Modal's dashboard shows the GPU a run uses. Data and tokenizers live in the Modal
volume cs336-data (data/<dataset>/, tokenizers/<name>/), uploaded once with `modal volume put`. Finished runs are saved
there under runs/<run_name>/, with budget_eval.json from the per-loop evaluation; fetch them with
`uv run modal volume get cs336-data runs/<run_name> runs/`. --detach keeps a run going if this terminal closes. The
configs/ scripts hold the flags of the experiment's runs.

Only modal and os are imported at the top: everything else runs remotely in the image.
"""
import os

import modal

GPU = os.environ.get("GPU", "A100-80GB")
TIMEOUT_MINUTES = 75  # hard stop: a 45-minute run plus its per-loop evaluation
PROJECT_DIR = "/root/project"
VOLUME_DIR = "/vol"
RUNS_DIR = "runs"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("gcc", "libc6-dev")  # torch.compile's Triton compiles a small C helper at runtime
    .uv_sync(extra_options="--no-dev")  # the locked dependencies, PyPI's Linux torch built for CUDA 13
    # expandable_segments lets the allocator join scattered free memory, so a large batch doesn't fail on fragmentation.
    .env({"PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(".", PROJECT_DIR, ignore=~modal.FilePatternMatcher("*.py", "configs/**"))  # the scripts and the configs, nothing else
)
volume = modal.Volume.from_name("cs336-data", create_if_missing=True)
app = modal.App("loopformer", image=image)


@app.function(gpu=GPU, timeout=TIMEOUT_MINUTES * 60, volumes={VOLUME_DIR: volume}, secrets=[modal.Secret.from_name("wandb")])
def train(dataset: str, run_name: str, args: list[str], git_commit: str):
    os.chdir(PROJECT_DIR)
    copy_inputs(dataset)
    print_gpu()
    try:
        run_script("train.py", "--runs_dir", RUNS_DIR, "--run_name", run_name, *args, env=dict(os.environ, GIT_COMMIT=git_commit))
        evaluate(run_name)
    finally:
        save_runs()  # whatever the run produced, also when it failed


@app.function(gpu=GPU, timeout=TIMEOUT_MINUTES * 60, volumes={VOLUME_DIR: volume})
def evaluate_run(dataset: str, run_name: str):
    """The per-loop evaluation on its own, for a run already on the volume."""
    os.chdir(PROJECT_DIR)
    copy_inputs(dataset)
    print_gpu()
    copy_run(run_name)
    try:
        evaluate(run_name)
    finally:
        save_runs()


@app.function(timeout=10 * 60)
def check():
    import sys

    import torch

    print(f"python {sys.executable}, torch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"architectures: {torch._C._cuda_getArchFlags()}")  # a B200 needs sm_100
    print(f"project files: {sorted(os.listdir(PROJECT_DIR))}")


def run_dir(run_name: str) -> str:
    return os.path.join(RUNS_DIR, run_name)  # train.py's rule: <runs_dir>/<run_name>


def evaluate(run_name: str):
    run_script("evaluate.py", "--run_dir", run_dir(run_name), "--device", "cuda", "--bf16")


def copy_inputs(dataset: str):
    # Local copies: np.memmap reads small random pieces, which is slow over a network volume.
    import shutil

    shutil.copytree(f"{VOLUME_DIR}/data/{dataset}", f"data/{dataset}")
    shutil.copytree(f"{VOLUME_DIR}/tokenizers", "tokenizers")


def copy_run(run_name: str):
    import shutil

    shutil.copytree(os.path.join(VOLUME_DIR, run_dir(run_name)), run_dir(run_name))


def save_runs():
    import shutil

    if os.path.isdir(RUNS_DIR):
        shutil.copytree(RUNS_DIR, os.path.join(VOLUME_DIR, RUNS_DIR), dirs_exist_ok=True)
        volume.commit()


def print_gpu():
    import subprocess

    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], check=False)


def run_script(script: str, *args: str, env: dict | None = None):
    import subprocess
    import sys

    returncode = subprocess.run([sys.executable, script, *args], env=env, check=False).returncode
    if returncode != 0:
        raise RuntimeError(f"{script} exited with code {returncode}")


@app.local_entrypoint()
def main(dataset: str, run_name: str, args: str):
    import shlex

    from version import git_commit  # runs locally, where Git is available; the run records the version

    train.remote(dataset, run_name, shlex.split(args), git_commit() or "unknown")


@app.local_entrypoint()
def reevaluate(run_name: str, dataset: str = "owt-32k"):
    evaluate_run.remote(dataset, run_name)
