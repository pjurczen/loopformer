# LoopFormer

A self-contained PyTorch project; `uv` manages its environment. Here the agent writes the code, runs the commands and
makes the commits.

- Run everything from the repository root: `uv run <script.py> ...`; tests: `uv run pytest`; lint: `uv run ruff check .`.
- Commit only the files you changed, named one by one. Never `git add -A` or `git add .`, never `git clean` (it would
  delete the data links, `results.tsv` and the run logs).
- The autonomous experiment loop is `autoresearch.md`, included below.

@autoresearch.md
