# loopformer/

This folder is the repository owner's side project, not CS336 coursework. The repository root `CLAUDE.md` is the
course's teaching policy for the assignment code (`cs336_basics/`, `tests/`) and does not apply here: in `loopformer/`
the agent writes the Python, runs the commands and makes the commits.

- Run everything from this folder: `env -u VIRTUAL_ENV uv run <script.py> ...`; tests: `env -u VIRTUAL_ENV uv run pytest`.
- Commit only paths under `loopformer/`, named one by one. Never `git add -A` at the repository root, never `git clean`.
- The autonomous experiment loop is `autoresearch.md`, included below.

@autoresearch.md
