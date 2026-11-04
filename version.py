"""The code version a run records: the short git commit, with "-dirty" when the tree has uncommitted changes."""
import os
import subprocess


def git_commit() -> str | None:
    # Inside a training image there is no Git; the cloud runner passes the version in as $GIT_COMMIT.
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], check=False).returncode != 0
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if dirty else commit
