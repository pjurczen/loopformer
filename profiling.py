"""Where the GPU memory goes: a snapshot of every allocation during one training step, with the code that made it
(CUDA only). The .pickle opens in https://pytorch.org/memory_viz as a timeline of allocations.

    uv run profiling.py runs/<run_name>/memory_snapshot.pickle    # the summary again, for a saved snapshot
"""
import os
import pickle
import sys
from collections import Counter

import torch

SNAPSHOT_FILE = "memory_snapshot.pickle"


def start_recording() -> None:
    torch.cuda.memory._record_memory_history(max_entries=200_000)


def dump_and_summarize(run_dir: str | os.PathLike, top: int = 15) -> None:
    """Writes the snapshot, stops recording and prints the summary."""
    path = os.path.join(run_dir, SNAPSHOT_FILE)
    torch.cuda.memory._dump_snapshot(path)
    torch.cuda.memory._record_memory_history(enabled=None)
    print(summarize(path, top))


def summarize(path: str | os.PathLike, top: int = 15, project_dir: str | None = None) -> str:
    """The live memory at the peak of the recorded step, grouped by the nearest frame of this project's code (or the
    first frame when none is). The snapshot's trace is replayed, so the peak is found whenever it happened."""
    with open(path, "rb") as f:
        snapshot = pickle.load(f)
    project_dir = project_dir or os.path.dirname(os.path.abspath(__file__))
    live: dict[int, tuple[int, str]] = {}  # address -> (size, origin)
    total = peak = 0
    at_peak: dict[int, tuple[int, str]] = {}
    for event in snapshot["device_traces"][0]:
        if event["action"] == "alloc":
            live[event["addr"]] = (event["size"], _origin(event.get("frames", []), project_dir))
            total += event["size"]
            if total > peak:
                peak, at_peak = total, dict(live)
        elif event["action"] == "free_completed" and event["addr"] in live:
            total -= live.pop(event["addr"])[0]
    by_origin: Counter[str] = Counter()
    for size, origin in at_peak.values():
        by_origin[origin] += size
    lines = [f"Memory snapshot: {peak / 1e9:.1f} GB live at the peak of the recorded step, by origin (full detail in {path}):"]
    lines += [f"  {size / 1e9:6.2f} GB  {origin}" for origin, size in by_origin.most_common(top)]
    return "\n".join(lines)


def _origin(frames: list[dict], project_dir: str) -> str:
    for frame in frames:
        if frame["filename"].startswith(project_dir):
            return f"{os.path.basename(frame['filename'])}:{frame['line']} {frame['name']}"
    if frames:
        return f"{os.path.basename(frames[0]['filename'])}:{frames[0]['line']} {frames[0]['name']}"
    return "unknown"


if __name__ == "__main__":
    print(summarize(sys.argv[1], project_dir="/root/project"))  # the frames name the paths inside the training image
