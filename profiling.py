"""Where the GPU memory goes: a snapshot of every live allocation at the peak of one training step, with the code
that made it (CUDA only). The .pickle opens in https://pytorch.org/memory_viz as a timeline of allocations."""
import os
import pickle
from collections import Counter

import torch

SNAPSHOT_FILE = "memory_snapshot.pickle"


def start_recording() -> None:
    torch.cuda.memory._record_memory_history(max_entries=200_000)


def dump_and_summarize(run_dir: str | os.PathLike, top: int = 15) -> None:
    """Writes the snapshot, stops recording and prints the largest groups of live memory at the peak, grouped by the
    nearest frame of this project's code (or the first frame when none is)."""
    path = os.path.join(run_dir, SNAPSHOT_FILE)
    torch.cuda.memory._dump_snapshot(path)
    torch.cuda.memory._record_memory_history(enabled=None)
    with open(path, "rb") as f:
        snapshot = pickle.load(f)
    by_frame: Counter[str] = Counter()
    for segment in snapshot["segments"]:
        for block in segment["blocks"]:
            if block["state"] == "active_allocated":
                by_frame[_origin(block.get("frames", []))] += block["size"]
    total = sum(by_frame.values())
    print(f"Memory snapshot: {total / 1e9:.1f} GB live at the end of the recorded step, by origin (full detail in {path}):")
    for origin, size in by_frame.most_common(top):
        print(f"  {size / 1e9:6.2f} GB  {origin}")


def _origin(frames: list[dict]) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for frame in frames:
        if frame["filename"].startswith(here):
            return f"{os.path.basename(frame['filename'])}:{frame['line']} {frame['name']}"
    if frames:
        return f"{os.path.basename(frames[0]['filename'])}:{frames[0]['line']} {frames[0]['name']}"
    return "unknown"
