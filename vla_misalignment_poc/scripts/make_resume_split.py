#!/usr/bin/env python
"""
Build a resume split containing only the frames that have NOT been extracted yet.

Used when consolidating work onto one GPU: scans the existing shard outputs, keeps every
sample_id that already has a record AND a readable hidden-state file, and writes the
remainder as a new split.

Frames are emitted in (clip, frame) order and the runner walks each clip from its start
anyway (for temporal-memory fidelity), so resuming mid-clip is safe — it simply replays
that clip's warm-up frames.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def done_ids(shard_dirs: list[str], require_hidden: bool = True) -> set[str]:
    done: set[str] = set()
    for sd in shard_dirs:
        rp = os.path.join(sd, "records.jsonl")
        if not os.path.exists(rp):
            continue
        with open(rp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue          # a truncated final line from a killed process
                if require_hidden:
                    hf = r.get("hidden_file")
                    if not hf or not os.path.exists(os.path.join(sd, hf)):
                        continue
                done.add(r["sample_id"])
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+",
                    default=[os.path.join(POC_DIR, "splits", "poc_gpu0.json"),
                             os.path.join(POC_DIR, "splits", "poc_gpu1.json")])
    ap.add_argument("--shards", nargs="+",
                    default=[os.path.join(POC_DIR, "outputs", "shard_gpu0"),
                             os.path.join(POC_DIR, "outputs", "shard_gpu1")])
    ap.add_argument("--out", default=os.path.join(POC_DIR, "splits", "poc_resume.json"))
    args = ap.parse_args()

    wanted: list[dict] = []
    seen: set[str] = set()
    for sp in args.splits:
        with open(sp) as f:
            for r in json.load(f):
                if r["sample_id"] not in seen:
                    seen.add(r["sample_id"])
                    wanted.append(r)

    done = done_ids(args.shards)
    remaining = [r for r in wanted if r["sample_id"] not in done]
    remaining.sort(key=lambda r: (r["clip_id"], r["frame_idx"]))

    with open(args.out, "w") as f:
        json.dump(remaining, f)

    clips = Counter(r["clip_id"] for r in remaining)
    print(f"total requested : {len(wanted)}")
    print(f"already done    : {len(done)}")
    print(f"remaining       : {len(remaining)}  over {len(clips)} clips")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
