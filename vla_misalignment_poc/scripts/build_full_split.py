#!/usr/bin/env python
"""
Full-coverage ORION split: every clip in Chat-B2D val, uniform temporal stride, no
per-scenario quota.

`build_subset.py` exists to hit a ~1800-frame budget, so it applies scenario quotas and a
rare-label carve-out that deliberately skew the sample.  For a full run neither is wanted:
the point is even coverage of the whole split, so this takes every clip and a plain stride,
and reports the resulting label balance rather than engineering it.

Edge frames are still dropped: ORION's temporal memory needs a run-up, and the last frames
of a clip have no future trajectory to score against.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_subset import CHATB2D_VAL, HERE, scan_clip, summarise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--edge-margin", type=int, default=35)
    ap.add_argument("--chatb2d", default=CHATB2D_VAL)
    ap.add_argument("--out", default=os.path.join(HERE, "splits", "full_all.json"))
    args = ap.parse_args()

    clips = sorted(d for d in glob.glob(os.path.join(args.chatb2d, "*")) if os.path.isdir(d))
    print(f"scanning {len(clips)} clips ...")
    records = []
    for cd in clips:
        records.extend(scan_clip(cd))
    print(f"  {len(records)} annotated frames")

    by_clip = defaultdict(list)
    for r in records:
        by_clip[r["clip_id"]].append(r)

    chosen = []
    dropped = {}
    for clip, rs in sorted(by_clip.items()):
        rs.sort(key=lambda r: r["frame_idx"])
        lo0, hi0 = rs[0]["frame_idx"], rs[-1]["frame_idx"]
        # The shortest clip in this split is 67 frames, so a flat 35-frame margin at both
        # ends empties it.  Shrink the margin for short clips instead of dropping them:
        # full coverage means every clip contributes.
        m = min(args.edge_margin, max(1, (hi0 - lo0) // 3))
        pool = [r for r in rs if lo0 + m <= r["frame_idx"] <= hi0 - m]
        if m != args.edge_margin:
            dropped[clip] = m
        chosen.extend(pool[:: args.stride])
    if dropped:
        print(f"  reduced edge margin on {len(dropped)} short clip(s): {dropped}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(chosen, f)
    n_clip = len({r["clip_id"] for r in chosen})
    print(f"  selected {len(chosen)} frames / {n_clip} clips -> {args.out}")

    stats = {"config": vars(args), "n_selected": len(chosen), "n_clips": n_clip,
             "overall": summarise(chosen),
             "per_clip": {c: sum(1 for r in chosen if r["clip_id"] == c)
                          for c in sorted(by_clip)}}
    with open(args.out.replace(".json", "_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    o = stats["overall"]
    print(f"  traffic_light={o.get('traffic_light')}  pedestrian={o.get('pedestrian')}  "
          f"lead_vehicle={o.get('lead_vehicle')}")


if __name__ == "__main__":
    main()
