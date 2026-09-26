#!/usr/bin/env python
"""
Build the PoC frame subset and the deterministic GPU shards.

Sampling policy (NO random frame sampling — see section 3 of the research spec):
  1. Temporal subsampling inside each clip (stride) so adjacent, near-duplicate frames
     do not both enter the set.
  2. Scenario-aware stratification: every scenario type gets a quota proportional to its
     clip count, floored at MIN_PER_SCENARIO so rare scenarios survive.
  3. Rare-label preservation: pedestrian-hazard frames (~497 in the whole val split) and
     traffic-light frames are force-included, bypassing the stride.
  4. Clip-aware sharding: a clip is assigned WHOLE to exactly one GPU shard, so no clip is
     split across shards and later probe splits can be grouped by clip without leakage.

Shard balance is by frame count, using a greedy longest-processing-time assignment over
clips, which keeps the two GPUs busy for roughly equal wall-clock.

Outputs
    splits/poc_all.json    every selected frame with its labels
    splits/poc_gpu0.json   shard A
    splits/poc_gpu1.json   shard B
    splits/subset_stats.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.chatb2d_parser import build_perception_labels, build_reasoning_labels, load_annotation

CHATB2D_VAL = "/root/VLA/orion/data/chat-B2D/val"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def scenario_of(clip: str) -> str:
    return clip.split("_Town")[0]


def scan_clip(clip_dir: str) -> list[dict]:
    """Parse every annotated frame of one clip into a record with its GT labels."""
    clip = os.path.basename(clip_dir)
    recs = []
    for path in sorted(glob.glob(os.path.join(clip_dir, "*.json"))):
        frame = int(os.path.basename(path)[:-5])
        try:
            rounds = load_annotation(path)
        except Exception as exc:                       # corrupt json -> skip, keep going
            print(f"  WARN unreadable {path}: {exc}", file=sys.stderr)
            continue
        P, objs = build_perception_labels(rounds)
        R = build_reasoning_labels(rounds)
        recs.append({
            "sample_id": f"{clip}_{frame}",
            "clip_id": clip,
            "frame_idx": frame,
            "scenario": scenario_of(clip),
            "gt_perception": P.to_dict(),
            "gt_reasoning": R.to_dict(),
            "gt_rounds": rounds,
        })
    return recs


def is_rare(rec: dict) -> bool:
    """
    Frames carrying a scarce label that must not be thinned out by the stride.

    Pedestrian hazards only (~497 / 12,806 = 3.9% of the val split). Traffic lights are
    deliberately NOT force-included: red+green cover ~27% of frames, so preserving them
    all would invert the natural class balance (red would outnumber 'none') and make the
    traffic-light probe trivially separable — an artefact of sampling, not of the model.
    Traffic-light frames are instead left to the normal stride, which preserves their
    natural prior.
    """
    return bool(rec["gt_perception"].get("pedestrian"))


def select(records: list[dict], target: int, stride: int, min_per_scenario: int,
           edge_margin: int) -> list[dict]:
    """Stratified, temporally-subsampled selection over all clips."""
    by_scen: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_scen[r["scenario"]].append(r)

    # Frames at a clip's start/end have padded (zeroed) future trajectories -- the dataset
    # zero-fills ego_fut_trajs when the 3 s horizon runs past the clip. Drop them so A
    # labels are never computed against padding.
    max_frame = {}
    for r in records:
        max_frame[r["clip_id"]] = max(max_frame.get(r["clip_id"], 0), r["frame_idx"])

    def usable(r: dict) -> bool:
        return edge_margin <= r["frame_idx"] <= max_frame[r["clip_id"]] - edge_margin

    total_usable = sum(1 for r in records if usable(r))
    if total_usable == 0:
        raise SystemExit("no usable frames after edge filtering")

    quotas = {}
    for scen, rs in by_scen.items():
        n_ok = sum(1 for r in rs if usable(r))
        quotas[scen] = max(min_per_scenario, round(target * n_ok / total_usable))

    chosen: list[dict] = []
    for scen, rs in sorted(by_scen.items()):
        pool = [r for r in rs if usable(r)]
        pool.sort(key=lambda r: (r["clip_id"], r["frame_idx"]))
        if not pool:
            continue
        # Rare-label frames first (all of them), then a temporal stride over the rest.
        picked = {r["sample_id"]: r for r in pool if is_rare(r)}
        for r in pool[::stride]:
            picked.setdefault(r["sample_id"], r)

        want = quotas[scen]
        if len(picked) > want:
            # Trim, but protect rare frames: drop non-rare ones evenly across the clip
            # rather than truncating the tail (which would bias toward clip beginnings).
            rare = [r for r in picked.values() if is_rare(r)]
            common = sorted((r for r in picked.values() if not is_rare(r)),
                            key=lambda r: (r["clip_id"], r["frame_idx"]))
            room = max(0, want - len(rare))
            if room and common:
                step = max(1, len(common) / room)
                common = [common[int(i * step)] for i in range(min(room, len(common)))]
            else:
                common = []
            sel = rare + common
        else:
            sel = list(picked.values())
        chosen.extend(sel)

    chosen.sort(key=lambda r: (r["clip_id"], r["frame_idx"]))
    return chosen


def shard_by_clip(records: list[dict], n_shards: int = 2) -> list[list[dict]]:
    """Assign whole clips to shards, greedily balancing frame counts (LPT)."""
    per_clip: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        per_clip[r["clip_id"]].append(r)
    # Largest clips first so the greedy assignment balances well.
    order = sorted(per_clip.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    shards: list[list[dict]] = [[] for _ in range(n_shards)]
    for _, rs in order:
        target = min(range(n_shards), key=lambda i: (len(shards[i]), i))
        shards[target].extend(rs)
    for s in shards:
        s.sort(key=lambda r: (r["clip_id"], r["frame_idx"]))
    return shards


def summarise(records: list[dict]) -> dict:
    tl = Counter(r["gt_perception"]["traffic_light"] for r in records)
    return {
        "n_frames": len(records),
        "n_clips": len({r["clip_id"] for r in records}),
        "n_scenarios": len({r["scenario"] for r in records}),
        "traffic_light": dict(tl),
        "lead_vehicle": sum(1 for r in records if r["gt_perception"]["lead_vehicle"]),
        "pedestrian": sum(1 for r in records if r["gt_perception"]["pedestrian"]),
        "critical_side": dict(Counter(r["gt_perception"]["critical_side"] for r in records)),
        "critical_motion": dict(Counter(r["gt_perception"]["critical_motion"] for r in records)),
        "reasoning_lon": dict(Counter(r["gt_reasoning"]["intent_lon"] for r in records)),
        "per_scenario": dict(Counter(r["scenario"] for r in records)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1800, help="approximate frame budget")
    ap.add_argument("--stride", type=int, default=8, help="temporal subsample stride")
    ap.add_argument("--min-per-scenario", type=int, default=12)
    ap.add_argument("--edge-margin", type=int, default=35,
                    help="drop frames this close to a clip start/end (padded GT horizon)")
    ap.add_argument("--chatb2d", default=CHATB2D_VAL)
    ap.add_argument("--outdir", default=os.path.join(HERE, "splits"))
    args = ap.parse_args()

    clips = sorted(d for d in glob.glob(os.path.join(args.chatb2d, "*")) if os.path.isdir(d))
    print(f"scanning {len(clips)} clips ...")
    records: list[dict] = []
    for cd in clips:
        records.extend(scan_clip(cd))
    print(f"  {len(records)} annotated frames total")

    chosen = select(records, args.target, args.stride, args.min_per_scenario, args.edge_margin)
    print(f"  selected {len(chosen)} frames")

    shards = shard_by_clip(chosen, 2)
    os.makedirs(args.outdir, exist_ok=True)

    with open(os.path.join(args.outdir, "poc_all.json"), "w") as f:
        json.dump(chosen, f)
    for i, s in enumerate(shards):
        with open(os.path.join(args.outdir, f"poc_gpu{i}.json"), "w") as f:
            json.dump(s, f)
        print(f"  shard gpu{i}: {len(s)} frames / {len({r['clip_id'] for r in s})} clips")

    # Hard guarantee: no clip and no frame appears in both shards.
    c0 = {r["clip_id"] for r in shards[0]}
    c1 = {r["clip_id"] for r in shards[1]}
    assert not (c0 & c1), f"clip overlap between shards: {c0 & c1}"
    ids = [r["sample_id"] for s in shards for r in s]
    assert len(ids) == len(set(ids)), "duplicate sample_id across shards"

    stats = {
        "config": vars(args),
        "overall": summarise(chosen),
        "shard_gpu0": summarise(shards[0]),
        "shard_gpu1": summarise(shards[1]),
        "source_total_frames": len(records),
    }
    with open(os.path.join(args.outdir, "subset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    o = stats["overall"]
    print(f"\n=== subset: {o['n_frames']} frames, {o['n_clips']} clips, "
          f"{o['n_scenarios']} scenarios ===")
    print("  traffic_light :", o["traffic_light"])
    print("  lead_vehicle  :", o["lead_vehicle"])
    print("  pedestrian    :", o["pedestrian"])
    print("  critical_side :", o["critical_side"])
    print("  reasoning_lon :", o["reasoning_lon"])
    print("  no clip overlap between shards: OK")


if __name__ == "__main__":
    main()
