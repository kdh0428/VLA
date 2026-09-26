#!/usr/bin/env python
"""
Merge the per-GPU shard outputs into outputs/merged/.

Validates what the spec asks to check after inference:
  * duplicate sample_ids across shards (must be none -- shards are clip-disjoint)
  * frames present in the split but missing from the records (inference failures)
  * NaN / empty / wrong-shape trajectories
  * missing or unreadable hidden-state files, and layer-set consistency across frames

Hidden-state .npz files are left in place; the merged records carry an absolute path so
nothing is copied twice (they are the bulk of the output).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [warn] {path}:{ln} unparseable ({exc}) -- skipped")
    return out


def check_traj(t, expect_steps: int = 6) -> str | None:
    if t is None or len(t) == 0:
        return "empty"
    a = np.asarray(t, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        return f"shape{list(a.shape)}"
    if a.shape[0] != expect_steps:
        return f"steps{a.shape[0]}"
    if not np.isfinite(a).all():
        return "nonfinite"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    # shard_resume holds the frames re-run after consolidating onto a single GPU; it is
    # disjoint from the other two by construction (make_resume_split.py excludes anything
    # already recorded), and the duplicate check below re-verifies that.
    ap.add_argument("--shards", nargs="+",
                    default=[os.path.join(POC_DIR, "outputs", "shard_gpu0"),
                             os.path.join(POC_DIR, "outputs", "shard_gpu1"),
                             os.path.join(POC_DIR, "outputs", "shard_resume")])
    ap.add_argument("--splits", nargs="+",
                    default=[os.path.join(POC_DIR, "splits", "poc_gpu0.json"),
                             os.path.join(POC_DIR, "splits", "poc_gpu1.json")])
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs", "merged"))
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    merged: list[dict] = []
    seen: dict[str, str] = {}
    dup: list[str] = []
    per_shard: dict[str, int] = {}
    metas: dict[str, dict] = {}

    for sd in args.shards:
        recs = read_jsonl(os.path.join(sd, "records.jsonl"))
        per_shard[sd] = len(recs)
        mp = os.path.join(sd, "run_meta.json")
        if os.path.exists(mp):
            with open(mp) as f:
                metas[sd] = json.load(f)
        for r in recs:
            sid = r["sample_id"]
            if sid in seen:
                dup.append(sid)
                continue
            seen[sid] = sd
            if r.get("hidden_file"):
                r["hidden_path"] = os.path.abspath(os.path.join(sd, r["hidden_file"]))
            merged.append(r)

    merged.sort(key=lambda r: (r["clip_id"], r["frame_idx"]))

    # --- coverage vs the requested splits -------------------------------------------
    requested: set[str] = set()
    for sp in args.splits:
        if os.path.exists(sp):
            with open(sp) as f:
                requested |= {r["sample_id"] for r in json.load(f)}
    got = set(seen)
    missing = sorted(requested - got)

    # --- integrity ------------------------------------------------------------------
    traj_issues: dict[str, list[str]] = defaultdict(list)
    hidden_missing, hidden_bad = [], []
    layer_sets: Counter = Counter()
    hid_dims: Counter = Counter()

    for r in merged:
        for name, expect in (("trajectory_pred", 6), ("trajectory_gt", 6)):
            prob = check_traj(r.get(name), expect)
            if prob:
                traj_issues[name].append(f"{r['sample_id']}:{prob}")
        hp = r.get("hidden_path")
        if not hp:
            hidden_missing.append(r["sample_id"])
        elif not os.path.exists(hp):
            hidden_missing.append(r["sample_id"])
        else:
            try:
                with np.load(hp) as z:
                    keys = tuple(sorted(int(k) for k in z.files))
                    layer_sets[keys] += 1
                    hid_dims[z[z.files[0]].shape[-1]] += 1
            except Exception as exc:
                hidden_bad.append(f"{r['sample_id']}:{type(exc).__name__}:{exc}")

    out_path = os.path.join(args.output, "records.jsonl")
    with open(out_path, "w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")

    scen = Counter(r["scenario"] for r in merged)
    report = {
        "n_merged": len(merged),
        "per_shard": per_shard,
        "n_requested": len(requested),
        "n_missing": len(missing),
        "missing_sample_ids": missing[:100],
        "n_duplicates": len(dup),
        "duplicate_sample_ids": dup[:50],
        "trajectory_issues": {k: v[:30] for k, v in traj_issues.items()},
        "n_trajectory_issues": {k: len(v) for k, v in traj_issues.items()},
        "hidden_missing": hidden_missing[:50],
        "n_hidden_missing": len(hidden_missing),
        "hidden_unreadable": hidden_bad[:20],
        "layer_sets": {str(k[:4]) + f"...n={len(k)}": v for k, v in layer_sets.items()},
        "hidden_dims": dict(hid_dims),
        "n_clips": len({r["clip_id"] for r in merged}),
        "n_scenarios": len(scen),
        "per_scenario": dict(scen.most_common()),
        "shard_meta": metas,
    }
    with open(os.path.join(args.output, "merge_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"merged {len(merged)} records -> {out_path}")
    for sd, n in per_shard.items():
        print(f"  {os.path.basename(sd)}: {n}")
    print(f"  requested={len(requested)} missing={len(missing)} duplicates={len(dup)}")
    print(f"  trajectory issues: {report['n_trajectory_issues']}")
    print(f"  hidden missing={len(hidden_missing)} unreadable={len(hidden_bad)}")
    print(f"  layer sets: {report['layer_sets']}  dims: {report['hidden_dims']}")
    print(f"  clips={report['n_clips']} scenarios={report['n_scenarios']}")
    if len(layer_sets) > 1:
        print("  [warn] inconsistent layer sets across frames")


if __name__ == "__main__":
    main()
