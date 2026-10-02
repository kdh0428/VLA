#!/usr/bin/env python
"""
Choose a navhard_two_stage subset that fits the disk and write its split configs + file lists (CPU only).

Groups (one stage-1 pair [orig t0, orig t-0.5 s] + its synthetic stage-2 pairs) are selected by whole log:
all logs whose original test images are already on disk first, then further logs in a seeded random order
until the synthetic images (CAM_F0 / CAM_L1 / CAM_R1, the cameras AutoVLA reads) reach --budget-gb.

Writes
  <navsim_v2>/.../train_test_split/navhard_half.yaml and scene_filter/navhard_half.yaml   (official format)
  <out>/groups.json, stage1_tokens.json, synthetic_tokens.json
  <out>/orig_files_missing.txt   original-scene images to fetch from navtest camera shards (by shard)
  <out>/synthetic_files.txt      synthetic images to extract from the navhard sensor archives
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pickle
import random

import yaml


class _Quoted(yaml.SafeDumper):
    """Quote every string: hex tokens such as '3e1234...' would otherwise be read as floats by OmegaConf."""


_Quoted.add_representer(str, lambda d, v: d.represent_scalar("tag:yaml.org,2002:str", v, style="'"))

V2 = "/root/VLA/navsim_v2/navsim/planning/script/config/common/train_test_split"
NAVHARD = "/root/VLA/navhard/navhard_two_stage"
LOGS = "/root/VLA/autovla/dataset/nuplan/navsim_logs/test"
SENS = "/root/VLA/autovla/dataset/nuplan/sensor_blobs/test"
CAMS = ("CAM_F0", "CAM_L1", "CAM_R1")
MB_PER_IMG = 0.345


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/VLA/autovla_misalignment_poc/outputs/navhard_eval/subset")
    ap.add_argument("--budget-gb", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-name", default="navhard_half", help="name of the written train_test_split / scene_filter configs")
    ap.add_argument("--logs", default=None, help="JSON list of logs: reuse a fixed selection (disk state then does not matter)")
    ap.add_argument("--shards", default="/tmp/claude-0/-root-VLA/3d5b895f-ebec-4409-a7c0-03065c2dc74b/scratchpad/shards.json")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    split = yaml.safe_load(open(f"{V2}/navhard_two_stage.yaml"))
    sf = yaml.safe_load(open(f"{V2}/scene_filter/navhard_two_stage.yaml"))
    by_scene = {r["synthetic_scene_token"]: r for r in csv.DictReader(open(f"{NAVHARD}/synthetic_scenes_attributes.csv"))}
    # configs use each synthetic scene's metadata initial_token; files / csv use scene_token
    init2scene = {}
    for fn in os.listdir(f"{NAVHARD}/synthetic_scene_pickles"):
        md = pickle.load(open(f"{NAVHARD}/synthetic_scene_pickles/{fn}", "rb"))["scene_metadata"]
        init2scene[md["initial_token"]] = md["scene_token"]
    rows = {i: by_scene[s] for i, s in init2scene.items()}
    json.dump(init2scene, open(f"{args.out}/synthetic_initial_to_scene_token.json", "w"))
    tok_log, frames_of = {}, {}
    for lg in sf["log_names"]:
        fr = pickle.load(open(f"{LOGS}/{lg}.pkl", "rb"))
        pos = {f["token"]: i for i, f in enumerate(fr)}
        for t in sf["tokens"]:
            if t in pos:
                tok_log[t] = lg
                frames_of[t] = [fr[j]["cams"][c]["data_path"] for j in range(pos[t] - 3, pos[t] + 1) for c in CAMS]
    groups = []
    for orig, prev, pairs in split["reactive_all_mapping"]:
        syn = [t for p in pairs for t in p]
        files = {rows[t][f"frame_{f}_{c.lower()}_path"] for t in syn for f in range(4) for c in CAMS}
        groups.append({"orig": orig, "prev": prev, "pairs": pairs, "log": tok_log[orig], "syn_files": files})
    by_log = collections.defaultdict(list)
    for g in groups:
        by_log[g["log"]].append(g)
    local = set(os.listdir(SENS))
    order = sorted(l for l in by_log if l in local)
    rest = sorted(l for l in by_log if l not in local)
    random.Random(args.seed).shuffle(rest)
    chosen, files = [], set()
    if args.logs:                                     # fixed selection
        chosen = json.load(open(args.logs))
        files = set().union(*(g["syn_files"] for lg in chosen for g in by_log[lg]))
    for lg in ([] if args.logs else order + rest):
        new = set().union(*(g["syn_files"] for g in by_log[lg]))
        if lg not in order and (len(files | new) * MB_PER_IMG / 1000) > args.budget_gb:
            continue
        chosen.append(lg); files |= new
    sel = [g for g in groups if g["log"] in set(chosen)]
    s1 = sorted({t for g in sel for t in (g["orig"], g["prev"])})
    s2 = sorted({t for g in sel for p in g["pairs"] for t in p})
    orig_files = sorted({f for t in s1 for f in frames_of[t]})
    missing = [f for f in orig_files if not os.path.exists(os.path.join(SENS, f))]
    shard_of_log = {l: k for k, v in json.load(open(args.shards)).items() for l in v}
    by_shard = collections.defaultdict(list)
    for f in missing:
        by_shard[shard_of_log[f.split("/")[0]]].append(f)
    # official-format split configs
    sf2 = dict(sf); sf2["log_names"] = sorted(chosen); sf2["tokens"] = s1; sf2["reactive_synthetic_initial_tokens"] = s2
    yaml.dump(sf2, open(f"{V2}/scene_filter/{args.split_name}.yaml", "w"), Dumper=_Quoted, sort_keys=False)
    split2 = {"defaults": [{"scene_filter": args.split_name}], "data_split": "test",
              "reactive_all_mapping": [[g["orig"], g["prev"], g["pairs"]] for g in sel]}
    yaml.dump(split2, open(f"{V2}/{args.split_name}.yaml", "w"), Dumper=_Quoted, sort_keys=False)
    json.dump([{k: g[k] for k in ("orig", "prev", "pairs", "log")} for g in sel], open(f"{args.out}/groups.json", "w"))
    json.dump(s1, open(f"{args.out}/stage1_tokens.json", "w")); json.dump(s2, open(f"{args.out}/synthetic_tokens.json", "w"))
    json.dump(by_shard, open(f"{args.out}/orig_files_missing_by_shard.json", "w"))
    open(f"{args.out}/synthetic_files.txt", "w").write("\n".join(sorted(files)) + "\n")
    print(f"groups {len(sel)}/{len(groups)}, logs {len(chosen)}/{len(by_log)} (local {len(order)}), stage-1 tokens {len(s1)}, "
          f"synthetic tokens {len(s2)}")
    print(f"synthetic images {len(files)} (~{len(files)*MB_PER_IMG/1000:.1f} GB); original images {len(orig_files)}, "
          f"missing {len(missing)} (~{len(missing)*MB_PER_IMG/1000:.2f} GB) in shards {sorted(by_shard, key=lambda s: int(s.split('_')[-1]))}")


if __name__ == "__main__":
    main()
