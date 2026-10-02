#!/usr/bin/env python
"""
Official NAVSIM v2 submission pickles for each selection rule (run with PYTHONPATH=/root/VLA/navsim_v2).

Reads best-of-N decode records (expanded_best_of_n format) of the navhard subset, picks one candidate per
scene with each pre-registered rule (normal, max_loglik, min_entropy, ranksum; analyze_best_of_n.pick) and
converts its first 8 action tokens (4 s) to ego-frame poses [x, y, heading] with the codebook rollout of
models/action_tokenizer.py. Stage-1 tokens go to first_stage_predictions, synthetic tokens to
second_stage_predictions, as run_pdm_score_from_submission.py expects.

  PYTHONPATH=/root/VLA/navsim_v2 python scripts/navhard_submission.py <decode_run_dir> <out_dir> [--picks picks.json]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, "/root/VLA/autovla")
SUBSET = "/root/VLA/autovla_misalignment_poc/outputs/navhard_eval/subset"
RULES = ["normal", "max_loglik", "min_entropy", "ranksum"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--picks", default=None, help="extra picks JSON {token: {rule: candidate_index}}")
    args = ap.parse_args()
    import navsim
    assert "navsim_v2" in navsim.__file__, "run with PYTHONPATH=/root/VLA/navsim_v2"
    from navsim.common.dataclasses import Trajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from models.action_tokenizer import ActionTokenizer
    from analyze_best_of_n import pick
    cb = torch.tensor(np.asarray(pickle.load(open("/root/VLA/autovla/codebook_cache/agent_vocab.pkl", "rb"))["token_all"]["veh"]),
                      dtype=torch.float32)
    stage1 = set(json.load(open(f"{SUBSET}/stage1_tokens.json")))
    extra = json.load(open(args.picks)) if args.picks else {}
    recs = [json.loads(l) for l in open(os.path.join(args.run, "records.jsonl"))]
    rules = RULES + sorted({k for v in extra.values() for k in v} - set(RULES))
    sub = {r: {"first_stage_predictions": [{}], "second_stage_predictions": [{}]} for r in rules}
    ts = TrajectorySampling(num_poses=8, interval_length=0.5)
    for r in recs:
        r["prev_traj"] = None
        for rule in rules:
            i = extra[r["token"]][rule] if rule in extra.get(r["token"], {}) else pick(r, rule)
            idx = r["candidates"][i]["action_idx"][:8]
            poses = ActionTokenizer.rollout(None, cb[idx], time_steps=len(idx))[0, 1:].numpy().astype(np.float32)
            key = "first_stage_predictions" if r["token"] in stage1 else "second_stage_predictions"
            sub[rule][key][0][r["token"]] = Trajectory(poses, ts)
    os.makedirs(args.out, exist_ok=True)
    for rule, s in sub.items():
        pickle.dump(s, open(os.path.join(args.out, f"submission_{rule}.pkl"), "wb"))
    n1 = len(sub[RULES[0]]["first_stage_predictions"][0]); n2 = len(sub[RULES[0]]["second_stage_predictions"][0])
    print(f"rules {rules}: stage-1 {n1}/{len(stage1)} tokens, stage-2 {n2} tokens -> {args.out}")


if __name__ == "__main__":
    main()
