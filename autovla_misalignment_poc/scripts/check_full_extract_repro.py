#!/usr/bin/env python
"""
Reproduction check of a regenerated full_extract run against the committed results (CPU only).

Reference numbers are read from git HEAD, so the check still works after the regenerated run
has overwritten the working-tree files:
  - natural_fast_mechanism/summary.json   per-arm n, A- count, step-0 token errors (N, C)
  - pra_comparison/outputs/summary.json   AutoVLA exclusions and P/R/A group counts
The regenerated records are relabelled with the same functions (natural_fast_mechanism.a_label,
pra_comparison) and compared count by count.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

POC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(POC)
PRA_DIR = os.path.join(REPO, "pra_comparison")
sys.path.insert(0, PRA_DIR)
import pra_labels as L                                  # noqa: E402
from run_pra_comparison import _pos_to_delta            # noqa: E402

N_ACT = 10
CFG = L.Config()


def git_json(path):
    return json.loads(subprocess.check_output(["git", "-C", REPO, "show", f"HEAD:{path}"]))


def a_label(traj, gt, horizon=10):
    s = L.Sample(model="", sample_id="", cluster="", scene=L.Scene(), desc_text="", desc_objs=[], rsn_text="",
                 decl_lon=None, decl_lat=None, pred_delta=_pos_to_delta(traj, horizon), gt_delta=_pos_to_delta(gt, horizon))
    return bool(L.label_A(s, CFG)[0])


def arm_counts(recs, src):
    """Same filtering as natural_fast_mechanism.load_source."""
    out = {"n": 0, "n_Aminus": 0, "n_Aminus_step0_token_wrong": 0, "n_Aplus_step0_token_wrong": 0}
    aminus = []
    for r in recs:
        arm = r["arms"][src]
        if src == "N" and (arm["cot_present"] or arm["runaway_action_tokens"]):
            continue
        if src == "C" and (arm["truncated"] or arm["runaway_action_tokens"]):
            continue
        traj, pred = arm["trajectory_pred"], arm["pred_action_idx"][:N_ACT]
        gt_traj, gt = r["trajectory_gt"], r["gt_action_idx"][:N_ACT]
        if len(traj) < N_ACT or len(pred) < N_ACT:
            continue
        ok = a_label(traj, gt_traj)
        out["n"] += 1
        wrong0 = pred[0] != gt[0]
        if ok:
            out["n_Aplus_step0_token_wrong"] += wrong0
        else:
            out["n_Aminus"] += 1
            out["n_Aminus_step0_token_wrong"] += wrong0
            aminus.append(r["token"])
    return out, aminus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(POC, "outputs/full_extract/records.jsonl"))
    ap.add_argument("--pra-out", default=None, help="dir with a regenerated pra_comparison summary.json")
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.records)]
    ref = git_json("autovla_misalignment_poc/outputs/natural_fast_mechanism/summary.json")
    ref_meta = git_json("autovla_misalignment_poc/outputs/full_extract/run_meta.json")

    rows, all_match = [], True

    def cmp(name, new, old):
        nonlocal all_match
        ok = new == old
        all_match &= ok
        rows.append(f"{'OK ' if ok else 'DIFF'} {name:48s} new={new!s:>8}  committed={old!s:>8}")

    cmp("records (n_ok)", len(recs), ref_meta["n_ok"])
    fork_straight = sum(not r["arms"]["N"]["cot_present"] for r in recs)
    cmp("arm N chooses ' straightforward'", fork_straight, 2747)
    res = {}
    for src in ("N", "C"):
        c, am = arm_counts(recs, src)
        res[src] = {"counts": c, "aminus_tokens": am}
        for k, v in c.items():
            cmp(f"arm {src}: {k}", v, ref[src][k])

    if args.pra_out:
        new = json.load(open(os.path.join(args.pra_out, "summary.json")))
        old = git_json("pra_comparison/outputs/summary.json")
        for k, v in old["autovla_excluded"].items():
            cmp(f"PRA AutoVLA excluded: {k}", new["autovla_excluded"].get(k), v)
        og, ng = old["results"]["headline"]["AutoVLA"]["groups_all8"], new["results"]["headline"]["AutoVLA"]["groups_all8"]
        for k in og:
            cmp(f"PRA AutoVLA {k}", ng[k]["count"], og[k]["count"])

    print("\n".join(rows))
    print("\nALL MATCH" if all_match else "\nMISMATCHES PRESENT")
    json.dump({"all_match": all_match, "rows": rows, "arms": res},
              open(os.path.join(os.path.dirname(args.records), "repro_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
