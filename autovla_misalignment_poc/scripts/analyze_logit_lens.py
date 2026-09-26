#!/usr/bin/env python
"""
Action logit lens: layer x action-generation-step analysis.

This is the analysis ORION could not support. AutoVLA emits its trajectory as ordinary LM
vocabulary tokens, so `lm_head` applied to any layer's hidden state yields a distribution
over the 2048 physical action tokens. For every (layer, step) we already recorded, during
inference, the GT action token's probability and rank, the argmax token, the margin, and
the entropy.

Reported here:
  * mean GT-token rank / probability as a layer x step grid, split by A+ and A-
  * the layer at which the argmax first locks onto the finally-generated token
    ("commitment depth") -- how early the model settles on its answer
  * the four error-emergence patterns from the spec, counted per failing sample:
        A  never-acquired      GT token never well-ranked at any layer
        B  late-overwrite      GT token good mid-stack, lost by the final layer
        C  propagation         first step correct, later steps degrade
        D  utilisation failure GT never leads, yet the model is confident in another token
  * autoregressive error propagation: P(error at step t | earlier steps correct) vs
    P(error at step t | an earlier step wrong)
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RANK_GOOD = 5          # GT token counts as "well ranked" inside the top-5
LATE_FRAC = 0.85       # layers at/after this fraction of depth are "late"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(POC_DIR, "outputs/gpu1/records.jsonl"))
    ap.add_argument("--taxonomy", default=os.path.join(POC_DIR, "outputs/taxonomy/taxonomy.json"))
    ap.add_argument("--outdir", default=os.path.join(POC_DIR, "outputs/logit_lens"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    A = {r["token"]: r["A"] for r in json.load(open(args.taxonomy))}
    R = [json.loads(l) for l in open(args.records) if l.strip()]
    R = [r for r in R if r.get("logit_lens")]
    print(f"{len(R)} records with logit lens")

    n_layers = len(R[0]["logit_lens"][0]["layers"])
    n_steps = max(len(r["logit_lens"]) for r in R)
    print(f"grid: {n_layers} layers x {n_steps} steps")

    grids = {}
    for tag in ("all", "A+", "A-"):
        rank = np.full((n_layers, n_steps), np.nan)
        prob = np.full((n_layers, n_steps), np.nan)
        istop = np.full((n_layers, n_steps), np.nan)
        cnt = np.zeros((n_layers, n_steps))
        acc_r = defaultdict(list)
        acc_p = defaultdict(list)
        acc_t = defaultdict(list)
        for r in R:
            a = A.get(r["token"])
            if tag == "A+" and a is not True:
                continue
            if tag == "A-" and a is not False:
                continue
            for st in r["logit_lens"]:
                s = st["step"]
                for L in st["layers"]:
                    acc_r[(L["layer"], s)].append(L["gt_rank"])
                    acc_p[(L["layer"], s)].append(L["gt_prob"])
                    acc_t[(L["layer"], s)].append(1.0 if L["top_is_gt"] else 0.0)
        for (l, s), v in acc_r.items():
            rank[l, s] = float(np.mean(v)); cnt[l, s] = len(v)
        for (l, s), v in acc_p.items():
            prob[l, s] = float(np.mean(v))
        for (l, s), v in acc_t.items():
            istop[l, s] = float(np.mean(v))
        grids[tag] = {"gt_rank": rank.tolist(), "gt_prob": prob.tolist(),
                      "top_is_gt": istop.tolist(), "count": cnt.tolist()}
        print(f"[{tag}] final-layer top_is_gt by step: "
              f"{np.round(istop[n_layers-1], 3).tolist()}")

    # ---- commitment depth: first layer whose argmax equals the generated token --------
    commit = {"A+": [], "A-": []}
    for r in R:
        a = A.get(r["token"])
        tag = "A+" if a is True else ("A-" if a is False else None)
        if tag is None:
            continue
        for st in r["logit_lens"]:
            first = None
            for L in st["layers"]:
                if L["top_is_generated"]:
                    first = L["layer"]
                    break
            if first is not None:
                commit[tag].append(first)
    commit_stats = {k: {"n": len(v), "mean": float(np.mean(v)) if v else float("nan"),
                        "median": float(np.median(v)) if v else float("nan"),
                        "normalised_median": float(np.median(v)) / (n_layers - 1) if v else float("nan")}
                    for k, v in commit.items()}
    print("commitment depth:", commit_stats)

    # ---- error-emergence patterns, per failing step ----------------------------------
    late0 = int(LATE_FRAC * (n_layers - 1))
    patterns = Counter()
    per_sample = []
    for r in R:
        if A.get(r["token"]) is not False:
            continue
        step_pat = []
        for st in r["logit_lens"]:
            layers = st["layers"]
            ranks = np.array([L["gt_rank"] for L in layers])
            final_ok = ranks[-1] < RANK_GOOD
            mid_ok = bool((ranks[:late0] < RANK_GOOD).any())
            ever_ok = bool((ranks < RANK_GOOD).any())
            top_prob_final = layers[-1]["top_prob"]
            if final_ok:
                p = "correct_here"
            elif mid_ok:
                p = "B_late_overwrite"
            elif not ever_ok and top_prob_final > 0.5:
                p = "D_confident_other"
            else:
                p = "A_never_acquired"
            step_pat.append(p)
            patterns[p] += 1
        # C: first step fine, a later step wrong
        if step_pat and step_pat[0] == "correct_here" and any(p != "correct_here" for p in step_pat[1:]):
            patterns["C_propagation_sample"] += 1
        per_sample.append({"token": r["token"], "steps": step_pat})

    # ---- autoregressive propagation ---------------------------------------------------
    tot_given_ok = Counter(); err_given_ok = Counter()
    tot_given_bad = Counter(); err_given_bad = Counter()
    for r in R:
        pred, gt = r.get("pred_action_idx") or [], r.get("gt_action_idx") or []
        n = min(len(pred), len(gt))
        prev_bad = False
        for t in range(n):
            wrong = pred[t] != gt[t]
            if t > 0:
                if prev_bad:
                    tot_given_bad[t] += 1
                    err_given_bad[t] += int(wrong)
                else:
                    tot_given_ok[t] += 1
                    err_given_ok[t] += int(wrong)
            prev_bad = prev_bad or wrong
    prop = {str(t): {
        "P(err|prev all correct)": (err_given_ok[t] / tot_given_ok[t]) if tot_given_ok[t] else None,
        "n_prev_ok": tot_given_ok[t],
        "P(err|some prev wrong)": (err_given_bad[t] / tot_given_bad[t]) if tot_given_bad[t] else None,
        "n_prev_bad": tot_given_bad[t],
    } for t in range(1, n_steps)}

    out = {"n_layers": n_layers, "n_steps": n_steps, "grids": grids,
           "commitment_depth": commit_stats,
           "error_patterns": dict(patterns.most_common()),
           "autoregressive_propagation": prop}
    with open(os.path.join(args.outdir, "logit_lens.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(args.outdir, "per_sample_patterns.json"), "w") as f:
        json.dump(per_sample, f, indent=1)

    print("\nerror patterns (failing samples, per step):")
    for k, v in patterns.most_common():
        print(f"  {k:24s} {v}")
    print("\nautoregressive propagation:")
    for t in range(1, min(n_steps, 6)):
        d = prop[str(t)]
        a1 = d["P(err|prev all correct)"]; a2 = d["P(err|some prev wrong)"]
        print(f"  step {t}: P(err|prev ok)={a1 if a1 is None else round(a1,3)} (n={d['n_prev_ok']})  "
              f"P(err|prev wrong)={a2 if a2 is None else round(a2,3)} (n={d['n_prev_bad']})")
    print(f"\nwrote -> {args.outdir}")


if __name__ == "__main__":
    main()
