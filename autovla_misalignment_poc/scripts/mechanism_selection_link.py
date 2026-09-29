#!/usr/bin/env python
"""
Does confidence-based selection reject rollouts caught in autoregressive feedback? (CPU only)

For every candidate of every scene in best-of-N runs (expanded / held-out format):
  t*                 first step whose token differs from the GT token (None = matches GT throughout)
  post_dev_entropy   mean decoding entropy over steps t*+1 .. 9 (the steps generated under the deviation)
  pre_dev_entropy    mean entropy over steps 0 .. t*
  err_growth         slope of the per-pose L2 error over poses t* .. 9 (m / step)
  realign            fraction of steps after t* whose token equals GT again
  ADE5 / FDE5, amplification = A- (P/R/A 5 s) and FDE5 > 3 m, a_minus = A-
  selector scores    mean entropy, summed log-prob, rank-sum rank (1 = selected by rank-sum)

Analyses (log-cluster bootstrap 95% CI throughout):
  1. selected (rank-sum pick) vs discarded (other candidates of the same scene): paired per-scene
     difference in post-deviation entropy, error growth, amplification rate, realignment, FDE
  2. candidate-level AUROC of each signal for amplification, pooled and within scene
     (scenes that contain both amplifying and non-amplifying candidates)
  3. mechanism: among candidates that deviate, how well post-deviation entropy and error growth separate
     amplifying from non-amplifying rollouts, and whether rank-sum's whole-plan entropy tracks them

  python scripts/mechanism_selection_link.py <out_dir> <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_action_history import AMP_FDE, a_eval   # noqa: E402

N_ACT = 10


def cand_stats(x, gt_tok, gt_xy):
    toks, ent, lp = x["action_idx"], np.asarray(x["entropy_steps"]), np.asarray(x["logprob_steps"])
    T = np.asarray(x["trajectory_pred"], float)
    err = np.linalg.norm(T - gt_xy, axis=1)
    t = next((k for k in range(N_ACT) if toks[k] != gt_tok[k]), None)
    ok = a_eval(T.tolist(), gt_xy.tolist())
    s = {"t_star": t, "ade5": float(err.mean()), "fde5": float(err[-1]), "a_minus": float(not ok),
         "amplification": float((not ok) and err[-1] > AMP_FDE),
         "ent_mean": float(ent.mean()), "lp_sum": float(lp.sum())}
    if t is not None and t < N_ACT - 1:
        s["post_dev_entropy"] = float(ent[t + 1:].mean())
        s["pre_dev_entropy"] = float(ent[:t + 1].mean())
        s["err_growth"] = float(np.polyfit(np.arange(t, N_ACT), err[t:], 1)[0])
        s["realign"] = float(np.mean([toks[k] == gt_tok[k] for k in range(t + 1, N_ACT)]))
    else:
        s["post_dev_entropy"] = s["pre_dev_entropy"] = s["err_growth"] = s["realign"] = None
    return s


def boot_ci(by, stat, reps=2000, seed=0):
    logs = sorted(by)
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        v = stat([x for lg in (rng.choice(logs) for _ in logs) for x in by[lg]])
        if v is not None and np.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def main() -> None:
    out_dir, runs = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    scenes = []
    for run in runs:
        for l in open(os.path.join(run, "records.jsonl")):
            r = json.loads(l)
            if r.get("stub_cot"):
                continue
            gt_xy = np.asarray(r["trajectory_gt"], float)[:, :2]
            C = [cand_stats(x, r["gt"], gt_xy) for x in r["candidates"]]
            re_ = rankdata([c["ent_mean"] for c in C]); rl_ = rankdata([-c["lp_sum"] for c in C])
            score = re_ + rl_ + 1e-6 * re_
            rank = rankdata(score, method="ordinal")
            for c, k in zip(C, rank):
                c["ranksum_rank"] = int(k)
                c["ranksum_score"] = float(-score[list(rank).index(k)])     # higher = more preferred
            for i, c in enumerate(C):
                c["ranksum_score"] = float(-score[i])
            scenes.append({"token": r["token"], "log": r["log"], "natural_fail": C[0]["a_minus"] == 1.0, "cands": C})
    S = {"n_scenes": len(scenes), "n_candidates": sum(len(s["cands"]) for s in scenes),
         "n_logs": len({s["log"] for s in scenes})}

    # ---- 1. selected vs discarded ---------------------------------------------------------------
    metrics = ["post_dev_entropy", "err_growth", "amplification", "a_minus", "realign", "fde5", "ade5", "pre_dev_entropy"]
    S["selected_vs_discarded"] = {}
    for subset, keep in (("all_scenes", lambda s: True), ("natural_failure_scenes", lambda s: s["natural_fail"])):
        by = defaultdict(list)
        for s in scenes:
            if not keep(s):
                continue
            sel = next(c for c in s["cands"] if c["ranksum_rank"] == 1)
            dis = [c for c in s["cands"] if c["ranksum_rank"] != 1]
            row = {}
            for m in metrics:
                dv = [c[m] for c in dis if c[m] is not None]
                row[m] = (sel[m], float(np.mean(dv))) if (sel[m] is not None and dv) else None
            by[s["log"]].append(row)
        blk = {"n_scenes": sum(len(v) for v in by.values())}
        for m in metrics:
            def stat(rows, m=m):
                v = [r[m][0] - r[m][1] for r in rows if r[m] is not None]
                return float(np.mean(v)) if v else None
            rows = [r for v in by.values() for r in v]
            pairs = [r[m] for r in rows if r[m] is not None]
            blk[m] = {"selected": float(np.mean([p[0] for p in pairs])), "discarded": float(np.mean([p[1] for p in pairs])),
                      "diff": stat(rows), "ci95": boot_ci(by, stat), "n": len(pairs)}
        S["selected_vs_discarded"][subset] = blk

    # ---- 2. AUROC for amplification ---------------------------------------------------------------
    signals = {"ranksum_score": -1, "lp_sum": -1, "ent_mean": 1, "post_dev_entropy": 1, "err_growth": 1, "pre_dev_entropy": 1}
    # sign: +1 = higher value predicts amplification; ranksum_score / lp_sum: lower predicts amplification
    by_c = defaultdict(list)
    for s in scenes:
        for c in s["cands"]:
            by_c[s["log"]].append(c)
    S["auroc_amplification"] = {}
    for sig, sgn in signals.items():
        def pooled(cs, sig=sig, sgn=sgn):
            cs = [c for c in cs if c[sig] is not None]
            y = [c["amplification"] for c in cs]
            if len(set(y)) < 2:
                return None
            return float(roc_auc_score(y, [sgn * c[sig] for c in cs]))
        within = []
        for s in scenes:
            cs = [c for c in s["cands"] if c[sig] is not None]
            y = [c["amplification"] for c in cs]
            if len(set(y)) == 2:
                within.append(roc_auc_score(y, [sgn * c[sig] for c in cs]))
        allc = [c for v in by_c.values() for c in v]
        S["auroc_amplification"][sig] = {"pooled": pooled(allc), "pooled_ci95": boot_ci(by_c, pooled, reps=500),
                                         "within_scene_mean": float(np.mean(within)) if within else None,
                                         "n_mixed_scenes": len(within)}

    # ---- 3. mechanism among deviating candidates ------------------------------------------------
    dev = [c for v in by_c.values() for c in v if c["post_dev_entropy"] is not None]
    amp = [c for c in dev if c["amplification"] == 1.0]
    non = [c for c in dev if c["amplification"] == 0.0]
    S["deviating_candidates"] = {
        "n": len(dev), "n_amplifying": len(amp),
        "post_dev_entropy": {"amplifying": float(np.mean([c["post_dev_entropy"] for c in amp])),
                             "non_amplifying": float(np.mean([c["post_dev_entropy"] for c in non]))},
        "pre_dev_entropy": {"amplifying": float(np.mean([c["pre_dev_entropy"] for c in amp])),
                            "non_amplifying": float(np.mean([c["pre_dev_entropy"] for c in non]))},
        "err_growth": {"amplifying": float(np.mean([c["err_growth"] for c in amp])),
                       "non_amplifying": float(np.mean([c["err_growth"] for c in non]))},
        "realign": {"amplifying": float(np.mean([c["realign"] for c in amp])),
                    "non_amplifying": float(np.mean([c["realign"] for c in non]))},
        "corr_ent_mean_vs_post_dev_entropy": float(np.corrcoef([c["ent_mean"] for c in dev], [c["post_dev_entropy"] for c in dev])[0, 1]),
        "corr_post_dev_entropy_vs_err_growth": float(np.corrcoef([c["post_dev_entropy"] for c in dev], [c["err_growth"] for c in dev])[0, 1]),
    }
    json.dump(S, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)

    print(f"scenes {S['n_scenes']}  candidates {S['n_candidates']}  logs {S['n_logs']}")
    for subset, blk in S["selected_vs_discarded"].items():
        print(f"\n== selected (rank-sum) vs discarded: {subset} ({blk['n_scenes']} scenes)")
        for m in metrics:
            e = blk[m]
            ci = e["ci95"] or [float("nan")] * 2
            print(f"  {m:18s} selected {e['selected']:8.4f}  discarded {e['discarded']:8.4f}  diff {e['diff']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]  n={e['n']}")
    print("\n== AUROC for amplification (candidate level)")
    for sig, e in S["auroc_amplification"].items():
        ci = e["pooled_ci95"] or [float("nan")] * 2
        print(f"  {sig:18s} pooled {e['pooled']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  within-scene {e['within_scene_mean']:.3f} (n={e['n_mixed_scenes']})")
    d = S["deviating_candidates"]
    print(f"\n== deviating candidates: n={d['n']}, amplifying {d['n_amplifying']}")
    for k in ("post_dev_entropy", "pre_dev_entropy", "err_growth", "realign"):
        print(f"  {k:18s} amplifying {d[k]['amplifying']:.4f}  non-amplifying {d[k]['non_amplifying']:.4f}")
    print(f"  corr(whole-plan entropy, post-dev entropy) {d['corr_ent_mean_vs_post_dev_entropy']:.3f}; "
          f"corr(post-dev entropy, error growth) {d['corr_post_dev_entropy_vs_err_growth']:.3f}")


if __name__ == "__main__":
    main()
