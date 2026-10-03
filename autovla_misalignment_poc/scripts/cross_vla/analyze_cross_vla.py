#!/usr/bin/env python
"""
Analysis of the OpenVLA cross-VLA replication (outputs/cross_vla_replication/PROTOCOL.md). CPU only.

Phase A/B (closed loop, LIBERO-Spatial 10 tasks x 10 inits):
  success rate per condition; paired vs natural by (task, init): McNemar exact, task-cluster bootstrap 95% CI
  (2,000, seed 0), exact task-level sign-flip permutation (2^10); among naturally successful episodes the share that
  fails under the perturbation (Q1); feedback vs corrected (Q3) and reverse vs natural (Q4) paired the same way.
  Within-step downstream deviation during the perturbation window, from the same-observation natural decode:
  D = sum_{j=1..5} |bin_j - natural bin_j|, amplified = max_j |dbin_j| >= |delta_eff|, recovered = D == 0.
Phase C (offline frames): D / amplified / recovered and entropy at d+1 per (d, |delta|, mode), task-cluster bootstrap CI.

  python analyze_cross_vla.py <cross_vla_replication_dir>
"""
from __future__ import annotations

import itertools
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest


def boot_tasks(vals_by_task, reps=2000):
    tasks = sorted(vals_by_task); rng = random.Random(0); out = []
    for _ in range(reps):
        v = [x for _ in tasks for x in vals_by_task[rng.choice(tasks)]]
        out.append(np.mean(v))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def perm_tasks(diff_by_task):
    m = np.array([np.mean(v) for _, v in sorted(diff_by_task.items())]); obs = abs(m.mean())
    allp = [abs((np.array(s) * m).mean()) for s in itertools.product([-1, 1], repeat=len(m))]
    return float(np.mean([p >= obs - 1e-12 for p in allp]))


def paired(E, a, b):
    """E: {(task, init, cond): success}. Success-rate difference a - b with tests."""
    keys = sorted({(t, i) for (t, i, c) in E if c == a} & {(t, i) for (t, i, c) in E if c == b})
    d = defaultdict(list); n10 = n01 = 0
    for t, i in keys:
        x, y = E[(t, i, a)], E[(t, i, b)]
        d[t].append(float(x) - float(y)); n10 += x and not y; n01 += y and not x
    p_mc = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
    return {"n": len(keys), "rate_a": float(np.mean([E[(t, i, a)] for t, i in keys])), "rate_b": float(np.mean([E[(t, i, b)] for t, i in keys])),
            "diff": float(np.mean([x for v in d.values() for x in v])), "ci95_task_boot": boot_tasks(d),
            "p_mcnemar": p_mc, "p_perm_tasks": perm_tasks(d), "a_only": int(n10), "b_only": int(n01)}


def window_stats(rec, mode):
    out = []
    for s in rec:
        if "natural" not in s:
            continue
        ex, nat, ctx = np.array(s["exec"]), np.array(s["natural"]), np.array(s["ctx"])
        de = abs(int(ctx[0] if mode == "reverse" else ex[0]) - int(nat[0]))
        dd = np.abs(ex[1:6] - nat[1:6])
        out.append({"delta_eff": de, "D": int(dd.sum()), "amp": bool(de > 0 and dd.max() >= de), "rec": bool(dd.sum() == 0)})
    return out


def main() -> None:
    root = sys.argv[1]
    res = {}
    A = [json.loads(l) for l in open(os.path.join(root, "rollouts/phaseA/episodes.jsonl"))]
    B = [json.loads(l) for l in open(os.path.join(root, "rollouts/phaseB/episodes.jsonl"))] if os.path.exists(os.path.join(root, "rollouts/phaseB/episodes.jsonl")) else []
    rp = os.path.join(root, "rollouts/phaseA_rep/episodes.jsonl")
    R = [dict(json.loads(l), cond="natural_rep") for l in open(rp)] if os.path.exists(rp) else []
    E = {(r["task"], r["init"], r["cond"]): bool(r["success"]) for r in A + B + R}
    conds = sorted({r["cond"] for r in B})
    res["natural_success"] = {"rate": float(np.mean([r["success"] for r in A])), "n": len(A),
                              "per_task": {t: int(sum(r["success"] for r in A if r["task"] == t)) for t in range(10)}}
    res["vs_natural"] = {c: paired(E, c, "natural") for c in conds}
    if R:   # protocol amendment: run-to-run variability of the natural policy under a different batch composition
        res["natural_rep_vs_natural"] = paired(E, "natural_rep", "natural")
        res["natural_rep_vs_natural"]["discordant_share"] = (res["natural_rep_vs_natural"]["a_only"] + res["natural_rep_vs_natural"]["b_only"]) / res["natural_rep_vs_natural"]["n"]
        res["vs_natural_rep"] = {c: paired(E, c, "natural_rep") for c in conds}
        for c in conds:
            v = res["vs_natural"][c]; v["discordant_share"] = (v["a_only"] + v["b_only"]) / v["n"]
    nat_ok = {(r["task"], r["init"]) for r in A if r["success"]}
    res["fail_given_natural_success"] = {c: float(np.mean([not E[(t, i, c)] for t, i in nat_ok if (t, i, c) in E])) for c in conds}
    for a_ in (8, 24):
        res[f"feedback_vs_corrected_d{a_}"] = paired(E, f"feedback_d{a_}", f"corrected_d{a_}")
        res[f"reverse_vs_natural_d{a_}"] = paired(E, f"reverse_d{a_}", "natural")
    W = {}
    for c in conds:
        mode = c.split("_d")[0]
        st = [x for r in B if r["cond"] == c for x in window_stats(r["rec"], mode)]
        if st:
            W[c] = {"n_steps": len(st), "mean_delta_eff": float(np.mean([x["delta_eff"] for x in st])),
                    "mean_D": float(np.mean([x["D"] for x in st])), "amplified": float(np.mean([x["amp"] for x in st])),
                    "recovered": float(np.mean([x["rec"] for x in st]))}
    res["closed_loop_window_token_stats"] = W
    # branch structure (Q2): per episode, success and mean within-window D
    res["branches"] = {}
    for c in [c for c in conds if c.startswith("feedback")]:
        ep = [(r["success"], np.mean([x["D"] for x in window_stats(r["rec"], "feedback")] or [0])) for r in B if r["cond"] == c and (r["task"], r["init"]) in nat_ok]
        res["branches"][c] = {"n_nat_success_eps": len(ep), "recovered_eps": int(sum(s for s, _ in ep)), "failed_eps": int(sum(not s for s, _ in ep)),
                              "mean_D_failed": float(np.mean([d for s, d in ep if not s])) if any(not s for s, _ in ep) else None,
                              "mean_D_recovered": float(np.mean([d for s, d in ep if s])) if any(s for s, _ in ep) else None}
    # phase C
    pc = os.path.join(root, "token_feedback/token_feedback.jsonl")
    if os.path.exists(pc):
        agg = defaultdict(lambda: defaultdict(list))
        for l in open(pc):
            f = json.loads(l); V = f["variants"]; ex = np.array(f["exec"]); ctx = np.array(f["ctx"]); ent = np.array(f["ent"])
            nat = ex[0]
            for k, v in enumerate(V[1:], 1):
                d, dl, mode = v.split(":"); d, dl = int(d), int(dl)
                de = abs(int(ctx[k, d] if mode == "reverse" else ex[k, d]) - int(nat[d]))
                if de == 0:
                    continue                                   # clipped at the bin range: no perturbation happened
                dd = np.abs(ex[k, d + 1:6] - nat[d + 1:6])
                key = f"d{d}_|{abs(dl)}|_{mode}"
                agg[key][f["task"]].append((dd.sum(), float(dd.max() >= de), float(dd.sum() == 0), ent[k, d + 1] - ent[0, d + 1]))
        PC = {}
        for key, bt in agg.items():
            col = lambda j: {t: [x[j] for x in v] for t, v in bt.items()}   # noqa: E731
            PC[key] = {"n": sum(len(v) for v in bt.values())}
            for j, nm in enumerate(("D", "amplified", "recovered", "d_entropy_next")):
                c_ = col(j); PC[key][nm] = float(np.mean([x for v in c_.values() for x in v])); PC[key][nm + "_ci95"] = boot_tasks(c_, 1000)
        res["phaseC"] = PC
    json.dump(res, open(os.path.join(root, "analysis.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "phaseC"}, indent=1)[:6000])
    if "phaseC" in res:
        print("\nphase C:")
        for k in sorted(res["phaseC"]):
            v = res["phaseC"][k]
            print(f"  {k:28s} n={v['n']:5d} D={v['D']:.2f} amp={v['amplified']:.3f} [{v['amplified_ci95'][0]:.3f},{v['amplified_ci95'][1]:.3f}] "
                  f"rec={v['recovered']:.3f} dEnt={v['d_entropy_next']:+.3f}")


if __name__ == "__main__":
    main()
