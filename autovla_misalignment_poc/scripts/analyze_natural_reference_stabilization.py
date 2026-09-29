#!/usr/bin/env python
"""
Analysis of natural_reference_stabilization (CPU only).

Per group (A- = natural failures, random = 300 uniformly sampled other scenes) and row:
  recovery (P/R/A A+ at 5 s), failure = A- AND FDE5 > 3 m (the amplification definition),
  ADE5 / FDE5, token realignment with GT; paired row - normal (McNemar / Wilcoxon, log-cluster CI).
For the random group the estimated population effect is also reported: the random sample stands
for the (2747 - 52) non-failing scenes, the A- group for the 52 failing ones.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_action_history import BINARY, cboot, unit_metrics      # noqa: E402
from analyze_prev_action_identity import paired                     # noqa: E402

BINARY.update({"a_minus", "failure"})     # same set object paired() consults -> McNemar for these

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_TOTAL, N_FAIL = 2747, 52


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(POC_DIR, "outputs/natural_reference_stabilization"))
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(args.run, "records.jsonl"))]
    rows = json.load(open(os.path.join(args.run, "run_meta.json")))["rows"] if os.path.exists(os.path.join(args.run, "run_meta.json")) \
        else list(recs[0]["conditions"])
    units = []
    for r in recs:
        u = {"token": r["token"], "log": r["log"], "group": r["group"], "repro": r["normal_reproduces_stored"],
             "avail": {s: r["ref"].get(s) is not None for s in ("prev", "consmed", "consmean", "gt")}}
        for c in rows:
            x = r["conditions"][c]
            m = unit_metrics(x["action_idx"], x["trajectory_pred"], x["entropy_steps"], r["gt"], r["trajectory_gt"], 0)
            m["failure"] = m.pop("amplification")
            m["a_minus"] = not m["recovery"]
            u[c] = m
        units.append(u)
    M = ["recovery", "a_minus", "failure", "ade5", "fde5", "realign"]
    S = {"n": {g: sum(u["group"] == g for u in units) for g in ("A-", "random")},
         "normal_reproduces_stored_natural_tokens": {g: float(np.mean([u["repro"] for u in units if u["group"] == g])) for g in ("A-", "random")},
         "groups": {}}
    for g in ("A-", "random"):
        us_all = [u for u in units if u["group"] == g]
        blk = {"conditions": {}, "vs_normal": {}}
        for c in rows:
            src = c.split("_")[0]
            us = [u for u in us_all if u["avail"].get(src, True)]
            blk["conditions"][c] = {m: cboot([(u["log"], float(u[c][m])) for u in us if u[c][m] is not None]) for m in M} | {"n": len(us)}
            if c != "normal":
                blk["vs_normal"][c] = {m: paired(us, c, "normal", m) for m in ("a_minus", "failure", "fde5", "ade5")}
        S["groups"][g] = blk
    # population estimate of the A- rate (natural failure rate) under each row
    S["population_a_minus_rate"] = {}
    for c in rows:
        a = S["groups"]["A-"]["conditions"][c]["a_minus"]["mean"]
        r_ = S["groups"]["random"]["conditions"][c]["a_minus"]["mean"]
        S["population_a_minus_rate"][c] = (N_FAIL * a + (N_TOTAL - N_FAIL) * r_) / N_TOTAL
    # entropy trigger: apply a stabilisation row only when the normal row's entropy statistic exceeds a
    # threshold chosen leave-one-log-out (cost = population-weighted A- rate + 0.05 * FDE5).
    # ent0 is causal (available before any correction); first4/mean/max look ahead over the plan.
    from sklearn.metrics import roc_auc_score
    wgt = {"A-": N_FAIL / max(1, S["n"]["A-"]), "random": (N_TOTAL - N_FAIL) / max(1, S["n"]["random"])}
    ent = {r["token"]: np.asarray(r["conditions"]["normal"]["entropy_steps"]) for r in recs}
    feats = {"ent0": lambda e: e[0], "ent_first4": lambda e: e[:4].mean(), "ent_mean": lambda e: e.mean(), "ent_max": lambda e: e.max()}
    S["entropy_trigger"] = {"auroc_A-_vs_random": {k: float(roc_auc_score([u["group"] == "A-" for u in units], [f(ent[u["token"]]) for u in units]))
                                                   for k, f in feats.items()}, "policies": {}}

    def pop(sel, pol):
        num = den = fs = 0.0
        for u in units:
            c = pol if sel(u) else "normal"
            num += wgt[u["group"]] * float(u[c]["a_minus"]); fs += wgt[u["group"]] * u[c]["fde5"]; den += wgt[u["group"]]
        return num / den, fs / den
    S["entropy_trigger"]["never"] = dict(zip(("a_minus", "fde5"), pop(lambda u: False, "normal")))
    logs_ = sorted({u["log"] for u in units})
    for k, f in feats.items():
        for pol in [c for c in ("prev_w4", "prev_all", "consmean_all") if c in rows]:
            S["entropy_trigger"]["policies"][f"{k}/{pol}"] = {"always": dict(zip(("a_minus", "fde5"), pop(lambda u: True, pol)))}
            taus = np.quantile([f(ent[u["token"]]) for u in units], np.linspace(0.5, 0.995, 40))
            num = den = fs = 0.0
            fired = {"A-": 0, "random": 0}
            for lg in logs_:
                tr = [u for u in units if u["log"] != lg]

                def cost(t):
                    return sum(wgt[u["group"]] * (float(u[pol if f(ent[u["token"]]) > t else "normal"]["a_minus"])
                                                  + 0.05 * u[pol if f(ent[u["token"]]) > t else "normal"]["fde5"]) for u in tr)
                best = min(taus, key=cost)
                for u in units:
                    if u["log"] != lg:
                        continue
                    c = pol if f(ent[u["token"]]) > best else "normal"
                    fired[u["group"]] += c != "normal"
                    num += wgt[u["group"]] * float(u[c]["a_minus"]); fs += wgt[u["group"]] * u[c]["fde5"]; den += wgt[u["group"]]
            S["entropy_trigger"]["policies"][f"{k}/{pol}"]["gated_lolo"] = {"a_minus": num / den, "fde5": fs / den, "fired": fired}

    json.dump(S, open(os.path.join(args.run, "summary.json"), "w"), indent=1)
    with open(os.path.join(args.run, "units.jsonl"), "w") as f:
        for u in units:
            f.write(json.dumps(u) + "\n")

    L = [f"units {S['n']}  normal reproduces stored natural tokens {S['normal_reproduces_stored_natural_tokens']}"]
    for g in ("A-", "random"):
        b = S["groups"][g]
        L.append(f"\n== {g} ==")
        L.append(f"{'row':14s} {'n':>4s} {'A- %':>20s} {'fail %':>8s} {'ADE5':>6s} {'FDE5':>6s} {'dA- %p (CI)':>24s} {'p':>8s} {'dFDE':>7s} {'p':>8s}")
        for c in rows:
            cd = b["conditions"][c]
            a = cd["a_minus"]
            v = b["vs_normal"].get(c)
            if v:
                d, f = v["a_minus"], v["fde5"]
                ds = f"{100*d['mean']:+.1f} [{100*d['ci95'][0]:+.1f},{100*d['ci95'][1]:+.1f}]"
                ps, fs, fp = f"{d['mcnemar']['p']:.2g}", f"{f['mean']:+.2f}", f"{f['wilcoxon_p']:.2g}"
            else:
                ds = ps = fs = fp = "–"
            L.append(f"{c:14s} {cd['n']:4d} {100*a['mean']:6.1f} [{100*a['ci95'][0]:5.1f},{100*a['ci95'][1]:5.1f}] "
                     f"{100*cd['failure']['mean']:8.1f} {cd['ade5']['mean']:6.2f} {cd['fde5']['mean']:6.2f} {ds:>24s} {ps:>8s} {fs:>7s} {fp:>8s}")
    L.append("\npopulation natural A- rate estimate: " + "  ".join(f"{c} {100*v:.2f}%" for c, v in S["population_a_minus_rate"].items()))
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(args.run, "analysis_console.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
