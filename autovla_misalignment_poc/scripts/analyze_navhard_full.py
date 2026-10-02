#!/usr/bin/env python
"""
Pre-registered navhard comparison (outputs/navhard_full_validation/PREREGISTRATION.md) from navhard_group_scores.py outputs.

Per group (= one official reactive_all_mapping entry: orig + prev stage-1 scene and their synthetic stage-2 scenes):
  EPDMS_g   = (s1(orig) * s2(orig) + s1(prev) * s2(prev)) / 2      (official; navhard EPDMS = mean over groups)
  stage-k metric m = mean over the two parts of the official weighted average of m
Rule vs no selection (`normal`):
  diff in means, log-cluster bootstrap 95% CI (2,000 resamples of logs, seed 0),
  paired log-level sign-flip permutation test on per-log mean differences (two-sided, 20,000 draws, seed 0) [primary],
  Wilcoxon signed-rank over groups (two-sided) [secondary],
  groups and stage-1 scenes better / worse / tied.

  python scripts/analyze_navhard_full.py <out_prefix> --half NAME SUBSET_DIR GROUP_SCORES_DIR [--half ...] --rules r1 r2 ...
Writes <out_prefix>.json and <out_prefix>.md; with several halves also reports the pooled set.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

METRICS = [("score", "score"), ("collision", "no_at_fault_collisions"), ("dac", "drivable_area_compliance"),
           ("ddc", "driving_direction_compliance"), ("tlc", "traffic_light_compliance"), ("progress", "ego_progress"),
           ("ttc", "time_to_collision_within_bound"), ("lane_keeping", "lane_keeping"),
           ("history_comfort", "history_comfort"), ("extended_comfort", "two_frame_extended_comfort")]
EPS = 1e-9


def load(subset, gdir, rules):
    log_of = {g["orig"]: g["log"] for g in json.load(open(os.path.join(subset, "groups.json")))}
    S = {r: {x["orig_token"]: x for x in json.load(open(os.path.join(gdir, f"g_{r}.json")))} for r in rules}
    toks = sorted(set.intersection(*(set(v) for v in S.values())))
    return [{"log": log_of[t], "token": t, **{r: S[r][t] for r in rules}} for t in toks]


def gval(x, key):
    if key == "epdms":
        return x["group_score"]
    stage, m = key.split(":")
    vals = [p[f"metrics_{stage}"][m] for p in x["parts"]]
    return float(np.mean(vals))


def compare(rows, rules):
    by = defaultdict(list)
    for i, x in enumerate(rows):
        by[x["log"]].append(i)
    logs = sorted(by)
    rng = random.Random(0)
    boots = [[rng.choice(logs) for _ in logs] for _ in range(2000)]
    prng = np.random.default_rng(0)
    signs = prng.choice([-1.0, 1.0], size=(20000, len(logs)))
    keys = ["epdms"] + [f"{st}:{m}" for st in ("stage1", "stage2") for _, m in METRICS]
    res = {"n_groups": len(rows), "n_logs": len(logs), "rules": {}}
    for r in rules:
        e = {}
        for k in keys:
            v = np.array([gval(x[r], k) for x in rows]); b = np.array([gval(x["normal"], k) for x in rows]); d = v - b
            bs = np.sort([np.mean([d[i] for lg in bt for i in by[lg]]) for bt in boots])
            e[k] = {"mean": float(v.mean()), "diff": float(d.mean()), "ci95": [float(bs[49]), float(bs[1949])]}
            if k in ("epdms", "stage1:score", "stage2:score") and r != "normal":
                lm = np.array([d[by[lg]].mean() for lg in logs])
                obs = abs(lm.mean()); perm = np.abs((signs * lm).mean(axis=1))
                e[k]["p_perm_log"] = float((1 + (perm >= obs - EPS).sum()) / (1 + len(perm)))
                nz = d[np.abs(d) > EPS]
                e[k]["p_wilcoxon_group"] = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
        d = np.array([x[r]["group_score"] - x["normal"]["group_score"] for x in rows])
        e["groups_better_worse_tied"] = [int((d > EPS).sum()), int((d < -EPS).sum()), int((np.abs(d) <= EPS).sum())]
        s1 = np.array([pr["stage1"] - pn["stage1"] for x in rows for pn, pr in zip(x["normal"]["parts"], x[r]["parts"])])
        e["stage1_scenes_better_worse_tied"] = [int((s1 > EPS).sum()), int((s1 < -EPS).sum()), int((np.abs(s1) <= EPS).sum())]
        res["rules"][r] = e
    return res


def table(name, res):
    L = [f"### {name}: {res['n_groups']} groups, {res['n_logs']} logs", "",
         "| rule | EPDMS | Δ [95% CI] | p (log perm) | p (Wilcoxon) | groups +/−/= | stage-1 | stage-2 |",
         "|---|---:|---|---:|---:|---|---:|---:|"]
    for r, e in res["rules"].items():
        x = e["epdms"]
        p1 = f"{x['p_perm_log']:.2g}" if "p_perm_log" in x else "–"
        p2 = f"{x['p_wilcoxon_group']:.2g}" if "p_wilcoxon_group" in x else "–"
        L.append(f"| {r} | {x['mean']:.4f} | {x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] | {p1} | {p2} | "
                 f"{'/'.join(map(str, e['groups_better_worse_tied']))} | {e['stage1:score']['mean']:.4f} | {e['stage2:score']['mean']:.4f} |")
    for st in ("stage1", "stage2"):
        L += ["", f"{st} sub-metrics (rate of violations = 1 − compliance; progress and comfort as scores), Δ vs normal [95% CI]:", "",
              "| rule | stage score | collision % | DAC viol. % | progress | history comfort | ext. comfort | TTC viol. % | DDC viol. % | stage-1 scenes +/−/= |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r, e in res["rules"].items():
            def c(m, viol=False, scale=1.0):
                x = e[f"{st}:{m}"]; mu, d, lo, hi = x["mean"], x["diff"], x["ci95"][0], x["ci95"][1]
                if viol:
                    return f"{(1 - mu) * 100:.2f} ({-d * 100 + 0.0:+.2f} [{-hi * 100 + 0.0:+.2f}, {-lo * 100 + 0.0:+.2f}])"
                return f"{mu:.4f} ({d:+.4f} [{lo:+.4f}, {hi:+.4f}])"
            cnt = "/".join(map(str, e["stage1_scenes_better_worse_tied"])) if st == "stage1" else ""
            L.append(f"| {r} | {c('score')} | {c('no_at_fault_collisions', True)} | {c('drivable_area_compliance', True)} | "
                     f"{c('ego_progress')} | {c('history_comfort')} | {c('two_frame_extended_comfort')} | "
                     f"{c('time_to_collision_within_bound', True)} | {c('driving_direction_compliance', True)} | {cnt} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--half", nargs=3, action="append", metavar=("NAME", "SUBSET", "GROUP_SCORES"), required=True)
    ap.add_argument("--rules", nargs="+", required=True)
    a = ap.parse_args()
    assert a.rules[0] == "normal"
    allrows, out, md = [], {}, []
    for name, subset, gdir in a.half:
        rows = load(subset, gdir, a.rules); allrows += rows
        out[name] = compare(rows, a.rules); md.append(table(name, out[name]))
    if len(a.half) > 1:
        assert len({x["log"] for x in allrows}) == sum(out[n]["n_logs"] for n, _, _ in a.half), "halves share logs"
        out["pooled"] = compare(allrows, a.rules); md.append(table("pooled (all halves)", out["pooled"]))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    open(a.out + ".md", "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
