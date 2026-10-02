#!/usr/bin/env python
"""
Safety-filter ablation on navhard (outputs/safety_filter_ablation/PROTOCOL.md): paired contrasts A − B over groups.

Group scores come from navhard_group_scores.py (g_<rule>.json), searched in the given dirs per half. Aliases map the
protocol names onto already-scored files; `filter_random` is the mean of its three seeds (group score and every metric).
Per contrast: mean difference, log-cluster bootstrap 95% CI (2,000, seed 0), log-level paired sign-flip permutation
(two-sided, 20,000, seed 0), group-level Wilcoxon signed-rank, groups better / worse / tied.

  python scripts/analyze_navhard_ablation.py <out_prefix> --half NAME SUBSET DIR[,DIR...] [--half ...]
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

ALIAS = {"filter_ranksum": "F1", "filter_maxll": "F1_maxll"}
SEEDS = {"filter_random": ["filter_random_s0", "filter_random_s1", "filter_random_s2"]}
CONDS = ["normal", "ranksum", "filter_only", "filter_random", "filter_maxll", "filter_ranksum",
         "filter_maxll_natfb", "filter_ranksum_natfb"]
PRIMARY = [("filter_only", "normal"), ("filter_ranksum", "filter_only"), ("filter_maxll", "filter_only")]
SECONDARY = [("filter_random", "filter_only"), ("filter_ranksum", "filter_random"), ("filter_maxll", "filter_random"),
             ("filter_ranksum_natfb", "filter_only"), ("filter_ranksum", "filter_ranksum_natfb"),
             ("filter_maxll_natfb", "filter_only"), ("filter_maxll", "filter_maxll_natfb"),
             ("ranksum", "normal"), ("filter_ranksum", "normal"), ("filter_maxll", "normal"), ("filter_random", "normal")]
# (label, stage, metric, violation-rate?)
METRICS = [("stage-1 score", "stage1", "score", False), ("stage-2 score", "stage2", "score", False),
           ("collision % s1", "stage1", "no_at_fault_collisions", True), ("collision % s2", "stage2", "no_at_fault_collisions", True),
           ("DAC viol. % s1", "stage1", "drivable_area_compliance", True), ("DAC viol. % s2", "stage2", "drivable_area_compliance", True),
           ("progress s1", "stage1", "ego_progress", False), ("progress s2", "stage2", "ego_progress", False),
           ("hist. comfort s2", "stage2", "history_comfort", False), ("ext. comfort s1", "stage1", "two_frame_extended_comfort", False),
           ("ext. comfort s2", "stage2", "two_frame_extended_comfort", False)]
EPS = 1e-9


def group_vec(x):
    """{'epdms': .., 'stage1:metric': .., 'stage2:metric': ..} for one group (stage metric = mean of the two parts)."""
    v = {"epdms": x["group_score"]}
    for st in ("stage1", "stage2"):
        for m in x["parts"][0][f"metrics_{st}"]:
            v[f"{st}:{m}"] = float(np.mean([p[f"metrics_{st}"][m] for p in x["parts"]]))
    v["_parts_s1"] = [p["stage1"] for p in x["parts"]]
    return v


def load_rule(dirs, name):
    for d in dirs:
        p = os.path.join(d, f"g_{name}.json")
        if os.path.exists(p):
            return {x["orig_token"]: group_vec(x) for x in json.load(open(p))}
    raise FileNotFoundError(f"g_{name}.json not in {dirs}")


def load_half(subset, dirs):
    log_of = {g["orig"]: g["log"] for g in json.load(open(os.path.join(subset, "groups.json")))}
    R = {}
    for c in CONDS:
        if c in SEEDS:
            parts = [load_rule(dirs, s) for s in SEEDS[c]]
            R[c] = {t: {k: (float(np.mean([p[t][k] for p in parts])) if k != "_parts_s1" else
                            list(np.mean([p[t][k] for p in parts], axis=0)))
                        for k in parts[0][t]} for t in parts[0]}
        else:
            R[c] = load_rule(dirs, ALIAS.get(c, c))
    toks = sorted(set.intersection(*(set(v) for v in R.values())))
    return [{"log": log_of[t], **{c: R[c][t] for c in CONDS}} for t in toks]


def stats(rows):
    by = defaultdict(list)
    for i, x in enumerate(rows):
        by[x["log"]].append(i)
    logs = sorted(by); rng = random.Random(0)
    boots = [[rng.choice(logs) for _ in logs] for _ in range(2000)]
    signs = np.random.default_rng(0).choice([-1.0, 1.0], size=(20000, len(logs)))
    idx = [np.array(by[lg]) for lg in logs]
    B = np.array([[logs.index(lg) for lg in bt] for bt in boots])          # (2000, n_logs) resampled log indices
    cnt = np.array([len(i) for i in idx], float)

    def contrast(a, b):
        out = {}
        for key in ["epdms"] + [f"{st}:{m}" for _, st, m, _ in METRICS]:
            d = np.array([x[a][key] - x[b][key] for x in rows])
            sums = np.array([d[i].sum() for i in idx])
            bs = np.sort(sums[B].sum(axis=1) / cnt[B].sum(axis=1))
            e = {"diff": float(d.mean()), "ci95": [float(bs[49]), float(bs[1949])]}
            if key == "epdms":
                lm = np.array([d[i].mean() for i in idx])
                e["p_perm_log"] = float((1 + (np.abs((signs * lm).mean(axis=1)) >= abs(lm.mean()) - EPS).sum()) / (1 + len(signs)))
                nz = d[np.abs(d) > EPS]
                e["p_wilcoxon_group"] = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
                e["groups_better_worse_tied"] = [int((d > EPS).sum()), int((d < -EPS).sum()), int((np.abs(d) <= EPS).sum())]
            out[key] = e
        return out
    means = {c: {k: float(np.mean([x[c][k] for x in rows])) for k in rows[0][c] if not k.startswith("_")} for c in CONDS}
    return {"n_groups": len(rows), "n_logs": len(logs), "means": means,
            "contrasts": {f"{a} - {b}": contrast(a, b) for a, b in PRIMARY + SECONDARY}}


def md(name, S):
    L = [f"### {name}: {S['n_groups']} groups, {S['n_logs']} logs", "", "Means:", "",
         "| condition | EPDMS | " + " | ".join(l for l, *_ in METRICS) + " |", "|---|---:|" + "---:|" * len(METRICS)]
    for c in CONDS:
        m = S["means"][c]
        cells = [f"{(1 - m[f'{st}:{k}']) * 100:.2f}" if viol else f"{m[f'{st}:{k}']:.4f}" for _, st, k, viol in METRICS]
        L.append(f"| {c} | {m['epdms']:.4f} | " + " | ".join(cells) + " |")
    L += ["", "Contrasts (A − B; violation rates in %p, positive = more violations):", "",
          "| contrast | Δ EPDMS [95% CI] | p (log perm) | p (Wilcoxon) | groups +/−/= | Δ collision s1 / s2 | Δ DAC viol. s1 / s2 | Δ progress s1 / s2 | Δ hist. comfort s2 |",
          "|---|---|---:|---:|---|---|---|---|---|"]
    for name_, e in S["contrasts"].items():
        x = e["epdms"]

        def v(st, k):
            y = e[f"{st}:{k}"]
            return f"{-y['diff'] * 100 + 0.0:+.2f} [{-y['ci95'][1] * 100 + 0.0:+.2f}, {-y['ci95'][0] * 100 + 0.0:+.2f}]"

        def s(st, k):
            y = e[f"{st}:{k}"]
            return f"{y['diff']:+.4f} [{y['ci95'][0]:+.4f}, {y['ci95'][1]:+.4f}]"
        L.append(f"| {name_} | {x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] | {x['p_perm_log']:.2g} | "
                 f"{x['p_wilcoxon_group']:.2g} | {'/'.join(map(str, x['groups_better_worse_tied']))} | "
                 f"{v('stage1', 'no_at_fault_collisions')} / {v('stage2', 'no_at_fault_collisions')} | "
                 f"{v('stage1', 'drivable_area_compliance')} / {v('stage2', 'drivable_area_compliance')} | "
                 f"{s('stage1', 'ego_progress')} / {s('stage2', 'ego_progress')} | {s('stage2', 'history_comfort')} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--half", nargs=3, action="append", metavar=("NAME", "SUBSET", "DIRS"), required=True)
    a = ap.parse_args()
    allrows, res, text = [], {}, []
    for name, subset, dirs in a.half:
        rows = load_half(subset, dirs.split(",")); allrows += rows
        res[name] = stats(rows); text.append(md(name, res[name]))
    if len(a.half) > 1:
        res["full"] = stats(allrows); text.insert(0, md("full navhard", res["full"]))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    open(a.out + ".md", "w").write("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
