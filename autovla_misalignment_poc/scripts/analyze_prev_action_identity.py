#!/usr/bin/env python
"""
Analysis of prev_action_identity_decomposition (CPU only).

Metrics per unit are exactly those of analyze_action_history (amplification = A- AND FDE5 > 3 m,
recovery = P/R/A accept rule at 5 s, ...). Reported per group (A-, A+):
  - condition means with log-cluster bootstrap CI
  - paired condition - normal (McNemar / Wilcoxon) and condition - recent_gt
  - fraction of the Recent-GT effect: (normal - cond) / (normal - recent_gt), jointly bootstrapped
  - substitution geometry: codebook distance / embedding cosine of the substitute to GT and to own
  - reproduction of action_history_causal normal / gt_history / recent_gt tokens
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_action_history import METRICS, BINARY, REPS, cboot, unit_metrics   # noqa: E402

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = ["normal", "gt_history", "recent_gt", "geo_nn_gt", "emb_nn_gt", "geo_nn_self", "emb_nn_self",
        "random_tok", "mean_emb"]
NAMES = {"normal": "Normal AR", "gt_history": "GT-history", "recent_gt": "Recent-GT (직전 1개 GT)",
         "geo_nn_gt": "GT의 기하 최근접 토큰", "emb_nn_gt": "GT의 embedding 최근접 토큰",
         "geo_nn_self": "자기 토큰의 기하 최근접", "emb_nn_self": "자기 토큰의 embedding 최근접",
         "random_tok": "무작위 토큰", "mean_emb": "[OOD] 평균 action embedding"}
SUBS = ["geo_nn_gt", "emb_nn_gt", "geo_nn_self", "emb_nn_self", "random_tok"]


def paired(us, c, ref, m):
    pairs = [(u["log"], u[c][m], u[ref][m]) for u in us if u[c][m] is not None and u[ref][m] is not None]
    d = cboot([(lg, float(a) - float(b)) for lg, a, b in pairs])
    if not d:
        return None
    if m in BINARY:
        gain = sum(bool(a) and not bool(b) for _, a, b in pairs)
        loss = sum(bool(b) and not bool(a) for _, a, b in pairs)
        d["mcnemar"] = {"cond_only": gain, "ref_only": loss,
                        "p": 1.0 if gain + loss == 0 else float(binomtest(gain, gain + loss, 0.5).pvalue)}
    else:
        diffs = np.array([float(a) - float(b) for _, a, b in pairs])
        d["wilcoxon_p"] = float(wilcoxon(diffs).pvalue) if np.any(diffs != 0) else 1.0
    return d


def frac_effect(us, c, m, reps=REPS, seed=7):
    """(normal - c) / (normal - recent_gt), logs resampled jointly."""
    by = defaultdict(list)
    for u in us:
        if all(u[x][m] is not None for x in ("normal", "recent_gt", c)):
            by[u["log"]].append((float(u["normal"][m]), float(u["recent_gt"][m]), float(u[c][m])))
    logs = list(by)
    if not logs:
        return None

    def f(ls):
        v = np.array([x for lg in ls for x in by[lg]])
        den = v[:, 0].mean() - v[:, 1].mean()
        return (v[:, 0].mean() - v[:, 2].mean()) / den if abs(den) > 1e-9 else np.nan
    rng = random.Random(seed)
    bs = [x for x in (f([rng.choice(logs) for _ in logs]) for _ in range(reps)) if np.isfinite(x)]
    return {"frac": float(f(logs)), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(POC_DIR, "outputs/prev_action_identity_decomposition"))
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(args.run, "records.jsonl"))]

    units, repro = [], defaultdict(list)
    for r in recs:
        t, gt, gtraj = r["t_star"], r["gt"], r["trajectory_gt"]
        u = {"token": r["token"], "log": r["log"], "group": r["group"], "pert": r["perturbation"], "t_star": t}
        for c in ROWS:
            x = r["conditions"][c]
            u[c] = unit_metrics(x["action_idx"], x["trajectory_pred"], x["entropy_steps"], gt, gtraj, t)
            subs = x.get("substitutions") or []
            if c in SUBS and subs:
                u[c + "_sub"] = {k: float(np.mean([s[k] for s in subs])) for k in
                                 ("geo_d_sub_gt", "geo_d_own_gt", "geo_d_sub_own", "cos_sub_gt", "cos_own_gt", "cos_sub_own")}
        if r.get("reproduces_action_history"):
            for c, ok in r["reproduces_action_history"].items():
                repro[c].append(bool(ok))
        units.append(u)

    S = {"n_units": {g: sum(u["group"] == g for u in units) for g in ("A-", "A+")},
         "n_scenes": {g: len({u["token"] for u in units if u["group"] == g}) for g in ("A-", "A+")},
         "reproduces_action_history_tokens": {c: float(np.mean(v)) for c, v in repro.items()},
         "groups": {}}
    for g in ("A-", "A+"):
        us = [u for u in units if u["group"] == g]
        blk = {"conditions": {c: {m: cboot([(u["log"], u[c][m]) for u in us]) for m in METRICS} for c in ROWS},
               "vs_normal": {c: {m: paired(us, c, "normal", m) for m in METRICS} for c in ROWS[1:]},
               "vs_recent_gt": {c: {m: paired(us, c, "recent_gt", m) for m in ("amplification", "fde5", "realign")}
                                for c in ROWS if c not in ("normal", "recent_gt")},
               "frac_of_recent_gt_effect": {c: {m: frac_effect(us, c, m) for m in ("amplification", "fde5")}
                                            for c in ROWS if c not in ("normal", "recent_gt")},
               "substitution_geometry": {}}
        for c in SUBS:
            vals = [u[c + "_sub"] for u in us if c + "_sub" in u]
            if vals:
                blk["substitution_geometry"][c] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        # dose-response inside the substitution rows: is FDE lower when the substitute is closer to GT?
        blk["dose_response"] = {}
        for c in SUBS:
            xs = [(u[c + "_sub"]["geo_d_sub_gt"], u[c + "_sub"]["cos_sub_gt"], u[c]["fde5"]) for u in us if c + "_sub" in u]
            if len(xs) > 10:
                a = np.array(xs)
                blk["dose_response"][c] = {"spearman_fde_vs_geo_d_sub_gt": float(spearmanr(a[:, 0], a[:, 2])[0]),
                                           "spearman_fde_vs_cos_sub_gt": float(spearmanr(a[:, 1], a[:, 2])[0]), "n": len(xs)}
        S["groups"][g] = blk

    # geometry vs embedding: outcome change regressed jointly on how much the substitute moved
    # toward GT in codebook distance and in embedding cosine (standardised, log-cluster bootstrap).
    def reg_rows(gs):
        X = []
        for u in units:
            if u["group"] not in gs:
                continue
            for c in ("geo_nn_gt", "emb_nn_gt", "geo_nn_self", "emb_nn_self"):
                s = u.get(c + "_sub")
                if s:
                    X.append((u["log"], s["geo_d_sub_gt"] - s["geo_d_own_gt"], s["cos_sub_gt"] - s["cos_own_gt"],
                              u[c]["fde5"] - u["normal"]["fde5"], float(u[c]["amplification"]) - float(u["normal"]["amplification"])))
        return X

    def fit(X, yi):
        a = np.array([x[1:] for x in X], float)
        Z = (a[:, :2] - a[:, :2].mean(0)) / a[:, :2].std(0)
        return np.linalg.lstsq(np.c_[np.ones(len(Z)), Z], a[:, yi], rcond=None)[0][1:]

    S["geometry_vs_embedding"] = {"note": "coef per 1 SD; dgeo = codebook dist(sub,GT) - dist(own,GT) (+ = farther), "
                                          "dcos = cos(sub,GT) - cos(own,GT) (+ = closer in embedding)"}
    for name, gs in (("A-", {"A-"}), ("A+", {"A+"}), ("both", {"A-", "A+"})):
        X = reg_rows(gs)
        by = defaultdict(list)
        for x in X:
            by[x[0]].append(x)
        logs = list(by)
        out = {"n": len(X), "corr_dgeo_dcos": float(np.corrcoef(np.array([x[1:3] for x in X]).T)[0, 1])}
        for yi, m in ((2, "fde5"), (3, "amplification")):
            rng = random.Random(0)
            bs = np.array([fit([x for lg in (rng.choice(logs) for _ in logs) for x in by[lg]], yi) for _ in range(1000)])
            b = fit(X, yi)
            out[m] = {"dgeo": float(b[0]), "dgeo_ci95": np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                      "dcos": float(b[1]), "dcos_ci95": np.percentile(bs[:, 1], [2.5, 97.5]).tolist()}
        S["geometry_vs_embedding"][name] = out

    with open(os.path.join(args.run, "summary.json"), "w") as f:
        json.dump(S, f, indent=1)
    with open(os.path.join(args.run, "units.jsonl"), "w") as f:
        for u in units:
            f.write(json.dumps(u) + "\n")

    # console table
    def fmt(x, pct=False):
        if not x:
            return "–"
        k = 100 if pct else 1
        return f"{x['mean']*k:.1f} [{x['ci95'][0]*k:.1f}, {x['ci95'][1]*k:.1f}]" if pct else f"{x['mean']:.2f}"
    lines = [f"units {S['n_units']}  scenes {S['n_scenes']}  reproduction {S['reproduces_action_history_tokens']}"]
    for g in ("A-", "A+"):
        blk = S["groups"][g]
        lines.append(f"\n== {g} ==")
        lines.append(f"{'condition':34s} {'amplif %':>22s} {'FDE5':>6s} {'realign':>7s} {'frac(amp)':>16s} {'frac(FDE)':>16s} {'p vs normal':>11s}")
        for c in ROWS:
            cd = blk["conditions"][c]
            fr = blk["frac_of_recent_gt_effect"].get(c, {})
            fa = fr.get("amplification") if fr else None
            ff = fr.get("fde5") if fr else None
            vn = blk["vs_normal"].get(c, {}).get("amplification") if c != "normal" else None
            fa_s = "%.2f" % fa["frac"] if fa else "–"
            ff_s = "%.2f" % ff["frac"] if ff else "–"
            vn_s = "%.2g" % vn["mcnemar"]["p"] if vn else "–"
            lines.append(f"{NAMES[c]:34s} {fmt(cd['amplification'], True):>22s} {fmt(cd['fde5']):>6s} "
                         f"{cd['realign']['mean'] if cd['realign'] else float('nan'):7.3f} "
                         f"{fa_s:>16s} {ff_s:>16s} "
                         f"{vn_s:>11s}")
        for c, v in blk["substitution_geometry"].items():
            lines.append(f"  sub {c:12s} geo_d(sub,gt)={v['geo_d_sub_gt']:.2f} geo_d(own,gt)={v['geo_d_own_gt']:.2f} "
                         f"geo_d(sub,own)={v['geo_d_sub_own']:.2f}  cos(sub,gt)={v['cos_sub_gt']:.3f} cos(own,gt)={v['cos_own_gt']:.3f}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(args.run, "analysis_console.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
