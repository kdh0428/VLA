#!/usr/bin/env python
"""
Analysis of reference_stabilization (CPU only). Metrics and statistics are those of
analyze_action_history (amplification = A- AND FDE5 > 3 m; log-cluster bootstrap; McNemar).
Per group: condition means, paired condition - normal, fraction of the GT effect at the same window
((normal - cond) / (normal - gt_same_window)), and reference quality (ADE/FDE vs GT, token match).
Units without a previous frame are excluded from the prev rows (and reported).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_action_history import METRICS, REPS, cboot, unit_metrics   # noqa: E402
from analyze_prev_action_identity import paired                          # noqa: E402

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRCS, WINS = ["gt", "kin", "prev"], ["w2", "w4", "all"]
ROWS = ["normal"] + [f"{s}_{w}" for s in SRCS for w in WINS]
NAMES = {"gt": "GT (oracle)", "kin": "운동학 외삽 (CTRA)", "prev": "이전 frame 계획"}


def frac(us, c, ref_c, m, reps=REPS, seed=11):
    by = defaultdict(list)
    for u in us:
        if all(u[x][m] is not None for x in ("normal", ref_c, c)):
            by[u["log"]].append((float(u["normal"][m]), float(u[ref_c][m]), float(u[c][m])))
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
    ap.add_argument("--run", default=os.path.join(POC_DIR, "outputs/reference_stabilization"))
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(args.run, "records.jsonl"))]
    meta_p = os.path.join(args.run, "run_meta.json")
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p))
        global ROWS, SRCS
        ROWS = meta["rows"]
        SRCS = meta.get("sources", SRCS)
    units = []
    for r in recs:
        t, gt, gtraj = r["t_star"], r["gt"], r["trajectory_gt"]
        u = {"token": r["token"], "log": r["log"], "group": r["group"], "pert": r["perturbation"], "t_star": t,
             "prev_available": r["prev_available"], "ref_quality": r["ref_quality"],
             "avail": r.get("refs_available") or {"prev": r["prev_available"]}}
        for c in ROWS:
            x = r["conditions"][c]
            u[c] = unit_metrics(x["action_idx"], x["trajectory_pred"], x["entropy_steps"], gt, gtraj, t)
        units.append(u)

    S = {"n_units": {g: sum(u["group"] == g for u in units) for g in ("A-", "A+")},
         "n_units_prev_available": {g: sum(u["group"] == g and u["prev_available"] for u in units) for g in ("A-", "A+")},
         "groups": {}}
    for g in ("A-", "A+"):
        us_all = [u for u in units if u["group"] == g]
        us_prev = [u for u in us_all if u["prev_available"]]
        blk = {"conditions": {}, "vs_normal": {}, "frac_of_gt_same_window": {}, "reference_quality": {}}
        for c in ROWS:
            src = c.split("_")[0]
            us = [u for u in us_all if u["avail"].get(src, True)]
            blk["conditions"][c] = {m: cboot([(u["log"], u[c][m]) for u in us]) for m in METRICS}
            blk["conditions"][c]["n"] = len(us)
            if c != "normal":
                blk["vs_normal"][c] = {m: paired(us, c, "normal", m) for m in ("amplification", "recovery", "fde5")}
                w = c.split("_")[1]
                if f"gt_{w}" in ROWS:
                    blk["frac_of_gt_same_window"][c] = {m: frac(us, c, f"gt_{w}", m) for m in ("amplification", "fde5")}
        # also the prev rows against normal on the SAME (prev-available) units, and gt restricted to them
        blk["prev_subset_reference"] = {c: {m: cboot([(u["log"], u[c][m]) for u in us_prev]) for m in ("amplification", "fde5")}
                                        for c in ("normal", "gt_w2", "gt_w4", "gt_all") if c in ROWS}
        seen = {}
        for u in us_all:
            seen.setdefault(u["token"], u["ref_quality"])
        for s in SRCS:
            q = [v[s] for v in seen.values() if s in v]
            if q:
                blk["reference_quality"][s] = {k: float(np.mean([x[k] for x in q])) for k in q[0]} | {"n_scenes": len(q)}
        S["groups"][g] = blk

    # dose-response on reference quality and GT-free gates (window positions t*+1 .. t*+4)
    import pickle
    from scipy.stats import spearmanr
    cb = np.asarray(pickle.load(open("/root/VLA/autovla/codebook_cache/agent_vocab.pkl", "rb"))["token_all"]["veh"]).reshape(2048, -1)
    dist = lambda a, b: float(np.linalg.norm(cb[a] - cb[b]))   # noqa: E731
    umap = {(u["token"], u["pert"]): u for u in units}
    feats = []
    for r in recs:
        t = r["t_star"]
        pos = list(range(t + 1, min(10, t + 5)))
        if not pos:
            continue
        u = umap[(r["token"], r["perturbation"])]
        own = r["conditions"]["normal"]["action_idx"]
        x = {"log": r["log"], "g": r["group"], "u": u}
        for s in [x for x in SRCS if x != "gt"]:
            if r["ref"].get(s) is not None:
                x[f"{s}_gt_dist"] = float(np.mean([dist(r["ref"][s][j], r["gt"][j]) for j in pos]))
        if r["ref"]["prev"] is not None:
            pv, kn = r["ref"]["prev"], r["ref"]["kin"]
            x["gate_prev_kin"] = float(np.mean([dist(pv[j], kn[j]) for j in pos]))
            x["gate_own_prev"] = float(np.mean([dist(own[j], pv[j]) for j in pos]))
            x["gate_forced_prev"] = dist(r["forced_token"], pv[t])
        feats.append(x)
    S["dose_response_w4"] = {}
    for g in ("A-", "A+"):
        for s in [x for x in SRCS if x != "gt" and f"{x}_w4" in ROWS]:
            xs = [x for x in feats if x["g"] == g and f"{s}_gt_dist" in x]
            if len(xs) < 6:
                continue
            d = np.array([x[f"{s}_gt_dist"] for x in xs])
            q = np.quantile(d, [0, 1 / 3, 2 / 3, 1])
            out = []
            for i in range(3):
                sel = [x for x, dd in zip(xs, d) if q[i] <= dd <= q[i + 1]]
                out.append({"mean_dist": float(np.mean([x[f"{s}_gt_dist"] for x in sel])), "n": len(sel),
                            "d_amplification": float(np.mean([x["u"][f"{s}_w4"]["amplification"] - x["u"]["normal"]["amplification"] for x in sel])),
                            "d_fde5": float(np.mean([x["u"][f"{s}_w4"]["fde5"] - x["u"]["normal"]["fde5"] for x in sel]))})
            S["dose_response_w4"][f"{g}/{s}"] = out
    xs = [x for x in feats if "gate_own_prev" in x] if "prev_w4" in ROWS else []
    logs_ = sorted({x["log"] for x in xs})
    S["gates_prev_w4"] = {}
    for k in ("gate_prev_kin", "gate_own_prev", "gate_forced_prev"):
        rho = float(spearmanr([x[k] for x in xs], [x["prev_gt_dist"] for x in xs])[0])
        taus = np.quantile([x[k] for x in xs], np.linspace(0.05, 1, 20))
        pick = lambda x, tau, m: x["u"]["prev_w4"][m] if x[k] < tau else x["u"]["normal"][m]   # noqa: E731
        res = []
        for lg in logs_:                                        # leave-one-log-out threshold choice (min mean FDE)
            tr = [x for x in xs if x["log"] != lg]
            best = min(taus, key=lambda tau: np.mean([pick(x, tau, "fde5") for x in tr]))
            res += [(x["g"], pick(x, best, "amplification"), pick(x, best, "fde5")) for x in xs if x["log"] == lg]
        S["gates_prev_w4"][k] = {"spearman_with_ref_error": rho} | {
            g: {"gated_amplification": float(np.mean([a for gg, a, _ in res if gg == g])),
                "gated_fde5": float(np.mean([f for gg, _, f in res if gg == g]))} for g in ("A-", "A+")}

    json.dump(S, open(os.path.join(args.run, "summary.json"), "w"), indent=1)
    with open(os.path.join(args.run, "units.jsonl"), "w") as f:
        for u in units:
            f.write(json.dumps(u) + "\n")

    L = [f"units {S['n_units']}  prev available {S['n_units_prev_available']}"]
    for g in ("A-", "A+"):
        b = S["groups"][g]
        L.append(f"\n== {g} ==  reference quality (ADE / FDE m, token==GT): " + "  ".join(
            f"{s}: {v['ade']:.2f}/{v['fde']:.2f}/{v['token_eq_gt']:.2f}" for s, v in b["reference_quality"].items()))
        L.append(f"{'row':12s} {'n':>5s} {'amplif %':>22s} {'FDE5':>6s} {'frac GT(amp)':>13s} {'frac GT(FDE)':>13s} {'p':>9s}")
        for c in ROWS:
            cd = b["conditions"][c]
            a = cd["amplification"]
            fr = b["frac_of_gt_same_window"].get(c, {})
            fa = "%.2f" % fr["amplification"]["frac"] if fr and fr.get("amplification") else "–"
            ff = "%.2f" % fr["fde5"]["frac"] if fr and fr.get("fde5") else "–"
            vn = b["vs_normal"].get(c, {}).get("amplification")
            p = "%.2g" % vn["mcnemar"]["p"] if vn else "–"
            L.append(f"{c:12s} {cd['n']:5d} {100*a['mean']:6.1f} [{100*a['ci95'][0]:5.1f},{100*a['ci95'][1]:5.1f}] "
                     f"{cd['fde5']['mean']:6.2f} {fa:>13s} {ff:>13s} {p:>9s}")
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(args.run, "analysis_console.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
