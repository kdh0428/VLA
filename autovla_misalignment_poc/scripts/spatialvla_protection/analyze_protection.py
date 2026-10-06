#!/usr/bin/env python
"""
Analysis of experiment 35 (outputs/spatialvla_feedback_protection_ablation/PROTOCOL.md). CPU only.

  python analyze_protection.py <out_dir>     -> analysis.json, curves.json, error_curves.png
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon

CFGS = ["r1e4", "r1e2", "r1e1", "r2e2", "r2e1", "r4e1"]
REDUCED = ["r1e2", "r1e1", "r2e2", "r2e1", "r4e1"]
MODES = ["natural", "feedback", "corrected", "reverse"]
TAU, HOLD, WIN_END, T = 0.02, 4, 24, 80
EXP34 = "/root/VLA/autovla_misalignment_poc/outputs/cross_domain_temporal_replication/closed_loop/episodes.jsonl"


def boot_mean(vals_by_key, reps=2000, seed=0):
    """vals_by_key: {(task, seed): value}; bootstrap stratified by task."""
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for (task, s), v in vals_by_key.items():
        by_task[task].append(v)
    bs = []
    for _ in range(reps):
        x = []
        for vs in by_task.values():
            x += [vs[rng.randrange(len(vs))] for _ in vs]
        bs.append(np.mean(x))
    bs.sort()
    return {"mean": float(np.mean(list(vals_by_key.values()))), "ci95": [float(bs[int(.025 * reps)]), float(bs[int(.975 * reps) - 1])],
            "n": len(vals_by_key)}


def mcnemar(a, b):
    g = int(sum(x > y for x, y in zip(a, b))); l_ = int(sum(y > x for x, y in zip(a, b)))
    return {"a_only": g, "b_only": l_, "p": float(binomtest(g, g + l_, 0.5).pvalue) if g + l_ else 1.0}


def wil(d):
    d = np.asarray(d, dtype=float)
    return float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0


def signflip(d, reps=10000, seed=0):
    d = np.asarray(d, dtype=float); rng = np.random.default_rng(seed)
    obs = abs(d.mean()); s = rng.choice([-1.0, 1.0], size=(reps, len(d)))
    return float((np.abs((s * d).mean(1)) >= obs - 1e-12).mean())


def holm(ps):
    order = np.argsort(ps); out = [0.0] * len(ps); m = len(ps); run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[j])); out[j] = run
    return out


def pos(ep):
    return np.array([r["pose"][:3] for r in ep["rec"]])


def quat(ep):
    return np.array([r["pose"][3:] for r in ep["rec"]])


def act_sum(ep):
    return np.cumsum(np.array([r["action"][:3] for r in ep["rec"]]), 0)


def pair_metrics(a, b):
    pa, pb = pos(a), pos(b); n = min(len(pa), len(pb))
    err = np.linalg.norm(pa[:n] - pb[:n], axis=1)
    qa, qb = quat(a)[:n], quat(b)[:n]
    ang = 2 * np.arccos(np.clip(np.abs((qa * qb).sum(1)), 0, 1))
    aa = np.array([r["action"][:3] for r in a["rec"]])[:n]; ab = np.array([r["action"][:3] for r in b["rec"]])[:n]
    adiff = np.linalg.norm(aa - ab, axis=1)
    ta = {r["t"]: r["exec"] for r in a["rec"] if "exec" in r}; tb = {r["t"]: r["exec"] for r in b["rec"] if "exec" in r}
    common = [t for t in ta if t in tb and t >= 8]
    cd = float(np.mean([np.mean([x[0] != y[0] for x, y in zip(ta[t], tb[t])]) for t in common])) if common else float("nan")
    re_t = None
    for t in range(WIN_END, n - HOLD + 1):
        if np.all(err[t:t + HOLD] <= TAU):
            re_t = t; break
    e24 = err[WIN_END]
    hl = next((t - WIN_END for t in range(WIN_END, n) if err[t] <= 0.5 * e24), None) if e24 > 1e-9 else 0
    return {"err": err, "ang": ang, "adiff": adiff,
            "traj_div": float(np.linalg.norm(act_sum(a)[-1] - act_sum(b)[-1])),
            "max_err": float(err.max()), "final_err": float(err[-1]), "mean_err": float(err[8:].mean()),
            "mean_ang": float(ang[8:].mean()), "final_ang": float(ang[-1]), "mean_adiff": float(adiff[8:].mean()),
            "chunk_disagree": cd, "realign_t": re_t, "realigned": float(re_t is not None), "half_life_ep": hl}


def ens_var(ep):
    v = [np.mean(np.sum((np.array(r["preds"]) - np.mean(r["preds"], 0)) ** 2, 1)) for r in ep["rec"] if r["t"] >= 8]
    return float(np.mean(v))


def failure_type(ep):
    fi, ev = ep["final_info"], ep["ever"]
    if ep["success"]:
        return "success"
    if "lifted_object" in fi:
        if not ev.get("is_grasped"):
            return "never_grasped"
        if not fi.get("is_grasped"):
            return "dropped_or_lost_grasp"
        return "grasped_not_lifted"
    if fi.get("moved_wrong_obj"):
        return "moved_wrong_obj"
    if not fi.get("all_obj_keep_height", True):
        return "object_lifted_or_fell"
    if not fi.get("moved_correct_obj"):
        return "target_not_moved"
    return "not_near_target"


def main() -> None:
    root = sys.argv[1]
    eps = [json.loads(l) for l in open(os.path.join(root, "rollouts", "episodes.jsonl"))]
    E = {(r["cond"], r["task"], r["seed"]): r for r in eps}
    keys = sorted({(r["task"], r["seed"]) for r in eps})
    res = {"n_episodes": len(eps), "configs": {}}
    curves = {}
    for cfg in CFGS:
        have = [m for m in MODES + ["natural_gen"] if all((f"{cfg}:{m}", *k) in E for k in keys)]
        if not all(m in have for m in MODES):
            continue
        g = lambda m, k: E[(f"{cfg}:{m}", *k)]
        R = {"success": {}, "success_by_task": {}, "failure_types": {}}
        for m in have:
            R["success"][m] = boot_mean({k: float(g(m, k)["success"]) for k in keys})
            R["success_by_task"][m] = {t: float(np.mean([g(m, k)["success"] for k in keys if k[0] == t])) for t in sorted({k[0] for k in keys})}
            ft = defaultdict(int)
            for k in keys:
                ft[failure_type(g(m, k))] += 1
            R["failure_types"][m] = dict(ft)
        S = {m: [float(g(m, k)["success"]) for k in keys] for m in have}
        con = {}
        for name, a, b in (("C-F", "corrected", "feedback"), ("N-R", "natural", "reverse"), ("N-F", "natural", "feedback"),
                           ("N-C", "natural", "corrected"), ("gen-N", "natural_gen", "natural")):
            if a in S and b in S:
                c = boot_mean({k: S[a][i] - S[b][i] for i, k in enumerate(keys)})
                c["mcnemar"] = mcnemar(S[a], S[b])
                c["by_task"] = {t: float(np.mean([S[a][i] - S[b][i] for i, k in enumerate(keys) if k[0] == t])) for t in sorted({k[0] for k in keys})}
                con[name] = c
        R["contrasts"] = con
        R["discordance_vs_natural"] = {m: float(np.mean([S[m][i] != S["natural"][i] for i in range(len(keys))])) for m in have if m != "natural"}
        # trajectory metrics
        pm = {}
        pairs = {"F_vs_C": ("feedback", "corrected"), "R_vs_N": ("reverse", "natural"), "F_vs_N": ("feedback", "natural"),
                 "C_vs_N": ("corrected", "natural")}
        if "natural_gen" in have:
            pairs["gen_vs_N"] = ("natural_gen", "natural")
        per_ep = {}
        for pname, (a, b) in pairs.items():
            ms = {k: pair_metrics(g(a, k), g(b, k)) for k in keys}
            per_ep[pname] = ms
            out = {}
            for f in ("traj_div", "max_err", "final_err", "mean_err", "mean_ang", "final_ang", "mean_adiff", "chunk_disagree", "realigned"):
                vals = {k: v[f] for k, v in ms.items() if not np.isnan(v[f])}
                out[f] = boot_mean(vals) if vals else None
            rts = [v["realign_t"] for v in ms.values() if v["realign_t"] is not None]
            out["realign_t_median"] = float(np.median(rts)) if rts else None
            hls = [v["half_life_ep"] for v in ms.values()]
            out["half_life_ep_median"] = float(np.median([h if h is not None else T for h in hls]))
            out["half_life_ep_not_reached"] = float(np.mean([h is None for h in hls]))
            mc = np.mean([v["err"][:T] for v in ms.values() if len(v["err"]) >= T], 0)
            out["half_life_meancurve"] = next((t - WIN_END for t in range(WIN_END, T) if mc[t] <= 0.5 * mc[WIN_END]), None)
            out["err_at"] = {str(t): float(mc[t]) for t in (8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 79)}
            curves.setdefault(cfg, {})[pname] = {"pos_err": mc.tolist(),
                                                 "ang": np.mean([v["ang"][:T] for v in ms.values()], 0).tolist(),
                                                 "adiff": np.mean([v["adiff"][:T] for v in ms.values()], 0).tolist()}
            pm[pname] = out
        R["trajectory"] = pm
        R["ensemble_var"] = {m: boot_mean({k: ens_var(g(m, k)) for k in keys}) for m in MODES}
        R["_per_ep"] = per_ep
        R["_S"] = S
        res["configs"][cfg] = R
    # Holm over reduced configs, interactions vs r1e4, persistence vs r1e4
    base = res["configs"].get("r1e4")
    red = [c for c in REDUCED if c in res["configs"]]
    for name in ("C-F", "N-R"):
        ps = [res["configs"][c]["contrasts"][name]["mcnemar"]["p"] for c in red]
        for c, h in zip(red, holm(ps)):
            res["configs"][c]["contrasts"][name]["holm_p"] = h
    if base:
        for c in red:
            R = res["configs"][c]; inter = {}
            for name, a, b in (("C-F", "corrected", "feedback"), ("N-R", "natural", "reverse")):
                d = [(R["_S"][a][i] - R["_S"][b][i]) - (base["_S"][a][i] - base["_S"][b][i]) for i in range(len(keys))]
                x = boot_mean({k: d[i] for i, k in enumerate(keys)}); x["signflip_p"] = signflip(d)
                inter[name] = x
            for pname in ("F_vs_C", "R_vs_N"):
                for f in ("mean_err", "traj_div", "final_err"):
                    d = [R["_per_ep"][pname][k][f] - base["_per_ep"][pname][k][f] for k in keys]
                    x = boot_mean({k: d[i] for i, k in enumerate(keys)}); x["wilcoxon_p"] = wil(d)
                    inter[f"{pname}:{f}"] = x
                d = [(R["_per_ep"][pname][k]["realign_t"] or T) - (base["_per_ep"][pname][k]["realign_t"] or T) for k in keys]
                x = boot_mean({k: d[i] for i, k in enumerate(keys)}); x["wilcoxon_p"] = wil(d)
                inter[f"{pname}:realign_t(unrealigned=80)"] = x
            R["vs_r1e4"] = inter
    # A: reproduction against experiment 34 (read only)
    if base and os.path.exists(EXP34):
        old = {(r["cond"], r["task"], r["seed"]): r["success"] for r in map(json.loads, open(EXP34))}
        mp = {"natural": "natural", "feedback": "feedback_opposite", "corrected": "corrected_opposite", "reverse": "reverse_opposite"}
        res["A_vs_exp34"] = {m: {"exp34": float(np.mean([old[(mp[m], *k)] for k in keys])), "now": base["success"][m]["mean"],
                                 "episode_agreement": float(np.mean([old[(mp[m], *k)] == base["_S"][m][i] for i, k in enumerate(keys)]))}
                             for m in MODES}
    for R in res["configs"].values():
        R.pop("_per_ep"); R.pop("_S")
    json.dump(res, open(os.path.join(root, "analysis.json"), "w"), indent=1, default=str)
    json.dump(curves, open(os.path.join(root, "curves.json"), "w"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
        cols = {"r1e4": "#1f4e9c", "r1e2": "#5b8bd6", "r1e1": "#9fbfee", "r2e2": "#b5651d", "r2e1": "#e8a15c", "r4e1": "#c0392b"}
        for ax, pname, title in ((axs[0], "F_vs_C", "feedback vs corrected (context-only effect)"),
                                 (axs[1], "R_vs_N", "reverse vs natural (context-only effect)")):
            for cfg in CFGS:
                if cfg in curves:
                    ax.plot(np.array(curves[cfg][pname]["pos_err"]) * 100, label=cfg, color=cols[cfg], lw=2)
            ax.axvspan(8, 24, color="#999", alpha=0.15, lw=0); ax.set_title(title, fontsize=10)
            ax.set_xlabel("control step"); ax.grid(alpha=0.3)
        axs[0].set_ylabel("mean TCP position difference (cm)"); axs[1].legend(fontsize=8, frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(root, "error_curves.png"), dpi=130)
    except Exception as e:  # noqa: BLE001
        print("plot failed", e)
    print(json.dumps({c: {m: round(v["mean"], 3) for m, v in R["success"].items()} for c, R in res["configs"].items()}, indent=1))


if __name__ == "__main__":
    main()
