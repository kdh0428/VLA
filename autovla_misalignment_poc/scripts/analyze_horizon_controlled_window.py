#!/usr/bin/env python
"""
Analysis of the horizon-controlled correction window (CPU only).

RELEASE-ALIGNED EVALUATION (removes the horizon confound)
  A window of length w (Normal: w = 0) last conditions a_{t*+w+1} on GT -> release pose
  r_w = t*+w+1. Every window is evaluated over exactly N free poses after its release,
  r_w+1 .. r_w+N (0.5 s each). For each w, Normal AR and Full GT-history are evaluated on the
  SAME poses (paired references), so win_w, Normal and Full share the free horizon AND the time.
  f_N(w) = (Normal - win_w) / (Normal - Full) on those poses.

TWO EVALUATION HORIZONS
  in-distribution (PRIMARY): only poses/tokens 0..9 (the trained 10-token plan). Feasible
    r_w + N <= 9: t* = 0 allows w <= 4 with N <= 4; t* = 1 allows w <= 4 with N <= 3.
  extended (SECONDARY): 20-token rollouts vs the 20-pose log GT, N up to 10. The model puts
    ~0 probability on action tokens for the first steps past the 10th token (it wants to end the
    answer), so the forced tokens there are out of distribution; reported only with diagnostics.

Metrics over the free window: fde (L2 at r_w+N), ade, amp (fde > 3 m; 2 m / 1 m sensitivity for
the short in-distribution windows), growth (slope of L2 over r_w..r_w+N, m/step), entropy (mean
over outputs r_w+1..r_w+N), realign (token == GT over those outputs), stab (L2 at r_w <= 1 m),
rediverge (stab AND fde > threshold).
Horizon-artefact test: for a fixed w, f_N(w) as N grows. If the short-window effect came from a
short remaining horizon, f_N(w) should fall as N increases.
Stats: paired McNemar / Wilcoxon, log-cluster bootstrap 95% CI; A- and A+ separately.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(POC_DIR, "scripts"))
from analyze_prev_action_state_patching import Boot  # noqa: E402

H_EXT = 20
WS_ALL = [1, 2, 3, 4, 5, 6]
MET = ["amp", "amp2", "amp1", "fde", "ade", "growth", "entropy", "realign", "stab", "rediverge", "rediverge2"]
BIN = {"amp", "amp2", "amp1", "stab", "rediverge", "rediverge2"}
# (label, horizon, N list, windows used for the critical window)
CONFIGS = [("in10", 10, [1, 2, 3, 4], [1, 2, 3, 4]), ("ext20", 20, [4, 6, 8, 10], [1, 2, 3, 4, 5, 6])]


def free_metrics(row, gt_traj, gt_tok, t, w, N, Hev):
    r = t + w + 1
    if r + N > Hev - 1:
        return None
    p = np.asarray(row["trajectory_pred"], float)[:Hev]
    g = np.asarray(gt_traj, float)[:Hev]
    e = np.linalg.norm(p - g, axis=1)
    poses = list(range(r + 1, r + N + 1))
    fde = float(e[r + N]); stab = bool(e[r] <= 1.0)
    return {"amp": fde > 3.0, "amp2": fde > 2.0, "amp1": fde > 1.0, "fde": fde, "ade": float(e[poses].mean()),
            "growth": float(np.polyfit(np.arange(r, r + N + 1), e[r:r + N + 1], 1)[0]),
            "entropy": float(np.mean([row["ent"][k - t - 1] for k in poses])),
            "realign": float(np.mean([row["action_idx"][k] == gt_tok[k] for k in poses])),
            "stab": stab, "rediverge": bool(stab and fde > 3.0), "rediverge2": bool(stab and fde > 2.0),
            "err_release": float(e[r]), "err_curve": [float(x) for x in e[r:r + N + 1]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(POC_DIR, "outputs/horizon_controlled_window"))
    ap.add_argument("--narrative", default=None)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.run, "run_meta.json")))
    recs = [json.loads(l) for l in open(os.path.join(args.run, "records.jsonl"))]
    ep = os.path.join(args.run, "errors.jsonl")
    errs = [json.loads(l) for l in open(ep)] if os.path.exists(ep) else []

    units = []
    for x in recs:
        t, g20, tok20 = x["t_star"], x["trajectory_gt20"], x["gt20"]
        u = {"token": x["token"], "log": x["log"], "group": x["group"], "pert": x["perturbation"], "t_star": t, "ev": {}}
        rows = x["rows"]
        for lab, Hev, NS, _ in CONFIGS:
            for N in NS:
                for w in [0] + WS_ALL:
                    name = "normal" if w == 0 else f"win{w}"
                    mw = free_metrics(rows[name], g20, tok20, t, w, N, Hev)
                    if mw is None:
                        continue
                    u["ev"][(lab, N, w)] = {"win": mw, "normal": free_metrics(rows["normal"], g20, tok20, t, w, N, Hev),
                                            "full": free_metrics(rows["full"], g20, tok20, t, w, N, Hev)}
        a = {n: rows[n]["action_idx"] for n in rows}
        p = x.get("prior_temporal_feedback_window") or {}
        u["checks"] = {
            "prefix_identical": all(v[:t + 1] == a["normal"][:t + 1] for v in a.values()),
            "first_output_identical": all(v[t + 1] == a["normal"][t + 1] for v in a.values()),
            "gt20_first10_eq_stored": x["gt20"][:10] == x["gt10_stored"],
            "gt_traj_first10_eq_stored": bool(np.abs(np.asarray(x["trajectory_gt20"])[:10] - np.asarray(x["trajectory_gt10_stored"])[:, :2]).max() < 1e-4),
            "win_saturated_first10_eq_full": all(a[f"win{w}"][:10] == a["full"][:10] for w in (5, 6, 8) if w >= 8 - t),
        }
        u["repro10"] = {c: (a[c2][:10] == p[c]) if c in p else None for c, c2 in
                        (("normal", "normal"), ("win1", "win1"), ("win2", "win2"), ("win3", "win3"), ("win4", "win4"), ("gt_history", "full"))}
        u["diag"] = {n: {"mass_by_k": rows[n]["p_action_mass"], "ent_by_k": rows[n]["ent"],
                         "top_non_by_k": [tn[0] for tn in rows[n]["top_non_action"]]} for n in ("normal", "full", "win4")}
        u["diag"]["full_err_by_pose"] = np.linalg.norm(np.asarray(rows["full"]["trajectory_pred"]) - np.asarray(g20), axis=1).tolist()
        u["diag"]["normal_err_by_pose"] = np.linalg.norm(np.asarray(rows["normal"]["trajectory_pred"]) - np.asarray(g20), axis=1).tolist()
        units.append(u)

    groups = ("A-", "A+")
    S = {"n_units": {g: sum(u["group"] == g for u in units) for g in groups},
         "n_scenes": {g: len({u["token"] for u in units if u["group"] == g}) for g in groups},
         "t_star": {g: {t: sum(u["group"] == g and u["t_star"] == t for u in units) for t in (0, 1)} for g in groups},
         "n_errors": len(errs), "errors": errs, "rows": meta["rows"]}
    S["sanity"] = {k: float(np.mean([u["checks"][k] for u in units])) for k in units[0]["checks"]}
    for c in ("normal", "win1", "win2", "win3", "win4", "gt_history"):
        v = [u["repro10"][c] for u in units if u["repro10"][c] is not None]
        S["sanity"][f"repro_tfw_{c}"] = float(np.mean(v)) if v else None

    # ---- extension diagnostics by absolute action step k
    D = {}
    for g in groups:
        us = [u for u in units if u["group"] == g]
        D[g] = {}
        for n in ("normal", "full", "win4"):
            mass = defaultdict(list); ent = defaultdict(list); top = defaultdict(Counter)
            for u in us:
                t = u["t_star"]
                for i, (m_, e_, tn) in enumerate(zip(u["diag"][n]["mass_by_k"], u["diag"][n]["ent_by_k"], u["diag"][n]["top_non_by_k"])):
                    k = t + 1 + i
                    mass[k].append(m_); ent[k].append(e_); top[k][tn] += 1
            D[g][n] = {"mass_by_k": {k: float(np.mean(v)) for k, v in sorted(mass.items())},
                       "ent_by_k": {k: float(np.mean(v)) for k, v in sorted(ent.items())},
                       "top_non_action_by_k": {k: top[k].most_common(1)[0] for k in sorted(top)}}
        D[g]["full_err_by_pose"] = np.mean([u["diag"]["full_err_by_pose"] for u in us], 0).tolist()
        D[g]["normal_err_by_pose"] = np.mean([u["diag"]["normal_err_by_pose"] for u in us], 0).tolist()
    S["extension_diagnostics"] = D

    def paired(uw, key_a, key_b, lab, N, w, m, seed):
        ok = [u for u in uw if u["ev"][(lab, N, w)][key_a] is not None and u["ev"][(lab, N, w)][key_b] is not None]
        if not ok:
            return None
        A_ = [u["ev"][(lab, N, w)][key_a][m] for u in ok]; B_ = [u["ev"][(lab, N, w)][key_b][m] for u in ok]
        c = Boot([u["log"] for u in ok], seed=seed)([float(a) - float(b) for a, b in zip(A_, B_)])
        if m in BIN:
            gn = sum(bool(a) and not bool(b) for a, b in zip(A_, B_)); ls = sum(bool(b) and not bool(a) for a, b in zip(A_, B_))
            c["mcnemar"] = {"a_only": gn, "b_only": ls, "p": 1.0 if gn + ls == 0 else float(binomtest(gn, gn + ls, .5).pvalue)}
        else:
            dd = np.array([float(a) - float(b) for a, b in zip(A_, B_)])
            c["wilcoxon_p"] = float(wilcoxon(dd).pvalue) if np.any(dd != 0) else 1.0
        return c

    def block(us, lab, NS, WC, seed=0):
        if not us:
            return None
        out = {"n": len(us), "n_scenes": len({u["token"] for u in us}), "N": {}}
        for N in NS:
            BN = {}
            for w in [0] + WS_ALL:
                uw = [u for u in us if (lab, N, w) in u["ev"]]
                if not uw:
                    continue
                bt = Boot([u["log"] for u in uw], seed=seed)
                d = {"n": len(uw), "n_scenes": len({u["token"] for u in uw}), "units": [(u["token"], u["pert"]) for u in uw],
                     "level": {}, "normal_same_poses": {}, "full_same_poses": {}, "vs_normal": {}, "vs_full": {}}
                for m in MET:
                    d["level"][m] = bt([u["ev"][(lab, N, w)]["win"][m] for u in uw])
                    d["normal_same_poses"][m] = bt([u["ev"][(lab, N, w)]["normal"][m] for u in uw])
                    d["full_same_poses"][m] = bt([u["ev"][(lab, N, w)]["full"][m] for u in uw])
                    if w > 0:
                        d["vs_normal"][m] = paired(uw, "win", "normal", lab, N, w, m, seed)
                    d["vs_full"][m] = paired(uw, "win", "full", lab, N, w, m, seed)
                st = [u for u in uw if u["ev"][(lab, N, w)]["win"]["stab"]]
                d["rediverge_given_stab"] = Boot([u["log"] for u in st], seed=seed)([u["ev"][(lab, N, w)]["win"]["rediverge2"] for u in st]) if st else None
                stf = [u for u in uw if u["ev"][(lab, N, w)]["full"]["stab"]]
                d["rediverge_given_stab_full"] = Boot([u["log"] for u in stf], seed=seed)([u["ev"][(lab, N, w)]["full"]["rediverge2"] for u in stf]) if stf else None
                d["fraction"] = {}
                nc = len(bt.labels)
                for m in ("amp", "amp2", "amp1", "fde", "ade", "growth"):
                    nv = np.array([float(u["ev"][(lab, N, w)]["normal"][m]) for u in uw]); wv = np.array([float(u["ev"][(lab, N, w)]["win"][m]) for u in uw])
                    fv = np.array([float(u["ev"][(lab, N, w)]["full"][m]) for u in uw])
                    sn = np.bincount(bt.cid, weights=nv - wv, minlength=nc); sg = np.bincount(bt.cid, weights=nv - fv, minlength=nc)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        bs = sn[bt.R].sum(1) / sg[bt.R].sum(1)
                    fin = bs[np.isfinite(bs)]
                    d["fraction"][m] = {"mean": float(sn.sum() / sg.sum()) if abs(sg.sum()) > 1e-12 else None, "boot": bs,
                                        "ci95": [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))] if len(fin) else None}
                BN[w] = d
            crit = {}
            wc = [w for w in WC if w in BN]
            same = len(wc) == len(WC) and len({frozenset(BN[w]["units"]) for w in wc}) == 1
            if same:
                for m in ("amp", "amp2", "fde", "ade"):
                    pts = {w: BN[w]["fraction"][m]["mean"] for w in wc}
                    pt = next((w for w in wc if pts[w] is not None and pts[w] >= 0.9), None)
                    R = len(BN[wc[0]]["fraction"][m]["boot"])
                    reps = np.array([next((w for w in wc if np.isfinite(BN[w]["fraction"][m]["boot"][i]) and BN[w]["fraction"][m]["boot"][i] >= 0.9), 99) for i in range(R)])
                    crit[m] = {"point": pt, "boot_median": float(np.median(reps)), "boot_ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
                               "boot_dist": {str(w): float(np.mean(reps == w)) for w in wc + [99]}}
            for w in BN:
                BN[w].pop("units")
                for m in BN[w]["fraction"]:
                    BN[w]["fraction"][m].pop("boot", None)
            out["N"][N] = {"windows": BN, "critical": crit, "critical_same_units": same}
        return out

    S["analyses"] = {}
    for lab, Hev, NS, WC in CONFIGS:
        S["analyses"][lab] = {}
        for sname, pred in (("t*<=1", lambda u: True), ("t*=0", lambda u: u["t_star"] == 0), ("t*=1", lambda u: u["t_star"] == 1)):
            S["analyses"][lab][sname] = {g: block([u for u in units if u["group"] == g and pred(u)], lab, NS, WC) for g in groups}

    # horizon-artefact test: fixed w, growing N (in-distribution, t*=0 units only so the unit set is fixed per w)
    trend = {}
    for g in groups:
        us0 = [u for u in units if u["group"] == g and u["t_star"] == 0]
        trend[g] = {}
        for w in (1, 2, 3, 4):
            Nmax = 8 - w                      # r_w + N <= 9 with t* = 0
            common = [u for u in us0 if all(("in10", N, w) in u["ev"] or False for N in [])]
            res = {}
            for N in range(1, Nmax + 1):
                ev = []
                for u in us0:
                    mw = free_metrics_cache(u, w, N)
                    if mw:
                        ev.append((u, mw))
                if not ev:
                    continue
                bt = Boot([u["log"] for u, _ in ev], seed=4)
                nc = len(bt.labels)
                res[N] = {"n": len(ev)}
                for m in ("amp2", "amp1", "fde", "ade"):
                    nv = np.array([float(e_["normal"][m]) for _, e_ in ev]); wv = np.array([float(e_["win"][m]) for _, e_ in ev]); fv = np.array([float(e_["full"][m]) for _, e_ in ev])
                    sn = np.bincount(bt.cid, weights=nv - wv, minlength=nc); sg = np.bincount(bt.cid, weights=nv - fv, minlength=nc)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        bs = sn[bt.R].sum(1) / sg[bt.R].sum(1)
                    fin = bs[np.isfinite(bs)]
                    res[N][m] = {"fraction": float(sn.sum() / sg.sum()) if abs(sg.sum()) > 1e-12 else None,
                                 "ci95": [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))] if len(fin) else None,
                                 "win": float(wv.mean()), "normal": float(nv.mean()), "full": float(fv.mean())}
                # slope of fraction vs N via bootstrap (fde)
            trend[g][w] = res
    S["horizon_trend_tstar0"] = trend
    json.dump(S, open(os.path.join(args.run, "summary.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    with open(os.path.join(args.run, "units.jsonl"), "w") as f:
        for u in units:
            uu = {k: v for k, v in u.items() if k != "diag"}; uu["ev"] = {f"{a_}_N{N}_w{w}": v for (a_, N, w), v in u["ev"].items()}
            f.write(json.dumps(uu, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    make_figures_and_report(args, S, units, D)


_RECS = {}


def free_metrics_cache(u, w, N):
    r = _RECS[(u["token"], u["pert"])]
    t = r["t_star"]
    name = "normal" if w == 0 else f"win{w}"
    mw = free_metrics(r["rows"][name], r["trajectory_gt20"], r["gt20"], t, w, N, 10)
    if mw is None:
        return None
    return {"win": mw, "normal": free_metrics(r["rows"]["normal"], r["trajectory_gt20"], r["gt20"], t, w, N, 10),
            "full": free_metrics(r["rows"]["full"], r["trajectory_gt20"], r["gt20"], t, w, N, 10)}


def make_figures_and_report(args, S, units, D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = ("A-", "A+")
    fd = os.path.join(args.run, "figures"); os.makedirs(fd, exist_ok=True)
    IN = S["analyses"]["in10"]; EX = S["analyses"]["ext20"]

    def pc(d):
        return "–" if d is None else f"{100*d['mean']:.1f}% [{100*d['ci95'][0]:.1f}, {100*d['ci95'][1]:.1f}]"

    def nm(d, nd=2):
        return "–" if d is None else f"{d['mean']:.{nd}f} [{d['ci95'][0]:.{nd}f}, {d['ci95'][1]:.{nd}f}]"

    def dl(d, m):
        if d is None:
            return "–"
        if m in BIN:
            return f"{100*d['mean']:+.1f}%p [{100*d['ci95'][0]:+.1f}, {100*d['ci95'][1]:+.1f}] p={d['mcnemar']['p']:.2g}"
        if m == "realign":
            return f"{100*d['mean']:+.1f}%p [{100*d['ci95'][0]:+.1f}, {100*d['ci95'][1]:+.1f}] p={d['wilcoxon_p']:.2g}"
        nd = 3 if m in ("growth", "entropy") else 2
        return f"{d['mean']:+.{nd}f} [{d['ci95'][0]:+.{nd}f}, {d['ci95'][1]:+.{nd}f}] p={d['wilcoxon_p']:.2g}"

    def fr(d):
        return "–" if not d or d.get("mean") is None else (f"{100*d['mean']:.0f}%" + (f" [{100*d['ci95'][0]:.0f}, {100*d['ci95'][1]:.0f}]" if d.get("ci95") else ""))

    # fig 1: in-distribution, t*=0, N=4 (w<=4 share the same units)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, m, lab_, sc in ((axs[0], "fde", "FDE at release+N (m)", 1), (axs[1], "amp2", "FDE_N > 2 m (%)", 100), (axs[2], "growth", "error growth (m/step)", 1)):
        for g, col in (("A-", "tab:red"), ("A+", "tab:blue")):
            if not IN["t*=0"][g]:
                continue
            B = IN["t*=0"][g]["N"][4]["windows"]
            xs = [w for w in (0, 1, 2, 3, 4) if w in B]
            ax.plot(xs, [B[w]["level"][m]["mean"] * sc for w in xs], "-o", color=col, label=f"{g} window (own release)")
            ax.fill_between(xs, [B[w]["level"][m]["ci95"][0] * sc for w in xs], [B[w]["level"][m]["ci95"][1] * sc for w in xs], color=col, alpha=.12)
            ax.plot(xs, [B[w]["normal_same_poses"][m]["mean"] * sc for w in xs], ":x", color=col, label=f"{g} Normal, same poses")
            ax.plot(xs, [B[w]["full_same_poses"][m]["mean"] * sc for w in xs], "--s", color=col, ms=3, label=f"{g} Full, same poses")
        ax.set_xlabel("correction window w (0 = Normal)"); ax.set_title(f"{lab_}\nt*=0, N=4 free steps after release for every w (in-distribution)", fontsize=9); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig1_equal_free_horizon_in_distribution.png"), dpi=150); plt.close(fig)

    # fig 2: horizon-artefact test (fraction vs N for fixed w)
    T = S["horizon_trend_tstar0"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for r_, g in enumerate(groups):
        for c_, m in enumerate(("fde", "ade")):
            ax = axs[r_, c_]
            for w, col in zip((1, 2, 3, 4), ("tab:gray", "tab:green", "tab:orange", "k")):
                Ns = sorted(T[g][w]); ys = [T[g][w][N][m]["fraction"] for N in Ns]
                ax.plot(Ns, ys, "-o", color=col, label=f"w={w}")
                ax.fill_between(Ns, [T[g][w][N][m]["ci95"][0] for N in Ns], [T[g][w][N][m]["ci95"][1] for N in Ns], color=col, alpha=.08)
            ax.axhline(.9, color="gray", ls="--", lw=1); ax.axhline(1, color="gray", ls=":", lw=1); ax.set_ylim(-.2, 1.3)
            ax.set_xlabel("N free steps after release"); ax.set_title(f"{g} t*=0: fraction of Normal→Full effect ({m}) vs free horizon", fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig2_fraction_vs_free_horizon_in_distribution.png"), dpi=150); plt.close(fig)

    # fig 3: extension diagnostics
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
    for g, ls in zip(groups, ("-", "--")):
        for n, col in (("normal", "tab:red"), ("full", "tab:blue"), ("win4", "tab:green")):
            ks = sorted(D[g][n]["mass_by_k"], key=int)
            axs[0].plot([int(k) for k in ks], [D[g][n]["mass_by_k"][k] for k in ks], ls, color=col, label=f"{g} {n}")
            axs[1].plot([int(k) for k in ks], [D[g][n]["ent_by_k"][k] for k in ks], ls, color=col, label=f"{g} {n}")
        axs[2].plot(range(H_EXT), D[g]["full_err_by_pose"], ls, color="tab:blue", label=f"{g} Full")
        axs[2].plot(range(H_EXT), D[g]["normal_err_by_pose"], ls, color="tab:red", label=f"{g} Normal")
    for ax, tt in zip(axs, ("P(action token) over full vocabulary", "action-row entropy (nats)", "mean L2 error to 10-s log GT (m)")):
        ax.axvline(9.5, color="gray", ls=":"); ax.set_xlabel("action step k / pose"); ax.set_title(tt + "\n(dotted: end of the trained 10-token plan)", fontsize=9); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig3_extension_out_of_distribution.png"), dpi=150); plt.close(fig)

    # fig 4: extended (secondary) N=10
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, m, sc, lab_ in ((axs[0], "fde", 1, "FDE at release+10 (m)"), (axs[1], "amp", 100, "FDE > 3 m (%)")):
        for g, col in (("A-", "tab:red"), ("A+", "tab:blue")):
            if not EX["t*<=1"][g]:
                continue
            B = EX["t*<=1"][g]["N"][10]["windows"]; xs = [w for w in [0] + WS_ALL if w in B]
            ax.plot(xs, [B[w]["level"][m]["mean"] * sc for w in xs], "-o", color=col, label=f"{g} window")
            ax.plot(xs, [B[w]["normal_same_poses"][m]["mean"] * sc for w in xs], ":x", color=col, label=f"{g} Normal same poses")
            ax.plot(xs, [B[w]["full_same_poses"][m]["mean"] * sc for w in xs], "--s", color=col, ms=3, label=f"{g} Full same poses")
        ax.set_xlabel("w"); ax.set_title(f"[SECONDARY, out-of-distribution past 10 tokens] {lab_}, N=10", fontsize=8); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig4_extended_horizon_secondary.png"), dpi=150); plt.close(fig)

    NAR = json.load(open(args.narrative)) if args.narrative and os.path.exists(args.narrative) else {}
    W_ = []; w_ = W_.append
    w_("# AutoVLA Natural/Fast — Horizon-Controlled Critical Window\n")
    w_("## 1. 결론\n"); w_(NAR.get("conclusion", "_(narrative 미작성)_") + "\n")

    w_("## 2. 요약 표 — window 길이별, 교정 종료 후 free step 수 동일\n")
    w_("**주 분석 (학습된 10-token 계획 안, t*=0, 모든 window에 N = 4 free step = 2 s).** 모든 window가 같은 단위 집합입니다. "
       "Normal/Full 값은 해당 window와 **같은 pose 구간**에서 잰 쌍대 기준값입니다. [ ]는 log cluster bootstrap 95% CI.\n")
    for g in groups:
        if not IN["t*=0"][g] or 4 not in IN["t*=0"][g]["N"][4]["windows"]:
            w_(f"### {g}: t*=0 단위 없음\n")
            continue
        B = IN["t*=0"][g]["N"][4]
        w_(f"### {g} (t*=0, 단위 {B['windows'][1]['n']}, 장면 {B['windows'][1]['n_scenes']})\n")
        w_("| window | FDE@release+4 (m) | 같은 구간 Normal | 같은 구간 Full | Full 효과 대비 (FDE) | ADE | Full 효과 대비 (ADE) | ΔFDE vs Full | FDE>2 m | 같은 구간 Normal / Full | error growth (m/step) | post-correction entropy | GT re-alignment | release 시 안정 (≤1 m) | 안정 후 재발산 (>2 m) |")
        w_("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for w in (0, 1, 2, 3, 4):
            if w not in B["windows"]:
                continue
            d = B["windows"][w]; L = d["level"]
            w_(f"| {'Normal AR' if w == 0 else f'{w}-step'} | {nm(L['fde'])} | {nm(d['normal_same_poses']['fde'])} | {nm(d['full_same_poses']['fde'])} | "
               f"{fr(d['fraction']['fde']) if w else '–'} | {nm(L['ade'])} | {fr(d['fraction']['ade']) if w else '–'} | {dl(d['vs_full']['fde'], 'fde')} | {pc(L['amp2'])} | "
               f"{100*d['normal_same_poses']['amp2']['mean']:.1f} / {100*d['full_same_poses']['amp2']['mean']:.1f}% | {nm(L['growth'], 3)} | {nm(L['entropy'], 3)} | {pc(L['realign'])} | {pc(L['stab'])} | {pc(d['rediverge_given_stab'])} |")
        d4 = B["windows"][4]
        w_(f"| Full GT-history (w=4 구간) | {nm(d4['full_same_poses']['fde'])} | | | 100% | {nm(d4['full_same_poses']['ade'])} | 100% | | {pc(d4['full_same_poses']['amp2'])} | | "
           f"{nm(d4['full_same_poses']['growth'], 3)} | {nm(d4['full_same_poses']['entropy'], 3)} | {pc(d4['full_same_poses']['realign'])} | {pc(d4['full_same_poses']['stab'])} | {pc(d4['rediverge_given_stab_full'])} |")
        w_("")
        for m in ("fde", "ade", "amp2"):
            x = B["critical"].get(m)
            if x:
                ds = ", ".join(f"{k if k != '99' else '>4'}:{100*v:.0f}%" for k, v in x["boot_dist"].items() if v >= .02)
                w_(f"- Full 효과 90% 도달 최소 window ({m}): **{x['point'] if x['point'] else '>4'}** (bootstrap 중앙값 {x['boot_median']:.0f}, 95% [{x['boot_ci95'][0]:.0f}, {x['boot_ci95'][1]:.0f}]; {ds})")
        w_("")
    w_("![fig1](figures/fig1_equal_free_horizon_in_distribution.png)\n")

    w_("## 3. Horizon 교란 검정: window 고정, free horizon N 증가 (t*=0, 10-token 안)\n")
    w_("짧은 remaining horizon 때문에 생긴 효과라면, 같은 window에서 N이 늘수록 Full 효과 대비 비율이 떨어져야 합니다. 셀: 비율 [95% CI] (win / Normal / Full FDE m).\n")
    w_("![fig2](figures/fig2_fraction_vs_free_horizon_in_distribution.png)\n")
    T = S["horizon_trend_tstar0"]
    for g in groups:
        w_(f"### {g}\n")
        allN = sorted({N for w in T[g] for N in T[g][w]})
        w_("| window | " + " | ".join(f"N={N}" for N in allN) + " |")
        w_("|---|" + "---|" * len(allN))
        for w in (1, 2, 3, 4):
            cells = []
            for N in allN:
                x = T[g][w].get(N)
                cells.append("–" if not x else f"{fr({'mean': x['fde']['fraction'], 'ci95': x['fde']['ci95']})} ({x['fde']['win']:.2f} / {x['fde']['normal']:.2f} / {x['fde']['full']:.2f})")
            w_(f"| {w}-step | " + " | ".join(cells) + " |")
        w_("")

    w_("## 4. 실험 설정\n"); w_(NAR.get("setup", "") + "\n")
    w_(f"- 단위: A− {S['n_units']['A-']} / A+ {S['n_units']['A+']} (장면 {S['n_scenes']['A-']} / {S['n_scenes']['A+']}; t*=0 {S['t_star']['A-'][0]} / {S['t_star']['A+'][0]}, t*=1 {S['t_star']['A-'][1]} / {S['t_star']['A+'][1]}). "
       f"제외 {S['n_errors']}: " + "; ".join(f"{e.get('token')}({e.get('group')}): {e.get('error')}" for e in S["errors"]) + "\n")

    w_("## 5. 주 분석 상세: 쌍대 검정 (10-token 안)\n")
    for sname, N in (("t*=0", 4), ("t*<=1", 3)):
        for g in groups:
            if not IN[sname][g]:
                continue
            B = IN[sname][g]["N"][N]["windows"]
            w_(f"### {g}, {sname}, N = {N}\n")
            w_("| window | n | FDE | Normal / Full 같은 구간 | ΔFDE vs Normal | ΔFDE vs Full | ΔADE vs Full | Δgrowth vs Full | Δentropy vs Full | Δre-align vs Full | ΔFDE>2 m vs Full | Full 효과 대비 (FDE / ADE) |")
            w_("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for w in (1, 2, 3, 4):
                if w not in B:
                    continue
                d = B[w]
                w_(f"| {w}-step | {d['n']} | {nm(d['level']['fde'])} | {d['normal_same_poses']['fde']['mean']:.2f} / {d['full_same_poses']['fde']['mean']:.2f} | {dl(d['vs_normal']['fde'], 'fde')} | "
                   f"{dl(d['vs_full']['fde'], 'fde')} | {dl(d['vs_full']['ade'], 'ade')} | {dl(d['vs_full']['growth'], 'growth')} | {dl(d['vs_full']['entropy'], 'entropy')} | "
                   f"{dl(d['vs_full']['realign'], 'realign')} | {dl(d['vs_full']['amp2'], 'amp2')} | {fr(d['fraction']['fde'])} / {fr(d['fraction']['ade'])} |")
            cr = IN[sname][g]["N"][N]["critical"]
            if cr:
                w_(f"\n90% window: FDE {cr['fde']['point']}, ADE {cr['ade']['point']} (같은 단위 집합)")
            w_("")

    w_("## 6. 보조 분석: 20-token 확장 horizon (N = 10, 학습 길이 초과 → 분포 밖)\n")
    w_("![fig3](figures/fig3_extension_out_of_distribution.png)\n\n![fig4](figures/fig4_extended_horizon_secondary.png)\n")
    w_("| group | 조건 | P(action) k=9 | k=10 | k=11 | k=12 | k=13 | k=15 | entropy k=9 / 10 / 11 / 12 / 13 | 가장 높은 비-action 토큰 k=10 |")
    w_("|---|---|---|---|---|---|---|---|---|---|")
    for g in groups:
        for n in ("normal", "full", "win4"):
            M = D[g][n]["mass_by_k"]; E = D[g][n]["ent_by_k"]
            gk = lambda d, k: d.get(k, d.get(str(k)))  # noqa: E731
            w_(f"| {g} | {n} | {gk(M, 9):.3f} | {gk(M, 10):.3f} | {gk(M, 11):.3f} | {gk(M, 12):.3f} | {gk(M, 13):.3f} | {gk(M, 15):.3f} | "
               f"{gk(E, 9):.2f} / {gk(E, 10):.2f} / {gk(E, 11):.2f} / {gk(E, 12):.2f} / {gk(E, 13):.2f} | {gk(D[g][n]['top_non_action_by_k'], 10)} |")
    w_("")
    for g in groups:
        if not EX["t*<=1"][g]:
            continue
        B = EX["t*<=1"][g]["N"][10]["windows"]
        w_(f"### {g} (확장, N = 10, 참고용)\n")
        w_("| window | FDE@release+10 | Normal / Full 같은 구간 | Full 효과 대비 (FDE) | FDE>3 m | Normal / Full | ΔFDE vs Full |")
        w_("|---|---|---|---|---|---|---|")
        for w in [0] + WS_ALL:
            if w not in B:
                continue
            d = B[w]
            w_(f"| {'Normal' if w == 0 else f'{w}-step'} | {nm(d['level']['fde'])} | {d['normal_same_poses']['fde']['mean']:.2f} / {d['full_same_poses']['fde']['mean']:.2f} | "
               f"{fr(d['fraction']['fde']) if w else '–'} | {pc(d['level']['amp'])} | {100*d['normal_same_poses']['amp']['mean']:.0f} / {100*d['full_same_poses']['amp']['mean']:.0f}% | {dl(d['vs_full']['fde'], 'fde')} |")
        w_("")
    w_("## 7. Sanity checks\n")
    w_("| 검사 | 결과 |"); w_("|---|---|")
    lab = {"prefix_identical": "t*까지 모든 조건 토큰 동일", "first_output_identical": "a_{t*+1} 모든 조건 동일",
           "gt20_first10_eq_stored": "확장 GT 토큰 앞 10개 = 기존 GT 토큰", "gt_traj_first10_eq_stored": "확장 GT 궤적 앞 10 pose = 장면 JSON GT (<1e-4 m)",
           "win_saturated_first10_eq_full": "10-token 안에서 포화된 window(w ≥ 8−t*) = Full (앞 10 토큰)",
           "repro_tfw_normal": "temporal_feedback_window Normal 앞 10 토큰 재현 (batch 9 vs 17 수치차)", "repro_tfw_win1": "… win1", "repro_tfw_win2": "… win2",
           "repro_tfw_win3": "… win3", "repro_tfw_win4": "… win4", "repro_tfw_gt_history": "… GT-history = Full"}
    for k, v in S["sanity"].items():
        w_(f"| {lab.get(k, k)} | {'–' if v is None else f'{100*v:.1f}%'} |")
    w_("")
    w_("## 8. 해석, claim 평가, 한계\n"); w_(NAR.get("interpretation", "") + "\n")
    open(os.path.join(args.run, "HORIZON_CONTROLLED_WINDOW.md"), "w").write("\n".join(W_))
    print("\n".join(W_))


if __name__ == "__main__":
    # records are needed by the horizon-trend helper
    _a = argparse.ArgumentParser(); _a.add_argument("--run", default=os.path.join(POC_DIR, "outputs/horizon_controlled_window")); _a.add_argument("--narrative", default=None)
    _ns, _ = _a.parse_known_args()
    for _l in open(os.path.join(_ns.run, "records.jsonl")):
        _x = json.loads(_l); _RECS[(_x["token"], _x["perturbation"])] = _x
    main()
