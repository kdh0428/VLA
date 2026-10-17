#!/usr/bin/env python
"""
A1 coverage study (review_followup4, 2026-10-17): which paired-binary interval is valid at the boundary?

*** POST-HOC method check. CPU only. Run BEFORE the real-data recomputation (a1_ci_audit.py reads the selection). ***

PRE-SPECIFIED SELECTION RULE (written before any real-data recomputation; the real data are not read here):
  R1  a method that returns a zero-width interval with positive probability while the true discordance probability
      is > 0 is not a valid population interval -> excluded.
  R2  iid paired units (OpenVLA initial states, SpatialVLA coke singletons): among the remaining candidates pick the
      method with the highest MINIMUM coverage over the n = 100 scenarios (required >= 0.925); ties (within 0.005)
      -> smaller mean |coverage - 0.95| -> smaller mean width. n = 10 coverage is reported (per-task OpenVLA rows) and,
      if the selected method's minimum n = 10 coverage is < 0.925, per-task intervals are labelled low-validity.
  R3  clustered units (SpatialVLA repeated states): same rule over the clustered scenarios for G = 53 and G = 10
      separately among {old sign-flip inversion, cluster-robust t, cluster bootstrap percentile, Rao-Scott Newcombe}.
      The naive (cluster-ignoring) Newcombe interval is shown for reference only and cannot be selected. If no method
      reaches 0.925 minimum coverage for a cluster regime, intervals in that regime are marked NR (no valid
      population interval) and used descriptively only.
  R4  equivalence: a TOST procedure must keep P(declare equivalence | true |delta| = margin) <= 0.05 (+ MC error).

iid scenarios: EXACT coverage by enumerating all 2x2 tables (multinomial), n = 10 and 100.
clustered scenarios: Monte Carlo (R = 1000 per scenario, seed fixed) with cluster cell probabilities ~ Dirichlet(kappa *
pi); cluster sizes = the real SpatialVLA move_near design (10 physical layouts; 53 instruction-inclusive initial-state
clusters) and 120 singletons; interaction scenarios with 4 correlated outcomes per unit.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python a1_coverage_sim.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from math import lgamma

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/review_followup2")
import a1_paired_ci as A  # noqa: E402
import stats_core as SC  # noqa: E402
import b_direct_effects as BDE  # noqa: E402

POC = "/root/VLA/autovla_misalignment_poc"
OUT = f"{POC}/outputs/review_followup4/A1_A4"
PAIRS = f"{POC}/outputs/review_followup2/B_spatialvla_direct/run_20261013/pairs.csv"   # design (cluster sizes) only
SEED = 20261017
R_CL = int(os.environ.get("A1_R_CL", 1000))
MC_INV_R = 5000
PRUNE = 1e-13

IID = {  # name: (p11, p10, p01, p00)   p10 = first-only success (a only), delta = p10 - p01
    "I1 high success, rare symmetric discordance": (0.97, 0.015, 0.015, 0.0),
    "I2 high success, one-sided discordance (p01=0)": (0.90, 0.05, 0.0, 0.05),
    "I3 high success, few discordant, asym": (0.88, 0.06, 0.02, 0.04),
    "I4 low success, few discordant": (0.05, 0.03, 0.02, 0.90),
    "I5 moderate success, symmetric discordance": (0.60, 0.10, 0.10, 0.20),
    "I6 moderate success, asym delta=+0.10": (0.55, 0.15, 0.05, 0.25),
    "I7 OpenVLA-like (~80% success, 12% discordant)": (0.75, 0.04, 0.08, 0.13),
    "I8 extreme: 99% concordant success, p01=0": (0.99, 0.01, 0.0, 0.0),
}
TOST_IID = {  # boundary scenarios: (pi, margin) with |delta| == margin
    "T1 n=100 delta=-0.10, m=0.10": ((0.70, 0.05, 0.15, 0.10), 0.10, 100),
    "T2 n=100 delta=+0.05, m=0.05": ((0.80, 0.06, 0.01, 0.13), 0.05, 100),
    "T3 n=120 delta=+0.10, m=0.10 (low discordance)": ((0.80, 0.11, 0.01, 0.08), 0.10, 120),
}
CL_PI = {
    "C1 low discordance, delta=0": (0.80, 0.06, 0.06, 0.08),
    "C2 low discordance, delta=+0.05": (0.80, 0.085, 0.035, 0.08),
    "C3 very low discordance, delta=+0.02": (0.85, 0.03, 0.01, 0.11),
    "C4 moderate discordance, delta=-0.10": (0.60, 0.05, 0.15, 0.20),
}
KAPPAS = (2.0, 10.0)


# ============================================================================================== iid exact
def tables(n):
    out = []
    for a in range(n + 1):
        for b in range(n - a + 1):
            for c in range(n - a - b + 1):
                out.append((a, b, c, n - a - b - c))
    return np.array(out, int)


def log_multinom(T, pi):
    n = T[0].sum()
    lp = np.full(len(T), lgamma(n + 1))
    for j in range(4):
        lp -= np.array([lgamma(x + 1) for x in T[:, j]])
        if pi[j] > 0:
            lp += T[:, j] * math.log(pi[j])
        else:
            lp = np.where(T[:, j] > 0, -np.inf, lp)
    return lp


_CACHE = {}


def ci_all(a, b, c, d, level=0.95):
    n = a + b + c + d
    key = (a, b, c, d, level)
    if key in _CACHE:
        return _CACHE[key]
    al = 1 - level
    kb = (b, c, n, level)
    if kb not in _CACHE:
        _CACHE[kb] = {"old_signflip_inv": A.signflip_inv_trinomial(b, c, n, level),
                      "wald": A.wald_paired(b, c, n, al), "tango": A.tango_ci(b, c, n, al)}
    r = dict(_CACHE[kb])
    r["newcombe10"] = A.newcombe10(a, b, c, d, al)
    _CACHE[key] = r
    return r


def iid_study():
    rows = []
    for n in (10, 100):
        T = tables(n)
        for name, pi in IID.items():
            lp = log_multinom(T, pi)
            P = np.exp(lp)
            keep = P > PRUNE
            Tk, Pk = T[keep], P[keep]
            delta = pi[1] - pi[2]
            acc = {}
            for (a, b, c, d), p in zip(Tk, Pk):
                cis = ci_all(int(a), int(b), int(c), int(d))
                for m, (lo, hi) in cis.items():
                    s = acc.setdefault(m, {"cov": 0.0, "miss_lo": 0.0, "miss_hi": 0.0, "width": 0.0, "zero_w": 0.0,
                                           "end_at_0_artifact": 0.0, "mass": 0.0})
                    lo_, hi_ = max(lo, -1.0), min(hi, 1.0)
                    s["mass"] += p
                    s["cov"] += p * (lo <= delta <= hi)
                    s["miss_lo"] += p * (delta < lo)
                    s["miss_hi"] += p * (delta > hi)
                    s["width"] += p * (hi_ - lo_)
                    s["zero_w"] += p * (hi - lo < 1e-9)
                    s["end_at_0_artifact"] += p * ((0 < abs(lo) < 1e-9) or (0 < abs(hi) < 1e-9))
            for m, s in acc.items():
                rows.append({"design": "iid paired units", "n": n, "scenario": name, "p11": pi[0], "p10": pi[1],
                             "p01": pi[2], "p00": pi[3], "true_delta": delta, "true_discordance": pi[1] + pi[2],
                             "method": m, "coverage": s["cov"] / s["mass"], "miss_below": s["miss_lo"] / s["mass"],
                             "miss_above": s["miss_hi"] / s["mass"], "mean_width": s["width"] / s["mass"],
                             "P_zero_width": s["zero_w"] / s["mass"], "P_endpoint_tie_artifact": s["end_at_0_artifact"] / s["mass"],
                             "enumerated_mass": s["mass"], "computation": f"exact enumeration of 2x2 tables (pruned P<{PRUNE})"})
        print(f"[iid] n={n} done", flush=True)
    return pd.DataFrame(rows)


def tost_iid():
    rows = []
    for name, (pi, m, n) in TOST_IID.items():
        T = tables(n)
        P = np.exp(log_multinom(T, pi))
        keep = P > PRUNE
        res = {"old_signflip_TOST (p_equiv<0.05)": 0.0, "newcombe10 90% CI inside +-m": 0.0,
               "tango score TOST (90% CI inside +-m)": 0.0, "wald 90% CI inside +-m": 0.0}
        mass = 0.0
        for (a, b, c, d), p in zip(T[keep], P[keep]):
            mass += p
            pe = A.signflip_tost_trinomial(int(b), int(c), n, m)[2]
            res["old_signflip_TOST (p_equiv<0.05)"] += p * (pe < 0.05)
            lo, hi = A.newcombe10(a, b, c, d, 0.10)
            res["newcombe10 90% CI inside +-m"] += p * (-m < lo and hi < m)
            lo, hi = A.tango_ci(int(b), int(c), n, 0.10)
            res["tango score TOST (90% CI inside +-m)"] += p * (-m < lo and hi < m)
            lo, hi = A.wald_paired(b, c, n, 0.10)
            res["wald 90% CI inside +-m"] += p * (-m < lo and hi < m)
        for k, v in res.items():
            rows.append({"design": "iid paired units", "scenario": name, "n": n, "true_delta": pi[1] - pi[2], "margin": m,
                         "procedure": k, "P_declare_equivalent": v / mass, "nominal_max": 0.05,
                         "computation": "exact enumeration"})
    return pd.DataFrame(rows)


# ============================================================================================== clustered MC
def design_sizes():
    p = pd.read_csv(PAIRS, usecols=["experiment", "config", "contrast", "direction", "task", "seed",
                                    "cluster_primary", "cluster_physical"])
    x = p[(p.experiment == "P1D") & (p.config == "r1e4") & (p.contrast == "R-N") & (p.task == "move_near")]
    return {"G=10 (move_near physical layouts)": x.groupby("cluster_physical").size().values,
            "G=53 (move_near instruction-inclusive initial states; exp35 design)":
                p[(p.experiment == "exp35") & (p.config == "r1e4") & (p.contrast == "R-N") &
                  (p.task == "move_near")].groupby("cluster_primary").size().values,
            "G=120 singletons (coke)": np.ones(120, int)}


def gen_cluster_units(rng, sizes, pi, kappa):
    ya, yb, cl = [], [], []
    for g, m in enumerate(sizes):
        pg = rng.dirichlet(kappa * np.asarray(pi)) if m > 1 else np.asarray(pi)
        cnt = rng.multinomial(m, pg)
        ya += [1] * cnt[0] + [1] * cnt[1] + [0] * cnt[2] + [0] * cnt[3]
        yb += [1] * cnt[0] + [0] * cnt[1] + [1] * cnt[2] + [0] * cnt[3]
        cl += [g] * m
    return np.array(ya), np.array(yb), np.array(cl)


def cluster_methods(ya, yb, cl, rng_boot, level=0.95, need_tost=None):
    n = len(ya)
    d = (ya - yb).astype(float)
    labs, inv = np.unique(cl, return_inverse=True)
    G = len(labs)
    N = np.bincount(inv, weights=d) / n
    D = np.bincount(inv).astype(float) / n
    th = float(N.sum() / D.sum())
    out = {}
    if G <= SC.EXACT_MAX:
        out["old_signflip_inv"] = tuple(SC.signflip_ci(N, D, th, level))
    else:
        out["old_signflip_inv"] = tuple(BDE.mc_inv_ci(N, D, th, level, _emat(G)))
    out["cluster_t"] = tuple(SC.cluster_t(N, D, level)[1])
    bs = SC.cluster_bootstrap(N, D, 1000, int(rng_boot.integers(1 << 30)))
    out["cluster_bootstrap_pct"] = tuple(SC.pct_ci(bs, level))
    a, b, c, dd = A.table_counts(ya, yb)
    out["naive_newcombe10 (ignores clusters; reference only)"] = A.newcombe10(a, b, c, dd, 1 - level)
    out["rs_newcombe10"] = A.rs_newcombe(ya, yb, cl, 1 - level)
    return th, out, (N, D)


_E = {}


def _emat(G):
    if G not in _E:
        _E[G] = np.random.default_rng(SEED + 99).integers(0, 2, size=(MC_INV_R, G)).astype(float) * 2 - 1
    return _E[G]


def cluster_study():
    sizes = design_sizes()
    rows, trows = [], []
    rng = np.random.default_rng(SEED)
    for sname, sz in sizes.items():
        for pname, pi in CL_PI.items():
            for kappa in (KAPPAS if sz.max() > 1 else (np.inf,)):
                t0 = time.time()
                delta = pi[1] - pi[2]
                stats = {}
                eq = {}
                tost_m = 0.10 if abs(abs(delta) - 0.10) < 1e-9 else (0.05 if abs(abs(delta) - 0.05) < 1e-9 else None)
                for r in range(R_CL):
                    ya, yb, cl = gen_cluster_units(rng, sz, pi, kappa)
                    th, cis, (N, D) = cluster_methods(ya, yb, cl, rng)
                    for m, (lo, hi) in cis.items():
                        s = stats.setdefault(m, {"cov": 0, "lo": 0, "hi": 0, "w": [], "zw": 0, "nan": 0})
                        if not (np.isfinite(lo) and np.isfinite(hi)):
                            s["nan"] += 1
                            s["cov"] += 1 if (np.isinf(lo) or np.isinf(hi)) and not (np.isnan(lo) or np.isnan(hi)) else 0
                            continue
                        s["cov"] += int(lo <= delta <= hi)
                        s["lo"] += int(delta < lo)
                        s["hi"] += int(delta > hi)
                        s["w"].append(min(hi, 1) - max(lo, -1))
                        s["zw"] += int(hi - lo < 1e-9)
                    if tost_m is not None:
                        G = len(N)
                        if G <= SC.EXACT_MAX:
                            t = SC.tost(N, D, tost_m, "signflip", B=20, seed=1)
                        else:
                            t = SC.tost(N, D, tost_m, "signflip", B=20, seed=1, reps=2000)
                        eq.setdefault("old_signflip_TOST (p_equiv<0.05)", []).append(t["p_equiv"] < 0.05)
                        lo, hi = A.rs_newcombe(ya, yb, cl, 0.10)
                        eq.setdefault("rs_newcombe10 90% CI inside +-m", []).append(-tost_m < lo and hi < tost_m)
                        lo, hi = SC.cluster_t(N, D, 0.90)[1]
                        eq.setdefault("cluster_t 90% CI inside +-m", []).append(-tost_m < lo and hi < tost_m)
                for m, s in stats.items():
                    rows.append({"design": sname, "G": len(sz), "n_units": int(sz.sum()), "scenario": pname,
                                 "kappa (Dirichlet; small = strong clustering)": kappa, "p11": pi[0], "p10": pi[1],
                                 "p01": pi[2], "p00": pi[3], "true_delta": delta, "method": m,
                                 "coverage": s["cov"] / R_CL, "mc_se": math.sqrt(0.95 * 0.05 / R_CL),
                                 "miss_below": s["lo"] / R_CL, "miss_above": s["hi"] / R_CL,
                                 "mean_width": float(np.mean(s["w"])) if s["w"] else np.nan,
                                 "P_zero_width": s["zw"] / R_CL, "P_nonfinite": s["nan"] / R_CL, "reps": R_CL,
                                 "computation": f"Monte Carlo R={R_CL}, seed {SEED}" +
                                                ("; old inversion = MC sign matrix R=5000" if len(sz) > SC.EXACT_MAX else "")})
                for k, v in eq.items():
                    trows.append({"design": sname, "scenario": pname, "kappa": kappa, "n": int(sz.sum()), "G": len(sz),
                                  "true_delta": delta, "margin": tost_m, "procedure": k,
                                  "P_declare_equivalent": float(np.mean(v)), "nominal_max": 0.05, "reps": R_CL,
                                  "computation": "Monte Carlo" + ("; old TOST MC sign-flip reps=2000" if len(sz) > SC.EXACT_MAX else "")})
                print(f"[cl] {sname[:5]} {pname[:3]} k={kappa} {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(trows)


def interaction_study():
    """4 outcomes per unit (N1, R1 in config 1; N2, R2 in config 2); interaction = (R2-N2) - (R1-N1)."""
    sizes = design_sizes()
    rng = np.random.default_rng(SEED + 5)
    off = {"N1": 0.0, "R1": -0.05, "N2": 0.0, "R2": 0.10}
    p_conc = 0.85

    def gen(sz, kappa, rng):
        Y = {k: [] for k in off}
        cl = []
        for g, m in enumerate(sz):
            q = rng.beta(kappa * 0.8, kappa * 0.2) if m > 1 else 0.8
            for _ in range(m):
                if rng.random() < p_conc:
                    v = int(rng.random() < q)
                    for k in off:
                        Y[k].append(v)
                else:
                    for k in off:
                        Y[k].append(int(rng.random() < min(max(q + off[k], 0), 1)))
                cl.append(g)
        return {k: np.array(v) for k, v in Y.items()}, np.array(cl)
    rows = []
    for sname, sz in sizes.items():
        if sz.max() == 1:
            continue
        for kappa in KAPPAS:
            # true interaction by large simulation of the generating process (cluster-level mean of the unit effect)
            big = []
            for _ in range(40):
                Yb, _ = gen(sz, kappa, rng)
                big.append(((Yb["R2"] - Yb["N2"]) - (Yb["R1"] - Yb["N1"])).mean())
            qs = rng.beta(kappa * 0.8, kappa * 0.2, size=2_000_000)
            true = float((1 - p_conc) * (np.clip(qs + 0.10, 0, 1) - qs - (np.clip(qs - 0.05, 0, 1) - qs)).mean())
            st = {}
            for r in range(R_CL):
                Y, cl = gen(sz, kappa, rng)
                d1, d2 = Y["R1"] - Y["N1"], Y["R2"] - Y["N2"]
                di = (d2 - d1).astype(float)
                n = len(di)
                labs, inv = np.unique(cl, return_inverse=True)
                N = np.bincount(inv, weights=di) / n
                D = np.bincount(inv).astype(float) / n
                th = float(di.mean())
                cis = {}
                cis["old_signflip_inv"] = tuple(SC.signflip_ci(N, D, th, .95)) if len(N) <= SC.EXACT_MAX else \
                    tuple(BDE.mc_inv_ci(N, D, th, .95, _emat(len(N))))
                cis["cluster_t"] = tuple(SC.cluster_t(N, D)[1])
                l1, u1 = A.rs_newcombe(Y["R1"], Y["N1"], cl)
                l2, u2 = A.rs_newcombe(Y["R2"], Y["N2"], cl)
                rr = A.cluster_corr(d1, d2, cl)
                _, L, U = A.mover_lincomb([(1, d2.mean(), l2, u2), (-1, d1.mean(), l1, u1)], [[1, rr], [rr, 1]])
                cis["rs_newcombe10 MOVER difference"] = (L, U)
                for m, (lo, hi) in cis.items():
                    s = st.setdefault(m, {"cov": 0, "w": [], "zw": 0})
                    s["cov"] += int(lo <= true <= hi)
                    s["w"].append(hi - lo)
                    s["zw"] += int(hi - lo < 1e-9)
            for m, s in st.items():
                rows.append({"design": sname + " INTERACTION (4 outcomes/unit)", "G": len(sz), "n_units": int(sz.sum()),
                             "scenario": "interaction (R2-N2)-(R1-N1), 85% fully concordant units",
                             "kappa (Dirichlet; small = strong clustering)": kappa, "true_delta": true, "method": m,
                             "coverage": s["cov"] / R_CL, "mc_se": math.sqrt(0.95 * 0.05 / R_CL),
                             "mean_width": float(np.mean(s["w"])), "P_zero_width": s["zw"] / R_CL, "reps": R_CL,
                             "computation": f"Monte Carlo R={R_CL}; true value by 2e6 cluster draws"})
            print(f"[int] {sname[:5]} k={kappa}", flush=True)
    return pd.DataFrame(rows)


# ============================================================================================== selection
def select(iid, cl, it):
    sel = {"rule": __doc__.split("PRE-SPECIFIED SELECTION RULE")[1].split("iid scenarios:")[0].strip()}
    cand = ["old_signflip_inv", "wald", "newcombe10", "tango"]
    x = iid[iid.n == 100]
    info = {}
    for m in cand:
        xm = x[x.method == m]
        zw = float(iid[(iid.method == m) & (iid.true_discordance > 0)].P_zero_width.max())
        info[m] = {"min_cov_n100": float(xm.coverage.min()), "mean_abs_dev_n100": float((xm.coverage - .95).abs().mean()),
                   "mean_width_n100": float(xm.mean_width.mean()), "max_P_zero_width_when_discordance>0": zw,
                   "min_cov_n10": float(iid[(iid.n == 10) & (iid.method == m)].coverage.min()),
                   "excluded_R1": zw > 0}
    ok = [m for m in cand if not info[m]["excluded_R1"] and info[m]["min_cov_n100"] >= 0.925]
    best = None
    if ok:
        top = max(info[m]["min_cov_n100"] for m in ok)
        tied = [m for m in ok if info[m]["min_cov_n100"] >= top - 0.005]
        tied.sort(key=lambda m: (info[m]["mean_abs_dev_n100"], info[m]["mean_width_n100"]))
        best = tied[0]
    sel["iid"] = {"candidates": info, "selected": best,
                  "per_task_n10_low_validity": bool(best is None or info[best]["min_cov_n10"] < 0.925)}
    sel["clustered"] = {}
    candc = ["old_signflip_inv", "cluster_t", "cluster_bootstrap_pct", "rs_newcombe10"]
    for G in sorted(cl.G.unique()):
        x = cl[cl.G == G]
        inf = {}
        for m in candc:
            xm = x[x.method == m]
            inf[m] = {"min_cov": float(xm.coverage.min()), "mean_abs_dev": float((xm.coverage - .95).abs().mean()),
                      "mean_width": float(xm.mean_width.mean()), "max_P_zero_width": float(xm.P_zero_width.max()),
                      "excluded_R1": bool(xm.P_zero_width.max() > 0)}
        ref = x[x.method.str.startswith("naive")]
        ok = [m for m in candc if not inf[m]["excluded_R1"] and inf[m]["min_cov"] >= 0.925]
        best = None
        if ok:
            top = max(inf[m]["min_cov"] for m in ok)
            tied = sorted([m for m in ok if inf[m]["min_cov"] >= top - 0.005],
                          key=lambda m: (inf[m]["mean_abs_dev"], inf[m]["mean_width"]))
            best = tied[0]
        sel["clustered"][f"G={G}"] = {"candidates": inf, "selected": best,
                                      "naive_newcombe_min_cov (reference)": float(ref.coverage.min()) if len(ref) else None,
                                      "regime_NR": best is None}
    if len(it):
        sel["interaction"] = {f"G={G}": {m: {"min_cov": float(g[g.method == m].coverage.min()),
                                             "max_P_zero_width": float(g[g.method == m].P_zero_width.max())}
                                         for m in g.method.unique()} for G, g in it.groupby("G")}
    return sel


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    iid = iid_study()
    iid.to_csv(f"{OUT}/A1_COVERAGE_SIM_iid.csv", index=False, float_format="%.6g")
    ti = tost_iid()
    cl, tc = cluster_study()
    it = interaction_study()
    pd.concat([cl, it], ignore_index=True).to_csv(f"{OUT}/A1_COVERAGE_SIM_cluster.csv", index=False, float_format="%.6g")
    pd.concat([ti, tc], ignore_index=True).to_csv(f"{OUT}/A1_TOST_SIZE_SIM.csv", index=False, float_format="%.6g")
    sel = select(iid, cl, it)
    sel["runtime_s"] = time.time() - t0
    sel["seed"] = SEED
    sel["R_cluster"] = R_CL
    json.dump(sel, open(f"{OUT}/A1_METHOD_SELECTION.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in sel.items() if k != "rule"}, indent=1, default=float))


if __name__ == "__main__":
    main()
