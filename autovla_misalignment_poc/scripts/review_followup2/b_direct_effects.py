#!/usr/bin/env python
"""
B (review_followup2, 2026-10-13): SpatialVLA success-rate DIRECT effects, interactions, equivalence (B2-B4).

*** POST-HOC ANALYSIS OF EXISTING DATA (experiments 34, 35, P1-D). CPU only. *** Nothing here is preregistered, except
that rows marked `orig_family` re-derive judgments of a preregistered family (P1-D S1-S4) or of P0-B (exp 34) so that
the original judgment can be compared with the clustered one. The +-5 pp margin is a post-hoc sensitivity margin
proposed by this review, not a field standard.

Input: pairs.csv / episodes.csv written by b_extract.py. Output (run dir):
  DIRECT_EFFECTS.csv           one row per experiment x config x contrast x task-level x clustering
  B4_TRAJECTORY_VS_SUCCESS.csv generated / executed / actual-TCP metrics next to the success direct effect
  INTERPRETATION.csv           interpretation rule per experiment x config x task (core rows)
  b_stats_meta.json            seeds, reps, methods

Estimand (kept from the original analyses): within a task, the unit-weighted mean of the paired success difference
(= difference of success rates over the same units); pooled = task-equal mean of the two task means.
Independent unit for resampling / sign flips: initial-state cluster (`primary` = initial_state_cluster_id, or the
conservative `physical` = physical_state_cluster_id: move_near 10 layouts), or `episode` (original unit, for
reproduction). All conditions / configs of a unit move together (pairing preserved); interactions are formed per unit
before clustering.
Inference:
  * sign-flip test (two-sided) on cluster totals, exact if G <= 26 else Monte Carlo 100,000 (p = (b+1)/(R+1));
    assumption: cluster totals independent and symmetric about 0 under H0 (conditional test, not a randomisation test).
  * 95% / 90% CI by sign-flip test inversion (exact if G <= 26, else Monte Carlo inversion with ONE fixed sign matrix,
    R = 20,000, seed 20261013); reported as the primary CI because it is the dual of the test and of the TOST.
    Cluster bootstrap percentile CI (task-stratified, B = 10,000) reported alongside (anti-conservative with 10 clusters).
  * TOST (stats_core.tost, method signflip): p_lower (H0 theta <= -m), p_upper (H0 theta >= m), p_equiv = max,
    m = 0.10 (original) and 0.05 (post-hoc sensitivity). Holm over p_equiv within the declared family; difference p
    Holm-adjusted within the same family.
  * judgments: difference from 0 (Holm sign-flip p < .05) and equivalence (Holm p_equiv < .05) in separate columns;
    'uncertain' when neither holds (stats_core.judge).

  nice -n 19 /root/miniforge3/envs/autovla/bin/python b_direct_effects.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stats_core as SC  # noqa: E402

POC = "/root/VLA/autovla_misalignment_poc"
OUT = os.path.join(POC, "outputs/review_followup2/B_spatialvla_direct/run_20261013")
SEED = 20261013
B = 10_000
MC = 100_000
MC_INV = 20_000
MARGINS = (0.10, 0.05)
CLUSTERINGS = ("primary", "physical", "episode")
TASKS = ("coke", "move_near")
STATUS = {"exp34": "preregistered experiment (PROTOCOL f578eae); this analysis post-hoc",
          "exp35": "preregistered experiment (PROTOCOL 8fc20bf); this analysis post-hoc",
          "P1D": "preregistered experiment (b212637); this analysis post-hoc"}
DEF = {"R-N": "delta_R = S(R) - S(N): context-perturbation direct effect (R executes natural step-1, context p)",
       "C-F": "delta_C = S(C) - S(F): context-correction direct effect (+ = correction benefit)",
       "F-C": "F - C (compatibility; = -delta_C)",
       "CD-FD": "delta_C under design B (dose-matched): S(CD) - S(FD)",
       "FD-CD": "FD - CD (compatibility; = -delta_C design B)",
       "F-N": "executed+context perturbation vs natural (positive control for execution)",
       "C-N": "executed perturbation only vs natural", "FD-N": "design B FD vs natural", "CD-N": "design B CD vs natural"}


# ----------------------------------------------------------------------------------------------------- inference
def cluster_arrays(df, ccol):
    """df: rows with d, task, cluster col. -> labels, task of cluster, N, D for the task-equal (or single-task)
    unit-weighted estimand: N_g = sum_u d_u / n_task, D_g = n_g / n_task."""
    nt = df.groupby("task").size()
    w = 1.0 / df["task"].map(nt).values
    x = pd.DataFrame({"c": df[ccol].values, "task": df["task"].values, "N": df["d"].values * w, "D": w})
    g = x.groupby("c", sort=True).agg(task=("task", "first"), N=("N", "sum"), D=("D", "sum"))
    if (x.groupby("c")["task"].nunique() > 1).any():
        raise ValueError("cluster spans tasks")
    return g.index.tolist(), g["task"].values, g["N"].values.astype(float), g["D"].values.astype(float)


def strat_boot(tasks_of, N, D, B=B, seed=SEED):
    rng = np.random.default_rng(seed)
    out = np.zeros(B)
    ts = sorted(set(tasks_of))
    for t in ts:
        ix = np.where(tasks_of == t)[0]
        C = rng.multinomial(len(ix), np.ones(len(ix)) / len(ix), size=B).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out += (C @ N[ix]) / (C @ D[ix])
    return out / len(ts)


def mc_inv_ci(N, D, theta, level, E):
    """sign-flip test inversion with a fixed Monte Carlo sign matrix E (R x G); p(delta) = (b+1)/(R+1)."""
    a = 1 - level

    def p(dl):
        T = N - dl * D
        obs = abs(T.sum())
        s = np.abs(E @ T)
        return (np.sum(s >= obs - 1e-12 * max(1.0, np.abs(T).sum())) + 1) / (E.shape[0] + 1)
    span = max(1.0, 4 * abs(theta))
    out = []
    for direction in (-1, 1):
        lo, hi = theta, theta + direction * span
        if p(hi) > a:
            out.append(float("inf") * direction)
            continue
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if p(mid) > a:
                lo = mid
            else:
                hi = mid
        out.append(lo)
    return [min(out), max(out)]


_EMAT = {}


def sign_matrix(G):
    if G not in _EMAT:
        _EMAT[G] = np.random.default_rng(SEED + 99).integers(0, 2, size=(MC_INV, G)).astype(float) * 2 - 1
    return _EMAT[G]


def infer(df, ccol):
    labels, tk, N, D = cluster_arrays(df, ccol)
    G = len(N)
    th = SC.ratio(N, D)
    r = {"n_units": int(len(df)), "n_clusters": G, "estimate": th}
    bs = strat_boot(tk, N, D)
    r["boot_ci95"], r["boot_ci90"] = SC.pct_ci(bs, .95), SC.pct_ci(bs, .90)
    sf = SC.signflip_test(N, "two-sided", SC.EXACT_MAX, MC, SEED + 7)
    r.update({"p_diff": sf["p"], "p_diff_method": sf["method"], "n_informative": sf["n_informative"],
              "p_min_attainable": sf["p_min_attainable"]})
    r["p_cluster_t"] = SC.cluster_t(N, D)[0]
    if G <= SC.EXACT_MAX:
        r["sf_ci95"], r["sf_ci90"] = SC.signflip_ci(N, D, th, .95), SC.signflip_ci(N, D, th, .90)
        r["ci_method"] = f"exact sign-flip inversion (G={G})"
    else:
        E = sign_matrix(G)
        r["sf_ci95"], r["sf_ci90"] = mc_inv_ci(N, D, th, .95, E), mc_inv_ci(N, D, th, .90, E)
        r["ci_method"] = f"MC sign-flip inversion (G={G}, R={MC_INV}, seed={SEED + 99})"
    for m in MARGINS:
        t = SC.tost(N, D, m, "signflip", B=200, seed=SEED, exact_max=SC.EXACT_MAX, reps=MC)
        r[f"tost{int(m * 100)}"] = {"p_lower": t["p_lower"], "p_upper": t["p_upper"], "p_equiv": t["p_equiv"]}
    return r


# ----------------------------------------------------------------------------------------------------- contrasts
def unit_d(pairs, exp, cfg, contrast, direction=None):
    """paired success differences for one experiment/config/contrast -> DataFrame(task, seed, d, clusters)."""
    sign = 1.0
    base = contrast
    if contrast in ("F-C", "FD-CD"):
        base, sign = {"F-C": "C-F", "FD-CD": "CD-FD"}[contrast], -1.0
    q = (pairs.experiment == exp) & (pairs.config == cfg) & (pairs.contrast == base)
    if direction is not None:
        q &= pairs.direction == direction
    x = pairs[q][["task", "seed", "d_success", "cluster_primary", "cluster_physical"]].copy()
    x["d"] = sign * x["d_success"]
    return x.drop(columns="d_success").reset_index(drop=True)


def interaction(pairs, exp, c1, cn1, c2, cn2, direction=None, direction2=None, max_seed=None):
    a = unit_d(pairs, exp, c2, cn2, direction2 if direction2 else direction)
    b = unit_d(pairs, exp, c1, cn1, direction)
    if max_seed is not None:
        a, b = a[a.seed <= max_seed], b[b.seed <= max_seed]
    a, b = a.set_index(["task", "seed"]), b.set_index(["task", "seed"])
    j = a.join(b[["d"]], rsuffix="_1", how="inner")
    if len(j) != len(a) or len(j) != len(b):
        raise ValueError(f"interaction unpaired units {exp} {c1} {c2}: {len(a)} {len(b)} {len(j)}")
    j["d"] = j["d"] - j["d_1"]
    return j.drop(columns="d_1").reset_index()


def rates(epi, exp, cfg, units, direction):
    """success rates (task-equal over the given units) of N, F, C, R (+ FD, CD) in this config."""
    out = {}
    for mode, key in (("natural", "S_N"), ("feedback", "S_F"), ("corrected", "S_C"), ("reverse", "S_R"),
                      ("feedbackD", "S_FD"), ("correctedD", "S_CD")):
        d = "-" if (exp == "exp34" and mode == "natural") else direction
        q = epi[(epi.experiment == exp) & (epi.config == cfg) & (epi["mode"] == mode) & (epi.direction == d)]
        q = q.merge(units, on=["task", "seed"])
        if len(q) == 0:
            continue
        per = q.groupby("task")["success"].mean()
        out[key] = {t: float(per[t]) for t in per.index}
        out[key]["pooled"] = float(per.mean())
    return out


def build_specs():
    """(id, experiment, kind, role, config label, contrast, family, builder kwargs, orig)"""
    S = []
    # ---- exp 34 (config r1e4 official; directions perp_left / opposite); family = P0-B closed-loop 4 contrasts
    for dr in ("perp_left", "opposite"):
        for cn in ("R-N", "C-F", "F-C"):
            S.append(dict(exp="exp34", kind="direct", role="core", config=f"r1e4/{dr}", contrast=cn,
                          family=None if cn == "F-C" else "exp34 closed-loop direct (P0-B 4 contrasts)",
                          src=("d", "exp34", "r1e4", cn, dr)))
        for cn in ("F-N", "C-N"):
            S.append(dict(exp="exp34", kind="aux", role="appendix", config=f"r1e4/{dr}", contrast=cn,
                          family="exp34 aux executed perturbation", src=("d", "exp34", "r1e4", cn, dr)))
    for cn in ("R-N", "C-F"):
        S.append(dict(exp="exp34", kind="interaction", role="appendix", config="opposite - perp_left (r1e4)",
                      contrast=f"{cn} interaction", family="exp34 direction interactions",
                      src=("i", "exp34", "r1e4", cn, "r1e4", cn, "perp_left", "opposite")))
    # ---- exp 35 direct
    for cfg in ("r1e4", "r4e1", "r1e2", "r1e1", "r2e2", "r2e1"):
        core = cfg in ("r1e4", "r4e1")
        for cn in ("R-N", "C-F", "F-C"):
            S.append(dict(exp="exp35", kind="direct", role="core" if core else "appendix", config=cfg, contrast=cn,
                          family=None if cn == "F-C" else ("exp35 core direct (seeds 0-119)" if core else
                                                          "exp35 other-config direct (seeds 0-39)"),
                          src=("d", "exp35", cfg, cn, "opposite")))
        for cn in ("F-N", "C-N"):
            S.append(dict(exp="exp35", kind="aux", role="appendix", config=cfg, contrast=cn,
                          family="exp35 aux executed perturbation", src=("d", "exp35", cfg, cn, "opposite")))
    for cn in ("R-N", "C-F"):
        S.append(dict(exp="exp35", kind="interaction", role="core", config="r4e1 - r1e4", contrast=f"{cn} interaction",
                      family="exp35 core interaction r4e1-r1e4 (preregistered main contrast)",
                      src=("i", "exp35", "r1e4", cn, "r4e1", cn, "opposite", None)))
    for c1, c2 in (("r1e1", "r2e1"), ("r1e1", "r4e1"), ("r2e1", "r4e1"), ("r1e1", "r1e2"), ("r1e1", "r1e4"),
                   ("r1e2", "r1e4"), ("r2e1", "r2e2")):
        for cn in ("R-N", "C-F"):
            S.append(dict(exp="exp35", kind="interaction", role="appendix", config=f"{c2} - {c1} (seeds 0-39)",
                          contrast=f"{cn} interaction", family="exp35 P0-C fixed-axis interactions (seeds 0-39)",
                          src=("i", "exp35", c1, cn, c2, cn, "opposite", None), seeds039=True))
    # ---- P1-D direct
    for cfg in ("r1e4", "r4e1", "r1e1", "r2e1"):
        core = cfg in ("r1e4", "r4e1")
        for cn in ("R-N", "C-F", "F-C"):
            S.append(dict(exp="P1D", kind="direct", role="core" if core else "appendix", config=cfg, contrast=cn,
                          family=None if cn == "F-C" else ("P1D core direct" if core else "P1D other direct"),
                          src=("d", "P1D", cfg, cn, "opposite")))
        for cn in ("F-N", "C-N"):
            S.append(dict(exp="P1D", kind="aux", role="appendix", config=cfg, contrast=cn,
                          family="P1D aux executed perturbation", src=("d", "P1D", cfg, cn, "opposite")))
    for cn in ("CD-FD", "FD-CD"):
        S.append(dict(exp="P1D", kind="direct", role="appendix", config="r1e4 (design B)", contrast=cn,
                      family=None if cn == "FD-CD" else "P1D other direct", src=("d", "P1D", "r1e4", cn, "opposite")))
    for cn in ("FD-N", "CD-N"):
        S.append(dict(exp="P1D", kind="aux", role="appendix", config="r1e4 (design B)", contrast=cn,
                      family="P1D aux executed perturbation", src=("d", "P1D", "r1e4", cn, "opposite")))
    # P1-D preregistered secondary success family S1-S4 (original sign F-C; here delta_C sign, equivalence identical)
    orig = {"S1": (-0.8, "equivalent +-10 (90% CI inside)"), "S2": (-0.8, "equivalent +-10 (90% CI inside)"),
            "S3": (0.4, "equivalent +-10 (90% CI inside)"), "S4": (-1.3, "equivalent +-10 (90% CI inside)")}
    for sid, c1, cn1, c2, cn2, lab in (("S1", "r1e1", "R-N", "r4e1", "R-N", "r4e1 - r1e1"),
                                       ("S2", "r1e1", "C-F", "r1e4", "CD-FD", "r1e4(design B) - r1e1"),
                                       ("S3", "r1e1", "C-F", "r4e1", "C-F", "r4e1 - r1e1"),
                                       ("S4", "r1e1", "R-N", "r1e4", "R-N", "r1e4 - r1e1")):
        S.append(dict(exp="P1D", kind="interaction", role="core", config=lab,
                      contrast=f"{sid}: {'delta_C' if cn1 == 'C-F' else 'delta_R'} interaction",
                      family="P1D S1-S4 (preregistered secondary success family)",
                      src=("i", "P1D", c1, cn1, c2, cn2, "opposite", None),
                      orig_est_pp=orig[sid][0] * (-1 if cn1 == "C-F" else 1), orig_judgment=orig[sid][1],
                      orig_note="original sign F-C; sign flipped here" if cn1 == "C-F" else ""))
    for cn in ("R-N", "C-F"):
        S.append(dict(exp="P1D", kind="interaction", role="core", config="r4e1 - r1e4 (cross-axis, post-hoc)",
                      contrast=f"{cn} interaction", family="P1D core interaction r4e1-r1e4 (post-hoc, cross-axis)",
                      src=("i", "P1D", "r1e4", cn, "r4e1", cn, "opposite", None)))
    for c1, cn1, c2, cn2, lab in (("r1e1", "R-N", "r2e1", "R-N", "r2e1 - r1e1"),
                                  ("r1e1", "C-F", "r2e1", "C-F", "r2e1 - r1e1"),
                                  ("r1e1", "C-F", "r1e4", "C-F", "r1e4(design A) - r1e1")):
        S.append(dict(exp="P1D", kind="interaction", role="appendix", config=lab, contrast=f"{cn1} interaction",
                      family="P1D other interactions", src=("i", "P1D", c1, cn1, c2, cn2, "opposite", None)))
    return S


def data_for(pairs, spec):
    s = spec["src"]
    if s[0] == "d":
        x = unit_d(pairs, s[1], s[2], s[3], s[4])
    else:
        _, exp, c1, cn1, c2, cn2, dr, dr2 = s
        if dr2 is not None:      # direction interaction (exp 34)
            x = interaction(pairs, exp, c1, cn1, c2, cn2, direction=dr, direction2=dr2)
        else:
            x = interaction(pairs, exp, c1, cn1, c2, cn2, direction=dr, max_seed=39 if spec.get("seeds039") else None)
    if spec.get("seeds039"):
        x = x[x.seed <= 39]
    return x


def main():
    t0 = time.time()
    pairs = pd.read_csv(os.path.join(OUT, "pairs.csv"))
    epi = pd.read_csv(os.path.join(OUT, "episodes.csv"))
    specs = build_specs()
    rows = []
    for sp in specs:
        x = data_for(pairs, sp)
        x["cluster_episode"] = x["task"] + "-s" + x["seed"].astype(str)
        units = x[["task", "seed"]]
        rt = {}
        if sp["kind"] in ("direct", "aux") and sp["src"][0] == "d":
            rt = rates(epi, sp["exp"], sp["src"][2], units, sp["src"][4])
        for lvl in TASKS + ("pooled",):
            xx = x if lvl == "pooled" else x[x.task == lvl]
            if len(xx) == 0:
                continue
            for cl in CLUSTERINGS:
                r = infer(xx, f"cluster_{cl}")
                row = {"model": "SpatialVLA-4B (spatialvla-4b-224-sft-fractal)", "experiment": sp["exp"],
                       "experiment_status": STATUS[sp["exp"]], "kind": sp["kind"], "role": sp["role"],
                       "config": sp["config"], "contrast": sp["contrast"],
                       "definition": DEF.get(sp["contrast"], "interaction = contrast(config_2) - contrast(config_1), "
                                                              "same unit"),
                       "task": lvl, "clustering": cl, "family": sp["family"] or "(compatibility row; not tested in family)",
                       "units": f"seeds {int(xx.seed.min())}-{int(xx.seed.max())}" if sp["exp"] != "P1D" else "P1-D units 0-119",
                       "n_units": r["n_units"], "n_clusters": r["n_clusters"], "n_informative_clusters": r["n_informative"]}
                for k, v in rt.items():
                    row[k] = round(100 * v.get(lvl, np.nan), 2)
                row["estimate_pp"] = 100 * r["estimate"]
                row["ci95_lo_pp"], row["ci95_hi_pp"] = [100 * v for v in r["sf_ci95"]]
                row["ci90_lo_pp"], row["ci90_hi_pp"] = [100 * v for v in r["sf_ci90"]]
                row["ci_method"] = r["ci_method"]
                row["ci_unbounded"] = bool(not np.all(np.isfinite(r["sf_ci95"])))
                row["boot_ci95_lo_pp"], row["boot_ci95_hi_pp"] = [100 * v for v in r["boot_ci95"]]
                row["boot_ci90_lo_pp"], row["boot_ci90_hi_pp"] = [100 * v for v in r["boot_ci90"]]
                row["p_diff_signflip"] = r["p_diff"]
                row["p_diff_method"] = r["p_diff_method"]
                row["p_min_attainable"] = r["p_min_attainable"]
                row["p_cluster_t"] = r["p_cluster_t"]
                for m in MARGINS:
                    k = int(m * 100)
                    t = r[f"tost{k}"]
                    row[f"margin{k}_pp"] = k
                    row[f"p_lower_m{k}"], row[f"p_upper_m{k}"], row[f"p_equiv_m{k}"] = t["p_lower"], t["p_upper"], t["p_equiv"]
                    row[f"ci90_inside_m{k}"] = bool(-k < row["ci90_lo_pp"] and row["ci90_hi_pp"] < k)
                for k in ("orig_est_pp", "orig_judgment", "orig_note"):
                    if k in sp:
                        row[k] = sp[k]
                rows.append(row)
        print(f"[{time.time() - t0:6.0f}s] {sp['exp']} {sp['config']} {sp['contrast']} n={len(x)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_pickle(os.path.join(OUT, "_raw_rows.pkl"))
    post(df, pairs, t0)


def post(df, pairs, t0):
    # ---- Holm within declared families (per task level x clustering); difference and equivalence separately
    df["p_diff_holm"] = np.nan
    for k in (10, 5):
        df[f"p_equiv_m{k}_holm"] = np.nan
    fam_mask = ~df["family"].str.startswith("(compat")
    for (fam, lvl, cl), g in df[fam_mask].groupby(["family", "task", "clustering"]):
        df.loc[g.index, "p_diff_holm"] = SC.holm(g["p_diff_signflip"].values)
        for k in (10, 5):
            df.loc[g.index, f"p_equiv_m{k}_holm"] = SC.holm(g[f"p_equiv_m{k}"].values)
        df.loc[g.index, "family_size"] = len(g)
    # compatibility rows inherit the adjusted p of their mirrored row (same |theta|, same tests)
    mirror = {"F-C": "C-F", "FD-CD": "CD-FD"}
    for i in df[~fam_mask].index:
        r = df.loc[i]
        j = df[(df.experiment == r.experiment) & (df.config == r.config) & (df.contrast == mirror[r.contrast]) &
               (df.task == r.task) & (df.clustering == r.clustering)].index[0]
        for c in ("p_diff_holm", "p_equiv_m10_holm", "p_equiv_m5_holm", "family_size"):
            df.loc[i, c] = df.loc[j, c]
        df.loc[i, "family"] = f"(compatibility: mirror of {mirror[r.contrast]}, family '{df.loc[j, 'family']}')"
        # exact sign / CI reversal check
        for a, b in (("estimate_pp", "estimate_pp"), ("ci95_lo_pp", "ci95_hi_pp"), ("ci90_lo_pp", "ci90_hi_pp")):
            if not np.isclose(df.loc[i, a], -df.loc[j, b], atol=1e-6):
                raise AssertionError(f"sign/CI reversal failed {r.experiment} {r.config} {r.contrast} {a}")
        exact = int(r.n_clusters) <= SC.EXACT_MAX
        for k in (10, 5):
            for a_, b_ in ((f"p_lower_m{k}", f"p_upper_m{k}"), (f"p_upper_m{k}", f"p_lower_m{k}")):
                pa, pb = df.loc[i, a_], df.loc[j, b_]
                tol = 1e-9 if exact else 6 * np.sqrt(max(pa, 1e-4) * (1 - min(pa, 0.9999)) / MC) + 2e-5
                if abs(pa - pb) > tol:
                    raise AssertionError(f"TOST side reversal failed {r.experiment} {r.config} {r.contrast} {pa} {pb}")
        df.loc[i, "reversal_check"] = ("estimate, 95/90% CI and TOST sides exactly reversed vs mirror row" if exact else
                                       "estimate and 95/90% CI exactly reversed; TOST sides reversed within Monte Carlo "
                                       "error (independent MC streams per side)")
    for i, r in df.iterrows():
        ci = [r.ci95_lo_pp, r.ci95_hi_pp]
        for k in (10, 5):
            j = SC.judge(r.estimate_pp, ci, r.p_diff_holm, 0.05, r[f"p_equiv_m{k}_holm"], k)
            df.loc[i, f"equiv_m{k}_holm"] = j["equivalent"]
            df.loc[i, f"equiv_m{k}_unadjusted"] = bool(r[f"p_equiv_m{k}"] < 0.05)
            df.loc[i, f"label_m{k}"] = j["label"]
        df.loc[i, "diff_from_0_holm"] = bool(r.p_diff_holm < 0.05)
        df.loc[i, "diff_from_0_unadjusted"] = bool(r.p_diff_signflip < 0.05)
        df.loc[i, "ci95_excludes_0"] = bool(np.isfinite(ci).all() and (ci[0] > 0 or ci[1] < 0))
        df.loc[i, "difference_judgment"] = ("different from 0 (Holm)" if r.p_diff_holm < 0.05 else
                                            ("different from 0 unadjusted only" if r.p_diff_signflip < 0.05 else
                                             "not different from 0"))
        for k in (10, 5):
            df.loc[i, f"equivalence_judgment_m{k}"] = (
                f"equivalent within +-{k} pp (Holm)" if df.loc[i, f"equiv_m{k}_holm"] else
                (f"equivalent within +-{k} pp unadjusted only" if df.loc[i, f"equiv_m{k}_unadjusted"] else
                 f"equivalence +-{k} pp not established"))
    df.to_csv(os.path.join(OUT, "DIRECT_EFFECTS.csv"), index=False, float_format="%.6g")
    print(f"[done] {len(df)} rows in {time.time() - t0:.0f}s")
    b4(pairs)
    interp(df)
    json.dump({"seed": SEED, "bootstrap_B": B, "signflip_MC": MC, "signflip_MC_inversion_R": MC_INV,
               "exact_max_clusters": SC.EXACT_MAX, "margins": MARGINS, "clusterings": CLUSTERINGS,
               "runtime_s": time.time() - t0}, open(os.path.join(OUT, "b_stats_meta.json"), "w"), indent=1)


# ----------------------------------------------------------------------------------------------------- B4
B4_METRICS = ["gen_suffix_tokfrac", "gen_suffix_trans_m", "ctx_exec_dev_trans_m", "cmd_final_m", "tcp_mean_cm",
              "tcp_area_cmstep", "tcp_final_cm"]


def mean_ci_cont(xx, col, ccol="cluster_primary"):
    v = xx[col]
    if v.isna().all():
        return np.nan, np.nan, np.nan, 0
    x = xx[v.notna()].copy()
    x["d"] = x[col]
    _, tk, N, D = cluster_arrays(x, ccol)
    bs = strat_boot(tk, N, D, B=2000, seed=SEED + 3)
    lo, hi = SC.pct_ci(bs, .95)
    return SC.ratio(N, D), lo, hi, len(x)


def b4(pairs):
    sel = [("exp34", "r1e4", c, d) for d in ("perp_left", "opposite") for c in ("R-N", "C-F", "F-N", "C-N")]
    sel += [("exp34", "r1e4", c, "-") for c in ("gen-N", "mix-N")]
    sel += [("exp35", cfg, c, "opposite") for cfg in ("r1e4", "r4e1", "r1e2", "r1e1", "r2e2", "r2e1")
            for c in ("R-N", "C-F", "F-N", "C-N")]
    sel += [("exp35", cfg, "gen-N", "opposite") for cfg in ("r1e4", "r4e1")]
    sel += [("P1D", cfg, c, "opposite") for cfg in ("r1e4", "r4e1", "r1e1", "r2e1") for c in ("R-N", "C-F", "F-N", "C-N")]
    sel += [("P1D", "r1e4", c, "opposite") for c in ("CD-FD", "FD-N", "CD-N")]
    sel += [("P1D", cfg, c, "opposite") for cfg in ("r1e1", "r4e1") for c in ("gen-N",)]
    sel += [("P1D", cfg, "null-N", "opposite") for cfg in ("r1e1", "r2e1", "r4e1", "r1e4")]
    rows = []
    for exp, cfg, cn, dr in sel:
        x = pairs[(pairs.experiment == exp) & (pairs.config == cfg) & (pairs.contrast == cn) & (pairs.direction == dr)]
        subsets = [("all units of the contrast", x)]
        if cn == "gen-N":     # same-subset comparison rows: R-N and C-F on the rerun units
            seeds = set(zip(x.task, x.seed))
            for cc in ("R-N", "C-F"):
                y = pairs[(pairs.experiment == exp) & (pairs.config == cfg) & (pairs.contrast == cc) &
                          (pairs.direction.isin(["opposite"]))]
                y = y[[k in seeds for k in zip(y.task, y.seed)]]
                subsets.append((f"{cc} on the rerun units", y))
        for sname, xs in subsets:
            if len(xs) == 0:
                continue
            for lvl in TASKS + ("pooled",):
                xx = xs if lvl == "pooled" else xs[xs.task == lvl]
                if len(xx) == 0:
                    continue
                c = cn if sname.startswith("all") else sname.split(" ")[0]
                xx = xx.copy()
                xx["d"] = xx["d_success"]
                r = infer_light(xx)
                row = {"experiment": exp, "config": cfg, "direction": dr, "contrast": c, "subset": sname, "task": lvl,
                       "n_units": len(xx), "n_clusters_primary": r["G"],
                       "success_effect_pp": 100 * r["est"], "success_ci95_lo_pp": 100 * r["ci"][0],
                       "success_ci95_hi_pp": 100 * r["ci"][1],
                       "discordance_pct": 100 * float(np.mean([*mean_ci_cont(xx, "discordant")[:1]])),
                       "prefix_identical_pct": 100 * float(xx["prefix_identical"].mean()),
                       "gen_suffix_source": xx["gen_suffix_source"].fillna("n/a").astype(str).value_counts().index[0]}
                for mcol in B4_METRICS:
                    mu, lo, hi, n = mean_ci_cont(xx, mcol)
                    row[mcol] = mu
                    row[f"{mcol}_ci95_lo"], row[f"{mcol}_ci95_hi"] = lo, hi
                    if mcol.startswith("gen_suffix"):
                        row["gen_suffix_n"] = n
                rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "B4_TRAJECTORY_VS_SUCCESS.csv"), index=False, float_format="%.6g")
    print(f"[b4] {len(rows)} rows")


def infer_light(xx):
    _, tk, N, D = cluster_arrays(xx, "cluster_primary")
    G = len(N)
    th = SC.ratio(N, D)
    if G <= SC.EXACT_MAX:
        ci = SC.signflip_ci(N, D, th, .95)
    else:
        ci = mc_inv_ci(N, D, th, .95, sign_matrix(G))
    return {"est": th, "ci": ci, "G": G}


# ----------------------------------------------------------------------------------------------------- rules
def interp(df):
    """Interpretation rule (prompt section B) per experiment x config x task, primary and physical clustering."""
    out = []
    core_dir = df[(df.role == "core") & (df.kind == "direct") & df.contrast.isin(["R-N", "C-F"])]
    for (exp, cfg, lvl, cl), g in core_dir.groupby(["experiment", "config", "task", "clustering"]):
        if cl == "episode":
            continue
        for _, r in g.iterrows():
            ints = df[(df.experiment == exp) & (df.kind == "interaction") & (df.role == "core") &
                      (df.task == lvl) & (df.clustering == cl) & df.contrast.str.contains(
                          "delta_R|R-N" if r.contrast == "R-N" else "delta_C|C-F")]
            ints = ints[ints.config.str.contains(cfg.split("/")[0])]
            int_eq10 = bool(len(ints)) and bool(ints["equiv_m10_holm"].all())
            width = r.ci95_hi_pp - r.ci95_lo_pp
            ep = df[(df.experiment == exp) & (df.config == cfg) & (df.contrast == r.contrast) & (df.task == lvl) &
                    (df.clustering == "episode")].iloc[0]
            rules = []
            if r.equiv_m10_holm:
                rules.append("R2: direct-effect equivalence holds (+-10 pp, Holm) -> success effect may be bounded "
                             "for this task/config/margin only" + ("; also +-5 pp" if r.equiv_m5_holm else
                                                                    "; +-5 pp NOT established"))
            elif r.diff_from_0_holm:
                rules.append("direct effect different from 0 (Holm)")
            else:
                rules.append(f"R3: direct-effect CI wide ({r.ci95_lo_pp:+.1f}, {r.ci95_hi_pp:+.1f}; width "
                             f"{width:.1f} pp, no equivalence at +-10) -> generation/trajectory deviation confirmed, "
                             "success effect uncertain")
            if int_eq10 and not r.equiv_m10_holm:
                rules.append("R1: interaction equivalence only -> write only 'difference between execution configs "
                             "within +-10 pp'")
            if bool(ep.equiv_m10_holm) and not bool(r.equiv_m10_holm):
                rules.append("R4: equivalence at episode level lost after initial-state clustering -> report loss")
            out.append({"experiment": exp, "config": cfg, "task": lvl, "clustering": cl, "contrast": r.contrast,
                        "estimate_pp": r.estimate_pp, "ci95": f"[{r.ci95_lo_pp:+.1f}, {r.ci95_hi_pp:+.1f}]",
                        "n_clusters": r.n_clusters, "direct_label_m10": r.label_m10, "direct_label_m5": r.label_m5,
                        "core_interactions_equiv_m10": int_eq10, "rules": " | ".join(rules)})
    pd.DataFrame(out).to_csv(os.path.join(OUT, "INTERPRETATION.csv"), index=False, float_format="%.4g")
    print(f"[interp] {len(out)} rows")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--post":      # re-run only the post-processing from the cached rows
        post(pd.read_pickle(os.path.join(OUT, "_raw_rows.pkl")), pd.read_csv(os.path.join(OUT, "pairs.csv")), time.time())
    else:
        main()
