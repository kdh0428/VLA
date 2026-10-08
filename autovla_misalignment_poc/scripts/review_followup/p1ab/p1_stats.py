#!/usr/bin/env python
"""
Cluster statistics for P1-A / P1-B / R8R9 (CPU). Estimators copied from P0-B
(outputs/review_followup/P0B_cluster_stats/run_20261008/p0b_analysis.py) so that results are
directly comparable:
  - primary estimand: scene-weighted mean of a paired per-unit difference d (units of a scene are
    averaged first, then scenes are pooled; sum over logs of scene means / number of scenes)
  - log block bootstrap (multinomial log weights, B = 10,000, percentile 95% and 90% CI)
  - log-level sign-flip test on the per-log sums of scene means (exact for <= 20 logs, else MC)
    -- conditional test under symmetry of per-log summaries; NOT exact randomization inference
  - cluster-robust t (df = G - 1) as a sensitivity
  - unit-weighted and log-uniform weights as sensitivities
  - Holm within a predeclared family; TOST equivalence via the 90% CI inside +-margin
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

SEED = 20261008
B = 10_000
MC = 200_000
EXACT_MAX = 20


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def sign_matrix(G):
    return (((np.arange(2 ** G)[:, None] >> np.arange(G)) & 1) * 2 - 1).astype(np.float64)


def signflip(vals, denom, seed, mc=MC):
    vals = np.asarray(vals, float); G = len(vals); obs = abs(vals.sum() / denom)
    if G <= EXACT_MAX:
        T = np.abs(sign_matrix(G) @ vals / denom)
        return float(np.mean(T >= obs - 1e-12)), f"exact 2^{G}"
    rng = np.random.default_rng(seed); cnt = 0; done = 0
    while done < mc:
        k = min(50_000, mc - done)
        E = rng.integers(0, 2, size=(k, G)) * 2 - 1
        cnt += int(np.sum(np.abs(E @ vals / denom) >= obs - 1e-12)); done += k
    return float((cnt + 1) / (mc + 1)), f"MC {mc}"


def log_arrays(df):
    sc = df.groupby(["log", "scene"])["d"].agg(["mean", "sum", "count"]).reset_index()
    logs = sorted(sc["log"].unique())
    g = sc.groupby("log")
    S = g["mean"].sum().reindex(logs).values.astype(float)
    n = g.size().reindex(logs).values.astype(float)
    U = g["sum"].sum().reindex(logs).values.astype(float)
    m = g["count"].sum().reindex(logs).values.astype(float)
    return logs, S, n, U, m, S / n


def cluster_t(num, w):
    G = len(num)
    if G < 2:
        return float("nan")
    th = num.sum() / w.sum(); r = num - th * w
    se = np.sqrt(G / (G - 1) * np.sum(r ** 2)) / w.sum()
    if se == 0:
        return 1.0 if th == 0 else 0.0
    return float(2 * tdist.sf(abs(th / se), G - 1))


def log_cluster_effects(df, seed=SEED, b=B, mc=MC, with_tests=True):
    """df: columns log, scene, d (paired per-unit difference, NaN rows must be dropped by caller)."""
    if len(df) == 0:
        return {"n_clusters": 0, "n_scenes": 0, "n_units": 0}
    logs, S, n, U, m, L = log_arrays(df)
    G = len(logs)
    est = {"scene": S.sum() / n.sum(), "unit": U.sum() / m.sum(), "log": L.mean()}
    rng = np.random.default_rng(seed)
    C = rng.multinomial(G, np.ones(G) / G, size=b).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = {"scene": C @ S / (C @ n), "unit": C @ U / (C @ m), "log": C @ L / G}
    res = {}
    for w in ("scene", "unit", "log"):
        x = bs[w][np.isfinite(bs[w])]
        r = {"estimate": float(est[w]), "ci95": [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))],
             "ci90": [float(np.percentile(x, 5)), float(np.percentile(x, 95))]}
        if with_tests:
            num, den = {"scene": (S, n), "unit": (U, m), "log": (L, np.ones(G))}[w]
            r["p_signflip"], r["sf_method"] = signflip(num, den.sum(), seed + 1, mc)
            r["p_cluster_t"] = cluster_t(num, den)
        res[w] = r
    res["n_clusters"] = G; res["n_scenes"] = int(n.sum()); res["n_units"] = int(m.sum())
    res["n_informative_logs"] = int(np.sum(np.abs(S) > 1e-12))
    return res


def paired_frame(units: pd.DataFrame, cond_a: str, cond_b: str, metric: str, mask=None) -> pd.DataFrame:
    """units: long table with unit_id, scene_id, log_id, condition, <metric>. Returns log, scene, uid, a, b, d
    on the paired common set (rows with NaN metric in either condition are dropped and counted)."""
    u = units if mask is None else units[mask]
    a = u[u.condition == cond_a].set_index("unit_id")
    b = u[u.condition == cond_b].set_index("unit_id")
    j = a[["scene_id", "log_id", metric]].join(b[[metric]], rsuffix="_b", how="inner")
    j = j.rename(columns={metric: "a", f"{metric}_b": "b", "scene_id": "scene", "log_id": "log"})
    j["a"] = pd.to_numeric(j["a"], errors="coerce").astype(float)
    j["b"] = pd.to_numeric(j["b"], errors="coerce").astype(float)
    dropped = int(j[["a", "b"]].isna().any(axis=1).sum())
    j = j.dropna(subset=["a", "b"]).copy()
    j["d"] = j["a"] - j["b"]
    j.attrs["dropped_nan"] = dropped
    j.attrs["missing_pairs"] = int(len(set(a.index) ^ set(b.index)))
    return j.reset_index()


def effect_row(family, contrast, group, metric, df, seed, margin=None, label=None):
    r = log_cluster_effects(df, seed=seed)
    row = {"family": family, "contrast": contrast, "group": group, "metric": metric, "label": label,
           "n_clusters": r.get("n_clusters"), "n_scenes": r.get("n_scenes"), "n_units": r.get("n_units"),
           "n_informative_logs": r.get("n_informative_logs"),
           "dropped_nan": df.attrs.get("dropped_nan"), "missing_pairs": df.attrs.get("missing_pairs")}
    for w in ("scene", "unit", "log"):
        if w in r:
            row[f"{w}_est"] = r[w]["estimate"]; row[f"{w}_ci95_lo"], row[f"{w}_ci95_hi"] = r[w]["ci95"]
            row[f"{w}_ci90_lo"], row[f"{w}_ci90_hi"] = r[w]["ci90"]
            row[f"{w}_p_signflip"] = r[w].get("p_signflip"); row[f"{w}_p_cluster_t"] = r[w].get("p_cluster_t")
    if "scene" in r:
        row["sf_method"] = r["scene"].get("sf_method")
    if margin is not None and "scene" in r:
        row["equiv_margin"] = margin
        row["equivalent_TOST90"] = bool(-margin < r["scene"]["ci90"][0] and r["scene"]["ci90"][1] < margin)
    return row


def add_holm(rows, family_name):
    idx = [i for i, r in enumerate(rows) if r["family"] == family_name and r.get("scene_p_signflip") is not None]
    if idx:
        adj = holm([rows[i]["scene_p_signflip"] for i in idx])
        for i, a in zip(idx, adj):
            rows[i]["holm_p"] = float(a); rows[i]["holm_m"] = len(idx)
    return rows


def loo_rows(family, contrast, group, metric, df, seed=SEED, b=2000):
    """leave-one-log-out for a paired frame (P0-B style): per left-out log, scene-weighted estimate,
    bootstrap CI (B = 2,000) and sign-flip p on the remaining logs."""
    out = []
    for lg in sorted(df["log"].unique()):
        sub = df[df["log"] != lg]
        if sub["log"].nunique() < 2:
            continue
        r = log_cluster_effects(sub, seed=seed, b=b, mc=20_000)
        out.append({"family": family, "contrast": contrast, "group": group, "metric": metric, "left_out_log": lg,
                    "estimate": r["scene"]["estimate"], "ci95_lo": r["scene"]["ci95"][0], "ci95_hi": r["scene"]["ci95"][1],
                    "p_signflip": r["scene"]["p_signflip"], "n_clusters": r["n_clusters"]})
    return out
