#!/usr/bin/env python
"""
stats_core -- shared cluster statistics for the review_followup2 (2026-10-13) POST-HOC analyses.

Pure numpy/scipy, CPU only, no project-specific imports, so every analysis agent (A AutoVLA/Impromptu cluster
re-analysis, B SpatialVLA direct effects/equivalence, C baseline/window recheck) can import it:

    sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/review_followup2")
    import stats_core as SC

DATA MODEL
----------
Paired rows: one row per (top-level cluster, ..., scene/initial state, perturbation/repeat) with the outcome under
condition a and under condition b (or directly d = a - b). Pairing is NEVER broken: every resampling / sign flip acts on
whole top-level clusters, so all rows of a cluster (all its logs, scenes, perturbations and both conditions) move
together.

  pair_rows(long_df, key_cols, cond_col, value_col, cond_a, cond_b)   long -> paired (reports unpaired keys)
  scene_table(paired, cluster, scene, d="d", w=None)                   paired rows -> one row per scene
  cluster_arrays(sc, weighting)                                        scene table -> per-cluster (labels, N, D)

Every estimand is written as a ratio  theta = sum_g N_g / sum_g D_g  over top-level clusters g:

  weighting "unit"          N_g = sum_s w_s * (sum of unit d in s)      D_g = sum_s w_s * n_units_s
                            (w_s = 1 -> plain mean over all paired rows = original "unit-weighted" estimate)
  weighting "scene"         N_g = sum_s w_s * mean_u d_su               D_g = sum_s w_s
                            (w_s = 1 -> scenes equally weighted; with design weights w_s -> Hajek estimator)
  weighting "cluster_equal" N_g = (scene-weighted ratio inside g)        D_g = 1
                            (each top-level cluster -- e.g. drive -- equally weighted; = "drive-equal")

These are DIFFERENT ESTIMANDS (they weight scenes/clusters differently); report them in separate columns, never pick
the one that looks best.

INFERENCE
---------
  cluster_bootstrap(N, D, B, seed)          resample top-level clusters with replacement (multinomial counts), the
                                            ratio is re-computed -> percentile CIs. Lower levels are NOT resampled
                                            separately: with a top-level-only bootstrap all within-cluster variance is
                                            carried by the cluster totals; resampling every level independently would
                                            double count it (Davison & Hinkley 1997, sec. 3.8; Field & Welsh 2007).
  signflip_test(T, alternative, ...)        sign-flip test on cluster totals T_g (= N_g - delta * D_g).
                                            EXACT enumeration of all 2^G sign patterns if G <= exact_max (meet in the
                                            middle), otherwise Monte Carlo with recorded reps/seed and p = (b+1)/(R+1).
      ASSUMPTIONS (state them in every report): under H0 (theta = delta) the cluster totals T_g are independent
      across clusters and each is symmetric about 0 (sign symmetry of the cluster-level difference distribution).
      Conditions in the VLA experiments are deterministic interventions on the same input, not randomised
      assignments, so this is NOT a randomization test; it is a conditional test under the symmetry assumption.
      The smallest attainable two-sided p is 2 / 2^(number of clusters with T_g != 0).
  signflip_ci(N, D, theta, level)           test-inversion CI {delta : p_two-sided(N - delta D) > 1 - level}
  cluster_t(N, D)                           ratio linearisation, cluster-robust SE, t with G-1 df (auxiliary)
  tost(N, D, margin, ...)                   paired-binary (or continuous) TOST for clustered data:
                                            p_lower: H0 theta <= -margin, p_upper: H0 theta >= +margin,
                                            p_equiv = max(p_lower, p_upper); reported with 90% and 95% CIs.
                                            method "signflip" (one-sided shifted sign-flip; assumption: the SHIFTED
                                            totals N_g - (+-margin) D_g are symmetric about 0 at the boundary) or
                                            "bootstrap" (p from the cluster-bootstrap distribution, (b+1)/(B+1)).
  holm(ps)                                  Holm step-down adjusted p values.
  leave_one_cluster_out(sc, weighting, ...) theta / CI / sign-flip p with each top-level cluster removed.
  judge(...)                                SEPARATE judgments: difference from 0 (adjusted p and CI) and
                                            equivalence (TOST p_equiv, possibly Holm-adjusted). Both can be true; when
                                            neither holds the label is "uncertain". Never an exclusive if/else.
  analyze(sc, ...)                          convenience wrapper returning all of the above for one weighting.

All randomness uses numpy default_rng(seed); seeds and reps are returned in the result dicts.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

DEFAULT_SEED = 20261013
DEFAULT_B = 10_000
DEFAULT_MC = 200_000
EXACT_MAX = 26          # 2^13 x 2^13 meet-in-the-middle; ~ms per call
WEIGHTINGS = ("unit", "scene", "cluster_equal")
_TOL = 1e-12


# ======================================================================================================== data shaping
def pair_rows(long_df: pd.DataFrame, key_cols: Sequence[str], cond_col: str, value_col: str, cond_a, cond_b,
              keep_cols: Sequence[str] = ()) -> (pd.DataFrame, Dict):
    """long (one row per key x condition) -> paired rows with columns key_cols + keep_cols + a, b, d (= a - b).
    Rows whose key lacks either condition (or has a missing value) are dropped and COUNTED in the report.
    Raises if a key has duplicate rows for one condition (that would silently change weights)."""
    key_cols = list(key_cols)
    sub = long_df[long_df[cond_col].isin([cond_a, cond_b])]
    dup = sub.duplicated(key_cols + [cond_col]).sum()
    if dup:
        raise ValueError(f"pair_rows: {dup} duplicate (key, condition) rows")
    keep = list(dict.fromkeys(keep_cols))
    piv = sub.set_index(key_cols + [cond_col])[value_col].unstack(cond_col)
    meta = sub.drop_duplicates(key_cols).set_index(key_cols)[keep] if keep else None
    n_keys = len(piv)
    for c in (cond_a, cond_b):
        if c not in piv.columns:
            piv[c] = np.nan
    ok = piv[cond_a].notna() & piv[cond_b].notna()
    out = piv[ok][[cond_a, cond_b]].rename(columns={cond_a: "a", cond_b: "b"})
    if meta is not None:
        out = out.join(meta, how="left")
    out = out.reset_index()
    out["d"] = out["a"].astype(float) - out["b"].astype(float)
    rep = {"n_keys": int(n_keys), "n_paired": int(ok.sum()), "n_unpaired_or_missing": int((~ok).sum())}
    return out, rep


def scene_table(paired: pd.DataFrame, cluster: str, scene: str, d: str = "d", w: Optional[str] = None,
                extra: Sequence[str] = ()) -> pd.DataFrame:
    """paired rows -> one row per (cluster, scene): n_units, dsum, dmean, w (design weight, default 1).
    `scene` is the averaging unit below the cluster (scene / initial state); `extra` columns (e.g. log, stratum) must
    be constant within a scene and are carried along."""
    if paired.empty:
        return pd.DataFrame(columns=["cluster", "scene", "n_units", "dsum", "dmean", "w", *extra])
    g = paired.groupby([cluster, scene], sort=True)
    sc = g[d].agg(n_units="size", dsum="sum").reset_index()
    sc["dmean"] = sc["dsum"] / sc["n_units"]
    if w is None:
        sc["w"] = 1.0
    else:
        ww = g[w].agg(["min", "max"]).reset_index()
        if not np.allclose(ww["min"], ww["max"]):
            raise ValueError("scene_table: weight not constant within scene")
        sc["w"] = ww["min"].values.astype(float)
    keys = {cluster: "cluster", scene: "scene"}
    for e in extra:
        if e in keys:
            continue
        ee = g[e].agg(["nunique"]).reset_index()
        if (ee["nunique"] > 1).any():
            raise ValueError(f"scene_table: column {e} not constant within scene")
        sc[e] = g[e].first().values
    sc = sc.rename(columns=keys)
    for e in extra:                      # an extra column that is also a key is kept under its own name too
        if e in keys:
            sc[e] = sc[keys[e]].values
    return sc


def cluster_arrays(sc: pd.DataFrame, weighting: str = "scene"):
    """scene table -> (labels, N, D) with theta = sum N / sum D (see module docstring)."""
    if weighting not in WEIGHTINGS:
        raise ValueError(weighting)
    x = pd.DataFrame({"cluster": sc["cluster"].values})
    if weighting == "unit":
        x["N"] = (sc["w"] * sc["dsum"]).values
        x["D"] = (sc["w"] * sc["n_units"]).values
    else:
        x["N"] = (sc["w"] * sc["dmean"]).values
        x["D"] = sc["w"].values
    g = x.groupby("cluster", sort=True)[["N", "D"]].sum()
    labels = g.index.tolist()
    N = g["N"].values.astype(float)
    D = g["D"].values.astype(float)
    if weighting == "cluster_equal":
        N = N / D
        D = np.ones_like(N)
    return labels, N, D


def ratio(N, D) -> float:
    N, D = np.asarray(N, float), np.asarray(D, float)
    return float(N.sum() / D.sum()) if D.sum() > 0 else float("nan")


# ======================================================================================================== bootstrap
def boot_counts(G: int, B: int = DEFAULT_B, seed: int = DEFAULT_SEED) -> np.ndarray:
    """B x G multinomial cluster counts (cluster bootstrap index); reuse the same counts across weightings/contrasts
    of one data set to keep the joint resampling."""
    return np.random.default_rng(seed).multinomial(G, np.ones(G) / G, size=B).astype(float)


def cluster_bootstrap(N, D, B: int = DEFAULT_B, seed: int = DEFAULT_SEED, counts: Optional[np.ndarray] = None):
    """distribution of theta* = (C N)/(C D) over cluster-bootstrap replicates (non-finite replicates dropped)."""
    N, D = np.asarray(N, float), np.asarray(D, float)
    C = counts if counts is not None else boot_counts(len(N), B, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = (C @ N) / (C @ D)
    return x[np.isfinite(x)]


def pct_ci(x, level: float = 0.95):
    if len(x) == 0:
        return [float("nan"), float("nan")]
    a = (1 - level) / 2 * 100
    return [float(np.percentile(x, a)), float(np.percentile(x, 100 - a))]


# ======================================================================================================== sign flip
def _half_sums(v):
    s = np.zeros(1)
    for x in v:
        s = np.concatenate([s + x, s - x])
    return s


def signflip_test(T, alternative: str = "two-sided", exact_max: int = EXACT_MAX, reps: int = DEFAULT_MC,
                  seed: int = DEFAULT_SEED) -> Dict:
    """Sign-flip test of H0: E sum T = 0 on cluster totals T (see module docstring for assumptions).
    alternative: 'two-sided' (|sum eps T| >= |sum T|), 'greater' (sum eps T >= sum T), 'less' (<=).
    Exact (all 2^G patterns, observed pattern included, p = count / 2^G) if G <= exact_max; else Monte Carlo
    p = (b + 1) / (R + 1)."""
    T = np.asarray(T, float)
    G = len(T)
    n_inf = int(np.sum(np.abs(T) > _TOL * max(1.0, np.abs(T).max() if G else 1.0)))
    res = {"n_clusters": G, "n_informative": n_inf, "alternative": alternative,
           "p_min_attainable": (2.0 ** (1 - n_inf) if alternative == "two-sided" else 2.0 ** (-n_inf)) if n_inf else 1.0}
    if G == 0:
        return res | {"p": float("nan"), "method": "none"}
    obs = float(T.sum())
    scale = max(1.0, float(np.abs(T).sum()))
    tol = _TOL * scale
    if G <= exact_max:
        a = _half_sums(T[: G // 2])
        b = np.sort(_half_sums(T[G // 2:]))
        tot = len(a) * len(b)
        if alternative == "two-sided":
            thr = abs(obs) - tol
            if thr <= 0:
                p = 1.0
            else:
                hi = len(b) - np.searchsorted(b, thr - a, side="left")
                lo = np.searchsorted(b, -thr - a, side="right")
                p = float((hi + lo).sum() / tot)
        elif alternative == "greater":
            cnt = len(b) - np.searchsorted(b, obs - tol - a, side="left")
            p = float(cnt.sum() / tot)
        elif alternative == "less":
            cnt = np.searchsorted(b, obs + tol - a, side="right")
            p = float(cnt.sum() / tot)
        else:
            raise ValueError(alternative)
        return res | {"p": min(1.0, p), "method": f"exact 2^{G}", "reps": None, "seed": None}
    rng = np.random.default_rng(seed)
    cnt = 0
    done = 0
    while done < reps:
        k = min(20_000, reps - done)
        E = rng.integers(0, 2, size=(k, G), dtype=np.int8).astype(float) * 2 - 1
        s = E @ T
        if alternative == "two-sided":
            cnt += int(np.sum(np.abs(s) >= abs(obs) - tol))
        elif alternative == "greater":
            cnt += int(np.sum(s >= obs - tol))
        else:
            cnt += int(np.sum(s <= obs + tol))
        done += k
    p = (cnt + 1) / (reps + 1)
    return res | {"p": float(p), "method": f"Monte Carlo R={reps} seed={seed}, p=(b+1)/(R+1)", "reps": reps, "seed": seed}


def signflip_ci(N, D, theta: float, level: float = 0.95, exact_max: int = EXACT_MAX, iters: int = 50,
                span: Optional[float] = None):
    """Test-inversion CI for theta = sum N / sum D: {delta : p_two-sided(N - delta D) > 1 - level}, by bisection on
    each side of theta (assumes the acceptance set is an interval). Exact enumeration only (returns nan if G is too
    large) -- Monte Carlo inversion is too noisy to be useful."""
    N, D = np.asarray(N, float), np.asarray(D, float)
    if len(N) < 2 or len(N) > exact_max or not np.isfinite(theta):
        return [float("nan"), float("nan")]
    a = 1 - level
    if 2.0 ** (1 - len(N)) > a:          # even the most extreme pattern cannot reject: the CI is unbounded
        return [float("-inf"), float("inf")]
    if span is None:
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(D > 0, N / np.where(D > 0, D, 1), 0)
        span = max(1.0, 4 * float(np.abs(r).max()), 4 * abs(theta))

    def acc(dl):
        return signflip_test(N - dl * D, "two-sided", exact_max)["p"] > a + 1e-15

    out = []
    for direction in (-1, 1):
        lo_, hi_ = theta, theta + direction * span
        if acc(hi_):
            out.append(hi_)
            continue
        for _ in range(iters):
            mid = 0.5 * (lo_ + hi_)
            if acc(mid):
                lo_ = mid
            else:
                hi_ = mid
        out.append(lo_)
    return [float(min(out)), float(max(out))]


# ======================================================================================================== cluster-t
def cluster_t(N, D, level: float = 0.95):
    N, D = np.asarray(N, float), np.asarray(D, float)
    G = len(N)
    if G < 2 or D.sum() <= 0:
        return float("nan"), [float("nan"), float("nan")]
    th = N.sum() / D.sum()
    r = N - th * D
    se = math.sqrt(G / (G - 1) * float(np.sum(r ** 2))) / D.sum()
    if se == 0:
        return (1.0 if th == 0 else 0.0), [float(th), float(th)]
    q = tdist.ppf(1 - (1 - level) / 2, G - 1)
    return float(2 * tdist.sf(abs(th / se), G - 1)), [float(th - q * se), float(th + q * se)]


# ======================================================================================================== TOST
def tost(N, D, margin: float, method: str = "signflip", B: int = DEFAULT_B, seed: int = DEFAULT_SEED,
         counts: Optional[np.ndarray] = None, exact_max: int = EXACT_MAX, reps: int = DEFAULT_MC) -> Dict:
    """Two one-sided tests for |theta| < margin on clustered paired data (binary or continuous outcomes).
    margin > 0, same scale as theta (e.g. 0.05 = 5 percentage points for a success-rate difference).
      p_lower : H0 theta <= -margin  vs  theta > -margin
      p_upper : H0 theta >= +margin  vs  theta < +margin
      p_equiv = max(p_lower, p_upper)   (equivalent at alpha iff p_equiv < alpha; uncorrected unless Holm is applied
                                         to p_equiv across a declared family)
    method 'signflip': one-sided sign-flip on shifted totals N_g -(+-margin) D_g (assumes symmetry at the boundary).
    method 'bootstrap': p_lower = (#{theta* <= -margin}+1)/(B+1), p_upper = (#{theta* >= margin}+1)/(B+1).
    Both 90% and 95% CIs (bootstrap percentile + sign-flip inversion) are returned; equivalence by the 90% CI being
    inside (-margin, margin) is the alpha = .05 TOST dual, the 95% CI is the alpha = .025-per-side version."""
    if margin <= 0:
        raise ValueError("margin must be > 0")
    N, D = np.asarray(N, float), np.asarray(D, float)
    th = ratio(N, D)
    bs = cluster_bootstrap(N, D, B, seed, counts)
    out = {"theta": th, "margin": margin, "method": method,
           "boot_ci90": pct_ci(bs, 0.90), "boot_ci95": pct_ci(bs, 0.95),
           "sf_ci90": signflip_ci(N, D, th, 0.90, exact_max), "sf_ci95": signflip_ci(N, D, th, 0.95, exact_max)}
    if method == "signflip":
        lo = signflip_test(N + margin * D, "greater", exact_max, reps, seed)
        up = signflip_test(N - margin * D, "less", exact_max, reps, seed + 1)
        out["p_lower"], out["p_upper"] = lo["p"], up["p"]
        out["p_method"] = lo["method"]
    elif method == "bootstrap":
        Bn = len(bs)
        out["p_lower"] = float((np.sum(bs <= -margin) + 1) / (Bn + 1))
        out["p_upper"] = float((np.sum(bs >= margin) + 1) / (Bn + 1))
        out["p_method"] = f"cluster bootstrap B={Bn} seed={seed}, (b+1)/(B+1)"
    else:
        raise ValueError(method)
    out["p_equiv"] = max(out["p_lower"], out["p_upper"])
    for k in ("boot", "sf"):
        ci = out[f"{k}_ci90"]
        out[f"equiv_{k}_ci90"] = bool(np.all(np.isfinite(ci)) and -margin < ci[0] and ci[1] < margin)
    return out


# ======================================================================================================== multiplicity
def holm(ps: Iterable[float]) -> np.ndarray:
    ps = np.asarray(list(ps), float)
    m = len(ps)
    adj = np.full(m, np.nan)
    ok = np.isfinite(ps)
    idx = np.where(ok)[0]
    order = idx[np.argsort(ps[idx], kind="mergesort")]
    mm = len(order)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (mm - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


# ======================================================================================================== wrappers
def analyze(sc: pd.DataFrame, weighting: str = "scene", B: int = DEFAULT_B, seed: int = DEFAULT_SEED,
            counts: Optional[np.ndarray] = None, ci_inversion: bool = True, exact_max: int = EXACT_MAX,
            reps: int = DEFAULT_MC, margin: Optional[float] = None, tost_method: str = "signflip") -> Dict:
    """Full inference for one scene table and one weighting (clusters = sc.cluster)."""
    labels, N, D = cluster_arrays(sc, weighting)
    G = len(N)
    out = {"weighting": weighting, "n_clusters": G, "n_scenes": int(len(sc)), "n_units": int(sc["n_units"].sum()),
           "theta": ratio(N, D), "B": B, "seed": seed}
    if counts is not None and counts.shape[1] != G:
        counts = None
    bs = cluster_bootstrap(N, D, B, seed, counts)
    out["boot_ci95"], out["boot_ci90"] = pct_ci(bs, 0.95), pct_ci(bs, 0.90)
    sf = signflip_test(N, "two-sided", exact_max, reps, seed + 7)
    out["p_signflip"], out["sf_method"] = sf["p"], sf["method"]
    out["n_informative"], out["p_min_attainable"] = sf["n_informative"], sf["p_min_attainable"]
    out["p_cluster_t"], out["clt_ci95"] = cluster_t(N, D)
    if ci_inversion:
        out["sf_ci95"] = signflip_ci(N, D, out["theta"], 0.95, exact_max)
        out["sf_ci90"] = signflip_ci(N, D, out["theta"], 0.90, exact_max)
    if margin is not None:
        out["tost"] = tost(N, D, margin, tost_method, B, seed, counts, exact_max, reps)
    out["_arrays"] = (labels, N, D)
    return out


def leave_one_cluster_out(sc: pd.DataFrame, weighting: str = "scene", B: int = 2_000, seed: int = DEFAULT_SEED,
                          exact_max: int = EXACT_MAX, reps: int = 20_000) -> pd.DataFrame:
    full = ratio(*cluster_arrays(sc, weighting)[1:])
    rows = []
    for c in sorted(sc["cluster"].unique()):
        s = sc[sc["cluster"] != c]
        if s["cluster"].nunique() < 2:
            continue
        r = analyze(s, weighting, B, seed, None, False, exact_max, reps)
        rows.append({"left_out": c, "theta": r["theta"], "influence": full - r["theta"],
                     "boot_ci95_lo": r["boot_ci95"][0], "boot_ci95_hi": r["boot_ci95"][1],
                     "p_signflip": r["p_signflip"], "sf_method": r["sf_method"], "n_clusters": r["n_clusters"],
                     "sign_change": bool(np.sign(r["theta"]) != np.sign(full) and full != 0),
                     "ci_includes_0": bool(r["boot_ci95"][0] <= 0 <= r["boot_ci95"][1])})
    return pd.DataFrame(rows)


def judge(theta: float, ci95: Sequence[float], p_adj: float, alpha: float = 0.05,
          p_equiv_adj: Optional[float] = None, margin: Optional[float] = None, predicted_sign: Optional[int] = None) -> Dict:
    """Separate (non-exclusive) judgments. diff_from_0: adjusted p < alpha (and CI excludes 0 is reported alongside);
    equivalent: p_equiv (adjusted if a family is declared) < alpha. label: one of
    'different_from_0', 'equivalent_within_margin', 'different_from_0_and_within_margin', 'uncertain'
    ('uncertain' = neither conclusion; if no margin was declared, equivalence is 'not_assessed')."""
    diff = bool(np.isfinite(p_adj) and p_adj < alpha)
    ci_ex0 = bool(np.all(np.isfinite(ci95)) and (ci95[0] > 0 or ci95[1] < 0))
    eq = None if (margin is None or p_equiv_adj is None) else bool(np.isfinite(p_equiv_adj) and p_equiv_adj < alpha)
    if diff and eq:
        label = "different_from_0_and_within_margin"
    elif diff:
        label = "different_from_0"
    elif eq:
        label = "equivalent_within_margin"
    else:
        label = "uncertain" if margin is not None else "not_detected (equivalence not assessed)"
    out = {"diff_from_0": diff, "ci95_excludes_0": ci_ex0, "equivalent": eq, "label": label}
    if predicted_sign is not None:
        out["sign_as_predicted"] = bool(np.sign(theta) == predicted_sign)
    return out
