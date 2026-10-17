#!/usr/bin/env python
"""
a1_paired_ci -- paired-binary confidence intervals / equivalence tests for review_followup4 A1 (2026-10-17).

*** POST-HOC statistical-method correction module. Pure numpy/scipy, CPU only. Nothing here modifies existing results. ***

Why: stats_core.signflip_ci / stats_core.tost(method="signflip") invert a sign-flip test on SHIFTED cluster totals
T_g = N_g - delta * D_g. The sign-flip test is valid at delta = 0 (H0: cluster totals symmetric about 0, McNemar-type
null for paired binary data), but for delta != 0 the shifted totals of a discrete paired-binary difference are NOT
symmetric (the zero differences all become -delta), so the inversion / shifted TOST rests on a false assumption. With
all differences 0 every delta != 0 makes all G shifted totals the same sign -> p = 2 / 2^G -> the CI collapses to
[0 +- tol] (zero width); with all informative differences of one sign the CI end on that side sits at 0 (tie artifact).

Methods implemented here (all keep the pairing; no independent-proportion formula is used):
  wald_paired(b, c, n)                    Newcombe (1998) method 1 (simple asymptotic); degenerates when b = c = 0.
  newcombe10(a, b, c, d)                  Newcombe (1998) method 10: MOVER on Wilson limits of the two marginal
                                          proportions with the phi correction (phi = 0 if a margin is 0; positive
                                          covariance ad - bc reduced by n/2). Counts may be non-integer (used for the
                                          design-effect-deflated table). Verified against R ratesci 1.1.0 reference
                                          values (statsmodels main test-suite table) in verify().
  tango_ci(b, c, n)                       Tango (1998) asymptotic score interval (inversion of the score test for
                                          H0: p10 - p01 = delta0 with the restricted MLE of p01 in closed form).
                                          Restricted MLE verified numerically against brute-force maximisation.
  tango_test(b, c, n, delta0, alt)        score test p-value for a non-zero null (used for paired TOST).
  ci_pvalue(cifun, bound, side)           p-value of a one-sided test of H0 (theta <= -m or >= m) by CI inversion:
                                          the alpha at which the (1 - 2 alpha) two-sided limit touches the margin.
  rs_newcombe(y_a, y_b, clusters)         cluster-appropriate version for repeated physical/initial states:
                                          Newcombe 10 on the 2x2 table deflated by a design effect (Rao-Scott style
                                          effective sample size, Rao & Scott 1992; Korn & Graubard 1998).
                                          deff = clip(V_cluster(dbar) / V_iid(dbar), 1, sum m_g^2 / n), and the
                                          ICC = 1 upper bound sum m_g^2 / n when V_iid = 0 (no discordant pair:
                                          within-cluster dependence of discordance cannot be estimated -> conservative).
  mover_lincomb(parts)                    MOVER (Zou & Donner 2008) for a linear combination of estimates with their
                                          CIs and correlations (pooled task-equal mean of independent tasks;
                                          interaction = difference of two correlated paired differences).
  signflip_inv_trinomial(b, c, n, level)  the OLD procedure (stats_core.signflip_ci semantics incl. the 1e-12 tie
                                          tolerance and span) for iid units with d in {-1, 0, 1}, computed exactly by
                                          convolution (any n). Equality with stats_core.signflip_ci is verified.
  signflip_tost_trinomial(b, c, n, m)     the OLD stats_core.tost(method="signflip") p_lower / p_upper, exact.
"""
from __future__ import annotations

import math
import sys
from typing import Callable, Sequence

import numpy as np
from scipy.stats import norm

sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/review_followup2")
_TOL = 1e-12


# ============================================================================================== basic paired intervals
def wilson(x, n, alpha=0.05):
    """Wilson score interval; x, n may be non-integer (effective counts)."""
    if n <= 0:
        return (0.0, 1.0)
    z = norm.isf(alpha / 2)
    p = x / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / den
    return (max(0.0, cen - half), min(1.0, cen + half))


def newcombe10(a, b, c, d, alpha=0.05):
    """Newcombe (1998) method 10 for p1 - p2, p1 = (a + b)/n, p2 = (a + c)/n (a = both success, b = first only,
    c = second only, d = neither). Returns (lo, hi)."""
    a, b, c, d = map(float, (a, b, c, d))
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, alpha)
    l2, u2 = wilson(a + c, n, alpha)
    mp = (a + b) * (c + d) * (a + c) * (b + d)
    if mp <= 0:
        phi = 0.0
    else:
        cov = a * d - b * c
        if cov > 0:
            cov = max(cov - n / 2, 0.0)
        phi = cov / math.sqrt(mp)
    dl, du = p1 - l1, u2 - p2
    lo = p1 - p2 - math.sqrt(max(dl * dl - 2 * phi * dl * du + du * du, 0.0))
    dl, du = u1 - p1, p2 - l2
    hi = p1 - p2 + math.sqrt(max(dl * dl - 2 * phi * dl * du + du * du, 0.0))
    return (max(-1.0, lo), min(1.0, hi))


def wald_paired(b, c, n, alpha=0.05):
    z = norm.isf(alpha / 2)
    th = (b - c) / n
    h = z * math.sqrt(max(b + c - (b - c) ** 2 / n, 0.0)) / n
    return (th - h, th + h)


# ============================================================================================== Tango score
def _tango_q01(b, c, n, d0):
    """restricted MLE of p01 (= P(first 0, second 1); 'c' cell) under p10 - p01 = d0 (Tango 1998)."""
    A = 2.0 * n
    B = -b - c + (2.0 * n - b + c) * d0
    C = -c * d0 * (1.0 - d0)
    disc = max(B * B - 4 * A * C, 0.0)
    return (math.sqrt(disc) - B) / (2 * A)


def tango_z(b, c, n, d0):
    q = _tango_q01(b, c, n, d0)
    var = n * (2 * q + d0 * (1 - d0))
    num = b - c - n * d0
    if var <= 0:
        return 0.0 if abs(num) < 1e-12 else math.copysign(np.inf, num)
    return num / math.sqrt(var)


def tango_test(b, c, n, d0, alt="two-sided"):
    z = tango_z(b, c, n, d0)
    if alt == "two-sided":
        return float(2 * norm.sf(abs(z)))
    if alt == "greater":          # H1: delta > d0
        return float(norm.sf(z))
    return float(norm.cdf(z))     # H1: delta < d0


def tango_ci(b, c, n, alpha=0.05, iters=80):
    zc = norm.isf(alpha / 2)
    th = (b - c) / n
    out = []
    for lo_end in (True, False):
        # lower limit: largest d0 < th with z(d0) >= zc ... solve z(d0) = +zc (lower) / -zc (upper)
        target = zc if lo_end else -zc
        a_, b_ = (-1.0 + 1e-15, th) if lo_end else (th, 1.0 - 1e-15)
        fa, fb = tango_z(b, c, n, a_) - target, tango_z(b, c, n, b_) - target
        if lo_end and fa <= 0:
            out.append(-1.0); continue
        if (not lo_end) and fb >= 0:
            out.append(1.0); continue
        for _ in range(iters):
            mid = 0.5 * (a_ + b_)
            fm = tango_z(b, c, n, mid) - target
            if (fm > 0) == (fa > 0):
                a_, fa = mid, fm
            else:
                b_ = mid
        out.append(0.5 * (a_ + b_))
    return (out[0], out[1])


# ============================================================================================== CI-inversion p
def ci_pvalue(cifun: Callable[[float], Sequence[float]], margin: float, side: str, iters: int = 60) -> float:
    """one-sided p for H0 theta <= -margin (side='lower') or theta >= +margin (side='upper') from a family of
    two-sided intervals cifun(alpha) (level 1 - alpha). The one-sided test at level a rejects iff the (1 - 2a)
    two-sided limit is beyond the margin; p = inf{a : rejects}. Bisection on a in (1e-12, 0.5)."""
    def rej(a):
        lo, hi = cifun(2 * a)
        return lo > -margin if side == "lower" else hi < margin
    lo_a, hi_a = 1e-12, 0.5
    if rej(lo_a):
        return lo_a
    if not rej(hi_a - 1e-12):
        return 1.0
    for _ in range(iters):
        mid = 0.5 * (lo_a + hi_a)
        if rej(mid):
            hi_a = mid
        else:
            lo_a = mid
    return hi_a


def ci_pvalue_two_sided_zero(cifun, iters=60):
    """two-sided p for H0 theta = 0 by CI inversion: smallest alpha whose (1 - alpha) CI excludes 0."""
    def excl(a):
        lo, hi = cifun(a)
        return lo > 0 or hi < 0
    lo_a, hi_a = 1e-12, 1 - 1e-12
    if not excl(hi_a):
        return 1.0
    if excl(lo_a):
        return lo_a
    for _ in range(iters):
        mid = 0.5 * (lo_a + hi_a)
        if excl(mid):
            hi_a = mid
        else:
            lo_a = mid
    return hi_a


# ============================================================================================== clustered (Rao-Scott)
def cluster_deff(d: np.ndarray, cl: np.ndarray):
    """design effect of the mean of d over units for clusters cl. Returns (deff_used, deff_raw, rule)."""
    d = np.asarray(d, float)
    n = len(d)
    _, inv, m = np.unique(cl, return_inverse=True, return_counts=True)
    G = len(m)
    cap = float((m ** 2).sum() / n)            # deff at ICC = 1 (every unit of a cluster identical)
    if G == n:
        return 1.0, 1.0, "singleton clusters (deff = 1)"
    dbar = d.mean()
    r = d - dbar
    v_iid = float((r ** 2).sum()) / n ** 2
    tg = np.bincount(inv, weights=r)
    v_cl = G / (G - 1) * float((tg ** 2).sum()) / n ** 2 if G > 1 else np.inf
    if v_iid <= 0:
        return cap, float("nan"), f"no variation in d: ICC=1 bound sum m^2/n = {cap:.3g}"
    raw = v_cl / v_iid
    return float(min(max(raw, 1.0), cap)), float(raw), "clip(V_cluster/V_iid, 1, sum m^2/n)"


def table_counts(ya, yb):
    ya, yb = np.asarray(ya, int), np.asarray(yb, int)
    return (int(((ya == 1) & (yb == 1)).sum()), int(((ya == 1) & (yb == 0)).sum()),
            int(((ya == 0) & (yb == 1)).sum()), int(((ya == 0) & (yb == 0)).sum()))


def rs_newcombe(ya, yb, cl, alpha=0.05, deff=None):
    """Newcombe 10 on the design-effect-deflated table (unit-weighted estimand within one stratum)."""
    a, b, c, d = table_counts(ya, yb)
    if deff is None:
        deff = cluster_deff(np.asarray(ya, float) - np.asarray(yb, float), cl)[0]
    return newcombe10(a / deff, b / deff, c / deff, d / deff, alpha)


def cluster_corr(x1, x2, cl):
    """cluster-robust correlation of the means of x1 and x2 (same units)."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    _, inv = np.unique(cl, return_inverse=True)
    t1 = np.bincount(inv, weights=x1 - x1.mean())
    t2 = np.bincount(inv, weights=x2 - x2.mean())
    v1, v2 = (t1 ** 2).sum(), (t2 ** 2).sum()
    if v1 <= 0 or v2 <= 0:
        return 0.0
    return float((t1 * t2).sum() / math.sqrt(v1 * v2))


def mover_lincomb(parts, corr=None):
    """parts: list of (w, est, lo, hi); corr: matrix of correlations of the estimates (default independent).
    MOVER (Zou & Donner 2008): L = sum w e - sqrt(sum_i sum_j r_ij l_i l_j), l_i = w_i (e_i - lo_i) if w_i > 0 else
    w_i (e_i - hi_i) ... implemented on the signed distances."""
    k = len(parts)
    R = np.eye(k) if corr is None else np.asarray(corr, float)
    sg = np.sign([w for w, _, _, _ in parts])
    R = R * np.outer(sg, sg)            # cross terms carry sign(w_i w_j): difference -> -2 r l_i l_j
    est = sum(w * e for w, e, _, _ in parts)
    dl = np.array([w * (e - lo) if w > 0 else w * (e - hi) for w, e, lo, hi in parts])     # >= 0 distances to L
    du = np.array([w * (hi - e) if w > 0 else w * (lo - e) for w, e, lo, hi in parts])     # >= 0 distances to U
    L = est - math.sqrt(max(float(dl @ R @ dl), 0.0))
    U = est + math.sqrt(max(float(du @ R @ du), 0.0))
    return est, L, U


# ============================================================================================== OLD sign-flip (exact)
def _signed_sum_dist(vals_counts):
    """distribution (values, probs) of sum_i v_i * (2K_i - n_i), K_i ~ Bin(n_i, 1/2), for groups (v_i, n_i)."""
    from math import comb
    vals = np.zeros(1)
    prob = np.ones(1)
    for v, k in vals_counts:
        if k == 0:
            continue
        x = v * (2 * np.arange(k + 1) - k)
        p = np.array([comb(k, j) for j in range(k + 1)], float) / 2.0 ** k
        vals = (vals[:, None] + x[None, :]).ravel()
        prob = (prob[:, None] * p[None, :]).ravel()
    return vals, prob


def signflip_p_trinomial(b, c, n, delta, alternative="two-sided"):
    """exact stats_core.signflip_test(T) for T = d - delta, d has b (+1), c (-1), n-b-c (0) entries (G = n)."""
    z = n - b - c
    v1, v2, v0 = 1 - delta, -1 - delta, -delta
    T_abs_sum = b * abs(v1) + c * abs(v2) + z * abs(v0)
    obs = b * v1 + c * v2 + z * v0
    tol = _TOL * max(1.0, T_abs_sum)
    vals, prob = _signed_sum_dist([(v1, b), (v2, c)])
    o = np.argsort(vals)
    vals, cprob = vals[o], np.cumsum(prob[o])
    tot = cprob[-1]
    from math import comb
    k0 = np.arange(z + 1)
    x0 = v0 * (2 * k0 - z)
    p0 = np.array([comb(z, j) for j in range(z + 1)], float) / 2.0 ** z

    def P_ge(t):         # P(A >= t) for array t
        idx = np.searchsorted(vals, t, side="left")
        return tot - np.where(idx > 0, cprob[np.maximum(idx - 1, 0)], 0.0)

    def P_le(t):
        idx = np.searchsorted(vals, t, side="right")
        return np.where(idx > 0, cprob[np.maximum(idx - 1, 0)], 0.0)
    if alternative == "two-sided":
        thr = abs(obs) - tol
        if thr <= 0:
            return 1.0
        p = float((p0 * (P_ge(thr - x0) + P_le(-thr - x0))).sum())
    elif alternative == "greater":
        p = float((p0 * P_ge(obs - tol - x0)).sum())
    else:
        p = float((p0 * P_le(obs + tol - x0)).sum())
    return min(1.0, p)


def signflip_inv_trinomial(b, c, n, level=0.95, iters=50):
    """stats_core.signflip_ci(d, ones, mean(d), level) for d with b (+1), c (-1), rest 0 -- exact for any n
    (stats_core itself returns nan for n > 26; review_followup2/3 then used a Monte Carlo inversion)."""
    a = 1 - level
    th = (b - c) / n
    if n < 2:
        return (float("nan"), float("nan"))
    if 2.0 ** (1 - n) > a:
        return (float("-inf"), float("inf"))
    span = max(1.0, 4.0 if (b + c) > 0 else 0.0, 4 * abs(th))

    def acc(dl):
        return signflip_p_trinomial(b, c, n, dl) > a + 1e-15
    out = []
    for direction in (-1, 1):
        lo_, hi_ = th, th + direction * span
        if acc(hi_):
            out.append(hi_); continue
        for _ in range(iters):
            mid = 0.5 * (lo_ + hi_)
            if acc(mid):
                lo_ = mid
            else:
                hi_ = mid
        out.append(lo_)
    return (min(out), max(out))


def signflip_tost_trinomial(b, c, n, m):
    """stats_core.tost(method='signflip') p_lower / p_upper / p_equiv, exact."""
    pl = signflip_p_trinomial(b, c, n, -m, "greater")
    pu = signflip_p_trinomial(b, c, n, m, "less")
    return pl, pu, max(pl, pu)


# ============================================================================================== verification
PAIRED_TABLES = [(10, 5, 2, 20), (3, 12, 7, 3), (10, 0, 0, 20), (0, 5, 3, 0), (0, 0, 4, 6), (5, 3, 0, 2), (1, 1, 1, 1),
                 (100, 20, 30, 850), (2, 0, 0, 0), (0, 3, 0, 0)]
# R 4.5.1, ratesci 1.1.0 moverpairci(type="wilson", corc=TRUE) -- reference values copied from the statsmodels main
# test suite (statsmodels/stats/tests/test_proportion.py, _PAIRED_NEWCOMB / _PAIRED_WALD); statsmodels 0.14.6 (the
# latest release installable for python 3.9) does NOT contain confint_proportions_paired, so it is not used here.
REF_NEWCOMBE = {0.05: [(-0.0643926051, 0.2211160179), (-0.1336023844, 0.4874632352), (-0.0722182713, 0.0722182713),
                       (-0.3885152108, 0.7263114284), (-0.6873262303, -0.0383858209), (-0.0529530013, 0.5665478379),
                       (-0.4949196078, 0.4949196078), (-0.0243592378, 0.0041365528), (-0.6576197725, 0.6576197725),
                       (0.2059232825, 1.0)],
                0.01: [(-0.1066889470, 0.2599315957), (-0.2266187694, 0.5539649645), (-0.0983942957, 0.0983942957),
                       (-0.5250595443, 0.7983788652), (-0.7518402591, 0.0828109031), (-0.1427568201, 0.6162757033),
                       (-0.5585157773, 0.5585157773), (-0.0291585105, 0.0087824399), (-0.7683817083, 0.7683817083),
                       (0.0261275085, 1.0)]}
REF_WALD = {0.05: [(-0.0566130752, 0.2187752374), (-0.1326169178, 0.5326169178), (0.0, 0.0), (-0.4209478039, 0.9209478039),
                   (-0.7036363149, -0.0963636851), (0.0159742349, 0.5840257651), (-0.6929519122, 0.6929519122),
                   (-0.0238451723, 0.0038451723), (0.0, 0.0), (1.0, 1.0)]}


def verify(verbose=True):
    """returns a list of check dicts; raises AssertionError on failure."""
    import stats_core as SC
    checks = []
    # 1. Newcombe 10 vs R ratesci
    mx = 0.0
    for al, refs in REF_NEWCOMBE.items():
        for t, r in zip(PAIRED_TABLES, refs):
            mine = newcombe10(*t, alpha=al)
            mx = max(mx, abs(mine[0] - r[0]), abs(mine[1] - r[1]))
    assert mx < 1e-9, mx
    checks.append({"check": "newcombe10 vs R ratesci 1.1.0 (20 table x alpha cases)", "max_abs_diff": mx, "pass": True})
    mx = 0.0
    for t, r in zip(PAIRED_TABLES, REF_WALD[0.05]):
        a, b, c, d = t
        mine = wald_paired(b, c, a + b + c + d)
        mine = (mine[0], mine[1])
        mx = max(mx, abs(mine[0] - r[0]), abs(mine[1] - r[1]))
    assert mx < 1e-9, mx
    checks.append({"check": "wald (Newcombe method 1) vs R PropCIs 0.3.0 (10 tables)", "max_abs_diff": mx, "pass": True})
    # 2. Tango restricted MLE vs brute force; score z = 0 at the MLE; CI contains the estimate
    from scipy.optimize import minimize_scalar
    rng = np.random.default_rng(1)
    mxq = 0.0
    for _ in range(300):
        n = int(rng.integers(5, 150))
        cnt = rng.multinomial(n, rng.dirichlet(np.ones(4)))
        a, b, c, d = map(int, cnt)
        d0 = float(rng.uniform(-0.9, 0.9))
        # maximise loglik over q = p01 with p10 = q + d0, p11 + p00 = 1 - 2q - d0 (concordant total; split irrelevant)

        def nll(q):
            p10, p01, pc = q + d0, q, 1 - 2 * q - d0
            if min(p10, p01, pc) <= 0:
                return 1e300
            return -(b * math.log(p10) + c * math.log(p01) + (a + d) * math.log(pc))
        lo_q, hi_q = max(0.0, -d0) + 1e-12, (1 - d0) / 2 - 1e-12
        if hi_q <= lo_q:
            continue
        res = minimize_scalar(nll, bounds=(lo_q, hi_q), method="bounded", options={"xatol": 1e-12})
        qb = res.x
        qt = _tango_q01(b, c, n, d0)
        if b > 0 and c > 0 and a + d > 0:
            mxq = max(mxq, abs(qb - qt))
    assert mxq < 1e-5, mxq
    checks.append({"check": "Tango restricted MLE of p01 vs brute-force maximisation (300 random tables, b,c,a+d>0)",
                   "max_abs_diff": mxq, "pass": True})
    # 3. OLD procedure: trinomial exact == stats_core.signflip_ci / signflip_test / tost for n <= 16
    mx = 0.0
    for _ in range(120):
        n = int(rng.integers(2, 17))
        d = rng.choice([-1, 0, 0, 0, 1], size=n).astype(float)
        b, c = int((d == 1).sum()), int((d == -1).sum())
        ref = SC.signflip_ci(d, np.ones(n), float(d.mean()), 0.95)
        mine = signflip_inv_trinomial(b, c, n, 0.95)
        if np.all(np.isfinite(ref)):
            mx = max(mx, abs(ref[0] - mine[0]), abs(ref[1] - mine[1]))
        else:
            assert not np.all(np.isfinite(mine))
        for m in (0.1, 0.3):
            t = SC.tost(d, np.ones(n), m, "signflip", B=20, seed=1)
            pl, pu, pe = signflip_tost_trinomial(b, c, n, m)
            mx = max(mx, abs(t["p_lower"] - pl), abs(t["p_upper"] - pu))
    assert mx < 1e-9, mx
    checks.append({"check": "exact trinomial sign-flip inversion/TOST == stats_core.signflip_ci/tost (120 random d, n<=16)",
                   "max_abs_diff": mx, "pass": True})
    # 4. reproduction of the reported degenerate OpenVLA per-task CI (all 10 differences 0)
    ref = SC.signflip_ci(np.zeros(10), np.ones(10), 0.0, 0.95)
    checks.append({"check": "stats_core.signflip_ci(all-zero d, n=10) reproduces +-2.4958e-13", "value": list(ref),
                   "pass": bool(abs(ref[1] - 2.495781359357352e-13) < 1e-20 and abs(ref[0] + 2.495781359357352e-13) < 1e-20)})
    assert checks[-1]["pass"]
    # 5. Newcombe never zero width; Tango never zero width (n>=1)
    for n in (1, 5, 10, 100):
        for t in ((n, 0, 0, 0), (0, 0, 0, n)):
            lo, hi = newcombe10(*t)
            assert hi - lo > 0
            lo, hi = tango_ci(0, 0, n)
            assert hi - lo > 0
    checks.append({"check": "Newcombe10 / Tango positive width at b = c = 0 (n = 1, 5, 10, 100)", "pass": True})
    if verbose:
        for c_ in checks:
            print(c_)
    return checks


if __name__ == "__main__":
    verify()
