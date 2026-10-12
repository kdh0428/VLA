#!/usr/bin/env python
"""
Checks for stats_core (prompt VLA_next_experiments_prompt_20261013.md, section F "통계 코드의 핵심 검증"):
  1. identical paired outcomes -> effect 0 (CI [0, 0], p = 1)
  2. swapping the conditions -> exact reversal of the sign, the CIs and the TOST sides
  3. duplicating rows inside clusters -> no extra clusters, CI does not shrink (naive row bootstrap shown as contrast)
  4. small simulated clustered binary data with boundary effects -> coverage / type-I sanity of the cluster bootstrap,
     the sign-flip test and the TOST
  5. direct effect present, interaction exactly zero -> the two judgments are not confused
plus correctness checks of the exact sign-flip (brute force), MC fallback, Holm.

Run:  nice -n 19 python -m pytest -q test_stats_core.py   (or: nice -n 19 python test_stats_core.py)
The simulation numbers are printed (-s) and written to the run directory by run_tests.sh.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats_core as SC  # noqa: E402

RNG_SEED = 13


def make_clustered(G=13, seed=0, p_base=0.3, delta=0.0, scenes=(1, 8), units=7, binary=True, re_sd=0.15):
    """paired long data: cluster -> log -> scene -> unit, outcomes a (treated) and b (control)."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(G):
        pg = np.clip(p_base + rng.normal(0, re_sd), 0.0, 1.0)
        for s in range(rng.integers(scenes[0], scenes[1] + 1)):
            ps = np.clip(pg + rng.normal(0, re_sd / 2), 0.0, 1.0)
            for u in range(units):
                if binary:
                    b = float(rng.random() < ps)
                    a = float(rng.random() < np.clip(ps + delta, 0, 1))
                else:
                    b = rng.normal(ps, 1); a = rng.normal(ps + delta, 1)
                rows.append({"drive": f"d{g:02d}", "log": f"d{g:02d}_l{s % 2}", "scene": f"d{g:02d}_s{s:02d}",
                             "unit": u, "a": a, "b": b})
    df = pd.DataFrame(rows)
    df["d"] = df["a"] - df["b"]
    return df


def sc_of(df, d="d"):
    return SC.scene_table(df, "drive", "scene", d=d)


# ------------------------------------------------------------------------------------------------------- 1 identical
def test_identical_pairs_give_zero():
    df = make_clustered(seed=1)
    df["a"] = df["b"]; df["d"] = 0.0
    sc = sc_of(df)
    for w in SC.WEIGHTINGS:
        r = SC.analyze(sc, w, B=2000, seed=1, margin=0.05)
        assert r["theta"] == 0.0
        assert r["boot_ci95"] == [0.0, 0.0] and r["boot_ci90"] == [0.0, 0.0]
        assert r["p_signflip"] == 1.0
        assert r["n_informative"] == 0
        assert abs(r["sf_ci95"][0]) < 1e-9 and abs(r["sf_ci95"][1]) < 1e-9
        # with no difference at all, equivalence within +-5 pp must be declared (p = 2^-G each side)
        assert r["tost"]["p_equiv"] <= 2.0 ** -sc.cluster.nunique() + 1e-15
    # pair_rows on identical long data
    long = pd.concat([df.assign(cond="A", y=df["b"]), df.assign(cond="B", y=df["b"])])
    pr, rep = SC.pair_rows(long, ["drive", "scene", "unit"], "cond", "y", "A", "B")
    assert rep["n_unpaired_or_missing"] == 0 and (pr["d"] == 0).all()


# ------------------------------------------------------------------------------------------------------- 2 swap
def test_condition_swap_reverses_exactly():
    df = make_clustered(seed=2, delta=0.12)
    sw = df.copy(); sw["a"], sw["b"] = df["b"], df["a"]; sw["d"] = sw["a"] - sw["b"]
    sc, scs = sc_of(df), sc_of(sw)
    for w in SC.WEIGHTINGS:
        r = SC.analyze(sc, w, B=4000, seed=5, margin=0.05)
        s = SC.analyze(scs, w, B=4000, seed=5, margin=0.05)
        assert s["theta"] == -r["theta"]
        for k in ("boot_ci95", "boot_ci90", "sf_ci95", "sf_ci90", "clt_ci95"):
            assert np.allclose(s[k], [-r[k][1], -r[k][0]], atol=1e-9), (w, k, r[k], s[k])
        assert s["p_signflip"] == r["p_signflip"]
        assert np.isclose(s["p_cluster_t"], r["p_cluster_t"])
        assert np.isclose(s["tost"]["p_lower"], r["tost"]["p_upper"]) and np.isclose(s["tost"]["p_upper"], r["tost"]["p_lower"])
        assert np.isclose(s["tost"]["p_equiv"], r["tost"]["p_equiv"])
        lr, ls = SC.leave_one_cluster_out(sc, w, B=500, seed=3), SC.leave_one_cluster_out(scs, w, B=500, seed=3)
        assert np.allclose(ls["theta"], -lr["theta"]) and np.allclose(ls["p_signflip"], lr["p_signflip"])
    # bootstrap TOST variant too
    tr = SC.tost(*SC.cluster_arrays(sc)[1:], 0.05, "bootstrap", 4000, 9)
    ts = SC.tost(*SC.cluster_arrays(scs)[1:], 0.05, "bootstrap", 4000, 9)
    assert np.isclose(tr["p_lower"], ts["p_upper"]) and np.isclose(tr["p_upper"], ts["p_lower"])


# ------------------------------------------------------------------------------------------------------- 3 duplication
def naive_row_bootstrap_ci(d, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    return SC.pct_ci(d[idx].mean(1))


def test_duplication_does_not_add_clusters_or_shrink_ci():
    df = make_clustered(seed=3, delta=0.1)
    sc = sc_of(df)
    # (i) every row of every cluster duplicated k = 3 times (uniform replication)
    dup = pd.concat([df] * 3, ignore_index=True)
    scd = sc_of(dup)
    for w in SC.WEIGHTINGS:
        r = SC.analyze(sc, w, B=4000, seed=11)
        s = SC.analyze(scd, w, B=4000, seed=11)
        assert s["n_clusters"] == r["n_clusters"]
        assert np.isclose(s["theta"], r["theta"])
        assert np.allclose(s["boot_ci95"], r["boot_ci95"]) and np.allclose(s["sf_ci95"], r["sf_ci95"])
        assert np.isclose(s["p_signflip"], r["p_signflip"])
        width_r = r["boot_ci95"][1] - r["boot_ci95"][0]; width_s = s["boot_ci95"][1] - s["boot_ci95"][0]
        assert width_s >= width_r - 1e-12
    # (ii) rows of ONE scene in ONE cluster duplicated 10x: scene / cluster-equal estimands unchanged, cluster count same
    one = df[df.scene == df.scene.iloc[0]]
    dup2 = pd.concat([df] + [one] * 9, ignore_index=True)
    sc2 = sc_of(dup2)
    for w in ("scene", "cluster_equal"):
        r = SC.analyze(sc, w, B=2000, seed=4, ci_inversion=False)
        s = SC.analyze(sc2, w, B=2000, seed=4, ci_inversion=False)
        assert s["n_clusters"] == r["n_clusters"] and np.isclose(s["theta"], r["theta"])
        assert np.allclose(s["boot_ci95"], r["boot_ci95"])
    # (iii) positive control: a naive row bootstrap WOULD shrink by ~sqrt(3) under (i)
    w0 = np.diff(naive_row_bootstrap_ci(df["d"].values))[0]
    w1 = np.diff(naive_row_bootstrap_ci(dup["d"].values))[0]
    assert w1 < 0.7 * w0
    print(f"\n[dup] naive row-bootstrap CI width {w0:.4f} -> {w1:.4f} after x3 duplication; cluster CI unchanged")


# ------------------------------------------------------------------------------------------------------- 4 simulation
def _sim(G, delta, p_base, nsim, B, seed, margin=None, re_sd=0.15):
    """coverage of the 95% cluster bootstrap CI (scene-weighted), sign-flip rejection rate at .05, TOST rejection rate.
    Truth = delta (additive on the probability scale; p_base chosen so clipping is rare except where intended)."""
    cov = rej = eq = 0
    cov_sf = 0
    for i in range(nsim):
        df = make_clustered(G=G, seed=seed + i, p_base=p_base, delta=delta, re_sd=re_sd)
        sc = sc_of(df)
        r = SC.analyze(sc, "scene", B=B, seed=seed + i, ci_inversion=(i % 5 == 0), margin=margin)
        lo, hi = r["boot_ci95"]
        # realised truth: per-scene probabilities can clip at 0/1, so use the expected scene-level difference
        cov += lo <= delta <= hi
        rej += r["p_signflip"] < 0.05
        if i % 5 == 0:
            cov_sf += r["sf_ci95"][0] <= delta <= r["sf_ci95"][1]
        if margin is not None:
            eq += r["tost"]["p_equiv"] < 0.05
    n_sf = len(range(0, nsim, 5))
    return {"G": G, "delta": delta, "p_base": p_base, "nsim": nsim, "coverage_boot95": cov / nsim,
            "coverage_sf95": cov_sf / n_sf, "reject_signflip_05": rej / nsim,
            "tost_equiv_rate_05": (eq / nsim) if margin is not None else None}


SIM_RESULTS = []


def test_simulated_coverage_and_boundary():
    # null with a low base rate (boundary: many zero-difference scenes / clusters, skewed totals)
    for G in (13, 24):
        r0 = _sim(G, 0.0, 0.08, 300, 999, 1000 * G, margin=0.05, re_sd=0.06)
        SIM_RESULTS.append(r0)
        assert r0["reject_signflip_05"] <= 0.09, r0           # type I of the sign-flip near nominal
        assert r0["coverage_boot95"] >= 0.84, r0              # percentile bootstrap under-covers with few clusters
        assert r0["coverage_sf95"] >= 0.85, r0
    # true effect exactly AT the equivalence margin (boundary for TOST): equivalence must rarely be declared
    for G in (13, 24):
        rb = _sim(G, 0.05, 0.08, 300, 999, 5000 + 1000 * G, margin=0.05, re_sd=0.06)
        SIM_RESULTS.append(rb)
        assert rb["tost_equiv_rate_05"] <= 0.10, rb
        assert rb["coverage_boot95"] >= 0.84, rb
    # moderate effect, mid base rate: power sanity
    r1 = _sim(13, 0.25, 0.3, 200, 999, 9000)
    SIM_RESULTS.append(r1)
    assert r1["reject_signflip_05"] >= 0.8 and r1["coverage_boot95"] >= 0.84, r1
    print("\n[sim] " + "\n[sim] ".join(str(x) for x in SIM_RESULTS))


# ------------------------------------------------------------------------------------------------------- 5 direct vs interaction
def test_direct_effect_vs_zero_interaction_not_confused():
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for g in range(20):
        pg = np.clip(0.6 + rng.normal(0, 0.1), 0.2, 0.95)
        for s in range(40):
            for rep in range(10):
                row = {"drive": f"g{g}", "scene": f"g{g}s{s}", "rep": rep}
                for cfg in (1, 2):                       # same direct effect -0.20 in both configs -> interaction 0
                    row[f"N{cfg}"] = float(rng.random() < pg)
                    row[f"R{cfg}"] = float(rng.random() < pg - 0.20)
                rows.append(row)
    df = pd.DataFrame(rows)
    df["dR1"] = df["R1"] - df["N1"]; df["dR2"] = df["R2"] - df["N2"]; df["inter"] = df["dR2"] - df["dR1"]
    m = 0.05
    direct = SC.analyze(SC.scene_table(df, "drive", "scene", d="dR1"), "scene", B=4000, seed=1, margin=m)
    inter = SC.analyze(SC.scene_table(df, "drive", "scene", d="inter"), "scene", B=4000, seed=1, margin=m)
    jd = SC.judge(direct["theta"], direct["boot_ci95"], direct["p_signflip"], 0.05, direct["tost"]["p_equiv"], m)
    ji = SC.judge(inter["theta"], inter["boot_ci95"], inter["p_signflip"], 0.05, inter["tost"]["p_equiv"], m)
    print(f"\n[direct/interaction] direct {direct['theta']:+.3f} {direct['boot_ci95']} p={direct['p_signflip']:.2g} "
          f"-> {jd['label']}; interaction {inter['theta']:+.3f} {inter['boot_ci95']} p={inter['p_signflip']:.2g} "
          f"p_equiv={inter['tost']['p_equiv']:.2g} -> {ji['label']}")
    assert jd["diff_from_0"] and not jd["equivalent"] and jd["label"] == "different_from_0"
    assert not ji["diff_from_0"] and ji["equivalent"] and ji["label"] == "equivalent_within_margin"
    # an effect can be both different from 0 and inside the margin: the judgments are not exclusive
    jb = SC.judge(0.02, [0.01, 0.03], 0.001, 0.05, 0.001, 0.05)
    assert jb["label"] == "different_from_0_and_within_margin"
    ju = SC.judge(0.02, [-0.08, 0.12], 0.6, 0.05, 0.4, 0.05)
    assert ju["label"] == "uncertain"


# ------------------------------------------------------------------------------------------------------- correctness
def test_exact_signflip_matches_brute_force():
    rng = np.random.default_rng(7)
    for G in (1, 2, 5, 8, 11):
        T = rng.normal(0.3, 1, G)
        T[0] = 0.0 if G > 3 else T[0]
        allE = np.array(list(itertools.product([-1, 1], repeat=G)), float)
        S = allE @ T; obs = T.sum()
        for alt, ref in (("two-sided", np.mean(np.abs(S) >= abs(obs) - 1e-12)),
                         ("greater", np.mean(S >= obs - 1e-12)), ("less", np.mean(S <= obs + 1e-12))):
            p = SC.signflip_test(T, alt)["p"]
            assert np.isclose(p, ref), (G, alt, p, ref)


def test_mc_fallback_close_to_exact_and_formula():
    rng = np.random.default_rng(8)
    T = rng.normal(0.4, 1, 16)
    pe = SC.signflip_test(T)["p"]
    pm = SC.signflip_test(T, exact_max=10, reps=200_000, seed=3)
    assert "Monte Carlo" in pm["method"] and pm["reps"] == 200_000 and pm["seed"] == 3
    assert abs(pm["p"] - pe) < 0.01
    # (b+1)/(R+1): an extreme statistic gives exactly 1/(R+1) when no flip reaches it
    T2 = np.full(40, 1.0)
    pm2 = SC.signflip_test(T2, exact_max=10, reps=1000, seed=1)
    assert np.isclose(pm2["p"], 1 / 1001)


def test_min_attainable_p_and_holm():
    r = SC.signflip_test(np.r_[np.ones(10), np.zeros(3)])
    assert np.isclose(r["p"], 2 / 2 ** 10) and np.isclose(r["p_min_attainable"], 2 / 2 ** 10) and r["n_informative"] == 10
    assert np.allclose(SC.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(SC.holm([0.5, np.nan, 0.01]), [0.5, np.nan, 0.02], equal_nan=True)


def test_weightings_are_different_estimands():
    # cluster 1: 1 scene with 1 unit d=1 ; cluster 2: 3 scenes x 3 units d=0, plus one scene with 1 unit d=1
    rows = [{"drive": "c1", "scene": "s1", "d": 1.0}]
    rows += [{"drive": "c2", "scene": f"t{s}", "d": 0.0} for s in range(3) for _ in range(3)]
    rows += [{"drive": "c2", "scene": "t9", "d": 1.0}]
    sc = SC.scene_table(pd.DataFrame(rows), "drive", "scene")
    th = {w: SC.ratio(*SC.cluster_arrays(sc, w)[1:]) for w in SC.WEIGHTINGS}
    assert np.isclose(th["unit"], 2 / 11) and np.isclose(th["scene"], 2 / 5) and np.isclose(th["cluster_equal"], (1 + 0.25) / 2)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
