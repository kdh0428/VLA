#!/usr/bin/env python
"""
P0-B (review_followup3, 2026-10-14): SpatialVLA per-task direct-effect tables (exp 35 and P1-D separately).

*** POST-HOC RE-ANALYSIS OF EXISTING DATA. CPU only, no model run. Existing files are read, never modified. ***

What it does
  1. Re-computes the CORE direct-effect rows of review_followup2/B (b_direct_effects.py, same functions, same seeds)
     for experiment in {exp35 (seeds 0-119), P1D (units 0-119)} x config {r1e4, r4e1} x contrast {R-N, C-F} x
     task level {coke, move_near, pooled} x clustering {primary, physical}, and ASSERTS that estimate, 95/90% CI,
     raw/Holm p and TOST p match DIRECT_EFFECTS.csv (Holm family = original 'exp35 core direct (seeds 0-119)' /
     'P1D core direct', m = 4 per task level x clustering, exactly as b_direct_effects.post defined it; not split).
  2. Adds what the review asked for: per-condition success counts / attempts, physical-state count, instruction-
     inclusive (= primary) cluster count, scene (instruction-free) cluster count, informative clusters.
  3. Equal-task mean (pooled) rows in a separate table.
  4. Task x intervention heterogeneity (coke effect - move_near effect) with a task-stratified cluster bootstrap CI
     (clusters independent across tasks; B = 10,000) and a Welch cluster-robust t; cancellation share
     1 - |pooled| / mean(|task effects|) (0 when the task effects have the same sign).
  5. Weighting sensitivity: ORIGINAL estimand (unit-weighted within task; task-equal pooled) vs physical-state
     equal-weight (mean within physical layout first, layouts equally weighted) and primary-state equal-weight.
     Inference for the sensitivity rows: sign-flip test / inversion and stratified bootstrap on the same clusters.
  6. Verification: unit pairing (every unit has N, F, C, R), cluster counts vs A2 state_manifest, arm-sign swap
     (R-N = -(N-R) and C-F = -(F-C) for estimate and CI), condition rates vs original exp35 RESULTS / P1-D
     condition_summary.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python p0_b_spatial_task_effects.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
RF2 = os.path.join(POC, "scripts/review_followup2")
sys.path.insert(0, RF2)
import stats_core as SC  # noqa: E402
import b_direct_effects as BDE  # noqa: E402  (functions only; its main() is never called)

BDIR = os.path.join(POC, "outputs/review_followup2/B_spatialvla_direct/run_20261013")
SMAN = os.path.join(POC, "outputs/review_followup2/A_spatialvla_states/run_20261013/state_manifest.csv")
OUT = os.path.join(POC, "outputs/review_followup3/P0")
SEED = 20261014
B = 10_000
EXPS = ("exp35", "P1D")
CFGS = ("r1e4", "r4e1")
CONTR = ("R-N", "C-F")
TASKS = ("coke", "move_near")
CLS = ("primary", "physical")
FAM = {"exp35": "exp35 core direct (seeds 0-119)", "P1D": "P1D core direct"}
MODES = {"N": "natural", "F": "feedback", "C": "corrected", "R": "reverse"}
REL = lambda p: os.path.relpath(p, POC)  # noqa: E731


def counts(epi, exp, cfg, task):
    out = {}
    for k, mode in MODES.items():
        q = epi[(epi.experiment == exp) & (epi.config == cfg) & (epi["mode"] == mode) & (epi.direction == "opposite")]
        if task != "pooled":
            q = q[q.task == task]
        out[f"succ_{k}"] = int(q.success.sum())
        out[f"att_{k}"] = int(len(q))
    return out


def wilson_free_rate(s, n):
    return 100.0 * s / n if n else np.nan


def welch(N1, D1, N2, D2):
    """difference of two independent ratio estimators, cluster-robust SE each, Welch df."""
    def one(N, D):
        th = N.sum() / D.sum()
        G = len(N)
        v = G / (G - 1) * np.sum((N - th * D) ** 2) / D.sum() ** 2
        return th, v, G
    a, va, ga = one(N1, D1)
    b, vb, gb = one(N2, D2)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (ga - 1) + vb ** 2 / (gb - 1)) if se > 0 else np.nan
    from scipy import stats
    t = (a - b) / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), df) if se > 0 else np.nan
    q = stats.t.ppf(0.975, df) if se > 0 else np.nan
    return a - b, (a - b - q * se, a - b + q * se), p, df


def boot_task(N, D, B=B, seed=SEED):
    rng = np.random.default_rng(seed)
    C = rng.multinomial(len(N), np.ones(len(N)) / len(N), size=B).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (C @ N) / (C @ D)


def infer_ND(N, D, tk):
    """same inference as b_direct_effects.infer but for arbitrary cluster totals (weighting sensitivity)."""
    G = len(N)
    th = SC.ratio(N, D)
    r = {"estimate": th, "n_clusters": G}
    bs = BDE.strat_boot(tk, N, D)
    r["boot_ci95"] = SC.pct_ci(bs, .95)
    sf = SC.signflip_test(N - 0 * D, "two-sided", SC.EXACT_MAX, BDE.MC, BDE.SEED + 7)
    r["p"] = sf["p"]
    r["n_inf"] = sf["n_informative"]
    if G <= SC.EXACT_MAX:
        r["ci95"] = SC.signflip_ci(N, D, th, .95)
    else:
        r["ci95"] = BDE.mc_inv_ci(N, D, th, .95, BDE.sign_matrix(G))
    for m in (0.10, 0.05):
        t = SC.tost(N, D, m, "signflip", B=200, seed=BDE.SEED, exact_max=SC.EXACT_MAX, reps=BDE.MC)
        r[f"p_equiv_m{int(m * 100)}"] = t["p_equiv"]
    return r


def equal_weight_arrays(x, level_col, ccol):
    """mean d within `level_col` group first, groups equally weighted within task; task-equal pooled.
    Cluster = ccol (must nest level_col groups). Returns labels, task, N, D."""
    keys = ["task", level_col] + ([ccol] if ccol != level_col else [])
    assert (x.groupby(level_col)[ccol].nunique() == 1).all(), "state level must nest in the inference cluster"
    g = x.groupby(keys, sort=True)["d"].mean().reset_index()
    ng = g.groupby("task")[level_col].nunique()
    g["w"] = 1.0 / g["task"].map(ng)
    g["N"] = g["d"] * g["w"]
    c = g.groupby(ccol, sort=True).agg(task=("task", "first"), N=("N", "sum"), D=("w", "sum"))
    return c.index.tolist(), c["task"].values, c["N"].values.astype(float), c["D"].values.astype(float)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    pairs = pd.read_csv(os.path.join(BDIR, "pairs.csv"))
    epi = pd.read_csv(os.path.join(BDIR, "episodes.csv"))
    de = pd.read_csv(os.path.join(BDIR, "DIRECT_EFFECTS.csv"))
    sm = pd.read_csv(SMAN)
    sm["task_s"] = sm.task.map({"google_robot_pick_coke_can": "coke", "google_robot_move_near": "move_near"})
    checks = {}

    # ---------------- verification: pairing (every unit has N, F, C, R in each core config)
    pair_ok = {}
    for exp in EXPS:
        for cfg in CFGS:
            q = epi[(epi.experiment == exp) & (epi.config == cfg) & (epi.direction == "opposite") &
                    epi["mode"].isin(MODES.values())]
            per = q.groupby(["task", "seed"])["mode"].nunique()
            dup = q.groupby(["task", "seed", "mode"]).size().max()
            pair_ok[f"{exp}/{cfg}"] = {"units": int(len(per)), "units_with_all_4_conditions": int((per == 4).sum()),
                                       "max_rows_per_unit_condition": int(dup)}
            assert (per == 4).all() and dup == 1 and len(per) == 240, (exp, cfg)
    checks["pairing"] = pair_ok

    # ---------------- recompute core rows and assert reproduction
    rows, pooled_rows = [], []
    xcache = {}
    for exp in EXPS:
        for cfg in CFGS:
            for cn in CONTR:
                x = BDE.unit_d(pairs, exp, cfg, cn, "opposite")
                assert len(x) == 240 and not x.duplicated(["task", "seed"]).any()
                # arm-sign swap test: reversed contrast from raw success columns
                raw = pairs[(pairs.experiment == exp) & (pairs.config == cfg) & (pairs.contrast == cn) &
                            (pairs.direction == "opposite")]
                assert np.allclose(raw.success_a - raw.success_b, raw.d_success)
                xcache[(exp, cfg, cn)] = x
                for lvl in TASKS + ("pooled",):
                    xx = x if lvl == "pooled" else x[x.task == lvl]
                    for cl in CLS:
                        r = BDE.infer(xx, f"cluster_{cl}")
                        sw = xx.copy()
                        sw["d"] = -sw["d"]
                        rs = SC.ratio(*BDE.cluster_arrays(sw, f"cluster_{cl}")[2:])
                        assert np.isclose(rs, -r["estimate"]), "arm-sign swap failed"
                        ref = de[(de.experiment == exp) & (de.config == cfg) & (de.contrast == cn) &
                                 (de.task == lvl) & (de.clustering == cl)]
                        assert len(ref) == 1
                        ref = ref.iloc[0]
                        mine = {"estimate_pp": 100 * r["estimate"], "ci95_lo_pp": 100 * r["sf_ci95"][0],
                                "ci95_hi_pp": 100 * r["sf_ci95"][1], "ci90_lo_pp": 100 * r["sf_ci90"][0],
                                "ci90_hi_pp": 100 * r["sf_ci90"][1], "boot_ci95_lo_pp": 100 * r["boot_ci95"][0],
                                "boot_ci95_hi_pp": 100 * r["boot_ci95"][1], "p_diff_signflip": r["p_diff"],
                                "p_equiv_m10": r["tost10"]["p_equiv"], "p_equiv_m5": r["tost5"]["p_equiv"],
                                "p_lower_m10": r["tost10"]["p_lower"], "p_upper_m10": r["tost10"]["p_upper"],
                                "p_lower_m5": r["tost5"]["p_lower"], "p_upper_m5": r["tost5"]["p_upper"]}
                        for k, v in mine.items():
                            if not np.isclose(v, ref[k], rtol=1e-5, atol=1e-6):
                                raise AssertionError(f"reproduction mismatch {exp} {cfg} {cn} {lvl} {cl} {k}: {v} vs {ref[k]}")
                        assert int(ref.n_clusters) == r["n_clusters"] and int(ref.n_units) == r["n_units"]
                        # unit / cluster counts
                        units = set(zip(xx.task, xx.seed))
                        smx = sm[(sm.experiment == exp) & sm.task_s.isin([lvl] if lvl != "pooled" else TASKS)]
                        smx = smx[[(t, s) in units for t, s in zip(smx.task_s, smx.seed_or_unit)]]
                        assert len(smx) == len(units)
                        n_prim, n_phys = xx.cluster_primary.nunique(), xx.cluster_physical.nunique()
                        assert n_prim == smx.initial_state_cluster_id.nunique()
                        assert n_phys == smx.physical_state_cluster_id.nunique()
                        row = {"row_id": f"P0B.{exp}.{cfg}.{cn}.{lvl}.{cl}", "experiment": exp, "config": cfg,
                               "contrast": cn, "task": lvl, "clustering": cl,
                               "contrast_definition": "S(R)-S(N)" if cn == "R-N" else "S(C)-S(F) (+ = correction benefit; = -(F-C))"}
                        row.update(counts(epi, exp, cfg, lvl))
                        for k in MODES:
                            row[f"rate_{k}_pct"] = wilson_free_rate(row[f"succ_{k}"], row[f"att_{k}"])
                        row.update({"n_units": r["n_units"], "n_primary_clusters_instruction_inclusive": n_prim,
                                    "n_scene_clusters_instruction_free": smx.scene_state_cluster_id.nunique(),
                                    "n_physical_states": n_phys, "n_clusters_used": r["n_clusters"],
                                    "n_informative_clusters": r["n_informative"],
                                    "p_min_attainable": r["p_min_attainable"]})
                        row.update(mine)
                        row.update({"ci_method": r["ci_method"], "p_diff_method": r["p_diff_method"],
                                    "family_id": FAM[exp], "family_scope": "Holm within family x task level x clustering "
                                    "(original b_direct_effects definition)",
                                    "p_diff_holm": ref.p_diff_holm, "family_m": int(ref.family_size),
                                    "p_equiv_m10_holm": ref.p_equiv_m10_holm, "p_equiv_m5_holm": ref.p_equiv_m5_holm,
                                    "margin_m10_status": "original (preregistered) margin",
                                    "margin_m5_status": "post-hoc sensitivity margin",
                                    "difference_judgment": ref.difference_judgment,
                                    "equivalence_judgment_m10": ref.equivalence_judgment_m10,
                                    "equivalence_judgment_m5": ref.equivalence_judgment_m5,
                                    "missing": "none (0 missing episodes; ITT, success at step 80)",
                                    "source_path": REL(os.path.join(BDIR, "DIRECT_EFFECTS.csv")),
                                    "source_row": int(ref.name) + 2, "reproduced": True})
                        (pooled_rows if lvl == "pooled" else rows).append(row)
                print(f"[{time.time() - t0:5.0f}s] {exp} {cfg} {cn}", flush=True)
    # recompute Holm from my own rows (same family definition) and assert equal to the file's Holm
    allr = pd.DataFrame(rows + pooled_rows)
    for (exp, lvl, cl), g in allr.groupby(["experiment", "task", "clustering"]):
        assert len(g) == 4
        h = SC.holm(g.p_diff_signflip.values)
        assert np.allclose(h, g.p_diff_holm.values, atol=1e-9)
        for k in (10, 5):
            assert np.allclose(SC.holm(g[f"p_equiv_m{k}"].values), g[f"p_equiv_m{k}_holm"].values, atol=1e-9)
    checks["holm_recomputed_equal"] = True
    checks["reproduced_rows"] = len(allr)
    RESTORE = {"exp35": "exp 35: NO restore verification (no state hash; object state inferred from seed -> reset code path; "
                        "condition pairs share an initial-state cluster but are not bit-verified; SpatialVLA P1 restore check absent)",
               "P1D": "P1-D: restore verified (init.state_hash 120/120 per task, restore.ok 3,840/3,840, null == N 320/320)"}
    for rr in rows + pooled_rows:
        rr["restore_verification"] = RESTORE[rr["experiment"]]
        rr["status"] = ("POST-HOC re-analysis (SpatialVLA; not subject to AutoVLA parity); " +
                        ("exp 35 restore NOT verified" if rr["experiment"] == "exp35" else "P1-D restore verified"))
        rr["ci95_width_pp"] = rr["ci95_hi_pp"] - rr["ci95_lo_pp"]
    task_df = pd.DataFrame(rows)
    task_df.to_csv(os.path.join(OUT, "SPATIAL_TASK_EFFECTS.csv"), index=False, float_format="%.6g")
    wsum = task_df.groupby(["experiment", "config", "clustering"]).ci95_width_pp.agg(["min", "max"]).reset_index()
    wsum.to_csv(os.path.join(OUT, "SPATIAL_TASK_EFFECTS_ci_width_summary.csv"), index=False, float_format="%.4g")
    pdf = pd.DataFrame(pooled_rows)
    pdf.insert(1, "estimand", "equal-task mean of the two task means (original pooled estimand)")
    pdf.to_csv(os.path.join(OUT, "SPATIAL_TASK_EFFECTS_equal_task_mean.csv"), index=False, float_format="%.6g")

    # ---------------- condition rates vs original outputs
    orig35 = {("r1e4", "N"): 82.9, ("r1e4", "F"): 67.1, ("r1e4", "C"): 60.0, ("r1e4", "R"): 82.1,
              ("r4e1", "N"): 81.7, ("r4e1", "F"): 69.6, ("r4e1", "C"): 65.0, ("r4e1", "R"): 75.4}
    cs = pd.read_csv(os.path.join(POC, "outputs/review_followup/P1D_dose_control/run_20261008/condition_summary.csv"))
    rc = []
    for (cfg, k), v in orig35.items():
        mine = pdf[(pdf.experiment == "exp35") & (pdf.config == cfg)].iloc[0][f"rate_{k}_pct"]
        rc.append({"experiment": "exp35", "config": cfg, "cond": k, "mine": mine, "orig": v, "ok": abs(mine - v) < 0.06})
    for cfg in CFGS:
        for k, mode in MODES.items():
            o = cs[(cs.config == cfg) & (cs.condition == mode) & (cs.task == "all")].success_rate.iloc[0] * 100
            mine = pdf[(pdf.experiment == "P1D") & (pdf.config == cfg)].iloc[0][f"rate_{k}_pct"]
            rc.append({"experiment": "P1D", "config": cfg, "cond": k, "mine": mine, "orig": o, "ok": abs(mine - o) < 0.01})
    checks["condition_rates_vs_original"] = rc
    assert all(r["ok"] for r in rc), rc

    # ---------------- heterogeneity (coke - move_near) and cancellation
    het = []
    for exp in EXPS:
        for cfg in CFGS:
            for cn in CONTR:
                x = xcache[(exp, cfg, cn)]
                for cl in CLS:
                    arrs = {}
                    for t in TASKS:
                        _, _, N, D = BDE.cluster_arrays(x[x.task == t], f"cluster_{cl}")
                        arrs[t] = (N, D)
                    (Nc, Dc), (Nm, Dm) = arrs["coke"], arrs["move_near"]
                    tc, tm = Nc.sum() / Dc.sum(), Nm.sum() / Dm.sum()
                    bc = boot_task(Nc, Dc, seed=SEED + 1)
                    bm = boot_task(Nm, Dm, seed=SEED + 2)
                    bd = bc - bm
                    diff, wci, wp, wdf = welch(Nc, Dc, Nm, Dm)
                    pooled = (tc + tm) / 2
                    mabs = (abs(tc) + abs(tm)) / 2
                    opp = np.sign(tc) * np.sign(tm) < 0
                    canc = 1 - abs(pooled) / mabs if (opp and mabs > 0) else 0.0
                    p_boot_opp = float(np.mean(np.sign(bc) * np.sign(bm) < 0))
                    het.append({"row_id": f"P0B.het.{exp}.{cfg}.{cn}.{cl}", "experiment": exp, "config": cfg,
                                "contrast": cn, "clustering": cl, "n_clusters_coke": len(Nc), "n_clusters_move_near": len(Nm),
                                "coke_pp": 100 * tc, "move_near_pp": 100 * tm,
                                "heterogeneity_coke_minus_move_near_pp": 100 * diff,
                                "het_boot_ci95_lo_pp": 100 * np.percentile(bd, 2.5),
                                "het_boot_ci95_hi_pp": 100 * np.percentile(bd, 97.5),
                                "het_welch_ci95_lo_pp": 100 * wci[0], "het_welch_ci95_hi_pp": 100 * wci[1],
                                "het_welch_p": wp, "het_welch_df": wdf,
                                "het_boot_p_two_sided": float(min(1.0, 2 * min(np.mean(bd <= 0), np.mean(bd >= 0)))),
                                "equal_task_mean_pp": 100 * pooled, "mean_abs_task_effect_pp": 100 * mabs,
                                "opposite_signs": bool(opp), "cancellation_share": canc,
                                "cancelled_magnitude_pp": 100 * (mabs - abs(pooled)),
                                "boot_prob_opposite_signs": p_boot_opp,
                                "note": "post-hoc; clusters independent across tasks (disjoint task-specific states); "
                                        "not in any original Holm family -> unadjusted"})
    hdf = pd.DataFrame(het)
    hdf.to_csv(os.path.join(OUT, "SPATIAL_TASK_EFFECTS_heterogeneity.csv"), index=False, float_format="%.6g")

    # ---------------- weighting sensitivity: original vs physical-equal vs primary-equal
    wrows = []
    for exp in EXPS:
        for cfg in CFGS:
            for cn in CONTR:
                x = xcache[(exp, cfg, cn)]
                for lvl in TASKS + ("pooled",):
                    xx = x if lvl == "pooled" else x[x.task == lvl]
                    orig = de[(de.experiment == exp) & (de.config == cfg) & (de.contrast == cn) & (de.task == lvl) &
                              (de.clustering == "physical")].iloc[0]
                    sizes = xx.groupby("cluster_physical").size()
                    base = {"experiment": exp, "config": cfg, "contrast": cn, "task": lvl, "inference_cluster": "physical",
                            "n_physical_states": int(len(sizes)), "units_per_physical_state_min": int(sizes.min()),
                            "units_per_physical_state_max": int(sizes.max())}
                    wrows.append(dict(base, row_id=f"P0B.w.{exp}.{cfg}.{cn}.{lvl}.original",
                                      weighting="ORIGINAL: unit-weighted within task (task-equal pooled)",
                                      estimate_pp=orig.estimate_pp, ci95_lo_pp=orig.ci95_lo_pp, ci95_hi_pp=orig.ci95_hi_pp,
                                      boot_ci95_lo_pp=orig.boot_ci95_lo_pp, boot_ci95_hi_pp=orig.boot_ci95_hi_pp,
                                      p_signflip=orig.p_diff_signflip, p_equiv_m10=orig.p_equiv_m10,
                                      p_equiv_m5=orig.p_equiv_m5, status="original estimand (from DIRECT_EFFECTS.csv)"))
                    for lvlcol, name in (("cluster_physical", "physical-state equal weight"),
                                         ("cluster_primary", "primary-state (instruction-inclusive) equal weight")):
                        _, tk, N, D = equal_weight_arrays(xx, lvlcol, "cluster_physical")
                        r = infer_ND(N, D, tk)
                        wrows.append(dict(base, row_id=f"P0B.w.{exp}.{cfg}.{cn}.{lvl}.{lvlcol.split('_')[1]}_equal",
                                          weighting=name + " (mean within state first; task-equal pooled)",
                                          estimate_pp=100 * r["estimate"], ci95_lo_pp=100 * r["ci95"][0],
                                          ci95_hi_pp=100 * r["ci95"][1], boot_ci95_lo_pp=100 * r["boot_ci95"][0],
                                          boot_ci95_hi_pp=100 * r["boot_ci95"][1], p_signflip=r["p"],
                                          p_equiv_m10=r["p_equiv_m10"], p_equiv_m5=r["p_equiv_m5"],
                                          status="post-hoc sensitivity (different estimand); not Holm-adjusted"))
    wdf = pd.DataFrame(wrows)
    wdf.to_csv(os.path.join(OUT, "SPATIAL_TASK_EFFECTS_weighting.csv"), index=False, float_format="%.6g")

    # ---------------- reconciliation of P1-D r4e1 R-N pooled +1.2 [-5.6, 8.0]
    pe = pd.read_csv(os.path.join(POC, "outputs/review_followup/P1D_dose_control/run_20261008/paired_effects.csv"))
    o = pe[(pe.name == "r4e1:R-N") & (pe.metric == "success")].iloc[0]
    rf2 = de[(de.experiment == "P1D") & (de.config == "r4e1") & (de.contrast == "R-N") & (de.task == "pooled")]
    recon = {"original_P1D_paired_effects_csv": {"estimate_pp": 100 * o.estimate, "ci95_pp": [100 * o.ci_lo, 100 * o.ci_hi],
                                                 "p": o.p, "test": o.test, "discordant": [int(o.x_only), int(o.y_only)],
                                                 "ci_method": "episode stratified bootstrap percentile (analyze_p1d.py)",
                                                 "source": "outputs/review_followup/P1D_dose_control/run_20261008/paired_effects.csv (name r4e1:R-N, metric success); analysis.md sec 7.1 '+1.3 [-5.0, +7.9], p = 0.81'"},
             "review_followup2_rows": {r.clustering: {"estimate_pp": r.estimate_pp, "ci95_signflip_inv": [r.ci95_lo_pp, r.ci95_hi_pp],
                                                      "boot_ci95": [r.boot_ci95_lo_pp, r.boot_ci95_hi_pp],
                                                      "p_signflip": r.p_diff_signflip, "source_row": int(r.name) + 2}
                                       for _, r in rf2.iterrows()}}
    checks["reconcile_P1D_r4e1_RN"] = recon

    # ---------------- numbers fragment
    nums = {}
    for _, r in pd.concat([task_df, pdf]).iterrows():
        nums[r.row_id] = {"value_pp": r.estimate_pp, "ci95_pp": [r.ci95_lo_pp, r.ci95_hi_pp], "p_raw": r.p_diff_signflip,
                          "p_holm": r.p_diff_holm, "family": r.family_id, "family_m": r.family_m,
                          "p_equiv_m10_holm": r.p_equiv_m10_holm, "p_equiv_m5_holm": r.p_equiv_m5_holm,
                          "n_units": r.n_units, "n_clusters": r.n_clusters_used,
                          "source_path": "outputs/review_followup3/P0/" + ("SPATIAL_TASK_EFFECTS.csv" if r.task != "pooled"
                                                                         else "SPATIAL_TASK_EFFECTS_equal_task_mean.csv"),
                          "upstream_source": f"{r.source_path}#row{r.source_row}"}
    for _, r in hdf.iterrows():
        nums[r.row_id] = {"value_pp": r.heterogeneity_coke_minus_move_near_pp,
                          "ci95_pp": [r.het_boot_ci95_lo_pp, r.het_boot_ci95_hi_pp], "p_raw": r.het_welch_p,
                          "cancellation_share": r.cancellation_share,
                          "source_path": "outputs/review_followup3/P0/SPATIAL_TASK_EFFECTS_heterogeneity.csv"}
    for _, r in wdf.iterrows():
        nums[r.row_id] = {"value_pp": r.estimate_pp, "ci95_pp": [r.ci95_lo_pp, r.ci95_hi_pp], "p_raw": r.p_signflip,
                          "weighting": r.weighting, "source_path": "outputs/review_followup3/P0/SPATIAL_TASK_EFFECTS_weighting.csv"}
    json.dump(nums, open(os.path.join(OUT, "_frag_numbers_B.json"), "w"), indent=1, default=float)
    json.dump(checks, open(os.path.join(OUT, "_checks_B.json"), "w"), indent=1, default=str)

    # ---------------- DATA_MANIFEST fragment (one row per experiment x task x unit/seed)
    man = []
    srcmap = {"exp34": "outputs/cross_domain_temporal_replication/closed_loop/episodes.jsonl",
              "exp35": "outputs/spatialvla_feedback_protection_ablation/rollouts{,_ext}/episodes.jsonl",
              "P1D": "outputs/review_followup/P1D_dose_control/run_20261008/rollouts/main/episodes.jsonl"}
    for (exp, task, seed), g in epi.groupby(["experiment", "task", "seed"]):
        s = sm[(sm.experiment == exp) & (sm.task_s == task) & (sm.seed_or_unit == seed)]
        assert len(s) == 1
        s = s.iloc[0]
        dr = g.direction.fillna("-").astype(str)
        cset = sorted(set(g.config.astype(str) + ":" + g["mode"].astype(str) + ("/" + dr).where(dr != "-", "")))
        man.append({"dataset": "SpatialVLA SimplerEnv (closed loop)", "experiment": exp,
                    "model": "SpatialVLA-4B (spatialvla-4b-224-sft-fractal)", "task": task,
                    "unit_id": f"{exp}:{task}-s{seed}", "log": "", "drive": "", "day": "", "vehicle": "", "day_vehicle": "",
                    "scene": s.scene_state_cluster_id,
                    "perturbation": ("step-1 translation token, d=0.3, directions " +
                                     "/".join(sorted(set(dr) - {"-"}))),
                    "initial_physical_state": s.physical_state_cluster_id,
                    "instruction": s.instruction, "rng_seed": int(seed),
                    "condition_set": ";".join(cset), "n_rows": int(len(g)), "included": True, "missing_reason": "",
                    "denominator_note": (f"primary(instruction-inclusive) cluster {s.initial_state_cluster_id}; "
                                         f"episode_id {'' if pd.isna(s.episode_id) else int(s.episode_id)}; "
                                         f"{s.restore_verification[:80]}; source {srcmap[exp]}")})
    pd.DataFrame(man).to_csv(os.path.join(OUT, "_frag_manifest_B.csv"), index=False)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
