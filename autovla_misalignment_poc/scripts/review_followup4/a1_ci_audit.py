#!/usr/bin/env python
"""
A1 CI validity audit (review_followup4, 2026-10-17): degenerate sign-flip-inversion CIs for paired success rates.

*** POST-HOC CORRECTION / SENSITIVITY. CPU only, no model run. Existing results are read, never modified; the
    original analysis stays the record and is shown side by side. ***

Inputs (read only)
  outputs/review_followup3/P0/OPENVLA_episode_records.csv, OPENVLA_REANALYSIS.csv, OPENVLA_REANALYSIS_per_task.csv
  outputs/review_followup2/B_spatialvla_direct/run_20261013/pairs.csv, DIRECT_EFFECTS.csv
  outputs/review_followup4/A1_A4/A1_METHOD_SELECTION.json   (written by a1_coverage_sim.py BEFORE this script)
Outputs (outputs/review_followup4/A1_A4/)
  A1_OPENVLA_2x2.csv            per-unit paired 2x2 counts for every OpenVLA contrast x task (+ all 100 states)
  CI_VALIDITY_AUDIT.csv         old vs new estimate / CI / p / margin / Holm family / judgments, one row per result row
  A1_JUDGMENT_CHANGES.csv       rows whose difference or equivalence judgment changes
  A1_CALLSITES.csv              every stats_core CI/TOST call site and whether it is affected
  A1_DEGENERACY_DEMO.json       step-by-step reproduction of the zero-width interval
  a1_audit_meta.json

  nice -n 19 /root/miniforge3/envs/autovla/bin/python a1_ci_audit.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/review_followup2")
import a1_paired_ci as A  # noqa: E402
import stats_core as SC  # noqa: E402
import b_direct_effects as BDE  # noqa: E402

POC = "/root/VLA/autovla_misalignment_poc"
P0 = f"{POC}/outputs/review_followup3/P0"
BDIR = f"{POC}/outputs/review_followup2/B_spatialvla_direct/run_20261013"
OUT = f"{POC}/outputs/review_followup4/A1_A4"
LABEL = "POST-HOC CORRECTION (review_followup4 A1); original analysis preserved"
SEL = json.load(open(f"{OUT}/A1_METHOD_SELECTION.json"))


def regime_of(G, n):
    if G == n:
        return "iid"
    return "G=10" if G <= 20 else "G=53"


def selected(reg):
    return SEL["iid"]["selected"] if reg == "iid" else SEL["clustered"][reg]["selected"]


# ============================================================================================== demo
def degeneracy_demo():
    d = np.zeros(10)
    out = {"input": "10 initial states, paired success difference d = 0 for all (e.g. C-F d8 task 5: 10/10 vs 10/10)",
           "stats_core.signflip_ci": SC.signflip_ci(d, np.ones(10), 0.0, 0.95)}
    steps = []
    for dl in (0.0, 1e-14, 1e-13, 2.4e-13, 2.6e-13, 1e-6, 0.05, 0.3):
        T = d - dl
        steps.append({"delta": dl, "shifted_totals": f"all 10 equal {-dl:g}",
                      "p_two_sided": SC.signflip_test(T, "two-sided")["p"]})
    out["p_of_delta"] = steps
    out["explanation"] = (
        "signflip_ci keeps delta iff p(N - delta D) > 0.05. At delta = 0 all totals are 0 -> p = 1. For any delta whose "
        "shifted totals exceed the 1e-12 * max(1, sum|T|) tie tolerance (|delta| > ~2.5e-13 here), all 10 totals are "
        "-delta, i.e. the same sign; the observed sum is the most extreme of the 2^10 sign patterns -> p = 2/1024 = "
        "0.002 < 0.05 -> rejected. Bisection therefore converges to the tolerance, giving +-2.4958e-13. The sign-flip "
        "null 'shifted totals symmetric about 0' can only hold at delta = 0 for a {-1, 0, 1} difference with zeros: the "
        "zeros become -delta, so H0: theta = delta != 0 never implies symmetry. The interval is an artefact of the "
        "tie tolerance, not a statement about any estimand.")
    d1 = np.r_[-1.0, np.zeros(9)]
    out["one_sided_case"] = {"input": "1 discordant state (-1), 9 concordant (e.g. C-F d8 task 7)",
                             "stats_core.signflip_ci": SC.signflip_ci(d1, np.ones(10), -0.1, 0.95),
                             "p_at_delta=+1e-6": SC.signflip_test(d1 - 1e-6)["p"],
                             "p_at_delta=0": SC.signflip_test(d1)["p"],
                             "explanation": "any delta > tol makes all 10 shifted totals negative -> p = 2/1024; the "
                                            "upper end is pinned at 0 (+tol) although only ONE state is discordant and "
                                            "the exact McNemar p is 1."}
    return out


# ============================================================================================== OpenVLA
def openvla():
    E = pd.read_csv(f"{P0}/OPENVLA_episode_records.csv")
    W = E.pivot_table(index=["task", "init"], columns="cond", values="success").reset_index()
    old_t = pd.read_csv(f"{P0}/OPENVLA_REANALYSIS_per_task.csv").set_index("row_id")
    old_s = pd.read_csv(f"{P0}/OPENVLA_REANALYSIS.csv").set_index("row_id")
    specs = [("C-F d8", "corrected_d8", "feedback_d8"), ("C-F d24", "corrected_d24", "feedback_d24")]
    for a in (8, 24):
        for m in ("feedback", "corrected", "reverse"):
            specs.append((f"{m[0].upper()}-N d{a}", f"{m}_d{a}", "natural"))
    for a in (8, 24):
        for m in ("feedback", "corrected", "reverse"):
            specs.append((f"{m[0].upper()}-Nrep d{a}", f"{m}_d{a}", "natural_rep"))
    specs.append(("Nrep-N", "natural_rep", "natural"))
    rows, tabs = [], []
    for cid, ca, cb in specs:
        for scope in ["all"] + list(range(10)):
            w = W if scope == "all" else W[W.task == scope]
            ya, yb = w[ca].astype(int).values, w[cb].astype(int).values
            a, b, c, d = A.table_counts(ya, yb)
            n = a + b + c + d
            rid = (f"P0E.succ.{cid.replace(' ', '_')}" if scope == "all" else
                   f"P0E.succ_task.{cid.replace(' ', '_')}.t{scope}")
            tabs.append({"row_id": rid, "contrast": cid, "cond_a": ca, "cond_b": cb, "task": scope, "n_states": n,
                         "both_success_a11": a, "a_only_success_b10": b, "b_only_success_c01": c, "both_fail_d00": d,
                         "succ_a": a + b, "succ_b": a + c, "n_discordant": b + c})
            est = (b - c) / n
            p_ex = A.signflip_p_trinomial(b, c, n, 0.0)   # exact sign-flip at 0 = exact McNemar, any n
            nw = A.newcombe10(a, b, c, d)
            tg = A.tango_ci(b, c, n)
            wd = A.wald_paired(b, c, n)
            sf_exact = A.signflip_inv_trinomial(b, c, n)
            reg = "iid"
            meth = selected(reg)
            new_ci = {"newcombe10": nw, "tango": tg, "wald": wd, "old_signflip_inv": sf_exact}[meth]
            r = {"block": "OpenVLA per-task" if scope != "all" else "OpenVLA all 100 states", "row_id": rid,
                 "model": "OpenVLA-7B / LIBERO-Spatial", "contrast": cid, "task_level": scope, "clustering": "initial state",
                 "unit": "initial state (1 paired episode per state & condition)", "n_units": n, "n_clusters": n,
                 "n_informative": b + c, "table_a11_b10_c01_d00": f"{a}/{b}/{c}/{d}", "estimate": est}
            if scope == "all":
                o = old_s.loc[rid]
                assert abs(o.estimate - est) < 1e-12 and int(o.a_only) == b and int(o.b_only) == c
                r.update({"old_ci_method": "Monte Carlo sign-flip inversion (R=20,000) [state_sf_inv_ci95]",
                          "old_ci95_lo": o.state_sf_inv_ci95_lo, "old_ci95_hi": o.state_sf_inv_ci95_hi,
                          "old_ci95_exact_inversion_lo": sf_exact[0], "old_ci95_exact_inversion_hi": sf_exact[1],
                          "old_aux_ci": f"state bootstrap [{o.state_boot_ci95_lo:.3f}, {o.state_boot_ci95_hi:.3f}]",
                          "old_p": o.p_signflip_state_exact, "old_p_method": "exact state sign-flip = exact McNemar",
                          "family": o.family, "old_p_holm": o.p_holm_state,
                          "old_p_equiv_m10": o.p_equiv_m10_posthoc, "margin": 0.10,
                          "margin_status": "post-hoc margin (review_followup3)"})
            else:
                o = old_t.loc[rid]
                assert abs(o.estimate - est) < 1e-12 and int(o.n_informative_states) == b + c
                ref = SC.signflip_ci((ya - yb).astype(float), np.ones(n), est, 0.95)
                assert np.allclose(ref, [o.sf_inv_ci95_lo, o.sf_inv_ci95_hi], atol=1e-15, rtol=1e-9)
                r.update({"old_ci_method": "exact sign-flip inversion 2^10 [sf_inv_ci95]",
                          "old_ci95_lo": o.sf_inv_ci95_lo, "old_ci95_hi": o.sf_inv_ci95_hi,
                          "old_ci95_exact_inversion_lo": sf_exact[0], "old_ci95_exact_inversion_hi": sf_exact[1],
                          "old_p": o["p_signflip_exact_2^10"], "old_p_method": "exact sign-flip 2^10 = exact McNemar",
                          "family": "none (per-task, descriptive)", "old_p_holm": np.nan, "old_p_equiv_m10": np.nan,
                          "margin": np.nan, "margin_status": "no equivalence test per task"})
            assert abs(r["old_p"] - p_ex) < 1e-12
            lo, hi = r["old_ci95_lo"], r["old_ci95_hi"]
            r["old_ci_degeneracy"] = ("ZERO-WIDTH (all differences 0)" if hi - lo < 1e-9 else
                                      ("ENDPOINT PINNED AT 0 (tie artefact)" if (0 < abs(lo) < 1e-9 or 0 < abs(hi) < 1e-9)
                                       else "non-degenerate (but rests on the shifted-symmetry assumption)"))
            r.update({"new_ci_method": f"{meth} (pre-specified selection, A1_METHOD_SELECTION.json)",
                      "new_ci95_lo": new_ci[0], "new_ci95_hi": new_ci[1],
                      "newcombe10_ci95": f"[{nw[0]:.4f}, {nw[1]:.4f}]", "tango_ci95": f"[{tg[0]:.4f}, {tg[1]:.4f}]",
                      "wald_ci95": f"[{wd[0]:.4f}, {wd[1]:.4f}]",
                      "new_p": p_ex, "new_p_method": "exact McNemar (= old sign-flip at delta=0; unchanged, valid)",
                      "new_p_holm": r["old_p_holm"]})
            if scope == "all":
                f = lambda al, a=a, b=b, c=c, d=d: A.newcombe10(a, b, c, d, al)  # noqa: E731
                pl, pu = A.ci_pvalue(f, 0.10, "lower"), A.ci_pvalue(f, 0.10, "upper")
                r.update({"new_p_lower_m10": pl, "new_p_upper_m10": pu, "new_p_equiv_m10": max(pl, pu),
                          "tango_p_equiv_m10": max(A.tango_test(b, c, n, -0.10, "greater"),
                                                   A.tango_test(b, c, n, 0.10, "less"))})
            r["estimand"] = ("difference in success probability for a new initial state from the same (LIBERO-Spatial "
                             "init-file) distribution, including run-to-run execution randomness; the 10 states/task are "
                             "the first 10 of the benchmark init file (not a random sample)")
            r["status"] = LABEL
            rows.append(r)
    R = pd.DataFrame(rows)
    # Holm on p_equiv within the original families (state-level rows only)
    st = R[R.task_level == "all"]
    for fam, g in st.groupby("family"):
        if fam.startswith(("F1", "F2", "F3")):
            R.loc[g.index, "old_p_equiv_m10_holm"] = SC.holm(g.old_p_equiv_m10.values)
            R.loc[g.index, "new_p_equiv_m10_holm"] = SC.holm(g.new_p_equiv_m10.values)
    for i, r in R.iterrows():
        if r.task_level == "all":
            R.loc[i, "old_difference_judgment"] = "different from 0 (Holm)" if (r.old_p_holm < .05) else (
                "different unadjusted only" if r.old_p < .05 else "not different from 0")
            R.loc[i, "new_difference_judgment"] = R.loc[i, "old_difference_judgment"]
            for k in ("old", "new"):
                pe, ph = r[f"{k}_p_equiv_m10"], R.loc[i].get(f"{k}_p_equiv_m10_holm", np.nan)
                R.loc[i, f"{k}_equivalence_judgment_m10"] = (
                    "equivalent +-10 pp (Holm)" if (np.isfinite(ph) and ph < .05) else
                    ("equivalent +-10 pp unadjusted only" if pe < .05 else "equivalence +-10 pp not established"))
            ci_old_ex0 = r.old_ci95_lo > 0 or r.old_ci95_hi < 0
            ci_new_ex0 = r.new_ci95_lo > 0 or r.new_ci95_hi < 0
            R.loc[i, "old_ci_excludes_0"] = ci_old_ex0
            R.loc[i, "new_ci_excludes_0"] = ci_new_ex0
        else:
            R.loc[i, "old_difference_judgment"] = "not different from 0" if r.old_p >= .05 else "different unadjusted"
            R.loc[i, "new_difference_judgment"] = R.loc[i, "old_difference_judgment"]
            R.loc[i, "old_ci_excludes_0"] = bool(r.old_ci95_lo > 0 or r.old_ci95_hi < 0)
            R.loc[i, "new_ci_excludes_0"] = bool(r.new_ci95_lo > 0 or r.new_ci95_hi < 0)
            R.loc[i, "old_equivalence_judgment_m10"] = "not tested"
            R.loc[i, "new_equivalence_judgment_m10"] = "not tested"
        R.loc[i, "use"] = ("descriptive only (10 states per task; Newcombe interval valid for the new-state estimand "
                           "but uninformative); old sign-flip interval INVALID (do not cite)" if r.task_level != "all"
                           else "inferential at state level (100 states)")
    return R, pd.DataFrame(tabs)


# ============================================================================================== SpatialVLA
def outcome_pairs(pairs, exp, cfg, contrast, direction):
    sign_swap = contrast in ("F-C", "FD-CD")
    base = {"F-C": "C-F", "FD-CD": "CD-FD"}.get(contrast, contrast)
    q = (pairs.experiment == exp) & (pairs.config == cfg) & (pairs.contrast == base)
    if direction is not None:
        q &= pairs.direction == direction
    x = pairs[q][["task", "seed", "success_a", "success_b", "cluster_primary", "cluster_physical"]].copy()
    if sign_swap:
        x["success_a"], x["success_b"] = x["success_b"].values, x["success_a"].values
    x["ya"], x["yb"] = x.success_a.astype(int), x.success_b.astype(int)
    return x.drop(columns=["success_a", "success_b"]).reset_index(drop=True)


def spec_data(pairs, sp):
    s = sp["src"]
    if s[0] == "d":
        x = outcome_pairs(pairs, s[1], s[2], s[3], s[4])
        if sp.get("seeds039"):
            x = x[x.seed <= 39]
        return "direct", x
    _, exp, c1, cn1, c2, cn2, dr, dr2 = s
    a = outcome_pairs(pairs, exp, c2, cn2, dr2 if dr2 else dr).set_index(["task", "seed"])
    b = outcome_pairs(pairs, exp, c1, cn1, dr).set_index(["task", "seed"])
    if sp.get("seeds039"):
        a, b = a[a.index.get_level_values(1) <= 39], b[b.index.get_level_values(1) <= 39]
    j = a.join(b[["ya", "yb"]], rsuffix="1", how="inner")
    assert len(j) == len(a) == len(b)
    j = j.rename(columns={"ya": "ya2", "yb": "yb2"}).reset_index()
    return "interaction", j


def task_ci(kind, xx, cl_col, alpha):
    """returns est, (lo, hi) with the RS-Newcombe (direct) or RS-Newcombe MOVER difference (interaction)."""
    cl = xx[cl_col].values
    if kind == "direct":
        dd = (xx.ya - xx.yb).values.astype(float)
        deff = A.cluster_deff(dd, cl)[0]
        return float(dd.mean()), A.rs_newcombe(xx.ya.values, xx.yb.values, cl, alpha, deff), deff
    d2 = (xx.ya2 - xx.yb2).values.astype(float)
    d1 = (xx.ya1 - xx.yb1).values.astype(float)
    f2 = A.cluster_deff(d2, cl)[0]
    f1 = A.cluster_deff(d1, cl)[0]
    l2, u2 = A.rs_newcombe(xx.ya2.values, xx.yb2.values, cl, alpha, f2)
    l1, u1 = A.rs_newcombe(xx.ya1.values, xx.yb1.values, cl, alpha, f1)
    rr = A.cluster_corr(d1, d2, cl)
    est, L, U = A.mover_lincomb([(1, d2.mean(), l2, u2), (-1, d1.mean(), l1, u1)], [[1, rr], [rr, 1]])
    return float(est), (L, U), (f1, f2, rr)


def level_ci(kind, x, lvl, cl_col, alpha):
    if lvl != "pooled":
        e, ci, info = task_ci(kind, x[x.task == lvl], cl_col, alpha)
        return e, ci, info
    parts, infos = [], []
    for t in ("coke", "move_near"):
        e, (lo, hi), info = task_ci(kind, x[x.task == t], cl_col, alpha)
        parts.append((0.5, e, lo, hi))
        infos.append(info)
    est, L, U = A.mover_lincomb(parts)
    return est, (L, U), infos


def cluster_t_ci(kind, x, lvl, cl_col, alpha):
    xx = x if lvl == "pooled" else x[x.task == lvl]
    y = xx.copy()
    y["d"] = (y.ya - y.yb) if kind == "direct" else ((y.ya2 - y.yb2) - (y.ya1 - y.yb1))
    _, tk, N, D = BDE.cluster_arrays(y, cl_col)
    return SC.cluster_t(N, D, 1 - alpha)[1]


def spatial():
    pairs = pd.read_csv(f"{BDIR}/pairs.csv")
    de = pd.read_csv(f"{BDIR}/DIRECT_EFFECTS.csv")
    rows = []
    for sp in BDE.build_specs():
        kind, x = spec_data(pairs, sp)
        for lvl in ("coke", "move_near", "pooled"):
            xx = x if lvl == "pooled" else x[x.task == lvl]
            if len(xx) == 0:
                continue
            for cl in BDE.CLUSTERINGS:
                ccol = f"cluster_{cl}"
                if cl == "episode":
                    x = x.copy()
                    x["cluster_episode"] = x["task"] + "-s" + x["seed"].astype(str)
                    xx = x if lvl == "pooled" else x[x.task == lvl]
                ref = de[(de.experiment == sp["exp"]) & (de.config == sp["config"]) & (de.contrast == sp["contrast"]) &
                         (de.task == lvl) & (de.clustering == cl)]
                assert len(ref) == 1, (sp["exp"], sp["config"], sp["contrast"], lvl, cl)
                o = ref.iloc[0]
                est, ci95, info = level_ci(kind, x, lvl, ccol, 0.05)
                assert abs(100 * est - o.estimate_pp) < 1e-5 * max(1, abs(o.estimate_pp)) + 1e-6, (sp, lvl, cl, est, o.estimate_pp)
                ci90 = level_ci(kind, x, lvl, ccol, 0.10)[1]
                # regime of the inference (per task G); pooled -> the most demanding of its two task regimes
                regs = []
                for t in (("coke", "move_near") if lvl == "pooled" else (lvl,)):
                    xt = x[x.task == t]
                    regs.append(regime_of(xt[ccol].nunique(), len(xt)))
                reg = "G=10" if "G=10" in regs else ("G=53" if "G=53" in regs else "iid")
                meth = selected(reg) if kind == "direct" or reg == "iid" else selected(reg)
                r = {"block": "SpatialVLA DIRECT_EFFECTS", "model": "SpatialVLA-4B", "experiment": sp["exp"],
                     "kind": kind, "role": sp["role"], "config": sp["config"], "contrast": sp["contrast"],
                     "task_level": lvl, "clustering": cl, "family": o.family, "family_m": o.family_size,
                     "row_id": (f"P0B.{sp['exp']}.{sp['config']}.{sp['contrast']}.{lvl}.{cl}"
                                if sp["exp"] in ("exp35", "P1D") and sp["role"] == "core" and kind == "direct"
                                and sp["config"] in ("r1e4", "r4e1") and sp["contrast"] in ("R-N", "C-F")
                                and cl != "episode" else ""),
                     "unit": "episode unit (task, seed); clusters = " + {"primary": "instruction-inclusive initial state",
                                                                         "physical": "physical layout",
                                                                         "episode": "none (original unit; not cluster-appropriate)"}[cl],
                     "n_units": int(o.n_units), "n_clusters": int(o.n_clusters),
                     "n_informative": int(o.n_informative_clusters), "estimate": est,
                     "estimand": ("within task: unit-weighted mean paired success difference over the 120 (or 40) "
                                  "units = difference of success rates on the same units; pooled: task-equal mean "
                                  "(ORIGINAL estimand weights kept); target = new units from the same initial-state "
                                  "distribution, cluster = repeated physical/initial state"),
                     "old_ci_method": o.ci_method, "old_ci95_lo": o.ci95_lo_pp / 100, "old_ci95_hi": o.ci95_hi_pp / 100,
                     "old_ci90_lo": o.ci90_lo_pp / 100, "old_ci90_hi": o.ci90_hi_pp / 100,
                     "old_p": o.p_diff_signflip, "old_p_holm": o.p_diff_holm,
                     "old_p_equiv_m10": o.p_equiv_m10, "old_p_equiv_m10_holm": o.p_equiv_m10_holm,
                     "old_p_equiv_m5": o.p_equiv_m5, "old_p_equiv_m5_holm": o.p_equiv_m5_holm,
                     "old_difference_judgment": o.difference_judgment,
                     "old_equivalence_judgment_m10": o.equivalence_judgment_m10,
                     "old_equivalence_judgment_m5": o.equivalence_judgment_m5,
                     "margin": "0.10 (original/prereg for P1D S1-S4) and 0.05 (post-hoc)",
                     "inference_regime": reg, "new_ci_method": None}
                lo, hi = r["old_ci95_lo"], r["old_ci95_hi"]
                r["old_ci_degeneracy"] = ("ZERO-WIDTH" if hi - lo < 1e-9 else
                                          ("ENDPOINT PINNED AT 0 (tie artefact)" if (0 < abs(lo) < 1e-8 or 0 < abs(hi) < 1e-8)
                                           else "non-degenerate (but rests on the shifted-symmetry assumption)"))
                r["rs_newcombe_ci95_lo"], r["rs_newcombe_ci95_hi"] = ci95
                r["rs_newcombe_ci90_lo"], r["rs_newcombe_ci90_hi"] = ci90
                r["deff_info"] = json.dumps(info, default=lambda v: round(float(v), 4))
                ct = cluster_t_ci(kind, x, lvl, ccol, 0.05)
                r["cluster_t_ci95_lo"], r["cluster_t_ci95_hi"] = ct
                if meth is None:
                    r["new_ci_method"] = f"NR: no method reached the coverage criterion in regime {reg}"
                    r["new_ci95_lo"] = r["new_ci95_hi"] = np.nan
                    for k in (10, 5):
                        r[f"new_p_equiv_m{k}"] = np.nan
                else:
                    if meth == "rs_newcombe10":
                        fci = lambda al, kind=kind, x=x, lvl=lvl, ccol=ccol: level_ci(kind, x, lvl, ccol, al)[1]  # noqa: E731
                        r["new_ci_method"] = ("Rao-Scott design-effect Newcombe 10" +
                                              (" + MOVER difference" if kind == "interaction" else "") +
                                              (" + MOVER task-equal mean" if lvl == "pooled" else "") +
                                              (" (deff = 1: singleton clusters)" if reg == "iid" else ""))
                    elif meth == "newcombe10":
                        fci = lambda al, kind=kind, x=x, lvl=lvl, ccol=ccol: level_ci(kind, x, lvl, ccol, al)[1]  # noqa: E731
                        r["new_ci_method"] = "Newcombe 10 (singleton clusters; deff = 1)" + (" + MOVER" if lvl == "pooled" or kind == "interaction" else "")
                    elif meth == "cluster_t":
                        fci = lambda al, kind=kind, x=x, lvl=lvl, ccol=ccol: cluster_t_ci(kind, x, lvl, ccol, al)  # noqa: E731
                        r["new_ci_method"] = "cluster-robust t (linearisation, G-1 df)"
                    else:
                        raise ValueError(f"selected method {meth} not implemented for SpatialVLA rows")
                    nci = fci(0.05)
                    r["new_ci95_lo"], r["new_ci95_hi"] = nci
                    n90 = fci(0.10)
                    r["new_ci90_lo"], r["new_ci90_hi"] = n90
                    for k, m in ((10, 0.10), (5, 0.05)):
                        pl, pu = A.ci_pvalue(fci, m, "lower"), A.ci_pvalue(fci, m, "upper")
                        r[f"new_p_lower_m{k}"], r[f"new_p_upper_m{k}"] = pl, pu
                        r[f"new_p_equiv_m{k}"] = max(pl, pu)
                    r["new_p_ci_inversion_at_0 (secondary)"] = A.ci_pvalue_two_sided_zero(fci)
                r["new_p"] = o.p_diff_signflip
                r["new_p_method"] = "sign-flip test at delta = 0 (unchanged: valid under the symmetry null at 0)"
                r["new_p_holm"] = o.p_diff_holm
                r["status"] = LABEL
                rows.append(r)
        print(f"[spatial] {sp['exp']} {sp['config']} {sp['contrast']}", flush=True)
    R = pd.DataFrame(rows)
    fam_mask = ~R.family.str.startswith("(compat")
    for (fam, lvl, cl), g in R[fam_mask].groupby(["family", "task_level", "clustering"]):
        for k in (10, 5):
            R.loc[g.index, f"new_p_equiv_m{k}_holm"] = SC.holm(g[f"new_p_equiv_m{k}"].values)
    mirror = {"F-C": "C-F", "FD-CD": "CD-FD"}
    for i in R[~fam_mask].index:
        r = R.loc[i]
        j = R[(R.experiment == r.experiment) & (R.config == r.config) & (R.contrast == mirror[r.contrast]) &
              (R.task_level == r.task_level) & (R.clustering == r.clustering)].index[0]
        for k in (10, 5):
            R.loc[i, f"new_p_equiv_m{k}_holm"] = R.loc[j, f"new_p_equiv_m{k}_holm"]
    for i, r in R.iterrows():
        R.loc[i, "new_difference_judgment"] = r.old_difference_judgment
        R.loc[i, "old_ci_excludes_0"] = bool(r.old_ci95_lo > 0 or r.old_ci95_hi < 0)
        R.loc[i, "new_ci_excludes_0"] = bool(np.isfinite(r.new_ci95_lo) and (r.new_ci95_lo > 0 or r.new_ci95_hi < 0))
        for k in (10, 5):
            ph, pe = R.loc[i, f"new_p_equiv_m{k}_holm"], r[f"new_p_equiv_m{k}"]
            if not np.isfinite(pe):
                R.loc[i, f"new_equivalence_judgment_m{k}"] = f"NR (no valid population interval; +-{k} pp not assessable)"
            else:
                R.loc[i, f"new_equivalence_judgment_m{k}"] = (
                    f"equivalent within +-{k} pp (Holm)" if ph < .05 else
                    (f"equivalent within +-{k} pp unadjusted only" if pe < .05 else f"equivalence +-{k} pp not established"))
        R.loc[i, "use"] = ("NR / descriptive only" if not np.isfinite(r.new_ci95_lo) else
                           ("reproduction of the original unit only (episode clustering ignores repeated states)"
                            if r.clustering == "episode" else "inferential (post-hoc corrected interval)"))
    return R


# ============================================================================================== call sites
CALLSITES = [
    ("scripts/review_followup2/stats_core.py:244 signflip_ci", "library", "test inversion on shifted totals",
     "AFFECTED for discrete/paired-binary outcomes with ties (zero-width / endpoint at 0); assumption false for delta != 0"),
    ("scripts/review_followup2/stats_core.py:310 tost(method='signflip')", "library", "one-sided sign-flip at +-margin",
     "AFFECTED for paired-binary outcomes: size up to 0.078 at the margin in A1_TOST_SIZE_SIM (nominal 0.05)"),
    ("scripts/review_followup2/stats_core.py:364 analyze(ci_inversion=True, margin=...)", "library", "calls signflip_ci/tost",
     "affected via signflip_ci/tost when outcome is binary"),
    ("scripts/review_followup2/stats_core.py:166 signflip_test (delta=0)", "library", "test at 0",
     "NOT affected: valid under its stated symmetry null at 0 (= exact McNemar for iid paired binary)"),
    ("scripts/review_followup2/b_direct_effects.py:143/147 infer (signflip_ci / mc_inv_ci)", "SpatialVLA success",
     "CIs of all 819 DIRECT_EFFECTS rows", "AFFECTED -> recomputed here (CI_VALIDITY_AUDIT.csv block SpatialVLA)"),
    ("scripts/review_followup2/b_direct_effects.py:150 SC.tost(signflip)", "SpatialVLA success", "TOST m=10/5 + Holm",
     "AFFECTED -> recomputed (new_p_equiv_*)"),
    ("scripts/review_followup2/b_direct_effects.py:481/483 infer_light", "SpatialVLA success (B4 table)",
     "success CI in B4_TRAJECTORY_VS_SUCCESS.csv", "AFFECTED (15 zero-width, 19 endpoint-at-0 rows); descriptive table; "
     "use CI_VALIDITY_AUDIT values for the same contrasts"),
    ("scripts/review_followup3/p0_b_spatial_task_effects.py:110/112/114 infer_ND", "SpatialVLA weighting sensitivity",
     "physical/primary-equal-weight rows (different estimand)", "AFFECTED (3 endpoint-at-0 rows); not recomputed: "
     "post-hoc sensitivity estimand -> descriptive only"),
    ("scripts/review_followup3/p0_b_spatial_task_effects.py (via BDE.infer)", "SpatialVLA core per-task/pooled",
     "SPATIAL_TASK_EFFECTS*.csv", "AFFECTED -> = DIRECT_EFFECTS core rows (row_id P0B.*) recomputed here"),
    ("scripts/review_followup3/p0_e_openvla.py:108 BDE.mc_inv_ci", "OpenVLA success, 100 states",
     "state_sf_inv_ci95", "AFFECTED (C-N d8 upper end -7e-14: tie artefact) -> recomputed"),
    ("scripts/review_followup3/p0_e_openvla.py:109 SC.tost(signflip, m=0.10)", "OpenVLA success",
     "p_equiv_m10_posthoc", "AFFECTED -> recomputed"),
    ("scripts/review_followup3/p0_e_openvla.py:205 SC.signflip_ci per task", "OpenVLA success, 10 states/task",
     "sf_inv_ci95 (150 rows)", "AFFECTED: 11 zero-width + 75 endpoint-at-0 -> recomputed"),
    ("scripts/review_followup2/a3_cluster_reanalysis.py:151/246 SC.analyze(margin=...)", "AutoVLA scene-level rates / "
     "continuous metrics, drive clusters", "sf_ci95/90, TOST", "PARTLY affected: binary-event contrasts with many zero "
     "drive totals (1 endpoint-at-0 row: P1C DM F unit-weighted sf_ci90 lower -0.0); continuous metrics unaffected by "
     "ties; not recomputed here (AutoVLA, out of A1 scope) -> flag only"),
    ("scripts/review_followup2/a3_cluster_reanalysis.py:362 SC.tost(bootstrap)", "AutoVLA", "drive bootstrap TOST",
     "not sign-flip; bootstrap percentile degenerates when all cluster totals equal (not observed in outputs)"),
    ("scripts/review_followup2/c_baseline_window.py:76/88/358 SC.analyze", "AutoVLA baseline/window metrics",
     "sf CI / TOST", "continuous metrics; no degenerate rows found by the CSV scan"),
    ("scripts/review_followup3/p0_a_population.py:347 SC.analyze(ci_inversion=True)", "AutoVLA population effects",
     "sf_ci95", "no degenerate rows found by the CSV scan"),
    ("scripts/review_followup3/p0_d_dependence.py:200/296/390 SC.analyze", "AutoVLA dependence subsets (6-7 drives)",
     "sf_ci95", "4 endpoint-at-0 rows in DEPENDENCE_disjoint_subsets.csv (vehicle/day-vehicle disjoint subsets, G=6-7) "
     "-> CI end at 0 is a tie artefact; read p (>= p_min = 2/2^G) not the CI; flag only"),
    ("scripts/review_followup3/p0_c_reconcile.py:156 SC.analyze", "AutoVLA reconciliation", "sf_ci95",
     "no degenerate rows found"),
    ("scripts/review_followup/p1c/analyze_p1c.py:226 signflip_ci (own copy)", "AutoVLA P1-C", "sf_ci95/90",
     "same construction; 1 endpoint-at-0 row in A_autovla_cluster/p1c_subsets_drive_level.csv (S2 day-vehicle disjoint, "
     "G=7, RN P, CI [-0.064, 0.0], p = 0.0625 = p_min); flag only"),
    ("scripts/review_followup3/p2_* MOTION_BASELINE_RESULTS (Geo rows)", "AutoVLA motion baseline", "ci95 [0,0]",
     "12 zero-width rows = privileged Geo control that is 0 BY CONSTRUCTION (structural zero, n_informative_drives = 0): "
     "descriptive; not evidence of equality"),
    ("scripts/review_followup3/cp2/p4_power_sim.py:123 diff_tost", "power planning (not run)", "-", "not a result"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    demo = degeneracy_demo()
    json.dump(demo, open(f"{OUT}/A1_DEGENERACY_DEMO.json", "w"), indent=1, default=float)
    ov, tabs = openvla()
    tabs.to_csv(f"{OUT}/A1_OPENVLA_2x2.csv", index=False)
    sv = spatial()
    allr = pd.concat([ov, sv], ignore_index=True, sort=False)
    first = ["block", "row_id", "model", "experiment", "kind", "config", "contrast", "task_level", "clustering", "unit",
             "n_units", "n_clusters", "n_informative", "table_a11_b10_c01_d00", "estimand", "estimate",
             "old_ci_method", "old_ci95_lo", "old_ci95_hi", "old_ci_degeneracy", "new_ci_method", "new_ci95_lo",
             "new_ci95_hi", "old_p", "new_p", "new_p_method", "family", "family_m", "old_p_holm", "new_p_holm", "margin",
             "old_p_equiv_m10", "new_p_equiv_m10", "old_p_equiv_m10_holm", "new_p_equiv_m10_holm", "old_p_equiv_m5",
             "new_p_equiv_m5", "old_p_equiv_m5_holm", "new_p_equiv_m5_holm", "old_difference_judgment",
             "new_difference_judgment", "old_ci_excludes_0", "new_ci_excludes_0", "old_equivalence_judgment_m10",
             "new_equivalence_judgment_m10", "old_equivalence_judgment_m5", "new_equivalence_judgment_m5", "use", "status"]
    cols = [c for c in first if c in allr.columns] + [c for c in allr.columns if c not in first]
    allr = allr[cols]
    allr.to_csv(f"{OUT}/CI_VALIDITY_AUDIT.csv", index=False, float_format="%.6g")
    ch = []
    for _, r in allr.iterrows():
        for k in ("difference_judgment", "equivalence_judgment_m10", "equivalence_judgment_m5"):
            o, n = r.get(f"old_{k}"), r.get(f"new_{k}")
            if isinstance(o, str) and isinstance(n, str) and o != n:
                ch.append({"block": r.block, "row_id": r.row_id, "experiment": r.get("experiment"), "kind": r.get("kind"),
                           "config": r.get("config"), "contrast": r.contrast, "task_level": r.task_level,
                           "clustering": r.clustering, "family": r.family, "judgment": k, "old": o, "new": n,
                           "estimate": r.estimate, "old_ci95": f"[{r.old_ci95_lo:.4f}, {r.old_ci95_hi:.4f}]",
                           "new_ci95": f"[{r.new_ci95_lo:.4f}, {r.new_ci95_hi:.4f}]",
                           "old_p_equiv": r.get(f"old_p_equiv_m{10 if 'm10' in k else 5}"),
                           "new_p_equiv": r.get(f"new_p_equiv_m{10 if 'm10' in k else 5}"),
                           "old_p_equiv_holm": r.get(f"old_p_equiv_m{10 if 'm10' in k else 5}_holm"),
                           "new_p_equiv_holm": r.get(f"new_p_equiv_m{10 if 'm10' in k else 5}_holm"),
                           "use": r.use, "status": LABEL})
        if r.old_ci_excludes_0 != r.new_ci_excludes_0:
            ch.append({"block": r.block, "row_id": r.row_id, "experiment": r.get("experiment"), "kind": r.get("kind"),
                       "config": r.get("config"), "contrast": r.contrast, "task_level": r.task_level,
                       "clustering": r.clustering, "family": r.family, "judgment": "95% CI excludes 0",
                       "old": bool(r.old_ci_excludes_0), "new": bool(r.new_ci_excludes_0), "estimate": r.estimate,
                       "old_ci95": f"[{r.old_ci95_lo:.4g}, {r.old_ci95_hi:.4g}]",
                       "new_ci95": f"[{r.new_ci95_lo:.4f}, {r.new_ci95_hi:.4f}]", "use": r.use, "status": LABEL})
    pd.DataFrame(ch).to_csv(f"{OUT}/A1_JUDGMENT_CHANGES.csv", index=False, float_format="%.6g")
    pd.DataFrame(CALLSITES, columns=["call_site", "data", "what", "affected"]).to_csv(f"{OUT}/A1_CALLSITES.csv", index=False)
    meta = {"selection": {k: v for k, v in SEL.items() if k != "rule"}, "n_rows": len(allr), "n_changes": len(ch),
            "label": LABEL}
    json.dump(meta, open(f"{OUT}/a1_audit_meta.json", "w"), indent=1, default=float)
    print(json.dumps({"rows": len(allr), "changes": len(ch)}, indent=1))


if __name__ == "__main__":
    main()
