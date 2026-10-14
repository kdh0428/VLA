#!/usr/bin/env python
"""
P0-C (review_followup3, 2026-10-14) -- NUMBER RECONCILIATION for the P1-A "+4.37 m" endpoint number and related
P1-A / P1-B / R8R9 / C1 / C2 / P1-C numbers quoted in EXPERIMENT_SUMMARY.md, PAPER_NUMBERS.md and review_followup2 docs.

POST-HOC audit of existing outputs. CPU only, no model inference. Existing files are only READ.

Steps
  1. Exact reproduction of the three versions of "+4.37 m" from their own code paths and seeds:
       V1  P1-A preregistered analysis (p1ab/p1a_analyze.py -> p1_stats.effect_row): OC - Geo, dev9_ref, family G, A-,
           scene-weighted, LOG-cluster bootstrap (seed 20261008 + 7), log sign-flip, Holm m = 3 (P1A_primary).
       V2  review_followup2 C drive re-analysis of the same contrast (c_baseline_window.c1_predeclared): DRIVE-cluster
           bootstrap (seed 20261013 + 5000 + 11), exact drive sign-flip, Holm m = 3 (predeclared family).
       V3  review_followup2 C1 post-hoc "excess model - Geo endpoint" (c_baseline_window.c1_units/c1_tables):
           ||OC9 - U9|| - ||Geo9 - r9||  -- a DIFFERENT per-unit metric (model counterfactual = U, not r), drive
           bootstrap with its own seed, Holm over the post-hoc 4-row excess family (m = 4).
  2. Independent recomputation of V2 with stats_core directly from units.parquet (no c_baseline code), bootstrap
     seed sensitivity (20 seeds), exact sign-flip and test-inversion CI, Holm m = 3.
  3. Per-unit comparison V2 metric vs V3 metric (where they differ and why).
  4. Machine check of quoted numbers in the docs against the source rows (single rounding of the full-precision
     source value at the quoted number of decimals).

Outputs (outputs/review_followup3/P0/):
  RECON_plus437_versions.csv, RECON_plus437_seed_sensitivity.csv, RECON_doc_number_check.csv,
  _frag_numbers_C_recon.json (merged into _frag_numbers_C.json by p0_c_window_weights.py)
Run:  nice -n 19 /root/miniforge3/envs/autovla/bin/python p0_c_reconcile.py
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
sys.path.insert(0, f"{POC}/scripts/review_followup2")
sys.path.insert(0, f"{POC}/scripts/review_followup/p1ab")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats_core as SC  # noqa: E402
import c_baseline_window as CBW  # noqa: E402  (module import only; its main() is NOT called, nothing is written)
import p1_stats as ST  # noqa: E402

RF = f"{POC}/outputs/review_followup"
RF2 = f"{POC}/outputs/review_followup2"
P1A = f"{RF}/P1A_gt_free_reference/run_20261008"
CDIR = f"{RF2}/C_baseline_window/run_20261013"
OUT = f"{POC}/outputs/review_followup3/P0"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def drive_of(l):
    p = l.split("_")
    return f"{p[0]}_{p[1]}"


def rnd(x, nd):
    """single half-up rounding of the full-precision value (what a correct table shows)."""
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def dbl_rnd(x, nd):
    """double rounding (first to nd+1 decimals, then to nd) -- the suspected transcription error."""
    return rnd(rnd(x, nd + 1), nd)


# ======================================================================================= 1. exact reproduction
def reproduce():
    rows = []
    U = pd.read_parquet(f"{P1A}/units.parquet")
    # ---- V1: P1-A preregistered, log level. seed: ST.SEED, G/A- first block, PRIMARY index 1 -> SEED + 7
    mask = (U.family == "G") & U.exclusion_reason.isna() & (U.group == "A-")
    v1 = {}
    for k, (cid, a, b, met) in enumerate((("P1A-1", "C1", "U", "suffix_loc"), ("P1A-2", "OC", "Geo", "dev9_ref"),
                                          ("P1A-3", "OC_refR", "OC", "dev9_ref"))):
        df = ST.paired_frame(U, a, b, met, mask)
        r = ST.effect_row("P1A_primary", cid, "A-", met, df, ST.SEED + 7 * k, margin=0.25)
        v1[cid] = r
    hol = ST.holm([v1[c]["scene_p_signflip"] for c in ("P1A-1", "P1A-2", "P1A-3")])
    stored = pd.read_csv(f"{P1A}/paired_effects.csv")
    st2 = stored[(stored.family == "P1A_primary") & stored.contrast.str.startswith("P1A-2")].iloc[0]
    r = v1["P1A-2"]
    rows.append({"version": "V1", "label": "P1-A preregistered analysis (original)", "contrast": "OC - Geo", "metric": "dev9_ref = ||X9 - r9|| (m)",
                 "family": "G", "group": "A-", "weighting": "scene (units averaged within scene, 52 scenes equal)",
                 "cluster_level": "log (16)", "bootstrap": "log multinomial, B = 10,000, seed 20261008+7, percentile",
                 "p_test": "log sign-flip exact 2^16", "holm_family": "P1A_primary m = 3 {P1A-1, P1A-2, P1A-3}",
                 "n_units": r["n_units"], "n_scenes": r["n_scenes"], "n_clusters": r["n_clusters"],
                 "estimate": r["scene_est"], "ci95_lo": r["scene_ci95_lo"], "ci95_hi": r["scene_ci95_hi"],
                 "p_raw": r["scene_p_signflip"], "p_holm": hol[1],
                 "stored_estimate": st2.scene_est, "stored_ci95_lo": st2.scene_ci95_lo, "stored_ci95_hi": st2.scene_ci95_hi,
                 "stored_p_holm": st2.holm_p, "source_file": f"{P1A}/paired_effects.csv (family P1A_primary, contrast P1A-2)",
                 "source_code": f"{POC}/scripts/review_followup/p1ab/p1a_analyze.py + p1_stats.py",
                 "quoted_in": "P1A analysis.md s0/s2: +4.37 [+3.34, +5.13], Holm 9.2e-5"})
    log("V1 reproduced", r["scene_est"], r["scene_ci95_lo"], r["scene_ci95_hi"], hol[1])

    # ---- V2: drive-level re-analysis of the predeclared contrast (exact c_baseline code path)
    PD = CBW.c1_predeclared()
    s = PD[(PD.source == "primary") & (PD.family == "G") & (PD.group == "A-") & PD.contrast.str.startswith("P1A-2")].iloc[0]
    stv2 = pd.read_csv(f"{CDIR}/c1_predeclared_and_gt_drive.csv")
    stv2 = stv2[(stv2.source == "primary") & (stv2.family == "G") & (stv2.group == "A-") & stv2.contrast.str.startswith("P1A-2")].iloc[0]
    rows.append({"version": "V2", "label": "review_followup2 C: drive-level re-analysis of the predeclared P1A-2", "contrast": "OC - Geo",
                 "metric": "dev9_ref = ||X9 - r9|| (m)", "family": "G", "group": "A-", "weighting": "scene",
                 "cluster_level": "drive (13)", "bootstrap": "drive multinomial, B = 10,000, seed 20261013+5000+11, percentile",
                 "p_test": "drive sign-flip exact 2^13", "holm_family": "predeclared P1A family m = 3 (drive p)",
                 "n_units": s.n_units, "n_scenes": s.n_scenes, "n_clusters": s.n_drives,
                 "estimate": s.est_scene, "ci95_lo": s.ci95_lo, "ci95_hi": s.ci95_hi, "p_raw": s.p_signflip, "p_holm": s.p_holm_m3_drive,
                 "sf_ci95_lo": s.sf_ci95_lo, "sf_ci95_hi": s.sf_ci95_hi,
                 "stored_estimate": stv2.est_scene, "stored_ci95_lo": stv2.ci95_lo, "stored_ci95_hi": stv2.ci95_hi, "stored_p_holm": stv2.p_holm_m3_drive,
                 "source_file": f"{CDIR}/c1_predeclared_and_gt_drive.csv (source=primary, family=G, group=A-, contrast='P1A-2 OC-Geo dev9_ref')",
                 "source_code": f"{POC}/scripts/review_followup2/c_baseline_window.py::c1_predeclared",
                 "quoted_in": "BASELINE_WINDOW_RECHECK.md C1.1 '[+3.34, +5.18]'; EXPERIMENT_SUMMARY.md s6 table '+4.37 m [+3.34, +5.18] ... Holm 7.3e-4'"})
    log("V2 reproduced", s.est_scene, s.ci95_lo, s.ci95_hi, s.p_holm_m3_drive)

    # ---- V3: post-hoc C1 excess model - Geo endpoint (exact c_baseline code path)
    UC, _ = CBW.c1_units()
    T = CBW.c1_tables(UC)
    t3 = T[(T.family == "G") & (T.group == "A-") & (T.metric == "ex_geo_end")].iloc[0]
    st3 = pd.read_csv(f"{CDIR}/c1_excess_effects.csv")
    st3 = st3[(st3.family == "G") & (st3.group == "A-") & (st3.metric == "ex_geo_end")].iloc[0]
    rows.append({"version": "V3", "label": "review_followup2 C1 post-hoc 'excess model - Geo, endpoint'", "contrast": "model change - Geo change",
                 "metric": "||OC9 - U9|| - ||Geo9 - r9|| (m)", "family": "G", "group": "A-", "weighting": "scene",
                 "cluster_level": "drive (13)", "bootstrap": "drive multinomial, B = 10,000, seed from c1_tables sequence (20261013 + 11 k), percentile",
                 "p_test": "drive sign-flip exact 2^13", "holm_family": "post-hoc C1 excess family m = 4 {ex_geo_end, ex_cm_end, ex_geo_sfx, ex_cm_sfx}",
                 "n_units": t3.n_units, "n_scenes": t3.n_scenes, "n_clusters": t3.n_drives,
                 "estimate": t3.est_scene, "ci95_lo": t3.ci95_lo, "ci95_hi": t3.ci95_hi, "p_raw": t3.p_signflip, "p_holm": t3.p_holm_posthoc_m4,
                 "sf_ci95_lo": t3.sf_ci95_lo, "sf_ci95_hi": t3.sf_ci95_hi,
                 "stored_estimate": st3.est_scene, "stored_ci95_lo": st3.ci95_lo, "stored_ci95_hi": st3.ci95_hi, "stored_p_holm": st3.p_holm_posthoc_m4,
                 "source_file": f"{CDIR}/c1_excess_effects.csv (family=G, group=A-, metric=ex_geo_end)",
                 "source_code": f"{POC}/scripts/review_followup2/c_baseline_window.py::c1_units/c1_tables",
                 "quoted_in": "BASELINE_WINDOW_RECHECK.md C1.2; EXPERIMENT_SUMMARY.md 3b; CLAIM_EVIDENCE_MATRIX.md; PAPER_NUMBERS.md claim 19 (labelled there as OC vs Geo)"})
    log("V3 reproduced", t3.est_scene, t3.ci95_lo, t3.ci95_hi, t3.p_holm_posthoc_m4)
    V = pd.DataFrame(rows)
    for c in ("estimate", "ci95_lo", "ci95_hi", "p_holm"):
        V[f"repro_absdiff_{c}"] = (V[c] - V[f"stored_{c}"]).abs()
    return V, U, UC


# ======================================================================================= 2. independent recompute
def independent(U, UC):
    u = U[(U.family == "G") & U.exclusion_reason.isna() & (U.group == "A-")].copy()
    u["drive_id"] = u.log_id.map(drive_of)
    res, sens = {}, []
    for cid, a, b, met in (("P1A-1", "C1", "U", "suffix_loc"), ("P1A-2", "OC", "Geo", "dev9_ref"), ("P1A-3", "OC_refR", "OC", "dev9_ref")):
        pr, rep = SC.pair_rows(u, ["unit_id"], "condition", met, a, b, keep_cols=["drive_id", "scene_id", "log_id"])
        assert rep["n_unpaired_or_missing"] == 0, rep
        sc = SC.scene_table(pr, "drive_id", "scene_id")
        r = SC.analyze(sc, "scene", B=10_000, seed=20261014)
        res[cid] = {"r": r, "pr": pr, "sc": sc}
        lab, N, D = SC.cluster_arrays(sc, "scene")
        for k in range(20):
            bs = SC.cluster_bootstrap(N, D, 10_000, 20261014 + 101 * k)
            lo, hi = SC.pct_ci(bs)
            sens.append({"contrast": cid, "seed": 20261014 + 101 * k, "B": 10_000, "ci95_lo": lo, "ci95_hi": hi})
        bs = SC.cluster_bootstrap(N, D, 200_000, 777)
        sens.append({"contrast": cid, "seed": 777, "B": 200_000, "ci95_lo": SC.pct_ci(bs)[0], "ci95_hi": SC.pct_ci(bs)[1]})
    hol = SC.holm([res[c]["r"]["p_signflip"] for c in ("P1A-1", "P1A-2", "P1A-3")])
    for c, h in zip(("P1A-1", "P1A-2", "P1A-3"), hol):
        res[c]["holm"] = h
    # swap test (arm sign): Geo - OC must give exactly the negated estimate
    pr = res["P1A-2"]["pr"].copy(); pr["d"] = -pr["d"]
    sw = SC.ratio(*SC.cluster_arrays(SC.scene_table(pr, "drive_id", "scene_id"), "scene")[1:])
    res["swap_P1A-2"] = sw
    # unit-level comparison of the V2 metric and the V3 metric
    v2 = res["P1A-2"]["pr"].set_index("unit_id")["d"]
    g = UC[(UC.source == "primary") & (UC.family == "G") & (UC.group == "A-")].set_index("unit_id")
    j = pd.DataFrame({"v2_dev9ref_OC_minus_Geo": v2, "v3_exgeo_end": g["ex_geo_end"], "u_dev9": g["u_dev9"]}).dropna()
    j["diff"] = j.v3_exgeo_end - j.v2_dev9ref_OC_minus_Geo
    cmp = {"n_units_joined": int(len(j)), "share_units_identical(|diff|<1e-9)": float((j["diff"].abs() < 1e-9).mean()),
           "share_units_U_eq_r(u_dev9 == 0)": float((j.u_dev9 < 1e-9).mean()),
           "mean_diff_v3_minus_v2_unit": float(j["diff"].mean()), "max_abs_diff": float(j["diff"].abs().max()),
           "corr": float(np.corrcoef(j.v2_dev9ref_OC_minus_Geo, j.v3_exgeo_end)[0, 1])}
    return res, pd.DataFrame(sens), cmp


# ======================================================================================= 4. document number check
def doc_checks():
    """quoted value -> source full-precision value -> single-rounded at the quoted decimals.  Each check is a
    (doc, regex fragment to locate the line, quoted numbers, source description, source values)."""
    CL = pd.read_csv(f"{CDIR}/c1_predeclared_and_gt_drive.csv")
    EX = pd.read_csv(f"{CDIR}/c1_excess_effects.csv")
    RC = pd.read_csv(f"{CDIR}/c2_recovery_fractions.csv")
    PO = pd.read_csv(f"{CDIR}/c2_position_length.csv")
    DEC = {(d["contrast"], d["estimand"]): d for d in json.load(open(f"{RF}/P1C_new_log_replication/run_20261008/analysis/decisions.json"))["decisions"]}
    PE = pd.read_csv(f"{P1A}/paired_effects.csv")

    def cl(cid, grp="A-", fam="G"):
        return CL[(CL.source == "primary") & (CL.family == fam) & (CL.group == grp) & CL.contrast.str.startswith(cid)].iloc[0]

    def ex(m, grp):
        return EX[(EX.family == "G") & (EX.group == grp) & (EX.metric == m)].iloc[0]

    def rc(run, cond, wt="scene", sub="all", met="amplification"):
        x = RC[RC.run.str.startswith(run) & (RC.condition == cond) & (RC.weighting == wt) & (RC.metric == met) & RC.subset.str.startswith(sub)]
        return x.iloc[0]

    def po(w):
        return PO[PO.analysis.str.startswith("total") & (PO.window_kind == "persistent") & (PO.contrast == "s1 - s3")
                  & (PO.metric == "amplification") & (PO.w == w)].iloc[0]

    p1a1, p1a2, p1a3 = cl("P1A-1"), cl("P1A-2"), cl("P1A-3")
    gamp = cl("GT amplification OC_refR-OC")
    pe1 = PE[(PE.family == "P1A_primary") & PE.contrast.str.startswith("P1A-1")].iloc[0]
    checks = [
        # id, doc, locate regex, quoted list, decimals, source label, source values, scale
        ("P1A-1 suffix (drive CI)", "EXPERIMENT_SUMMARY.md", r"suffix local motion \+0\.57 m/seg \[", [0.57, 0.44, 0.68], 2,
         "c1_predeclared P1A-1 drive (est, ci95)", [p1a1.est_scene, p1a1.ci95_lo, p1a1.ci95_hi]),
        ("P1A-1 suffix (drive CI)", "PAPER_NUMBERS.md", r"\+0\.571 m/seg \[", [0.571, 0.435, 0.682], 3,
         "c1_predeclared P1A-1 drive", [p1a1.est_scene, p1a1.ci95_lo, p1a1.ci95_hi]),
        ("P1A-1 suffix (log CI, original)", "P1A analysis.md", r"\+0\.571 m/segment \[", [0.571, 0.430, 0.677], 3,
         "paired_effects P1A-1 log", [pe1.scene_est, pe1.scene_ci95_lo, pe1.scene_ci95_hi]),
        ("P1A-2 (drive CI, predeclared contrast)", "EXPERIMENT_SUMMARY.md", r"endpoint \+4\.37 m \[\+3\.34, \+5\.18\]", [4.37, 3.34, 5.18], 2,
         "c1_predeclared P1A-2 drive", [p1a2.est_scene, p1a2.ci95_lo, p1a2.ci95_hi]),
        ("P1A-2 (drive CI, predeclared contrast)", "BASELINE_WINDOW_RECHECK.md", r"P1A-2: \[", [3.34, 5.18], 2,
         "c1_predeclared P1A-2 drive", [p1a2.ci95_lo, p1a2.ci95_hi]),
        ("P1A-2 Holm m=3 drive", "EXPERIMENT_SUMMARY.md", r"Holm 7\.3e-4\)\. 원 비교", [7.3e-4], None,
         "c1_predeclared p_holm_m3_drive (all 3 P1A contrasts)", [p1a2.p_holm_m3_drive]),
        ("C1 excess model-Geo endpoint (post-hoc)", "EXPERIMENT_SUMMARY.md", r"endpoint가 \+4\.37 m \[\+3\.32, \+5\.19\]", [4.37, 3.32, 5.19], 2,
         "c1_excess ex_geo_end", [ex("ex_geo_end", "A-").est_scene, ex("ex_geo_end", "A-").ci95_lo, ex("ex_geo_end", "A-").ci95_hi]),
        ("C1 excess model-Geo endpoint Holm m=4", "EXPERIMENT_SUMMARY.md", r"Holm 9\.8e-4\)", [9.8e-4], None,
         "c1_excess ex_geo_end p_holm_posthoc_m4", [ex("ex_geo_end", "A-").p_holm_posthoc_m4]),
        ("C1 excess model-Geo labelled as 'Output + context vs Geo'", "PAPER_NUMBERS.md", r"endpoint \*\*\+4\.37 m \[\+3\.32, \+5\.19\]\*\*", [4.37, 3.32, 5.19], 2,
         "c1_excess ex_geo_end (NOT the P1A-2 dev9_ref contrast)", [ex("ex_geo_end", "A-").est_scene, ex("ex_geo_end", "A-").ci95_lo, ex("ex_geo_end", "A-").ci95_hi]),
        ("P1A-3 OC_refR - OC drive", "BASELINE_WINDOW_RECHECK.md", r"P1A-3: \[", [-4.29, -2.59], 2, "c1_predeclared P1A-3 drive",
         [p1a3.ci95_lo, p1a3.ci95_hi]),
        ("P1A-3 estimate", "EXPERIMENT_SUMMARY.md", r"reference 교정 −3\.50 m", [-3.50], 2, "c1_predeclared P1A-3", [p1a3.est_scene]),
        ("GT amp OC_refR-OC A- (drive)", "PAPER_NUMBERS.md", r"\+19\.7 pp \[\+9\.0, \+29\.6\]", [19.7, 9.0, 29.6], 1, "c1_predeclared GT amplification OC_refR-OC x100",
         [100 * gamp.est_scene, 100 * gamp.ci95_lo, 100 * gamp.ci95_hi]),
        ("model-CM endpoint A-", "EXPERIMENT_SUMMARY.md", r"\+2\.35 m \[\+1\.27, \+3\.19\]", [2.35, 1.27, 3.19], 2, "c1_excess ex_cm_end A-",
         [ex("ex_cm_end", "A-").est_scene, ex("ex_cm_end", "A-").ci95_lo, ex("ex_cm_end", "A-").ci95_hi]),
        ("model-CM endpoint A- raw p / Holm", "PAPER_NUMBERS.md", r"A− \+2\.35 m \[\+1\.27, \+3\.19\]\*\* \(p = 7\.3e-4, Holm 9\.8e-4\)", [7.3e-4, 9.8e-4], None,
         "c1_excess ex_cm_end p_signflip, p_holm_posthoc_m4", [ex("ex_cm_end", "A-").p_signflip, ex("ex_cm_end", "A-").p_holm_posthoc_m4]),
        ("model-CM endpoint A+", "EXPERIMENT_SUMMARY.md", r"A\+ −0\.52 m \[−0\.82, −0\.26\]", [-0.52, -0.82, -0.26], 2, "c1_excess ex_cm_end A+",
         [ex("ex_cm_end", "A+").est_scene, ex("ex_cm_end", "A+").ci95_lo, ex("ex_cm_end", "A+").ci95_hi]),
        ("model-CM endpoint all", "EXPERIMENT_SUMMARY.md", r"전체 \+0\.20 m \[−0\.32, \+0\.56\]", [0.20, -0.32, 0.56], 2, "c1_excess ex_cm_end all",
         [ex("ex_cm_end", "all").est_scene, ex("ex_cm_end", "all").ci95_lo, ex("ex_cm_end", "all").ci95_hi]),
        ("R9 recovery w1..w4 (scene)", "EXPERIMENT_SUMMARY.md", r"1/2/3/4 step = 47 / 77 / \*\*90 \[81, 99\]\*\* / 96%", [47, 77, 90, 81, 99, 96], 0,
         "c2_recovery R9 all A- scene amplification x100",
         [100 * rc("R9", "win1").fraction, 100 * rc("R9", "win2").fraction, 100 * rc("R9", "win3").fraction, 100 * rc("R9", "win3").ci95_lo,
          100 * rc("R9", "win3").ci95_hi, 100 * rc("R9", "win4").fraction]),
        ("R9 recovery w1..w4 with CIs", "PAPER_NUMBERS.md", r"1/2/3/4 steps = 46\.7 \[29\.6, 62\.1\]", [46.7, 29.6, 62.1, 76.6, 62.7, 88.0, 90.1, 81.3, 99.0, 95.8, 93.3, 100],
         1, "c2_recovery R9 all A- scene amplification x100",
         sum([[100 * rc("R9", f"win{w}").fraction, 100 * rc("R9", f"win{w}").ci95_lo, 100 * rc("R9", f"win{w}").ci95_hi] for w in (1, 2, 3, 4)], [])),
        ("R8 Recent-GT share (unit)", "EXPERIMENT_SUMMARY.md", r"92\.9% \[89\.4, 99\.1\]", [92.9, 89.4, 99.1], 1, "c2_recovery R8 recent_gt unit x100",
         [100 * rc("R8", "recent_gt", "unit").fraction, 100 * rc("R8", "recent_gt", "unit").ci95_lo, 100 * rc("R8", "recent_gt", "unit").ci95_hi]),
        ("R8 Recent-GT share (scene)", "PAPER_NUMBERS.md", r"scene-weighted 93\.6% \[90\.3, 99\.8\]", [93.6, 90.3, 99.8], 1, "c2_recovery R8 recent_gt scene x100",
         [100 * rc("R8", "recent_gt").fraction, 100 * rc("R8", "recent_gt").ci95_lo, 100 * rc("R8", "recent_gt").ci95_hi]),
        ("P1-B early-late w1..w3", "EXPERIMENT_SUMMARY.md", r"\+10\.7 / \+4\.3 / \+0\.5%p", [10.7, 4.3, 0.5], 1, "c2_position total persistent s1-s3 amplification x100",
         [100 * po(w).est_scene for w in (1, 2, 3)]),
        ("P1-B w2 TOST Holm", "EXPERIMENT_SUMMARY.md", r"w2는 보정 전만\(Holm 0\.064\)", [0.064], 3, "c2_position w2 tost_p_equiv_holm_m3", [po(2).tost_p_equiv_holm_m3]),
        ("P1-C RN F", "EXPERIMENT_SUMMARY.md", r"RN F −34\.2 \[−38\.3, −29\.5\]", [-34.2, -38.3, -29.5], 1, "decisions.json RN F x100",
         [100 * DEC[("RN", "F")]["theta"], 100 * DEC[("RN", "F")]["boot_ci95_lo"], 100 * DEC[("RN", "F")]["boot_ci95_hi"]]),
        ("P1-C RV F", "EXPERIMENT_SUMMARY.md", r"RV F \+26\.8 \[\+21\.9, \+30\.7\]", [26.8, 21.9, 30.7], 1, "decisions.json RV F x100",
         [100 * DEC[("RV", "F")]["theta"], 100 * DEC[("RV", "F")]["boot_ci95_lo"], 100 * DEC[("RV", "F")]["boot_ci95_hi"]]),
        ("P1-C RN/RV P", "PAPER_NUMBERS.md", r"RN −4\.1 pp \[−6\.4, −2\.4\], RV \+3\.6 pp \[\+2\.2, \+5\.5\]", [-4.1, -6.4, -2.4, 3.6, 2.2, 5.5], 1,
         "decisions.json RN P, RV P x100",
         [100 * DEC[("RN", "P")]["theta"], 100 * DEC[("RN", "P")]["boot_ci95_lo"], 100 * DEC[("RN", "P")]["boot_ci95_hi"],
          100 * DEC[("RV", "P")]["theta"], 100 * DEC[("RV", "P")]["boot_ci95_lo"], 100 * DEC[("RV", "P")]["boot_ci95_hi"]]),
        ("P1-C DM F / P", "EXPERIMENT_SUMMARY.md", r"DM F \+6\.2 \[\+0\.6, \+12\.1\], sign-flip p = 0\.073 / P \+2\.6, p = 0\.10", [6.2, 0.6, 12.1, 0.073, 2.6, 0.10], None,
         "decisions.json DM F (theta, boot CI, p) and DM P (theta, p)",
         [100 * DEC[("DM", "F")]["theta"], 100 * DEC[("DM", "F")]["boot_ci95_lo"], 100 * DEC[("DM", "F")]["boot_ci95_hi"], DEC[("DM", "F")]["p_signflip"],
          100 * DEC[("DM", "P")]["theta"], DEC[("DM", "P")]["p_signflip"]]),
    ]
    docs = {"EXPERIMENT_SUMMARY.md": "/root/VLA/EXPERIMENT_SUMMARY.md", "PAPER_NUMBERS.md": "/root/VLA/paper_quantitative_package/PAPER_NUMBERS.md",
            "BASELINE_WINDOW_RECHECK.md": f"{RF2}/BASELINE_WINDOW_RECHECK.md", "P1A analysis.md": f"{P1A}/analysis.md",
            "CLAIM_EVIDENCE_MATRIX.md": f"{RF2}/CLAIM_EVIDENCE_MATRIX.md"}
    out = []
    for cid, doc, pat, quoted, nd, slab, src in checks:
        lines = open(docs[doc], encoding="utf-8").read().split("\n")
        hit = [i + 1 for i, l in enumerate(lines) if re.search(pat, l)]
        for q, s in zip(quoted, src):
            if nd is None:     # p values: compare at 2 significant digits
                ok = abs(q - float(f"{s:.2g}")) <= 1e-12 + 1e-9 * abs(q) or float(f"{s:.2g}") == q
                ok = ok or abs(q - s) / max(abs(s), 1e-300) < 0.06
                single, dbl = float(f"{s:.2g}"), np.nan
            else:
                single, dbl = rnd(s, nd), dbl_rnd(s, nd)
                ok = abs(single - q) < 1e-9
            out.append({"check_id": cid, "doc": doc, "doc_path": docs[doc], "doc_line": ",".join(map(str, hit)) if hit else "NOT FOUND",
                        "quoted": q, "source": slab, "source_full_precision": s, "source_single_rounded": single,
                        "source_double_rounded": dbl, "match_single_rounding": bool(ok),
                        "explained_by_double_rounding": bool((not ok) and nd is not None and abs(dbl - q) < 1e-9)})
    return pd.DataFrame(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    V, U, UC = reproduce()
    res, sens, cmp = independent(U, UC)
    r2 = res["P1A-2"]["r"]
    # authoritative row (independent recompute, seed 20261014) appended
    auth = {"version": "AUTH", "label": "AUTHORITATIVE: predeclared P1A-2 estimand, drive-cluster inference (independent recompute, review_followup3)",
            "contrast": "OC - Geo", "metric": "dev9_ref = ||X9 - r9|| (m)", "family": "G", "group": "A-", "weighting": "scene",
            "cluster_level": "drive (13)", "bootstrap": "drive multinomial, B = 10,000, seed 20261014, percentile", "p_test": "drive sign-flip exact 2^13",
            "holm_family": "predeclared P1A family m = 3 (drive p)", "n_units": r2["n_units"], "n_scenes": r2["n_scenes"], "n_clusters": r2["n_clusters"],
            "estimate": r2["theta"], "ci95_lo": r2["boot_ci95"][0], "ci95_hi": r2["boot_ci95"][1], "p_raw": r2["p_signflip"], "p_holm": res["P1A-2"]["holm"],
            "sf_ci95_lo": r2["sf_ci95"][0], "sf_ci95_hi": r2["sf_ci95"][1],
            "source_file": f"{P1A}/units.parquet (family G, A-, exclusion_reason null; conditions OC, Geo; metric dev9_ref)",
            "source_code": f"{POC}/scripts/review_followup3/p0_c_reconcile.py::independent", "quoted_in": "-"}
    V = pd.concat([V, pd.DataFrame([auth])], ignore_index=True)
    V.to_csv(f"{OUT}/RECON_plus437_versions.csv", index=False)
    sens.to_csv(f"{OUT}/RECON_plus437_seed_sensitivity.csv", index=False)
    import p0_c_doc_checks as DCK          # string-exact checks incl. tables.tex / MANUSCRIPT_REVISION (supersedes doc_checks())
    D = DCK.run()
    D.to_csv(f"{OUT}/RECON_doc_number_check.csv", index=False)
    s2 = sens[(sens.contrast == "P1A-2") & (sens.B == 10_000)]
    summ = {"seed_spread_P1A2_ci95_lo": [float(s2.ci95_lo.min()), float(s2.ci95_lo.max())],
            "seed_spread_P1A2_ci95_hi": [float(s2.ci95_hi.min()), float(s2.ci95_hi.max())],
            "P1A2_B200000": sens[(sens.contrast == "P1A-2") & (sens.B == 200_000)][["ci95_lo", "ci95_hi"]].iloc[0].tolist(),
            "swap_test_P1A2(Geo-OC)": res["swap_P1A-2"], "estimate": r2["theta"],
            "unit_metric_comparison_V2_vs_V3": cmp,
            "independent_all_three": {c: {"theta": res[c]["r"]["theta"], "ci95": res[c]["r"]["boot_ci95"], "sf_ci95": res[c]["r"]["sf_ci95"],
                                          "p": res[c]["r"]["p_signflip"], "holm_m3": res[c]["holm"], "n_drives": res[c]["r"]["n_clusters"],
                                          "n_informative": res[c]["r"]["n_informative"]} for c in ("P1A-1", "P1A-2", "P1A-3")}}
    json.dump(summ, open(f"{OUT}/RECON_plus437_summary.json", "w"), indent=1, default=float)
    frag = {}
    for _, r in V.iterrows():
        frag[f"P0C.plus437.{r.version}"] = {"label": r.label, "value": r.estimate, "ci95": [r.ci95_lo, r.ci95_hi], "p": r.p_raw, "p_holm": r.p_holm,
                                            "holm_family": r.holm_family, "cluster": r.cluster_level, "units": "m",
                                            "source_path": r.source_file, "source_code": r.source_code}
    for c in ("P1A-1", "P1A-3"):
        rr = res[c]["r"]
        frag[f"P0C.P1A.{c}.drive.AUTH"] = {"value": rr["theta"], "ci95": rr["boot_ci95"], "sf_ci95": rr["sf_ci95"], "p": rr["p_signflip"],
                                           "p_holm": res[c]["holm"], "holm_family": "predeclared P1A m = 3", "cluster": "drive (13)",
                                           "source_path": f"{P1A}/units.parquet", "source_code": f"{POC}/scripts/review_followup3/p0_c_reconcile.py"}
    json.dump(frag, open(f"{OUT}/_frag_numbers_C_recon.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    log("\n" + V[["version", "estimate", "ci95_lo", "ci95_hi", "p_raw", "p_holm", "repro_absdiff_estimate", "repro_absdiff_ci95_lo",
                   "repro_absdiff_ci95_hi", "repro_absdiff_p_holm"]].to_string())
    log(json.dumps(summ, indent=1, default=float))
    log("\n" + D[D.verdict != "OK"][["check_id", "doc", "doc_line", "quoted", "source_full_precision", "source_py_format",
                                      "verdict"]].to_string())
    log("doc lines not found:", D[D.doc_line == "NOT FOUND"].check_id.unique().tolist())


if __name__ == "__main__":
    main()
