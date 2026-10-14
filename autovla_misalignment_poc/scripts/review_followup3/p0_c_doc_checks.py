#!/usr/bin/env python
"""
P0-C document-number check (review_followup3, 2026-10-14).  POST-HOC audit, CPU only, read-only on every document.

Every check = (document, line pattern, list of (quoted string, format), source label, source values, source path).
The quoted string is compared character-exactly (after normalising the unicode minus) with the source value formatted
two ways:
  py  : Python's format spec (round-half-even on the binary value -- what the generating scripts use), e.g. 1.25 -> '1.2'
  up  : single half-up rounding of the decimal repr, e.g. 1.25 -> '1.3'
  dbl : double rounding (first to one more decimal, then half-up) -- the suspected transcription error
format: 'fN' = N decimals, 'gN' = N significant digits (p values), 'pctN' = x100 with N decimals, 'int' = integer.
Verdict: OK (py or up matches), DOUBLE_ROUNDING (only dbl matches), MISMATCH, NOT_FOUND (pattern not in doc).
"""
from __future__ import annotations

import json
import re
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
RF = f"{POC}/outputs/review_followup"
RF2 = f"{POC}/outputs/review_followup2"
CDIR = f"{RF2}/C_baseline_window/run_20261013"
P1A = f"{RF}/P1A_gt_free_reference/run_20261008"
P0 = f"{POC}/outputs/review_followup3/P0"
DOCS = {"EXPERIMENT_SUMMARY.md": "/root/VLA/EXPERIMENT_SUMMARY.md",
        "PAPER_NUMBERS.md": "/root/VLA/paper_quantitative_package/PAPER_NUMBERS.md",
        "tables.tex": "/root/VLA/paper_quantitative_package/tables.tex",
        "MANUSCRIPT_REVISION.md": "/root/VLA/paper_quantitative_package/MANUSCRIPT_REVISION.md",
        "CONCLUSIONS.md": "/root/VLA/CONCLUSIONS.md",
        "BASELINE_WINDOW_RECHECK.md": f"{RF2}/BASELINE_WINDOW_RECHECK.md",
        "CLAIM_EVIDENCE_MATRIX.md": f"{RF2}/CLAIM_EVIDENCE_MATRIX.md",
        "P1A analysis.md": f"{P1A}/analysis.md"}


def _norm(s):
    return s.replace("−", "-").replace("+", "").strip()


def fmt(x, f, mode):
    if f == "int":
        return str(int(round(x)))
    if f.startswith("pct"):
        x, f = 100 * x, "f" + f[3:]
    if f.startswith("g"):
        n = int(f[1:])
        if mode == "py":
            return f"{x:.{n}g}"
        e = int(np.floor(np.log10(abs(x)))) if x != 0 else 0
        return fmt(x / 10 ** e, f"f{n - 1}", mode) + (f"e{e}" if e else "")
    n = int(f[1:])
    if mode == "py":
        return f"{x:.{n}f}"
    q = Decimal(1).scaleb(-n)
    if mode == "up":
        return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))
    d1 = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-(n + 1)), rounding=ROUND_HALF_UP)
    return str(d1.quantize(q, rounding=ROUND_HALF_UP))


def eqnum(q, s):
    """compare quoted string with formatted source (numeric compare to avoid '-0.0' / 'e-04' cosmetics)."""
    try:
        return abs(float(_norm(q)) - float(_norm(s))) <= 1e-12 * max(1, abs(float(_norm(s))))
    except ValueError:
        return False


def sources():
    S = {}
    CL = pd.read_csv(f"{CDIR}/c1_predeclared_and_gt_drive.csv")
    EX = pd.read_csv(f"{CDIR}/c1_excess_effects.csv")
    RC = pd.read_csv(f"{CDIR}/c2_recovery_fractions.csv")
    PO = pd.read_csv(f"{CDIR}/c2_position_length.csv")
    PE = pd.read_csv(f"{P1A}/paired_effects.csv")
    DEC = {(d["contrast"], d["estimand"]): d for d in json.load(open(f"{RF}/P1C_new_log_replication/run_20261008/analysis/decisions.json"))["decisions"]}
    CR = pd.read_csv(f"{RF2}/CLUSTER_REANALYSIS.csv").set_index("contrast_id")
    DE = pd.read_csv(f"{RF2}/B_spatialvla_direct/run_20261013/DIRECT_EFFECTS.csv")
    OV = json.load(open(f"{POC}/outputs/cross_vla_replication/analysis.json"))
    FL = pd.read_csv(f"{P0}/POPULATION_selection_flow.csv")
    u = pd.read_parquet(f"{RF}/P0B_cluster_stats/run_20261008/units.parquet")
    e7 = u[(u.experiment == "exp07_action_history_causal") & (u.group == "A-")].groupby("condition").amplification.mean()

    def cl(cid, grp="A-", fam="G"):
        return CL[(CL.source == "primary") & (CL.family == fam) & (CL.group == grp) & CL.contrast.str.startswith(cid)].iloc[0]

    def ex(m, grp):
        return EX[(EX.family == "G") & (EX.group == grp) & (EX.metric == m)].iloc[0]

    def rc(run, cond, wt="scene", sub="all", met="amplification"):
        return RC[RC.run.str.startswith(run) & (RC.condition == cond) & (RC.weighting == wt) & (RC.metric == met) & RC.subset.str.startswith(sub)].iloc[0]

    def po(w):
        return PO[PO.analysis.str.startswith("total") & (PO.window_kind == "persistent") & (PO.contrast == "s1 - s3") & (PO.metric == "amplification") & (PO.w == w)].iloc[0]

    def de(exp, cfg, cn, task, clus="primary"):
        return DE[(DE.experiment == exp) & (DE.config == cfg) & (DE.contrast == cn) & (DE.task == task) & (DE.clustering == clus)].iloc[0]

    S.update(cl=cl, ex=ex, rc=rc, po=po, de=de, DEC=DEC, CR=CR, OV=OV, FL=FL, PE=PE, e7=e7)
    return S


def checks(S):
    cl, ex, rc, po, de, DEC, CR, OV, FL, PE, e7 = (S[k] for k in ("cl", "ex", "rc", "po", "de", "DEC", "CR", "OV", "FL", "PE", "e7"))
    p1a1, p1a2, p1a3 = cl("P1A-1"), cl("P1A-2"), cl("P1A-3")
    pe1 = PE[(PE.family == "P1A_primary") & PE.contrast.str.startswith("P1A-1")].iloc[0]
    av1, av2, av3 = CR.loc["AV-P1 Recent-Normal"], CR.loc["AV-P2 Reverse-Full"], CR.loc["AV-P3 Direction-Magnitude"]
    flow = FL.set_index("step")
    n2747 = flow.loc[[s for s in flow.index if s.startswith("1 non-CoT")][0], "n_scenes"]
    nAm = flow.loc[[s for s in flow.index if s.startswith("2A- natural A-")][0], "n_scenes"]
    P = f"{CDIR}/c1_predeclared_and_gt_drive.csv"
    X = f"{CDIR}/c1_excess_effects.csv"
    R = f"{CDIR}/c2_recovery_fractions.csv"
    D7 = f"{RF2}/B_spatialvla_direct/run_20261013/DIRECT_EFFECTS.csv"
    C = []

    def add(cid, doc, pat, quoted, fmts, slab, vals, spath):
        C.append((cid, doc, pat, quoted, fmts, slab, vals, spath))

    # ---- +4.37 family (P1-A)
    add("P1A-1 suffix drive CI", "EXPERIMENT_SUMMARY.md", r"suffix local motion \+0\.57 m/seg \[", ["0.57", "0.44", "0.68"], "f2", "c1_predeclared P1A-1 drive", [p1a1.est_scene, p1a1.ci95_lo, p1a1.ci95_hi], P)
    add("P1A-1 suffix drive CI", "PAPER_NUMBERS.md", r"\+0\.571 m/seg \[", ["0.571", "0.435", "0.682"], "f3", "c1_predeclared P1A-1 drive", [p1a1.est_scene, p1a1.ci95_lo, p1a1.ci95_hi], P)
    add("P1A-1 suffix log CI (original prereg.)", "P1A analysis.md", r"\+0\.571 m/segment \[", ["0.571", "0.430", "0.677"], "f3", "paired_effects P1A-1 log", [pe1.scene_est, pe1.scene_ci95_lo, pe1.scene_ci95_hi], f"{P1A}/paired_effects.csv")
    add("P1A-2 OC-Geo dev9_ref drive CI (predeclared estimand)", "EXPERIMENT_SUMMARY.md", r"endpoint \+4\.37 m \[\+3\.34, \+5\.18\]", ["4.37", "3.34", "5.18"], "f2", "c1_predeclared P1A-2 drive", [p1a2.est_scene, p1a2.ci95_lo, p1a2.ci95_hi], P)
    add("P1A-2 OC-Geo dev9_ref drive CI (predeclared estimand)", "BASELINE_WINDOW_RECHECK.md", r"P1A-2: \[", ["3.34", "5.18"], "f2", "c1_predeclared P1A-2 drive", [p1a2.ci95_lo, p1a2.ci95_hi], P)
    add("P1A family Holm m=3 drive (all three P1A contrasts at the 2/2^13 floor)", "EXPERIMENT_SUMMARY.md", r"Holm 7\.3e-4\)\. 원 비교", ["7.3e-4"], "g2", "c1_predeclared p_holm_m3_drive (P1A-1/2/3 identical)", [p1a2.p_holm_m3_drive], P)
    add("C1 excess model-Geo endpoint (post-hoc estimand)", "EXPERIMENT_SUMMARY.md", r"endpoint가 \+4\.37 m \[\+3\.32, \+5\.19\]", ["4.37", "3.32", "5.19"], "f2", "c1_excess ex_geo_end A-", [ex("ex_geo_end", "A-").est_scene, ex("ex_geo_end", "A-").ci95_lo, ex("ex_geo_end", "A-").ci95_hi], X)
    add("C1 excess model-Geo Holm m=4 (post-hoc family)", "EXPERIMENT_SUMMARY.md", r"Holm 9\.8e-4\)", ["9.8e-4"], "g2", "c1_excess ex_geo_end p_holm_posthoc_m4", [ex("ex_geo_end", "A-").p_holm_posthoc_m4], X)
    add("C1 excess model-Geo labelled 'Output + context vs Geo'", "PAPER_NUMBERS.md", r"endpoint \*\*\+4\.37 m \[\+3\.32, \+5\.19\]\*\*", ["4.37", "3.32", "5.19"], "f2", "c1_excess ex_geo_end A- (NOT the P1A-2 dev9_ref estimand)", [ex("ex_geo_end", "A-").est_scene, ex("ex_geo_end", "A-").ci95_lo, ex("ex_geo_end", "A-").ci95_hi], X)
    add("C1 excess model-Geo in manuscript text", "MANUSCRIPT_REVISION.md", r"4\.37 m \[3\.32, 5\.19\]", ["4.37", "3.32", "5.19"], "f2", "c1_excess ex_geo_end A-", [ex("ex_geo_end", "A-").est_scene, ex("ex_geo_end", "A-").ci95_lo, ex("ex_geo_end", "A-").ci95_hi], X)
    add("C1 excess model-Geo, CLAIM_EVIDENCE_MATRIX", "CLAIM_EVIDENCE_MATRIX.md", r"\+4\.37 m \[\+3\.32, \+5\.19\] \(Holm 9\.8e-4\)", ["4.37", "3.32", "5.19", "9.8e-4"], ["f2", "f2", "f2", "g2"], "c1_excess ex_geo_end", [ex("ex_geo_end", "A-").est_scene, ex("ex_geo_end", "A-").ci95_lo, ex("ex_geo_end", "A-").ci95_hi, ex("ex_geo_end", "A-").p_holm_posthoc_m4], X)
    add("P1A-3 OC_refR-OC drive CI", "BASELINE_WINDOW_RECHECK.md", r"P1A-3: \[", ["-4.29", "-2.59"], "f2", "c1_predeclared P1A-3", [p1a3.ci95_lo, p1a3.ci95_hi], P)
    add("P1A-3 estimate", "EXPERIMENT_SUMMARY.md", r"reference 교정 −3\.50 m", ["-3.50"], "f2", "c1_predeclared P1A-3", [p1a3.est_scene], P)
    add("model-CM endpoint A- (post-hoc)", "EXPERIMENT_SUMMARY.md", r"\+2\.35 m \[\+1\.27, \+3\.19\]", ["2.35", "1.27", "3.19"], "f2", "c1_excess ex_cm_end A-", [ex("ex_cm_end", "A-").est_scene, ex("ex_cm_end", "A-").ci95_lo, ex("ex_cm_end", "A-").ci95_hi], X)
    add("model-CM endpoint A- raw p / Holm (the 'p = 7.3e-4' near +4.37)", "PAPER_NUMBERS.md", r"\(p = 7\.3e-4, Holm 9\.8e-4\)", ["7.3e-4", "9.8e-4"], "g2", "c1_excess ex_cm_end p_signflip, p_holm_posthoc_m4", [ex("ex_cm_end", "A-").p_signflip, ex("ex_cm_end", "A-").p_holm_posthoc_m4], X)
    add("predeclared family Holm attached to C1 number", "PAPER_NUMBERS.md", r"predeclared family Holm 7\.3e-4", ["7.3e-4"], "g2", "c1_predeclared p_holm_m3_drive of P1A-2 (different estimand from the +4.37 [3.32, 5.19] it is attached to)", [p1a2.p_holm_m3_drive], P)
    add("model-CM endpoint A+ / all", "EXPERIMENT_SUMMARY.md", r"A\+ −0\.52 m \[−0\.82, −0\.26\]", ["-0.52", "-0.82", "-0.26", "0.20", "-0.32", "0.56"], "f2", "c1_excess ex_cm_end A+, all",
        [ex("ex_cm_end", "A+").est_scene, ex("ex_cm_end", "A+").ci95_lo, ex("ex_cm_end", "A+").ci95_hi, ex("ex_cm_end", "all").est_scene, ex("ex_cm_end", "all").ci95_lo, ex("ex_cm_end", "all").ci95_hi], X)
    # ---- R9 / R8 / P1-B
    add("R9 recovery w1..w4 (scene, drive)", "EXPERIMENT_SUMMARY.md", r"1/2/3/4 step = 47 / 77 / \*\*90 \[81, 99\]\*\* / 96%", ["47", "77", "90", "81", "99", "96"], "pct0", "c2_recovery R9 all A- scene",
        [rc("R9", "win1").fraction, rc("R9", "win2").fraction, rc("R9", "win3").fraction, rc("R9", "win3").ci95_lo, rc("R9", "win3").ci95_hi, rc("R9", "win4").fraction], R)
    add("R9 recovery w1..w4 with CIs", "PAPER_NUMBERS.md", r"1/2/3/4 steps = 46\.7 \[29\.6, 62\.1\]", ["46.7", "29.6", "62.1", "76.6", "62.7", "88.0", "90.1", "81.3", "99.0", "95.8", "93.3", "100"], ["pct1"] * 11 + ["pct0"], "c2_recovery R9 all A- scene",
        sum([[rc("R9", f"win{w}").fraction, rc("R9", f"win{w}").ci95_lo, rc("R9", f"win{w}").ci95_hi] for w in (1, 2, 3, 4)], []), R)
    add("R9 recovery in tables.tex Table 3", "tables.tex", r"1/2/3/4 step: 47/77/90/96", ["47", "77", "90", "96", "90", "81", "99"], "pct0", "c2_recovery R9 all A- scene",
        [rc("R9", "win1").fraction, rc("R9", "win2").fraction, rc("R9", "win3").fraction, rc("R9", "win4").fraction, rc("R9", "win3").fraction, rc("R9", "win3").ci95_lo, rc("R9", "win3").ci95_hi], R)
    add("R8 Recent-GT share (unit)", "EXPERIMENT_SUMMARY.md", r"92\.9% \[89\.4, 99\.1\]", ["92.9", "89.4", "99.1"], "pct1", "c2_recovery R8 recent_gt unit", [rc("R8", "recent_gt", "unit").fraction, rc("R8", "recent_gt", "unit").ci95_lo, rc("R8", "recent_gt", "unit").ci95_hi], R)
    add("R8 Recent-GT share (unit), Table 2 caption", "tables.tex", r"It recovers 92\.9\\% \[89\.4, 99\.1\]", ["92.9", "89.4", "99.1"], "pct1", "c2_recovery R8 recent_gt unit", [rc("R8", "recent_gt", "unit").fraction, rc("R8", "recent_gt", "unit").ci95_lo, rc("R8", "recent_gt", "unit").ci95_hi], R)
    add("R8 Recent-GT share (scene)", "PAPER_NUMBERS.md", r"scene-weighted 93\.6% \[90\.3, 99\.8\]", ["93.6", "90.3", "99.8"], "pct1", "c2_recovery R8 recent_gt scene", [rc("R8", "recent_gt").fraction, rc("R8", "recent_gt").ci95_lo, rc("R8", "recent_gt").ci95_hi], R)
    add("P1-B early-late w1..w3 (total horizon)", "EXPERIMENT_SUMMARY.md", r"\+10\.7 / \+4\.3 / \+0\.5%p", ["10.7", "4.3", "0.5"], "pct1", "c2_position total persistent s1-s3", [po(w).est_scene for w in (1, 2, 3)], f"{CDIR}/c2_position_length.csv")
    add("P1-B w2 TOST Holm", "EXPERIMENT_SUMMARY.md", r"w2는 보정 전만\(Holm 0\.064\)", ["0.064"], "f3", "c2_position w2 tost_p_equiv_holm_m3", [po(2).tost_p_equiv_holm_m3], f"{CDIR}/c2_position_length.csv")
    # ---- AutoVLA primary family / Table 2
    T2 = f"{RF2}/CLUSTER_REANALYSIS.csv"
    add("Table 2 Normal / Recent-GT unit rates (exp 7, A-)", "tables.tex", r"Normal \(own history\) & 47\.4", ["47.4", "7.1"], "pct1", "exp 7 A- unit-weighted amplification rate", [e7["normal"], e7["recent_gt"]], f"{RF}/P0B_cluster_stats/run_20261008/units.parquet")
    add("Table 2 GT-history unit rate", "tables.tex", r"GT-history \(all context := GT\) & 4\.1", ["4.1"], "pct1", "exp 7 A- gt_history", [e7["gt_history"]], f"{RF}/P0B_cluster_stats/run_20261008/units.parquet")
    add("Table 2 Recent-GT drive effect", "tables.tex", r"mathbf\{-39\.8\}\$ \[\$-49\.0\$, \$-29\.9\$\]", ["-39.8", "-49.0", "-29.9"], "pct1", "CLUSTER_REANALYSIS AV-P1 drive", [av1.drive_estimate_scene, av1.drive_boot_ci95_lo, av1.drive_boot_ci95_hi], T2)
    add("Table 2 Reverse drive effect", "tables.tex", r"mathbf\{\+28\.6\}\$ \[\$\+20\.9\$, \$\+36\.1\$\]", ["28.6", "20.9", "36.1"], "pct1", "CLUSTER_REANALYSIS AV-P2 drive", [av2.drive_estimate_scene, av2.drive_boot_ci95_lo, av2.drive_boot_ci95_hi], T2)
    add("Table 2 p_drive / Holm", "tables.tex", r"\$2\.4\\times10\^\{-4\}\$ \[\$7\.3\\times10\^\{-4\}\$\]", ["2.4e-4", "7.3e-4"], "g2", "CLUSTER_REANALYSIS AV-P1 drive p, Holm", [av1.drive_p_signflip, av1.drive_holm_p], T2)
    add("Direction-Magnitude drive", "tables.tex", r"\$\+9\.9\$ \[\$\+3\.9\$, \$\+21\.0\$\], \$p=0\.0056\$", ["9.9", "3.9", "21.0"], "pct1", "CLUSTER_REANALYSIS AV-P3 drive", [av3.drive_estimate_scene, av3.drive_boot_ci95_lo, av3.drive_boot_ci95_hi], T2)
    add("Manuscript contribution 1 numbers", "MANUSCRIPT_REVISION.md", r"lowers amplification by 39\.8 pp \[−49\.0, −29\.9\]", ["39.8", "-49.0", "-29.9", "28.6", "20.9", "36.1"], "pct1", "CLUSTER_REANALYSIS AV-P1 (|est|), AV-P2 drive",
        [-av1.drive_estimate_scene, av1.drive_boot_ci95_lo, av1.drive_boot_ci95_hi, av2.drive_estimate_scene, av2.drive_boot_ci95_lo, av2.drive_boot_ci95_hi], T2)
    # ---- population
    add("natural 2,747 / A- 52 (1.89%)", "PAPER_NUMBERS.md", r"natural 2,747 장면 중 A−", ["2747", "52", "1.89"], ["int", "int", "f2"], "POPULATION_selection_flow (dev natural)", [n2747, nAm, 100 * nAm / n2747], f"{P0}/POPULATION_selection_flow.csv")
    add("P1-C RN F / RV F", "EXPERIMENT_SUMMARY.md", r"RN F −34\.2 \[−38\.3, −29\.5\]", ["-34.2", "-38.3", "-29.5", "26.8", "21.9", "30.7"], "pct1", "decisions.json RN F, RV F",
        [DEC[("RN", "F")]["theta"], DEC[("RN", "F")]["boot_ci95_lo"], DEC[("RN", "F")]["boot_ci95_hi"], DEC[("RV", "F")]["theta"], DEC[("RV", "F")]["boot_ci95_lo"], DEC[("RV", "F")]["boot_ci95_hi"]], "decisions.json")
    add("P1-C RN/RV P (population estimand)", "PAPER_NUMBERS.md", r"RN −4\.1 pp \[−6\.4, −2\.4\], RV \+3\.6 pp \[\+2\.2, \+5\.5\]", ["-4.1", "-6.4", "-2.4", "3.6", "2.2", "5.5"], "pct1", "decisions.json RN P, RV P",
        [DEC[("RN", "P")]["theta"], DEC[("RN", "P")]["boot_ci95_lo"], DEC[("RN", "P")]["boot_ci95_hi"], DEC[("RV", "P")]["theta"], DEC[("RV", "P")]["boot_ci95_lo"], DEC[("RV", "P")]["boot_ci95_hi"]], "decisions.json")
    add("P1-C DM F / P", "EXPERIMENT_SUMMARY.md", r"DM F \+6\.2 \[\+0\.6, \+12\.1\], sign-flip p = 0\.073 / P \+2\.6, p = 0\.10", ["6.2", "0.6", "12.1", "0.073", "2.6", "0.10"], ["pct1", "pct1", "pct1", "f3", "pct1", "f2"], "decisions.json DM F, DM P",
        [DEC[("DM", "F")]["theta"], DEC[("DM", "F")]["boot_ci95_lo"], DEC[("DM", "F")]["boot_ci95_hi"], DEC[("DM", "F")]["p_signflip"], DEC[("DM", "P")]["theta"], DEC[("DM", "P")]["p_signflip"]], "decisions.json")
    # ---- SpatialVLA P1-D Table 7 and exp 35 Table 6
    add("Table 7 r4e1 dR pooled", "tables.tex", r"r4e1 & \$\\delta_R\$", ["1.7", "-8.3", "11.5", "0.8", "-8.3", "10.1", "-8.1", "10.0", "1.2", "-5.6", "8.0"], "f1", "DIRECT_EFFECTS P1D r4e1 R-N coke / move_near primary / physical / pooled primary",
        [de("P1D", "r4e1", "R-N", "coke").estimate_pp, de("P1D", "r4e1", "R-N", "coke").ci95_lo_pp, de("P1D", "r4e1", "R-N", "coke").ci95_hi_pp,
         de("P1D", "r4e1", "R-N", "move_near").estimate_pp, de("P1D", "r4e1", "R-N", "move_near").ci95_lo_pp, de("P1D", "r4e1", "R-N", "move_near").ci95_hi_pp,
         de("P1D", "r4e1", "R-N", "move_near", "physical").ci95_lo_pp, de("P1D", "r4e1", "R-N", "move_near", "physical").ci95_hi_pp,
         de("P1D", "r4e1", "R-N", "pooled").estimate_pp, de("P1D", "r4e1", "R-N", "pooled").ci95_lo_pp, de("P1D", "r4e1", "R-N", "pooled").ci95_hi_pp], D7)
    add("PAPER_NUMBERS r4e1 dR row", "PAPER_NUMBERS.md", r"\| r4e1 \| δR \| \+1\.7", ["1.7", "-8.3", "11.5", "0.8", "-8.3", "10.1", "-8.1", "10.0", "1.2", "-5.6", "8.0"], "f1", "DIRECT_EFFECTS P1D r4e1 R-N",
        [de("P1D", "r4e1", "R-N", "coke").estimate_pp, de("P1D", "r4e1", "R-N", "coke").ci95_lo_pp, de("P1D", "r4e1", "R-N", "coke").ci95_hi_pp,
         de("P1D", "r4e1", "R-N", "move_near").estimate_pp, de("P1D", "r4e1", "R-N", "move_near").ci95_lo_pp, de("P1D", "r4e1", "R-N", "move_near").ci95_hi_pp,
         de("P1D", "r4e1", "R-N", "move_near", "physical").ci95_lo_pp, de("P1D", "r4e1", "R-N", "move_near", "physical").ci95_hi_pp,
         de("P1D", "r4e1", "R-N", "pooled").estimate_pp, de("P1D", "r4e1", "R-N", "pooled").ci95_lo_pp, de("P1D", "r4e1", "R-N", "pooled").ci95_hi_pp], D7)
    add("Table 7 r4e1 dC pooled ('-0.0')", "tables.tex", r"r4e1 & \$\\delta_C\$", ["-4.2", "-14.3", "5.8", "4.2", "-5.0", "13.3", "-10.5", "18.8", "-0.0", "-6.7", "6.8"], "f1", "DIRECT_EFFECTS P1D r4e1 C-F",
        [de("P1D", "r4e1", "C-F", "coke").estimate_pp, de("P1D", "r4e1", "C-F", "coke").ci95_lo_pp, de("P1D", "r4e1", "C-F", "coke").ci95_hi_pp,
         de("P1D", "r4e1", "C-F", "move_near").estimate_pp, de("P1D", "r4e1", "C-F", "move_near").ci95_lo_pp, de("P1D", "r4e1", "C-F", "move_near").ci95_hi_pp,
         de("P1D", "r4e1", "C-F", "move_near", "physical").ci95_lo_pp, de("P1D", "r4e1", "C-F", "move_near", "physical").ci95_hi_pp,
         de("P1D", "r4e1", "C-F", "pooled").estimate_pp, de("P1D", "r4e1", "C-F", "pooled").ci95_lo_pp, de("P1D", "r4e1", "C-F", "pooled").ci95_hi_pp], D7)
    add("Table 6 exp35 clustered pooled r4e1 dR", "tables.tex", r"r4e1 \$\\delta_R\$ \$-6\.2\$ \[\$-13\.6\$, \$\+1\.1\$\]", ["-6.2", "-13.6", "1.1"], "f1", "DIRECT_EFFECTS exp35 r4e1 R-N pooled primary",
        [de("exp35", "r4e1", "R-N", "pooled").estimate_pp, de("exp35", "r4e1", "R-N", "pooled").ci95_lo_pp, de("exp35", "r4e1", "R-N", "pooled").ci95_hi_pp], D7)
    # ---- OpenVLA
    o = OV["phaseC"]["d0_|8|_feedback"]
    fc8, fc24 = OV["feedback_vs_corrected_d8"], OV["feedback_vs_corrected_d24"]
    OVP = f"{POC}/outputs/cross_vla_replication/analysis.json"
    add("OpenVLA within-step 92.6 [89.8, 94.4]", "tables.tex", r"within-step 92\.6 \[89\.8, 94\.4\]", ["92.6", "89.8", "94.4"], "pct1", "analysis.json phaseC d0_|8|_feedback", [o["amplified"]] + o["amplified_ci95"], OVP)
    add("OpenVLA within-step 92.6, 4,741 units", "PAPER_NUMBERS.md", r"\*\*92\.6% \[89\.8, 94\.4\]\*\* of 4,741", ["92.6", "89.8", "94.4", "4741"], ["pct1", "pct1", "pct1", "int"], "analysis.json phaseC", [o["amplified"]] + o["amplified_ci95"] + [o["n"]], OVP)
    add("OpenVLA feedback-corrected +3 [-4, +9], p=0.65", "tables.tex", r"feedback\$-\$corrected \$\+3\$ \[\$-4\$, \$\+9\$\], \$p=0\.65\$", ["3", "-4", "9", "0.65"], ["pct0", "pct0", "pct0", "f2"], "analysis.json feedback_vs_corrected_d8",
        [fc8["diff"]] + fc8["ci95_task_boot"] + [fc8["p_mcnemar"]], OVP)
    add("OpenVLA re-run disagreement 20%", "tables.tex", r"20\\% \(success\)", ["20"], "pct0", "analysis.json natural_rep_vs_natural discordant_share", [OV["natural_rep_vs_natural"]["discordant_share"]], OVP)
    return C


def run():
    S = sources()
    out = []
    for cid, doc, pat, quoted, fmts, slab, vals, spath in checks(S):
        txt = open(DOCS[doc], encoding="utf-8").read().split("\n")
        hit = [i + 1 for i, l in enumerate(txt) if re.search(pat, l)]
        fmts = fmts if isinstance(fmts, list) else [fmts] * len(quoted)
        assert len(fmts) == len(quoted) == len(vals), cid
        for q, f, v in zip(quoted, fmts, vals):
            py, up, dbl = fmt(v, f, "py"), fmt(v, f, "up"), fmt(v, f, "dbl") if f[0] in "fp" else ""
            if not hit:
                verdict = "NOT_FOUND"
            elif eqnum(q, py) or eqnum(q, up):
                verdict = "OK" if eqnum(q, py) and eqnum(q, up) else ("OK_PY_ROUNDING(half-even tie)" if eqnum(q, py) else "OK_HALF_UP_only")
            elif dbl and eqnum(q, dbl):
                verdict = "DOUBLE_ROUNDING"
            else:
                verdict = "MISMATCH"
            out.append({"check_id": cid, "doc": doc, "doc_path": DOCS[doc], "doc_line": ",".join(map(str, hit)) or "NOT FOUND",
                        "quoted": q, "format": f, "source": slab, "source_path": spath, "source_full_precision": float(v),
                        "source_py_format": py, "source_half_up": up, "source_double_rounded": dbl, "verdict": verdict})
    return pd.DataFrame(out)


if __name__ == "__main__":
    d = run()
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 50)
    print(d[d.verdict != "OK"][["check_id", "doc", "doc_line", "quoted", "source_full_precision", "source_py_format", "source_half_up", "verdict"]].to_string())
    print(d.verdict.value_counts())
