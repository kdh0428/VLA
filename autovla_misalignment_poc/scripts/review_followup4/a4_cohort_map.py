#!/usr/bin/env python
"""
A4 (review_followup4, 2026-10-17): cohort / estimand map and wording-violation list.

*** Documentation audit. CPU only. Reads source CSVs / code / documents; modifies nothing. ***
Outputs (outputs/review_followup4/A1_A4/):
  COHORT_ESTIMAND_MAP.csv     every headline number -> its set (cohort), denominators, estimand, outcome definition,
                              horizon, source; numbers are re-read from the source tables and asserted
  A4_WORDING_VIOLATIONS.csv   document file:line, rule violated, current text (verified to be on that line),
                              proposed replacement (documents are NOT edited)

  nice -n 19 /root/miniforge3/envs/autovla/bin/python a4_cohort_map.py
"""
from __future__ import annotations

import csv
import os

import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
P0 = f"{POC}/outputs/review_followup3/P0"
P2 = f"{POC}/outputs/review_followup3/P2"
OUT = f"{POC}/outputs/review_followup4/A1_A4"
A1 = f"{OUT}/CI_VALIDITY_AUDIT.csv"


def flow(cohort, step_prefix):
    f = pd.read_csv(f"{P0}/POPULATION_selection_flow.csv")
    r = f[(f.cohort.str.startswith(cohort)) & (f.step.str.startswith(step_prefix))]
    assert len(r) == 1, (cohort, step_prefix, len(r))
    return r.iloc[0]


def build_map():
    rows = []

    def add(**k):
        rows.append(k)
    dev = flow("dev natural", "1 non-CoT")
    am = flow("dev natural", "2A- natural A- (all")
    sev = flow("dev natural", "2A-s")
    mech = flow("dev natural", "5 selected total")
    ap = flow("dev natural", "5 selected A+")
    st1 = flow("P1-C stage 1", "0 stage-1")
    E = flow("P1-C stage 1", "3 eligible E")
    F = flow("P1-C stage 1", "3F eligible")
    S = flow("P1-C stage 1", "3S eligible")
    sel = flow("P1-C stage 1", "4 selected total")
    sS = flow("P1-C stage 1", "4S selected")
    assert (dev.n_scenes, am.n_scenes, sev.n_scenes, mech.n_scenes, ap.n_scenes) == (2747, 52, 40, 208, 156)
    assert (E.n_scenes, F.n_scenes, sel.n_scenes, F.n_drives, sel.n_drives, st1.n_scenes) == (2191, 105, 353, 18, 24, 5671)
    den = pd.read_csv(f"{P0}/POPULATION_p1c_denominators.csv").set_index(["contrast", "estimand"])
    assert den.loc[("RN", "F"), "n_units"] == 748 and den.loc[("RN", "P"), "n_units"] == 2436
    dec = pd.read_csv(f"{P0}/POPULATION_decomposition.csv").set_index("contrast")
    sh = {c: (dec.loc[c, "F_contribution_to_theta_P"] / dec.loc[c, "theta_P"],
              dec.loc[c, "S_contribution_to_theta_P"] / dec.loc[c, "theta_P"]) for c in dec.index}
    assert abs(sh["RN"][1] - 0.6026) < 1e-3
    mb = pd.read_csv(f"{P2}/MOTION_BASELINE_RESULTS.csv")
    gs = mb[mb.family == "G"].groupby("group").n_scenes.max()
    assert (gs["all"], gs["A-"], gs["A+"]) == (208, 52, 156)
    pc = pd.read_csv(f"{P0}/OPENVLA_REANALYSIS_phaseC.csv").set_index("key").loc["d0_|8|_feedback"]
    u = pd.read_csv(f"{P0}/OPENVLA_phaseC_feedback_units.csv")
    g = u[(u.d == 0) & (u.abs_delta == 8)]
    k = g[~g.clipped_out]
    n_partial = int((k.delta_eff_bins < 8).sum())
    tr_bins = float((k[["dbin_y", "dbin_z"]].max(1) >= k.delta_eff_bins).mean())
    rot_bins = float((k[["dbin_roll", "dbin_pitch", "dbin_yaw"]].max(1) >= k.delta_eff_bins).mean())
    assert int(pc.n_units) == 4741 and abs(pc.amplified_rate - 0.925754) < 1e-6
    phys = pd.read_csv(f"{P0}/OPENVLA_physical_units.csv").set_index("dim")
    het = pd.read_csv(f"{P0}/SPATIAL_TASK_EFFECTS_heterogeneity.csv").set_index("row_id")
    h1 = het.loc["P0B.het.P1D.r4e1.C-F.primary"]
    h2 = het.loc["P0B.het.exp35.r1e4.R-N.primary"]
    S_ = "outputs/review_followup3/P0/POPULATION_selection_flow.csv"
    common_def = ("A- = NOT pra_labels.label_A(pred, GT) on the 10-pose (5 s) plan [analyze_action_history.a_eval, "
                  "_pos_to_delta(traj, 10)]; amplification/severe = A- AND FDE5 > 3.0 m (AMP_FDE)")
    # ------------------------------------------------------------------ cohorts
    add(id="SET-01", item="dev natural set", value=f"{dev.n_scenes} scenes", cohort="dev natural (full_extract arm N, navtest shards 0-5)",
        n_scenes=dev.n_scenes, n_units="1 natural plan per scene", n_logs=dev.n_logs, n_drives=dev.n_drives,
        n_day_vehicles=dev.n_day_vehicles, estimand_weighting="unweighted scene frequency",
        outcome_definition=common_def, horizon="5 s (10 poses)", source=S_,
        use_rule="the only set for natural (evaluation-set) frequencies of the dev logs; NOT a deployment / natural failure rate",
        where_used="EXPERIMENT_SUMMARY.md:413-418; tables.tex:198 (Table 8 caption); MANUSCRIPT_REVISION.md:87 (K2)")
    add(id="SET-02", item="dev mechanism set (equal-distance / selected)", value=f"{mech.n_scenes} selected scenes = {am.n_scenes} A- (all, pi=1) + {ap.n_scenes} A+ (3 per A- at the same t*, Random(0))",
        cohort="dev mechanism set", n_scenes=mech.n_scenes, n_units="A- stratum: 365 perturbation units (52 scenes)",
        n_logs=mech.n_logs, n_drives=mech.n_drives, n_day_vehicles=mech.n_day_vehicles,
        estimand_weighting="selected; A- share 52/208 = 25% vs 1.89% natural -> F/A+ ratio differs from the natural mix",
        outcome_definition=common_def, horizon="5 s", source=S_,
        use_rule="mechanism contrasts (exps 6-11, 32, R8R9, P1-A) are conditional on the selected A- (52 scenes / 13 drives) or A+ strata; never present as natural-set or population rates",
        where_used="tables.tex:37 (Table 2, A- 52/365/13); EXPERIMENT_SUMMARY.md:25, 125-150")
    add(id="SET-03", item="P1-C stage-1 natural pass", value=f"{st1.n_scenes} scenes", cohort="P1-C 64 eval logs, stage 1",
        n_scenes=st1.n_scenes, n_units="1 natural plan", n_logs=st1.n_logs, n_drives=st1.n_drives,
        n_day_vehicles=st1.n_day_vehicles, estimand_weighting="unweighted", outcome_definition=common_def, horizon="5 s",
        source=S_, use_rule="A- 107/5,671 = 1.89% is an evaluation-set frequency", where_used="tables.tex:198; EXPERIMENT_SUMMARY.md:419")
    add(id="SET-04", item="P1-C target E (eligible frame)", value=f"{E.n_scenes} eligible scenes (t* <= 7)", cohort="P1-C target E",
        n_scenes=E.n_scenes, n_units="-", n_logs=E.n_logs, n_drives=E.n_drives, n_day_vehicles=E.n_day_vehicles,
        estimand_weighting="target of the scene-weighted Hajek estimand P (weights reproduce 2,191)",
        outcome_definition=common_def, horizon="5 s", source=S_,
        use_rule="name: 'evaluation-population (E) estimand'; A- 105/2,191 = 4.79% is the eligible-frame frequency; not deployment",
        where_used="tables.tex:198; MANUSCRIPT_REVISION.md:87; EXPERIMENT_SUMMARY.md:420-423")
    add(id="SET-05", item="P1-C intervention sample", value=f"{sel.n_scenes} selected scenes (105 F + {sS.n_scenes} S, m = 4 per segment), 2,436 units",
        cohort="P1-C intervention sample", n_scenes=sel.n_scenes, n_units=int(den.loc[("RN", "P"), "n_units"]),
        n_logs=sel.n_logs, n_drives=sel.n_drives, n_day_vehicles=sel.n_day_vehicles,
        estimand_weighting="F weight 1; S weight N_S,seg / m_seg (Hajek) -> estimates the 2,191-scene E estimand",
        outcome_definition=common_def, horizon="5 s", source="outputs/review_followup3/P0/POPULATION_p1c_denominators.csv",
        use_rule="P effects (-4.1 / +3.6 pp) are weighted estimates from 353 sampled scenes, 24 drives (21 informative for RN)",
        where_used="tables.tex:210-212 (Table 8 rows, '353 / 2436')")
    add(id="SET-06", item="P1-C failure stratum F", value=f"{F.n_scenes} scenes / 748 units / {F.n_logs} logs / {F.n_drives} drives",
        cohort="P1-C failure stratum F (all eligible A-, perturbed)", n_scenes=F.n_scenes, n_units=748, n_logs=F.n_logs,
        n_drives=F.n_drives, n_day_vehicles=F.n_day_vehicles, estimand_weighting="scene-weighted mean over F (weight 1)",
        outcome_definition=common_def, horizon="5 s", source=S_,
        use_rule="F effects (-34.2 / +26.8 / +6.2 pp) are on 18 drives (6 of the 24 P1-C drives have no F scene); never '24 (independent) failure drives'",
        where_used="tables.tex:51, 207-209")
    add(id="SET-07", item="P1-C S frame / sample", value=f"{S.n_scenes} eligible A+; {sS.n_scenes} sampled", cohort="P1-C S stratum",
        n_scenes=f"{S.n_scenes} frame / {sS.n_scenes} sample", n_units=1688, n_logs=S.n_logs, n_drives=S.n_drives,
        n_day_vehicles=S.n_day_vehicles, estimand_weighting="Hajek", outcome_definition=common_def, horizon="5 s",
        source=S_, use_rule="contributes ~60% of RN P (see NUM-07)", where_used="POPULATION_EFFECTS.md")
    add(id="SET-08", item="motion-baseline 'all'", value=f"{gs['all']} selected scenes = A- {gs['A-']} + A+ {gs['A+']}",
        cohort="dev mechanism set (= SET-02), P1-A rollouts", n_scenes=int(gs["all"]), n_units="P1-A units",
        n_logs="26", n_drives="19 (A-: 13)", n_day_vehicles="13", estimand_weighting="scene-weighted over the SELECTED mix (25% A-)",
        outcome_definition="Delta_model - Delta_control, M1 endpoint displacement at 5 s (m)", horizon="5 s",
        source="outputs/review_followup3/P2/MOTION_BASELINE_RESULTS.csv (family G, group all)",
        use_rule="'all' = 208 selected scenes, NOT the 2,747 natural / 2,191 eligible scenes or a population; P2 family/column label 'E' = learned extrapolator, unrelated to P1-C target E",
        where_used="tables.tex:230 (Table 9 row 'all'); EXPERIMENT_SUMMARY.md:34, 361, 488; MANUSCRIPT_REVISION.md:18, 97")
    # ------------------------------------------------------------------ numbers
    add(id="NUM-01", item="RN dev A-", value="-39.8 pp [-49.0, -29.9], p = 2.4e-4", cohort="SET-02 A- stratum",
        n_scenes=52, n_units=365, n_logs=16, n_drives=13, n_day_vehicles=10, estimand_weighting="scene-weighted",
        outcome_definition="amplification (A- AND FDE5 > 3 m)", horizon="5 s", source="numbers.json P0A.DEV.RN.Am",
        use_rule="failure-stratum conditional effect", where_used="tables.tex:46, 204; MANUSCRIPT_REVISION.md:11, 17, 83")
    add(id="NUM-02", item="RN / RV P1-C F", value="RN -34.2 [-38.3, -29.5]; RV +26.8 [+21.9, +30.7]", cohort="SET-06",
        n_scenes=105, n_units=748, n_logs=41, n_drives=18, n_day_vehicles=10, estimand_weighting="scene-weighted (F)",
        outcome_definition="amplification", horizon="5 s", source="numbers.json P0A.P1C.RN.F / RV.F",
        use_rule="'replicated on new drives: failure stratum 105 scenes / 18 drives (0 drives, 10 day-vehicles shared with dev)'",
        where_used="tables.tex:51, 207-208; MANUSCRIPT_REVISION.md:17, 83; CLAIM_EVIDENCE_MATRIX.md:47")
    add(id="NUM-03", item="RN / RV P1-C P (evaluation-population)", value="RN -4.1 [-6.4, -2.4]; RV +3.6 [+2.2, +5.5]",
        cohort="SET-04 estimand estimated from SET-05", n_scenes="2,191 target / 353 sampled", n_units=2436, n_logs=64,
        n_drives=24, n_day_vehicles=16, estimand_weighting="Hajek scene-weighted", outcome_definition="amplification", horizon="5 s",
        source="numbers.json P0A.P1C.RN.P / RV.P", use_rule="'evaluation-population (E) estimand'; not population/deployment",
        where_used="tables.tex:210-211; MANUSCRIPT_REVISION.md:17, 87")
    add(id="NUM-07", item="P = F + S decomposition", value=(f"RN: F {100 * sh['RN'][0]:.1f}% / S (A+) {100 * sh['RN'][1]:.1f}%; "
                                                         f"RV: F {100 * sh['RV'][0]:.1f}% / S {100 * sh['RV'][1]:.1f}%; "
                                                         f"DM: F {100 * sh['DM'][0]:.1f}% / S {100 * sh['DM'][1]:.1f}%"),
        cohort="SET-04/05", n_scenes="105 F (weight 105) + 248 S (weight 2,086)", n_units=2436, n_logs=64, n_drives=24,
        n_day_vehicles=16, estimand_weighting="Hajek", outcome_definition="amplification", horizon="5 s",
        source="outputs/review_followup3/P0/POPULATION_decomposition.csv",
        use_rule="keep 'about 60% of the RN evaluation-population effect comes from the A+ (non-failure) stratum'; do NOT describe -4.1 pp as recovery of originally failing scenes",
        where_used="EXPERIMENT_SUMMARY.md:423; PAPER_NUMBERS.md:379")
    add(id="NUM-08", item="A- frequency dev", value=f"52/2,747 = 1.89% [1.16, 2.64] (drive clusters)", cohort="SET-01",
        n_scenes=2747, n_units="-", n_logs=28, n_drives=20, n_day_vehicles=14, estimand_weighting="unweighted",
        outcome_definition="A-", horizon="5 s", source="outputs/review_followup3/P0/POPULATION_prevalence.csv",
        use_rule="evaluation-set frequency, not deployment / natural failure rate; severe 40/2,747 = 1.46%",
        where_used="tables.tex:198; MANUSCRIPT_REVISION.md:87, 155")
    add(id="NUM-09", item="exp 15 / 18-19 'population' A- (re-weighted)", value="1.24 -> 2.60% (3080 Ti), 1.31 -> 2.27% (5090)",
        cohort="A- 52 + random A+ 297, re-weighted to the dev composition 52 / 2,695", n_scenes="349 run, weighted to 2,747",
        n_units="-", n_logs=28, n_drives=20, n_day_vehicles=14, estimand_weighting="N_FAIL=52, N_TOTAL=2747 (analyze_natural_reference_stabilization.py:27)",
        outcome_definition="A-", horizon="5 s", source="scripts/analyze_natural_reference_stabilization.py",
        use_rule="'dev evaluation-set re-weighted A- frequency', not population/deployment",
        where_used="EXPERIMENT_SUMMARY.md:157; PAPER_EVIDENCE.md:802-806, 817-819, 887; SOURCE_MAP.md:287, 291")
    add(id="NUM-11", item="motion baseline model - CM", value="A- +2.35 [+1.27, +3.19]; A+ -0.52; all +0.20 [-0.32/-0.33, +0.56] p = 0.51",
        cohort="SET-08", n_scenes="52 / 156 / 208", n_units="P1-A units", n_logs="16 / - / 26", n_drives="13 / 19 / 19",
        n_day_vehicles="10 / - / 13", estimand_weighting="scene-weighted", outcome_definition="M1 endpoint displacement change (m)",
        horizon="5 s", source="numbers.json P2A.excess.G.*", use_rule="'all' = 208 selected; n.s. = no difference detected, equivalence not tested",
        where_used="tables.tex:228-230")
    add(id="NUM-12", item="+4.37 m GT-free endpoint", value="+4.37 m [+3.34, +5.17] (P1A-2) / [+3.32, +5.19] (post-hoc C1)",
        cohort="SET-02 A- stratum", n_scenes=52, n_units=364, n_logs=16, n_drives=13, n_day_vehicles=10, estimand_weighting="scene-weighted",
        outcome_definition="dev9_ref: endpoint (pose index 9 = 5 s) distance to own unperturbed reference, OC - Geo", horizon="5 s",
        source="numbers.json P0C.plus437.AUTHORITATIVE", use_rule="A- stratum only", where_used="MANUSCRIPT_REVISION.md:18, 91")
    add(id="NUM-13", item="R9 correction-length 90%", value="47 / 77 / 90 [81, 99] / 96% (scene-weighted)", cohort="SET-02 A- stratum",
        n_scenes=52, n_units=365, n_logs=16, n_drives=13, n_day_vehicles=10, estimand_weighting="scene-weighted (unit-weighted 47/76/90/96 separate)",
        outcome_definition="(Normal - win w) / (Normal - GT-history) amplification; persistent fixed-position GT window",
        horizon="5 s", source="numbers.json P0C.R9.*", use_rule="see A2 for window semantics", where_used="MANUSCRIPT_REVISION.md:101")
    add(id="NUM-14", item="OpenVLA 92.6% within-step", value=(f"92.6% (task CI [89.8, 94.4]; state CI [90.8, 94.0]); 4,741 units = 2,462 frames x sign; "
                                                           f"183 fully clipped units dropped; {n_partial} units partially clipped (effective shift < 8 bins)"),
        cohort="OpenVLA phase C offline decode of the 100 natural phase-A episodes (every 5th step)", n_scenes="100 initial states / 10 tasks",
        n_units=4741, n_logs="-", n_drives="-", n_day_vehicles="-", estimand_weighting="unit mean (equal-state 92.6%)",
        outcome_definition=("amplified = max over downstream dims y, z, roll, pitch, yaw (gripper excluded) of |bin - natural bin| >= effective injected "
                            "|x shift| in bins (after clipping to [0, 254]); i.e. AT LEAST ONE later component; bins mix m and rad"),
        horizon="1 control step (within-step decode)", source="outputs/review_followup3/P0/OPENVLA_REANALYSIS_phaseC.csv; p0_e_openvla.py",
        use_rule=(f"not 'all later dimensions', not task failure; translation-only (y,z) share {100 * tr_bins:.1f}% (bins) / "
                  f"{100 * pc.phys_amp_translation_rate:.1f}% (commanded m); rotation-only {100 * rot_bins:.1f}% (bins)"),
        where_used="tables.tex:69; MANUSCRIPT_REVISION.md:121; EXPERIMENT_SUMMARY.md:36, 88, 211, 464")
    add(id="NUM-15", item="OpenVLA physical units", value=(f"8 bins: x {1000 * phys.loc['x', 'delta8_physical']:.2f} mm, y {1000 * phys.loc['y', 'delta8_physical']:.2f} mm, "
                                                        f"z {1000 * phys.loc['z', 'delta8_physical']:.2f} mm; roll {1000 * phys.loc['roll', 'delta8_physical']:.2f} mrad, "
                                                        f"pitch {1000 * phys.loc['pitch', 'delta8_physical']:.2f} mrad, yaw {1000 * phys.loc['yaw', 'delta8_physical']:.2f} mrad; "
                                                        f"gripper binarised (bin change {100 * pc.gripper_bin_changed_rate:.1f}%)"),
        cohort="OpenVLA LIBERO-Spatial norm stats (q01/q99) x OSC_POSE scaling", n_scenes="-", n_units="-", n_logs="-", n_drives="-",
        n_day_vehicles="-", estimand_weighting="-", outcome_definition="COMMANDED per-step delta (OSC input), not realised TCP motion",
        horizon="1 step", source="outputs/review_followup3/P0/OPENVLA_physical_units.csv",
        use_rule="report translation (m), rotation (rad) and gripper (binary) separately; no TCP was recorded for OpenVLA phase C",
        where_used="EXPERIMENT_SUMMARY.md:465")
    add(id="NUM-16", item="SpatialVLA trajectory metrics", value="P1-D R-N actual TCP 2.67 / 5.95 cm (rerun 1.24 / 2.00); exp35 +1.84 cm actual TCP vs +0.116 m commanded; exp34 0.21-0.23 m commanded only",
        cohort="P1-D (restore verified) / exp35 / exp34", n_scenes="240 units per config", n_units="-", n_logs="-", n_drives="-",
        n_day_vehicles="-", estimand_weighting="episode mean", outcome_definition="actual TCP distance vs cumulative commanded world_vector (not TCP)",
        horizon="episode", source="PAPER_NUMBERS.md Claims 10, 17, 18",
        use_rule="label every metre-scale number as 'commanded'; cm-scale TCP numbers as 'actual TCP'", where_used="tables.tex:163, 185; EXP35_RESULTS.md:36, 56-59")
    add(id="NUM-17", item="SpatialVLA per-task opposite point estimates", value=(f"P1-D r4e1 C-F coke {h1.coke_pp:+.1f} vs move_near {h1.move_near_pp:+.1f}; "
                                                                               f"het {h1.heterogeneity_coke_minus_move_near_pp:+.1f} [{h1.het_boot_ci95_lo_pp:+.1f}, {h1.het_boot_ci95_hi_pp:+.1f}], p = {h1.het_welch_p:.2f}; "
                                                                               f"exp35 r1e4 R-N {h2.coke_pp:+.1f} vs {h2.move_near_pp:+.1f}, het CI [{h2.het_boot_ci95_lo_pp:+.1f}, {h2.het_boot_ci95_hi_pp:+.1f}]; exp34: heterogeneity not tested"),
        cohort="SpatialVLA closed loop, primary clusters", n_scenes="120 + 120 units", n_units="-", n_logs="-", n_drives="-",
        n_day_vehicles="-", estimand_weighting="task-equal", outcome_definition="success difference",
        horizon="episode", source="outputs/review_followup3/P0/SPATIAL_TASK_EFFECTS_heterogeneity.csv",
        use_rule="opposite point estimates are not established heterogeneity or cancellation (all het CIs include 0); 'cancellation share' is arithmetic on point estimates",
        where_used="EXPERIMENT_SUMMARY.md:280, 472, 575; PAPER_NUMBERS.md:197, 328; CLAIM_EVIDENCE_MATRIX.md:70")
    add(id="NUM-18", item="drive-disjoint vs day/vehicle-disjoint", value="P1-C: 0 drives shared with the dev mechanism set, 10 day-vehicles shared, 6 connected day-vehicle components (exact Holm floor 0.094); 0 day-and-vehicle-disjoint logs exist; exp 21 held-out shares 22 drives (85% of scenes); navhard halves share 16 drives",
        cohort="P1-C / held-out / navhard", n_scenes="-", n_units="-", n_logs="64 / 52 / 76", n_drives="24 / 28 / -", n_day_vehicles="16",
        estimand_weighting="-", outcome_definition="-", horizon="-", source="outputs/review_followup3/P0/DEPENDENCE_*.csv, DISJOINT_LOG_INVENTORY*",
        use_rule="P1-C = 'new drives (days and vehicles partly shared)'; never 'external' or 'day/vehicle-independent'; held-out = 'log-disjoint, not drive-disjoint'",
        where_used="MANUSCRIPT_REVISION.md:11, 58, 148; tables.tex:51")
    # ------------------------------------------------------------------ definitions
    add(id="DEF-01", item="A- (failure) / A+ (recovery)", value="A- = NOT label_A", cohort="all AutoVLA cohorts",
        n_scenes="-", n_units="-", n_logs="-", n_drives="-", n_day_vehicles="-", estimand_weighting="-",
        outcome_definition=("label_A = lon_ok AND lat_ok (pra_comparison/pra_labels.py:469). Kinematics from vla_misalignment_poc/analysis/trajectory_metrics.py: "
                            "DT = 0.5 s; v_start = mean speed of first 2 steps, v_end = mean of last 2 steps. Lon class: STOP if total path < 1.0 m AND v_end < 0.5 m/s; "
                            "if v_start < 0.5 m/s: ACCELERATE iff v_end - v_start > 0.8 m/s else MAINTAIN; else r = v_end/v_start >= 1.25 ACCELERATE, <= 0.75 DECELERATE "
                            "(STOP if v_end < 0.5), else MAINTAIN. Pred lon class (computed with the GT start speed) must be in the GT tolerance set: GT v_end +- max(0.5 m/s, 15% x max(v_start, v_end)) (11-point grid). "
                            "Lat class: LEFT/RIGHT if |final lateral offset| > 2.0 m or |final-step heading| > 15 deg (heading ignored if last step < 0.25 m), else STRAIGHT; "
                            "pred lat class must be in the GT tolerance set (+-1.0 m, +-10 deg). Invalid/short plan -> A- (P1-C: amplification 0, recovery 0)"),
        horizon=("applied to the 10-pose / 5 s plan (_pos_to_delta(traj, 10) in analyze_action_history.a_eval) although trajectory_metrics.py documents "
                 "these thresholds for a 3 s horizon (HORIZON = 6, 'total path length below this over 3 s'); run_pra_comparison.py has a 3 s sensitivity (horizon 6) only for the P/R/A comparison"),
        source="pra_comparison/pra_labels.py; vla_misalignment_poc/analysis/trajectory_metrics.py:20-35; scripts/analyze_action_history.py:51-54",
        use_rule="state the formula, thresholds and the 3 s-documented / 5 s-applied fact wherever A- is defined", where_used="tables.tex:37, 62; PAPER_EVIDENCE.md:23; SOURCE_MAP.md:272; EXPERIMENT_SUMMARY.md:296, 567")
    add(id="DEF-02", item="amplification = severe", value="A- AND FDE5 > 3.0 m", cohort="AutoVLA / Impromptu",
        n_scenes="-", n_units="-", n_logs="-", n_drives="-", n_day_vehicles="-", estimand_weighting="-",
        outcome_definition="FDE5 = ||pred pose 10 - GT pose 10|| (5 s endpoint); AMP_FDE = 3.0 (analyze_action_history.py:45); 'severe' in POPULATION_* = the same formula on the natural plan",
        horizon="5 s", source="scripts/analyze_action_history.py:45-63; review_followup/p1c/analyze_p1c.py:9",
        use_rule="say 'amplification (A- and FDE5 > 3 m)'; dev severe 40/2,747", where_used="tables.tex:37, 62")
    add(id="DEF-03", item="GT-free severe_ref{2,3,4}", value="dev9_ref > 2 / 3 / 4 m", cohort="P1-A / P1-B (A- stratum)",
        n_scenes="-", n_units="-", n_logs="-", n_drives="-", n_day_vehicles="-", estimand_weighting="-",
        outcome_definition="dev9_ref = distance at pose index 9 (5 s) from the model's own unperturbed reference rollout (not GT)",
        horizon="5 s", source="scripts/review_followup/p1ab/p1a_analyze.py:22, 138", use_rule="never mix with GT-based severe/amplification", where_used="P1-A/P1-B tables")
    return rows


VIOL = [
    # file, line, snippet (must be on that line), rule, issue, replacement
    ("/root/VLA/EXPERIMENT_SUMMARY.md", 11, "p·CI는 cluster sign-flip(역산 CI)", "A1 CI validity",
     "sign-flip inversion CI is invalid for binary success outcomes (zero-width / end pinned at 0)",
     "p는 cluster sign-flip(δ = 0, 유효), CI는 사후 수정(Rao–Scott design-effect Newcombe 10; 과제 평균은 MOVER) — review_followup4 A1"),
    ("/root/VLA/EXPERIMENT_SUMMARY.md", 26, "새 24 drive**(P1-C)에서 재현: 실패 조건부 RN −34.2%p", "set/denominator; F = 105 scenes / 18 drives; evaluation-population",
     "failure-conditional effects are on 18 drives, not 24; '모집단' is the evaluation frame",
     "새 drive(P1-C; 0 drive 공유, day–vehicle 10 공유)에서 재현: 실패 층 F(105 장면 / 18 drive) RN −34.2%p, RV +26.8%p; 평가 집합 E 추정량(적격 2,191 장면, 선정 353 장면 가중, 24 drive) −4.1 / +3.6%p — RN의 약 60%는 A+ 층 기여"),
    ("/root/VLA/EXPERIMENT_SUMMARY.md", 34, "전체에서는 +0.20 m(p = 0.51)로 외삽과 구별되지 않습니다", "motion-baseline all = 208 selected; non-significance != equivalence",
     "'전체' = 208 selected mechanism scenes (25% A-), not all scenes",
     "선정 기전 집합 전체(208 장면 = A− 52 + A+ 156)에서는 +0.20 m [−0.32, +0.56](p = 0.51)로 차이가 검출되지 않았습니다(동등성 미검정)"),
    ("/root/VLA/EXPERIMENT_SUMMARY.md", 36, "나머지 차원이 93%에서 그 이상 움직이고", "OpenVLA 92.6% exact definition; units",
     "implies all remaining dimensions; bins not physical units",
     "x token을 ±8 bin(명령 약 2.6 mm/step) 바꾸면 단위의 92.6%(4,741 단위; 초기 상태 군집 CI [90.8, 94.0])에서 하위 성분(y, z, roll, pitch, yaw; gripper 제외) 중 적어도 하나의 bin 변화가 실제 주입 bin 이상이었고(translation 성분만으로는 72%)"),
    ("/root/VLA/EXPERIMENT_SUMMARY.md", 38, "closed-loop 기저 변동(자연 재실행끼리도 결과 20% 불일치) 안에 있었습니다", "non-significance != equivalence; 20% rerun disagreement is not a detection threshold",
     "uses rerun disagreement as a yardstick for 'no effect'",
     "LIBERO 과제 성공률 효과는 검출되지 않았고(C − F 8-bin −3%p [−11.9, +5.9], 동등성 미성립), 자연 재실행끼리도 결과가 20% 달랐습니다"),
    ("/root/VLA/EXPERIMENT_SUMMARY.md", 88, "하위 차원 93% 증폭", "OpenVLA 92.6% exact definition", "vague 'downstream dims 93% amplified'",
     "x token ±8 bin 교란 → 하위 성분 중 적어도 하나가 주입 bin 이상 변한 단위 92.6%"),
    ("/root/VLA/EXPERIMENT_SUMMARY.md", 157, "모집단 A− 1.24 → 2.60%로 악화", "evaluation set vs population",
     "re-weighted to the dev composition 52/2,695 -> dev evaluation-set frequency",
     "dev 자연 평가 집합(2,747 장면: A− 52 / A+ 2,695)으로 재가중한 A− 빈도 1.24 → 2.60%로 악화(배포 빈도 아님)"),
    ("/root/VLA/EXPERIMENT_SUMMARY.md", 211, "하위 차원이 δ 이상 움직이는 비율 **92.6% [89.8, 94.4]**", "OpenVLA 92.6% exact definition; units",
     "'하위 차원이' without 'at least one'; frames vs units",
     "하위 성분(y..yaw; gripper 제외) 중 적어도 하나의 |bin 변화| ≥ 실제 주입 bin(clip 후)인 단위 비율 92.6% (4,741 단위 = 2,462 프레임 × 부호; task CI [89.8, 94.4], 초기 상태 CI [90.8, 94.0]); bin은 m·rad 혼합"),
    ("/root/VLA/EXPERIMENT_SUMMARY.md", 280, "pooled 평균에서 상쇄됩니다", "opposite point estimates != heterogeneity",
     "asserts cancellation from opposite-signed, non-significant task point estimates",
     "과제별 점추정의 부호가 반대(δR −10.0 vs +7.5%p, δC +7.5 vs −10.0%p)지만 과제 간 차이는 검정하지 않았고 과제별 CI가 넓어, 이질성이나 상쇄가 확인된 것은 아닙니다"),
    ("/root/VLA/EXPERIMENT_SUMMARY.md", 361, "전체 +0.20 m [−0.32, +0.56] (p = 0.51)", "motion-baseline all = 208 selected",
     "'전체' undefined", "선정 208 장면 전체(A− 52 + A+ 156) +0.20 m [−0.32, +0.56] (p = 0.51; 동등성 미검정)"),
    ("/root/VLA/EXPERIMENT_SUMMARY.md", 364, "RN F −34.2 [−38.3, −29.5] / P −4.1", "F = 105 scenes / 18 drives; P = 353 weighted",
     "row header '64 log / 24 drive' attached to F effects",
     "RN F(105 장면 / 18 drive) −34.2 [−38.3, −29.5] / P(E 2,191 적격 장면, 353 선정 장면 가중, 24 drive) −4.1"),
    ("/root/VLA/EXPERIMENT_SUMMARY.md", 421, "**모집단 추정량.**", "evaluation-population", "'모집단' label",
     "**평가 집합(E) 추정량.**"),
    ("/root/VLA/EXPERIMENT_SUMMARY.md", 472, "과제 효과가 반대 부호로 평균에서 100% 상쇄됩니다", "opposite point estimates != heterogeneity",
     "cancellation share is arithmetic on point estimates; heterogeneity CI includes 0",
     "P1-D r4e1 C−F의 과제별 점추정은 −4.2 / +4.2%p로 부호가 반대지만, 과제 간 차이 −8.3%p [−21.7, +5.0](p = 0.23)는 0을 포함하므로 이질성·상쇄는 확인되지 않았습니다"),
    ("/root/VLA/EXPERIMENT_SUMMARY.md", 488, "| 전체 | CM +0.20 (n.s.)", "motion-baseline all = 208 selected", "'전체' undefined",
     "| 선정 208 장면 전체 (A− 52 + A+ 156) | CM +0.20 (n.s.; 동등성 미검정)"),
    ("/root/VLA/EXPERIMENT_SUMMARY.md", 490, "A+와 전체에서는 운동 외삽과 구별되지 않습니다", "all = 208; non-significance != equivalence",
     "'구별되지 않는다' + undefined '전체'", "A+와 선정 208 장면 전체에서는 운동 외삽과의 차이가 검출되지 않았습니다(동등성 미검정)"),
    ("/root/VLA/EXPERIMENT_SUMMARY.md", 542, "새 24 drive에서 재현", "F = 18 drives", "failure-stratum replication is on 18 drives",
     "새 drive의 실패 층(105 장면 / 18 drive)에서 재현"),
    ("/root/VLA/EXPERIMENT_SUMMARY.md", 567, "평가 horizon은 5 s(10 token)이며", "A-/severe exact formula; 3 s thresholds applied at 5 s",
     "A- thresholds documented for 3 s are applied to the 5 s plan; not stated anywhere",
     "평가 horizon은 5 s(10 token)입니다. A−는 P/R/A 행동 범주 판정(pra_labels.label_A: 종방향 v_end/v_start 1.25/0.75, 저속 0.5 m/s·가속 0.8 m/s, STOP = 경로 < 1.0 m & v_end < 0.5 m/s, 허용 max(0.5 m/s, 15%); 횡방향 최종 횡변위 > 2.0 m 또는 마지막 step heading > 15°, 허용 ±1 m/±10°)이고, 증폭(severe) = A− ∧ FDE5 > 3 m입니다. 이 임계값은 3 s horizon(HORIZON = 6)용으로 문서화되었지만 10 pose(5 s) 계획에 그대로 적용됐습니다"),
    ("/root/VLA/EXPERIMENT_SUMMARY.md", 575, "과제별로 반대 부호 효과가 일부 상쇄됩니다", "opposite point estimates != heterogeneity",
     "asserts partial cancellation", "과제별 점추정의 부호가 일부 반대지만 과제 간 이질성은 검출되지도 배제되지도 않았습니다"),
    ("/root/VLA/EXPERIMENT_SUMMARY.md", 47, "과제별로는 CI 폭 최대 18–29%p(P1-D r4e1)로 불확실", "A1 CI validity",
     "task CI widths come from the invalid sign-flip inversion (r1e4 6.9 pp lower bound is a tie artefact)",
     "과제별로는 CI 폭 최대 18–25%p(P1-D r4e1; r1e4 13–20%p; A1 사후 수정 구간)로 불확실"),
    ("/root/VLA/paper_quantitative_package/MANUSCRIPT_REVISION.md", 11, "Both effects replicated in a preregistered test on 24 new drives that share no drive with the development set", "F = 105 scenes / 18 drives; drive- vs day/vehicle-disjoint; cohort of the 40 pp",
     "40 pp is the A- stratum effect (52 scenes); replication is in the P1-C failure stratum on 18 drives; days/vehicles shared",
     "'...reduced trajectory amplification in failure-stratum (A−) scenes by 40 pp (52 scenes, 13 drives; ...)' and 'Both effects replicated in a preregistered test on new drives (failure stratum: 105 scenes, 18 drives; no drive shared with the development set, but days and vehicles partly shared)'"),
    ("/root/VLA/paper_quantitative_package/MANUSCRIPT_REVISION.md", 17, "A preregistered replication on 24 new drives (day/vehicle partly shared) reproduces both contrasts (failure-conditional −34.2 / +26.8 pp; population −4.1 / +3.6 pp)", "F = 18 drives; evaluation-population; 60% A+",
     "F on 18 drives; 'population' is the E estimand",
     "A preregistered replication on new drives (days/vehicles partly shared) reproduces both contrasts: failure stratum (105 scenes, 18 drives) −34.2 / +26.8 pp; evaluation-frame estimand over 2,191 eligible scenes (353 sampled, weighted; 24 drives) −4.1 / +3.6 pp, about 60% of the RN effect coming from the non-failure (A+) stratum"),
    ("/root/VLA/paper_quantitative_package/MANUSCRIPT_REVISION.md", 18, "across all scenes it is not distinguishable from extrapolation (+0.20 m, p = 0.51)", "motion-baseline all = 208; non-significance != equivalence",
     "'all scenes' = 208 selected mechanism scenes", "across the 208 selected mechanism scenes (52 A− + 156 A+; not the natural mix) no difference from extrapolation was detected (+0.20 m [−0.32, +0.56], p = 0.51; equivalence not tested)"),
    ("/root/VLA/paper_quantitative_package/MANUSCRIPT_REVISION.md", 49, "across all scenes it was not distinguishable from extrapolation", "motion-baseline all = 208", "same",
     "...; across the 208 selected mechanism scenes no difference from extrapolation was detected (equivalence not tested)."),
    ("/root/VLA/paper_quantitative_package/MANUSCRIPT_REVISION.md", 58, "\"replicated on 24 new drives (day/vehicle partly shared)\"", "F = 18 drives", "failure-stratum replication counted on 24 drives",
     "\"replicated on new drives (failure stratum 105 scenes / 18 drives; evaluation frame 24 drives; days and vehicles partly shared)\""),
    ("/root/VLA/paper_quantitative_package/MANUSCRIPT_REVISION.md", 83, "both effects replicated in a preregistered test on 24 new drives (−34.2 and +26.8 pp)", "F = 18 drives", "K1 attaches 24 drives to F effects",
     "both effects replicated in a preregistered test on new drives (failure stratum, 105 scenes / 18 drives; −34.2 and +26.8 pp; days and vehicles partly shared)"),
    ("/root/VLA/paper_quantitative_package/MANUSCRIPT_REVISION.md", 87, "On the evaluation population of the new-drive logs (2,191 eligible scenes)", "P1-C intervention sample 353 weighted; 60% A+",
     "omits that P is estimated from 353 sampled scenes and that ~60% comes from A+",
     "On the evaluation frame of the new-drive logs (2,191 eligible scenes; estimated from 353 sampled scenes with design weights, 24 drives), the corresponding effects were −4.1 pp [−6.4, −2.4] and +3.6 pp [+2.2, +5.5] (about 60% of the RN effect from non-failure scenes);"),
    ("/root/VLA/paper_quantitative_package/MANUSCRIPT_REVISION.md", 97, "across all scenes and in non-failure scenes it was not distinguishable from motion extrapolation", "all = 208; non-significance",
     "'all scenes' = 208 selected", "in non-failure scenes and across the 208 selected mechanism scenes no excess over motion extrapolation was detected (equivalence not tested)"),
    ("/root/VLA/paper_quantitative_package/MANUSCRIPT_REVISION.md", 121, "moved a later dimension by at least 8 bins in 92.6% of units", "OpenVLA 92.6% exact definition; units",
     "threshold is the effective injected shift (30 units < 8 bins after clipping); bins mix m and rad",
     "an 8-bin x-token shift (≈ 2.6 mm commanded) changed at least one later component (y, z, roll, pitch, yaw; gripper excluded) by at least the effective injected number of bins in 92.6% of 4,741 units (initial-state-cluster CI [90.8, 94.0]; translation components alone 72%)"),
    ("/root/VLA/paper_quantitative_package/MANUSCRIPT_REVISION.md", 148, "\"replicated on 24 new drives (days and vehicles partly shared)", "F = 18 drives", "same as line 58",
     "\"replicated on new drives (failure stratum 105 scenes / 18 drives; days and vehicles partly shared); no external (day/vehicle-disjoint) validation was possible\""),
    ("/root/VLA/paper_quantitative_package/tables.tex", 37, "Amplification = A$-$ and FDE$_{5\\,s}>3$\\,m.", "A-/severe exact formula; 3 s thresholds at 5 s",
     "A- rule and horizon caveat missing",
     "Amplification = A$-$ and FDE$_{5\\,s}>3$\\,m, where A$-$ = failure of the coarse action-category rule (longitudinal class from start/end speed with tolerance $\\max(0.5$\\,m/s$, 15\\%)$; lateral class from final offset $>2$\\,m or final heading $>15^\\circ$, tolerance $\\pm1$\\,m/$\\pm10^\\circ$) applied to the 5\\,s plan with thresholds originally specified for a 3\\,s horizon."),
    ("/root/VLA/paper_quantitative_package/tables.tex", 51, "0 drives shared with this set", "drive- vs day/vehicle-disjoint; F cohort", "day-vehicle sharing and F scene count missing",
     "New drives (P1-C, pre-registered; failure stratum 105 scenes / 18 drives; 0 drives but 10 day--vehicle combinations shared with this set): ..."),
    ("/root/VLA/paper_quantitative_package/tables.tex", 69, "Normal amplification & 47.4 [38, 57] & 63.4 [60.6, 66.7] & within-step 92.6", "OpenVLA 92.6% exact definition",
     "92.6% is not a Normal-condition rate; it is the perturbed within-step dependency share",
     "OpenVLA cell: 'perturbed within-step dependency: $\\ge$1 later component (y..yaw, gripper excl.) shifted $\\ge$ injected bins, 92.6 (state CI [90.8, 94.0]); bins, not physical units' (or move it to its own row)"),
    ("/root/VLA/paper_quantitative_package/tables.tex", 174, "pp with sign-flip-inversion 95\\% CI on initial-state clu", "A1 CI validity",
     "sign-flip-inversion CIs invalid for binary outcomes (e.g. move_near physical r1e4 delta_C [-2.6, 4.3] -> [-6.9, +8.6])",
     "pp with post-hoc corrected 95% CI (Rao--Scott design-effect Newcombe; MOVER for the two-task mean) on initial-state clusters; original sign-flip-inversion CIs in the supplement"),
    ("/root/VLA/paper_quantitative_package/tables.tex", 187, "task-level CIs span up to 18--29 pp (r4e1; r1e4 6.9--21.6 pp)", "A1 CI validity",
     "widths from invalid intervals", "task-level CIs span up to 18--25 pp (r4e1; r1e4 13--20 pp; post-hoc corrected)"),
    ("/root/VLA/paper_quantitative_package/tables.tex", 230, "all & +0.20 [$-$0.33, +0.56]", "motion-baseline all = 208 selected",
     "row 'all' undefined in caption", "row label 'all (208 selected = 52 A$-$ + 156 A+)'; caption: '\\emph{all} is the selected mechanism set, not the natural set'; also rename column 'E' (learned extrapolator) to avoid confusion with P1-C target E"),
    ("/root/VLA/paper_quantitative_package/tables.tex", 240, "pp, sign-flip-inversion 95\\% CI on initial-state clusters", "A1 CI validity",
     "same as Table 7", "pp, post-hoc corrected 95% CI (Rao--Scott design-effect Newcombe) on initial-state clusters"),
    ("/root/VLA/paper_quantitative_package/PAPER_NUMBERS.md", 85, "population estimand P (24 drives, Hajek-weighted)", "evaluation-population; 353 sampled",
     "'population'", "evaluation-frame estimand P (2,191 eligible scenes; 353 sampled, Hajek-weighted; 24 drives)"),
    ("/root/VLA/paper_quantitative_package/PAPER_NUMBERS.md", 91, "(13 drives; replicated on 24 new drives)", "F = 18 drives", "F replication counted on 24 drives",
     "(A− stratum, 52 scenes / 13 drives; replicated in the P1-C failure stratum, 105 scenes / 18 new drives)"),
    ("/root/VLA/paper_quantitative_package/PAPER_NUMBERS.md", 180, "amplifies a downstream dimension in **92.6% [89.8, 94.4]**", "OpenVLA 92.6% exact definition",
     "'amplifies' + threshold/units unstated", "changes at least one downstream component (y..yaw; gripper excluded) by ≥ the effective injected bin shift in 92.6% [89.8, 94.4] (bins; translation-only 72%)"),
    ("/root/VLA/paper_quantitative_package/PAPER_NUMBERS.md", 185, "no task-level effect distinguishable from re-run variability", "non-significance != equivalence",
     "uses rerun disagreement as the yardstick", "no task-level effect was detected (C−F −3 pp [−11.9, +5.9]; equivalence not established), while re-running alone changed 20% of outcomes"),
    ("/root/VLA/paper_quantitative_package/PAPER_NUMBERS.md", 197, "opposite-signed effects that cancel in the pooled mean", "opposite point estimates != heterogeneity",
     "asserts cancellation", "opposite-signed task point estimates (heterogeneity not tested; task CIs wide), so the pooled mean may hide task differences"),
    ("/root/VLA/paper_quantitative_package/PAPER_NUMBERS.md", 220, "OpenVLA shows no task effect distinguishable from rerun variability", "non-significance != equivalence",
     "same as line 185", "OpenVLA showed no detectable task effect (equivalence not established)"),
    ("/root/VLA/paper_quantitative_package/PAPER_NUMBERS.md", 328, "coke S3 −9.2 vs move_near +8.3 partly cancel", "opposite point estimates != heterogeneity",
     "asserts partial cancellation", "coke S3 −9.2 vs move_near +8.3 (opposite point estimates; heterogeneity not established)"),
    ("/root/VLA/paper_quantitative_package/PAPER_NUMBERS.md", 350, "**all scenes +0.20 m [−0.32, +0.56], p = 0.51**", "motion-baseline all = 208",
     "'all scenes' = 208 selected", "**all 208 selected mechanism scenes (52 A− + 156 A+) +0.20 m [−0.32, +0.56], p = 0.51**"),
    ("/root/VLA/paper_quantitative_package/PAPER_NUMBERS.md", 356, "across all scenes it was not distinguishable from extrapolation", "motion-baseline all = 208",
     "same", "across the 208 selected mechanism scenes no difference from extrapolation was detected (equivalence not tested)"),
    ("/root/VLA/paper_quantitative_package/PAPER_NUMBERS.md", 405, "across all scenes and in non-failure scenes it was not disting", "motion-baseline all = 208",
     "same", "in non-failure scenes and across the 208 selected mechanism scenes no excess was detected"),
    ("/root/VLA/paper_quantitative_package/PAPER_EVIDENCE.md", 558, "하위 차원 증폭이 92.6%", "OpenVLA 92.6% exact definition", "vague",
     "하위 성분 중 적어도 하나가 주입 bin 이상 변한 단위가 92.6%"),
    ("/root/VLA/paper_quantitative_package/PAPER_EVIDENCE.md", 716, "closed-loop task 효과가 없다는 것은", "non-significance != equivalence", "'효과가 없다'",
     "closed-loop task 효과가 검출되지 않았다는 것은(동등성 미검정)"),
    ("/root/VLA/paper_quantitative_package/PAPER_EVIDENCE.md", 802, "모집단 A− 1.24 → 2.60%로 악화", "evaluation set vs population", "re-weighted dev frequency",
     "dev 평가 집합(52/2,695) 재가중 A− 1.24 → 2.60%로 악화"),
    ("/root/VLA/paper_quantitative_package/PAPER_EVIDENCE.md", 817, "모집단 가중 52/2,695", "evaluation set vs population", "label", "dev 평가 집합 가중 52/2,695"),
    ("/root/VLA/paper_quantitative_package/EXP35_RESULTS.md", 18, "task 수준에서는 교정 효과가 없고", "non-significance != equivalence", "'no correction effect'",
     "task 수준에서는 교정 효과가 검출되지 않았고(동등성 미검정; 군집 후 불확실)"),
    ("/root/VLA/paper_quantitative_package/EXP35_RESULTS.md", 56, "궤적 발산", "commanded delta vs actual TCP", "column 'trajectory divergence' is cumulative commanded action",
     "'실행 명령 누적 끝점 거리 (commanded, TCP 아님)'"),
    ("/root/VLA/paper_quantitative_package/EXP35_RESULTS.md", 36, "trajectory_divergence F 대 C / R 대 N (m)", "commanded delta vs actual TCP", "same",
     "commanded_cumulative_divergence F 대 C / R 대 N (m; TCP 아님)"),
    ("/root/VLA/paper_quantitative_package/CONSISTENCY_AUDIT.md", 47, "결론(문맥 효과 0, 실행 효과 있음)", "non-significance != equivalence; heterogeneity from per-task significance",
     "'context effect 0'; 'concentrated in coke' from one significant / one non-significant task",
     "결론(문맥 효과 검출 안 됨, 실행 효과 있음). 과제별 점추정 차이(coke −37.5 vs move_near −2.5%p)는 이질성 검정 없이 '집중'으로 쓰지 않음"),
    ("/root/VLA/paper_quantitative_package/CONSISTENCY_AUDIT.md", 75, "→ 효과 없음 ✔", "non-significance != equivalence", "'효과 없음'",
     "→ 효과 검출 안 됨(동등성 미검정)"),
    ("/root/VLA/paper_quantitative_package/CONSISTENCY_AUDIT.md", 457, "단 A−에서는 등속 외삽 baseline과 구별되지 않는다", "stale; contradicts review_followup3 P2",
     "withdrawn comparison", "단 등속 외삽 초과는 A−(실패 선택 층)에서만 성립한다(P2)"),
    ("/root/VLA/paper_quantitative_package/SOURCE_MAP.md", 291, "상시 적용 모집단 1.24 → 2.60%", "evaluation set vs population", "label",
     "상시 적용 dev 평가 집합 재가중 A− 1.24 → 2.60%"),
    ("/root/VLA/review_followup3/CLAIM_EVIDENCE_MATRIX.md", 16, "| VERIFIED_MACHINERY_MANIFEST_SCOPE | usable with the manifest-scope caveat |", "verified path != statistical validity",
     "status could be read as validating the estimate / CI", "| VERIFIED_MACHINERY_MANIFEST_SCOPE | intervention code path verified on the manifest; NOT a certification of the estimand or interval (see review_followup4 A1) |"),
    ("/root/VLA/review_followup3/CLAIM_EVIDENCE_MATRIX.md", 47, "On 24 new drives (preregistered) the reduction was −34.2 pp", "F = 18 drives", "F counted on 24 drives",
     "In the P1-C failure stratum (preregistered; 105 scenes, 18 new drives) the reduction was −34.2 pp"),
    ("/root/VLA/review_followup3/CLAIM_EVIDENCE_MATRIX.md", 49, "On the P1-C evaluation frame (2,191 eligible scenes, 24 drives)", "intervention sample 353 weighted",
     "sample size behind the estimate missing", "On the P1-C evaluation frame (2,191 eligible scenes; 353 sampled, design-weighted; 24 drives)"),
    ("/root/VLA/review_followup3/CLAIM_EVIDENCE_MATRIX.md", 56, "Across A+ scenes and all scenes it was not distinguishable from motion extrapolation", "all = 208; non-significance",
     "'all scenes' = 208 selected", "Across A+ scenes and the 208 selected mechanism scenes no excess over motion extrapolation was detected (equivalence not tested)"),
    ("/root/VLA/review_followup3/CLAIM_EVIDENCE_MATRIX.md", 70, "Opposite signs cancel up to 100 %", "opposite point estimates != heterogeneity",
     "cancellation share on point estimates", "Opposite-signed task point estimates (P1-D r4e1 C−F −4.2 / +4.2 pp; heterogeneity −8.3 pp [−21.7, +5.0]); heterogeneity neither detected nor excluded"),
    ("/root/VLA/review_followup3/CLAIM_EVIDENCE_MATRIX.md", 75, "moved a later dimension by ≥ 8 bins in 92.6 % of units", "OpenVLA 92.6% exact definition",
     "threshold = effective injected shift", "changed at least one later component (y..yaw; gripper excluded) by ≥ the effective injected shift (8 bins unless clipped) in 92.6 % of units"),
    ("/root/VLA/review_followup3/CP2_REPORT.md", 45, "In A+ and over all scenes it is not distinguished", "all = 208; non-significance", "'all scenes' = 208 selected",
     "In A+ and over the 208 selected mechanism scenes no excess was detected (equivalence not tested)."),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = build_map()
    cols = ["id", "item", "value", "cohort", "n_scenes", "n_units", "n_logs", "n_drives", "n_day_vehicles",
            "estimand_weighting", "outcome_definition", "horizon", "use_rule", "where_used", "source"]
    pd.DataFrame(rows)[cols].to_csv(f"{OUT}/COHORT_ESTIMAND_MAP.csv", index=False)
    out = []
    for f, ln, snip, rule, issue, rep in VIOL:
        lines = open(f, encoding="utf-8").read().split("\n")
        line = lines[ln - 1]
        ok = snip in line
        if not ok:
            raise AssertionError(f"snippet not found at {f}:{ln}: {snip!r}")
        out.append({"file": f, "line": ln, "rule": rule, "issue": issue, "current_text": snip, "proposed_replacement": rep,
                    "verified_on_line": ok, "edited": False})
    pd.DataFrame(out).to_csv(f"{OUT}/A4_WORDING_VIOLATIONS.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"map rows {len(rows)}, violations {len(out)}")


if __name__ == "__main__":
    main()
