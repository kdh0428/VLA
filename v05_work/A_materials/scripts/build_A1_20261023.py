#!/usr/bin/env python
"""Assemble A1 driving table + FDE-threshold table from existing derived files (+ RV decomposition computed by
rv_metrics_20261023.py from raw records). No number is typed by hand except via the source files read here."""
import json, ast
import pandas as pd, numpy as np
POC = "/root/VLA/autovla_misalignment_poc/outputs"
V4 = "/root/VLA/v04_apply/work/CVPR_INTEGRATED_REVISION_20261023_v04/paper/evidence"
OUT = "/root/VLA/v05_work/A_materials"
S = {"final": f"{V4}/FINAL_PAPER_NUMBERS.csv", "ready": f"{V4}/latest_validation/PAPER_READY_NUMBERS.csv",
     "pop": f"{POC}/review_followup3/P0/POPULATION_EFFECTS.csv",
     "b2": f"{POC}/review_followup4/B/B2/p1c_analysis/MATCHED_PREFIX_RESULTS.csv",
     "p1c": f"{POC}/review_followup/P1C_new_log_replication/run_20261008/analysis/paired_effects.csv",
     "pdms": "/root/VLA/vla0_work/pkg/paper/pdms_paired_cluster_t.csv",
     "pdms_units": f"{POC}/cvpr_review_and_contribution_20261020/data/pdms_units_P1C.csv ; pdms_units_B2.csv",
     "pdms_recheck": f"{OUT}/A1_pdms_ci_recheck_20261023.csv",
     "a2": f"{POC}/cvpr_review_and_contribution_20261020/tables/A2_metric_decomposition.csv",
     "a3": f"{POC}/cvpr_review_and_contribution_20261020/tables/A3_threshold_sensitivity.csv",
     "rv": f"{OUT}/A1_rv_metric_decomposition_computed_20261023.csv",
     "r8": f"{POC}/review_followup/R8R9_raw_recovery/run_20261008/paired_effects.csv"}
F = pd.read_csv(S["final"]).set_index("claim_id"); R = pd.read_csv(S["ready"]).set_index("claim_id")
B2 = pd.read_csv(S["b2"]); P1 = pd.read_csv(S["p1c"]); PD = pd.read_csv(S["pdms"]); A2 = pd.read_csv(S["a2"]); RV = pd.read_csv(S["rv"])
R8 = pd.read_csv(S["r8"])
SIGN = {"RN": "Recent-GT minus Normal (repeated latest-position GT correction); negative severe/FDE = improvement; positive PDMS = improvement",
        "RV": "Reverse-N minus Full-GT-history (Normal-branch token reinserted into GT context); positive severe/FDE = harm; positive PDMS would be improvement",
        "C_GT": "GT_NSRC minus GT_REF (frozen Normal-source latest value vs GT, behind GT older position) = GT->Normal-source substitution; positive severe/FDE = harm; negative PDMS = harm (opposite sign convention to RN)",
        "C_NAT": "NAT_NSRC minus NAT_REF (same, behind natural older position) = GT->Normal-source substitution; positive severe/FDE = harm; negative PDMS = harm (opposite sign convention to RN)"}
POPS = [  # key, contrast, cohort label, estimand/weights, final/ready id, b2/p1c/pdms keys, a2 frame
    ("RN_devF", "RN", "development F (A- failure-selected; dev mechanism set)", "F: equal scene weights", "REP.P0A.DEV.RN.Am", None, "dev mechanism set A- (failure-selected)"),
    ("RV_devF", "RV", "development F (A- failure-selected; dev mechanism set)", "F: equal scene weights", "REP.P0A.DEV.RV.Am", None, "dev mechanism set A- (failure-selected)"),
    ("RN_newF", "RN", "new-drive P1-C F (failure stratum)", "F: equal scene weights", "REP.P0A.P1C.RN.F", None, "P1-C F"),
    ("RV_newF", "RV", "new-drive P1-C F (failure stratum)", "F: equal scene weights", "REP.P0A.P1C.RV.F", None, "P1-C F"),
    ("RN_E", "RN", "P1-C weighted eligible frame E (2,191 eligible; 353 sampled)", "E: Hajek inverse-inclusion weights", "CVPR.E_RN", None, "P1-C E (Hajek)"),
    ("RV_E", "RV", "P1-C weighted eligible frame E (2,191 eligible; 353 sampled)", "E: Hajek inverse-inclusion weights", "CVPR.E_RV", None, "P1-C E (Hajek)"),
    ("CGT_F", "C_GT", "matched-prefix F (B2, P1-C drives, a=t*+3, one older position)", "F: equal scene weights", "B2.C_GT.F.drive", ("C_GT", "F"), "B2 F"),
    ("CGT_E4", "C_GT", "matched-prefix weighted E4 (1,775 eligible; 312 sampled)", "E4: Hajek inverse-inclusion weights", "CVPR.E4_CGT", ("C_GT", "P"), "B2 E4 (Hajek)"),
    ("CNAT_F", "C_NAT", "matched-prefix F (B2, P1-C drives, a=t*+3, one older position)", "F: equal scene weights", "CVPR.F_CNAT", ("C_NAT", "F"), "B2 F"),
    ("CNAT_E4", "C_NAT", "matched-prefix weighted E4 (1,775 eligible; 312 sampled)", "E4: Hajek inverse-inclusion weights", "CVPR.E4_CNAT", ("C_NAT", "P"), "B2 E4 (Hajek)")]
DISPLAY = {"RN_devF": "shown (approximate)", "RV_devF": "shown (approximate)", "RN_newF": "shown (approximate)", "RV_newF": "shown (approximate)",
           "RN_E": "shown (approximate eligible-frame interval)", "RV_E": "WITHHELD in v04 (point only; D2 new-drive min coverage 0.886)",
           "CGT_F": "shown (approximate; not directly simulated)", "CGT_E4": "shown in supplement text only (approximate eligible-frame interval)",
           "CNAT_F": "shown (approximate)", "CNAT_E4": "WITHHELD in v04 (point only; D2 new-drive min coverage 0.890)"}
PDMSK = {"RN_newF": ("RN", "F"), "RN_E": ("RN", "E (Hajek)"), "CGT_F": ("C_GT", "F"), "CGT_E4": ("C_GT", "E4 (Hajek)"),
         "CNAT_F": ("C_NAT", "F"), "CNAT_E4": ("C_NAT", "E4 (Hajek)")}
P1FDE = {"RN_newF": ("RN", "F"), "RV_newF": ("RV", "F"), "RN_E": ("RN", "P"), "RV_E": ("RV", "P")}
rows = []
def add(key, con, coh, est, metric, horizon, val, unit, lo, hi, meth, disp, praw, padj, n_u, n_s, n_d, src, srow, note=""):
    rows.append(dict(row_id=f"{key}.{metric}", contrast=con, sign_convention=SIGN[con], cohort=coh, estimand_weights=est, metric=metric,
                     horizon=horizon, estimate=val, unit=unit, ci95_low=lo, ci95_high=hi, ci_method=meth, ci_display_status_v04=disp,
                     p_raw=praw, p_holm=padj, n_units=n_u, n_scenes=n_s, n_drives=n_d, source_file=src, source_row=srow, note=note))
for key, con, coh, est, cid, b2k, a2f in POPS:
    T = R if cid in R.index else F
    r = T.loc[cid]; srcf = S["ready"] if cid in R.index else S["final"]
    lo, hi = r.ci_low, r.ci_high
    if pd.isna(lo) and b2k:  # E4 C_GT / C_NAT interval from MATCHED_PREFIX_RESULTS
        x = B2[(B2.n_old == 1) & (B2.cluster == "drive") & (B2.subset == "all") & (B2.metric == "severe") & (B2.contrast == b2k[0]) & (B2.estimand == b2k[1])].iloc[0]
        lo, hi = [100 * v for v in ast.literal_eval(x.clt_ci95)]
    add(key, con, coh, est, "severe3 = rule-rejected high-FDE plan (A=0 and FDE_5s > 3 m)", "5 s plan (10 poses x 0.5 s)", r.estimate, "pp",
        lo, hi, "cluster-robust t (G-1 df), drive clusters; approximate" if not pd.isna(lo) else "none", DISPLAY[key], r.p_raw, r.p_adjusted,
        r.n_units, r.n_scenes, r.n_drives, srcf + f" (origin: {r.source_file})", cid)
    # PDMS
    if key in PDMSK:
        x = PD[(PD.contrast == PDMSK[key][0]) & (PD.estimand == PDMSK[key][1])].iloc[0]
        add(key, con, coh, est, "PDMS (NAVSIM v1 non-reactive, official scorer, in-memory metric cache)", "4 s", x.delta_PDMS, "score [0,1]",
            x.ci95_lo, x.ci95_hi, "cluster-robust t (G-1 df), drive clusters, ratio linearisation; approximate", "shown (approximate; E/E4 coverage not validated)",
            "NA", "NA", "NA", x.n_scenes, x.n_drives, S["pdms"] + " ; recheck " + S["pdms_recheck"] + " ; per-unit " + S["pdms_units"], f"{PDMSK[key][0]}|{PDMSK[key][1]}",
            "interval re-derived exactly from per-unit PDMS in this task (A1_pdms_ci_recheck)")
    else:
        add(key, con, coh, est, "PDMS (NAVSIM v1 non-reactive)", "4 s", "NOT_FOUND", "score [0,1]", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA",
            "NOT_FOUND", "", "PDMS was scored only for P1-C H7 normal/recent_gt and B2 cells (cvpr_review_and_contribution_20261020/data/pdms_raw.jsonl); RV and development cohorts were never scored")
    # FDE / ADE
    for m in ("fde5", "ade5"):
        if con == "RV":
            x = RV[(RV.target_population == a2f) & (RV.metric == m)].iloc[0]; val, nu, ns, nd = x.delta, x.n_units, x.n_scenes, x.n_drives
            src = S["rv"] + " (computed here from raw records with ra_a_metrics.metrics; severe3/fde5 reproduce stored values exactly)"
        else:
            x = A2[(A2.contrast == con) & (A2.target_population == a2f) & (A2.metric == m)].iloc[0]
            val, nu, ns, nd = x.delta_intervention_minus_reference, x.n_complete, x.n_scenes, x.n_clusters; src = S["a2"]
        lo = hi = "NA"; meth = "not computed"; note = ""
        if m == "fde5":
            if key in P1FDE:
                y = P1[(P1.kind == "secondary_fde5") & (P1.contrast == P1FDE[key][0]) & (P1.estimand == P1FDE[key][1])].iloc[0]
                lo, hi, meth = y.clt_ci95_lo, y.clt_ci95_hi, "cluster-robust t, drive clusters (P1-C analyze_p1c secondary_fde5); approximate, coverage not simulated"
                note = f"interval source {S['p1c']} kind=secondary_fde5; point there = {y.theta:.6f}"
            elif b2k:
                y = B2[(B2.n_old == 1) & (B2.cluster == "drive") & (B2.subset == "all") & (B2.metric == "fde5") & (B2.contrast == b2k[0]) & (B2.estimand == b2k[1])].iloc[0]
                lo, hi = ast.literal_eval(y.clt_ci95); meth = "cluster-robust t, drive clusters (MATCHED_PREFIX_RESULTS fde5); approximate, coverage not simulated"
                note = f"interval source {S['b2']}; point there = {y.theta:.6f}"
            elif key == "RV_devF" or key == "RN_devF":
                lab = "reverse@emb - gt_history" if con == "RV" else "recent_gt - normal"
                y = R8[(R8.family == "R8_secondary_A-") & (R8.contrast == lab) & (R8.metric == "fde5")].iloc[0]
                note = (f"R8 rerun reports scene_est {y.scene_est:.4f} with interval [{y.scene_ci95_lo:.3f},{y.scene_ci95_hi:.3f}] over {int(y.n_clusters)} LOG clusters "
                        f"({S['r8']}); not a drive-cluster interval" + ("; R8 recent_gt-normal is a rerun distinct from the exp-7 records behind the paper's RN" if con == "RN" else ""))
        add(key, con, coh, est, m.upper().replace("5", "_5s") + " (m, L2 to GT)", "5 s", val, "m", lo, hi, meth,
            "decomposition table (Supp. Tab. tab:decomp) shows points only" if m == "fde5" else "points only", "NA", "NA", nu, ns, nd, src, f"{con}|{a2f}|{m}", note)
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/A1_driving_table_20261023.csv", index=False)
# threshold sensitivity
A3 = pd.read_csv(S["a3"]); th = []
for key, con, coh, est, cid, b2k, a2f in POPS:
    for tau in (2, 3, 4, 5):
        if con == "RV":
            sv = RV[(RV.target_population == a2f) & (RV.metric == f"severe{tau}")].iloc[0].delta
            fg = RV[(RV.target_population == a2f) & (RV.metric == f"fde_gt{tau}")].iloc[0].delta; src = S["rv"]
            n = RV[(RV.target_population == a2f)].iloc[0]; nu, ns, nd = n.n_units, n.n_scenes, n.n_drives
        else:
            q = A3[(A3.contrast == con) & (A3.target_population == a2f)]
            sv = q[q.metric == f"severe{tau}"].iloc[0].delta_intervention_minus_reference
            fg = q[q.metric == f"fde_gt{tau}"].iloc[0].delta_intervention_minus_reference; src = S["a3"]
            nu, ns, nd = q.iloc[0].n_complete, q.iloc[0].n_scenes, q.iloc[0].n_clusters
        th.append(dict(contrast=con, cohort=coh, estimand_weights=est, tau_m=tau, severe_tau_pp=sv, fde_gt_tau_only_pp=fg,
                       primary=(tau == 3), n_units=nu, n_scenes=ns, n_drives=nd, interval="not computed (descriptive)",
                       sign_convention=SIGN[con].split(";")[0], source_file=src))
pd.DataFrame(th).to_csv(f"{OUT}/A1_fde_threshold_20261023.csv", index=False)
json.dump({"files": S, "scripts": {"rv": f"{OUT}/scripts/rv_metrics_20261023.py", "pdms_recheck": f"{OUT}/scripts/pdms_ci_check_20261023.py",
           "build": f"{OUT}/scripts/build_A1_20261023.py"}}, open(f"{OUT}/_build_paths.json", "w"), indent=1)
print(T[["row_id", "estimate", "ci95_low", "ci95_high", "n_scenes"]].to_string()); print(pd.DataFrame(th)[["contrast", "cohort", "tau_m", "severe_tau_pp", "fde_gt_tau_only_pp"]].to_string())
