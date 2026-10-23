#!/usr/bin/env python
"""CVPR next validation: FINAL_INTERVAL_POLICY.csv from SIM_RESULTS.csv (this round) and the author-query coverage check
(outputs/manuscript_author_queries_20261019/COVERAGE_SUMMARY.csv). Decision rule from STATISTICAL_CHECK_PLAN.md §8.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python policy.py
"""
import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
OUT = f"{POC}/outputs/cvpr_next_validation_20261019"
d = pd.read_csv(f"{OUT}/SIM_RESULTS.csv")
prev = pd.read_csv(f"{POC}/outputs/manuscript_author_queries_20261019/COVERAGE_SUMMARY.csv")
EST = {"E_RN": ("P1-C E (Hajek; 2,191 frame, 353 sampled, 24 drives)", "Recent-GT - Normal, amplification"),
       "E_RV": ("P1-C E", "Reverse-N - GT-history, amplification"),
       "E4_CNAT": ("B2 E4 (Hajek; 1,775 frame, 312 sampled, 24 drives)", "NAT_NSRC - NAT_REF, severe"),
       "E4_CGT": ("B2 E4", "GT_NSRC - GT_REF, severe"), "E4_INT": ("B2 E4", "INT = C_GT - C_NAT (unit-level), severe"),
       "F_RN": ("P1-C F (105 scenes, 18 drives)", "Recent-GT - Normal"), "F_CNAT": ("B2 F (100 scenes, 18 drives)", "NAT_NSRC - NAT_REF"),
       "F_INT": ("B2 F", "INT (unit-level)")}
rows = []
for cid, (coh, con) in EST.items():
    for meth in ("cluster_t_95", "drive_bootstrap_95"):
        x = d[(d.contrast == cid) & (d.method == meth)]
        m3, m2 = x[x.design == "D3"], x[x.design == "D2"]
        allx = pd.concat([m3, m2])
        adequate = bool((allx.mc_wilson95_hi >= 0.94).all())
        mn = allx.coverage.min()
        dec = "ADEQUATE_IN_CHECKED_SCENARIOS" if adequate else ("APPROXIMATE_ONLY" if mn >= 0.90 else "WITHHOLD")
        d3s = f"D3 finite-population min {m3.coverage.min():.3f} / max {m3.coverage.max():.3f}" if len(m3) else "D3 not run (F census)"
        d2s = f"D2 superpopulation (new drives) min {m2.coverage.min():.3f} / max {m2.coverage.max():.3f}"
        if meth == "cluster_t_95":
            if dec == "WITHHOLD" and len(m3) and m3.coverage.min() >= 0.90:
                wording = ("finite-population (eligible-frame) cluster-t interval may be shown as approximate; no interval is claimed "
                           "for new drives (superpopulation coverage below 0.90 in at least one checked scenario)")
            elif dec == "WITHHOLD":
                wording = "withhold the interval; report estimate and the drive sign-flip p"
            elif dec == "APPROXIMATE_ONLY":
                wording = "report the cluster-t 95% interval labelled approximate (simulated coverage roughly 0.90-0.95 for nominal 0.95)"
            else:
                wording = "report the cluster-t 95% interval; coverage near nominal in the checked scenarios (not a guarantee)"
        else:
            wording = "do not use as the primary interval (lower coverage than cluster-t in every contrast)"
        rows.append({"contrast_id": cid, "cohort_estimand": coh, "contrast": con, "interval_method": meth,
                     "evidence_scope": "post-hoc simulation, plan locked (STATISTICAL_CHECK_PLAN.md): pseudo-population built from the real frame, weights, unit counts and drives; within-unit label swaps (q = 0.25/0.5/0.75/1); couplings U/S; " + ("D3 + D2" if len(m3) else "D2"),
                     "actual_coverage": f"{d3s}; {d2s}", "MC_uncertainty": f"R = {int(allx.reps.iloc[0])} per scenario; MC SE <= {allx.mc_se.max():.4f}; min Wilson upper {allx.mc_wilson95_hi.min():.3f}",
                     "decision": dec, "recommended_wording": wording,
                     "limitations": "non-sampled S outcomes imputed by cycling the sampled S scenes of the log; new-drive variation bounded by the 24 (18) observed drives; drive clusters only (day-vehicle / component dependence not modelled)"})
for _, r in prev.iterrows():
    if r.method != "cluster_t_95":
        continue
    rows.append({"contrast_id": r.contrast_id, "cohort_estimand": "failure stratum (fixed scenes)", "contrast": "see author-query COVERAGE_STRUCTURES.csv",
                 "interval_method": r.method, "evidence_scope": "author-query check 2026-10-19 (fixed scenes, contrast-specific reference arm, 2 couplings, +-MME / observed)",
                 "actual_coverage": f"min {r.min_cov:.3f} / max {r.max_cov:.3f}", "MC_uncertainty": f"R = 5,000; min Wilson upper {r.min_mc_hi:.3f}",
                 "decision": "APPROXIMATE_ONLY", "recommended_wording": "cluster-t 95% interval labelled approximate",
                 "limitations": "fixed-scene (conditional) check; superpopulation checked separately only for F_RN / F_CNAT / F_INT"})
rows.append({"contrast_id": "dev A- RV, dev DM, P1-C F RV/DM (superpop), B2 C_PROV", "cohort_estimand": "various", "contrast": "-", "interval_method": "cluster_t_95",
             "evidence_scope": "not simulated directly", "actual_coverage": "NA", "MC_uncertainty": "NA", "decision": "APPROXIMATE_ONLY (by analogy)",
             "recommended_wording": "label approximate; do not call verified", "limitations": "no direct simulation"})
rows.append({"contrast_id": "day-vehicle / component clustering", "cohort_estimand": "B2, P1-C", "contrast": "-", "interval_method": "none",
             "evidence_scope": "G = 6-16 clusters", "actual_coverage": "NA", "MC_uncertainty": "NA", "decision": "WITHHOLD",
             "recommended_wording": "report p and attainable floor only", "limitations": "-"})
pd.DataFrame(rows).to_csv(f"{OUT}/FINAL_INTERVAL_POLICY.csv", index=False)
print(pd.DataFrame(rows)[["contrast_id", "interval_method", "decision", "actual_coverage"]].to_string())
