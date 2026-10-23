#!/usr/bin/env python
"""Author query Q1: CI_RECOMMENDATION.csv from COVERAGE_STRUCTURES / COVERAGE_SUMMARY (this round) and the review_followup5
coverage study (A_METHOD_SELECTION.json). No new estimation.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python q1_recommendation.py
"""
import json
import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
OUT = f"{POC}/outputs/manuscript_author_queries_20261019"
st = pd.read_csv(f"{OUT}/COVERAGE_STRUCTURES.csv").set_index("contrast_id")
sm = pd.read_csv(f"{OUT}/COVERAGE_SUMMARY.csv")
rf5 = json.load(open(f"{POC}/outputs/review_followup5/A_STATS/A_METHOD_SELECTION.json"))["selection"]


def mc(cid, lev):
    x = sm[(sm.contrast_id == cid) & (sm.method == f"cluster_t_{lev}")].iloc[0]
    return f"min coverage {x.min_cov:.3f} (MC SE ~0.0036; Wilson upper {x.min_mc_hi:.3f}), max {x.max_cov:.3f}, 16 scenarios x 5,000 reps"


rows = []
for cid, coh, est in (("DEV_RN", "dev A- (exp 7)", "A-"), ("P1C_F_RN", "P1-C", "F"), ("P1C_F_RV", "P1-C", "F"), ("B2_F_CNAT", "P1-C B2", "F")):
    s = st.loc[cid]
    common = dict(contrast_id=cid, cohort=coh, estimand=est, reference_arm=s.reference_arm, unit="paired unit -> scene mean",
                  n_scenes=int(s.n_scenes), n_drives=int(s.n_drives), weighting="scene-weighted (weights 1 in this stratum)")
    rows.append({**common, "interval_method": "cluster-robust t 95% (G-1 df)",
                 "evidence_scope": "post-hoc contrast-specific simulation (this round): real scene/drive structure, real reference-arm rates and discordance; two coupling models (empirical, independence), +-MME and observed-scale effects, drive heterogeneity 0 / 0.05",
                 "mc_uncertainty": mc(cid, 95), "recommended_use": "primary",
                 "limitation": "approximate, not guaranteed: near-null homogeneous scenarios cover ~0.93 (< 0.95); fixed-scene simulation, between-scene sampling only through drive heterogeneity; observed-scale constant shifts not reachable (realised theta smaller), empirical-cell alternative covers ~0.99"})
    rows.append({**common, "interval_method": "cluster-robust t 90% (G-1 df)", "evidence_scope": "same", "mc_uncertainty": mc(cid, 90),
                 "recommended_use": "sensitivity", "limitation": "min coverage ~0.875 (< 0.90); use only for margin checks with this caveat"})
    rows.append({**common, "interval_method": "drive percentile bootstrap 95%", "evidence_scope": "review_followup5 study (Normal-arm model, R = 400)",
                 "mc_uncertainty": "R = 400 -> MC SE ~0.013", "recommended_use": "descriptive",
                 "limitation": "under-covered (min 0.91 P1-C F, 0.908 dev A-) in the review_followup5 study; not re-run here"})
for cid, coh, est, ref in (("DEV_RV", "dev A- (R8)", "A-", "gt_history"), ("DEV_DM", "dev A- (exp 32)", "A-", "dir_ok_mag_wrong"),
                           ("P1C_F_DM", "P1-C", "F", "dir_ok_mag_wrong"), ("B2_F_CGT", "P1-C B2", "F", "GT_REF"), ("B2_F_CPROV", "P1-C B2", "F", "GT_NSRC")):
    rows.append(dict(contrast_id=cid, cohort=coh, estimand=est, reference_arm=ref, unit="paired unit -> scene mean", n_scenes=None, n_drives=None,
                     weighting="scene-weighted", interval_method="cluster-robust t 95%", evidence_scope="same cluster structure as a checked contrast; not itself simulated",
                     mc_uncertainty="NA", recommended_use="sensitivity", limitation="by analogy only; report with the approximate label"))
rows.append(dict(contrast_id="B2_F_INT", cohort="P1-C B2", estimand="F", reference_arm="C_NAT (difference of contrasts)", unit="unit-level difference of two paired contrasts",
                 n_scenes=100, n_drives=18, weighting="scene-weighted", interval_method="cluster-robust t 95%", evidence_scope="not simulated",
                 mc_uncertainty="NA", recommended_use="sensitivity", limitation="joint-cell covariance of the two paired contrasts not modelled; never read as equality"))
for cid, coh, est in (("P1C_P_RN/RV/DM", "P1-C evaluation frame", "P (Hajek, 353 sampled / 2,191)"), ("B2_E4_all", "P1-C B2", "E4 (Hajek)"),
                      ("P1C_S", "P1-C", "S"), ("dayvehicle/component/subsets", "dev, P1-C", "any")):
    rows.append(dict(contrast_id=cid, cohort=coh, estimand=est, reference_arm="various", unit="paired unit -> scene mean", n_scenes=None,
                     n_drives=None, weighting="design-weighted / various", interval_method="none", evidence_scope="review_followup5 study: no method reached the historical threshold (P: cluster-t min 0.922, bootstrap 0.900; day-vehicle 0.908; G<=7 0.902; R = 400)",
                     mc_uncertainty="R = 400 -> MC SE ~0.013 (classification near 0.925 not robust)", recommended_use="NR",
                     limitation="report estimate and sign-flip p only; not re-checked here (CI not forced)"))
rows.append(dict(contrast_id="dev A+ / all / t*-reweighted", cohort="dev", estimand="A+, all", reference_arm="normal", unit="paired unit -> scene mean",
                 n_scenes=None, n_drives=19, weighting="scene-weighted", interval_method="drive bootstrap 95%", evidence_scope="not simulated",
                 mc_uncertainty="NA", recommended_use="descriptive", limitation="structure not covered by any coverage check"))
rows.append(dict(contrast_id="GT-free endpoint excess (B_NUMERIC, continuous m)", cohort="P1-A family G", estimand="A-, A+, all", reference_arm="baseline response (fixed)",
                 unit="unit -> scene mean", n_scenes=208, n_drives=19, weighting="scene-weighted", interval_method="drive bootstrap 95%",
                 evidence_scope="no coverage check for continuous endpoint metrics", mc_uncertainty="NA", recommended_use="sensitivity",
                 limitation="inference by exact sign-flip p with Holm (12-test family); interval descriptive of uncertainty"))
rows.append(dict(contrast_id="binary sign-flip inversion CIs / shifted sign-flip TOST", cohort="all AutoVLA binary contrasts", estimand="any", reference_arm="-",
                 unit="-", n_scenes=None, n_drives=None, weighting="-", interval_method="sign-flip test inversion", evidence_scope="assumption fails with ties",
                 mc_uncertainty="NA", recommended_use="NR", limitation="never cite"))
pd.DataFrame(rows).to_csv(f"{OUT}/CI_RECOMMENDATION.csv", index=False)
print(len(rows))
