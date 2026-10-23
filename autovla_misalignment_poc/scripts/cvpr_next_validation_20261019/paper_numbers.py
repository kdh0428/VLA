#!/usr/bin/env python
"""CVPR next validation: PAPER_READY_NUMBERS.csv — only rows whose interval status changes or is newly specified in this
round. Estimates and p values are copied unchanged from the stored outputs (hashes included).

  nice -n 19 /root/miniforge3/envs/autovla/bin/python paper_numbers.py
"""
import hashlib, json
import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
O = f"{POC}/outputs"
OUT = f"{O}/cvpr_next_validation_20261019"
POP = f"{O}/review_followup3/P0/POPULATION_EFFECTS.csv"
MPR = f"{O}/review_followup4/B/B2/p1c_analysis/MATCHED_PREFIX_RESULTS.csv"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
SEV = "severe = A- AND FDE5 > 3 m at 5 s; pp of units, scene-weighted"
rows = []
pop = pd.read_csv(POP).set_index("row_id")
for rid, cid, status, prev in (("P0A.P1C.RN.P", "E_RN", "APPROXIMATE_ONLY (finite eligible frame and new drives; superpop min coverage 0.908)", "NR (review_followup5)"),
                               ("P0A.P1C.RV.P", "E_RV", "finite eligible frame only: APPROXIMATE (D3 min 0.953); new drives: WITHHOLD (min 0.886)", "NR (review_followup5)"),
                               ("P0A.P1C.DM.P", "E_DM", "NR (not simulated)", "NR (review_followup5)")):
    r = pop.loc[rid]
    show = not status.startswith(("NR", "WITHHOLD"))
    rows.append({"claim_id": f"CVPR.{cid}", "cohort": "P1-C eligible frame E (2,191 scenes; 353 sampled)", "estimand": "finite-population E effect (Hajek)",
                 "outcome": SEV, "sign": "treatment - reference (RN negative = reduction; RV positive = restoration)", "estimate": 100 * r.theta, "unit": "pp",
                 "ci_status": status, "previous_ci_status": prev, "ci_method": "cluster-robust t (G-1 df), drive clusters" if show else "none",
                 "ci_low": 100 * r.clt_ci95_lo if show else None, "ci_high": 100 * r.clt_ci95_hi if show else None,
                 "p_raw": r.p_signflip_raw, "p_adjusted": r.p_holm, "adjustment": "Holm", "family": f"{r.family_id} (m = {r.family_m})", "alpha": r.family_alpha,
                 "n_units": int(r.n_units), "n_scenes": int(r.n_scenes), "n_drives": int(r.n_drives), "n_components": 6,
                 "source_file": POP, "source_row": rid, "source_hash": sha(POP)})
m = pd.read_csv(MPR)
for con, est, cid, status, prev in (("C_NAT", "P", "E4_CNAT", "finite domain only: APPROXIMATE (D3 min 0.940); new drives: WITHHOLD (min 0.890)", "NR (review_followup5)"),
                                     ("C_GT", "P", "E4_CGT", "APPROXIMATE_ONLY (D3 min 0.930, D2 min 0.927)", "NR (review_followup5)"),
                                     ("INT", "P", "E4_INT", "APPROXIMATE_ONLY (D3 min 0.928, D2 min 0.912); joint four-cell structure simulated", "NR (review_followup5)"),
                                     ("C_PROV", "P", "E4_CPROV", "NR (not simulated)", "NR (review_followup5)"),
                                     ("INT", "F", "F_INT", "WITHHOLD for new drives (D2 min 0.886); never read as equality", "approximate sensitivity (review_followup5 / author queries)"),
                                     ("C_NAT", "F", "F_CNAT", "APPROXIMATE (fixed-scene min 0.931; superpop 0.936-0.943)", "approximate (author queries)")):
    r = m[(m.n_old == 1) & (m.metric == "severe") & (m.subset == "all") & (m.cluster == "drive") & (m.estimand == est) & (m.contrast == con)].iloc[0]
    lo, hi = json.loads(r.clt_ci95)
    show = not status.startswith(("NR", "WITHHOLD"))
    rows.append({"claim_id": f"CVPR.{cid}", "cohort": "P1-C B2 cohort (existing drives re-run)", "estimand": "E4 finite-population effect (Hajek; 1,775 frame, 312 sampled)" if est == "P" else "F (100 scenes, census)",
                 "outcome": SEV, "sign": "cell difference as defined (INT = C_GT - C_NAT)", "estimate": 100 * r.theta, "unit": "pp",
                 "ci_status": status, "previous_ci_status": prev, "ci_method": "cluster-robust t (G-1 df), drive clusters" if show else "none",
                 "ci_low": 100 * lo if show else None, "ci_high": 100 * hi if show else None, "p_raw": r.p_signflip, "p_adjusted": r.holm_p,
                 "adjustment": "Holm" if pd.notna(r.holm_p) else "none", "family": r.family if isinstance(r.family, str) else None, "alpha": r.family_alpha,
                 "n_units": int(r.n_units), "n_scenes": int(r.n_scenes), "n_drives": int(r.n_clusters), "n_components": 6,
                 "source_file": MPR, "source_row": f"{con}|{est}|drive|n_old=1|severe|all", "source_hash": sha(MPR)})
pd.DataFrame(rows).to_csv(f"{OUT}/PAPER_READY_NUMBERS.csv", index=False)
print(pd.DataFrame(rows)[["claim_id", "estimate", "ci_low", "ci_high", "ci_status"]].round(2).to_string())
