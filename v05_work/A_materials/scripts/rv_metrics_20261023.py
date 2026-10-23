#!/usr/bin/env python
"""CPU-only, post hoc: RV (Reverse@emb - GT-history) metric decomposition (ADE5/FDE5/rule/severe_tau) from stored
raw records, using the identical metric/aggregation code as cvpr_review_and_contribution_20261020/scripts/ra_a_metrics.py.
Writes only to /root/VLA/v05_work/A_materials/. Validates severe3 / fde5 against stored values."""
import json, sys, os
import numpy as np, pandas as pd
POC = "/root/VLA/autovla_misalignment_poc"
sys.path.insert(0, f"{POC}/outputs/cvpr_review_and_contribution_20261020/scripts")
import ra_common as R  # noqa
import ra_a_metrics as A  # noqa

OUT = "/root/VLA/v05_work/A_materials"
DEV = f"{POC}/outputs/review_followup/R8R9_raw_recovery/run_20261008/raw/exp8/records.jsonl"
P1C = f"{POC}/outputs/review_followup/P1C_new_log_replication/run_20261008/mech/prev_action_state_patching/records.jsonl"
CONDS = ("reverse@emb", "gt_history")

def rows(path, frame):
    out = []
    for line in open(path):
        r = json.loads(line); gt = r["trajectory_gt"]
        for c in CONDS:
            m = A.metrics(r["rows"][c]["trajectory_pred"], gt)
            out.append({"frame": frame, "scene_id": r["token"], "unit_id": f"{r['token']}|{r['perturbation']}",
                        "drive_id": R.drive_of(r["log"]), "group": r.get("group"), "cond": c, **m})
    return pd.DataFrame(out)

dev = rows(DEV, "dev"); dev = dev[dev.group == "A-"]
p1 = rows(P1C, "P1C")
U = pd.read_parquet(f"{R.P1C}/analysis/units.parquet")
U8 = U[(U.block == "H8") & U.condition.isin(CONDS)][["unit_id", "condition", "stratum", "weight", "amplification", "fde5"]]
chk = p1.merge(U8, left_on=["unit_id", "cond"], right_on=["unit_id", "condition"], suffixes=("", "_stored"))
checks = {"P1C_rows": len(p1), "P1C_matched": len(chk), "P1C_severe3_eq_stored": int((chk.severe3 == chk.amplification).sum()),
          "P1C_fde5_maxabs": float((chk.fde5 - chk.fde5_stored).abs().max())}
p1 = p1.merge(U8[["unit_id", "condition", "stratum", "weight"]].rename(columns={"condition": "cond"}), on=["unit_id", "cond"])
D8 = pd.read_parquet(f"{POC}/outputs/review_followup/R8R9_raw_recovery/run_20261008/units.parquet")
D8 = D8[(D8.experiment == "R8") & D8.condition.isin(CONDS) & (D8.group == "A-")]
chk2 = dev.merge(D8[["unit_id", "condition", "amplification", "fde5", "ade5"]], left_on=["unit_id", "cond"],
                 right_on=["unit_id", "condition"], suffixes=("", "_stored"))
checks.update({"DEV_rows": len(dev), "DEV_matched": len(chk2), "DEV_severe3_eq_stored": int((chk2.severe3 == chk2.amplification).sum()),
               "DEV_fde5_maxabs": float((chk2.fde5 - chk2.fde5_stored).abs().max()),
               "DEV_ade5_maxabs": float((chk2.ade5 - chk2.ade5_stored).abs().max())})
specs = [("dev mechanism set A- (failure-selected)", dev, None), ("P1-C F", p1[p1.stratum == "F"], None), ("P1-C E (Hajek)", p1, "weight")]
res = []
for frame, df, w in specs:
    D, Aa, Bb, na, nb, nc = A.contrast(df, "reverse@emb", "gt_history")
    D["weight"] = df[df.cond == "reverse@emb"].set_index("unit_id").loc[D.index, "weight"] if w else 1.0
    est, ns, nd, ws = A.aggregate(D, "weight" if w else None)
    for m in A.MET:
        sc = 1.0 if m in ("ade5", "fde5") else 100.0
        res.append({"contrast": "RV", "intervention": "reverse@emb", "reference": "gt_history", "target_population": frame,
                    "metric": m, "unit": "m" if sc == 1 else "pp", "n_units": nc, "n_scenes": ns, "n_drives": nd,
                    "weighting": "Hajek (stored H8 weights)" if w else "equal scene", "delta": sc * est[m]})
pd.DataFrame(res).to_csv(f"{OUT}/A1_rv_metric_decomposition_computed_20261023.csv", index=False)
json.dump(checks, open(f"{OUT}/A1_rv_metric_checks_20261023.json", "w"), indent=1)
print(json.dumps(checks, indent=1)); print(pd.DataFrame(res).query("metric in ['ade5','fde5','rule_fail','severe2','severe3','severe4','severe5']").to_string())
