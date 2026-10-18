#!/usr/bin/env python
"""P1-C input verification (protocol section 2): bank Normal executed tokens vs stored P1-C H7 'normal' tokens.
Exit 0 if share >= 0.95, else exit 4 (STOP)."""
import json, sys
RUN = "/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1C_new_log_replication/run_20261008"
bank = [json.loads(l) for l in open(sys.argv[1])]
st = {}
for l in open(f"{RUN}/mech/action_history_causal/records.jsonl"):
    x = json.loads(l)
    for pn, pr in x["perturbations"].items():
        st[f"{x['token']}|{pn}"] = pr["conditions"]["normal"].get("action_idx")
eq = [b["normal_acts"] == st.get(b["unit_id"]) for b in bank if b["unit_id"] in st]
res = {"units_compared": len(eq), "equal": int(sum(eq)), "share": sum(eq) / len(eq) if eq else None, "threshold": 0.95,
       "not_in_stored": sum(b["unit_id"] not in st for b in bank)}
res["pass"] = bool(eq) and res["share"] >= 0.95
print(json.dumps(res))
json.dump(res, open(sys.argv[2], "w"), indent=1)
sys.exit(0 if res["pass"] else 4)
