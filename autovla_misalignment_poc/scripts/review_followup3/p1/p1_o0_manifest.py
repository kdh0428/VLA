#!/usr/bin/env python
"""AMENDMENT P1-03 manifest extension (CPU only, no outcome read): the Stage-0 manifest's alt0 units (same fixed
sha256 selection), run on the H32 and P1B paths; P1B units carry the P1-A reference tokens of the scene."""
import json, os, sys
P1 = "/root/VLA/autovla_misalignment_poc/outputs/review_followup3/P1"
REF = "/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1A_gt_free_reference/run_20261008/reference/references.jsonl"
out = sys.argv[1] if len(sys.argv) > 1 else f"{P1}/stage0_manifest_O0.json"
if os.path.exists(out):
    raise SystemExit("exists")
man = json.load(open(f"{P1}/stage0_manifest.json"))
ref = {x["token"]: x["ref_tokens"] for x in map(json.loads, open(REF))}
units = []
for u in man["units"]:
    if u["unit_id"].endswith("|alt0"):
        units.append(dict(u, paths=["H32", "P1B"], ref_tokens=ref[u["scene"]]))
json.dump({**{k: v for k, v in man.items() if k != "units"}, "extension": "AMENDMENT_P1_03 (O0): alt0 units on H32, P1B",
           "n_units": len(units), "units": units}, open(out, "w"), indent=1)
print(len(units), [u["unit_id"] for u in units])
