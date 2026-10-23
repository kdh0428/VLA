#!/usr/bin/env python
"""review_followup5 B_NUMERIC manifest (CPU, no model, no outcome read): every family-G unit of the original P1-A run,
with its own sector perturbation token (frozen from the stored perturbation_info) and frozen reference tokens.

  python b_numeric_manifest.py <out.json>
"""
import hashlib
import json
import os
import sys

POC = "/root/VLA/autovla_misalignment_poc"
RAW = f"{POC}/outputs/review_followup/P1A_gt_free_reference/run_20261008/raw/records.jsonl"
STRATA = {"early": (0, 1), "middle": (2, 3), "late": (4, 5, 6, 7)}


def main():
    out = sys.argv[1]
    if os.path.exists(out):
        raise SystemExit("exists")
    units = []
    for line in open(RAW):
        if '"unit_id"' not in line[:200]:
            continue
        x = json.loads(line)
        if x["family"] != "G":
            continue
        pi = x["perturbation_info"]
        st = next(k for k, v in STRATA.items() if x["t_star"] in v)
        units.append({"unit_id": x["unit_id"], "scene": x["token"], "log": x["log"], "group": x["group"], "t_star": x["t_star"],
                      "t_p": x["t_p"], "stratum": st, "paths": ["P1A"], "family": "G", "sector": pi["sector"],
                      "p_token": pi["token_p"], "ref_tokens": x["ref_tokens"], "pilot": False,
                      "stored_rows_acts": {n: v["action_idx"] for n, v in x["rows"].items()}})
    man = {"rule": __doc__, "raw": RAW, "raw_sha256": hashlib.sha256(open(RAW, "rb").read()).hexdigest(),
           "n_units": len(units), "n_scenes": len({u["scene"] for u in units}),
           "groups": {g: [sum(u["group"] == g for u in units), len({u["scene"] for u in units if u["group"] == g})] for g in ("A-", "A+")},
           "units": units}
    json.dump(man, open(out, "w"))
    print(man["n_units"], man["n_scenes"], man["groups"])


if __name__ == "__main__":
    main()
