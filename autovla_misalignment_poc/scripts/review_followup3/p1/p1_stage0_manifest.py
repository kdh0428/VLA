#!/usr/bin/env python
"""
Stage-0 / P1-B AutoVLA verification manifest (CPU only, no model, no outcome read).

Fixed selection rule (written before any control was run; the rule never reads a perturbation outcome,
an amplification label, or any control result):
  population  = the 208 scenes of outputs/equal_distance_perturbation/records.jsonl (dev mechanism set,
                the unit source of exps 7, 8, 9, 32 and of P1-A)
  strata      = intervention position t* (inherited first-mismatch step):
                  early  t* in {0, 1}
                  middle t* in {2, 3}
                  late   t* in {4, 5, 6, 7}
  order       = sha256("P1-STAGE0-20261014|" + scene_token) ascending, within stratum
  take        = first 3 scenes per stratum  -> 9 scenes
  units per scene (one per code path family; never chosen by outcome):
    H7/H8/H9  (exp 7 action_history_causal = Recent-GT/Full-GT, exp 8 prev_action_state_patching = Reverse-N
               (reverse@emb), exp 9 temporal_feedback_window = windows / Full-GT) : perturbation "alt0"
               (first distance-matched alternative, present in every scene)
    P1A       (P1-A self-reference path, rows U/OC/O1/C1/OC_ref*/O_refH/OC_gt*): family G at t_p = t*,
               sector = lowest sector index available in the recorded perturbation_support (GT-free rule)
    P1A-GF3   (P1-A fixed-position / coordinate-accumulation path, prefix = reference tokens r[:3]) for the
               3 EARLY scenes only (t_p = 3 > t*, so the prefix contains reference-continuation tokens)
  pilot       = the first scene of each stratum (3 scenes) -> control-5 pilot only
The A-/A+ group is recorded but never used for selection. A- share is reported, not balanced.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

POC = "/root/VLA/autovla_misalignment_poc"
ED = f"{POC}/outputs/equal_distance_perturbation/records.jsonl"
FULL = f"{POC}/outputs/full_extract/records.jsonl"
P1A_RUN = f"{POC}/outputs/review_followup/P1A_gt_free_reference/run_20261008"
REF = f"{P1A_RUN}/reference/references.jsonl"
SUPPORT = f"{P1A_RUN}/raw/perturbation_support.jsonl"
SALT = "P1-STAGE0-20261014|"
STRATA = {"early": {0, 1}, "middle": {2, 3}, "late": {4, 5, 6, 7}}
SECTORS = ["faster", "faster-left", "left", "slower-left", "slower", "slower-right", "right", "faster-right"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else f"{POC}/outputs/review_followup3/P1/stage0_manifest.json"
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite")
    ed = [json.loads(l) for l in open(ED)]
    ref = {x["token"]: x for x in map(json.loads, open(REF))}
    sup = {x["token"]: x for x in map(json.loads, open(SUPPORT))}
    want = set()
    picks = {}
    for name, ts in STRATA.items():
        cand = sorted((r for r in ed if r["t_star"] in ts), key=lambda r: hashlib.sha256((SALT + r["token"]).encode()).hexdigest())
        picks[name] = cand[:3]
        want |= {r["token"] for r in cand[:3]}
    units = []
    for name, rs in picks.items():
        for i, r in enumerate(rs):
            tok = r["token"]
            secs = sup[tok]["sectors_available"]
            first_sector = min(secs, key=lambda s: SECTORS.index(s))
            common = {"scene": tok, "log": r["log"], "group": r["group"], "t_star": r["t_star"], "stratum": name,
                      "rank_in_stratum": i, "pilot": i == 0, "hash": hashlib.sha256((SALT + tok).encode()).hexdigest()}
            units.append({**common, "unit_id": f"{tok}|alt0", "paths": ["H7", "H8", "H9"], "perturbation": "alt0",
                          "forced_token": r["alternatives"][0]["token"], "t_p": r["t_star"]})
            units.append({**common, "unit_id": f"{tok}|G:G_{first_sector}", "paths": ["P1A"], "family": "G",
                          "sector": first_sector, "t_p": r["t_star"], "ref_tokens": ref[tok]["ref_tokens"]})
            if name == "early":
                units.append({**common, "unit_id": f"{tok}|GF3:GF3_{first_sector}", "paths": ["P1A"], "family": "GF3",
                              "sector": first_sector, "t_p": 3, "ref_tokens": ref[tok]["ref_tokens"],
                              "note": "sector chosen by the same lowest-index rule at t_p=t*; GF3 sector support is re-derived by the harness at t_p=3"})
    man = {"rule": __doc__, "salt": SALT, "strata": {k: sorted(v) for k, v in STRATA.items()},
           "inputs": {"ed_records": ED, "ed_sha256": sha(ED), "references": REF, "references_sha256": sha(REF),
                      "perturbation_support": SUPPORT},
           "n_scenes": len(want), "n_units": len(units),
           "group_counts": {g: sum(1 for u in units if u["group"] == g and u["paths"] != ["P1A"]) for g in ("A-", "A+")},
           "units": units}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(man, open(out, "w"), indent=1)
    for u in units:
        print(u["stratum"], u["unit_id"], u["group"], "t*", u["t_star"], "t_p", u["t_p"], u["paths"], "pilot" if u["pilot"] else "")
    print("scenes", len(want), "units", len(units), man["group_counts"])


if __name__ == "__main__":
    main()
