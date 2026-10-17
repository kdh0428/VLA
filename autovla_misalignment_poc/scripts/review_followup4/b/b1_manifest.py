#!/usr/bin/env python
"""B1 manifest (CPU, no model, no outcome read): Stage-0 rule (p1_stage0_manifest.py) with take = 8 per t* stratum.
population 208 dev ED scenes; strata early {0,1} / middle {2,3} / late {4..7}; order sha256("P1-STAGE0-20261014|"+token);
units: alt0 on H7/H8; P1A family G at t_p = t*, lowest available sector. No GF3."""
import hashlib, json, os, sys
sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/review_followup3/p1")
import p1_stage0_manifest as S0  # noqa: E402

TAKE = int(os.environ.get("B1_TAKE", "8"))


def main():
    out = sys.argv[1]
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite")
    ed = [json.loads(l) for l in open(S0.ED)]
    sup = {x["token"]: x for x in map(json.loads, open(S0.SUPPORT))}
    ref = {x["token"]: x for x in map(json.loads, open(S0.REF))}
    units = []
    for name, ts in S0.STRATA.items():
        cand = sorted((r for r in ed if r["t_star"] in ts), key=lambda r: hashlib.sha256((S0.SALT + r["token"]).encode()).hexdigest())
        for i, r in enumerate(cand[:TAKE]):
            tok = r["token"]
            fs = min(sup[tok]["sectors_available"], key=lambda s: S0.SECTORS.index(s))
            common = {"scene": tok, "log": r["log"], "group": r["group"], "t_star": r["t_star"], "stratum": name,
                      "rank_in_stratum": i, "pilot": False, "in_stage0": i < 3}
            units.append({**common, "unit_id": f"{tok}|alt0", "paths": ["H7", "H8"], "perturbation": "alt0",
                          "forced_token": r["alternatives"][0]["token"], "t_p": r["t_star"]})
            units.append({**common, "unit_id": f"{tok}|G:G_{fs}", "paths": ["P1A"], "family": "G", "sector": fs,
                          "t_p": r["t_star"], "ref_tokens": ref[tok]["ref_tokens"]})
    man = {"rule": __doc__, "take_per_stratum": TAKE, "inputs": {"ed_sha256": S0.sha(S0.ED), "references_sha256": S0.sha(S0.REF)},
           "n_scenes": len({u["scene"] for u in units}), "n_units": len(units),
           "group_counts_scenes": {g: len({u["scene"] for u in units if u["group"] == g}) for g in ("A-", "A+")}, "units": units}
    json.dump(man, open(out, "w"), indent=1)
    print(man["n_scenes"], man["n_units"], man["group_counts_scenes"])


if __name__ == "__main__":
    main()
