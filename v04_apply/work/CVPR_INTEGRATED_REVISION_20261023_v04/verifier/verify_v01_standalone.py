"""Standalone CPU verifier for the v01 SpatialVLA within-call pilot (no GPU, no simulator, no transformers).
  python verify_v01_standalone.py <records.jsonl> <processor_config.json> <token_id_ranges.json> [out.json]"""
import json, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from svla_standalone_decoder import Decoder
REC, PC, RG = sys.argv[1:4]; OUT = sys.argv[4] if len(sys.argv) > 4 else None
dec = Decoder(PC, RG); E = dec.eos
D = lambda x, y: float(np.mean(np.abs(x[:, :6] - y[:, :6])))  # noqa: E731
recs = [json.loads(l) for l in open(REC)]
cnt = defaultdict(int); rows = []; nassert = 0
for r in recs:
    cnt["records"] += 1
    n = r["official"]["ids"]; a12 = n[:12]
    v = len(a12) == 12 and dec.type_valid(a12) and E not in a12
    assert v == r["natural_valid"]; nassert += 1
    if not v: continue
    cnt["natural_valid"] += 1
    assert r["official"]["ids"] == r["official"]["ids"]; 
    if r["R"]["ids"][:6] != a12[6:12] or r["R"]["ids"][6] != (n[12] if len(n) > 12 else None):
        continue
    cnt["R_identity_with_eos"] += 1
    uN = dec.decode(a12)
    if "u_N" in r: assert np.allclose(np.array(r["u_N"]), uN, atol=1e-6); nassert += 1
    sel = None
    for s in r.get("sources", []):
        a6 = s["ids"][:6]
        ok = len(a6) == 6 and dec.type_valid(a6) and E not in a6
        jv = False
        if ok:
            us = dec.decode(a6 + a12[6:12])
            dO, dL = float(np.mean(np.abs(us[0, :6] - uN[0, :6]))), float(np.mean(np.abs(us[1, :6] - uN[1, :6])))
            jv = a6[2] == a12[2] and a6[5] == a12[5] and dO > 0 and dL > 0
            if "d_O" in s: assert abs(s["d_O"] - dO) < 1e-9 and abs(s["d_L"] - dL) < 1e-9; nassert += 2
        assert bool(jv) == bool(s["joint_valid"]); nassert += 1
        cnt["sources"] += 1; cnt["sources_joint_valid"] += int(jv)
        if jv and sel is None: sel = (s["j"], dO, dL, a6)
    if sel is None:
        continue
    cnt["donor_found"] += 1
    assert sel[0] == r["donor"]["j"]; nassert += 1
    a6 = sel[3]
    cond = {"O": a6[:3] + a12[3:6], "L": a12[:3] + a6[3:6], "OL": a6[:6]}
    ua = {}
    for k in ("O", "L", "OL"):
        assert r["arms"][k]["cond"] == cond[k]; nassert += 1
        g = r["arms"][k]["ids"][:6]
        if not (dec.type_valid(g) and E not in g): break
        ua[k] = dec.decode(a12[:6] + g)[2:4]
    if len(ua) < 3: continue
    uR = dec.decode(a12[:6] + r["R"]["ids"][:6])[2:4]
    EL, EO, EOL = D(ua["L"], uR), D(ua["O"], uR), D(ua["OL"], uR)
    rows.append({"task": r["task"], "seed": r["seed"], "Delta": EL - EO, "E_L": EL, "E_O": EO, "D_OL_L": D(ua["OL"], ua["L"]),
                 "matched": 0.8 <= sel[2] / sel[1] <= 1.25, "chg_L": r["arms"]["L"]["ids"][:6] != r["R"]["ids"][:6],
                 "chg_O": r["arms"]["O"]["ids"][:6] != r["R"]["ids"][:6]})


def agg(rs, col, seed):
    lay = defaultdict(list)
    for x in rs: lay[(x["task"], x["seed"])].append(x[col])
    tasks = sorted({t for t, _ in lay})
    by = {t: np.array([np.mean(lay[k]) for k in sorted(lay) if k[0] == t]) for t in tasks}
    tm = {t: float(by[t].mean()) for t in tasks}
    two = float(np.mean(list(tm.values()))) if len(tasks) == 2 else None
    rng = np.random.default_rng(seed); bt = []
    for _ in range(10000):
        ms = [float(np.mean(rng.choice(by[t], len(by[t])))) for t in tasks]
        bt.append(np.mean(ms))
    ci = None if (len(tasks) < 2 or min(len(v) for v in by.values()) <= 1) else [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]
    return {"two_task": two, "ci": ci, "n_obs": len(rs), "n_layouts": {t: len(v) for t, v in by.items()}, "task_means": tm}


res = {"counts": dict(cnt), "analyzed": len(rows), "matched": sum(x["matched"] for x in rows),
       "primary_matched_Delta": agg([x for x in rows if x["matched"]], "Delta", 20261022),
       "secondary_all_Delta": agg(rows, "Delta", 20261022),
       "token_change_L": f"{sum(x['chg_L'] for x in rows)}/{len(rows)}", "token_change_O": f"{sum(x['chg_O'] for x in rows)}/{len(rows)}",
       "assertions": nassert}
print(json.dumps(res, indent=1, default=float))
if OUT: json.dump(res, open(OUT, "w"), indent=1, default=float)
