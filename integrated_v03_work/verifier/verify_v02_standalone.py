"""Independent standalone CPU verifier for the v02 matched follow-up and the drawer task (second implementation; NumPy only).
  python verify_v02_standalone.py <records.jsonl> <summary.json> <processor_config.json> <token_id_ranges.json> [out.json]"""
import json, sys, os
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svla_standalone_decoder import Decoder
REC, SUM, PC, RG = sys.argv[1:5]; OUT = sys.argv[5] if len(sys.argv) > 5 else None
dec = Decoder(PC, RG); E = dec.eos; S = json.load(open(SUM))
md = lambda x, y: float(np.abs(x[:, :6] - y[:, :6]).mean())  # noqa: E731
cnt = defaultdict(int); obs = []; n_assert = 0
for l in open(REC):
    r = json.loads(l); cnt["records"] += 1
    n = r["official"]["ids"]; a12 = n[:12]
    if not (len(a12) == 12 and dec.type_valid(a12) and E not in a12): continue
    cnt["natural_valid"] += 1
    if not (r["R"]["ids"][:6] == a12[6:12] and len(n) > 12 and r["R"]["ids"][6] == n[12] and r["official"]["equals_collect_predict_action"] in (True, None)
            and r["adapter_natural"]["ids"][:13] == n[:13]): continue
    cnt["R_pass"] += 1
    uN = dec.decode(a12); pick = None
    for s in r.get("sources", []):
        cnt["sources"] += 1
        a6 = s["ids"][:6]
        if not (len(a6) == 6 and dec.type_valid(a6) and E not in a6) or a6[2] != a12[2] or a6[5] != a12[5]: continue
        u = dec.decode(a6 + a12[6:12]); dO = np.abs(u[0, :6] - uN[0, :6]).mean(); dL = np.abs(u[1, :6] - uN[1, :6]).mean()
        if dO > 0 and dL > 0 and 0.8 <= dL / dO <= 1.25:
            pick = (s["j"], a6, dO, dL); break
    if pick is None:
        assert r["status"] == "MATCHED_SOURCE_MISSING"; n_assert += 1; continue
    assert pick[0] == r["donor"]["j"] and len(r["sources"]) == pick[0] + 1; n_assert += 2
    cnt["matched"] += 1
    a6 = pick[1]
    arms = {}
    for k, cond in (("O", a6[:3] + a12[3:6]), ("L", a12[:3] + a6[3:6]), ("OL", list(a6))):
        assert r["arms"][k]["cond"] == cond; n_assert += 1
        g = r["arms"][k]["ids"][:6]
        if not (dec.type_valid(g) and E not in g): break
        arms[k] = dec.decode(a12[:6] + g)[2:4]
    if len(arms) < 3: continue
    uR = dec.decode(a12[:6] + r["R"]["ids"][:6])[2:4]
    obs.append((r["task"], r["seed"], md(arms["L"], uR) - md(arms["O"], uR)))
cnt["analyzed"] = len(obs)
lay = defaultdict(list)
for t, s, d in obs: lay[(t, s)].append(d)
tasks = sorted({t for t, _ in lay})
by = {t: np.array([np.mean(v) for (tt, s), v in sorted(lay.items()) if tt == t]) for t in tasks}
est = float(np.mean([by[t].mean() for t in tasks])) if tasks else None
rng = np.random.default_rng(S["seed"]); bt = []
for _ in range(10000):
    bt.append(np.mean([np.mean(rng.choice(by[t], len(by[t]))) for t in tasks]))
ci = None if (not tasks or min(len(v) for v in by.values()) <= 1) else [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]
P = S["primary_Delta"]
ok = {"estimate": abs(est - P["estimate"]) < 1e-12, "ci": (ci is None and P["ci"] is None) or (ci is not None and P["ci"] is not None and max(abs(a - b) for a, b in zip(ci, P["ci"])) < 1e-12),
      "n_obs": cnt["analyzed"] == P["n_obs"], "n_layouts": {t: len(v) for t, v in by.items()} == P["n_layouts"],
      "status_OK": cnt["analyzed"] == S["status_counts"].get("OK", 0), "sources": cnt["sources"] == S["n_sources"]}
res = {"counts": dict(cnt), "estimate": est, "ci": ci, "checks": ok, "assertions": n_assert, "ALL_MATCH": all(ok.values())}
print(json.dumps(res, indent=1, default=float))
if OUT: json.dump(res, open(OUT, "w"), indent=1, default=float)
