#!/usr/bin/env python
"""CPU analysis for the v02 distance-matched follow-up and the drawer supplementary task (no GPU, no simulator, no
transformers): raw token ids -> standalone NumPy decoder (bins from the snapshotted processor_config.json).
  python analyze_v02.py <records.jsonl> <out_dir> <processor_config.json> <token_id_ranges.json> [--seed 20261023]
Primary (pre-registered): Delta = E_L - E_O over all analyzed observations (every analyzed observation has a donor that
satisfies 0.8 <= d_L/d_O <= 1.25 by construction). Aggregation: layout mean -> equal-weight layouts within task -> equal-
weight tasks. Bootstrap: 10,000 resamples of layouts within each task (task fixed), 95% percentile. NA rules: two-task
point estimate NA if a task has no analyzed observation; interval NA if any task has <= 1 layout."""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "verifier"))
from svla_standalone_decoder import Decoder  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("rec"); ap.add_argument("out"); ap.add_argument("pc"); ap.add_argument("rg")
ap.add_argument("--seed", type=int, default=20261023); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dec = Decoder(a.pc, a.rg); E = dec.eos
D = lambda x, y: float(np.mean(np.abs(x[:, :6] - y[:, :6])))  # noqa: E731
recs = [json.loads(l) for l in open(a.rec)]
flow, rows, srcrows = [], [], []
for r in recs:
    f = {"obs_id": r["obs_id"], "task": r["task"], "seed": r["seed"], "call": r["call"], "status": r["status"]}
    n = r["official"]["ids"]; a12 = n[:12]
    f["natural_valid"] = len(a12) == 12 and dec.type_valid(a12) and E not in a12
    assert f["natural_valid"] == r["natural_valid"]
    if f["natural_valid"]:
        f["R_pass"] = (r["R"]["ids"][:6] == a12[6:12] and len(n) > 12 and r["R"]["ids"][6] == n[12]
                       and r["official"]["equals_collect_predict_action"] in (True, None) and r["adapter_natural"]["token_equal_13"])
    uN = dec.decode(a12) if f["natural_valid"] else None
    sel = None
    for s in r.get("sources", []):
        a6 = s["ids"][:6]
        ok = len(a6) == 6 and dec.type_valid(a6) and E not in a6
        reason = "invalid_type_or_eos_in_a1a2"
        if ok:
            us = dec.decode(a6 + a12[6:12])
            dO = float(np.mean(np.abs(us[0, :6] - uN[0, :6]))); dL = float(np.mean(np.abs(us[1, :6] - uN[1, :6])))
            if not (a6[2] == a12[2] and a6[5] == a12[5]): reason = "gripper_mismatch"
            elif dO == 0 or dL == 0: reason = "zero_distance"
            elif not (0.8 <= dL / dO <= 1.25): reason = "ratio_outside_0.8_1.25"
            else: reason = None
            srcrows.append({"obs_id": r["obs_id"], "task": r["task"], "j": s["j"], "d_O": dO, "d_L": dL, "reason": reason})
        else:
            srcrows.append({"obs_id": r["obs_id"], "task": r["task"], "j": s["j"], "reason": reason})
        assert reason == s["reason"], (r["obs_id"], s["j"], reason, s["reason"])
        if reason is None and sel is None:
            sel = (s["j"], dO, dL, a6)
    f["n_sources_tried"] = len(r.get("sources", []))
    f["matched_donor"] = sel is not None
    if sel is not None:
        assert sel[0] == r["donor"]["j"]
        a6 = sel[3]
        cond = {"O": a6[:3] + a12[3:6], "L": a12[:3] + a6[3:6], "OL": a6[:6]}
        ua, okA = {}, True
        for k in ("O", "L", "OL"):
            assert r["arms"][k]["cond"] == cond[k] and r["arms"][k]["emitted_prefix"] == a12[:6]
            g = r["arms"][k]["ids"][:6]
            if not (dec.type_valid(g) and E not in g): okA = False; break
            ua[k] = dec.decode(a12[:6] + g)[2:4]
        f["arms_valid"] = okA
        if okA and f["R_pass"]:
            uR = dec.decode(a12[:6] + r["R"]["ids"][:6])[2:4]
            ct1 = [int(a6[0] != a12[0]), int(a6[1] != a12[1])]; ct2 = [int(a6[3] != a12[3]), int(a6[4] != a12[4])]
            row = {**f, "d_O": sel[1], "d_L": sel[2], "ratio": sel[2] / sel[1], "E_L": D(ua["L"], uR), "E_O": D(ua["O"], uR),
                   "E_OL": D(ua["OL"], uR), "D_OL_L": D(ua["OL"], ua["L"]), "same_change_type": ct1 == ct2,
                   "change_type_a1": "".join(map(str, ct1)), "change_type_a2": "".join(map(str, ct2)),
                   "chg_L": r["arms"]["L"]["ids"][:6] != r["R"]["ids"][:6], "chg_O": r["arms"]["O"]["ids"][:6] != r["R"]["ids"][:6]}
            row["Delta"] = row["E_L"] - row["E_O"]; row["J"] = row["E_OL"] - row["E_O"] - row["E_L"]
            for t, nm in ((0, "a3"), (1, "a4")):
                row[f"E_L_{nm}"] = float(np.mean(np.abs(ua["L"][t, :6] - uR[t, :6]))); row[f"E_O_{nm}"] = float(np.mean(np.abs(ua["O"][t, :6] - uR[t, :6])))
            for part, sl in (("trans", slice(0, 3)), ("rot", slice(3, 6))):
                row[f"E_L_{part}"] = float(np.mean(np.abs(ua["L"][:, sl] - uR[:, sl]))); row[f"E_O_{part}"] = float(np.mean(np.abs(ua["O"][:, sl] - uR[:, sl])))
            for k in ("O", "L", "OL"):
                row[f"grip_change_{k}"] = float(np.mean(ua[k][:, 6] != uR[:, 6]))
            rows.append(row)
    flow.append(f)


def agg(rs, col, seed):
    lay = defaultdict(list)
    for x in rs: lay[(x["task"], x["seed"])].append(x[col])
    tasks = sorted({t for t, _ in lay})
    if not tasks: return {"estimate": None, "ci": None, "n_obs": 0, "n_layouts": {}, "task_means": {}, "task_ci": {}}
    by = {t: np.array([np.mean(lay[k]) for k in sorted(lay) if k[0] == t]) for t in tasks}
    tm = {t: float(by[t].mean()) for t in tasks}
    est = float(np.mean(list(tm.values())))
    rng = np.random.default_rng(seed); bt, bts = [], {t: [] for t in tasks}
    for _ in range(10000):
        ms = {t: float(np.mean(rng.choice(by[t], len(by[t])))) for t in tasks}
        for t in tasks: bts[t].append(ms[t])
        bt.append(np.mean(list(ms.values())))
    ci = None if min(len(v) for v in by.values()) <= 1 else [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]
    tci = {t: (None if len(by[t]) <= 1 else [float(np.percentile(bts[t], 2.5)), float(np.percentile(bts[t], 97.5))]) for t in tasks}
    return {"estimate": est, "ci": ci, "n_obs": len(rs), "n_layouts": {t: len(v) for t, v in by.items()}, "task_means": tm, "task_ci": tci}


tasks_all = sorted({r["task"] for r in recs})
S = {"tasks_in_records": tasks_all, "seed": a.seed}
prim = agg(rows, "Delta", a.seed)
if len(tasks_all) == 2 and len(prim["n_layouts"]) < 2:
    prim["estimate"] = None; prim["ci"] = None; prim["note"] = "a task has no analyzed observation -> two-task estimate NA"
S["primary_Delta"] = prim
for c in ("E_L", "E_O", "E_OL", "D_OL_L", "J", "E_L_a3", "E_O_a3", "E_L_a4", "E_O_a4", "E_L_trans", "E_O_trans", "E_L_rot", "E_O_rot",
          "grip_change_L", "grip_change_O", "grip_change_OL"):
    S.setdefault("secondary", {})[c] = agg(rows, c, a.seed)
S["secondary"]["Delta_same_change_type"] = agg([x for x in rows if x["same_change_type"]], "Delta", a.seed)
fl = defaultdict(lambda: defaultdict(int))
for f in flow:
    t = f["task"]; fl[t]["collected"] += 1; fl[t]["natural_valid"] += int(f["natural_valid"]); fl[t]["R_pass"] += int(f.get("R_pass", False))
    fl[t]["matched_donor"] += int(f["matched_donor"]); fl[t]["arms_valid"] += int(f.get("arms_valid", False))
for x in rows: fl[x["task"]]["analyzed"] += 1
S["flow"] = {t: dict(v) for t, v in fl.items()}
S["flow_layouts"] = {t: {"collected": len({f["seed"] for f in flow if f["task"] == t}),
                         "analyzed": len({x["seed"] for x in rows if x["task"] == t})} for t in tasks_all}
S["status_counts"] = {s: sum(1 for f in flow if f["status"] == s) for s in sorted({f["status"] for f in flow})}
rs = defaultdict(int)
for s in srcrows: rs[s["reason"] or "selected"] += 1
S["source_reasons"] = dict(rs); S["n_sources"] = len(srcrows)
S["token_change"] = {"L": f"{sum(x['chg_L'] for x in rows)}/{len(rows)}", "O": f"{sum(x['chg_O'] for x in rows)}/{len(rows)}"}
S["donor_distance"] = {"mean_d_L": float(np.mean([x["d_L"] for x in rows])) if rows else None, "mean_d_O": float(np.mean([x["d_O"] for x in rows])) if rows else None,
                       "ratio_quantiles": (np.quantile([x["ratio"] for x in rows], [0, 0.25, 0.5, 0.75, 1]).round(3).tolist() if rows else None)}
json.dump(S, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
import csv  # noqa: E402
for name, data in (("flow_per_observation.csv", flow), ("per_observation.csv", rows), ("sources_all.csv", srcrows)):
    if data:
        keys = sorted({k for d in data for k in d})
        with open(os.path.join(a.out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, keys); w.writeheader(); [w.writerow(d) for d in data]
lay = defaultdict(list)
for x in rows: lay[(x["task"], x["seed"])].append(x)
with open(os.path.join(a.out, "per_layout.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["task", "seed", "n_obs", "Delta", "E_L", "E_O"])
    for (t, s), v in sorted(lay.items()):
        w.writerow([t, s, len(v), np.mean([x["Delta"] for x in v]), np.mean([x["E_L"] for x in v]), np.mean([x["E_O"] for x in v])])
print(json.dumps({k: S[k] for k in ("primary_Delta", "flow", "flow_layouts", "status_counts", "source_reasons", "token_change", "donor_distance")}, indent=1, default=float))
