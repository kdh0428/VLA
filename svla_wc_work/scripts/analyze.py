#!/usr/bin/env python
"""CPU analysis / independent verifier (no GPU, no simulator). Recomputes everything from raw token ids in
run/<phase>/records.jsonl with the official action tokenizer (processor loaded on CPU).
  python analyze.py <records.jsonl> <out_dir>
"""
import json, os, sys
import numpy as np, pandas as pd

REC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
from transformers import AutoProcessor  # noqa: E402
P = AutoProcessor.from_pretrained("/root/VLA/spatialvla/spatialvla-4b-224-sft-fractal", trust_remote_code=True)
TOK = P.action_tokenizer
R0 = [(TOK.translation_tokenizer.token_start_idx, TOK.translation_tokenizer.token_end_idx),
      (TOK.rotation_tokenizer.token_start_idx, TOK.rotation_tokenizer.token_end_idx),
      (TOK.gripper_tokenizer.token_start_idx, TOK.gripper_tokenizer.token_end_idx)]
EOS, B, SEED = 1, 10000, 20261022


def tvalid(ids):
    return all(R0[j % 3][0] <= int(t) <= R0[j % 3][1] for j, t in enumerate(ids))


def u(ids12):
    return TOK.decode_token_ids_to_actions(np.asarray(ids12, dtype=np.int64).reshape(-1, 3))


def D(x, y):  # x, y: (2, 7) for a3, a4; motion dims only
    return float(np.mean(np.abs(x[:, :6] - y[:, :6])))


recs = [json.loads(l) for l in open(REC)]
rows, flow = [], []
for r in recs:
    f = {"obs_id": r["obs_id"], "task": r["task"], "seed": r["seed"], "call": r["call"], "status": r["status"]}
    off = r["official"]["ids"]; a12 = off[:12]
    f["natural_valid_recomputed"] = len(a12) == 12 and tvalid(a12) and EOS not in a12
    assert f["natural_valid_recomputed"] == r["natural_valid"], r["obs_id"]
    if not r["natural_valid"]:
        flow.append(f); continue
    f["R_identity_recomputed"] = r["R"]["ids"][:6] == a12[6:12]
    uN = u(a12)
    if "sources" in r:
        # recompute donor eligibility and selection from raw source ids
        sel = None
        for s in r["sources"]:
            a6 = s["ids"][:6]
            ok = len(a6) == 6 and tvalid(a6) and EOS not in a6
            if ok:
                us = u(a6 + a12[6:12])
                dO = float(np.mean(np.abs(us[0, :6] - uN[0, :6]))); dL = float(np.mean(np.abs(us[1, :6] - uN[1, :6])))
                jv = a6[2] == a12[2] and a6[5] == a12[5] and dO > 0 and dL > 0
            else:
                jv = False
            assert bool(jv) == bool(s["joint_valid"]), (r["obs_id"], s["j"])
            if jv and sel is None:
                sel = (s["j"], dO, dL, a6)
        f["donor_found"] = sel is not None
        if sel is not None:
            assert sel[0] == r["donor"]["j"]
            f.update({"donor_j": sel[0], "d_O": sel[1], "d_L": sel[2], "ratio": sel[2] / sel[1]})
            # verify conditioning prefixes and replacement indices
            d6 = sel[3]
            exp = {"O": d6[:3] + a12[3:6], "L": a12[:3] + d6[3:6], "OL": d6[:6]}
            for k, v in exp.items():
                assert r["arms"][k]["cond"] == v and r["arms"][k]["emitted_prefix"] == a12[:6]
            uR = u(a12[:6] + r["R"]["ids"][:6])[2:4]
            arms_ok = all(tvalid(r["arms"][k]["ids"][:6]) and EOS not in r["arms"][k]["ids"][:6] for k in ("O", "L", "OL"))
            f["arms_valid"] = arms_ok
            if arms_ok and f["R_identity_recomputed"]:
                ua = {k: u(a12[:6] + r["arms"][k]["ids"][:6])[2:4] for k in ("O", "L", "OL")}
                for k in ("O", "L", "OL"):
                    if "u_a34" in r["arms"][k]:
                        assert np.allclose(np.array(r["arms"][k]["u_a34"]), ua[k])
                EL, EO, EOL = D(ua["L"], uR), D(ua["O"], uR), D(ua["OL"], uR)
                f.update({"E_L": EL, "E_O": EO, "E_OL": EOL, "Delta": EL - EO, "D_OL_L": D(ua["OL"], ua["L"]),
                          "J": EOL - EO - EL, "matched": 0.8 <= f["ratio"] <= 1.25})
                for t, nm in ((0, "a3"), (1, "a4")):
                    f[f"E_L_{nm}"] = float(np.mean(np.abs(ua["L"][t, :6] - uR[t, :6]))); f[f"E_O_{nm}"] = float(np.mean(np.abs(ua["O"][t, :6] - uR[t, :6])))
                for part, sl in (("trans", slice(0, 3)), ("rot", slice(3, 6))):
                    f[f"E_L_{part}"] = float(np.mean(np.abs(ua["L"][:, sl] - uR[:, sl]))); f[f"E_O_{part}"] = float(np.mean(np.abs(ua["O"][:, sl] - uR[:, sl])))
                for k in ("O", "L", "OL"):
                    f[f"grip_change_{k}"] = float(np.mean(ua[k][:, 6] != uR[:, 6]))
                f["any_change_L"] = r["arms"]["L"]["ids"][:6] != r["R"]["ids"][:6]; f["any_change_O"] = r["arms"]["O"]["ids"][:6] != r["R"]["ids"][:6]
                rows.append(f)
    flow.append(f)
F = pd.DataFrame(flow); A = pd.DataFrame(rows)
F.to_csv(os.path.join(OUT, "flow_per_observation.csv"), index=False)
A.to_csv(os.path.join(OUT, "per_observation.csv"), index=False)


def agg(df, col):
    lay = df.groupby(["task", "seed"])[col].mean().reset_index()
    tasks = sorted(df.task.unique())
    res = {"n_obs": int(len(df)), "n_layouts": {t: int((lay.task == t).sum()) for t in tasks},
           "task_means": {t: float(lay[lay.task == t][col].mean()) for t in tasks}}
    both = len(tasks) == 2
    res["two_task"] = float(np.mean(list(res["task_means"].values()))) if both else None
    rng = np.random.default_rng(SEED)
    by = {t: lay[lay.task == t][col].values for t in tasks}
    bs = {t: [] for t in tasks}; two = []
    for _ in range(B):
        ms = {t: float(np.mean(rng.choice(v, len(v)))) for t, v in by.items()}
        for t in tasks: bs[t].append(ms[t])
        if both: two.append(np.mean(list(ms.values())))
    res["task_ci"] = {t: (None if len(by[t]) <= 1 else [float(np.percentile(bs[t], 2.5)), float(np.percentile(bs[t], 97.5))]) for t in tasks}
    res["two_task_ci"] = (None if (not both or min(len(v) for v in by.values()) <= 1) else [float(np.percentile(two, 2.5)), float(np.percentile(two, 97.5))])
    return res, lay


summary = {}
if len(A):
    Mt = A[A.matched]
    TASKS = ["google_robot_move_near", "google_robot_pick_coke_can"]
    for name, df in (("primary_matched", Mt), ("all_joint_valid", A)):
        if df.task.nunique() < 2:
            summary[name] = {"two_task": None, "note": "a task has no observations -> NA", "n_obs": int(len(df)),
                             "n_layouts": df.groupby("task").seed.nunique().to_dict()}
            if len(df): summary[name]["task_means_layout_weighted"] = df.groupby(["task", "seed"]).Delta.mean().groupby("task").mean().to_dict()
            continue
        summary[name] = {c: agg(df, c)[0] for c in ("Delta", "E_L", "E_O", "D_OL_L", "J", "E_OL")}
        agg(df, "Delta")[1].to_csv(os.path.join(OUT, f"per_layout_{name}.csv"), index=False)
    for c in ("E_L_a3", "E_O_a3", "E_L_a4", "E_O_a4", "E_L_trans", "E_O_trans", "E_L_rot", "E_O_rot", "grip_change_L", "grip_change_O", "grip_change_OL"):
        summary.setdefault("secondary_all_joint_valid", {})[c] = agg(A, c)[0] if A.task.nunique() == 2 else None
    summary["donor_distance_vs_response"] = {"corr_dL_EL": float(np.corrcoef(A.d_L, A.E_L)[0, 1]) if len(A) > 2 else None,
                                             "corr_dO_EO": float(np.corrcoef(A.d_O, A.E_O)[0, 1]) if len(A) > 2 else None}
summary["flow"] = {"records": len(recs), "natural_valid": int(F.natural_valid_recomputed.sum()),
                   "R_identity": int(F.get("R_identity_recomputed", pd.Series(dtype=bool)).fillna(False).sum()),
                   "donor_found": int(F.get("donor_found", pd.Series(dtype=bool)).fillna(False).sum()),
                   "arms_valid": int(F.get("arms_valid", pd.Series(dtype=bool)).fillna(False).sum()),
                   "analyzed_joint_valid": int(len(A)), "matched": int(A.matched.sum()) if len(A) else 0,
                   "status_counts": F.status.value_counts().to_dict(),
                   "by_task": F.groupby("task").size().to_dict()}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
print(json.dumps(summary["flow"], indent=1))
if "primary_matched" in summary:
    print(json.dumps({k: summary[k] for k in ("primary_matched", "all_joint_valid") if k in summary}, indent=1, default=float)[:3000])
