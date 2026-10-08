#!/usr/bin/env python
"""
P1-D analysis (protocol: outputs/review_followup/P1D_dose_control/<run_id>/protocol.md). CPU only.

  nice -n 19 python analyze_p1d.py --run RUN_DIR --episodes RUN_DIR/rollouts/main/episodes.jsonl [...] [--smoke]

Outputs in RUN_DIR (plan section 2 layout): units.csv (one row per episode), intervention_trace.jsonl (one row per
injection), verification.json (restoration / no-op / re-run identity), dose_table.csv (budget, dose, exposure per
config x mode), condition_summary.csv, paired_effects.csv (within-config contrasts and DiDs with task-stratified
episode bootstrap CI, exact McNemar / Wilcoxon / sign-flip, Holm within the preregistered families).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon

T_END = 80; WIN_END = 24; TAU, HOLD = 0.02, 4
B_REPS, PERM_REPS = 2000, 10000
CONTRASTS = {"R-N": ("reverse", "natural"), "F-C": ("feedback", "corrected"), "FD-CD": ("feedbackD", "correctedD"),
             "C-N": ("corrected", "natural"), "F-N": ("feedback", "natural"), "CD-N": ("correctedD", "natural"),
             "FD-N": ("feedbackD", "natural"), "null-N": ("null", "natural"), "gen-N": ("gen", "natural")}
# preregistered families (protocol section 5); DiD = later config minus earlier config, same episode
PRIMARY = [("H1", "area_dev_cmstep", "R-N", "r1e1", "r4e1"),            # replanning axis, context budget fixed (A)
           ("H2", "area_dev_cmstep", "FD-CD|F-C", "r1e1", "r1e4")]      # ensemble axis, executed dose matched (B)
SECONDARY_SUCCESS = [("S1", "success", "R-N", "r1e1", "r4e1"), ("S2", "success", "FD-CD|F-C", "r1e1", "r1e4"),
                     ("S3", "success", "F-C", "r1e1", "r4e1"), ("S4", "success", "R-N", "r1e1", "r1e4")]
SECONDARY_TRAJ = [("T1", "area_dev_cmstep", "R-N", "r1e1", "r2e1"), ("T2", "area_dev_cmstep", "F-C", "r1e1", "r4e1"),
                  ("T3", "area_dev_cmstep", "R-N", "r1e1", "r1e4"), ("T4", "area_dev_cmstep", "F-C", "r1e1", "r1e4"),
                  ("T5", "max_dev_cm", "R-N", "r1e1", "r4e1"), ("T6", "realign_t80", "R-N", "r1e1", "r4e1")]
MEANINGFUL = {"success": 0.10, "area_dev_cmstep": 100.0}


def load(paths):
    E = {}
    for p in paths:
        for line in open(p):
            ep = json.loads(line)
            E[(ep["cond"], ep["task"], ep["seed"])] = ep
    return E


def sticky_onsets(a6):
    on = np.abs(a6) > 0.5; ons = []; run_ = 0
    for t in range(len(a6)):
        if on[t]:
            if run_ % 10 == 0:
                ons.append(t)
            run_ += 1
        else:
            run_ = 0
    return ons


def compact(ep):
    rec = ep["rec"]
    pos = np.array([r["pose"][:3] for r in rec]); act = np.array([r["action"] for r in rec])
    inj = [dict(r["inj"], t=r["t"]) for r in rec if "inj" in r]
    ctx = np.array([np.linalg.norm(r["ctx_dev"][:3]) if "ctx_dev" in r else 0.0 for r in rec])
    ctxr = np.array([np.linalg.norm(r["ctx_dev"][3:6]) if "ctx_dev" in r else 0.0 for r in rec])
    ctxg = np.array([abs(r["ctx_dev"][6]) if "ctx_dev" in r else 0.0 for r in rec])
    inj_t = {i["t"] for i in inj}
    w_exp = 0.0                           # weight of executed predictions that came from injected chunks at age >= 1
    for r in rec:
        w = np.exp(0.8 * np.arange(len(r["ages"]))); w = w / w.sum()
        for wi, ag in zip(w, r["ages"]):
            if (r["t"] - ag) in inj_t and ag >= 1:
                w_exp += wi
    first = lambda key: next((r["t"] for r in rec if r.get("info", {}).get(key)), None)
    contact_t = next((r["t"] for r in rec if r.get("contacts")), None)
    return {"task": ep["task"], "seed": ep["seed"], "cfg": ep["cfg"], "mode": ep["mode"], "cond": ep["cond"],
            "success": float(ep["success"]), "pos": pos, "act": act, "inj": inj, "steps": ep["steps"],
            "ctx_trans_m": float(ctx.sum()), "ctx_rot_rad": float(ctxr.sum()), "ctx_grip": float(ctxg.sum()),
            "ctx_trans_win_m": float(ctx[:WIN_END].sum()), "ctx_weight_exposure": w_exp,
            "direct_exec_m": float(sum(np.linalg.norm(i["direct_exec"]) for i in inj)),
            "n_inj": len(inj), "n_gen": ep["n_gen_calls"], "wall_s": ep["wall_s"], "decode_s": ep["decode_s"],
            "restore_ok": None if ep["restore"] is None else bool(ep["restore"]["ok"]),
            "first_grasp_t": first("is_grasped"), "ever_grasped": bool(ep["ever"].get("is_grasped", False)),
            "first_contact_t": contact_t, "sticky_onsets": sticky_onsets(act[:, 6]),
            "state_hashes": [r["state_hash"] for r in rec], "episode_id": ep.get("episode_id")}


def pair(a, b):
    n = min(len(a["pos"]), len(b["pos"]))
    d = np.linalg.norm(a["pos"][:n] - b["pos"][:n], axis=1)
    c = np.linalg.norm(np.cumsum(a["act"][:n, :3], 0) - np.cumsum(b["act"][:n, :3], 0), axis=1)
    re_t = None
    for t in range(WIN_END, n - HOLD + 1):
        if np.all(d[t:t + HOLD] <= TAU):
            re_t = t; break
    ft = lambda x: T_END if x is None else x
    return {"success": a["success"] - b["success"], "mean_dev_cm": 100 * d[8:].mean(), "final_dev_cm": 100 * d[n - 1],
            "max_dev_cm": 100 * d.max(), "area_dev_cmstep": 100 * d[8:].sum(), "post_area_dev_cmstep": 100 * d[WIN_END:].sum(),
            "cmd_final_m": c[-1], "cmd_max_m": c.max(), "realign_t80": float(re_t if re_t is not None else T_END),
            "first_grasp_shift": ft(a["first_grasp_t"]) - ft(b["first_grasp_t"]),
            "ever_grasped_diff": float(a["ever_grasped"]) - float(b["ever_grasped"]),
            "first_contact_shift": ft(a["first_contact_t"]) - ft(b["first_contact_t"]),
            "sticky_onsets_diff": float(len(a["sticky_onsets"]) - len(b["sticky_onsets"])),
            "identical": float(np.array_equal(a["act"], b["act"]) and a["state_hashes"] == b["state_hashes"])}


def boot_idx(keys, seed=0):
    rng = np.random.default_rng(seed); by = defaultdict(list)
    for i, k in enumerate(keys):
        by[k[0]].append(i)
    return np.concatenate([np.array(ix)[rng.integers(0, len(ix), size=(B_REPS, len(ix)))] for ix in by.values()], 1)


def strat_mean(v, keys):
    by = defaultdict(list)
    for i, k in enumerate(keys):
        by[k[0]].append(v[i])
    return float(np.mean([np.mean(x) for x in by.values()]))


def mcnemar(d):
    a = int(np.sum(d > 0)); b = int(np.sum(d < 0))
    return a, b, (float(binomtest(a, a + b, 0.5).pvalue) if a + b else 1.0)


def wil(d):
    d = np.asarray(d, float)
    return float(wilcoxon(d).pvalue) if np.any(d != 0) and len(d) > 1 else 1.0


def signflip(d, seed=0):
    d = np.asarray(d, float); rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=(PERM_REPS, len(d))); obs = abs(d.mean())
    return float((np.abs((s * d).mean(1)) >= obs - 1e-12).mean())


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); out = np.zeros(m); run_ = 0.0
    for i, j in enumerate(o):
        run_ = max(run_, min(1.0, (m - i) * ps[j])); out[j] = run_
    return out


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--smoke", action="store_true", help="skip inferential statistics")
    a = ap.parse_args()
    raw = load(a.episodes)
    C = {k: compact(v) for k, v in raw.items()}
    # ---- units + intervention trace
    units, trace = [], []
    for (cond, task, seed), c in sorted(C.items()):
        units.append({"log_id": None, "scene_id": c["episode_id"], "task_id": task, "episode_id": f"{task}_s{seed}", "seed": seed,
                      "model": "spatialvla-4b-224-sft-fractal", "condition": cond, "config": c["cfg"], "mode": c["mode"],
                      "decode_j": None, "control_t": None, "reference_branch": "natural" if c["mode"] != "natural" else None,
                      "perturbed_span": ",".join(str(i["t"]) for i in c["inj"]) or None,
                      "perturbation_requested": "opposite 0.3 (norm. space), step-1 translation token" if c["inj"] else None,
                      "perturbation_achieved": float(np.mean([i["dnorm"] for i in c["inj"]])) if c["inj"] else None,
                      "intervention_count": c["n_inj"], "parser_status": "ok", "exclusion_reason": None,
                      "primary_outcome": c["success"], "restore_ok": c["restore_ok"], "n_gen_calls": c["n_gen"],
                      "direct_exec_dose_m": c["direct_exec_m"], "ctx_exec_dev_trans_m": c["ctx_trans_m"],
                      "ctx_exec_dev_rot_rad": c["ctx_rot_rad"], "ctx_exec_dev_grip": c["ctx_grip"],
                      "ctx_weight_exposure": c["ctx_weight_exposure"], "flip_frac": float(np.mean([i["flip"] for i in c["inj"]])) if c["inj"] else None,
                      "first_grasp_t": c["first_grasp_t"], "ever_grasped": c["ever_grasped"], "first_contact_t": c["first_contact_t"],
                      "wall_s": c["wall_s"], "decode_s": c["decode_s"]})
        for i in c["inj"]:
            trace.append(dict(i, task=task, seed=seed, condition=cond))
    write_csv(os.path.join(a.run, "units.csv"), units)
    with open(os.path.join(a.run, "intervention_trace.jsonl"), "w") as f:
        for r in trace:
            f.write(json.dumps(r) + "\n")
    # ---- verification
    ver = {"restore_ok": {}, "null_identical_to_N": {}, "rerun_identical_to_N": {}, "prefix_identical_to_N": {}}
    for (cond, task, seed), c in C.items():
        if c["mode"] == "natural" and "#rerun" not in cond:
            continue
        nk = (f"{c['cfg']}:natural", task, seed)
        if nk not in C:
            continue
        n = C[nk]; T0 = raw[(cond, task, seed)]["T0"]
        ver["prefix_identical_to_N"].setdefault(cond, []).append(c["state_hashes"][:T0] == n["state_hashes"][:T0] and np.array_equal(c["act"][:T0], n["act"][:T0]))
        if c["restore_ok"] is not None:
            ver["restore_ok"].setdefault(cond, []).append(c["restore_ok"])
        if c["mode"] == "null":
            ver["null_identical_to_N"].setdefault(cond, []).append(bool(pair(c, n)["identical"]))
        if "#rerun" in cond:
            ver["rerun_identical_to_N"].setdefault(cond, []).append(bool(pair(c, n)["identical"]))
    ver = {k: {c: f"{sum(v)}/{len(v)}" for c, v in d.items()} for k, d in ver.items()}
    json.dump(ver, open(os.path.join(a.run, "verification.json"), "w"), indent=1)
    print(json.dumps(ver, indent=1))
    # ---- dose / budget / exposure table
    by = defaultdict(list)
    for c in C.values():
        by[(c["cfg"], c["cond"].split(":")[1])].append(c)
    dose = []
    for (cfg, mode), cs in sorted(by.items()):
        inj = [i for c in cs for i in c["inj"]]
        dose.append({"config": cfg, "condition": mode, "n_episodes": len(cs), "gen_calls_per_ep": np.mean([c["n_gen"] for c in cs]),
                     "injections_per_ep": np.mean([c["n_inj"] for c in cs]),
                     "achieved_dnorm_mean": np.mean([i["dnorm"] for i in inj]) if inj else None,
                     "flip_frac": np.mean([i["flip"] for i in inj]) if inj else None,
                     "w0_mean": np.mean([i["w0"] for i in inj]) if inj else None,
                     "delta_phys_m_mean": np.mean([np.linalg.norm(i["delta_phys"]) for i in inj]) if inj else None,
                     "direct_exec_dose_per_ep_m": np.mean([c["direct_exec_m"] for c in cs]),
                     "ctx_exec_dev_trans_per_ep_m": np.mean([c["ctx_trans_m"] for c in cs]),
                     "ctx_exec_dev_rot_per_ep_rad": np.mean([c["ctx_rot_rad"] for c in cs]),
                     "ctx_exec_dev_grip_per_ep": np.mean([c["ctx_grip"] for c in cs]),
                     "ctx_weight_exposure_per_ep": np.mean([c["ctx_weight_exposure"] for c in cs]),
                     "success_rate": np.mean([c["success"] for c in cs]),
                     "wall_s_per_ep": np.mean([c["wall_s"] for c in cs]), "decode_s_per_ep": np.mean([c["decode_s"] for c in cs])})
    write_csv(os.path.join(a.run, "dose_table.csv"), [{k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()} for r in dose])
    if a.smoke:
        return
    # ---- paired contrasts (episode = (task, seed), common to all configs)
    tasks = sorted({c["task"] for c in C.values()})
    seeds = {t: sorted({c["seed"] for c in C.values() if c["task"] == t and c["mode"] == "natural"}) for t in tasks}
    keys_all = [(t, s) for t in tasks for s in seeds[t]]

    def contrast_vals(cfg, cn, metric, keys):
        alts = cn.split("|")
        out = []
        for t, s in keys:
            v = None
            for alt in alts:
                x, y = CONTRASTS[alt]
                ka, kb = (f"{cfg}:{x}", t, s), (f"{cfg}:{y}", t, s)
                if ka in C and kb in C:
                    v = pair(C[ka], C[kb])[metric]; break
            out.append(np.nan if v is None else v)
        return np.array(out, float)

    rows = []

    def estimate(name, v, keys, metric, kind):
        ok = ~np.isnan(v); v, keys = v[ok], [k for k, o in zip(keys, ok) if o]
        if len(v) < 2:
            return None
        Bi = boot_idx(keys)
        bm = np.array([strat_mean(v[b], [keys[i] for i in b]) for b in Bi[:B_REPS]]) if len(v) < 50 else None
        if bm is None:
            # vectorised stratified bootstrap
            tk = np.array([k[0] for k in keys]); ms = []
            for t in sorted(set(tk)):
                ix = np.nonzero(tk == t)[0]
                ms.append(v[ix][np.random.default_rng(0).integers(0, len(ix), size=(B_REPS, len(ix)))].mean(1))
            bm = np.mean(ms, 0)
        est = strat_mean(v, keys)
        r = {"name": name, "metric": metric, "kind": kind, "n": len(v), "estimate": est,
             "ci_lo": float(np.percentile(bm, 2.5)), "ci_hi": float(np.percentile(bm, 97.5)),
             "ci90_lo": float(np.percentile(bm, 5)), "ci90_hi": float(np.percentile(bm, 95))}
        if metric == "success" and kind == "within":
            a_, b_, p = mcnemar(v); r.update({"test": "exact McNemar", "x_only": a_, "y_only": b_, "p": p})
        elif metric == "success":
            r.update({"test": "sign-flip", "p": signflip(v)})
        else:
            r.update({"test": "sign-flip permutation (mean); Wilcoxon reported", "p": signflip(v), "p_wilcoxon": wil(v)})
        if metric in MEANINGFUL:
            mg = MEANINGFUL[metric]
            r["meaningful_threshold"] = mg
            r["equivalent_within_threshold(90%CI)"] = bool(r["ci90_lo"] > -mg and r["ci90_hi"] < mg)
        for t in tasks:
            ix = [i for i, k in enumerate(keys) if k[0] == t]
            r[f"{t}_est"] = float(np.mean(v[ix])) if ix else None
        rows.append(r)
        return r
    cfgs = sorted({c["cfg"] for c in C.values()})
    metrics = ["success", "area_dev_cmstep", "max_dev_cm", "final_dev_cm", "mean_dev_cm", "post_area_dev_cmstep", "cmd_final_m",
               "realign_t80", "first_grasp_shift", "ever_grasped_diff", "first_contact_shift", "sticky_onsets_diff"]
    for cfg in cfgs:
        for cn in CONTRASTS:
            for mt in metrics:
                v = contrast_vals(cfg, cn, mt, keys_all)
                if np.all(np.isnan(v)):
                    continue
                estimate(f"{cfg}:{cn}", v, keys_all, mt, "within")
    fam_rows = {}
    for fam, spec in (("primary", PRIMARY), ("secondary_success", SECONDARY_SUCCESS), ("secondary_traj", SECONDARY_TRAJ)):
        for hid, mt, cn, c1, c2 in spec:
            v = contrast_vals(c2, cn, mt, keys_all) - contrast_vals(c1, cn, mt, keys_all)
            r = estimate(f"{hid}: DiD {c2}-{c1} [{cn}]", v, keys_all, mt, "did")
            if r is not None:
                r["family"] = fam; fam_rows.setdefault(fam, []).append(r)
    for fam, rs in fam_rows.items():
        for r, ph in zip(rs, holm([r["p"] for r in rs])):
            r["p_holm_family"] = float(ph)
    write_csv(os.path.join(a.run, "paired_effects.csv"), [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
    cs = []
    for (cfg, mode), lst in sorted(by.items()):
        for t in tasks + ["all"]:
            l2 = [c for c in lst if t == "all" or c["task"] == t]
            if l2:
                cs.append({"config": cfg, "condition": mode, "task": t, "n": len(l2), "success_rate": round(float(np.mean([c["success"] for c in l2])), 4),
                           "ever_grasped": round(float(np.mean([c["ever_grasped"] for c in l2])), 4)})
    write_csv(os.path.join(a.run, "condition_summary.csv"), cs)
    print("wrote", a.run)


if __name__ == "__main__":
    main()
