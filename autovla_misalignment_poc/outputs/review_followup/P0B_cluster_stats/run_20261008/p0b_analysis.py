#!/usr/bin/env python
"""
P0-B cluster-statistics reanalysis (protocol.md in this directory, written before any of these numbers were computed).
CPU only, no model inference. Reads existing raw records; writes only into this directory.

Metrics are taken from the original analysis code (imported, not redefined):
  AutoVLA   analyze_action_history.unit_metrics (amplification = A- and FDE5 > 3 m)
  Impromptu analyze_temporal.metrics
  SpatialVLA per-unit token metrics: the per-unit formulas of analyze_svla.token_level (copied below, translation
             table from analyze_svla.table(), i.e. the processor; no weights).

  PYTHONPATH=/root/VLA/spatialvla/site nice -n 19 python p0b_analysis.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from scipy.stats import t as tdist

OUT = os.path.dirname(os.path.abspath(__file__))
POC = "/root/VLA/autovla_misalignment_poc"
O = os.path.join(POC, "outputs")
sys.path.insert(0, os.path.join(POC, "scripts"))
sys.path.insert(0, os.path.join(POC, "scripts/cross_vla_temporal"))
sys.path.insert(0, os.path.join(POC, "scripts/cross_domain_temporal"))
from analyze_action_history import unit_metrics, AMP_FDE   # noqa: E402
import analyze_temporal as AT                               # noqa: E402

SEED = 20261008
B = 10_000
B_LOO = 2_000
MC = 200_000
MC_LOO = 20_000
EXACT_MAX = 20
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------------------------------------- generic statistics
def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def sign_matrix(G):
    return (((np.arange(2 ** G)[:, None] >> np.arange(G)) & 1) * 2 - 1).astype(np.float64)


def signflip(vals, denom, seed, mc=MC):
    """two-sided p for statistic sum(eps * vals) / denom; exact if G <= EXACT_MAX."""
    vals = np.asarray(vals, float); G = len(vals); obs = abs(vals.sum() / denom)
    if G <= EXACT_MAX:
        T = np.abs(sign_matrix(G) @ vals / denom)
        return float(np.mean(T >= obs - 1e-12)), f"exact 2^{G}"
    rng = np.random.default_rng(seed); cnt = 0; done = 0
    while done < mc:
        k = min(50_000, mc - done)
        E = rng.integers(0, 2, size=(k, G)) * 2 - 1
        cnt += int(np.sum(np.abs(E @ vals / denom) >= obs - 1e-12)); done += k
    return float((cnt + 1) / (mc + 1)), f"MC {mc}"


def log_arrays(df):
    """df: columns log, scene, d -> per-log arrays."""
    sc = df.groupby(["log", "scene"])["d"].agg(["mean", "sum", "count"]).reset_index()
    logs = sorted(sc["log"].unique())
    g = sc.groupby("log")
    S = g["mean"].sum().reindex(logs).values.astype(float)
    n = g.size().reindex(logs).values.astype(float)
    U = g["sum"].sum().reindex(logs).values.astype(float)
    m = g["count"].sum().reindex(logs).values.astype(float)
    return logs, S, n, U, m, S / n


def cluster_t(num, w):
    """ratio estimator sum(num)/sum(w), linearised cluster-robust SE, df G-1."""
    G = len(num); th = num.sum() / w.sum(); r = num - th * w
    se = np.sqrt(G / (G - 1) * np.sum(r ** 2)) / w.sum()
    if se == 0:
        return 1.0 if th == 0 else 0.0
    return float(2 * tdist.sf(abs(th / se), G - 1))


def log_cluster_effects(df, seed, b=B, mc=MC, with_tests=True):
    logs, S, n, U, m, L = log_arrays(df)
    G = len(logs)
    est = {"scene": S.sum() / n.sum(), "unit": U.sum() / m.sum(), "log": L.mean()}
    rng = np.random.default_rng(seed)
    C = rng.multinomial(G, np.ones(G) / G, size=b).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = {"scene": C @ S / (C @ n), "unit": C @ U / (C @ m), "log": C @ L / G}
    res = {}
    for w in ("scene", "unit", "log"):
        x = bs[w][np.isfinite(bs[w])]
        r = {"estimate": float(est[w]), "ci95": [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))],
             "ci90": [float(np.percentile(x, 5)), float(np.percentile(x, 95))]}
        if with_tests:
            if w == "scene":
                r["p_signflip"], r["sf_method"] = signflip(S, n.sum(), seed + 1, mc)
                r["p_cluster_t"] = cluster_t(S, n)
            elif w == "unit":
                r["p_signflip"], r["sf_method"] = signflip(U, m.sum(), seed + 2, mc)
                r["p_cluster_t"] = cluster_t(U, m)
            else:
                r["p_signflip"], r["sf_method"] = signflip(L, G, seed + 3, mc)
                r["p_cluster_t"] = cluster_t(L, np.ones(G))
        res[w] = r
    res["n_clusters"] = G; res["n_scenes"] = int(n.sum()); res["n_units"] = int(m.sum()); res["logs"] = logs
    res["arrays"] = (S, n, U, m, L)
    return res


def unit_level_test(a, b, binary):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if binary:
        g = int(np.sum(a > b)); l_ = int(np.sum(b > a))
        return (float(binomtest(g, g + l_, 0.5).pvalue) if g + l_ else 1.0), f"McNemar exact ({g} vs {l_})"
    d = a - b
    return (float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0), "Wilcoxon signed-rank (units)"


# ---------------------------------------------------------------------------------------------- AutoVLA raw
LONG = []          # units table rows
JOIN = {}          # join / exclusion report


def add_long(**kw):
    base = dict(model=None, experiment=None, unit_id=None, scene_id=None, log_id=None, task_id=None, episode_id=None,
                seed=None, group=None, perturbation=None, condition=None, amplification=None, fde5=None, success=None,
                exclusion_reason=None, note=None)
    base.update(kw); LONG.append(base)


W = {}   # W[exp][uid] = {"log","scene","group","pert", cond: {"amplification","fde5"}}

E7_CONDS = ["normal", "hist_attn_mask", "recent_attn_mask", "hist_emb_neutral", "recent_emb_neutral", "gt_history", "recent_gt"]
E11_CONDS = ["normal", "gt_history", "recent_gt", "geo_nn_gt", "emb_nn_gt", "geo_nn_self", "emb_nn_self", "random_tok", "mean_emb"]
E32_CONDS = ["normal", "recent_gt", "dir_ok_mag_wrong", "dir_wrong_mag_ok", "mirror_same_dist", "random_mag_matched"]
OOD_NOTE = {"hist_emb_neutral": "OOD condition in original analysis (not used for verdicts)",
            "recent_emb_neutral": "OOD condition in original analysis (not used for verdicts)",
            "mean_emb": "OOD condition in original analysis"}


def load_autovla():
    for exp, path, conds in (("exp07_action_history_causal", "action_history_causal/units.jsonl", E7_CONDS),
                             ("exp11_prev_action_identity", "prev_action_identity_decomposition/units.jsonl", E11_CONDS)):
        W[exp] = {}
        for line in open(os.path.join(O, path)):
            u = json.loads(line); uid = f"{u['token']}|{u['pert']}"
            W[exp][uid] = {"log": u["log"], "scene": u["token"], "group": u["group"], "pert": u["pert"]}
            for c in conds:
                x = u.get(c)
                amp = None if x is None else x.get("amplification"); fde = None if x is None else x.get("fde5")
                reason = "condition_missing" if x is None else ("metric_missing" if amp is None else None)
                W[exp][uid][c] = {"amplification": None if amp is None else float(amp), "fde5": fde}
                add_long(model="AutoVLA", experiment=exp, unit_id=uid, scene_id=u["token"], log_id=u["log"], group=u["group"],
                         perturbation=u["pert"], condition=c, amplification=W[exp][uid][c]["amplification"], fde5=fde,
                         exclusion_reason=reason, note=OOD_NOTE.get(c))
    exp = "exp32_motion_semantics"; W[exp] = {}; nosub = defaultdict(int)
    for line in open(os.path.join(O, "motion_semantics_ablation/records.jsonl")):
        r = json.loads(line); t = r["t_star"]; uid = f"{r['token']}|{r['perturbation']}"
        W[exp][uid] = {"log": r["log"], "scene": r["token"], "group": r["group"], "pert": r["perturbation"], "no_sub": {}}
        for c in E32_CONDS:
            x = r["conditions"].get(c)
            if x is None:
                W[exp][uid][c] = {"amplification": None, "fde5": None}; reason = "condition_missing"
            else:
                mm = unit_metrics(x["action_idx"], x["trajectory_pred"], x["entropy_steps"], r["gt"], r["trajectory_gt"], t)
                W[exp][uid][c] = {"amplification": float(mm["amplification"]), "fde5": mm["fde5"]}; reason = None
            note = None
            if c in E32_CONDS[2:] and x is not None and not x.get("substitutions"):
                note = "no substitution recorded (row = own context)"; nosub[(r["group"], c)] += 1
                W[exp][uid]["no_sub"][c] = True
            add_long(model="AutoVLA", experiment=exp, unit_id=uid, scene_id=r["token"], log_id=r["log"], group=r["group"],
                     perturbation=r["perturbation"], condition=c, amplification=W[exp][uid][c]["amplification"],
                     fde5=W[exp][uid][c]["fde5"], exclusion_reason=reason, note=note)
    JOIN["exp32_units_without_substitution"] = {f"{g}|{c}": v for (g, c), v in sorted(nosub.items())}
    exp = "exp06_equal_distance"; W[exp] = {}
    for line in open(os.path.join(O, "equal_distance_perturbation/rows.jsonl")):
        x = json.loads(line); uid = f"{x['token']}|{x['cond']}"
        amp = x.get("amplification")
        W[exp][uid] = {"log": x["log"], "scene": x["token"], "group": x["group"], "pert": x["cond"], "kind": x["kind"],
                       "valid": x["valid"], "amplification": None if amp is None else float(amp), "fde5": x.get("fde5")}
        add_long(model="AutoVLA", experiment=exp, unit_id=uid, scene_id=x["token"], log_id=x["log"], group=x["group"],
                 perturbation=x["cond"], condition=x["kind"], amplification=W[exp][uid]["amplification"], fde5=x.get("fde5"),
                 exclusion_reason=None, note=None if x["valid"] else "invalid model output (original: counted, amplification False)")
    # ---- join report
    rep = {}
    ref = {k for k in W["exp06_equal_distance"] if W["exp06_equal_distance"][k]["kind"] in ("original", "alt")}
    for exp in ("exp07_action_history_causal", "exp11_prev_action_identity", "exp32_motion_semantics"):
        ids = set(W[exp]); rep[exp] = {"n_units": len(ids), "missing_vs_exp06_units": len(ref - ids), "extra_vs_exp06_units": len(ids - ref)}
        mism = 0
        for k in ids & ref:
            a, b_ = W[exp][k], W["exp06_equal_distance"][k]
            mism += (a["log"] != b_["log"]) or (a["group"] != b_["group"])
        rep[exp]["log_or_group_mismatch_vs_exp06"] = mism
        for g in ("A-", "A+"):
            us = [u for u in W[exp].values() if u["group"] == g]
            rep[exp][g] = {"units": len(us), "scenes": len({u["scene"] for u in us}), "logs": len({u["log"] for u in us})}
    ids7, ids11, ids32 = set(W["exp07_action_history_causal"]), set(W["exp11_prev_action_identity"]), set(W["exp32_motion_semantics"])
    rep["common_units_exp07_exp11_exp32"] = len(ids7 & ids11 & ids32)
    JOIN["autovla"] = rep


def autovla_pair_df(exp, group, a, b, metric, drop_nosub=False):
    rows, miss = [], 0
    for uid, u in W[exp].items():
        if u["group"] != group:
            continue
        if drop_nosub and (u.get("no_sub", {}).get(a) or u.get("no_sub", {}).get(b)):
            continue
        x, y = u[a][metric], u[b][metric]
        if x is None or y is None:
            miss += 1; continue
        rows.append({"log": u["log"], "scene": u["scene"], "uid": uid, "a": float(x), "b": float(y)})
    df = pd.DataFrame(rows); df["d"] = df["a"] - df["b"]
    return df, miss


# ---------------------------------------------------------------------------------------------- Impromptu raw
IMP = []


def load_impromptu():
    n_ref = n_pf = 0
    for line in open(os.path.join(O, "cross_vla_temporal_replication/exp_cde/records.jsonl")):
        r = json.loads(line); gt = r["gt"]; refu = next(u for u in r["units"] if u["dir"] is None)
        for u in r["units"]:
            if u["dir"] is None:
                n_ref += 1; continue
            uid = f"{r['token']}|{u['dir']}"
            if u["parse_fail"] or refu["parse_fail"]:
                n_pf += 1
                add_long(model="Impromptu-VLA-3B", experiment="exp33_impromptu_cde", unit_id=uid, scene_id=r["token"], log_id=r["log"],
                         perturbation=str(u["dir"]), condition=None, exclusion_reason="parse_fail (original rule)")
                continue
            x = {"log": r["log"], "scene": r["token"], "uid": uid}
            for row in AT.ROWS:
                mm = AT.metrics(u["rows"][row], gt); x[row] = mm
                add_long(model="Impromptu-VLA-3B", experiment="exp33_impromptu_cde", unit_id=uid, scene_id=r["token"], log_id=r["log"],
                         perturbation=str(u["dir"]), condition=row, amplification=mm["amp"], fde5=mm["fde"])
            IMP.append(x)
    JOIN["impromptu"] = {"units_used": len(IMP), "reference_units_dir_none_excluded": n_ref, "parse_fail_excluded": n_pf,
                         "scenes": len({x["scene"] for x in IMP}), "logs": len({x["log"] for x in IMP})}


def imp_pair_df(a, b):
    df = pd.DataFrame([{"log": x["log"], "scene": x["scene"], "uid": x["uid"], "a": x[a]["amp"], "b": x[b]["amp"]} for x in IMP])
    df["d"] = df["a"] - df["b"]; return df


# ---------------------------------------------------------------------------------------------- SpatialVLA raw
SV_ROWS = ["normal", "recent_ref", "full_ref", "win1", "reverse", "ref_trans", "near_ref", "dir_ok_mag_wrong",
           "dir_wrong_mag_ok", "random_mag_matched"]
EPS = {}
TOK = []


def load_svla():
    d = os.path.join(O, "cross_domain_temporal_replication")
    eps = [json.loads(l) for l in open(os.path.join(d, "closed_loop/episodes.jsonl"))]
    conds = sorted({r["cond"] for r in eps})
    for r in eps:
        EPS[(r["task"], r["seed"], r["cond"])] = float(r["success"])
    keys = sorted({(r["task"], r["seed"]) for r in eps})
    complete = [k for k in keys if all(k + (c,) in EPS for c in conds)]
    for k in keys:
        for c in conds:
            ok = k in complete
            add_long(model="SpatialVLA-4B", experiment="exp34_closed_loop", unit_id=f"{k[0]}|{k[1]}", task_id=k[0],
                     episode_id=f"{k[0]}|{k[1]}", seed=k[1], condition=c, success=EPS.get(k + (c,)),
                     exclusion_reason=None if ok else "episode missing a condition")
    JOIN["svla_closed_loop"] = {"episodes": len(keys), "complete_episodes": len(complete), "conditions": conds,
                                "per_task": {t: sum(k[0] == t for k in complete) for t in sorted({k[0] for k in keys})}}
    import analyze_svla as SV
    V = SV.table()
    gen_eq = []
    for l in open(os.path.join(d, "token_level/units.jsonl")):
        f = json.loads(l); gen_eq.append(f["gen_equal"])
        # ---- formulas copied from analyze_svla.token_level (amp, rec, dev4 only; D/A/E4 not needed) ----
        R = np.array(f["R"]); r = np.array([V[int(t)] for t in R[:, 0]])
        for ui, u in enumerate(f["units"]):
            it = {"cl": (f["task"], f["seed"]), "frame": f["frame"], "d": u["d"], "dir": u["dir"], "subs": u["subs"]}
            for row in SV_ROWS:
                S = np.array(u["rows"][row]); v = np.array([V[int(t)] for t in S[:, 0]])
                inj = np.linalg.norm(v[0] - r[0]); add = np.linalg.norm((v[1:] - r[1:]).sum(0))
                it[row] = {"amp": float(add / max(inj, 1e-9) >= 1), "rec": float(np.array_equal(S[1:, 0], R[1:, 0])),
                           "dev4": float(np.linalg.norm(v[3] - r[3]))}
            # ---- end copied formulas ----
            it["motion_unit"] = it["subs"]["dir_ok_mag_wrong"][0]["e"] > 0
            uid = f"{f['frame']}|{u['d']}|{u['dir']}|{ui}"
            it["uid"] = uid
            TOK.append(it)
            for row in SV_ROWS:
                add_long(model="SpatialVLA-4B", experiment="exp34_token_level", unit_id=uid, scene_id=f["frame"],
                         task_id=f["task"], episode_id=f"{f['task']}|{f['seed']}", seed=f["seed"], perturbation=f"{u['d']}|{u['dir']}",
                         condition=row, amplification=it[row]["amp"],
                         note=None if row not in ("dir_ok_mag_wrong", "dir_wrong_mag_ok") or it["motion_unit"] else "motion contrast excludes (e=0)")
    JOIN["svla_token_level"] = {"units": len(TOK), "frames": len(gen_eq), "episodes": len({t["cl"] for t in TOK}),
                                "motion_units_e_gt_0": int(sum(t["motion_unit"] for t in TOK))}


def strat_episode_effects(ep_vals, seed, b=B, mc=MC, weights=None):
    """ep_vals: dict task -> array of per-episode values (episode-level summaries).
    weights: dict task -> (sums, counts) for unit-weighted estimator (optional).
    Estimate = mean over tasks of task means (episode-weighted). Stratified episode bootstrap; stratified sign-flip."""
    tasks = sorted(ep_vals); T = len(tasks)
    est = np.mean([ep_vals[t].mean() for t in tasks])
    rng = np.random.default_rng(seed)
    bs = np.zeros(b); bsU_num = np.zeros(b); bsU_den = np.zeros(b)
    for t in tasks:
        v = ep_vals[t]; k = len(v)
        C = rng.multinomial(k, np.ones(k) / k, size=b).astype(float)
        bs += (C @ v / k) / T
        if weights:
            s_, c_ = weights[t]; bsU_num += C @ s_; bsU_den += C @ c_
    # sign-flip: statistic = sum_t sum_e eps * v_e / (T k_t)
    allv = np.concatenate([ep_vals[t] / (T * len(ep_vals[t])) for t in tasks])
    p_sf, meth = signflip(allv, 1.0, seed + 1, mc)
    out = {"estimate": float(est), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           "ci90": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))], "p_signflip": p_sf, "sf_method": meth + " (task-stratified)",
           "n_clusters": int(sum(len(ep_vals[t]) for t in tasks))}
    if weights:
        num = np.concatenate([weights[t][0] for t in tasks]); den = np.concatenate([weights[t][1] for t in tasks])
        x = bsU_num / bsU_den
        pU, mU = signflip(num, den.sum(), seed + 2, mc)
        out["unit"] = {"estimate": float(num.sum() / den.sum()), "ci95": [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))],
                       "ci90": [float(np.percentile(x, 5)), float(np.percentile(x, 95))], "p_signflip": pU, "sf_method": mU + " (episode sums)"}
    return out


# ---------------------------------------------------------------------------------------------- main
def main():
    load_autovla(); log("AutoVLA loaded", {k: len(v) for k, v in W.items()})
    load_impromptu(); log("Impromptu loaded", JOIN["impromptu"])
    load_svla(); log("SpatialVLA loaded", JOIN["svla_closed_loop"]["complete_episodes"], JOIN["svla_token_level"])

    orig = {}
    s7 = json.load(open(os.path.join(O, "action_history_causal/summary.json")))
    s11 = json.load(open(os.path.join(O, "prev_action_identity_decomposition/summary.json")))
    rc32 = json.load(open(os.path.join(O, "motion_semantics_ablation/row_contrasts.json")))
    s8 = json.load(open(os.path.join(O, "prev_action_state_patching/summary.json")))
    s6 = json.load(open(os.path.join(O, "equal_distance_perturbation/summary.json")))
    a33 = json.load(open(os.path.join(O, "cross_vla_temporal_replication/analysis.json")))["CDE"]["all_units"]
    a34 = json.load(open(os.path.join(O, "cross_domain_temporal_replication/analysis.json")))

    def o(d, pkey="mcnemar", neg=False, src=""):
        sg = -1 if neg else 1
        p = d.get(pkey, {}).get("p") if pkey == "mcnemar" else d.get("wilcoxon_p")
        ci = [sg * d["ci95"][0], sg * d["ci95"][1]]
        return {"estimate": sg * d["mean"], "ci95": sorted(ci), "p": p, "n": d.get("n"), "source": src}

    ROWS_OUT, LOO_OUT, CMP = [], [], []

    # ------------------------------------------------ AutoVLA contrasts
    av_specs = [
        # id, family, exp, group, a, b, metric, original
        ("AV-P1 Recent-Normal", "AutoVLA primary", "exp07_action_history_causal", "A-", "recent_gt", "normal", "amplification",
         o(s7["subsets"]["all perturbations"]["A-"]["vs_normal"]["recent_gt"]["amplification"], src="action_history_causal/summary.json")),
        ("AV-P3 Direction-Magnitude", "AutoVLA primary", "exp32_motion_semantics", "A-", "dir_wrong_mag_ok", "dir_ok_mag_wrong", "amplification",
         o(rc32["A- dir_wrong_mag_ok-dir_ok_mag_wrong amplification"], src="motion_semantics_ablation/row_contrasts.json")),
        ("AV-S Recent-Normal FDE", "AutoVLA secondary", "exp07_action_history_causal", "A-", "recent_gt", "normal", "fde5",
         o(s7["subsets"]["all perturbations"]["A-"]["vs_normal"]["recent_gt"]["fde5"], "wilcoxon", src="action_history_causal/summary.json")),
        ("AV-S Direction-Magnitude FDE", "AutoVLA secondary", "exp32_motion_semantics", "A-", "dir_wrong_mag_ok", "dir_ok_mag_wrong", "fde5",
         o(rc32["A- dir_wrong_mag_ok-dir_ok_mag_wrong fde5"], "wilcoxon", src="motion_semantics_ablation/row_contrasts.json")),
        ("AV-S GT-history-Normal", "AutoVLA secondary", "exp07_action_history_causal", "A-", "gt_history", "normal", "amplification",
         o(s7["subsets"]["all perturbations"]["A-"]["vs_normal"]["gt_history"]["amplification"], src="action_history_causal/summary.json")),
        ("AV-S Recent-Normal (exp11 harness)", "AutoVLA secondary", "exp11_prev_action_identity", "A-", "recent_gt", "normal", "amplification",
         o(s11["groups"]["A-"]["vs_normal"]["recent_gt"]["amplification"], src="prev_action_identity_decomposition/summary.json")),
        ("AV-S GeoNN(GT)-RecentGT [equivalence]", "AutoVLA secondary", "exp11_prev_action_identity", "A-", "geo_nn_gt", "recent_gt", "amplification",
         o(s11["groups"]["A-"]["vs_recent_gt"]["geo_nn_gt"]["amplification"], src="prev_action_identity_decomposition/summary.json")),
        ("AV-S Mirror-Normal", "AutoVLA secondary", "exp32_motion_semantics", "A-", "mirror_same_dist", "normal", "amplification",
         o(rc32["A- mirror_same_dist-normal amplification"], src="motion_semantics_ablation/row_contrasts.json")),
        ("AV-S Recent-Normal A+", "AutoVLA secondary", "exp07_action_history_causal", "A+", "recent_gt", "normal", "amplification",
         o(s7["subsets"]["all perturbations"]["A+"]["vs_normal"]["recent_gt"]["amplification"], src="action_history_causal/summary.json")),
        ("AV-S Direction-Magnitude A+", "AutoVLA secondary", "exp32_motion_semantics", "A+", "dir_wrong_mag_ok", "dir_ok_mag_wrong", "amplification", None),
    ]
    av_res = {}
    for i, (cid, fam, exp, g, a, b, met, ori) in enumerate(av_specs):
        df, miss = autovla_pair_df(exp, g, a, b, met)
        r = log_cluster_effects(df, SEED + 100 * i)
        pu, pmeth = unit_level_test(df["a"], df["b"], met == "amplification")
        av_res[cid] = (r, df)
        for w in ("scene", "unit", "log"):
            rr = r[w]
            ROWS_OUT.append({"contrast_id": cid, "family": fam, "model": "AutoVLA", "experiment": exp, "group": g, "cond_a": a, "cond_b": b,
                             "metric": met, "weighting": {"scene": "scene-weighted (primary)", "unit": "unit-weighted", "log": "log-equal"}[w],
                             "estimate": rr["estimate"], "ci95_lo": rr["ci95"][0], "ci95_hi": rr["ci95"][1], "ci90_lo": rr["ci90"][0],
                             "ci90_hi": rr["ci90"][1], "ci_method": f"log block bootstrap B={B} percentile",
                             "p_signflip": rr["p_signflip"], "signflip_method": rr["sf_method"] + " over log summaries",
                             "p_cluster_t": rr["p_cluster_t"], "p_unit_level": pu, "unit_level_method": pmeth,
                             "n_clusters": r["n_clusters"], "cluster_unit": "navtest log", "n_scenes": r["n_scenes"], "n_units": r["n_units"],
                             "n_paired_missing": miss})
        if ori:
            CMP.append({"contrast_id": cid, "experiment": exp, "group": g, "metric": met, "orig_estimate": ori["estimate"],
                        "orig_ci95_lo": ori["ci95"][0], "orig_ci95_hi": ori["ci95"][1], "orig_p": ori["p"],
                        "orig_method": "unit mean; log bootstrap 2000 seed 0; unit-level " + ("McNemar" if met == "amplification" else "Wilcoxon"),
                        "orig_n_units": ori["n"], "orig_source": ori["source"],
                        "new_unit_estimate": r["unit"]["estimate"], "unit_estimate_matches": abs(r["unit"]["estimate"] - ori["estimate"]) < 1e-9,
                        "new_unit_level_p": pu,
                        "new_scene_estimate": r["scene"]["estimate"], "new_scene_ci95_lo": r["scene"]["ci95"][0], "new_scene_ci95_hi": r["scene"]["ci95"][1],
                        "new_scene_p_signflip": r["scene"]["p_signflip"], "new_scene_p_cluster_t": r["scene"]["p_cluster_t"],
                        "new_log_estimate": r["log"]["estimate"], "new_log_ci95_lo": r["log"]["ci95"][0], "new_log_ci95_hi": r["log"]["ci95"][1],
                        "new_log_p_signflip": r["log"]["p_signflip"],
                        "n_clusters": r["n_clusters"], "n_scenes": r["n_scenes"], "n_units": r["n_units"]})
        log("done", cid, f"scene {r['scene']['estimate']:+.4f} {r['scene']['ci95']} p={r['scene']['p_signflip']:.3g}")
    # exp 32 sensitivity: drop units without recorded substitution
    df, miss = autovla_pair_df("exp32_motion_semantics", "A-", "dir_wrong_mag_ok", "dir_ok_mag_wrong", "amplification", drop_nosub=True)
    r = log_cluster_effects(df, SEED + 999)
    ROWS_OUT.append({"contrast_id": "AV-P3 Direction-Magnitude (drop units w/o substitution)", "family": "AutoVLA sensitivity", "model": "AutoVLA",
                     "experiment": "exp32_motion_semantics", "group": "A-", "cond_a": "dir_wrong_mag_ok", "cond_b": "dir_ok_mag_wrong",
                     "metric": "amplification", "weighting": "scene-weighted (primary)", "estimate": r["scene"]["estimate"],
                     "ci95_lo": r["scene"]["ci95"][0], "ci95_hi": r["scene"]["ci95"][1], "ci90_lo": r["scene"]["ci90"][0], "ci90_hi": r["scene"]["ci90"][1],
                     "ci_method": f"log block bootstrap B={B} percentile", "p_signflip": r["scene"]["p_signflip"],
                     "signflip_method": r["scene"]["sf_method"] + " over log summaries", "p_cluster_t": r["scene"]["p_cluster_t"],
                     "p_unit_level": unit_level_test(df["a"], df["b"], True)[0], "unit_level_method": "McNemar exact",
                     "n_clusters": r["n_clusters"], "cluster_unit": "navtest log", "n_scenes": r["n_scenes"], "n_units": r["n_units"], "n_paired_missing": miss})
    # Reverse-Full: raw unavailable
    rv = s8["groups"]["A-"]["all"]["rows"]["reverse@emb"]["vs_base"]["amplification"]
    ROWS_OUT.append({"contrast_id": "AV-P2 Reverse-Full", "family": "AutoVLA primary", "model": "AutoVLA", "experiment": "exp08_prev_action_state_patching",
                     "group": "A-", "cond_a": "reverse@emb", "cond_b": "gt_history", "metric": "amplification", "weighting": "RAW UNAVAILABLE",
                     "estimate": None, "ci_method": "not computable: no per-unit records (only summary.json)", "n_clusters": 16,
                     "cluster_unit": "navtest log", "n_scenes": 52, "n_units": 365})
    CMP.append({"contrast_id": "AV-P2 Reverse-Full", "experiment": "exp08_prev_action_state_patching", "group": "A-", "metric": "amplification",
                "orig_estimate": rv["mean"], "orig_ci95_lo": rv["ci95"][0], "orig_ci95_hi": rv["ci95"][1], "orig_p": rv["mcnemar"]["p"],
                "orig_method": "unit mean; log bootstrap 2000 seed 0; unit-level McNemar", "orig_n_units": rv["n"],
                "orig_source": "prev_action_state_patching/summary.json groups.A-.all.rows.reverse@emb.vs_base",
                "note": "RAW UNAVAILABLE: cluster reanalysis impossible"})

    # equal-distance (exp 6): alternative amplification rates, descriptive
    ed = pd.DataFrame([{"log": u["log"], "scene": u["scene"], "group": u["group"], "d": u["amplification"] or 0.0}
                       for u in W["exp06_equal_distance"].values() if u["kind"] == "alt"])
    ed_res = {}
    for g in ("A-", "A+"):
        r = log_cluster_effects(ed[ed.group == g], SEED + 600 + (g == "A+"), with_tests=False); ed_res[g] = r
        for w in ("scene", "unit", "log"):
            ROWS_OUT.append({"contrast_id": f"ED alt amplification rate {g}", "family": "AutoVLA secondary (level, not contrast)", "model": "AutoVLA",
                             "experiment": "exp06_equal_distance", "group": g, "cond_a": "alt", "cond_b": None, "metric": "amplification",
                             "weighting": {"scene": "scene-weighted (primary)", "unit": "unit-weighted", "log": "log-equal"}[w],
                             "estimate": r[w]["estimate"], "ci95_lo": r[w]["ci95"][0], "ci95_hi": r[w]["ci95"][1], "ci90_lo": r[w]["ci90"][0],
                             "ci90_hi": r[w]["ci90"][1], "ci_method": f"log block bootstrap B={B} percentile", "n_clusters": r["n_clusters"],
                             "cluster_unit": "navtest log", "n_scenes": r["n_scenes"], "n_units": r["n_units"]})
        tk = s6["table"][g]["alt (all)"]["amplification"]
        CMP.append({"contrast_id": f"ED alt amplification rate {g}", "experiment": "exp06_equal_distance", "group": g, "metric": "amplification",
                    "orig_estimate": tk["mean"], "orig_ci95_lo": tk["ci95"][0], "orig_ci95_hi": tk["ci95"][1], "orig_p": None,
                    "orig_method": "unit mean; log bootstrap 2000 seed 0", "orig_n_units": tk["n"], "orig_source": "equal_distance_perturbation/summary.json table",
                    "new_unit_estimate": r["unit"]["estimate"], "unit_estimate_matches": abs(r["unit"]["estimate"] - tk["mean"]) < 1e-9,
                    "new_scene_estimate": r["scene"]["estimate"], "new_scene_ci95_lo": r["scene"]["ci95"][0], "new_scene_ci95_hi": r["scene"]["ci95"][1],
                    "new_log_estimate": r["log"]["estimate"], "new_log_ci95_lo": r["log"]["ci95"][0], "new_log_ci95_hi": r["log"]["ci95"][1],
                    "n_clusters": r["n_clusters"], "n_scenes": r["n_scenes"], "n_units": r["n_units"]})
    # A- minus A+ (scene-weighted), joint log bootstrap over the union of logs
    logsU = sorted(set(ed.log)); idx = {lg: i for i, lg in enumerate(logsU)}; G = len(logsU)
    arr = {}
    for g in ("A-", "A+"):
        S = np.zeros(G); n = np.zeros(G)
        sc = ed[ed.group == g].groupby(["log", "scene"])["d"].mean().reset_index()
        for _, x in sc.iterrows():
            S[idx[x["log"]]] += x["d"]; n[idx[x["log"]]] += 1
        arr[g] = (S, n)
    rng = np.random.default_rng(SEED + 610); C = rng.multinomial(G, np.ones(G) / G, size=B).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = C @ arr["A-"][0] / (C @ arr["A-"][1]) - C @ arr["A+"][0] / (C @ arr["A+"][1])
    diff = diff[np.isfinite(diff)]
    est = arr["A-"][0].sum() / arr["A-"][1].sum() - arr["A+"][0].sum() / arr["A+"][1].sum()
    ROWS_OUT.append({"contrast_id": "ED alt amplification A- minus A+", "family": "AutoVLA secondary (between-group)", "model": "AutoVLA",
                     "experiment": "exp06_equal_distance", "group": "A- vs A+", "cond_a": "alt", "metric": "amplification",
                     "weighting": "scene-weighted (primary)", "estimate": float(est), "ci95_lo": float(np.percentile(diff, 2.5)),
                     "ci95_hi": float(np.percentile(diff, 97.5)), "ci90_lo": float(np.percentile(diff, 5)), "ci90_hi": float(np.percentile(diff, 95)),
                     "ci_method": f"joint log block bootstrap over {G} logs B={B} (resamples with no A-/A+ log dropped: {len(diff)})",
                     "n_clusters": G, "cluster_unit": "navtest log (union)", "n_scenes": int(arr['A-'][1].sum() + arr['A+'][1].sum()),
                     "n_units": int(len(ed))})

    # ------------------------------------------------ AutoVLA LOO (A- primary + exp11 Recent)
    for cid in ("AV-P1 Recent-Normal", "AV-P3 Direction-Magnitude", "AV-S Recent-Normal (exp11 harness)", "AV-S Recent-Normal FDE",
                "AV-S Direction-Magnitude FDE"):
        r, df = av_res[cid]; full = r["scene"]["estimate"]
        for j, lg in enumerate(r["logs"]):
            sub = df[df.log != lg]
            rr = log_cluster_effects(sub, SEED + 5000 + j, b=B_LOO, mc=MC_LOO)
            LOO_OUT.append({"contrast_id": cid, "experiment": [s[2] for s in av_specs if s[0] == cid][0], "dropped_log": lg,
                            "dropped_log_n_scenes": int(df[df.log == lg].scene.nunique()), "dropped_log_n_units": int((df.log == lg).sum()),
                            "dropped_log_scene_effect_mean": float(df[df.log == lg].groupby("scene")["d"].mean().mean()),
                            "full_estimate": full, "loo_estimate": rr["scene"]["estimate"], "influence": full - rr["scene"]["estimate"],
                            "loo_ci95_lo": rr["scene"]["ci95"][0], "loo_ci95_hi": rr["scene"]["ci95"][1], "loo_p_signflip": rr["scene"]["p_signflip"],
                            "loo_signflip_method": rr["scene"]["sf_method"], "loo_n_clusters": rr["n_clusters"], "loo_n_scenes": rr["n_scenes"],
                            "loo_n_units": rr["n_units"], "loo_bootstrap_B": B_LOO})
        log("LOO done", cid)

    # ------------------------------------------------ Impromptu
    imp_specs = [("IMP Recent-Normal", "recent_gt", "normal", a33["vs_normal"]["recent_gt"]["amp"]),
                 ("IMP GT-history-Normal", "gt_history", "normal", a33["vs_normal"]["gt_history"]["amp"]),
                 ("IMP Reverse-Full", "reverse", "gt_history", a33["reverse_vs_gt_history"]["amp"]),
                 ("IMP Direction-Magnitude", "dir_wrong_mag_ok", "dir_ok_mag_wrong", a33["motion"]["dir_wrong_mag_ok - dir_ok_mag_wrong"]["amp"])]
    for i, (cid, a, b, od) in enumerate(imp_specs):
        df = imp_pair_df(a, b); r = log_cluster_effects(df, SEED + 700 + 10 * i)
        pu, pm = unit_level_test(df["a"], df["b"], True)
        for w in ("scene", "unit", "log"):
            rr = r[w]
            ROWS_OUT.append({"contrast_id": cid, "family": "Impromptu secondary (own Holm m=4)", "model": "Impromptu-VLA-3B", "experiment": "exp33_impromptu_cde",
                             "group": "all units", "cond_a": a, "cond_b": b, "metric": "amplification",
                             "weighting": {"scene": "scene-weighted (primary)", "unit": "unit-weighted", "log": "log-equal"}[w],
                             "estimate": rr["estimate"], "ci95_lo": rr["ci95"][0], "ci95_hi": rr["ci95"][1], "ci90_lo": rr["ci90"][0], "ci90_hi": rr["ci90"][1],
                             "ci_method": f"log block bootstrap B={B} percentile", "p_signflip": rr["p_signflip"], "signflip_method": rr["sf_method"] + " over log summaries",
                             "p_cluster_t": rr["p_cluster_t"], "p_unit_level": pu, "unit_level_method": pm, "n_clusters": r["n_clusters"],
                             "cluster_unit": "navtest log", "n_scenes": r["n_scenes"], "n_units": r["n_units"], "n_paired_missing": 0})
        ori = o(od, src="cross_vla_temporal_replication/analysis.json CDE.all_units")
        CMP.append({"contrast_id": cid, "experiment": "exp33_impromptu_cde", "group": "all", "metric": "amplification", "orig_estimate": ori["estimate"],
                    "orig_ci95_lo": ori["ci95"][0], "orig_ci95_hi": ori["ci95"][1], "orig_p": ori["p"], "orig_method": "unit mean; log bootstrap 2000 seed 0; McNemar",
                    "orig_n_units": ori["n"], "orig_source": ori["source"], "new_unit_estimate": r["unit"]["estimate"],
                    "unit_estimate_matches": abs(r["unit"]["estimate"] - ori["estimate"]) < 1e-9, "new_unit_level_p": pu,
                    "new_scene_estimate": r["scene"]["estimate"], "new_scene_ci95_lo": r["scene"]["ci95"][0], "new_scene_ci95_hi": r["scene"]["ci95"][1],
                    "new_scene_p_signflip": r["scene"]["p_signflip"], "new_scene_p_cluster_t": r["scene"]["p_cluster_t"],
                    "new_log_estimate": r["log"]["estimate"], "new_log_ci95_lo": r["log"]["ci95"][0], "new_log_ci95_hi": r["log"]["ci95"][1],
                    "new_log_p_signflip": r["log"]["p_signflip"], "n_clusters": r["n_clusters"], "n_scenes": r["n_scenes"], "n_units": r["n_units"]})
        full = r["scene"]["estimate"]
        for j, lg in enumerate(r["logs"]):
            sub = df[df.log != lg]; rr = log_cluster_effects(sub, SEED + 8000 + 100 * i + j, b=B_LOO, mc=MC_LOO)
            LOO_OUT.append({"contrast_id": cid, "experiment": "exp33_impromptu_cde", "dropped_log": lg,
                            "dropped_log_n_scenes": int(df[df.log == lg].scene.nunique()), "dropped_log_n_units": int((df.log == lg).sum()),
                            "dropped_log_scene_effect_mean": float(df[df.log == lg].groupby("scene")["d"].mean().mean()),
                            "full_estimate": full, "loo_estimate": rr["scene"]["estimate"], "influence": full - rr["scene"]["estimate"],
                            "loo_ci95_lo": rr["scene"]["ci95"][0], "loo_ci95_hi": rr["scene"]["ci95"][1], "loo_p_signflip": rr["scene"]["p_signflip"],
                            "loo_signflip_method": rr["scene"]["sf_method"], "loo_n_clusters": rr["n_clusters"], "loo_n_scenes": rr["n_scenes"],
                            "loo_n_units": rr["n_units"], "loo_bootstrap_B": B_LOO})
        log("Impromptu done", cid)

    # ------------------------------------------------ SpatialVLA closed loop
    cl = a34["closed_loop"]
    tasks = sorted({k[0] for k in EPS})
    seeds = {t: sorted({k[1] for k in EPS if k[0] == t}) for t in tasks}
    cl_specs = []
    for dr in ("opposite", "perp_left"):
        cl_specs.append((f"SV-CL Feedback-Corrected {dr}", f"feedback_{dr}", f"corrected_{dr}", o(cl[f"corrected_vs_feedback_{dr}"], neg=True,
                         src="cross_domain_temporal_replication/analysis.json closed_loop (sign flipped)")))
        cl_specs.append((f"SV-CL Reverse-Natural {dr}", f"reverse_{dr}", "natural", o(cl["vs_natural"][f"reverse_{dr}"],
                         src="cross_domain_temporal_replication/analysis.json closed_loop.vs_natural")))
    for i, (cid, a, b, ori) in enumerate(cl_specs):
        ep = {t: np.array([EPS[(t, s, a)] - EPS[(t, s, b)] for s in seeds[t]]) for t in tasks}
        r = strat_episode_effects(ep, SEED + 900 + 10 * i)
        A = np.concatenate([[EPS[(t, s, a)] for s in seeds[t]] for t in tasks]); Bv = np.concatenate([[EPS[(t, s, b)] for s in seeds[t]] for t in tasks])
        pu, pm = unit_level_test(A, Bv, True)
        ROWS_OUT.append({"contrast_id": cid, "family": "SpatialVLA primary", "model": "SpatialVLA-4B", "experiment": "exp34_closed_loop",
                         "group": "pooled (task-equal)", "cond_a": a, "cond_b": b, "metric": "success", "weighting": "episode (task-stratified)",
                         "estimate": r["estimate"], "ci95_lo": r["ci95"][0], "ci95_hi": r["ci95"][1], "ci90_lo": r["ci90"][0], "ci90_hi": r["ci90"][1],
                         "ci_method": f"task-stratified episode bootstrap B={B}", "p_signflip": r["p_signflip"], "signflip_method": r["sf_method"],
                         "p_unit_level": pu, "unit_level_method": pm + " (episodes)", "n_clusters": r["n_clusters"], "cluster_unit": "episode (task, seed)",
                         "n_scenes": None, "n_units": r["n_clusters"], "equivalence_margin": 0.10})
        for t in tasks:
            rt = strat_episode_effects({t: ep[t]}, SEED + 950 + 10 * i + tasks.index(t))
            At = np.array([EPS[(t, s, a)] for s in seeds[t]]); Bt = np.array([EPS[(t, s, b)] for s in seeds[t]])
            put, pmt = unit_level_test(At, Bt, True)
            ROWS_OUT.append({"contrast_id": cid, "family": "SpatialVLA per-task (secondary)", "model": "SpatialVLA-4B", "experiment": "exp34_closed_loop",
                             "group": t, "cond_a": a, "cond_b": b, "metric": "success", "weighting": "episode",
                             "estimate": rt["estimate"], "ci95_lo": rt["ci95"][0], "ci95_hi": rt["ci95"][1], "ci90_lo": rt["ci90"][0], "ci90_hi": rt["ci90"][1],
                             "ci_method": f"episode bootstrap within task B={B}", "p_signflip": rt["p_signflip"], "signflip_method": rt["sf_method"],
                             "p_unit_level": put, "unit_level_method": pmt + " (episodes)", "n_clusters": rt["n_clusters"], "cluster_unit": "episode (task, seed)",
                             "n_units": rt["n_clusters"], "equivalence_margin": 0.10})
        CMP.append({"contrast_id": cid, "experiment": "exp34_closed_loop", "group": "pooled", "metric": "success", "orig_estimate": ori["estimate"],
                    "orig_ci95_lo": ori["ci95"][0], "orig_ci95_hi": ori["ci95"][1], "orig_p": ori["p"],
                    "orig_method": "episode mean; episode bootstrap 2000 seed 0 UNSTRATIFIED; McNemar", "orig_n_units": ori["n"], "orig_source": ori["source"],
                    "new_unit_estimate": r["estimate"], "unit_estimate_matches": abs(r["estimate"] - ori["estimate"]) < 1e-9, "new_unit_level_p": pu,
                    "new_scene_estimate": r["estimate"], "new_scene_ci95_lo": r["ci95"][0], "new_scene_ci95_hi": r["ci95"][1],
                    "new_scene_p_signflip": r["p_signflip"], "n_clusters": r["n_clusters"], "n_units": r["n_clusters"],
                    "note": "new = task-stratified episode bootstrap; 'scene' columns hold the episode-level primary estimate"})
        log("closed loop done", cid, r["estimate"], r["ci95"], r["p_signflip"])

    # ------------------------------------------------ SpatialVLA token level
    tk = a34["token_level"]
    tok_specs = [("SV-TOK Correction (recent_ref-normal)", "SpatialVLA primary", "recent_ref", "normal", False, tk["all"]["vs_normal"]["recent_ref"]["amp"]),
                 ("SV-TOK Reverse (reverse-full_ref)", "SpatialVLA primary", "reverse", "full_ref", False, tk["all"]["reverse_vs_full_ref"]["amp"]),
                 ("SV-TOK Full-ref-normal", "SpatialVLA secondary", "full_ref", "normal", False, tk["all"]["vs_normal"]["full_ref"]["amp"]),
                 ("SV-TOK Direction-Magnitude", "SpatialVLA secondary", "dir_wrong_mag_ok", "dir_ok_mag_wrong", True,
                  tk["motion"]["dir_wrong_mag_ok - dir_ok_mag_wrong"]["amp"])]
    for i, (cid, fam, a, b, motion_only, od) in enumerate(tok_specs):
        us = [u for u in TOK if (u["motion_unit"] or not motion_only)]
        byep = defaultdict(list)
        for u in us:
            byep[u["cl"]].append(u[a]["amp"] - u[b]["amp"])
        ep_t = {t: sorted(k for k in byep if k[0] == t) for t in tasks}

        def run(tsel, seed):
            ev = {t: np.array([np.mean(byep[k]) for k in ep_t[t]]) for t in tsel}
            wt = {t: (np.array([np.sum(byep[k]) for k in ep_t[t]]), np.array([len(byep[k]) for k in ep_t[t]], float)) for t in tsel}
            return strat_episode_effects(ev, seed, weights=wt)
        r = run(tasks, SEED + 1200 + 10 * i)
        A = np.array([u[a]["amp"] for u in us]); Bv = np.array([u[b]["amp"] for u in us]); pu, pm = unit_level_test(A, Bv, True)
        n_frames = len({u["frame"] for u in us})
        for w, rr in (("episode-weighted (primary)", r), ("unit-weighted", r["unit"])):
            ROWS_OUT.append({"contrast_id": cid, "family": fam, "model": "SpatialVLA-4B", "experiment": "exp34_token_level", "group": "pooled (task-equal)",
                             "cond_a": a, "cond_b": b, "metric": "token amplification (A>=1)", "weighting": w, "estimate": rr["estimate"],
                             "ci95_lo": rr["ci95"][0], "ci95_hi": rr["ci95"][1], "ci90_lo": rr["ci90"][0], "ci90_hi": rr["ci90"][1],
                             "ci_method": f"task-stratified episode bootstrap B={B}", "p_signflip": rr["p_signflip"], "signflip_method": rr["sf_method"],
                             "p_unit_level": pu, "unit_level_method": pm, "n_clusters": r["n_clusters"], "cluster_unit": "episode (task, seed)",
                             "n_scenes": n_frames, "n_units": len(us), "equivalence_margin": 0.02})
        for t in tasks:
            rt = run([t], SEED + 1300 + 10 * i + tasks.index(t))
            ut = [u for u in us if u["cl"][0] == t]
            put, pmt = unit_level_test([u[a]["amp"] for u in ut], [u[b]["amp"] for u in ut], True)
            for w, rr in (("episode-weighted (primary)", rt), ("unit-weighted", rt["unit"])):
                ROWS_OUT.append({"contrast_id": cid, "family": "SpatialVLA per-task (secondary)", "model": "SpatialVLA-4B", "experiment": "exp34_token_level",
                                 "group": t, "cond_a": a, "cond_b": b, "metric": "token amplification (A>=1)", "weighting": w, "estimate": rr["estimate"],
                                 "ci95_lo": rr["ci95"][0], "ci95_hi": rr["ci95"][1], "ci90_lo": rr["ci90"][0], "ci90_hi": rr["ci90"][1],
                                 "ci_method": f"episode bootstrap within task B={B}", "p_signflip": rr["p_signflip"], "signflip_method": rr["sf_method"],
                                 "p_unit_level": put, "unit_level_method": pmt, "n_clusters": rt["n_clusters"], "cluster_unit": "episode (task, seed)",
                                 "n_scenes": len({u["frame"] for u in ut}), "n_units": len(ut), "equivalence_margin": 0.02})
        ori = o(od, src="cross_domain_temporal_replication/analysis.json token_level")
        CMP.append({"contrast_id": cid, "experiment": "exp34_token_level", "group": "pooled", "metric": "token amplification", "orig_estimate": ori["estimate"],
                    "orig_ci95_lo": ori["ci95"][0], "orig_ci95_hi": ori["ci95"][1], "orig_p": ori["p"],
                    "orig_method": "unit mean; episode-cluster bootstrap 2000 seed 0 UNSTRATIFIED; unit-level McNemar", "orig_n_units": ori["n"],
                    "orig_source": ori["source"], "new_unit_estimate": r["unit"]["estimate"], "unit_estimate_matches": abs(r["unit"]["estimate"] - ori["estimate"]) < 1e-9,
                    "new_unit_level_p": pu, "new_scene_estimate": r["estimate"], "new_scene_ci95_lo": r["ci95"][0], "new_scene_ci95_hi": r["ci95"][1],
                    "new_scene_p_signflip": r["p_signflip"], "new_log_estimate": r["unit"]["estimate"], "new_log_ci95_lo": r["unit"]["ci95"][0],
                    "new_log_ci95_hi": r["unit"]["ci95"][1], "new_log_p_signflip": r["unit"]["p_signflip"], "n_clusters": r["n_clusters"],
                    "n_scenes": n_frames, "n_units": len(us),
                    "note": "scene columns = episode-weighted primary; log columns = unit-weighted with task-stratified episode bootstrap"})
        log("token done", cid, r["estimate"], r["ci95"], r["p_signflip"])

    # ------------------------------------------------ Holm and equivalence
    pe = pd.DataFrame(ROWS_OUT)
    pe["p_holm"] = np.nan
    prim_av = pe[(pe.family == "AutoVLA primary") & (pe.weighting == "scene-weighted (primary)")]
    ps = list(prim_av.p_signflip.values) + [1.0]           # Reverse-Full untestable -> p = 1, m = 3
    adj = holm(ps)
    pe.loc[prim_av.index, "p_holm"] = adj[:len(prim_av)]
    pe.loc[(pe.contrast_id == "AV-P2 Reverse-Full"), "p_holm"] = np.nan
    sv = pe[(pe.family == "SpatialVLA primary") & pe.weighting.isin(["episode (task-stratified)", "episode-weighted (primary)"])]
    pe.loc[sv.index, "p_holm"] = holm(sv.p_signflip.values)
    im = pe[(pe.experiment == "exp33_impromptu_cde") & (pe.weighting == "scene-weighted (primary)")]
    pe.loc[im.index, "p_holm"] = holm(im.p_signflip.values)
    pe.loc[pe.contrast_id == "AV-S GeoNN(GT)-RecentGT [equivalence]", "equivalence_margin"] = 0.05

    def eq(r):
        mgn = r.get("equivalence_margin")
        if mgn is None or not np.isfinite(mgn) or not np.isfinite(r.get("ci90_lo", np.nan)):
            return None
        if r["ci90_lo"] > -mgn and r["ci90_hi"] < mgn:
            return f"equivalent within +-{mgn:g} (90% CI inside)"
        if r["ci95_lo"] > mgn or r["ci95_hi"] < -mgn:
            return f"effect exceeds margin +-{mgn:g}"
        return f"not established (90% CI crosses +-{mgn:g})"
    pe["equivalence_verdict"] = pe.apply(eq, axis=1)
    pe.to_csv(os.path.join(OUT, "paired_effects.csv"), index=False)
    pd.DataFrame(LOO_OUT).to_csv(os.path.join(OUT, "loo_logs.csv"), index=False)
    cmp = pd.DataFrame(CMP)
    # attach Holm p of new primary
    hp = pe[pe.p_holm.notna()].groupby("contrast_id").p_holm.first()
    cmp["new_p_holm"] = cmp.contrast_id.map(hp)
    cmp["orig_sig_0.05"] = cmp.orig_p.apply(lambda p: None if p is None or not np.isfinite(p) else bool(p < 0.05))
    cmp["new_sig_0.05_signflip"] = cmp.new_scene_p_signflip.apply(lambda p: None if p is None or not np.isfinite(p) else bool(p < 0.05))
    cmp["orig_ci_excludes_0"] = (cmp.orig_ci95_lo > 0) | (cmp.orig_ci95_hi < 0)
    cmp["new_scene_ci_excludes_0"] = (cmp.new_scene_ci95_lo > 0) | (cmp.new_scene_ci95_hi < 0)
    cmp["p_ratio_new_over_orig"] = pd.to_numeric(cmp.new_scene_p_signflip, errors="coerce") / pd.to_numeric(cmp.orig_p, errors="coerce")
    cmp.to_csv(os.path.join(OUT, "comparison_with_original.csv"), index=False)

    # ------------------------------------------------ condition summary
    CS = []
    for exp, conds in (("exp07_action_history_causal", E7_CONDS), ("exp11_prev_action_identity", E11_CONDS), ("exp32_motion_semantics", E32_CONDS)):
        for g in ("A-", "A+"):
            for c in conds:
                df = pd.DataFrame([{"log": u["log"], "scene": u["scene"], "d": u[c]["amplification"]} for u in W[exp].values()
                                   if u["group"] == g and u[c]["amplification"] is not None])
                r = log_cluster_effects(df, SEED + 3000, b=2000, with_tests=False)
                CS.append({"model": "AutoVLA", "experiment": exp, "group": g, "condition": c, "metric": "amplification",
                           "rate_scene_weighted": r["scene"]["estimate"], "ci95_lo": r["scene"]["ci95"][0], "ci95_hi": r["scene"]["ci95"][1],
                           "rate_unit_weighted": r["unit"]["estimate"], "rate_log_equal": r["log"]["estimate"], "n_clusters": r["n_clusters"],
                           "cluster_unit": "navtest log", "n_scenes": r["n_scenes"], "n_units": r["n_units"], "note": OOD_NOTE.get(c)})
    for row in AT.ROWS:
        df = pd.DataFrame([{"log": x["log"], "scene": x["scene"], "d": x[row]["amp"]} for x in IMP])
        r = log_cluster_effects(df, SEED + 3100, b=2000, with_tests=False)
        CS.append({"model": "Impromptu-VLA-3B", "experiment": "exp33_impromptu_cde", "group": "all", "condition": row, "metric": "amplification",
                   "rate_scene_weighted": r["scene"]["estimate"], "ci95_lo": r["scene"]["ci95"][0], "ci95_hi": r["scene"]["ci95"][1],
                   "rate_unit_weighted": r["unit"]["estimate"], "rate_log_equal": r["log"]["estimate"], "n_clusters": r["n_clusters"],
                   "cluster_unit": "navtest log", "n_scenes": r["n_scenes"], "n_units": r["n_units"]})
    for c in JOIN["svla_closed_loop"]["conditions"]:
        for t in tasks + ["pooled"]:
            ts = tasks if t == "pooled" else [t]
            ev = {tt: np.array([EPS[(tt, s, c)] for s in seeds[tt]]) for tt in ts}
            r = strat_episode_effects(ev, SEED + 3200, b=2000, mc=1000)
            CS.append({"model": "SpatialVLA-4B", "experiment": "exp34_closed_loop", "group": t, "condition": c, "metric": "success",
                       "rate_scene_weighted": r["estimate"], "ci95_lo": r["ci95"][0], "ci95_hi": r["ci95"][1], "rate_unit_weighted": r["estimate"],
                       "n_clusters": r["n_clusters"], "cluster_unit": "episode (task, seed)", "n_units": r["n_clusters"],
                       "note": "rate_scene_weighted = task-equal episode mean"})
    for row in SV_ROWS:
        for t in tasks + ["pooled"]:
            ts = tasks if t == "pooled" else [t]
            byep = defaultdict(list)
            for u in TOK:
                if u["cl"][0] in ts:
                    byep[u["cl"]].append(u[row]["amp"])
            ev = {tt: np.array([np.mean(byep[k]) for k in sorted(byep) if k[0] == tt]) for tt in ts}
            wt = {tt: (np.array([np.sum(byep[k]) for k in sorted(byep) if k[0] == tt]), np.array([len(byep[k]) for k in sorted(byep) if k[0] == tt], float)) for tt in ts}
            r = strat_episode_effects(ev, SEED + 3300, b=2000, mc=1000, weights=wt)
            CS.append({"model": "SpatialVLA-4B", "experiment": "exp34_token_level", "group": t, "condition": row, "metric": "token amplification (A>=1)",
                       "rate_scene_weighted": r["estimate"], "ci95_lo": r["ci95"][0], "ci95_hi": r["ci95"][1], "rate_unit_weighted": r["unit"]["estimate"],
                       "n_clusters": r["n_clusters"], "cluster_unit": "episode (task, seed)", "n_units": int(sum(len(v) for v in byep.values())),
                       "note": "rate_scene_weighted = episode-weighted (task-equal)"})
    pd.DataFrame(CS).to_csv(os.path.join(OUT, "condition_summary.csv"), index=False)

    # ------------------------------------------------ units table
    U = pd.DataFrame(LONG)
    for c in ("seed",):
        U[c] = U[c].astype("Int64")
    U.to_parquet(os.path.join(OUT, "units.parquet"), index=False)
    json.dump({"join": JOIN, "runtime_s": time.time() - T0}, open(os.path.join(OUT, "join_report.json"), "w"), indent=1, default=str)
    log("all written", len(U), "unit rows")


if __name__ == "__main__":
    main()
