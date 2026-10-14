#!/usr/bin/env python
"""
P0-E (review_followup3, 2026-10-14): OpenVLA-7B / LIBERO-Spatial (exp 28) re-analysis with INITIAL-STATE clusters.

*** POST-HOC re-analysis of existing unit-level records. CPU only, no model inference, no existing file modified. ***
Inputs (read only):
  outputs/cross_vla_replication/rollouts/{phaseA,phaseA_rep,phaseB}/episodes.jsonl   (closed-loop episodes)
  outputs/cross_vla_replication/token_feedback/token_feedback.jsonl                  (phase C, offline frames)
  outputs/cross_vla_replication/analysis.json                                        (original results)
  scripts/cross_vla/analyze_cross_vla.py (imported for exact reproduction; its main() is not called)
  /root/VLA/openvla/LIBERO/libero/libero/init_files/libero_spatial/*.pruned_init     (initial physical states)
  outputs/review_followup3/P0/_external/openvla-7b-finetuned-libero-spatial_config.json (norm_stats; HF commit
      962318ce = the commit in the local HF cache refs/main used for the runs)
  /root/VLA/openvla/site/robosuite/controllers/config/osc_pose.json                  (OSC_POSE scaling used by LIBERO)

1. unit records: episode -> (task, init) -> initial state (LIBERO init vector, sha1) -> task; pairing check (every
   state has exactly one episode in each of the 8 conditions), sign rule check (delta > 0 for even init).
2. success contrasts (C-F at |delta| 8/24; every perturbation - natural; - natural_rep; natural_rep - natural):
   OLD (exact reproduction of analyze_cross_vla.paired: episode mean, task-cluster bootstrap 2,000 Random(0), task
   sign-flip on task means, McNemar) side by side with NEW:
     state cluster (100 initial states; 1 paired episode each): mean, cluster bootstrap B = 10,000, exact sign-flip
     p (= exact two-sided sign test on discordant states), Monte Carlo sign-flip inversion 95% CI;
     task-stratified bootstrap (states resampled within task; equal-task weights, 10 states per task => same point);
     per-task estimates (10 states each: exact 2^10 sign-flip, exact inversion CI) -- low precision by design.
   Holm (post-hoc families): F1 = {C-F d8, C-F d24}; F2 = 6 perturbation - natural; F3 = 6 perturbation - natural_rep.
3. phase C "92.6% [89.8, 94.4]": exact definition from analyze_cross_vla.main (reproduced to machine precision),
   then initial-state cluster CI, per-dimension decomposition, physical units (m, rad) via the dataset q01/q99
   normalisation and the OSC_POSE input->output scaling, gripper handling (excluded from the metric; binarised in env).

Run: nice -n 19 /root/miniforge3/envs/autovla/bin/python p0_e_openvla.py
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from math import comb

import numpy as np
import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
sys.path.insert(0, f"{POC}/scripts/review_followup2")
sys.path.insert(0, f"{POC}/scripts/cross_vla")
import stats_core as SC  # noqa: E402
import b_direct_effects as BDE  # noqa: E402  (helpers only)
import analyze_cross_vla as ACV  # noqa: E402  (original analysis functions; main() not called)

ROOT = f"{POC}/outputs/cross_vla_replication"
OUT = f"{POC}/outputs/review_followup3/P0"
CFG = f"{OUT}/_external/openvla-7b-finetuned-libero-spatial_config.json"
OSC = "/root/VLA/openvla/site/robosuite/controllers/config/osc_pose.json"
INIT = "/root/VLA/openvla/LIBERO/libero/libero/init_files/libero_spatial"
TASKS = ["pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
         "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"]   # libero_suite_task_map order
SEED, B, MC = 20261014, 10_000, 200_000
STATUS = "POST-HOC (OpenVLA; not affected by AutoVLA parity; OpenVLA intervention path not parity-verified, no new run)"
DIMS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def load_init_states():
    import torch
    out, vecs = {}, []
    for t, n in enumerate(TASKS):
        x = np.asarray(torch.load(os.path.join(INIT, n + ".pruned_init"), weights_only=False))
        for i in range(10):
            v = np.ascontiguousarray(x[i])
            out[(t, i)] = hashlib.sha1(v.tobytes()).hexdigest()[:16]
            vecs.append(v)
    A = np.stack(vecs)
    d = np.abs(A[:, None, :] - A[None, :, :]).max(-1)
    np.fill_diagonal(d, np.inf)
    return out, float(d.min())


def exact_sign_test(d):
    """two-sided exact sign-flip p for per-cluster differences in {-1, 0, 1} (= exact sign test / McNemar)."""
    d = np.asarray(d)
    n = int(np.sum(d != 0))
    if n == 0:
        return 1.0, 0
    s = abs(int(d.sum()))
    # S = 2K - n with K ~ Bin(n, 1/2);  P(|S| >= s)
    p = sum(comb(n, k) for k in range(n + 1) if abs(2 * k - n) >= s) / 2 ** n
    return float(min(1.0, p)), n


def state_level(d, tasks, seed):
    d = np.asarray(d, float)
    N, D = d.copy(), np.ones_like(d)
    th = float(d.mean())
    bs = SC.cluster_bootstrap(N, D, B, seed)
    sb = BDE.strat_boot(np.asarray(tasks), N, D, B=B, seed=seed + 1)
    p_ex, n_inf = exact_sign_test(d)
    p_mc = SC.signflip_test(N, "two-sided", SC.EXACT_MAX, MC, seed + 2)["p"]
    E = np.random.default_rng(seed + 3).integers(0, 2, size=(20_000, len(N))).astype(float) * 2 - 1
    inv = BDE.mc_inv_ci(N, D, th, 0.95, E)
    t10 = SC.tost(N, D, 0.10, "signflip", B=200, seed=seed + 4, reps=MC)
    return {"estimate": th, "boot_ci95": SC.pct_ci(bs), "strat_boot_ci95": SC.pct_ci(sb), "p_exact": p_ex,
            "p_mc": p_mc, "n_informative_states": n_inf, "sf_inv_ci95_mc": inv, "p_equiv_m10": t10["p_equiv"],
            "p_min_attainable": 2.0 ** (1 - n_inf) if n_inf else 1.0}


def main():
    os.makedirs(OUT, exist_ok=True)
    checks = {}
    # ------------------------------------------------------------------------------------------- 1. unit records
    sid, min_state_dist = load_init_states()
    checks["init_states_distinct"] = {"n_states": len(set(sid.values())), "min_pairwise_maxabs_diff": min_state_dist}
    ep = []
    for ph, fn in (("A", "phaseA"), ("A_rep", "phaseA_rep"), ("B", "phaseB")):
        for l in open(f"{ROOT}/rollouts/{fn}/episodes.jsonl"):
            r = json.loads(l)
            cond = "natural_rep" if ph == "A_rep" else r["cond"]
            ep.append({"phase": ph, "cond": cond, "task": r["task"], "init": r["init"], "delta": r.get("delta", 0),
                       "success": int(bool(r["success"])), "steps": r["steps"], "n_rec": len(r["rec"])})
    E = pd.DataFrame(ep)
    E["initial_state_id"] = [f"t{t}_i{i}:{sid[(t, i)]}" for t, i in zip(E.task, E.init)]
    E["task_name"] = E.task.map(dict(enumerate(TASKS)))
    per = E.groupby(["task", "init"]).cond.agg(["nunique", "size"])
    checks["pairing"] = {"n_states": len(per), "conds_per_state": sorted(per["nunique"].unique().tolist()),
                         "episodes_per_state": sorted(per["size"].unique().tolist()),
                         "episodes_total": len(E), "conditions": sorted(E.cond.unique())}
    assert len(per) == 100 and (per["nunique"] == 8).all() and (per["size"] == 8).all()
    pb = E[E.phase == "B"]
    sign_ok = ((np.sign(pb.delta) == np.where(pb.init % 2 == 0, 1, -1)) & (pb.delta.abs() == pb.cond.str.split("_d").str[1].astype(int))).all()
    checks["delta_sign_rule_even_init_positive"] = bool(sign_ok)
    assert sign_ok
    E.to_csv(f"{OUT}/OPENVLA_episode_records.csv", index=False)

    # ------------------------------------------------------------------------------------------- 2. success contrasts
    Ed = {(r.task, r.init, r.cond): bool(r.success) for r in E.itertuples()}
    old = json.load(open(f"{ROOT}/analysis.json"))
    W = E.pivot_table(index=["task", "init"], columns="cond", values="success").reset_index()
    specs = [("C-F d8", "corrected_d8", "feedback_d8", "F1 {C-F d8, C-F d24} (post-hoc Holm m=2)"),
             ("C-F d24", "corrected_d24", "feedback_d24", "F1 {C-F d8, C-F d24} (post-hoc Holm m=2)")]
    for a in (8, 24):
        for m in ("feedback", "corrected", "reverse"):
            specs.append((f"{m[0].upper()}-N d{a}", f"{m}_d{a}", "natural", "F2 perturbation - natural (post-hoc Holm m=6)"))
    for a in (8, 24):
        for m in ("feedback", "corrected", "reverse"):
            specs.append((f"{m[0].upper()}-Nrep d{a}", f"{m}_d{a}", "natural_rep", "F3 perturbation - natural_rep (post-hoc Holm m=6)"))
    specs.append(("Nrep-N", "natural_rep", "natural", "baseline variability (no family)"))
    rows, task_rows = [], []
    seed = SEED
    for cid, a, b, fam in specs:
        # OLD exact reproduction (episode level, task bootstrap Random(0), task sign-flip, McNemar)
        po = ACV.paired(Ed, a, b)
        ref = None
        if a.startswith("corrected") and b.startswith("feedback"):
            o = old[f"feedback_vs_corrected_d{a.split('_d')[1]}"]
            ref = {"diff": -o["diff"], "ci": [-o["ci95_task_boot"][1], -o["ci95_task_boot"][0]], "p_mc": o["p_mcnemar"], "p_perm": o["p_perm_tasks"]}
        elif b == "natural" and a in old["vs_natural"]:
            o = old["vs_natural"][a]; ref = {"diff": o["diff"], "ci": o["ci95_task_boot"], "p_mc": o["p_mcnemar"], "p_perm": o["p_perm_tasks"]}
        elif b == "natural_rep" and a in old.get("vs_natural_rep", {}):
            o = old["vs_natural_rep"][a]; ref = {"diff": o["diff"], "ci": o["ci95_task_boot"], "p_mc": o["p_mcnemar"], "p_perm": o["p_perm_tasks"]}
        elif cid == "Nrep-N":
            o = old["natural_rep_vs_natural"]; ref = {"diff": o["diff"], "ci": o["ci95_task_boot"], "p_mc": o["p_mcnemar"], "p_perm": o["p_perm_tasks"]}
        if ref is not None:
            assert abs(po["diff"] - ref["diff"]) < 1e-12 and np.allclose(po["ci95_task_boot"], ref["ci"], atol=1e-9) \
                and abs(po["p_mcnemar"] - ref["p_mc"]) < 1e-12 and abs(po["p_perm_tasks"] - ref["p_perm"]) < 1e-12, (cid, po, ref)
        # NEW
        d = (W[a] - W[b]).values
        tk = W.task.values
        r = state_level(d, tk, seed); seed += 10
        eq_task = float(np.mean([d[tk == t].mean() for t in range(10)]))
        assert abs(eq_task - r["estimate"]) < 1e-12           # 10 states per task -> equal-task == state mean
        # arm-sign swap
        r_sw = exact_sign_test(-d)
        assert abs(r_sw[0] - r["p_exact"]) < 1e-12
        n10, n01 = int(((W[a] == 1) & (W[b] == 0)).sum()), int(((W[a] == 0) & (W[b] == 1)).sum())
        assert abs(r["p_exact"] - po["p_mcnemar"]) < 1e-9 or n10 + n01 == 0, (cid, r["p_exact"], po["p_mcnemar"])
        rows.append({"row_id": f"P0E.succ.{cid.replace(' ', '_')}", "contrast": cid, "cond_a": a, "cond_b": b,
                     "definition": f"success({a}) - success({b}), paired by initial state", "family": fam,
                     "succ_a": int(W[a].sum()), "n_a": len(W), "succ_b": int(W[b].sum()), "n_b": len(W),
                     "rate_a": W[a].mean(), "rate_b": W[b].mean(), "a_only": n10, "b_only": n01,
                     "n_initial_states": len(W), "n_tasks": 10, "states_per_task": 10,
                     "n_informative_states": r["n_informative_states"], "p_min_attainable": r["p_min_attainable"],
                     "estimate": r["estimate"], "estimate_equal_task": eq_task,
                     "state_boot_ci95_lo": r["boot_ci95"][0], "state_boot_ci95_hi": r["boot_ci95"][1],
                     "state_sf_inv_ci95_lo": r["sf_inv_ci95_mc"][0], "state_sf_inv_ci95_hi": r["sf_inv_ci95_mc"][1],
                     "task_strat_boot_ci95_lo": r["strat_boot_ci95"][0], "task_strat_boot_ci95_hi": r["strat_boot_ci95"][1],
                     "p_signflip_state_exact": r["p_exact"], "p_signflip_state_mc": r["p_mc"],
                     "p_equiv_m10_posthoc": r["p_equiv_m10"],
                     "OLD_estimate": po["diff"], "OLD_task_boot_ci95_lo": po["ci95_task_boot"][0],
                     "OLD_task_boot_ci95_hi": po["ci95_task_boot"][1], "OLD_p_mcnemar": po["p_mcnemar"],
                     "OLD_p_perm_tasks": po["p_perm_tasks"],
                     "OLD_reproduced_from_analysis_json": ref is not None,
                     "OLD_method": "episode mean; task-cluster bootstrap 2,000 (random.Random(0)); task-mean sign-flip 2^10; exact McNemar",
                     "status": STATUS, "source_path": f"{OUT}/OPENVLA_episode_records.csv (from {ROOT}/rollouts/*/episodes.jsonl)"})
        for t in range(10):
            dt = d[tk == t].astype(float)
            p_t = SC.signflip_test(dt, "two-sided", SC.EXACT_MAX)["p"]
            ci_t = SC.signflip_ci(dt, np.ones_like(dt), float(dt.mean()), 0.95)
            task_rows.append({"row_id": f"P0E.succ_task.{cid.replace(' ', '_')}.t{t}", "contrast": cid, "task": t,
                              "task_name": TASKS[t], "n_states": len(dt), "succ_a": int(W[a][tk == t].sum()),
                              "succ_b": int(W[b][tk == t].sum()), "estimate": float(dt.mean()), "p_signflip_exact_2^10": p_t,
                              "sf_inv_ci95_lo": ci_t[0], "sf_inv_ci95_hi": ci_t[1],
                              "n_informative_states": int(np.sum(dt != 0)),
                              "precision_note": "10 initial states per task: low precision; descriptive",
                              "status": STATUS})
    S = pd.DataFrame(rows)
    S["p_holm_state"] = np.nan
    for fam, g in S.groupby("family"):
        if fam.startswith(("F1", "F2", "F3")):
            S.loc[g.index, "p_holm_state"] = SC.holm(g.p_signflip_state_exact.values)
    T = pd.DataFrame(task_rows)

    # ------------------------------------------------------------------------------------------- 3. phase C
    cfg = json.load(open(CFG))
    st = cfg["norm_stats"]["libero_spatial"]["action"]
    q01, q99, mask = np.array(st["q01"]), np.array(st["q99"]), np.array(st["mask"])
    osc = json.load(open(OSC))
    out_max = np.array(osc["output_max"] + [1.0])                  # m (xyz), rad (rpy); gripper binarised
    bins = np.linspace(-1, 1, cfg.get("n_action_bins", 256))
    centers = (bins[:-1] + bins[1:]) / 2.0                         # 255 centres, index = bin
    per_bin_norm = (q99 - q01) / 255.0                             # env-input units per bin (mask dims)
    per_bin_phys = per_bin_norm * out_max / osc["input_max"]       # m or rad per bin (commanded per-step delta)

    def unnorm(b):
        n = centers[np.asarray(b)]
        return np.where(mask, 0.5 * (n + 1) * (q99 - q01) + q01, n)

    def env_grip(b):
        g = centers[int(b)]                                        # mask False -> raw normalised value
        return -np.sign(2 * g - 1)                                 # normalize_gripper_action(binarize) + invert

    units = []
    agg_orig = defaultdict(lambda: defaultdict(list))
    for l in open(f"{ROOT}/token_feedback/token_feedback.jsonl"):
        f = json.loads(l)
        V, ex, ctx = f["variants"], np.array(f["exec"]), np.array(f["ctx"])
        tt, ii = f["frame"].split("/")[0].split("_")
        task, init, step = int(tt[1:]), int(ii[1:]), int(f["frame"].split("/")[1][:-4])
        assert task == f["task"]
        nat = ex[0]
        for k, v in enumerate(V[1:], 1):
            dd_, dl, mode = v.split(":")
            if mode != "feedback":
                continue
            dd_, dl = int(dd_), int(dl)
            de = abs(int(ex[k, dd_]) - int(nat[dd_]))
            if de == 0:
                units.append({"frame": f["frame"], "task": task, "init": init, "step": step, "d": dd_, "delta": dl,
                              "abs_delta": abs(dl), "clipped_out": True})
                continue
            dd = np.abs(ex[k, dd_ + 1:6] - nat[dd_ + 1:6])
            amp = float(dd.max() >= de)
            agg_orig[f"d{dd_}_|{abs(dl)}|"][task].append(amp)
            dphys = np.abs(unnorm(ex[k]) - unnorm(nat))[:6] * out_max[:6]
            rec = {"frame": f["frame"], "task": task, "init": init, "step": step, "d": dd_, "delta": dl, "abs_delta": abs(dl),
                   "clipped_out": False, "delta_eff_bins": de, "D_bins": int(dd.sum()), "amplified": amp,
                   "recovered": float(dd.sum() == 0), "injected_phys": float(dphys[dd_]),
                   "injected_phys_unit": "m" if dd_ < 3 else "rad"}
            for j in range(7):
                rec[f"dbin_{DIMS[j]}"] = int(abs(int(ex[k, j]) - int(nat[j])))
            for j in range(6):
                rec[f"dphys_{DIMS[j]}"] = float(dphys[j])
            rec["gripper_env_flip"] = float(env_grip(ex[k, 6]) != env_grip(nat[6]))
            lo = [j for j in range(dd_ + 1, 3)]
            rec["down_trans_phys_m"] = float(np.linalg.norm(dphys[lo])) if lo else np.nan
            rec["down_rot_phys_rad"] = float(np.linalg.norm(dphys[3:6]))
            rec["phys_amp_translation"] = float(rec["down_trans_phys_m"] >= rec["injected_phys"]) if lo and dd_ < 3 else np.nan
            units.append(rec)
    U = pd.DataFrame(units)
    U["initial_state"] = "t" + U.task.astype(str) + "_i" + U.init.astype(str)
    U.to_csv(f"{OUT}/OPENVLA_phaseC_feedback_units.csv", index=False)
    # exact reproduction of 92.6 [89.8, 94.4]
    c_ = agg_orig["d0_|8|"]
    m_orig = float(np.mean([x for v in c_.values() for x in v]))
    ci_orig = ACV.boot_tasks(c_, 1000)
    o = old["phaseC"]["d0_|8|_feedback"]
    assert abs(m_orig - o["amplified"]) < 1e-12 and np.allclose(ci_orig, o["amplified_ci95"], atol=1e-12)
    assert sum(len(v) for v in c_.values()) == o["n"]
    checks["phaseC_92.6_reproduced"] = {"mean": m_orig, "ci95_task_boot_1000": ci_orig, "n_units": o["n"]}
    PC, DEC = [], []
    seed = SEED + 500
    for (d_, a_), g in U[~U.clipped_out].groupby(["d", "abs_delta"]):
        key = f"d{d_}_|{a_}|_feedback"
        gs = g.groupby("initial_state")
        N, D = gs.amplified.sum().values, gs.size().values.astype(float)
        bs = SC.cluster_bootstrap(N, D, B, seed)
        gt = g.groupby("task")
        bt = SC.cluster_bootstrap(gt.amplified.sum().values, gt.size().values.astype(float), B, seed + 1)
        _, clt = SC.cluster_t(N, D)
        eqs = float((N / D).mean())
        bse = SC.cluster_bootstrap(N / D, np.ones_like(N), B, seed + 2)
        row = {"row_id": f"P0E.phaseC.{key}", "key": key, "intervened_dim": DIMS[d_], "abs_delta_bins": a_,
               "definition": ("unit = (saved frame, sign of delta); frames = every 5th control step of the 100 natural "
                              "phase-A episodes; comparison = natural greedy decode of the SAME observation; "
                              f"amplified = max over dims {d_ + 1}..5 (gripper excluded) of |bin - natural bin| >= effective "
                              "|delta| (after clipping to [0,254]); units with effective delta 0 (clipped) dropped"),
               "n_units": int(len(g)), "n_units_clipped_out": int(U[(U.d == d_) & (U.abs_delta == a_) & U.clipped_out].shape[0]),
               "n_frames": g.frame.nunique(), "n_initial_states": int(len(N)), "n_tasks": g.task.nunique(),
               "frames_per_state_min": int(g.groupby("initial_state").frame.nunique().min()),
               "frames_per_state_max": int(g.groupby("initial_state").frame.nunique().max()),
               "amplified_events": int(N.sum()), "amplified_rate": float(N.sum() / D.sum()),
               "state_boot_ci95_lo": SC.pct_ci(bs)[0], "state_boot_ci95_hi": SC.pct_ci(bs)[1],
               "state_clt_ci95_lo": clt[0], "state_clt_ci95_hi": clt[1],
               "task_boot_B10000_ci95_lo": SC.pct_ci(bt)[0], "task_boot_B10000_ci95_hi": SC.pct_ci(bt)[1],
               "equal_state_weight_rate": eqs, "equal_state_ci95_lo": SC.pct_ci(bse)[0], "equal_state_ci95_hi": SC.pct_ci(bse)[1],
               "recovered_rate": float(g.recovered.mean()), "mean_D_bins": float(g.D_bins.mean()),
               "injected_phys_mean": float(g.injected_phys.mean()), "injected_phys_unit": g.injected_phys_unit.iloc[0],
               "down_trans_phys_mean_m": float(g.down_trans_phys_m.mean()), "down_rot_phys_mean_rad": float(g.down_rot_phys_rad.mean()),
               "phys_amp_translation_rate": float(g.phys_amp_translation.mean()) if g.phys_amp_translation.notna().any() else np.nan,
               "gripper_bin_changed_rate": float((g.dbin_gripper > 0).mean()), "gripper_env_flip_rate": float(g.gripper_env_flip.mean()),
               "status": STATUS, "source_path": f"{OUT}/OPENVLA_phaseC_feedback_units.csv"}
        if key in old["phaseC"]:
            oo = old["phaseC"][key]
            row.update({"OLD_amplified": oo["amplified"], "OLD_task_boot_ci95_lo": oo["amplified_ci95"][0],
                        "OLD_task_boot_ci95_hi": oo["amplified_ci95"][1], "OLD_n": oo["n"],
                        "OLD_method": "unit mean; task-cluster bootstrap 1,000 (random.Random(0))"})
            assert oo["n"] == len(g) and abs(oo["amplified"] - row["amplified_rate"]) < 1e-12
        PC.append(row)
        seed += 10
        for j in range(d_ + 1, 7):
            nm = DIMS[j]
            x = g[f"dbin_{nm}"]
            dec = {"row_id": f"P0E.dim.{key}.{nm}", "key": key, "downstream_dim": nm, "n_units": len(g),
                   "share_dbin_ge_delta": float((x >= g.delta_eff_bins).mean()) if j < 6 else np.nan,
                   "share_dbin_gt0": float((x > 0).mean()), "mean_abs_dbin": float(x.mean()),
                   "share_dim_is_argmax_of_downstream": np.nan,
                   "per_bin_physical": float(per_bin_phys[j]) if j < 6 else np.nan,
                   "physical_unit": ("m" if j < 3 else "rad") if j < 6 else "binary (env: sign(2n-1), inverted)",
                   "mean_abs_phys": float(g[f"dphys_{nm}"].mean()) if j < 6 else np.nan,
                   "p95_abs_phys": float(g[f"dphys_{nm}"].quantile(.95)) if j < 6 else np.nan,
                   "gripper_env_flip_rate": float(g.gripper_env_flip.mean()) if j == 6 else np.nan,
                   "in_amplification_metric": j < 6, "status": STATUS}
            if j < 6:
                cols = [f"dbin_{DIMS[k]}" for k in range(d_ + 1, 6)]
                arg = g[cols].values.argmax(1) + d_ + 1
                dec["share_dim_is_argmax_of_downstream"] = float((arg == j).mean())
            DEC.append(dec)
    PCd, DECd = pd.DataFrame(PC), pd.DataFrame(DEC)
    phys = pd.DataFrame({"dim": DIMS[:6], "q01": q01[:6], "q99": q99[:6], "mask": mask[:6],
                         "env_input_per_bin": per_bin_norm[:6], "osc_output_max": out_max[:6],
                         "physical_per_bin": per_bin_phys[:6], "unit": ["m"] * 3 + ["rad"] * 3,
                         "delta8_physical": 8 * per_bin_phys[:6], "delta24_physical": 24 * per_bin_phys[:6]})
    checks["physical_conversion"] = phys.to_dict("records")
    checks["gripper"] = {"q01": float(q01[6]), "q99": float(q99[6]), "mask": bool(mask[6]),
                         "handling": "excluded from the amplification metric (dims d+1..5 only); in the env the gripper "
                                     "value is binarised: a = sign(2n - 1) then inverted (to_env_action, run_libero_eval)"}
    # ------------------------------------------------------------------------------------------- write
    S.to_csv(f"{OUT}/OPENVLA_REANALYSIS.csv", index=False)
    T.to_csv(f"{OUT}/OPENVLA_REANALYSIS_per_task.csv", index=False)
    PCd.to_csv(f"{OUT}/OPENVLA_REANALYSIS_phaseC.csv", index=False)
    DECd.to_csv(f"{OUT}/OPENVLA_REANALYSIS_phaseC_dims.csv", index=False)
    phys.to_csv(f"{OUT}/OPENVLA_physical_units.csv", index=False)
    json.dump(checks, open(f"{OUT}/_checks_E.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(S[["row_id", "succ_a", "succ_b", "a_only", "b_only", "estimate", "state_boot_ci95_lo", "state_boot_ci95_hi",
             "state_sf_inv_ci95_lo", "state_sf_inv_ci95_hi", "task_strat_boot_ci95_lo", "task_strat_boot_ci95_hi",
             "p_signflip_state_exact", "p_holm_state", "p_equiv_m10_posthoc", "OLD_task_boot_ci95_lo", "OLD_task_boot_ci95_hi", "OLD_p_mcnemar"]].to_string())
    print(T[T.contrast.str.startswith("C-F")][["row_id", "succ_a", "succ_b", "estimate", "p_signflip_exact_2^10", "sf_inv_ci95_lo", "sf_inv_ci95_hi"]].to_string())
    print(PCd.drop(columns=["definition", "status", "source_path"]).T.to_string())
    print(DECd[DECd.key == "d0_|8|_feedback"].to_string())
    print(phys.to_string())
    print(json.dumps({k: v for k, v in checks.items() if k != "physical_conversion"}, indent=1, default=float))


if __name__ == "__main__":
    main()
