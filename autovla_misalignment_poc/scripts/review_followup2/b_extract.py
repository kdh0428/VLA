#!/usr/bin/env python
"""
B (review_followup2, 2026-10-13): extract per-episode and per-pair tables for the SpatialVLA direct-effect analysis.

*** POST-HOC ANALYSIS OF EXISTING DATA. CPU only, no model. *** Reads (never modifies):
  exp 34  outputs/cross_domain_temporal_replication/closed_loop/episodes.jsonl            (config r1e4, 2 directions)
  exp 35  outputs/spatialvla_feedback_protection_ablation/rollouts{,_ext}/episodes.jsonl   (6 configs, opposite)
  P1-D    outputs/review_followup/P1D_dose_control/run_20261008/rollouts/main/episodes.jsonl (streamed, 4 configs)
  clusters outputs/review_followup2/A_spatialvla_states/run_20261013/state_manifest.csv
  token table outputs/review_followup/P0C_exp35_fixed_axis/run_20261008/token_table.npz (translation token -> phys xyz)
Writes to outputs/review_followup2/B_spatialvla_direct/run_20261013/:
  episodes.csv  one row per episode (intention-to-treat: every recorded episode, success as recorded)
  pairs.csv     one row per (experiment, config, direction, task, seed, contrast a-b) with success difference and the
                B4 trajectory metrics (definitions in DIRECT_EFFECTS.md, section B4)

Trajectory metric definitions (identical to analyze_protection.pair_metrics / analyze_p1d.pair):
  d(t) = || p_a(t) - p_b(t) ||, p(t) = actual TCP position (m) recorded AFTER env.step at control step t (t = 0..79)
  tcp_mean_cm   = 100 * mean_{t=8..79} d(t)        (72 steps; denominator 72)
  tcp_area_cmstep = 100 * sum_{t=8..79} d(t)       (cm*step, same 72 steps)
  tcp_final_cm  = 100 * d(79)
  cmd_final_m   = || sum_{t=0..79} a_a(t)[:3] - sum_{t=0..79} a_b(t)[:3] ||   executed world-vector command accumulation
                  (commanded delta translation, m; a proxy, NOT the actual TCP)
  gen_suffix_*  generated-action suffix change (tokens generated after the step-1 translation token):
     P1-D ("shadow"): at each injected generation call (t = 8, 12, 16, 20) the a-episode's executed chunk vs its
                  counterfactual-context chunk decoded at the same observation (cf_exec);
     exp 34/35 ("t8"): the a- vs b-episode chunk generated at t = 8 (first injected chunk), only where the executed
                  actions at t < 8 are identical (prefix_identical), so both chunks see the same observation.
     gen_suffix_tokfrac = fraction of the 11 tokens after the step-1 translation token that differ
     gen_suffix_trans_m = sum over chunk steps 2-4 of || phys(trans_a) - phys(trans_b) || (m), mean per call
  ctx_exec_dev_trans_m (P1-D only) = sum_t || actual - counterfactual-context pre-sticky ensemble translation ||
                  of the a-episode (context-attributable executed translation deviation, m per episode)

  nice -n 19 /root/miniforge3/envs/autovla/bin/python b_extract.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
OUT = os.path.join(POC, "outputs/review_followup2/B_spatialvla_direct/run_20261013")
SRC = {
    "exp34": [os.path.join(POC, "outputs/cross_domain_temporal_replication/closed_loop/episodes.jsonl")],
    "exp35": [os.path.join(POC, "outputs/spatialvla_feedback_protection_ablation/rollouts/episodes.jsonl"),
              os.path.join(POC, "outputs/spatialvla_feedback_protection_ablation/rollouts_ext/episodes.jsonl")],
    "P1D": [os.path.join(POC, "outputs/review_followup/P1D_dose_control/run_20261008/rollouts/main/episodes.jsonl")],
}
MANIFEST = os.path.join(POC, "outputs/review_followup2/A_spatialvla_states/run_20261013/state_manifest.csv")
TOKTAB = os.path.join(POC, "outputs/review_followup/P0C_exp35_fixed_axis/run_20261008/token_table.npz")
T0, WIN_END = 8, 24
TASK_SHORT = {"google_robot_pick_coke_can": "coke", "google_robot_move_near": "move_near"}

# contrast name -> (mode a, mode b); success difference = S(a) - S(b)
CONTRASTS = {"R-N": ("reverse", "natural"), "C-F": ("corrected", "feedback"), "F-N": ("feedback", "natural"),
             "C-N": ("corrected", "natural"), "CD-FD": ("correctedD", "feedbackD"), "FD-N": ("feedbackD", "natural"),
             "CD-N": ("correctedD", "natural"), "gen-N": ("gen", "natural"), "null-N": ("null", "natural"),
             "mix-N": ("natural_mix", "natural")}

tt = np.load(TOKTAB)
ID2PHYS = {int(i): tt["phys"][k] for k, i in enumerate(tt["ids"])}


def parse_cond(exp, ep):
    """-> (config, direction, mode)"""
    c = ep["cond"]
    if exp == "exp34":
        if c in ("natural", "natural_mix", "natural_gen"):
            return "r1e4", "-", {"natural_gen": "gen"}.get(c, c)
        mode, direction = c.split("_", 1)
        return "r1e4", direction, mode
    cfg, mode = c.split(":")
    mode = mode.split("#")[0]
    if mode == "natural_gen":
        mode = "gen"
    return cfg, "opposite", mode


def compact(exp, ep):
    rec = ep["rec"]
    act = np.array([r["action"][:3] for r in rec], dtype=float)
    pos = np.array([r["pose"][:3] for r in rec], dtype=float) if "pose" in rec[0] else None
    chunks = {r["t"]: np.asarray(r["exec"], dtype=np.int64) for r in rec if "exec" in r and T0 <= r["t"] < WIN_END}
    cf = {r["t"]: np.asarray(r["cf_exec"], dtype=np.int64) for r in rec if "cf_exec" in r}
    ctxdev = float(sum(np.linalg.norm(r["ctx_dev"][:3]) for r in rec if "ctx_dev" in r)) if exp == "P1D" else np.nan
    restore = ep.get("restore")
    return {"success": float(ep["success"]), "steps": int(ep["steps"]), "act": act, "pos": pos, "chunks": chunks,
            "cf": cf, "ctxdev": ctxdev, "restore_ok": (None if restore is None else bool(restore["ok"])),
            "full_act_prefix": act[:T0].copy()}


def suffix_change(ca, cb):
    """ca, cb: (4, 3) token arrays. -> (fraction of the 11 post-step-1-translation tokens that differ,
    sum over steps 2-4 of translation phys distance in m)."""
    fa, fb = ca.reshape(-1)[1:], cb.reshape(-1)[1:]
    frac = float(np.mean(fa != fb))
    tr = float(sum(np.linalg.norm(ID2PHYS.get(int(ca[k, 0]), np.zeros(3)) - ID2PHYS.get(int(cb[k, 0]), np.zeros(3)))
                   for k in range(1, 4)))
    return frac, tr


def pair_metrics(exp, a, b):
    out = {"success_a": a["success"], "success_b": b["success"], "d_success": a["success"] - b["success"],
           "discordant": float(a["success"] != b["success"])}
    n = min(len(a["act"]), len(b["act"]))
    out["cmd_final_m"] = float(np.linalg.norm(a["act"][:n].sum(0) - b["act"][:n].sum(0)))
    pre = bool(np.allclose(a["full_act_prefix"], b["full_act_prefix"], atol=1e-9, rtol=0))
    out["prefix_identical"] = float(pre)
    if a["pos"] is not None and b["pos"] is not None:
        d = np.linalg.norm(a["pos"][:n] - b["pos"][:n], axis=1)
        out.update({"tcp_mean_cm": 100 * float(d[T0:].mean()), "tcp_area_cmstep": 100 * float(d[T0:].sum()),
                    "tcp_final_cm": 100 * float(d[n - 1]), "tcp_steps": int(n - T0)})
    else:
        out.update({"tcp_mean_cm": np.nan, "tcp_area_cmstep": np.nan, "tcp_final_cm": np.nan, "tcp_steps": 0})
    if exp == "P1D":
        vals = [suffix_change(a["chunks"][t], a["cf"][t]) for t in sorted(a["cf"]) if t in a["chunks"]]
        out["gen_suffix_source"] = "shadow (a vs counterfactual-context chunk, t=8,12,16,20)" if vals else "n/a"
        out["ctx_exec_dev_trans_m"] = a["ctxdev"]
    else:
        vals = [suffix_change(a["chunks"][T0], b["chunks"][T0])] if (pre and T0 in a["chunks"] and T0 in b["chunks"]) else []
        out["gen_suffix_source"] = "t8 a-vs-b chunk (prefix identical)" if vals else "n/a (prefix diverged or no chunk)"
        out["ctx_exec_dev_trans_m"] = np.nan
    out["gen_suffix_tokfrac"] = float(np.mean([v[0] for v in vals])) if vals else np.nan
    out["gen_suffix_trans_m"] = float(np.mean([v[1] for v in vals])) if vals else np.nan
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    man = pd.read_csv(MANIFEST)
    man["exp"] = man["experiment"].replace({"P1D": "P1D"})
    cl = {(r.exp, r.task, int(r.seed_or_unit)): (r.initial_state_cluster_id, r.physical_state_cluster_id, r.restore_verification)
          for r in man.itertuples()}
    E, rows = {}, []
    for exp, paths in SRC.items():
        for p in paths:
            with open(p) as f:
                for line in f:
                    ep = json.loads(line)
                    cfg, direction, mode = parse_cond(exp, ep)
                    key = (exp, cfg, direction, mode, ep["task"], int(ep["seed"]))
                    if key in E:
                        raise RuntimeError(f"duplicate episode {key}")
                    c = compact(exp, ep)
                    E[key] = c
                    prim, phys, rv = cl[(exp, ep["task"], int(ep["seed"]))]
                    rows.append({"experiment": exp, "config": cfg, "direction": direction, "mode": mode,
                                 "task": TASK_SHORT[ep["task"]], "seed": int(ep["seed"]), "success": c["success"],
                                 "steps": c["steps"], "restore_ok": c["restore_ok"], "cluster_primary": prim,
                                 "cluster_physical": phys, "source": os.path.relpath(p, POC)})
        print(f"[load] {exp}: {sum(1 for k in E if k[0] == exp)} episodes", flush=True)
    epi = pd.DataFrame(rows)
    epi.to_csv(os.path.join(OUT, "episodes.csv"), index=False)
    prs = []
    for key, a in E.items():
        exp, cfg, direction, mode, task, seed = key
        for cname, (ma, mb) in CONTRASTS.items():
            if mode != ma:
                continue
            bdir = "-" if (exp == "exp34" and mb == "natural") else direction
            bk = (exp, cfg, bdir, mb, task, seed)
            if bk not in E:
                continue
            r = {"experiment": exp, "config": cfg, "direction": direction if not (exp == "exp34" and mode in ("gen", "natural_mix")) else "-",
                 "task": TASK_SHORT[task], "seed": seed, "contrast": cname, "mode_a": ma, "mode_b": mb}
            r.update(pair_metrics(exp, a, E[bk]))
            prim, phys, _ = cl[(exp, task, seed)]
            r["cluster_primary"], r["cluster_physical"] = prim, phys
            prs.append(r)
    pairs = pd.DataFrame(prs)
    pairs.to_csv(os.path.join(OUT, "pairs.csv"), index=False)
    print(epi.groupby(["experiment", "config", "direction", "mode"]).size().to_string())
    print(pairs.groupby(["experiment", "config", "direction", "contrast"]).agg(
        n=("d_success", "size"), d=("d_success", "mean"), pre=("prefix_identical", "mean"),
        tcp=("tcp_mean_cm", "mean"), cmd=("cmd_final_m", "mean"), suf=("gen_suffix_tokfrac", "mean")).to_string())


if __name__ == "__main__":
    main()
