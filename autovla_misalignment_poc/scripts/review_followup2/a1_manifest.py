#!/usr/bin/env python
"""
A1 (review_followup2, 2026-10-13): analysis manifest for every AutoVLA / Impromptu unit that enters the PoC-mechanism,
P1-A, P1-B, R8R9 and P1-C analyses. POST-HOC bookkeeping of existing data; reads units tables only (no inference).

One manifest row = (experiment, pair/unit, condition = intervention, outcome). Required columns of the prompt (A1) are
all present, null where not applicable; a few extra descriptive columns are appended (group_or_stratum,
perturbation, t_star, design_weight, block). Column definitions, missing rates, unique-key and duplicate counts go to
manifest_columns.csv and manifest_keys.json.

Usage: nice -n 19 python a1_manifest.py
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
RF = os.path.join(POC, "outputs/review_followup")
OUT = os.path.join(POC, "outputs/review_followup2/A_autovla_cluster/run_20261013")
os.makedirs(OUT, exist_ok=True)


def drive_of(l):
    p = l.split("_"); return f"{p[0]}_{p[1]}"


def git_last(path):
    try:
        return subprocess.run(["git", "-C", "/root/VLA", "log", "-1", "--format=%h", "--", path], capture_output=True,
                              text=True, timeout=30).stdout.strip() or None
    except Exception:
        return None


# experiment -> (run_id, split, execution_config, protocol_commit, original raw path relative to outputs/)
EXP = {
    "exp06_equal_distance": ("outputs/equal_distance_perturbation", "equal_distance_perturbation.py; forced alt token at t*; output-level "
                             "(executed trajectory contains the forced token); T=0.01; bf16; 3080Ti/5090 per original run_meta",
                             "none (exploratory; no preregistration)", "equal_distance_perturbation/rows.jsonl"),
    "exp07_action_history_causal": ("outputs/action_history_causal", "action_history_causal.py; context-only interventions "
                                    "(executed output = model's own tokens); T=0.01; bf16", "none (exploratory)",
                                    "action_history_causal/units.jsonl"),
    "exp11_prev_action_identity": ("outputs/prev_action_identity_decomposition", "prev_action_identity_decomposition.py; context-only; "
                                   "T=0.01; bf16", "none (exploratory)", "prev_action_identity_decomposition/units.jsonl"),
    "exp32_motion_semantics": ("outputs/motion_semantics_ablation", "motion_semantics_ablation.py; context-only sliding k-1 "
                               "substitution; T=0.01; bf16", "af2eb10 (protocol committed with scripts, before run)",
                               "motion_semantics_ablation/records.jsonl"),
    "exp33_impromptu_cde": ("outputs/cross_vla_temporal_replication/exp_cde", "Impromptu VLA 3B exp_cde.py; waypoint-text context "
                            "edits; RTX 5090", "038a6d3 (preregistration)", "cross_vla_temporal_replication/exp_cde/records.jsonl"),
    "R8": ("outputs/review_followup/R8R9_raw_recovery/run_20261008", "prev_action_state_patching.py unmodified (93-row batch); RTX 5090; "
           "recovery re-run of exp 8", "b212637 (R8R9 protocol; recovery, not confirmatory)", "R8R9_raw_recovery/run_20261008/raw/exp8"),
    "R9": ("outputs/review_followup/R8R9_raw_recovery/run_20261008", "temporal_feedback_window.py unmodified (17-row batch); RTX 5090; "
           "recovery re-run of exp 9", "b212637 (R8R9 protocol; recovery, not confirmatory)", "R8R9_raw_recovery/run_20261008/raw/exp9"),
    "P1A_gt_free_reference": ("outputs/review_followup/P1A_gt_free_reference/run_20261008", "p1a_gt_free_reference.py; GT-free model-reference "
                              "branches; RTX 5090", "b212637 (preregistered)", "P1A_gt_free_reference/run_20261008/raw"),
    "P1B_start_position": ("outputs/review_followup/P1B_start_position/run_20261008", "p1b_start_position.py; fixed-length correction windows "
                           "at start s, length w; RTX 5090", "b212637 (preregistered)", "P1B_start_position/run_20261008/raw"),
    "P1C": ("outputs/review_followup/P1C_new_log_replication/run_20261008", "unchanged dev harness per block (H7/H8/H9/H32); "
            "AutoVLA_PDMS_89.ckpt; T=0.01 seed 0; bf16; RTX 5090; H8 = attempt 2 (amendment 2)",
            "b212637 + b216920 (amend. 1) + 9b3d4af (amend. 2)", "P1C_new_log_replication/run_20261008/mech"),
}
SPLIT_POC = "poc_mechanism_dev (navtest shards 0-5; 28 logs / 20 drives; mechanism units on 26 logs / 19 drives)"
SPLIT_P1C = "p1c_eval (navtest shards 6-31; 64 logs / 24 drives; drive-disjoint from PoC)"
OUTCOME_UNIT = {"amplification": "binary 0/1 (not A+ and FDE5 > 3 m)", "fde5": "m", "dev9_ref": "m (deviation from model reference)",
                "recovery": "binary 0/1 (A+)"}


def frame(df, exp_key, experiment, outcomes, *, model, split, block=None, group_col="group", w_col=None, seed_col=None,
          tstar_col=None, src_parquet):
    run_id, cfg, prot, raw = EXP[exp_key]
    base = pd.DataFrame({
        "model": model, "experiment": experiment, "run_id": run_id, "split": split, "task": "navtest open-loop planning",
        "drive_id": df["log_id"].map(drive_of).values, "log_id": df["log_id"].values, "scene_id": df["scene_id"].values,
        "intervention": df["condition"].values, "execution_config": cfg,
        "simulator_seed": pd.array([pd.NA] * len(df), dtype="Int64"),
        "policy_rng_seed": (df[seed_col].astype("Int64").values if seed_col else pd.array([pd.NA] * len(df), dtype="Int64")),
        "repeat_id": 0,
        "initial_state_cluster_id": ("navtest:" + df["scene_id"].astype(str)).values,
        "complete_restore_hash": None,
        "pair_id": (experiment + "|" + df["unit_id"].astype(str)).values,
        "source_file": f"{src_parquet} <- outputs/{raw}", "protocol_commit": prot,
        "code_commit": None,
        "group_or_stratum": df[group_col].values if group_col else None,
        "perturbation": df["unit_id"].astype(str).str.split("|").str[-1].values,
        "t_star": df[tstar_col].astype("Int64").values if tstar_col else pd.array([pd.NA] * len(df), dtype="Int64"),
        "design_weight": df[w_col].astype(float).values if w_col else 1.0,
        "block": block if block is not None else None,
    })
    outs = []
    for oname, col in outcomes:
        x = base.copy()
        x["outcome_name"] = oname
        x["outcome_value"] = df[col].astype(float).values
        x["unit"] = OUTCOME_UNIT.get(oname, "")
        outs.append(x)
    return pd.concat(outs, ignore_index=True)


def main():
    parts = []
    # ---- PoC mechanism experiments (P0-B units table)
    p0b = os.path.join(RF, "P0B_cluster_stats/run_20261008/units.parquet")
    u = pd.read_parquet(p0b)
    u = u[u["model"].isin(["AutoVLA", "Impromptu-VLA-3B"]) & u["condition"].notna()].copy()
    # t* per scene from R8R9 (same scenes; first natural mismatch)
    r = pd.read_parquet(os.path.join(RF, "R8R9_raw_recovery/run_20261008/units.parquet"))
    ts = r.groupby("scene_id")["t_star"].agg(["min", "max"])
    assert (ts["min"] == ts["max"]).all()
    tmap = ts["min"]
    for exp in ["exp06_equal_distance", "exp07_action_history_causal", "exp11_prev_action_identity", "exp32_motion_semantics",
                "exp33_impromptu_cde"]:
        d = u[u["experiment"] == exp].copy()
        d["t_star_m"] = d["scene_id"].map(tmap) if exp != "exp33_impromptu_cde" else np.nan
        model = "Impromptu-VLA-3B" if exp == "exp33_impromptu_cde" else "AutoVLA"
        parts.append(frame(d, exp, exp, [("amplification", "amplification"), ("fde5", "fde5")], model=model, split=SPLIT_POC,
                           group_col=None if exp == "exp33_impromptu_cde" else "group",
                           tstar_col=None if exp == "exp33_impromptu_cde" else "t_star_m",
                           src_parquet="review_followup/P0B_cluster_stats/run_20261008/units.parquet"))
    # ---- R8 / R9
    for exp in ("R8", "R9"):
        d = r[r["experiment"] == exp]
        parts.append(frame(d, exp, exp, [("amplification", "amplification"), ("fde5", "fde5")], model="AutoVLA", split=SPLIT_POC,
                           seed_col="seed", tstar_col="t_star",
                           src_parquet="review_followup/R8R9_raw_recovery/run_20261008/units.parquet"))
    # ---- P1-A / P1-B
    a = pd.read_parquet(os.path.join(RF, "P1A_gt_free_reference/run_20261008/units.parquet"),
                        columns=["unit_id", "log_id", "scene_id", "group", "condition", "seed", "t_star", "primary_outcome",
                                 "amplification", "fde5"])
    parts.append(frame(a, "P1A_gt_free_reference", "P1A_gt_free_reference",
                       [("dev9_ref", "primary_outcome"), ("amplification", "amplification"), ("fde5", "fde5")], model="AutoVLA",
                       split=SPLIT_POC, seed_col="seed", tstar_col="t_star",
                       src_parquet="review_followup/P1A_gt_free_reference/run_20261008/units.parquet"))
    b = pd.read_parquet(os.path.join(RF, "P1B_start_position/run_20261008/units.parquet"),
                        columns=["unit_id", "log_id", "scene_id", "group", "condition", "seed", "t_star", "amplification", "fde5"])
    parts.append(frame(b, "P1B_start_position", "P1B_start_position", [("amplification", "amplification"), ("fde5", "fde5")],
                       model="AutoVLA", split=SPLIT_POC, seed_col="seed", tstar_col="t_star",
                       src_parquet="review_followup/P1B_start_position/run_20261008/units.parquet"))
    # ---- P1-C (one experiment per block, each block has its own Normal)
    c = pd.read_parquet(os.path.join(RF, "P1C_new_log_replication/run_20261008/analysis/units.parquet"))
    for blk in ("H7", "H8", "H9", "H32"):
        d = c[c["block"] == blk]
        parts.append(frame(d, "P1C", f"P1C_{blk}", [("amplification", "amplification"), ("fde5", "fde5"), ("recovery", "recovery")],
                           model="AutoVLA", split=SPLIT_P1C, block=blk, group_col="stratum", w_col="weight", seed_col="seed",
                           tstar_col="t_star", src_parquet="review_followup/P1C_new_log_replication/run_20261008/analysis/units.parquet"))
    m = pd.concat(parts, ignore_index=True)
    # code commit = last commit touching the experiment's run directory (results commit)
    cc = {e: git_last(os.path.join("autovla_misalignment_poc", EXP[k][0])) for e, k in
          [(x, x) for x in ["exp06_equal_distance", "exp07_action_history_causal", "exp11_prev_action_identity",
                            "exp32_motion_semantics", "exp33_impromptu_cde", "R8", "R9", "P1A_gt_free_reference", "P1B_start_position"]]}
    cpc = git_last("autovla_misalignment_poc/outputs/review_followup/P1C_new_log_replication/run_20261008/analysis")
    m["code_commit"] = m["experiment"].map(lambda e: cpc if e.startswith("P1C") else cc.get(e))
    m["anchor_id"] = m["scene_id"].astype(str) + "@t*=" + m["t_star"].astype("string").fillna("NA")
    req = ["model", "experiment", "run_id", "split", "task", "drive_id", "log_id", "scene_id", "anchor_id", "intervention",
           "execution_config", "simulator_seed", "policy_rng_seed", "repeat_id", "initial_state_cluster_id",
           "complete_restore_hash", "pair_id", "outcome_name", "outcome_value", "unit", "source_file", "protocol_commit",
           "code_commit"]
    extra = ["group_or_stratum", "perturbation", "t_star", "design_weight", "block"]
    m = m[req + extra]
    for col in ["model", "experiment", "run_id", "split", "task", "drive_id", "log_id", "intervention", "execution_config",
                "outcome_name", "unit", "source_file", "protocol_commit", "code_commit", "group_or_stratum", "block"]:
        m[col] = m[col].astype("category")
    m.to_parquet(os.path.join(OUT, "manifest_autovla_impromptu.parquet"), index=False)

    # ---------------------------------------------------------------- column definitions, missing, keys, duplicates
    DEF = {
        "model": "AutoVLA (AutoVLA_PDMS_89.ckpt, Qwen2.5-VL-3B, 2048-token action codebook) or Impromptu-VLA-3B",
        "experiment": "experiment id; P1-C split per mechanism block (each block has its own Normal row)",
        "run_id": "run directory (relative to autovla_misalignment_poc/)",
        "split": "data split the unit belongs to (PoC mechanism dev set vs P1-C drive-disjoint eval set)",
        "task": "task family; all rows are open-loop navtest trajectory planning (no closed-loop simulator)",
        "drive_id": "<date.time>_veh-NN = first two '_' fields of the navsim log name (same rule as P1-C split.csv)",
        "log_id": "navsim log segment <date.time>_veh-NN_<start>_<end>",
        "scene_id": "navtest scene token (anchor frame)",
        "anchor_id": "scene token + t* (first natural token mismatch, the injection position); t* from the units table "
                     "(R8R9 for exp 6/7/11/32 scenes; NA for Impromptu, whose units are direction perturbations at a fixed step)",
        "intervention": "condition name as in the original script (definitions in DATA_AUDIT_autovla)",
        "execution_config": "harness script and fixed settings (temperature, precision, GPU, executed-vs-context intervention)",
        "simulator_seed": "null: no simulator (open-loop on logged frames)",
        "policy_rng_seed": "sampling seed recorded in the units table (0 where present); null where the source table did not "
                           "store it (exp 6/7/11/32/33 via P0-B units) -- not inferred",
        "repeat_id": "0: one decode per (unit, condition); there are no stochastic repeats in these experiments",
        "initial_state_cluster_id": "physical initial state = logged navtest frame + ego state at the scene token; identical "
                                    "for all perturbations/conditions/models of one scene (navtest:<scene_id>)",
        "complete_restore_hash": "null: not applicable -- open-loop forward pass on logged inputs, no simulator/controller "
                                 "state to restore; reproducibility of the driving harness is audited by re-running "
                                 "(R8R9 220/220 rows exact)",
        "pair_id": "experiment|unit_id (unit = scene|perturbation); rows with the same pair_id are the paired conditions",
        "outcome_name": "amplification (primary binary), fde5 (m), dev9_ref (P1-A primary, m), recovery (P1-C, binary)",
        "outcome_value": "value of the outcome (amplification/recovery 0/1; invalid outputs count 0 per the dev rule)",
        "unit": "measurement unit of outcome_value",
        "source_file": "units table used here <- original raw location",
        "protocol_commit": "commit of the governing protocol / preregistration ('none' = exploratory experiment)",
        "code_commit": "last git commit touching the run directory (results commit); null if untracked",
        "group_or_stratum": "AutoVLA A-/A+ group (PoC) or P1-C stratum F/S; null for Impromptu",
        "perturbation": "perturbation id inside the scene (original / alt_k / direction)",
        "t_star": "first natural mismatch position (integer)",
        "design_weight": "P1-C Hajek weight 1/pi (F = 1); 1 elsewhere",
        "block": "P1-C mechanism block (H7/H8/H9/H32); null elsewhere",
    }
    cols = []
    for ccol in m.columns:
        s = m[ccol]
        miss = float(s.isna().mean()) if s.dtype != object else float(s.isna().mean())
        cols.append({"column": ccol, "required_by_prompt": ccol in req, "definition": DEF.get(ccol, ""),
                     "dtype": str(s.dtype), "missing_rate": round(miss, 6), "n_unique": int(s.nunique(dropna=True))})
    pd.DataFrame(cols).to_csv(os.path.join(OUT, "manifest_columns.csv"), index=False)
    key = ["experiment", "pair_id", "intervention", "outcome_name", "repeat_id"]
    dup = int(m.duplicated(key).sum())
    per_exp = m[m.outcome_name == "amplification"].groupby("experiment", observed=True).agg(
        rows=("pair_id", "size"), pairs=("pair_id", "nunique"), scenes=("scene_id", "nunique"), logs=("log_id", "nunique"),
        drives=("drive_id", "nunique"), interventions=("intervention", "nunique"),
        init_states=("initial_state_cluster_id", "nunique"))
    per_exp.to_csv(os.path.join(OUT, "manifest_counts_by_experiment.csv"))
    dup_by_exp = m[m.duplicated(key, keep=False)].groupby("experiment", observed=True).size().to_dict()
    info = {"status": "POST-HOC bookkeeping of existing data", "rows": int(len(m)), "unique_key": key,
            "duplicate_rows_on_unique_key": dup, "duplicates_by_experiment": {k: int(v) for k, v in dup_by_exp.items()},
            "secondary_keys": {
                "(experiment, pair_id, intervention) per outcome": "unique if duplicate_rows_on_unique_key == 0",
                "initial_state_cluster_id": int(m["initial_state_cluster_id"].nunique()),
                "drives": int(m["drive_id"].nunique()), "logs": int(m["log_id"].nunique()),
                "scenes": int(m["scene_id"].nunique())},
            "notes": ["simulator_seed and complete_restore_hash are null by construction (no simulator).",
                      "policy_rng_seed null for exp 6/7/11/32/33 rows: not stored in the P0-B units table.",
                      "Each P1-C block is a separate experiment value because each contrast uses its own block Normal."]}
    json.dump(info, open(os.path.join(OUT, "manifest_keys.json"), "w"), indent=1)
    print(json.dumps(info, indent=1)); print(per_exp)


if __name__ == "__main__":
    main()
