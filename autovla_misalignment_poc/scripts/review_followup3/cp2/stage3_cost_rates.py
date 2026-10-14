#!/usr/bin/env python
"""Measured throughput for Stage-3 cost planning (CP2, 2026-10-14). CPU only; reads run logs / run_meta files.

Writes outputs/review_followup3/STAGE3_COST_RATES.csv: one row per measured run (model, rate id, source file,
quantity, n, wall minutes, rate per minute). Nothing is hand-typed: every n and minute value is parsed from the file
named in `source`. Rates are wall-clock on the RTX 5090 (GPU 1), which is shared with an external process.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python stage3_cost_rates.py
"""
import json
import os
import re
from datetime import datetime

import pandas as pd

P = "/root/VLA/autovla_misalignment_poc/outputs"
OUT = f"{P}/review_followup3/STAGE3_COST_RATES.csv"


def meta(path, nkey):
    m = json.load(open(path))
    n = m.get(nkey)
    if n is None:
        for k in ("n_units", "n_perturbations"):
            if m.get(k):
                n = m[k]
                break
    return float(n), float(m["elapsed_min"])


def main():
    rows = []

    def add(model, rid, src, qty, n, minutes, note=""):
        rows.append({"model": model, "rate_id": rid, "source": src, "quantity": qty, "n": n, "minutes": minutes,
                     "rate_per_min": n / minutes, "note": note})

    # AutoVLA: intervention harnesses (units = perturbation units; each unit decodes all rows of the batch)
    for rid, rel, qty, note in [
        ("AV_H7_P1C_B1", "review_followup/P1C_new_log_replication/run_20261008/mech/action_history_causal/run_meta.json",
         "units (7 conditions, B=1 sequential)", "P1-C, 353 scenes"),
        ("AV_H8_P1C_B93", "review_followup/P1C_new_log_replication/run_20261008/mech/prev_action_state_patching/run_meta.json",
         "units (93 rows batched)", "P1-C, 353 scenes"),
        ("AV_H9_P1C_B17", "review_followup/P1C_new_log_replication/run_20261008/mech/temporal_feedback_window/run_meta.json",
         "units (17 rows batched)", "P1-C, 353 scenes"),
        ("AV_H32_P1C_B6", "review_followup/P1C_new_log_replication/run_20261008/mech/motion_semantics_ablation/run_meta.json",
         "units (6 rows batched)", "P1-C, 353 scenes"),
        ("AV_H8_dev_B93", "review_followup/R8R9_raw_recovery/run_20261008/raw/exp8/run_meta.json",
         "units (93 rows batched)", "dev 208 scenes"),
        ("AV_H9_dev_B17", "review_followup/R8R9_raw_recovery/run_20261008/raw/exp9/run_meta.json",
         "units (17 rows batched)", "dev 208 scenes"),
        ("AV_P1A_raw_B10", "review_followup/P1A_gt_free_reference/run_20261008/raw/run_meta.json",
         "units (10 rows batched)", "dev 208 scenes"),
        ("AV_P1B_raw", "review_followup/P1B_start_position/run_20261008/raw/run_meta.json",
         "units", "dev 208 scenes (start-position script)"),
    ]:
        n, mins = meta(f"{P}/{rel}", "n_units")
        add("AutoVLA", rid, rel, qty, n, mins, note)

    # AutoVLA natural pass (P1-C stage 1: one plan per navtest scene of the 64 eval logs) and unit building
    rel = "review_followup/P1C_new_log_replication/run_20261008/gpu_time.jsonl"
    steps = [json.loads(l) for l in open(f"{P}/{rel}")]
    flow = pd.read_csv(f"{P}/review_followup3/P0/POPULATION_selection_flow.csv")
    n_s1 = float(flow[flow.cohort.str.startswith("P1-C stage 1") & flow.step.str.startswith("0 ")].n_scenes.iloc[0])
    add("AutoVLA", "AV_P1C_stage1_natural", rel, "scenes (natural plan)", n_s1,
        sum(r["s"] for r in steps if r["step"].startswith("stage1")) / 60.0, "26 navtest shards, 64 eval logs")
    n_an = float(flow[flow.cohort.str.startswith("P1-C stage 1") & flow.step.str.startswith("4 selected total")].n_scenes.iloc[0])
    add("AutoVLA", "AV_P1C_units_build", rel, "scenes analysed (unit building)", n_an,
        sum(r["s"] for r in steps if r["step"] == "units") / 60.0, "analysed scenes from the P0-A flow table")

    # AutoVLA Stage-0 gate: per path-unit x decoding seconds from the harness log (all controls c1-c5)
    for run, ctrl in (("run1", "c1-c5"), ("run2", "c5 only")):
        rel = f"review_followup3/P1/gate_stage0/{run}.log"
        txt = open(f"{P}/{rel}").read()
        secs = [(m.group(1), float(m.group(2))) for m in re.finditer(r"\[p1b\] \S+ (H7|H8|H9|P1A) \S+ ([\d.]+)s", txt)]
        tot = float(re.search(r"\[done\] ([\d.]+) min", txt).group(1))
        add("AutoVLA", f"AV_gate_{run}_total", rel, f"path-unit x decoding ({ctrl})", float(len(secs)), tot,
            "21 path-units x 2 decodings, RTX 5090")
        for path in ("H7", "H8", "H9", "P1A"):
            s = [x for p, x in secs if p == path]
            add("AutoVLA", f"AV_gate_{run}_{path}", rel, f"path-unit x decoding ({ctrl})", float(len(s)), sum(s) / 60.0)

    # SpatialVLA closed-loop episodes
    rel = "review_followup3/P1/svla_gate/gate_run1.log"
    m = re.search(r"\[done\] (\d+) episodes in ([\d.]+) min", open(f"{P}/{rel}").read())
    add("SpatialVLA", "SV_P1gate_4workers", rel, "episodes (80 steps)", float(m.group(1)), float(m.group(2)),
        "4 SAPIEN workers (A4: 8 workers thrashed the page cache)")
    rel = "review_followup/P1D_dose_control/run_20261008/analysis.md"
    txt = open(f"{P}/{rel}").read()
    blocks = [(float(a.replace(",", "")), float(b)) for a, b in re.findall(r"([\d,]+\.\d)분\(([\d.]+) ep/min\)", txt)]
    eps = sum(r * mnt for mnt, r in blocks)
    add("SpatialVLA", "SV_P1D_8workers", rel, "episodes (80 steps)", round(eps), sum(mnt for mnt, _ in blocks),
        "P1-D main run, 2 blocks, 8 workers, GPU shared with an external process")

    # Impromptu exp 33 exp_cde (approximate: file mtimes; ran concurrently with exp_ab)
    d = f"{P}/cross_vla_temporal_replication/exp_cde"
    t0 = os.path.getmtime(f"{d}/selected_scenes.json")
    t1 = os.path.getmtime(f"{d}/records.jsonl")
    n_units = sum(len(json.loads(l)["units"]) for l in open(f"{d}/records.jsonl"))
    add("Impromptu", "IM_exp33_cde_mtime", "cross_vla_temporal_replication/exp_cde (mtimes)", "units (12 rows)",
        float(n_units), (t1 - t0) / 60.0, "APPROXIMATE: from file mtimes; concurrent with exp_ab")
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, float_format="%.4f")
    print(df.to_string())


if __name__ == "__main__":
    main()
