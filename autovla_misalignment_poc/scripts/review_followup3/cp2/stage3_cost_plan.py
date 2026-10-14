#!/usr/bin/env python
"""Stage-3 cost plan (CP2, 2026-10-14). PLANNING ONLY, CPU only. Nothing here is run on the GPU.

Combines measured rates (STAGE3_COST_RATES.csv, from stage3_cost_rates.py), unit counts read from P0 tables, and the
P4 power simulation (P4/power/power_cost.csv) into outputs/review_followup3/STAGE3_COST_PLAN.csv.
Every GPU-hour figure is a range [optimistic, conservative] from two measured rates; assumptions that are not measured
are listed in the `assumption` column and come from PROTOCOL_P1.md §9 (model load ~3 min; Impromptu harness ~1 h).

  nice -n 19 /root/miniforge3/envs/autovla/bin/python stage3_cost_plan.py
"""
import pandas as pd

R3 = "/root/VLA/autovla_misalignment_poc/outputs/review_followup3"
LOAD_MIN = 3.0               # PROTOCOL_P1 §9.2: model load ~3 min each (assumption, not re-measured)
IMP_HARNESS_H = 1.0          # PROTOCOL_P1 §9.4: Impromptu no-op harness ~1 h GPU for 20 scenes (estimate)
P3_ROWS_NOTE = "<= 12 rows per unit batched (6 factorial cells + Normal + Full-GT + 2 no-op/leak rows + spares)"


def main():
    r = pd.read_csv(f"{R3}/STAGE3_COST_RATES.csv").set_index("rate_id")
    rate = r.rate_per_min
    den = pd.read_csv(f"{R3}/P0/POPULATION_p1c_denominators.csv")
    p1c_units = int(den[(den.contrast == "RN") & (den.estimand == "P")].n_units.iloc[0])
    p1c_f_units = int(den[(den.contrast == "RN") & (den.estimand == "F")].n_units.iloc[0])
    pe = pd.read_csv(f"{R3}/P0/POPULATION_EFFECTS.csv")
    dev_units = int(pe[(pe.row_id == "P0A.DEV.RN.ALL_unweighted")].n_units.iloc[0])
    inv = pd.read_csv(f"{R3}/P0/DISJOINT_LOG_INVENTORY_summary.csv")
    unused = inv[inv.subset.str.startswith("test logs not used by dev / P1-C")].iloc[0]
    flow = pd.read_csv(f"{R3}/P0/POPULATION_selection_flow.csv")
    s1 = float(flow[flow.cohort.str.startswith("P1-C stage 1") & flow.step.str.startswith("0 ")].n_scenes.iloc[0])
    an = float(flow[flow.cohort.str.startswith("P1-C stage 1") & flow.step.str.startswith("4 selected total")].n_scenes.iloc[0])

    gate_fast = max(rate[f"AV_gate_run1_{p}"] for p in ("H7", "H8", "H9"))
    gate_slow = rate["AV_gate_run1_P1A"]
    c5_fast, c5_slow = rate["AV_gate_run2_H7"], rate["AV_gate_run2_P1A"]
    rows = []

    def add(option, item, n, unit, fast, slow, assumption="", moves=""):
        rows.append({"option": option, "item": item, "n": n, "n_unit": unit,
                     "gpu_h_optimistic": n / fast / 60.0 if fast else 0.0,
                     "gpu_h_conservative": n / slow / 60.0 if slow else 0.0, "assumption": assumption,
                     "review_items_moved": moves})

    # Option 0: extend the AutoVLA gate to H32 (DM) and the P1-B start-position path
    n = 9 * 2 * 2  # 9 manifest scenes x 2 new paths x 2 decodings (same manifest rule as Stage 0)
    add("O0 gate extension (H32, P1-B)", "controls c1-c5 run1", n, "path-unit x decoding", gate_fast, gate_slow)
    add("O0 gate extension (H32, P1-B)", "c5 run2 (separate process)", n, "path-unit x decoding", c5_fast, c5_slow)
    add("O0 gate extension (H32, P1-B)", "model loads", 2, "loads", 1 / LOAD_MIN, 1 / LOAD_MIN, "load ~3 min")
    # Option 1: P3 AutoVLA matched reverse
    o = "O1 P3 AutoVLA matched Reverse"
    add(o, "P3 code-path no-op gate (Stage-0 manifest, new rows)", 9 * 2, "path-unit x decoding", gate_fast, gate_slow,
        moves="M6 (new path)")
    add(o, "P3 code-path c5 run2", 9 * 2, "path-unit x decoding", c5_fast, c5_slow)
    for setname, n_u in (("P1-C 353 scenes (primary)", p1c_units), ("dev 208 scenes (secondary)", dev_units)):
        add(o, f"frozen source bank (Normal + corrected branch), {setname}", n_u, "units",
            rate["AV_H9_P1C_B17"], rate["AV_H7_P1C_B1"])
        add(o, f"one-shot single-anchor factorial (primary block), {setname}", n_u, "units",
            rate["AV_P1A_raw_B10"], rate["AV_H7_P1C_B1"], P3_ROWS_NOTE, "M2 (AutoVLA part)")
        add(o, f"repeated-schedule factorial (secondary block), {setname}", n_u, "units",
            rate["AV_P1A_raw_B10"], rate["AV_H7_P1C_B1"], P3_ROWS_NOTE)
    add(o, "model loads", 8, "loads", 1 / LOAD_MIN, 1 / LOAD_MIN, "load ~3 min")
    # Option 2: P3 Impromptu add-on
    o = "O2 P3 Impromptu add-on"
    add(o, "Impromptu no-op harness + gate (20 scenes)", 1, "gate", 1 / (IMP_HARNESS_H * 60), 1 / (2 * IMP_HARNESS_H * 60),
        "PROTOCOL_P1 §9.4 estimate 1 h (conservative x2)", "M6 (Impromptu path)")
    for blk in ("frozen source bank", "one-shot factorial (primary)", "repeated-schedule factorial (secondary)"):
        add(o, f"{blk}, exp 33 exp_cde scenes", 2700, "units", rate["IM_exp33_cde_mtime"], rate["IM_exp33_cde_mtime"] / 2,
            "exp 33 rate from file mtimes (approximate; conservative = half rate)", "M2 (Impromptu part)" if "primary" in blk else "")
    # Option 3: P4 SpatialVLA new physical states (from the power simulation)
    pc = pd.read_csv(f"{R3}/P4/power/power_cost.csv", keep_default_na=False)
    for scen, m, crit, label in (("true0", 10.0, "all 8 pass, Holm m=8 >= 0.8", "+/-10 pp per task, Holm m=8, true 0"),
                                 ("het_state", 10.0, "all 8 pass, Holm m=8 >= 0.8", "+/-10 pp per task, Holm m=8, heterogeneous"),
                                 ("stress35", 10.0, "all 8 pass, Holm m=8 >= 0.8", "+/-10 pp per task, Holm m=8, discordance 0.35"),
                                 ("true0", 7.5, "all 8 pass, Holm m=8 >= 0.8", "+/-7.5 pp per task, Holm m=8, true 0"),
                                 ("true0", 5.0, "all 8 pass, Holm m=8 >= 0.8", "+/-5 pp per task, Holm m=8, true 0")):
        x = pc[(pc.scenario == scen) & (pc.margin_pp.astype(float) == m) & (pc.criterion == crit)].iloc[0]
        rows.append({"option": "O3 P4 SpatialVLA variant states", "item": f"{label}: {int(float(x.n_states_per_task_required))} states/task",
                     "n": int(float(x.episodes_total)), "n_unit": "episodes (incl. 5% control arms + 2x16-state pilot)",
                     "gpu_h_optimistic": float(x.gpu_h_at_SV_P1gate_4workers),
                     "gpu_h_conservative": float(x.gpu_h_at_SV_P1D_8workers),
                     "assumption": "rates: P1 gate 4 workers (optimistic) / P1-D 8 workers (conservative)",
                     "review_items_moved": "M4 (new-state part)" if m == 10.0 else "M4 (+/-5 pp target)" if m == 5.0 else "M4"})
    # Option 4: held-out-drive (NOT day/vehicle-disjoint) AutoVLA replication on the 53 unused test logs
    o = "O4 held-out-drive replication (not P5)"
    n_sc = float(unused.navtest_scenes)
    add(o, "natural pass (all scenes of the 53 unused test logs)", n_sc, "scenes", rate["AV_P1C_stage1_natural"],
        rate["AV_P1C_stage1_natural"], f"images present locally for {int(unused.images_local_logs)}/{int(unused.n_logs)} logs: fetch needed")
    scale = n_sc / s1
    n_u = p1c_units * scale
    for rid in ("AV_H7_P1C_B1", "AV_H8_P1C_B93", "AV_H9_P1C_B17", "AV_H32_P1C_B6"):
        add(o, f"mechanism run {rid} (units scaled from P1-C by scene count)", round(n_u), "units", rate[rid], rate[rid],
            "units = P1-C units x (scenes / P1-C stage-1 scenes)", "none resolved (M5 needs day/vehicle-disjoint logs)")
    df = pd.DataFrame(rows)
    # O3 rows are alternative designs (one is chosen), so they are not summed
    tot = df[~df.option.str.startswith("O3")].groupby("option")[["gpu_h_optimistic", "gpu_h_conservative"]].sum().reset_index()
    tot["item"] = "TOTAL"
    df = pd.concat([df, tot], ignore_index=True)
    df.to_csv(f"{R3}/STAGE3_COST_PLAN.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 250)
    print(df[["option", "item", "n", "gpu_h_optimistic", "gpu_h_conservative"]].to_string())
    print({"p1c_units": p1c_units, "p1c_f_units": p1c_f_units, "dev_units": dev_units, "unused_scenes": n_sc,
           "analysed_scenes_p1c": an})


if __name__ == "__main__":
    main()
