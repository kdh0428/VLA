#!/usr/bin/env python
"""
A2 (review_followup2, 2026-10-13): drive / log / scene overlap between every AutoVLA-side data set.
POST-HOC audit of existing files (no inference). drive_id = <date.time>_veh-NN (first two '_' fields of the log name),
the same rule as P1-C split.csv (verified below row by row).

Sets
  mech_poc_units      logs/scenes in the PoC mechanism units (exp 6/7/11/32, R8R9, P1-A, P1-B; P0-B/R8R9/P1A/P1B units)
  mech_poc_Aminus     A- subset of the above (the 16-log / 52-scene set of the AutoVLA primary family)
  impromptu_exp33     Impromptu exp 33 C-E units (same PoC frames)
  poc_dev_discovery   split.csv role dev_discovery (28 PoC logs, shards 0-5)
  sel_dev             expanded_best_of_n records (selection dev, shards 6-17)
  sel_heldout         heldout_best_of_n records (selection held-out, shards 18-31)
  navhard_src         navhard_two_stage source logs (76; design/work/navhard_logs.json)
  navhard_half1/2     navhard source logs of exp 25 / exp 26 halves
  p1c_eval_all        P1-C eval logs (split.csv role eval) and their stage-1 scenes
  p1c_eval_units      P1-C analysed scenes (353)
  p1c_pilot           split.csv role pilot_harness_check (+ pilot units)
"""
from __future__ import annotations

import itertools
import json
import os

import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
O = os.path.join(POC, "outputs")
RF = os.path.join(O, "review_followup")
OUT = os.path.join(POC, "outputs/review_followup2/A_autovla_cluster/run_20261013")


def drive_of(l):
    p = l.split("_"); return f"{p[0]}_{p[1]}"


def dayveh_of(l):
    p = l.split("_"); return f"{p[0][:10]}_{p[1]}"


def jsonl_token_log(paths):
    sc, lg = set(), set()
    pairs = set()
    for p in paths:
        with open(p) as f:
            for line in f:
                # records are large; only the leading keys are needed
                i = line.find('"token"'); j = line.find('"log"')
                r = json.loads(line) if (i < 0 or j < 0) else None
                if r is None:
                    t = line[i:].split('"')[3]; l_ = line[j:].split('"')[3]
                else:
                    t, l_ = r["token"], r["log"]
                sc.add(t); lg.add(l_); pairs.add((t, l_))
    return sc, lg, pairs


def main():
    S = {}   # name -> {"logs": set, "scenes": set or None}
    split = pd.read_csv(os.path.join(RF, "P1C_new_log_replication/design_20261008/split.csv"), dtype=str)
    # verify split.csv drive definition
    assert (split["drive"] == split["log"].map(drive_of)).all(), "split.csv drive != log prefix"
    assert (split["day_vehicle"] == split["log"].map(dayveh_of)).all()
    u = pd.read_parquet(os.path.join(RF, "P0B_cluster_stats/run_20261008/units.parquet"),
                        columns=["model", "experiment", "log_id", "scene_id", "group"])
    av = u[u.model == "AutoVLA"]
    frames = [av[["log_id", "scene_id", "group"]]]
    for p in ("R8R9_raw_recovery/run_20261008/units.parquet", "P1A_gt_free_reference/run_20261008/units.parquet",
              "P1B_start_position/run_20261008/units.parquet"):
        frames.append(pd.read_parquet(os.path.join(RF, p), columns=["log_id", "scene_id", "group"]))
    mech = pd.concat(frames).drop_duplicates()
    S["mech_poc_units"] = {"logs": set(mech.log_id), "scenes": set(mech.scene_id)}
    am = mech[mech.group == "A-"]
    S["mech_poc_Aminus"] = {"logs": set(am.log_id), "scenes": set(am.scene_id)}
    im = u[u.experiment == "exp33_impromptu_cde"].dropna(subset=["log_id"])
    S["impromptu_exp33"] = {"logs": set(im.log_id), "scenes": set(im.scene_id)}
    S["poc_dev_discovery"] = {"logs": set(split.loc[split.role == "dev_discovery", "log"]), "scenes": None}
    for name, d in (("sel_dev", "expanded_best_of_n"), ("sel_heldout", "heldout_best_of_n")):
        paths = [os.path.join(O, d, g, "records.jsonl") for g in ("gpu0", "gpu1")]
        sc, lg, _ = jsonl_token_log(paths)
        S[name] = {"logs": lg, "scenes": sc}
    nh = json.load(open(os.path.join(RF, "P1C_new_log_replication/design_20261008/work/navhard_logs.json")))
    S["navhard_src"] = {"logs": set(nh["navhard_src"]), "scenes": None}
    S["navhard_half1_exp25"] = {"logs": set(json.load(open(os.path.join(O, "navhard_eval/subset/chosen_logs.json")))), "scenes": None}
    S["navhard_half2_exp26"] = {"logs": set(json.load(open(os.path.join(O, "navhard_full_validation/half2/subset/second_half_logs.json")))),
                                "scenes": None}
    ev = set(split.loc[split.role == "eval", "log"])
    st1 = set()
    with open(os.path.join(RF, "P1C_new_log_replication/run_20261008/stage1/records.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            st1.add((r["token"], r.get("log", r.get("log_name"))))
    S["p1c_eval_all"] = {"logs": ev, "scenes": {t for t, l_ in st1 if l_ in ev}}
    c = pd.read_parquet(os.path.join(RF, "P1C_new_log_replication/run_20261008/analysis/units.parquet"),
                        columns=["log_id", "scene_id", "stratum", "drive_id"])
    S["p1c_eval_units"] = {"logs": set(c.log_id), "scenes": set(c.scene_id)}
    S["p1c_eval_units_F"] = {"logs": set(c.loc[c.stratum == "F", "log_id"]), "scenes": set(c.loc[c.stratum == "F", "scene_id"])}
    pil = set(split.loc[split.role == "pilot_harness_check", "log"])
    psc = set()
    with open(os.path.join(RF, "P1C_new_log_replication/pilot_20261008/units/records.jsonl")) as f:
        for line in f:
            psc.add(json.loads(line)["token"])
    S["p1c_pilot"] = {"logs": pil, "scenes": psc}
    S["excluded_drive_adjacent_to_dev"] = {"logs": set(split.loc[split.role == "excluded_drive_adjacent_to_dev", "log"]), "scenes": None}
    for k in S:
        S[k]["drives"] = {drive_of(l) for l in S[k]["logs"]}
        S[k]["dayveh"] = {dayveh_of(l) for l in S[k]["logs"]}

    # ------------------------------------------------------------------ summary + pairwise overlap
    summ = [{"set": k, "n_logs": len(v["logs"]), "n_drives": len(v["drives"]), "n_day_vehicles": len(v["dayveh"]),
             "n_scenes": (len(v["scenes"]) if v["scenes"] is not None else None)} for k, v in S.items()]
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "overlap_set_summary.csv"), index=False)
    rows = []
    for a, b in itertools.permutations(S, 2):
        A, B = S[a], S[b]
        rows.append({"set_a": a, "set_b": b,
                     "shared_drives": len(A["drives"] & B["drives"]), "shared_logs": len(A["logs"] & B["logs"]),
                     "shared_day_vehicles": len(A["dayveh"] & B["dayveh"]),
                     "shared_scenes": (len(A["scenes"] & B["scenes"]) if A["scenes"] is not None and B["scenes"] is not None else None),
                     "a_logs_in_shared_drives": len({l for l in A["logs"] if drive_of(l) in B["drives"]}),
                     "a_n_logs": len(A["logs"]), "a_n_drives": len(A["drives"])})
    ov = pd.DataFrame(rows)
    ov.to_csv(os.path.join(OUT, "overlap_pairwise.csv"), index=False)
    # wide drive matrix
    names = list(S)
    for lev in ("drives", "logs", "scenes"):
        mat = pd.DataFrame(index=names, columns=names, dtype=object)
        for a in names:
            for b in names:
                A, B = S[a][lev], S[b][lev]
                mat.loc[a, b] = None if A is None or B is None else len(A & B)
        mat.to_csv(os.path.join(OUT, f"overlap_matrix_{lev}.csv"))
    # per-drive membership
    alld = sorted(set().union(*[v["drives"] for v in S.values()]))
    mem = pd.DataFrame([{"drive_id": d, **{k: len({l for l in S[k]["logs"] if drive_of(l) == d}) for k in names}} for d in alld])
    mem.to_csv(os.path.join(OUT, "drive_membership.csv"), index=False)
    # A- 16 logs -> 13 drives
    amap = sorted(S["mech_poc_Aminus"]["logs"])
    amd = pd.DataFrame({"log_id": amap, "drive_id": [drive_of(l) for l in amap],
                        "n_Aminus_scenes": [am[am.log_id == l].scene_id.nunique() for l in amap]})
    amd.to_csv(os.path.join(OUT, "Aminus_log_to_drive.csv"), index=False)
    # P1-C drive_id column check
    p1c_drive_ok = bool((c["drive_id"] == c["log_id"].map(drive_of)).all())
    I = lambda a, b, lev="drives": sorted(S[a][lev] & S[b][lev])  # noqa: E731
    checks = {
        "status": "POST-HOC audit of existing files",
        "drive_definition": "drive_id = first two '_' fields of the log name; identical to split.csv `drive` (all 147 rows) and to "
                            "P1-C units drive_id" + (" (verified)" if p1c_drive_ok else " (MISMATCH)"),
        "Aminus_logs": len(amap), "Aminus_drives": amd.drive_id.nunique(),
        "Aminus_multi_log_drives": amd.groupby("drive_id").filter(lambda x: len(x) > 1).groupby("drive_id")["log_id"].apply(list).to_dict(),
        "mech_poc_units_logs_drives": [len(S["mech_poc_units"]["logs"]), len(S["mech_poc_units"]["drives"])],
        "impromptu_logs_drives": [len(S["impromptu_exp33"]["logs"]), len(S["impromptu_exp33"]["drives"])],
        "impromptu_vs_autovla_mech_scene_overlap": len(S["impromptu_exp33"]["scenes"] & S["mech_poc_units"]["scenes"]),
        "p1c_eval_vs_mech_poc_shared_drives": I("p1c_eval_all", "mech_poc_units"),
        "p1c_eval_vs_poc_dev_discovery_shared_drives": I("p1c_eval_all", "poc_dev_discovery"),
        "p1c_eval_vs_impromptu_shared_drives": I("p1c_eval_all", "impromptu_exp33"),
        "p1c_eval_vs_pilot_shared_drives": I("p1c_eval_all", "p1c_pilot"),
        "p1c_eval_vs_poc_shared_day_vehicles": I("p1c_eval_all", "poc_dev_discovery", "dayveh"),
        "p1c_eval_units_logs_in_sel_dev_or_heldout": len(S["p1c_eval_units"]["logs"] & (S["sel_dev"]["logs"] | S["sel_heldout"]["logs"])),
        "p1c_eval_units_scenes_in_sel_dev_or_heldout": len(S["p1c_eval_units"]["scenes"] & (S["sel_dev"]["scenes"] | S["sel_heldout"]["scenes"])),
        "p1c_eval_logs_navhard_source": len(S["p1c_eval_all"]["logs"] & S["navhard_src"]["logs"]),
        "p1c_eval_drives_with_navhard_source_log": len(S["p1c_eval_all"]["drives"] & S["navhard_src"]["drives"]),
        "sel_dev_vs_sel_heldout_shared_logs": len(S["sel_dev"]["logs"] & S["sel_heldout"]["logs"]),
        "sel_dev_vs_sel_heldout_shared_drives": I("sel_dev", "sel_heldout"),
        "sel_dev_vs_sel_heldout_shared_day_vehicles": len(S["sel_dev"]["dayveh"] & S["sel_heldout"]["dayveh"]),
        "sel_heldout_logs_in_drives_shared_with_sel_dev": len({l for l in S["sel_heldout"]["logs"] if drive_of(l) in S["sel_dev"]["drives"]}),
        "sel_heldout_scenes_in_drives_shared_with_sel_dev": None,
        "sel_dev_vs_mech_poc_shared_drives": len(S["sel_dev"]["drives"] & S["mech_poc_units"]["drives"]),
        "sel_heldout_vs_mech_poc_shared_drives": len(S["sel_heldout"]["drives"] & S["mech_poc_units"]["drives"]),
        "navhard_half1_vs_half2_shared_logs": len(S["navhard_half1_exp25"]["logs"] & S["navhard_half2_exp26"]["logs"]),
        "navhard_half1_vs_half2_shared_drives": I("navhard_half1_exp25", "navhard_half2_exp26"),
        "navhard_src_vs_mech_poc_shared_logs": len(S["navhard_src"]["logs"] & S["mech_poc_units"]["logs"]),
        "navhard_src_vs_mech_poc_shared_drives": len(S["navhard_src"]["drives"] & S["mech_poc_units"]["drives"]),
    }
    # held-out scenes that sit in a drive also used by the dev selection set
    sel_pairs_h = jsonl_token_log([os.path.join(O, "heldout_best_of_n", g, "records.jsonl") for g in ("gpu0", "gpu1")])[2]
    checks["sel_heldout_scenes_in_drives_shared_with_sel_dev"] = sum(drive_of(l) in S["sel_dev"]["drives"] for _, l in sel_pairs_h)
    checks["sel_heldout_n_scenes"] = len(sel_pairs_h)
    json.dump(checks, open(os.path.join(OUT, "overlap_checks.json"), "w"), indent=1, default=list)
    print(json.dumps(checks, indent=1, default=list))
    print(pd.DataFrame(summ).to_string())


if __name__ == "__main__":
    main()
