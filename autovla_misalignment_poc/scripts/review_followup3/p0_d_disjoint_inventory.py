#!/usr/bin/env python
"""
P0-D "P5 preparation" (review_followup3, 2026-10-14): inventory of logs that are disjoint in DAY and VEHICLE from
  (a) development  = PoC dev_discovery logs (navtest shards 0-5; contains every mechanism / A- scene),
  (b) P1-C         = P1-C eval (64 logs) + pilot (2 logs),
  (c) training     = the AutoVLA checkpoint's training split as defined by the repository config.

POST-HOC data inventory. CPU only, no model inference, no existing file modified.

Training-split determination (recorded, not assumed):
  * /root/VLA/autovla/config/training/qwen2.5-vl-3B-mix-sft.yaml: train json_dataset_path ./dataset/nuplan/trainval
    (+ ./dataset/nuscenes/nuscenes_train), val = navtest.
  * /root/VLA/autovla/config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml: train scene_filter
    train_test_split/trainval.yaml (data_split trainval, scene_filter all_scenes = every trainval log), val = navtest.
  * The released checkpoint (HF Zewei-Zhou/AutoVLA @ a7d7ba3e, file AutoVLA_PDMS_89.ckpt) has no model card text on
    its training data; the repo config is the only official definition available. => training logs := OpenScene v1.1
    trainval (1,310 logs; streamed by p0_d_openscene_log_metadata.py). nuScenes train is a different dataset
    (Boston/Singapore, different vehicles naming) and is flagged separately.
  * Whether the checkpoint was model-selected on navtest (the configs' validation split) is NOT determinable.

Universe of non-training nuPlan logs on record: OpenScene v1.1 test = 147 logs (= P1-C split.csv universe; 136 of
them are in navtest with 12,146 scenes, 11 have no navtest scene). Other splits on disk: navmini / warmup_test_e2e
(log lists only) are checked against trainval/test; private_test_e2e has no public log list.

Expected eligible scenes use the P1-C stage-1 rates (64 eval logs): eligible E (first mismatch at t* <= 7) =
2,191/5,671 scenes; natural A- with t* <= 7 (stratum F) = 105/5,671. They are planning approximations, not data.

Run: nice -n 19 /root/miniforge3/envs/autovla/bin/python p0_d_disjoint_inventory.py
Out: outputs/review_followup3/P0/DISJOINT_LOG_INVENTORY.csv (one row per log), DISJOINT_LOG_INVENTORY_summary.csv,
     DISJOINT_LOG_INVENTORY_meta.json
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd
import yaml

POC = "/root/VLA/autovla_misalignment_poc"
OUT = f"{POC}/outputs/review_followup3/P0"
EXT = f"{OUT}/_external"
RF = f"{POC}/outputs/review_followup"
TTS = "/root/VLA/autovla/navsim/navsim/planning/script/config/common/train_test_split"
SPLIT = f"{RF}/P1C_new_log_replication/design_20261008/split.csv"
MEMB = f"{POC}/outputs/review_followup2/A_autovla_cluster/run_20261013/drive_membership.csv"
STAGE1_N, STAGE1_E, STAGE1_F = 5671, 2191, 105          # P1-C stage-1 counts (POPULATION_selection_flow.csv)


def ids(log):
    f = log.split("_")
    return f"{f[0]}_{f[1]}", f[0][:10], f[1], f"{f[0][:10]}_{f[1]}"


def frame(logs, **cols):
    d = pd.DataFrame({"log": sorted(set(logs))})
    x = d.log.map(ids)
    d["drive"], d["day"], d["vehicle"], d["day_vehicle"] = zip(*x) if len(d) else ([], [], [], [])
    for k, v in cols.items():
        d[k] = v
    return d


def main():
    tv = pd.read_csv(f"{EXT}/openscene_v1.1_trainval_log_metadata.csv")
    te = pd.read_csv(f"{EXT}/openscene_v1.1_test_log_metadata.csv")
    prov = {s: json.load(open(f"{EXT}/openscene_v1.1_{s}_log_metadata.json")) for s in ("trainval", "test")}
    split = pd.read_csv(SPLIT, dtype=str)
    nt = yaml.safe_load(open(f"{TTS}/scene_filter/navtest.yaml"))
    navtest_logs = set(nt["log_names"])
    # reference sets ---------------------------------------------------------------------------------------------
    dev = set(split[split.role == "dev_discovery"].log)
    p1c = set(split[split.role.isin(["eval", "pilot_harness_check"])].log)
    adj = set(split[split.role == "excluded_drive_adjacent_to_dev"].log)
    train = set(tv.log_name)
    ref = {}
    for nm, logs in (("dev", dev), ("p1c", p1c), ("train", train)):
        f = frame(logs)
        ref[nm] = {k: set(f[k]) for k in ("log", "drive", "day", "vehicle", "day_vehicle")}
    mem = pd.read_csv(MEMB)
    sel_drives = set(mem[(mem.sel_dev > 0) | (mem.sel_heldout > 0)].drive_id)
    nh_drives = set(mem[mem.navhard_src > 0].drive_id)

    # candidate rows: all OpenScene test logs + logs of the other splits on disk -------------------------------------
    rows = []
    city = dict(zip(te.log_name, te.map_location)) | dict(zip(tv.log_name, tv.map_location))
    nfr = dict(zip(te.log_name, te.n_frames)) | dict(zip(tv.log_name, tv.n_frames))
    sp = split.set_index("log")
    other = {}
    for nm in ("navmini", "warmup_test_e2e", "private_test_e2e"):
        y = yaml.safe_load(open(f"{TTS}/scene_filter/{nm}.yaml"))
        other[nm] = set(y.get("log_names") or [])
    cand = [(l, "OpenScene v1.1 test (nuPlan test)") for l in sorted(te.log_name)]
    for nm, ls in other.items():
        cand += [(l, f"navsim {nm} log list") for l in sorted(ls)]
    for log, src in cand:
        drv, day, veh, dv = ids(log)
        r = {"log": log, "source_split": src, "drive": drv, "day": day, "vehicle": veh, "day_vehicle": dv,
             "city": city.get(log, "unknown (log not in trainval/test metadata)"), "n_frames": nfr.get(log),
             "in_openscene_trainval": log in train, "in_openscene_test": log in set(te.log_name),
             "in_navtest": log in navtest_logs,
             "p1c_split_role": sp.role.get(log, "") if log in sp.index else "",
             "n_scenes_navtest": int(sp.n_scenes_navtest.get(log, 0)) if log in sp.index else 0,
             "images_local": (sp.images_local.get(log) == "True") if log in sp.index else False,
             "used_in_selection_expts_drive": drv in sel_drives, "navhard_source_drive": drv in nh_drives}
        for nm in ("dev", "p1c", "train"):
            for k in ("log", "drive", "day", "vehicle", "day_vehicle"):
                r[f"shares_{k}_with_{nm}"] = (log if k == "log" else r[k]) in ref[nm][k]
        r["day_disjoint_all3"] = not any(r[f"shares_day_with_{n}"] for n in ("dev", "p1c", "train"))
        r["vehicle_disjoint_all3"] = not any(r[f"shares_vehicle_with_{n}"] for n in ("dev", "p1c", "train"))
        r["day_and_vehicle_disjoint_all3"] = r["day_disjoint_all3"] and r["vehicle_disjoint_all3"]
        r["day_and_vehicle_disjoint_dev_p1c_only"] = not any(r[f"shares_{k}_with_{n}"] for n in ("dev", "p1c")
                                                             for k in ("day", "vehicle"))
        r["day_and_vehicle_disjoint_train_only"] = not (r["shares_day_with_train"] or r["shares_vehicle_with_train"])
        r["dayvehicle_disjoint_all3"] = not any(r[f"shares_day_vehicle_with_{n}"] for n in ("dev", "p1c", "train"))
        r["exp_eligible_scenes_E"] = r["n_scenes_navtest"] * STAGE1_E / STAGE1_N
        r["exp_F_stratum_scenes"] = r["n_scenes_navtest"] * STAGE1_F / STAGE1_N
        city_ref = {c for nm in ("dev", "p1c") for c in [city.get(l) for l in (dev if nm == "dev" else p1c)]}
        r["other_city_than_dev_p1c"] = r["city"] not in city_ref
        rows.append(r)
    inv = pd.DataFrame(rows)
    inv["p5_tier"] = "not eligible"
    inv.loc[inv.dayvehicle_disjoint_all3, "p5_tier"] = "T3 day-vehicle-disjoint only (shares day or vehicle)"
    inv.loc[inv.day_and_vehicle_disjoint_dev_p1c_only & ~inv.day_and_vehicle_disjoint_all3, "p5_tier"] = \
        "T2 day+vehicle-disjoint from dev and P1-C, NOT from training"
    inv.loc[inv.day_and_vehicle_disjoint_all3, "p5_tier"] = "T1 day+vehicle-disjoint from dev, P1-C and training"
    inv.loc[inv.in_openscene_trainval, "p5_tier"] = "excluded: training log"
    inv.to_csv(f"{OUT}/DISJOINT_LOG_INVENTORY.csv", index=False)

    # summary ----------------------------------------------------------------------------------------------------
    S = []

    def summ(name, q):
        S.append({"subset": name, "n_logs": len(q), "n_drives": q.drive.nunique(), "n_days": q.day.nunique(),
                  "n_vehicles": q.vehicle.nunique(), "n_day_vehicles": q.day_vehicle.nunique(),
                  "cities": ";".join(f"{k}:{v}" for k, v in q.city.value_counts().items()),
                  "navtest_scenes": int(q.n_scenes_navtest.sum()), "exp_eligible_E": round(q.exp_eligible_scenes_E.sum(), 1),
                  "exp_F": round(q.exp_F_stratum_scenes.sum(), 1),
                  "images_local_logs": int(q.images_local.sum()),
                  "logs": ";".join(q.log) if len(q) <= 30 else f"{len(q)} logs (see per-log rows)"})

    T = inv[inv.source_split.str.startswith("OpenScene")]
    summ("ALL OpenScene test logs", T)
    summ("ref: dev_discovery", T[T.p1c_split_role == "dev_discovery"])
    summ("ref: P1-C eval + pilot", T[T.p1c_split_role.isin(["eval", "pilot_harness_check"])])
    summ("test logs not used by dev / P1-C (adjacent-to-dev + reserve)", T[T.p1c_split_role.isin(["excluded_drive_adjacent_to_dev", "reserve_non_navtest_not_used"])])
    for col, lab in (("day_and_vehicle_disjoint_all3", "T1 day AND vehicle disjoint from dev, P1-C, training"),
                     ("day_and_vehicle_disjoint_dev_p1c_only", "day AND vehicle disjoint from dev and P1-C (training ignored)"),
                     ("day_and_vehicle_disjoint_train_only", "day AND vehicle disjoint from training only"),
                     ("day_disjoint_all3", "day-disjoint from all three (vehicle may be shared)"),
                     ("vehicle_disjoint_all3", "vehicle-disjoint from all three (day may be shared)"),
                     ("dayvehicle_disjoint_all3", "day-vehicle-disjoint from all three")):
        summ(lab, T[T[col]])
    for c in ("dev", "p1c", "train"):
        summ(f"test logs sharing NO day with {c}", T[~T[f"shares_day_with_{c}"]])
        summ(f"test logs sharing NO vehicle with {c}", T[~T[f"shares_vehicle_with_{c}"]])
    for nm in other:
        q = inv[inv.source_split == f"navsim {nm} log list"]
        summ(f"other split: {nm} (all)", q)
        summ(f"other split: {nm}, in trainval", q[q.in_openscene_trainval])
        summ(f"other split: {nm}, T1 day+vehicle disjoint", q[q.day_and_vehicle_disjoint_all3])
    sm = pd.DataFrame(S)
    sm.to_csv(f"{OUT}/DISJOINT_LOG_INVENTORY_summary.csv", index=False)
    meta = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "POST-HOC inventory",
            "training_split_definition": "OpenScene v1.1 trainval, all scenes (AutoVLA repo configs mix-sft + nuplan-grpo-cot); "
                                         "+ nuScenes train (different dataset). Checkpoint model card silent; navtest is the "
                                         "configs' validation split (model selection on navtest not determinable).",
            "openscene_provenance": prov,
            "trainval": {"n_logs": len(tv), "n_days": tv.log_name.str[:10].nunique(),
                         "vehicles": sorted(tv.vehicle_name.unique()), "cities": tv.map_location.value_counts().to_dict()},
            "test": {"n_logs": len(te), "n_days": te.log_name.str[:10].nunique(), "vehicles": sorted(te.vehicle_name.unique()),
                     "cities": te.map_location.value_counts().to_dict(), "navtest_logs": len(navtest_logs),
                     "navtest_logs_in_test": len(navtest_logs & set(te.log_name)), "navtest_tokens_yaml": len(nt.get("tokens") or []),
                     "navtest_scenes_split_csv_sum": int(T.n_scenes_navtest.sum())},
            "test_vs_trainval": {"shared_logs": len(train & set(te.log_name)),
                                 "test_days_not_in_trainval": sorted(set(te.log_name.str[:10]) - set(tv.log_name.str[:10])),
                                 "test_vehicles_not_in_trainval": sorted(set(te.vehicle_name) - set(tv.vehicle_name))},
            "ref_sets": {nm: {k: len(v) for k, v in ref[nm].items()} for nm in ref},
            "ref_days": {nm: sorted(ref[nm]["day"]) for nm in ("dev", "p1c")},
            "ref_vehicles": {nm: sorted(ref[nm]["vehicle"]) for nm in ("dev", "p1c")},
            "rates_used": {"E_per_scene": STAGE1_E / STAGE1_N, "F_per_scene": STAGE1_F / STAGE1_N,
                           "source": "P1-C stage 1 (POPULATION_selection_flow.csv)"},
            "inputs": [SPLIT, MEMB, f"{TTS}/scene_filter/navtest.yaml", f"{EXT}/openscene_v1.1_trainval_log_metadata.csv",
                       f"{EXT}/openscene_v1.1_test_log_metadata.csv"]}
    json.dump(meta, open(f"{OUT}/DISJOINT_LOG_INVENTORY_meta.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(sm.drop(columns=["logs"]).to_string())
    print(json.dumps({k: meta[k] for k in ("test_vs_trainval", "ref_days", "ref_vehicles")}, indent=1))


if __name__ == "__main__":
    main()
