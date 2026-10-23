#!/usr/bin/env python
"""review_followup5 C_DISJOINT (CPU; protocol §4): inventory and day/vehicle overlap audit of every evaluation source.

Sources:
  * review_followup3 P0-D per-log inventory (OpenScene v1.1 test = navtest logs, navmini, warmup_test_e2e; trainval days /
    vehicles from the streamed trainval metadata), kept as is;
  * NAVSIM v2 navhard two-stage metadata on disk (/root/VLA/navhard/navhard_two_stage/openscene_meta_datas, log names);
  * NEW: OpenScene v1.1 private_test_hard metadata (179 kB, streamed from the pinned dataset commit, never written to disk;
    only log-level fields kept).
Eligibility for an independent day/vehicle replication: the log's day and its vehicle both lie outside every connected
day-vehicle component of dev + P1-C (B2 component map), and outside the known training split (OpenScene trainval days /
vehicles; unknown -> flagged). If none: BLOCKED_NO_DISJOINT_DATA.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python c_disjoint.py
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import pickle
import tarfile
import urllib.request

import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
OUT = f"{POC}/outputs/review_followup5/C_DISJOINT"
INV = f"{POC}/outputs/review_followup3/P0/DISJOINT_LOG_INVENTORY.csv"
INVM = f"{POC}/outputs/review_followup3/P0/DISJOINT_LOG_INVENTORY_meta.json"
COMP = f"{POC}/outputs/review_followup5/A_B2_CLUSTER/b2_cluster_meta.json"
NAVHARD = "/root/VLA/navhard/navhard_two_stage/openscene_meta_datas"
COMMIT = "a76f840b65e972bc45e56c2adced897498e9a026"
URL = f"https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{COMMIT}/openscene-v1.1/openscene_metadata_private_test_hard.tar.gz"


def dv_of(log):
    import re
    m = re.search(r"(\d{4}\.\d{2}\.\d{2})\.[\d.]+_(veh-\d+)", str(log))
    return (m.group(1), m.group(2)) if m else (None, None)


def datetime_utc(ts):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts / 1e6, timezone.utc).strftime("%Y.%m.%d") if ts else None


def stream_private():
    h = hashlib.sha256()
    data = urllib.request.urlopen(URL, timeout=120).read()
    h.update(data)
    logs = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf:
            if not m.name.endswith(".pkl"):
                continue
            frames = pickle.load(tf.extractfile(m))
            f0 = frames[0] if isinstance(frames, list) and frames else {}
            ln = f0.get("log_name", os.path.basename(m.name)[:-4])
            for fr in frames:
                ln = fr.get("log_name")
                logs.setdefault(ln, {"log": ln, "n_frames": 0, "map_location": fr.get("map_location"),
                                     "vehicle_name": fr.get("vehicle_name"), "utc_day": datetime_utc(fr.get("timestamp"))})
                logs[ln]["n_frames"] += 1
    return pd.DataFrame(logs.values()), {"url": URL, "dataset_commit": COMMIT, "bytes": len(data), "stream_sha256": h.hexdigest(),
                                         "note": "streamed, never written to disk; only log-level fields kept"}


def main():
    os.makedirs(OUT, exist_ok=True)
    inv = pd.read_csv(INV)
    meta = json.load(open(INVM))
    comp = json.load(open(COMP))["combined_graph"]["components_dev_plus_p1c"]
    days, vehs = set(), set()
    for k, dvs in comp.items():
        for x in dvs:
            d, v = x.split("_", 1); days.add(d); vehs.add(v)
    tr_days = set(meta["trainval"].get("days", [])) if isinstance(meta["trainval"].get("days"), list) else set()
    tr_veh = set(meta["trainval"]["vehicles"])
    rows = []
    for _, r in inv.iterrows():
        d, v = dv_of(r.log)
        rows.append({"source": r.source_split, "log": r.log, "day": d, "vehicle": v, "city": r.city,
                     "day_in_dev_p1c_components": d in days, "vehicle_in_dev_p1c_components": v in vehs,
                     "vehicle_in_trainval": v in tr_veh, "shares_day_with_train (P0-D)": r.shares_day_with_train})
    from datetime import datetime, timezone, timedelta
    TZ = {"us-nv-las-vegas-strip": -7, "us-ma-boston": -4, "us-pa-pittsburgh-hazelwood": -4, "sg-one-north": 8}

    def local_day(ts, city):
        if ts is None or city not in TZ:
            return None
        return (datetime.fromtimestamp(ts / 1e6, timezone.utc) + timedelta(hours=TZ[city])).strftime("%Y.%m.%d")
    nh = sorted(glob.glob(f"{NAVHARD}/*.pkl"))
    for p in nh:
        fr = pickle.load(open(p, "rb"))
        f0 = fr[0]
        d, v, city = local_day(f0.get("timestamp"), f0.get("map_location")), f0.get("vehicle_name"), f0.get("map_location")
        rows.append({"source": "navsim v2 navhard_two_stage (on disk; anonymised log, day from timestamp + city timezone)",
                     "log": f0.get("log_name"), "day": d, "vehicle": v, "city": city,
                     "day_in_dev_p1c_components": d in days, "vehicle_in_dev_p1c_components": v in vehs,
                     "vehicle_in_trainval": v in tr_veh, "shares_day_with_train (P0-D)": None})
    priv_info = {}
    try:
        pv, priv_info = stream_private()
        for _, r in pv.iterrows():
            d, v = None, None      # log name, vehicle and map are anonymised in private_test_hard: day / vehicle unknown
            rows.append({"source": "OpenScene v1.1 private_test_hard (metadata streamed)", "log": r.log, "day": d, "vehicle": v,
                         "city": r.map_location, "day_in_dev_p1c_components": d in days, "vehicle_in_dev_p1c_components": v in vehs,
                         "vehicle_in_trainval": v in tr_veh, "shares_day_with_train (P0-D)": None})
    except Exception as e:                 # network / format problem: recorded, not fatal
        priv_info = {"error": repr(e), "url": URL}
    df = pd.DataFrame(rows)
    df["day_vehicle_parsed"] = df.day.notna() & df.vehicle.notna()
    df["eligible_day_and_vehicle_disjoint_from_dev_p1c"] = df.day_vehicle_parsed & ~df.day_in_dev_p1c_components & ~df.vehicle_in_dev_p1c_components
    df["eligible_also_vehicle_not_in_trainval"] = df.eligible_day_and_vehicle_disjoint_from_dev_p1c & ~df.vehicle_in_trainval
    tv = set(inv[inv.in_openscene_trainval.astype(bool)].log)
    df["log_in_openscene_trainval (training data)"] = df.log.isin(tv)
    df["eligible_final"] = df.eligible_day_and_vehicle_disjoint_from_dev_p1c & ~df["log_in_openscene_trainval (training data)"]
    df.to_csv(f"{OUT}/DATA_INVENTORY.csv", index=False)
    summ = df.groupby("source").agg(n_logs=("log", "nunique"), n_days=("day", "nunique"), n_vehicles=("vehicle", "nunique"),
                                    eligible_vs_dev_p1c=("eligible_day_and_vehicle_disjoint_from_dev_p1c", "sum"),
                                    eligible_also_not_trainval_vehicle=("eligible_also_vehicle_not_in_trainval", "sum"),
                                    in_training_data=("log_in_openscene_trainval (training data)", "sum"),
                                    day_vehicle_unknown=("day_vehicle_parsed", lambda x: int((~x).sum())),
                                    eligible_final=("eligible_final", "sum")).reset_index()
    summ.to_csv(f"{OUT}/DATA_INVENTORY_summary.csv", index=False)
    el = df[df.eligible_final]
    status = "ELIGIBLE_DATA_FOUND" if len(el) else "BLOCKED_NO_DISJOINT_DATA"
    out = {"status": status, "dev_p1c_component_days": sorted(days), "dev_p1c_component_vehicles": sorted(vehs),
           "n_components_dev_plus_p1c": len(comp), "trainval_vehicles": sorted(tr_veh), "navhard_pickles_on_disk": len(nh),
           "navhard_scenes": len(nh), "private_test_hard": priv_info,
           "eligible_logs": el[["source", "log", "day", "vehicle", "city", "vehicle_in_trainval"]].to_dict("records"),
           "summary": summ.to_dict("records")}
    json.dump(out, open(f"{OUT}/overlap_audit.json", "w"), indent=1, default=str)
    print(summ.to_string()); print(status, len(el))
    if len(el):
        print(el[["source", "log", "day", "vehicle", "city", "vehicle_in_trainval"]].drop_duplicates().to_string())


if __name__ == "__main__":
    main()
