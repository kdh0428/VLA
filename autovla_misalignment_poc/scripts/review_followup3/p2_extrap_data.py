#!/usr/bin/env python
"""
P2-B data (review_followup3, 2026-10-14).  Spec: outputs/review_followup3/P2/LEARNED_EXTRAPOLATOR_PROTOCOL.md (section 5).

Stage 1 (stream):  streams OpenScene v1.1 openscene_metadata_trainval.tgz (dataset commit a76f840b, the AutoVLA
                   checkpoint training split) through tarfile 'r|gz' -- the tarball is NEVER written -- and keeps only
                   per-frame ego fields -> P2/_extrap_data/trainval_ego_frames.parquet
Stage 2 (samples): drive split (drive-disjoint from every OpenScene test drive = dev, P1-C, P5 candidates), plan
                   frames (even frame_idx, 10 future frames, dt in [0.45, 0.55] s, step <= 15 m), GT future in the
                   plan frame, AutoVLA GT->token projection (numpy port) + decode -> P2/_extrap_data/samples.npz,
                   P2/extrap_drive_separation.json
Run:  nice -n 19 python p2_extrap_data.py stream ; nice -n 19 python p2_extrap_data.py samples
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import tarfile
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p2_extrap_common as X  # noqa: E402

POC = "/root/VLA/autovla_misalignment_poc"
OUT = f"{POC}/outputs/review_followup3/P2"
DD = f"{OUT}/_extrap_data"
REV = "a76f840b65e972bc45e56c2adced897498e9a026"
URL = f"https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{REV}/openscene-v1.1/openscene_metadata_trainval.tgz"
TEST_META = f"{POC}/outputs/review_followup3/P0/_external/openscene_v1.1_test_log_metadata.csv"
DRIVE_MAP = f"{POC}/outputs/review_followup2/A_autovla_cluster/run_20261013/drive_membership.csv"
INVENTORY = f"{POC}/outputs/review_followup3/P0/DISJOINT_LOG_INVENTORY.csv"
C1U = f"{POC}/outputs/review_followup2/C_baseline_window/run_20261013/c1_units_counterfactual.parquet"
VAL_FRAC = 0.15
SPLIT_SEED = 20261014


def drive_of(log):
    p = log.split("_")
    return f"{p[0]}_{p[1]}"


def check_lock():
    lock = json.load(open(f"{OUT}/extrap_protocol_lock.json"))
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert sha(f"{OUT}/LEARNED_EXTRAPOLATOR_PROTOCOL.md") == lock["LEARNED_EXTRAPOLATOR_PROTOCOL.md_sha256"], "protocol changed"
    assert sha(os.path.join(HERE, "p2_extrap_common.py")) == lock["p2_extrap_common.py_sha256"], "common changed"
    return lock


def stream():
    os.makedirs(DD, exist_ok=True)
    t0 = time.time()
    p = subprocess.Popen(["curl", "-sL", "--fail", URL], stdout=subprocess.PIPE)
    h = hashlib.sha256()

    class Tee:
        def read(self, n=-1):
            b = p.stdout.read(n)
            h.update(b)
            return b

    rows, nlog = [], 0
    with tarfile.open(fileobj=Tee(), mode="r|gz") as tf:
        for m in tf:
            if not (m.isfile() and m.name.endswith(".pkl")):
                continue
            d = pickle.load(tf.extractfile(m))
            nlog += 1
            for fr in d:
                e = fr["ego_dynamic_state"]; t = fr["ego2global_translation"]
                rows.append((fr["log_name"], fr["token"], int(fr["frame_idx"]), int(fr["timestamp"]), float(t[0]), float(t[1]),
                             X.quat_yaw(fr["ego2global_rotation"]), float(e[0]), float(e[1]), float(e[2]), float(e[3]),
                             int(np.argmax(fr["driving_command"])), fr.get("vehicle_name"), fr.get("map_location")))
            if nlog % 100 == 0:
                print(f"[{time.time() - t0:6.0f}s] {nlog} logs {len(rows)} frames", flush=True)
    rc = p.wait()
    df = pd.DataFrame(rows, columns=["log_name", "token", "frame_idx", "timestamp", "x", "y", "yaw", "vx", "vy", "ax", "ay",
                                     "cmd", "vehicle", "city"])
    df.to_parquet(f"{DD}/trainval_ego_frames.parquet", index=False)
    prov = {"url": URL, "curl_returncode": rc, "stream_sha256": h.hexdigest(), "n_logs": nlog, "n_frames": len(df),
            "seconds": round(time.time() - t0, 1), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "streamed (7.05 GB network), tarball never written; only per-frame ego fields kept"}
    json.dump(prov, open(f"{DD}/trainval_ego_frames_provenance.json", "w"), indent=1)
    print(json.dumps(prov, indent=1))
    assert rc == 0


def _tok_worker(P):
    cb, _, _ = X.C.codebook_geometry()
    return X.tokenize_gt(P, cb)


def samples():
    lock = check_lock()
    F = pd.read_parquet(f"{DD}/trainval_ego_frames.parquet")
    F["drive"] = F.log_name.map(drive_of)
    F["day"] = F.drive.str[:10]; F["veh"] = F.drive.str.split("_").str[1]
    # ---------------- drive separation
    test = pd.read_csv(TEST_META); test_drives = set(test.log_name.map(drive_of))
    dm = pd.read_csv(DRIVE_MAP)
    dev_drives = set(dm.loc[dm.poc_dev_discovery > 0, "drive_id"]) if "poc_dev_discovery" in dm else set()
    p1c_drives = set(dm.loc[(dm.p1c_eval_all > 0) | (dm.p1c_pilot > 0), "drive_id"])
    inv = pd.read_csv(INVENTORY)
    # P5: no eligible candidate exists (P0: P5 NOT_RUN); the whole P5 candidate POOL = every inventory log that is not a
    # trainval log (OpenScene test + navmini / warmup logs outside trainval) is excluded
    p5_drives = set(inv.loc[~inv.in_openscene_trainval.astype(bool), "drive"])
    test_drives = test_drives | p5_drives
    eval_drives = set(pd.read_parquet(C1U, columns=["drive_id"]).drive_id)
    all_tv = sorted(F.drive.unique())
    excluded = sorted(set(all_tv) & test_drives)
    elig = [d for d in all_tv if d not in test_drives]
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(elig))
    n_val = int(math.ceil(VAL_FRAC * len(elig)))
    val = sorted(elig[i] for i in perm[:n_val]); tr = sorted(elig[i] for i in perm[n_val:])
    for ref, nm in ((dev_drives, "dev"), (p1c_drives, "p1c"), (p5_drives, "p5"), (eval_drives, "p1a_eval"), (test_drives, "test")):
        assert not (set(tr) | set(val)) & ref, f"overlap with {nm}"
    assert not set(tr) & set(val)
    days = lambda s: {d[:10] for d in s}; vehs = lambda s: {d.split("_")[1] for d in s}
    sep = {"protocol_lock": lock, "n_trainval_drives": len(all_tv), "n_excluded_test_drives": len(excluded),
           "excluded_drives": excluded, "n_train_drives": len(tr), "n_val_drives": len(val),
           "train_drives": tr, "val_drives": val,
           "reference_sets": {"dev_discovery": len(dev_drives), "p1c_eval_or_pilot": len(p1c_drives), "p5_candidates": len(p5_drives),
                              "p1a_eval_units": len(eval_drives), "openscene_test_all": len(test_drives)},
           "drive_overlap": {nm: len((set(tr) | set(val)) & ref) for ref, nm in
                             ((dev_drives, "dev"), (p1c_drives, "p1c"), (p5_drives, "p5"), (eval_drives, "p1a_eval"), (test_drives, "test"))},
           "train_val_drive_overlap": len(set(tr) & set(val)),
           "descriptive_day_overlap_train_vs_p1a_eval": sorted(days(tr) & days(eval_drives)),
           "descriptive_vehicle_overlap_train_vs_p1a_eval": sorted(vehs(tr) & vehs(eval_drives)),
           "descriptive_day_overlap_val_vs_p1a_eval": sorted(days(val) & days(eval_drives)),
           "split_rule": f"rng({SPLIT_SEED}).permutation over sorted eligible drives; first ceil({VAL_FRAC}*n) = validation"}
    # ---------------- plan frames
    # frame_idx is the index WITHIN an OpenScene scene (0..40) and restarts per scene, so frames are ordered by timestamp
    # inside each log and de-duplicated on (log, timestamp); temporal adjacency is checked with timestamps only
    # (implementation fix made before any sample/training existed: a first run with a frame_idx-consecutive check
    #  rejected every candidate because sorting by frame_idx interleaved scenes)
    n_dup = int(F.duplicated(["log_name", "timestamp"]).sum())
    F = F.drop_duplicates(["log_name", "timestamp"]).sort_values(["log_name", "timestamp"]).reset_index(drop=True)
    sep["n_duplicate_frames_dropped"] = n_dup
    recs = {"P": [], "ego": [], "split": [], "drive": [], "log": [], "token": []}
    n_cand = n_dt = n_step = 0
    for log, g in F.groupby("log_name", sort=True):
        d = g.drive.iloc[0]
        if d not in tr and d not in val:
            continue
        x, y, h, ts, fi = g.x.values, g.y.values, g.yaw.values, g.timestamp.values, g.frame_idx.values
        for i in range(len(g) - 10):
            if fi[i] % 2:
                continue
            n_cand += 1
            dt = np.diff(ts[i:i + 11]) / 1e6
            if not ((dt >= 0.45) & (dt <= 0.55)).all():
                n_dt += 1; continue
            c, s = math.cos(h[i]), math.sin(h[i])
            dx, dy = x[i + 1:i + 11] - x[i], y[i + 1:i + 11] - y[i]
            P = np.c_[c * dx + s * dy, -s * dx + c * dy, X.wrap(h[i + 1:i + 11] - h[i])]
            st = np.linalg.norm(np.diff(np.vstack([[0, 0], P[:, :2]]), axis=0), axis=1)
            if st.max() > 15.0:
                n_step += 1; continue
            recs["P"].append(P)
            recs["ego"].append(X.ego_vector([g.vx.values[i], g.vy.values[i]], [g.ax.values[i], g.ay.values[i]],
                                            np.eye(4)[g.cmd.values[i]])[0])
            recs["split"].append(0 if d in tr else 1); recs["drive"].append(d); recs["log"].append(log)
            recs["token"].append(g.token.values[i])
    P = np.asarray(recs["P"])
    print(f"plan frames {len(P)} (candidates {n_cand}, dt/frame gaps {n_dt}, step>15m {n_step})", flush=True)
    t0 = time.time()
    chunks = np.array_split(P, 64)
    with Pool(8) as pool:
        res = pool.map(_tok_worker, chunks)
    T = np.concatenate([r[0] for r in res]); D = np.concatenate([r[1] for r in res])
    print(f"tokenized in {time.time() - t0:.0f}s", flush=True)
    quant = np.linalg.norm(D[..., :2] - P[..., :2], axis=-1)
    np.savez_compressed(f"{DD}/samples.npz", gt=P.astype(np.float32), tok=T.astype(np.int16), dec=D.astype(np.float64),
                        ego=np.asarray(recs["ego"], np.float32), split=np.asarray(recs["split"], np.int8),
                        drive=np.asarray(recs["drive"]), log=np.asarray(recs["log"]), token=np.asarray(recs["token"]))
    sep.update({"n_plan_frames": int(len(P)), "n_plan_frames_train": int((np.asarray(recs['split']) == 0).sum()),
                "n_plan_frames_val": int((np.asarray(recs['split']) == 1).sum()), "n_candidates_even": n_cand,
                "n_dropped_dt_or_gap": n_dt, "n_dropped_step_gt_15m": n_step,
                "token_quantization_residual_m": {"median": float(np.median(quant)), "p99": float(np.quantile(quant, .99)),
                                                  "max": float(quant.max())}})
    json.dump(sep, open(f"{OUT}/extrap_drive_separation.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in sep.items() if not isinstance(v, list)}, indent=1))


if __name__ == "__main__":
    {"stream": stream, "samples": samples}[sys.argv[1]]()
