"""
P1-C shared definitions (CPU only; no model import).

Everything here is fixed by the committed preregistration (commit b212637):
  outputs/review_followup/P1C_new_log_replication/design_20261008/{protocol.md, analysis.md, split.csv}
  - eligibility E: natural plan has 10 poses, no CoT stub, no runaway, t* exists and t* <= 7      (protocol §3)
  - strata: F = eligible A- (pi = 1); S = eligible A+, per log segment the first m_seg = min(4, N_S,seg)
    scenes in sha256("P1C-20261008|" + scene_token) order, pi = m_seg / N_S,seg, w = 1 / pi          (protocol §3)
  - cluster = drive (<date.time>_veh-NN)                                                            (protocol §2)
Nothing in this module reads a perturbation outcome.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re

ROOT = "/root/VLA"
POC = f"{ROOT}/autovla_misalignment_poc"
SCRIPTS = f"{POC}/scripts"
AUTOVLA_DIR = f"{ROOT}/autovla"
NUP = f"{AUTOVLA_DIR}/dataset/nuplan"
SCENES = f"{NUP}/navtest_ext"
SENSOR = f"{NUP}/sensor_blobs/test"
P1C = f"{POC}/outputs/review_followup/P1C_new_log_replication"
DESIGN = f"{P1C}/design_20261008"
SPLIT = f"{DESIGN}/split.csv"
POC_FILTER = f"{POC}/configs/scene_filter_navtest_subset.yaml"
ANNO_DIRS = [f"{NUP}/_stream_gpu1/anno", f"{NUP}/_stream_gpu0/anno", f"{POC}/outputs/annotations"]

M_S = 4                      # A+ scenes per log segment
T_STAR_MAX = 7               # eligibility: t* <= 7
N_ACT = 10
SALT = "P1C-20261008|"
CAMS = ("CAM_F0", "CAM_L1", "CAM_R1")   # the only cameras AutoVLA reads (models/autovla.py camera_types)
LOG_RE = re.compile(r"^20[0-9]{2}\.[0-9]{2}\.[0-9]{2}\.[0-9.]+_veh-[0-9]+_[0-9]+_[0-9]+$")


def sha256_file(p, bs=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def s_order_key(token: str) -> str:
    return hashlib.sha256((SALT + token).encode()).hexdigest()


def load_split(path=SPLIT):
    return {r["log"]: r for r in csv.DictReader(open(path))}


def logs_with_role(role, path=SPLIT):
    return sorted(l for l, r in load_split(path).items() if r["role"] == role)


def protected_logs():
    import yaml
    return set(yaml.safe_load(open(POC_FILTER))["log_names"])


def log_of_scene(scene: dict) -> str:
    return scene["front_camera_paths"][0].split("/")[0]


def scene_index(cache=f"{P1C}/work_scene_index.json"):
    """token -> log for every preprocessed navtest scene (built once from navtest_ext, cached)."""
    if os.path.exists(cache):
        return json.load(open(cache))
    out = {}
    for f in sorted(os.listdir(SCENES)):
        if f.endswith(".json"):
            out[f[:-5]] = log_of_scene(json.load(open(os.path.join(SCENES, f))))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(out, open(cache, "w"))
    return out


def tokens_of_logs(logs):
    logs = set(logs)
    idx = scene_index()
    return sorted(t for t, l in idx.items() if l in logs)


def image_files_of_scene(scene: dict):
    """relative paths (under sensor_blobs/test) of the images AutoVLA loads for this scene: 3 cameras x 4 frames."""
    out = []
    for k in ("front_camera_paths", "left_camera_paths", "right_camera_paths"):
        out += list(scene[k])
    return out


def stage1_labels(pred, gt, traj, gtraj, cot, runaway, a_eval_fn):
    """Stage-1 per-scene labels from the natural plan only (protocol §3). Returns dict with t_star, a_eval, fde5,
    eligible, exclusion_reason. a_eval_fn = analyze_action_history.a_eval (unchanged dev rule)."""
    import numpy as np
    out = {"t_star": None, "a_eval": None, "fde5": None, "eligible": False, "exclusion_reason": None}
    if traj is None or len(traj) < N_ACT or len(pred) < N_ACT:
        out["exclusion_reason"] = "invalid_plan"
        return out
    p, g = np.asarray(traj, float)[:, :2], np.asarray(gtraj, float)[:, :2]
    out["fde5"] = float(np.linalg.norm(p[-1] - g[-1]))
    out["a_eval"] = "A+" if a_eval_fn(traj, gtraj) else "A-"
    t = next((k for k in range(N_ACT) if pred[k] != gt[k]), None)
    out["t_star"] = t
    if cot:
        out["exclusion_reason"] = "cot_stub"
    elif runaway:
        out["exclusion_reason"] = "runaway"
    elif t is None:
        out["exclusion_reason"] = "no_mismatch"
    elif t > T_STAR_MAX:
        out["exclusion_reason"] = "t_star_gt_7"
    else:
        out["eligible"] = True
    return out


def select_strata(stage1_rows, m=M_S):
    """stage1_rows: list of dicts with token, log, eligible, a_eval, exclusion_reason (stage-1 output only).
    Returns (selected rows with stratum/pi/weight/N_S_seg/m_seg, per-log denominator table).
    F = all eligible A-; S = first min(m, N_S,seg) eligible A+ per log segment in sha256 order."""
    from collections import defaultdict
    by_log = defaultdict(list)
    for r in stage1_rows:
        by_log[r["log"]].append(r)
    sel, den = [], {}
    for log in sorted(by_log):
        rows = by_log[log]
        F = sorted((r for r in rows if r["eligible"] and r["a_eval"] == "A-"), key=lambda r: r["token"])
        S_all = sorted((r for r in rows if r["eligible"] and r["a_eval"] == "A+"), key=lambda r: s_order_key(r["token"]))
        m_seg = min(m, len(S_all))
        S = S_all[:m_seg]
        for r in F:
            sel.append({"token": r["token"], "log": log, "stratum": "F", "pi": 1.0, "weight": 1.0,
                        "N_S_seg": len(S_all), "m_seg": m_seg, "t_star": r["t_star"]})
        for r in S:
            pi = m_seg / len(S_all)
            sel.append({"token": r["token"], "log": log, "stratum": "S", "pi": pi, "weight": 1.0 / pi,
                        "N_S_seg": len(S_all), "m_seg": m_seg, "t_star": r["t_star"], "s_order_key": s_order_key(r["token"])})
        reasons = defaultdict(int)
        for r in rows:
            if not r["eligible"]:
                reasons[r["exclusion_reason"] or "unknown"] += 1
        den[log] = {"scenes": len(rows), "eligible": sum(r["eligible"] for r in rows), "F": len(F), "S_eligible": len(S_all),
                    "S_sampled": m_seg, "not_eligible_by_reason": dict(reasons)}
    return sel, den
