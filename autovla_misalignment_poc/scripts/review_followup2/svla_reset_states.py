#!/usr/bin/env python
"""A2 (SpatialVLA): re-create the initial states of experiments 34/35 and P1-D on CPU (no policy) and record the
physical state right after reset(). One process per task. Writes <out>/raw_<task>.jsonl with records of kind
  canon   : reset with reconfigure=True as the canonical state of (experiment-group, seed/unit)
  repeat  : canonical reset repeated later in the same process (determinism)
  drift_* : default reset (no options, as experiments 34/35) after another seed, compared with canon
plus exp-34 t=0 frame comparison. See outputs/review_followup2/A_spatialvla_states/run_20261013/METHOD.md.

  source /root/VLA/simpler/env.sh; export CUDA_VISIBLE_DEVICES=""
  nice -n 19 python svla_reset_states.py --task google_robot_move_near --out OUT
  nice -n 19 python svla_reset_states.py --task google_robot_move_near --out OUT --fresh-check 0,7,1006   # new process
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

ROOT = "/root/VLA/autovla_misalignment_poc"
UNITS = f"{ROOT}/outputs/review_followup/P1D_dose_control/run_20261008/units.json"
FRAMES = f"{ROOT}/outputs/cross_domain_temporal_replication/closed_loop/frames"
H = lambda x: hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]   # same as p1d_runner.H


def canon_q(q):
    q = [float(x) for x in q]
    for x in q:
        if x != 0.0:
            return q if x > 0 else [-y for y in q]
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fresh-check", default="", help="comma list of seeds/units: canonical reset in this fresh process only")
    ap.add_argument("--mixed", type=int, default=0, help="N default resets with BOTH task envs in one process, random "
                    "task/seed order (as the exp 34/35 workers); compared with canonical states in raw_<task>.jsonl")
    ap.add_argument("--stepped", type=int, default=0, help="N trials: replay a recorded exp-35 episode (default reset + "
                    "its executed actions) in one of the two task envs, then default-reset a target seed 0-39 and compare "
                    "with the canonical state and the exp-34 t=0 frame (history with steps, as in the exp 34/35 workers)")
    a = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ["LP_NUM_THREADS"] = "2"
    import cv2 as cv
    cv.setNumThreads(1)
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    if a.stepped:
        return stepped(a, cv, simpler_env, get_image_from_maniskill2_obs_dict)
    if a.mixed:
        return mixed(a, cv, simpler_env, get_image_from_maniskill2_obs_dict)
    task = a.task
    units = json.load(open(UNITS))[task]
    p1d_units = sorted(int(k) for k in units)
    env = simpler_env.make(task)
    u = env.unwrapped

    def reset(seed, reconfigure, eid=None):
        opts = None
        if reconfigure or eid is not None:
            opts = {"reconfigure": bool(reconfigure)}
            if eid is not None:
                opts["obj_init_options"] = {"episode_id": int(eid)}
        obs, info = env.reset(seed=seed, options=opts) if opts is not None else env.reset(seed=seed)
        img = cv.resize(get_image_from_maniskill2_obs_dict(env, obs), (224, 224), interpolation=cv.INTER_AREA)
        st = np.asarray(u.get_state(), dtype=np.float64)
        actors = []
        for act in sorted(u._scene.get_all_actors(), key=lambda x: x.name):
            if not hasattr(act, "velocity"):
                continue                                    # static actors (arena) never move
            actors.append({"name": act.name, "p": [float(x) for x in act.pose.p], "q": canon_q(act.pose.q),
                           "v": [float(x) for x in act.velocity], "w": [float(x) for x in act.angular_velocity]})
        rob = u.agent.robot
        rec = {
            "task": task, "seed": int(seed), "reconfigure": bool(reconfigure), "eid_opt": eid,
            "instr": env.get_language_instruction(),
            "episode_id": int(info["episode_id"]) if "episode_id" in info else None,
            "source": info.get("episode_source_obj_name"), "target": info.get("episode_target_obj_name"),
            "orientation": info.get("orientation"),
            "model_ids": list(info.get("episode_model_ids", [info.get("model_id")])),
            "model_scales": [float(x) for x in info.get("episode_model_scales", [info.get("model_scale")])],
            "urdf_version": u.urdf_version, "rgb_overlay_path": os.path.basename(str(u.rgb_overlay_path)),
            "scene_name": info.get("scene_name"),
            "actors": actors,
            "robot": {"p": [float(x) for x in rob.pose.p], "q": canon_q(rob.pose.q),
                      "qpos": [float(x) for x in rob.get_qpos()], "qvel": [float(x) for x in rob.get_qvel()]},
            "tcp_pose": [float(x) for x in u.tcp.pose.p] + [float(x) for x in u.tcp.pose.q],
            "state_hash": H(st), "img_hash": H(img), "state": st.tolist(),
        }
        return rec, img

    os.makedirs(a.out, exist_ok=True)
    if a.fresh_check:
        out = open(os.path.join(a.out, f"fresh_{task}.jsonl"), "w")
        for s in map(int, a.fresh_check.split(",")):
            eid = units.get(str(s), {}).get("episode_id")
            rec, _ = reset(s, True, eid)
            rec["kind"] = "fresh"
            out.write(json.dumps(rec) + "\n"); out.flush()
        return

    out = open(os.path.join(a.out, f"raw_{task}.jsonl"), "w")
    W = lambda r: (out.write(json.dumps(r) + "\n"), out.flush())
    canon = {}
    # 1. experiments 34/35: seeds 0-119, canonical (reconfigure=True)
    for s in range(120):
        rec, img = reset(s, True)
        rec["kind"], rec["group"] = "canon", "exp34_35"
        if s < 40:                                          # exp-34 saved t=0 frame
            fp = os.path.join(FRAMES, f"{task}_s{s}", "000.png")
            if os.path.exists(fp):
                fr = cv.imread(fp)[:, :, ::-1]               # runner wrote img[:, :, ::-1] (RGB->BGR)
                d = np.abs(fr.astype(int) - img.astype(int))
                rec["exp34_frame"] = {"exact": bool((d == 0).all()), "max_abs": int(d.max()),
                                      "n_px_diff": int((d.max(axis=2) > 0).sum())}
            else:
                rec["exp34_frame"] = None
        canon[("exp", s)] = rec["state"]; W(rec)
    # 2. P1-D units
    for s in p1d_units:
        eid = units[str(s)].get("episode_id")
        rec, _ = reset(s, True, eid)
        rec["kind"], rec["group"] = "canon", "p1d"
        rec["unit_overlay"] = units[str(s)].get("overlay")
        canon[("p1d", s)] = rec["state"]; W(rec)
    # 3. determinism: canonical repeats after everything else ran in this process
    rng = np.random.default_rng(20261013)
    rep = [("exp", int(s)) for s in rng.choice(120, 8, replace=False)] + \
          [("p1d", int(s)) for s in rng.choice(p1d_units, 6, replace=False)]
    for g, s in rep:
        eid = units[str(s)].get("episode_id") if g == "p1d" else None
        rec, _ = reset(s, True, eid)
        rec["kind"], rec["group"] = "repeat", g
        rec["max_abs_vs_canon"] = float(np.max(np.abs(np.asarray(rec["state"]) - np.asarray(canon[(g, s)]))))
        W(rec)
    # 4. drift of default resets (exp 34/35 style): random predecessors, then a sequential chain 0..119
    for s in [int(x) for x in rng.choice(120, 12, replace=False)]:
        for pred in [int(x) for x in rng.choice(120, 5, replace=False)]:
            reset(pred, False)
            rec, _ = reset(s, False)
            rec["kind"], rec["group"], rec["pred"] = "drift_pred", "exp34_35", pred
            if len(rec["state"]) == len(canon[("exp", s)]):
                rec["max_abs_vs_canon"] = float(np.max(np.abs(np.asarray(rec["state"]) - np.asarray(canon[("exp", s)]))))
            W(rec)
    for s in range(120):
        rec, _ = reset(s, False)
        rec["kind"], rec["group"] = "drift_chain", "exp34_35"
        if len(rec["state"]) == len(canon[("exp", s)]):
            rec["max_abs_vs_canon"] = float(np.max(np.abs(np.asarray(rec["state"]) - np.asarray(canon[("exp", s)]))))
        W(rec)
    out.close()
    env.close()


def mixed(a, cv, simpler_env, get_img):
    """Exp 34/35 workers held one env per task in the same process and served a shared queue (default resets)."""
    tasks = ("google_robot_pick_coke_can", "google_robot_move_near")
    canon = {}
    for t in tasks:
        for l in open(os.path.join(a.out, f"raw_{t}.jsonl")):
            r = json.loads(l)
            if r["kind"] == "canon" and r["group"] == "exp34_35":
                canon[(t, r["seed"])] = r
    envs = {t: simpler_env.make(t) for t in tasks}
    rng = np.random.default_rng(20261014)
    out = open(os.path.join(a.out, "mixed_drift.jsonl"), "w")
    for i in range(a.mixed):
        t = tasks[int(rng.integers(2))]; s = int(rng.integers(40)) if rng.random() < 0.5 else int(rng.integers(120))
        env = envs[t]; u = env.unwrapped
        obs, info = env.reset(seed=s)
        img = cv.resize(get_img(env, obs), (224, 224), interpolation=cv.INTER_AREA)
        st = np.asarray(u.get_state(), dtype=np.float64)
        c = canon[(t, s)]
        rec = {"i": i, "task": t, "seed": s, "state_hash": H(st), "img_hash": H(img),
               "same_hash": H(st) == c["state_hash"], "same_img": H(img) == c["img_hash"],
               "max_abs_vs_canon": float(np.max(np.abs(st - np.asarray(c["state"])))) if len(st) == len(c["state"]) else None,
               "state": st.tolist()}
        if s < 40:
            fp = os.path.join(FRAMES, f"{t}_s{s}", "000.png")
            fr = cv.imread(fp)[:, :, ::-1]
            rec["frame_exact"] = bool((fr == img).all())
            rec["frame_max_abs"] = int(np.abs(fr.astype(int) - img.astype(int)).max())
        out.write(json.dumps(rec) + "\n"); out.flush()


def stepped(a, cv, simpler_env, get_img):
    tasks = ("google_robot_pick_coke_can", "google_robot_move_near")
    canon = {}
    for t in tasks:
        for l in open(os.path.join(a.out, f"raw_{t}.jsonl")):
            r = json.loads(l)
            if r["kind"] == "canon" and r["group"] == "exp34_35":
                canon[(t, r["seed"])] = r
    eps = []
    for l in open(f"{ROOT}/outputs/spatialvla_feedback_protection_ablation/rollouts/episodes.jsonl"):
        r = json.loads(l)
        eps.append((r["task"], r["seed"], r["cond"], [x["action"] for x in r["rec"]]))
    envs = {t: simpler_env.make(t) for t in tasks}
    rng = np.random.default_rng(20261015)
    out = open(os.path.join(a.out, "stepped_drift.jsonl"), "w")
    for i in range(a.stepped):
        tp, sp, cp, acts = eps[int(rng.integers(len(eps)))]
        envs[tp].reset(seed=sp)
        for act in acts:
            envs[tp].step(np.asarray(act, dtype=np.float64))
        t = tasks[int(rng.integers(2))] if rng.random() < 0.3 else tp     # often the same env, as in a worker
        s = int(rng.integers(40))
        env = envs[t]; u = env.unwrapped
        obs, _ = env.reset(seed=s)
        img = cv.resize(get_img(env, obs), (224, 224), interpolation=cv.INTER_AREA)
        st = np.asarray(u.get_state(), dtype=np.float64)
        c = canon[(t, s)]
        fr = cv.imread(os.path.join(FRAMES, f"{t}_s{s}", "000.png"))[:, :, ::-1]
        rec = {"i": i, "pred": [tp, sp, cp, len(acts)], "task": t, "seed": s, "same_hash": H(st) == c["state_hash"],
               "same_img": H(img) == c["img_hash"],
               "max_abs_vs_canon": float(np.max(np.abs(st - np.asarray(c["state"])))) if len(st) == len(c["state"]) else None,
               "frame_exact": bool((fr == img).all()), "frame_max_abs": int(np.abs(fr.astype(int) - img.astype(int)).max()),
               "state": st.tolist()}
        out.write(json.dumps(rec) + "\n"); out.flush()


if __name__ == "__main__":
    main()
