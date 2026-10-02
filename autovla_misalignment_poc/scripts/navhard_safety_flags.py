#!/usr/bin/env python
"""
Current-frame safety flags for every navhard candidate (NAVSIM v2; run with PYTHONPATH=/root/VLA/navsim_v2).

The official v2 metric-cache computation is reused with its two future-reading steps replaced:
  _interpolate_gt_observation       -> current tracked objects extrapolated at constant velocity (0..4 s, 0.1 s)
  _interpolate_traffic_light_status -> current traffic-light status held constant
Map, route, drivable area and centerline are computed exactly as in the official cache. Each candidate is
then scored with navsim's pdm_score using the ConstantVelocityTrafficAgents policy, and we keep per
candidate: no_at_fault_collisions, drivable_area_compliance, driving_direction_compliance,
time_to_collision_within_bound. No log frame after the current one and no synthetic future is read.

  PYTHONPATH=/root/VLA/navsim_v2 python scripts/navhard_safety_flags.py <decode_run_dir> <out_dir> --workers 4
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

SUBSET = os.environ.get("NAVHARD_SUBSET", "/root/VLA/autovla_misalignment_poc/outputs/navhard_eval/subset")   # other navhard subsets via env
SYN = "/root/VLA/navhard/navhard_two_stage/synthetic_scene_pickles"
LOGS = "/root/VLA/autovla/dataset/nuplan/navsim_logs/test"
FLAGS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "time_to_collision_within_bound"]
_W = {}


def _init(stage1_dicts_path):
    import torch
    sys.path.insert(1, "/root/VLA/autovla")
    from models.action_tokenizer import ActionTokenizer
    from nuplan.common.actor_state.agent import Agent
    from nuplan.common.actor_state.oriented_box import OrientedBox
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
    from nuplan.common.actor_state.static_object import StaticObject
    from nuplan.common.actor_state.tracked_objects import TrackedObjects
    from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.traffic_agents_policies.constant_velocity_traffic_agents import ConstantVelocityTrafficAgents
    torch.set_num_threads(1)
    ps = TrajectorySampling(num_poses=40, interval_length=0.1)
    proc = MetricCacheProcessor(cache_path=None, force_feature_computation=True, proposal_sampling=ps)

    def cv_obs(scenario):
        objs = scenario.get_tracked_objects_at_iteration(0).tracked_objects
        out = []
        for t in np.arange(0, ps.time_horizon + 1e-9, ps.interval_length):
            cur = []
            for o in objs:
                if o.tracked_object_type in AGENT_TYPES:
                    vx, vy = o.velocity.x, o.velocity.y
                    box = OrientedBox(StateSE2(o.center.x + vx * t, o.center.y + vy * t, o.center.heading), o.box.length, o.box.width, o.box.height)
                    cur.append(Agent(tracked_object_type=o.tracked_object_type, oriented_box=box, velocity=StateVector2D(vx, vy), metadata=o.metadata))
                else:
                    cur.append(StaticObject(tracked_object_type=o.tracked_object_type, oriented_box=o.box, metadata=o.metadata))
            out.append(DetectionsTracks(TrackedObjects(cur)))
        return out

    def tl_hold(scenario):
        cur = list(scenario.get_traffic_light_status_at_iteration(0))
        return [cur for _ in np.arange(0, ps.time_horizon + 1e-9, ps.interval_length)]
    proc._interpolate_gt_observation = cv_obs
    proc._interpolate_traffic_light_status = tl_hold
    _W.update(proc=proc, ps=ps, simulator=PDMSimulator(ps), scorer=PDMScorer(ps, PDMScorerConfig(human_penalty_filter=False)),   # the filter reads the human future: off
              policy=ConstantVelocityTrafficAgents(ps), rollout=ActionTokenizer.rollout,
              cb=torch.tensor(np.asarray(pickle.load(open("/root/VLA/autovla/codebook_cache/agent_vocab.pkl", "rb"))["token_all"]["veh"]), dtype=torch.float32),
              stage1=pickle.load(open(stage1_dicts_path, "rb")),
              init2scene=json.load(open(f"{SUBSET}/synthetic_initial_to_scene_token.json")))


def _flags(job):
    token, cands = job
    from navsim.common.dataclasses import Scene, SensorConfig, Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    try:
        W = _W
        if token in W["stage1"]:
            scene = Scene.from_scene_dict_list(W["stage1"][token], None, num_history_frames=4, num_future_frames=8,
                                               sensor_config=SensorConfig.build_no_sensors())
        else:
            scene = Scene.load_from_disk(Path(f"{SYN}/{W['init2scene'][token]}.pkl"), None, SensorConfig.build_no_sensors())
        scenario = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")
        mc = W["proc"].compute_metric_cache(scenario)
        fl, cache = [], {}
        for idx in cands:
            key = tuple(idx[:8])
            if key not in cache:
                poses = W["rollout"](None, W["cb"][list(key)], time_steps=8)[0, 1:].numpy().astype(np.float32)
                df = pdm_score(mc, Trajectory(poses, TrajectorySampling(num_poses=8, interval_length=0.5)), W["ps"],
                               W["simulator"], W["scorer"], W["policy"])
                df = df[0] if isinstance(df, tuple) else df          # v2 returns (pdm_result, simulated_states)
                row = df.iloc[0]
                cache[key] = [float(row[f]) for f in FLAGS]
            fl.append(cache[key])
        return {"token": token, "flags": fl}
    except Exception as exc:
        return {"token": token, "error": repr(exc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import navsim
    assert "navsim_v2" in navsim.__file__, "run with PYTHONPATH=/root/VLA/navsim_v2"
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader
    os.makedirs(args.out, exist_ok=True)
    groups = json.load(open(f"{SUBSET}/groups.json"))
    s1 = json.load(open(f"{SUBSET}/stage1_tokens.json"))
    loader = SceneLoader(data_path=Path(LOGS), original_sensor_path=None,
                         scene_filter=SceneFilter(num_history_frames=4, num_future_frames=8, frame_interval=1, has_route=True,
                                                  include_synthetic_scenes=False, log_names=sorted({g["log"] for g in groups}), tokens=s1),
                         sensor_config=SensorConfig.build_no_sensors())
    s1_path = os.path.join(args.out, "_stage1_dicts.pkl")
    pickle.dump(dict(loader.scene_frames_dicts), open(s1_path, "wb"))
    dst = os.path.join(args.out, "flags.jsonl")
    done = {json.loads(l)["token"] for l in open(dst)} if os.path.exists(dst) else set()
    jobs = [(r["token"], [c["action_idx"] for c in r["candidates"]]) for r in map(json.loads, open(os.path.join(args.run, "records.jsonl")))
            if r["token"] not in done]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"[plan] {len(jobs)} scenes ({len(done)} done), {args.workers} workers", flush=True)
    t0 = time.time(); n = 0
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(s1_path,)) as pool, open(dst, "a") as f:
        for res in pool.imap_unordered(_flags, jobs, chunksize=2):
            f.write(json.dumps(res) + "\n"); f.flush(); n += 1
            if n % 100 == 0:
                print(f"[prog] {n}/{len(jobs)} {(time.time()-t0)/n:.2f}s/scene", flush=True)
    print(f"[done] {n} scenes in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
