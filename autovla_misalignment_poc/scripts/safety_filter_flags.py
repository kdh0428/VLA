#!/usr/bin/env python
"""
Current-frame safety flags for every best-of-N candidate (experiment 22; CPU only; no future information).

For each scene: PDM-Closed is initialised from the current frame (map, route roadblocks, current objects,
traffic lights) to obtain the centerline, route lanes and drivable-area map; the observation used for
scoring is built from the CURRENT tracked objects only, extrapolated at constant velocity over 5 s
(static objects stay in place). All candidates are simulated and scored together with navsim's
PDMSimulator / PDMScorer (default scoring parameters), and per candidate we keep
  pred_no_collision      NO_COLLISION multiplier (1 ok, 0.5 / 0 predicted collision)
  pred_drivable_area     DRIVABLE_AREA multiplier (1 ok, 0 leaves drivable area)
  pred_driving_direction DRIVING_DIRECTION metric (1 ok, 0.5 / 0 oncoming-lane travel)
  pred_ttc               TTC metric
No log frame after the current one is read.

  python scripts/safety_filter_flags.py --out <dir> --workers 3 <run_dir> [<run_dir> ...]
Output: <dir>/flags.jsonl  {token, log, flags: [[no_collision, drivable_area, driving_direction, ttc] x n_candidates]}
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdm_score_candidates as P     # noqa: E402  (worker init, codebook rollout)


def cv_observation(proc, scenario, horizon=5.0, step=0.5):
    """PDMObservation from current tracked objects extrapolated at constant velocity (no future frames)."""
    from nuplan.common.actor_state.agent import Agent
    from nuplan.common.actor_state.oriented_box import OrientedBox
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
    from nuplan.common.actor_state.static_object import StaticObject
    from nuplan.common.actor_state.tracked_objects import TrackedObjects
    from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
    objs = scenario.get_tracked_objects_at_iteration(0).tracked_objects
    tracks = []
    for t in np.arange(0, horizon + 1e-9, 0.1):
        cur = []
        for o in objs:
            if o.tracked_object_type in AGENT_TYPES:
                vx, vy = o.velocity.x, o.velocity.y
                box = OrientedBox(StateSE2(o.center.x + vx * t, o.center.y + vy * t, o.center.heading),
                                  o.box.length, o.box.width, o.box.height)
                cur.append(Agent(tracked_object_type=o.tracked_object_type, oriented_box=box,
                                 velocity=StateVector2D(vx, vy), metadata=o.metadata))
            else:
                cur.append(StaticObject(tracked_object_type=o.tracked_object_type, oriented_box=o.box, metadata=o.metadata))
        tracks.append(DetectionsTracks(TrackedObjects(cur)))
    obs = PDMObservation(proc._future_sampling, proc._proposal_sampling, proc._map_radius, observation_sample_res=1)
    obs.update_detections_tracks(tracks)
    return obs


def _flags(job):
    token, log, cands = job
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    try:
        W = P._W
        scene = W["loader"].get_scene_from_token(token)
        scenario = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version=os.environ.get("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0"))
        proc = W["proc"]
        pin, pinit = proc._get_planner_inputs(scenario)
        proc._pdm_closed.initialize(pinit)
        pdm_traj = proc._pdm_closed.compute_planner_trajectory(pin)
        obs = cv_observation(proc, scenario)
        ego = scenario.initial_ego_state
        fs = W["simulator"].proposal_sampling
        states = [get_trajectory_as_array(pdm_traj, fs, ego.time_point)]
        for idx in cands:
            t = Trajectory(P._poses(idx).astype(np.float32), TrajectorySampling(num_poses=10, interval_length=0.5))
            states.append(get_trajectory_as_array(transform_trajectory(t, ego), fs, ego.time_point))
        sim = W["simulator"].simulate_proposals(np.stack(states), ego)
        sc = W["scorer"]
        sc.score_proposals(sim, obs, proc._pdm_closed._centerline, list(proc._pdm_closed._route_lane_dict.keys()),
                           proc._pdm_closed._drivable_area_map)
        fl = [[float(sc._multi_metrics[MultiMetricIndex.NO_COLLISION, i]), float(sc._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, i]),
               float(sc._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, i]), float(sc._weighted_metrics[WeightedMetricIndex.TTC, i])]
              for i in range(1, len(states))]
        return {"token": token, "log": log, "flags": fl}
    except Exception as exc:
        return {"token": token, "log": log, "error": repr(exc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, "flags.jsonl")
    done = {json.loads(l)["token"] for l in open(dst)} if os.path.exists(dst) else set()
    jobs = []
    for run in args.runs:
        for l in open(os.path.join(run, "records.jsonl")):
            r = json.loads(l)
            if r.get("stub_cot") or r["token"] in done:
                continue
            jobs.append((r["token"], r["log"], [x["action_idx"] for x in r["candidates"]]))
    if args.limit:
        jobs = jobs[:args.limit]
    logs = sorted({j[1] for j in jobs})
    print(f"[plan] {len(jobs)} scenes ({len(done)} done), {len(logs)} logs, {args.workers} workers", flush=True)
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(args.workers, initializer=P._init, initargs=(logs,)) as pool, open(dst, "a") as f:
        for res in pool.imap_unordered(_flags, jobs, chunksize=4):
            f.write(json.dumps(res) + "\n"); f.flush()
            n += 1
            if n % 200 == 0:
                el = time.time() - t0
                print(f"[prog] {n}/{len(jobs)} {el/n:.2f}s/scene eta {(len(jobs)-n)*el/n/60:.1f}min", flush=True)
    print(f"[done] {n} scenes in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
