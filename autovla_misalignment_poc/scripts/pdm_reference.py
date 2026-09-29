#!/usr/bin/env python
"""
PDM-Closed reference trajectories for the stabilisation experiments (CPU only).

PDM-Closed (the rule-based planner NAVSIM uses for metric caching) is run on each scene from the
log at the current frame only: map, route roadblocks, current tracked objects and traffic lights,
initial ego state. No future frames, no sensors. It uses exactly the planner configuration of
navsim's MetricCacheProcessor. The 5 s trajectory (rear axle) is expressed in the current ego frame
at 0.5 s steps, i.e. the same format as scene["gt_trajectory"], so it is tokenised with the GT routine.

Output: outputs/frame_index/pdm_refs.json  {token: {"poses": [[x, y, heading] x 10], "ade_vs_gt": float}}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

AUTOVLA_DIR = "/root/VLA/autovla"
POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AUTOVLA_DIR)
sys.path.insert(0, os.path.join(AUTOVLA_DIR, "navsim"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ed-records", default=os.path.join(POC_DIR, "outputs/equal_distance_perturbation/records.jsonl"))
    ap.add_argument("--units", default=os.path.join(POC_DIR, "outputs/frame_index/natural_units.json"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--out", default=os.path.join(POC_DIR, "outputs/frame_index/pdm_refs.json"))
    args = ap.parse_args()
    os.chdir(AUTOVLA_DIR)

    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor

    toks = {}
    for l in open(args.ed_records):
        r = json.loads(l)
        toks[r["token"]] = r["log"]
    if os.path.exists(args.units):
        for u in json.load(open(args.units)):
            toks[u["token"]] = u["log"]
    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    todo = sorted(t for t in toks if t not in out)
    print(f"[plan] {len(toks)} scenes, {len(todo)} to compute", flush=True)
    if not todo:
        return
    loader = SceneLoader(data_path=Path("dataset/nuplan/navsim_logs/test"), sensor_blobs_path=None,
                         scene_filter=SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1,
                                                  has_route=False, log_names=sorted({toks[t] for t in todo}), tokens=todo),
                         sensor_config=SensorConfig.build_no_sensors())
    proc = MetricCacheProcessor(cache_path=None, force_feature_computation=True)
    maps_root, maps_ver = os.environ["NUPLAN_MAPS_ROOT"], os.environ.get("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    t0, n_ok, n_fail = time.time(), 0, 0
    for i, tok in enumerate(todo, 1):
        try:
            scene = loader.get_scene_from_token(tok)
            scenario = NavSimScenario(scene, map_root=maps_root, map_version=maps_ver)
            pin, pinit = proc._get_planner_inputs(scenario)
            proc._pdm_closed.initialize(pinit)
            traj = proc._pdm_closed.compute_planner_trajectory(pin)
            states = traj.get_sampled_trajectory()                  # EgoStates at 0.1 s, index 0 = now
            o = states[0].rear_axle
            c, s = np.cos(o.heading), np.sin(o.heading)
            poses = []
            for k in range(5, 51, 5):                               # 0.5 .. 5.0 s
                p = states[min(k, len(states) - 1)].rear_axle
                dx, dy = p.x - o.x, p.y - o.y
                poses.append([c * dx + s * dy, -s * dx + c * dy, float(np.arctan2(np.sin(p.heading - o.heading), np.cos(p.heading - o.heading)))])
            gt = np.asarray(json.load(open(os.path.join(args.scenes, f"{tok}.json")))["gt_trajectory"])[:, :2] \
                if os.path.exists(os.path.join(args.scenes, f"{tok}.json")) else None
            ade = float(np.linalg.norm(np.asarray(poses)[:, :2] - gt, axis=1).mean()) if gt is not None else None
            out[tok] = {"poses": poses, "ade_vs_gt": ade}
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            if n_fail <= 5:
                import traceback; traceback.print_exc()
        if i % 50 == 0:
            json.dump(out, open(args.out, "w"))
            el = time.time() - t0
            print(f"[prog] {i}/{len(todo)} ok={n_ok} fail={n_fail} {el/i:.2f}s/scene", flush=True)
    json.dump(out, open(args.out, "w"))
    ades = [v["ade_vs_gt"] for v in out.values() if v["ade_vs_gt"] is not None]
    print(f"[done] ok={n_ok} fail={n_fail}; PDM ADE vs GT: mean {np.mean(ades):.2f} median {np.median(ades):.2f} m", flush=True)


if __name__ == "__main__":
    main()
