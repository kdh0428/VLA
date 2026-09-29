#!/usr/bin/env python
"""
NAVSIM PDM Score of best-of-N selections (experiment 21; CPU only).

For each scene of expanded_best_of_n / heldout_best_of_n runs, a NAVSIM metric cache is built in memory
exactly as navsim's MetricCacheProcessor does (PDM-Closed reference, ego state, interpolated future
observations, centerline, route lanes, drivable-area map) and the plans chosen by each selection rule
are scored with navsim's pdm_score using the default scoring parameters
(config/pdm_scoring/default_scoring_parameters.yaml: 4 s / 0.1 s proposal sampling, progress 5, ttc 5,
comfort 2, driving direction 0). The PDM Score is the non-reactive simulation metric AutoVLA reports.

Trajectories (x, y, heading at 0.5 s x 10) are rebuilt from the stored action indices with the codebook
rollout of models/action_tokenizer.py, so heading is exact.

  python scripts/pdm_score_candidates.py --out <dir> --workers 4 <run_dir> [<run_dir> ...]
Output: <dir>/pdm_scores.jsonl  {token, log, rule: {picked, score, no_at_fault_collisions, ...}}
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # BLAS thread spinning made scoring ~30x slower per CPU second

import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

AUTOVLA_DIR = "/root/VLA/autovla"
POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (AUTOVLA_DIR, os.path.join(AUTOVLA_DIR, "navsim"), os.path.dirname(os.path.abspath(__file__))):
    sys.path.insert(0, p)

RULES = ["normal", "max_loglik", "min_entropy", "ranksum", "oracle"]
_W = {}


def _init(log_names):
    os.chdir(AUTOVLA_DIR)
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from models.action_tokenizer import ActionTokenizer
    torch.set_num_threads(1)
    cfg = OmegaConf.load(os.path.join(AUTOVLA_DIR, "navsim/navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"))
    _W["simulator"] = instantiate(cfg.simulator)
    _W["scorer"] = instantiate(cfg.scorer)
    _W["proc"] = MetricCacheProcessor(cache_path=None, force_feature_computation=True)
    _W["loader"] = SceneLoader(data_path=Path("dataset/nuplan/navsim_logs/test"), sensor_blobs_path=None,
                               scene_filter=SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1,
                                                        has_route=False, log_names=log_names),
                               sensor_config=SensorConfig.build_no_sensors())
    _W["cb"] = torch.tensor(np.asarray(pickle.load(open("codebook_cache/agent_vocab.pkl", "rb"))["token_all"]["veh"]), dtype=torch.float32)
    _W["rollout"] = ActionTokenizer.rollout


def _poses(idx):
    t = _W["cb"][list(idx)]
    return _W["rollout"](None, t, time_steps=len(idx))[0, 1:].numpy()     # (10, 3) x, y, heading


def _score(job):
    token, log, picks = job
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.metric_caching.metric_cache import MetricCache
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    try:
        scene = _W["loader"].get_scene_from_token(token)
        scenario = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version=os.environ.get("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0"))
        proc = _W["proc"]
        pin, pinit = proc._get_planner_inputs(scenario)
        proc._pdm_closed.initialize(pinit)
        traj = proc._pdm_closed.compute_planner_trajectory(pin)
        mc = MetricCache(Path("/dev/null"), traj, scenario.initial_ego_state, proc._interpolate_gt_observation(scenario),
                         proc._pdm_closed._centerline, list(proc._pdm_closed._route_lane_dict.keys()), proc._pdm_closed._drivable_area_map)
        out, cache = {"token": token, "log": log}, {}
        for rule, (ci, idx) in picks.items():
            key = tuple(idx)
            if key not in cache:
                t = Trajectory(_poses(idx).astype(np.float32), TrajectorySampling(num_poses=10, interval_length=0.5))
                res = pdm_score(metric_cache=mc, model_trajectory=t, future_sampling=_W["simulator"].proposal_sampling,
                                simulator=_W["simulator"], scorer=_W["scorer"])
                cache[key] = {k: float(v) for k, v in res.__dict__.items()}
            out[rule] = {"picked": int(ci), **cache[key]}
        return out
    except Exception as exc:
        return {"token": token, "log": log, "error": repr(exc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--picks-file", default=None, help="JSON {token: {rule_name: candidate_index}} to score instead of the rules")
    ap.add_argument("--ncurve-picks", default=None,
                    help="prefix_picks.json of candidate_count_curve: score the rank-sum pick at each N instead of the rules")
    args = ap.parse_args()
    ap_picks = None
    ncurve = json.load(open(args.ncurve_picks)) if args.ncurve_picks else None
    if args.picks_file:
        ap_picks = json.load(open(args.picks_file))
    from analyze_best_of_n import pick
    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, "pdm_scores.jsonl")
    done = {json.loads(l)["token"] for l in open(dst)} if os.path.exists(dst) else set()
    jobs = []
    for run in args.runs:
        for l in open(os.path.join(run, "records.jsonl")):
            r = json.loads(l)
            if r.get("stub_cot") or r["token"] in done:
                continue
            r["prev_traj"] = None
            picks = {}
            if ap_picks is not None:
                if r["token"] not in ap_picks:
                    continue
                for rule, i in ap_picks[r["token"]].items():
                    picks[rule] = (i, r["candidates"][i]["action_idx"])
            elif ncurve is not None:
                if r["token"] not in ncurve:
                    continue
                for n, i in ncurve[r["token"]].items():
                    picks[f"ranksum_N{n}"] = (i, r["candidates"][i]["action_idx"])
            else:
                for rule in RULES:
                    i = pick(r, rule)
                    picks[rule] = (i, r["candidates"][i]["action_idx"])
            jobs.append((r["token"], r["log"], picks))
    if args.limit:
        jobs = jobs[:args.limit]
    logs = sorted({j[1] for j in jobs})
    print(f"[plan] {len(jobs)} scenes ({len(done)} already scored), {len(logs)} logs, {args.workers} workers", flush=True)
    t0 = time.time()
    n = 0
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(logs,)) as pool, open(dst, "a") as f:
        for res in pool.imap_unordered(_score, jobs, chunksize=4):
            f.write(json.dumps(res) + "\n"); f.flush()
            n += 1
            if n % 100 == 0:
                el = time.time() - t0
                print(f"[prog] {n}/{len(jobs)} {el/n:.2f}s/scene eta {(len(jobs)-n)*el/n/60:.1f}min", flush=True)
    print(f"[done] {n} scenes in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
