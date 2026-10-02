#!/usr/bin/env python
"""
NAVSIM v2 metric caches for the navhard subset, using the official caching functions (run with
PYTHONPATH=/root/VLA/navsim_v2). Same computation as navsim's cache_scenarios
(Scene -> NavSimScenario -> MetricCacheProcessor.compute_and_save_metric_cache, proposal sampling 40 x 0.1 s),
but it is handed exactly the needed scenes: the official driver re-reads all 5,462 synthetic pickles for
every log it processes, which is ~2 pickles/s.

  PYTHONPATH=/root/VLA/navsim_v2 python scripts/navhard_metric_cache.py --workers 4
Writes caches + metadata csv under --cache (the layout MetricCacheLoader and run_pdm_score_from_submission read).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

SUBSET = "/root/VLA/autovla_misalignment_poc/outputs/navhard_eval/subset"
SYN = "/root/VLA/navhard/navhard_two_stage/synthetic_scene_pickles"
LOGS = "/root/VLA/autovla/dataset/nuplan/navsim_logs/test"
_P = {}


def _init(cache):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    _P["proc"] = MetricCacheProcessor(cache_path=cache, force_feature_computation=True,
                                      proposal_sampling=TrajectorySampling(num_poses=40, interval_length=0.1))


def _scenario(scene):
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    return NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")


def _synthetic(path):
    from navsim.common.dataclasses import Scene, SensorConfig
    try:
        scene = Scene.load_from_disk(Path(path), None, SensorConfig.build_no_sensors())
        return _P["proc"].compute_and_save_metric_cache(_scenario(scene))
    except Exception as exc:
        return repr(exc)


def _original(scene_dict):
    from navsim.common.dataclasses import Scene, SensorConfig
    try:
        scene = Scene.from_scene_dict_list(scene_dict, None, num_history_frames=4, num_future_frames=8,
                                           sensor_config=SensorConfig.build_no_sensors())
        return _P["proc"].compute_and_save_metric_cache(_scenario(scene))
    except Exception as exc:
        return repr(exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/root/VLA/navhard/exp/metric_cache_half")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    import navsim
    assert "navsim_v2" in navsim.__file__, f"run with PYTHONPATH=/root/VLA/navsim_v2 (got {navsim.__file__})"
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader
    from nuplan.planning.training.experiments.cache_metadata_entry import CacheMetadataEntry, save_cache_metadata
    groups = json.load(open(f"{SUBSET}/groups.json"))
    s1 = json.load(open(f"{SUBSET}/stage1_tokens.json"))
    init2scene = json.load(open(f"{SUBSET}/synthetic_initial_to_scene_token.json"))
    syn_paths = [f"{SYN}/{init2scene[t]}.pkl" for t in json.load(open(f"{SUBSET}/synthetic_tokens.json"))]
    loader = SceneLoader(data_path=Path(LOGS), original_sensor_path=None,
                         scene_filter=SceneFilter(num_history_frames=4, num_future_frames=8, frame_interval=1, has_route=True,
                                                  include_synthetic_scenes=False, log_names=sorted({g["log"] for g in groups}), tokens=s1),
                         sensor_config=SensorConfig.build_no_sensors())
    dicts = list(loader.scene_frames_dicts.values())
    print(f"[plan] stage-1 scenes {len(dicts)}/{len(s1)}, synthetic {len(syn_paths)}, workers {args.workers}", flush=True)
    t0 = time.time(); entries, errors = [], []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.cache,)) as pool:
        for i, res in enumerate(pool.imap_unordered(_original, dicts, chunksize=2), 1):
            (entries if isinstance(res, CacheMetadataEntry) else errors).append(res)
        print(f"[stage1] {len(entries)} ok, {len(errors)} errors, {(time.time()-t0)/60:.1f} min", flush=True)
        for i, res in enumerate(pool.imap_unordered(_synthetic, syn_paths, chunksize=4), 1):
            (entries if isinstance(res, CacheMetadataEntry) else errors).append(res)
            if i % 250 == 0:
                print(f"[synthetic] {i}/{len(syn_paths)} {(time.time()-t0)/60:.1f} min", flush=True)
    save_cache_metadata(entries, Path(args.cache), 0)
    json.dump([str(e) for e in errors], open(f"{args.cache}/errors.json", "w"))
    print(f"[done] {len(entries)} caches, {len(errors)} errors in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
