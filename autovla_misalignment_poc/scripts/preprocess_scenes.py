#!/usr/bin/env python
"""
Build AutoVLA scene JSONs for the PoC subset, plus a parallel GT annotation file.

Two differences from `tools/preprocessing/nocot_sample_generation.py`, both deliberate:

1. It skips the base64 image encoding. That step exists only to feed the 72B CoT
   annotation model; the no-CoT path writes just camera PATHS. Decoding + re-encoding
   4 frames x 8 cameras per scene dominates the runtime and none of it is used here.

2. It additionally dumps the per-frame navsim `Annotations` (boxes, names, velocity_3d)
   to `annotations/<token>.json`. AutoVLA's own scene JSON carries no object annotations
   (nuplan_dataset.py:229-241), so the perception (P) labels have to come from the
   dataset ground truth -- which is exactly where the spec says they must come from.

The emitted scene JSON has the same keys the AutoVLA feature builder reads
(navsim/agents/autovla_agent.py:240-285), so downstream inference is unchanged.
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

CAM_LIST = ["front", "front_left", "front_right", "back", "back_left", "back_right",
            "left", "right"]

#: Root the camera paths are made relative to (must match the runner's sensor_data_path).
SENSOR_ROOT = "dataset/nuplan/sensor_blobs/test"


def to_list(x):
    if x is None:
        return []
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-filter",
                    default=os.path.join(POC_DIR, "configs/scene_filter_navtest_subset.yaml"))
    ap.add_argument("--out", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--anno-out", default=os.path.join(POC_DIR, "outputs/annotations"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    os.chdir(AUTOVLA_DIR)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.anno_out, exist_ok=True)

    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader
    from navsim.agents.vla_agent import VlaAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from dataset_utils.preprocessing.nuplan_dataset import get_action_instruction

    interval = 0.5
    agent = VlaAgent(trajectory_sampling=TrajectorySampling(time_horizon=5,
                                                            interval_length=interval))
    scene_filter: SceneFilter = instantiate(OmegaConf.load(args.scene_filter))
    loader = SceneLoader(
        data_path=Path("dataset/nuplan/navsim_logs/test"),
        sensor_blobs_path=Path("dataset/nuplan/sensor_blobs/test"),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config())

    tokens = loader.tokens[args.start:]
    if args.limit:
        tokens = tokens[: args.limit]
    print(f"scenes to process: {len(tokens)}", flush=True)

    t0 = time.time()
    n_ok = n_fail = 0
    for i, token in enumerate(tokens, 1):
        try:
            scene = loader.get_scene_from_token(token)
            agent_input = scene.get_agent_input()
            feats = {}
            for b in agent.get_feature_builders():
                feats.update(b.compute_features(agent_input))
            gt = agent.get_target_builders()[0].compute_targets(scene)

            vel = to_list(feats["vehicle_velocity"])
            acc = to_list(feats["vehicle_acceleration"])
            his = to_list(feats["history_trajectory"])
            gtl = to_list(gt)

            rec = {
                "token": token,
                "dataset_name": "nuplan",
                "cot_output": [],
                "velocity": vel,
                "acceleration": acc,
                "instruction": feats["driving_command"].lower(),
                "gt_trajectory": gtl,
                "his_trajectory": his,
            }
            # Camera paths, stored RELATIVE to the sensor-blobs root.
            #
            # navsim's Camera.camera_path is already sensor_blobs_path / log / cam / file,
            # i.e. it embeds the root we passed to SceneLoader. AutoVLA's get_prompt then
            # prepends `sensor_data_path` again (models/autovla.py:544-546), producing
            # ".../sensor_blobs/test/dataset/nuplan/sensor_blobs/test/...". Strip the root
            # here so the JSON stays portable and the official prefixing works unchanged.
            images = feats["images"]
            for side in CAM_LIST:
                cams = images.get(f"{side}_camera") or []
                paths = []
                for c in cams:
                    cp = getattr(c, "camera_path", None)
                    if cp:
                        cp = os.path.relpath(str(cp), SENSOR_ROOT) \
                            if str(cp).startswith(SENSOR_ROOT) else str(cp)
                    paths.append(cp)
                rec[f"{side}_camera_paths"] = paths

            # AutoVLA's own coarse action label for the GT future (used later for A)
            gtp = np.asarray(gtl, dtype=np.float64)[:, :2]
            v = np.diff(gtp, axis=0) / interval
            v = np.concatenate([v, v[-1:]], axis=0) if len(v) else np.zeros((1, 2))
            rec["gt_action_instruction"] = get_action_instruction(gtp, v)

            with open(os.path.join(args.out, f"{token}.json"), "w") as f:
                json.dump(rec, f)

            # ---- ground-truth annotations for the P labels ----
            fi = scene.scene_metadata.num_history_frames - 1     # the "current" frame
            frame = scene.frames[fi]
            ann = frame.annotations
            with open(os.path.join(args.anno_out, f"{token}.json"), "w") as f:
                json.dump({
                    "token": token,
                    "log_name": scene.scene_metadata.log_name,
                    "map_name": scene.scene_metadata.map_name,
                    "boxes": np.asarray(ann.boxes, dtype=np.float32).tolist(),
                    "names": list(ann.names),
                    "velocity_3d": np.asarray(ann.velocity_3d, dtype=np.float32).tolist(),
                    "track_tokens": list(ann.track_tokens),
                    "ego_velocity": vel,
                    "ego_acceleration": acc,
                    "driving_command": rec["instruction"],
                }, f)
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            if n_fail <= 5:
                print(f"[fail] {token}: {type(exc).__name__}: {exc}", flush=True)
        if i % 100 == 0:
            el = time.time() - t0
            print(f"[prog] {i}/{len(tokens)} ok={n_ok} fail={n_fail} "
                  f"{el/i:.2f}s/scene eta {(len(tokens)-i)*el/i/60:.1f}min", flush=True)

    print(f"\ndone: ok={n_ok} fail={n_fail} in {(time.time()-t0)/60:.1f}min -> {args.out}")


if __name__ == "__main__":
    main()
