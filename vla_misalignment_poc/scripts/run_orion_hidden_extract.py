#!/usr/bin/env python
"""
ORION CoT inference + layer-wise planning-token extraction for one GPU shard.

Run one of these per GPU (see README):
    CUDA_VISIBLE_DEVICES=0 python scripts/run_orion_hidden_extract.py \
        --split splits/poc_gpu0.json --output outputs/shard_gpu0

TEMPORAL FIDELITY
-----------------
ORION carries a temporal memory bank inside `pts_bbox_head` / `map_head`
(`orion_head.pre_update_memory`), refreshed per frame and invalidated when the scene
token changes or more than 2 s elapse. Feeding it only the sub-sampled frames would give
the model a 0.6 s-spaced history instead of the 0.1 s-spaced history it was evaluated
with, which is a confound on the trajectory it predicts.

So this runner walks EVERY frame of each clip in order, and for frames that are not in
the subset it runs the vision/detection/map path only (which is what updates the memory)
while skipping the expensive LLM. `history_query`, the only memory input that could have
come from the language side, is produced by `memory_decoder_cq` from the detection memory
(`orion_head.py:783`), not by the LLM — so skipping the LLM leaves the memory chain exact.

OUTPUTS (per shard directory)
    records.jsonl        one json record per selected frame (no tensors)
    hidden/<id>.npz      per-frame layer-wise planning tokens, fp16
    run_meta.json        seed, versions, config, GPU, timings, checksums
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
import torch

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORION_DIR = "/root/VLA/orion"
sys.path.insert(0, POC_DIR)
sys.path.insert(0, ORION_DIR)

import analysis.numpy_compat  # noqa: E402,F401  (restores np.bool etc. before mmcv loads)
from analysis.chatb2d_parser import (build_perception_labels, build_reasoning_labels,  # noqa: E402
                                     classify_question, rounds_from_annotation)
from analysis.orion_hooks import install_planning_capture  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="shard json from build_subset.py")
    ap.add_argument("--output", required=True, help="output directory for this shard")
    ap.add_argument("--config", default=os.path.join(POC_DIR, "configs", "orion_poc_cot_fp16.py"))
    ap.add_argument("--checkpoint", default=os.path.join(ORION_DIR, "ckpts", "Orion.pth"))
    ap.add_argument("--layers", default="all",
                    help="'all' or comma-separated layer indices (0 = embedding output)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N selected frames (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-warmup", action="store_true",
                    help="DEBUG ONLY: skip the memory warm-up frames (breaks temporal fidelity)")
    ap.add_argument("--verify-identical", action="store_true",
                    help="check the capture hook leaves the trajectory bit-identical")
    return ap.parse_args()


def build_everything(args):
    """Build dataset + model exactly the way the official test.py does."""
    from mmcv.utils import Config, load_checkpoint, set_random_seed
    from mmcv.datasets import build_dataset
    from mmcv.models import build_model

    # The ORION configs reference 'ckpts/pretrain_qformer/' and 'data/others/...' relative
    # to the repo root (that is how orion_dist_eval.sh runs them). Match that so paths in
    # the inherited config resolve identically; our own data roots are absolute.
    os.chdir(ORION_DIR)

    cfg = Config.fromfile(args.config)
    set_random_seed(args.seed, deterministic=True)

    # Mirrors test.py:141-152 -- import any custom modules the config asks for.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
    cfg.model.train_cfg = None

    dataset = build_dataset(cfg.data.test)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

    if cfg.get("fp16", None) is not None:
        # test.py:256-261 -- keep the det/map heads in fp32, everything else fp16.
        for m in model.modules():
            if hasattr(m, "fp16_enabled"):
                m.fp16_enabled = True
        for name, v in dict(map_head=False, pts_bbox_head=False).items():
            if name in model._modules:
                model._modules[name].fp16_enabled = v

    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.CLASSES = getattr(dataset, "CLASSES", None)
    model = model.cuda().eval()

    # test.py wraps the model in torch's DataParallel even for a single GPU, and relies on
    # its scatter to move the collated batch onto the device (this vendored mmcv has no
    # `mmcv.parallel.scatter`). We do the same so the data reaches the model in exactly
    # the shape the official pipeline produces. device_ids=[0] is the only visible GPU
    # because the caller sets CUDA_VISIBLE_DEVICES.
    from torch.nn import DataParallel
    wrapped = DataParallel(model, device_ids=[0])
    return cfg, dataset, model, wrapped


def index_dataset(dataset):
    """(folder, frame_idx) -> dataset index, plus per-clip ordered index lists."""
    key2idx: dict[tuple[str, int], int] = {}
    per_clip: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, info in enumerate(dataset.data_infos):
        folder = info["folder"]
        clip = folder.split("/")[-1]
        fi = int(info["frame_idx"])
        key2idx[(clip, fi)] = i
        per_clip[clip].append((fi, i))
    for clip in per_clip:
        per_clip[clip].sort()
    return key2idx, per_clip


def collate_one(dataset, idx):
    """Fetch one sample and collate it exactly as build_dataloader would (batch of 1)."""
    from mmcv.parallel import collate
    return collate([dataset[idx]], samples_per_gpu=1)


class LMToggle:
    """
    Temporarily disable ORION's language head.

    `Orion.with_lm_head` is a property reading `self.lm_head`; setting the module slot to
    None makes it False, so `simple_test_pts` runs the vision/detection/map path (which
    updates the temporal memory) and returns before the LLM. Fully reversible; the repo is
    untouched.
    """

    def __init__(self, model):
        self.model = model
        self.saved = None

    def __enter__(self):
        self.saved = self.model._modules.get("lm_head", None)
        self.model._modules["lm_head"] = None
        return self

    def __exit__(self, *exc):
        self.model._modules["lm_head"] = self.saved
        return False


def extract_answers(text_out) -> dict[str, str]:
    """ORION `text_out` -> {round_name: answer}, using the same router as the GT parser."""
    rounds: dict[str, str] = {}
    if not text_out:
        return rounds
    for item in text_out:
        q = item.get("Q", "")
        a = item.get("A", "")
        if isinstance(a, list):
            a = a[0] if a else ""
        if isinstance(q, (list, tuple)):
            q = q[0] if q else ""
        name = classify_question(str(q))
        rounds.setdefault(name, str(a).strip())
    return rounds


def first_tensor(v):
    """
    Unwrap the nesting `collate` produces.

    Depending on how the pipeline wrapped a key (MultiScaleFlipAug3D adds a TTA list, and
    DataContainer adds another level), a value arrives as a Tensor, a list of Tensors, or
    a list of lists. Descend until we reach the first Tensor.
    """
    for _ in range(4):
        if v is None or isinstance(v, torch.Tensor):
            return v
        if isinstance(v, (list, tuple)):
            if not v:
                return None
            v = v[0]
        else:
            return v
    return v


class _LimitReached(Exception):
    """Internal signal that --limit was hit; not an error."""


def command_index(v) -> int | None:
    """`ego_fut_cmd` is a 6-way one-hot (turn L/R, straight, lane-follow, change L/R)."""
    t = first_tensor(v)
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return None
    a = t.detach().float().cpu().numpy().reshape(-1)
    if a.size % 6 == 0:
        a = a[:6]
    return int(np.argmax(a))


def decumsum(traj: list) -> list:
    """Cumulative positions -> per-step deltas (inverse of np.cumsum along the step axis)."""
    if not traj:
        return []
    a = np.asarray(traj, dtype=np.float32)
    return np.diff(a, axis=0, prepend=np.zeros((1, a.shape[1]), dtype=a.dtype)).tolist()


def to_numpy_traj(t) -> list:
    t = first_tensor(t)
    if t is None:
        return []
    if isinstance(t, torch.Tensor):
        t = t.detach().float().cpu().numpy()
    a = np.asarray(t, dtype=np.float32)
    if a.size == 0 or a.size % 2 != 0:
        return []
    return a.reshape(-1, 2).tolist()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    hidden_dir = os.path.join(args.output, "hidden")
    os.makedirs(hidden_dir, exist_ok=True)

    with open(args.split) as f:
        shard = json.load(f)
    wanted = {(r["clip_id"], r["frame_idx"]): r for r in shard}
    print(f"[shard] {len(shard)} selected frames over "
          f"{len({r['clip_id'] for r in shard})} clips", flush=True)

    layers = None if args.layers == "all" else [int(x) for x in args.layers.split(",")]

    t0 = time.time()
    cfg, dataset, model, wrapped = build_everything(args)
    print(f"[build] model + dataset ready in {time.time() - t0:.1f}s "
          f"({len(dataset)} dataset frames)", flush=True)

    capture = install_planning_capture(model, layers=layers)
    key2idx, per_clip = index_dataset(dataset)

    # Only clips that exist in BOTH the shard and the built infos can be processed.
    shard_clips = sorted({r["clip_id"] for r in shard})
    usable = [c for c in shard_clips if c in per_clip]
    missing = [c for c in shard_clips if c not in per_clip]
    if missing:
        print(f"[warn] {len(missing)} shard clips absent from b2d_infos_val.pkl: "
              f"{missing[:4]}{' ...' if len(missing) > 4 else ''}", flush=True)

    records_path = os.path.join(args.output, "records.jsonl")
    fout = open(records_path, "w")
    n_done = n_fail = n_warm = 0
    fail_reasons: dict[str, int] = defaultdict(int)
    layer_shape_seen = None
    t_start = time.time()

    try:
      for clip in usable:
        frames = per_clip[clip]
        sel_in_clip = {fi for (c, fi) in wanted if c == clip}
        if not sel_in_clip:
            continue
        last_needed = max(sel_in_clip)

        # New scene -> reset the temporal memory, exactly as forward_test would on a
        # scene change (orion.py:690-695).
        if model.with_pts_bbox:
            model.pts_bbox_head.reset_memory()
        if model.with_map_head:
            model.map_head.reset_memory()

        for fi, didx in frames:
            if fi > last_needed:
                break                      # nothing left to capture in this clip
            selected = fi in sel_in_clip
            if not selected and args.skip_warmup:
                continue
            try:
                data = collate_one(dataset, didx)
                if not selected:
                    # Memory warm-up only: vision path, no LLM.
                    with torch.no_grad(), LMToggle(model):
                        wrapped(data, return_loss=False)
                    n_warm += 1
                    continue

                with torch.no_grad():
                    out = wrapped(data, return_loss=False)

                res = out["bbox_results"][0] if isinstance(out, dict) else out[0]
                pts = res.get("pts_bbox", {})
                # `ego_fut_preds` in the result dict is ALREADY cumulative: orion.py:932
                # does `ego_fut_pred = ego_fut_preds.cumsum(dim=-2)` before storing it,
                # whereas `ego_fut_trajs` (GT) stays as per-step deltas. Convert the
                # prediction back to deltas so both trajectories in a record share one
                # convention and downstream code can cumsum both, exactly as ORION's own
                # compute_planner_metric_stp3 does.
                pred = decumsum(to_numpy_traj(pts.get("ego_fut_preds")))
                text_out = res.get("text_out", [])
                model_rounds = extract_answers(text_out)

                rec_in = wanted[(clip, fi)]
                info = dataset.data_infos[didx]

                # `collate` yields these as either a tensor or a 1-element list holding
                # one (per test-time-augmentation nesting), so unwrap generically.
                gt_delta = to_numpy_traj(first_tensor(data.get("ego_fut_trajs")))
                fut_mask = first_tensor(data.get("ego_fut_masks"))
                if isinstance(fut_mask, torch.Tensor):
                    fut_mask = fut_mask.detach().float().cpu().numpy().reshape(-1).tolist()

                P_model, _ = build_perception_labels(model_rounds)
                R_model = build_reasoning_labels(model_rounds)

                sid = rec_in["sample_id"]
                if capture.planning_by_layer:
                    arrs = {str(li): v.numpy() for li, v in capture.planning_by_layer.items()}
                    np.savez_compressed(os.path.join(hidden_dir, f"{sid}.npz"), **arrs)
                    if layer_shape_seen is None:
                        li0 = sorted(capture.planning_by_layer)[0]
                        layer_shape_seen = {
                            "n_layers_plus_embed": capture.n_layers + 1 if capture.n_layers else None,
                            "n_saved_layers": len(capture.planning_by_layer),
                            "per_layer_shape": list(capture.planning_by_layer[li0].shape),
                            "seq_len": capture.seq_len,
                            "n_waypoint_positions": capture.n_waypoint_positions,
                        }
                        print(f"[shape] {layer_shape_seen}", flush=True)

                rec = {
                    "sample_id": sid,
                    "clip_id": clip,
                    "frame_idx": fi,
                    "scenario": rec_in["scenario"],
                    "dataset_index": didx,
                    "town": info.get("town_name"),
                    "command": command_index(data.get("ego_fut_cmd")),
                    "trajectory_pred": pred,
                    "trajectory_gt": gt_delta,
                    "fut_mask": fut_mask,
                    "fut_valid_flag": bool(pts.get("fut_valid_flag", False)),
                    "model_rounds": model_rounds,
                    "gt_rounds": rec_in["gt_rounds"],
                    "model_perception": P_model.to_dict(),
                    "model_reasoning": R_model.to_dict(),
                    "gt_perception": rec_in["gt_perception"],
                    "gt_reasoning": rec_in["gt_reasoning"],
                    "hidden_file": f"hidden/{sid}.npz" if capture.planning_by_layer else None,
                    "capture_error": capture.error,
                }
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n_done += 1

                if args.limit and n_done >= args.limit:
                    print(f"[limit] reached {args.limit} selected frames", flush=True)
                    raise _LimitReached

                if n_done % 25 == 0:
                    el = time.time() - t_start
                    print(f"[prog] {n_done}/{len(shard)} selected "
                          f"({n_warm} warm-up) {el/max(n_done,1):.2f}s/sel "
                          f"eta {(len(shard)-n_done)*el/max(n_done,1)/60:.1f}min", flush=True)

            except _LimitReached:
                raise
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                n_fail += 1
                key = f"{type(exc).__name__}"
                fail_reasons[key] += 1
                print(f"[fail] {clip}#{fi}: {exc}", flush=True)
                if n_fail <= 3:
                    traceback.print_exc()
                if n_fail > 50 and n_fail > 3 * max(n_done, 1):
                    print("[abort] too many failures", flush=True)
                    raise

        if args.limit and n_done >= args.limit:
            break
    except _LimitReached:
        pass

    fout.close()
    meta = {
        "shard_split": os.path.abspath(args.split),
        "output": os.path.abspath(args.output),
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "seed": args.seed,
        "layers": args.layers,
        "n_selected_done": n_done,
        "n_warmup_frames": n_warm,
        "n_failed": n_fail,
        "fail_reasons": dict(fail_reasons),
        "hidden_shape": layer_shape_seen,
        "elapsed_sec": time.time() - t_start,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "clips_missing_from_infos": missing,
    }
    with open(os.path.join(args.output, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[done] selected={n_done} warmup={n_warm} failed={n_fail} "
          f"elapsed={meta['elapsed_sec']/60:.1f}min -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
