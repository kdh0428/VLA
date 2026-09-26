#!/usr/bin/env python
"""
AutoVLA inference + hidden-state / action-logit extraction.

For every scene we record, in one forward pass:

  * the generated CoT text and the declared final action (R, secondary)
  * the physical action tokens and the decoded trajectory (A)
  * GT action tokens (`gt_idx`) and GT trajectory
  * hidden states at TWO kinds of position, kept separate as the spec requires:
        prefill        -- last prompt position, i.e. the state the first action token is
                          generated from
        action_step_k  -- the state that produced the k-th action token
    for all 37 layers (embedding + 36 blocks), fp16
  * the action-token logit lens: lm_head applied to EVERY layer at EVERY action step,
    reduced on the fly to per-(layer, step) statistics about the GT action token
    (probability, rank, margin against the argmax, entropy). Storing the raw
    37 x 10 x 2048 logits per scene would be ~1.5 MB/scene of mostly-unused numbers.

Runs on whichever single GPU CUDA_VISIBLE_DEVICES exposes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

AUTOVLA_DIR = "/root/VLA/autovla"
POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AUTOVLA_DIR)
sys.path.insert(0, os.path.join(AUTOVLA_DIR, "navsim"))
sys.path.insert(0, POC_DIR)

from src.labeling.labels import (action_correct, action_semantics,  # noqa: E402
                                 build_perception_labels, parse_declared_action)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--annotations", default=os.path.join(POC_DIR, "outputs/annotations"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/gpu1"))
    ap.add_argument("--config", default=os.path.join(
        AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-hidden", action="store_true", default=True)
    ap.add_argument("--no-save-hidden", dest="save_hidden", action="store_false")
    return ap.parse_args()


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Resolve every user path BEFORE chdir'ing into the AutoVLA repo (the repo's configs
    # and the sensor blobs are addressed relative to that root, so the chdir is required).
    args.output = os.path.abspath(args.output)
    args.scenes = os.path.abspath(args.scenes)
    args.annotations = os.path.abspath(args.annotations)
    args.checkpoint = os.path.abspath(args.checkpoint)
    args.config = os.path.abspath(args.config)
    os.makedirs(args.output, exist_ok=True)
    hid_dir = os.path.join(args.output, "hidden")
    os.makedirs(hid_dir, exist_ok=True)
    os.chdir(AUTOVLA_DIR)

    from models.autovla import AutoVLA
    from navsim.agents.autovla_agent import AutoVLAAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from dataset_utils.preprocessing.nuplan_dataset import get_action_instruction

    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["pretrained_model_path"] = os.path.join(AUTOVLA_DIR, "Qwen2.5-VL-3B-Instruct")
    cfg["model"]["codebook_cache_path"] = os.path.join(AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl")

    t0 = time.time()
    model = AutoVLA(cfg, inference=True, device="cuda:0")
    sd = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    model.load_state_dict({k[len("autovla."):]: v for k, v in sd.items()
                           if k.startswith("autovla.")}, strict=False)
    model.to("cuda:0").eval()
    del sd
    llm = model.vlm
    A0 = model.action_start_id
    n_layers = llm.config.num_hidden_layers if hasattr(llm.config, "num_hidden_layers") \
        else llm.config.text_config.num_hidden_layers
    print(f"[build] {time.time()-t0:.0f}s  layers={n_layers} "
          f"lm_head={tuple(llm.lm_head.weight.shape)}", flush=True)

    agent = AutoVLAAgent(
        trajectory_sampling=TrajectorySampling(
            time_horizon=cfg["model"]["trajectory"]["time_horizon"],
            interval_length=cfg["model"]["trajectory"]["interval_length"]),
        sensor_data_path=Path("dataset/nuplan/sensor_blobs/test"),
        codebook_cache_path=cfg["model"]["codebook_cache_path"],
        skip_model_load=True)

    files = sorted(Path(args.scenes).glob("*.json"))[args.start:]
    if args.limit:
        files = files[: args.limit]
    print(f"[shard] {len(files)} scenes -> {args.output}", flush=True)

    # lm_head weight restricted to the action-token rows: the logit lens only ever needs
    # the 2048 action logits, so this avoids a 153k-wide matmul per (layer, step).
    W_act = llm.lm_head.weight[A0:].detach()                      # (2048, hidden)
    gen_conf = cfg["inference"]["sample"]

    out_path = os.path.join(args.output, "records.jsonl")
    fout = open(out_path, "w")
    n_ok = n_fail = 0
    shape_logged = False
    tstart = time.time()

    for i, sp in enumerate(files, 1):
        token = sp.stem
        try:
            scene_data = json.load(open(sp))
            anno_p = os.path.join(args.annotations, f"{token}.json")
            anno = json.load(open(anno_p)) if os.path.exists(anno_p) else {}

            feats, targs = {}, {}
            for b in agent.get_feature_builders():
                feats.update(b.compute_features(scene_data))
            for b in agent.get_target_builders():
                targs.update(b.compute_targets(scene_data))

            inputs = model.get_prompt(feats)
            n_in = int(inputs["input_ids"].shape[1])
            mi = {k: v.to("cuda:0") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                gen = llm.generate(**mi, max_length=gen_conf["max_length"],
                                   do_sample=True, temperature=gen_conf["temperature"],
                                   top_k=gen_conf["top_k"], top_p=gen_conf["top_p"],
                                   return_dict_in_generate=True, output_hidden_states=True)

            new_ids = gen.sequences[0][n_in:]
            text = model.processor.decode(new_ids, skip_special_tokens=False)
            act_mask = new_ids >= A0
            act_ids = new_ids[act_mask]
            act_pos = torch.nonzero(act_mask).flatten().tolist()

            # ---- trajectories -------------------------------------------------------
            pred = model.action_tokenizer.decode_token_ids_to_trajectory(act_ids.cpu())
            pred_xy = np.asarray(pred[0, 1:], dtype=np.float32) if len(pred) else np.zeros((0, 3))
            gt_xy = np.asarray(scene_data["gt_trajectory"], dtype=np.float32)
            gt_idx = np.asarray(targs["gt_idx"], dtype=np.int64).reshape(-1)

            sem_p = action_semantics(pred_xy, get_action_instruction) if len(pred_xy) else None
            sem_g = action_semantics(gt_xy, get_action_instruction)
            verdict = action_correct(sem_p, sem_g, pred_xy, gt_xy) if sem_p else None

            P, _ = build_perception_labels(anno)
            lon_d, lat_d, cot_present = parse_declared_action(text)

            # ---- hidden states ------------------------------------------------------
            hs = gen.hidden_states           # [step][layer] -> (1, seq_or_1, hidden)
            saved = {}
            if args.save_hidden and act_pos:
                # prefill: last prompt position (state the first generated token came from)
                saved["prefill"] = np.stack(
                    [hs[0][l][0, -1, :].float().cpu().numpy() for l in range(len(hs[0]))]
                ).astype(np.float16)
                for k, t in enumerate(act_pos):
                    saved[f"action_{k}"] = np.stack(
                        [hs[t][l][0, -1, :].float().cpu().numpy() for l in range(len(hs[t]))]
                    ).astype(np.float16)

            # ---- action logit lens --------------------------------------------------
            lens = []
            n_cmp = min(len(act_pos), len(gt_idx))
            for k in range(n_cmp):
                t = act_pos[k]
                gt_a = int(gt_idx[k])
                pred_a = int(act_ids[k].item() - A0)
                per_layer = []
                for l in range(len(hs[t])):
                    h = hs[t][l][0, -1, :].to(W_act.dtype)
                    lg = (W_act @ h).float()
                    pr = torch.softmax(lg, dim=-1)
                    top = int(lg.argmax())
                    rank = int((lg > lg[gt_a]).sum())
                    per_layer.append({
                        "layer": l,
                        "gt_prob": float(pr[gt_a]),
                        "gt_rank": rank,
                        "top_token": top,
                        "top_prob": float(pr[top]),
                        "margin_gt_minus_top": float(lg[gt_a] - lg[top]),
                        "entropy": entropy(pr.cpu().numpy()),
                        "top_is_gt": bool(top == gt_a),
                        "top_is_generated": bool(top == pred_a),
                    })
                lens.append({"step": k, "gt_action": gt_a, "generated_action": pred_a,
                             "layers": per_layer})

            if saved:
                np.savez_compressed(os.path.join(hid_dir, f"{token}.npz"), **saved)
            if not shape_logged and saved:
                print(f"[shape] prefill {saved['prefill'].shape}  "
                      f"steps {len(act_pos)}  lens layers {len(lens[0]['layers']) if lens else 0}",
                      flush=True)
                shape_logged = True

            rec = {
                "token": token,
                "log_name": anno.get("log_name"),
                "map_name": anno.get("map_name"),
                "instruction": scene_data.get("instruction"),
                "n_input_tokens": n_in,
                "n_generated": int(len(new_ids)),
                "action_positions": act_pos,
                "action_token_ids": act_ids.tolist(),
                "pred_action_idx": [int(a) - A0 for a in act_ids.tolist()],
                "gt_action_idx": gt_idx.tolist(),
                "generated_text": text,
                "cot_present": cot_present,
                "declared_lon": lon_d, "declared_lat": lat_d,
                "trajectory_pred": pred_xy[:, :2].tolist(),
                "trajectory_gt": gt_xy[:, :2].tolist(),
                "gt_action_instruction": scene_data.get("gt_action_instruction"),
                "perception": P.to_dict(),
                "action_pred": sem_p.to_dict() if sem_p else None,
                "action_gt": sem_g.to_dict(),
                "verdict": verdict.to_dict() if verdict else None,
                "logit_lens": lens,
                "hidden_file": f"hidden/{token}.npz" if saved else None,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            if n_fail <= 5:
                import traceback
                print(f"[fail] {token}: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
        if i % 100 == 0:
            el = time.time() - tstart
            print(f"[prog] {i}/{len(files)} ok={n_ok} fail={n_fail} "
                  f"{el/i:.2f}s/scene eta {(len(files)-i)*el/i/60:.1f}min", flush=True)

    fout.close()
    meta = {
        "n_ok": n_ok, "n_fail": n_fail, "n_layers": int(n_layers),
        "action_start_id": int(A0), "seed": args.seed,
        "checkpoint": args.checkpoint, "config": args.config,
        "elapsed_min": (time.time() - tstart) / 60,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
    }
    with open(os.path.join(args.output, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[done] ok={n_ok} fail={n_fail} in {meta['elapsed_min']:.1f}min", flush=True)


if __name__ == "__main__":
    main()
