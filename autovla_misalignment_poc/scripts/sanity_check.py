#!/usr/bin/env python
"""
AutoVLA Phase-1 sanity check.

Loads the official checkpoint, runs a handful of real navtest scenarios, and records
everything the later analysis depends on:

  * input token sequence length, visual vs text token counts
  * generated CoT text and whether the model chose to reason at all (adaptive thinking)
  * physical action token ids and their positions in the generated sequence
  * decoded trajectory vs GT trajectory
  * hidden-state tensor shapes, layer count, hidden dim
  * action-token logits shape from lm_head applied to each layer (logit-lens feasibility)

Nothing in /root/VLA/autovla is modified; this only reads the repo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

AUTOVLA_DIR = "/root/VLA/autovla"
POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AUTOVLA_DIR)
sys.path.insert(0, os.path.join(AUTOVLA_DIR, "navsim"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(AUTOVLA_DIR,
                    "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--scene-filter",
                    default=os.path.join(POC_DIR, "configs/scene_filter_navtest_subset.yaml"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--scenes", default="/root/VLA/autovla/dataset/nuplan/navtest_poc")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(POC_DIR, "outputs/sanity"))
    args = ap.parse_args()
    os.chdir(AUTOVLA_DIR)                    # configs use repo-relative paths
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoProcessor
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader
    from navsim.agents.autovla_agent import AutoVLAAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from models.autovla import AutoVLA

    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["pretrained_model_path"] = os.path.join(AUTOVLA_DIR, "Qwen2.5-VL-3B-Instruct")
    cfg["model"]["codebook_cache_path"] = os.path.join(AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl")

    print("=== building model ===", flush=True)
    model = AutoVLA(cfg, inference=True, device=args.device)
    tok = model.processor.tokenizer
    print("vocab (with action tokens):", len(tok))
    print("action_start_id:", model.action_start_id)
    print("use_cot:", model.use_cot)

    sd = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    sd = {k[len("autovla."):]: v for k, v in sd.items() if k.startswith("autovla.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded checkpoint: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(args.device).eval()

    llm = model.vlm
    n_layers = llm.config.num_hidden_layers if hasattr(llm.config, "num_hidden_layers") \
        else llm.config.text_config.num_hidden_layers
    hidden = llm.config.hidden_size if hasattr(llm.config, "hidden_size") \
        else llm.config.text_config.hidden_size
    print(f"decoder layers: {n_layers}  hidden dim: {hidden}")
    print("lm_head:", tuple(llm.lm_head.weight.shape))

    # ---- data ----
    # The AutoVLA feature builder consumes the PREPROCESSED scene dict (not a navsim
    # Scene object) -- same path the official nusc_eval.py takes.
    agent = AutoVLAAgent(
        trajectory_sampling=TrajectorySampling(
            time_horizon=cfg["model"]["trajectory"]["time_horizon"],
            interval_length=cfg["model"]["trajectory"]["interval_length"]),
        sensor_data_path=Path("dataset/nuplan/sensor_blobs/test"),
        codebook_cache_path=cfg["model"]["codebook_cache_path"],
        skip_model_load=True)
    scene_files = sorted(Path(args.scenes).glob("*.json"))
    print("preprocessed scenes:", len(scene_files))

    records = []
    for i, sp in enumerate(scene_files[: args.n]):
        token = sp.stem
        print(f"\n=== sample {i}: {token} ===", flush=True)
        scene_data = json.load(open(sp))
        feats, targets = {}, {}
        for b in agent.get_feature_builders():
            feats.update(b.compute_features(scene_data))
        for b in agent.get_target_builders():
            targets.update(b.compute_targets(scene_data))

        inputs = model.get_prompt(feats)
        n_in = int(inputs["input_ids"].shape[1])
        ids = inputs["input_ids"][0]
        # Qwen marks visual positions with the video/image pad token
        vid_tok = getattr(llm.config, "video_token_id", None)
        img_tok = getattr(llm.config, "image_token_id", None)
        n_vis = int(sum((ids == t).sum().item() for t in (vid_tok, img_tok) if t is not None))
        print(f"input tokens: {n_in}  (visual {n_vis}, text {n_in - n_vis})")

        model_inputs = {k: v.to(args.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            gen = llm.generate(**model_inputs,
                               max_length=cfg["inference"]["sample"]["max_length"],
                               do_sample=True,
                               temperature=cfg["inference"]["sample"]["temperature"],
                               top_k=cfg["inference"]["sample"]["top_k"],
                               top_p=cfg["inference"]["sample"]["top_p"],
                               return_dict_in_generate=True,
                               output_hidden_states=True,
                               output_scores=True)

        seq = gen.sequences[0]
        new_ids = seq[n_in:]
        text = model.processor.decode(new_ids, skip_special_tokens=False)
        act_mask = new_ids >= model.action_start_id
        act_ids = new_ids[act_mask]
        act_pos = torch.nonzero(act_mask).flatten().tolist()
        print(f"generated tokens: {len(new_ids)}  action tokens: {len(act_ids)}")
        print(f"action token ids: {act_ids.tolist()[:12]}")
        print(f"action positions in generation: {act_pos[:12]}")
        print(f"CoT present: {'Chain-of-Thought is not needed' not in text}")
        print(f"text head: {text[:220]!r}")

        traj = model.action_tokenizer.decode_token_ids_to_trajectory(act_ids.cpu())[0, 1:]
        gt = targets["trajectory"].poses if hasattr(targets.get("trajectory", None), "poses") \
            else targets.get("gt_pos_raw")
        traj_np = np.asarray(traj.cpu() if torch.is_tensor(traj) else traj, dtype=np.float32)
        gt_np = np.asarray(gt.cpu() if torch.is_tensor(gt) else gt, dtype=np.float32)
        print(f"decoded trajectory: {traj_np.shape}  GT: {gt_np.shape}")
        print(f"  pred[:3] {np.round(traj_np[:3, :2], 3).tolist()}")
        print(f"  gt  [:3] {np.round(gt_np[:3, :2], 3).tolist()}")

        # hidden states: tuple per generation step, each a tuple per layer
        hs = gen.hidden_states
        print(f"hidden_states: {len(hs)} steps; step0 has {len(hs[0])} layer tensors, "
              f"shape {tuple(hs[0][0].shape)}")
        print(f"  last step layer tensor shape {tuple(hs[-1][0].shape)}")
        print(f"scores: {len(gen.scores)} steps, shape {tuple(gen.scores[0].shape)}")

        # logit-lens feasibility: apply lm_head to a mid-layer state at an action step
        if act_pos:
            t = act_pos[0]
            h = hs[t][n_layers // 2][0, -1, :].to(llm.lm_head.weight.dtype)
            lg = llm.lm_head(h)
            act_lg = lg[model.action_start_id:]
            print(f"  logit-lens @ step {t}, layer {n_layers//2}: full {tuple(lg.shape)}, "
                  f"action slice {tuple(act_lg.shape)}, argmax action "
                  f"{int(act_lg.argmax())} (id {model.action_start_id + int(act_lg.argmax())}), "
                  f"actual generated {int(act_ids[0])}")

        records.append({
            "token": token, "n_input_tokens": n_in, "n_visual_tokens": n_vis,
            "n_generated": int(len(new_ids)), "n_action_tokens": int(len(act_ids)),
            "action_token_ids": act_ids.tolist(), "action_positions": act_pos,
            "cot_present": bool("Chain-of-Thought is not needed" not in text),
            "generated_text": text,
            "trajectory_pred": traj_np.tolist(), "trajectory_gt": gt_np.tolist(),
            "n_layers": int(n_layers), "hidden_dim": int(hidden),
            "n_gen_steps": len(hs), "layer_tensors_per_step": len(hs[0]),
        })

    with open(os.path.join(args.out, "sanity.json"), "w") as f:
        json.dump(records, f, indent=1)
    print(f"\nwrote {args.out}/sanity.json")


if __name__ == "__main__":
    main()
