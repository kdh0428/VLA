#!/usr/bin/env python
"""
Teacher-forced read-out for the CoT intervention (GPU 1).

Free generation compounds: once a later action token differs, every token after it differs,
so trajectory change mixes the CoT's direct effect with sampling divergence at near-ties.
Here every condition is scored on the SAME action sequences:

  gt_seq    the GT action tokens
  orig_seq  the action tokens the in-harness `original` condition generated

For each condition and step: log p(token | prefix, earlier forced tokens) over the full
vocabulary, the GT-token margin inside the action vocabulary, and the action-token logits
under orig_seq (so JS vs `original` is measured without compounding).

Reads outputs/cot_intervention/records.jsonl; writes outputs/cot_intervention/teacher_forced/.
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

CONDS = ["natural_nocot", "original", "template_original", "corrected_decision",
         "corrected_full", "counter_decision", "counter_full"]
N_ACT = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(POC_DIR, "outputs/cot_intervention"))
    ap.add_argument("--full-records", default=os.path.join(POC_DIR, "outputs/full_extract/records.jsonl"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--config", default=os.path.join(AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" and not os.environ.get("VLA_ANY_GPU"):
        raise SystemExit("pinned to GPU 1: launch with CUDA_VISIBLE_DEVICES=1")
    for k in ("run", "full_records", "scenes", "config"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    out_dir = os.path.join(args.run, "teacher_forced")
    os.makedirs(out_dir, exist_ok=True)

    recs = [json.loads(l) for l in open(os.path.join(args.run, "records.jsonl"))]
    if args.limit:
        recs = recs[:args.limit]
    want = {r["token"] for r in recs}
    full = {}
    for line in open(args.full_records):
        x = json.loads(line)
        if x["token"] in want:
            full[x["token"]] = x

    os.chdir(AUTOVLA_DIR)
    from models.autovla import AutoVLA
    from navsim.agents.autovla_agent import AutoVLAAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["pretrained_model_path"] = os.path.join(AUTOVLA_DIR, "Qwen2.5-VL-3B-Instruct")
    cfg["model"]["codebook_cache_path"] = os.path.join(AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl")
    model = AutoVLA(cfg, inference=True, device="cuda:0")
    sd = torch.load(args.checkpoint, map_location="cpu", mmap=True)["state_dict"]
    model.load_state_dict({k[len("autovla."):]: v for k, v in sd.items() if k.startswith("autovla.")}, strict=False)
    model.to("cuda:0").eval()
    del sd
    llm, A0, tok = model.vlm, model.action_start_id, model.processor.tokenizer

    vis_box = {}
    orig_visual = llm.visual.forward

    def visual_cached(pixel_values, grid_thw=None, **kw):
        if vis_box.get("v") is None:
            vis_box["v"] = orig_visual(pixel_values, grid_thw=grid_thw, **kw)
        return vis_box["v"]
    llm.visual.forward = visual_cached

    agent = AutoVLAAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=cfg["model"]["trajectory"]["time_horizon"],
                                               interval_length=cfg["model"]["trajectory"]["interval_length"]),
        sensor_data_path=Path("dataset/nuplan/sensor_blobs/test"),
        codebook_cache_path=cfg["model"]["codebook_cache_path"], skip_model_load=True)

    t0 = time.time()
    for i, r in enumerate(recs, 1):
        token = r["token"]
        fr = full[token]
        scene = json.load(open(os.path.join(args.scenes, f"{token}.json")))
        vis_box.clear()
        feats = {}
        for b in agent.get_feature_builders():
            feats.update(b.compute_features(scene))
        mi = {k: v.to("cuda:0") for k, v in model.get_prompt(feats).items() if isinstance(v, torch.Tensor)}
        prompt_ids = mi["input_ids"][0].tolist()

        c_ids, n_ids = fr["arms"]["C"]["token_ids"], fr["arms"]["N"]["token_ids"]
        prefixes = {
            "natural_nocot": n_ids[:next(j for j, t in enumerate(n_ids) if t >= A0)],
            "original": c_ids[:next(j for j, t in enumerate(c_ids) if t >= A0)],
        }
        for name in CONDS[2:]:
            c = r["conditions"][name]
            prefixes[name] = None if c.get("skipped") else tok.encode(c["prefix_text"], add_special_tokens=False)

        gt_seq = [A0 + a for a in r["gt_action_idx"][:N_ACT]]
        o = r["conditions"]["original"]
        orig_seq = [A0 + a for a in o["action_idx"]] if (not o.get("skipped") and o["n_action"] == N_ACT) else None

        C = len(CONDS)
        gt_lp = np.full((C, N_ACT), np.nan, np.float32)
        gt_margin = np.full((C, N_ACT), np.nan, np.float32)
        orig_lp = np.full((C, N_ACT), np.nan, np.float32)
        orig_logits = np.full((C, N_ACT, 2048), np.nan, np.float16)
        valid = np.zeros(C, bool)
        for ci, name in enumerate(CONDS):
            pids = prefixes[name]
            if pids is None or orig_seq is None:
                continue
            for seq, which in ((gt_seq, "gt"), (orig_seq, "orig")):
                ids = torch.tensor([prompt_ids + list(pids) + seq], device="cuda:0")
                m2 = dict(mi); m2["input_ids"] = ids; m2["attention_mask"] = torch.ones_like(ids)
                with torch.no_grad():
                    lg = llm(**m2, use_cache=False).logits[0]
                start = len(prompt_ids) + len(pids) - 1
                step = lg[start:start + N_ACT].float()                     # (10, vocab)
                lp = torch.log_softmax(step, -1)
                tgt = torch.tensor(seq, device="cuda:0")
                picked = lp.gather(1, tgt[:, None])[:, 0].cpu().numpy()
                if which == "gt":
                    gt_lp[ci] = picked
                    act = step[:, A0:]
                    g = tgt - A0
                    gl = act.gather(1, g[:, None])[:, 0]
                    act_m = act.clone()
                    act_m.scatter_(1, g[:, None], -1e9)
                    gt_margin[ci] = (gl - act_m.max(1).values).cpu().numpy()
                else:
                    orig_lp[ci] = picked
                    orig_logits[ci] = step[:, A0:].cpu().numpy().astype(np.float16)
                del lg
            valid[ci] = True
        np.savez_compressed(os.path.join(out_dir, f"{token}.npz"), gt_lp=gt_lp, gt_margin=gt_margin,
                            orig_lp=orig_lp, orig_logits=orig_logits, valid=valid, conditions=np.array(CONDS))
        if i == 1 or i % 20 == 0:
            el = time.time() - t0
            print(f"[prog] {i}/{len(recs)} {el/i:.1f}s/scene eta {(len(recs)-i)*el/i/60:.1f}min | "
                  f"gt_lp sum orig={np.nansum(gt_lp[1]):.2f} corr_full={np.nansum(gt_lp[4]):.2f} cf_full={np.nansum(gt_lp[6]):.2f}", flush=True)
    print(f"[done] {len(recs)} scenes in {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
