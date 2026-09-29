#!/usr/bin/env python
"""
Where does the action position look when it gets the first token wrong?

The counterfactual established that a wrong action token propagates. It said nothing about
why the FIRST token was wrong. If the "encoded but unused" reading is right, the perceptual
evidence is present in the sequence but the action position fails to draw on it — which
should show up as attention mass on the visual span.

Measurement
-----------
At a chosen action-generation step the model attends over a sequence made of four spans:

    visual    the video pad tokens (3 cameras x 4 frames)
    text      the rest of the prompt (system, camera captions, ego state, instruction)
    scaffold  the generated CoT / "<answer> The final output action is:" tokens
    action    the action tokens generated so far

For the query position that emits the step's token we record, per (layer, head), how much
attention mass falls on each span. Two comparisons:

  between-sample  at action step 0: samples whose first action token is wrong vs correct
  within-sample   at the first-error step t vs the immediately preceding step t-1, which
                  was correct by construction — same scene, same image, adjacent positions,
                  so it controls for scene difficulty almost perfectly

Only the last query row of each attention matrix is kept; everything else is discarded
immediately so memory stays bounded.
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

SPANS = ("visual", "text", "scaffold", "action")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(POC_DIR, "outputs/gpu1/records.jsonl"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/attention"))
    ap.add_argument("--config", default=os.path.join(
        AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--max-samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    for k in ("output", "records", "scenes"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.makedirs(args.output, exist_ok=True)
    os.chdir(AUTOVLA_DIR)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    from models.autovla import AutoVLA
    from navsim.agents.autovla_agent import AutoVLAAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["pretrained_model_path"] = os.path.join(AUTOVLA_DIR, "Qwen2.5-VL-3B-Instruct")
    cfg["model"]["codebook_cache_path"] = os.path.join(AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl")

    model = AutoVLA(cfg, inference=True, device="cuda:0")
    sd = torch.load(args.checkpoint, map_location="cpu", mmap=True)["state_dict"]
    model.load_state_dict({k[len("autovla."):]: v for k, v in sd.items()
                           if k.startswith("autovla.")}, strict=False)
    model.to("cuda:0").eval()
    del sd
    llm, A0 = model.vlm, model.action_start_id
    # eager attention is required for output_attentions to be populated
    llm.config._attn_implementation = "eager"
    if hasattr(llm.config, "text_config"):
        llm.config.text_config._attn_implementation = "eager"
    VIDEO_ID = llm.config.video_token_id
    IMAGE_ID = llm.config.image_token_id
    gen_conf = cfg["inference"]["sample"]

    agent = AutoVLAAgent(
        trajectory_sampling=TrajectorySampling(
            time_horizon=cfg["model"]["trajectory"]["time_horizon"],
            interval_length=cfg["model"]["trajectory"]["interval_length"]),
        sensor_data_path=Path("dataset/nuplan/sensor_blobs/test"),
        codebook_cache_path=cfg["model"]["codebook_cache_path"],
        skip_model_load=True)

    recs = []
    for line in open(args.records):
        if line.strip():
            r = json.loads(line)
            if r.get("action_positions") and r.get("gt_action_idx"):
                recs.append(r)
    idx = rng.permutation(len(recs))[: args.max_samples]
    chosen = [recs[i] for i in sorted(idx)]
    print(f"[select] {len(chosen)} samples", flush=True)

    def span_masses(attn_layers, spans, qpos):
        """attn_layers: tuple per layer of (1, H, L, L) -> array (n_layers, H, 4)."""
        out = np.zeros((len(attn_layers), attn_layers[0].shape[1], len(SPANS)), np.float32)
        for li, A in enumerate(attn_layers):
            row = A[0, :, qpos, :].float()               # (H, L)
            for si, s in enumerate(SPANS):
                m = spans[s]
                out[li, :, si] = row[:, m].sum(-1).cpu().numpy() if m.any() else 0.0
        return out

    results = []
    t0 = time.time()
    for i, r in enumerate(chosen, 1):
        try:
            scene = json.load(open(os.path.join(args.scenes, f"{r['token']}.json")))
            feats = {}
            for b in agent.get_feature_builders():
                feats.update(b.compute_features(scene))
            inputs = model.get_prompt(feats)
            mi = {k: v.to("cuda:0") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            prompt_ids = mi["input_ids"][0]
            n_prompt = prompt_ids.shape[0]

            # regenerate on-policy so the prefix is the model's own
            with torch.no_grad():
                warm = llm.generate(**mi, max_length=gen_conf["max_length"], do_sample=True,
                                    temperature=gen_conf["temperature"],
                                    top_k=gen_conf["top_k"], top_p=gen_conf["top_p"])
            new_ids = warm[0][n_prompt:]
            apos = torch.nonzero(new_ids >= A0).flatten().tolist()
            gt = r["gt_action_idx"]
            if len(apos) < 2 or len(gt) < 2:
                continue
            pred = [int(new_ids[p].item()) - A0 for p in apos]
            n = min(len(pred), len(gt))
            diff = [k for k in range(n) if pred[k] != gt[k]]
            first_err = diff[0] if diff else None

            # query positions to probe: step 0 always; plus first-error step and the step
            # before it when that exists and is >0
            probe = {0: ("step0", pred[0] == gt[0])}
            if first_err is not None and first_err >= 1:
                probe[first_err] = ("first_error", False)
                probe[first_err - 1] = ("pre_error", True)

            for step, (kind, correct) in probe.items():
                cut = n_prompt + apos[step]        # absolute index of that action token
                ids = warm[0][:cut].unsqueeze(0)   # everything before it
                am = torch.ones_like(ids)
                kw = {k: v for k, v in mi.items() if k not in ("input_ids", "attention_mask")}
                with torch.no_grad():
                    out = llm(input_ids=ids, attention_mask=am, **kw, output_attentions=True)
                L = ids.shape[1]
                seq = ids[0]
                spans = {
                    "visual": ((seq == VIDEO_ID) | (seq == IMAGE_ID)),
                    "action": (seq >= A0),
                }
                gen_mask = torch.zeros(L, dtype=torch.bool, device=seq.device)
                gen_mask[n_prompt:] = True
                spans["scaffold"] = gen_mask & ~spans["action"]
                spans["text"] = ~(spans["visual"] | spans["action"] | spans["scaffold"])
                masses = span_masses(out.attentions, spans, L - 1)
                del out
                results.append({
                    "token": r["token"], "log": r.get("log_name"),
                    "step": int(step), "kind": kind, "correct": bool(correct),
                    "n_visual": int(spans["visual"].sum()), "seq_len": int(L),
                    "mass": masses.tolist(),
                })
            torch.cuda.empty_cache()
        except Exception as exc:
            if i <= 5:
                import traceback
                print(f"[fail] {r['token']}: {exc}", flush=True)
                traceback.print_exc()
        if i % 25 == 0:
            el = time.time() - t0
            print(f"[prog] {i}/{len(chosen)} rows={len(results)} "
                  f"{el/i:.2f}s/sample eta {(len(chosen)-i)*el/i/60:.1f}min", flush=True)

    np.save(os.path.join(args.output, "mass.npy"),
            np.array([r.pop("mass") for r in results], dtype=np.float32))
    with open(os.path.join(args.output, "rows.json"), "w") as f:
        json.dump(results, f)
    print(f"\n[done] {len(results)} probe rows in {(time.time()-t0)/60:.1f}min -> {args.output}")


if __name__ == "__main__":
    main()
