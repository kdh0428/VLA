#!/usr/bin/env python
"""
Does explicit CoT help AutoVLA drive?  An unbiased, paired evaluation (GPU 1).

Scenes are drawn from the whole navtest PoC set (2748) by map-stratified random sampling
with a fixed seed. No label, failure flag, or earlier model output is used to choose them.

Each scene runs three conditions in ONE harness, with the same prompt, vision tensor and
per-scene seed, and AutoVLA's own decoding config (do_sample, T=0.01, top_k=0, top_p=1,
max_length=2048):

  natural   free generation from the prompt -- the model decides whether to reason
  fast      teacher-forced no-CoT stub, then action tokens
            "<think>\\nThis is a straightforward scenario, and a direct decision can be made.\\n</think>\\n<answer>\\nThe final output action is: "
  cot       teacher-forced "<think>\\nThis is a complex", then FREE generation of the whole CoT and
            the action tokens (runaway / max_length outcomes are kept, not dropped)

All three conditions are regenerated rather than reusing the stored runs. A one-shot prefill
and the original incremental decode differ numerically enough to change later action
tokens, so a paired comparison is only clean inside a single harness. The stored natural
run is used afterwards as a reproduction check and as an independent uncertainty
measurement for stratification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

AUTOVLA_DIR = "/root/VLA/autovla"
POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AUTOVLA_DIR)
sys.path.insert(0, os.path.join(AUTOVLA_DIR, "navsim"))
sys.path.insert(0, POC_DIR)

CONDS = ["natural", "fast", "cot"]
N_ACT = 10
TOK_COMPLEX = 6351
FAST_STUB = ("<think>\nThis is a straightforward scenario, and a direct decision can be made.\n</think>\n"
             "<answer>\nThe final output action is: ")
COT_STEM = "<think>\nThis is a"


def sample_scenes(records_path: str, n: int, seed: int) -> list[str]:
    """Map-stratified random sample, proportional allocation; nothing outcome-related is read."""
    by_map = defaultdict(list)
    for line in open(records_path):
        r = json.loads(line)
        by_map[r["map_name"]].append(r["token"])
    total = sum(len(v) for v in by_map.values())
    rng = random.Random(seed)
    alloc = {m: int(round(n * len(v) / total)) for m, v in by_map.items()}
    # fix rounding so the allocation sums to n exactly
    while sum(alloc.values()) != n:
        m = max(by_map, key=lambda k: len(by_map[k]) / total * n - alloc[k]) if sum(alloc.values()) < n \
            else min(by_map, key=lambda k: len(by_map[k]) / total * n - alloc[k])
        alloc[m] += 1 if sum(alloc.values()) < n else -1
    picked = []
    for m in sorted(by_map):
        toks = sorted(by_map[m])
        rng.shuffle(toks)
        picked += toks[:alloc[m]]
    rng.shuffle(picked)
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(POC_DIR, "outputs/full_extract/records.jsonl"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/fast_vs_cot_unbiased"))
    ap.add_argument("--config", default=os.path.join(AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("pinned to GPU 1: launch with CUDA_VISIBLE_DEVICES=1")
    for k in ("records", "scenes", "output", "config"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.makedirs(os.path.join(args.output, "logits"), exist_ok=True)

    tokens = sample_scenes(args.records, args.n, args.seed)
    with open(os.path.join(args.output, "sample.json"), "w") as f:
        json.dump({"seed": args.seed, "n": len(tokens), "method": "map-stratified random, proportional",
                   "tokens": tokens}, f, indent=1)
    if args.limit:
        tokens = tokens[:args.limit]
    meta_rec = {}
    for line in open(args.records):
        r = json.loads(line)
        meta_rec[r["token"]] = {k: r[k] for k in ("log_name", "map_name", "instruction", "gt_action_instruction",
                                                  "velocity", "gt_action_idx", "trajectory_gt")}
        meta_rec[r["token"]]["stored_armN_action_idx"] = r["arms"]["N"]["pred_action_idx"][:N_ACT]
        meta_rec[r["token"]]["stored_armC_action_idx"] = r["arms"]["C"]["pred_action_idx"][:N_ACT]
    print(f"[plan] {len(tokens)} scenes x {len(CONDS)} conditions -> {args.output}", flush=True)

    os.chdir(AUTOVLA_DIR)
    from models.autovla import AutoVLA
    from navsim.agents.autovla_agent import AutoVLAAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["pretrained_model_path"] = os.path.join(AUTOVLA_DIR, "Qwen2.5-VL-3B-Instruct")
    cfg["model"]["codebook_cache_path"] = os.path.join(AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl")
    gc = cfg["inference"]["sample"]
    model = AutoVLA(cfg, inference=True, device="cuda:0")
    sd = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    model.load_state_dict({k[len("autovla."):]: v for k, v in sd.items() if k.startswith("autovla.")}, strict=False)
    model.to("cuda:0").eval()
    del sd
    llm, A0, tok = model.vlm, model.action_start_id, model.processor.tokenizer
    fast_ids = tok.encode(FAST_STUB, add_special_tokens=False)
    cot_ids = tok.encode(COT_STEM, add_special_tokens=False) + [TOK_COMPLEX]
    end_think = tok.encode("</think>", add_special_tokens=False)

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

    def find_sub(seq, pat):
        for i in range(len(seq) - len(pat) + 1):
            if seq[i:i + len(pat)] == pat:
                return i
        return -1

    fout = open(os.path.join(args.output, "records.jsonl"), "w")
    t0 = time.time()
    n_ok = n_fail = 0
    for i, token in enumerate(tokens, 1):
        try:
            m = meta_rec[token]
            scene = json.load(open(os.path.join(args.scenes, f"{token}.json")))
            vis_box.clear()
            feats = {}
            for b in agent.get_feature_builders():
                feats.update(b.compute_features(scene))
            mi = {k: v.to("cuda:0") for k, v in model.get_prompt(feats).items() if isinstance(v, torch.Tensor)}
            prompt_ids = mi["input_ids"][0].tolist()
            seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{token}".encode()).digest()[:4], "little")
            gt0 = int(m["gt_action_idx"][0])
            store = np.full((len(CONDS), 2048), np.nan, np.float16)
            conds = {}
            for ci, name in enumerate(CONDS):
                head = [] if name == "natural" else (fast_ids if name == "fast" else cot_ids)
                ids = torch.tensor([prompt_ids + head], device="cuda:0")
                m2 = dict(mi); m2["input_ids"] = ids; m2["attention_mask"] = torch.ones_like(ids)
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                kw = dict(do_sample=True, temperature=gc["temperature"], top_k=gc["top_k"], top_p=gc["top_p"],
                          return_dict_in_generate=True, output_logits=True)
                if name == "fast":
                    kw["max_new_tokens"] = 16
                else:
                    kw["max_length"] = gc["max_length"]
                with torch.no_grad():
                    g = llm.generate(**m2, **kw)
                new = g.sequences[0][ids.shape[1]:].tolist()
                full_gen = head + new                                   # what the assistant turn contains
                first_act = next((j for j, t in enumerate(new) if t >= A0), None)
                all_act = [t for t in new if t >= A0]
                act = all_act[:N_ACT]
                rec_c = {"n_generated": len(new), "n_action_tokens_total": len(all_act),
                         "action_idx": [t - A0 for t in act], "valid": len(act) == N_ACT,
                         "truncated": bool(new) and new[-1] != tok.eos_token_id and len(prompt_ids) + len(full_gen) >= gc["max_length"],
                         "runaway": len(all_act) > N_ACT}
                rec_c["cot_present"] = TOK_COMPLEX in full_gen[:8]
                # Count on text: "<think>\n" tokenizes differently in context than "<think>" alone,
                # so an id-sequence search for the tag misses it.
                txt = tok.decode(full_gen, skip_special_tokens=False)
                body = txt.split("</think>")[0] if "</think>" in txt else txt.split("<answer>")[0]
                rec_c["think_tokens"] = len(tok.encode(body, add_special_tokens=False))
                if first_act is not None:
                    lg = g.logits[first_act][0].float()
                    lp = torch.log_softmax(lg, -1)
                    p = lp.exp()
                    act_lg = lg[A0:]
                    act_lp = torch.log_softmax(act_lg, -1)
                    others = act_lg.clone(); others[gt0] = -1e9
                    rec_c.update({
                        "first_action": act[0] - A0 if act else None,
                        "first_entropy_full": float(-(p * lp).sum()),
                        "first_entropy_action": float(-(act_lp.exp() * act_lp).sum()),
                        "first_top_prob_action": float(act_lp.exp().max()),
                        "gt_margin0": float(act_lg[gt0] - others.max()),
                        "gt_prob0": float(act_lp[gt0].exp()),
                        "gt_rank0": int((act_lg > act_lg[gt0]).sum()),
                        "first_is_generated_argmax": int(act_lg.argmax()) == (act[0] - A0 if act else -1),
                    })
                    store[ci] = act_lg.cpu().numpy().astype(np.float16)
                if len(act) == N_ACT:
                    tr = model.action_tokenizer.decode_token_ids_to_trajectory(torch.tensor(act))
                    rec_c["trajectory_pred"] = np.asarray(tr[0, 1:], np.float32)[:, :2].tolist()
                if name == "cot":
                    rec_c["text"] = tok.decode(full_gen, skip_special_tokens=False)
                elif name == "natural" and rec_c["cot_present"]:
                    rec_c["text"] = tok.decode(full_gen, skip_special_tokens=False)
                conds[name] = rec_c
                del g
            np.savez_compressed(os.path.join(args.output, "logits", f"{token}.npz"),
                                first_action_logits=store, conditions=np.array(CONDS))
            fout.write(json.dumps({"token": token, **{k: m[k] for k in ("log_name", "map_name", "instruction",
                                   "gt_action_instruction", "velocity", "gt_action_idx")},
                                   "trajectory_gt": [p[:2] for p in m["trajectory_gt"]],
                                   "stored_armN_action_idx": m["stored_armN_action_idx"],
                                   "stored_armC_action_idx": m["stored_armC_action_idx"],
                                   "conditions": conds}) + "\n")
            fout.flush()
            n_ok += 1
            if args.sanity:
                for name in CONDS:
                    c = conds[name]
                    print(f"  [{token}] {name:7s} gen={c['n_generated']:4d} valid={c['valid']} cot={c['cot_present']} "
                          f"think={c['think_tokens']:4d} first={c.get('first_action')} H={c.get('first_entropy_action', float('nan')):.3f} "
                          f"gtm={c.get('gt_margin0', float('nan')):+.2f} trunc={c['truncated']} runaway={c['runaway']}", flush=True)
                print(f"     stored natural first={m['stored_armN_action_idx'][:1]} stored cot first={m['stored_armC_action_idx'][:1]}")
        except Exception as exc:
            n_fail += 1
            import traceback; traceback.print_exc()
            print(f"[fail] {token}: {exc}", flush=True)
        if i % 25 == 0 or args.sanity:
            el = time.time() - t0
            print(f"[prog] {i}/{len(tokens)} ok={n_ok} fail={n_fail} {el/i:.1f}s/scene eta {(len(tokens)-i)*el/i/60:.1f}min", flush=True)
    fout.close()
    json.dump({"n_ok": n_ok, "n_fail": n_fail, "conditions": CONDS, "gen_conf": gc, "seed": args.seed,
               "fast_stub": FAST_STUB, "cot_stem": COT_STEM + " complex", "gpu": torch.cuda.get_device_name(0),
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "elapsed_min": (time.time() - t0) / 60}, open(os.path.join(args.output, "run_meta.json"), "w"), indent=2)
    print(f"[done] ok={n_ok} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
