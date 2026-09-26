#!/usr/bin/env python
"""
Temporal feedback window (GPU 1, natural/fast only).

Question: for how many steps after the first mismatch must the previous-action feedback be
corrected before the rollout keeps itself on track?

Units: the equal-distance perturbation set exactly as in action_history_causal /
prev_action_state_patching: A- = first mismatch AND trajectory failure (52 scenes), A+ = first
mismatch AND trajectory fine, 3 per A- scene at the SAME first-mismatch step (156 scenes);
unit = (scene, perturbation) with the forced token at t*. No new perturbations.

Harness (unchanged): prefix KV cache = prompt + fast stub + model's own tokens before t*; at
every step k the post-t* span [a_t*, c_{t*+1}, ..., c_{k-1}] is recomputed and the logits at its
last position give a_k (action rows only, T = 0.01, seed sha256(f"{seed}:{token}:{pert}:{k}"),
identical for all rows).

CORRECTION WINDOW (same definition as the GT-history condition of action_history_causal)
  Context position j = t* + o (o >= 1) holds  gt[j]            if start <= o < start + length
                                              model's own a_j  otherwise.
  The EXECUTED action at every step is the model's own sampled token (the trajectory is never
  teacher-forced); only what the model conditions on is corrected. After the window the model
  conditions on its own outputs again (free autoregression).
  length 0 = Normal AR; start 1, length >= 8 - t* = Full GT-history.
  Note a_{t*+1} is produced from context [a_t*] only, so no window can change it; the first step
  a window can change is a_{t*+2} (via context position t*+1).

ROWS (one batched forward per step)
  normal                      length 0
  win{w}   w = 1..8           start 1, length w          (w-step correction t*+1 .. t*+w)
  gt_history                  start 1, length 99         (must equal win8 and every win{w>=8-t*})
  delay{s}_len{l}             start s in {2,3,4}, length 1; start s in {2,3}, length 2
  skipfirst{s}_full           start s in {2,3}, length 99  (all correction except the first s-1)
"""
from __future__ import annotations

import argparse
import hashlib
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

N_ACT = 10


def build_rows():
    rows = [{"name": "normal", "start": 1, "length": 0}]
    rows += [{"name": f"win{w}", "start": 1, "length": w} for w in range(1, 9)]
    rows.append({"name": "gt_history", "start": 1, "length": 99})
    rows += [{"name": f"delay{s}_len1", "start": s, "length": 1} for s in (2, 3, 4)]
    rows += [{"name": f"delay{s}_len2", "start": s, "length": 2} for s in (2, 3)]
    rows += [{"name": f"skipfirst{s}_full", "start": s, "length": 99} for s in (2, 3)]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ed-records", default=os.path.join(POC_DIR, "outputs/equal_distance_perturbation/records.jsonl"))
    ap.add_argument("--ah-records", default=os.path.join(POC_DIR, "outputs/action_history_causal/records.jsonl"))
    ap.add_argument("--full-records", default=os.path.join(POC_DIR, "outputs/full_extract/records.jsonl"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/temporal_feedback_window"))
    ap.add_argument("--config", default=os.path.join(AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-alts", type=int, default=0)
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("pinned to GPU 1: launch with CUDA_VISIBLE_DEVICES=1")
    for k in ("ed_records", "ah_records", "full_records", "scenes", "output", "config"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.makedirs(args.output, exist_ok=True)
    if os.path.exists(os.path.join(args.output, "records.jsonl")):
        raise SystemExit(f"{args.output}/records.jsonl exists; refusing to overwrite")

    ROWS = build_rows()
    B = len(ROWS)
    idx = {r["name"]: i for i, r in enumerate(ROWS)}

    ed = [json.loads(l) for l in open(args.ed_records)]
    ed.sort(key=lambda r: (r["group"] != "A-", r["token"]))
    if args.limit:
        a_m = [r for r in ed if r["group"] == "A-"][:max(1, args.limit // 2)]
        a_p = [r for r in ed if r["group"] == "A+"][:args.limit - len(a_m)]
        ed = a_m + a_p
    want = {r["token"] for r in ed}
    stub_of = {}
    for line in open(args.full_records):
        x = json.loads(line)
        if x["token"] in want:
            stub_of[x["token"]] = x["arms"]["N"]["token_ids"]
    prior = {}
    for line in open(args.ah_records):
        x = json.loads(line)
        if x["token"] in want:
            for pname, pr in x["perturbations"].items():
                prior[(x["token"], pname)] = {c: pr["conditions"][c]["action_idx"] for c in ("normal", "gt_history")}
    print(f"[plan] {len(ed)} scenes (A- {sum(r['group']=='A-' for r in ed)}, A+ {sum(r['group']=='A+' for r in ed)}), "
          f"{B} rows per batched step", flush=True)

    os.chdir(AUTOVLA_DIR)
    from models.autovla import AutoVLA
    from navsim.agents.autovla_agent import AutoVLAAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["pretrained_model_path"] = os.path.join(AUTOVLA_DIR, "Qwen2.5-VL-3B-Instruct")
    cfg["model"]["codebook_cache_path"] = os.path.join(AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl")
    temp = float(cfg["inference"]["sample"]["temperature"])
    model = AutoVLA(cfg, inference=True, device="cuda:0")
    sd = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    model.load_state_dict({k[len("autovla."):]: v for k, v in sd.items() if k.startswith("autovla.")}, strict=False)
    model.to("cuda:0").eval()
    del sd
    llm, A0 = model.vlm, model.action_start_id

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

    fout = open(os.path.join(args.output, "records.jsonl"), "w")
    ferr = open(os.path.join(args.output, "errors.jsonl"), "w")
    t0 = time.time()
    n_ok = n_fail = n_units = n_unit_fail = 0
    for si, r in enumerate(ed, 1):
        try:
            token, t, gt, pred = r["token"], r["t_star"], r["gt"], r["pred"]
            scene = json.load(open(os.path.join(args.scenes, f"{token}.json")))
            vis_box.clear()
            feats = {}
            for b in agent.get_feature_builders():
                feats.update(b.compute_features(scene))
            mi = {k: v.to("cuda:0") for k, v in model.get_prompt(feats).items() if isinstance(v, torch.Tensor)}
            n_ids = stub_of[token]
            stub = n_ids[:next(j for j, x in enumerate(n_ids) if x >= A0)]
            prefix = mi["input_ids"][0].tolist() + stub + [A0 + a for a in pred[:t]]
            P = len(prefix)
            ids = torch.tensor([prefix], device="cuda:0")
            llm.rope_deltas = None
            with torch.no_grad():
                out = llm(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values_videos=mi.get("pixel_values_videos"),
                          video_grid_thw=mi.get("video_grid_thw"), use_cache=True, cache_position=torch.arange(P, device="cuda:0"))
            cache = out.past_key_values
            del out
            for li in range(len(cache.key_cache)):
                cache.key_cache[li] = cache.key_cache[li].expand(B, -1, -1, -1).contiguous()
                cache.value_cache[li] = cache.value_cache[li].expand(B, -1, -1, -1).contiguous()
            prefix_checksum = float(cache.key_cache[0][0, :, :P].float().abs().sum())
        except Exception as exc:
            n_fail += 1
            import traceback; traceback.print_exc()
            ferr.write(json.dumps({"token": r.get("token"), "stage": "scene", "error": repr(exc)}) + "\n"); ferr.flush()
            continue

        perts = [("original", pred[t])] + [(f"alt{j}", a["token"]) for j, a in enumerate(r["alternatives"])]
        if args.max_alts:
            perts = perts[:1 + args.max_alts]
        for pname, ptok in perts:
            try:
                choices = [[] for _ in range(B)]
                ents = [[] for _ in range(B)]
                pgts = [[] for _ in range(B)]
                ctx_gt_mask = [[] for _ in range(B)]          # per step: which context positions were GT
                for k in range(t + 1, N_ACT):
                    spans = []
                    for i, row in enumerate(ROWS):
                        ctx, mask = [], []
                        for j in range(t + 1, k):
                            o = j - t
                            use_gt = row["start"] <= o < row["start"] + row["length"]
                            ctx.append(gt[j] if use_gt else choices[i][o - 1])
                            mask.append(use_gt)
                        spans.append([A0 + ptok] + [A0 + a for a in ctx])
                        ctx_gt_mask[i].append(mask)
                    m = len(spans[0])
                    sids = torch.tensor(spans, device="cuda:0")
                    with torch.no_grad():
                        o_ = llm(input_ids=sids, attention_mask=torch.ones((B, P + m), device="cuda:0", dtype=torch.long),
                                 past_key_values=cache, use_cache=True, cache_position=torch.arange(P, P + m, device="cuda:0"))
                    cache.crop(P)
                    lg = o_.logits[:, -1, A0:A0 + 2048].double()
                    del o_
                    lp = torch.log_softmax(lg, -1)
                    ent = -(lp.exp() * lp).sum(-1)
                    seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{token}:{pname}:{k}".encode()).digest()[:4], "little")
                    probs_t = torch.softmax(lg / temp, -1).cpu()
                    gen = torch.Generator()
                    for i in range(B):
                        gen.manual_seed(seed)
                        choices[i].append(int(torch.multinomial(probs_t[i], 1, generator=gen)))
                        ents[i].append(round(float(ent[i]), 5))
                        pgts[i].append(round(float(lp[i, gt[k]].exp()), 6))
                chk = float(cache.key_cache[0][0, :, :P].float().abs().sum())
                assert abs(chk - prefix_checksum) < 1e-3 * max(1.0, prefix_checksum), "prefix cache was modified"
                assert cache.get_seq_length() == P, "cache not cropped back to the prefix"

                rows_out = {}
                for name, i in idx.items():
                    acts = list(pred[:t]) + [ptok] + choices[i]
                    tr = model.action_tokenizer.decode_token_ids_to_trajectory(torch.tensor([A0 + a for a in acts]))
                    rows_out[name] = {"action_idx": acts, "trajectory_pred": np.asarray(tr[0, 1:], np.float32)[:, :2].tolist(),
                                      "ent": ents[i], "p_gt": pgts[i],
                                      "n_gt_context_positions": int(sum(ctx_gt_mask[i][-1])),
                                      "gt_context_offsets": [o + 1 for o, g_ in enumerate(ctx_gt_mask[i][-1]) if g_]}
                pri = prior.get((token, pname))
                rec = {"token": token, "group": r["group"], "t_star": t, "log": r["log"], "perturbation": pname,
                       "forced_token": ptok, "gt": gt, "pred_prefix": pred[:t], "trajectory_gt": r["trajectory_gt"],
                       "rows": rows_out, "prior_action_history": pri}
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                n_units += 1
            except Exception as exc:
                n_unit_fail += 1
                import traceback; traceback.print_exc()
                ferr.write(json.dumps({"token": r["token"], "perturbation": pname, "stage": "unit", "error": repr(exc)}) + "\n"); ferr.flush()
        del cache
        n_ok += 1
        if si % 10 == 0 or args.limit:
            el = time.time() - t0
            print(f"[prog] {si}/{len(ed)} scenes ok={n_ok} fail={n_fail} units={n_units} unit_fail={n_unit_fail} "
                  f"{el/si:.1f}s/scene eta {(len(ed)-si)*el/si/60:.1f}min", flush=True)
    fout.close(); ferr.close()
    json.dump({"n_scenes_ok": n_ok, "n_scene_fail": n_fail, "n_units": n_units, "n_unit_fail": n_unit_fail,
               "rows": ROWS, "temperature": temp, "seed": args.seed, "gpu": torch.cuda.get_device_name(0),
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "elapsed_min": (time.time() - t0) / 60},
              open(os.path.join(args.output, "run_meta.json"), "w"), indent=2)
    print(f"[done] scenes ok={n_ok} fail={n_fail} units={n_units} unit_fail={n_unit_fail}", flush=True)


if __name__ == "__main__":
    main()
