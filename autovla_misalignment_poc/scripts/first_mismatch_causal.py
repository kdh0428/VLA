#!/usr/bin/env python
"""
Does a small first action-token mismatch cause the trajectory failure? (GPU 1, natural/fast only)

Scenes: full_extract arm N (natural fast path), the ones whose generated action tokens differ
from GT somewhere. Groups come from the STORED arm N run, fixed before any intervention:
  A-  first token mismatch exists AND the trajectory fails  (P/R/A accept rule, 5 s)
  A+  first token mismatch exists AND the trajectory is fine

At the first mismatch step t*, with the prefix identical to the stored run
(prompt + the model's own fast stub + the model's own action tokens before t*):
  original  write the model's own predicted token, then free generation
  gt        write the GT token, then free generation
  nn        write GT's nearest codebook neighbour (by segment displacement), excluding GT and
            the original token, then free generation -- a different deviation of similar size

Free generation uses AutoVLA's decoding config (do_sample, T=0.01, top_k=0, top_p=1) with one
seed per scene shared by the three conditions. All three are generated in this harness, so they
are compared with each other, not with the stored run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

AUTOVLA_DIR = "/root/VLA/autovla"
POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRA_DIR = "/root/VLA/pra_comparison"
sys.path.insert(0, AUTOVLA_DIR)
sys.path.insert(0, os.path.join(AUTOVLA_DIR, "navsim"))
sys.path.insert(0, POC_DIR)
sys.path.insert(0, PRA_DIR)

import pra_labels as L                                  # noqa: E402
from run_pra_comparison import _pos_to_delta            # noqa: E402

N_ACT = 10
CONDS = ["original", "gt", "nn"]
CFG = L.Config()


def a_label(traj, gt):
    s = L.Sample(model="", sample_id="", cluster="", scene=L.Scene(), desc_text="", desc_objs=[], rsn_text="",
                 decl_lon=None, decl_lat=None, pred_delta=_pos_to_delta(traj, 10), gt_delta=_pos_to_delta(gt, 10))
    return bool(L.label_A(s, CFG)[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(POC_DIR, "outputs/full_extract/records.jsonl"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/first_mismatch_causal"))
    ap.add_argument("--config", default=os.path.join(AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("pinned to GPU 1: launch with CUDA_VISIBLE_DEVICES=1")
    for k in ("records", "scenes", "output", "config"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.makedirs(os.path.join(args.output, "logits"), exist_ok=True)

    cb = np.asarray(pickle.load(open(os.path.join(AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl"), "rb"))["token_all"]["veh"], np.float32)
    disp = cb[:, -1].mean(1)

    # ---- group assignment from the stored natural run ------------------------------------
    picks = []
    for line in open(args.records):
        r = json.loads(line)
        n = r["arms"]["N"]
        if n["cot_present"] or n["runaway_action_tokens"] or len(n["trajectory_pred"]) < N_ACT:
            continue
        pred, gt = n["pred_action_idx"][:N_ACT], r["gt_action_idx"][:N_ACT]
        t = next((k for k in range(N_ACT) if pred[k] != gt[k]), None)
        if t is None:
            continue
        A = a_label(n["trajectory_pred"], r["trajectory_gt"])
        d_gt = np.linalg.norm(disp - disp[gt[t]], axis=1)
        order = [int(i) for i in np.argsort(d_gt, kind="stable") if i != gt[t] and i != pred[t]]
        picks.append({"token": r["token"], "group": "A+" if A else "A-", "t_star": t, "log": r["log_name"],
                      "pred": pred, "gt": gt, "nn_token": order[0],
                      "d_pred_gt": float(d_gt[pred[t]]), "d_nn_gt": float(d_gt[order[0]]),
                      "nn_rank_pred": int(np.argsort(np.argsort(d_gt, kind="stable"), kind="stable")[pred[t]]),
                      "n_ids": n["token_ids"], "velocity": r["velocity"], "trajectory_gt": [p[:2] for p in r["trajectory_gt"]],
                      "stored_traj": n["trajectory_pred"], "map_name": r["map_name"],
                      "gt_action_instruction": r["gt_action_instruction"], "instruction": r["instruction"]})
    # A- first so a limited sanity run sees both groups
    picks.sort(key=lambda p: (p["group"] != "A-", p["token"]))
    if args.limit:
        picks = picks[:max(1, args.limit // 2)] + [p for p in picks if p["group"] == "A+"][:args.limit - max(1, args.limit // 2)]
    print(f"[plan] {len(picks)} scenes (A- {sum(p['group']=='A-' for p in picks)}, A+ {sum(p['group']=='A+' for p in picks)}) x {len(CONDS)}", flush=True)

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
    t0 = time.time()
    n_ok = n_fail = 0
    for i, p in enumerate(picks, 1):
        try:
            scene = json.load(open(os.path.join(args.scenes, f"{p['token']}.json")))
            vis_box.clear()
            feats = {}
            for b in agent.get_feature_builders():
                feats.update(b.compute_features(scene))
            mi = {k: v.to("cuda:0") for k, v in model.get_prompt(feats).items() if isinstance(v, torch.Tensor)}
            prompt_ids = mi["input_ids"][0].tolist()
            fa = next(j for j, x in enumerate(p["n_ids"]) if x >= A0)
            stub = p["n_ids"][:fa]
            t = p["t_star"]
            before = [A0 + a for a in p["pred"][:t]]
            forced = {"original": p["pred"][t], "gt": p["gt"][t], "nn": p["nn_token"]}
            seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{p['token']}".encode()).digest()[:4], "little")
            remaining = N_ACT - t - 1
            store = np.full((len(CONDS), N_ACT, 2048), np.nan, np.float16)
            out = {}
            for ci, name in enumerate(CONDS):
                head = stub + before + [A0 + forced[name]]
                ids = torch.tensor([prompt_ids + head], device="cuda:0")
                m2 = dict(mi); m2["input_ids"] = ids; m2["attention_mask"] = torch.ones_like(ids)
                acts = [p["pred"][k] for k in range(t)] + [forced[name]]
                gen_logits = []
                if remaining > 0:
                    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                    with torch.no_grad():
                        g = llm.generate(**m2, max_new_tokens=remaining + 4, do_sample=True,
                                         temperature=gc["temperature"], top_k=gc["top_k"], top_p=gc["top_p"],
                                         return_dict_in_generate=True, output_logits=True)
                    new = g.sequences[0][ids.shape[1]:].tolist()
                    for j, x in enumerate(new):
                        if len(acts) >= N_ACT:
                            break
                        if x < A0:
                            break                              # left the action span early
                        acts.append(x - A0)
                        store[ci, len(acts) - 1] = g.logits[j][0, A0:].float().cpu().numpy().astype(np.float16)
                    del g
                valid = len(acts) == N_ACT
                rec = {"forced_token": forced[name], "action_idx": acts, "valid": valid}
                if valid:
                    tr = model.action_tokenizer.decode_token_ids_to_trajectory(torch.tensor([A0 + a for a in acts]))
                    rec["trajectory_pred"] = np.asarray(tr[0, 1:], np.float32)[:, :2].tolist()
                out[name] = rec
            np.savez_compressed(os.path.join(args.output, "logits", f"{p['token']}.npz"),
                                step_action_logits=store, conditions=np.array(CONDS))
            keep = {k: p[k] for k in ("token", "group", "t_star", "log", "pred", "gt", "nn_token", "d_pred_gt", "d_nn_gt",
                                      "nn_rank_pred", "velocity", "trajectory_gt", "stored_traj", "map_name",
                                      "gt_action_instruction", "instruction")}
            fout.write(json.dumps(keep | {"conditions": out}) + "\n"); fout.flush()
            n_ok += 1
            if args.sanity:
                o = out["original"]
                print(f"  [{p['token']} {p['group']} t*={t}] stored pred={p['pred']}\n     gt={p['gt']}  nn={p['nn_token']} "
                      f"(d_pred {p['d_pred_gt']:.3f} m, d_nn {p['d_nn_gt']:.3f} m, pred NN-rank {p['nn_rank_pred']})", flush=True)
                for name in CONDS:
                    print(f"     {name:8s} valid={out[name]['valid']} acts={out[name]['action_idx']}", flush=True)
                print(f"     original reproduces stored tokens: {o['action_idx'] == p['pred']}", flush=True)
        except Exception as exc:
            n_fail += 1
            import traceback; traceback.print_exc()
            print(f"[fail] {p['token']}: {exc}", flush=True)
        if i % 50 == 0 or args.sanity:
            el = time.time() - t0
            print(f"[prog] {i}/{len(picks)} ok={n_ok} fail={n_fail} {el/i:.2f}s/scene eta {(len(picks)-i)*el/i/60:.1f}min", flush=True)
    fout.close()
    json.dump({"n_ok": n_ok, "n_fail": n_fail, "conditions": CONDS, "gen_conf": gc, "seed": args.seed,
               "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "elapsed_min": (time.time() - t0) / 60}, open(os.path.join(args.output, "run_meta.json"), "w"), indent=2)
    print(f"[done] ok={n_ok} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
