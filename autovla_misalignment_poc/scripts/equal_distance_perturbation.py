#!/usr/bin/env python
"""
Equal-distance action-token perturbations at the first mismatch (GPU 1, natural/fast only).

Question: do geometrically similar initial deviations lead to different autoregressive
outcomes?

Scenes (full_extract arm N, natural fast path; groups fixed from the stored run)
  A-  first token mismatch AND trajectory failure         -> all of them
  A+  first token mismatch AND trajectory fine            -> 3 per A- scene, same first-mismatch step

Codebook geometry
  disp[a] = mean of the last sub-step's corners of token a = the (forward, lateral[+left])
  displacement the token contributes over its 0.5 s segment. d(a, b) = ||disp[a] - disp[b]||.

At the first mismatch step t*, with the prefix identical to the stored run, the forced token is
  original          the model's own predicted token
  original_reseed   the same token, different sampling seed      (decoding-noise floor)
  gt                the GT token
  alt_*             tokens with |d(GT, alt) - d(GT, pred)| <= max(5 mm, rel * d(GT, pred)),
                    rel = 0.10, widened to 0.20 / 0.35 only if fewer than 5 directions exist;
                    one per 45-degree direction sector of (disp[alt] - disp[GT]), the best
                    distance match in each sector. NOT nearest neighbours: matched to the
                    original error's distance.
then the remaining action tokens are generated freely (AutoVLA decoding config, one seed per
scene shared by all conditions except original_reseed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
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
PRA_DIR = "/root/VLA/pra_comparison"
sys.path.insert(0, AUTOVLA_DIR)
sys.path.insert(0, os.path.join(AUTOVLA_DIR, "navsim"))
sys.path.insert(0, POC_DIR)
sys.path.insert(0, PRA_DIR)

import pra_labels as L                                  # noqa: E402
from run_pra_comparison import _pos_to_delta            # noqa: E402

N_ACT = 10
CFG = L.Config()
SECTORS = ["faster", "faster-left", "left", "slower-left", "slower", "slower-right", "right", "faster-right"]
REL_LEVELS = (0.10, 0.20, 0.35)
ABS_TOL = 0.005
MIN_ALTS = 5


def a_label(traj, gt):
    s = L.Sample(model="", sample_id="", cluster="", scene=L.Scene(), desc_text="", desc_objs=[], rsn_text="",
                 decl_lon=None, decl_lat=None, pred_delta=_pos_to_delta(traj, 10), gt_delta=_pos_to_delta(gt, 10))
    return bool(L.label_A(s, CFG)[0])


def sector_of(v) -> int:
    ang = math.atan2(float(v[1]), float(v[0]))
    return int(((ang + math.pi / 8) % (2 * math.pi)) // (math.pi / 4)) % 8


def select_alternatives(disp, cb, gt, pred):
    d_all = np.linalg.norm(disp - disp[gt], axis=1)
    d_pred = float(d_all[pred])
    vec = disp - disp[gt]
    chosen, used_rel, tol = {}, None, None
    for rel in REL_LEVELS:
        tol = max(ABS_TOL, rel * d_pred)
        idx = [int(a) for a in np.where(np.abs(d_all - d_pred) <= tol)[0] if a != gt and a != pred]
        by_sec = {}
        for a in idx:
            s = sector_of(vec[a])
            key = (abs(d_all[a] - d_pred), a)
            if s not in by_sec or key < by_sec[s][0]:
                by_sec[s] = (key, a)
        chosen, used_rel = {s: v[1] for s, v in by_sec.items()}, rel
        if len(chosen) >= MIN_ALTS:
            break
    alts = [chosen[s] for s in sorted(chosen)]
    if len(alts) < MIN_ALTS:
        # The codebook is sparse near some GT tokens, so even the widest tolerance can leave
        # fewer than MIN_ALTS. Fill from the WHOLE codebook ranked by distance match
        # |d(GT, a) - d(GT, pred)| -- still distance-matched, never nearest-to-GT -- taking
        # uncovered directions first. `matched_strict` marks which picks met the tolerance.
        covered = {sector_of(vec[a]) for a in alts}
        ranked = [int(a) for a in np.argsort(np.abs(d_all - d_pred), kind="stable")
                  if a != gt and a != pred and a not in alts]
        for a in ranked:
            if len(alts) >= MIN_ALTS:
                break
            if sector_of(vec[a]) not in covered:
                alts.append(a); covered.add(sector_of(vec[a]))
        for a in ranked:
            if len(alts) >= MIN_ALTS:
                break
            if a not in alts:
                alts.append(a)
    pred_sector = sector_of(vec[pred])
    strict_tol = max(ABS_TOL, REL_LEVELS[-1] * d_pred)
    out = []
    for a in alts:
        s = sector_of(vec[a])
        out.append({"token": a, "d_gt": float(d_all[a]), "rel_err": float(abs(d_all[a] - d_pred) / max(d_pred, 1e-6)),
                    "matched_strict": bool(abs(d_all[a] - d_pred) <= strict_tol),
                    "sector": SECTORS[s], "same_sector_as_pred": s == pred_sector,
                    "vec_forward": float(vec[a, 0]), "vec_left": float(vec[a, 1]),
                    "shape_d_gt": float(np.linalg.norm(cb[a] - cb[gt], axis=-1).mean()),
                    "shape_d_pred": float(np.linalg.norm(cb[a] - cb[pred], axis=-1).mean()),
                    "d_pred_token": float(np.linalg.norm(disp[a] - disp[pred]))})
    return out, {"d_pred": d_pred, "tol_m": tol, "rel_used": used_rel, "pred_sector": SECTORS[pred_sector],
                 "pred_vec_forward": float(vec[pred, 0]), "pred_vec_left": float(vec[pred, 1]),
                 "pred_shape_d_gt": float(np.linalg.norm(cb[pred] - cb[gt], axis=-1).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(POC_DIR, "outputs/full_extract/records.jsonl"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/equal_distance_perturbation"))
    ap.add_argument("--config", default=os.path.join(AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--aplus-per-aminus", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" and not os.environ.get("VLA_ANY_GPU"):
        raise SystemExit("pinned to GPU 1: launch with CUDA_VISIBLE_DEVICES=1")
    for k in ("records", "scenes", "output", "config"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.makedirs(args.output, exist_ok=True)

    cb = np.asarray(pickle.load(open(os.path.join(AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl"), "rb"))["token_all"]["veh"], np.float32)
    disp = cb[:, -1].mean(1)

    # ---- groups from the stored natural run ---------------------------------------------
    neg, pos_by_t = [], defaultdict(list)
    for line in open(args.records):
        r = json.loads(line)
        n = r["arms"]["N"]
        if n["cot_present"] or n["runaway_action_tokens"] or len(n["trajectory_pred"]) < N_ACT:
            continue
        pred, gt = n["pred_action_idx"][:N_ACT], r["gt_action_idx"][:N_ACT]
        t = next((k for k in range(N_ACT) if pred[k] != gt[k]), None)
        if t is None:
            continue
        item = {"token": r["token"], "t_star": t, "log": r["log_name"], "pred": pred, "gt": gt, "n_ids": n["token_ids"],
                "velocity": r["velocity"], "trajectory_gt": [p[:2] for p in r["trajectory_gt"]], "map_name": r["map_name"]}
        if a_label(n["trajectory_pred"], r["trajectory_gt"]):
            item["group"] = "A+"; pos_by_t[t].append(item)
        else:
            item["group"] = "A-"; neg.append(item)
    rng = random.Random(args.seed)
    picks = sorted(neg, key=lambda x: x["token"])
    need = defaultdict(int)
    for x in neg:
        need[x["t_star"]] += args.aplus_per_aminus
    for t, k in sorted(need.items()):
        pool = sorted(pos_by_t[t], key=lambda x: x["token"])
        rng.shuffle(pool)
        picks += pool[:k]
    if args.limit:
        a_m = [p for p in picks if p["group"] == "A-"][:max(1, args.limit // 2)]
        a_p = [p for p in picks if p["group"] == "A+"][:args.limit - len(a_m)]
        picks = a_m + a_p
    print(f"[plan] {len(picks)} scenes (A- {sum(p['group']=='A-' for p in picks)}, A+ {sum(p['group']=='A+' for p in picks)})", flush=True)

    os.chdir(AUTOVLA_DIR)
    from models.autovla import AutoVLA
    from navsim.agents.autovla_agent import AutoVLAAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["pretrained_model_path"] = os.path.join(AUTOVLA_DIR, "Qwen2.5-VL-3B-Instruct")
    cfg["model"]["codebook_cache_path"] = os.path.join(AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl")
    gc = cfg["inference"]["sample"]
    model = AutoVLA(cfg, inference=True, device="cuda:0")
    sd = torch.load(args.checkpoint, map_location="cpu", mmap=True)["state_dict"]
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
            t, gt, pred = p["t_star"], p["gt"], p["pred"]
            alts, geo = select_alternatives(disp, cb, gt[t], pred[t])
            base_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{p['token']}".encode()).digest()[:4], "little")
            conds = [("original", pred[t], base_seed), ("original_reseed", pred[t], base_seed + 7919), ("gt", gt[t], base_seed)]
            conds += [(f"alt{j}", a["token"], base_seed) for j, a in enumerate(alts)]
            remaining = N_ACT - t - 1
            out = {}
            for name, tok_forced, seed in conds:
                head = stub + [A0 + a for a in pred[:t]] + [A0 + tok_forced]
                ids = torch.tensor([prompt_ids + head], device="cuda:0")
                m2 = dict(mi); m2["input_ids"] = ids; m2["attention_mask"] = torch.ones_like(ids)
                acts = list(pred[:t]) + [tok_forced]
                ents, gtlp = [], []
                if remaining > 0:
                    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                    with torch.no_grad():
                        g = llm.generate(**m2, max_new_tokens=remaining + 4, do_sample=True, temperature=gc["temperature"],
                                         top_k=gc["top_k"], top_p=gc["top_p"], return_dict_in_generate=True, output_logits=True)
                    new = g.sequences[0][ids.shape[1]:].tolist()
                    for j, x in enumerate(new):
                        if len(acts) >= N_ACT or x < A0:
                            break
                        lg = g.logits[j][0, A0:].float()
                        lp = torch.log_softmax(lg, -1)
                        ents.append(float(-(lp.exp() * lp).sum()))
                        gtlp.append(float(lp[gt[len(acts)]]))
                        acts.append(x - A0)
                    del g
                rec = {"forced_token": tok_forced, "action_idx": acts, "valid": len(acts) == N_ACT,
                       "entropy_steps": ents, "gt_logprob_steps": gtlp}
                if rec["valid"]:
                    tr = model.action_tokenizer.decode_token_ids_to_trajectory(torch.tensor([A0 + a for a in acts]))
                    rec["trajectory_pred"] = np.asarray(tr[0, 1:], np.float32)[:, :2].tolist()
                out[name] = rec
            keep = {k: p[k] for k in ("token", "group", "t_star", "log", "pred", "gt", "velocity", "trajectory_gt", "map_name")}
            fout.write(json.dumps(keep | {"geometry": geo, "alternatives": alts, "conditions": out}) + "\n"); fout.flush()
            n_ok += 1
            if args.sanity:
                print(f"  [{p['token']} {p['group']} t*={t}] d_pred={geo['d_pred']:.3f} m tol={geo['tol_m']:.3f} (rel {geo['rel_used']}) "
                      f"pred sector={geo['pred_sector']} n_alt={len(alts)}", flush=True)
                for j, a in enumerate(alts):
                    print(f"     alt{j}: tok={a['token']} d_gt={a['d_gt']:.3f} rel_err={a['rel_err']:.2f} {a['sector']:12s} "
                          f"same_as_pred={a['same_sector_as_pred']} acts={out[f'alt{j}']['action_idx']}", flush=True)
                for name in ("original", "original_reseed", "gt"):
                    print(f"     {name:15s} acts={out[name]['action_idx']}", flush=True)
        except Exception as exc:
            n_fail += 1
            import traceback; traceback.print_exc()
            print(f"[fail] {p['token']}: {exc}", flush=True)
        if i % 20 == 0 or args.sanity:
            el = time.time() - t0
            print(f"[prog] {i}/{len(picks)} ok={n_ok} fail={n_fail} {el/i:.1f}s/scene eta {(len(picks)-i)*el/i/60:.1f}min", flush=True)
    fout.close()
    json.dump({"n_ok": n_ok, "n_fail": n_fail, "gen_conf": gc, "seed": args.seed, "sectors": SECTORS,
               "rel_levels": REL_LEVELS, "abs_tol_m": ABS_TOL, "min_alts": MIN_ALTS,
               "aplus_per_aminus": args.aplus_per_aminus, "gpu": torch.cuda.get_device_name(0),
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "elapsed_min": (time.time() - t0) / 60},
              open(os.path.join(args.output, "run_meta.json"), "w"), indent=2)
    print(f"[done] ok={n_ok} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
