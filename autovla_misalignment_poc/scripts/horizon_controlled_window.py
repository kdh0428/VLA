#!/usr/bin/env python
"""
Horizon-controlled correction window (GPU 1, natural/fast only).

Question: is the 3-4 step critical window of temporal_feedback_window a real stabilization
window, or an artefact of the 5-s horizon (longer corrections leave fewer free steps)?

Units: equal-distance perturbation set restricted to early first mismatches (t* in {0, 1}):
A- (first mismatch AND failure) and step-matched A+; unit = (scene, perturbation). Scenes need
>= 20 future GT frames in the nuPlan/navsim log (10 s); others are logged to errors.jsonl.

Extended horizon: the rollout generates H = 20 action tokens (10 s) instead of 10. Extended
GT poses come from the navsim log (ego2global of the following frames, transformed into the
current ego frame; first 10 poses match the scene JSON to < 4e-6 m) and are tokenized with
AutoVLA's own target builder (sequential codebook matching; first 10 tokens == stored GT).

Harness as temporal_feedback_window (prefix KV cache, post-t* span recomputed each step, action
rows only, T = 0.01, seed sha256(f"{seed}:{token}:{pert}:{k}") shared by all rows), so the first
10 tokens of every row reproduce the 10-token experiment.

Correction: context position t*+o is GT iff 1 <= o <= length; executed actions are always the
model's own samples. After the window, free autoregression.
  normal (length 0), win1..win6, win8, full (every context position = GT, through step 19).
Release: a window of length w last conditions a_{t*+w+1} on GT; the first output conditioned on a
self-generated token after the window is a_{t*+w+2}. The analysis evaluates every window over
exactly N free steps after release (poses t*+w+2 .. t*+w+1+N), so all windows get the same free
horizon; generating to 20 and truncating is identical to stopping early (causal decoding).

Also logged per step (to judge whether generating past the trained 10-token length is
in-distribution): full-vocabulary probability mass on action tokens and the top non-action token.
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

H = 20


def build_rows():
    rows = [{"name": "normal", "length": 0}]
    rows += [{"name": f"win{w}", "length": w} for w in (1, 2, 3, 4, 5, 6, 8)]
    rows.append({"name": "full", "length": 99})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ed-records", default=os.path.join(POC_DIR, "outputs/equal_distance_perturbation/records.jsonl"))
    ap.add_argument("--tfw-records", default=os.path.join(POC_DIR, "outputs/temporal_feedback_window/records.jsonl"))
    ap.add_argument("--full-records", default=os.path.join(POC_DIR, "outputs/full_extract/records.jsonl"))
    ap.add_argument("--ext-gt", default=os.path.join(POC_DIR, "outputs/horizon_controlled_window/extended_gt.json"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/horizon_controlled_window"))
    ap.add_argument("--config", default=os.path.join(AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tstar", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-alts", type=int, default=0)
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" and not os.environ.get("VLA_ANY_GPU"):
        raise SystemExit("pinned to GPU 1: launch with CUDA_VISIBLE_DEVICES=1")
    for k in ("ed_records", "tfw_records", "full_records", "ext_gt", "scenes", "output", "config"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.makedirs(args.output, exist_ok=True)
    if os.path.exists(os.path.join(args.output, "records.jsonl")):
        raise SystemExit(f"{args.output}/records.jsonl exists; refusing to overwrite")

    ROWS = build_rows()
    B = len(ROWS)
    idx = {r["name"]: i for i, r in enumerate(ROWS)}
    ext = json.load(open(args.ext_gt))

    ed = [json.loads(l) for l in open(args.ed_records)]
    ed = [r for r in ed if r["t_star"] <= args.max_tstar]
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
    if os.path.exists(args.tfw_records):
        for line in open(args.tfw_records):
            x = json.loads(line)
            if x["token"] in want:
                prior[(x["token"], x["perturbation"])] = {c: x["rows"][c]["action_idx"] for c in ("normal", "win1", "win2", "win3", "win4", "gt_history")}
    print(f"[plan] {len(ed)} scenes with t* <= {args.max_tstar} (A- {sum(r['group']=='A-' for r in ed)}, "
          f"A+ {sum(r['group']=='A+' for r in ed)}), H={H}, {B} rows per batched step", flush=True)

    os.chdir(AUTOVLA_DIR)
    from models.autovla import AutoVLA
    from navsim.agents.autovla_agent import AutoVLAAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["pretrained_model_path"] = os.path.join(AUTOVLA_DIR, "Qwen2.5-VL-3B-Instruct")
    cfg["model"]["codebook_cache_path"] = os.path.join(AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl")
    temp = float(cfg["inference"]["sample"]["temperature"])
    model = AutoVLA(cfg, inference=True, device="cuda:0")
    sd = torch.load(args.checkpoint, map_location="cpu", mmap=True)["state_dict"]
    model.load_state_dict({k[len("autovla."):]: v for k, v in sd.items() if k.startswith("autovla.")}, strict=False)
    model.to("cuda:0").eval()
    del sd
    llm, A0 = model.vlm, model.action_start_id
    tok = model.processor.tokenizer if hasattr(model, "processor") else None

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
            token, t, pred = r["token"], r["t_star"], r["pred"]
            e = ext.get(token) or {}
            gt = e.get("gt_tokens20")
            if gt is None or len(gt) < H:
                raise ValueError(f"insufficient future GT: {e.get('n_future')} frames < {H}")
            if gt[:10] != r["gt"]:
                raise ValueError("extended GT tokens disagree with stored GT tokens")
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
            ferr.write(json.dumps({"token": r.get("token"), "group": r.get("group"), "stage": "scene", "error": repr(exc)}) + "\n"); ferr.flush()
            print(f"[skip] {r.get('token')}: {exc!r}", flush=True)
            continue

        perts = [("original", pred[t])] + [(f"alt{j}", a["token"]) for j, a in enumerate(r["alternatives"])]
        if args.max_alts:
            perts = perts[:1 + args.max_alts]
        for pname, ptok in perts:
            try:
                choices = [[] for _ in range(B)]
                ents = [[] for _ in range(B)]
                pgts = [[] for _ in range(B)]
                pact = [[] for _ in range(B)]
                top_non = [[] for _ in range(B)]
                for k in range(t + 1, H):
                    spans = []
                    for i, row in enumerate(ROWS):
                        ctx = [gt[j] if 1 <= j - t <= row["length"] else choices[i][j - t - 1] for j in range(t + 1, k)]
                        spans.append([A0 + ptok] + [A0 + a for a in ctx])
                    m = len(spans[0])
                    sids = torch.tensor(spans, device="cuda:0")
                    with torch.no_grad():
                        o_ = llm(input_ids=sids, attention_mask=torch.ones((B, P + m), device="cuda:0", dtype=torch.long),
                                 past_key_values=cache, use_cache=True, cache_position=torch.arange(P, P + m, device="cuda:0"))
                    cache.crop(P)
                    full_lg = o_.logits[:, -1].double()
                    del o_
                    full_lp = torch.log_softmax(full_lg, -1)
                    mass = full_lp[:, A0:A0 + 2048].exp().sum(-1)
                    non = full_lp.clone(); non[:, A0:A0 + 2048] = -float("inf")
                    nv, ni = non.max(-1)
                    lg = full_lg[:, A0:A0 + 2048]
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
                        pact[i].append(round(float(mass[i]), 6))
                        top_non[i].append([int(ni[i]), round(float(nv[i].exp()), 6)])
                chk = float(cache.key_cache[0][0, :, :P].float().abs().sum())
                assert abs(chk - prefix_checksum) < 1e-3 * max(1.0, prefix_checksum), "prefix cache was modified"
                assert cache.get_seq_length() == P, "cache not cropped back to the prefix"
                rows_out = {}
                for name, i in idx.items():
                    acts = list(pred[:t]) + [ptok] + choices[i]
                    tr = model.action_tokenizer.decode_token_ids_to_trajectory(torch.tensor([A0 + a for a in acts]))
                    rows_out[name] = {"action_idx": acts, "trajectory_pred": np.asarray(tr[0, 1:], np.float32)[:, :2].tolist(),
                                      "ent": ents[i], "p_gt": pgts[i], "p_action_mass": pact[i], "top_non_action": top_non[i]}
                rec = {"token": token, "group": r["group"], "t_star": t, "log": r["log"], "perturbation": pname,
                       "forced_token": ptok, "gt20": gt, "gt10_stored": r["gt"], "pred_prefix": pred[:t],
                       "trajectory_gt20": [p[:2] for p in e["traj20_xyh"]], "trajectory_gt10_stored": r["trajectory_gt"],
                       "rows": rows_out, "prior_temporal_feedback_window": prior.get((token, pname))}
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
    json.dump({"n_scenes_ok": n_ok, "n_scene_fail": n_fail, "n_units": n_units, "n_unit_fail": n_unit_fail, "H": H,
               "max_tstar": args.max_tstar, "rows": ROWS, "temperature": temp, "seed": args.seed,
               "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "elapsed_min": (time.time() - t0) / 60}, open(os.path.join(args.output, "run_meta.json"), "w"), indent=2)
    print(f"[done] scenes ok={n_ok} fail={n_fail} units={n_units} unit_fail={n_unit_fail}", flush=True)


if __name__ == "__main__":
    main()
