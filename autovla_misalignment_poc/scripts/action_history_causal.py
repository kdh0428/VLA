#!/usr/bin/env python
"""
Is self-conditioning on previously generated action tokens what sustains error amplification?
(GPU 1, natural/fast only)

Scenes and perturbations are taken verbatim from outputs/equal_distance_perturbation:
  208 scenes (A- 52 unstable, A+ 156 stable), and per scene the forced token at the first
  mismatch step t*: the model's original token and its distance-matched alternatives.

For every (scene, perturbation) the prefix (prompt + fast stub + action tokens before t*) and
the forced token at t* are identical. Only how the remaining action tokens are produced changes:

  normal              plain autoregression
  hist_attn_mask      action queries from t* on cannot ATTEND to earlier action tokens from t*
                      on (self, visual and text keys untouched). NOTE: the token a_{k-1} is the
                      query's own input, so it still reaches the query through the residual
                      stream -- this removes attention-mediated history only.
  recent_attn_mask    each action query cannot attend to the span token right before its own
                      position (the closest attention-level analogue of "block the previous
                      action": a_{k-1} itself is the query's input and cannot be masked)
  hist_emb_neutral    [OOD stress test] input embedding of every action token from t* on replaced
                      by the mean action embedding. The sanity run showed first-step entropy
                      jumping from ~1.2 to ~4.5 nats and different perturbations collapsing to
                      the same degenerate sequence: this measures damage from an unseen input,
                      not removal of self-conditioning. Reported, never used for the verdict.
  recent_emb_neutral  [OOD stress test] only the most recent action token neutralised
  gt_history          the perturbation at t* is kept, but every later context token is the GT
                      token instead of the model's own choice (the output at each step is still
                      the model's choice)
  recent_gt           like gt_history but only the MOST RECENT generated context token is GT;
                      earlier generated tokens are kept (graded, in-distribution)

Decoding: one shared prefix KV cache per scene; at each step a forward over the post-t* span
(<= 10 tokens) with that cache, cropped back afterwards. Sampling from the action-token rows
only, T = 0.01 as AutoVLA's config, with a per-(scene, perturbation, step) seed that is the same
for every condition. Outputs are the model's chosen tokens in every condition.
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
CONDS = ["normal", "hist_attn_mask", "recent_attn_mask", "hist_emb_neutral", "recent_emb_neutral", "gt_history", "recent_gt"]


class ST:
    block = None             # None | "hist" | "recent" attention block for this forward
    neutral = None           # list of span positions whose input embedding is neutralised
    mean_emb = None
    n_blocked_calls = 0
    n_neutral_calls = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ed-records", default=os.path.join(POC_DIR, "outputs/equal_distance_perturbation/records.jsonl"))
    ap.add_argument("--full-records", default=os.path.join(POC_DIR, "outputs/full_extract/records.jsonl"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/action_history_causal"))
    ap.add_argument("--config", default=os.path.join(AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-alts", type=int, default=0, help="0 = all alternatives")
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" and not os.environ.get("VLA_ANY_GPU"):
        raise SystemExit("pinned to GPU 1: launch with CUDA_VISIBLE_DEVICES=1")
    for k in ("ed_records", "full_records", "scenes", "output", "config"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.makedirs(args.output, exist_ok=True)

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
    print(f"[plan] {len(ed)} scenes (A- {sum(r['group']=='A-' for r in ed)}, A+ {sum(r['group']=='A+' for r in ed)}) x perturbations x {len(CONDS)}", flush=True)

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
    emb = llm.model.embed_tokens
    ST.mean_emb = emb.weight[A0:A0 + 2048].detach().mean(0)

    def embed_hook(_m, _inp, out):
        if ST.neutral:
            out = out.clone()
            out[:, ST.neutral] = ST.mean_emb.to(out.dtype)
            ST.n_neutral_calls += 1
            return out
        return out
    emb.register_forward_hook(embed_hook)

    def make_attn(orig):
        def fwd(hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False,
                use_cache=False, cache_position=None, position_embeddings=None, **kw):
            q = hidden_states.shape[1]
            if ST.block and q > 1:
                if attention_mask is None:
                    raise RuntimeError("expected a 4D causal mask for a multi-token cached forward")
                am = attention_mask.clone()
                kv = am.shape[-1]
                p = kv - q
                minv = torch.finfo(am.dtype).min
                rows = torch.arange(q, device=am.device)
                if ST.block == "hist":
                    tri = rows[:, None] > rows[None, :]             # every earlier span token
                else:
                    tri = rows[:, None] == rows[None, :] + 1        # only the span token just before
                sub = am[..., :, p:p + q]
                am[..., :, p:p + q] = torch.where(tri, torch.full_like(sub, minv), sub)
                attention_mask = am
                ST.n_blocked_calls += 1
            return orig(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                        past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
                        cache_position=cache_position, position_embeddings=position_embeddings, **kw)
        return fwd
    for layer in llm.model.layers:
        layer.self_attn.forward = make_attn(layer.self_attn.forward)

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
    n_ok = n_fail = n_pert = 0
    for i, r in enumerate(ed, 1):
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
            ST.block, ST.neutral = None, None
            with torch.no_grad():
                out = llm(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values_videos=mi.get("pixel_values_videos"),
                          video_grid_thw=mi.get("video_grid_thw"), use_cache=True,
                          cache_position=torch.arange(P, device="cuda:0"))
            cache = out.past_key_values
            del out

            perts = [("original", pred[t])] + [(f"alt{j}", a["token"]) for j, a in enumerate(r["alternatives"])]
            if args.max_alts:
                perts = perts[:1 + args.max_alts]
            stored = {name: r["conditions"][name]["action_idx"] for name, _ in perts}
            res = {}
            for pname, ptok in perts:
                res[pname] = {"forced_token": ptok, "stored_normal_action_idx": stored[pname], "conditions": {}}
                for cname in CONDS:
                    choices, ents, gtlp = [], [], []
                    for k in range(t + 1, N_ACT):
                        if cname == "gt_history":
                            ctx = list(gt[t + 1:k])
                        elif cname == "recent_gt":
                            # choices[-1] is the token at step k-1; only it becomes GT
                            ctx = choices[:-1] + [gt[k - 1]] if choices else []
                        else:
                            ctx = choices
                        span = [A0 + ptok] + [A0 + a for a in ctx]
                        m = len(span)
                        ST.block = ("hist" if cname == "hist_attn_mask" else "recent" if cname == "recent_attn_mask" else None)
                        ST.neutral = (list(range(m)) if cname == "hist_emb_neutral"
                                      else [m - 1] if cname == "recent_emb_neutral" else None)
                        sids = torch.tensor([span], device="cuda:0")
                        with torch.no_grad():
                            o = llm(input_ids=sids, attention_mask=torch.ones((1, P + m), device="cuda:0", dtype=torch.long),
                                    past_key_values=cache, use_cache=True,
                                    cache_position=torch.arange(P, P + m, device="cuda:0"))
                        cache.crop(P)
                        ST.block, ST.neutral = None, None
                        lg = o.logits[0, -1, A0:A0 + 2048].double()
                        del o
                        lp = torch.log_softmax(lg, -1)
                        ents.append(float(-(lp.exp() * lp).sum()))
                        gtlp.append(float(lp[gt[k]]))
                        pr = torch.softmax(lg / temp, -1).cpu()
                        gen = torch.Generator().manual_seed(
                            int.from_bytes(hashlib.sha256(f"{args.seed}:{token}:{pname}:{k}".encode()).digest()[:4], "little"))
                        choices.append(int(torch.multinomial(pr, 1, generator=gen)))
                    acts = list(pred[:t]) + [ptok] + choices
                    tr = model.action_tokenizer.decode_token_ids_to_trajectory(torch.tensor([A0 + a for a in acts]))
                    res[pname]["conditions"][cname] = {"action_idx": acts, "entropy_steps": ents, "gt_logprob_steps": gtlp,
                                                       "trajectory_pred": np.asarray(tr[0, 1:], np.float32)[:, :2].tolist()}
                n_pert += 1
            keep = {k: r[k] for k in ("token", "group", "t_star", "log", "pred", "gt", "trajectory_gt", "geometry")}
            fout.write(json.dumps(keep | {"alternatives": r["alternatives"], "perturbations": res}) + "\n"); fout.flush()
            n_ok += 1
            del cache
            if args.sanity:
                print(f"  [{token} {r['group']} t*={t}] blocked_calls={ST.n_blocked_calls} neutral_calls={ST.n_neutral_calls} prefix={P}", flush=True)
                for pname in list(res)[:3]:
                    pr_ = res[pname]
                    rep = pr_["conditions"]["normal"]["action_idx"] == pr_["stored_normal_action_idx"]
                    print(f"     {pname:9s} stored normal={pr_['stored_normal_action_idx']} reproduced={rep}", flush=True)
                    for cname in CONDS:
                        c = pr_["conditions"][cname]
                        print(f"        {cname:18s} acts={c['action_idx']} H1={c['entropy_steps'][0] if c['entropy_steps'] else float('nan'):.3f}", flush=True)
        except Exception as exc:
            n_fail += 1
            import traceback; traceback.print_exc()
            print(f"[fail] {r['token']}: {exc}", flush=True)
        if i % 10 == 0 or args.sanity:
            el = time.time() - t0
            print(f"[prog] {i}/{len(ed)} scenes ok={n_ok} fail={n_fail} perturbations={n_pert} {el/i:.1f}s/scene "
                  f"eta {(len(ed)-i)*el/i/60:.1f}min", flush=True)
    fout.close()
    json.dump({"n_ok": n_ok, "n_fail": n_fail, "n_perturbations": n_pert, "conditions": CONDS, "temperature": temp,
               "seed": args.seed, "blocked_attention_calls": ST.n_blocked_calls, "neutralised_embedding_calls": ST.n_neutral_calls,
               "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "elapsed_min": (time.time() - t0) / 60}, open(os.path.join(args.output, "run_meta.json"), "w"), indent=2)
    print(f"[done] ok={n_ok} fail={n_fail} perturbations={n_pert}", flush=True)


if __name__ == "__main__":
    main()
