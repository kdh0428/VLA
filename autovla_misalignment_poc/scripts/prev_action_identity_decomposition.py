#!/usr/bin/env python
"""
Which component of the previous action token's identity carries the 1-step feedback?
(GPU 1, natural/fast only.)

prev_action_state_patching showed that the amplification after a first mismatch is carried by
the identity of the immediately preceding action token (patch@emb == Recent-GT recovers ~93%).
It could not say WHAT about that identity matters. Two candidates:

  geometry   the codebook displacement the token stands for (what the car would do), which a
             model could read through any token with a similar motion;
  identity   the token-specific embedding vector itself, unrelated to its motion.

Offline check (see run_meta["embedding_geometry"]): the action embeddings (tied to lm_head)
encode almost no codebook geometry (poly-3 ridge CV R^2 ~ 0.03-0.09; embedding/codebook
distance Spearman 0.06), and 378 + 32 tokens share one near-identical embedding (untrained).
So geometry and embedding proximity can be separated by picking substitute tokens.

Data and harness: identical to prev_action_state_patching / action_history_causal
(equal-distance set, 208 scenes; unit = (scene, perturbation) with the forced token at t*;
prefix KV cache; action-row sampling at T = 0.01 with seed sha256(f"{seed}:{token}:{pert}:{k}")).
From k = t*+2 on, the LAST context token (the one the next action is conditioned on) is
replaced per row; earlier context stays the row's own tokens, exactly like Recent-GT.
At k = t*+1 the preceding token is the perturbation itself and is never replaced.

ROWS
  normal        plain autoregression
  gt_history    context after t* is GT
  recent_gt     last context token := GT                             (reference: full effect)
  geo_nn_gt     last := codebook-nearest token to GT     (motion ~ GT, embedding != GT)
  emb_nn_gt     last := embedding-nearest token to GT    (embedding ~ GT, motion arbitrary)
  geo_nn_self   last := codebook-nearest token to own    (motion ~ own, identity changed)
  emb_nn_self   last := embedding-nearest token to own   (small embedding change, control)
  random_tok    last := uniform random trained token     (identity destroyed, nonsense motion)
  mean_emb      last position embedding := mean trained action embedding   [OOD, hook]
Candidates for every substitution exclude GT, the row's own token and the untrained group.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
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
N_CODES = 2048
ROWS = ["normal", "gt_history", "recent_gt", "geo_nn_gt", "emb_nn_gt", "geo_nn_self", "emb_nn_self",
        "random_tok", "mean_emb"]
SUB_ROWS = {"geo_nn_gt", "emb_nn_gt", "geo_nn_self", "emb_nn_self", "random_tok"}
DUP_COS = 0.999


def token_spaces(emb: np.ndarray, codebook: np.ndarray):
    """Nearest-neighbour tables in codebook (motion) and embedding space, untrained group removed."""
    en = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    cos = en @ en.T
    np.fill_diagonal(cos, 0.0)
    untrained = (cos > DUP_COS).any(1)
    g = codebook.reshape(N_CODES, -1).astype(np.float64)
    from scipy.spatial.distance import cdist
    gd = cdist(g, g)
    np.fill_diagonal(cos, 1.0)
    # sorted candidate lists (trained tokens only, self excluded later at lookup time)
    ok = np.where(~untrained)[0]
    geo_order = {i: ok[np.argsort(gd[i, ok])] for i in range(N_CODES)}
    emb_order = {i: ok[np.argsort(-cos[i, ok])] for i in range(N_CODES)}
    return untrained, gd, cos, geo_order, emb_order


def first_not(order, exclude):
    for c in order:
        if int(c) not in exclude:
            return int(c)
    raise RuntimeError("no candidate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ed-records", default=os.path.join(POC_DIR, "outputs/equal_distance_perturbation/records.jsonl"))
    ap.add_argument("--ah-records", default=os.path.join(POC_DIR, "outputs/action_history_causal/records.jsonl"))
    ap.add_argument("--full-records", default=os.path.join(POC_DIR, "outputs/full_extract/records.jsonl"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/prev_action_identity_decomposition"))
    ap.add_argument("--config", default=os.path.join(AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-alts", type=int, default=0)
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" and not os.environ.get("VLA_ANY_GPU"):
        raise SystemExit("pinned to GPU 1: launch with CUDA_VISIBLE_DEVICES=1")
    for k in ("ed_records", "ah_records", "full_records", "scenes", "output", "config"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.makedirs(args.output, exist_ok=True)
    if os.path.exists(os.path.join(args.output, "records.jsonl")) and not args.limit:
        raise SystemExit(f"{args.output}/records.jsonl exists; refusing to overwrite")

    B = len(ROWS)
    idx = {n: i for i, n in enumerate(ROWS)}

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
    if os.path.exists(args.ah_records):
        for line in open(args.ah_records):
            x = json.loads(line)
            if x["token"] in want:
                for pname, pr in x["perturbations"].items():
                    prior[(x["token"], pname)] = {c: pr["conditions"][c]["action_idx"] for c in ("normal", "gt_history", "recent_gt")}
    print(f"[plan] {len(ed)} scenes (A- {sum(r['group']=='A-' for r in ed)}, A+ {sum(r['group']=='A+' for r in ed)}), "
          f"{B} rows per batched step, prior action_history rows for {len(prior)} units", flush=True)

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
    layers = llm.model.layers

    emb_w = llm.get_input_embeddings().weight
    act_emb = emb_w[A0:A0 + N_CODES].detach().float().cpu().numpy()
    codebook = np.asarray(pickle.load(open(cfg["model"]["codebook_cache_path"], "rb"))["token_all"]["veh"])
    untrained, gd, cos, geo_order, emb_order = token_spaces(act_emb, codebook)
    trained_ids = np.where(~untrained)[0]
    mean_vec = emb_w[A0 + torch.as_tensor(trained_ids, device=emb_w.device)].float().mean(0)
    print(f"[tokens] untrained group {int(untrained.sum())}, trained {len(trained_ids)}", flush=True)

    # mean_emb: replace the layer-0 input at the last span position of that row
    MEAN = {"on": False}

    def pre_hook_emb(_m, a, kw):
        if not MEAN["on"]:
            return None
        hs = a[0] if a else kw.get("hidden_states")
        hs = hs.clone()
        hs[idx["mean_emb"], -1] = mean_vec.to(hs.dtype)
        if a:
            return (hs,) + tuple(a[1:]), kw
        kw = dict(kw); kw["hidden_states"] = hs
        return a, kw
    layers[0].register_forward_pre_hook(pre_hook_emb, with_kwargs=True)

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
            MEAN["on"] = False
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
                ents = {n: [] for n in ROWS}
                subs_log = {n: [] for n in SUB_ROWS | {"mean_emb"}}
                for k in range(t + 1, N_ACT):
                    useed = int.from_bytes(hashlib.sha256(f"{args.seed}:{token}:{pname}:{k}:rand".encode()).digest()[:4], "little")
                    rng = np.random.default_rng(useed)
                    spans = []
                    for i, name in enumerate(ROWS):
                        own = choices[i]
                        if name == "gt_history":
                            ctx = list(gt[t + 1:k])
                        elif name == "normal" or name == "mean_emb" or not own:
                            ctx = list(own)
                        else:
                            prev_own, g_prev = own[-1], gt[k - 1]
                            excl = {int(g_prev), int(prev_own)}
                            if name == "recent_gt":
                                sub = g_prev
                            elif name == "geo_nn_gt":
                                sub = first_not(geo_order[g_prev], excl)
                            elif name == "emb_nn_gt":
                                sub = first_not(emb_order[g_prev], excl)
                            elif name == "geo_nn_self":
                                sub = first_not(geo_order[prev_own], excl)
                            elif name == "emb_nn_self":
                                sub = first_not(emb_order[prev_own], excl)
                            elif name == "random_tok":
                                cand = trained_ids[~np.isin(trained_ids, list(excl))]
                                sub = int(rng.choice(cand))
                            ctx = own[:-1] + [int(sub)]
                            if name in SUB_ROWS:
                                subs_log[name].append({
                                    "k": k, "own": int(prev_own), "gt": int(g_prev), "sub": int(sub),
                                    "geo_d_sub_gt": round(float(gd[sub, g_prev]), 4), "geo_d_own_gt": round(float(gd[prev_own, g_prev]), 4),
                                    "geo_d_sub_own": round(float(gd[sub, prev_own]), 4),
                                    "cos_sub_gt": round(float(cos[sub, g_prev]), 5), "cos_own_gt": round(float(cos[prev_own, g_prev]), 5),
                                    "cos_sub_own": round(float(cos[sub, prev_own]), 5)})
                        if name == "mean_emb" and own:
                            subs_log["mean_emb"].append({"k": k, "own": int(own[-1]), "gt": int(gt[k - 1])})
                        spans.append([A0 + ptok] + [A0 + a for a in ctx])
                    m = len(spans[0])
                    assert all(len(s) == m for s in spans), "span lengths must match across rows"
                    sids = torch.tensor(spans, device="cuda:0")
                    MEAN["on"] = m > 1
                    with torch.no_grad():
                        o = llm(input_ids=sids, attention_mask=torch.ones((B, P + m), device="cuda:0", dtype=torch.long),
                                past_key_values=cache, use_cache=True, cache_position=torch.arange(P, P + m, device="cuda:0"))
                    MEAN["on"] = False
                    cache.crop(P)
                    lg = o.logits[:, -1, A0:A0 + N_CODES].double()
                    del o
                    lp = torch.log_softmax(lg, -1)
                    ent = -(lp.exp() * lp).sum(-1)
                    seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{token}:{pname}:{k}".encode()).digest()[:4], "little")
                    probs_t = torch.softmax(lg / temp, -1).cpu()
                    gen = torch.Generator()
                    for i, name in enumerate(ROWS):
                        gen.manual_seed(seed)
                        choices[i].append(int(torch.multinomial(probs_t[i], 1, generator=gen)))
                        ents[name].append(round(float(ent[i]), 5))
                chk = float(cache.key_cache[0][0, :, :P].float().abs().sum())
                assert abs(chk - prefix_checksum) < 1e-3 * max(1.0, prefix_checksum), "prefix cache was modified"
                assert cache.get_seq_length() == P, "cache not cropped back to the prefix"

                conds = {}
                for i, name in enumerate(ROWS):
                    acts = list(pred[:t]) + [ptok] + choices[i]
                    tr = model.action_tokenizer.decode_token_ids_to_trajectory(torch.tensor([A0 + a for a in acts]))
                    conds[name] = {"action_idx": acts, "trajectory_pred": np.asarray(tr[0, 1:], np.float32)[:, :2].tolist(),
                                   "entropy_steps": ents[name]}
                    if name in subs_log:
                        conds[name]["substitutions"] = subs_log[name]
                pri = prior.get((token, pname))
                rec = {"token": token, "group": r["group"], "t_star": t, "log": r["log"], "perturbation": pname,
                       "forced_token": ptok, "gt": gt, "pred_prefix": pred[:t], "trajectory_gt": r["trajectory_gt"],
                       "conditions": conds,
                       "reproduces_action_history": ({c: pri[c] == conds[c]["action_idx"] for c in pri} if pri else None)}
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
               "rows": ROWS, "temperature": temp, "seed": args.seed,
               "n_untrained_tokens": int(untrained.sum()), "dup_cos_threshold": DUP_COS,
               "embedding_geometry": {"note": "offline, CPU: ridge CV R^2 of action embeddings from codebook geometry",
                                      "geom_dxdyyaw_linear": 0.026, "geom_poly3": 0.088, "codebook48_linear": -0.021,
                                      "codebook48_poly2": 0.080, "rsa_spearman": 0.06, "tied_embeddings": True},
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "elapsed_min": (time.time() - t0) / 60}, open(os.path.join(args.output, "run_meta.json"), "w"), indent=2)
    print(f"[done] scenes ok={n_ok} fail={n_fail} units={n_units} unit_fail={n_unit_fail}", flush=True)


if __name__ == "__main__":
    main()
