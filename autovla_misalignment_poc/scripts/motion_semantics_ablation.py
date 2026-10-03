#!/usr/bin/env python
"""
Previous-action motion semantics ablation (P5, outputs/motion_semantics_ablation/PROTOCOL.md). RTX 5090.

Same harness as prev_action_identity_decomposition.py (experiment 11: equal-distance set, 208 scenes / 1,408 units,
forced token at t*, prefix KV cache, rows sampled at T = 0.01 with the same seeds). From k = t*+2 on, the LAST context
token is replaced per row; every substitute is chosen from the token's motion m = (dx, dy) of the codebook end pose
(as ActionTokenizer.rollout), relative to the GT token g and the row's own token o (e = |m_o - m_g|, the own error):
  normal              no replacement (model-generated previous action)
  recent_gt           g                                                       (GT previous action)
  dir_ok_mag_wrong    direction of g, speed error = e:  min (dphi/1deg)^2 + ((|dmag| - e)/(0.1 max(e, 0.1 m)))^2
  dir_wrong_mag_ok    speed of g, displacement error = e:  min (dmag/0.05 m)^2 + ((|m - m_g| - e)/(0.1 max(e, 0.1 m)))^2
  mirror_same_dist    same distance e on the opposite side of g:  nearest to 2 m_g - m_o
  random_mag_matched  uniform random token with |mag - mag_g| <= max(0.05 mag_g, 0.05 m), any direction
Candidates exclude g, o and the untrained embedding group. Achieved dphi / dmag / distance are logged per substitution.
Per row: action tokens, entropy and log-prob of every generated step.
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
ROWS = ["normal", "recent_gt", "dir_ok_mag_wrong", "dir_wrong_mag_ok", "mirror_same_dist", "random_mag_matched"]
SUB_ROWS = {"dir_ok_mag_wrong", "dir_wrong_mag_ok", "mirror_same_dist", "random_mag_matched"}


def token_motion(codebook: np.ndarray) -> np.ndarray:
    """(dx, dy) of each token's end pose: mean of the last contour, as ActionTokenizer.rollout (local frame)."""
    return codebook[:, -1].mean(axis=1).astype(np.float64)


def angdiff(a, b):
    return np.abs(np.degrees(np.arctan2(np.sin(a - b), np.cos(a - b))))


def motion_sub(name, g, o, M, ok_ids, rng):
    mg, mo = M[g], M[o]; e = float(np.linalg.norm(mo - mg))
    c = ok_ids[(ok_ids != g) & (ok_ids != o)]; mc = M[c]
    mag_c, mag_g = np.linalg.norm(mc, axis=1), float(np.linalg.norm(mg))
    phi_c, phi_g = np.arctan2(mc[:, 1], mc[:, 0]), float(np.arctan2(mg[1], mg[0]))
    dist = np.linalg.norm(mc - mg, axis=1); sc = 0.1 * max(e, 0.1)
    if name == "dir_ok_mag_wrong":
        cost = (angdiff(phi_c, phi_g) / 1.0) ** 2 + ((np.abs(mag_c - mag_g) - e) / sc) ** 2
        return int(c[np.argmin(cost)])
    if name == "dir_wrong_mag_ok":
        cost = ((mag_c - mag_g) / 0.05) ** 2 + ((dist - e) / sc) ** 2
        return int(c[np.argmin(cost)])
    if name == "mirror_same_dist":
        return int(c[np.argmin(np.linalg.norm(mc - (2 * mg - mo), axis=1))])
    if name == "random_mag_matched":
        pool = c[np.abs(mag_c - mag_g) <= max(0.05 * mag_g, 0.05)]
        if len(pool) == 0:
            pool = c[np.argsort(np.abs(mag_c - mag_g))[:5]]
        return int(rng.choice(pool))
    raise ValueError(name)
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
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/motion_semantics_ablation"))
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
    print(f"[tokens] untrained group {int(untrained.sum())}, trained {len(trained_ids)}", flush=True)

    MOT = token_motion(codebook)

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
                ents = {n: [] for n in ROWS}
                lps = {n: [] for n in ROWS}
                subs_log = {n: [] for n in SUB_ROWS}
                for k in range(t + 1, N_ACT):
                    useed = int.from_bytes(hashlib.sha256(f"{args.seed}:{token}:{pname}:{k}:rand".encode()).digest()[:4], "little")
                    rng = np.random.default_rng(useed)
                    spans = []
                    for i, name in enumerate(ROWS):
                        own = choices[i]
                        if name == "normal" or not own:
                            ctx = list(own)
                        else:
                            prev_own, g_prev = int(own[-1]), int(gt[k - 1])
                            sub = g_prev if name == "recent_gt" else motion_sub(name, g_prev, prev_own, MOT, trained_ids, rng)
                            ctx = own[:-1] + [int(sub)]
                            if name in SUB_ROWS:
                                mg, mo, ms = MOT[g_prev], MOT[prev_own], MOT[sub]
                                subs_log[name].append({
                                    "k": k, "own": prev_own, "gt": g_prev, "sub": int(sub),
                                    "e_own": round(float(np.linalg.norm(mo - mg)), 4), "d_sub_gt": round(float(np.linalg.norm(ms - mg)), 4),
                                    "dmag_sub_gt": round(float(np.linalg.norm(ms) - np.linalg.norm(mg)), 4),
                                    "dphi_sub_gt_deg": round(float(angdiff(np.arctan2(ms[1], ms[0]), np.arctan2(mg[1], mg[0]))), 3),
                                    "mag_gt": round(float(np.linalg.norm(mg)), 4)})
                        spans.append([A0 + ptok] + [A0 + a for a in ctx])
                    m = len(spans[0])
                    assert all(len(s) == m for s in spans), "span lengths must match across rows"
                    sids = torch.tensor(spans, device="cuda:0")
                    with torch.no_grad():
                        o = llm(input_ids=sids, attention_mask=torch.ones((B, P + m), device="cuda:0", dtype=torch.long),
                                past_key_values=cache, use_cache=True, cache_position=torch.arange(P, P + m, device="cuda:0"))
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
                        a_ = int(torch.multinomial(probs_t[i], 1, generator=gen))
                        choices[i].append(a_)
                        ents[name].append(round(float(ent[i]), 5)); lps[name].append(round(float(lp[i, a_]), 5))
                chk = float(cache.key_cache[0][0, :, :P].float().abs().sum())
                assert abs(chk - prefix_checksum) < 1e-3 * max(1.0, prefix_checksum), "prefix cache was modified"
                assert cache.get_seq_length() == P, "cache not cropped back to the prefix"

                conds = {}
                for i, name in enumerate(ROWS):
                    acts = list(pred[:t]) + [ptok] + choices[i]
                    tr = model.action_tokenizer.decode_token_ids_to_trajectory(torch.tensor([A0 + a for a in acts]))
                    conds[name] = {"action_idx": acts, "trajectory_pred": np.asarray(tr[0, 1:], np.float32)[:, :2].tolist(),
                                   "entropy_steps": ents[name], "logprob_steps": lps[name]}
                    if name in subs_log:
                        conds[name]["substitutions"] = subs_log[name]
                pri = prior.get((token, pname))
                pri = {"normal": pri["normal"], "recent_gt": pri["recent_gt"]} if pri else None
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
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "elapsed_min": (time.time() - t0) / 60}, open(os.path.join(args.output, "run_meta.json"), "w"), indent=2)
    print(f"[done] scenes ok={n_ok} fail={n_fail} units={n_units} unit_fail={n_unit_fail}", flush=True)


if __name__ == "__main__":
    main()
