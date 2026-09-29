#!/usr/bin/env python
"""
Best-of-N on NEW navtest scenes (experiment 20: more natural failures for statistical power).

Same candidate set as best_of_n_selection (row 0 = natural plan at T = 0.01, rows 1..N at
--temperature), but for scenes that have no full_extract record: the natural stub is generated here
("<think>\\nThis is a" stem + free decoding until the first action token, exactly as _planner.Planner)
and the GT tokens come from the same tokenisation routine as every other experiment. Every scene is
decoded, so the population effect is measured directly (no failure / normal weighting).

  --scenes  directory of preprocessed scene JSONs;  --tokens  optional JSON list restricting it
Output: <output>/records.jsonl in the best_of_n_selection format (+ "stub_cot").
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _planner import N_ACT, POC_DIR, Planner, seed_of   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--log-of", default=None, help="JSON {token: log}; else taken from camera paths")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1") and not os.environ.get("VLA_ANY_GPU"):
        raise SystemExit("launch with CUDA_VISIBLE_DEVICES=1 (RTX 5090) or 0 (RTX 3080 Ti)")
    for k in ("scenes", "output"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.makedirs(args.output, exist_ok=True)
    toks = json.load(open(args.tokens)) if args.tokens else sorted(f[:-5] for f in os.listdir(args.scenes) if f.endswith(".json"))
    done = set()
    rec_path = os.path.join(args.output, "records.jsonl")
    if os.path.exists(rec_path):                       # resumable: skip scenes already written
        done = {json.loads(l)["token"] for l in open(rec_path)}
    toks = [t for t in toks if t not in done]
    if args.limit:
        toks = toks[:args.limit]
    B = args.n + 1
    print(f"[plan] {len(toks)} scenes to decode ({len(done)} already done), {B} candidates", flush=True)

    pl = Planner()
    llm, A0, model = pl.llm, pl.A0, pl.model
    temps = torch.tensor([pl.temp] + [args.temperature] * args.n, dtype=torch.float64)[:, None]
    fout = open(rec_path, "a")
    ferr = open(os.path.join(args.output, "errors.jsonl"), "a")
    t0 = time.time()
    n_ok = n_fail = 0
    for si, token in enumerate(toks, 1):
        try:
            scene = json.load(open(os.path.join(args.scenes, f"{token}.json")))
            log = scene["front_camera_paths"][0].split("/")[0]
            gt = pl.tokenize(scene["gt_trajectory"])
            pl._vis.clear()
            feats = {}
            for b in pl.agent.get_feature_builders():
                feats.update(b.compute_features(scene))
            mi = {k: v.to("cuda:0") for k, v in model.get_prompt(feats).items() if isinstance(v, torch.Tensor)}
            prompt = mi["input_ids"][0].tolist()
            stem_ids = torch.tensor([prompt + pl.stem], device="cuda:0")
            m1 = dict(mi); m1["input_ids"] = stem_ids; m1["attention_mask"] = torch.ones_like(stem_ids)
            torch.manual_seed(seed_of(args.seed, token, "stub"))
            with torch.no_grad():
                g = llm.generate(**m1, max_new_tokens=160, do_sample=True, temperature=pl.gen_conf["temperature"],
                                 top_k=pl.gen_conf["top_k"], top_p=pl.gen_conf["top_p"])
            new = g[0][stem_ids.shape[1]:].tolist()
            fa = next((j for j, x in enumerate(new) if x >= A0), None)
            if fa is None:
                raise RuntimeError("no action token within 160 generated tokens")
            stub = pl.stem + new[:fa]
            prefix = prompt + stub
            P = len(prefix)
            ids = torch.tensor([prefix], device="cuda:0")
            llm.rope_deltas = None
            with torch.no_grad():
                out = llm(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values_videos=mi.get("pixel_values_videos"),
                          video_grid_thw=mi.get("video_grid_thw"), use_cache=True, cache_position=torch.arange(P, device="cuda:0"))
            cache, last0 = out.past_key_values, out.logits[:, -1, A0:A0 + 2048].double()
            del out
            for li in range(len(cache.key_cache)):
                cache.key_cache[li] = cache.key_cache[li].expand(B, -1, -1, -1).contiguous()
                cache.value_cache[li] = cache.value_cache[li].expand(B, -1, -1, -1).contiguous()
            choices = [[] for _ in range(B)]
            lps, ents = [[] for _ in range(B)], [[] for _ in range(B)]
            for k in range(N_ACT):
                if k == 0:
                    lg = last0.expand(B, -1)
                else:
                    sids = torch.tensor([[A0 + a for a in c] for c in choices], device="cuda:0")
                    with torch.no_grad():
                        o = llm(input_ids=sids, attention_mask=torch.ones((B, P + k), device="cuda:0", dtype=torch.long),
                                past_key_values=cache, use_cache=True, cache_position=torch.arange(P, P + k, device="cuda:0"))
                    cache.crop(P)
                    lg = o.logits[:, -1, A0:A0 + 2048].double()
                    del o
                lp = torch.log_softmax(lg, -1).cpu()
                ent = -(lp.exp() * lp).sum(-1)
                probs = torch.softmax(lg.cpu() / temps, -1)
                gen = torch.Generator()
                for i in range(B):
                    key = f"{args.seed}:{token}:natural:{k}" if i == 0 else f"{args.seed}:{token}:bon{i}:{k}"
                    gen.manual_seed(int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little"))
                    a = int(torch.multinomial(probs[i], 1, generator=gen))
                    choices[i].append(a)
                    lps[i].append(round(float(lp[i, a]), 5))
                    ents[i].append(round(float(ent[i]), 5))
            del cache
            cands = [{"action_idx": choices[i], "trajectory_pred": pl.decode(choices[i])[:, :2].tolist(),
                      "logprob_steps": lps[i], "entropy_steps": ents[i], "temperature": float(temps[i, 0])} for i in range(B)]
            fout.write(json.dumps({"token": token, "log": log, "gt": gt,
                                   "trajectory_gt": np.asarray(scene["gt_trajectory"])[:, :2].tolist(),
                                   "stub_cot": "complex" in pl.tok.decode(stub), "candidates": cands}) + "\n")
            fout.flush()
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            ferr.write(json.dumps({"token": token, "error": repr(exc)}) + "\n"); ferr.flush()
        if si % 50 == 0:
            el = time.time() - t0
            print(f"[prog] {si}/{len(toks)} ok={n_ok} fail={n_fail} {el/si:.2f}s/scene eta {(len(toks)-si)*el/si/60:.1f}min", flush=True)
    fout.close(); ferr.close()
    print(f"[done] ok={n_ok} fail={n_fail} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
