#!/usr/bin/env python
"""
Teacher-forced re-scoring of existing best-of-N candidates (P4, outputs/instability_detection_baselines). RTX 5090.

For every scene of an expanded_best_of_n run, the natural stub is regenerated exactly as in expanded_best_of_n.py
(same seed key, same GPU), the prompt + stub prefix is encoded once, and all 17 recorded candidates' action tokens are
fed in one batch. No new sampling. Per candidate and step we keep:
  logprob of the recorded token (check against the stored logprob_steps), entropy, top-1 / top-2 probability margin
  (action vocabulary, untempered), and for deviating candidates (first token != GT at t* < 9) hidden states of two
  layers at   pre  = the position that emits token t* (context = tokens 0 .. t*-1)
              post = mean over positions emitting tokens t*+1 .. 9 (context contains the deviating token)
Outputs: <out>/steps.jsonl (per scene: margins, entropies, logprobs per candidate) and <out>/hidden_{pre,post}_L{l}.npy
with <out>/hidden_index.jsonl (token, candidate, t*) in the same row order.

  CUDA_VISIBLE_DEVICES=1 python scripts/teacher_force_candidates.py <run_dir> <out_dir> [--layers 18 36]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _planner import N_ACT, Planner, seed_of   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--scenes", default="/root/VLA/autovla/dataset/nuplan/navtest_ext")
    ap.add_argument("--layers", type=int, nargs="+", default=[18, 36])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("RTX 5090 only: CUDA_VISIBLE_DEVICES=1")
    os.makedirs(a.out, exist_ok=True)
    recs = [json.loads(l) for l in open(os.path.join(a.run, "records.jsonl"))]
    sp = os.path.join(a.out, "steps.jsonl")
    done = {json.loads(l)["token"] for l in open(sp)} if os.path.exists(sp) else set()
    recs = [r for r in recs if r["token"] not in done]
    if a.limit:
        recs = recs[:a.limit]
    print(f"[plan] {len(recs)} scenes ({len(done)} done)", flush=True)
    pl = Planner(); llm, A0, model = pl.llm, pl.A0, pl.model
    hfiles = {(w, l): open(os.path.join(a.out, f"hidden_{w}_L{l}.f16"), "ab") for w in ("pre", "post") for l in a.layers}
    fidx = open(os.path.join(a.out, "hidden_index.jsonl"), "a"); fst = open(sp, "a")
    ferr = open(os.path.join(a.out, "errors.jsonl"), "a")
    t0 = time.time()
    for si, r in enumerate(recs, 1):
        token = r["token"]
        try:
            scene = json.load(open(os.path.join(a.scenes, f"{token}.json")))
            pl._vis.clear(); feats = {}
            for b in pl.agent.get_feature_builders():
                feats.update(b.compute_features(scene))
            mi = {k: v.to("cuda:0") for k, v in model.get_prompt(feats).items() if isinstance(v, torch.Tensor)}
            prompt = mi["input_ids"][0].tolist()
            stem_ids = torch.tensor([prompt + pl.stem], device="cuda:0")
            m1 = dict(mi); m1["input_ids"] = stem_ids; m1["attention_mask"] = torch.ones_like(stem_ids)
            torch.manual_seed(seed_of(a.seed, token, "stub"))
            with torch.no_grad():
                g = llm.generate(**m1, max_new_tokens=160, do_sample=True, temperature=pl.gen_conf["temperature"],
                                 top_k=pl.gen_conf["top_k"], top_p=pl.gen_conf["top_p"])
            new = g[0][stem_ids.shape[1]:].tolist()
            fa = next(j for j, x in enumerate(new) if x >= A0)
            prefix = prompt + pl.stem + new[:fa]; P = len(prefix)
            ids = torch.tensor([prefix], device="cuda:0"); llm.rope_deltas = None
            with torch.no_grad():
                out = llm(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values_videos=mi.get("pixel_values_videos"),
                          video_grid_thw=mi.get("video_grid_thw"), use_cache=True, cache_position=torch.arange(P, device="cuda:0"),
                          output_hidden_states=True)
            cache = out.past_key_values
            last_lg = out.logits[:, -1, A0:A0 + 2048].float()
            last_h = {l: out.hidden_states[l][:, -1].float() for l in a.layers}
            del out
            C = r["candidates"]; B = len(C)
            for li in range(len(cache.key_cache)):
                cache.key_cache[li] = cache.key_cache[li].expand(B, -1, -1, -1).contiguous()
                cache.value_cache[li] = cache.value_cache[li].expand(B, -1, -1, -1).contiguous()
            acts = torch.tensor([[A0 + x for x in c["action_idx"][:N_ACT - 1]] for c in C], device="cuda:0")
            with torch.no_grad():
                o = llm(input_ids=acts, attention_mask=torch.ones((B, P + N_ACT - 1), device="cuda:0", dtype=torch.long),
                        past_key_values=cache, use_cache=False, cache_position=torch.arange(P, P + N_ACT - 1, device="cuda:0"),
                        output_hidden_states=True)
            # logits / hidden predicting step k: k = 0 from the prefix, k >= 1 from position k-1 of the forced block
            lg = torch.cat([last_lg.expand(B, -1)[:, None], o.logits[:, :, A0:A0 + 2048].float()], 1)       # (B, 10, 2048)
            hs = {l: torch.cat([last_h[l].expand(B, -1)[:, None], o.hidden_states[l].float()], 1) for l in a.layers}
            del o, cache
            lp = torch.log_softmax(lg, -1); p = lp.exp()
            ent = -(p * lp).sum(-1); top2 = torch.topk(p, 2, -1).values; margin = top2[..., 0] - top2[..., 1]
            tok = torch.tensor([c["action_idx"] for c in C], device="cuda:0")
            lpt = lp.gather(-1, tok[..., None])[..., 0]
            gt = r["gt"]
            for i, c in enumerate(C):
                ts = next((k for k in range(N_ACT) if c["action_idx"][k] != gt[k]), None)
                if ts is None or ts >= N_ACT - 1:
                    continue
                for l in a.layers:
                    hfiles[("pre", l)].write(hs[l][i, ts].half().cpu().numpy().tobytes())
                    hfiles[("post", l)].write(hs[l][i, ts + 1:].mean(0).half().cpu().numpy().tobytes())
                fidx.write(json.dumps({"token": token, "cand": i, "t_star": ts}) + "\n")
            fst.write(json.dumps({"token": token, "margin": np.round(margin.cpu().numpy(), 5).tolist(),
                                  "entropy": np.round(ent.cpu().numpy(), 5).tolist(),
                                  "logprob": np.round(lpt.cpu().numpy(), 5).tolist(),
                                  "lp_check_maxabs": float(np.max(np.abs(lpt[0].cpu().numpy() - np.asarray(C[0]["logprob_steps"]))))}) + "\n")
            if si % 25 == 0:
                for f in list(hfiles.values()) + [fidx, fst]:
                    f.flush()
        except Exception as exc:
            ferr.write(json.dumps({"token": token, "error": repr(exc)}) + "\n"); ferr.flush()
        if si % 100 == 0:
            el = time.time() - t0
            print(f"[prog] {si}/{len(recs)} {el / si:.2f}s/scene eta {(len(recs) - si) * el / si / 60:.0f} min", flush=True)
    for f in list(hfiles.values()) + [fidx, fst, ferr]:
        f.close()
    print(f"[done] {len(recs)} scenes in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
