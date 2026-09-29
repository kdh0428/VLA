#!/usr/bin/env python
"""
Selection instead of conditioning: best-of-N over the model's own samples (deployment condition).

Experiments 15-16 showed that CONDITIONING on an external motion reference drags the model onto
that reference (66-89% of the way), which rescues failing scenes but degrades normal ones, so the
population failure rate gets worse. Here references are used only to SELECT among plans the model
samples itself; nothing is injected into the context.

Units and references: the natural units of natural_reference_stabilization (52 natural failures +
297 random scenes) and the previous-frame plan stored there (ref["prev"]), so no new references.

Candidates per scene (one batched decode, prefix = prompt + stored natural stub):
  row 0      T = 0.01, the same per-step seed as every experiment      (= the normal plan)
  rows 1..N  T = --temperature, independent per-row seeds
For every candidate the per-step log-probability of its own token under the T = 1 distribution
and the entropy are stored. Selection rules are applied in the analysis script.
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
from _planner import AUTOVLA_DIR, N_ACT, POC_DIR, Planner   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nat-records", default=os.path.join(POC_DIR, "outputs/natural_reference_stabilization/records.jsonl"))
    ap.add_argument("--full-records", default=os.path.join(POC_DIR, "outputs/full_extract/records.jsonl"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/best_of_n_selection"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1") and not os.environ.get("VLA_ANY_GPU"):
        raise SystemExit("launch with CUDA_VISIBLE_DEVICES=1 (RTX 5090) or 0 (RTX 3080 Ti)")
    for k in ("nat_records", "full_records", "scenes", "output"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.makedirs(args.output, exist_ok=True)
    if os.path.exists(os.path.join(args.output, "records.jsonl")) and not args.limit:
        raise SystemExit(f"{args.output}/records.jsonl exists; refusing to overwrite")

    nat = [json.loads(l) for l in open(args.nat_records)]
    if args.limit:
        nat = [r for r in nat if r["group"] == "A-"][:args.limit // 2] + [r for r in nat if r["group"] != "A-"][:args.limit - args.limit // 2]
    want = {r["token"] for r in nat}
    stub_of = {}
    for line in open(args.full_records):
        x = json.loads(line)
        if x["token"] in want:
            stub_of[x["token"]] = x["arms"]["N"]["token_ids"]
    B = args.n + 1
    print(f"[plan] {len(nat)} scenes, {B} candidates (1 at T=0.01 + {args.n} at T={args.temperature})", flush=True)

    pl = Planner()
    llm, A0, model = pl.llm, pl.A0, pl.model
    temps = torch.tensor([pl.temp] + [args.temperature] * args.n, dtype=torch.float64)[:, None]
    fout = open(os.path.join(args.output, "records.jsonl"), "w")
    ferr = open(os.path.join(args.output, "errors.jsonl"), "w")
    t0 = time.time()
    n_ok = n_fail = 0
    for si, r in enumerate(nat, 1):
        token = r["token"]
        try:
            scene = json.load(open(os.path.join(args.scenes, f"{token}.json")))
            pl._vis.clear()
            feats = {}
            for b in pl.agent.get_feature_builders():
                feats.update(b.compute_features(scene))
            mi = {k: v.to("cuda:0") for k, v in model.get_prompt(feats).items() if isinstance(v, torch.Tensor)}
            ids_n = stub_of[token]
            stub = ids_n[:next(j for j, x in enumerate(ids_n) if x >= A0)]
            prefix = mi["input_ids"][0].tolist() + stub
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
            prev = r["ref"].get("prev")
            fout.write(json.dumps({"token": token, "group": r["group"], "log": r["log"], "gt": r["gt"], "trajectory_gt": r["trajectory_gt"],
                                   "prev_ref": prev, "prev_traj": pl.decode(prev)[:, :2].tolist() if prev else None,
                                   "candidates": cands}) + "\n")
            fout.flush()
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            import traceback; traceback.print_exc()
            ferr.write(json.dumps({"token": token, "error": repr(exc)}) + "\n"); ferr.flush()
        if si % 10 == 0 or args.limit:
            el = time.time() - t0
            print(f"[prog] {si}/{len(nat)} ok={n_ok} fail={n_fail} {el/si:.1f}s/scene eta {(len(nat)-si)*el/si/60:.1f}min", flush=True)
    fout.close(); ferr.close()
    json.dump({"n_ok": n_ok, "n_fail": n_fail, "n": args.n, "temperature": args.temperature, "seed": args.seed,
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "elapsed_min": (time.time() - t0) / 60},
              open(os.path.join(args.output, "run_meta.json"), "w"), indent=2)
    print(f"[done] ok={n_ok} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
