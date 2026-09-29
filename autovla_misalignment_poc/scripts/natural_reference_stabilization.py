#!/usr/bin/env python
"""
Trigger-free reference stabilisation on NATURAL decoding (deployment condition).

Experiments 12-14 corrected the context only after an oracle first mismatch t* on forced
perturbations. A deployed system knows neither t* nor whether a mismatch happened. Here the
policy is applied unconditionally from the first action: context positions 0 .. w-1 (or all)
hold the reference token instead of the model's own, executed actions stay the model's samples,
and nothing is forced. Outcomes are the natural A-label / FDE against GT.

Units (outputs/frame_index/natural_units.json):
  A-      the 52 scenes whose natural plan fails (arm N, P/R/A 5 s rule) -> can the policy rescue them?
  random  300 scenes sampled uniformly from the rest (seed 0)          -> does it harm normal driving?

Prefix = prompt + the stored natural stub (full_extract arm N), identical to every harness experiment;
step 0 is produced from the prefix alone and is the same in every row. Rows share one batched
forward per step and the same per-step seed (T = 0.01).

ROWS  normal, prev_w2, prev_w4, prev_all, consmed_all, consmean_all, gt_w4 (oracle), gt_all (oracle)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _planner import AUTOVLA_DIR, N_ACT, POC_DIR, Planner   # noqa: E402
from reference_stabilization import consensus                 # noqa: E402

ROWS = ["normal", "prev_w2", "prev_w4", "prev_all", "consmed_all", "consmean_all", "gt_w4", "gt_all"]
WIN = {"w2": 2, "w4": 4, "all": 99}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default=os.path.join(POC_DIR, "outputs/frame_index/natural_units.json"))
    ap.add_argument("--index", default=os.path.join(POC_DIR, "outputs/frame_index/natural_index.json"))
    ap.add_argument("--full-records", default=os.path.join(POC_DIR, "outputs/full_extract/records.jsonl"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--frame-scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc_frames"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/natural_reference_stabilization"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1") and not os.environ.get("VLA_ANY_GPU"):
        raise SystemExit("launch with CUDA_VISIBLE_DEVICES=1 (RTX 5090) or 0 (RTX 3080 Ti); all rows of a run share one GPU")
    for k in ("units", "index", "full_records", "scenes", "frame_scenes", "output"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.makedirs(args.output, exist_ok=True)
    if os.path.exists(os.path.join(args.output, "records.jsonl")) and not args.limit:
        raise SystemExit(f"{args.output}/records.jsonl exists; refusing to overwrite")

    units = json.load(open(args.units))
    if args.limit:
        units = [u for u in units if u["group"] == "A-"][:args.limit // 2] + \
                [u for u in units if u["group"] != "A-"][:args.limit - args.limit // 2]
    index = json.load(open(args.index))
    want = {u["token"] for u in units}
    stored = {}
    for line in open(args.full_records):
        x = json.loads(line)
        if x["token"] in want:
            stored[x["token"]] = {"ids": x["arms"]["N"]["token_ids"], "pred": x["arms"]["N"]["pred_action_idx"][:N_ACT],
                                  "gt": x["gt_action_idx"][:N_ACT], "traj_gt": x["trajectory_gt"]}
    B = len(ROWS)
    print(f"[plan] {len(units)} scenes ({sum(u['group']=='A-' for u in units)} A-), {B} rows", flush=True)

    pl = Planner()
    llm, A0, model = pl.llm, pl.A0, pl.model
    cb = np.asarray(pickle.load(open(os.path.join(AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl"), "rb"))["token_all"]["veh"],
                    np.float64).reshape(2048, -1)
    emb = llm.get_input_embeddings().weight[A0:A0 + 2048].detach().float().cpu().numpy()
    en = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    cosm = en @ en.T
    np.fill_diagonal(cosm, 0)
    trained = np.where(~(cosm > 0.999).any(1))[0]
    del cosm

    fout = open(os.path.join(args.output, "records.jsonl"), "w")
    ferr = open(os.path.join(args.output, "errors.jsonl"), "w")
    t0 = time.time()
    n_ok = n_fail = 0
    for si, u in enumerate(units, 1):
        token = u["token"]
        try:
            st = stored[token]
            gt = st["gt"]
            scene = json.load(open(os.path.join(args.scenes, f"{token}.json")))
            # references: previous-frame plans shifted by k (fresh prompt, natural path)
            past = index[token]["past"]
            ref = {"gt": list(gt)}
            plans = {}
            for kk in (1, 2, 3):
                tk = past.get(str(kk))
                path = os.path.join(args.frame_scenes, f"{tk}.json") if tk else None
                if path and os.path.exists(path):
                    tt = pl.plan(json.load(open(path)), key=(args.seed, token, "prevplan") if kk == 1 else (args.seed, token, f"prevplan{kk}"))["tokens"]
                    plans[kk] = tt[kk:] + tt[-1:] * kk
            if 1 not in plans:
                raise RuntimeError("no previous-frame plan")
            ref["prev"] = plans[1]
            if len(plans) == 3:
                ref["consmed"] = consensus([plans[1], plans[2], plans[3]], cb, trained, "med")
                ref["consmean"] = consensus([plans[1], plans[2], plans[3]], cb, trained, "mean")
            else:
                ref["consmed"] = ref["consmean"] = None

            pl._vis.clear()
            feats = {}
            for b in pl.agent.get_feature_builders():
                feats.update(b.compute_features(scene))
            mi = {k: v.to("cuda:0") for k, v in model.get_prompt(feats).items() if isinstance(v, torch.Tensor)}
            ids_n = st["ids"]
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
            ents = {n: [] for n in ROWS}
            for k in range(N_ACT):
                if k == 0:
                    lg = last0.expand(B, -1)
                else:
                    spans = []
                    for i, name in enumerate(ROWS):
                        if name == "normal":
                            ctx = choices[i][:k]
                        else:
                            s, w = name.split("_")
                            rs = ref.get(s)
                            ctx = [rs[j] if (rs is not None and j < WIN[w]) else choices[i][j] for j in range(k)]
                        spans.append([A0 + a for a in ctx])
                    sids = torch.tensor(spans, device="cuda:0")
                    with torch.no_grad():
                        o = llm(input_ids=sids, attention_mask=torch.ones((B, P + k), device="cuda:0", dtype=torch.long),
                                past_key_values=cache, use_cache=True, cache_position=torch.arange(P, P + k, device="cuda:0"))
                    cache.crop(P)
                    lg = o.logits[:, -1, A0:A0 + 2048].double()
                    del o
                lp = torch.log_softmax(lg, -1)
                ent = -(lp.exp() * lp).sum(-1)
                seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{token}:natural:{k}".encode()).digest()[:4], "little")
                probs_t = torch.softmax(lg / pl.temp, -1).cpu()
                gen = torch.Generator()
                for i, name in enumerate(ROWS):
                    gen.manual_seed(seed)
                    choices[i].append(int(torch.multinomial(probs_t[i], 1, generator=gen)))
                    ents[name].append(round(float(ent[i]), 5))
            del cache
            conds = {name: {"action_idx": choices[i], "trajectory_pred": pl.decode(choices[i])[:, :2].tolist(),
                            "entropy_steps": ents[name]} for i, name in enumerate(ROWS)}
            gt_xy = np.asarray(st["traj_gt"], float)[:, :2]
            rq = {s: {"ade": float(np.linalg.norm(pl.decode(ref[s])[:, :2] - gt_xy, axis=1).mean()),
                      "token_eq_gt": float(np.mean([a == b for a, b in zip(ref[s], gt)]))}
                  for s in ("prev", "consmed", "consmean") if ref.get(s) is not None}
            fout.write(json.dumps({"token": token, "group": u["group"], "log": u["log"], "gt": gt, "trajectory_gt": st["traj_gt"],
                                   "stored_pred": st["pred"], "normal_reproduces_stored": choices[0] == st["pred"],
                                   "conditions": conds, "ref": ref, "ref_quality": rq}) + "\n")
            fout.flush()
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            import traceback; traceback.print_exc()
            ferr.write(json.dumps({"token": token, "error": repr(exc)}) + "\n"); ferr.flush()
        if si % 10 == 0 or args.limit:
            el = time.time() - t0
            print(f"[prog] {si}/{len(units)} ok={n_ok} fail={n_fail} {el/si:.1f}s/scene eta {(len(units)-si)*el/si/60:.1f}min", flush=True)
    fout.close(); ferr.close()
    json.dump({"n_ok": n_ok, "n_fail": n_fail, "rows": ROWS, "temperature": pl.temp, "seed": args.seed,
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "elapsed_min": (time.time() - t0) / 60},
              open(os.path.join(args.output, "run_meta.json"), "w"), indent=2)
    print(f"[done] ok={n_ok} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
