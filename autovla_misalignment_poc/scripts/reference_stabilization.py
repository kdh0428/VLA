#!/usr/bin/env python
"""
Feedback stabilisation with NON-oracle motion references (GPU 1, natural/fast only).

Experiments 9-11 corrected the post-mismatch context with GT tokens and showed that any token
with GT-like MOTION works. A deployable system has no GT, so here the corrected context comes
from references available at inference time, tokenised to the codebook by motion:

  gt     GT tokens                                                     (oracle, = temporal window)
  kin    constant-turn-rate-and-acceleration extrapolation of the current ego state
         (speed, longitudinal acceleration, yaw rate of the last history interval), tokenised with
         the same routine that produces the GT tokens
  prev   the model's own plan from the previous frame (0.5 s earlier, natural path, fresh prompt),
         shifted by one step: tokens are local per-step motions, so prev_plan[j + 1] is the motion
         the previous plan intended for current step j

Harness and window definition are exactly temporal_feedback_window's: equal-distance units,
prefix KV cache, context position t* + o holds ref[t* + o] for 1 <= o < 1 + w and the model's own
token otherwise; executed actions are always the model's own samples.

ROWS  normal, {gt, kin, prev} x {w2, w4, all}      ("all" = every post-t* context position)
The timing (t*) is still oracle; only the CONTENT of the correction is non-oracle here.
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
from _planner import AUTOVLA_DIR, N_ACT, POC_DIR, Planner, ctra_poses, seed_of   # noqa: E402

SRCS = ["gt", "kin", "prev"]
ALL_SRCS = ["gt", "kin", "prev", "prev2", "prev3", "consmed", "consmean", "consmedkin", "pdm", "prevpdm"]
WINS = {"w2": 2, "w4": 4, "all": 99}


def consensus(cands, cb, trained, how):
    """Per-step motion consensus of candidate token lists (codebook-space medoid or snapped mean)."""
    out = []
    for j in range(N_ACT):
        c = [x[j] for x in cands]
        if how == "med":
            out.append(min(c, key=lambda a: sum(np.linalg.norm(cb[a] - cb[b]) for b in c)))
        else:
            mu = np.mean([cb[a] for a in c], 0)
            out.append(int(trained[np.argmin(np.linalg.norm(cb[trained] - mu, axis=1))]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ed-records", default=os.path.join(POC_DIR, "outputs/equal_distance_perturbation/records.jsonl"))
    ap.add_argument("--full-records", default=os.path.join(POC_DIR, "outputs/full_extract/records.jsonl"))
    ap.add_argument("--index", default=os.path.join(POC_DIR, "outputs/frame_index/index.json"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--frame-scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc_frames"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/reference_stabilization"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-alts", type=int, default=0)
    ap.add_argument("--sources", default=",".join(SRCS), help=f"subset of {ALL_SRCS}")
    ap.add_argument("--windows", default=",".join(WINS))
    ap.add_argument("--temperature", type=float, default=None, help="override the config's sampling temperature (0.01)")
    ap.add_argument("--pdm-refs", default=os.path.join(POC_DIR, "outputs/frame_index/pdm_refs.json"))
    args = ap.parse_args()
    srcs = args.sources.split(",")
    wins = {w: WINS[w] for w in args.windows.split(",")}
    ROWS = ["normal"] + [f"{s}_{w}" for s in srcs for w in wins]
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1") and not os.environ.get("VLA_ANY_GPU"):
        raise SystemExit("launch with CUDA_VISIBLE_DEVICES=1 (RTX 5090) or 0 (RTX 3080 Ti); all rows of a run share one GPU")
    for k in ("ed_records", "full_records", "index", "scenes", "frame_scenes", "output"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.makedirs(args.output, exist_ok=True)
    if os.path.exists(os.path.join(args.output, "records.jsonl")) and not args.limit:
        raise SystemExit(f"{args.output}/records.jsonl exists; refusing to overwrite")

    B = len(ROWS)
    ed = [json.loads(l) for l in open(args.ed_records)]
    ed.sort(key=lambda r: (r["group"] != "A-", r["token"]))
    if args.limit:
        a_m = [r for r in ed if r["group"] == "A-"][:max(1, args.limit // 2)]
        ed = a_m + [r for r in ed if r["group"] == "A+"][:args.limit - len(a_m)]
    want = {r["token"] for r in ed}
    stub_of = {}
    for line in open(args.full_records):
        x = json.loads(line)
        if x["token"] in want:
            stub_of[x["token"]] = x["arms"]["N"]["token_ids"]
    index = json.load(open(args.index))
    print(f"[plan] {len(ed)} scenes (A- {sum(r['group']=='A-' for r in ed)}), {B} rows", flush=True)

    pl = Planner()
    if args.temperature is not None:
        pl.temp = float(args.temperature)          # action rows AND the previous-frame plans
        pl.gen_conf = dict(pl.gen_conf, temperature=float(args.temperature))
    llm, A0, model = pl.llm, pl.A0, pl.model
    import pickle
    cb = np.asarray(pickle.load(open(os.path.join(AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl"), "rb"))["token_all"]["veh"],
                    np.float64).reshape(2048, -1)
    emb = llm.get_input_embeddings().weight[A0:A0 + 2048].detach().float().cpu().numpy()
    en = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    cosm = en @ en.T
    np.fill_diagonal(cosm, 0)
    trained = np.where(~(cosm > 0.999).any(1))[0]          # same untrained-group rule as experiment 11
    del cosm
    pdm_refs = json.load(open(args.pdm_refs)) if any(x.startswith(("pdm", "prevpdm")) for x in srcs) else None

    fout = open(os.path.join(args.output, "records.jsonl"), "w")
    ferr = open(os.path.join(args.output, "errors.jsonl"), "w")
    t0 = time.time()
    n_ok = n_fail = n_units = n_unit_fail = 0
    for si, r in enumerate(ed, 1):
        try:
            token, t, gt, pred = r["token"], r["t_star"], r["gt"], r["pred"]
            scene = json.load(open(os.path.join(args.scenes, f"{token}.json")))
            # ---- references ------------------------------------------------------------------
            kin_poses = ctra_poses(scene)
            ref = {"gt": list(gt), "kin": pl.tokenize(kin_poses)}
            past = dict(index.get(token, {}).get("past", {}))
            if index.get(token, {}).get("prev"):
                past.setdefault("1", index[token]["prev"])
            prev_plan = None
            need_k = {1} | ({2, 3} if any(x.startswith(("prev2", "prev3", "cons")) for x in srcs) else set())
            for kk in sorted(need_k):
                name_k = "prev" if kk == 1 else f"prev{kk}"
                tk = past.get(str(kk))
                path = os.path.join(args.frame_scenes, f"{tk}.json") if tk else None
                if path and os.path.exists(path):
                    pk = pl.plan(json.load(open(path)), key=(args.seed, token, "prevplan") if kk == 1 else (args.seed, token, f"prevplan{kk}"))
                    if kk == 1:
                        prev_plan = pk
                    tt = pk["tokens"]
                    ref[name_k] = tt[kk:] + tt[-1:] * kk          # plan made kk steps ago, shifted by kk
                else:
                    ref[name_k] = None
            ps = [ref.get(x) for x in ("prev", "prev2", "prev3")]
            ok3 = all(x is not None for x in ps)
            ref["consmed"] = consensus(ps, cb, trained, "med") if ok3 else None
            ref["consmean"] = consensus(ps, cb, trained, "mean") if ok3 else None
            ref["consmedkin"] = consensus(ps + [ref["kin"]], cb, trained, "med") if ok3 else None
            if pdm_refs is not None:
                pr = pdm_refs.get(token)
                ref["pdm"] = pl.tokenize(pr["poses"]) if pr else None
                ref["prevpdm"] = consensus([ref["prev"], ref["pdm"]], cb, trained, "mean") \
                    if (ref["pdm"] is not None and ref["prev"] is not None) else None
            gt_xy = np.asarray(r["trajectory_gt"], float)[:, :2]
            ref_quality = {}
            for s in srcs:
                if ref.get(s) is None:
                    continue
                e = np.linalg.norm(pl.decode(ref[s])[:, :2] - gt_xy, axis=1)
                ref_quality[s] = {"ade": float(e.mean()), "fde": float(e[-1]),
                                  "token_eq_gt": float(np.mean([a == b for a, b in zip(ref[s], gt)]))}

            # ---- shared prefix (same as every harness experiment) --------------------------
            pl._vis.clear()
            feats = {}
            for b in pl.agent.get_feature_builders():
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
                for k in range(t + 1, N_ACT):
                    spans = []
                    for i, name in enumerate(ROWS):
                        ctx = []
                        for o, j in enumerate(range(t + 1, k), start=1):
                            if name == "normal":
                                ctx.append(choices[i][o - 1])
                                continue
                            s, w = name.split("_")
                            rs = ref.get(s)
                            use = rs is not None and o < 1 + WINS[w]
                            ctx.append(rs[j] if use else choices[i][o - 1])
                        spans.append([A0 + ptok] + [A0 + a for a in ctx])
                    m = len(spans[0])
                    sids = torch.tensor(spans, device="cuda:0")
                    with torch.no_grad():
                        o_ = llm(input_ids=sids, attention_mask=torch.ones((B, P + m), device="cuda:0", dtype=torch.long),
                                 past_key_values=cache, use_cache=True, cache_position=torch.arange(P, P + m, device="cuda:0"))
                    cache.crop(P)
                    lg = o_.logits[:, -1, A0:A0 + 2048].double()
                    del o_
                    lp = torch.log_softmax(lg, -1)
                    ent = -(lp.exp() * lp).sum(-1)
                    seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{token}:{pname}:{k}".encode()).digest()[:4], "little")
                    probs_t = torch.softmax(lg / pl.temp, -1).cpu()
                    gen = torch.Generator()
                    for i, name in enumerate(ROWS):
                        gen.manual_seed(seed)
                        choices[i].append(int(torch.multinomial(probs_t[i], 1, generator=gen)))
                        ents[name].append(round(float(ent[i]), 5))
                assert cache.get_seq_length() == P
                conds = {}
                for i, name in enumerate(ROWS):
                    acts = list(pred[:t]) + [ptok] + choices[i]
                    conds[name] = {"action_idx": acts, "trajectory_pred": pl.decode(acts)[:, :2].tolist(),
                                   "entropy_steps": ents[name]}
                fout.write(json.dumps({"token": token, "group": r["group"], "t_star": t, "log": r["log"], "perturbation": pname,
                                       "forced_token": ptok, "gt": gt, "trajectory_gt": r["trajectory_gt"], "conditions": conds,
                                       "ref": ref, "ref_quality": ref_quality, "prev_available": ref["prev"] is not None,
                                       "refs_available": {x: ref.get(x) is not None for x in srcs},
                                       "prev_plan_cot": prev_plan["cot"] if prev_plan else None}) + "\n")
                fout.flush()
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
    json.dump({"n_scenes_ok": n_ok, "n_scene_fail": n_fail, "n_units": n_units, "n_unit_fail": n_unit_fail, "rows": ROWS,
               "sources": srcs, "windows": wins, "temperature": pl.temp, "seed": args.seed, "gpu": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "elapsed_min": (time.time() - t0) / 60}, open(os.path.join(args.output, "run_meta.json"), "w"), indent=2)
    print(f"[done] scenes ok={n_ok} fail={n_fail} units={n_units} unit_fail={n_unit_fail}", flush=True)


if __name__ == "__main__":
    main()
