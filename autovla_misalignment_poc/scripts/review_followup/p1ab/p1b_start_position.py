#!/usr/bin/env python
"""
P1-B: same correction budget, different start position (GPU, RTX 5090 only).

Units: the equal-distance set exactly as exps 7-11 (208 scenes, 1,408 units = scene x forced token f
at t*). Executed output of every row = pred[:t*] + [f] + own samples (context-only corrections, as
exps 7-10). Harness and seed key as exp 9 (sha256(f"{seed}:{token}:{pert}:{k}"), shared by all rows
of a unit); 10-token horizon only (in-distribution).

Context position j = t* + o (o >= 1) at step k (positions t*+1 .. k-1 are read when producing a_k):
  persistent  P_{src}_s{s}_w{w}: o in [s, s+w-1] holds src[j] at every step (exp 9 definition: a
              corrected position stays corrected in the context); w corrected positions.
  transient   T_{src}_s{s}_w{w}: src[j] only while j is the MOST RECENT context position (j = k-1) and
              o in [s, s+w-1]; earlier corrected positions revert to own tokens (exp 7 Recent-GT rule
              restricted to w steps); w corrected forwards, each with one corrected position.
  s in {1,2,3,4}, w in {1,2,3};  src in {gt, self}  (self = fixed model reference from p1_reference.py)
  anchors: normal, full_gt, full_self (all positions), recent_gt, recent_self (sliding, all steps)
Per row the corrected (step, position) pairs are logged; windows reaching beyond position 8 are
flagged 'truncated' (only the common subset t* <= 2 has every window complete).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1_common as C  # noqa: E402

STARTS = (1, 2, 3, 4)
LENGTHS = (1, 2, 3)


def build_rows():
    rows = [{"name": "normal", "kind": "none", "src": None},
            {"name": "full_gt", "kind": "full", "src": "gt"}, {"name": "full_self", "kind": "full", "src": "self"},
            {"name": "recent_gt", "kind": "recent", "src": "gt"}, {"name": "recent_self", "kind": "recent", "src": "self"}]
    for kind, tag in (("persistent", "P"), ("transient", "T")):
        for src in ("gt", "self"):
            for s in STARTS:
                for w in LENGTHS:
                    rows.append({"name": f"{tag}_{src}_s{s}_w{w}", "kind": kind, "src": src, "s": s, "w": w})
    return rows


def ctx_tokens(row, k, t, f, own, gt, ref):
    toks, src_tags = [f], ["forced"]
    for j in range(t + 1, k):
        o = j - t
        kind = row["kind"]
        use = False
        if kind == "full":
            use = True
        elif kind == "recent":
            use = j == k - 1
        elif kind == "persistent":
            use = row["s"] <= o <= row["s"] + row["w"] - 1
        elif kind == "transient":
            use = j == k - 1 and row["s"] <= o <= row["s"] + row["w"] - 1
        if use:
            toks.append(gt[j] if row["src"] == "gt" else ref[j]); src_tags.append(row["src"])
        else:
            toks.append(own[o - 1]); src_tags.append("own")
    return toks, src_tags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-scenes", type=int, default=0)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--max-alts", type=int, default=0)
    ap.add_argument("--trace-units", type=int, default=24)
    args = ap.parse_args()
    C.gpu_guard()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "records.jsonl")):
        raise SystemExit(f"{out}/records.jsonl exists; refusing to overwrite")
    man = json.load(open(os.path.join(args.reference, "manifest.json")))
    ref_path = os.path.join(args.reference, "references.jsonl")
    ref_sha = C.sha256_file(ref_path)
    if man["files"]["references.jsonl"] != ref_sha:
        raise SystemExit("reference sha256 mismatch")
    REF = {x["token"]: x for x in map(json.loads, open(ref_path))}

    ROWS = build_rows()
    B = len(ROWS)
    ed = C.load_ed(args.limit_scenes, args.tokens.split(",") if args.tokens else None)
    ed = [r for r in ed if r["token"] in REF]
    stub_of, _, _ = C.load_stubs_and_token_freq({r["token"] for r in ed})
    torch, model, llm, A0, temp, agent, vis_box, cfg = C.load_model()
    gpu = C.assert_5090(torch)
    fout = open(os.path.join(out, "records.jsonl"), "w")
    ftr = open(os.path.join(out, "intervention_trace.jsonl"), "w")
    ferr = open(os.path.join(out, "errors.jsonl"), "w")
    t0 = time.time()
    n_units = n_traced = 0
    unit_times = []
    for si, r in enumerate(ed, 1):
        token, t, gt, pred = r["token"], r["t_star"], r["gt"], r["pred"]
        ref = REF[token]["ref_tokens"]
        try:
            cache, P, chk, _ = C.build_prefix_cache(torch, model, llm, agent, vis_box, A0, token, stub_of[token], pred[:t], B)
        except Exception as exc:
            ferr.write(C.jdump({"token": token, "stage": "scene", "error": repr(exc)}) + "\n"); ferr.flush()
            continue
        perts = [("original", pred[t])] + [(f"alt{j}", a["token"]) for j, a in enumerate(r["alternatives"])]
        if args.max_alts:
            perts = perts[:1 + args.max_alts]
        for pname, f in perts:
            tu = time.time()
            try:
                own = [[] for _ in range(B)]
                ents = [[] for _ in range(B)]
                pgt = [[] for _ in range(B)]
                pref = [[] for _ in range(B)]
                corrected = [[] for _ in range(B)]          # (k, j, token, differs_from_own)
                trace_this = n_traced < args.trace_units
                steps_trace = []
                for k in range(t + 1, C.N_ACT):
                    spans, tags = [], []
                    for i, row in enumerate(ROWS):
                        toks, st = ctx_tokens(row, k, t, f, own[i], gt, ref)
                        spans.append([A0 + a for a in toks]); tags.append(st)
                        for jj, tg in enumerate(st[1:]):
                            if tg != "own":
                                j = t + 1 + jj
                                corrected[i].append([k, j, toks[1 + jj], toks[1 + jj] != own[i][jj]])
                    full = C.forward_spans(torch, llm, cache, P, spans)
                    lg = full[:, A0:A0 + C.N_TOK]
                    lp = torch.log_softmax(lg, -1)
                    ent = -(lp.exp() * lp).sum(-1)
                    picks = C.sample_rows(torch, lg, temp, C.seed_int(args.seed, token, pname, k))
                    for i in range(B):
                        own[i].append(picks[i])
                        ents[i].append(round(float(ent[i]), 5))
                        pgt[i].append(round(float(lp[i, gt[k]].exp()), 6))
                        pref[i].append(round(float(lp[i, ref[k]].exp()), 6))
                    cc = C.check_cache(cache, P, chk)
                    if trace_this:
                        steps_trace.append({"k": k, "span_len": len(spans[0]), "cache_len_after_crop": cache.get_seq_length(),
                                            "prefix_checksum_ok": cc["prefix_checksum_ok"],
                                            "rows": {ROWS[i]["name"]: {"context_tokens_pos_t..k-1": [x - A0 for x in spans[i]],
                                                                       "context_source": tags[i], "sampled": picks[i]}
                                                     for i in range(B)}})
                rows_out = {}
                locked_all = True
                for i, row in enumerate(ROWS):
                    acts = list(pred[:t]) + [f] + own[i]
                    locked_all &= acts[:t + 1] == list(pred[:t]) + [f]
                    poses, origin = C.decode(torch, model, A0, acts)
                    win = None
                    if row["kind"] in ("persistent", "transient"):
                        last = t + row["s"] + row["w"] - 1
                        win = {"s": row["s"], "w": row["w"], "positions": [t + o for o in range(row["s"], row["s"] + row["w"])],
                               "truncated": last > C.N_ACT - 2, "release_pose": t + row["s"] + row["w"]}
                    rows_out[row["name"]] = {"action_idx": acts, "traj_xyh": np.round(poses, 6).tolist(), "ent": ents[i],
                                             "p_gt": pgt[i], "p_ref": pref[i], "window": win,
                                             "n_corrected_forwards": len({c[0] for c in corrected[i]}),
                                             "n_corrected_positions": len({c[1] for c in corrected[i]}),
                                             "n_corrected_pairs": len(corrected[i]),
                                             "n_corrected_pairs_effective": int(sum(c[3] for c in corrected[i])),
                                             "corrected_pairs": corrected[i] if trace_this else None}
                rec = {"unit_id": f"{token}|{pname}", "token": token, "log": r["log"], "group": r["group"], "t_star": t,
                       "perturbation": pname, "forced_token": f, "seed": args.seed, "seed_key": "perturbation (exp 9 rule)",
                       "gt": gt, "trajectory_gt": r["trajectory_gt"], "ref_tokens": ref, "ref_traj_xyh": REF[token]["ref_traj_xyh"],
                       "rows": rows_out, "checks": {"output_prefix_locked_all_rows": bool(locked_all)}, "prefix_len": P}
                fout.write(C.jdump(rec) + "\n"); fout.flush()
                if trace_this:
                    ftr.write(C.jdump({"unit_id": rec["unit_id"], "t_star": t, "forced_token": f, "prefix_len": P,
                                       "steps": steps_trace, "checks": rec["checks"]}) + "\n"); ftr.flush()
                    n_traced += 1
                n_units += 1
                unit_times.append(time.time() - tu)
            except Exception as exc:
                import traceback; traceback.print_exc()
                ferr.write(C.jdump({"token": token, "unit": pname, "stage": "unit", "error": repr(exc)}) + "\n"); ferr.flush()
        del cache
        if si % 10 == 0 or args.limit_scenes or args.tokens:
            el = time.time() - t0
            print(f"[p1b] {si}/{len(ed)} scenes units={n_units} {el/si:.1f}s/scene eta {(len(ed)-si)*el/si/60:.1f} min", flush=True)
    fout.close(); ftr.close(); ferr.close()
    meta = C.env_record({"script": os.path.abspath(__file__), "rows": ROWS, "n_scenes": len(ed), "n_units": n_units,
                         "seed": args.seed, "temperature": temp, "batch_rows": B, "reference_dir": os.path.abspath(args.reference),
                         "reference_sha256": ref_sha, "elapsed_min": (time.time() - t0) / 60,
                         "sec_per_unit_mean": float(np.mean(unit_times)) if unit_times else None,
                         "gpu_checked": gpu, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9})
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=1)
    print(f"[done] units={n_units} elapsed {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
