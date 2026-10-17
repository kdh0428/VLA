#!/usr/bin/env python
"""
B1 numerical sensitivity runner (RTX 5090 only; protocol outputs/review_followup4/B/B1/NUMERICAL_SENSITIVITY_PROTOCOL.md).

One model load; settings run sequentially: S0 (bf16, sdpa, experiment batch) -> S0r (repeat) -> SB (batch shape) ->
SE (eager attention) -> SF (fp32 compute, bf16 weights upcast exactly). Decoding t001 (experiments' T = 0.01 CRN).
Re-uses the P1-verified engine (scripts/review_followup3/p1/p1b_engine.py) unchanged; adds only
  - reverse_b1(): Reverse-N at B = 1 by token substitution of the B = 1 Normal row's token (C4-verified equivalence),
  - set_attn()/set_precision(): in-place backend / precision switches.
Stores per-row tokens, decoded plan and logit-difference summaries only (no logit arrays on disk).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/review_followup3/p1")
import p1b_engine as E  # noqa: E402
import p1b_harness as HN  # noqa: E402
C = E.C

H7_ROWS = ["normal", "recent_gt", "gt_history"]
H8_ROWS = ["normal", "gt_history", "recent_gt", "reverse@emb"]
P1A_ROWS = [r["name"] for r in E.path_rows("P1A")]


def disk_free_gb(p="/root"):
    return shutil.disk_usage(p).free / 1e9


def slim(res, keep):
    """per-row record: own, acts, poses, spans (relative), logits (in memory only)."""
    out = {}
    for n in keep:
        if n in res["rows"]:
            r = res["rows"][n]
            out[n] = {"own": [int(x) for x in r["own"]], "acts": [int(x) for x in r["acts"]], "poses": np.asarray(r["poses"], np.float64),
                      "span": r["span"], "logits": r["logits"], "margin": r["margin"], "ent": r["ent"], "B": r.get("B")}
    return out


def eff_spans(rows):
    """effective input of H8 reverse@emb = span with the last token replaced by the Normal row's token (layer-0 patch)."""
    if "reverse@emb" in rows and "normal" in rows:
        rv, nm = rows["reverse@emb"], rows["normal"]
        rv["eff_span"] = [s[:-1] + [nm["span"][i][-1]] if len(s) >= 2 else s for i, s in enumerate(rv["span"])]
    for r in rows.values():
        r.setdefault("eff_span", r["span"])
    return rows


def reverse_b1(eng, unit, ed, stub, normal_own, seed):
    """Reverse-N at B = 1: context = [p] + gt[t+1 .. k-2] + [normal_own(k-1)] (== H8 reverse@emb, layer-0 patch of the
    last position with the Normal row's token); same CRN key as H8."""
    torch, A0 = eng.torch, eng.A0
    token, t, p, key = unit["scene"], unit["t_p"], unit["forced_token"], unit["perturbation"]
    gt, pred = ed["gt"], ed["pred"]
    cache, P, chk, info, mi, prefix = eng.build_prefix(token, stub, pred[:t], 1)
    own, span, logits, margin, ent = [], [], [], [], []
    for k in range(t + 1, C.N_ACT):
        ctx = list(gt[t + 1:k - 1]) + ([normal_own[k - t - 2]] if k >= t + 2 else [])
        sp = [p] + ctx
        full = eng.fwd_span(cache, P, [[A0 + a for a in sp]])
        C.check_cache(cache, P, chk)
        lg = full[:, A0:A0 + C.N_TOK]
        lp = torch.log_softmax(lg, -1)
        srt = torch.sort(lg, -1, descending=True).values
        pick = C.sample_rows(torch, lg, eng.temp, C.seed_int(seed, token, key, k))[0]
        own.append(int(pick)); span.append(sp); logits.append(lg[0].float().cpu().numpy())
        margin.append(float(srt[0, 0] - srt[0, 1])); ent.append(float(-(lp.exp() * lp).sum(-1)[0]))
    del cache
    torch.cuda.empty_cache()
    acts = list(pred[:t]) + [p] + own
    poses, _ = C.decode(torch, eng.model, A0, acts)
    return {"own": own, "acts": acts, "poses": np.asarray(poses, np.float64), "span": span, "eff_span": span,
            "logits": np.stack(logits), "margin": margin, "ent": ent, "B": 1}


def set_attn(eng, impl):
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as M
    llm = eng.llm
    for layer in llm.model.layers:
        layer.self_attn.__class__ = M.QWEN2_5_VL_ATTENTION_CLASSES[impl]
    for blk in llm.visual.blocks:
        blk.attn.__class__ = M.QWEN2_5_VL_VISION_ATTENTION_CLASSES[impl]
        if not hasattr(blk.attn, "head_dim"):
            blk.attn.head_dim = blk.attn.qkv.in_features // blk.attn.num_heads
    for cfg in {id(llm.config): llm.config, id(llm.model.config): llm.model.config, id(llm.visual.config): llm.visual.config}.values():
        cfg._attn_implementation = impl
    llm.model._attn_implementation = impl
    return {"lm_attn_class": type(llm.model.layers[0].self_attn).__name__, "vis_attn_class": type(llm.visual.blocks[0].attn).__name__,
            "config_attn": llm.config._attn_implementation}


def set_precision(eng, dtype_name):
    torch = eng.torch
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype_name]
    n = 0
    with torch.no_grad():
        for p in eng.model.parameters():
            if p.dtype != dt and p.dtype in (torch.float32, torch.bfloat16):
                p.data = p.data.to(dt); n += 1
    torch.cuda.empty_cache()
    return {"params_cast": n, "param_dtypes": sorted({str(p.dtype) for p in eng.model.parameters()})}


def param_checksum(eng):
    torch = eng.torch
    s = 0.0
    with torch.no_grad():
        for p in list(eng.llm.parameters())[::40]:
            s += float(p.double().abs().sum())
    return s


def cmp_rows(a, b):
    """free-run comparison of one row: logits only on steps with identical effective input span."""
    n = min(len(a["own"]), len(b["own"]))
    steps = [i for i in range(n) if a["eff_span"][i] == b["eff_span"][i]]
    d = [float(np.abs(a["logits"][i].astype(np.float64) - b["logits"][i].astype(np.float64)).max()) for i in steps]
    fd = next((i for i in range(n) if a["own"][i] != b["own"][i]), None)
    flips = [i for i in steps if int(np.argmax(a["logits"][i])) != int(np.argmax(b["logits"][i]))]
    return {"n_steps": n, "n_steps_compared": len(steps), "logit_maxabs": max(d) if d else None,
            "logit_maxabs_per_step": d, "bitwise": (all(x == 0.0 for x in d) if d else None),
            "suffix_equal": a["own"] == b["own"], "first_div": fd, "argmax_flip_steps": flips,
            "margin_at_first_div_ref": (a["margin"][fd] if fd is not None else None),
            "endpoint_dev_m": float(np.linalg.norm(a["poses"][-1, :2] - b["poses"][-1, :2])),
            "pose_maxabs": float(np.abs(a["poses"] - b["poses"]).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--settings", default="S0,S0r,SB,SE,SF")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-scenes", type=int, default=0)
    ap.add_argument("--cap-min", type=float, default=90.0)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "records.jsonl")):
        raise SystemExit("records.jsonl exists; refusing to overwrite")
    if disk_free_gb() < 1.0:
        raise SystemExit("disk < 1 GB free")
    man, units, ed, stub_of = HN.prepare_units(args.manifest, False)
    if args.limit_scenes:
        keep = []
        for u in units:
            if u["scene"] not in keep:
                keep.append(u["scene"])
        keep = set(keep[:args.limit_scenes])
        units = [u for u in units if u["scene"] in keep]
    eng = E.Engine()
    settings = args.settings.split(",")
    meta = {"script": os.path.abspath(__file__), "argv": sys.argv, "units": [u["unit_id"] for u in units], "settings": settings,
            "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **eng.env(), "switches": {}, "skipped": {}}
    fr = open(os.path.join(out, "records.jsonl"), "w")
    fl = open(os.path.join(out, "logit_cmp.jsonl"), "w")
    store = {}            # (setting, unit_id, path, row) -> slim row
    t0 = time.time()
    timing = []
    ck0 = param_checksum(eng)
    kw = dict(decoding="t001", seed=args.seed)

    def emit(setting, u, path, rows):
        for n, r in rows.items():
            store[(setting, u["unit_id"], path, n)] = r
            fr.write(C.jdump({"setting": setting, "unit_id": u["unit_id"], "scene": u["scene"], "log": u["log"], "group": u["group"],
                              "stratum": u["stratum"], "t_p": u["t_p"], "path": path, "row": n, "B": r.get("B"),
                              "own": r["own"], "acts": r["acts"], "poses": np.round(r["poses"], 6).tolist(),
                              "ent": r["ent"], "margin": r["margin"]}) + "\n")
        fr.flush()

    for setting in settings:
        if (time.time() - t0) / 60 > args.cap_min:
            meta["skipped"][setting] = "GPU-time cap reached"; break
        if disk_free_gb() < 1.0:
            meta["skipped"][setting] = "disk < 1 GB"; break
        if setting == "SE":
            meta["switches"]["SE"] = set_attn(eng, "eager")
        if setting == "SF":
            need = 17e9
            ok = False
            for _ in range(30):
                free, _tot = eng.torch.cuda.mem_get_info()
                if free >= need:
                    ok = True; break
                time.sleep(30)
            if not ok:
                meta["skipped"]["SF"] = f"insufficient GPU memory for fp32 at experiment batch (free {free/1e9:.1f} GB)"
                continue
            meta["switches"]["SF"] = set_precision(eng, "fp32")
        ts = time.time()
        for u in units:
            e, stub = ed[u["scene"]], stub_of[u["scene"]]
            tu = time.time()
            ok_unit = False
            for attempt in range(21):
                try:
                    for path in u["paths"]:
                        if path == "H7":
                            if setting == "SB":
                                continue                                      # SB H7 rows = the H8 B = 93 rows (written under H8)
                            res = eng.decode("H7", u, e, stub, row_names=H7_ROWS, **kw)
                            emit(setting, u, "H7", eff_spans(slim(res, H7_ROWS)))
                        elif path == "H8":
                            if setting == "SB":
                                nb1 = store[("S0", u["unit_id"], "H7", "normal")]
                                rows = {n: store[("S0", u["unit_id"], "H7", n)] for n in ("normal", "gt_history", "recent_gt")}
                                rows["reverse@emb"] = reverse_b1(eng, u, e, stub, nb1["own"], args.seed)
                                emit("SB", u, "H8", {n: dict(r) for n, r in rows.items()})
                            else:
                                res = eng.decode("H8", u, e, stub, hooks="experiment", **kw)
                                emit(setting, u, "H8", eff_spans(slim(res, H8_ROWS)))
                        elif path == "P1A":
                            res = eng.decode("P1A", u, e, stub, batch=("1" if setting == "SB" else "exp"),
                                             row_names=(P1A_ROWS if setting == "SB" else None), **kw)
                            emit(setting, u, "P1A", eff_spans(slim(res, P1A_ROWS)))
                    ok_unit = True
                    break
                except eng.torch.OutOfMemoryError:
                    eng.torch.cuda.empty_cache()
                    print(f"[b1] OOM {setting} {u['unit_id']} attempt {attempt}; waiting 30 s", flush=True)
                    time.sleep(30)
            if not ok_unit:
                meta["skipped"].setdefault(f"{setting}_units_oom", []).append(u["unit_id"])
                continue
            # logit / suffix comparison vs S0 (same unit, same path, same row)
            if setting != "S0":
                for (s, uid, path, n), r in list(store.items()):
                    if s != setting or uid != u["unit_id"]:
                        continue
                    ref = store.get(("S0", uid, path, n))
                    if ref is not None:
                        fl.write(C.jdump({"setting": setting, "unit_id": uid, "path": path, "row": n, "B_ref": ref.get("B"),
                                          "B_alt": r.get("B"), **cmp_rows(ref, r)}) + "\n")
                fl.flush()
            timing.append({"setting": setting, "unit": u["unit_id"], "sec": time.time() - tu})
            print(f"[b1] {setting} {u['unit_id']} {time.time()-tu:.1f}s total {(time.time()-t0)/60:.1f} min", flush=True)
        meta.setdefault("setting_min", {})[setting] = (time.time() - ts) / 60
        if setting == "SE":
            meta["switches"]["SE_restore"] = set_attn(eng, "sdpa")
        if setting == "SF":
            meta["switches"]["SF_restore"] = set_precision(eng, "bf16")
            meta["param_checksum_restored_equal"] = param_checksum(eng) == ck0
        # free logits of non-reference settings to bound memory
        for key in [k for k in store if k[0] not in ("S0",)]:
            store[key] = {kk: vv for kk, vv in store[key].items() if kk != "logits"} | {"logits": None}
    fr.close(); fl.close()
    meta.update({"end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "elapsed_min": (time.time() - t0) / 60,
                 "timing": timing, "peak_mem_gb": eng.torch.cuda.max_memory_allocated() / 1e9, "param_checksum_start": ck0})
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=1, default=str)
    print(f"[done] {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
