#!/usr/bin/env python
"""
P1-C stage 1: natural-only pass (protocol §3, decision D7a). GPU 1 (RTX 5090), bf16, batch 1.

Procedure = `_planner.Planner.plan`, UNCHANGED: forced stem "<think>\\nThis is a", free decoding of the stub until the
first action token (HF generate, AutoVLA sample config, torch seed sha256("<seed>:<token>:stub")), then 10 action tokens
from the action rows at T = 0.01 with per-step seed sha256("<seed>:<token>:<k>") (key = f"{seed}:{token}").
The stub token ids are not returned by Planner.plan, so this script wraps `llm.generate` only to RECORD its output
(the call itself is untouched) and rebuilds the stub exactly as Planner.plan does (stem + new[:first action]).

Output: <output>/records.jsonl, one line per scene, in the `full_extract` arm-N layout that
equal_distance_perturbation / action_history_causal / prev_action_state_patching / temporal_feedback_window /
motion_semantics_ablation read (token, log_name, map_name, velocity, trajectory_gt, gt_action_idx,
arms.N.{token_ids, pred_action_idx, cot_present, runaway_action_tokens, trajectory_pred}), plus a "p1c" block with the
stage-1 labels (t*, a_eval, FDE5, eligibility, exclusion reason) computed from the natural plan only.
Arm C is not run. Planner emits exactly 10 action tokens, so runaway is 0 by construction (recorded as such).

Resumable: scenes already in records.jsonl are skipped; existing records are never rewritten.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1c_common as C  # noqa: E402

sys.path.insert(0, C.SCRIPTS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True, help="JSON list of scene tokens")
    ap.add_argument("--scenes", default=C.SCENES)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sensor-root", default=None, help="override sensor_blobs/test (P3 camera check only)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("pinned to the RTX 5090: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1")
    args.output = os.path.abspath(args.output)
    os.makedirs(args.output, exist_ok=True)
    rec_path = os.path.join(args.output, "records.jsonl")
    done = set()
    if os.path.exists(rec_path):
        done = {json.loads(l)["token"] for l in open(rec_path) if l.strip()}
    toks = [t for t in json.load(open(args.tokens)) if t not in done]
    if args.limit:
        toks = toks[:args.limit]
    print(f"[plan] {len(toks)} scenes to decode ({len(done)} already in {rec_path})", flush=True)
    if not toks:
        print("[done] nothing to do", flush=True)
        return

    import torch
    from _planner import Planner
    from analyze_action_history import a_eval

    t_load = time.time()
    pl = Planner()
    load_s = time.time() - t_load
    if args.sensor_root:
        from pathlib import Path
        pl.agent.sensor_data_path = Path(os.path.abspath(args.sensor_root))
    llm, A0 = pl.llm, pl.A0
    print(f"[load] {load_s:.0f}s gpu={torch.cuda.get_device_name(0)} dtype={next(llm.parameters()).dtype} "
          f"sensor={pl.agent.sensor_data_path}", flush=True)

    box = {}
    orig_generate = llm.generate

    def generate_recorded(*a, **kw):       # records the stub generation; the call is unchanged
        g = orig_generate(*a, **kw)
        box["seq"] = g[0].tolist()
        box["n_in"] = int(kw["input_ids"].shape[1])
        return g
    llm.generate = generate_recorded

    fout = open(rec_path, "a")
    ferr = open(os.path.join(args.output, "errors.jsonl"), "a")
    ftime = open(os.path.join(args.output, "timing.jsonl"), "a")
    t0 = time.time()
    n_ok = n_fail = 0
    for i, token in enumerate(toks, 1):
        ts = time.time()
        try:
            scene = json.load(open(os.path.join(args.scenes, f"{token}.json")))
            log = C.log_of_scene(scene)
            box.clear()
            res = pl.plan(scene, key=f"{args.seed}:{token}")
            new = box["seq"][box["n_in"]:]
            fa = next(j for j, x in enumerate(new) if x >= A0)
            stub = list(pl.stem) + new[:fa]
            assert len(stub) == res["stub_len"], "stub reconstruction mismatch"
            pred = [int(a) for a in res["tokens"]]
            traj = np.asarray(pl.decode(pred), np.float32)[:, :2].tolist()
            gt = pl.tokenize(scene["gt_trajectory"])
            stub_text = pl.tok.decode(stub)
            cot = bool(res["cot"])
            lab = C.stage1_labels(pred, gt, traj, scene["gt_trajectory"], cot, 0, a_eval)
            anno = next((json.load(open(p)) for p in (os.path.join(d, f"{token}.json") for d in C.ANNO_DIRS)
                         if os.path.exists(p)), {})
            rec = {
                "token": token, "log_name": log, "map_name": anno.get("map_name"),
                "instruction": scene.get("instruction"), "velocity": scene.get("velocity"),
                "acceleration": scene.get("acceleration"),
                "trajectory_gt": scene["gt_trajectory"],
                "gt_action_idx": gt, "gt_action_token_ids": [A0 + a for a in gt],
                "arms": {"N": {
                    "token_ids": stub + [A0 + a for a in pred],
                    "action_token_ids": [A0 + a for a in pred],
                    "pred_action_idx": pred,
                    "cot_present": cot,
                    "runaway_action_tokens": 0,
                    "truncated": False,
                    "trajectory_pred": traj,
                    "generated_text": stub_text,
                    "tensor_file": None,
                }},
                "p1c": {"stage": "stage1_natural", "procedure": "_planner.Planner.plan", "key": f"{args.seed}:{token}",
                        "seed": args.seed, "stub_len": len(stub), "entropy_steps": [round(float(e), 5) for e in res["entropy"]],
                        **lab, "runtime_s": round(time.time() - ts, 3)},
            }
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            import traceback; traceback.print_exc()
            ferr.write(json.dumps({"token": token, "stage": "stage1", "error": repr(exc)}) + "\n"); ferr.flush()
        ftime.write(json.dumps({"token": token, "s": round(time.time() - ts, 3)}) + "\n"); ftime.flush()
        if i % 25 == 0 or i == len(toks):
            el = time.time() - t0
            print(f"[prog] {i}/{len(toks)} ok={n_ok} fail={n_fail} {el/i:.2f}s/scene eta {(len(toks)-i)*el/i/60:.1f}min", flush=True)
    fout.close(); ferr.close(); ftime.close()
    meta_p = os.path.join(args.output, "run_meta.jsonl")
    with open(meta_p, "a") as f:
        f.write(json.dumps({"n_ok": n_ok, "n_fail": n_fail, "n_requested": len(toks), "load_s": round(load_s, 1),
                            "decode_s": round(time.time() - t0, 1), "seed": args.seed, "gpu": torch.cuda.get_device_name(0),
                            "torch": torch.__version__, "dtype": str(next(llm.parameters()).dtype),
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "sensor_root": str(pl.agent.sensor_data_path), "utc": time.strftime("%FT%TZ", time.gmtime())}) + "\n")
    print(f"[done] ok={n_ok} fail={n_fail} decode {(time.time()-t0)/60:.1f} min (load {load_s:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
