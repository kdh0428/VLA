#!/usr/bin/env python
"""
Receding-horizon replanning on logged observations (GPU 1, natural/fast only).

Experiment 10 left one question open: stabilising the previous-action feedback delays but does not
prevent re-divergence, and beyond the 5 s plan nothing could be tested in distribution. A real
driving stack replans. Here the model replans every m steps from the LOGGED camera frames at the
replanning time, executes only the first m tokens of each plan, and the executed pieces are
stitched into an 8 s trajectory evaluated against the log.

  ego state fed to the prompt (speed / acceleration magnitudes only; the prompt has no history
  trajectory):
    log   the logged state at the replanning frame           (pure open-loop replanning)
    exec  the logged state corrected by the executed path:   v = v_log + (v_exec - v_log),
          finite differences over the last 0.5 s              (pseudo closed loop)
  cameras always come from the log, so the observation does not follow the executed deviation;
  the executed-vs-logged pose gap at every replanning point is recorded as the size of that
  inconsistency.

  stabilisation (prev): at every replan r >= 1, context positions 0 .. w-1 are replaced by the
  previous plan shifted by m (ref[j] = prev_plan[j + m]); executed tokens stay the model's own.
  No GT and no oracle timing is used.

CONDITIONS  ol (single plan, 5 s), m{2,4,8}_{log,exec}, m{2,4,8}_exec_prev (w = min(3, m - 1)).
Plan 0 is identical in every condition (logged state, no previous plan) and is computed once.
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
from _planner import AUTOVLA_DIR, DT, N_ACT, POC_DIR, Planner, to_global, to_local   # noqa: E402

EVAL = 16
CONDS = [("ol", 10, "log", 0)] + [(f"m{m}_{st}", m, st, 0) for m in (2, 4, 8) for st in ("log", "exec")] + \
        [(f"m{m}_exec_prev", m, "exec", min(3, m - 1)) for m in (2, 4, 8)]


def fd_speed_accel(poses):
    """speed and |accel| from the last three poses at DT (global frame)."""
    p = np.asarray(poses, float)
    v1 = np.hypot(*(p[-1, :2] - p[-2, :2])) / DT
    v0 = np.hypot(*(p[-2, :2] - p[-3, :2])) / DT if len(p) >= 3 else v1
    return v1, (v1 - v0) / DT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(POC_DIR, "outputs/frame_index/index.json"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--frame-scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc_frames"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/receding_horizon_replanning"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" and not os.environ.get("VLA_ANY_GPU"):
        raise SystemExit("pinned to GPU 1: launch with CUDA_VISIBLE_DEVICES=1")
    for k in ("index", "scenes", "frame_scenes", "output"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.makedirs(args.output, exist_ok=True)
    if os.path.exists(os.path.join(args.output, "records.jsonl")) and not args.limit:
        raise SystemExit(f"{args.output}/records.jsonl exists; refusing to overwrite")

    index = json.load(open(args.index))
    toks = sorted(index, key=lambda k: (index[k]["group"] != "A-", k))
    if args.limit:
        a_m = [k for k in toks if index[k]["group"] == "A-"][:max(1, args.limit // 2)]
        toks = a_m + [k for k in toks if index[k]["group"] == "A+"][:args.limit - len(a_m)]
    print(f"[plan] {len(toks)} scenes, conditions {[c[0] for c in CONDS]}", flush=True)

    pl = Planner()
    fout = open(os.path.join(args.output, "records.jsonl"), "w")
    ferr = open(os.path.join(args.output, "errors.jsonl"), "w")
    t0 = time.time()
    n_ok = n_fail = n_plans = 0

    def scene_at(f0, off):
        if off == 0:
            return json.load(open(os.path.join(args.scenes, f"{f0}.json")))
        tk = index[f0]["future"].get(str(off))
        p = os.path.join(args.frame_scenes, f"{tk}.json") if tk else None
        return json.load(open(p)) if p and os.path.exists(p) else None

    for si, f0 in enumerate(toks, 1):
        try:
            ent = index[f0]
            log_pose = np.asarray(ent["log_pose"], float)          # frames f0 .. f0+16 (global)
            horizon = min(EVAL, len(log_pose) - 1)
            sc0 = scene_at(f0, 0)
            plan0 = pl.plan(sc0, key=(args.seed, f0, 0, "plan0"))
            n_plans += 1
            rec = {"token": f0, "group": ent["group"], "log": ent["log"], "horizon_steps": horizon,
                   "log_pose": log_pose[:horizon + 1].tolist(), "conditions": {}}
            for name, m, state, w in CONDS:
                exec_g = [log_pose[0]]                            # executed global poses, step 0 = f0
                plans, gaps = [], []
                prev_tokens = None
                off = 0
                while off < (N_ACT if name == "ol" else horizon):
                    if off == 0:
                        p = plan0
                        sp = ac = None
                    else:
                        sc = scene_at(f0, off)
                        if sc is None or off >= len(log_pose):
                            break
                        sp = ac = None
                        if state == "exec":
                            hist_e = exec_g[-3:] if len(exec_g) >= 3 else exec_g
                            hist_l = log_pose[max(0, off - 2):off + 1]
                            if len(hist_e) >= 2 and len(hist_l) >= 2:
                                ve, ae = fd_speed_accel(hist_e if len(hist_e) >= 3 else [hist_e[0]] + list(hist_e))
                                vl, al = fd_speed_accel(hist_l if len(hist_l) >= 3 else [hist_l[0]] + list(hist_l))
                                sp = max(0.0, float(np.hypot(*sc["velocity"][:2])) + (ve - vl))
                                ac = max(0.0, float(np.hypot(*sc["acceleration"][:2])) + (ae - al))
                        ref, pos = None, ()
                        if w and prev_tokens is not None:
                            ref = prev_tokens[m:] + [prev_tokens[-1]] * m
                            pos = range(w)
                        p = pl.plan(sc, key=(args.seed, f0, off, name), speed=sp, accel=ac, ref=ref, ref_positions=pos)
                        n_plans += 1
                        gaps.append({"off": off, "pos_gap_m": float(np.hypot(*(exec_g[-1][:2] - log_pose[off][:2]))),
                                     "speed_in": sp, "speed_log": float(np.hypot(*sc["velocity"][:2]))})
                    n_exec = N_ACT if name == "ol" else min(m, horizon - off)
                    local = pl.decode(p["tokens"])[:n_exec]
                    exec_g.extend(to_global(local, exec_g[-1]))
                    plans.append({"off": off, "tokens": p["tokens"], "cot": p["cot"], "entropy": p["entropy"][:n_exec]})
                    prev_tokens = p["tokens"]
                    off += n_exec
                    if name == "ol":
                        break
                exec_g = np.asarray(exec_g)
                n = len(exec_g) - 1
                err = np.hypot(*(exec_g[1:n + 1, :2] - log_pose[1:n + 1, :2]).T)
                rec["conditions"][name] = {
                    "executed_local": to_local(exec_g[1:], log_pose[0]).tolist(), "n_steps": int(n),
                    "err": err.tolist(), "plans": plans, "replan_gaps": gaps}
            rec["log_local"] = to_local(log_pose[1:horizon + 1], log_pose[0]).tolist()
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            import traceback; traceback.print_exc()
            ferr.write(json.dumps({"token": f0, "error": repr(exc)}) + "\n"); ferr.flush()
        if si % 5 == 0 or args.limit:
            el = time.time() - t0
            print(f"[prog] {si}/{len(toks)} ok={n_ok} fail={n_fail} plans={n_plans} {el/si:.1f}s/scene "
                  f"eta {(len(toks)-si)*el/si/60:.1f}min", flush=True)
    fout.close(); ferr.close()
    json.dump({"n_ok": n_ok, "n_fail": n_fail, "n_plans": n_plans, "conditions": CONDS, "eval_steps": EVAL,
               "seed": args.seed, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "elapsed_min": (time.time() - t0) / 60},
              open(os.path.join(args.output, "run_meta.json"), "w"), indent=2)
    print(f"[done] ok={n_ok} fail={n_fail} plans={n_plans}", flush=True)


if __name__ == "__main__":
    main()
