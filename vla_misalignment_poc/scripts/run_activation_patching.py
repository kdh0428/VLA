#!/usr/bin/env python
"""
Activation patching / steering on the planning token, with controls.

For each failure sample and each candidate layer l:
    bad input -> blocks 1..l-1 -> WRITE h_l[<waypoint_ego>] -> blocks l..L -> planner
The write is either
    patch  : h_l <- h_l(donor)      donor = a matched P+R+A+ sample
    steer  : h_l <- h_l + alpha*d   d = mean(correct) - mean(failure) at layer l
    random : h_l <- h_l + alpha*r   r = random direction, SAME NORM as d   (control)

Everything after the write is the model's own computation and the planner still receives a
genuine final-layer vector, so a trajectory change is attributable to the representation
rather than to feeding the planner an out-of-distribution input.

Also measured (spec 14.8): COLLATERAL DAMAGE -- the same intervention applied to samples
that were already correct, to see how many it breaks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORION_DIR = "/root/VLA/orion"
sys.path.insert(0, POC_DIR)
sys.path.insert(0, ORION_DIR)

import analysis.numpy_compat  # noqa: E402,F401
from analysis.activation_patching import (make_random_direction, patch_planning_token,  # noqa: E402
                                          steering_direction)
from analysis.orion_hooks import _resolve_causal_lm, _waypoint_mask, install_planning_capture  # noqa: E402
from analysis.probing import build_matrix, load_hidden  # noqa: E402
from analysis.trajectory_metrics import action_correct, action_semantics  # noqa: E402

sys.path.insert(0, os.path.join(POC_DIR, "scripts"))
from run_orion_hidden_extract import (LMToggle, build_everything, collate_one,  # noqa: E402
                                      decumsum, first_tensor, index_dataset, to_numpy_traj)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", default=os.path.join(POC_DIR, "outputs", "taxonomy", "taxonomy.json"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs", "intervention"))
    ap.add_argument("--config", default=os.path.join(POC_DIR, "configs", "orion_poc_cot_fp16.py"))
    ap.add_argument("--checkpoint", default=os.path.join(ORION_DIR, "ckpts", "Orion.pth"))
    ap.add_argument("--layers", default="", help="layers to patch; default = a spread")
    ap.add_argument("--modes", default="patch,steer,random",
                    help="subset of patch,steer,random")
    ap.add_argument("--alphas", default="1.0,2.0,4.0")
    ap.add_argument("--max-failures", type=int, default=40)
    ap.add_argument("--max-controls", type=int, default=40,
                    help="already-correct samples, for collateral damage")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def run_llm_and_plan(wrapped, data):
    """One full forward; returns (pred_delta, model_rounds_text_out)."""
    with torch.no_grad():
        out = wrapped(data, return_loss=False)
    res = out["bbox_results"][0] if isinstance(out, dict) else out[0]
    pts = res.get("pts_bbox", {})
    return decumsum(to_numpy_traj(pts.get("ego_fut_preds"))), res.get("text_out", [])


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with open(args.taxonomy) as f:
        rows = json.load(f)
    rows = [r for r in rows if r.get("hidden_path") and os.path.exists(r["hidden_path"])]

    fails = [r for r in rows if r["P"] is True and r["A"] is False]
    ctrls = [r for r in rows if r["P"] is True and r["A"] is True]
    print(f"P+A- failures={len(fails)}  P+A+ controls={len(ctrls)}")
    if len(fails) < 5 or len(ctrls) < 5:
        raise SystemExit("not enough samples for intervention")

    cfg, dataset, model, wrapped = build_everything(args)
    capture = install_planning_capture(model, layers=None)
    _, per_clip = index_dataset(dataset)
    key2idx = {(c, fi): di for c, lst in per_clip.items() for fi, di in lst}
    causal_lm = _resolve_causal_lm(model.lm_head)
    n_blocks = len(causal_lm.model.layers)

    # A patch at layer L rewrites the INPUT to block L, so blocks L..n_blocks-1 still run
    # on the modified state. Valid L is therefore 1..n_blocks-1. L = n_blocks would be the
    # final hidden state with no transformer left to re-process it -- that is just
    # replacing the planner's input directly, the out-of-distribution query this design
    # deliberately avoids.
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else sorted({max(1, int(round((n_blocks - 1) * f)))
                           for f in (0.25, 0.5, 0.75, 0.94)}))
    dropped = [L for L in layers if not (1 <= L < n_blocks)]
    layers = [L for L in layers if 1 <= L < n_blocks]
    if dropped:
        print(f"[note] dropped out-of-range patch layers {dropped} "
              f"(valid 1..{n_blocks - 1}; {n_blocks} would bypass all remaining blocks)")
    modes = args.modes.split(",")
    alphas = [float(a) for a in args.alphas.split(",")]
    print(f"blocks={n_blocks} patch layers={layers} modes={modes} alphas={alphas}")

    # Steering directions per layer, from the FULL groups (not the sampled subsets).
    #
    # IMPORTANT: keep the RAW mean-difference, do not unit-normalise it. The residual-stream
    # norm grows steeply with depth (measured ||h||: 10.9 @L8 -> 95.5 @L30), so a fixed
    # alpha on a unit vector is a ~9% perturbation early and only ~1% deep -- an earlier run
    # done that way produced no effect at any layer simply because the deep interventions
    # were negligible. With the raw difference, alpha=1 means "move by the full observed
    # correct-minus-failure offset at this layer", which is comparable across depth.
    dirs: dict[int, np.ndarray] = {}
    dir_scale: dict[int, dict] = {}
    for L in layers:
        Xg, _ = build_matrix(ctrls, L, "mean")
        Xb, _ = build_matrix(fails, L, "mean")
        if len(Xg) >= 5 and len(Xb) >= 5:
            dirs[L] = steering_direction(Xg, Xb, normalise=False)
            hn = float(np.linalg.norm(Xb, axis=1).mean())
            dn = float(np.linalg.norm(dirs[L]))
            dir_scale[L] = {"hidden_norm": hn, "dir_norm": dn,
                            "frac_of_norm_at_alpha1": dn / max(hn, 1e-8)}
            print(f"  L{L}: ||h||={hn:.2f} ||d||={dn:.3f} "
                  f"(alpha=1 moves {100*dn/max(hn,1e-8):.1f}% of the state)")

    sel_fail = list(rng.permutation(len(fails))[:args.max_failures])
    sel_ctrl = list(rng.permutation(len(ctrls))[:args.max_controls])
    targets = ([("failure", fails[i]) for i in sel_fail]
               + [("control", ctrls[i]) for i in sel_ctrl])
    # Walk targets in (clip, frame) order so the temporal memory can be advanced
    # INCREMENTALLY. Resetting and replaying each clip from frame 0 per target would
    # dominate the runtime (~30 s of warm-up per sample); in clip order we replay each
    # clip at most once.
    targets.sort(key=lambda kv: (kv[1]["clip_id"], kv[1]["frame_idx"]))
    print(f"running {len(targets)} samples x {len(layers)} layers "
          f"({len({t[1]['clip_id'] for t in targets})} clips)")

    # Donor pool for hard patching, keyed by GT *longitudinal* action only.
    # Keying on (scenario, full GT action) as in the first run left almost no donors --
    # only 1 of 25 failures found a match -- because control and failure samples sit in
    # largely different scenarios. Matching on the longitudinal action keeps the donor
    # semantically appropriate ("this is what a correct STOP looks like here") while
    # actually yielding a pool. Same-clip donors are still excluded at draw time.
    donors: dict[str, list[dict]] = defaultdict(list)
    for c in ctrls:
        donors[(c.get("high_level_action_gt") or "").split("+")[0]].append(c)
    print("donor pool sizes:", {k: len(v) for k, v in sorted(donors.items())})

    results = []
    t0 = time.time()
    cur_clip, cur_frame = None, -1
    for n, (kind, r) in enumerate(targets, 1):
        clip, fi = r["clip_id"], r["frame_idx"]
        if (clip, fi) not in key2idx:
            continue
        didx = key2idx[(clip, fi)]

        # Advance this clip's temporal memory up to `fi`. Because targets are visited in
        # (clip, frame) order we only reset when the clip changes, and then replay just
        # the frames between the previous target and this one.
        if clip != cur_clip:
            if model.with_pts_bbox:
                model.pts_bbox_head.reset_memory()
            if model.with_map_head:
                model.map_head.reset_memory()
            cur_clip, cur_frame = clip, -1
        for wfi, wdi in per_clip[clip]:
            if wfi <= cur_frame:
                continue
            if wfi >= fi:
                break
            with torch.no_grad(), LMToggle(model):
                wrapped(collate_one(dataset, wdi), return_loss=False)
        cur_frame = fi

        data = collate_one(dataset, didx)
        base_pred, _ = run_llm_and_plan(wrapped, data)
        # Taxonomy rows carry metrics, not trajectories -- read the GT straight from the
        # dataset sample we just built (per-step deltas, current-LiDAR frame).
        gt_delta = to_numpy_traj(first_tensor(data.get("ego_fut_trajs")))
        gtp = {"traffic_light": r.get("gt_traffic_light"),
               "lead_vehicle": r.get("gt_lead_vehicle"),
               "pedestrian": r.get("gt_pedestrian")}
        v0 = action_correct(base_pred, gt_delta, gtp)

        # The waypoint mask for THIS frame, recorded by the capture hook during the
        # unpatched pass (positions are identical for the patched re-runs).
        mask = capture.last_mask

        for L in layers:
            for mode in modes:
                alist = [0.0] if mode == "patch" else alphas
                for alpha in alist:
                    kw = {}
                    if mode == "patch":
                        key = (r.get("high_level_action_gt") or "").split("+")[0]
                        pool = [d for d in donors.get(key, []) if d["clip_id"] != clip]
                        if not pool:
                            continue
                        donor = pool[int(rng.integers(len(pool)))]
                        rep = load_hidden(donor["hidden_path"], L, "mean")
                        if rep is None:
                            continue
                        kw["replacement"] = torch.from_numpy(rep)
                    elif mode == "steer":
                        if L not in dirs:
                            continue
                        kw["direction"] = torch.from_numpy(dirs[L])
                        kw["alpha"] = alpha
                    else:
                        if L not in dirs:
                            continue
                        kw["direction"] = torch.from_numpy(
                            make_random_direction(dirs[L], seed=args.seed + L))
                        kw["alpha"] = alpha

                    if mask is None:
                        continue
                    try:
                        with patch_planning_token(causal_lm, L, mask, **kw):
                            pred, _ = run_llm_and_plan(wrapped, data)
                    except Exception as exc:
                        print(f"  [warn] {r['sample_id']} L{L} {mode}: {exc}", flush=True)
                        continue
                    v1 = action_correct(pred, gt_delta, gtp)
                    results.append({
                        "sample_id": r["sample_id"], "kind": kind, "scenario": r["scenario"],
                        "layer": L, "mode": mode, "alpha": alpha,
                        "ade_before": v0.ade, "ade_after": v1.ade,
                        "fde_before": v0.fde, "fde_after": v1.fde,
                        "action_before": v0.sem_pred, "action_after": v1.sem_pred,
                        "action_gt": v0.sem_gt,
                        "correct_before": v0.correct, "correct_after": v1.correct,
                        "recovered": (not v0.correct) and v1.correct,
                        "broken": v0.correct and (not v1.correct),
                    })
        if n % 5 == 0:
            el = time.time() - t0
            print(f"[prog] {n}/{len(targets)} samples, {len(results)} interventions, "
                  f"{el/60:.1f}min", flush=True)

    with open(os.path.join(args.output, "interventions.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)

    # --- summary: effect curve per (mode, layer, alpha) -----------------------------
    summary: dict[str, dict] = {}
    for row in results:
        k = f"{row['mode']}|L{row['layer']}|a{row['alpha']}"
        s = summary.setdefault(k, {"mode": row["mode"], "layer": row["layer"],
                                   "alpha": row["alpha"],
                                   "n_fail": 0, "recovered": 0, "d_ade_fail": [],
                                   "n_ctrl": 0, "broken": 0, "d_ade_ctrl": []})
        d = row["ade_after"] - row["ade_before"]
        if row["kind"] == "failure":
            s["n_fail"] += 1
            s["recovered"] += int(row["recovered"])
            s["d_ade_fail"].append(d)
        else:
            s["n_ctrl"] += 1
            s["broken"] += int(row["broken"])
            s["d_ade_ctrl"].append(d)
    for s in summary.values():
        s["mean_dADE_failure"] = float(np.mean(s["d_ade_fail"])) if s["d_ade_fail"] else float("nan")
        s["mean_dADE_control"] = float(np.mean(s["d_ade_ctrl"])) if s["d_ade_ctrl"] else float("nan")
        s["recovery_rate"] = s["recovered"] / s["n_fail"] if s["n_fail"] else float("nan")
        s["breakage_rate"] = s["broken"] / s["n_ctrl"] if s["n_ctrl"] else float("nan")
        s.pop("d_ade_fail"), s.pop("d_ade_ctrl")
    with open(os.path.join(args.output, "intervention_summary.json"), "w") as f:
        json.dump({"perturbation_scale": dir_scale, "results": summary}, f, indent=2)

    print("\n=== intervention summary (recovery / breakage) ===")
    for k in sorted(summary, key=lambda x: (summary[x]["mode"], summary[x]["layer"], summary[x]["alpha"])):
        s = summary[k]
        print(f"  {k:28s} rec={s['recovery_rate']:.2f} ({s['recovered']}/{s['n_fail']}) "
              f"brk={s['breakage_rate']:.2f} dADE_f={s['mean_dADE_failure']:+.3f} "
              f"dADE_c={s['mean_dADE_control']:+.3f}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
