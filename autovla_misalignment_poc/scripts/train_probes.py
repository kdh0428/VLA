#!/usr/bin/env python
"""
Layer-wise probes on AutoVLA hidden states, at prefill and at each action-generation step.

Two probe families, and the second one exists because of a mistake found in the ORION PoC:

  perception  h -> GT scene attribute (lead vehicle, pedestrian, critical side/motion).
              Labels come from navsim Annotations, independent of the model's output, so
              this is not circular.

  action      h -> coarse longitudinal action, decoded TWICE:
                 * `gt`   : the ground-truth action
                 * `pred` : the model's OWN predicted action
              Reporting only `gt` is misleading. A+/A- is *defined* by gt==pred agreement,
              so "gt is decodable in A+ but not in A-" largely restates the definition. The
              informative question is whether the failure group encodes ANY action
              confidently (high `pred`) while failing to encode the correct one (low `gt`).

Splits are grouped by nuPlan log (the AutoVLA analogue of ORION's clip grouping); scalers
are fit on train folds only; every probe reports a label-shuffled null on the same folds
and its clip-level support, so a score resting on a handful of logs is visibly flagged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

# Cap BLAS pools before numpy loads them (same reason as the ORION runner).
_T = os.environ.get("VLA_PROBE_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _T

import numpy as np  # noqa: E402

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, POC_DIR)

from src.probes.probing import probe_binary  # noqa: E402


def load_hidden(path: str, key: str) -> np.ndarray | None:
    """One (37, hidden) stack for a given position key ('prefill' / 'action_k')."""
    try:
        with np.load(path) as z:
            if key not in z.files:
                return None
            return z[key].astype(np.float32)
    except Exception:
        return None


def build_X(rows: list[dict], key: str, layer: int) -> tuple[np.ndarray, list[int]]:
    vecs, keep = [], []
    for i, r in enumerate(rows):
        hp = r.get("_hpath")
        if not hp:
            continue
        a = load_hidden(hp, key)
        if a is None or layer >= a.shape[0]:
            continue
        vecs.append(a[layer])
        keep.append(i)
    if not vecs:
        return np.zeros((0, 0), np.float32), []
    return np.stack(vecs), keep


def lon_binary(v):
    """STOP/DECELERATE -> 1, MAINTAIN/ACCELERATE -> 0 (the ORION `stop_or_slow` label)."""
    return 1.0 if v in ("STOP", "DECELERATE") else (0.0 if v in ("MAINTAIN", "ACCELERATE") else np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", default=os.path.join(POC_DIR, "outputs/taxonomy/taxonomy.json"))
    ap.add_argument("--hidden-dir", default=os.path.join(POC_DIR, "outputs/gpu1/hidden"))
    ap.add_argument("--outdir", default=os.path.join(POC_DIR, "outputs/probes"))
    ap.add_argument("--layers", default="0,4,8,12,16,20,24,28,32,36")
    ap.add_argument("--positions", default="prefill,action_0,action_3,action_6,action_9")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = json.load(open(args.taxonomy))
    for r in rows:
        hf = r.get("hidden_file")
        r["_hpath"] = os.path.join(os.path.dirname(args.hidden_dir), hf) if hf else None
    rows = [r for r in rows if r["_hpath"] and os.path.exists(r["_hpath"])]
    print(f"{len(rows)} samples with hidden states, {len({r['log_name'] for r in rows})} logs")

    layers = [int(x) for x in args.layers.split(",")]
    positions = args.positions.split(",")
    seeds = tuple(int(x) for x in args.seeds.split(","))

    ok = [r for r in rows if r["A"] is True]
    bad = [r for r in rows if r["A"] is False]
    print(f"A+={len(ok)}  A-={len(bad)}")

    groups_key = "log_name"
    perception_labels = {
        "lead_vehicle": lambda r: (1.0 if r["lead_vehicle"] is True
                                   else (0.0 if r["lead_vehicle"] is False else np.nan)),
        "pedestrian": lambda r: (1.0 if r["pedestrian"] is True
                                 else (0.0 if r["pedestrian"] is False else np.nan)),
        "critical_side_left": lambda r: (1.0 if r["critical_side"] == "left"
                                         else (0.0 if r["critical_side"] in ("center", "right") else np.nan)),
        "critical_motion_moving": lambda r: (1.0 if r["critical_motion"] == "moving"
                                             else (0.0 if r["critical_motion"] == "static" else np.nan)),
    }
    action_labels = {
        "action_gt": lambda r: lon_binary(r["lon_gt"]),
        "action_pred": lambda r: lon_binary(r["lon_pred"]),
    }

    out = {"config": vars(args),
           "group_sizes": {"all": len(rows), "A+": len(ok), "A-": len(bad)},
           "results": {}}

    for tag, subset in (("all", rows), ("A+", ok), ("A-", bad)):
        if len(subset) < 40:
            print(f"[skip] group {tag}: only {len(subset)} samples")
            continue
        print(f"\n=== group {tag} (n={len(subset)}) ===", flush=True)
        g_all = np.array([r[groups_key] for r in subset])
        out["results"][tag] = {}
        for pos in positions:
            out["results"][tag][pos] = {}
            for layer in layers:
                X, keep = build_X(subset, pos, layer)
                if len(keep) < 40:
                    continue
                gk = g_all[keep]
                entry = {"n": len(keep)}
                for name, fn in {**perception_labels, **action_labels}.items():
                    y = np.array([fn(subset[i]) for i in keep], dtype=float)
                    m = np.isfinite(y)
                    if m.sum() < 40:
                        continue
                    r = probe_binary(X[m], y[m].astype(int), gk[m], layer, name, seeds=seeds)
                    entry[name] = r.to_dict()
                out["results"][tag][pos][str(layer)] = entry
                lv = entry.get("lead_vehicle", {})
                ag = entry.get("action_gt", {})
                apd = entry.get("action_pred", {})
                print(f"  [{tag}] {pos:10s} L{layer:>2} n={len(keep):5d} "
                      f"lead={lv.get('auroc', float('nan')):.3f} "
                      f"act_gt={ag.get('auroc', float('nan')):.3f} "
                      f"act_pred={apd.get('auroc', float('nan')):.3f}", flush=True)

    path = os.path.join(args.outdir, "layerwise_probes.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
