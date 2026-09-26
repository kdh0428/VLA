#!/usr/bin/env python
"""
Layer-wise perception and action probes on the planning-token hidden states.

Perception probes (spec section 8)
    traffic_light_red, traffic_light_present, lead_vehicle, pedestrian,
    critical_side_left / _right, critical_motion_moving
Action probes (spec section 9)
    action_lon (4-class), action_lateral (3-class), plus binary stop-vs-go.

The headline analysis is failure-conditioned: every probe is run on
  * ALL samples,
  * the P+R+A+ control group,
  * the P+R+A- failure group,
so we can ask "does perception stay decodable in the failure group?" rather than only
"is perception decodable on average".

All splits are clip-grouped; scalers are fit on train folds only; each probe reports a
label-shuffled chance level measured on the same splits.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

# Cap the BLAS pools BEFORE numpy is imported -- once numpy/OpenBLAS initialises, the pool
# size is fixed and later env changes are ignored. Setting it only inside analysis.probing
# was too late (numpy was already imported here), which let 5 worker processes each spawn
# 28 threads and drove the load average past 125 on a 28-core box, starving the GPU job.
_THREADS = os.environ.get("VLA_PROBE_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _THREADS

import numpy as np  # noqa: E402

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, POC_DIR)

from analysis.probing import HiddenCache, build_matrix, probe_binary, probe_multiclass  # noqa: E402


def load_taxonomy(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def binary_labels(rows: list[dict]) -> dict[str, np.ndarray]:
    """GT-derived binary targets. None entries become NaN and are dropped per-probe."""
    def g(r, k):
        return r.get(k)

    tl = [g(r, "gt_traffic_light") for r in rows]
    lead = [g(r, "gt_lead_vehicle") for r in rows]
    ped = [g(r, "gt_pedestrian") for r in rows]

    out = {
        "traffic_light_red": np.array([1.0 if v == "red" else (0.0 if v in ("none", "green", "yellow") else np.nan)
                                       for v in tl]),
        "traffic_light_present": np.array([0.0 if v == "none" else (1.0 if v in ("red", "green", "yellow") else np.nan)
                                           for v in tl]),
        "lead_vehicle": np.array([1.0 if v is True else (0.0 if v is False else np.nan) for v in lead]),
        "pedestrian": np.array([1.0 if v is True else (0.0 if v is False else np.nan) for v in ped]),
    }
    return out


def side_motion_labels(rows: list[dict], records_by_id: dict[str, dict]) -> dict[str, np.ndarray]:
    """critical-object side / motion, read back from the merged records' GT perception."""
    side, motion = [], []
    for r in rows:
        gp = (records_by_id.get(r["sample_id"], {}) or {}).get("gt_perception", {})
        side.append(gp.get("critical_side"))
        motion.append(gp.get("critical_motion"))
    return {
        "critical_side_left": np.array([1.0 if s == "left" else (0.0 if s in ("center", "right") else np.nan)
                                        for s in side]),
        "critical_side_right": np.array([1.0 if s == "right" else (0.0 if s in ("center", "left") else np.nan)
                                         for s in side]),
        "critical_motion_moving": np.array([1.0 if m == "moving" else (0.0 if m == "static" else np.nan)
                                            for m in motion]),
    }


def action_labels(rows: list[dict]) -> dict[str, np.ndarray]:
    """
    Coarse action semantics of the GT trajectory (what the model SHOULD do).

    `high_level_action_gt` is e.g. "DECELERATE" or "DECELERATE+LEFT"; split it back.
    """
    lon, lat = [], []
    for r in rows:
        s = r.get("high_level_action_gt") or ""
        if not s:
            lon.append(None)
            lat.append(None)
            continue
        parts = s.split("+")
        lon.append(parts[0])
        lat.append(parts[1] if len(parts) > 1 else "STRAIGHT")
    return {"action_lon": np.array(lon, dtype=object),
            "action_lateral": np.array(lat, dtype=object),
            "action_stop_or_slow": np.array(
                [1.0 if v in ("STOP", "DECELERATE") else (0.0 if v in ("MAINTAIN", "ACCELERATE") else np.nan)
                 for v in lon])}


# Set once in the parent before forking workers; children inherit it copy-on-write so the
# ~0.7 GB hidden-state cache is not pickled or duplicated per worker.
_CTX: dict = {}


def _probe_one_layer(layer: int) -> tuple[int, dict]:
    """Run every probe for a single layer. Executed in a forked worker."""
    rows = _CTX["rows"]
    X, keep = build_matrix(rows, layer, _CTX["pool"], cache=_CTX["cache"])
    if len(keep) < 30:
        return layer, {}
    gk = _CTX["groups_all"][keep]
    seeds = _CTX["seeds"]
    entry: dict[str, dict] = {"n_with_hidden": len(keep), "dim": int(X.shape[1])}

    for name, yall in _CTX["bin_lbls"].items():
        y = np.asarray(yall, dtype=float)[keep]
        m = np.isfinite(y)
        if m.sum() < 30:
            continue
        r = probe_binary(X[m], y[m].astype(int), gk[m], layer, name, seeds=seeds)
        entry[name] = r.to_dict()

    for name, yall in _CTX["multi_lbls"].items():
        y = np.asarray(yall, dtype=object)[keep]
        m = np.array([v is not None for v in y])
        if m.sum() < 40:
            continue
        r = probe_multiclass(X[m], y[m], gk[m], layer, name, seeds=seeds)
        entry[name] = r.to_dict()
    return layer, entry


def run_group(rows: list[dict], layers: list[int], pool: str, seeds: tuple[int, ...],
              records_by_id: dict[str, dict], tag: str,
              cache: HiddenCache | None = None, workers: int = 1) -> dict:
    """
    Run every probe at every layer for one sample group.

    Layers are independent, so they are fanned out over processes. Each worker keeps a
    small BLAS pool (see analysis/probing.py) -- many small fits run far better as
    process-parallel than as one process with a wide thread pool.
    """
    result: dict[str, dict] = {"tag": tag, "n_rows": len(rows), "layers": {}}
    if len(rows) < 30:
        result["skipped"] = f"too few samples ({len(rows)})"
        return result

    _CTX.clear()
    _CTX.update(
        rows=rows, pool=pool, cache=cache, seeds=seeds,
        groups_all=np.array([r["clip_id"] for r in rows]),
        bin_lbls={**binary_labels(rows), **side_motion_labels(rows, records_by_id),
                  **{k: v for k, v in action_labels(rows).items() if k == "action_stop_or_slow"}},
        multi_lbls={k: v for k, v in action_labels(rows).items()
                    if k in ("action_lon", "action_lateral")},
    )

    def _record(layer: int, entry: dict) -> None:
        if not entry:
            return
        result["layers"][str(layer)] = entry
        top = entry.get("traffic_light_red", {})
        print(f"  [{tag}] L{layer:>2} n={entry['n_with_hidden']} "
              f"tl_red AUROC={top.get('auroc', float('nan')):.3f} "
              f"(shuf {top.get('auroc_shuffled', float('nan')):.3f})", flush=True)

    if workers <= 1:
        for layer in layers:
            _record(*_probe_one_layer(layer))
    else:
        import concurrent.futures as cf
        import multiprocessing as mp
        ctx = mp.get_context("fork")     # fork so the cache is inherited, not pickled
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for layer, entry in ex.map(_probe_one_layer, layers):
                _record(layer, entry)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", default=os.path.join(POC_DIR, "outputs", "taxonomy", "taxonomy.json"))
    ap.add_argument("--records", default=os.path.join(POC_DIR, "outputs", "merged", "records.jsonl"))
    ap.add_argument("--outdir", default=os.path.join(POC_DIR, "outputs", "probes"))
    ap.add_argument("--layers", default="", help="comma list; default = all layers found")
    ap.add_argument("--pool", default="mean", choices=["mean", "first", "last"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workers", type=int, default=8,
                    help="processes to fan layers over (each keeps a small BLAS pool)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = load_taxonomy(args.taxonomy)
    rows = [r for r in rows if r.get("hidden_path") and os.path.exists(r["hidden_path"])]
    print(f"{len(rows)} samples with hidden states")
    if not rows:
        raise SystemExit("no hidden states found -- run the extraction first")

    records_by_id = {}
    with open(args.records) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records_by_id[r["sample_id"]] = r

    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        with np.load(rows[0]["hidden_path"]) as z:
            layers = sorted(int(k) for k in z.files)
    print(f"layers: {layers}")
    seeds = tuple(int(x) for x in args.seeds.split(","))

    print("loading hidden states into memory ...", flush=True)
    cache = HiddenCache(args.pool).load(rows)
    print(f"  cached {len(rows)} frames x {len(cache.layers)} layers", flush=True)

    ctrl = [r for r in rows if r["group"] == "P+R+A+"]
    fail = [r for r in rows if r["group"] == "P+R+A-"]
    fail_pa = [r for r in rows if r["P"] is True and r["A"] is False]
    perc_fail = [r for r in rows if r["P"] is False and r["A"] is False]
    print(f"groups: all={len(rows)} P+R+A+={len(ctrl)} P+R+A-={len(fail)} "
          f"P+A-={len(fail_pa)} P-A-={len(perc_fail)}")
    print("group counts:", dict(Counter(r["group"] for r in rows).most_common()))

    out = {
        "config": vars(args),
        "group_sizes": {"all": len(rows), "P+R+A+": len(ctrl), "P+R+A-": len(fail),
                        "P+A-": len(fail_pa), "P-A-": len(perc_fail)},
        "results": {},
    }
    for tag, subset in (("all", rows), ("P+R+A+", ctrl), ("P+R+A-", fail), ("P+A-", fail_pa)):
        print(f"\n=== probing group {tag} (n={len(subset)}) ===")
        out["results"][tag] = run_group(subset, layers, args.pool, seeds, records_by_id,
                                        tag, cache=cache, workers=args.workers)

    path = os.path.join(args.outdir, "layerwise_probes.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
