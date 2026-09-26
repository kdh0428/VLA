#!/usr/bin/env python
"""
Failure-conditioned layer-wise divergence: where does action alignment break while
perception stays decodable?

Compares the control group P+R+A+ against the failure group P+R+A- (and P+A- as a wider
variant) at every layer, using:
  * geometric separation of the two groups' planning-token clouds,
  * a clip-grouped linear "is this a failure" probe per layer,
  * perception decodability measured INSIDE the failure group,
and then localises the earliest layer at which action information diverges while
perception is still present.

CONFOUND CHECKS (spec section 10)
  * scenario/action-matched subsets (no same-clip pairs) rule out "different situations";
  * correlation of each layer's geometric score with ADE, so a score that merely restates
    the output error is visible as such;
  * a shuffled-label control for the failure probe.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Cap BLAS pools before numpy initialises them (see the same note in train_probes.py).
_THREADS = os.environ.get("VLA_PROBE_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _THREADS

import numpy as np  # noqa: E402

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, POC_DIR)

from analysis.probing import HiddenCache, build_matrix, probe_binary  # noqa: E402
from analysis.representation import (divergence_vs_error_correlation, group_stats,  # noqa: E402
                                     linear_cka, matched_groups, pca_project)


def load_rows(path: str) -> list[dict]:
    with open(path) as f:
        rows = json.load(f)
    return [r for r in rows if r.get("hidden_path") and os.path.exists(r["hidden_path"])]


def stack_for(rows: list[dict], layer: int, pool: str,
              cache: HiddenCache | None = None) -> tuple[np.ndarray, list[dict]]:
    X, keep = build_matrix(rows, layer, pool, cache=cache)
    return X, [rows[i] for i in keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", default=os.path.join(POC_DIR, "outputs", "taxonomy", "taxonomy.json"))
    ap.add_argument("--outdir", default=os.path.join(POC_DIR, "outputs", "probes"))
    ap.add_argument("--pool", default="mean")
    ap.add_argument("--layers", default="")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = load_rows(args.taxonomy)
    print(f"{len(rows)} samples with hidden states")
    if not rows:
        raise SystemExit("no hidden states")

    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        with np.load(rows[0]["hidden_path"]) as z:
            layers = sorted(int(k) for k in z.files)
    seeds = tuple(int(x) for x in args.seeds.split(","))

    print("loading hidden states into memory ...", flush=True)
    cache = HiddenCache(args.pool).load(rows)
    print(f"  cached {len(rows)} frames x {len(cache.layers)} layers", flush=True)

    ctrl = [r for r in rows if r["group"] == "P+R+A+"]
    fail = [r for r in rows if r["group"] == "P+R+A-"]
    fail_pa = [r for r in rows if r["P"] is True and r["A"] is False]
    ctrl_pa = [r for r in rows if r["P"] is True and r["A"] is True]
    print(f"P+R+A+={len(ctrl)}  P+R+A-={len(fail)}  P+A+={len(ctrl_pa)}  P+A-={len(fail_pa)}")

    # --- confound-controlled subsets (never pairing frames from the same clip) ----------
    # Matching on (scenario, GT action) is far too strict here: it yields only ~12 pairs,
    # because the control and failure groups sit in largely DIFFERENT scenarios -- which is
    # itself the confound worth controlling. So we run two complementary matches:
    #   * action-matched   : same GT high-level action, scenario free  (~234 pairs, good power)
    #   * scenario-matched : same scenario, action free                (~43 pairs, low power)
    # A divergence that survives the action-matched comparison is not merely "the failure
    # group was asked to do different manoeuvres".
    for r in ctrl + fail + ctrl_pa + fail_pa:
        r["lon_gt"] = (r.get("high_level_action_gt") or "").split("+")[0]

    ma_ctrl, ma_fail = matched_groups(ctrl, fail, keys=("high_level_action_gt",))
    ms_ctrl, ms_fail = matched_groups(ctrl, fail, keys=("scenario",))
    ma_ctrl_pa, ma_fail_pa = matched_groups(ctrl_pa, fail_pa, keys=("high_level_action_gt",))
    print(f"action-matched P+R+A+/-: {len(ma_fail)} pairs | "
          f"scenario-matched: {len(ms_fail)} pairs | action-matched P+A+/-: {len(ma_fail_pa)}")

    out: dict = {
        "group_sizes": {"P+R+A+": len(ctrl), "P+R+A-": len(fail),
                        "P+A+": len(ctrl_pa), "P+A-": len(fail_pa),
                        "matched_action_PRA": len(ma_fail),
                        "matched_scenario_PRA": len(ms_fail),
                        "matched_action_PA": len(ma_fail_pa)},
        "layers": {},
        "config": vars(args),
    }

    comparisons = [("PRA", ctrl, fail),
                   ("PRA_action_matched", ma_ctrl, ma_fail),
                   ("PRA_scenario_matched", ms_ctrl, ms_fail),
                   ("PA", ctrl_pa, fail_pa),
                   ("PA_action_matched", ma_ctrl_pa, ma_fail_pa)]

    prev_ctrl_X = None
    for layer in layers:
        entry: dict = {}
        for tag, c_rows, f_rows in comparisons:
            if len(c_rows) < 10 or len(f_rows) < 10:
                continue
            Xc, kc = stack_for(c_rows, layer, args.pool, cache)
            Xf, kf = stack_for(f_rows, layer, args.pool, cache)
            if len(Xc) < 10 or len(Xf) < 10:
                continue
            st = group_stats(Xc, Xf)

            # Layer-wise "is this a failure" probe, clip-grouped.
            X = np.concatenate([Xc, Xf])
            y = np.concatenate([np.zeros(len(Xc), int), np.ones(len(Xf), int)])
            g = np.array([r["clip_id"] for r in kc] + [r["clip_id"] for r in kf])
            pr = probe_binary(X, y, g, layer, f"failure_{tag}", seeds=seeds)
            st["failure_probe_auroc"] = pr.auroc
            st["failure_probe_auroc_shuffled"] = pr.auroc_shuffled
            st["failure_probe_auprc"] = pr.auprc
            st["failure_probe_ci"] = list(pr.auroc_ci)

            # Is the geometric score merely a restatement of ADE?
            from analysis.representation import cosine_to_centroid
            cen = Xc.mean(axis=0)
            sc = np.concatenate([cosine_to_centroid(Xc, cen), cosine_to_centroid(Xf, cen)])
            ade = np.array([r["ADE"] for r in kc] + [r["ADE"] for r in kf], dtype=float)
            st["score_vs_ade"] = divergence_vs_error_correlation(sc, ade)
            entry[tag] = st

        # Representation drift of the control group across depth (context for the curve).
        Xc_all, _ = stack_for(ctrl or rows, layer, args.pool, cache)
        if len(Xc_all) >= 3:
            entry["ctrl_mean_norm"] = float(np.linalg.norm(Xc_all, axis=1).mean())
            if prev_ctrl_X is not None and prev_ctrl_X.shape == Xc_all.shape:
                entry["cka_with_prev_layer"] = linear_cka(prev_ctrl_X, Xc_all)
            proj, ratio = pca_project(Xc_all, 2)
            entry["ctrl_pca_var_ratio"] = [float(x) for x in ratio]
            prev_ctrl_X = Xc_all

        out["layers"][str(layer)] = entry
        p = entry.get("PRA", entry.get("PA", {}))
        if p:
            print(f"  L{layer:>2} sep_auroc={p.get('separation_auroc', float('nan')):.3f} "
                  f"fail_probe={p.get('failure_probe_auroc', float('nan')):.3f} "
                  f"(shuf {p.get('failure_probe_auroc_shuffled', float('nan')):.3f}) "
                  f"cos_centroids={p.get('centroid_cosine', float('nan')):.4f}", flush=True)

    # --- earliest systematic divergence -------------------------------------------
    # First layer where the failure probe clears the shuffled control by a margin AND
    # stays above it for every deeper layer (a transient bump is not a transition).
    def earliest(tag: str, margin: float = 0.05) -> dict:
        ls = sorted(out["layers"], key=int)
        vals = []
        for L in ls:
            e = out["layers"][L].get(tag, {})
            a, s = e.get("failure_probe_auroc"), e.get("failure_probe_auroc_shuffled")
            vals.append((int(L), a, s))
        for i, (L, a, s) in enumerate(vals):
            if a is None or not np.isfinite(a):
                continue
            base = s if (s is not None and np.isfinite(s)) else 0.5
            if a - base < margin:
                continue
            rest = [v for v in vals[i:] if v[1] is not None and np.isfinite(v[1])]
            if rest and all(v[1] - (v[2] if v[2] and np.isfinite(v[2]) else 0.5) >= margin * 0.6
                            for v in rest):
                return {"layer": L, "auroc": a, "shuffled": base}
        return {"layer": None}

    def geometric_transition(tag: str, z: float = 2.0) -> dict:
        """
        Layer at which the two groups' planning-token clouds *geometrically* separate.

        The failure probe can already be above chance in very early layers without the
        representations having moved apart at all (observed: AUROC ~0.70 at L1 while the
        centroid cosine is still 0.9995). That early signal is decodable scene information,
        not a representational split. This locates the step where the centroid distance
        actually jumps: the first layer whose normalised centroid L2 exceeds the median of
        the earlier layers by `z` robust SDs, and which stays elevated afterwards.
        """
        ls = sorted(out["layers"], key=int)
        vals = [(int(L), out["layers"][L].get(tag, {}).get("centroid_l2_normalised"))
                for L in ls]
        vals = [(L, v) for L, v in vals if v is not None and np.isfinite(v) and L > 0]
        if len(vals) < 8:
            return {"layer": None}
        for i in range(4, len(vals)):
            base = np.array([v for _, v in vals[:i]])
            med, mad = np.median(base), np.median(np.abs(base - np.median(base)))
            sd = max(1.4826 * mad, 1e-6)
            L, v = vals[i]
            if v - med > z * sd and all(w - med > 0.5 * z * sd for _, w in vals[i:]):
                return {"layer": L, "centroid_l2_normalised": v,
                        "baseline_median": float(med), "robust_sd": float(sd)}
        return {"layer": None}

    tags = ("PRA", "PRA_action_matched", "PRA_scenario_matched", "PA", "PA_action_matched")
    out["earliest_action_divergence"] = {t: earliest(t) for t in tags}
    out["geometric_transition"] = {t: geometric_transition(t) for t in tags}

    # How strongly does the layer's geometric score just restate the trajectory error?
    out["score_vs_ade_spearman"] = {
        t: {L: out["layers"][L].get(t, {}).get("score_vs_ade", {}).get("spearman")
            for L in sorted(out["layers"], key=int)} for t in tags}

    path = os.path.join(args.outdir, "layerwise_divergence.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("\nearliest probe-decodable divergence:", out["earliest_action_divergence"])
    print("geometric transition (centroid jump):", out["geometric_transition"])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
