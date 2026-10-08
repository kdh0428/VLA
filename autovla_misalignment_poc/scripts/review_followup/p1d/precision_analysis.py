#!/usr/bin/env python
"""P1-D sample-size / precision evaluation from experiment 35 raw rollouts (read-only). CPU only, no inference.

  nice -n 19 python precision_analysis.py <run_dir>

Per-episode (task, seed) paired contrasts are built exactly as in P0-C (success_X - success_Y; TCP deviation area
sum_{t>=8} ||p_X(t) - p_Y(t)|| in cm*step) and differences-in-differences (DiD) between configurations on the same
episode. For each estimand: per-episode SD (task-stratified episode bootstrap 2,000x, seed 0, gives a 90% upper bound),
CI half-width at n episodes (normal approximation with the stratified variance), n needed for target half-widths, and
power for given true effects at two-sided alpha (also at a Holm-first-step alpha/m). Writes <run_dir>/precision/.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import norm

SRC = "/root/VLA/autovla_misalignment_poc/outputs/spatialvla_feedback_protection_ablation"
FILES = [os.path.join(SRC, "rollouts", "episodes.jsonl"), os.path.join(SRC, "rollouts_ext", "episodes.jsonl")]
TASKS = ["google_robot_move_near", "google_robot_pick_coke_can"]
B = 2000


def load():
    S, P = {}, {}
    for f in FILES:
        with open(f) as fh:
            for line in fh:
                ep = json.loads(line)
                cfg, mode = ep["cond"].split(":")
                k = (cfg, mode, ep["task"], ep["seed"])
                S[k] = float(ep["success"])
                P[k] = np.array([r["pose"][:3] for r in ep["rec"]], dtype=np.float64)
    return S, P


def contrast(S, P, cfg, x, y, keys, metric):
    out = []
    for t, s in keys:
        a, b = (cfg, x, t, s), (cfg, y, t, s)
        if metric == "success":
            out.append(S[a] - S[b])
        else:
            d = np.linalg.norm(P[a] - P[b], axis=1)
            out.append(100 * d[8:].sum())
    return np.array(out)


def strat_sd(v, keys, rng):
    """SD of the task-stratified mean, expressed per episode: sqrt(n * Var(stratified mean)) with equal task sizes."""
    by = defaultdict(list)
    for i, (t, _) in enumerate(keys):
        by[t].append(i)
    w = {t: len(ix) / len(keys) for t, ix in by.items()}
    var_mean = sum(w[t] ** 2 * v[ix].var(ddof=1) / len(ix) for t, ix in by.items())
    sd = np.sqrt(var_mean * len(keys))
    bs = []
    for _ in range(B):
        vb = np.concatenate([v[np.array(ix)[rng.integers(0, len(ix), len(ix))]] for ix in by.values()])
        kb = [k for ix in by.values() for k in [keys[i] for i in ix]]
        byb = defaultdict(list)
        for i, (t, _) in enumerate(kb):
            byb[t].append(i)
        vm = sum(w[t] ** 2 * vb[ix].var(ddof=1) / len(ix) for t, ix in byb.items())
        bs.append(np.sqrt(vm * len(kb)))
    return float(sd), float(np.percentile(bs, 90)), float(np.mean(v)), float(np.mean(v != 0))


def main():
    run = sys.argv[1]
    out = os.path.join(run, "precision"); os.makedirs(out, exist_ok=True)
    S, P = load()
    k80 = [(t, s) for t in TASKS for s in range(40)]
    k240 = [(t, s) for t in TASKS for s in range(120)]
    dedupe = len(sys.argv) > 2 and sys.argv[2] == "--dedupe-move-near"
    if dedupe:
        # move_near seeds map to a finite set of (episode_id, overlay) configurations; keep the first seed of each
        # (initial_states.jsonl from check_initial_states.py)
        combo = {}
        for line in open(os.path.join(run, "verification", "initial_states.jsonl")):
            r = json.loads(line)
            if r.get("task") == "google_robot_move_near" and "seed" in r:
                combo[r["seed"]] = (r["episode_id"], r["urdf_version"])

        def dd(keys):
            seen, out = set(), []
            for t, s in keys:
                if t == "google_robot_move_near":
                    if combo[s] in seen:
                        continue
                    seen.add(combo[s])
                out.append((t, s))
            return out
        k80, k240 = dd(k80), dd(k240)
        out = os.path.join(run, "precision_dedup"); os.makedirs(out, exist_ok=True)
        print("dedup:", len(k80), len(k240))
    rng = np.random.default_rng(0)
    rows = []

    def add(name, v, keys, unit, note):
        sd, sd90, mean, nz = strat_sd(v, keys, rng)
        rows.append({"estimand": name, "n_source": len(keys), "unit": unit, "observed_mean": round(mean, 4),
                     "frac_nonzero": round(nz, 3), "sd_per_ep": round(sd, 4), "sd_per_ep_boot90": round(sd90, 4), "note": note})
        return v

    C = {"R-N": ("reverse", "natural"), "F-C": ("feedback", "corrected"), "N-F": ("natural", "feedback")}
    cfgs80 = ["r1e1", "r1e2", "r1e4", "r2e1", "r2e2", "r4e1"]
    within = {}
    for metric, unit in (("success", "pp/100"), ("area", "cm*step")):
        for cfg in cfgs80:
            for cn, (x, y) in C.items():
                if metric == "area" and cn == "N-F":
                    continue
                v = contrast(S, P, cfg, x, y, k80, metric); within[(metric, cfg, cn, 80)] = v
                add(f"{metric}:{cfg}:{cn}", v, k80, unit, "exp35 seed 0-39 (16/8/4 perturbed chunks for r1/r2/r4)")
        for cfg in ("r1e4", "r4e1"):
            for cn in ("R-N", "F-C"):
                x, y = C[cn]
                v = contrast(S, P, cfg, x, y, k240, metric); within[(metric, cfg, cn, 240)] = v
                add(f"{metric}:{cfg}:{cn}", v, k240, unit, "exp35 seed 0-119")
        # noise (rerun): natural_gen vs natural
        for cfg in ("r1e4", "r4e1"):
            v = contrast(S, P, cfg, "natural_gen", "natural", k80, metric)
            add(f"{metric}:{cfg}:gen-N", v, k80, unit, "numeric-path rerun noise")
        for a, b in (("r1e1", "r2e1"), ("r1e1", "r4e1"), ("r1e1", "r1e2"), ("r1e1", "r1e4"), ("r1e2", "r1e4")):
            for cn in ("R-N", "F-C"):
                v = within[(metric, b, cn, 80)] - within[(metric, a, cn, 80)]
                add(f"{metric}:DiD {b}-{a}:{cn}", v, k80, unit, "exp35 seed 0-39")
        for cn in ("R-N", "F-C"):
            v = within[(metric, "r4e1", cn, 240)] - within[(metric, "r1e4", cn, 240)]
            add(f"{metric}:DiD r4e1-r1e4:{cn}", v, k240, unit, "exp35 seed 0-119 (cross-axis)")

    # sample size / power table for the candidate primary estimands
    def need(sd, h):
        return int(np.ceil((1.959964 * sd / h) ** 2))

    def power(sd, n, delta, alpha):
        z = norm.ppf(1 - alpha / 2); se = sd / np.sqrt(n)
        return float(norm.cdf(abs(delta) / se - z) + norm.cdf(-abs(delta) / se - z))
    plan = []
    for r in rows:
        sd = r["sd_per_ep_boot90"]
        rec = {"estimand": r["estimand"], "sd_used(boot90)": sd}
        if r["unit"] == "pp/100":
            for h in (0.05, 0.075, 0.10):
                rec[f"n_for_halfwidth_{int(h * 1000) / 10}pp"] = need(sd, h)
            for n in (80, 160, 240, 320, 480):
                rec[f"halfwidth_pp_at_n{n}"] = round(100 * 1.959964 * sd / np.sqrt(n), 1)
            for d in (0.05, 0.10):
                for n in (160, 240, 320, 480):
                    rec[f"power_d{int(d * 100)}pp_n{n}_a.05"] = round(power(sd, n, d, 0.05), 2)
                    rec[f"power_d{int(d * 100)}pp_n{n}_a.0125"] = round(power(sd, n, d, 0.0125), 2)
        else:
            for h in (25, 50):
                rec[f"n_for_halfwidth_{h}"] = need(sd, h)
            for n in (80, 160, 240):
                rec[f"halfwidth_at_n{n}"] = round(1.959964 * sd / np.sqrt(n), 1)
            for d in (50, 100):
                for n in (80, 160, 240):
                    rec[f"power_d{d}_n{n}_a.0125"] = round(power(sd, n, d, 0.0125), 2)
        plan.append(rec)
    for name, rr in (("sd_table.csv", rows), ("sample_size_table.csv", plan)):
        keys = []
        for r in rr:
            keys += [k for k in r if k not in keys]
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rr)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
