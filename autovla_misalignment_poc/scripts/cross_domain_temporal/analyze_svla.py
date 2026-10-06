#!/usr/bin/env python
"""
Analysis of experiment 34 (PROTOCOL.md). CPU only (the translation table is rebuilt from the processor, no model weights).

Closed loop: success per condition, paired differences (McNemar exact) and episode bootstrap CI (2,000, seed 0, stratified
by task), executed-trajectory divergence (endpoint of the summed world_vector actions) vs `natural`, rerun variability.
Token level: D, amplification ratio A, amplified (A >= 1), recovered, step-4 deviation per row; paired contrasts over units
with episode-cluster bootstrap CI and Wilcoxon / McNemar; matched-magnitude branches; motion substitutions.

  source /root/VLA/simpler/env.sh; python analyze_svla.py <out_dir>
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROWS = ["normal", "recent_ref", "full_ref", "win1", "reverse", "ref_trans", "near_ref", "dir_ok_mag_wrong",
        "dir_wrong_mag_ok", "random_mag_matched"]


def table():
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained("/root/VLA/spatialvla/spatialvla-4b-224-sft-fractal", trust_remote_code=True)
    tt = proc.action_tokenizer.translation_tokenizer
    ids = np.arange(tt.token_start_idx, tt.token_end_idx + 1)
    tab = {int(t): v for t, v in zip(ids, tt.decode_token_ids_to_actions(ids))}
    lo, hi = int(ids[0]), int(ids[-1])

    class Clip(dict):                                   # out-of-range ids decode as the tokenizer clips them
        def __getitem__(self, k):
            return dict.__getitem__(self, min(max(int(k), lo), hi))
    return Clip(tab)


def boot(pairs, reps=2000):
    by = defaultdict(list)
    for c, v in pairs:
        by[c].append(v)
    cl = sorted(by); rng = random.Random(0)
    bs = sorted(np.mean([x for _ in cl for x in by[rng.choice(cl)]]) for _ in range(reps))
    return {"mean": float(np.mean([v for _, v in pairs])), "ci95": [float(bs[int(.025 * reps)]), float(bs[int(.975 * reps) - 1])], "n": len(pairs)}


def paired(items, a, b, key, binary):
    pr = [(it["cl"], it[a][key], it[b][key]) for it in items]
    d = boot([(c, float(x) - float(y)) for c, x, y in pr])
    if binary:
        g = sum(x > y for _, x, y in pr); l_ = sum(y > x for _, x, y in pr)
        d["mcnemar"] = {"a_only": int(g), "b_only": int(l_), "p": float(binomtest(g, g + l_, 0.5).pvalue) if g + l_ else 1.0}
    else:
        diffs = np.array([float(x) - float(y) for _, x, y in pr])
        d["wilcoxon_p"] = float(wilcoxon(diffs).pvalue) if np.any(diffs != 0) else 1.0
    return d


def closed_loop(d):
    eps = [json.loads(l) for l in open(os.path.join(d, "episodes.jsonl"))]
    E = {(r["task"], r["seed"], r["cond"]): r for r in eps}
    conds = sorted({r["cond"] for r in eps})
    keys = sorted({(r["task"], r["seed"]) for r in eps})

    def traj(r):
        return np.cumsum(np.array([x["action"][:3] for x in r["rec"]]), 0)
    out = {"success": {c: boot([(k, float(E[k + (c,)]["success"])) for k in keys if k + (c,) in E]) for c in conds}}
    items = [{"cl": k, **{c: {"s": float(E[k + (c,)]["success"])} for c in conds if k + (c,) in E}} for k in keys]
    items = [it for it in items if all(c in it for c in conds)]
    out["n_paired_episodes"] = len(items)
    out["vs_natural"] = {c: paired(items, c, "natural", "s", True) for c in conds if c != "natural"}
    out["discordant_vs_natural"] = {c: float(np.mean([it[c]["s"] != it["natural"]["s"] for it in items])) for c in conds if c != "natural"}
    for dname in ("perp_left", "opposite"):
        f, co, rv = f"feedback_{dname}", f"corrected_{dname}", f"reverse_{dname}"
        if all(x in conds for x in (f, co, rv)):
            out[f"corrected_vs_feedback_{dname}"] = paired(items, co, f, "s", True)
            out[f"reverse_vs_corrected_{dname}"] = paired(items, rv, co, "s", True)
    dv = {}
    for c in conds:
        if c == "natural":
            continue
        dv[c] = boot([(k, float(np.linalg.norm(traj(E[k + (c,)])[-1] - traj(E[k + ("natural",)])[-1]))) for k in keys
                      if k + (c,) in E and k + ("natural",) in E])
    out["final_traj_divergence_vs_natural_m"] = dv
    return out


def token_level(d, V):
    units, gen_eq = [], []
    for l in open(os.path.join(d, "units.jsonl")):
        f = json.loads(l); gen_eq.append(f["gen_equal"])
        R = np.array(f["R"]); r = np.array([V[int(t)] for t in R[:, 0]])
        for u in f["units"]:
            it = {"cl": (f["task"], f["seed"]), "frame": f["frame"], "d": u["d"], "dir": u["dir"], "subs": u["subs"]}
            for row in ROWS:
                S = np.array(u["rows"][row]); v = np.array([V[int(t)] for t in S[:, 0]])
                inj = np.linalg.norm(v[0] - r[0]); add = np.linalg.norm((v[1:] - r[1:]).sum(0))
                it[row] = {"D": float(np.linalg.norm(v[1:] - r[1:], axis=1).sum()), "A": float(add / max(inj, 1e-9)),
                           "amp": float(add / max(inj, 1e-9) >= 1), "rec": float(np.array_equal(S[1:, 0], R[1:, 0])),
                           "dev4": float(np.linalg.norm(v[3] - r[3])), "E4": float(np.linalg.norm((v - r).sum(0)))}
            units.append(it)
    out = {"n_units": len(units), "n_frames": len(gen_eq), "generate_vs_stepwise_chunk_equal": float(np.mean(gen_eq))}

    def block(us):
        b = {"means": {row: {k: boot([(u["cl"], u[row][k]) for u in us]) for k in ("amp", "rec", "D", "dev4", "A")} for row in ROWS}}
        b["vs_normal"] = {row: {k: paired(us, row, "normal", k, k in ("amp", "rec")) for k in ("amp", "rec", "D", "dev4")}
                          for row in ("recent_ref", "full_ref", "win1", "reverse")}
        b["reverse_vs_full_ref"] = {k: paired(us, "reverse", "full_ref", k, k in ("amp", "rec")) for k in ("amp", "rec", "D", "dev4")}
        b["win1_vs_full_ref"] = {k: paired(us, "win1", "full_ref", k, k in ("amp", "rec")) for k in ("amp", "D", "dev4")}
        return b
    out["all"] = block(units)
    for dd in (0.15, 0.30):
        out[f"d{dd}"] = block([u for u in units if u["d"] == dd])
    amp_units = [u for u in units if u["normal"]["amp"] == 1]
    out["normal_amplified"] = block(amp_units) if len(amp_units) > 20 else {"n": len(amp_units)}
    out["normal_amplified_n"] = len(amp_units)
    # matched-magnitude branches (experiment B)
    B = {}
    for dd in (0.15, 0.30):
        fr = defaultdict(list)
        for u in units:
            if u["d"] == dd:
                fr[u["frame"]].append(u)
        co = [(us[0]["cl"], float(any(x["normal"]["rec"] == 1 for x in us) and any(x["normal"]["amp"] == 1 for x in us)))
              for us in fr.values() if len(us) >= 2]
        la = [[np.log(max(x["normal"]["A"], 1e-3)) for x in us] for us in fr.values() if len(us) >= 2]
        between = float(np.var([np.mean(x) for x in la])); within = float(np.mean([np.var(x) for x in la]))
        B[f"d{dd}"] = {"coexist_recovered_and_amplified": boot(co), "share_between_frames": between / (between + within),
                       "amp_by_direction": {dr: float(np.mean([u["normal"]["amp"] for u in units if u["d"] == dd and u["dir"] == dr]))
                                            for dr in ("along", "opposite", "perp_left", "perp_right", "up", "down")},
                       "rec_by_direction": {dr: float(np.mean([u["normal"]["rec"] for u in units if u["d"] == dd and u["dir"] == dr]))
                                            for dr in ("along", "opposite", "perp_left", "perp_right", "up", "down")}}
    out["B"] = B
    # motion semantics (experiment F): units whose first substitution has e > 0
    mu = [u for u in units if u["subs"]["dir_ok_mag_wrong"][0]["e"] > 0]
    out["motion_n"] = len(mu)
    out["motion"] = {f"{a} - {c}": {k: paired(mu, a, c, k, k in ("amp", "rec")) for k in ("amp", "rec", "D", "dev4")}
                     for a, c in (("dir_wrong_mag_ok", "dir_ok_mag_wrong"), ("near_ref", "ref_trans"), ("random_mag_matched", "dir_ok_mag_wrong"),
                                  ("dir_ok_mag_wrong", "ref_trans"), ("dir_wrong_mag_ok", "ref_trans"))}
    out["motion_means"] = {row: {k: boot([(u["cl"], u[row][k]) for u in mu]) for k in ("amp", "rec", "D", "dev4")}
                           for row in ("ref_trans", "near_ref", "dir_ok_mag_wrong", "dir_wrong_mag_ok", "random_mag_matched", "normal")}
    out["motion_geometry_first_sub"] = {row: {k: float(np.mean([u["subs"][row][0][k] for u in mu])) for k in ("e", "d_sub_ref")}
                                        for row in ("ref_trans", "near_ref", "dir_ok_mag_wrong", "dir_wrong_mag_ok", "random_mag_matched")}
    return out


def main() -> None:
    root = sys.argv[1]; res = {}
    if os.path.exists(os.path.join(root, "closed_loop/episodes.jsonl")):
        res["closed_loop"] = closed_loop(os.path.join(root, "closed_loop"))
    if os.path.exists(os.path.join(root, "token_level/units.jsonl")):
        res["token_level"] = token_level(os.path.join(root, "token_level"), table())
    json.dump(res, open(os.path.join(root, "analysis.json"), "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str)[:5000])


if __name__ == "__main__":
    main()
