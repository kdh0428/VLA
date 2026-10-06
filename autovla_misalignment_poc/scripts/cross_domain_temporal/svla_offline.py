#!/usr/bin/env python
"""
Token-level experiments B-F for SpatialVLA (PROTOCOL.md section 4). RTX 5090.

For every frame saved by the natural closed-loop pass: reference chunk R (greedy, step-wise path; also compared with
generate), perturbations p(g, d, u) of the step-1 translation token (d in {0.15, 0.30}, 6 directions), and the rows
normal / recent_ref / full_ref / win1 / reverse / motion substitutions (ref_trans, near_ref, dir_ok_mag_wrong,
dir_wrong_mag_ok, random_mag_matched). Writes <out>/units.jsonl (one line per frame, resumable).

  python svla_offline.py --rollouts DIR --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import cv2 as cv
import numpy as np

from spatialvla_core import SpatialVLA, decode_chunks, translation_table
from svla_rollouts import perturb_token

DS, DIRS = (0.15, 0.30), ("along", "opposite", "perp_left", "perp_right", "up", "down")
MOTION = ("ref_trans", "near_ref", "dir_ok_mag_wrong", "dir_wrong_mag_ok", "random_mag_matched")


def motion_sub(T, rt, ot, name, key):
    """Substitute translation token for the last context step (see PROTOCOL.md). rt = reference token, ot = own token."""
    lo, hi = int(T["ids"][0]), int(T["ids"][-1])     # out-of-range ids are clipped, as the tokenizer's decode does
    jr, jo = T["id2row"][min(max(rt, lo), hi)], T["id2row"][min(max(ot, lo), hi)]
    N, Bn = T["norm"], T["bins"]
    e = float(np.linalg.norm(N[jo] - N[jr]))
    dist = np.linalg.norm(N - N[jr], axis=1)
    same_r = Bn[:, 2] == Bn[jr, 2]; same_dir = (Bn[:, 0] == Bn[jr, 0]) & (Bn[:, 1] == Bn[jr, 1])
    notself = np.arange(len(N)) != jr
    if name == "ref_trans":
        j = jr
    elif name == "near_ref":
        one = same_r & notself & (((Bn[:, 0] == Bn[jr, 0]) & (np.abs(Bn[:, 1] - Bn[jr, 1]) == 1)) |
                                  ((Bn[:, 1] == Bn[jr, 1]) & (np.abs(Bn[:, 0] - Bn[jr, 0]) == 1)))
        c = np.where(one)[0]; j = int(c[np.argmin(dist[c])])
    elif name == "dir_ok_mag_wrong":
        c = np.where(same_dir & notself)[0]; j = int(c[np.argmin(np.abs(dist[c] - e))])
    elif name == "dir_wrong_mag_ok":
        c = np.where(same_r & ~same_dir)[0]; j = int(c[np.argmin(np.abs(dist[c] - e))])
    else:
        c = np.where(same_r & ~same_dir)[0]
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little"))
        j = int(rng.choice(c))
    return int(T["ids"][j]), {"e": round(e, 4), "d_sub_ref": round(float(dist[j]), 4),
                              "same_r": bool(Bn[j, 2] == Bn[jr, 2]), "same_dir": bool(Bn[j, 0] == Bn[jr, 0] and Bn[j, 1] == Bn[jr, 1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "units.jsonl")
    done = {json.loads(l)["frame"] for l in open(dst)} if os.path.exists(dst) else set()
    instr = {(r["task"], r["seed"]): r["instr"] for r in map(json.loads, open(os.path.join(a.rollouts, "episodes.jsonl"))) if r["cond"] == "natural"}
    frames = sorted(os.path.join(d, f) for d in os.listdir(os.path.join(a.rollouts, "frames"))
                    for f in os.listdir(os.path.join(a.rollouts, "frames", d)) if f.endswith(".png"))
    frames = [f for f in frames if f not in done]
    if a.limit:
        frames = frames[:a.limit]
    print(f"[plan] {len(frames)} frames", flush=True)
    m = SpatialVLA(); T = translation_table(m); t0 = time.time()
    with open(dst, "a") as fo:
        for n, rel in enumerate(frames, 1):
            ep, fname = rel.split("/"); task, seed = ep.rsplit("_s", 1); seed = int(seed)
            img = cv.imread(os.path.join(a.rollouts, "frames", rel))[:, :, ::-1].copy()
            inp = m.inputs(img, instr[(task, seed)])
            R = m.decode_chunk(inp)[0]
            gen_ids, _ = m.generate_reference(img, instr[(task, seed)])
            g = int(R[0, 0])
            units = []
            for d in DS:
                for u in DIRS:
                    p, dist, _ = perturb_token(T, g, d, u)
                    if p != g and 0.5 * d <= dist <= 1.5 * d:
                        units.append({"d": d, "dir": u, "p": p, "dist": round(dist, 4)})
            if not units:
                fo.write(json.dumps({"frame": rel, "task": task, "seed": seed, "R": R.tolist(), "gen_equal": bool(np.array_equal(gen_ids, R)), "units": []}) + "\n")
                continue
            U = len(units)
            exe, _, _ = decode_chunks(m, [inp], [0] * U, lambda r, gg: (units[r]["p"], units[r]["p"]))   # normal rows
            S1p = [list(map(int, exe[i, 0])) for i in range(U)]
            own2 = [list(map(int, exe[i, 1])) for i in range(U)]; N3 = [list(map(int, exe[i, 2])) for i in range(U)]
            R2, R3 = list(map(int, R[1])), list(map(int, R[2]))
            # step 3: corrected context [S1p, R2] and motion substitutions of step 2
            ctx3, idx3, subs = [], [], {}
            for i in range(U):
                ctx3.append([S1p[i], R2]); idx3.append((i, "ref"))
                for mn in MOTION:
                    st, info = motion_sub(T, R2[0], own2[i][0], mn, f"{rel}:{i}:{mn}:3")
                    ctx3.append([S1p[i], [st, own2[i][1], own2[i][2]]]); idx3.append((i, mn)); subs[(i, mn, 3)] = info
            o3, _ = m.decode_step(inp, ctx3)
            A3, M3 = {}, {}
            for (i, k), row in zip(idx3, o3):
                (A3 if k == "ref" else M3)[(i, k) if k != "ref" else i] = list(map(int, row))
            # step 4
            ctx4, idx4 = [], []
            for i in range(U):
                for name, c in (("recent_ref", [S1p[i], own2[i], R3]), ("full_ref", [S1p[i], R2, R3]),
                                ("win1", [S1p[i], R2, A3[i]]), ("reverse", [S1p[i], R2, N3[i]])):
                    ctx4.append(c); idx4.append((i, name))
                for mn in MOTION:
                    own3 = M3[(i, mn)]
                    st, info = motion_sub(T, R3[0], own3[0], mn, f"{rel}:{i}:{mn}:4")
                    ctx4.append([S1p[i], own2[i], [st, own3[1], own3[2]]]); idx4.append((i, mn)); subs[(i, mn, 4)] = info
            o4, _ = m.decode_step(inp, ctx4)
            rows = {i: {"normal": [S1p[i], own2[i], N3[i], list(map(int, exe[i, 3]))]} for i in range(U)}
            for (i, k), row in zip(idx4, o4):
                s3 = A3[i] if k in ("recent_ref", "full_ref", "win1", "reverse") else M3[(i, k)]
                if k == "recent_ref":
                    s3 = A3[i]
                rows[i][k] = [S1p[i], own2[i], s3, list(map(int, row))]
            for i, uu in enumerate(units):
                uu["rows"] = rows[i]
                uu["subs"] = {mn: [subs[(i, mn, 3)], subs[(i, mn, 4)]] for mn in MOTION}
            fo.write(json.dumps({"frame": rel, "task": task, "seed": seed, "R": R.tolist(), "gen_equal": bool(np.array_equal(gen_ids, R)),
                                 "units": units}) + "\n")
            if n % 50 == 0:
                fo.flush(); el = time.time() - t0
                print(f"[prog] {n}/{len(frames)} {el / n:.2f}s/frame eta {(len(frames) - n) * el / n / 60:.0f} min", flush=True)
    print(f"[done] {len(frames)} frames in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
