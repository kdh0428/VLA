#!/usr/bin/env python
"""A2 (SpatialVLA): initial-state clusters for experiments 34, 35 and P1-D from the re-created reset states written by
svla_reset_states.py. Tolerances are fixed in METHOD.md (written before this script was run) and must not be changed.

  python cluster_states.py --run outputs/review_followup2/A_spatialvla_states/run_20261013
Writes state_manifest.csv, cluster_counts.csv, cluster_overlap.csv, cluster_sizes.csv, drift_summary.csv,
determinism.json, verification_summary.json into --run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

ROOT = "/root/VLA/autovla_misalignment_poc"
EXP34 = f"{ROOT}/outputs/cross_domain_temporal_replication/closed_loop/episodes.jsonl"
EXP35 = [f"{ROOT}/outputs/spatialvla_feedback_protection_ablation/rollouts/episodes.jsonl",
         f"{ROOT}/outputs/spatialvla_feedback_protection_ablation/rollouts_ext/episodes.jsonl"]
P1D = f"{ROOT}/outputs/review_followup/P1D_dose_control/run_20261008/rollouts/main/episodes.jsonl"
TASKS = ("google_robot_pick_coke_can", "google_robot_move_near")
SHORT = {"google_robot_pick_coke_can": "coke", "google_robot_move_near": "mnear"}
# ---- fixed in METHOD.md §4 (do not change) ----
TOL_POS, TOL_LINVEL, TOL_ANGVEL = 1e-4, 1e-3, 1e-2


def vectors(r):
    """(positions/quaternions/qpos, linear velocities, angular velocities/qvel) as flat arrays."""
    pos, lin, ang = [], [], []
    for a in r["actors"]:
        pos += a["p"] + a["q"]; lin += a["v"]; ang += a["w"]
    rb = r["robot"]
    pos += rb["p"] + rb["q"] + rb["qpos"]; ang += rb["qvel"]
    return np.array(pos), np.array(lin), np.array(ang)


def dev(r1, r2):
    """max deviation in units of the tolerance (<= 1 means 'same' for all components); inf if layouts differ."""
    a, b = vectors(r1), vectors(r2)
    if any(x.shape != y.shape for x, y in zip(a, b)) or [x["name"] for x in r1["actors"]] != [x["name"] for x in r2["actors"]]:
        return float("inf")
    return max(float(np.max(np.abs(a[0] - b[0]), initial=0)) / TOL_POS, float(np.max(np.abs(a[1] - b[1]), initial=0)) / TOL_LINVEL,
               float(np.max(np.abs(a[2] - b[2]), initial=0)) / TOL_ANGVEL)


def dev_state(st, r):
    """deviation of a raw get_state vector from a canonical record using the raw-state layout is not typed; used only
    for drift records that store the raw state: map back through the same ordering as the canonical record."""
    return float(np.max(np.abs(np.asarray(st) - np.asarray(r["state"]))))


def key(r, level):
    base = (r["task"], tuple(r["model_ids"]), tuple(r["model_scales"]), r["scene_name"])
    if level == "physical":
        return base
    k = base + (r["urdf_version"], r["rgb_overlay_path"])
    if level == "scene":
        return k
    return k + (r["instr"], r["source"], r["target"], r["orientation"])


def cluster(recs, level):
    """single-linkage components of dev <= 1 within identical discrete keys. Returns list of cluster index per rec."""
    n = len(recs)
    parent = list(range(n))

    def f(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    groups = defaultdict(list)
    for i, r in enumerate(recs):
        groups[key(r, level)].append(i)
    for idx in groups.values():
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                if dev(recs[idx[x]], recs[idx[y]]) <= 1.0:
                    parent[f(idx[x])] = f(idx[y])
    roots = [f(i) for i in range(n)]
    order = {}
    for i in range(n):
        order.setdefault(roots[i], len(order))
    # diagnostics
    within, between = 0.0, float("inf")
    for idx in groups.values():
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                d = dev(recs[idx[x]], recs[idx[y]])
                if roots[idx[x]] == roots[idx[y]]:
                    within = max(within, d)
                else:
                    between = min(between, d)
    return [order[r] for r in roots], within, between


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    raw_dir = os.path.join(a.run, "raw")
    raw = {t: [json.loads(l) for l in open(os.path.join(raw_dir, f"raw_{t}.jsonl"))] for t in TASKS}
    canon = {}                                              # (group, task, seed) -> rec
    for t in TASKS:
        for r in raw[t]:
            if r["kind"] == "canon":
                canon[(r["group"], t, r["seed"])] = r

    # ---------------- episodes per experiment / seed, and in-run records used for verification
    ep = {"exp34": defaultdict(lambda: {"n": 0, "instr": set()}), "exp35": defaultdict(lambda: {"n": 0, "instr": set(), "pose0": []})}
    for l in open(EXP34):
        r = json.loads(l); d = ep["exp34"][(r["task"], r["seed"])]; d["n"] += 1; d["instr"].add(r["instr"])
    for fp in EXP35:
        for l in open(fp):
            r = json.loads(l); d = ep["exp35"][(r["task"], r["seed"])]; d["n"] += 1; d["instr"].add(r["instr"])
            d["pose0"].append(r["pose0"])
    cache = os.path.join(a.run, "p1d_episode_index.json")
    if os.path.exists(cache):
        p1d = {tuple(json.loads(k)): v for k, v in json.load(open(cache)).items()}
    else:
        p1d = {}
        for l in open(P1D):                                  # 315 MB, streamed
            r = json.loads(l)
            k = (r["task"], r["seed"])
            d = p1d.setdefault(k, {"n": 0, "state_hash": [], "img_hash": [], "episode_id": [], "overlay": [], "instr": [],
                                   "restore_ok": 0, "restore_n": 0, "pose": []})
            d["n"] += 1
            for f_, v in (("state_hash", r["init"]["state_hash"]), ("img_hash", r["init"]["img_hash"]),
                          ("episode_id", r["episode_id"]), ("overlay", r["overlay"]), ("instr", r["instr"])):
                if v not in d[f_]:
                    d[f_].append(v)
            if r["init"]["pose"] not in d["pose"]:
                d["pose"].append(r["init"]["pose"])
            if r.get("restore") is not None:
                d["restore_n"] += 1; d["restore_ok"] += int(bool(r["restore"].get("ok")))
        json.dump({json.dumps(list(k)): v for k, v in p1d.items()}, open(cache, "w"))

    # ---------------- drift: worst deviation of non-canonical (history-dependent) resets, in tolerance units
    drift_rows = []
    drift_recs = []
    for t in TASKS:
        for r in raw[t]:
            if r["kind"].startswith("drift"):
                drift_recs.append((r["kind"], t, r))
    drift_by_kind = defaultdict(list)
    for kind, t, r in drift_recs:
        c = canon[("exp34_35", t, r["seed"])]
        drift_by_kind[(kind, t)].append((r["seed"], dev(r, c), r["state_hash"] == c["state_hash"],
                                          [x["name"] for x in r["actors"]]))
    # mixed / stepped drift records store only the raw state vector: typed comparison via the canonical record layout
    for fname, kind in (("mixed_drift.jsonl", "mixed"), ("stepped_drift.jsonl", "stepped")):
        fp = os.path.join(raw_dir, fname)
        if not os.path.exists(fp):
            continue
        for l in open(fp):
            r = json.loads(l)
            c = canon[("exp34_35", r["task"], r["seed"])]
            drift_by_kind[(kind, r["task"])].append((r["seed"], raw_dev_typed(r["state"], c), r["same_hash"],
                                                     [x["name"] for x in c["actors"]], r.get("frame_exact")))
    # layout check of raw_dev_typed: on drift records that carry both typed and raw state, both must agree
    chk = [abs(dev(r, canon[("exp34_35", t, r["seed"])]) - raw_dev_typed(r["state"], canon[("exp34_35", t, r["seed"])]))
           for _, t, r in drift_recs]
    print("raw-layout check max |typed - raw| (tol units):", max(chk))
    for (kind, t), v in sorted(drift_by_kind.items()):
        ds = [x[1] for x in v]
        over = [x for x in v if x[1] > 1.0]
        drift_rows.append({"kind": kind, "task": t, "n_resets": len(v), "n_bit_identical": sum(x[2] for x in v),
                           "n_exceeding_tol": len(over), "max_dev_in_tol_units": round(max(ds), 3),
                           "seeds_exceeding_tol": " ".join(sorted({str(x[0]) for x in over}, key=int)),
                           "objects_in_exceeding_layouts": " ".join(sorted({n for x in over for n in x[3]})),
                           "n_frame_exact": sum(1 for x in v if len(x) > 4 and x[4]) if kind in ("mixed", "stepped") else ""})
    drift_actors = defaultdict(dict)                         # task -> actor name -> max deviation (tol units), typed tests
    for kind, t, r in drift_recs:
        c = canon[("exp34_35", t, r["seed"])]
        for x, y in zip(r["actors"], c["actors"]):
            d = max(max(abs(p - q) for p, q in zip(x["p"] + x["q"], y["p"] + y["q"])) / TOL_POS,
                    max(abs(p - q) for p, q in zip(x["v"], y["v"])) / TOL_LINVEL, max(abs(p - q) for p, q in zip(x["w"], y["w"])) / TOL_ANGVEL)
            if d > 1.0:
                drift_actors[t][x["name"]] = max(d, drift_actors[t].get(x["name"], 0.0))
    phys = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])         # (kind, task) -> max |dp|, |dq|, |dv|, |dw| over actors
    for kind, t, r in drift_recs:
        for nm, dp, dq, dv, dw in actor_raw_diffs(r["state"], canon[("exp34_35", t, r["seed"])]):
            phys[(kind, t)] = [max(x, y) for x, y in zip(phys[(kind, t)], (dp, dq, dv, dw))]
    for fname, kind in (("mixed_drift.jsonl", "mixed"), ("stepped_drift.jsonl", "stepped")):
        fp = os.path.join(raw_dir, fname)
        if not os.path.exists(fp):
            continue
        for l in open(fp):
            r = json.loads(l); t = r["task"]; c = canon[("exp34_35", t, r["seed"])]
            for nm, dp, dq, dv, dw in actor_raw_diffs(r["state"], c):
                phys[(kind, t)] = [max(x, y) for x, y in zip(phys[(kind, t)], (dp, dq, dv, dw))]
                d = max(max(dp, dq) / TOL_POS, dv / TOL_LINVEL, dw / TOL_ANGVEL)
                if d > 1.0:
                    drift_actors[t][nm] = max(d, drift_actors[t].get(nm, 0.0))
    print("actors exceeding tol under history-dependent resets:", dict(drift_actors))
    drift_seeds = defaultdict(set)                           # (task) -> seeds whose default reset exceeded tol in any test
    for (kind, t), v in drift_by_kind.items():
        for x in v:
            if x[1] > 1.0:
                drift_seeds[t].add(x[0])

    # ---------------- clustering on the union of canonical states
    keys = sorted(canon, key=lambda k: (TASKS.index(k[1]), 0 if k[0] == "exp34_35" else 1, k[2]))
    recs = [canon[k] for k in keys]
    levels = ("primary", "scene", "physical")
    cid, diag = {}, {}
    for lev in levels:
        lab, w, b = cluster(recs, lev)
        diag[lev] = {"max_within_dev_tol_units": w, "min_between_dev_tol_units": b}
        for k, l_ in zip(keys, lab):
            cid[(lev, k)] = l_
    # readable ids, numbered per task in order of first appearance
    names = {}
    for lev, tag in zip(levels, ("S", "L", "P")):
        per = defaultdict(dict)
        for k in keys:
            c = cid[(lev, k)]
            if c not in per[k[1]]:
                per[k[1]][c] = f"{SHORT[k[1]]}-{tag}{len(per[k[1]]) + 1:03d}"
            names[(lev, k)] = per[k[1]][c]

    # ---------------- manifest
    rows = []
    exp_seeds = {"exp34": range(40), "exp35": range(120)}
    for t in TASKS:
        for e in ("exp34", "exp35"):
            for s in exp_seeds[e]:
                c = canon[("exp34_35", t, s)]; k = ("exp34_35", t, s)
                d = ep[e].get((t, s), {"n": 0, "instr": set()})
                instr_ok = d["instr"] == {c["instr"]}
                if e == "exp34":
                    fr = c.get("exp34_frame")
                    v = ("no in-run state/restore hash; per-condition resets without reconfigure (history-dependent); "
                         f"recorded instr {'matches' if instr_ok else 'MISMATCH'}; t0 frame "
                         + ("missing" if not fr else ("pixel-exact" if fr["exact"] else f"differs (max {fr['max_abs']}/255 on {fr['n_px_diff']} px, object region)")))
                else:
                    p0 = np.array(d["pose0"]); md = float(np.max(np.abs(p0 - np.array(c["tcp_pose"])))) if len(p0) else float("nan")
                    v = ("no in-run state/restore hash; per-condition resets without reconfigure (history-dependent); "
                         f"recorded instr {'matches' if instr_ok else 'MISMATCH'}; recorded TCP pose0 max|diff| {md:.1e} "
                         "(robot only; object state not recorded)")
                prone = sorted(set(x["name"] for x in c["actors"]) & set(drift_actors[t]))
                if prone:
                    v += ("; physical state uncertain beyond tol: layout contains " + "/".join(prone) +
                          " which drifted up to " + "/".join(f"{drift_actors[t][n]:.0f}" for n in prone) +
                          "x tol under no-reconfigure resets")
                if s in drift_seeds[t]:
                    v += " (drift>tol reproduced for this seed)"
                rows.append(row(t, e, str(s), c, names, k, d["n"], v))
        for s in sorted(x[2] for x in canon if x[0] == "p1d" and x[1] == t):
            c = canon[("p1d", t, s)]; k = ("p1d", t, s)
            d = p1d.get((t, s))
            if d is None:
                v = "no P1-D episodes found"; n = 0
            else:
                n = d["n"]
                sh = d["state_hash"] == [c["state_hash"]]; ih = d["img_hash"] == [c["img_hash"]]
                v = (f"in-run: state+img hash recorded for {n} episodes ({len(d['state_hash'])} distinct state hash), replay restore.ok "
                     f"{d['restore_ok']}/{d['restore_n']}; re-created state_hash {'MATCH' if sh else 'MISMATCH'}, img_hash {'MATCH' if ih else 'MISMATCH'}")
                if t == "google_robot_move_near" and (d["episode_id"] != [c["episode_id"]] or d["overlay"] != [c["urdf_version"]]):
                    v += "; episode_id/overlay MISMATCH"
            rows.append(row(t, "P1D", str(s), c, names, k, n, v))
    fields = list(rows[0].keys())
    with open(os.path.join(a.run, "state_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    # ---------------- counts, sizes, overlap
    cnt = []
    for t in TASKS:
        for e in ("exp34", "exp35", "exp35_seeds0_39", "exp35_seeds40_119", "exp34+exp35", "P1D", "all"):
            sel = [r for r in rows if r["task"] == t and (
                r["experiment"] == e or (e == "exp35_seeds0_39" and r["experiment"] == "exp35" and int(r["seed_or_unit"]) < 40)
                or (e == "exp35_seeds40_119" and r["experiment"] == "exp35" and int(r["seed_or_unit"]) >= 40)
                or (e == "exp34+exp35" and r["experiment"] in ("exp34", "exp35")) or e == "all")]
            for lev, col in zip(levels, ("initial_state_cluster_id", "scene_state_cluster_id", "physical_state_cluster_id")):
                eps_per = Counter()
                seeds_per = defaultdict(set)
                for r in sel:
                    eps_per[r[col]] += int(r["n_episodes"]); seeds_per[r[col]].add((r["experiment"] == "P1D", r["seed_or_unit"]))
                nseed = len({(r["experiment"] == "P1D", r["seed_or_unit"]) for r in sel})
                ss = [len(v) for v in seeds_per.values()]
                cnt.append({"task": t, "experiment": e, "level": lev, "n_seeds_or_units": nseed,
                            "n_episodes": sum(int(r["n_episodes"]) for r in sel),
                            "n_unique_clusters": len(seeds_per), "max_seeds_per_cluster": max(ss),
                            "n_clusters_with_gt1_seed": sum(x > 1 for x in ss),
                            "max_episodes_per_cluster": max(eps_per.values()),
                            "median_episodes_per_cluster": float(np.median(list(eps_per.values())))})
    # exp34+exp35 / all double-count exp34 seeds 0-39 rows (same seeds, separate episodes): episodes add, seeds dedup above
    with open(os.path.join(a.run, "cluster_counts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cnt[0].keys())); w.writeheader(); w.writerows(cnt)
    ov = []
    for t in TASKS:
        for lev, col in zip(levels, ("initial_state_cluster_id", "scene_state_cluster_id", "physical_state_cluster_id")):
            S = {e: {r[col] for r in rows if r["task"] == t and r["experiment"] == e} for e in ("exp34", "exp35", "P1D")}
            for x, y in (("exp34", "exp35"), ("exp34", "P1D"), ("exp35", "P1D")):
                ov.append({"task": t, "level": lev, "exp_a": x, "exp_b": y, "n_a": len(S[x]), "n_b": len(S[y]),
                           "n_shared": len(S[x] & S[y]), "n_only_a": len(S[x] - S[y]), "n_only_b": len(S[y] - S[x])})
            ov.append({"task": t, "level": lev, "exp_a": "all3", "exp_b": "", "n_a": len(S["exp34"] | S["exp35"] | S["P1D"]),
                       "n_b": "", "n_shared": len(S["exp34"] & S["exp35"] & S["P1D"]), "n_only_a": "", "n_only_b": ""})
    with open(os.path.join(a.run, "cluster_overlap.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ov[0].keys())); w.writeheader(); w.writerows(ov)
    # cluster membership table
    mem = []
    for t in TASKS:
        for lev, col in zip(levels, ("initial_state_cluster_id", "scene_state_cluster_id", "physical_state_cluster_id")):
            by = defaultdict(lambda: defaultdict(list))
            for r in rows:
                if r["task"] == t:
                    by[r[col]][r["experiment"]].append(r["seed_or_unit"])
            for c in sorted(by):
                d = by[c]
                mem.append({"task": t, "level": lev, "cluster_id": c,
                            "exp34_seeds": " ".join(d.get("exp34", [])), "exp35_seeds": " ".join(d.get("exp35", [])),
                            "P1D_units": " ".join(d.get("P1D", [])),
                            "n_episodes": sum(int(r["n_episodes"]) for r in rows if r["task"] == t and r[col] == c)})
    with open(os.path.join(a.run, "cluster_sizes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mem[0].keys())); w.writeheader(); w.writerows(mem)
    for r in drift_rows:
        dp, dq, dv, dw = phys[(r["kind"], r["task"])]
        r.update({"max_obj_pos_diff_m": f"{dp:.2e}", "max_obj_quat_diff": f"{dq:.2e}", "max_obj_linvel_diff": f"{dv:.2e}",
                  "max_obj_angvel_diff": f"{dw:.2e}"})
    with open(os.path.join(a.run, "drift_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(drift_rows[0].keys())); w.writeheader(); w.writerows(drift_rows)

    # ---------------- determinism, verification summaries
    det = {"tolerances": {"pos_quat_qpos": TOL_POS, "lin_vel": TOL_LINVEL, "ang_vel_qvel": TOL_ANGVEL},
           "cluster_diagnostics": diag}
    for t in TASKS:
        rep = [r for r in raw[t] if r["kind"] == "repeat"]
        det[t] = {"repeat_same_process_bit_identical": f"{sum(r['state_hash'] == canon[(('exp34_35' if r['group'] == 'exp' else 'p1d'), t, r['seed'])]['state_hash'] and r['img_hash'] == canon[(('exp34_35' if r['group'] == 'exp' else 'p1d'), t, r['seed'])]['img_hash'] for r in rep)}/{len(rep)}"}
        fp = os.path.join(raw_dir, f"fresh_{t}.jsonl")
        if os.path.exists(fp):
            fr = [json.loads(l) for l in open(fp)]
            ok = 0
            for r in fr:
                g = "p1d" if ("p1d", t, r["seed"]) in canon and r["seed"] >= 120 else "exp34_35"
                c = canon[(g, t, r["seed"])]
                ok += int(r["state_hash"] == c["state_hash"] and r["img_hash"] == c["img_hash"])
            det[t]["fresh_process_bit_identical"] = f"{ok}/{len(fr)}"
    json.dump(det, open(os.path.join(a.run, "determinism.json"), "w"), indent=1, default=str)
    ver = {}
    for t in TASKS:
        p = [r for r in rows if r["task"] == t and r["experiment"] == "P1D"]
        e34 = [canon[("exp34_35", t, s)].get("exp34_frame") for s in range(40)]
        ver[t] = {"P1D_state_hash_match": sum("state_hash MATCH" in r["restore_verification"] for r in p), "P1D_units": len(p),
                  "P1D_img_hash_match": sum("img_hash MATCH" in r["restore_verification"] for r in p),
                  "exp34_t0_frame_pixel_exact": sum(bool(x and x["exact"]) for x in e34), "exp34_seeds": 40,
                  "exp34_frame_mismatch_seeds": [s for s in range(40) if e34[s] and not e34[s]["exact"]],
                  "instr_mismatch_rows": [(r["experiment"], r["seed_or_unit"]) for r in rows if r["task"] == t and "MISMATCH" in r["restore_verification"]]}
    json.dump(ver, open(os.path.join(a.run, "verification_summary.json"), "w"), indent=1)
    for r in cnt:
        if r["level"] == "primary" or r["experiment"] in ("exp34", "exp35", "P1D", "all"):
            print(r)
    print(json.dumps(diag)); print(json.dumps(ver)); print(json.dumps(det)[:2000])
    for r in drift_rows:
        print(r)


def raw_dev_typed(st, c):
    """typed deviation of a raw get_state vector against canonical record c. The raw layout (ManiSkill2 get_state):
    for each actor in scene order [p(3) q(4) v(3) w(3)], then each articulation [root p q v w (13), qpos, qvel].
    Actor order = c['state'] order, which we recover by matching against the canonical typed vectors: we compare the
    raw vectors element-wise and attribute each element's type from the canonical layout built here."""
    st = np.asarray(st); cs = np.asarray(c["state"])
    if st.shape != cs.shape:
        return float("inf")
    n_act = (len(cs) - 13 - 2 * len(c["robot"]["qpos"])) // 13
    typ = []
    for _ in range(n_act):
        typ += ["p"] * 7 + ["l"] * 3 + ["a"] * 3
    typ += ["p"] * 7 + ["l"] * 3 + ["a"] * 3 + ["p"] * len(c["robot"]["qpos"]) + ["a"] * len(c["robot"]["qvel"])
    typ = np.array(typ)
    d = np.abs(st - cs)
    # quaternion sign: raw states are not sign-canonicalised; identical pipelines give identical signs
    return max(float(d[typ == "p"].max(initial=0)) / TOL_POS, float(d[typ == "l"].max(initial=0)) / TOL_LINVEL,
               float(d[typ == "a"].max(initial=0)) / TOL_ANGVEL)


def actor_raw_diffs(st, c):
    """per dynamic actor of canonical record c: (name, max|dp| m, max|dq|, max|dv| m/s, max|dw| rad/s) from raw
    get_state vectors (actor blocks of 13 in scene order; matched to names via the canonical positions)."""
    st = np.asarray(st); cs = np.asarray(c["state"]); out = []
    n_act = (len(cs) - 13 - 2 * len(c["robot"]["qpos"])) // 13
    for a in c["actors"]:
        i = [j for j in range(n_act) if np.allclose(cs[13 * j:13 * j + 3], a["p"], atol=0, rtol=0)]
        assert len(i) == 1, (a["name"], i)
        b = 13 * i[0]; d = np.abs(st[b:b + 13] - cs[b:b + 13])
        out.append((a["name"], float(d[0:3].max()), float(d[3:7].max()), float(d[7:10].max()), float(d[10:13].max())))
    return out


def row(t, e, s, c, names, k, n_eps, v):
    return {"task": t, "experiment": e, "seed_or_unit": s,
            "reset_call": ("reset(seed) [no reconfigure]" if e != "P1D" else
                           ("reset(seed, reconfigure=True, episode_id)" if t == "google_robot_move_near" else "reset(seed, reconfigure=True)")),
            "episode_id": c["episode_id"] if c["episode_id"] is not None else "", "overlay_urdf_version": c["urdf_version"],
            "rgb_overlay_path": c["rgb_overlay_path"], "instruction": c["instr"],
            "source_obj": c["source"] or "", "target_obj": c["target"] or "", "coke_orientation": c["orientation"] or "",
            "object_model_ids": "|".join(c["model_ids"]), "object_scales": "|".join(str(x) for x in c["model_scales"]),
            "object_poses_xyz_wxyz": "|".join(a["name"] + ":" + ",".join(f"{x:.6f}" for x in a["p"] + a["q"]) for a in c["actors"]),
            "object_max_speed": f"{max([float(np.linalg.norm(a['v'])) for a in c['actors']] + [0]):.2e}",
            "robot_base_xyz_wxyz": ",".join(f"{x:.6f}" for x in c["robot"]["p"] + c["robot"]["q"]),
            "robot_qpos": ",".join(f"{x:.6f}" for x in c["robot"]["qpos"]),
            "robot_max_abs_qvel": f"{max(abs(x) for x in c['robot']['qvel']):.2e}",
            "tcp_pose": ",".join(f"{x:.6f}" for x in c["tcp_pose"]),
            "reset_state_sha1": c["state_hash"], "reset_img_sha1": c["img_hash"],
            "initial_state_cluster_id": names[("primary", k)], "scene_state_cluster_id": names[("scene", k)],
            "physical_state_cluster_id": names[("physical", k)], "n_episodes": n_eps, "restore_verification": v}


if __name__ == "__main__":
    main()
