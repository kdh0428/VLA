#!/usr/bin/env python
"""v05 C: SpatialVLA same within-call intervention evaluated over full SimplerEnv episodes
(PROTOCOL_C_SVLA_FULL_EPISODE_20261023_v01.md). RTX 5090 (physical GPU 1). source /root/VLA/simpler/env.sh first.

Per unit (task, seed): N (natural, official per-call decoding, executor r{K}e{E}) from reset; at the anchor policy call the
natural chunk, adapter/SELF parity and the donor search are computed once; every other arm (SELF validation, O, L) resets
the same seed, replays N's float64 actions before the anchor with state/image hash checks, restores N's executor copy and
runs closed loop from the anchor with its own observations. At the anchor call the executed chunk is natural a1,a2 + the
arm's a3,a4 continuation (conditioning copy edited for O / L only). Records: units.jsonl (anchor data) and
episodes.jsonl (per-step actions, ensemble weights, TCP pose, hashes, info)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/spatialvla_protection")
sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/cross_domain_temporal")

SRC_SAMPLE = {"temperature": 0.7, "top_p": 1.0, "top_k": 0}
MAX_SRC = 64
H = lambda x: hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]  # noqa: E731
T0 = time.time()


def worker(conn):
    """p1d_runner.worker (reset with reconfigure=True + replay with per-step observation records), unchanged logic."""
    for k, v in (("LP_NUM_THREADS", "2"), ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1")):
        os.environ[k] = v
    import cv2 as cv
    cv.setNumThreads(1)
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    envs = {}

    def observe(env, obs):
        u = env.unwrapped
        img = cv.resize(get_image_from_maniskill2_obs_dict(env, obs), (224, 224), interpolation=cv.INTER_AREA)
        p = u.tcp.pose
        return img, {"pose": [float(x) for x in p.p] + [float(x) for x in p.q], "grip": float(u.agent.get_gripper_closedness()),
                     "state_hash": H(np.asarray(u.get_state(), dtype=np.float64)), "img_hash": H(img)}

    def scal(info):
        return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in info.items()
                if isinstance(v, (bool, int, float, np.bool_, np.integer, np.floating))}
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            task, seed, replay = arg
            if task not in envs:
                envs[task] = simpler_env.make(task)
            env = envs[task]
            obs, info0 = env.reset(seed=seed, options={"reconfigure": True})
            img, ob = observe(env, obs)
            trace = [ob]
            for a in replay:
                obs, _, done, trunc, info = env.step(np.asarray(a, dtype=np.float64))
                img, ob = observe(env, obs); ob["info"] = scal(info); trace.append(ob)
            mx = getattr(env, "_max_episode_steps", None) or getattr(getattr(env, "spec", None), "max_episode_steps", None)
            conn.send((img, env.get_language_instruction(), trace, {"max_episode_steps": mx}))
        elif cmd == "step":
            task, action = arg
            env = envs[task]
            obs, _, done, trunc, info = env.step(np.asarray(action, dtype=np.float64))
            img, ob = observe(env, obs); ob["info"] = scal(info)
            conn.send((img, bool(done), bool(trunc), ob))
        elif cmd == "close":
            conn.send(None); return


def src_seed(unit_id, j, tag):
    return int.from_bytes(hashlib.sha256(f"svla_v05_C_src{tag}:{unit_id}:{j}".encode()).digest()[:4], "little")


def anchor_protocol(m, inp, unit_id, tag, cap_check):
    """official natural chunk, adapter/SELF parity and v02 donor rule at the anchor observation."""
    from svla_wc_core import EOS
    off, off_lg = m.official(inp)
    a12 = off[:12]
    rec = {"official_ids": off, "natural_valid": bool(len(a12) == 12 and all(m.type_valid(a12)) and EOS not in a12)}
    if not rec["natural_valid"]:
        rec["status"] = "NATURAL_INVALID_AT_ANCHOR"; return rec
    adn, adn_lg = m.forced(inp, [], n_gen=13); k = min(13, len(off))
    rec["adapter_natural_equal"] = adn[:k] == off[:k]
    rid, r_lg = m.forced(inp, a12[:6], n_gen=7)
    rec["SELF_ids"] = rid
    rec["SELF_a34_identity"] = rid[:6] == a12[6:12]
    rec["SELF_term_equal"] = len(off) > 12 and rid[6] == off[12]
    rec["u_N"] = m.u_norm(a12).tolist()
    if not (rec["adapter_natural_equal"] and rec["SELF_a34_identity"] and rec["SELF_term_equal"]):
        rec["status"] = "PARITY_FAIL"; return rec
    uN = np.asarray(rec["u_N"])
    srcs, sel = [], None
    for j in range(MAX_SRC):
        sd = src_seed(unit_id, j, tag); ids, _ = m.official(inp, sample=SRC_SAMPLE, seed=sd)
        s = {"j": j, "seed": sd, "ids": ids}; a6 = ids[:6]
        if not (len(a6) == 6 and all(m.type_valid(a6 + [0] * 6)[:6]) and EOS not in a6):
            s["reason"] = "invalid_type_or_eos_in_a1a2"
        else:
            us = m.u_norm(a6 + a12[6:12])
            s["d_O"] = float(np.mean(np.abs(us[0, :6] - uN[0, :6]))); s["d_L"] = float(np.mean(np.abs(us[1, :6] - uN[1, :6])))
            s["change_type_a1"] = [int(a6[0] != a12[0]), int(a6[1] != a12[1])]
            s["change_type_a2"] = [int(a6[3] != a12[3]), int(a6[4] != a12[4])]
            if not (a6[2] == a12[2] and a6[5] == a12[5]):
                s["reason"] = "gripper_mismatch"
            elif s["d_O"] == 0 or s["d_L"] == 0:
                s["reason"] = "zero_distance"
            else:
                s["ratio_L_over_O"] = s["d_L"] / s["d_O"]
                s["reason"] = None if 0.8 <= s["ratio_L_over_O"] <= 1.25 else "ratio_outside_0.8_1.25"
        srcs.append(s)
        if s["reason"] is None:
            sel = s; break
    rec["sources"] = srcs; rec["n_sources_tried"] = len(srcs)
    if sel is None:
        rec["status"] = "SOURCE_MISSING"; return rec
    d6 = sel["ids"][:6]
    rec["donor"] = {k: sel[k] for k in ("j", "seed", "d_O", "d_L", "ratio_L_over_O", "change_type_a1", "change_type_a2")}
    rec["donor"]["a12"] = d6
    rec["cond"] = {"SELF": a12[:6], "O": d6[:3] + a12[3:6], "L": a12[:3] + d6[3:6]}
    rec["status"] = "OK"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/root/VLA/v05_work/svla_C/manifest/C_manifest_20261023_v01.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=["val", "main", "exec_r4e1"], required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cap-min", type=float, default=200.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    M = json.load(open(a.manifest))
    from svla_protection import Executor, TEMP
    if a.phase == "exec_r4e1":
        K, E, anchor, tag = 4, 1, 8, "_r4e1"
    else:
        K, E, anchor, tag = 1, 4, M["anchor_call"], ""                     # r1e4: env step = policy call
    val = {u for v in M["self_validation_units"].values() for u in v}
    by_task = {}
    for u in M["units"]:
        by_task.setdefault(u["task"], []).append(u)
    order = []                                                           # round-robin across tasks, manifest order
    for i in range(max(len(v) for v in by_task.values())):
        for t in by_task:
            if i < len(by_task[t]):
                order.append(by_task[t][i])
    if a.phase == "val":
        units = [u for u in order if u["unit_id"] in val]
    elif a.phase == "main":
        units = [u for u in order if u["unit_id"] not in val]
    else:
        units = [u for t in by_task for u in by_task[t][:5]]
    uf, ef = os.path.join(a.out, "units.jsonl"), os.path.join(a.out, "episodes.jsonl")
    done = {json.loads(l)["unit_id"] for l in open(uf) if json.loads(l).get("unit_complete")} if os.path.exists(uf) else set()
    units = [u for u in units if u["unit_id"] not in done]
    print(f"[plan] phase={a.phase} K={K} E={E} anchor={anchor} units={len(units)}", flush=True)
    import torch
    from svla_wc_core import SVLA
    m = SVLA()
    run_id = int(T0)
    json.dump({"run_id": run_id, "phase": a.phase, "K": K, "E": E, "anchor": anchor, "gpu": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "transformers": __import__("transformers").__version__,
               "generation_config": m.vla.generation_config.to_dict(), "load_s": time.time() - T0,
               "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "device_order": os.environ.get("CUDA_DEVICE_ORDER")},
              open(os.path.join(a.out, f"run_meta_{run_id}.json"), "w"), indent=1, default=str)
    ctx = mp.get_context("spawn"); pipes = []
    for _ in range(min(a.workers, len(units))):
        p_, c_ = ctx.Pipe(); pr = ctx.Process(target=worker, args=(c_,), daemon=True); pr.start(); pipes.append(p_)
    queue = list(units); active = {}; U = {}
    fu, fe = open(uf, "a"), open(ef, "a")
    gpu_min = lambda: (time.time() - T0) / 60  # noqa: E731
    stop = {"cap": False}

    def jobs_for(u):
        return ["N"] + (["SELF"] if (u["unit_id"] in val and a.phase == "val") else []) + ["O", "L"]

    def begin(w, u, arm):
        uid = u["unit_id"]; st = U[uid]
        ep = {"unit": u, "arm": arm, "rec": [], "calls": 0, "invalid_calls": 0, "t": 0, "success_first": None, "ever_success": False}
        if arm == "N":
            pipes[w].send(("reset", (u["task"], u["seed"], [])))
            img, instr, trace, meta = pipes[w].recv()
            ep.update(img=img, instr=instr, ex=Executor(K, E), trace0=trace, meta=meta)
            st["N_trace"] = [trace[0]]
        else:
            pre = st["N_actions"][:anchor]
            pipes[w].send(("reset", (u["task"], u["seed"], pre)))
            img, instr, trace, meta = pipes[w].recv()
            ref = st["N_trace"][:anchor + 1]
            ok = len(trace) == len(ref) and all(x["state_hash"] == y["state_hash"] and x["img_hash"] == y["img_hash"] for x, y in zip(trace, ref))
            ep.update(img=img, instr=instr, ex=copy.deepcopy(st["ex_at_anchor"]), t=anchor, restore_ok=bool(ok), meta=meta,
                      rec=[dict(x, prefix_from_N=True) for x in copy.deepcopy(st["N_rec"][:anchor])])
            ep["restore_detail"] = {"n_compared": len(ref), "n_trace": len(trace)}
        active[w] = ep

    def next_job(w):
        """advance worker w to its unit's next job or a new unit."""
        cur = active.pop(w, None)
        if cur is not None:
            u = cur["unit"]; st = U[u["unit_id"]]
            st["pending"].pop(0)
            while st["pending"] and st["pending"][0] != "N" and st["anchor"].get("status") != "OK":
                st["skipped"].append(st["pending"].pop(0))
            if st["pending"]:
                begin(w, u, st["pending"][0]); return
            fu.write(json.dumps({"unit_id": u["unit_id"], "run_id": run_id, "phase": a.phase, "unit_complete": True,
                                 "task": u["task"], "seed": u["seed"], "anchor": st["anchor"], "skipped_arms": st["skipped"],
                                 "gpu_min_at_end": gpu_min()}) + "\n"); fu.flush()
            print(f"[unit] {u['unit_id']} {st['anchor'].get('status')} skipped={st['skipped']} gpu_min={gpu_min():.1f}", flush=True)
        if queue and not stop["cap"]:
            if gpu_min() > a.cap_min:
                stop["cap"] = True; print("[cap] GPU cap reached; finishing active units (PARTIAL)", flush=True); return
            u = queue.pop(0)
            U[u["unit_id"]] = {"pending": jobs_for(u), "anchor": {}, "skipped": []}
            begin(w, u, "N")

    for w in range(len(pipes)):
        next_job(w)
    while active:
        ws = sorted(active)
        for w in ws:
            ep = active[w]; u = ep["unit"]; st = U[u["unit_id"]]; t = ep["t"]; arm = ep["arm"]
            step = {"t": t}
            if ep["ex"].plans_at(t):
                inp = m.inputs(ep["img"], ep["instr"])
                if t == anchor and arm == "N":
                    st["ex_at_anchor"] = copy.deepcopy(ep["ex"])
                    st["anchor"] = anchor_protocol(m, inp, u["unit_id"], tag, None)
                    st["anchor"]["anchor_img_hash"] = H(ep["img"])
                    ids = st["anchor"]["official_ids"]
                elif t == anchor:
                    off, _ = m.official(inp)
                    an = st["anchor"]
                    a12 = an["official_ids"][:12]
                    cont, lg = m.forced(inp, an["cond"][arm], n_gen=7)
                    from svla_wc_core import EOS
                    ep["anchor_branch"] = {"official_equal_N": off == an["official_ids"], "img_hash_equal_N": H(ep["img"]) == an["anchor_img_hash"],
                                           "cont_ids": cont, "cont_type_valid": all(m.type_valid(cont[:6] + [0] * 6)[:6]),
                                           "cont_eos_in_a34": EOS in cont[:6], "cont_term_eos": cont[6] == EOS}
                    ids = a12[:6] + cont[:6] + [cont[6]]
                    step["executed_cond"] = an["cond"][arm]
                else:
                    ids, _ = m.official(inp)
                ep["calls"] += 1
                a12 = ids[:12]
                valid = len(a12) == 12 and all(m.type_valid(a12))
                if not valid:
                    ep["invalid_calls"] += 1
                try:
                    acts = np.asarray(m.unnorm(a12 if len(a12) >= 3 else ids), dtype=np.float64).reshape(-1, 7)
                    ep["last_acts"] = acts
                except Exception as exc:                                    # undecodable output: reuse the last decoded chunk (recorded)
                    acts = ep.get("last_acts", np.zeros((4, 7))); step["decode_error"] = repr(exc)[:200]
                if acts.shape[0] < 4:                                       # pad short chunks with their last action (recorded)
                    acts = np.concatenate([acts, np.repeat(acts[-1:], 4 - acts.shape[0], 0)], 0); step["padded"] = True
                ep["ex"].add(t, acts[:4])
                step["ids"] = ids; step["valid"] = valid
            env_action, preds, ages = ep["ex"].step(t)
            w_ = np.exp(-TEMP * np.arange(len(ages))); w_ = w_ / w_.sum()
            step["action"] = env_action.tolist(); step["ages"] = ages; step["weights"] = w_.tolist()
            step["chunk_tgen"] = [int(t - g) for g in ages]
            ep["pending_step"] = step
            pipes[w].send(("step", (u["task"], env_action)))
        for w in ws:
            ep = active[w]; u = ep["unit"]; st = U[u["unit_id"]]
            img, dn, tr, ob = pipes[w].recv()
            step = ep.pop("pending_step"); step.update(ob)
            ep["rec"].append(step); ep["img"] = img; ep["t"] += 1
            if dn:
                ep["ever_success"] = True
                if ep["success_first"] is None:
                    ep["success_first"] = ep["t"]
            if ep["arm"] == "N" and ep["t"] <= anchor:
                st["N_trace"].append({k: ob[k] for k in ("state_hash", "img_hash")})
            if tr or ep["t"] >= 400:
                if ep["arm"] == "N":
                    st["N_actions"] = [s["action"] for s in ep["rec"]]
                    st["N_rec"] = ep["rec"]
                e = {"unit_id": u["unit_id"], "run_id": run_id, "phase": a.phase, "K": K, "E": E, "anchor": anchor, "arm": ep["arm"],
                     "task": u["task"], "seed": u["seed"], "layout_id": u["layout_id"], "cluster_id": u["cluster_id"],
                     "instruction": ep["instr"], "steps": ep["t"], "truncated": bool(tr), "success": bool(dn),
                     "ever_success": ep["ever_success"], "first_success_step": ep["success_first"], "calls": ep["calls"],
                     "invalid_calls": ep["invalid_calls"], "restore_ok": ep.get("restore_ok"), "anchor_branch": ep.get("anchor_branch"),
                     "meta": ep["meta"], "rec": ep["rec"]}
                fe.write(json.dumps(e) + "\n"); fe.flush()
                print(f"[ep] {u['unit_id']} {ep['arm']} success={dn} steps={ep['t']} restore={ep.get('restore_ok')} gpu_min={gpu_min():.1f}", flush=True)
                next_job(w)
    for p_ in pipes:
        p_.send(("close", None)); p_.recv()
    json.dump({"run_id": run_id, "phase": a.phase, "gpu_minutes": gpu_min(), "cap_reached": stop["cap"], "remaining_queue": [u["unit_id"] for u in queue]},
              open(os.path.join(a.out, f"phase_end_{run_id}.json"), "w"), indent=1)
    print(f"[done] gpu_min={gpu_min():.1f} cap={stop['cap']}", flush=True)


if __name__ == "__main__":
    main()
