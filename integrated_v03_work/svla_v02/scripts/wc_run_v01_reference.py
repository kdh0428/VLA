#!/usr/bin/env python
"""SpatialVLA within-call experiment runner (2026-10-22 v01). GPU (RTX 5090 only); requires --enable-gpu.

  --phase feasibility : collect validation episodes (natural official policy), then parity + self-patch on the first 12
                        valid candidates (manifest/validation_candidates.json)
  --phase pilot       : collect pilot episodes (manifest/pilot_layouts.json), then N/R/O/L/OL arms + 4 donor sources per
                        observation

Every observation record stores raw token ids (official greedy, adapter natural, all arms, all sources), validity, logits
summaries, donor selection and timing (records.jsonl). Observation images are saved losslessly (obs/*.png + sha256).
"""
from __future__ import annotations

import argparse, hashlib, json, multiprocessing as mp, os, platform, subprocess, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/cross_domain_temporal")
from svla_wc_core import SVLA, CKPT, EOS, seed_of, margin_top2  # noqa: E402

CALLS = (0, 4, 8)
SHORT = {"google_robot_pick_coke_can": "coke", "google_robot_move_near": "move_near"}   # must match manifest obs_id
SRC_SAMPLE = {"temperature": 0.7, "top_p": 1.0, "top_k": 0}


def sha_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint(args):
    fp = {"ckpt_files": {f: sha_file(os.path.join(CKPT, f)) for f in sorted(os.listdir(CKPT))
                         if os.path.isfile(os.path.join(CKPT, f)) and not f.endswith(".pyc")},
          "scripts": {f: sha_file(os.path.join(HERE, f)) for f in sorted(os.listdir(HERE)) if f.endswith(".py")},
          "manifests": {f: sha_file(os.path.join(args.work, "manifest", f)) for f in sorted(os.listdir(os.path.join(args.work, "manifest")))},
          "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "platform": platform.platform()}
    import transformers
    fp["transformers"] = transformers.__version__ + " @ " + transformers.__file__
    try:
        fp["git_head"] = subprocess.check_output(["git", "-C", "/root/VLA", "rev-parse", "HEAD"], text=True).strip()
        d = subprocess.check_output(["git", "-C", "/root/VLA", "diff", "--stat"], text=True)
        fp["git_tracked_diff_stat_sha256"] = hashlib.sha256(d.encode()).hexdigest(); fp["git_tracked_diff_stat_tail"] = d.strip().splitlines()[-1:]
    except Exception as e:  # noqa: BLE001
        fp["git_error"] = str(e)
    return fp


class Ledger:
    def __init__(self, path, cap):
        self.path, self.cap = path, cap
        self.d = json.load(open(path)) if os.path.exists(path) else {"cap_min": cap, "attempts": []}
        self.d["attempts"].append({"pid": os.getpid(), "phase": None, "start": time.time(), "last": time.time()})
        self.save()

    def tick(self, phase=None):
        a = self.d["attempts"][-1]; a["last"] = time.time()
        if phase: a["phase"] = phase
        self.save()

    def used(self):
        return sum(a["last"] - a["start"] for a in self.d["attempts"]) / 60

    def save(self):
        json.dump(self.d, open(self.path + ".tmp", "w"), indent=1); os.replace(self.path + ".tmp", self.path)


# ---------------- observation collection ----------------
def collect(m, episodes, out_dir, led, workers=4):
    """episodes: list of dict(task, seed, tag). Natural official policy (predict_action every control step, official
    adapter execution). Saves the image before generation calls in CALLS. Returns list of observation dicts."""
    from svla_rollouts import worker
    from spatialvla_policy import PolicyState
    import cv2 as cv
    os.makedirs(os.path.join(out_dir, "obs"), exist_ok=True)
    ctx = mp.get_context("spawn"); pipes = []
    for _ in range(min(workers, len(episodes))):
        p_, c_ = ctx.Pipe(); pr = ctx.Process(target=worker, args=(c_,), daemon=True); pr.start(); pipes.append(p_)
    queue = list(episodes); obs_list = []; active = {}

    def start(w):
        ep = queue.pop(0); pipes[w].send(("reset", (ep["task"], ep["seed"]))); img, instr = pipes[w].recv()
        active[w] = dict(ep, img=img, instr=instr, t=0, pol=PolicyState(), done=False, calls=[])
    for w in range(len(pipes)):
        if queue: start(w)
    while active:
        for w in sorted(active):
            ep = active[w]; t = ep["t"]
            inp = m.inputs(ep["img"], ep["instr"])
            ids = m.predict_action_ids(inp)
            if t in CALLS:
                oid = f"{ep['tag']}_{SHORT[ep['task']]}_s{ep['seed']}_c{t}"
                p = os.path.join(out_dir, "obs", oid + ".png"); cv.imwrite(p, ep["img"][:, :, ::-1])
                obs_list.append({"obs_id": oid, "task": ep["task"], "seed": ep["seed"], "call": t, "env_step": t,
                                 "instruction": ep["instr"], "img_sha256": hashlib.sha256(np.ascontiguousarray(ep["img"]).tobytes()).hexdigest(),
                                 "png": os.path.relpath(p, out_dir), "collect_predict_action_ids": ids})
            acts = m.unnorm(ids[:12]) if len(ids) >= 12 else m.unnorm(ids)
            env_action, _ = ep["pol"].to_env(acts)
            pipes[w].send(("step", (ep["task"], env_action)))
            ep["calls"].append({"t": t, "n_ids": len(ids)})
        for w in sorted(active):
            ep = active[w]; img, dn, tr = pipes[w].recv(); ep["img"] = img; ep["t"] += 1
            if dn or tr or ep["t"] > max(CALLS):
                ep["end"] = {"t": ep["t"], "done": dn, "trunc": tr}
                obs_list.append({"episode_end": True, "task": ep["task"], "seed": ep["seed"], "tag": ep["tag"], **ep["end"]})
                del active[w]
                if queue: start(w)
        led.tick()
    for p_ in pipes:
        p_.send(("close", None)); p_.recv()
    return obs_list


# ---------------- arms ----------------
def run_obs(m, o, out_dir, with_arms=True):
    import cv2 as cv
    img = cv.imread(os.path.join(out_dir, o["png"]))[:, :, ::-1].copy()
    assert hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest() == o["img_sha256"]
    inp = m.inputs(img, o["instruction"])
    t0 = time.time()
    rec = {k: o[k] for k in ("obs_id", "task", "seed", "call", "env_step", "instruction", "img_sha256")}
    off, off_lg = m.official(inp)                                   # predict_action-equivalent greedy + logits
    rec["official"] = {"ids": off, "n_generated": len(off), "eos_at": (off.index(EOS) if EOS in off else None)}
    rec["official"]["equals_collect_predict_action"] = off == o["collect_predict_action_ids"]
    a12 = off[:12]
    tv = m.type_valid(a12) if len(a12) == 12 else []
    valid = len(a12) == 12 and all(tv) and EOS not in a12
    rec["natural_valid"] = bool(valid)
    rec["natural_checks"] = {"len_ge_12": len(off) >= 12, "type_valid_12": bool(tv) and all(tv), "eos_inside_12": EOS in a12,
                             "token_13_is_eos": len(off) >= 13 and off[12] == EOS}
    if not valid:
        rec["status"] = "NATURAL_INVALID"; rec["t_s"] = time.time() - t0
        return rec
    # adapter natural (no forced tokens) vs official, 13 tokens
    adn, adn_lg = m.forced(inp, [], n_gen=13)
    k = min(13, len(off))
    rec["adapter_natural"] = {"ids": adn, "token_equal_13": adn[:k] == off[:k],
                              "max_abs_logit_diff": float((adn_lg[:k] - off_lg[:k]).abs().max()),
                              "argmax_mismatch": int((adn_lg[:k].argmax(-1) != off_lg[:k].argmax(-1)).sum())}
    # R self-patch: natural a1,a2 forced
    rid, r_lg = m.forced(inp, a12[:6], n_gen=7)
    kk = min(7, len(off) - 6)
    rec["R"] = {"ids": rid, "a34_token_identity": rid[:6] == a12[6:12], "term_equal": (len(off) > 12 and rid[6] == off[12]),
                "max_abs_logit_diff_vs_official": float((r_lg[:kk] - off_lg[6:6 + kk]).abs().max()),
                "argmax_mismatch": int((r_lg[:kk].argmax(-1) != off_lg[6:6 + kk].argmax(-1)).sum()),
                "min_top2_margin_a34": float(min(margin_top2(r_lg[:6]))),
                "decoded_equal": bool(np.array_equal(m.unnorm(a12[:6] + rid[:6]), m.unnorm(a12)))}
    rec["u_N"] = m.u_norm(a12).tolist()
    if not with_arms:
        rec["status"] = "OK" if rec["R"]["a34_token_identity"] else "R_MISMATCH"; rec["t_s"] = time.time() - t0
        return rec
    # donor sources: 4 independent seeds
    srcs = []
    for j in range(4):
        sd = seed_of(f"svla_wc_src:{o['obs_id']}:{j}")
        ids, _ = m.official(inp, sample=SRC_SAMPLE, seed=sd)
        s = {"j": j, "seed": sd, "ids": ids}
        a6 = ids[:6]
        ok_type = len(a6) == 6 and all(m.type_valid(a6 + [0] * 6)[:6]) and EOS not in a6
        s["a12_type_valid"] = ok_type
        if ok_type:
            us = m.u_norm(a6 + a12[6:12])                                  # pad with natural a3,a4 only to decode a1,a2
            s["grip_equal_a1"] = a6[2] == a12[2]; s["grip_equal_a2"] = a6[5] == a12[5]
            s["d_O"] = float(np.mean(np.abs(us[0, :6] - np.array(rec["u_N"])[0, :6])))
            s["d_L"] = float(np.mean(np.abs(us[1, :6] - np.array(rec["u_N"])[1, :6])))
            s["joint_valid"] = bool(s["grip_equal_a1"] and s["grip_equal_a2"] and s["d_O"] > 0 and s["d_L"] > 0)
        else:
            s["joint_valid"] = False; s["reason"] = "invalid_type_or_eos_in_a1a2"
        if not s["joint_valid"] and "reason" not in s:
            s["reason"] = "gripper_mismatch" if not (s["grip_equal_a1"] and s["grip_equal_a2"]) else "zero_distance"
        srcs.append(s)
    rec["sources"] = srcs
    sel = next((s for s in srcs if s["joint_valid"]), None)
    if sel is None:
        rec["status"] = "SOURCE_MISSING"; rec["t_s"] = time.time() - t0
        return rec
    rec["donor"] = {"j": sel["j"], "d_O": sel["d_O"], "d_L": sel["d_L"], "ratio_L_over_O": sel["d_L"] / sel["d_O"]}
    d6 = sel["ids"][:6]
    conds = {"O": d6[:3] + a12[3:6], "L": a12[:3] + d6[3:6], "OL": d6[:6]}
    rec["arms"] = {}
    for name, cp in conds.items():
        ids, lg = m.forced(inp, cp, n_gen=7)
        tvv = m.type_valid(ids[:6] + [0] * 6)[:6]
        rec["arms"][name] = {"cond": cp, "ids": ids, "type_valid_a34": all(tvv), "eos_in_a34": EOS in ids[:6],
                             "term_is_eos": ids[6] == EOS, "emitted_prefix": a12[:6],
                             "cond_changed_positions": [i for i in range(6) if cp[i] != a12[i]],
                             "min_top2_margin_a34": float(min(margin_top2(lg[:6])))}
        if all(tvv) and EOS not in ids[:6]:
            rec["arms"][name]["u_a34"] = m.u_norm(a12[:6] + ids[:6])[2:4].tolist()
    rec["u_R_a34"] = m.u_norm(a12[:6] + rid[:6])[2:4].tolist()
    okR = rec["R"]["a34_token_identity"]
    okA = all(a["type_valid_a34"] and not a["eos_in_a34"] for a in rec["arms"].values())
    rec["status"] = "OK" if (okR and okA) else ("R_MISMATCH" if not okR else "ARM_INVALID")
    rec["t_s"] = time.time() - t0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["feasibility", "pilot"], required=True)
    ap.add_argument("--work", default="/root/VLA/svla_wc_work")
    ap.add_argument("--cap-min", type=float, default=150.0)
    ap.add_argument("--phase-cap-min", type=float, required=True)
    ap.add_argument("--enable-gpu", action="store_true")
    a = ap.parse_args()
    if not a.enable_gpu:
        sys.exit("--enable-gpu not given")
    out = os.path.join(a.work, "run", a.phase)
    if os.path.isdir(out) and any(os.scandir(out)):
        sys.exit("output dir not empty (no resume): " + out)
    os.makedirs(out)
    fp = fingerprint(a)                                            # before model load
    json.dump(fp, open(os.path.join(out, "RUN_FINGERPRINT.json"), "w"), indent=1)
    led = Ledger(os.path.join(a.work, "run", "gpu_ledger.json"), a.cap_min)
    led.tick(a.phase)
    t_phase = time.time()
    m = SVLA()
    meta = {"gpu": torch.cuda.get_device_name(0), "generation_config": m.vla.generation_config.to_dict(),
            "attn_implementation": getattr(m.vla.config.text_config, "_attn_implementation", None), "dtype": "bfloat16",
            "load_s": time.time() - t_phase}
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=1, default=str)
    status = {"phase": a.phase}
    S = lambda: json.dump({**status, "gpu_min_total": led.used(), "phase_min": (time.time() - t_phase) / 60},  # noqa: E731
                          open(os.path.join(out, "STATUS.json"), "w"), indent=1)
    recf = open(os.path.join(out, "records.jsonl"), "w")
    if a.phase == "feasibility":
        V = json.load(open(os.path.join(a.work, "manifest", "validation_candidates.json")))["candidates"]
        eps = [{"task": t, "seed": s, "tag": "val"} for t, s in dict.fromkeys((c["task"], c["seed"]) for c in V)]
        obs = collect(m, eps, out, led)
        json.dump(obs, open(os.path.join(out, "collected.json"), "w"), indent=1)
        byid = {o["obs_id"]: o for o in obs if "obs_id" in o}
        used, cand_log = [], []
        for c in V:
            o = byid.get(c["obs_id"])
            if o is None:
                cand_log.append({"cand": c["cand"], "obs_id": c["obs_id"], "used": False, "reason": "not_collected (episode ended)"}); continue
            if len(used) >= 12:
                cand_log.append({"cand": c["cand"], "obs_id": c["obs_id"], "used": False, "reason": "12 already selected"}); continue
            r = run_obs(m, o, out, with_arms=True)
            recf.write(json.dumps(r) + "\n"); recf.flush(); led.tick()
            if r["natural_valid"]:
                used.append(r)
            cand_log.append({"cand": c["cand"], "obs_id": c["obs_id"], "used": bool(r["natural_valid"]),
                             "reason": None if r["natural_valid"] else "natural_invalid", "status": r["status"]})
            if (time.time() - t_phase) / 60 > a.phase_cap_min:
                status["stop_reason"] = "phase cap"; break
        n = len(used)
        status.update({"n_valid": n, "candidates": cand_log,
                       "official_eq_collect": sum(r["official"]["equals_collect_predict_action"] for r in used),
                       "adapter_natural_token_equal": sum(r["adapter_natural"]["token_equal_13"] for r in used),
                       "R_a34_identity": sum(r["R"]["a34_token_identity"] for r in used),
                       "R_decoded_equal": sum(r["R"]["decoded_equal"] for r in used),
                       "R_term_equal": sum(r["R"]["term_equal"] for r in used),
                       "max_abs_logit_diff_adapter_natural": max((r["adapter_natural"]["max_abs_logit_diff"] for r in used), default=None),
                       "max_abs_logit_diff_R": max((r["R"]["max_abs_logit_diff_vs_official"] for r in used), default=None),
                       "argmax_mismatch_R": sum(r["R"]["argmax_mismatch"] for r in used)})
        status["FEASIBILITY"] = "PASS" if (n == 12 and status["R_a34_identity"] == 12 and status["adapter_natural_token_equal"] == 12
                                           and status["R_decoded_equal"] == 12) else "STOP"
        S()
    else:
        L = json.load(open(os.path.join(a.work, "manifest", "pilot_layouts.json")))["layouts"]
        eps = [{"task": x["task"], "seed": x["seed"], "tag": "pil"} for x in L]
        obs = collect(m, eps, out, led)
        json.dump(obs, open(os.path.join(out, "collected.json"), "w"), indent=1)
        status["n_collected"] = sum(1 for o in obs if "obs_id" in o); S()
        done = 0
        for o in [o for o in obs if "obs_id" in o]:
            if (time.time() - t_phase) / 60 > a.phase_cap_min or led.used() > a.cap_min:
                status["stop_reason"] = "budget cap"; break
            r = run_obs(m, o, out, with_arms=True)
            recf.write(json.dumps(r) + "\n"); recf.flush(); led.tick(); done += 1
            if done % 10 == 0:
                status["n_done"] = done; S()
        status["n_done"] = done; S()
    led.tick(); S()


if __name__ == "__main__":
    main()
