#!/usr/bin/env python
"""SpatialVLA within-call follow-up v02 (2026-10-23). GPU (RTX 5090 only); --enable-gpu required.
Same model, fresh-cache official path, adapter, arms (N/R/O/L/OL), action numbering and 6-D normalized response as v01
(svla_wc_core.py copied unchanged from the v01 final run). Changes (pre-registered in PROTOCOL_20261023_v02.md):
generation calls 2/6/10; donor = first source among at most 64 sequential official T=0.7 samples that is joint-valid AND
distance-matched (0.8 <= d_L/d_O <= 1.25); seeds sha256("svla_wc_v02_src:" + obs_id + ":" + j)[:4] little-endian.
  --phase smoke   : parity only (native / adapter / R) on the smoke list (coke + move_near; no arms)
  --phase main    : collect panel (manifest) + all arms
"""
from __future__ import annotations
import argparse, hashlib, json, multiprocessing as mp, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/cross_domain_temporal")
from svla_wc_core import SVLA, EOS, margin_top2  # noqa: E402
sys.path.insert(0, "/root/VLA/svla_wc_work/scripts")
from wc_run import fingerprint as _fp_v01, Ledger, sha_file  # noqa: E402

SRC_SAMPLE = {"temperature": 0.7, "top_p": 1.0, "top_k": 0}
MAX_SRC = 64
SHORT = {"google_robot_pick_coke_can": "coke", "google_robot_move_near": "move_near", "google_robot_open_drawer": "open_drawer"}


def src_seed(obs_id, j):
    return int.from_bytes(hashlib.sha256(f"svla_wc_v02_src:{obs_id}:{j}".encode()).digest()[:4], "little")


def fingerprint(work):
    fp = {"ckpt_files": {}, "scripts": {f: sha_file(os.path.join(HERE, f)) for f in sorted(os.listdir(HERE)) if f.endswith(".py")},
          "manifests": {f: sha_file(os.path.join(work, "manifest", f)) for f in sorted(os.listdir(os.path.join(work, "manifest"))) if f.endswith(".json")}}
    from svla_wc_core import CKPT
    for f in sorted(os.listdir(CKPT)):
        p = os.path.join(CKPT, f)
        if os.path.isfile(p) and not f.endswith(".pyc"):
            fp["ckpt_files"][f] = sha_file(p)
    import transformers, platform, subprocess
    fp.update({"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "platform": platform.platform(),
               "transformers": transformers.__version__ + " @ " + transformers.__file__})
    try:
        fp["git_head"] = subprocess.check_output(["git", "-C", "/root/VLA", "rev-parse", "HEAD"], text=True).strip()
    except Exception as e:  # noqa: BLE001
        fp["git_error"] = str(e)
    return fp


def collect(m, episodes, calls, out_dir, led, cap_check, workers=4):
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
        active[w] = dict(ep, img=img, instr=instr, t=0, pol=PolicyState())
    for w in range(len(pipes)):
        if queue: start(w)
    while active:
        if cap_check():
            obs_list.append({"collection_stopped": "budget cap"}); break
        for w in sorted(active):
            ep = active[w]; t = ep["t"]
            inp = m.inputs(ep["img"], ep["instr"]); ids = m.predict_action_ids(inp)
            if t in calls:
                oid = f"{ep['tag']}_{SHORT[ep['task']]}_s{ep['seed']}_c{t}"
                p = os.path.join(out_dir, "obs", oid + ".png"); cv.imwrite(p, ep["img"][:, :, ::-1])
                obs_list.append({"obs_id": oid, "task": ep["task"], "seed": ep["seed"], "call": t, "env_step": t, "instruction": ep["instr"],
                                 "img_sha256": hashlib.sha256(np.ascontiguousarray(ep["img"]).tobytes()).hexdigest(),
                                 "png": os.path.relpath(p, out_dir), "collect_predict_action_ids": ids})
            acts = m.unnorm(ids[:12]) if len(ids) >= 12 else m.unnorm(ids)
            env_action, _ = ep["pol"].to_env(acts)
            pipes[w].send(("step", (ep["task"], env_action)))
        for w in sorted(active):
            ep = active[w]; img, dn, tr = pipes[w].recv(); ep["img"] = img; ep["t"] += 1
            if dn or tr or ep["t"] > max(calls):
                obs_list.append({"episode_end": True, "task": ep["task"], "seed": ep["seed"], "tag": ep["tag"], "t": ep["t"], "done": dn, "trunc": tr})
                del active[w]
                if queue: start(w)
        led.tick()
    for p_ in pipes:
        p_.send(("close", None)); p_.recv()
    return obs_list


def trans_rot_changed(a, b):
    return [int(a[0] != b[0]), int(a[1] != b[1])]


def run_obs(m, o, out_dir, cap_check, with_arms=True):
    import cv2 as cv
    img = cv.imread(os.path.join(out_dir, o["png"]))[:, :, ::-1].copy()
    assert hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest() == o["img_sha256"]
    inp = m.inputs(img, o["instruction"]); t0 = time.time()
    rec = {k: o[k] for k in ("obs_id", "task", "seed", "call", "env_step", "instruction", "img_sha256")}
    off, off_lg = m.official(inp)
    rec["official"] = {"ids": off, "n_generated": len(off), "equals_collect_predict_action": (None if o.get("collect_predict_action_ids") is None else off == o["collect_predict_action_ids"])}
    a12 = off[:12]; tv = m.type_valid(a12) if len(a12) == 12 else []
    rec["natural_valid"] = bool(len(a12) == 12 and all(tv) and EOS not in a12)
    rec["natural_checks"] = {"token_13_is_eos": len(off) >= 13 and off[12] == EOS}
    if not rec["natural_valid"]:
        rec["status"] = "NATURAL_INVALID"; return rec
    adn, adn_lg = m.forced(inp, [], n_gen=13); k = min(13, len(off))
    rec["adapter_natural"] = {"ids": adn, "token_equal_13": adn[:k] == off[:k], "max_abs_logit_diff": float((adn_lg[:k] - off_lg[:k]).abs().max())}
    rid, r_lg = m.forced(inp, a12[:6], n_gen=7); kk = min(7, len(off) - 6)
    rec["R"] = {"ids": rid, "a34_token_identity": rid[:6] == a12[6:12], "term_equal": len(off) > 12 and rid[6] == off[12],
                "max_abs_logit_diff_vs_official": float((r_lg[:kk] - off_lg[6:6 + kk]).abs().max()),
                "decoded_equal": bool(np.array_equal(m.unnorm(a12[:6] + rid[:6]), m.unnorm(a12)))}
    uN = m.u_norm(a12); rec["u_N"] = uN.tolist()
    rok = rec["R"]["a34_token_identity"] and rec["R"]["term_equal"] and rec["official"]["equals_collect_predict_action"] in (True, None) and rec["adapter_natural"]["token_equal_13"]
    if not with_arms:
        rec["status"] = "OK" if rok else "PARITY_FAIL"; rec["t_s"] = time.time() - t0; return rec
    if not rok:
        rec["status"] = "PARITY_FAIL"; rec["t_s"] = time.time() - t0; return rec
    srcs, sel = [], None
    for j in range(MAX_SRC):
        if cap_check():
            rec["status"] = "BUDGET_STOP_IN_SOURCES"; rec["sources"] = srcs; return rec
        sd = src_seed(o["obs_id"], j); ids, _ = m.official(inp, sample=SRC_SAMPLE, seed=sd)
        s = {"j": j, "seed": sd, "ids": ids}; a6 = ids[:6]
        ok = len(a6) == 6 and all(m.type_valid(a6 + [0] * 6)[:6]) and EOS not in a6
        if not ok:
            s["reason"] = "invalid_type_or_eos_in_a1a2"
        else:
            us = m.u_norm(a6 + a12[6:12])
            s["d_O"] = float(np.mean(np.abs(us[0, :6] - uN[0, :6]))); s["d_L"] = float(np.mean(np.abs(us[1, :6] - uN[1, :6])))
            s["grip_equal"] = [a6[2] == a12[2], a6[5] == a12[5]]
            s["change_type_a1"] = trans_rot_changed(a6[:3], a12[:3]); s["change_type_a2"] = trans_rot_changed(a6[3:6], a12[3:6])
            if not all(s["grip_equal"]):
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
        rec["status"] = "MATCHED_SOURCE_MISSING"; rec["t_s"] = time.time() - t0; return rec
    rec["donor"] = {k: sel[k] for k in ("j", "seed", "d_O", "d_L", "ratio_L_over_O", "change_type_a1", "change_type_a2")}
    d6 = sel["ids"][:6]
    conds = {"O": d6[:3] + a12[3:6], "L": a12[:3] + d6[3:6], "OL": d6[:6]}
    rec["arms"] = {}
    for name, cp in conds.items():
        ids, lg = m.forced(inp, cp, n_gen=7)
        tvv = m.type_valid(ids[:6] + [0] * 6)[:6]
        rec["arms"][name] = {"cond": cp, "ids": ids, "type_valid_a34": all(tvv), "eos_in_a34": EOS in ids[:6], "term_is_eos": ids[6] == EOS,
                             "emitted_prefix": a12[:6], "cond_changed_positions": [i for i in range(6) if cp[i] != a12[i]],
                             "min_top2_margin_a34": float(min(margin_top2(lg[:6])))}
    okA = all(a["type_valid_a34"] and not a["eos_in_a34"] for a in rec["arms"].values())
    rec["status"] = "OK" if okA else "ARM_INVALID"; rec["t_s"] = time.time() - t0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke", "main", "recovered"], required=True)
    ap.add_argument("--recovered-json", default=None)
    ap.add_argument("--work", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--phase-cap-min", type=float, required=True)
    ap.add_argument("--global-cap-min", type=float, default=150.0)
    ap.add_argument("--prior-gpu-min", type=float, default=20.0918)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-obs", type=int, default=105)
    ap.add_argument("--enable-gpu", action="store_true")
    a = ap.parse_args()
    if not a.enable_gpu: sys.exit("--enable-gpu not given")
    out = os.path.join(a.work, "run", a.phase)
    if os.path.isdir(out) and any(os.scandir(out)): sys.exit("output dir not empty: " + out)
    os.makedirs(out)
    json.dump(fingerprint(a.work), open(os.path.join(out, "RUN_FINGERPRINT.json"), "w"), indent=1)
    led = Ledger(a.ledger, a.global_cap_min); led.tick(f"{a.tag}_{a.phase}")
    t_phase = time.time()
    cap_check = lambda: ((time.time() - t_phase) / 60 > a.phase_cap_min) or (a.prior_gpu_min + led.used() > a.global_cap_min)  # noqa: E731
    m = SVLA()
    json.dump({"gpu": torch.cuda.get_device_name(0), "generation_config": m.vla.generation_config.to_dict(),
               "attn_implementation": getattr(m.vla.config.text_config, "_attn_implementation", None), "load_s": time.time() - t_phase},
              open(os.path.join(out, "run_meta.json"), "w"), indent=1, default=str)
    M = json.load(open(a.manifest)); calls = tuple(M["calls"])
    status = {"phase": a.phase, "tag": a.tag}
    S = lambda: json.dump({**status, "ledger_min_this_work": led.used(), "phase_min": (time.time() - t_phase) / 60},  # noqa: E731
                          open(os.path.join(out, "STATUS.json"), "w"), indent=1)
    recf = open(os.path.join(out, "records.jsonl"), "w")
    if a.phase == "recovered":   # A1 (drawer): observations recovered from an interrupted collection; no new collection
        obs = json.load(open(a.recovered_json))["observations"]
        for o in obs:
            o["png"] = os.path.join(a.work, o["png_source"])
    else:
        eps = [{"task": x["task"], "seed": x["seed"], "tag": a.tag} for x in M["layouts"]]
        obs = collect(m, eps, calls, out, led, cap_check)
    json.dump(obs, open(os.path.join(out, "collected.json"), "w"), indent=1)
    olist = [o for o in obs if "obs_id" in o][:a.max_obs]
    status["n_collected"] = len(olist); S()
    done = 0
    for o in olist:
        if cap_check():
            status["stop_reason"] = "budget cap"; break
        r = run_obs(m, o, out, cap_check, with_arms=(a.phase in ("main", "recovered")))
        recf.write(json.dumps(r) + "\n"); recf.flush(); led.tick(); done += 1
        if r["status"] == "PARITY_FAIL":
            status.setdefault("parity_fail", []).append(r["obs_id"])
        if done % 10 == 0:
            status["n_done"] = done; S()
    status["n_done"] = done; S(); led.tick()


if __name__ == "__main__":
    main()
