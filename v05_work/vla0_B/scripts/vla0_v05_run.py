#!/usr/bin/env python
"""v05 B/D: VLA-0 repetition-penalty control, donor follow-up and bridge regeneration (protocol PROTOCOL_B_D_VLA0_20261023_v01.md).
Reuses vla0_work/scripts/vla0_adapter.py unchanged (ForcePrefix forced decoding on the native incremental KV path).
Phases (one process, sequential, resumable by obs_id): validate -> main -> bridge -> donor. One JSON line per unit."""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/root/VLA/vla0_work/scripts")
import vla0_adapter as V  # noqa: E402

ACT, H, ANCHOR, BINS = 7, 8, 3, 1000
PENS = (1.05, 1.0)
ARMS = (("SELF", []), ("O", [2]), ("L", [3]), ("OL", [2, 3]))
T0 = time.time()


def sha(x):
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()[:16]


def seed_of(s):
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "little")


def set_pen(a, p):
    a.model.generation_config.repetition_penalty = p


@torch.no_grad()
def gen(a, mi, prefix_ids=None, temperature=0.0, seed=None, num_return=1, diag=False):
    """V.generate with optional raw/processed logits capture (diag)."""
    if not diag:
        return V.generate(a, mi, prefix_ids=prefix_ids, temperature=temperature, seed=seed, num_return=num_return), None
    m = dict(mi)
    L = m["input_ids"].shape[1]
    k = len(prefix_ids or [])
    procs = [a.logits_processor] + ([V.ForcePrefix(L, prefix_ids)] if k else [])
    out = a.model.generate(**m, logits_processor=procs, max_new_tokens=a.num_tokens + k, do_sample=False,
                           output_logits=True, output_scores=True, return_dict_in_generate=True)
    seq = out.sequences[0, L:].tolist()
    assert seq[:k] == list(prefix_ids or [])
    full = out.sequences[0].tolist()
    d = []
    for i in range(k, min(k + 8, len(out.logits))):
        raw, pr = out.logits[i][0].float(), out.scores[i][0].float()
        ch = seq[i]
        hist = set(full[:L + i])
        rt, pt = torch.topk(raw, 10), torch.topk(pr, 10)
        d.append({"step": i - k, "chosen": ch, "raw_chosen": float(raw[ch]), "proc_chosen": float(pr[ch]),
                  "raw_top10": [[int(t), float(v)] for v, t in zip(rt.values, rt.indices)],
                  "proc_top10": [[int(t), float(v)] for v, t in zip(pt.values, pt.indices)],
                  "n_masked": int(torch.isinf(pr).sum()), "history_size": len(hist), "chosen_in_history": ch in hist,
                  "raw_argmax_in_history": int(rt.indices[0]) in hist, "raw_argmax_equals_chosen": int(rt.indices[0]) == ch})
    return [seq[k:]], d


def edited_prefix(a, A, donors, which):
    mod = list(A[:ANCHOR * ACT])
    for j in which:
        mod[(j - 1) * ACT:j * ACT] = donors[j]
    return mod, V.tok(a, V.prefix_text(mod, ANCHOR * ACT))


def invariants(a, A, mod_ids, ref_ids, which):
    spans = V.vector_spans(a, A, ANCHOR)
    ok_len = len(mod_ids) == len(ref_ids)
    ok_out = ok_len and all(mod_ids[t] == ref_ids[t] for t in range(len(ref_ids))
                            if not any(spans[j - 1][0] <= t < spans[j - 1][1] for j in which))
    return ok_len, ok_out


def suffix_ok(au, n=(H - ANCHOR) * ACT):
    return au["n_numbers"] == n and au["n_bad_fields"] == 0 and au["in_range"] and au["eos"]


def first_mismatch(x, y):
    for i, (u, v) in enumerate(zip(x, y)):
        if u != v:
            return i
    return None if len(x) == len(y) else min(len(x), len(y))


def load_obs(a, kv, obs_dir, r):
    z = np.load(os.path.join(obs_dir, f"{r['obs_id']}.npz"))
    rgb = V.obs_to_rgb(z["agentview"], z["wrist"], kv)
    return rgb, V.prompt_inputs(a, rgb, r["language"])


def done_ids(p):
    return {json.loads(l)["obs_id"] for l in open(p)} if os.path.exists(p) else set()


def gpu_min():
    return (time.time() - T0) / 60


def phase_validate(a, kv, g, panel):
    dst = os.path.join(g.out, "validation.jsonl")
    done = done_ids(dst)
    for r in panel[:8]:
        if r["obs_id"] in done:
            continue
        rgb, mi = load_obs(a, kv, g.obs, r)
        A = r["reference"]["audit"]["numbers"]
        ref_pids = V.tok(a, V.prefix_text(A, ANCHOR * ACT))
        rec = {"obs_id": r["obs_id"], "rgb_sha_equal": hashlib.sha256(rgb.cpu().numpy().tobytes()).hexdigest()[:16] == r["rgb_sha"],
               "prefix_ids_equal_stored": ref_pids == r["arms"]["R"]["prefix_ids"], "by_pen": {}}
        for p in PENS:
            set_pen(a, p)
            c = {"effective_repetition_penalty": a.model.generation_config.repetition_penalty}
            free = V.generate(a, mi)[0]
            fau = V.audit(a, free)
            nat = a.forward(rgb=rgb, instr=[r["language"]], get_loss=False, get_action=True, generate_temperature=0.0)
            c["native_text_equal_adapter"] = nat["pred_action_txt"][0] == V.decode(a, free)
            c["free_complete"] = fau["complete"] and fau["eos"]
            c["free_prefix_equals_fixed_prefix"] = fau["numbers"][:ANCHOR * ACT] == A[:ANCHOR * ACT]
            fp = V.tok(a, V.prefix_text(fau["numbers"], ANCHOR * ACT))
            c["free_prefix_retokenises"] = fp == free[:len(fp)]
            own = V.generate(a, mi, prefix_ids=fp)[0]
            c["selfpatch_own_prefix_identity"] = own == free[len(fp):]
            s1, dg = gen(a, mi, prefix_ids=ref_pids, diag=True)
            s2 = V.generate(a, mi, prefix_ids=ref_pids)[0]
            c["self_fixed_prefix_deterministic"] = s1[0] == s2
            c["self_fixed_equals_free_if_same_prefix"] = (s1[0] == free[len(ref_pids):]) if c["free_prefix_equals_fixed_prefix"] else None
            if p == 1.05:
                c["self_fixed_equals_stored_R"] = s1[0] == r["arms"]["R"]["ids"]
            Lmod, Lids = edited_prefix(a, A, {3: r["donors"]["3"]["vector"]}, [3])
            _, dgL = gen(a, mi, prefix_ids=Lids, diag=True)
            c["diag_SELF"] = dg
            c["diag_L"] = dgL
            c["free_ids"] = free
            rec["by_pen"][str(p)] = c
        set_pen(a, 1.05)
        core = []
        for p in PENS:
            c = rec["by_pen"][str(p)]
            core += [c["native_text_equal_adapter"], c["selfpatch_own_prefix_identity"], c["self_fixed_prefix_deterministic"],
                     c["self_fixed_equals_free_if_same_prefix"] in (True, None), c["free_prefix_retokenises"]]
        core += [rec["rgb_sha_equal"], rec["prefix_ids_equal_stored"]]
        rec["gate_pass"] = all(core)
        rec["gpu_min_at_end"] = gpu_min()
        open(dst, "a").write(json.dumps(rec) + "\n")
        print("VAL", r["obs_id"], rec["gate_pass"], flush=True)
    V_ = [json.loads(l) for l in open(dst)]
    s = {"n": len(V_), "n_pass": sum(v["gate_pass"] for v in V_), "status": "PASS" if len(V_) == 8 and all(v["gate_pass"] for v in V_) else "FAIL"}
    json.dump(s, open(os.path.join(g.out, "VALIDATION_SUMMARY.json"), "w"), indent=1)
    return s


def run_arms(a, mi, A, donors, ref_pids, arms, p):
    set_pen(a, p)
    res = {}
    for name, which in arms:
        mod, pids = edited_prefix(a, A, donors, which)
        ok_len, ok_out = invariants(a, A, pids, ref_pids, which)
        t = time.time()
        ids = V.generate(a, mi, prefix_ids=pids)[0]
        au = V.audit(a, ids, expected=(H - ANCHOR) * ACT)
        res[name] = {"edited_vectors": which, "prefix_numbers": mod, "prefix_ids": pids, "ids": ids, "audit": au,
                     "invariant_len": ok_len, "invariant_outside": ok_out, "suffix_ok": suffix_ok(au), "latency_s": time.time() - t}
    set_pen(a, 1.05)
    return res


def phase_main(a, kv, g, panel):
    dst = os.path.join(g.out, "main.jsonl")
    done = done_ids(dst)
    for r in panel:
        if r["obs_id"] in done:
            continue
        if gpu_min() > g.cap_main:
            print("CAP main", flush=True); return "PARTIAL"
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        rgb, mi = load_obs(a, kv, g.obs, r)
        A = r["reference"]["audit"]["numbers"]
        ref_pids = V.tok(a, V.prefix_text(A, ANCHOR * ACT))
        donors = {2: r["donors"]["2"]["vector"], 3: r["donors"]["3"]["vector"]}
        rec = {"obs_id": r["obs_id"], "suite": r["suite"], "task_index": r["task_index"],
               "rgb_sha_equal": hashlib.sha256(rgb.cpu().numpy().tobytes()).hexdigest()[:16] == r["rgb_sha"],
               "prefix_sha": sha(ref_pids), "donor_sha": sha([donors[2], donors[3]]), "by_pen": {}}
        for p in PENS:
            rec["by_pen"][str(p)] = run_arms(a, mi, A, donors, ref_pids, ARMS, p)
        old = {"SELF": "R", "O": "O", "L": "L", "OL": "OL"}
        rec["replay_1.05_vs_stored"] = {k: {"equal": rec["by_pen"]["1.05"][k]["ids"] == r["arms"][old[k]]["ids"],
                                            "first_mismatch": first_mismatch(rec["by_pen"]["1.05"][k]["ids"], r["arms"][old[k]]["ids"])}
                                        for k in old}
        rec["latency_s"] = time.time() - t0
        rec["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        rec["n_generate_calls"] = 8
        open(dst, "a").write(json.dumps(rec) + "\n")
        print("MAIN", r["obs_id"], {k: v["equal"] for k, v in rec["replay_1.05_vs_stored"].items()}, round(rec["latency_s"], 1),
              round(gpu_min(), 1), flush=True)
    return "DONE"


def joint_valid(a, A, cand_au, ref_pids):
    if not (cand_au["complete"] and cand_au["in_range"] and cand_au["eos"]):
        return False, "incomplete"
    n = cand_au["numbers"]
    v2, v3 = n[ACT:2 * ACT], n[2 * ACT:3 * ACT]
    if v2 == A[ACT:2 * ACT] or v3 == A[2 * ACT:3 * ACT]:
        return False, "identical_vector"
    dons = {2: v2, 3: v3}
    for which in ([2], [3], [2, 3]):
        _, pids = edited_prefix(a, A, dons, which)
        l, o = invariants(a, A, pids, ref_pids, which)
        if not (l and o):
            return False, f"invariant_fail_{which}"
    dO = float(np.abs(np.array(v2[:6]) - np.array(A[ACT:ACT + 6])).mean() / BINS)
    dL = float(np.abs(np.array(v3[:6]) - np.array(A[2 * ACT:2 * ACT + 6])).mean() / BINS)
    if dO <= 0 or dL <= 0:
        return False, "zero_motion_distance"
    if not (0.8 <= dL / dO <= 1.25):
        return False, "ratio_out_of_range"
    return True, "accepted"


def phase_donor(a, kv, g, panel):
    dst = os.path.join(g.out, "donor_followup.jsonl")
    done = done_ids(dst)
    main = {json.loads(l)["obs_id"]: json.loads(l) for l in open(os.path.join(g.out, "main.jsonl"))}
    for r in panel:
        if r["obs_id"] in done or r["obs_id"] not in main:
            continue
        if gpu_min() > g.cap_donor:
            print("CAP donor", flush=True); return "PARTIAL"
        t0 = time.time()
        rgb, mi = load_obs(a, kv, g.obs, r)
        A = r["reference"]["audit"]["numbers"]
        ref_pids = V.tok(a, V.prefix_text(A, ANCHOR * ACT))
        set_pen(a, 1.05)
        log, chosen, n_samp = [], None, 0
        for b in range(8):
            sd = seed_of(f"vla0_v05_donor:{r['obs_id']}:{b}")
            sids = V.generate(a, mi, temperature=0.7, seed=sd, num_return=8)
            for i, s in enumerate(sids):
                n_samp += 1
                au = V.audit(a, s)
                ok, why = joint_valid(a, A, au, ref_pids)
                log.append({"batch": b, "index": i, "seed": sd, "reason": why, "ids_sha": sha(s)})
                if ok and chosen is None:
                    chosen = {"batch": b, "index": i, "seed": sd, "ids": s, "numbers": au["numbers"]}
            if chosen is not None:
                break
        rec = {"obs_id": r["obs_id"], "suite": r["suite"], "task_index": r["task_index"], "n_samples": n_samp, "search_log": log}
        if chosen is None:
            rec["status"] = "SOURCE_MISSING"
        else:
            n = chosen["numbers"]
            donors = {2: n[ACT:2 * ACT], 3: n[2 * ACT:3 * ACT]}
            rec["donor"] = chosen
            rec["by_pen"] = {str(p): run_arms(a, mi, A, donors, ref_pids, ARMS[1:], p) for p in PENS}
            rec["status"] = "OK"
        rec["latency_s"] = time.time() - t0
        open(dst, "a").write(json.dumps(rec) + "\n")
        print("DONOR", r["obs_id"], rec["status"], n_samp, round(rec["latency_s"], 1), round(gpu_min(), 1), flush=True)
    return "DONE"


def phase_bridge(a, kv, g, panel):
    dst = os.path.join(g.out, "bridge.jsonl")
    done = done_ids(dst)
    seen, sel = set(), []
    for r in sorted(panel, key=lambda x: x["obs_id"]):
        t = (r["suite"], r["task_index"])
        if t in seen or r["donors"].get("2") is None:
            continue
        seen.add(t); sel.append(r)
    json.dump([r["obs_id"] for r in sel], open(os.path.join(g.out, "bridge_panel_ids.json"), "w"), indent=1)
    set_pen(a, 1.05)
    for r in sel:
        if r["obs_id"] in done:
            continue
        if gpu_min() > g.cap_bridge:
            print("CAP bridge", flush=True); return "PARTIAL"
        rgb, mi = load_obs(a, kv, g.obs, r)
        A = r["reference"]["audit"]["numbers"]
        ref_ids = r["reference"]["ids"]
        nat14 = V.tok(a, V.prefix_text(A, 2 * ACT))
        mod = list(A[:2 * ACT]); mod[ACT:2 * ACT] = r["donors"]["2"]["vector"]
        reg14 = V.tok(a, V.prefix_text(mod, 2 * ACT))
        rec = {"obs_id": r["obs_id"], "suite": r["suite"], "task_index": r["task_index"],
               "nat14_retokenises": nat14 == ref_ids[:len(nat14)], "len_equal": len(reg14) == len(nat14)}
        s_nat = V.generate(a, mi, prefix_ids=nat14)[0]
        rec["bridge_self_parity"] = s_nat == ref_ids[len(nat14):]
        s_reg = V.generate(a, mi, prefix_ids=reg14)[0]
        rec["BRIDGE_SELF"] = {"ids": s_nat, "audit": V.audit(a, s_nat, expected=(H - 2) * ACT)}
        rec["BRIDGE_REGEN"] = {"prefix_numbers": mod, "prefix_ids": reg14, "ids": s_reg, "audit": V.audit(a, s_reg, expected=(H - 2) * ACT)}
        rec["status"] = "OK" if rec["bridge_self_parity"] and rec["nat14_retokenises"] and suffix_ok(rec["BRIDGE_REGEN"]["audit"], (H - 2) * ACT) else "BLOCKED"
        open(dst, "a").write(json.dumps(rec) + "\n")
        print("BRIDGE", r["obs_id"], rec["status"], round(gpu_min(), 1), flush=True)
    return "DONE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="/root/VLA/vla0_work/run/pilot/records.jsonl")
    ap.add_argument("--obs", default="/root/VLA/vla0_work/obs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--phases", default="validate,main,donor,bridge")
    ap.add_argument("--cap-main", type=float, default=120.0)
    ap.add_argument("--cap-donor", type=float, default=225.0)
    ap.add_argument("--cap-bridge", type=float, default=135.0)
    ap.add_argument("--release-shm-weights", action="store_true")
    g = ap.parse_args()
    os.makedirs(g.out, exist_ok=True)
    panel = sorted([json.loads(l) for l in open(g.records)], key=lambda x: x["obs_id"])
    panel = [r for r in panel if r["status"] == "OK"]
    assert len(panel) == 149, len(panel)
    a, cfg, kv = V.build_actor()
    if g.release_shm_weights:
        for f in sorted(os.listdir(V.WEIGHTS)):
            if f.endswith(".safetensors"):
                os.remove(os.path.join(V.WEIGHTS, f))
    env = {"start_unix": T0, "gpu": torch.cuda.get_device_name(0), "vram_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
           "torch": torch.__version__, "transformers": __import__("transformers").__version__,
           "generation_config": a.model.generation_config.to_dict(), "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "device_order": os.environ.get("CUDA_DEVICE_ORDER"), "load_s": time.time() - T0}
    json.dump(env, open(os.path.join(g.out, f"run_env_{int(T0)}.json"), "w"), indent=1, default=str)
    st = {}
    ph = g.phases.split(",")
    if "validate" in ph:
        s = phase_validate(a, kv, g, panel)
        st["validate"] = s
        if s["status"] != "PASS":
            print("VALIDATION FAIL", s, flush=True)
            json.dump(st, open(os.path.join(g.out, f"phase_status_{int(T0)}.json"), "w"), indent=1)
            return
    for name, fn in (("main", phase_main), ("bridge", phase_bridge), ("donor", phase_donor)):
        if name in ph:
            st[name] = fn(a, kv, g, panel)
    st["gpu_minutes_this_process"] = gpu_min()
    json.dump(st, open(os.path.join(g.out, f"phase_status_{int(T0)}.json"), "w"), indent=1)
    print("ALL", st, flush=True)


if __name__ == "__main__":
    main()
