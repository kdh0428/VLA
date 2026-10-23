#!/usr/bin/env python
"""VLA-0 native-path validation (--mode validate) and within-call pilot (--mode pilot). GPU; requires --enable-new-inference.
Protocol: pkg/PROTOCOL_BEFORE_VLA0_GPU.md. Records: one JSON line per observation (raw ids/text, audits, donors, arms, cost)."""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vla0_adapter as V  # noqa: E402

ACT, H, ANCHOR = 7, 8, 3


def seed_of(s):
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "little")


def donors_for(a, A, cands, ref_ids):
    """first candidate per target vector satisfying the protocol rule; returns {j: (cand_idx, vector)} and audit list."""
    out, log = {}, []
    for j in (1, 2, 3):
        chosen = None
        for ci, c in enumerate(cands):
            if not c["audit"]["in_range"] or c["audit"]["n_bad_fields"]:
                continue
            nums = c["audit"]["numbers"]
            if len(nums) < j * ACT:
                continue
            vec = nums[(j - 1) * ACT:j * ACT]
            ref_vec = A[(j - 1) * ACT:j * ACT]
            if vec == ref_vec:
                continue
            mod = list(A[:ANCHOR * ACT]); mod[(j - 1) * ACT:j * ACT] = vec
            mod_ids = V.tok(a, V.prefix_text(mod, ANCHOR * ACT))
            spans = V.vector_spans(a, A, ANCHOR)
            s, e = spans[j - 1]
            ok_len = len(mod_ids) == len(ref_ids)
            ok_out = ok_len and mod_ids[:s] == ref_ids[:s] and mod_ids[e:] == ref_ids[e:]
            log.append({"target_vector": j, "cand": ci, "len_match": ok_len, "outside_identical": ok_out})
            if ok_len and ok_out:
                chosen = (ci, vec); break
        out[j] = chosen
    return out, log


def run_arm(a, mi, A, donors, which):
    mod = list(A[:ANCHOR * ACT])
    for j in which:
        mod[(j - 1) * ACT:j * ACT] = donors[j][1]
    pids = V.tok(a, V.prefix_text(mod, ANCHOR * ACT))
    t0 = time.time()
    ids = V.generate(a, mi, prefix_ids=pids)[0]
    au = V.audit(a, ids, expected=(H - ANCHOR) * ACT)
    return {"edited_vectors": list(which), "prefix_numbers": mod, "prefix_ids": pids, "ids": ids, "audit": au,
            "latency_s": time.time() - t0}


HF_REVISION = "6c62a511ad56042b2b3ea0a90c4def33e1ea3b96"
UPSTREAM_COMMIT = "b78db19eda67b02844fe82b8929e1f94de7f2dcc"
UPSTREAM_TARBALL_SHA = "b8c1ce72b557a762ed74a3b531637c4edd68f4317a31189c1da73e9484324968"
VERIFIED_SHARDS = {"model-00001-of-00002.safetensors": "2021480ff1b4fb6e635195591615d7cd93d7fba492865375d6f49722b4415e4a",
                   "model-00002-of-00002.safetensors": "2fd9786a84cf652f432173bb4f69951eaaca3d840995f730cd7ec4496b98d659"}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
                   "special_tokens_map.json", "chat_template.json", "preprocessor_config.json", "generation_config.json", "config.json",
                   "model.safetensors.index.json")


def _h(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint(g):
    """Computed BEFORE model loading (2026-10-21 fix). Weight identity = streaming sha256 of the actual shards, which must equal the
    verified HF LFS hashes of revision HF_REVISION; tokenizer/processor files, upstream code tarball, config, stats, manifests, code."""
    here = os.path.dirname(os.path.abspath(__file__))
    shards = {f: _h(os.path.join(V.WEIGHTS, f)) for f in VERIFIED_SHARDS}
    bad = [f for f, x in shards.items() if x != VERIFIED_SHARDS[f]]
    if bad:
        sys.exit(f"weight shards do not match the verified revision {HF_REVISION}: {bad}")
    tarball = os.path.join(os.path.dirname(V.REPO), "vla0_b78db19.tar.gz")
    if not os.path.exists(tarball) or _h(tarball) != UPSTREAM_TARBALL_SHA:
        sys.exit("upstream code tarball missing or not the recorded b78db19 tarball")
    ref = os.path.join(V.HF, "model_last")          # persistent copy of the revision's tokenizer/processor/config files
    tok = {}
    for f in TOKENIZER_FILES:
        x = _h(os.path.join(V.WEIGHTS, f)) if os.path.exists(os.path.join(V.WEIGHTS, f)) else "MISSING"
        if x != _h(os.path.join(ref, f)):
            sys.exit(f"tokenizer/processor/config file differs from the persistent revision copy: {f}")
        tok[f] = x
    return {"run_script": _h(os.path.abspath(__file__)), "adapter": _h(os.path.join(here, "vla0_adapter.py")),
            "upstream_commit": UPSTREAM_COMMIT, "upstream_tarball": UPSTREAM_TARBALL_SHA,
            "hf_revision": HF_REVISION, "weight_shards": shards, "tokenizer_processor": tok,
            "config_yaml": _h(os.path.join(V.HF, "config.yaml")), "dataset_stats": _h(os.path.join(V.HF, "dataset_stats.pkl")),
            "val_manifest": _h(g.val_manifest), "pilot_manifest": _h(g.pilot_manifest)}


def preflight(g):
    """Fail fast before loading the model or touching any file. Returns (fingerprint, resume: bool)."""
    fpp = os.path.join(g.out, "RUN_FINGERPRINT.json")
    nonempty = os.path.isdir(g.out) and any(os.scandir(g.out))
    if nonempty and not os.path.exists(fpp):
        sys.exit("output dir has results but no RUN_FINGERPRINT.json (legacy/completed run): refuse; use a NEW --out directory")
    fp = fingerprint(g)
    if os.path.exists(fpp):
        if json.load(open(fpp)) != fp:
            sys.exit("RUN_FINGERPRINT.json differs from the current code/model/config/stats/manifests: refuse to resume")
        return fp, True
    return fp, False


def validation_summary(records, n_expected):
    """Final validation gate (2026-10-21): every observation status OK and all gates true, including the strict arm gates;
    donor policy: the different-source check is REQUIRED (missing donor for vector 3 = FAIL)."""
    gates = ["native_text_equal", "native_parsed_equal", "raw_complete", "prefix_retokenises_to_generated",
             "prefix_equals_vector_concat", "selfpatch_token_identity", "suffix_starts_at_timestep4",
             "arms_complete_exact35_eos", "arm_invariants", "donor_check_invariants"]
    per, ok = [], len(records) == n_expected
    for r in records:
        c = r.get("checks", {})
        failed = [k for k in gates if c.get(k) is not True]
        if r.get("status") != "OK":
            failed.append(f"status={r.get('status')}")
        ok = ok and not failed
        per.append({"obs_id": r.get("obs_id"), "failed": failed})
    return {"status": "PASS" if ok else "FAIL", "gates": gates + ["status==OK", f"n_records=={n_expected}"],
            "donor_policy": "donor for vector 3 required in every validation observation", "per_obs": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--pilot-manifest", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--enable-new-inference", action="store_true", default=False)
    ap.add_argument("--gpu-cap-min", type=float, default=120.0)
    ap.add_argument("--release-shm-weights", action="store_true", default=False,
                    help="after loading to GPU, unlink the RAM-backed weight files (container RAM limit)")
    g = ap.parse_args()
    if not g.enable_new_inference:
        sys.exit("enable_new_inference=false")
    fp, resume = preflight(g)                       # before model load, weight release, or any metadata write
    t_start = time.time()
    a, cfg, kv = V.build_actor()
    released = []
    if g.release_shm_weights:
        for f in sorted(os.listdir(V.WEIGHTS)):
            if f.endswith(".safetensors"):
                os.remove(os.path.join(V.WEIGHTS, f)); released.append(f)
    os.makedirs(g.out, exist_ok=True)
    base = a.model.generation_config.to_dict()
    eff_greedy = dict(base, do_sample=False, max_new_tokens=a.num_tokens, logits_processor="NumberSpaceOnlyProcessor (+ForcePrefix for prefixes)")
    eff_source = dict(base, do_sample=True, temperature=0.7, num_return_sequences=4, max_new_tokens=a.num_tokens,
                      logits_processor="NumberSpaceOnlyProcessor", seed='sha256("vla0src:"+obs_id)[:4]')
    meta = {"generation_config_base": base, "effective_greedy": eff_greedy, "effective_source_sampling": eff_source,
            "load_s": time.time() - t_start, "released_shm_weight_files": released}
    if resume:                                       # never overwrite metadata of an existing run
        json.dump(meta, open(os.path.join(g.out, f"run_meta_resume_{int(time.time())}.json"), "w"), indent=1, default=str)
    else:
        json.dump(fp, open(os.path.join(g.out, "RUN_FINGERPRINT.json"), "w"), indent=1)
        json.dump(meta, open(os.path.join(g.out, "run_meta.json"), "w"), indent=1, default=str)
    vdir = os.path.join(g.out, "validation")
    vsp = os.path.join(vdir, "VALIDATION_SUMMARY.json")
    if not (resume and os.path.exists(vsp) and json.load(open(vsp))["status"] == "PASS"):
        run_mode(a, kv, "validate", g.val_manifest, g.obs, vdir, t_start, g.gpu_cap_min)
    vs = json.load(open(vsp))
    if vs["status"] != "PASS":
        print("validation FAIL: pilot not run", flush=True)
        return
    run_mode(a, kv, "pilot", g.pilot_manifest, g.obs, os.path.join(g.out, "pilot"), t_start, g.gpu_cap_min)
    json.dump({"gpu_minutes_total": (time.time() - t_start) / 60}, open(os.path.join(g.out, f"gpu_time_{int(t_start)}.json"), "w"))


def run_mode(a, kv, mode, manifest, obs, out, t_start, cap):
    a_ = argparse.Namespace(mode=mode, manifest=manifest, obs=obs, out=out, gpu_cap_min=cap, gpu_used_min=0.0)
    M = json.load(open(a_.manifest))
    os.makedirs(a_.out, exist_ok=True)
    dst = os.path.join(a_.out, "records.jsonl")
    done = {json.loads(l)["obs_id"] for l in open(dst)} if (os.path.exists(dst) and a_.mode == "pilot") else set()
    if a_.mode == "validate" and os.path.exists(dst):
        os.replace(dst, dst + f".prev_{int(time.time())}")
    for x in M["observations"]:
        if x["obs_id"] in done:
            continue
        used = a_.gpu_used_min + (time.time() - t_start) / 60
        if used > a_.gpu_cap_min:
            print("GPU cap reached; stopping (PARTIAL)", flush=True)
            break
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        z = np.load(os.path.join(a_.obs, f"{x['obs_id']}.npz"))
        rgb = V.obs_to_rgb(z["agentview"], z["wrist"], kv)
        mi = V.prompt_inputs(a, rgb, x["language"])
        rec = {"obs_id": x["obs_id"], "suite": x["suite"], "task_index": x["task_index"], "task_name": x["task_name"],
               "init_state_index": x["init_state_index"], "language": x["language"],
               "rgb_sha": hashlib.sha256(rgb.cpu().numpy().tobytes()).hexdigest()[:16]}
        # reference (adapter greedy) and, in validation, the native forward
        t1 = time.time()
        ids = V.generate(a, mi)[0]
        rec["ref_latency_s"] = time.time() - t1
        au = V.audit(a, ids)
        A = au["numbers"]
        rec["reference"] = {"ids": ids, "audit": au}
        chk = {}
        if a_.mode == "validate":
            with torch.no_grad():
                nat = a.forward(rgb=rgb, instr=[x["language"]], get_loss=False, get_action=True, generate_temperature=0.0)
            chk["native_text_equal"] = nat["pred_action_txt"][0] == V.decode(a, ids)
            chk["native_parsed_equal"] = bool(torch.equal(nat["out_ori_act"].cpu(),
                                                          a.get_action_from_text_action([V.decode(a, ids)])))
            rec["native"] = {"text": nat["pred_action_txt"][0], "out_ori_act": nat["out_ori_act"].cpu().numpy().tolist(),
                             "gt_out_ori_act_field_is_prediction": bool(torch.equal(nat["gt_out_ori_act"].cpu(), nat["out_ori_act"].cpu()))}
        chk["raw_complete"] = au["complete"] and au["in_range"] and au["eos"]
        if not chk["raw_complete"]:
            rec["status"] = "REFERENCE_INCOMPLETE"
            rec["checks"] = chk
            open(dst, "a").write(json.dumps(rec) + "\n")
            continue
        ref_txt = V.prefix_text(A, ANCHOR * ACT)
        ref_pids = V.tok(a, ref_txt)
        spans = V.vector_spans(a, A, ANCHOR)
        concat = sum((V.tok(a, V.prefix_text(A[j * ACT:(j + 1) * ACT], ACT)) for j in range(ANCHOR)), [])
        chk["prefix_retokenises_to_generated"] = ref_pids == ids[:len(ref_pids)]
        chk["prefix_equals_vector_concat"] = concat == ref_pids
        R = run_arm(a, mi, A, {}, [])
        chk["selfpatch_token_identity"] = R["ids"] == ids[len(ref_pids):]
        chk["suffix_starts_at_timestep4"] = R["audit"]["numbers"][:1] == A[ANCHOR * ACT:ANCHOR * ACT + 1]
        rec["arms"] = {"R": R}
        # source bank
        t2 = time.time()
        sids = V.generate(a, mi, temperature=0.7, seed=seed_of("vla0src:" + x["obs_id"]), num_return=4)
        rec["sources"] = [{"ids": s, "audit": V.audit(a, s)} for s in sids]
        rec["source_latency_s"] = time.time() - t2
        donors, dlog = donors_for(a, A, rec["sources"], ref_pids)
        rec["donor_search"] = dlog
        rec["donors"] = {str(j): (None if d is None else {"cand": d[0], "vector": d[1]}) for j, d in donors.items()}
        arms = {"O": [2], "L": [3], "OL": [2, 3], "H": [1, 2], "HL": [1, 2, 3]}
        if a_.mode == "validate":
            arms = {"L": [3]}
        for name, which in arms.items():
            if all(donors[j] is not None for j in which):
                arm = run_arm(a, mi, A, donors, which)
                s, e = spans[which[0] - 1][0], spans[which[-1] - 1][1]
                arm["invariant_len"] = len(arm["prefix_ids"]) == len(ref_pids)
                arm["invariant_outside"] = all(arm["prefix_ids"][k] == ref_pids[k] for k in range(len(ref_pids))
                                               if not any(spans[j - 1][0] <= k < spans[j - 1][1] for j in which))
                rec["arms"][name] = arm
        if "L" in rec["arms"] and a_.mode == "validate":
            chk["donor_check_invariants"] = rec["arms"]["L"]["invariant_len"] and rec["arms"]["L"]["invariant_outside"]
        arm_ok = all(v["audit"]["n_numbers"] == (H - ANCHOR) * ACT and v["audit"]["n_bad_fields"] == 0 and v["audit"]["in_range"]
                     and v["audit"]["eos"] for v in rec["arms"].values())
        # R is the self-patch (no edited span); every edited arm must carry both invariants explicitly (no default True)
        inv_ok = all(v.get("invariant_len") is True and v.get("invariant_outside") is True for k, v in rec["arms"].items() if k != "R")
        chk["arms_complete_exact35_eos"] = arm_ok
        chk["arm_invariants"] = inv_ok
        rec["checks"] = chk
        core = [k for k in ("raw_complete", "prefix_retokenises_to_generated", "prefix_equals_vector_concat",
                            "selfpatch_token_identity", "suffix_starts_at_timestep4", "arms_complete_exact35_eos", "arm_invariants")]
        rec["status"] = "OK" if all(chk.get(k) for k in core) else "CHECK_FAILED"
        rec["latency_s"] = time.time() - t0
        rec["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        rec["n_generate_calls"] = 1 + 1 + len(rec["arms"]) + (1 if a_.mode == "validate" else 0)
        open(dst, "a").write(json.dumps(rec) + "\n")
        print(x["obs_id"], {k: v for k, v in chk.items()}, round(rec["latency_s"], 1), flush=True)
    if a_.mode == "validate":
        R_ = [json.loads(l) for l in open(dst)]
        json.dump(validation_summary(R_, len(M["observations"])), open(os.path.join(a_.out, "VALIDATION_SUMMARY.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
