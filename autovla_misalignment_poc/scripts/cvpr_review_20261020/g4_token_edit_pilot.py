#!/usr/bin/env python
"""
G4 — action-token edit gain pilot for AutoVLA (utility pilot).  v2, 2026-10-20.  GPU path implemented, NOT executed here.

Intervention type (differs from the B2 mechanism experiment): an alternative token is EMITTED at one action step, i.e. it is
both the executed output and the context for later steps; the prefix before that step is the keep prefix. B2 instead fixes
the emitted prefix and edits conditioning only.

Sub-commands
  manifest   (CPU) 600-scene development manifest (stratified sample, pi, weights, seeds, split, scene hashes, fetch patterns)
  subset     (CPU) derive a small validation manifest (default: first 5 scenes in manifest order, both strata covered)
  smoke      (CPU) manifest schema/hashes, runtime camera check, record schema definition, opt-in guard (no model forward)
  run        (GPU) native harness decoding. Refuses unless --enable-new-inference AND, at run time, every selected scene has all
                   model camera files, all scene hashes match, and an existing output dir has an identical run fingerprint.
                   --mode validate (writes to <out>/validation, requires --check-identity, max 5 scenes, no resume skipping;
                                    gate = token identity vs Planner.plan + candidate invariants + schema; entropy recorded only)
                   --mode pilot    (writes to <out>/pilot; requires <out>/validation/VALIDATION_SUMMARY.json with status PASS whose
                                    model/config/code fingerprint equals the current one and whose parent manifest is this manifest;
                                    any invariant failure is logged to errors.jsonl and stops the run; only valid scenes are
                                    recorded and only those count as done on resume)
  analyze    (CPU) pre-specified gate analysis at equal WEIGHTED budgets + CONTINUE / STOP / NOT_DECIDED (PDMS required)
  evaluate   (CPU) tokens -> trajectories (codebook rollout) -> FDE5 / rule / severe / optional official non-reactive PDMS ->
                   Hajek-weighted frame estimates and F-census estimates for keep, token-edit candidates and the equal-budget
                   whole-plan baseline.
  selftest-eval (CPU) checks the evaluation path on STORED best-of-N keep plans (reconstruction and metrics), not on G4 outputs.

Action index convention: 0-based. Step s emits segment (0.5 s, s*0.5 .. (s+1)*0.5 s]. k_hat is chosen in s = 1..5 inclusive
(2nd to 6th token; never s = 0) as the step with maximal keep entropy (ties -> smallest s).
Budget per scene: keep (1 decode, stores top-5 logits at s = 1..5) + 4 token-edit candidates = 5 decodes; equal-budget baseline =
keep + 4 whole-plan samples at T = 1.0 from the same stub (seed keys "<key>:wp<j>"). Each decode re-runs the prompt prefill
(prefill count is recorded); timing and peak VRAM are recorded per scene.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time

POC = "/root/VLA/autovla_misalignment_poc"
REPO = "/root/VLA"
SCENES = f"{REPO}/autovla/dataset/nuplan/navtest_ext"
BLOBS = f"{REPO}/autovla/dataset/nuplan/sensor_blobs/test"
SPLIT = f"{POC}/outputs/review_followup/P1C_new_log_replication/design_20261008/split.csv"
DEV = [f"{POC}/outputs/expanded_best_of_n/gpu0/records.jsonl", f"{POC}/outputs/expanded_best_of_n/gpu1/records.jsonl"]
CKPT = f"{REPO}/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt"
CONFIG = f"{REPO}/autovla/config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"
PLANNER = f"{POC}/scripts/_planner.py"
N_PILOT, TOPK, KHAT_RANGE, N_WHOLEPLAN = 600, 4, (1, 5), 4
MODEL_CAMS = ("front_camera_paths", "left_camera_paths", "right_camera_paths")   # nuPlan: AutoVLAAgent uses left/right (CAM_L1/CAM_R1)
ARM_FIELDS = ("tokens", "entropy", "logprob", "edit_step", "edit_token", "temperature", "seed_key", "latency_s")
RECORD_FIELDS = ("token", "log", "stratum", "pi", "weight", "key", "k_hat", "top_at_steps", "keep", "candidates", "wholeplan",
                 "keep_matches_stored_candidate0", "prefix_sha", "stub_len", "stub_generated_tokens", "cot", "n_decodes",
                 "action_prefills", "action_rows_decoded", "pool_latency_s", "latency_s", "peak_vram_gb", "run_fingerprint")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def fingerprint_large(p, chunk=64 << 20):
    """size + first/last 64 MB sha256 (a full 16 GB hash per run is avoided; documented as partial)."""
    n = os.path.getsize(p)
    h = hashlib.sha256(str(n).encode())
    with open(p, "rb") as f:
        h.update(f.read(chunk)); f.seek(max(0, n - chunk)); h.update(f.read(chunk))
    return h.hexdigest()


def order_key(token):
    return hashlib.sha256(f"g4pilot:{token}".encode()).hexdigest()


def scene_cams(scene):
    return [c for k in MODEL_CAMS for c in scene[k]]


def cams_present(scene_json):
    sc = json.load(open(scene_json))
    cams = scene_cams(sc)
    return sum(os.path.exists(f"{BLOBS}/{c}") for c in cams), len(cams)


# ------------------------------------------------------------------ CPU: manifest / subset
def build_manifest(out):
    sys.path.insert(0, f"{POC}/scripts")
    import numpy as np
    from analyze_action_history import a_eval  # noqa: E402  (outcome used ONLY for stratification)
    shard_of = {r["log"]: r["shard"] for r in csv.DictReader(open(SPLIT))}
    rows = []
    for p in DEV:
        for line in open(p):
            r = json.loads(line)
            if r.get("stub_cot"):
                continue
            k = r["candidates"][0]
            ok = a_eval(k["trajectory_pred"], r["trajectory_gt"])
            f = float(np.linalg.norm(np.asarray(k["trajectory_pred"])[-1] - np.asarray(r["trajectory_gt"])[-1, :2]))
            rows.append({"token": r["token"], "log": r["log"], "natural_rule_failure": int(not ok),
                         "natural_severe": int((not ok) and f > 3.0), "stored_keep_tokens": json.dumps(k["action_idx"])})
    F = [x for x in rows if x["natural_rule_failure"]]
    S = sorted([x for x in rows if not x["natural_rule_failure"]], key=lambda x: order_key(x["token"]))
    m_s = N_PILOT - len(F)
    pi_s = m_s / len(S)
    sel = [dict(x, stratum="F_rule_failure_census", pi=1.0, weight=1.0) for x in F] + \
          [dict(x, stratum="S_sha_order", pi=pi_s, weight=1.0 / pi_s) for x in S[:m_s]]
    man, need_bytes, patterns = [], 0, {}
    for x in sel:
        sp = f"{SCENES}/{x['token']}.json"
        have, n = cams_present(sp)
        sh = shard_of.get(x["log"], "UNKNOWN")
        for c in scene_cams(json.load(open(sp))):
            patterns.setdefault(sh, set()).add("*/" + c)
        need_bytes += 220_000 * (n - have)
        man.append({**x, "split": "development_pilot", "shard": sh, "key": f"0:{x['token']}", "seed": 0,
                    "scene_json": os.path.relpath(sp, REPO), "scene_json_sha256": sha(sp),
                    "n_model_camera_files": n, "n_present_at_manifest_time": have})
    os.makedirs(out, exist_ok=True)
    meta = {"n_scenes": len(man), "n_F_census": len(F), "n_F_severe_subset": sum(x["natural_severe"] for x in F),
            "n_S_frame": len(S), "m_S": m_s, "pi_S": pi_s, "frame_scenes": len(rows),
            "frame": "expanded_best_of_n development scenes without CoT stub",
            "estimator": "Hajek weighted mean (weights 1/pi, sum = frame size); F census reported separately",
            "F_definition": "natural plan rejected by the coarse rule (A=0); severe (A=0 and FDE5>3 m) is a subset",
            "camera_files_missing_scenes_at_manifest_time": sum(x["n_present_at_manifest_time"] < x["n_model_camera_files"] for x in man),
            "camera_check_note": "informational only; `run` re-checks every scene's camera files at run time",
            "estimated_fetch_bytes": need_bytes, "estimate_basis": "220 kB per missing jpg (file-count estimate, not measured)",
            "action_index_convention": "0-based; k_hat in s = 1..5 inclusive",
            "input_hashes": {os.path.relpath(p, REPO): sha(p) for p in DEV + [SPLIT]}}
    json.dump({"meta": meta, "scenes": man}, open(f"{out}/g4_manifest.json", "w"), indent=1)
    with open(f"{out}/g4_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0])); w.writeheader(); w.writerows(man)
    for sh, pats in patterns.items():
        open(f"{out}/g4_fetch_patterns_shard_{sh}.txt", "w").write("\n".join(sorted(pats)) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "input_hashes"}, indent=1))


def make_subset(manifest, out, n=5):
    M = json.load(open(manifest))
    F = [x for x in M["scenes"] if x["stratum"].startswith("F")][:2]
    S = [x for x in M["scenes"] if x["stratum"].startswith("S")][:n - len(F)]
    sub = {"meta": dict(M["meta"], n_scenes=len(F + S), role="validation subset (implementation check only; not analysed)",
                        parent_manifest_sha256=sha(manifest)), "scenes": F + S}
    json.dump(sub, open(out, "w"), indent=1)
    print(out, [x["token"] for x in F + S])


# ------------------------------------------------------------------ checks shared by smoke and run
def runtime_checks(M, need_cameras=True):
    errs, missing = [], []
    for x in M["scenes"]:
        p = f"{REPO}/{x['scene_json']}"
        if not os.path.exists(p) or sha(p) != x["scene_json_sha256"]:
            errs.append(f"scene json changed/missing: {x['token']}")
            continue
        have, n = cams_present(p)
        if have < n:
            missing.append(x["token"])
    if need_cameras and missing:
        errs.append(f"{len(missing)} scenes lack camera files at run time")
    return errs, missing


def run_fingerprint(manifest):
    """JSON-normalised (tuples -> lists) so a saved fingerprint compares equal after a round trip."""
    fp = {"manifest_sha256": sha(manifest), "checkpoint_partial_sha256": fingerprint_large(CKPT), "config_sha256": sha(CONFIG),
          "planner_sha256": sha(PLANNER), "script_sha256": sha(os.path.abspath(__file__)),
          "topk": TOPK, "khat_range": list(KHAT_RANGE), "n_wholeplan": N_WHOLEPLAN}
    return json.loads(json.dumps(fp))


MODEL_FIELDS = ("checkpoint_partial_sha256", "config_sha256", "planner_sha256", "script_sha256", "topk", "khat_range", "n_wholeplan")


def parent_of(manifest):
    """sha256 of the full manifest a run belongs to (a validation subset points to its parent)."""
    m = json.load(open(manifest))["meta"]
    return m.get("parent_manifest_sha256") or sha(manifest)


def smoke(manifest, out_json):
    M = json.load(open(manifest))
    errs = []
    req = {"token", "log", "stratum", "pi", "weight", "split", "shard", "key", "seed", "scene_json", "scene_json_sha256"}
    for x in M["scenes"]:
        if not req <= set(x):
            errs.append(f"missing fields {req - set(x)} in {x.get('token')}")
        if not (0 < x["pi"] <= 1) or abs(x["weight"] * x["pi"] - 1) > 1e-9:
            errs.append(f"bad pi/weight {x['token']}")
    toks = [x["token"] for x in M["scenes"]]
    if len(set(toks)) != len(toks):
        errs.append("duplicate tokens")
    for p, h in M["meta"]["input_hashes"].items():
        if sha(f"{REPO}/{p}") != h:
            errs.append(f"input hash changed: {p}")
    rerrs, missing = runtime_checks(M, need_cameras=False)
    errs += rerrs
    code = os.system(f"{sys.executable} {os.path.abspath(__file__)} run --mode validate --manifest {manifest} "
                     f"--out /nonexistent_g4 > /dev/null 2>&1")
    rep = {"manifest": manifest, "scenes": len(toks), "errors": errs, "gpu_entry_refuses_without_opt_in": code != 0,
           "scenes_missing_camera_files_now": len(missing), "record_fields": list(RECORD_FIELDS), "arm_fields": list(ARM_FIELDS),
           "status_definitions": "PASS only for executed checks; GPU checks are NOT_RUN until `run --mode validate` is executed",
           "gpu_checks": {"identity_vs_Planner.plan": "NOT_RUN", "candidate_prefix_identity": "NOT_RUN",
                          "four_valid_candidates": "NOT_RUN", "latency_vram": "NOT_RUN"}}
    json.dump(rep, open(out_json, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("record_fields", "arm_fields")}, indent=1))
    return 0 if not errs and code != 0 else 1


# ------------------------------------------------------------------ GPU: native harness
def make_edit_planner():
    sys.path.insert(0, f"{POC}/scripts")
    import torch
    from _planner import N_ACT, Planner, seed_of

    class EditPlanner(Planner):
        """Planner.plan() loop plus (i) an optional emitted token at edit_step, (ii) top-k logits at chosen steps,
        (iii) an optional temperature override (whole-plan samples). Without options it equals Planner.plan."""

        @torch.no_grad()
        def prompt_inputs(self, scene):
            feats = {}
            for b in self.agent.get_feature_builders():
                feats.update(b.compute_features(dict(scene)))
            self._vis.clear()
            return {k: v.to(self.dev) for k, v in self.model.get_prompt(feats).items() if isinstance(v, torch.Tensor)}

        @torch.no_grad()
        def plan_edit(self, scene, key, mi, stub=None, edit_step=None, edit_token=None, topk_steps=(), temperature=None):
            t0 = time.time()
            llm, A0 = self.llm, self.A0
            prompt = mi["input_ids"][0].tolist()
            if stub is None:
                stem_ids = torch.tensor([prompt + self.stem], device=self.dev)
                m1 = dict(mi); m1["input_ids"] = stem_ids; m1["attention_mask"] = torch.ones_like(stem_ids)
                torch.manual_seed(seed_of(key, "stub"))
                g = llm.generate(**m1, max_new_tokens=160, do_sample=True, temperature=self.gen_conf["temperature"],
                                 top_k=self.gen_conf["top_k"], top_p=self.gen_conf["top_p"])
                new = g[0][stem_ids.shape[1]:].tolist()
                fa = next((j for j, x in enumerate(new) if x >= A0), None)
                if fa is None:
                    raise RuntimeError("no action token within 160 generated tokens")
                stub = self.stem + new[:fa]
            pre = prompt + stub
            P = len(pre)
            ids = torch.tensor([pre], device=self.dev)
            llm.rope_deltas = None
            out = llm(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values_videos=mi.get("pixel_values_videos"),
                      video_grid_thw=mi.get("video_grid_thw"), use_cache=True, cache_position=torch.arange(P, device=self.dev))
            cache, last = out.past_key_values, out.logits[:, -1, A0:A0 + 2048].double()
            del out
            temp = self.temp if temperature is None else float(temperature)
            own, ents, lps, top = [], [], [], {}
            for s in range(N_ACT):
                if s > 0:
                    sids = torch.tensor([[A0 + a for a in own[:s]]], device=self.dev)
                    o = llm(input_ids=sids, attention_mask=torch.ones((1, P + s), device=self.dev, dtype=torch.long),
                            past_key_values=cache, use_cache=True, cache_position=torch.arange(P, P + s, device=self.dev))
                    cache.crop(P)
                    last = o.logits[:, -1, A0:A0 + 2048].double()
                    del o
                lp = torch.log_softmax(last[0], -1)
                ents.append(float(-(lp.exp() * lp).sum()))
                if s in topk_steps:
                    v, i = torch.topk(last[0], TOPK + 1)
                    top[s] = {"logits": v.tolist(), "tokens": i.tolist()}
                if edit_step is not None and s == edit_step:
                    tok = int(edit_token)
                else:
                    gen = torch.Generator(); gen.manual_seed(seed_of(key, s))
                    tok = int(torch.multinomial(torch.softmax(last[0] / temp, -1).cpu(), 1, generator=gen))
                own.append(tok); lps.append(float(lp[tok]))
            del cache
            return {"tokens": own, "entropy": ents, "logprob": lps, "top": top, "stub": stub, "stub_len": len(stub),
                    "cot": "complex" in self.tok.decode(stub), "edit_step": edit_step, "edit_token": edit_token,
                    "temperature": temp, "seed_key": key, "latency_s": time.time() - t0}
    return EditPlanner, torch


def run(a):
    if not a.enable_new_inference:
        sys.exit("enable_new_inference=false: GPU pilot not approved; nothing was run.")
    if a.mode == "validate" and not a.check_identity:
        sys.exit("--mode validate requires --check-identity")
    M = json.load(open(a.manifest))
    scenes = M["scenes"][: a.max_scenes] if a.max_scenes else M["scenes"]
    if a.mode == "validate" and len(scenes) > 5:
        sys.exit("validation runs at most 5 scenes (use `subset` or --max-scenes 5)")
    errs, _ = runtime_checks({"scenes": scenes}, need_cameras=True)
    if errs:
        sys.exit("runtime input check failed: " + "; ".join(errs[:5]))
    out = os.path.join(a.out, "validation" if a.mode == "validate" else "pilot")
    fp = run_fingerprint(a.manifest)
    if a.mode == "pilot":
        vs = os.path.join(a.out, "validation", "VALIDATION_SUMMARY.json")
        if not os.path.exists(vs):
            sys.exit("pilot requires <out>/validation/VALIDATION_SUMMARY.json with status PASS")
        V = json.load(open(vs))
        if V.get("status") != "PASS":
            sys.exit("validation summary is not PASS")
        bad = [k for k in MODEL_FIELDS if V.get("run_fingerprint", {}).get(k) != fp[k]]
        if bad:
            sys.exit(f"validation PASS was produced with different {bad}: re-validate")
        if V.get("parent_manifest_sha256") != parent_of(a.manifest):
            sys.exit("validation PASS belongs to a different manifest family")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        sys.exit("launch with CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1")
    os.makedirs(out, exist_ok=True)
    fpp = os.path.join(out, "RUN_FINGERPRINT.json")
    if a.mode == "pilot" and os.path.exists(fpp) and json.load(open(fpp)) != fp:
        sys.exit("existing output was produced with a different manifest/checkpoint/config/code: refuse to append")
    json.dump(fp, open(fpp, "w"), indent=1)
    EditPlanner, torch = make_edit_planner()
    pl = EditPlanner()
    dst = os.path.join(out, "records.jsonl")
    errlog = os.path.join(out, "errors.jsonl")
    if a.mode == "validate":
        for f in (dst, errlog):
            if os.path.exists(f):
                os.replace(f, f + f".prev_{int(time.time())}")
        done = set()
    else:
        done = {json.loads(l)["token"] for l in open(dst)} if os.path.exists(dst) else set()   # successful scenes only
    checks = []
    for x in scenes:
        if x["token"] in done:
            continue
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        scene = json.load(open(f"{REPO}/{x['scene_json']}"))
        mi = pl.prompt_inputs(scene)
        lo, hi = KHAT_RANGE
        keep = pl.plan_edit(scene, x["key"], mi, topk_steps=tuple(range(lo, hi + 1)))
        n_dec = 1
        chk = {"token": x["token"]}
        if a.check_identity:
            ref = pl.plan(scene, key=x["key"]); n_dec += 1
            chk["identity_tokens"] = ref["tokens"] == keep["tokens"]                     # the gate (token identity only)
            chk["entropy_maxabs_recorded_not_gated"] = max(abs(u - v) for u, v in zip(ref["entropy"], keep["entropy"]))
        k_hat = lo + max(range(hi - lo + 1), key=lambda j: (keep["entropy"][lo + j], -j))
        alts = [t for t in keep["top"][k_hat]["tokens"] if t != keep["tokens"][k_hat]][:TOPK]
        cands = []
        for t in alts:
            c = pl.plan_edit(scene, x["key"], mi, stub=keep["stub"], edit_step=k_hat, edit_token=t); n_dec += 1
            cands.append({f: c[f] for f in ARM_FIELDS})
        wp = []
        for j in range(N_WHOLEPLAN):
            c = pl.plan_edit(scene, f"{x['key']}:wp{j}", mi, stub=keep["stub"], temperature=1.0); n_dec += 1
            wp.append({f: c[f] for f in ARM_FIELDS})
        chk["candidate_prefix_identity"] = all(c["tokens"][:k_hat] == keep["tokens"][:k_hat] for c in cands)
        chk["four_valid_candidates"] = len(cands) == TOPK and all(len(c["tokens"]) == 10 for c in cands) and \
            all(c["tokens"][k_hat] == t for c, t in zip(cands, alts)) and len(set(alts)) == TOPK
        rec = {"token": x["token"], "log": x["log"], "stratum": x["stratum"], "pi": x["pi"], "weight": x["weight"],
               "key": x["key"], "k_hat": k_hat, "top_at_steps": keep["top"], "keep": {f: keep[f] for f in ARM_FIELDS},
               "candidates": cands, "wholeplan": wp,
               "keep_matches_stored_candidate0": keep["tokens"] == json.loads(x["stored_keep_tokens"]),
               "prefix_sha": hashlib.sha256(str(keep["tokens"][:k_hat]).encode()).hexdigest()[:16],
               "stub_len": keep["stub_len"], "stub_generated_tokens": keep["stub_len"] - len(pl.stem), "cot": keep["cot"],
               "n_decodes": n_dec, "action_prefills": n_dec, "action_rows_decoded": 10 * n_dec,
               "cost_note": "action_prefills / action_rows_decoded count action decoding only; the stub generation "
                            "(llm.generate, once per scene) is reported as stub_generated_tokens",
               "pool_latency_s": {"token_edit": keep["latency_s"] + sum(c["latency_s"] for c in cands),
                                  "wholeplan": keep["latency_s"] + sum(c["latency_s"] for c in wp)},
               "latency_s": time.time() - t0, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
               "invariants": chk, "run_fingerprint": fp["script_sha256"][:16]}
        chk["schema_ok"] = all(f in rec for f in RECORD_FIELDS)
        failed = [k for k in ("candidate_prefix_identity", "four_valid_candidates", "schema_ok") if not chk[k]]
        if a.check_identity and not chk["identity_tokens"]:
            failed.append("identity_tokens")
        if failed:
            with open(errlog, "a") as fe:
                fe.write(json.dumps({"token": x["token"], "failed": failed, "checks": chk}) + "\n")
            if a.mode == "pilot":
                raise RuntimeError(f"invariant failed for {x['token']}: {failed} (not recorded as a completed scene)")
        else:
            with open(dst, "a") as fout:
                fout.write(json.dumps(rec) + "\n")
        checks.append({**chk, "failed": failed, "latency_s": rec["latency_s"], "peak_vram_gb": rec["peak_vram_gb"]})
    if a.mode == "validate":
        ok = len(checks) == len(scenes) and all(not c["failed"] for c in checks)
        json.dump({"status": "PASS" if ok else "FAIL", "gate": "token identity vs Planner.plan + candidate invariants + schema; "
                   "entropy differences are recorded, not gated", "checks": checks,
                   "run_fingerprint": fp, "validation_manifest_sha256": sha(a.manifest),
                   "parent_manifest_sha256": parent_of(a.manifest),
                   "note": "keep vs stored best-of-N candidate 0 may differ (B = 17 batch numerics); strata stay as in the manifest"},
                  open(os.path.join(out, "VALIDATION_SUMMARY.json"), "w"), indent=1)


# ------------------------------------------------------------------ CPU: evaluation
def _decoder():
    sys.path.insert(0, f"{REPO}/autovla")
    import pickle
    import numpy as np
    import torch
    from models.action_tokenizer import ActionTokenizer
    cb = torch.tensor(np.asarray(pickle.load(open(f"{REPO}/autovla/codebook_cache/agent_vocab.pkl", "rb"))["token_all"]["veh"]),
                      dtype=torch.float32)
    return lambda idx: ActionTokenizer.rollout(None, cb[list(idx)], time_steps=len(idx))[0, 1:].numpy()


def metrics(poses, gt):
    import numpy as np
    sys.path.insert(0, f"{POC}/scripts")
    from analyze_action_history import a_eval
    p = np.asarray(poses)[:, :2]; g = np.asarray(gt)[:, :2]
    e = np.linalg.norm(p - g, axis=1)
    ok = a_eval(p.tolist(), g.tolist())
    return {"fde5": float(e[-1]), "ade5": float(e.mean()), "rule_fail": int(not ok), "severe": int((not ok) and e[-1] > 3)}


def selftest_eval(out_json, n=5):
    """Evaluation-path check on STORED best-of-N keep plans: codebook reconstruction vs stored trajectory, metrics recomputed."""
    import numpy as np
    dec = _decoder()
    rows = []
    for i, line in enumerate(open(DEV[0])):
        if i >= n:
            break
        r = json.loads(line)
        k = r["candidates"][0]
        P = dec(k["action_idx"])
        rows.append({"token": r["token"], "recon_maxabs_m": float(np.abs(P[:, :2] - np.asarray(k["trajectory_pred"])).max()),
                     **metrics(P, r["trajectory_gt"])})
    rep = {"what": "evaluation path on stored keep plans (not G4 outputs)", "rows": rows,
           "status": "PASS" if all(x["recon_maxabs_m"] < 1e-3 for x in rows) else "FAIL"}
    json.dump(rep, open(out_json, "w"), indent=1)
    print(json.dumps(rep, indent=1))


def completeness(records_path, manifest_path):
    """Is the records file a complete run of the manifest? Returns (complete: bool, report: dict)."""
    M = json.load(open(manifest_path))
    want = {x["token"]: x for x in M["scenes"]}
    recs = [json.loads(l) for l in open(records_path)]
    toks = [r["token"] for r in recs]
    rep = {"manifest_scenes": len(want), "records": len(recs), "unique_records": len(set(toks)),
           "records_not_in_manifest": len(set(toks) - set(want)), "manifest_scenes_missing": len(set(want) - set(toks)),
           "bad_pool_sizes": sum(len(r["candidates"]) != TOPK or len(r["wholeplan"]) != N_WHOLEPLAN for r in recs),
           "weight_sum_records": float(sum(want[t]["weight"] for t in set(toks) if t in want)),
           "weight_sum_frame": float(M["meta"].get("frame_scenes", float("nan")))}
    ok = (rep["unique_records"] == rep["records"] == rep["manifest_scenes"] and rep["records_not_in_manifest"] == 0
          and rep["bad_pool_sizes"] == 0 and abs(rep["weight_sum_records"] - rep["weight_sum_frame"]) < 1e-6)
    return ok, rep


def evaluate(a):
    """records.jsonl -> per-arm metrics -> pool comparison. Hajek frame estimates are produced ONLY for a complete run of the
    600-scene manifest; otherwise every output is labelled PARTIAL_DIAGNOSTIC."""
    import numpy as np
    import pandas as pd
    complete, crep = completeness(a.records, a.manifest)
    label = "COMPLETE" if complete else "PARTIAL_DIAGNOSTIC"
    os.makedirs(a.out, exist_ok=True)
    json.dump({"status": label, **crep}, open(os.path.join(a.out, "g4_completeness.json"), "w"), indent=1)
    dec = _decoder()
    gt, rows = {}, []
    for line in open(a.records):
        r = json.loads(line)
        if r["token"] not in gt:
            gt[r["token"]] = json.load(open(f"{SCENES}/{r['token']}.json"))["gt_trajectory"]
        arms = [("keep", r["keep"])] + [(f"edit{j}", c) for j, c in enumerate(r["candidates"])] + \
               [(f"wholeplan{j}", c) for j, c in enumerate(r["wholeplan"])]
        for name, arm in arms:
            P = dec(arm["tokens"])
            m = metrics(P, gt[r["token"]])
            rows.append({"token": r["token"], "log": r["log"], "stratum": r["stratum"], "weight": r["weight"], "arm": name,
                         "mean_entropy": float(np.mean(arm["entropy"])), "max_entropy": float(np.max(arm["entropy"])),
                         "sum_logprob": float(np.sum(arm["logprob"])), "min_logprob": float(np.min(arm["logprob"])),
                         "final_x": float(P[-1, 0]), "final_y": float(P[-1, 1]), **m})
    D = pd.DataFrame(rows)
    if a.pdms:
        os.environ.setdefault("NUPLAN_MAPS_ROOT", f"{REPO}/autovla/dataset/nuplan/maps")
        import multiprocessing as mp
        sys.path.insert(0, f"{POC}/scripts")
        import pdm_score_candidates as PS
        recs = [json.loads(l) for l in open(a.records)]
        jobs = [(r["token"], r["log"], {n: (0, arm["tokens"]) for n, arm in
                 [("keep", r["keep"])] + [(f"edit{j}", c) for j, c in enumerate(r["candidates"])] +
                 [(f"wholeplan{j}", c) for j, c in enumerate(r["wholeplan"])]}) for r in recs]
        with mp.get_context("spawn").Pool(a.workers, initializer=PS._init, initargs=(sorted({j[1] for j in jobs}),)) as pool:
            res = pool.map(PS._score, jobs)
        pm = {(x["token"], k): v["score"] for x in res if "error" not in x for k, v in x.items() if k not in ("token", "log")}
        D["pdms"] = [pm.get((t, n), np.nan) for t, n in zip(D.token, D.arm)]
    D["status"] = label
    D.to_csv(os.path.join(a.out, "g4_arm_metrics.csv"), index=False)
    out = []
    for pool, prefix in (("token-edit (keep + 4 edits)", "edit"), ("whole-plan (keep + 4 samples)", "wholeplan")):
        pick = rank_sum_pick(D, prefix)
        keep = D[D.arm == "keep"].set_index("token").loc[pick.index]
        for est, mask in (("Hajek frame (development)" if complete else "PARTIAL_DIAGNOSTIC (weights not interpretable)",
                           np.ones(len(pick), bool)),
                          ("F census (rule failures)", keep.stratum.str.startswith("F").values)):
            w = keep.weight.values[mask] if not est.startswith("F census") else np.ones(mask.sum())
            changed = (pick.arm.values[mask] != "keep").astype(float)
            for m in ("fde5", "severe") + (("pdms",) if a.pdms else ()):
                d = pick[m].values[mask] - keep[m].values[mask]
                out.append({"status": label, "pool": pool, "estimand": est, "metric": m, "n_scenes": int(mask.sum()),
                            "delta_pick_minus_keep": float(np.average(d, weights=w)),
                            "weighted_share_changed": float(np.average(changed, weights=w)),
                            "unweighted_share_changed": float(changed.mean())})
    pd.DataFrame(out).to_csv(os.path.join(a.out, "g4_pool_comparison.csv"), index=False)
    print(label, json.dumps(crep)); print(pd.DataFrame(out).to_string())


def rank_sum_pick(D, prefix):
    sub = D[(D.arm == "keep") | D.arm.str.startswith(prefix)].copy()
    sub["r"] = sub.groupby("token").mean_entropy.rank() + sub.groupby("token").sum_logprob.rank(ascending=False)
    return sub.loc[sub.groupby("token").r.idxmin()].set_index("token")


RATES = (0.05, 0.10, 0.20, 0.30, 0.50)


def tie_key(token):
    return hashlib.sha256(f"g4tie:{token}".encode()).hexdigest()


def weighted_budget(scores, w, tokens, q):
    """Equal WEIGHTED budget, fixed before results: rank by score (desc), ties by sha256('g4tie:'+token); trigger scenes fully
    while cumulative weight <= q*W; the boundary scene is triggered with probability p = remaining budget / its weight.
    Returns (expected-policy trigger probabilities, deterministic floor trigger, weighted rate of the floor trigger).
    This is a batch-ranking / expected-policy evaluation at an exact weighted budget, not an online fixed-threshold evaluation."""
    import numpy as np
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], tie_key(tokens[i])))
    W, budget = float(np.sum(w)), q * float(np.sum(w))
    prob = np.zeros(len(scores)); cum = 0.0
    for i in order:
        if cum + w[i] <= budget + 1e-9:
            prob[i] = 1.0; cum += w[i]
        else:
            prob[i] = max(0.0, (budget - cum) / w[i]); break
    floor = prob >= 1.0
    return prob, floor, float(np.sum(w * floor) / W)


def decide(F, P, S, pdms_status):
    """F/P/S: dict gate -> {q: weighted delta (pick - keep) or (gate - keep)}.  pdms_status: 'complete' | 'absent' | 'missing'."""
    reasons = []
    if pdms_status != "complete":
        return {"decision": "NOT_DECIDED", "reasons": [f"PDMS {pdms_status}: the PDMS criterion cannot be evaluated"]}
    qs = [q for q in RATES if (F["gain"][q] <= F["entropy"][q] - 0.01 and F["gain"][q] <= F["random"][q] - 0.01 and
                                P["gain"][q] >= P["entropy"][q] + 0.003 and P["gain"][q] >= P["random"][q] + 0.003)]
    sev_ok = all(S["gain"][q] <= 1e-12 for q in RATES)
    if len(qs) < 3:
        reasons.append(f"FDE and PDMS jointly met at {len(qs)} of 5 budgets (need >= 3): {qs}")
    if not sev_ok:
        reasons.append("gain gate increases weighted severe events at >= 1 budget")
    return {"decision": "CONTINUE" if not reasons else "STOP", "joint_pass_budgets": qs, "severe_not_increased_all": sev_ok,
            "reasons": reasons}


def analyze(a):
    """Pre-specified gate analysis on the COMPLETE development pilot (PROTOCOL_BEFORE_GPU.md). Gates differ only in the
    trigger and apply the rank-sum pick of the token-edit pool. Log-grouped 5-fold CV; equal WEIGHTED budgets 5/10/20/30/50 %;
    Hajek-weighted effects; decision CONTINUE / STOP / NOT_DECIDED."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    st = json.load(open(os.path.join(a.eval_dir, "g4_completeness.json")))
    if st["status"] != "COMPLETE" and not a.allow_partial:
        sys.exit("analysis requires a COMPLETE evaluation (use --allow-partial only for a labelled diagnostic)")
    D = pd.read_csv(os.path.join(a.eval_dir, "g4_arm_metrics.csv"))
    pick = rank_sum_pick(D, "edit")
    wpick = rank_sum_pick(D, "wholeplan")
    keep = D[D.arm == "keep"].set_index("token").loc[pick.index]
    wpick = wpick.loc[pick.index]
    if "pdms" not in D.columns:
        pdms_status = "absent"
    elif not np.isfinite(D["pdms"].astype(float)).all():
        pdms_status = "missing"
    else:
        pdms_status = "complete"
    metrics_ = ["fde5", "severe"] + (["pdms"] if "pdms" in D.columns else [])
    feats = pd.DataFrame({"keep_ent_mean": keep.mean_entropy, "keep_ent_max": keep.max_entropy, "keep_lp_sum": keep.sum_logprob,
                          "keep_lp_min": keep.min_logprob, "pick_ent_mean": pick.mean_entropy, "pick_lp_sum": pick.sum_logprob,
                          "pick_is_keep": (pick.arm == "keep").astype(float),
                          "pick_keep_end_dist": np.hypot(pick.final_x - keep.final_x, pick.final_y - keep.final_y)})
    feats.to_csv(os.path.join(a.eval_dir, "g4_runtime_features.csv"))
    w = keep.weight.values.astype(float)
    tokens = list(pick.index)
    y_gain = (keep.fde5 - pick.fde5).values
    y_risk = keep.severe.values
    groups = keep.log.values
    oof = {"gain": np.zeros(len(feats)), "risk": np.zeros(len(feats))}
    folds = np.zeros(len(feats), int)
    for f, (tr, te) in enumerate(GroupKFold(5).split(feats, groups=groups)):
        folds[te] = f
        oof["gain"][te] = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(feats.values[tr], y_gain[tr]).predict(feats.values[te])
        if len(set(y_risk[tr])) > 1:
            oof["risk"][te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(
                feats.values[tr], y_risk[tr]).predict_proba(feats.values[te])[:, 1]
    pd.DataFrame({"token": tokens, "log": groups, "fold": folds, "weight": w, "oof_gain": oof["gain"], "oof_risk": oof["risk"],
                  "entropy_score": feats.keep_ent_mean.values}).to_csv(os.path.join(a.eval_dir, "g4_oof_predictions.csv"), index=False)
    scores = {"gain": oof["gain"], "risk": oof["risk"], "entropy": feats.keep_ent_mean.values}
    W = w.sum()
    rows, F, P, S = [], {}, {}, {}
    changed = (pick.arm.values != "keep").astype(float)
    for g in list(scores) + ["random", "always rank-sum", "whole-plan rank-sum (always)"]:
        F[g], P[g], S[g] = {}, {}, {}
        for q in RATES:
            if g in scores:
                prob, floor, frate = weighted_budget(scores[g], w, tokens, q)
            elif g == "random":
                prob, floor, frate = np.full(len(w), q), np.zeros(len(w), bool), float("nan")
            else:
                prob, floor, frate = np.ones(len(w)), np.ones(len(w), bool), 1.0
            src = wpick if g.startswith("whole-plan") else pick
            row = {"gate": g, "budget_q": q, "expected_weighted_trigger_rate": float(np.sum(w * prob) / W),
                   "deterministic_floor_weighted_rate": frate,
                   "expected_weighted_plan_change_rate": float(np.sum(w * prob * ((src.arm.values != "keep").astype(float))) / W)}
            for m in metrics_:
                d = prob * (src[m].values - keep[m].values)
                row[f"weighted_delta_{m}"] = float(np.sum(w * d) / W)
                dfl = np.where(floor, src[m].values - keep[m].values, 0.0)
                row[f"floor_weighted_delta_{m}"] = float(np.sum(w * dfl) / W)
            harm = (keep.severe.values == 0) & (src.severe.values == 1)
            ben = (keep.severe.values == 1) & (src.severe.values == 0)
            row["expected_weighted_harm_flip_rate"] = float(np.sum(w * prob * harm) / W)
            row["expected_weighted_benefit_flip_rate"] = float(np.sum(w * prob * ben) / W)
            rows.append(row)
            F[g][q] = row["weighted_delta_fde5"]; S[g][q] = row["weighted_delta_severe"]
            P[g][q] = row.get("weighted_delta_pdms", float("nan"))
    R = pd.DataFrame(rows)
    R["evaluation"] = "equal weighted budget, batch-ranking / expected-policy evaluation (not an online fixed threshold)"
    R["status"] = st["status"]
    R.to_csv(os.path.join(a.eval_dir, "g4_gate_analysis.csv"), index=False)
    if st["status"] != "COMPLETE":
        dec = {"decision": "NOT_DECIDED", "reasons": ["evaluation incomplete (PARTIAL_DIAGNOSTIC)"]}
    else:
        dec = decide(F, P, S, pdms_status)
    dec.update({"status": st["status"], "pdms_status": pdms_status,
                "always_rank_sum_fde": F["always rank-sum"], "wholeplan_fde": F["whole-plan rank-sum (always)"],
                "always_rank_sum_pdms": P["always rank-sum"], "wholeplan_pdms": P["whole-plan rank-sum (always)"]})
    json.dump(dec, open(os.path.join(a.eval_dir, "g4_continue_stop.json"), "w"), indent=1, default=float)
    print(R[["gate", "budget_q", "expected_weighted_trigger_rate", "weighted_delta_fde5", "weighted_delta_severe"] +
            (["weighted_delta_pdms"] if "pdms" in D else [])].round(4).to_string())
    print(json.dumps(dec, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("manifest"); p.add_argument("--out", required=True)
    p = sp.add_parser("subset"); p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=5)
    p = sp.add_parser("smoke"); p.add_argument("--manifest", required=True); p.add_argument("--out-json", required=True)
    p = sp.add_parser("run"); p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["validate", "pilot"], required=True)
    p.add_argument("--max-scenes", type=int, default=0)
    p.add_argument("--enable-new-inference", action="store_true", default=False)
    p.add_argument("--check-identity", action="store_true", default=False)
    p = sp.add_parser("evaluate"); p.add_argument("--records", required=True); p.add_argument("--out", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--pdms", action="store_true"); p.add_argument("--workers", type=int, default=6)
    p = sp.add_parser("analyze"); p.add_argument("--eval-dir", required=True); p.add_argument("--allow-partial", action="store_true")
    p = sp.add_parser("selftest-eval"); p.add_argument("--out-json", required=True)
    a = ap.parse_args()
    if a.cmd == "manifest":
        build_manifest(a.out)
    elif a.cmd == "subset":
        make_subset(a.manifest, a.out, a.n)
    elif a.cmd == "smoke":
        sys.exit(smoke(a.manifest, a.out_json))
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "evaluate":
        evaluate(a)
    elif a.cmd == "analyze":
        analyze(a)
    else:
        selftest_eval(a.out_json)


if __name__ == "__main__":
    main()
