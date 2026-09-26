#!/usr/bin/env python
"""
Mode-conditioned steering of the L35 MLP.

The geometry of the required corrections (see analyse output) predicts this will fail: the
per-sample directions w_GT - w_pred are near-orthogonal (mean pairwise cosine +0.002, 54
effective dimensions), and grouping them by GT motion primitive barely raises within-group
cosine. This experiment tests that prediction directly, because a geometric prediction is
not a result until the intervention is actually run.

Modes are the GT token's motion primitive (MAINTAIN / ACCELERATE / STOP) -- the coarse
"STOP->ACCELERATE" style modes the brief proposed do not exist at step 0: 74% of failures
are MAINTAIN->MAINTAIN and the predicted token is the GT token's 8th nearest codebook
neighbour (median), 0.079 m away.

Arms per alpha: global mean direction, own-mode direction, a deliberately WRONG mode's
direction, and a matched-norm random direction. Directions are fit on logs disjoint from
the test logs.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

AUTOVLA_DIR = "/root/VLA/autovla"
POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AUTOVLA_DIR)
sys.path.insert(0, os.path.join(AUTOVLA_DIR, "navsim"))
sys.path.insert(0, POC_DIR)

from src.labeling.labels import ade, fde  # noqa: E402
from src.labeling.token_primitive import token_primitive  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(POC_DIR, "outputs/gpu1/records.jsonl"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/mode_steering"))
    ap.add_argument("--config", default=os.path.join(
        AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--layer", type=int, default=35)
    ap.add_argument("--alphas", default="2.0,4.0")
    ap.add_argument("--n-fit-per-mode", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=90)
    ap.add_argument("--n-ctrl", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for k in ("output", "records", "scenes"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.makedirs(args.output, exist_ok=True)
    os.chdir(AUTOVLA_DIR)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    from models.autovla import AutoVLA
    from navsim.agents.autovla_agent import AutoVLAAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["pretrained_model_path"] = os.path.join(AUTOVLA_DIR, "Qwen2.5-VL-3B-Instruct")
    cfg["model"]["codebook_cache_path"] = os.path.join(AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl")
    model = AutoVLA(cfg, inference=True, device="cuda:0")
    sd = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    model.load_state_dict({k[len("autovla."):]: v for k, v in sd.items()
                           if k.startswith("autovla.")}, strict=False)
    model.to("cuda:0").eval(); del sd
    llm, A0 = model.vlm, model.action_start_id
    gen_conf = cfg["inference"]["sample"]
    mlp = llm.model.layers[args.layer].mlp
    cbk = pickle.load(open(cfg["model"]["codebook_cache_path"], "rb"))["token_all"]["veh"]

    EDIT = {"vec": None}
    CAP = {}

    def hook(_m, _a, out):
        o = out[0] if isinstance(out, tuple) else out
        CAP["v"] = o[0, -1, :].detach().float().clone()
        if EDIT["vec"] is None:
            return out
        o = o.clone()
        o[0, -1, :] = o[0, -1, :] + EDIT["vec"].to(o.dtype).to(o.device)
        return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o

    h = mlp.register_forward_hook(hook)

    agent = AutoVLAAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=5, interval_length=0.5),
        sensor_data_path=Path("dataset/nuplan/sensor_blobs/test"),
        codebook_cache_path=cfg["model"]["codebook_cache_path"], skip_model_load=True)

    recs = [json.loads(l) for l in open(args.records) if l.strip()]
    recs = [r for r in recs if r.get("action_positions") and r.get("gt_action_idx")]
    ego = {}
    for r in recs:
        s = json.load(open(os.path.join(args.scenes, f"{r['token']}.json")))
        ego[r["token"]] = float(np.linalg.norm(s["velocity"][:2]))
    for r in recs:
        r["_prim"] = token_primitive(cbk, r["gt_action_idx"][0], ego[r["token"]])
        r["_wrong0"] = r["pred_action_idx"][0] != r["gt_action_idx"][0]

    logs = sorted({r["log_name"] for r in recs}); rng.shuffle(logs)
    fit_logs = set(logs[: len(logs) // 2])
    MODES = ["MAINTAIN", "ACCELERATE", "STOP"]

    def prep(r):
        s = json.load(open(os.path.join(args.scenes, f"{r['token']}.json")))
        feats = {}
        for b in agent.get_feature_builders():
            feats.update(b.compute_features(s))
        mi = {k: v.to("cuda:0") for k, v in model.get_prompt(feats).items()
              if isinstance(v, torch.Tensor)}
        npr = mi["input_ids"].shape[1]
        EDIT["vec"] = None
        with torch.no_grad():
            warm = llm.generate(**mi, max_length=gen_conf["max_length"], do_sample=True,
                                temperature=gen_conf["temperature"],
                                top_k=gen_conf["top_k"], top_p=gen_conf["top_p"])
        ap_ = torch.nonzero(warm[0][npr:] >= A0).flatten().tolist()
        return (mi, warm, npr, ap_) if ap_ else None

    def mlp_out(r):
        p = prep(r)
        if p is None:
            return None
        mi, warm, npr, ap_ = p
        ids = warm[0][: npr + ap_[0]].unsqueeze(0)
        CAP.clear(); EDIT["vec"] = None
        with torch.no_grad():
            llm(input_ids=ids, attention_mask=torch.ones_like(ids),
                **{k: v for k, v in mi.items() if k not in ("input_ids", "attention_mask")})
        return CAP["v"].cpu().numpy()

    # ---- fit per-mode directions on the fit logs ------------------------------------
    print("[fit] collecting per-mode MLP means ...", flush=True)
    acc = defaultdict(lambda: {"c": [], "f": []})
    for tag, want in (("c", False), ("f", True)):
        for m in MODES:
            pool = [r for r in recs if r["log_name"] in fit_logs and r["_prim"] == m
                    and r["_wrong0"] == want]
            rng.shuffle(pool)
            n = args.n_fit_per_mode if tag == "c" else max(12, args.n_fit_per_mode // 3)
            got = 0
            for r in pool:
                if got >= n:
                    break
                v = mlp_out(r)
                if v is not None:
                    acc[m][tag].append(v); got += 1
            print(f"  {m:11s} {'correct' if tag=='c' else 'failure'}: {got}", flush=True)

    dirs = {}
    for m in MODES:
        if len(acc[m]["c"]) >= 8 and len(acc[m]["f"]) >= 8:
            d = np.stack(acc[m]["c"]).mean(0) - np.stack(acc[m]["f"]).mean(0)
            dirs[m] = torch.tensor(d, device="cuda:0")
    allc = np.concatenate([acc[m]["c"] for m in MODES if acc[m]["c"]])
    allf = np.concatenate([acc[m]["f"] for m in MODES if acc[m]["f"]])
    d_glob = torch.tensor(allc.mean(0) - allf.mean(0), device="cuda:0")
    SCALE = float(d_glob.norm())
    print(f"[fit] modes with a direction: {list(dirs)}  ||d_global||={SCALE:.2f}")
    cosm = {m: float(torch.dot(dirs[m] / dirs[m].norm(), d_glob / d_glob.norm())) for m in dirs}
    pair = {f"{a}|{b}": float(torch.dot(dirs[a] / dirs[a].norm(), dirs[b] / dirs[b].norm()))
            for a in dirs for b in dirs if a < b}
    print(f"[fit] cos(mode, global): { {k: round(v,3) for k,v in cosm.items()} }")
    print(f"[fit] cos(mode_i, mode_j): { {k: round(v,3) for k,v in pair.items()} }")

    alphas = [float(a) for a in args.alphas.split(",")]

    def run_arm(mi, warm, npr, ap_, vec, gt_seq):
        ids = warm[0][: npr + ap_[0]].unsqueeze(0)
        kw = {k: v for k, v in mi.items() if k not in ("input_ids", "attention_mask")}
        EDIT["vec"] = vec
        with torch.no_grad():
            out = llm(input_ids=ids, attention_mask=torch.ones_like(ids), **kw)
        EDIT["vec"] = None
        lg = out.logits[0, -1, A0:].float()
        tok = int(lg.argmax())
        pref = torch.cat([ids[0], torch.tensor([tok + A0], device="cuda:0",
                                               dtype=ids.dtype)]).unsqueeze(0)
        with torch.no_grad():
            cont = llm.generate(input_ids=pref, attention_mask=torch.ones_like(pref), **kw,
                                max_length=gen_conf["max_length"], do_sample=True,
                                temperature=gen_conf["temperature"],
                                top_k=gen_conf["top_k"], top_p=gen_conf["top_p"])
        nw = cont[0][pref.shape[1]:]
        acts = ([tok] + [int(x) - A0 for x in nw[nw >= A0].tolist()])[: len(gt_seq)]
        tr = model.action_tokenizer.decode_token_ids_to_trajectory(
            torch.tensor([a + A0 for a in acts]))
        tr = np.asarray(tr[0, 1:], np.float32) if len(tr) else np.zeros((0, 3))
        gtr = np.asarray(model.action_tokenizer.decode_token_ids_to_trajectory(
            torch.tensor([a + A0 for a in gt_seq]))[0, 1:], np.float32)
        n = min(len(acts), len(gt_seq))
        return {"tok": tok, "is_gt": tok == gt_seq[0],
                "margin": float(lg[gt_seq[0]] - lg[tok]) if tok != gt_seq[0] else 0.0,
                "downstream": float(np.mean([acts[j] != gt_seq[j] for j in range(1, n)])) if n > 1 else None,
                "ade": ade(tr, gtr), "fde": fde(tr, gtr)}

    test = [r for r in recs if r["log_name"] not in fit_logs and r["_prim"] in dirs]
    tw = [r for r in test if r["_wrong0"]][: args.n_test]
    tc = [r for r in test if not r["_wrong0"]][: args.n_ctrl]
    print(f"[run] {len(tw)} failures + {len(tc)} controls", flush=True)

    rows = []
    t0 = time.time()
    for i, (kind, r) in enumerate([("failure", x) for x in tw] + [("control", x) for x in tc], 1):
        try:
            p = prep(r)
            if p is None:
                continue
            mi, warm, npr, ap_ = p
            gt = [int(x) for x in r["gt_action_idx"]]
            m = r["_prim"]
            other = [k for k in dirs if k != m]
            wrong_m = other[int(rng.integers(len(other)))] if other else m
            arms = {"baseline": run_arm(mi, warm, npr, ap_, None, gt)}
            for a in alphas:
                arms[f"global_a{a}"] = run_arm(mi, warm, npr, ap_,
                                               d_glob / d_glob.norm() * (a * SCALE), gt)
                arms[f"mode_a{a}"] = run_arm(mi, warm, npr, ap_,
                                             dirs[m] / dirs[m].norm() * (a * SCALE), gt)
                arms[f"wrongmode_a{a}"] = run_arm(mi, warm, npr, ap_,
                                                  dirs[wrong_m] / dirs[wrong_m].norm() * (a * SCALE), gt)
                rv = torch.randn_like(d_glob); rv /= rv.norm()
                arms[f"random_a{a}"] = run_arm(mi, warm, npr, ap_, rv * (a * SCALE), gt)
            rows.append({"token": r["token"], "log": r["log_name"], "kind": kind,
                         "mode": m, "wrong_mode": wrong_m, "arms": arms})
        except Exception as exc:
            if i <= 3:
                import traceback; print(f"[fail] {exc}"); traceback.print_exc()
        if i % 20 == 0:
            el = time.time() - t0
            print(f"[prog] {i} ok={len(rows)} {el/i:.1f}s/s eta "
                  f"{(len(tw)+len(tc)-i)*el/i/60:.1f}min", flush=True)
    h.remove()
    with open(os.path.join(args.output, "mode_steering_raw.json"), "w") as f:
        json.dump({"alphas": alphas, "modes": list(dirs), "scale": SCALE,
                   "cos_mode_global": cosm, "cos_mode_pairs": pair, "rows": rows}, f, default=str)
    print(f"\n[done] {len(rows)} in {(time.time()-t0)/60:.1f}min -> {args.output}")


if __name__ == "__main__":
    main()
