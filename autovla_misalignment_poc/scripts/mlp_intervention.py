#!/usr/bin/env python
"""
Causal test of the L35 MLP: does editing it fix the first action token?

Addendum 2 showed the final-layer MLP *writes* the wrong action (attribution). This turns
that into an intervention. At the step-0 action query position we replace or steer the L35
MLP output, take the resulting token, and then hand generation back to the model with the
hook removed — so exactly one component at one position is edited, and everything after is
the model's own free decoding (the same shape as the first-error counterfactual).

Arms
    baseline    no edit (re-generated, so every arm shares the same prefix)
    mean_swap   L35 MLP output <- mean MLP output of CORRECT samples
    null_swap   L35 MLP output <- mean MLP output of FAILURE samples   (negative control)
    steer       + alpha * d,  d = mean_correct - mean_failure          (group-level, no GT)
    random      + alpha * r,  r random with the SAME norm as d         (control)
    oracle      + alpha * (w_GT - w_rival) direction                   (uses GT: ceiling only)

The group means and the steering direction are fit on a DISJOINT set of logs from the ones
tested, so the intervention never sees the samples it is evaluated on.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

AUTOVLA_DIR = "/root/VLA/autovla"
POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AUTOVLA_DIR)
sys.path.insert(0, os.path.join(AUTOVLA_DIR, "navsim"))
sys.path.insert(0, POC_DIR)

from src.labeling.labels import ade, fde, parse_action_instruction  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(POC_DIR, "outputs/gpu1/records.jsonl"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/mlp_intervention"))
    ap.add_argument("--config", default=os.path.join(
        AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--layer", type=int, default=35)
    ap.add_argument("--n-fit", type=int, default=120, help="samples per group for the means")
    ap.add_argument("--n-test", type=int, default=120, help="failure samples to intervene on")
    ap.add_argument("--n-ctrl", type=int, default=80, help="correct samples, for collateral damage")
    ap.add_argument("--alphas", default="1.0,2.0")
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
    from dataset_utils.preprocessing.nuplan_dataset import get_action_instruction

    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["pretrained_model_path"] = os.path.join(AUTOVLA_DIR, "Qwen2.5-VL-3B-Instruct")
    cfg["model"]["codebook_cache_path"] = os.path.join(AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl")
    model = AutoVLA(cfg, inference=True, device="cuda:0")
    sd = torch.load(args.checkpoint, map_location="cpu", mmap=True)["state_dict"]
    model.load_state_dict({k[len("autovla."):]: v for k, v in sd.items()
                           if k.startswith("autovla.")}, strict=False)
    model.to("cuda:0").eval(); del sd
    llm, A0 = model.vlm, model.action_start_id
    gen_conf = cfg["inference"]["sample"]
    mlp_mod = llm.model.layers[args.layer].mlp
    W = llm.lm_head.weight[A0:].detach().float()

    agent = AutoVLAAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=5, interval_length=0.5),
        sensor_data_path=Path("dataset/nuplan/sensor_blobs/test"),
        codebook_cache_path=cfg["model"]["codebook_cache_path"], skip_model_load=True)

    # ---- one shared edit slot the hook reads; None means "pass through" --------------
    EDIT = {"mode": None, "vec": None}

    def hook(_m, _a, out):
        o = out[0] if isinstance(out, tuple) else out
        if EDIT["mode"] is None:
            return out
        o = o.clone()
        v = EDIT["vec"].to(o.dtype).to(o.device)
        o[0, -1, :] = v if EDIT["mode"] == "replace" else o[0, -1, :] + v
        return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o

    handle = mlp_mod.register_forward_hook(hook)
    capture = {}

    def cap_hook(_m, _a, out):
        o = out[0] if isinstance(out, tuple) else out
        capture["v"] = o[0, -1, :].detach().float().clone()

    cap_handle = mlp_mod.register_forward_hook(cap_hook)

    # ---- data, split by log so the fit set and the test set share no scene -----------
    recs = [json.loads(l) for l in open(args.records) if l.strip()]
    recs = [r for r in recs if r.get("action_positions") and r.get("gt_action_idx")]
    logs = sorted({r.get("log_name") for r in recs})
    rng.shuffle(logs)
    fit_logs = set(logs[: len(logs) // 2])
    is_wrong = lambda r: r["pred_action_idx"][0] != r["gt_action_idx"][0]
    fit_w = [r for r in recs if r["log_name"] in fit_logs and is_wrong(r)]
    fit_c = [r for r in recs if r["log_name"] in fit_logs and not is_wrong(r)]
    test_w = [r for r in recs if r["log_name"] not in fit_logs and is_wrong(r)]
    test_c = [r for r in recs if r["log_name"] not in fit_logs and not is_wrong(r)]
    for L in (fit_w, fit_c, test_w, test_c):
        rng.shuffle(L)
    print(f"[split] fit logs {len(fit_logs)}/{len(logs)}  "
          f"fit: wrong {len(fit_w)} correct {len(fit_c)}  "
          f"test: wrong {len(test_w)} correct {len(test_c)}", flush=True)

    def prep(r):
        scene = json.load(open(os.path.join(args.scenes, f"{r['token']}.json")))
        feats = {}
        for b in agent.get_feature_builders():
            feats.update(b.compute_features(scene))
        mi = {k: v.to("cuda:0") for k, v in model.get_prompt(feats).items()
              if isinstance(v, torch.Tensor)}
        n_prompt = mi["input_ids"].shape[1]
        EDIT["mode"] = None
        with torch.no_grad():
            warm = llm.generate(**mi, max_length=gen_conf["max_length"], do_sample=True,
                                temperature=gen_conf["temperature"],
                                top_k=gen_conf["top_k"], top_p=gen_conf["top_p"])
        new_ids = warm[0][n_prompt:]
        apos = torch.nonzero(new_ids >= A0).flatten().tolist()
        if not apos:
            return None
        return mi, warm, n_prompt, apos, new_ids

    # ---- fit the group means on the fit logs ----------------------------------------
    def collect(pool, n):
        acc = []
        for r in pool[: n * 2]:
            if len(acc) >= n:
                break
            try:
                p = prep(r)
                if p is None:
                    continue
                mi, warm, n_prompt, apos, _ = p
                ids = warm[0][: n_prompt + apos[0]].unsqueeze(0)
                capture.clear(); EDIT["mode"] = None
                with torch.no_grad():
                    llm(input_ids=ids, attention_mask=torch.ones_like(ids),
                        **{k: v for k, v in mi.items() if k not in ("input_ids", "attention_mask")})
                acc.append(capture["v"].cpu().numpy())
            except Exception:
                continue
        return np.stack(acc) if acc else None

    print("[fit] collecting group means ...", flush=True)
    Vc = collect(fit_c, args.n_fit)
    Vw = collect(fit_w, args.n_fit)
    mean_c = torch.tensor(Vc.mean(0), device="cuda:0")
    mean_w = torch.tensor(Vw.mean(0), device="cuda:0")
    d = mean_c - mean_w
    d_unit = d / d.norm()
    print(f"[fit] n_correct={len(Vc)} n_failure={len(Vw)}  ||d||={float(d.norm()):.3f}  "
          f"||mean_c||={float(mean_c.norm()):.3f}", flush=True)

    alphas = [float(a) for a in args.alphas.split(",")]
    ADD_SCALE = float(d.norm())        # steer/random/oracle all move by alpha * ||d||

    def coarse(tr):
        p = np.asarray(tr, float)[:, :2]
        if len(p) < 2:
            return (None, None)
        v = np.diff(p, axis=0) / 0.5
        v = np.concatenate([v, v[-1:]], axis=0)
        return parse_action_instruction(get_action_instruction(p, v))

    def run_arm(mi, warm, n_prompt, apos, mode, vec, gt_seq):
        """Edit only the step-0 position, then free generation with the hook off."""
        ids = warm[0][: n_prompt + apos[0]].unsqueeze(0)
        kw = {k: v for k, v in mi.items() if k not in ("input_ids", "attention_mask")}
        EDIT["mode"], EDIT["vec"] = mode, vec
        with torch.no_grad():
            out = llm(input_ids=ids, attention_mask=torch.ones_like(ids), **kw)
        EDIT["mode"] = None
        # temperature is 0.01, i.e. effectively greedy; argmax keeps the arm deterministic
        tok = int(out.logits[0, -1, A0:].float().argmax())
        pref = torch.cat([ids[0], torch.tensor([tok + A0], device="cuda:0",
                                               dtype=ids.dtype)]).unsqueeze(0)
        with torch.no_grad():
            cont = llm.generate(input_ids=pref, attention_mask=torch.ones_like(pref), **kw,
                                max_length=gen_conf["max_length"], do_sample=True,
                                temperature=gen_conf["temperature"],
                                top_k=gen_conf["top_k"], top_p=gen_conf["top_p"])
        new = cont[0][pref.shape[1]:]
        acts = [tok] + [int(x) - A0 for x in new[new >= A0].tolist()]
        acts = acts[: len(gt_seq)]
        tr = model.action_tokenizer.decode_token_ids_to_trajectory(
            torch.tensor([a + A0 for a in acts]))
        tr = np.asarray(tr[0, 1:], np.float32) if len(tr) else np.zeros((0, 3))
        gtr = model.action_tokenizer.decode_token_ids_to_trajectory(
            torch.tensor([a + A0 for a in gt_seq]))
        gtr = np.asarray(gtr[0, 1:], np.float32)
        lon, lat = coarse(tr); glon, glat = coarse(gtr)
        n = min(len(acts), len(gt_seq))
        return {"token0": tok, "token0_is_gt": tok == gt_seq[0],
                "actions": acts,
                "downstream_error_rate": float(np.mean([acts[j] != gt_seq[j]
                                                        for j in range(1, n)])) if n > 1 else None,
                "ade": ade(tr, gtr), "fde": fde(tr, gtr),
                "coarse_match": bool(lon == glon and lat == glat)}

    targets = ([("failure", r) for r in test_w[: args.n_test]] +
               [("control", r) for r in test_c[: args.n_ctrl]])
    print(f"[run] {len(targets)} test samples", flush=True)

    results = []
    t0 = time.time()
    for i, (kind, r) in enumerate(targets, 1):
        try:
            p = prep(r)
            if p is None:
                continue
            mi, warm, n_prompt, apos, _ = p
            gt_seq = [int(x) for x in r["gt_action_idx"]]
            # rival for the oracle arm = the token the unedited run produced
            base_tok = int(warm[0][n_prompt + apos[0]].item()) - A0
            arms = {}
            arms["baseline"] = run_arm(mi, warm, n_prompt, apos, None, None, gt_seq)
            arms["mean_swap"] = run_arm(mi, warm, n_prompt, apos, "replace", mean_c, gt_seq)
            arms["null_swap"] = run_arm(mi, warm, n_prompt, apos, "replace", mean_w, gt_seq)
            for a in alphas:
                arms[f"steer_a{a}"] = run_arm(mi, warm, n_prompt, apos, "add",
                                              d_unit * (a * ADD_SCALE), gt_seq)
                rv = torch.randn_like(d_unit)
                rv = rv / rv.norm()
                arms[f"random_a{a}"] = run_arm(mi, warm, n_prompt, apos, "add",
                                               rv * (a * ADD_SCALE), gt_seq)
                od = (W[gt_seq[0]] - W[base_tok])
                od = od / od.norm()
                arms[f"oracle_a{a}"] = run_arm(mi, warm, n_prompt, apos, "add",
                                               od * (a * ADD_SCALE), gt_seq)
            results.append({"token": r["token"], "log": r["log_name"], "kind": kind,
                            "gt0": gt_seq[0], "base_token": base_tok, "arms": arms})
        except Exception as exc:
            if i <= 3:
                import traceback; print(f"[fail] {r['token']}: {exc}", flush=True); traceback.print_exc()
        if i % 20 == 0:
            el = time.time() - t0
            print(f"[prog] {i}/{len(targets)} ok={len(results)} {el/i:.2f}s/s "
                  f"eta {(len(targets)-i)*el/i/60:.1f}min", flush=True)

    handle.remove(); cap_handle.remove()
    with open(os.path.join(args.output, "mlp_intervention_raw.json"), "w") as f:
        json.dump({"layer": args.layer, "alphas": alphas,
                   "d_norm": float(d.norm()), "rows": results}, f, default=str)
    print(f"\n[done] {len(results)} in {(time.time()-t0)/60:.1f}min -> {args.output}")


if __name__ == "__main__":
    main()
