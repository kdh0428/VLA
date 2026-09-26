#!/usr/bin/env python
"""
Follow-up: the late MLP is the culprit — what is it writing?

Captures the raw MLP output vector at the action query position for the last few layers and
asks three things:

  1. Is the failure-group MLP output pointing somewhere else, or just scaled differently?
     (norm and cosine between the two group means)
  2. How much of its action-logit effect is explained by one direction, the readout
     direction  w_gt - w_comp  that the margin is measured along?
  3. Does the MLP output in failures align with the COMPETITOR's readout row — i.e. is it
     actively writing the wrong action rather than merely failing to write the right one?
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(POC_DIR, "outputs/gpu1/records.jsonl"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/decomposition"))
    ap.add_argument("--config", default=os.path.join(
        AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--layers", default="34,35")
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
    layers = llm.model.layers
    W = llm.lm_head.weight[A0:].detach().float()
    probe = [int(x) for x in args.layers.split(",")]

    cap = {}
    handles = [layers[li].mlp.register_forward_hook(
        (lambda li: (lambda _m, _a, o: cap.__setitem__(li, (o[0] if isinstance(o, tuple) else o)[0, -1, :].detach().float().clone())))(li))
        for li in probe]

    agent = AutoVLAAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=5, interval_length=0.5),
        sensor_data_path=Path("dataset/nuplan/sensor_blobs/test"),
        codebook_cache_path=cfg["model"]["codebook_cache_path"], skip_model_load=True)

    recs = [json.loads(l) for l in open(args.records) if l.strip()]
    recs = [r for r in recs if r.get("action_positions") and r.get("gt_action_idx")]
    wrong = [r for r in recs if r["pred_action_idx"][0] != r["gt_action_idx"][0]]
    right = [r for r in recs if r["pred_action_idx"][0] == r["gt_action_idx"][0]]
    rng.shuffle(wrong); rng.shuffle(right)
    chosen = wrong[: args.n] + right[: args.n]
    print(f"[select] {len(chosen)} (wrong {min(len(wrong),args.n)} / correct {min(len(right),args.n)})", flush=True)

    data = {li: {"vec": [], "correct": [], "gt": [], "comp": []} for li in probe}
    t0 = time.time()
    for i, r in enumerate(chosen, 1):
        try:
            scene = json.load(open(os.path.join(args.scenes, f"{r['token']}.json")))
            feats = {}
            for b in agent.get_feature_builders():
                feats.update(b.compute_features(scene))
            mi = {k: v.to("cuda:0") for k, v in model.get_prompt(feats).items()
                  if isinstance(v, torch.Tensor)}
            n_prompt = mi["input_ids"].shape[1]
            with torch.no_grad():
                warm = llm.generate(**mi, max_length=gen_conf["max_length"], do_sample=True,
                                    temperature=gen_conf["temperature"],
                                    top_k=gen_conf["top_k"], top_p=gen_conf["top_p"])
            new_ids = warm[0][n_prompt:]
            apos = torch.nonzero(new_ids >= A0).flatten().tolist()
            if not apos:
                continue
            gen0 = int(new_ids[apos[0]].item()) - A0
            gt0 = int(r["gt_action_idx"][0])
            correct = gen0 == gt0
            ids = warm[0][: n_prompt + apos[0]].unsqueeze(0)
            cap.clear()
            with torch.no_grad():
                out = llm(input_ids=ids, attention_mask=torch.ones_like(ids),
                          **{k: v for k, v in mi.items() if k not in ("input_ids", "attention_mask")})
            fl = out.logits[0, -1, A0:].float()
            comp = gen0 if not correct else int(
                next(a for a in torch.argsort(fl, descending=True).tolist() if a != gt0))
            for li in probe:
                if li in cap:
                    data[li]["vec"].append(cap[li].cpu().numpy())
                    data[li]["correct"].append(correct)
                    data[li]["gt"].append(gt0); data[li]["comp"].append(comp)
        except Exception:
            pass
        if i % 50 == 0:
            print(f"[prog] {i}/{len(chosen)} {(time.time()-t0)/i:.2f}s/s", flush=True)
    for h in handles:
        h.remove()

    Wc = W.cpu().numpy()
    res = {}
    for li in probe:
        V = np.stack(data[li]["vec"]); c = np.array(data[li]["correct"])
        gt = np.array(data[li]["gt"]); cp = np.array(data[li]["comp"])
        vc, vf = V[c], V[~c]
        mc, mf = vc.mean(0), vf.mean(0)
        cos = float(mc @ mf / (np.linalg.norm(mc) * np.linalg.norm(mf)))
        # projection onto the readout direction the margin is measured along
        d = Wc[gt] - Wc[cp]                        # (N, hidden)
        dn = d / np.linalg.norm(d, axis=1, keepdims=True)
        proj = (V * dn).sum(1)
        # alignment with the competitor's own readout row
        cn = Wc[cp] / np.linalg.norm(Wc[cp], axis=1, keepdims=True)
        gn = Wc[gt] / np.linalg.norm(Wc[gt], axis=1, keepdims=True)
        a_comp = (V * cn).sum(1)
        a_gt = (V * gn).sum(1)
        res[li] = {
            "n_correct": int(c.sum()), "n_failure": int((~c).sum()),
            "norm_correct": float(np.linalg.norm(vc, axis=1).mean()),
            "norm_failure": float(np.linalg.norm(vf, axis=1).mean()),
            "cos_group_means": cos,
            "proj_on_gt_minus_comp_correct": float(proj[c].mean()),
            "proj_on_gt_minus_comp_failure": float(proj[~c].mean()),
            "align_gt_correct": float(a_gt[c].mean()), "align_gt_failure": float(a_gt[~c].mean()),
            "align_comp_correct": float(a_comp[c].mean()), "align_comp_failure": float(a_comp[~c].mean()),
        }
        r_ = res[li]
        print(f"\n=== L{li} MLP output ===")
        print(f"  n: correct {r_['n_correct']}  failure {r_['n_failure']}")
        print(f"  ||out||        correct {r_['norm_correct']:.3f}   failure {r_['norm_failure']:.3f}")
        print(f"  cos(mean_c, mean_f)                 {cos:+.4f}")
        print(f"  projection on (w_gt - w_comp)  correct {r_['proj_on_gt_minus_comp_correct']:+.4f}"
              f"   failure {r_['proj_on_gt_minus_comp_failure']:+.4f}")
        print(f"  alignment with w_gt            correct {r_['align_gt_correct']:+.4f}"
              f"   failure {r_['align_gt_failure']:+.4f}")
        print(f"  alignment with w_competitor    correct {r_['align_comp_correct']:+.4f}"
              f"   failure {r_['align_comp_failure']:+.4f}")

    with open(os.path.join(args.output, "mlp_direction.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {args.output}/mlp_direction.json")


if __name__ == "__main__":
    main()
