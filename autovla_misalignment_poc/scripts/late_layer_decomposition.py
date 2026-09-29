#!/usr/bin/env python
"""
Which late-layer component writes the wrong action into the residual stream?

A Qwen2.5-VL decoder layer updates the residual stream twice
(modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer.forward):

    h_in                                   <- layer input
    h_attn = h_in   + self_attn(ln1(h_in))     <- after attention
    h_out  = h_attn + mlp(ln2(h_attn))         <- after MLP == layer output

We read each of those three states at the action-token query position through the model's
final norm and lm_head, and track

    margin = logit(GT action token) - logit(competitor action token)

The competitor is the token the model actually generated when that is wrong, and the
runner-up token when the generation was correct — in both cases "the strongest rival to the
right answer", so the two groups are measuring the same thing.

Per layer this yields two attributable deltas:

    d_attn = margin(h_attn) - margin(h_in)
    d_mlp  = margin(h_out)  - margin(h_attn)

NOTE on the readout: the final RMSNorm is applied before lm_head here. The earlier logit
lens in this project applied lm_head directly to raw hidden states (the naive variant);
for an attribution the normalised readout is the right one, since that is what the model
itself does at the end of the stack.
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(POC_DIR, "outputs/gpu1/records.jsonl"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/decomposition"))
    ap.add_argument("--config", default=os.path.join(
        AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--n-wrong", type=int, default=250)
    ap.add_argument("--n-correct", type=int, default=250)
    ap.add_argument("--last-layers", type=int, default=8,
                    help="decompose this many final layers (report focuses on the last 4)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
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
    sd = torch.load(args.checkpoint, map_location="cpu", mmap=True)["state_dict"]
    model.load_state_dict({k[len("autovla."):]: v for k, v in sd.items()
                           if k.startswith("autovla.")}, strict=False)
    model.to("cuda:0").eval()
    del sd
    llm, A0 = model.vlm, model.action_start_id
    gen_conf = cfg["inference"]["sample"]

    body = llm.model
    layers = body.layers
    final_norm = body.norm
    # fp32 copy of the action rows: the bf16 matmul quantises logits to ~1/8, which is
    # coarser than the per-component margin shifts we are trying to attribute.
    W_act = llm.lm_head.weight[A0:].detach().float()   # (2048, hidden)
    nL = len(layers)
    probe_layers = list(range(nL - args.last_layers, nL))
    print(f"[build] {nL} layers, decomposing {probe_layers}", flush=True)

    # ---- hooks: capture layer input, attn output and mlp output at the last position ----
    cap = {}

    def mk_layer_pre(li):
        def f(_m, args_, kwargs_):
            hs = kwargs_.get("hidden_states")
            if hs is None and args_:
                hs = args_[0]
            if hs is not None:
                cap[("in", li)] = hs[0, -1, :].detach().clone()
            return None
        return f

    def mk_attn(li):
        def f(_m, _a, out):
            o = out[0] if isinstance(out, tuple) else out
            cap[("attn", li)] = o[0, -1, :].detach().clone()
        return f

    def mk_mlp(li):
        def f(_m, _a, out):
            o = out[0] if isinstance(out, tuple) else out
            cap[("mlp", li)] = o[0, -1, :].detach().clone()
        return f

    handles = []
    for li in probe_layers:
        handles.append(layers[li].register_forward_pre_hook(mk_layer_pre(li), with_kwargs=True))
        handles.append(layers[li].self_attn.register_forward_hook(mk_attn(li)))
        handles.append(layers[li].mlp.register_forward_hook(mk_mlp(li)))

    @torch.no_grad()
    def readout(h):
        """final norm -> lm_head, restricted to the action-token rows, computed in fp32."""
        return W_act @ final_norm(h.unsqueeze(0)).squeeze(0).float()

    agent = AutoVLAAgent(
        trajectory_sampling=TrajectorySampling(
            time_horizon=cfg["model"]["trajectory"]["time_horizon"],
            interval_length=cfg["model"]["trajectory"]["interval_length"]),
        sensor_data_path=Path("dataset/nuplan/sensor_blobs/test"),
        codebook_cache_path=cfg["model"]["codebook_cache_path"],
        skip_model_load=True)

    # ---- pick a balanced set on step-0 correctness ----------------------------------
    recs = []
    for line in open(args.records):
        if line.strip():
            r = json.loads(line)
            if r.get("action_positions") and r.get("gt_action_idx") and r.get("pred_action_idx"):
                recs.append(r)
    wrong0 = [r for r in recs if r["pred_action_idx"][0] != r["gt_action_idx"][0]]
    right0 = [r for r in recs if r["pred_action_idx"][0] == r["gt_action_idx"][0]]
    rng.shuffle(wrong0); rng.shuffle(right0)
    chosen = wrong0[: args.n_wrong] + right0[: args.n_correct]
    print(f"[select] step-0 wrong {min(len(wrong0), args.n_wrong)} / "
          f"correct {min(len(right0), args.n_correct)} "
          f"(pool {len(wrong0)}/{len(right0)})", flush=True)

    rows = []
    t0 = time.time()
    for i, r in enumerate(chosen, 1):
        try:
            scene = json.load(open(os.path.join(args.scenes, f"{r['token']}.json")))
            feats = {}
            for b in agent.get_feature_builders():
                feats.update(b.compute_features(scene))
            inputs = model.get_prompt(feats)
            mi = {k: v.to("cuda:0") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
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
            correct = (gen0 == gt0)

            cut = n_prompt + apos[0]                 # query position that emits step 0
            ids = warm[0][:cut].unsqueeze(0)
            cap.clear()
            with torch.no_grad():
                llm(input_ids=ids, attention_mask=torch.ones_like(ids),
                    **{k: v for k, v in mi.items() if k not in ("input_ids", "attention_mask")})

            # competitor: the generated token when wrong, otherwise the runner-up
            last_in = cap[("in", probe_layers[0])]
            final_logits = readout(cap[("in", probe_layers[-1])] +
                                   cap[("attn", probe_layers[-1])] +
                                   cap[("mlp", probe_layers[-1])])
            if correct:
                order = torch.argsort(final_logits, descending=True).tolist()
                comp = next(int(a) for a in order if a != gt0)
            else:
                comp = gen0

            per_layer = []
            for li in probe_layers:
                h_in = cap[("in", li)]
                h_attn = h_in + cap[("attn", li)]
                h_out = h_attn + cap[("mlp", li)]
                m = []
                for h in (h_in, h_attn, h_out):
                    lg = readout(h)
                    m.append(float(lg[gt0] - lg[comp]))
                rk = int((readout(h_out) > readout(h_out)[gt0]).sum())
                per_layer.append({"layer": li,
                                  "margin_in": m[0], "margin_attn": m[1], "margin_mlp": m[2],
                                  "d_attn": m[1] - m[0], "d_mlp": m[2] - m[1],
                                  "gt_rank_out": rk})
            rows.append({"token": r["token"], "log": r.get("log_name"),
                         "correct": bool(correct), "gt": gt0, "competitor": comp,
                         "layers": per_layer})
        except Exception as exc:
            if i <= 5:
                import traceback
                print(f"[fail] {r['token']}: {exc}", flush=True)
                traceback.print_exc()
        if i % 50 == 0:
            el = time.time() - t0
            print(f"[prog] {i}/{len(chosen)} ok={len(rows)} {el/i:.2f}s/sample "
                  f"eta {(len(chosen)-i)*el/i/60:.1f}min", flush=True)

    for h in handles:
        h.remove()
    with open(os.path.join(args.output, "decomposition_raw.json"), "w") as f:
        json.dump({"probe_layers": probe_layers, "n_layers": nL, "rows": rows}, f)
    print(f"\n[done] {len(rows)} samples in {(time.time()-t0)/60:.1f}min -> {args.output}")


if __name__ == "__main__":
    main()
