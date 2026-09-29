#!/usr/bin/env python
"""
First-error teacher-forcing counterfactual for AutoVLA action tokens.

Question: does the FIRST wrong action token causally cause the later ones, or do hard
scenes simply keep being wrong (the confound behind the observed 4% -> 73% jump)?

Design — fully paired, one token apart
-------------------------------------
For each sample we locate the first generated action token that differs from the GT action
token (step t). All three conditions then share the *identical* prefix

    prompt  +  generated tokens up to (not including) the step-t action token

and differ only in the single token written at step t:

    original : the model's own wrong token   (re-generated, not the recorded run, so all
                                              three arms sit on exactly the same footing)
    gt       : the ground-truth token
    random   : a uniformly random valid action token != GT and != original, several seeds

After that one token, generation is handed back to the model — free autoregressive decoding
to the end, same decoding config as the baseline run. Nothing downstream is teacher-forced.

Continuation is done by calling `generate` on the extended prefix. That recomputes the
prefix KV rather than splicing a cache, which is slower but semantically exactly what a
normal decode would have produced from that prefix; splicing risks silent position/mrope
bugs with Qwen2.5-VL's vision tokens.
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(POC_DIR, "outputs/gpu1/records.jsonl"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/counterfactual"))
    ap.add_argument("--config", default=os.path.join(
        AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--random-seeds", type=int, default=3)
    ap.add_argument("--min-downstream", type=int, default=2,
                    help="require at least this many action steps after the first error")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def coarse(traj_xy, get_action_instruction):
    """Coarse (longitudinal, lateral) action of a decoded trajectory."""
    p = np.asarray(traj_xy, dtype=np.float64)[:, :2]
    if len(p) < 2:
        return (None, None)
    v = np.diff(p, axis=0) / 0.5
    v = np.concatenate([v, v[-1:]], axis=0)
    return parse_action_instruction(get_action_instruction(p, v))


def main() -> None:
    args = parse_args()
    args.output = os.path.abspath(args.output)
    args.records = os.path.abspath(args.records)
    args.scenes = os.path.abspath(args.scenes)
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
    model.to("cuda:0").eval()
    del sd
    llm, A0 = model.vlm, model.action_start_id
    n_actions = model.action_tokenizer.code_book.shape[0]
    gen_conf = cfg["inference"]["sample"]
    print(f"[build] action vocab {n_actions}, start id {A0}", flush=True)

    agent = AutoVLAAgent(
        trajectory_sampling=TrajectorySampling(
            time_horizon=cfg["model"]["trajectory"]["time_horizon"],
            interval_length=cfg["model"]["trajectory"]["interval_length"]),
        sensor_data_path=Path("dataset/nuplan/sensor_blobs/test"),
        codebook_cache_path=cfg["model"]["codebook_cache_path"],
        skip_model_load=True)

    # ---- select samples that actually have a first error with room downstream ----------
    cand = []
    for line in open(args.records):
        if not line.strip():
            continue
        r = json.loads(line)
        pred, gt = r.get("pred_action_idx") or [], r.get("gt_action_idx") or []
        n = min(len(pred), len(gt))
        if n < 3 or not r.get("action_positions"):
            continue
        diff = [i for i in range(n) if pred[i] != gt[i]]
        if not diff:
            continue
        t = diff[0]
        if n - 1 - t < args.min_downstream:
            continue
        cand.append({"token": r["token"], "t": t, "n": n,
                     "pred": pred[:n], "gt": gt[:n],
                     "act_pos": r["action_positions"], "log": r.get("log_name")})
    print(f"[select] {len(cand)} samples with a first error and "
          f">={args.min_downstream} downstream steps", flush=True)

    idx = rng.permutation(len(cand))[: args.max_samples]
    chosen = [cand[i] for i in sorted(idx)]
    print(f"[select] running {len(chosen)} "
          f"({len({c['log'] for c in chosen})} logs)", flush=True)

    def continue_from(prefix_ids, mi):
        """Free autoregressive generation from an explicit prefix."""
        kw = {k: v for k, v in mi.items() if k not in ("input_ids", "attention_mask")}
        ids = prefix_ids.unsqueeze(0)
        am = torch.ones_like(ids)
        with torch.no_grad():
            out = llm.generate(input_ids=ids, attention_mask=am, **kw,
                               max_length=gen_conf["max_length"], do_sample=True,
                               temperature=gen_conf["temperature"], top_k=gen_conf["top_k"],
                               top_p=gen_conf["top_p"])
        return out[0][ids.shape[1]:]

    def actions_of(new_ids):
        a = new_ids[new_ids >= A0]
        return [int(x) - A0 for x in a.tolist()]

    def traj_of(action_idx):
        if not action_idx:
            return np.zeros((0, 2))
        ids = torch.tensor([a + A0 for a in action_idx])
        tr = model.action_tokenizer.decode_token_ids_to_trajectory(ids)
        return np.asarray(tr[0, 1:], dtype=np.float32) if len(tr) else np.zeros((0, 2))

    results = []
    examples = []
    t0 = time.time()
    for i, c in enumerate(chosen, 1):
        try:
            scene = json.load(open(os.path.join(args.scenes, f"{c['token']}.json")))
            feats = {}
            for b in agent.get_feature_builders():
                feats.update(b.compute_features(scene))
            inputs = model.get_prompt(feats)
            mi = {k: v.to("cuda:0") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            prompt_ids = mi["input_ids"][0]
            n_prompt = prompt_ids.shape[0]

            t = c["t"]
            gt_seq, pred_seq = c["gt"], c["pred"]
            # absolute position of the step-t action token inside the FULL sequence
            cut = n_prompt + c["act_pos"][t]

            # Reconstruct the shared prefix by regenerating the prompt and splicing the
            # recorded generated tokens before step t. The recorded ids are the model's own
            # previous output, so the prefix is on-policy.
            gen_recorded = torch.tensor(
                [A0 + a for a in pred_seq[:t]], device="cuda:0", dtype=prompt_ids.dtype)
            # tokens between prompt and first action token (the CoT/answer scaffolding)
            head_len = c["act_pos"][0]
            # we do not have the scaffolding ids stored; regenerate them deterministically
            # by decoding the prompt once and taking the model's own prefix.
            with torch.no_grad():
                warm = llm.generate(**mi, max_length=gen_conf["max_length"], do_sample=True,
                                    temperature=gen_conf["temperature"],
                                    top_k=gen_conf["top_k"], top_p=gen_conf["top_p"])
            warm_new = warm[0][n_prompt:]
            wpos = torch.nonzero(warm_new >= A0).flatten().tolist()
            if len(wpos) <= t:
                continue                      # regeneration produced too few action tokens
            scaffold = warm_new[: wpos[0]]    # CoT / "<answer> The final output action is:"
            base_prefix = torch.cat([prompt_ids, scaffold,
                                     warm_new[wpos[0]: wpos[0] + t]])

            orig_tok = int(warm_new[wpos[t]].item()) - A0
            gt_tok = int(gt_seq[t])
            arms = {"original": [orig_tok], "gt": [gt_tok]}

            rands = []
            for s in range(args.random_seeds):
                r_ = int(rng.integers(n_actions))
                while r_ == gt_tok or r_ == orig_tok:
                    r_ = int(rng.integers(n_actions))
                rands.append(r_)
            arms["random"] = rands

            # Tighter control: the model's OWN next-best action token at this position.
            # A uniformly random token out of 2048 lands far off any plausible trajectory
            # (ADE ~3 m), so beating it only shows "GT != nonsense". The runner-up token is
            # an alternative the model itself considered, which is the control the causal
            # claim actually needs.
            with torch.no_grad():
                logits = llm(input_ids=base_prefix.unsqueeze(0),
                             attention_mask=torch.ones_like(base_prefix).unsqueeze(0),
                             **{k: v for k, v in mi.items()
                                if k not in ("input_ids", "attention_mask")}).logits[0, -1]
            act_logits = logits[A0:]
            order = torch.argsort(act_logits, descending=True).tolist()
            plausible = next((int(a) for a in order if a != gt_tok and a != orig_tok), None)
            if plausible is not None:
                arms["plausible"] = [plausible]

            rec = {"token": c["token"], "log": c["log"], "t": t, "n": c["n"],
                   "gt_seq": gt_seq, "orig_token": orig_tok, "gt_token": gt_tok,
                   "arms": {}}
            for arm, toks in arms.items():
                per = []
                for tok in toks:
                    pref = torch.cat([base_prefix,
                                      torch.tensor([tok + A0], device="cuda:0",
                                                   dtype=prompt_ids.dtype)])
                    cont = continue_from(pref, mi)
                    # full action sequence = forced prefix actions + forced token + continuation
                    acts = pred_seq[:t] + [tok] + actions_of(cont)
                    acts = acts[: c["n"]]
                    down = [j for j in range(t + 1, min(len(acts), c["n"]))]
                    err_down = [int(acts[j] != gt_seq[j]) for j in down]
                    nxt = err_down[0] if err_down else None
                    tr = traj_of(acts)
                    gtr = traj_of(gt_seq)
                    lon, lat = coarse(tr, get_action_instruction)
                    glon, glat = coarse(gtr, get_action_instruction)
                    per.append({
                        "forced_token": tok,
                        "actions": acts,
                        "next_token_error": nxt,
                        "downstream_error_rate": float(np.mean(err_down)) if err_down else None,
                        "remaining_accuracy": float(1 - np.mean(err_down)) if err_down else None,
                        "sequence_recovered": bool(err_down and sum(err_down) == 0),
                        "ade": ade(tr, gtr), "fde": fde(tr, gtr),
                        "lon": lon, "lat": lat, "lon_gt": glon, "lat_gt": glat,
                        "coarse_match": bool(lon == glon and lat == glat),
                    })
                rec["arms"][arm] = per
            results.append(rec)
            if len(examples) < 10:
                examples.append(rec)
        except Exception as exc:
            if len(results) < 5:
                import traceback
                print(f"[fail] {c['token']}: {exc}", flush=True)
                traceback.print_exc()
        if i % 25 == 0:
            el = time.time() - t0
            print(f"[prog] {i}/{len(chosen)} ok={len(results)} "
                  f"{el/i:.2f}s/sample eta {(len(chosen)-i)*el/i/60:.1f}min", flush=True)

    with open(os.path.join(args.output, "counterfactual_raw.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    with open(os.path.join(args.output, "examples.json"), "w") as f:
        json.dump(examples, f, indent=1, default=str)
    print(f"\n[done] {len(results)} samples in {(time.time()-t0)/60:.1f}min -> {args.output}")


if __name__ == "__main__":
    main()
