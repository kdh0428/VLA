#!/usr/bin/env python
"""
One-pass AutoVLA extraction: everything later analyses need, so that no further GPU run
is required.

WHY TWO ARMS
------------
Natural decoding almost never reasons: all 2748 scenes open `<think>` and then fork on a
single token at generated position 6 --- ` straightforward` (30339) in 2747 scenes,
` complex` (6351) in exactly one.  So CoT text simply does not exist to be analysed.

    <think>\\nThis is a| straightforward, and a direct decision can be made.\\n</think>...
    <think>\\nThis is a| complex scenario requiring additional reasoning.\\n### **Scene ...

The 6 tokens before the fork are identical in every scene, so this script forces them,
reads the full-vocabulary distribution at the fork, and then runs BOTH continuations:

    arm N ("natural")  : let the model choose (it picks ` straightforward`)
    arm C ("cot")      : force ` complex`, then free generation

Arm C is a counterfactual, NOT a recovery of latent reasoning: the CoT never existed in
arm N, and the action tokens arm C emits differ from arm N's.  Never pair arm C's CoT with
arm N's action and call it one computation.  The pairing that IS valid is arm C vs arm N as
two complete runs of the same model on the same scene.

WHAT IS STORED WITHOUT LOSS
---------------------------
A decoder layer updates the residual stream twice:

    h_out[l] = h_in[l] + attn_out[l] + mlp_out[l]
    h_in[l+1] = h_out[l]                       h_in[0] = embeddings

so storing (h_in, attn_out, mlp_out) for the 36 layers reconstructs all 37 residual-stream
states and both component contributions exactly.  Storing h_attn / h_out separately would
be pure redundancy, so it is not done --- `derive.py` helpers rebuild them.

Q/K/V are captured at the projection outputs, i.e. BEFORE mRoPE.  V is unaffected by RoPE
so value-contribution analysis is exact; Q/K are pre-rotation, and the rotation-dependent
quantity (the attention map itself) is stored separately and in full.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import yaml

AUTOVLA_DIR = "/root/VLA/autovla"
POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AUTOVLA_DIR)
sys.path.insert(0, os.path.join(AUTOVLA_DIR, "navsim"))
sys.path.insert(0, POC_DIR)

FORK_STEP = 6                      # generated index of the ` straightforward`/` complex` token
TOK_STRAIGHT = 30339
TOK_COMPLEX = 6351
DETAIL_LO, DETAIL_HI = 28, 36      # layers kept at full component detail
VIS_LAYERS = (0, 8, 16, 24, 28, 29, 30, 31, 32, 33, 34, 35)   # visual-token hidden states
MAX_ACT = 10                       # action tokens the codebook rollout uses; caps runaway arms
MAX_DETAIL_ROWS = 12               # attention/MLP-internal rows per sample

SPANS = ["prompt_text", "prompt_visual", "gen_think", "gen_cot", "gen_answer", "gen_action"]
SPAN_ID = {s: i for i, s in enumerate(SPANS)}
END_THINK: list = []               # filled once the tokenizer is loaded


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--annotations", default=os.path.join(POC_DIR, "outputs/annotations"))
    ap.add_argument("--records", default=os.path.join(POC_DIR, "outputs/gpu1/records.jsonl"),
                    help="previous natural run; used only to stratify the heavy-dump subset")
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/full_extract"))
    ap.add_argument("--config", default=os.path.join(
        AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--detail-n", type=int, default=250,
                    help="samples that additionally get every-token CoT hidden states and "
                         "multi-layer visual hidden states")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sanity", action="store_true", help="print shape/consistency checks")
    ap.add_argument("--force-heavy", action="store_true", help="heavy dump for every sample")
    ap.add_argument("--tensor-arms", default="NC",
                    help="arms whose .npz tensors are written (records keep both arms). "
                         "'N' alone needs ~18 GB instead of ~56 GB for 2748 scenes")
    return ap.parse_args()


# --------------------------------------------------------------------------------------
# capture plumbing
# --------------------------------------------------------------------------------------
class Capture:
    """Hook state for one teacher-forced analysis forward."""

    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.on = False
        self.keep = None          # positions retained for h_in (residual stream)
        self.keep_comp = None     # positions retained for attn_out / mlp_out
        self.keep_detail = None   # positions retained for MLP internals / QKV / attention rows
        self.keep_vis = None      # visual positions, retained at VIS_LAYERS only
        self.h_vis = {}
        self.h_in = {}
        self.attn_out = {}
        self.mlp_out = {}
        self.gate = {}
        self.up = {}
        self.gated = {}
        self.q = {}
        self.k = {}
        self.v = {}
        self.attn_map = {}

    def reset(self):
        for d in (self.h_in, self.attn_out, self.mlp_out, self.gate, self.up,
                  self.gated, self.q, self.k, self.v, self.attn_map, self.h_vis):
            d.clear()
        self.keep = self.keep_comp = self.keep_detail = self.keep_vis = None


def install_hooks(layers, cap: Capture):
    handles = []

    def sel(t, idx):
        return t.index_select(0, idx).detach().to(torch.float16).cpu()

    def mk_pre(li):
        def f(_m, a, kw):
            if not cap.on:
                return None
            hs = kw.get("hidden_states")
            if hs is None and a:
                hs = a[0]
            if hs is not None:
                cap.h_in[li] = sel(hs[0], cap.keep)
                if cap.keep_vis is not None and li in VIS_LAYERS:
                    cap.h_vis[li] = sel(hs[0], cap.keep_vis)
            return None
        return f

    def mk_attn(li):
        def f(_m, _a, out):
            if not cap.on:
                return
            o = out[0] if isinstance(out, tuple) else out
            cap.attn_out[li] = sel(o[0], cap.keep_comp)
        return f

    def mk_mlp(li):
        def f(_m, _a, out):
            if not cap.on:
                return
            o = out[0] if isinstance(out, tuple) else out
            cap.mlp_out[li] = sel(o[0], cap.keep_comp)
        return f

    def mk_lin(li, store):
        def f(_m, _a, out):
            if not cap.on or cap.keep_detail is None:
                return
            store[li] = sel(out[0], cap.keep_detail)
        return f

    for li, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(mk_pre(li), with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(mk_attn(li)))
        handles.append(layer.mlp.register_forward_hook(mk_mlp(li)))
        if DETAIL_LO <= li < DETAIL_HI:
            handles.append(layer.mlp.gate_proj.register_forward_hook(mk_lin(li, cap.gate)))
            handles.append(layer.mlp.up_proj.register_forward_hook(mk_lin(li, cap.up)))
            handles.append(layer.self_attn.q_proj.register_forward_hook(mk_lin(li, cap.q)))
            handles.append(layer.self_attn.k_proj.register_forward_hook(mk_lin(li, cap.k)))
            handles.append(layer.self_attn.v_proj.register_forward_hook(mk_lin(li, cap.v)))
    return handles


def patch_eager_attention(layers, cap: Capture):
    """
    transformers 4.49 picks the attention class at construction time, so the sdpa module
    cannot report weights.  For the detail layers we call the eager parent implementation
    with output_attentions=True, keep the rows we need, and hand back the normal tuple so
    the rest of the stack is unaffected.  Layers outside the detail range keep sdpa.
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention
    eager = Qwen2_5_VLAttention.forward

    def mk(li):
        def fwd(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kw):
            if not cap.on or cap.keep_detail is None:
                return eager(self, hidden_states, attention_mask, position_ids,
                             past_key_value, False, use_cache, cache_position,
                             position_embeddings)
            out = eager(self, hidden_states, attention_mask, position_ids,
                        past_key_value, True, use_cache, cache_position,
                        position_embeddings)
            w = out[1]                                    # (1, heads, q, k)
            cap.attn_map[li] = w[0].index_select(1, cap.keep_detail).detach().to(
                torch.float16).cpu()
            return (out[0], None) + tuple(out[2:])
        return fwd

    for li in range(DETAIL_LO, DETAIL_HI):
        m = layers[li].self_attn
        m.forward = types.MethodType(mk(li), m)


def patch_vision_cache(llm, box):
    """The visual tower output is fixed per scene; recompute it once, reuse for every pass."""
    orig = llm.visual.forward

    def fwd(pixel_values, grid_thw=None, **kw):
        if box.get("v") is not None:
            return box["v"]
        v = orig(pixel_values, grid_thw=grid_thw, **kw)
        box["v"] = v
        return v

    llm.visual.forward = fwd


# --------------------------------------------------------------------------------------
def find_sub(seq, pat):
    """First index of `pat` in `seq`, or -1. Token-id search, so no re-tokenisation drift."""
    n, m = len(seq), len(pat)
    for i in range(n - m + 1):
        if seq[i:i + m] == pat:
            return i
    return -1


def span_array(ids, n_prompt, vis_ids, a0, cot_end):
    """
    Per-position span label.

    gen_think  the 6-token stem shared by both arms, plus the fork token
    gen_cot    the reasoning body up to and including `</think>` (in arm N this is the
               one-line "straightforward" stub, which is the same slot)
    gen_answer everything after `</think>`
    gen_action action tokens, which override whatever span they fall in
    """
    sp = np.full(len(ids), SPAN_ID["prompt_text"], dtype=np.int8)
    arr = np.asarray(ids)
    vis_mask = np.isin(arr[:n_prompt], list(vis_ids))
    sp[:n_prompt] = np.where(vis_mask, SPAN_ID["prompt_visual"], SPAN_ID["prompt_text"])
    sp[n_prompt:] = SPAN_ID["gen_answer"]
    sp[n_prompt:n_prompt + FORK_STEP + 1] = SPAN_ID["gen_think"]
    if cot_end > FORK_STEP + 1:
        sp[n_prompt + FORK_STEP + 1:n_prompt + cot_end] = SPAN_ID["gen_cot"]
    sp[arr >= a0] = SPAN_ID["gen_action"]
    return sp


def topk_stats(logits_row, k):
    """logprob / entropy / top-k over the full vocabulary, computed in fp32."""
    lg = logits_row.float()
    lp = torch.log_softmax(lg, -1)
    p = lp.exp()
    ent = float(-(p * lp).sum())
    v, i = lp.topk(k)
    return ent, i.cpu().numpy().astype(np.int32), v.cpu().numpy().astype(np.float16)


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" and not os.environ.get("VLA_ANY_GPU"):
        raise SystemExit("this run is pinned to GPU 1 (RTX 5090): launch with CUDA_VISIBLE_DEVICES=1")
    for k in ("output", "scenes", "annotations", "records", "config"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.makedirs(os.path.join(args.output, "tensors"), exist_ok=True)
    os.chdir(AUTOVLA_DIR)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from models.autovla import AutoVLA
    from navsim.agents.autovla_agent import AutoVLAAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from dataset_utils.preprocessing.nuplan_dataset import get_action_instruction
    from src.labeling.labels import action_semantics, action_correct, build_perception_labels

    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["pretrained_model_path"] = os.path.join(AUTOVLA_DIR, "Qwen2.5-VL-3B-Instruct")
    cfg["model"]["codebook_cache_path"] = os.path.join(AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl")
    gen_conf = cfg["inference"]["sample"]

    t0 = time.time()
    model = AutoVLA(cfg, inference=True, device="cuda:0")
    sd = torch.load(args.checkpoint, map_location="cpu", mmap=True)["state_dict"]
    missing = model.load_state_dict(
        {k[len("autovla."):]: v for k, v in sd.items() if k.startswith("autovla.")}, strict=False)
    model.to("cuda:0").eval()
    del sd
    llm, A0 = model.vlm, model.action_start_id
    tok = model.processor.tokenizer
    body, layers, final_norm = llm.model, llm.model.layers, llm.model.norm
    nL = len(layers)
    W_act = llm.lm_head.weight[A0:].detach().float()          # (2048, hidden), fp32 readout
    vis_ids = {llm.config.video_token_id, llm.config.image_token_id}
    print(f"[build] {time.time()-t0:.0f}s  layers={nL}  A0={A0}  "
          f"unexpected_keys={len(missing.unexpected_keys)}  vis_ids={sorted(vis_ids)}", flush=True)

    cap = Capture(nL)
    install_hooks(layers, cap)
    patch_eager_attention(layers, cap)
    vis_box = {}
    patch_vision_cache(llm, vis_box)

    forced = tok.encode("<think>\nThis is a", add_special_tokens=False)
    assert len(forced) == FORK_STEP, f"forced prefix is {len(forced)} tokens, expected {FORK_STEP}"
    global END_THINK
    END_THINK = tok.encode("</think>", add_special_tokens=False)

    agent = AutoVLAAgent(
        trajectory_sampling=TrajectorySampling(
            time_horizon=cfg["model"]["trajectory"]["time_horizon"],
            interval_length=cfg["model"]["trajectory"]["interval_length"]),
        sensor_data_path=Path("dataset/nuplan/sensor_blobs/test"),
        codebook_cache_path=cfg["model"]["codebook_cache_path"],
        skip_model_load=True)

    files = sorted(Path(args.scenes).glob("*.json"))[args.start:]
    if args.limit:
        files = files[: args.limit]

    # heavy-dump subset, stratified on step-0 correctness of the previous natural run
    detail_tokens = set()
    if args.detail_n and os.path.exists(args.records):
        prev = [json.loads(l) for l in open(args.records) if l.strip()]
        wrong = [r["token"] for r in prev
                 if r.get("pred_action_idx") and r.get("gt_action_idx")
                 and r["pred_action_idx"][0] != r["gt_action_idx"][0]]
        right = [r["token"] for r in prev
                 if r.get("pred_action_idx") and r.get("gt_action_idx")
                 and r["pred_action_idx"][0] == r["gt_action_idx"][0]]
        rng = np.random.default_rng(args.seed)
        rng.shuffle(wrong); rng.shuffle(right)
        h = args.detail_n // 2
        detail_tokens = set(wrong[:h]) | set(right[: args.detail_n - h])
    print(f"[plan] {len(files)} scenes, heavy dump for {len(detail_tokens)}", flush=True)

    @torch.no_grad()
    def readout_all(states):
        """(L, n, hidden) fp16 cpu -> (L, n, 2048) action logits via final norm + lm_head."""
        outs = []
        for s in states:
            h = s.to("cuda:0", torch.bfloat16)
            outs.append((W_act @ final_norm(h.unsqueeze(0)).squeeze(0).float().T).T
                        .to(torch.float16).cpu())
        return torch.stack(outs)

    fout = open(os.path.join(args.output, "records.jsonl"), "w")
    n_ok = n_fail = 0
    tstart = time.time()

    for i, sp_path in enumerate(files, 1):
        token = sp_path.stem
        try:
            scene = json.load(open(sp_path))
            anno_p = os.path.join(args.annotations, f"{token}.json")
            anno = json.load(open(anno_p)) if os.path.exists(anno_p) else {}
            vis_box.clear()

            feats, targs = {}, {}
            for b in agent.get_feature_builders():
                feats.update(b.compute_features(scene))
            for b in agent.get_target_builders():
                targs.update(b.compute_targets(scene))
            gt_idx = np.asarray(targs["gt_idx"], dtype=np.int64).reshape(-1)

            inputs = model.get_prompt(feats)
            mi = {k: v.to("cuda:0") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            prompt_ids = mi["input_ids"][0].tolist()
            n_prompt = len(prompt_ids)

            # ---- pass 1: forced 6-token stem, then let the model choose (arm N) --------
            stem = torch.tensor([prompt_ids + forced], device="cuda:0")
            mi1 = dict(mi); mi1["input_ids"] = stem
            mi1["attention_mask"] = torch.ones_like(stem)
            g1 = llm.generate(**mi1, max_length=gen_conf["max_length"], do_sample=True,
                              temperature=gen_conf["temperature"], top_k=gen_conf["top_k"],
                              top_p=gen_conf["top_p"], return_dict_in_generate=True,
                              output_logits=True)
            n_stem = stem.shape[1]
            armN_new = g1.sequences[0][n_stem:].tolist()
            fork_logits = g1.logits[0][0]                       # full-vocab at the fork
            flp = torch.log_softmax(fork_logits.float(), -1)
            fork_ent, fork_ids, fork_lps = topk_stats(fork_logits, args.topk)

            # ---- pass 2: force ` complex` (arm C) --------------------------------------
            stem_c = torch.tensor([prompt_ids + forced + [TOK_COMPLEX]], device="cuda:0")
            mi2 = dict(mi); mi2["input_ids"] = stem_c
            mi2["attention_mask"] = torch.ones_like(stem_c)
            g2 = llm.generate(**mi2, max_length=gen_conf["max_length"], do_sample=True,
                              temperature=gen_conf["temperature"], top_k=gen_conf["top_k"],
                              top_p=gen_conf["top_p"], return_dict_in_generate=True,
                              output_logits=True)
            armC_new = g2.sequences[0][stem_c.shape[1]:].tolist()

            arms = {}
            for name, gen_ids, forced_head, logits_seq in (
                ("N", armN_new, list(forced), g1.logits),
                ("C", armC_new, list(forced) + [TOK_COMPLEX], g2.logits),
            ):
                full_new = forced_head + gen_ids
                seq = prompt_ids + full_new
                ids_t = torch.tensor([seq], device="cuda:0")

                all_act = [n_prompt + j for j, t in enumerate(full_new) if t >= A0]
                # a forced-CoT arm can run away and emit hundreds of action tokens; only the
                # first MAX_ACT drive the trajectory, so tensors stop there.  The full id list
                # stays in the record so nothing is lost.
                act_pos = all_act[:MAX_ACT]
                truncated = bool(full_new and full_new[-1] != tok.eos_token_id)
                runaway = len(all_act) > MAX_ACT
                text = tok.decode(full_new, skip_special_tokens=False)
                think_end = find_sub(full_new, END_THINK)
                cot_end = think_end + len(END_THINK) if think_end >= 0 else len(full_new)

                # ---- analysis forward: teacher-forced, hooks on ------------------------
                # Action token k is emitted by the logits AT POSITION act_pos[k] - 1, not at
                # act_pos[k].  Action tokens are contiguous, so the union of "states that
                # generate an action" and "states sitting on an action" is just one extra
                # position at the front: qpos[j] generates action j, and qpos[j] is also the
                # state at action token j-1.
                qpos = ([act_pos[0] - 1] + act_pos) if act_pos else [len(seq) - 1]
                heavy = args.force_heavy or token in detail_tokens
                cot_pos = ([n_prompt + j for j in range(FORK_STEP, cot_end)]
                           if heavy and name == "C" else [])
                keep_list = sorted(set(qpos) | set(cot_pos))
                cap.reset()
                cap.keep = torch.tensor(keep_list, device="cuda:0", dtype=torch.long)
                cap.keep_comp = torch.tensor(qpos, device="cuda:0", dtype=torch.long)
                cap.keep_detail = (torch.tensor(qpos[:MAX_DETAIL_ROWS], device="cuda:0",
                                                dtype=torch.long)
                                   if act_pos and name == "C" else None)
                vis_pos = (np.where(np.isin(np.asarray(prompt_ids), list(vis_ids)))[0]
                           if heavy and name == "C" else None)
                cap.keep_vis = (torch.tensor(vis_pos, device="cuda:0", dtype=torch.long)
                                if vis_pos is not None and len(vis_pos) else None)
                cap.on = True
                mi3 = dict(mi); mi3["input_ids"] = ids_t
                mi3["attention_mask"] = torch.ones_like(ids_t)
                with torch.no_grad():
                    fw = llm(**mi3, use_cache=False)
                cap.on = False

                h_in = torch.stack([cap.h_in[l] for l in range(nL)])       # (36, n_keep, H)
                a_out = torch.stack([cap.attn_out[l] for l in range(nL)])  # (36, n_act,  H)
                m_out = torch.stack([cap.mlp_out[l] for l in range(nL)])

                kept = np.asarray(keep_list, dtype=np.int64)
                pos_of = {int(p): j for j, p in enumerate(kept)}
                q_rows = np.array([pos_of[p] for p in qpos], dtype=np.int64)
                h_q = h_in[:, q_rows]                                      # (36, n_q, H)
                h_all_q = torch.cat([h_q, (h_q[-1] + a_out[-1] + m_out[-1]).unsqueeze(0)])
                act_logits = readout_all(h_all_q)                          # (37, n_q, 2048)

                spans = span_array(seq, n_prompt, vis_ids, A0, cot_end)
                pred_idx = [t - A0 for t in full_new if t >= A0]
                head_ids = [t for t in full_new if t >= A0][:MAX_ACT]
                traj = (model.action_tokenizer.decode_token_ids_to_trajectory(
                    torch.tensor(head_ids)) if head_ids else None)
                traj = np.asarray(traj[0, 1:], np.float32) if traj is not None \
                    else np.zeros((0, 3), np.float32)

                # per-generated-token stats over the whole arm
                ent, tk_i, tk_v, lps = [], [], [], []
                for s_i, lg in enumerate(logits_seq):
                    e, ii, vv = topk_stats(lg[0], args.topk)
                    ent.append(e); tk_i.append(ii); tk_v.append(vv)
                    nxt = full_new[len(forced_head) + s_i] if len(forced_head) + s_i < len(full_new) else None
                    lps.append(float(torch.log_softmax(lg[0].float(), -1)[nxt]) if nxt is not None else np.nan)

                blob = {
                    "h_in": h_in.numpy(), "attn_out": a_out.numpy(), "mlp_out": m_out.numpy(),
                    "action_logits": act_logits.numpy(),
                    "kept_positions": kept.astype(np.int32),
                    "q_positions": np.asarray(qpos, np.int32),
                    "q_rows_in_kept": q_rows.astype(np.int32),
                    # row j of attn_out/mlp_out/action_logits/attn_map generates action
                    # token `generates_action[j]`; -1 means the row generates a non-action
                    "generates_action": np.asarray(
                        list(range(len(act_pos))) + [-1] if act_pos else [-1], np.int32),
                    "action_positions": np.asarray(act_pos, np.int32),
                    "span_ids": spans,
                    "input_ids": np.asarray(seq, np.int32),
                    "gen_entropy": np.asarray(ent, np.float32),
                    "gen_topk_ids": np.asarray(tk_i, np.int32),
                    "gen_topk_logprob": np.asarray(tk_v, np.float16),
                    "gen_chosen_logprob": np.asarray(lps, np.float32),
                }
                if cap.keep_detail is not None:
                    dl = list(range(DETAIL_LO, DETAIL_HI))
                    blob["detail_layers"] = np.asarray(dl, np.int32)
                    blob["mlp_gate"] = torch.stack([cap.gate[l] for l in dl]).numpy()
                    blob["mlp_up"] = torch.stack([cap.up[l] for l in dl]).numpy()
                    blob["q_proj"] = torch.stack([cap.q[l] for l in dl]).numpy()
                    blob["k_proj"] = torch.stack([cap.k[l] for l in dl]).numpy()
                    blob["v_proj"] = torch.stack([cap.v[l] for l in dl]).numpy()
                    blob["attn_map"] = torch.stack([cap.attn_map[l] for l in dl]).numpy()
                if cap.keep_vis is not None:
                    blob["visual_layers"] = np.asarray(VIS_LAYERS, np.int32)
                    blob["visual_positions"] = vis_pos.astype(np.int32)
                    blob["visual_hidden"] = torch.stack(
                        [cap.h_vis[l] for l in VIS_LAYERS]).numpy()
                if cot_pos:
                    blob["cot_positions"] = np.asarray(cot_pos, np.int32)

                if name in args.tensor_arms:
                    np.savez(os.path.join(args.output, "tensors", f"{token}_{name}.npz"), **blob)
                arms[name] = {
                    "generated_text": text,
                    "n_generated": len(full_new),
                    "token_ids": full_new,
                    "action_positions": act_pos,
                    "action_token_ids": [t for t in full_new if t >= A0],
                    "pred_action_idx": pred_idx,
                    "cot_end": int(cot_end),
                    "cot_present": "complex scenario" in text,
                    "truncated": truncated,
                    "runaway_action_tokens": len(all_act) if runaway else 0,
                    "trajectory_pred": traj[:, :2].tolist(),
                    "tensor_file": (f"tensors/{token}_{name}.npz"
                                    if name in args.tensor_arms else None),
                    "heavy_dump": bool(heavy and name == "C"),
                    "has_detail": cap.keep_detail is not None,
                }
                if args.sanity and act_pos:
                    ref = fw.logits[0, qpos[0], A0:].float()
                    ours = act_logits[-1, 0].to("cuda:0").float()
                    hits = sum(int(act_logits[-1, j].argmax()) == pred_idx[j]
                               for j in range(min(len(act_pos), MAX_ACT)))
                    gt0, pr0 = int(gt_idx[0]), pred_idx[0]
                    amask = np.asarray(spans)[np.asarray(qpos)] == SPAN_ID["gen_action"]
                    print(f"  [{name}] seq={len(seq)} n_act={len(all_act)} keep={len(kept)} "
                          f"h_in{tuple(h_in.shape)} comp{tuple(a_out.shape)} "
                          f"logits{tuple(act_logits.shape)} "
                          f"maxdiff={float((ref - ours).abs().max()):.3f} "
                          f"argmax==generated {hits}/{min(len(act_pos), MAX_ACT)} "
                          f"margin(GT-pred)@0={float(ours[gt0] - ours[pr0]):+.2f} "
                          f"qspan_action={int(amask.sum())}/{len(qpos)} "
                          f"vis{tuple(blob['visual_hidden'].shape) if 'visual_hidden' in blob else '-'} "
                          f"attn{tuple(blob['attn_map'].shape) if 'attn_map' in blob else '-'} "
                          f"trunc={truncated}", flush=True)

            P, _ = build_perception_labels(anno)
            sem_g = action_semantics(np.asarray(scene["gt_trajectory"], np.float32),
                                     get_action_instruction)
            rec = {
                "token": token,
                "log_name": anno.get("log_name"), "map_name": anno.get("map_name"),
                "instruction": scene.get("instruction"),
                "gt_action_instruction": scene.get("gt_action_instruction"),
                "velocity": scene.get("velocity"), "acceleration": scene.get("acceleration"),
                "his_trajectory": scene.get("his_trajectory"),
                "trajectory_gt": scene["gt_trajectory"],
                "gt_action_idx": gt_idx.tolist(),
                "gt_action_token_ids": (gt_idx + A0).tolist(),
                "camera_paths": {k: scene[k] for k in scene if k.endswith("_camera_paths")},
                "prompt_text": tok.decode(prompt_ids, skip_special_tokens=False),
                "n_prompt_tokens": n_prompt,
                "n_visual_tokens": int(np.isin(np.asarray(prompt_ids), list(vis_ids)).sum()),
                "fork": {
                    "step": FORK_STEP,
                    "p_complex": float(flp[TOK_COMPLEX].exp()),
                    "p_straightforward": float(flp[TOK_STRAIGHT].exp()),
                    "logit_complex": float(fork_logits[TOK_COMPLEX]),
                    "logit_straightforward": float(fork_logits[TOK_STRAIGHT]),
                    "entropy": fork_ent,
                    "topk_ids": fork_ids.tolist(),
                    "topk_logprob": fork_lps.astype(np.float32).tolist(),
                    "argmax_id": int(fork_logits.argmax()),
                },
                "perception": P.to_dict(),
                "action_gt": sem_g.to_dict(),
                "arms": arms,
                "span_names": SPANS,
            }
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            if n_fail <= 5:
                import traceback; traceback.print_exc()
                print(f"[fail] {token}: {type(exc).__name__}: {exc}", flush=True)
        if i % 25 == 0 or args.sanity:
            el = time.time() - tstart
            print(f"[prog] {i}/{len(files)} ok={n_ok} fail={n_fail} {el/i:.2f}s/sample "
                  f"eta {(len(files)-i)*el/i/3600:.2f}h", flush=True)

    fout.close()
    meta = {
        "n_ok": n_ok, "n_fail": n_fail, "n_layers": nL, "action_start_id": A0,
        "forced_prefix_ids": forced, "tok_complex": TOK_COMPLEX,
        "tok_straightforward": TOK_STRAIGHT, "detail_layers": [DETAIL_LO, DETAIL_HI],
        "spans": SPANS, "topk": args.topk, "seed": args.seed,
        "checkpoint": args.checkpoint, "config": args.config,
        "sample_config": gen_conf,
        "detail_tokens": sorted(detail_tokens),
        "elapsed_h": (time.time() - tstart) / 3600,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "tensor_arms": args.tensor_arms,
        "layout": {
            "h_in": "(36, n_kept, 2048) fp16 - layer input; h_in[0] = embeddings",
            "attn_out": "(36, n_kept, 2048) fp16 - self_attn residual contribution",
            "mlp_out": "(36, n_kept, 2048) fp16 - mlp residual contribution",
            "identity": "h_out[l] = h_in[l] + attn_out[l] + mlp_out[l] = h_in[l+1]",
            "action_logits": "(37, n_action, 2048) fp16 - final_norm + lm_head[A0:], fp32 math",
            "attn_map": "(8, 16, n_action, seq) fp16 - rows are action queries",
            "q_proj/k_proj/v_proj": "(8, n_action, 2048|256) fp16 - PRE-RoPE projections",
            "mlp_gate/mlp_up": "(8, n_action, 11008) fp16; gated = silu(gate)*up",
            "kept_positions": "absolute token positions of the rows in h_in/attn_out/mlp_out",
            "span_ids": "(seq,) int8 index into spans",
        },
    }
    with open(os.path.join(args.output, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] ok={n_ok} fail={n_fail} in {meta['elapsed_h']:.2f}h", flush=True)


if __name__ == "__main__":
    main()
