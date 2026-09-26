#!/usr/bin/env python
"""
Instrumented ORION extraction: one pass, everything later CPU analyses need.

Nothing in /root/VLA/orion is edited.  The LLM forward inside `inference_ego` is wrapped at
runtime and read with hooks; the value `inference_ego` returns is untouched, so ORION's own
planner output stays bit-for-bit identical to the uninstrumented run.  `--verify-against`
checks exactly that against the previous run's records.

WHAT ORION DOES AND DOES NOT SHARE WITH AutoVLA
----------------------------------------------
ORION emits ONE `<waypoint_ego>` token whose final-layer state is handed to a VAE planner
(orion.py:794-819).  There is no action-token vocabulary and no autoregressive action
sequence, so the AutoVLA analyses that depend on those --- first-action error, error
propagation, action logit lens, the ` straightforward`/` complex` CoT fork --- have no
ORION counterpart and are not faked here.  What does transfer is captured in full:
residual-stream decomposition, Q/K/V, MLP internals, head-wise attention, and probing
features.

DETERMINISM (measured, not assumed)
-----------------------------------
ORION's trajectory is stochastic: the QA rounds are sampled (orion.py:883, do_sample=True,
T=0.1, top_p=0.75) and each sampled answer is appended to the context the final
`<waypoint_ego>` planning round reads.  Measured behaviour:

  same split, run twice            -> 6/6 frames bit-identical
  same split, instrumented vs not  -> 10/10 frames bit-identical
  stride 6 vs stride 3, same frame -> only the first frame after reset_memory matches

So the model replays exactly when the frame schedule is replayed, but a frame's result
depends on which earlier frames ran the LLM.  The LLM block writes no persistent state
(`self.fut_ts = 6` is its only assignment, and it is idempotent) and the detection/map
heads update memory identically on warm-up frames, which leaves allocator/workspace state
changing fp16 kernel selection, amplified into different tokens by the low-temperature
sampling, as the remaining explanation.

Consequence for later work: an intervention experiment must replay this exact schedule, and
collateral damage may only be read off frames at or before the patched one.  Per-frame
reseeding (see `seed_frame`) removes the RNG channel so that at least the draw is a
function of (model, frame, seed); it is a deliberate deviation from the official loop,
which seeds once globally.

Because of this, trajectories here do NOT reproduce the earlier stride-6 PoC run, and
`--verify-against` only quantifies the gap rather than asserting equality.

RESIDUAL STREAM
---------------
    h_out[l] = h_in[l] + attn_out[l] + mlp_out[l]     h_in[l+1] = h_out[l]

so (h_in, attn_out, mlp_out) over the 32 layers reconstructs all 33 states exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from analysis.orion_hooks import _resolve_causal_lm, _waypoint_mask
from analysis.chatb2d_parser import build_perception_labels, build_reasoning_labels  # noqa: F401

IMAGE_TOKEN_INDEX = -200
DETAIL_LO, DETAIL_HI = 24, 32
VIS_LAYERS = (0, 8, 16, 20, 24, 25, 26, 27, 28, 29, 30, 31)
TOPK = 20


def seed_frame(sample_id: str, base: int) -> None:
    """Deterministic per-frame RNG state, independent of how many frames ran before."""
    import hashlib
    import random
    h = hashlib.sha256(f"{base}:{sample_id}".encode()).digest()
    s = int.from_bytes(h[:4], "little")
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class Cap:
    def __init__(self):
        self.on = False
        self.want_vis = False     # heavy samples also keep visual-token hidden states
        self.keep = None          # <waypoint_ego> positions
        self.keep_vis = None      # visual positions (heavy samples only)
        self.h_in, self.attn_out, self.mlp_out = {}, {}, {}
        self.gate, self.up, self.q, self.k, self.v, self.attn_map = {}, {}, {}, {}, {}, {}
        self.h_vis = {}
        self.seq_len = None
        self.n_vis = None
        self.err = None

    def reset(self):
        for d in (self.h_in, self.attn_out, self.mlp_out, self.gate, self.up,
                  self.q, self.k, self.v, self.attn_map, self.h_vis):
            d.clear()
        self.keep = self.keep_vis = None
        self.err = None


def install(causal_lm, cap: Cap):
    """Hooks on every decoder layer, plus eager attention capture on the detail layers."""
    layers = causal_lm.model.layers

    def sel(t, idx):
        return t.index_select(0, idx).detach().to(torch.float16).cpu()

    def mk_pre(li):
        def f(_m, a, kw):
            if not cap.on:
                return None
            hs = kw.get("hidden_states") if kw else None
            if hs is None and a:
                hs = a[0]
            if hs is None:
                return None
            cap.h_in[li] = sel(hs[0], cap.keep)
            if cap.keep_vis is not None and li in VIS_LAYERS:
                cap.h_vis[li] = sel(hs[0], cap.keep_vis)
            return None
        return f

    def mk_out(li, store):
        def f(_m, _a, out):
            if not cap.on:
                return
            o = out[0] if isinstance(out, tuple) else out
            store[li] = sel(o[0], cap.keep)
        return f

    def mk_lin(li, store):
        def f(_m, _a, out):
            if not cap.on:
                return
            store[li] = sel(out[0], cap.keep)
        return f

    handles = []
    for li, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(mk_pre(li), with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(mk_out(li, cap.attn_out)))
        handles.append(layer.mlp.register_forward_hook(mk_out(li, cap.mlp_out)))
        if DETAIL_LO <= li < DETAIL_HI:
            handles.append(layer.mlp.gate_proj.register_forward_hook(mk_lin(li, cap.gate)))
            handles.append(layer.mlp.up_proj.register_forward_hook(mk_lin(li, cap.up)))
            handles.append(layer.self_attn.q_proj.register_forward_hook(mk_lin(li, cap.q)))
            handles.append(layer.self_attn.k_proj.register_forward_hook(mk_lin(li, cap.k)))
            handles.append(layer.self_attn.v_proj.register_forward_hook(mk_lin(li, cap.v)))

    # transformers 4.31 LlamaAttention returns its weights only when asked; force it on the
    # detail layers only, so the other 24 keep the cheap path and the model's own
    # `output_attentions` config is left alone.
    def mk_attn_fwd(li, orig):
        def fwd(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False, **kw):
            if not cap.on:
                return orig(hidden_states, attention_mask, position_ids, past_key_value,
                            output_attentions, use_cache, **kw)
            out = orig(hidden_states, attention_mask, position_ids, past_key_value,
                       True, use_cache, **kw)
            w = out[1]
            if w is not None:
                cap.attn_map[li] = w[0].index_select(1, cap.keep).detach().to(
                    torch.float16).cpu()
            return (out[0], None) + tuple(out[2:])
        return fwd

    for li in range(DETAIL_LO, DETAIL_HI):
        m = layers[li].self_attn
        m.forward = types.MethodType(mk_attn_fwd(li, m.forward), m)
    return handles


def patch_inference_ego(model, cap: Cap):
    """
    Same contract as `analysis.orion_hooks.install_planning_capture`, but the waypoint mask
    is resolved BEFORE the LLM forward so the hooks can slice to those positions instead of
    retaining the whole 919-token sequence at 33 layers.
    """
    causal_lm = _resolve_causal_lm(model.lm_head)
    original = causal_lm.inference_ego

    def patched(inputs=None, images=None, image_sizes=None, return_ego_feature=False, **kwargs):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _,
             new_input_ids) = causal_lm.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images,
                image_sizes=image_sizes)
        else:
            inputs_embeds = causal_lm.get_model().embed_tokens(inputs)
            new_input_ids = inputs

        loc = _waypoint_mask(new_input_ids, causal_lm.config.waypoint_token_idx)
        cap.reset()
        cap.seq_len = int(new_input_ids.shape[1])
        wp = torch.nonzero(loc[0]).flatten()
        vis = torch.nonzero(new_input_ids[0] == IMAGE_TOKEN_INDEX).flatten()
        cap.n_vis = int(vis.numel())
        if wp.numel() == 0:
            cap.err = "no <waypoint_ego> position"
            cap.on = False
        else:
            cap.keep = wp.to(inputs_embeds.device)
            if cap.want_vis and vis.numel():
                cap.keep_vis = vis.to(inputs_embeds.device)
            cap.on = True

        outputs = causal_lm.model(
            input_ids=inputs, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=None, inputs_embeds=inputs_embeds, use_cache=True,
            output_attentions=causal_lm.config.output_attentions,
            output_hidden_states=False, return_dict=True,
        )
        cap.on = False

        hidden_states = outputs[0]
        selected_hidden_states = hidden_states[loc.to(hidden_states.device)]
        if return_ego_feature:
            return selected_hidden_states
        raise AssertionError("inference_ego called with return_ego_feature=False")

    causal_lm.inference_ego = patched
    return causal_lm, original


def main() -> None:
    import scripts.run_orion_hidden_extract as base

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=os.path.join(HERE, "splits/full_all.json"))
    ap.add_argument("--output", default=os.path.join(HERE, "outputs/full_extract"))
    ap.add_argument("--config", default=os.path.join(HERE, "configs/orion_poc_cot_fp16.py"))
    ap.add_argument("--checkpoint", default="/root/VLA/orion/ckpts/Orion.pth")
    ap.add_argument("--heavy-n", type=int, default=250)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-against", default=os.path.join(HERE, "outputs/merged/records.jsonl"),
                    help="previous run; with per-frame reseeding trajectories are EXPECTED to\n                             differ (different RNG draw), this only quantifies by how much")
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise SystemExit("pinned to GPU 0: launch with CUDA_VISIBLE_DEVICES=0")
    # build_everything() chdir's into the ORION repo, so every path must be resolved first.
    for k in ("split", "output", "config", "checkpoint", "verify_against"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.makedirs(os.path.join(args.output, "tensors"), exist_ok=True)

    with open(args.split) as f:
        shard = json.load(f)
    if args.limit:
        shard = shard[: args.limit]
    rng = np.random.default_rng(args.seed)
    heavy = set(rng.choice([r["sample_id"] for r in shard],
                           size=min(args.heavy_n, len(shard)), replace=False).tolist())

    prev = {}
    if args.verify_against and os.path.exists(args.verify_against):
        for line in open(args.verify_against):
            if line.strip():
                r = json.loads(line)
                prev[r["sample_id"]] = r.get("trajectory_pred")
    print(f"[plan] {len(shard)} frames, heavy {len(heavy)}, "
          f"verify pool {len(prev)}", flush=True)

    ns = argparse.Namespace(config=args.config, checkpoint=args.checkpoint, seed=args.seed,
                            gpu=0, layers="all")
    _cfg, dataset, model, wrapped = base.build_everything(ns)
    cap = Cap()
    cap.want_vis = False
    causal_lm, _orig = patch_inference_ego(model, cap)
    install(causal_lm, cap)

    _key2idx, per_clip = base.index_dataset(dataset)
    wanted = {(r["clip_id"], r["frame_idx"]): r for r in shard}
    usable = [c for c in per_clip if any(k[0] == c for k in wanted)]

    fout = open(os.path.join(args.output, "records.jsonl"), "w")
    n_done = n_warm = n_fail = n_mismatch = 0
    t0 = time.time()

    for clip in usable:
        frames = per_clip[clip]
        sel_in_clip = {fi for (c, fi) in wanted if c == clip}
        last_needed = max(sel_in_clip)
        if model.with_pts_bbox:
            model.pts_bbox_head.reset_memory()
        if model.with_map_head:
            model.map_head.reset_memory()

        for fi, didx in frames:
            if fi > last_needed:
                break
            selected = fi in sel_in_clip
            try:
                data = base.collate_one(dataset, didx)
                if not selected:
                    with torch.no_grad(), base.LMToggle(model):
                        wrapped(data, return_loss=False)
                    n_warm += 1
                    continue

                rec_in = wanted[(clip, fi)]
                sid = rec_in["sample_id"]
                cap.want_vis = sid in heavy
                # ORION's QA rounds are SAMPLED (orion.py:883, do_sample=True, T=0.1,
                # top_p=0.75) and each sampled answer is appended to the context that the
                # final `<waypoint_ego>` planning round reads.  The trajectory is therefore
                # stochastic, and with one global seed it depends on how many generate()
                # calls preceded this frame --- i.e. on the stride.  Reseeding from the
                # sample id makes each frame's draw a function of (model, frame, seed)
                # alone, so the extraction is reproducible, comparable across strides, and
                # usable as an intervention baseline where only the patched frame may move.
                seed_frame(sid, args.seed)
                with torch.no_grad():
                    out = wrapped(data, return_loss=False)

                res = out["bbox_results"][0] if isinstance(out, dict) else out[0]
                pts = res.get("pts_bbox", {})
                pred = base.decumsum(base.to_numpy_traj(pts.get("ego_fut_preds")))
                rounds = base.extract_answers(res.get("text_out", []))

                if sid in prev and prev[sid] is not None:
                    a = np.asarray(prev[sid], np.float32)
                    b = np.asarray(pred, np.float32)
                    if a.shape != b.shape or not np.array_equal(a, b):
                        n_mismatch += 1

                h_in = torch.stack([cap.h_in[l] for l in range(len(causal_lm.model.layers))])
                a_out = torch.stack([cap.attn_out[l] for l in range(len(causal_lm.model.layers))])
                m_out = torch.stack([cap.mlp_out[l] for l in range(len(causal_lm.model.layers))])
                dl = list(range(DETAIL_LO, DETAIL_HI))
                blob = {
                    "h_in": h_in.numpy(), "attn_out": a_out.numpy(), "mlp_out": m_out.numpy(),
                    "waypoint_positions": cap.keep.cpu().numpy().astype(np.int32),
                    "seq_len": np.int32(cap.seq_len), "n_visual_tokens": np.int32(cap.n_vis),
                    "detail_layers": np.asarray(dl, np.int32),
                    "mlp_gate": torch.stack([cap.gate[l] for l in dl]).numpy(),
                    "mlp_up": torch.stack([cap.up[l] for l in dl]).numpy(),
                    "q_proj": torch.stack([cap.q[l] for l in dl]).numpy(),
                    "k_proj": torch.stack([cap.k[l] for l in dl]).numpy(),
                    "v_proj": torch.stack([cap.v[l] for l in dl]).numpy(),
                    "attn_map": torch.stack([cap.attn_map[l] for l in dl]).numpy(),
                }
                if cap.h_vis:
                    blob["visual_layers"] = np.asarray(VIS_LAYERS, np.int32)
                    blob["visual_hidden"] = torch.stack(
                        [cap.h_vis[l] for l in VIS_LAYERS]).numpy()
                np.savez(os.path.join(args.output, "tensors", f"{sid}.npz"), **blob)

                info = dataset.data_infos[didx]
                P_m, _ = build_perception_labels(rounds)
                R_m = build_reasoning_labels(rounds)
                gt_delta = base.to_numpy_traj(base.first_tensor(data.get("ego_fut_trajs")))
                rec = {
                    "sample_id": sid, "clip_id": clip, "frame_idx": fi,
                    "scenario": rec_in["scenario"], "dataset_index": didx,
                    "town": info.get("town_name"),
                    "command": base.command_index(data.get("ego_fut_cmd")),
                    "trajectory_pred": pred, "trajectory_gt": gt_delta,
                    "fut_valid_flag": bool(pts.get("fut_valid_flag", False)),
                    "model_rounds": rounds, "gt_rounds": rec_in["gt_rounds"],
                    "model_perception": P_m.to_dict(), "model_reasoning": R_m.to_dict(),
                    "gt_perception": rec_in["gt_perception"],
                    "gt_reasoning": rec_in["gt_reasoning"],
                    "seq_len": cap.seq_len, "n_visual_tokens": cap.n_vis,
                    "n_waypoint_positions": int(cap.keep.numel()),
                    "tensor_file": f"tensors/{sid}.npz",
                    "heavy_dump": sid in heavy,
                    "capture_error": cap.err,
                }
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                n_done += 1
                if args.sanity and n_done <= 5:
                    print(f"  [{sid}] seq={cap.seq_len} vis={cap.n_vis} wp={cap.keep.tolist()} "
                          f"h_in{tuple(h_in.shape)} attn{tuple(blob['attn_map'].shape)} "
                          f"gate{tuple(blob['mlp_gate'].shape)} "
                          f"vis{tuple(blob['visual_hidden'].shape) if 'visual_hidden' in blob else '-'} "
                          f"traj_match={'n/a' if sid not in prev else np.array_equal(np.asarray(prev[sid],np.float32), np.asarray(pred,np.float32))}",
                          flush=True)
                if n_done % 25 == 0:
                    el = time.time() - t0
                    print(f"[prog] {n_done}/{len(shard)} ({n_warm} warm) "
                          f"{el/n_done:.2f}s/sel eta {(len(shard)-n_done)*el/n_done/3600:.2f}h "
                          f"differs_from_prev={n_mismatch}", flush=True)
            except Exception as exc:
                n_fail += 1
                if n_fail <= 5:
                    import traceback; traceback.print_exc()
                    print(f"[fail] {clip}:{fi} {type(exc).__name__}: {exc}", flush=True)

    fout.close()
    meta = {
        "split": os.path.abspath(args.split), "n_selected": n_done, "n_warmup": n_warm,
        "n_failed": n_fail, "n_differs_from_previous_run": n_mismatch,
        "n_layers": len(causal_lm.model.layers), "detail_layers": [DETAIL_LO, DETAIL_HI],
        "visual_layers": list(VIS_LAYERS), "heavy_n": len(heavy),
        "checkpoint": args.checkpoint, "config": os.path.abspath(args.config),
        "seed": args.seed, "elapsed_h": (time.time() - t0) / 3600,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "layout": {
            "h_in": "(32, n_wp, 4096) fp16 layer input; h_in[0] = embeddings",
            "attn_out": "(32, n_wp, 4096) self_attn residual contribution",
            "mlp_out": "(32, n_wp, 4096) mlp residual contribution",
            "identity": "h_out[l] = h_in[l] + attn_out[l] + mlp_out[l] = h_in[l+1]",
            "attn_map": "(8, 32 heads, n_wp, seq) fp16, rows are <waypoint_ego> queries",
            "q_proj/k_proj/v_proj": "(8, n_wp, 4096) fp16, pre-RoPE",
            "mlp_gate/mlp_up": "(8, n_wp, 11008) fp16; gated = silu(gate)*up",
            "visual_hidden": "(12, n_vis, 4096) fp16 at visual_layers, heavy samples only",
        },
    }
    with open(os.path.join(args.output, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] selected={n_done} warm={n_warm} fail={n_fail} "
          f"differs_from_prev={n_mismatch} in {meta['elapsed_h']:.2f}h", flush=True)


if __name__ == "__main__":
    main()
