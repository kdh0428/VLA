#!/usr/bin/env python
"""
P1-B AutoVLA no-op / cache parity engine (RTX 5090 only). Imported by p1b_harness.py.

It re-implements, for a fixed manifest unit, the per-step decoding of the four AutoVLA intervention code paths
(rows and context rules are IMPORTED from the experiment scripts where they are module-level; closures that live inside
the scripts' main() are copied verbatim, with the source line range sha256-checked at import time, see VERBATIM):

  path  experiment script (HEAD 817d166)                     batch B   rows                     hooks in experiment
  H7    scripts/action_history_causal.py  (exp 7, P1-C H7)    1         normal/recent_gt/gt_history  embed fwd-hook + attention wrappers (all layers)
  H8    scripts/prev_action_state_patching.py (exp 8, P1-C H8) 93       build_rows() (93 rows)   layer-0 pre-hook + 36 layer post-hooks (patch plan)
  H9    scripts/temporal_feedback_window.py (exp 9, P1-C H9)  17        build_rows() (17 rows)   none
  P1A   scripts/review_followup/p1ab/p1a_gt_free_reference.py 10        ROWS (10 rows)           none
All paths: prefix = prompt + stored fast stub + prefix actions; prefix KV once per scene (expanded to B rows); every step
the post-t* span is re-encoded on top of the prefix cache and the cache is cropped back to P; logits over the 2,048
action rows in float64; greedy = argmax (first index on ties) or the experiments' T = 0.01 sampling with a torch.Generator
re-seeded per row from sha256(f"{seed}:{token}:{key}:{k}") (common random numbers).

Forward modes (what the controls vary):
  span         the experiments' cached-edit path (crop to P, re-encode the whole post-t* span)        [C1, C2, C5]
  incremental  standard AR KV decoding: only the newest token is fed, cache grows (append-only rows)  [C3a]
  deep         crop the cache back to the stub start, re-encode stub + prefix actions with the same
               tokens, then 'span'                                                                     [C3b]
  full         no cache: prompt + stub + prefix + span re-encoded from position 0 every step, B = 1   [C4]
Hook modes:  none | installed (experiment hooks, no edit) | selfpatch (edit writes back the read value) | experiment
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import time

import numpy as np

POC = "/root/VLA/autovla_misalignment_poc"
SCRIPTS = f"{POC}/scripts"
P1AB = f"{SCRIPTS}/review_followup/p1ab"
for _p in (P1AB, SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import p1_common as C  # noqa: E402

# ---------------------------------------------------------------------------------------------- verbatim provenance
# (file, first line, last line, sha256 of those lines at HEAD 817d166) for every block re-implemented below.
VERBATIM = {
    "H7_context_rule": ("action_history_causal.py", 207, 215),
    "H7_embed_hook": ("action_history_causal.py", 118, 128),
    "H7_attn_wrapper": ("action_history_causal.py", 130, 155),
    "H7_sampling": ("action_history_causal.py", 227, 235),
    "H8_context_rule": ("prev_action_state_patching.py", 288, 296),
    "H8_patch_plan": ("prev_action_state_patching.py", 164, 172),
    "H8_apply_patch_and_hooks": ("prev_action_state_patching.py", 174, 213),
    "H8_sampling": ("prev_action_state_patching.py", 319, 326),
    "H9_context_rule": ("temporal_feedback_window.py", 189, 197),
    "P1A_context_for": ("review_followup/p1ab/p1a_gt_free_reference.py", 72, 85),
    "P1A_exec_acts": ("review_followup/p1ab/p1a_gt_free_reference.py", 206, 208),
    "prefix_cache": ("review_followup/p1ab/p1_common.py", 231, 256),
    "forward_spans": ("review_followup/p1ab/p1_common.py", 268, 281),
}


def verbatim_hashes():
    out = {}
    for k, (f, a, b) in VERBATIM.items():
        lines = open(os.path.join(SCRIPTS, f)).read().split("\n")[a - 1:b]
        out[k] = {"file": f, "lines": f"{a}-{b}", "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest()}
    return out


def h(x) -> str:
    if hasattr(x, "detach"):
        x = x.detach().cpu().contiguous().numpy()
    if isinstance(x, np.ndarray):
        return hashlib.sha256(x.tobytes() + str(x.dtype).encode() + str(x.shape).encode()).hexdigest()[:16]
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------- rows per path
def path_rows(path):
    if path == "H7":
        return [{"name": "normal"}, {"name": "recent_gt"}, {"name": "gt_history"}]
    if path == "H8":
        import prev_action_state_patching as PASP
        return PASP.build_rows()
    if path == "H9":
        import temporal_feedback_window as TFW
        return TFW.build_rows()
    if path == "P1A":
        import p1a_gt_free_reference as P1
        return [dict(r) for r in P1.ROWS]
    if path == "H32":                                                    # AMENDMENT P1-03
        import motion_semantics_ablation as MSA
        return [{"name": n} for n in MSA.ROWS]
    if path == "P1B":                                                    # AMENDMENT P1-03
        import p1b_start_position as PB
        return PB.build_rows()
    raise ValueError(path)


KEY_ROWS = {
    "H7": ["normal", "recent_gt", "gt_history"],
    "H8": ["normal", "gt_history", "recent_gt", "reverse@emb", "patch_full@emb", "selfpatch@emb", "selfpatch@L16",
           "selfpatch@L35", "reverse@L16", "patch_full@L35"],
    "H9": ["normal", "win1", "win3", "win4", "gt_history"],
    "P1A": ["U", "OC", "O1", "C1", "OC_ref1", "OC_refR", "OC_refH", "O_refH", "OC_gtR", "OC_gtH"],
    "H32": ["normal", "recent_gt", "dir_ok_mag_wrong", "dir_wrong_mag_ok", "mirror_same_dist", "random_mag_matched"],
    "P1B": ["normal", "full_gt", "full_self", "recent_gt", "recent_self", "P_gt_s1_w1", "T_gt_s1_w1", "P_self_s2_w2",
            "T_self_s2_w2", "P_gt_s4_w3"],
}

_MSA = {}


def msa_tables(llm, A0):
    """motion_semantics_ablation.py:166-171: token motion table and trained-token ids (from the live embeddings)."""
    if not _MSA:
        import pickle
        import motion_semantics_ablation as MSA
        act_emb = llm.get_input_embeddings().weight[A0:A0 + MSA.N_CODES].detach().float().cpu().numpy()
        codebook = np.asarray(pickle.load(open(C.CODEBOOK, "rb"))["token_all"]["veh"])
        untrained, _, _, _, _ = MSA.token_spaces(act_emb, codebook)
        _MSA.update(MOT=MSA.token_motion(codebook), trained=np.where(~untrained)[0], MSA=MSA)
    return _MSA


def base_of(path, row):
    """context family of a row (patch rows of H8 share their base's context)."""
    if path == "H8":
        return {"normal": "normal", "gt": "gt_history", "recent_gt": "recent_gt"}[row["base"]]
    return row["name"]


def context(path, row, k, t, own, gt, ref, p, self_override=False):
    """(span tokens relative, sources) for step k; the span covers action positions t..k-1.
    self_override (C2 token-level self-patch): every position the row's rule would override is filled through the same
    rule with the row's OWN token at that position, i.e. the override reads and writes back the same value."""
    n_own = k - t - 1
    assert len(own) == n_own

    def dyn(base):                     # base[j] for j <= t, own[j-t-1] for j > t
        return list(base[:t + 1]) + list(own) + [None] * (C.N_ACT - t - 1 - n_own)
    if path == "P1A":                                                    # p1a_gt_free_reference.py:72-85 (imported)
        import p1a_gt_free_reference as P1
        r, g = (dyn(ref), dyn(gt)) if self_override else (ref, gt)
        toks, src = P1.context_for(row, k, t, own, r, g, p)
        return toks, src
    if path in ("H7", "H8"):            # action_history_causal.py:207-215 == prev_action_state_patching.py:288-296
        g = dyn(gt) if self_override else gt
        b = row["name"] if path == "H7" else {"normal": "normal", "gt": "gt_history", "recent_gt": "recent_gt"}[row["base"]]
        if b == "gt_history":
            ctx = list(g[t + 1:k]); src = ["gt"] * len(ctx)
        elif b == "recent_gt":
            ctx = (own[:-1] + [g[k - 1]]) if own else []
            src = (["own"] * (len(own) - 1) + ["gt"]) if own else []
        else:
            ctx, src = list(own), ["own"] * len(own)
        return [p] + ctx, ["pert"] + src
    if path == "H32":                                                    # motion_semantics_ablation.py:234-251
        name = row["name"]
        if name == "normal" or not own:
            return [p] + list(own), ["pert"] + ["own"] * len(own)
        T = _MSA
        prev_own, g_prev = int(own[-1]), int(gt[k - 1])
        if self_override:
            sub = prev_own
        elif name == "recent_gt":
            sub = g_prev
        else:
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{row['_seed']}:{row['_token']}:{row['_pert']}:{k}:rand".encode()).digest()[:4], "little"))
            sub = T["MSA"].motion_sub(name, g_prev, prev_own, T["MOT"], T["trained"], rng)
        return [p] + own[:-1] + [int(sub)], ["pert"] + ["own"] * (len(own) - 1) + ["sub"]
    if path == "P1B":                                                    # p1b_start_position.py:50-68 (imported)
        import p1b_start_position as PB
        g, r = (dyn(gt), dyn(ref)) if self_override else (gt, ref)
        toks, tags = PB.ctx_tokens(row, k, t, p, own, g, r)
        return toks, ["pert"] + tags[1:]
    if path == "H9":                                                     # temporal_feedback_window.py:189-197
        g = dyn(gt) if self_override else gt
        ctx, src = [], []
        for j in range(t + 1, k):
            o = j - t
            use_gt = row["start"] <= o < row["start"] + row["length"]
            ctx.append(g[j] if use_gt else own[o - 1]); src.append("gt" if use_gt else "own")
        return [p] + ctx, ["pert"] + src
    raise ValueError(path)


def exec_acts(path, row, t, own, pred_or_ref, p):
    if path == "P1A":                                                    # p1a_gt_free_reference.py:206-208
        out_t = p if row["out"] == "p" else pred_or_ref[t]
        return list(pred_or_ref[:t]) + [out_t] + list(own)
    return list(pred_or_ref[:t]) + [p] + list(own)                        # e.g. action_history_causal.py:236


# ---------------------------------------------------------------------------------------------- engine
class Engine:
    def __init__(self):
        C.gpu_guard()
        self.torch, self.model, self.llm, self.A0, self.temp, self.agent, self.vis_box, self.cfg = C.load_model()
        self.gpu = C.assert_5090(self.torch)
        self.layers = self.llm.model.layers
        self.hooks = None
        self._stub_of = None

    def env(self):
        t = self.torch
        return {"gpu": self.gpu, "attn_implementation": getattr(self.llm.config, "_attn_implementation", None),
                "dtype": str(next(self.llm.parameters()).dtype), "temperature": self.temp,
                "deterministic_algorithms": t.are_deterministic_algorithms_enabled(),
                "cudnn_deterministic": t.backends.cudnn.deterministic, "cudnn_benchmark": t.backends.cudnn.benchmark,
                "tf32_matmul": t.backends.cuda.matmul.allow_tf32, "bf16_reduced_precision_reduction":
                    t.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "verbatim_sources": verbatim_hashes(), **C.env_record()}

    # -------------------------------------------------------------------- prefix (= p1_common.build_prefix_cache + hashes)
    def build_prefix(self, token, stub_ids, prefix_actions, B):
        torch, llm, A0 = self.torch, self.llm, self.A0
        scene = json.load(open(os.path.join(C.SCENES, f"{token}.json")))
        self.vis_box.clear()
        feats = {}
        for b in self.agent.get_feature_builders():
            feats.update(b.compute_features(scene))
        mi = {k: v.to("cuda:0") for k, v in self.model.get_prompt(feats).items() if isinstance(v, torch.Tensor)}
        stub = stub_ids[:next(j for j, x in enumerate(stub_ids) if x >= A0)]
        prompt = mi["input_ids"][0].tolist()
        prefix = prompt + stub + [A0 + a for a in prefix_actions]
        P = len(prefix)
        ids = torch.tensor([prefix], device="cuda:0")
        llm.rope_deltas = None
        with torch.no_grad():
            out = llm(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values_videos=mi.get("pixel_values_videos"),
                      video_grid_thw=mi.get("video_grid_thw"), use_cache=True, cache_position=torch.arange(P, device="cuda:0"))
        cache = out.past_key_values
        lg_prefix = out.logits[0, -1, A0:A0 + C.N_TOK].double().cpu()
        del out
        if B > 1:
            for li in range(len(cache.key_cache)):
                cache.key_cache[li] = cache.key_cache[li].expand(B, -1, -1, -1).contiguous()
                cache.value_cache[li] = cache.value_cache[li].expand(B, -1, -1, -1).contiguous()
        checksum = float(cache.key_cache[0][0, :, :P].float().abs().sum())
        rd = llm.rope_deltas
        info = {"P": P, "prompt_len": len(prompt), "stub_len": len(stub), "n_prefix_actions": len(prefix_actions),
                "prefix_ids_sha": h(prefix), "pixel_sha": h(mi["pixel_values_videos"].float()) if mi.get("pixel_values_videos") is not None else None,
                "video_grid_thw": mi["video_grid_thw"].tolist() if mi.get("video_grid_thw") is not None else None,
                "prompt_keys": sorted(mi), "second_per_grid_ts": (mi["second_per_grid_ts"].tolist() if mi.get("second_per_grid_ts") is not None else None),
                "rope_deltas": rd.tolist() if rd is not None else None, "prefix_checksum": checksum,
                "lg_prefix_sha": h(lg_prefix.numpy())}
        return cache, P, checksum, info, mi, prefix

    # -------------------------------------------------------------------- hooks
    def install_hooks(self, path, mode):
        """mode: installed | selfpatch | experiment. Returns nothing; self.uninstall_hooks() restores the model."""
        assert self.hooks is None, "hooks already installed"
        torch, llm, A0 = self.torch, self.llm, self.A0
        handles, wrapped = [], []
        if path == "H7":
            # ---- action_history_causal.py:118-155 (verbatim except the C2 'self' branches marked SELF)
            class ST:
                block = None
                neutral = None
                selfemb = None           # SELF: positions whose embedding is read and written back unchanged
                mean_emb = None
                n_blocked_calls = 0
                n_neutral_calls = 0
                n_self_calls = 0
            emb = llm.model.embed_tokens
            ST.mean_emb = emb.weight[A0:A0 + 2048].detach().mean(0)

            def embed_hook(_m, _inp, out):
                if ST.neutral:
                    out = out.clone()
                    out[:, ST.neutral] = ST.mean_emb.to(out.dtype)
                    ST.n_neutral_calls += 1
                    return out
                if ST.selfemb:                                            # SELF
                    out = out.clone()
                    out[:, ST.selfemb] = out[:, ST.selfemb].clone()
                    ST.n_self_calls += 1
                    return out
                return out
            handles.append(emb.register_forward_hook(embed_hook))

            def make_attn(orig):
                def fwd(hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False,
                        use_cache=False, cache_position=None, position_embeddings=None, **kw):
                    q = hidden_states.shape[1]
                    if ST.block and q > 1:
                        if attention_mask is None:
                            raise RuntimeError("expected a 4D causal mask for a multi-token cached forward")
                        am = attention_mask.clone()
                        kv = am.shape[-1]
                        p = kv - q
                        minv = torch.finfo(am.dtype).min
                        rows = torch.arange(q, device=am.device)
                        if ST.block == "hist":
                            tri = rows[:, None] > rows[None, :]
                        elif ST.block == "self":                          # SELF: read the mask, write it back unchanged
                            tri = torch.zeros((q, q), dtype=torch.bool, device=am.device)
                        else:
                            tri = rows[:, None] == rows[None, :] + 1
                        sub = am[..., :, p:p + q]
                        am[..., :, p:p + q] = torch.where(tri, torch.full_like(sub, minv), sub)
                        attention_mask = am
                        ST.n_blocked_calls += 1
                    return orig(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                                past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
                                cache_position=cache_position, position_embeddings=position_embeddings, **kw)
                return fwd
            for layer in self.layers:
                layer.self_attn.forward = make_attn(layer.self_attn.forward)
                wrapped.append(layer.self_attn)
            self.ST = ST
        elif path == "H8":
            # ---- prev_action_state_patching.py:164-213 (verbatim; the plan is chosen by mode)
            class PATCH:
                on = False
                plan = {}
                debug = None
                n_calls = 0
            rows = path_rows("H8")
            idx = {r["name"]: i for i, r in enumerate(rows)}
            src_of = {"gt_history": idx["gt_history"], "normal": idx["normal"], "recent_gt": idx["recent_gt"]}
            plan = {}
            if mode in ("experiment", "selfpatch"):
                for i, r in enumerate(rows):
                    if "layer" in r:
                        s = src_of[r["src"]] if mode == "experiment" else i          # selfpatch: source = the target row itself
                        plan.setdefault(r["layer"], []).append((i, s, r["alpha"]))
            PATCH.plan = {lk: (torch.tensor([t for t, _, _ in v], device="cuda:0"),
                               torch.tensor([s for _, s, _ in v], device="cuda:0"),
                               torch.tensor([a for _, _, a in v], device="cuda:0", dtype=torch.float32)[:, None])
                          for lk, v in plan.items()}

            def apply_patch(hs, lk):
                if not PATCH.on or lk not in PATCH.plan:
                    return hs
                tgt, src, alpha = PATCH.plan[lk]
                h_src = hs[src, -1].float()
                h_tgt = hs[tgt, -1].float()
                new_last = (h_tgt + alpha * (h_src - h_tgt)).to(hs.dtype)
                hs = hs.clone()
                hs[tgt, -1] = new_last
                PATCH.n_calls += 1
                return hs

            def pre_hook_emb(_m, a, kw):
                if a:
                    hs = apply_patch(a[0], "emb")
                    return (hs,) + tuple(a[1:]), kw
                if kw.get("hidden_states") is not None:
                    kw = dict(kw); kw["hidden_states"] = apply_patch(kw["hidden_states"], "emb")
                return a, kw
            handles.append(self.layers[0].register_forward_pre_hook(pre_hook_emb, with_kwargs=True))

            def make_post(i):
                def hook(_m, _a, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    hs2 = apply_patch(hs, f"L{i}")
                    if hs2 is hs:
                        return out
                    return (hs2,) + tuple(out[1:]) if isinstance(out, tuple) else hs2
                return hook
            for i, layer in enumerate(self.layers):
                handles.append(layer.register_forward_hook(make_post(i)))
            self.PATCH = PATCH
        else:
            raise ValueError(f"path {path} has no hooks in its experiment script")
        self.hooks = {"path": path, "mode": mode, "handles": handles, "wrapped": wrapped}

    def uninstall_hooks(self):
        if self.hooks is None:
            return
        for hd in self.hooks["handles"]:
            hd.remove()
        for m in self.hooks["wrapped"]:
            del m.forward                      # instance attribute -> class method restored
        self.hooks = None
        self.ST = self.PATCH = None

    def hooks_clean(self):
        m = self.llm
        n_hooks = sum(len(x._forward_hooks) + len(x._forward_pre_hooks) for x in m.modules())
        n_wrapped = sum("forward" in x.__dict__ for x in self.layers.modules() if hasattr(x, "q_proj"))
        return {"n_module_hooks": n_hooks, "n_wrapped_attn": n_wrapped}

    # -------------------------------------------------------------------- one forward
    def fwd_span(self, cache, P, spans_abs):
        """p1_common.forward_spans (cached-edit path)."""
        return C.forward_spans(self.torch, self.llm, cache, P, spans_abs)

    def fwd_full(self, mi, prefix, span_abs):
        """C4: no cache; the whole sequence from position 0 (B = 1). Returns logits (1, V) float64 and position ids."""
        torch, llm = self.torch, self.llm
        ids = torch.tensor([prefix + span_abs], device="cuda:0")
        llm.rope_deltas = None
        with torch.no_grad():
            pos, _ = llm.get_rope_index(ids, None, mi.get("video_grid_thw"), None, torch.ones_like(ids))
            o = llm(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values_videos=mi.get("pixel_values_videos"),
                    video_grid_thw=mi.get("video_grid_thw"), use_cache=False, cache_position=torch.arange(ids.shape[1], device="cuda:0"))
        lg = o.logits[:, -1].double()
        del o
        return lg, pos[:, 0, len(prefix):].cpu().numpy()

    @staticmethod
    def cached_positions(P, m, rope_delta):
        """position ids the cached span forward uses (modeling_qwen2_5_vl.py:1849-1859): cache_position + rope_deltas,
        identical on the 3 M-RoPE axes."""
        return np.stack([np.arange(P, P + m) + int(rope_delta)] * 3)

    # -------------------------------------------------------------------- decode one unit on one path
    def decode(self, path, unit, ed, stub_ids, *, decoding, seed=0, batch="exp", forward="span", hooks="none",
               self_override=False, force=None, row_names=None):
        """Decode one manifest unit on one path.
        batch   'exp' = the experiment's batch composition (H7: each condition alone, B = 1; H8: 93; H9: 17; P1A: 10)
                '1'   = every row alone (B = 1)
        forward 'span' | 'incremental' | 'deep' | 'full'   (non-span modes always run rows alone, B = 1)
        hooks   'none' | 'installed' | 'selfpatch' | 'experiment'  (H7/H8 only)
        force   {row_name: own tokens} teacher forcing (picks are still recorded); in 'full' mode the H8 emb-patch rows
                are evaluated by token substitution with the source row's forced token (exact for a layer-0 input patch).
        row_names restricts the rows for non-'exp' batches."""
        torch, A0 = self.torch, self.A0
        rows_all = path_rows(path)
        alone = batch == "1" or forward != "span" or path == "H7"
        rows = [r for r in rows_all if row_names is None or r["name"] in row_names] if alone else rows_all
        if path == "H8" and alone and (hooks in ("none", "installed", "selfpatch") or forward != "span"):
            keep = []
            for r in rows:
                if "layer" not in r:
                    keep.append(r)
                elif forward == "full" and r["layer"] == "emb" and r["alpha"] == 1.0 and force is not None:
                    keep.append(r)                                  # token-substitution equivalent
            rows = keep
        token, t = unit["scene"], unit["t_p"]
        gt, pred = ed["gt"], ed["pred"]
        if path == "P1A":
            ref, p = unit["ref_tokens"], unit["p_token"]
            prefix_actions, key, base_seq = ref[:t], "original", ref
        else:
            ref, p = unit.get("ref_tokens") if path == "P1B" else None, unit["forced_token"]
            prefix_actions, key, base_seq = pred[:t], unit["perturbation"], pred
        if path == "H32":
            msa_tables(self.llm, A0)
            rows = [dict(r, _seed=seed, _token=token, _pert=key) for r in rows]
        groups = [[r] for r in rows] if alone else [rows]
        res = {"rows": {}, "path": path, "unit_id": unit["unit_id"], "decoding": decoding, "batch": batch, "forward": forward,
               "hooks": hooks, "self_override": self_override, "teacher_forced": force is not None,
               "B_used": 1 if alone else len(rows)}
        if hooks != "none":
            self.install_hooks(path, hooks)
        try:
            for grp in groups:
                B = len(grp)
                cache, P, chk, info, mi, prefix = self.build_prefix(token, stub_ids, prefix_actions, B)
                rd = info["rope_deltas"][0][0] if isinstance(info["rope_deltas"][0], list) else info["rope_deltas"][0]
                res["prefix"] = info
                if forward == "deep":                               # C3b: invalidate stub + prefix actions, recompute same tokens
                    P_inv = info["prompt_len"]
                    cache.crop(P_inv)
                    with torch.no_grad():
                        self.llm(input_ids=torch.tensor([prefix[P_inv:]], device="cuda:0"),
                                 attention_mask=torch.ones((1, P), device="cuda:0", dtype=torch.long), past_key_values=cache,
                                 use_cache=True, cache_position=torch.arange(P_inv, P, device="cuda:0"))
                    chk = float(cache.key_cache[0][0, :, :P].float().abs().sum())
                    res["deep_invalidated_range"] = [P_inv, P]
                own = [[] for _ in grp]
                st = [{"logits": [], "lsha": [], "argmax": [], "pick": [], "margin": [], "ent": [], "span": [], "src": [],
                       "span_sha": [], "pos_sha": [], "pos_full_eq_cached": [], "override_pos": [], "override_effective": [],
                       "substituted": []} for _ in grp]
                fed = []
                for k in range(t + 1, C.N_ACT):
                    spans, srcs = [], []
                    for i, r in enumerate(grp):
                        toks, src = context(path, r, k, t, own[i], gt, ref, p, self_override)
                        sub = None
                        if forward == "full" and path == "H8" and "layer" in r and len(toks) >= 2:
                            s_name = r["src"]                       # layer-0 input patch of the last position == token substitution
                            s_tok = force[s_name][k - t - 2] if s_name == "normal" else gt[k - 1]
                            sub = (toks[-1], s_tok); toks = toks[:-1] + [s_tok]; src = src[:-1] + [f"patch:{s_name}"]
                        spans.append([A0 + a for a in toks]); srcs.append(src)
                        ov = [t + jj for jj, sname in enumerate(src) if sname not in ("own", "pert")]
                        eff = [j for j in ov if j > t and toks[j - t] != own[i][j - t - 1]]
                        st[i]["override_pos"].append(ov); st[i]["override_effective"].append(eff); st[i]["substituted"].append(sub)
                    m = len(spans[0])
                    pos_eq = [None] * len(grp)
                    pos = [self.cached_positions(P, m, rd)] * len(grp)
                    if forward in ("span", "deep"):
                        if hooks == "selfpatch" and path == "H7":
                            self.ST.block, self.ST.selfemb = "self", list(range(m))
                        if hooks != "none" and path == "H8":
                            self.PATCH.on = True
                        full = self.fwd_span(cache, P, spans)
                        if hooks != "none" and path == "H7":
                            self.ST.block, self.ST.selfemb = None, None
                        if hooks != "none" and path == "H8":
                            self.PATCH.on = False
                        C.check_cache(cache, P, chk)
                    elif forward == "incremental":
                        sp = spans[0]
                        if sp[:len(fed)] != fed:
                            raise RuntimeError("row is not append-only; incremental decoding undefined")
                        new = sp[len(fed):]
                        L = P + len(fed)
                        with torch.no_grad():
                            o = self.llm(input_ids=torch.tensor([new], device="cuda:0"),
                                         attention_mask=torch.ones((1, L + len(new)), device="cuda:0", dtype=torch.long),
                                         past_key_values=cache, use_cache=True,
                                         cache_position=torch.arange(L, L + len(new), device="cuda:0"))
                        full = o.logits[:, -1].double(); del o
                        fed = list(sp)
                    elif forward == "full":
                        full, posf = self.fwd_full(mi, prefix, spans[0])
                        pos_eq = [bool(np.array_equal(posf, pos[0]))]
                        pos = [posf]
                    lg = full[:, A0:A0 + C.N_TOK]
                    lp = torch.log_softmax(lg, -1)
                    ent = -(lp.exp() * lp).sum(-1)
                    srt = torch.sort(lg, -1, descending=True).values
                    am = torch.argmax(lg, -1).tolist()
                    picks = am if decoding == "greedy" else C.sample_rows(torch, lg, self.temp, C.seed_int(seed, token, key, k))
                    lgn = lg.float().cpu().numpy()
                    for i, r in enumerate(grp):
                        nxt = force[r["name"]][k - t - 1] if force is not None else picks[i]
                        own[i].append(int(nxt))
                        s = st[i]
                        s["logits"].append(lgn[i]); s["lsha"].append(h(lgn[i])); s["argmax"].append(int(am[i])); s["pick"].append(int(picks[i]))
                        s["margin"].append(float(srt[i, 0] - srt[i, 1])); s["ent"].append(float(ent[i]))
                        s["span"].append([x - A0 for x in spans[i]]); s["src"].append(srcs[i]); s["span_sha"].append(h(spans[i]))
                        s["pos_sha"].append(h(np.ascontiguousarray(pos[i]).astype(np.int64))); s["pos_full_eq_cached"].append(pos_eq[i])
                if forward == "incremental":
                    cache.crop(P)
                for i, r in enumerate(grp):
                    acts = exec_acts(path, r, t, own[i], base_seq, p)
                    poses, origin = C.decode(torch, self.model, A0, acts)
                    s = st[i]
                    s["logits"] = np.stack(s["logits"]) if s["logits"] else np.zeros((0, C.N_TOK), np.float32)
                    res["rows"][r["name"]] = {"own": own[i], "acts": acts, "poses": poses, "decoder_origin": origin.tolist(),
                                              "attn_mask": f"ones(B={len(grp) if forward in ('span', 'deep') else 1}, P+m)",
                                              "B": len(grp) if forward in ("span", "deep") else 1, **s}
                del cache
                torch.cuda.empty_cache()
        finally:
            if hooks != "none":
                res["hook_calls"] = ({"blocked": self.ST.n_blocked_calls, "neutral": self.ST.n_neutral_calls, "self": self.ST.n_self_calls}
                                     if path == "H7" else {"patch_calls": self.PATCH.n_calls})
                self.uninstall_hooks()
        res["hooks_after"] = self.hooks_clean()
        return res
