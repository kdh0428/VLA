"""
SpatialVLA-4B (PaliGemma2 / Gemma2 backbone; IPEC-COMMUNITY/spatialvla-4b-224-sft-fractal) for the cross-domain temporal
replication (outputs/cross_domain_temporal_replication). RTX 5090 only; transformers 4.47 (as required by the model code).

Action chunk = 4 future steps; every step is 3 tokens (translation = one of 32 x 16 x 8 spherical (phi, theta, r) bins,
rotation = one of 16^3 (roll, pitch, yaw) bins, gripper = 2 bins), generated greedily left to right after the
image + prompt prefix: step k's tokens are conditioned on the tokens of steps < k (causal suffix).

The model's inference mask makes the whole *input* bidirectional, so forced action tokens must not be fed together with
the prompt. decode_steps() therefore reproduces generate(): prefill the prompt once, then feed action tokens one by one
through the cache. Interventions are expressed per step: for step k the caller supplies the context steps 1..k-1
(3 tokens each), the step's 3 tokens are decoded greedily and returned.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

CKPT = "/root/VLA/spatialvla/spatialvla-4b-224-sft-fractal"
UNNORM = "fractal20220817_data/0.1.0"


class SpatialVLA:
    def __init__(self):
        from transformers import AutoModel, AutoProcessor
        assert torch.cuda.device_count() == 1 and "5090" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
        self.proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
        self.vla = AutoModel.from_pretrained(CKPT, torch_dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()
        self.tok = self.proc.action_tokenizer

    def inputs(self, image: np.ndarray, instruction: str):
        return self.proc(images=[Image.fromarray(image)], text=instruction, unnorm_key=UNNORM, return_tensors="pt",
                         do_normalize=False)

    @torch.no_grad()
    def generate_reference(self, image, instruction):
        """The model's own predict_action (generate, greedy): (4, 3) token ids and (4, 7) actions."""
        g = self.vla.predict_action(self.inputs(image, instruction))
        out = self.proc.decode_actions(g, unnorm_key=UNNORM)
        return out["action_ids"], out["actions"]

    @torch.no_grad()
    def _prefill(self, inp, B):
        """Prefill the prompt once (batch 1, vision encoders run once) and copy the cache to B rows."""
        from transformers import HybridCache
        ids = inp["input_ids"].cuda(); am = inp["attention_mask"].cuda()
        pv = inp["pixel_values"].cuda().to(torch.bfloat16)
        P = ids.shape[1]
        cfg = self.vla.config.text_config if hasattr(self.vla.config, "text_config") else self.vla.config
        c1 = HybridCache(config=cfg, max_batch_size=1, max_cache_len=P + 16, device="cuda", dtype=torch.bfloat16)   # as generate()
        out = self.vla(input_ids=ids, attention_mask=am, pixel_values=pv, intrinsic=inp["intrinsic"].cuda(), use_cache=True,
                       past_key_values=c1, cache_position=torch.arange(P, device="cuda"))
        if B == 1:
            return c1, out.logits[:, -1].float(), P
        cb = HybridCache(config=cfg, max_batch_size=B, max_cache_len=P + 16, device="cuda", dtype=torch.bfloat16)
        for li in range(len(c1.key_cache)):
            cb.key_cache[li].copy_(c1.key_cache[li].expand_as(cb.key_cache[li]))
            cb.value_cache[li].copy_(c1.value_cache[li].expand_as(cb.value_cache[li]))
        del c1
        return cb, out.logits[:, -1].float().expand(B, -1), P

    @torch.no_grad()
    def _feed(self, cache, tokens, pos, B):
        ids = torch.as_tensor(tokens, device="cuda").view(B, 1)
        out = self.vla(input_ids=ids, past_key_values=cache, use_cache=True,
                       attention_mask=torch.ones(B, pos + 1, device="cuda", dtype=torch.long),
                       cache_position=torch.tensor([pos], device="cuda"))
        return out.logits[:, -1].float()

    @torch.no_grad()
    def decode_step(self, inp, contexts):
        """contexts: list (rows) of lists of previous steps, each step = 3 token ids. All rows must have the same number
        of context steps. Returns (rows, 3) greedy token ids of the next step and the logits' entropy of its first token."""
        B = len(contexts); n_prev = len(contexts[0])
        assert all(len(c) == n_prev for c in contexts)
        cache, logits, P = self._prefill(inp, B)
        pos = P
        for s in range(n_prev):
            for j in range(3):
                logits = self._feed(cache, [c[s][j] for c in contexts], pos, B); pos += 1
        step, ent = [], None
        for j in range(3):
            nxt = torch.argmax(logits, -1)
            if j == 0:
                p = torch.softmax(logits, -1); ent = (-(p * torch.log(p.clamp_min(1e-12))).sum(-1)).cpu().numpy()
            step.append(nxt.cpu().numpy())
            if j < 2:
                logits = self._feed(cache, nxt.tolist(), pos, B); pos += 1
        del cache
        return np.stack(step, 1), ent

    def decode_chunk(self, inp, n_rows=1):
        """Free greedy decoding of the 4 steps through the step-wise path (natural reference)."""
        ctx = [[] for _ in range(n_rows)]
        for _ in range(4):
            st, _ = self.decode_step(inp, ctx)
            ctx = [c + [list(map(int, st[i]))] for i, c in enumerate(ctx)]
        return np.array(ctx)

    def tokens_to_actions(self, ids: np.ndarray) -> np.ndarray:
        """(n, 3) token ids -> (n, 7) unnormalised actions, exactly as processor.decode_actions."""
        g = torch.as_tensor(np.asarray(ids).reshape(1, -1))
        return self.proc.decode_actions(g, unnorm_key=UNNORM)["actions"]


def batch_inputs(m: SpatialVLA, images, instructions):
    """Processor outputs for several observations, grouped by prompt length (no padding needed inside a group)."""
    inps = [m.inputs(im, ins) for im, ins in zip(images, instructions)]
    groups = {}
    for i, inp in enumerate(inps):
        groups.setdefault(int(inp["input_ids"].shape[1]), []).append(i)
    return inps, groups


@torch.no_grad()
def decode_chunks(m: SpatialVLA, inps, rows, override):
    """
    One greedy pass per chunk for several rows (row r uses observation inps[rows[r]]), prompt prefilled once.
    override(r, g) is called when row r reaches the step-1 translation token with its greedy value g and returns
    (exec_token, context_token) — None keeps (g, g). Returns exec tokens (R, 4, 3), context tokens (R, 4, 3) and the natural
    step-1 translation tokens g (R,).
    """
    from transformers import HybridCache
    R = len(rows)
    exe = np.zeros((R, 4, 3), dtype=np.int64); ctx = np.zeros((R, 4, 3), dtype=np.int64); nat1 = np.zeros(R, dtype=np.int64)
    by_len = {}
    for r, i in enumerate(rows):
        by_len.setdefault(int(inps[i]["input_ids"].shape[1]), []).append(r)
    for P, rs in by_len.items():
        B = len(rs)
        if len({rows[r] for r in rs}) == 1:                 # one observation, several rows: prefill once
            cache, logits, P = m._prefill(inps[rows[rs[0]]], B); logits = logits.clone(); pos = P
            exe_g, ctx_g, nat_g = _decode_loop(m, cache, logits, pos, rs, override, B)
            exe[rs], ctx[rs], nat1[rs] = exe_g, ctx_g, nat_g
            continue
        ids = torch.cat([inps[rows[r]]["input_ids"] for r in rs]).cuda()
        am = torch.cat([inps[rows[r]]["attention_mask"] for r in rs]).cuda()
        pv = torch.cat([inps[rows[r]]["pixel_values"] for r in rs]).cuda().to(torch.bfloat16)
        cfg = m.vla.config.text_config if hasattr(m.vla.config, "text_config") else m.vla.config
        cache = HybridCache(config=cfg, max_batch_size=B, max_cache_len=P + 16, device="cuda", dtype=torch.bfloat16)
        out = m.vla(input_ids=ids, attention_mask=am, pixel_values=pv, intrinsic=inps[rows[rs[0]]]["intrinsic"].cuda(),
                    use_cache=True, past_key_values=cache, cache_position=torch.arange(P, device="cuda"))
        logits = out.logits[:, -1].float(); pos = P
        exe_g, ctx_g, nat_g = _decode_loop(m, cache, logits, pos, rs, override, B)
        exe[rs], ctx[rs], nat1[rs] = exe_g, ctx_g, nat_g
        del cache
    return exe, ctx, nat1


@torch.no_grad()
def _decode_loop(m, cache, logits, pos, rs, override, B):
    exe = np.zeros((B, 4, 3), dtype=np.int64); ctx = np.zeros((B, 4, 3), dtype=np.int64); nat1 = np.zeros(B, dtype=np.int64)
    if True:
        for k in range(12):
            g = torch.argmax(logits, -1).cpu().numpy()
            e, c = g.copy(), g.copy()
            if k == 0:
                for b, r in enumerate(rs):
                    nat1[b] = g[b]
                    o = override(r, int(g[b]))
                    if o is not None:
                        e[b], c[b] = o
            exe[:, k // 3, k % 3] = e; ctx[:, k // 3, k % 3] = c
            if k == 11:
                break
            o2 = m.vla(input_ids=torch.as_tensor(c, device="cuda").view(B, 1), past_key_values=cache, use_cache=True,
                       attention_mask=torch.ones(B, pos + 1, device="cuda", dtype=torch.long),
                       cache_position=torch.tensor([pos], device="cuda"))
            logits = o2.logits[:, -1].float(); pos += 1
    return exe, ctx, nat1


def translation_table(m: SpatialVLA):
    """Every translation token (4,096 = theta 16 x phi 32 x r 8 bins): token ids, normalised xyz (model space, decoded
    exactly as the tokenizer), unnormalised xyz (as processor.decode_actions), and the (theta, phi, r) bin indices."""
    tt = m.tok.translation_tokenizer
    ids = np.arange(tt.token_start_idx, tt.token_end_idx + 1)
    norm = tt.decode_token_ids_to_actions(ids)
    st = m.proc.statistics[UNNORM]["action"]
    lo, hi = np.array(st["q01"])[:3], np.array(st["q99"])[:3]
    phys = 0.5 * (norm + 1) * (hi - lo) + lo
    k = ids - tt.token_start_idx
    bins = np.stack([k // tt.NP, (k % tt.NP) // tt.num_r_bins, k % tt.num_r_bins], 1)     # theta, phi, r
    return {"ids": ids, "norm": norm, "phys": phys, "bins": bins, "id2row": {int(t): j for j, t in enumerate(ids)}}
