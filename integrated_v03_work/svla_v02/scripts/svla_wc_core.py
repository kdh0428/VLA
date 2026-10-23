"""SpatialVLA within-call intervention core (2026-10-22 v01).

Official path: SpatialVLAForConditionalGeneration.predict_action = generate(**inputs.to(bf16), max_new_tokens=256,
do_sample=False) with the checkpoint's generation_config (cache_implementation="hybrid").

Adapter path (used for every arm, including the natural reference R): it reproduces generate() step by step.
1. Prefill the prompt once in a single forward with a HybridCache of max_cache_len = P + 256 (= generate's max_length).
   The model's _update_causal_mask builds the official prompt mask (bidirectional inside the prompt).
2. Feed action tokens one per forward through the cache, with a 2-D all-ones attention mask of the current length and
   cache_position = [pos]. Position ids are then cache_position + 1, as in generate. Forced (conditioning) tokens use
   this same path, so they are causal action tokens, never part of the bidirectional prompt.
Batch size is always 1, as in predict_action.

Action layout (1-based actions a1..a4, each 3 tokens): translation (ids 257153-261248), rotation (261249-265344),
gripper (265345-265346). Zero-based token index of action k, field f = 3*(k-1) + f.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from PIL import Image

CKPT = "/root/VLA/spatialvla/spatialvla-4b-224-sft-fractal"
UNNORM = "fractal20220817_data/0.1.0"
MAX_NEW = 256
EOS = 1


def seed_of(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "little")


class SVLA:
    def __init__(self):
        from transformers import AutoModel, AutoProcessor
        assert torch.cuda.device_count() == 1 and "5090" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
        self.proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
        self.vla = AutoModel.from_pretrained(CKPT, torch_dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()
        T = self.proc.action_tokenizer
        self.T = T
        self.ranges = [(T.translation_tokenizer.token_start_idx, T.translation_tokenizer.token_end_idx),
                       (T.rotation_tokenizer.token_start_idx, T.rotation_tokenizer.token_end_idx),
                       (T.gripper_tokenizer.token_start_idx, T.gripper_tokenizer.token_end_idx)]

    # ---------- inputs ----------
    def inputs(self, image: np.ndarray, instruction: str):
        """Official SimplerEnv adapter call, then predict_action's cast (.to(bf16).to(device))."""
        inp = self.proc(images=[Image.fromarray(image)], text=instruction, unnorm_key=UNNORM, return_tensors="pt",
                        do_normalize=False)
        return inp.to(torch.bfloat16).to(self.vla.device)

    # ---------- official ----------
    def _fresh_cache(self):
        """A3: generate() keeps self._cache and reuses it whenever it is at least as long as needed, so a call after a
        longer prompt ran with a larger HybridCache (different attention shapes, different bf16 numerics). Deleting it
        makes every official call build its own P+256 cache, exactly as the first call of a fresh process."""
        if hasattr(self.vla, "_cache"):
            del self.vla._cache

    @torch.no_grad()
    def official(self, inp, sample=None, seed=None):
        """predict_action-equivalent generate; also returns per-step logits (output_logits does not change the computation).
        sample=None -> greedy (exactly predict_action's arguments)."""
        kw = dict(max_new_tokens=MAX_NEW, do_sample=False)
        if sample is not None:
            kw = dict(max_new_tokens=MAX_NEW, do_sample=True, **sample)
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        L = inp["input_ids"].shape[-1]
        self._fresh_cache()
        out = self.vla.generate(**inp, **kw, output_logits=True, return_dict_in_generate=True)
        ids = out.sequences[0, L:].tolist()
        logits = torch.stack(out.logits, 0)[:, 0].float().cpu() if sample is None else None   # (steps, V)
        return ids, logits

    @torch.no_grad()
    def predict_action_ids(self, inp):
        """The model's own predict_action, unchanged (fresh generate cache, A3)."""
        self._fresh_cache()
        return self.vla.predict_action(inp)[0].tolist()

    # ---------- adapter ----------
    @torch.no_grad()
    def _prefill(self, inp):
        from transformers import HybridCache
        P = inp["input_ids"].shape[1]
        cache = HybridCache(config=self.vla.config.text_config, max_batch_size=1, max_cache_len=P + MAX_NEW,
                            device=self.vla.device, dtype=torch.bfloat16)
        out = self.vla(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], pixel_values=inp["pixel_values"],
                       intrinsic=inp["intrinsic"], use_cache=True, past_key_values=cache,
                       cache_position=torch.arange(P, device=self.vla.device))
        return cache, out.logits[:, -1].float(), P

    @torch.no_grad()
    def _feed(self, cache, tok, pos):
        out = self.vla(input_ids=torch.tensor([[int(tok)]], device=self.vla.device), past_key_values=cache, use_cache=True,
                       attention_mask=torch.ones(1, pos + 1, device=self.vla.device, dtype=torch.long),
                       cache_position=torch.tensor([pos], device=self.vla.device))
        return out.logits[:, -1].float()

    @torch.no_grad()
    def forced(self, inp, cond_prefix, n_gen=7):
        """Feed cond_prefix (conditioning token ids) one by one, then greedy-decode n_gen tokens (6 continuation tokens
        + 1 termination token). Returns generated ids and their logits (n_gen, V) on CPU."""
        cache, logits, pos = self._prefill(inp)
        for t in cond_prefix:
            logits = self._feed(cache, t, pos); pos += 1
        gen, lg = [], []
        for k in range(n_gen):
            lg.append(logits[0].cpu()); nxt = int(torch.argmax(logits, -1))
            gen.append(nxt)
            if k < n_gen - 1:
                logits = self._feed(cache, nxt, pos); pos += 1
        del cache
        return gen, torch.stack(lg, 0)

    # ---------- action helpers ----------
    def type_valid(self, ids12):
        """per-token validity of action type at its slot (translation, rotation, gripper)."""
        return [self.ranges[j % 3][0] <= int(t) <= self.ranges[j % 3][1] for j, t in enumerate(ids12)]

    def u_norm(self, ids12):
        """(4, 7) normalized actions before unnormalization (official action tokenizer); [:, :6] = motion u, [:, 6] gripper."""
        return self.T.decode_token_ids_to_actions(np.asarray(ids12, dtype=np.int64).reshape(-1, 3))

    def unnorm(self, ids12):
        g = torch.as_tensor(np.asarray(ids12, dtype=np.int64)).reshape(1, -1)
        return self.proc.decode_actions(g, unnorm_key=UNNORM)["actions"]


def margin_top2(lg):
    v, _ = torch.topk(lg, 2, dim=-1)
    return (v[:, 0] - v[:, 1]).tolist()
