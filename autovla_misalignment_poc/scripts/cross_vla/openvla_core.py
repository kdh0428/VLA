"""
OpenVLA-7B (LIBERO fine-tune) core for the cross-VLA replication (outputs/cross_vla_replication).

- Model: openvla/openvla-7b-finetuned-libero-spatial, bf16, the HF modeling code vendored from the openvla repo
  (/root/VLA/openvla/openvla_hf, unchanged). RTX 5090 only (env.sh: CUDA_VISIBLE_DEVICES=1,2 -> CUDA sees only index 1).
- Preprocessing replicates experiments/robot/libero (run_libero_eval.py) without TensorFlow:
  agentview 256x256 rotated 180 deg -> JPEG round trip (quality 95) -> Lanczos resize to 224 -> center crop with
  area 0.9 resized back to 224 by bilinear crop_and_resize (TF semantics) -> prompt "In: What action should the robot
  take to {task}?\\nOut:" + empty token 29871.
- Decoding: greedy over the full vocabulary, 7 action tokens (x, y, z, roll, pitch, yaw, gripper), one token per
  dimension, each conditioned on the previous ones. Token t <-> bin b: b = clip(32000 - t - 1, 0, 254), t = 31999 - b.

Interventions (decode_batch): for each sample an optional (dim d, delta) and a mode
  natural    : no change
  feedback   : executed and context token at d = greedy + delta (later dims see the perturbed token)
  corrected  : executed token at d = greedy + delta, later dims decoded with the ORIGINAL token in context
  reverse    : executed token at d = original greedy, later dims decoded with the PERTURBED token in context
  window_w   : executed token at d = greedy + delta; dims d+1..d+w decoded with the original token in context,
               dims > d+w with the perturbed token in context (KV cache re-written at position d)
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import torch
from PIL import Image

CKPT = "/root/VLA/openvla/openvla-7b-finetuned-libero-spatial"
sys.path.insert(0, "/root/VLA/openvla")
NB = 255            # bin centers
VOCAB = 32000


from openvla_preproc import preprocess, tf_crop_and_resize   # noqa: E402,F401


class OpenVLA:
    def __init__(self, unnorm_key: str = "libero_spatial"):
        from transformers import AutoTokenizer
        from openvla_hf.modeling_prismatic import OpenVLAForActionPrediction
        from openvla_hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        assert torch.cuda.device_count() == 1 and "5090" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
        self.dev = torch.device("cuda:0")
        self.model = OpenVLAForActionPrediction.from_pretrained(
            CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa",
            device_map={"": 0}).eval()                  # straight to the GPU: keeps host RAM (14 GB) free for simulators
        self.proc = PrismaticProcessor(image_processor=PrismaticImageProcessor.from_pretrained(CKPT),
                                       tokenizer=AutoTokenizer.from_pretrained(CKPT))
        st = self.model.norm_stats[unnorm_key]["action"]
        self.lo, self.hi = np.array(st["q01"]), np.array(st["q99"])
        self.mask = np.array(st.get("mask", np.ones(7, bool)))
        self.centers = self.model.bin_centers
        self.lm = self.model.language_model

    # ---- token / action conversion ----
    @staticmethod
    def tok2bin(t):
        return np.clip(VOCAB - np.asarray(t) - 1, 0, NB - 1)

    @staticmethod
    def bin2tok(b):
        return VOCAB - 1 - np.asarray(b)

    def bins2action(self, bins: np.ndarray) -> np.ndarray:
        n = self.centers[np.asarray(bins)]
        return np.where(self.mask, 0.5 * (n + 1) * (self.hi - self.lo) + self.lo, n)

    # ---- model calls ----
    @torch.no_grad()
    def prefix(self, images, task: str):
        prompt = f"In: What action should the robot take to {task.lower()}?\nOut:"
        enc = self.proc([prompt] * len(images), images, return_tensors="pt")
        ids = enc["input_ids"].to(self.dev)
        if not torch.all(ids[:, -1] == 29871):
            ids = torch.cat([ids, torch.full((ids.shape[0], 1), 29871, device=self.dev, dtype=ids.dtype)], dim=1)
        pv = enc["pixel_values"].to(self.dev, dtype=torch.bfloat16)
        out = self.model(input_ids=ids, pixel_values=pv, use_cache=True)
        return out.logits[:, -1].float(), out.past_key_values

    @torch.no_grad()
    def extend(self, cache, tokens: torch.Tensor):
        """Feed tokens (B, k) after `cache`; returns logits (B, k, V) and the new cache."""
        out = self.lm(input_ids=tokens, past_key_values=cache, use_cache=True)
        return out.logits.float(), out.past_key_values

    @staticmethod
    def crop(cache, L):
        return tuple((k[:, :, :L], v[:, :, :L]) for k, v in cache)

    @staticmethod
    def stats(logits):
        """Entropy over the 255 action tokens (renormalised), greedy prob over the full vocab, margin top1-top2."""
        p = torch.softmax(logits, -1)
        pa = p[..., VOCAB - NB:VOCAB]; pa = pa / pa.sum(-1, keepdim=True)
        ent = -(pa * torch.log(pa.clamp_min(1e-12))).sum(-1)
        top = torch.topk(p, 2, dim=-1).values
        return ent.cpu().numpy(), top[..., 0].cpu().numpy(), (top[..., 0] - top[..., 1]).cpu().numpy()

    @torch.no_grad()
    def decode_batch(self, images, task: str, interventions=None):
        """interventions: list (per sample) of None or dict(dim, delta, mode). Returns dict of arrays (B, 7)."""
        iv = interventions or [None] * len(images)
        B = len(iv)
        if len(images) == 1 and B > 1:                  # one observation, several interventions: share the prefix
            logits, cache = self.prefix(images, task)
            logits = logits.expand(B, -1).contiguous()
            cache = tuple((k.expand(B, -1, -1, -1).contiguous(), v.expand(B, -1, -1, -1).contiguous()) for k, v in cache)
        else:
            assert len(images) == B
            logits, cache = self.prefix(images, task)
        L0 = cache[0][0].shape[2]
        exec_b = np.zeros((B, 7), int); ctx_b = np.zeros((B, 7), int); orig_b = np.full((B, 7), -1)
        ent = np.zeros((B, 7)); pmax = np.zeros((B, 7)); marg = np.zeros((B, 7))
        pending = {}                                    # sample -> (perturbed bin, original bin, window end)
        for k in range(7):
            e, pm, mg = self.stats(logits); ent[:, k], pmax[:, k], marg[:, k] = e, pm, mg
            g = self.tok2bin(torch.argmax(logits, -1).cpu().numpy())
            ex, cx = g.copy(), g.copy()
            for i in range(B):
                if iv[i] is None or iv[i]["dim"] != k:
                    continue
                pb = int(np.clip(g[i] + iv[i]["delta"], 0, NB - 1)); orig_b[i, k] = g[i]
                m = iv[i]["mode"]
                if m == "feedback":
                    ex[i] = cx[i] = pb
                elif m == "corrected":
                    ex[i] = pb
                elif m == "reverse":
                    cx[i] = pb
                elif m.startswith("window_"):
                    ex[i] = pb; pending[i] = (pb, k, k + int(m.split("_")[1]))
                else:
                    raise ValueError(m)
            exec_b[:, k], ctx_b[:, k] = ex, cx
            if k == 6:
                break
            logits, cache = self.extend(cache, torch.as_tensor(self.bin2tok(cx), device=self.dev)[:, None])
            logits = logits[:, -1]
            # window: after dims d+1..d+w have been decoded with the original token at d, put the perturbed token
            # back at position d (re-feed d..k with the perturbed token) before decoding dim k+1
            for i, (pb, d, end) in list(pending.items()):
                if k == end:
                    seq = ctx_b[i, d:k + 1].copy(); seq[0] = pb
                    c_i = tuple((kk[i:i + 1, :, :L0 + d], vv[i:i + 1, :, :L0 + d]) for kk, vv in cache)
                    lg, c_i = self.extend(c_i, torch.as_tensor(self.bin2tok(seq), device=self.dev)[None])
                    logits[i] = lg[0, -1]
                    cache = tuple((torch.cat([kk[:i], ck, kk[i + 1:]]), torch.cat([vv[:i], cv, vv[i + 1:]]))
                                  for (kk, vv), (ck, cv) in zip(cache, c_i))
                    del pending[i]
        return {"exec_bins": exec_b, "ctx_bins": ctx_b, "orig_bins": orig_b, "entropy": ent, "pmax": pmax, "margin": marg,
                "action": np.stack([self.bins2action(b) for b in exec_b])}


def to_env_action(a: np.ndarray) -> list:
    """normalize_gripper_action(binarize=True) then invert_gripper_action, as run_libero_eval.py."""
    a = a.copy()
    a[-1] = 2 * (a[-1] - 0.0) / (1.0 - 0.0) - 1
    a[-1] = np.sign(a[-1])
    a[-1] = -a[-1]
    return a.tolist()
