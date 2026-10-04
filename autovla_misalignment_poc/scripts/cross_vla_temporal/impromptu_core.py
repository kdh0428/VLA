"""
Impromptu VLA (Qwen2.5-VL-3B, "3B Base+Impromptu" = aaaaaap/ImpromptuVLAModel/3B_AD) on NAVSIM navtest scenes, for the
cross-VLA temporal replication (outputs/cross_vla_temporal_replication). RTX 5090 only.

Prompt = the NAVSIM trajectory question of the Impromptu data engine (data_qa_generate/.../navsim/q7_navsim.py), rebuilt
from the navsim logs: front / front-right / front-left camera of the current frame (CAM_F0, CAM_R0, CAM_L0), ego statuses
at -1.5 / -1.0 / -0.5 s (position in the current ego frame, acceleration, velocity), LLaMA-Factory `qwen2_vl` template,
images regularised to <= 262,144 pixels as in the released training / inference configs.
Answer = 10 future waypoints (0.5 s apart, 5 s) as text "[x, y]" with 2 decimals, generated left to right: every
waypoint's digit tokens are conditioned on all earlier waypoints (causal LM).

Decoding: greedy with the checkpoint's generation_config repetition_penalty 1.05 (its top_k = 1 makes sampling greedy).
Interventions are applied by writing an assistant-side prefix (earlier waypoints as text) and letting the model continue.
"""
from __future__ import annotations

import math
import os
import pickle
import re

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

CKPT = "/root/VLA/impromptu/3B_AD"
LOGS = "/root/VLA/autovla/dataset/nuplan/navsim_logs/test"
SENS = "/root/VLA/autovla/dataset/nuplan/sensor_blobs/test"
MAX_PIXELS = 262144
ANSWER_HEAD = ("<PLANNING>Predicted future movement details for the next 5 seconds (sampled at 0.5-second intervals), "
               "including BEV location in x and y directions (in meters). Positive x means forward direction while positive "
               "y means leftwards. The output is formatted as [x, y]:")       # data-engine "x-y"; ": [" tokenises as ":" + " ["
WP_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")
_LOG_CACHE: dict = {}


def fmt_wp(p) -> str:
    """Waypoint text exactly as the training answers: python round(., 2), no -0.0."""
    x, y = round(float(p[0]), 2), round(float(p[1]), 2)
    x = 0.0 if x == -0.0 else x; y = 0.0 if y == -0.0 else y
    return f"[{x}, {y}]"


def prefix_text(wps) -> str:
    """Assistant-side text after ANSWER_HEAD holding the given waypoints, ready for the next one."""
    return "".join(" " + fmt_wp(w) + "," for w in wps)    # training text ": [x, y], [x, y], ...": the next token is " ["


def parse_wps(text: str):
    return [(float(a), float(b)) for a, b in WP_RE.findall(text)]


def _frames(log):
    if log not in _LOG_CACHE:
        _LOG_CACHE.clear()
        _LOG_CACHE[log] = pickle.load(open(os.path.join(LOGS, f"{log}.pkl"), "rb"))
    return _LOG_CACHE[log]


def _pose(f):
    t = f["ego2global_translation"]
    return t[0], t[1], Quaternion(*f["ego2global_rotation"]).yaw_pitch_roll[0]


def _to_ego(gx, gy, ex, ey, yaw):
    dx, dy = gx - ex, gy - ey; c, s = math.cos(yaw), math.sin(yaw)
    return dx * c + dy * s, -dx * s + dy * c


def scene_inputs(log: str, token: str):
    """Question text, the 3 images (front, front-right, front-left) and the logged 5 s future (10 x 2, ego frame)."""
    fr = _frames(log)
    i = next(k for k, f in enumerate(fr) if f["token"] == token)
    ex, ey, eyaw = _pose(fr[i])
    # data-engine format (loaders/pipelines: metadata "3v" + ego_status "x-y"): 4 statuses t-1.5 .. t-0.0, python round()
    st = []
    for k, j in enumerate((i - 3, i - 2, i - 1, i)):
        gx, gy, _ = _pose(fr[j]); x, y = _to_ego(gx, gy, ex, ey, eyaw)
        x, y = round(float(x), 2), round(float(y), 2)
        x = 0.0 if x == -0.0 else x; y = 0.0 if y == -0.0 else y
        vx, vy, ax, ay = (round(float(v), 2) for v in fr[j]["ego_dynamic_state"])
        st.append(f"(t-{1.5 - k * 0.5}s) [{x}, {y}], Acceleration: X {ax}, Y {ay} m/s^2, Velocity: X {vx}, Y {vy} m/s")
    q = ("You are an autonomous driving agent. You have access to multi-view camera images of a vehicle: "
         "(1) front view (which you should focus on with the most attention) <image>, (2) front right view <image>, "
         "and (3) front left view <image>. Your task is to do your best to predict future waypoints for the vehicle "
         "over the next 10 timesteps, given the vehicle's intent inferred from the images."
         "Provided are the previous ego vehicle statuses recorded over the last 1.5 seconds (at 0.5-second intervals). "
         "This includes the x and y coordinates of the ego vehicle. Positive x means forward direction while positive y "
         "means leftwards. The data is presented in the format [x, y]:\n" + ", ".join(st) + "\n")
    imgs = []
    for cam in ("CAM_F0", "CAM_R0", "CAM_L0"):
        im = Image.open(os.path.join(SENS, fr[i]["cams"][cam]["data_path"])).convert("RGB")
        if im.width * im.height > MAX_PIXELS:                       # LLaMA-Factory _regularize_images
            r = math.sqrt(MAX_PIXELS / (im.width * im.height))
            im = im.resize((int(im.width * r), int(im.height * r)))
        imgs.append(im)
    fut = []
    for j in range(i + 1, i + 11):
        gx, gy, _ = _pose(fr[j]); fut.append(_to_ego(gx, gy, ex, ey, eyaw))
    return q, imgs, np.array(fut)


class Impromptu:
    def __init__(self):
        from transformers import Qwen2_5_VLForConditionalGeneration
        assert torch.cuda.device_count() == 1 and "5090" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            CKPT, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}).eval()
        from transformers import AutoTokenizer, Qwen2VLImageProcessor, Qwen2_5_VLProcessor
        # transformers 4.49 has no "Qwen2_5_VLImageProcessor" name; Qwen2.5-VL uses the Qwen2-VL image processor class
        self.proc = Qwen2_5_VLProcessor(image_processor=Qwen2VLImageProcessor.from_pretrained(CKPT),
                                        tokenizer=AutoTokenizer.from_pretrained(CKPT))
        self.proc.tokenizer.padding_side = "left"
        # every row of a batch carries the same 3 images: encode them once and tile the visual tokens per row (exact; the
        # vision encoder attends within each image). Also avoids the full-batch vision attention mask (OOM).
        vis = self.model.visual; orig = vis.forward

        def vis_forward(pixel_values, grid_thw=None, **kw):
            n = grid_thw.shape[0]
            if n > 3 and n % 3 == 0 and torch.equal(grid_thw[:3].repeat(n // 3, 1), grid_thw):
                first = int((grid_thw[:3, 0] * grid_thw[:3, 1] * grid_thw[:3, 2]).sum())
                return orig(pixel_values[:first], grid_thw=grid_thw[:3], **kw).repeat(n // 3, 1)
            return orig(pixel_values, grid_thw=grid_thw, **kw)
        vis.forward = vis_forward

    def _text(self, q: str, assistant_prefix: str) -> str:
        q = q.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        return (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n"
                f"<|im_start|>assistant\n{ANSWER_HEAD}{assistant_prefix}")

    @torch.no_grad()
    def continue_batch(self, q, imgs, prefixes, one_waypoint=False, chunk=24):
        """Greedy continuation of each assistant prefix. one_waypoint: stop each row after its next ']'. Returns texts."""
        outs = []
        for c0 in range(0, len(prefixes), chunk):
            texts = [self._text(q, p) for p in prefixes[c0:c0 + chunk]]
            enc = self.proc(text=texts, images=[im for _ in texts for im in imgs], return_tensors="pt", padding=True).to("cuda:0")
            kw = dict(max_new_tokens=20, stop_strings=["]"], tokenizer=self.proc.tokenizer) if one_waypoint else dict(max_new_tokens=200)
            out = self.model.generate(**enc, do_sample=False, repetition_penalty=1.05,
                                      pad_token_id=self.proc.tokenizer.pad_token_id, **kw)
            outs += self.proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return outs


class FastWaypointDecoder:
    """
    One-waypoint greedy continuation with a per-scene prompt KV cache (same idea as the AutoVLA harness): the prompt
    (images + question + ANSWER_HEAD) is encoded once; for each row only its assistant suffix is fed, then tokens are
    decoded until ']' (max 20). Rows are grouped by suffix length so no padding is needed. Repetition penalty 1.05 is
    applied exactly like transformers' RepetitionPenaltyLogitsProcessor over all ids of the row (prompt + suffix + generated).
    """

    def __init__(self, imp: Impromptu, penalty: float = 1.05):
        self.m, self.tok, self.pen = imp, imp.proc.tokenizer, penalty
        self.close_ids = {i for t, i in self.tok.get_vocab().items() if "]" in self.tok.convert_tokens_to_string([t])}

    @torch.no_grad()
    def set_scene(self, q, imgs):
        text = self.m._text(q, "")
        enc = self.m.proc(text=[text], images=imgs, return_tensors="pt").to("cuda:0")
        self.m.model.rope_deltas = None
        out = self.m.model(**enc, use_cache=True)
        self.cache, self.P = out.past_key_values, enc["input_ids"].shape[1]
        self.prompt_ids = enc["input_ids"][0]
        self.last_logits = out.logits[0, -1].float()

    def _expand(self, B):
        from transformers import DynamicCache
        c = DynamicCache()
        for li in range(len(self.cache.key_cache)):
            c.update(self.cache.key_cache[li].expand(B, -1, -1, -1).contiguous(),
                     self.cache.value_cache[li].expand(B, -1, -1, -1).contiguous(), li)
        return c

    def _penalise(self, logits, seen):
        # seen: (B, V) bool
        s = logits.clone()
        pos = s > 0
        s = torch.where(seen & pos, s / self.pen, torch.where(seen & ~pos, s * self.pen, s))
        return s

    @torch.no_grad()
    def next_waypoint(self, suffixes, chunk=32):
        """suffixes: assistant texts after ANSWER_HEAD. Returns generated text per row (up to and including ']')."""
        res = [None] * len(suffixes)
        ids = [self.tok(s, add_special_tokens=False)["input_ids"] if s else [] for s in suffixes]
        V = self.last_logits.shape[0]
        groups = {}
        for i, x in enumerate(ids):
            groups.setdefault(len(x), []).append(i)
        for L, members in groups.items():
            for c0 in range(0, len(members), chunk):
                mem = members[c0:c0 + chunk]; B = len(mem)
                cache = self._expand(B)
                seen = torch.zeros(B, V, dtype=torch.bool, device="cuda:0")
                seen[:, self.prompt_ids] = True
                if L:
                    suf = torch.tensor([ids[i] for i in mem], device="cuda:0")
                    seen.scatter_(1, suf, True)
                    out = self.m.model(input_ids=suf, past_key_values=cache, use_cache=True,
                                       attention_mask=torch.ones(B, self.P + L, device="cuda:0", dtype=torch.long),
                                       cache_position=torch.arange(self.P, self.P + L, device="cuda:0"))
                    logits = out.logits[:, -1].float()
                else:
                    logits = self.last_logits.expand(B, -1)
                gen = [[] for _ in range(B)]; done = torch.zeros(B, dtype=torch.bool, device="cuda:0")
                cur = self.P + L
                for _ in range(20):
                    nxt = torch.argmax(self._penalise(logits, seen), -1)
                    for b in range(B):
                        if not done[b]:
                            gen[b].append(int(nxt[b]))
                            if int(nxt[b]) in self.close_ids:
                                done[b] = True
                    if bool(done.all()):
                        break
                    seen.scatter_(1, nxt[:, None], True)
                    out = self.m.model(input_ids=nxt[:, None], past_key_values=cache, use_cache=True,
                                       attention_mask=torch.ones(B, cur + 1, device="cuda:0", dtype=torch.long),
                                       cache_position=torch.arange(cur, cur + 1, device="cuda:0"))
                    logits = out.logits[:, -1].float(); cur += 1
                for b, i in enumerate(mem):
                    res[i] = self.tok.decode(gen[b])
                del cache
        return res
