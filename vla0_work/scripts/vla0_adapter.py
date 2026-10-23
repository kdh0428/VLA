#!/usr/bin/env python
"""VLA-0 adapter (NVlabs/vla0 @ b78db19, HF export ankgoyal/vla0-libero@6c62a51 model_last/).

Builds the official `QwenActor` wrapper WITHOUT its constructor's base-model download (the constructor would first load
Qwen/Qwen2.5-VL-3B-Instruct and then overwrite it); instead the identical attributes are set from config.yaml and the
fine-tuned HF export is loaded directly (bf16, B = 1, sdpa). Native preprocessing is reproduced with the repository's own
functions (`image_unifier_transform`, `QwenActor.get_imgs/get_qwen_inputs/forward`).
"""
import os
import pickle
import sys
import types

import numpy as np
import torch
import yaml

REPO = "/root/VLA/vla0_work/vla0-b78db19eda67b02844fe82b8929e1f94de7f2dcc"
HF = "/root/VLA/vla0_work/hf_6c62a51"
WEIGHTS = "/dev/shm/vla0/model_last"
EOS = 151645

# rv_train.utils.train_utils imports tensorboard tooling only needed for training; stub it (ForkedPdb is a debugger alias)
_stub = types.ModuleType("rv_train.utils.train_utils")
_stub.ForkedPdb = object
sys.modules["rv_train.utils.train_utils"] = _stub
for p in (REPO, os.path.join(REPO, "libs/RoboVerse")):
    if p not in sys.path:
        sys.path.insert(0, p)
from einops import rearrange  # noqa: E402
from torch import nn  # noqa: E402
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration  # noqa: E402

from rv_train.models.qwen.model import QwenActor, NumberSpaceOnlyProcessor  # noqa: E402
# import the native image unifier without executing roboverse/__init__.py (it only re-exports the training dataset builder,
# which needs yacs/lerobot); the module code itself is unchanged
_rv = types.ModuleType("roboverse"); _rv.__path__ = [os.path.join(REPO, "libs/RoboVerse/roboverse")]
sys.modules.setdefault("roboverse", _rv)
from roboverse.unifiers.image_unifier import image_unifier_transform  # noqa: E402


def load_cfg():
    cfg = yaml.safe_load(open(os.path.join(HF, "config.yaml")))
    opts = cfg["DATALOADER"]["ROBOVERSE"]["cfg_opts"].split(":")
    kv = dict(zip(opts[0::2], opts[1::2]))
    return cfg, kv


def build_actor(device="cuda:0"):
    cfg, kv = load_cfg()
    q = cfg["MODEL"]["QWEN"]
    a = QwenActor.__new__(QwenActor)
    nn.Module.__init__(a)
    a.load_param_before_ddp = True
    a.qwen_model_id = WEIGHTS
    a.action_type = q["action_type"]
    a.original_action_dim = q["original_action_dim"]
    a.horizon = q["horizon"]
    a.history = q["history"]
    a.use_lora = q["use_lora"]
    a.use_qlora = q["use_qlora"]
    a.num_cam = q["num_cam"]
    a.lora_config = q["lora_config"]
    a.lora_rank = q["lora_rank"]
    a.rgb_input = q["rgb_input"]
    a.rgb_img_size = tuple(q["rgb_img_size"])
    a.add_vision_id = q["add_vision_id"]
    a.tiled_rgb_imgs = q["tiled_rgb_imgs"]
    a.num_bins_actions = q["num_bins_actions"]
    a.use_flash_attention_2 = q["use_flash_attention_2"]
    a.action_mask_aug_per = q["action_mask_aug_per"]
    a.attention_dropout = q["attention_dropout"]
    assert not a.use_lora and not a.use_qlora and a.action_type == "original"
    a.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(WEIGHTS, torch_dtype=torch.bfloat16,
                                                                 device_map={"": device})
    a.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    a.min_pixel = a.max_pixel = a.rgb_img_size[0] * a.rgb_img_size[1]
    if a.rgb_input and a.tiled_rgb_imgs:
        a.min_pixel *= a.history * a.num_cam
        a.max_pixel *= a.history * a.num_cam
    a.processor = QwenActor.load_qwen_model_processor(qwen_model_id=WEIGHTS, min_pixel=a.min_pixel, max_pixel=a.max_pixel,
                                                      padding_side=None)
    a.logits_processor = NumberSpaceOnlyProcessor(a.processor.tokenizer)
    a.act_dim = a.original_action_dim
    a.system_message = (f"Analyze the input image and predict robot actions for the next {a.horizon} timesteps. Each action has "
                        f"{a.act_dim} dimensions. Output a single sequence of {a.horizon * a.act_dim} integers (0-{a.num_bins_actions} "
                        f"each), representing the {a.horizon} timesteps sequentially. Provide only space separated numbers. Nothing else.")
    a.num_tokens = 1024
    a.original_dataset_stats = None
    a.dataset_stats = None
    a._sysuser_len = None
    a.cache_sysuser_len = False
    a.set_dataset_stats(pickle.load(open(os.path.join(HF, "dataset_stats.pkl"), "rb")))
    a.eval()
    return a, cfg, kv


def obs_to_rgb(agentview, wrist, kv):
    """Native libero_to_lerobot_format + le_sample_to_rv_sample + image_unifier_transform(eval=True)."""
    def to_t(img):
        return torch.tensor(np.ascontiguousarray(img[::-1, ::-1]) / 255.0, dtype=torch.float32).permute(2, 0, 1)
    rgb = [to_t(agentview) * 255, to_t(wrist) * 255]                       # le cams: image, wrist_image -> 3p1, 3p2
    rgb = [rearrange(x, "c h w -> 1 1 h w c") for x in rgb]
    sample = {"rgb": torch.cat(rgb, dim=1).numpy()}
    cam_list = list(eval(kv["IMAGE.cam_list"]))
    icfg = types.SimpleNamespace(IMAGE=types.SimpleNamespace(cam_list=cam_list, crop_img=float(kv["IMAGE.crop_img"]),
                                                             img_size=int(kv["IMAGE.img_size"]), brightness_aug=0.2,
                                                             contrast_aug=0.2, saturation_aug=0.2, hue_aug=0.05))
    sample = image_unifier_transform(icfg, sample, sample_cam_list=["3p1", "3p2"], eval=True)
    return torch.tensor(sample["rgb"][None], dtype=torch.float32).to(0)    # (1, history=1, cams=2, 224, 224, 3)


def prompt_inputs(a, rgb, instr):
    imgs = a.get_imgs(bs=1, pc=None, rgb_pc=None, rgb=rgb)
    mi, _ = a.get_qwen_inputs(bs=1, imgs=imgs, instr=[instr], action_txt=[[]], drop_assistant=True, add_generation_prompt=True)
    return mi


class ForcePrefix:
    """Forces the first len(forced) generated tokens (native incremental KV path, one token per step), then releases."""
    def __init__(self, prompt_len, forced):
        self.L, self.forced = prompt_len, list(forced)

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.L
        if step < len(self.forced):
            keep = scores[:, self.forced[step]].clone()
            scores = torch.full_like(scores, float("-inf"))
            scores[:, self.forced[step]] = torch.where(torch.isfinite(keep), keep, torch.zeros_like(keep))
        return scores


@torch.no_grad()
def generate(a, mi, prefix_ids=None, temperature=0.0, seed=None, num_return=1, max_new=None):
    """Native generate. A conditioning prefix is fed by FORCED DECODING through the same incremental KV path the native policy
    uses (not by prefill concatenation, which changed digits through bf16 path differences in validation run 1).
    Returned ids exclude the forced prefix."""
    m = dict(mi)
    L = m["input_ids"].shape[1]
    procs = [a.logits_processor]
    k = 0
    if prefix_ids is not None and len(prefix_ids):
        procs = [a.logits_processor, ForcePrefix(L, prefix_ids)]
        k = len(prefix_ids)
    args = {"do_sample": False} if temperature <= 0 else {"do_sample": True, "temperature": temperature}
    if seed is not None:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    out = a.model.generate(**m, logits_processor=procs, max_new_tokens=(max_new or a.num_tokens) + k,
                           num_return_sequences=num_return, **args)
    res = []
    for o in out:
        g = o[L:].tolist()
        assert g[:k] == list(prefix_ids or [])[:k], "forced prefix not reproduced"
        res.append(g[k:])
    return res


def strip_eos(ids):
    return [t for t in ids if t not in (EOS, 151643)]


def decode(a, ids):
    return a.processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def audit(a, ids, expected=56):
    txt = decode(a, ids).strip()
    parts = txt.split(" ") if txt else []
    nums, bad = [], 0
    for x in parts:
        if x.isdigit():
            nums.append(int(x))
        else:
            bad += 1
    return {"text": txt, "n_numbers": len(nums), "n_bad_fields": bad, "eos": bool(ids) and ids[-1] in (EOS, 151643),
            "in_range": all(0 <= v <= a.num_bins_actions for v in nums), "complete": len(nums) == expected and bad == 0,
            "numbers": nums}


def prefix_text(numbers, n):
    return " ".join(str(v) for v in numbers[:n]) + " "


def tok(a, text):
    return a.processor.tokenizer(text, add_special_tokens=False)["input_ids"]


def vector_spans(a, numbers, n_vec, act_dim=7):
    """token index spans of each of the first n_vec vectors inside tok(prefix_text(numbers, n_vec*act_dim))."""
    spans, pos = [], 0
    for j in range(n_vec):
        t = tok(a, prefix_text(numbers[j * act_dim:(j + 1) * act_dim], act_dim))
        spans.append((pos, pos + len(t))); pos += len(t)
    return spans
