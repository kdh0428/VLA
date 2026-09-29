"""
Shared single-plan machinery for the replanning / reference-stabilisation experiments.

A plan follows the natural/fast path used by every earlier experiment: the forced 6-token stem
"<think>\\nThis is a", free decoding of the stub until the first action token, then 10 action
tokens sampled from the action rows at T = 0.01 with a per-step seed. Action steps are decoded
with a prefix KV cache (prompt + stub) and the post-stub span recomputed each step, so any
context position can be replaced by a reference token (the stabilisation used in experiments
9-12: the model conditions on the reference, the EXECUTED action stays its own sample).
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

AUTOVLA_DIR = "/root/VLA/autovla"
POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (AUTOVLA_DIR, os.path.join(AUTOVLA_DIR, "navsim"), POC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

N_ACT = 10
DT = 0.5
CKPT = "/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt"
CONFIG = os.path.join(AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml")
STEM = "<think>\nThis is a"


def seed_of(*parts) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:4], "little")


class Planner:
    def __init__(self, device: str = "cuda:0"):
        os.chdir(AUTOVLA_DIR)
        from models.autovla import AutoVLA
        from navsim.agents.autovla_agent import AutoVLAAgent, TrajectoryTargetBuilder
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

        cfg = yaml.safe_load(open(CONFIG))
        cfg["model"]["pretrained_model_path"] = os.path.join(AUTOVLA_DIR, "Qwen2.5-VL-3B-Instruct")
        cfg["model"]["codebook_cache_path"] = os.path.join(AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl")
        self.cfg, self.dev = cfg, device
        self.temp = float(cfg["inference"]["sample"]["temperature"])
        self.gen_conf = cfg["inference"]["sample"]
        # VLA_LOW_MEM=1 (small / shared GPU): skip the mean-initialisation of the 2048 new action
        # embeddings in resize_token_embeddings. It needs a ~1.2 GB GPU transient, and those rows are
        # overwritten by the checkpoint right after (load reports missing = 0), so weights are identical.
        if os.environ.get("VLA_LOW_MEM"):
            from transformers import PreTrainedModel
            _orig_resize = PreTrainedModel.resize_token_embeddings

            def _resize(self, new_num_tokens=None, pad_to_multiple_of=None, mean_resizing=True):
                return _orig_resize(self, new_num_tokens, pad_to_multiple_of, mean_resizing=False)
            PreTrainedModel.resize_token_embeddings = _resize
        build = "cpu" if os.environ.get("VLA_BUILD_ON_CPU") else device
        model = AutoVLA(cfg, inference=True, device=build)
        sd = torch.load(CKPT, map_location="cpu", mmap=True)["state_dict"]
        model.load_state_dict({k[len("autovla."):]: v for k, v in sd.items() if k.startswith("autovla.")}, strict=False)
        del sd
        model.to(device).eval()
        torch.cuda.empty_cache()
        self.model, self.llm, self.A0 = model, model.vlm, model.action_start_id
        self.tok = model.processor.tokenizer
        self.stem = self.tok.encode(STEM, add_special_tokens=False)
        ts = TrajectorySampling(time_horizon=cfg["model"]["trajectory"]["time_horizon"],
                                interval_length=cfg["model"]["trajectory"]["interval_length"])
        self.agent = AutoVLAAgent(trajectory_sampling=ts, sensor_data_path=Path("dataset/nuplan/sensor_blobs/test"),
                                  codebook_cache_path=cfg["model"]["codebook_cache_path"], skip_model_load=True)
        self.target = TrajectoryTargetBuilder(ts, cfg["model"]["codebook_cache_path"])

        self._vis = {}
        orig = self.llm.visual.forward

        def visual_cached(pixel_values, grid_thw=None, **kw):
            if self._vis.get("v") is None:
                self._vis["v"] = orig(pixel_values, grid_thw=grid_thw, **kw)
            return self._vis["v"]
        self.llm.visual.forward = visual_cached

    # ---- trajectories <-> tokens ------------------------------------------------------------
    def tokenize(self, poses) -> list:
        """(10, 3) local poses [x, y, heading] -> 10 codebook indices (same routine as the GT tokens)."""
        out = self.target.compute_targets({"gt_trajectory": np.asarray(poses, np.float64).tolist()})
        return [int(x) for x in np.asarray(out["gt_idx"]).reshape(-1)]

    def decode(self, idx) -> np.ndarray:
        """codebook indices -> (n, 3) local poses [x, y, heading] (pose after each token)."""
        tr = self.model.action_tokenizer.decode_token_ids_to_trajectory(torch.tensor([self.A0 + int(a) for a in idx]))
        return np.asarray(tr[0, 1:], np.float64)

    # ---- one plan ---------------------------------------------------------------------------
    @torch.no_grad()
    def plan(self, scene: dict, key, speed=None, accel=None, ref=None, ref_positions=()) -> dict:
        """
        scene: preprocessed scene dict. speed/accel override the prompt's ego state (magnitudes).
        ref / ref_positions: context positions j in ref_positions are replaced by ref[j].
        Returns tokens (executed samples), entropies, stub, and whether the stub reasoned (CoT).
        """
        sc = dict(scene)
        if speed is not None:
            sc["velocity"] = [float(speed), 0.0]
        if accel is not None:
            sc["acceleration"] = [float(accel), 0.0]
        feats = {}
        for b in self.agent.get_feature_builders():
            feats.update(b.compute_features(sc))
        self._vis.clear()
        mi = {k: v.to(self.dev) for k, v in self.model.get_prompt(feats).items() if isinstance(v, torch.Tensor)}
        prompt = mi["input_ids"][0].tolist()
        llm, A0 = self.llm, self.A0

        # stub: forced stem, then free decoding until the first action token
        stem_ids = torch.tensor([prompt + self.stem], device=self.dev)
        m1 = dict(mi); m1["input_ids"] = stem_ids; m1["attention_mask"] = torch.ones_like(stem_ids)
        torch.manual_seed(seed_of(key, "stub"))
        g = llm.generate(**m1, max_new_tokens=160, do_sample=True, temperature=self.gen_conf["temperature"],
                         top_k=self.gen_conf["top_k"], top_p=self.gen_conf["top_p"])
        new = g[0][stem_ids.shape[1]:].tolist()
        first_act = next((j for j, x in enumerate(new) if x >= A0), None)
        if first_act is None:
            raise RuntimeError("no action token within 160 generated tokens")
        stub = self.stem + new[:first_act]
        cot = "complex" in self.tok.decode(stub)

        prefix = prompt + stub
        P = len(prefix)
        ids = torch.tensor([prefix], device=self.dev)
        llm.rope_deltas = None
        out = llm(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values_videos=mi.get("pixel_values_videos"),
                  video_grid_thw=mi.get("video_grid_thw"), use_cache=True, cache_position=torch.arange(P, device=self.dev))
        cache, last = out.past_key_values, out.logits[:, -1, A0:A0 + 2048].double()
        del out
        own, ents = [], []
        ref_positions = set(ref_positions)
        for k in range(N_ACT):
            if k > 0:
                ctx = [ref[j] if (j in ref_positions and ref is not None and j < len(ref)) else own[j] for j in range(k)]
                sids = torch.tensor([[A0 + a for a in ctx]], device=self.dev)
                o = llm(input_ids=sids, attention_mask=torch.ones((1, P + k), device=self.dev, dtype=torch.long),
                        past_key_values=cache, use_cache=True, cache_position=torch.arange(P, P + k, device=self.dev))
                cache.crop(P)
                last = o.logits[:, -1, A0:A0 + 2048].double()
                del o
            lp = torch.log_softmax(last[0], -1)
            ents.append(float(-(lp.exp() * lp).sum()))
            gen = torch.Generator(); gen.manual_seed(seed_of(key, k))
            own.append(int(torch.multinomial(torch.softmax(last[0] / self.temp, -1).cpu(), 1, generator=gen)))
        del cache
        return {"tokens": own, "entropy": ents, "stub_len": len(stub), "cot": cot}


# ---- kinematic reference ----------------------------------------------------------------------
def ctra_poses(scene: dict, n: int = N_ACT, dt: float = DT) -> np.ndarray:
    """Constant turn rate and acceleration from the ego state in the scene (no future information):
    speed = |velocity|, longitudinal acceleration = acceleration[0], yaw rate from the last history
    interval. Integrated at 0.05 s, sampled every dt. Speed is clipped at 0 (no reversing)."""
    v = float(np.hypot(*scene["velocity"][:2]))
    a = float(scene["acceleration"][0])
    his = np.asarray(scene["his_trajectory"], np.float64)
    w = float(math.atan2(math.sin(his[-1, 2] - his[-2, 2]), math.cos(his[-1, 2] - his[-2, 2]))) / dt
    x = y = h = 0.0
    out, sub = [], 10
    for _ in range(n):
        for _ in range(sub):
            s = dt / sub
            x += v * math.cos(h) * s
            y += v * math.sin(h) * s
            h += w * s
            v = max(0.0, v + a * s)
        out.append([x, y, h])
    return np.asarray(out)


# ---- SE2 helpers ------------------------------------------------------------------------------
def to_global(local: np.ndarray, pose: np.ndarray) -> np.ndarray:
    c, s = math.cos(pose[2]), math.sin(pose[2])
    x = pose[0] + c * local[:, 0] - s * local[:, 1]
    y = pose[1] + s * local[:, 0] + c * local[:, 1]
    return np.stack([x, y, pose[2] + local[:, 2]], 1)


def to_local(glob: np.ndarray, pose: np.ndarray) -> np.ndarray:
    c, s = math.cos(pose[2]), math.sin(pose[2])
    dx, dy = glob[:, 0] - pose[0], glob[:, 1] - pose[1]
    h = np.arctan2(np.sin(glob[:, 2] - pose[2]), np.cos(glob[:, 2] - pose[2]))
    return np.stack([c * dx + s * dy, -s * dx + c * dy, h], 1)
