"""Standalone NumPy decoder for SpatialVLA action tokens (no transformers / AutoProcessor). Reimplements the official
action_tokenizer.decode_token_ids_to_actions using bin edges from processor_config.json and fixed token-id ranges."""
import json
import numpy as np


class Decoder:
    def __init__(self, processor_config_path, ranges_path):
        pc = json.load(open(processor_config_path)); rg = json.load(open(ranges_path))
        bp = pc["bin_policy"]; nb = pc["action_config"]["num_bins"]
        self.tb, self.pb, self.rb = (np.array(bp["translation"][k]) for k in ("theta_bins", "phi_bins", "r_bins"))
        self.rlb, self.ptb, self.ywb = (np.array(bp["rotation"][k]) for k in ("roll_bins", "pitch_bins", "yaw_bins"))
        self.nphi, self.nr = nb["translation"]["phi_bins"], nb["translation"]["r_bins"]
        self.npitch, self.nyaw = nb["rotation"]["pitch_bins"], nb["rotation"]["yaw_bins"]
        self.R = [tuple(rg["translation"]), tuple(rg["rotation"]), tuple(rg["gripper"])]
        self.eos = rg["eos"]
        assert pc["action_config"]["use_spherical"] is True

    def type_valid(self, ids):
        return all(self.R[i % 3][0] <= int(t) <= self.R[i % 3][1] for i, t in enumerate(ids))

    def decode(self, ids):
        """ids: flat list (3k) -> (k, 7) normalized actions [x, y, z, roll, pitch, yaw, gripper] (official clipping kept)."""
        a = np.asarray(ids, dtype=np.int64).reshape(-1, 3)
        t = np.clip(a[:, 0], *self.R[0]) - self.R[0][0]
        NP = self.nphi * self.nr
        th, ph, r = t // NP, (t % NP) // self.nr, t % self.nr
        theta = 0.5 * (self.tb[th] + self.tb[th + 1]); phi = 0.5 * (self.pb[ph] + self.pb[ph + 1]); rr = 0.5 * (self.rb[r] + self.rb[r + 1])
        x, y, z = rr * np.sin(theta) * np.cos(phi), rr * np.sin(theta) * np.sin(phi), rr * np.cos(theta)
        x, y, z = np.clip([x, y, z], -1, 1)
        q = np.clip(a[:, 1], *self.R[1]) - self.R[1][0]
        NPr = self.npitch * self.nyaw
        ro, pi, ya = q // NPr, (q % NPr) // self.nyaw, q % self.nyaw
        roll = 0.5 * (self.rlb[ro] + self.rlb[ro + 1]); pitch = 0.5 * (self.ptb[pi] + self.ptb[pi + 1]); yaw = 0.5 * (self.ywb[ya] + self.ywb[ya + 1])
        g = np.where(np.clip(a[:, 2], *self.R[2]) - self.R[2][0] == 0, 0.0, 1.0)
        return np.stack([x, y, z, roll, pitch, yaw, g], 1)
