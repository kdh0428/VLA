"""
SpatialVLA closed-loop policy for SimplerEnv Google Robot tasks with chunk-level interventions.

Execution follows the official SpatialVLA SimplerEnv adapter (DelinQu/SimplerEnv-OpenVLA,
simpler_env/policies/spatialvla/spatialvla_model.py) for policy_setup = "google_robot": 224 x 224 INTER_AREA resize,
the 4-step chunk is predicted every control step, the executed action is the ActionEnsembler (temp -0.8) average of the
predictions for the current step from the last 4 chunks, rotation euler -> axis-angle, sticky relative gripper (10 repeats).
The only change is that the chunk tokens come from SpatialVLA.decode_step (identical to generate) so that an intervention
can be applied while the chunk is generated:

  natural          free decoding
  feedback         step-1 translation token := perturbed token p; steps 2-4 generated after p (p in context and executed)
  corrected        step-1 executed with p, but steps 2-4 generated with the natural step-1 token in context
  reverse          step-1 executed with the natural token, steps 2-4 generated after p
Interventions are active only for control steps in [win_start, win_end).
"""
from __future__ import annotations

import numpy as np
from transforms3d.euler import euler2axangle

from action_ensemble import ActionEnsembler


class PolicyState:
    def __init__(self):
        self.ens = ActionEnsembler(4, -0.8)
        self.sticky_on, self.repeat, self.sticky_action, self.prev_grip = False, 0, 0.0, None

    def to_env(self, chunk_actions):
        a = self.ens.ensemble_action(chunk_actions)
        world = np.array(a[:3]); roll, pitch, yaw = np.asarray(a[3:6], dtype=np.float64)
        ax, ang = euler2axangle(roll, pitch, yaw); rot = ax * ang
        cur = np.array(a[6:7])
        if self.prev_grip is None:
            rel = np.array([0.0]); self.prev_grip = cur
        else:
            rel = self.prev_grip - cur
        if np.abs(rel) > 0.5 and not self.sticky_on:
            self.sticky_on, self.sticky_action, self.prev_grip = True, rel, cur
        if self.sticky_on:
            self.repeat += 1; rel = self.sticky_action
        if self.repeat == 10:
            self.sticky_on, self.repeat, self.sticky_action = False, 0, 0.0
        return np.concatenate([world, rot, np.asarray(rel, dtype=np.float64).reshape(1)]), a


def chunks_with_interventions(m, inp, specs):
    """specs: list (rows) of None or dict(mode, p) where p = perturbed step-1 translation token.
    Returns (rows, 4, 3) executed token ids and (rows, 4, 3) natural-context tokens for logging."""
    B = len(specs)
    s1, _ = m.decode_step(inp, [[] for _ in range(B)])                       # natural step-1 for every row
    exec1, ctx1 = s1.copy(), s1.copy()
    for i, sp in enumerate(specs):
        if sp is None:
            continue
        if sp["mode"] == "feedback":
            exec1[i, 0] = ctx1[i, 0] = sp["p"]
        elif sp["mode"] == "corrected":
            exec1[i, 0] = sp["p"]
        elif sp["mode"] == "reverse":
            ctx1[i, 0] = sp["p"]
        else:
            raise ValueError(sp["mode"])
    ctx = [[list(map(int, ctx1[i]))] for i in range(B)]
    exe = [[list(map(int, exec1[i]))] for i in range(B)]
    for _ in range(3):
        st, _ = m.decode_step(inp, ctx)
        for i in range(B):
            ctx[i].append(list(map(int, st[i]))); exe[i].append(list(map(int, st[i])))
    return np.array(exe), np.array(ctx), s1
