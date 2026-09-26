"""
Motion primitive of a single AutoVLA action token.

The L35 intervention edits ONE token, which covers one 0.5 s segment, so the failure mode
has to be defined at that granularity too — a sample-level coarse action would lump
together token errors that mean quite different things.

Codebook geometry (verified against `ActionTokenizer.rollout`): entry `a` has shape
(6, 4, 2) — 6 sub-steps x 4 bbox corners x (longitudinal, lateral), in the frame of the
segment's start. `rollout` advances the pose by the mean of the LAST sub-step's corners,
so that mean is exactly the displacement the token contributes.

Speed change is expressed relative to the ego's current speed, which is what makes
"ACCELERATE" / "DECELERATE" meaningful rather than a bare speed bucket. The thresholds
match `src/labeling/labels.py` so the vocabulary is the same one used everywhere else.
"""
from __future__ import annotations

import numpy as np

DT = 0.5
STOP_SPEED = 0.5        # m/s below which the token is stationary
ACCEL_RATIO = 1.25
DECEL_RATIO = 0.75
MIN_REF_SPEED = 0.5     # m/s; below this a ratio is meaningless
LATERAL_M = 0.35        # lateral displacement within one 0.5 s segment


def token_displacement(codebook: np.ndarray, idx: int) -> np.ndarray:
    """(longitudinal, lateral) displacement contributed by one action token."""
    return np.asarray(codebook[idx][-1], dtype=np.float64).mean(axis=0)


def token_primitive(codebook: np.ndarray, idx: int, ego_speed: float) -> str:
    """
    Coarse primitive of one action token: '<LONGITUDINAL>' or '<LONGITUDINAL>+<LATERAL>'.

    Longitudinal is relative to `ego_speed`; lateral is the segment's own sideways motion.
    """
    d = token_displacement(codebook, idx)
    v = float(np.linalg.norm(d)) / DT
    ref = max(float(ego_speed), 0.0)

    if v < STOP_SPEED:
        lon = "STOP"
    elif ref < MIN_REF_SPEED:
        lon = "ACCELERATE"                      # moving off from rest
    else:
        r = v / ref
        lon = ("ACCELERATE" if r >= ACCEL_RATIO
               else "DECELERATE" if r <= DECEL_RATIO
               else "MAINTAIN")

    lat = ("STRAIGHT" if abs(d[1]) <= LATERAL_M
           else ("RIGHT" if d[1] < 0 else "LEFT"))
    return lon if lat == "STRAIGHT" else f"{lon}+{lat}"


def failure_mode(codebook: np.ndarray, gt_idx: int, pred_idx: int, ego_speed: float) -> str:
    """'GT primitive -> predicted primitive'."""
    return (f"{token_primitive(codebook, gt_idx, ego_speed)}"
            f"->{token_primitive(codebook, pred_idx, ego_speed)}")
