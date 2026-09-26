"""
Activation patching on the planning token, and steering directions.

WHY PATCHING RATHER THAN LAYER-WISE PLANNER DECODING
----------------------------------------------------
Feeding h_l (l < final) straight into ORION's planner is an out-of-distribution query:
the planner only ever saw final-layer statistics during training, so a degraded trajectory
mostly measures that distribution shift. Instead we replace the planning-token state at
layer l and let the model finish normally:

    bad input -> layers 1..l -> REPLACE h_l[waypoint] -> layers l+1..L -> planner -> traj

Everything downstream of the patch is the model's own computation, and the planner still
receives a genuine final-layer vector, so a change in the trajectory is attributable to
the patched representation rather than to feeding the planner something it cannot parse.

CONTROLS (spec section 14.7-14.8)
  * random-direction control: same norm, random direction -> `make_random_direction`
  * random-layer control: patch a layer other than the hypothesised one
  * collateral damage: apply the same intervention to already-correct samples and measure
    how many break -> handled by the caller passing a control group

IMPLEMENTATION
A forward pre-hook on decoder layer l+1 rewrites the rows of the hidden-state tensor at
the `<waypoint_ego>` positions. Hooks are installed/removed per call, so nothing leaks
between runs.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import torch


def _decoder_layers(causal_lm) -> torch.nn.ModuleList:
    """The transformer block list of the (possibly LoRA-wrapped) Llama body."""
    body = causal_lm.model
    for attr in ("layers", "decoder"):
        mod = getattr(body, attr, None)
        if isinstance(mod, torch.nn.ModuleList):
            return mod
        if mod is not None and hasattr(mod, "layers"):
            return mod.layers
    raise AttributeError("could not locate decoder layer list")


@contextlib.contextmanager
def patch_planning_token(causal_lm, layer: int, positions: torch.Tensor,
                         replacement: torch.Tensor | None = None,
                         direction: torch.Tensor | None = None, alpha: float = 1.0):
    """
    Patch the planning-token hidden state entering block `layer`.

    Exactly one of `replacement` (hard swap, h <- h_good) or `direction` (steering,
    h <- h + alpha*d) must be given.

    `layer` uses the same indexing as the captured states: the value written is the INPUT
    to block `layer`, i.e. the output of block `layer-1`. Patching at `layer` therefore
    means blocks `layer..L` still run on the modified state.

    `positions` is the boolean mask over the sequence produced by `_waypoint_mask`.
    """
    if (replacement is None) == (direction is None):
        raise ValueError("pass exactly one of `replacement` or `direction`")

    layers = _decoder_layers(causal_lm)
    if not (0 <= layer < len(layers)):
        raise IndexError(f"layer {layer} out of range (0..{len(layers)-1})")

    def pre_hook(_module, args, kwargs):
        hs = kwargs.get("hidden_states", None)
        use_kw = hs is not None
        if not use_kw:
            if not args:
                return None
            hs = args[0]
        mask = positions.to(hs.device)
        if mask.shape != hs.shape[:2]:
            return None
        new = hs.clone()
        if replacement is not None:
            rep = replacement.to(device=hs.device, dtype=hs.dtype)
            n = int(mask.sum().item())
            if rep.ndim == 1:
                rep = rep.unsqueeze(0).expand(n, -1)
            elif rep.shape[0] != n:
                # Broadcast/trim so a donor with a different number of waypoint slots
                # still applies cleanly.
                rep = rep[:1].expand(n, -1) if rep.shape[0] >= 1 else rep
            new[mask] = rep
        else:
            d = direction.to(device=hs.device, dtype=hs.dtype)
            new[mask] = new[mask] + alpha * d
        if use_kw:
            kwargs["hidden_states"] = new
            return args, kwargs
        return (new,) + tuple(args[1:]), kwargs

    handle = layers[layer].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def make_random_direction(reference: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Random direction with the same norm as `reference`.

    The norm match matters: an intervention's effect scales with magnitude, so a random
    control of a different length would not be a fair comparison.
    """
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(reference.shape[-1]).astype(np.float32)
    d /= max(np.linalg.norm(d), 1e-8)
    return d * float(np.linalg.norm(reference))


def steering_direction(X_good: np.ndarray, X_bad: np.ndarray,
                       normalise: bool = False) -> np.ndarray:
    """
    Mean-difference direction d = mean(good) - mean(bad).

    This is the standard difference-of-means estimator. `normalise=True` returns a unit
    vector, which makes the alpha sweep comparable across layers (whose norms differ a
    lot in a Llama residual stream).
    """
    if len(X_good) == 0 or len(X_bad) == 0:
        raise ValueError("both groups must be non-empty")
    d = X_good.mean(axis=0) - X_bad.mean(axis=0)
    if normalise:
        d = d / max(np.linalg.norm(d), 1e-8)
    return d.astype(np.float32)


@dataclass
class InterventionOutcome:
    sample_id: str
    layer: int
    mode: str                 # patch / steer / random_dir / none
    alpha: float
    ade_before: float
    ade_after: float
    fde_before: float
    fde_after: float
    action_before: str
    action_after: str
    action_gt: str
    recovered: bool           # was wrong, is now right
    broken: bool              # was right, is now wrong

    def to_dict(self) -> dict:
        return self.__dict__.copy()
