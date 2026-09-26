"""
Non-invasive hooks into ORION for layer-wise planning-token extraction.

Nothing in /root/VLA/orion is edited. Everything here is applied at runtime by rebinding
methods on the loaded model object.

WHAT IS CAPTURED
----------------
ORION's planner is driven by a single vector: the hidden state at the `<waypoint_ego>`
token position, taken from the LLM's FINAL layer (`llava_llama.py:295-311`). That vector
is `ego_feature`, and `Orion.simple_test_pts` feeds it to the VAE planner
(`orion.py:794-819`). We re-run the same selection mask against EVERY layer's hidden
state, giving `h_l_plan in R^4096` for l = 0..32 (0 = embedding output).

We also capture the last-token state of each generated reasoning answer, so that
perception information can be probed at the point where the model has finished
verbalising the scene.

WHY WE DO NOT FEED INTERMEDIATE LAYERS TO THE PLANNER
-----------------------------------------------------
The planner was trained only on final-layer statistics. Feeding h_l for l < 32 straight
into it is an out-of-distribution query, and any resulting "trajectory degradation" mostly
measures that distribution shift rather than a property of the representation. Layer-wise
planner decoding is therefore recorded only as a descriptive diagnostic; the causal claim
comes from activation patching (see `activation_patching.py`), where the patched state is
run through the REMAINING transformer layers before reaching the planner.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class CaptureState:
    """Per-frame buffer filled by the patched `inference_ego`."""
    planning_by_layer: dict[int, torch.Tensor] = field(default_factory=dict)
    n_layers: int | None = None
    hidden_dim: int | None = None
    n_waypoint_positions: int | None = None
    seq_len: int | None = None
    error: str | None = None
    # Boolean (batch, seq) mask of the <waypoint_ego> positions for the frame just run.
    # Activation patching reuses it so the patched re-run writes to exactly the positions
    # the planner reads from.
    last_mask: torch.Tensor | None = None

    def clear(self) -> None:
        self.planning_by_layer = {}
        self.n_waypoint_positions = None
        self.seq_len = None
        self.error = None


def _resolve_causal_lm(lm_head: Any) -> Any:
    """
    Unwrap PEFT/LoRA wrappers to reach the real LlavaLlamaForCausalLM.

    IMPORTANT: a `hasattr(m, "inference_ego")` test is NOT enough. PeftModel forwards
    unknown attribute lookups to the module it wraps, so the wrapper also answers True and
    we would stop one level too early. Its `.model` is then the CausalLM (not the Llama
    body), and calling it returns logits (vocab=32001) instead of hidden states (4096).

    Attribute-chasing is also fragile here: HuggingFace's `PreTrainedModel.base_model` is a
    property returning the DECODER BODY (`self.model`), so following `.base_model` from the
    LoRA wrapper jumps straight past the CausalLM we want. Searching the module tree by
    class name avoids both traps. The real nesting is
    `PeftModelForCausalLM -> LoraModel -> LlavaLlamaForCausalLM`.
    """
    if type(lm_head).__name__ == "LlavaLlamaForCausalLM":
        return lm_head
    for _, mod in lm_head.named_modules():
        if type(mod).__name__ == "LlavaLlamaForCausalLM":
            return mod
    raise AttributeError(
        f"could not locate LlavaLlamaForCausalLM under lm_head (got {type(lm_head).__name__})")


def _waypoint_mask(new_input_ids: torch.Tensor, waypoint_idx) -> torch.Tensor:
    """
    Reproduce the upstream `<waypoint_ego>` position mask exactly.

    Mirrors llava_llama.py:297-310 for both the scalar and list forms of
    `config.waypoint_token_idx`.
    """
    if not isinstance(waypoint_idx, list):
        return new_input_ids == waypoint_idx
    masks = []
    for row in new_input_ids:
        m = torch.zeros_like(row, dtype=torch.bool)
        for tok in waypoint_idx:
            if tok in row:
                m = torch.logical_or(m, row == tok)
        masks.append(m)
    return torch.stack(masks, dim=0)


def install_planning_capture(model: Any, layers: list[int] | None = None,
                             store_dtype: torch.dtype = torch.float16) -> CaptureState:
    """
    Patch `inference_ego` so it records per-layer planning-token hidden states.

    The patched version returns EXACTLY the same value as upstream (the final-layer
    selected hidden states), so ORION's own trajectory output is bit-for-bit unchanged —
    the capture is a pure side effect. Verified by `--verify-identical` in the runner.

    Args:
        model: the built `Orion` detector.
        layers: which layer indices to keep (None = all). Index 0 is the embedding output,
            index L is the output of transformer block L.
        store_dtype: dtype for the stored vectors (fp16 keeps files small; analysis
            upcasts to fp32).
    """
    causal_lm = _resolve_causal_lm(model.lm_head)
    state = CaptureState()
    original = causal_lm.inference_ego

    def patched(inputs=None, images=None, image_sizes=None, return_ego_feature=False, **kwargs):
        # Rebuild the multimodal inputs exactly as upstream does.
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _,
             new_input_ids) = causal_lm.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images,
                image_sizes=image_sizes)
        else:
            inputs_embeds = causal_lm.get_model().embed_tokens(inputs)
            new_input_ids = inputs

        outputs = causal_lm.model(
            input_ids=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=causal_lm.config.output_attentions,
            output_hidden_states=True,          # <-- the only behavioural change
            return_dict=True,
        )

        hidden_states = outputs[0]
        loc = _waypoint_mask(new_input_ids, causal_lm.config.waypoint_token_idx)
        loc = loc.to(device=hidden_states.device)

        state.clear()
        state.last_mask = loc.detach().cpu().clone()
        try:
            all_hidden = outputs.hidden_states          # tuple: n_layers+1 tensors
            state.n_layers = len(all_hidden) - 1
            state.hidden_dim = all_hidden[0].shape[-1]
            state.seq_len = all_hidden[0].shape[1]
            state.n_waypoint_positions = int(loc.sum().item())
            wanted = range(len(all_hidden)) if layers is None else layers
            for li in wanted:
                if li < 0 or li >= len(all_hidden):
                    continue
                sel = all_hidden[li][loc]               # (n_positions, hidden)
                if sel.numel() == 0:
                    continue
                # Several <waypoint_ego> positions can occur; the planner consumes them
                # all as a (n, 4096) block, so keep the block rather than pooling.
                state.planning_by_layer[li] = sel.detach().to(store_dtype).cpu().clone()
        except Exception as exc:                        # never break inference for capture
            state.error = f"{type(exc).__name__}: {exc}"

        selected_hidden_states = hidden_states[loc]
        if return_ego_feature:
            return selected_hidden_states
        raise AssertionError("inference_ego called with return_ego_feature=False")

    causal_lm.inference_ego = patched
    causal_lm._orion_poc_original_inference_ego = original
    return state


def uninstall_planning_capture(model: Any) -> None:
    causal_lm = _resolve_causal_lm(model.lm_head)
    orig = getattr(causal_lm, "_orion_poc_original_inference_ego", None)
    if orig is not None:
        causal_lm.inference_ego = orig
        del causal_lm._orion_poc_original_inference_ego


# --------------------------------------------------------------------------------------
# Reasoning-token capture
# --------------------------------------------------------------------------------------

@dataclass
class ReasoningCapture:
    """Last-token hidden state of each generated reasoning answer, per layer."""
    by_round: list[dict[int, torch.Tensor]] = field(default_factory=list)

    def clear(self) -> None:
        self.by_round = []


def install_reasoning_capture(model: Any, layers: list[int] | None = None,
                              store_dtype: torch.dtype = torch.float16) -> ReasoningCapture:
    """
    Capture the hidden state at the final position of each *generated* answer.

    `generate()` is left untouched (patching it risks changing sampling). Instead we take
    the last decoding step's hidden states via a forward hook on the LLM body, which fires
    once per generated token; keeping the most recent one yields the state at the answer's
    final token.
    """
    causal_lm = _resolve_causal_lm(model.lm_head)
    cap = ReasoningCapture()
    body = causal_lm.model
    latest: dict[int, torch.Tensor] = {}

    def hook(_module, _args, output):
        hs = getattr(output, "hidden_states", None)
        if hs is None:
            return
        latest.clear()
        wanted = range(len(hs)) if layers is None else layers
        for li in wanted:
            if 0 <= li < len(hs):
                latest[li] = hs[li][:, -1, :].detach().to(store_dtype).cpu().clone()

    handle = body.register_forward_hook(hook)
    cap._handle = handle           # type: ignore[attr-defined]
    cap._latest = latest           # type: ignore[attr-defined]
    return cap


@contextlib.contextmanager
def output_hidden_states(model: Any, enabled: bool = True):
    """Temporarily force `config.output_hidden_states` (used around `generate`)."""
    causal_lm = _resolve_causal_lm(model.lm_head)
    prev = causal_lm.config.output_hidden_states
    causal_lm.config.output_hidden_states = enabled
    try:
        yield
    finally:
        causal_lm.config.output_hidden_states = prev
