# MODEL_SELECTION_20261023_v03 — why these policies, and why not the others

Full evidence table with per-cell source tags (LOCAL / PUBLIC / UNVERIFIED) and revisions: `model_selection/MODEL_SELECTION_TABLE_20261023.md`; sources in `model_selection/model_selection_sources.json`. No surveyed-only policy is counted as an experimental model.

## Scientific question → required property
The within-call estimand needs a policy that, inside **one** call, emits several **time steps** autoregressively, so that a later step is conditioned on the decoder's copy of earlier steps. That copy can then be edited at an older or the latest position while the emitted actions are held fixed.

| Policy | Backbone family | Time step ↔ token position | Native older-vs-latest within-call edit | Status in this paper |
| --- | --- | --- | --- | --- |
| AutoVLA | Qwen2.5-VL-3B | 10 motion tokens at 0.5 s, one per step | Yes (matched-prefix latest-position contrast; older position GT/natural) | Experimental (driving, open loop) |
| VLA-0 | Qwen2.5-VL-3B | 8 vectors × 7 integers as text, one vector per step | Yes (vector 3 vs vector 2) | Experimental (LIBERO open loop + paired rollouts) |
| SpatialVLA-4B | SigLIP + Gemma 2 (PaliGemma 2) | 4 actions × 3 spatial tokens, one action per step | Yes (a2 vs a1; verified path, R identity 117/117 in v01 and all v02 checks) | Experimental (SimplerEnv, command level) |
| pi0-FAST | SigLIP + Gemma-2B (JAX) | DCT + BPE compressed chunk; a token is a span of frequency coefficients, **not a time step** | No (only a low→high-frequency edit, a different estimand) | Surveyed only; not suitable |
| OpenVLA | DINOv2/SigLIP + Llama-2-7B | One time step per call (7 tokens), no action history in the prompt | No (no earlier step inside a call) | Surveyed only; not suitable |
| OpenVLA-OFT | OpenVLA-7B + LoRA, bidirectional fork | Parallel decoding over zero-embedded slots, regression head | No (emitted actions never condition later ones) | Surveyed only; not suitable |
| WorldVLA (paper v1 code, fork `Stars1233/WorldVLA@6a9af08`) | Chameleon-7B | Autoregressive blocks, one step per block | Only as a structural null control: its inference attention mask removes block-to-earlier-block attention | Surveyed only; deferred (null control, needs retraining for a matched pair) |

## Selection reason (scientific)
- **SpatialVLA** is the only available policy outside the Qwen2.5-VL family whose official decoding exposes an autoregressive, time-ordered within-call channel. It is therefore the policy that tests whether the position-specific response is specific to Qwen2.5-VL policies.
- **The other candidates** either lack that channel (pi0-FAST, OpenVLA, OFT) or remove it by design (WorldVLA). They cannot provide the same estimand.

## Implementation (separate from the scientific reason)
- **Reuse.** SpatialVLA, its checkpoint and its SimplerEnv evaluation were already installed and verified on this server (P1-D), and the open-drawer task uses the same embodiment, adapter and checkpoint.
- **Disk.** Every surveyed alternative needs 14–25 GB of new weights, and some need new frameworks (JAX, a transformers fork). Disk free space was about 2.2–2.4 GB during this work. This would have blocked them in practice, but it is not the reason for any verdict above.

## Limits
- Showing a response in a second backbone family is not the same as isolating a causal effect of the backbone: the policies also differ in data, action representation, chunk length, domain and training.
- The SpatialVLA evidence is command-level, on 2 + 1 tasks.
