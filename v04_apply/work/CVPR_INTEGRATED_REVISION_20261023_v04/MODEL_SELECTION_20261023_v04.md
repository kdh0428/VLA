# MODEL_SELECTION_20261023_v04 — why these policies, and why not the others

Full evidence table with per-cell source tags (LOCAL / PUBLIC / UNVERIFIED) and revisions: `model_selection/MODEL_SELECTION_TABLE_20261023_v04.md`; sources in `model_selection/model_selection_sources.json`. No surveyed-only policy is counted as an experimental model. Local environment and public-source observations are those reported by the server agent at the recorded revisions; they were not re-fetched during the CPU review.

## Scientific question → required property
The within-call estimand needs a policy that, inside **one** call, emits several **time steps** autoregressively, so that a later step is conditioned on the decoder's copy of earlier steps. That copy can then be edited at an older or the latest position while the emitted actions are held fixed.

| Policy | Backbone family | Time step ↔ token position | Native older-vs-latest within-call edit | Status in this paper |
| --- | --- | --- | --- | --- |
| AutoVLA | Qwen2.5-VL-3B | 10 motion tokens at 0.5 s, one per step | Yes (matched-prefix latest-position contrast; older position GT/natural) | Experimental (driving, open loop) |
| VLA-0 | Qwen2.5-VL-3B | 8 vectors × 7 integers as text, one vector per step | Yes (vector 3 vs vector 2) | Experimental (LIBERO open loop + paired rollouts) |
| SpatialVLA-4B | SigLIP + Gemma 2 (PaliGemma 2) | 4 actions × 3 spatial tokens, one action per step | Yes (a2 vs a1; verified path, R identity 117/117 in v01 and all v02 checks) | Experimental (SimplerEnv, command level) |
| pi0-FAST | SigLIP + Gemma-2B (JAX) | DCT + BPE compressed chunk; a token is a span of frequency coefficients, **not a time step** | No (token- or coefficient-domain edits require a different estimand) | Surveyed only; not suitable |
| OpenVLA | DINOv2/SigLIP + Llama-2-7B | One time step per call (7 tokens), no action history in the prompt | No (no earlier step inside a call) | Surveyed only; not suitable |
| OpenVLA-OFT | OpenVLA-7B + LoRA, bidirectional fork | Parallel decoding over zero-embedded slots, regression head | No (emitted actions never condition later ones) | Surveyed only; not suitable |
| WorldVLA (paper v1 code, fork `Stars1233/WorldVLA@6a9af08`) | Chameleon-7B | Autoregressive blocks, one step per block | Only as a structural null control: its inference attention mask removes block-to-earlier-block attention | Surveyed only; deferred (null control, needs retraining for a matched pair) |

## Selection reason (scientific)
- **SpatialVLA**, among the examined candidates, provides an accessible, time-aligned autoregressive interface outside Qwen2.5-VL. It therefore tests whether the position-specific response also occurs in a policy from another backbone family.
- **The other candidates** do not directly implement this physical-time-step substitution: FAST is autoregressive but temporally nonlocal, original OpenVLA emits one step, OFT is parallel, and the inspected WorldVLA mask blocks prior action blocks.

## Implementation (separate from the scientific reason)
- **Reuse.** SpatialVLA, its checkpoint and its SimplerEnv evaluation were already installed and verified on this server (P1-D), and the open-drawer task uses the same embodiment, adapter and checkpoint.
- **Disk.** The source report lists about 10.85 GB for the FAST LIBERO checkpoint and about 14–16 GB for individual checkpoints of the other candidates, before extra environments. Disk free space on the experiment server was reported as about 2.2–2.4 GB. This constrained implementation, independently of the scientific interface comparison.

## Limits
- Showing a response in a second backbone family is not the same as isolating a causal effect of the backbone: the policies also differ in data, action representation, chunk length, domain and training.
- The SpatialVLA evidence is command-level, on 2 + 1 tasks.


FAST has autoregressive action dependence. The limitation concerns the present time-aligned intervention, not whether action-token interventions can be studied at all. This is a comparison of the inspected implementations, not an exhaustive assertion that no other suitable policy exists.
