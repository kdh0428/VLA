# FEASIBILITY_20261022_v01 — SpatialVLA within-call path

**Result: PASS.** The final run is `run/feasibility`, attempt 3, after Amendment A3. It passed the same gates as attempt 2, which also passed (`run/feasibility_attempt2_pre_A3`).

## Model and path facts (code + actual output)

| Item | Finding | How checked |
| --- | --- | --- |
| Model | IPEC-COMMUNITY/spatialvla-4b-224-sft-fractal (local copy, file sha256 in `run/*/RUN_FINGERPRINT.json`): SigLIP vision + PaliGemma2/Gemma 2 language model, bf16, attention implementation `eager` | `config.json`, `run_meta.json` |
| Code version | transformers 4.47.0 from `/root/VLA/spatialvla/site` (the model's required version); torch 2.8.0+cu128, CUDA 12.8; RTX 5090 | fingerprint |
| Chunk | 4 actions × 3 tokens. Every natural output in this study (12 validation + 105 pilot) has exactly 13 generated tokens: 12 action tokens + EOS (id 1) | records (`n_generated`, `token_13_is_eos`) |
| Token types and ID ranges | Translation 257153–261248 (4,096 = θ16 × φ32 × r8 spherical bins); rotation 261249–265344 (roll/pitch/yaw 16³); gripper 265345–265346 | action tokenizer |
| Order | Per action: translation, rotation, gripper. Actions a1→a4 are generated left to right, one token per decoding step (`generate` with cache): a3–a4 depend on a1–a2 only through the causal decoding context | code + R identity with bitwise logits |
| Inputs | Official SimplerEnv adapter: 224×224 INTER_AREA image, instruction prompt, intrinsic matrix from the processor. `predict_action` casts all float inputs, including intrinsic, to bf16. The prompt mask is bidirectional inside the prompt; generated tokens are causal | `processing_spatialvla.py`, `_update_causal_mask` |
| Normalization | `decode_token_ids_to_actions` gives normalized translation xyz (spherical bin centres), roll/pitch/yaw (bin centres) and gripper 0/1. Unnormalization uses fractal q01/q99. **u = first 6 normalized components** | action tokenizer source |
| Generation settings | `generate(max_new_tokens=256, do_sample=False)`, cache_implementation `hybrid`, EOS 1, pad 0. No repetition penalty or custom logits processor | `generation_config.json`, `predict_action` |
| Parser | Takes the first 12 generated tokens and asserts token 12 ≠ EOS. A shorter output is **zero-padded**; out-of-range IDs are **clipped**. The study therefore checks type validity itself and never pads: a short or invalid output is NATURAL_INVALID (0 cases) | `decode_actions`; `natural_checks` |
| Adapter | Prefill once with a HybridCache of P + 256 (= `generate`'s max_length), then one token per forward pass with a 2-D all-ones mask and `cache_position`. Forced conditioning tokens use the same causal path and never enter the prompt | `scripts/svla_wc_core.py` |

## Parity and self-patch (12 validation observations)
- **Panel.** Candidates are coke seeds 3000–3007 × calls 0/4/8 (24, frozen). The first 12 were all valid (seeds 3000–3003). Seeds 3004–3007 were collected but not needed. The physical layouts are disjoint from the pilot.

| Check | Result |
| --- | --- |
| Official `generate` (with logits output) = collection-time `predict_action` | 12/12 |
| Adapter natural 13 tokens = official | 12/12 (max \|logit difference\| 0.125; 0 argmax changes) |
| **R self-patch a3–a4 raw token identity** | **12/12** |
| R termination token (EOS) = official | 12/12 |
| R decoded actions = official | 12/12 |
| R continuation logits vs official | max \|difference\| **0.0** (bitwise); 0 argmax mismatches |
| Replacement index, emitted prefix, a3–a4 type validity, native EOS | All arms: conditioning differs from natural only in the intended 3-token action. Recorded output prefix = natural a1–a2. a3–a4 type-valid with termination token EOS (192/192 arm continuations in the pilot) |

- **Unexplained difference.** The 0.125 adapter-natural logit difference occurs at a decoding step where the tokens nonetheless match. Its cause was not isolated. R's continuation logits (steps 7–13) are bitwise equal, so it does not affect the intervention comparisons.

## Attempts and amendments (all GPU time counted)
1. **Attempt 1** (1.16 min). A naming bug (`val_pick_coke_can_*` vs the manifest's `val_coke_*`) meant no check ran (A1).
2. **Attempt 2** (1.65 min). PASS with the same numbers as above.
3. **Pilot attempt 1** (6.94 min). R identity failed in 4/105 observations, and the record-time official output differed from collection in 7/105. Diagnosis (A3): `generate` keeps and reuses a larger HybridCache from earlier, longer move_near prompts.
   - A fresh-process diagnostic was fully consistent (10/10).
   - Deterministic algorithms changed tokens and were rejected.
   - The fix gives every official call a fresh cache.
4. **Attempt 3** (1.63 min, after A3). PASS with identical numbers. The pilot was then re-run.
