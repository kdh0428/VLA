# FEASIBILITY — section 6: π₀-FAST (main candidate), WorldVLA (side note) (review_followup4)

**Status: FEASIBILITY CHECK ONLY (2026-10-17).** Nothing was downloaded except code and small metadata files. Nothing ran on a GPU, nothing was fine-tuned, and no model was run. Every statement below comes from reading official code or metadata at the commits listed in §0, or from public bucket and HF listings. Items marked **UNVERIFIED** need the smoke test in §9 before anyone relies on them.

**Verdict: feasible with conditions. Not feasible on this machine today.**
- **Main blocker: disk.** About 3.1 GB is free. The minimum footprint is about 20–25 GB (§8).
- **Second risk: the RTX 5090 (sm_120) under JAX** (§3).
- **No fine-tuning is needed.** An official `pi0_fast_libero` JAX checkpoint, with its matching norm stats, exists in the public bucket and can be read anonymously. However, the current README no longer lists it, and nothing documents which LIBERO numbers it reproduces.
- **Scope is narrower than AutoVLA.** π₀-FAST has **no action history in its context**. Each inference conditions only on images, the prompt and the discretised state. So all *cross-step action-context* interventions (the AutoVLA H/P/RN/RV family) are **NA** for π₀-FAST. Only **within-chunk** token interventions exist: earlier tokens hold lower DCT frequencies of the *same* chunk.

---

## 0. Sources and snapshots (accessed 2026-10-17)

| Source | Snapshot |
|---|---|
| openpi GitHub `Physical-Intelligence/openpi` | HEAD `215abfb217dbac7d5f1273282331b9b1866c0479` (2026-08-24). Shallow plus blobless clone in scratchpad. 215 commits in history. |
| openpi checkpoint bucket | `https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/` (anonymous JSON listing, metadata only) |
| FAST tokenizer (HF) | `https://huggingface.co/physical-intelligence/fast`, sha `ec4d7aa71691cac0b8bed6942be45684db2110f4` (lastModified 2025-01-16). Read `processing_action_tokenizer.py`, `processor_config.json`, `README.md`. |
| FAST paper | arXiv 2501.09747 **v1** (only version), `https://arxiv.org/abs/2501.09747`, HTML `https://arxiv.org/html/2501.09747` |
| LeRobot PyTorch port (third party, not PI) | `https://huggingface.co/lerobot/pi0fast-libero` sha `840f4b50…` (2026-02-20); `lerobot/pi0fast-libero-v044` sha `2bb8d559…`; `lerobot/pi0fast-base` sha `308eb844…` |
| WorldVLA / RynnVLA-002 GitHub | `https://github.com/alibaba-damo-academy/WorldVLA` now redirects (API repo id 1006839665) to `alibaba-damo-academy/RynnVLA-002`, HEAD `e548ccc2977ba4fc06013de18192fa6245e454a4` (2025-12-02) |
| Original WorldVLA code (third-party fork snapshot) | `https://github.com/Stars1233/WorldVLA` HEAD `6a9af0811df9663de1f6da765f865f18d3787201` (2025-10-10). Contains upstream commits `128eb0a` (2025-06-23) … `3b71772` (2025-07-05) by the WorldVLA authors. |
| WorldVLA checkpoints (HF) | `https://huggingface.co/Alibaba-DAMO-Academy/WorldVLA` sha `5952d61278d52608b2f141ef5b155d42e237165b` (2025-10-10) |
| WorldVLA paper | arXiv 2506.21539 **v1**, `https://arxiv.org/html/2506.21539v1` |
| PolaRiS (DROID real-to-sim) | arXiv 2512.16881, `https://arxiv.org/abs/2512.16881v1` |
| JAX sm_120 miscompile | `https://github.com/jax-ml/jax/issues/39336` (open, created 2026-07-21) |
| OpenVLA LIBERO-Spatial (HF) | `https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial` sha `962318cec55ac10993ff0f5f43eda9a270b4c873` |

---

## 1. Do the benchmark checkpoints exist, and can we get them?

Public bucket `gs://openpi-assets/checkpoints/` (anonymous listing; the norm-stats JSON downloads without credentials):

| Checkpoint | Type | Size | Object dates | Notes |
|---|---|---|---|---|
| `pi0_fast_libero` | π₀-FAST, LIBERO | 10.85 GB (16 objects) | 2025-05-28 | **Exists and is public, but the current README does not list it.** Contains `params/` (OCDBT) and `assets/physical-intelligence/libero/norm_stats.json` (1964 B, md5 `L2Ad9ZXQvpnr6JwAYQpbfw==`). No `_CHECKPOINT_METADATA`. |
| `pi0_fast_droid` | π₀-FAST, DROID (EEF-velocity action space) | 10.84 GB | 2025-05-28 | Listed in README. DROID needs a real Franka; openpi ships no official sim. |
| `pi0_fast_droid_jointpos` | π₀-FAST, DROID joint position | 10.84 GB | 2025-12-13 | Not in README |
| `polaris/pi0_fast_droid_jointpos_polaris` | π₀-FAST for PolaRiS (Isaac Sim + Gaussian splatting) | not sized | — | PolaRiS also contains `paligemma_binning_droid_jointpos_polaris`, which uses the same PaliGemma backbone with **binning** tokens. That would be a very clean FAST-vs-binning comparator, but it needs Isaac Sim (§7). |
| `pi0_fast_base` | base model | 10.85 GB | 2025-05-28 | Only for fine-tuning |
| ALOHA | — | — | — | **No π₀-FAST ALOHA checkpoint.** `pi0_aloha_sim` and the other ALOHA checkpoints are flow-based π₀ models. |

**Provenance of `pi0_fast_libero`: UNVERIFIED.**
- Until commit `35106c9` (2025-09-03), `examples/libero/README.md` said that following the `pi0_fast_libero` config gives "π0-FAST @ 30k (finetuned) 96.4 / 96.8 / 88.6 / 60.2 / 85.5" (Spatial / Object / Goal / 10 / avg). The README called these "not tuned" and noted that "π₀-FAST does not use a FAST tokenizer optimized for Libero".
- That commit deleted those numbers.
- No file links the bucket checkpoint to those numbers or to a training step.
- The baseline success rate must therefore be **measured by us**, not cited.

**Third-party alternative: LeRobot (HF), PyTorch.**
- `lerobot/pi0fast-libero` is a 7.73 GB safetensors checkpoint. Its card claims "82.5 % SR on all tasks".
- It is a **different checkpoint and a different recipe**: 20k steps, batch size 32, a custom action tokenizer `jadechoghari/tokenizer-lib-mean`, MEAN_STD normalisation, `validate_action_token_prefix=True`.
- It is not the PI model and not the universal FAST+ tokenizer. Use it only as a clearly labelled third-party port.

## 2. Do checkpoint, tokenizer and normaliser form a matched set?

At openpi `215abfb`, `config.py` `pi0_fast_libero` defines:
- `Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180)`;
- `LeRobotLiberoDataConfig(repo_id="physical-intelligence/libero", extra_delta_transform=True)`;
- tokenizer = default `FASTTokenizer`: the PaliGemma SentencePiece `gs://big_vision/paligemma_tokenizer.model` plus HF `physical-intelligence/fast`;
- `use_quantile_norm=True` for every non-π₀ model type (`config.py:187`).

The checkpoint's `assets/physical-intelligence/libero/norm_stats.json` has `q01`/`q99` for state (8-d) and actions (7-d), so quantile normalisation can run. **The parts match by naming convention only; no hash ties them together.**
- The tokenizer revision is not pinned in code (`AutoProcessor.from_pretrained("physical-intelligence/fast")`). HF main `ec4d7aa` (2025-01-16) predates the checkpoint objects, and the model has had no commits since. Pin `revision=ec4d7aa…` in any run.
- `extra_delta_transform=True` is load-bearing.
  - In training, the state was subtracted from already-delta LIBERO actions on dims 0–5.
  - At inference, `AbsoluteActions` adds the state back.
  - The action norm stats confirm this. For example, action dim 3 has mean −2.97 and q01..q99 of −3.31..−1.49, which is state-relative.
  - Running the checkpoint without this transform would be a mismatched pipeline.
- `max_token_len=180` and `action_horizon=10` must match the config. The decoder reshapes to (10, 7).
- Known code history:
  - `6348104` / `3b39ac1` (2025-06-07/10): "fix missing EOS bug for pi0-FAST". Training targets lacked EOS before this fix.
  - The released checkpoint objects date from 2025-05-28. **UNVERIFIED:** whether the checkpoint emits EOS. If it does not, every inference decodes the full `max_decoding_steps=256` (cost impact in §8).
  - `1824d36` / `5d26774` (2025-07): fix for temperature > 0 under JIT. The default inference is greedy (`temperature=0.0`), so it uses no RNG.

## 3. Does the baseline run on the official JAX / supported backend?

- **Official backend: JAX.**
  - `pyproject.toml` pins `jax[cuda12]==0.5.3` and `requires-python >=3.11`.
  - The LIBERO client runs in a separate Python 3.8 venv (`examples/libero`, torch 1.11 cu113, robosuite 1.4.1) and talks to the policy server over a websocket.
- **Not run here (no GPU, no download).** Status: **UNVERIFIED on this machine.** Specific risks:
  1. **RTX 5090 / sm_120 under JAX.**
     - It is UNVERIFIED whether jaxlib 0.5.3 CUDA-12 wheels have sm_120 kernels or fall back to PTX JIT.
     - JAX issue #39336 (open; reported on jaxlib 0.10.2 / RTX 5090) describes a **silent, deterministic GEMM-epilogue miscompile**: `alpha*(A@B+bias)` is computed as `alpha*A@B+bias` under GPU jit at the default autotune level. A reported mitigation is `--xla_gpu_autotune_level<=3`.
     - Any π₀-FAST baseline on the 5090 therefore needs a CPU-vs-GPU logit parity check on a few fixed inputs before rollouts, plus a run with the autotune flag.
  2. **Python and environment.** Our `autovla` env is Python 3.9 and has no JAX. `uv` is not installed. openpi needs a new Python 3.11 env.
     - We already have a working LIBERO plus Mesa-EGL client stack at `/root/VLA/openvla/{LIBERO,site,mesa}`. It could serve as the client if `openpi-client` is installed there. This deviates from the official Docker client, so record it as a deviation.
  3. **Host RAM (14.4 GB container limit).** The params are stored at ~fp32 size (10.85 GB for ~2.9 B params). `restore_params(..., dtype=bfloat16)` restores directly to device arrays. **UNVERIFIED:** whether peak host RAM during the orbax restore stays under 14.4 GB.
  4. **Shared GPU memory.** An external process holds 13–26 GB on the 5090 (9.5 GB at the time of this check). JAX preallocates 75% by default, so set `XLA_PYTHON_CLIENT_PREALLOCATE=false` or a small `XLA_PYTHON_CLIENT_MEM_FRACTION`.

## 4. Does official openpi support π₀-FAST in PyTorch? Checked: **NO**

- README at `215abfb`, section "PyTorch Support": "A few features are currently not supported … **The π₀-FAST model**".
- `src/openpi/models_pytorch/` contains only `pi0_pytorch.py`, `gemma_pytorch.py`, `preprocessing_pytorch.py` and `transformers_replace/`. No file in `models_pytorch/` references `Pi0FAST` or `PI0_FAST`.
- PyTorch-related commits since `5546575` (2025-08-28 "add pytorch support"), up to `e4429ad` (2026-04-14), cover π₀ and π₀.₅ only.
- openpi issue #840 (open) reports that π₀/π₀.₅ performance drops after JAX→PyTorch conversion. This is a further reason to treat PyTorch as non-baseline even for π₀.
- The only PyTorch π₀-FAST is LeRobot's third-party `PI0FastPolicy` (§1): a different checkpoint, tokenizer and normaliser. Treating it as "openpi π₀-FAST" would be a mislabel.
- **Implication:** our existing intervention tooling is torch-based, so we would have to reimplement the hooks in JAX/flax-nnx (§5).

## 5. Token positions → DCT coefficients. Can the decoder effect be separated from suffix regeneration?

**Encoding** (`processing_action_tokenizer.py` at `ec4d7aa`; paper v1 §IV):
1. Normalise the chunk A[T=10, D=7] (quantile, roughly [−1, 1]).
2. `dct(A, axis=0, norm="ortho")` along time, then `round(·×scale)` with `scale=10`.
3. Shift by `min_token=−354` and map each integer to `chr`.
4. Flatten **row-major over (frequency k, dim d)**: [k0·d0…d6, k1·d0…d6, …]. The paper calls this "column-first flattening, i.e. concatenate the lowest-frequency components for each dimension first".
5. Apply byte-level BPE (vocab 2048). Map BPE id `t` to PaliGemma id `257152 − 1 − 128 − t`.
6. Output template: `"Action: " + tokens + "|" + EOS`.

**Position → coefficient map:**
- Token *i* covers the contiguous coefficient index range [c_i, c_i + len_i), where len_i is the character length of BPE token i's string and c_i is the sum of the earlier lengths. Flat index j corresponds to (k = j // 7, d = j % 7).
- So the map is **deterministic given the whole emitted sequence, but not fixed per position**. It varies per sample, and one token can span several dims and frequencies. Early tokens hold the lowest frequencies (the chunk mean and slow trend) across all dims.
- Every analysis must record the per-token span (k, d set) from the actual decoded strings, not a nominal position.

**Separability: yes, for a restricted intervention class.**
- **Decoder-only effect (CPU, no model).** Replace token i, keep all other tokens fixed, and decode with the official decoder. The difference from the original decode is the pure DCT/BPE-decoder effect of the replacement.
- **Suffix-regeneration effect (needs the model).** Teacher-force tokens < i, insert the replacement at i, and let the model regenerate the tokens after i. The total minus the decoder-only effect is the model's reaction.
- **Restriction:** a replacement token whose string length differs from the original shifts every later coefficient to a different (k, d) slot. It usually also makes the total length differ from 70, which triggers the decode fallback (§6).
  - Restrict the primary class to **length-preserving replacements**: same character length, ideally single-coefficient tokens.
  - Report every length-changing edit and every decode failure as **its own outcome category**. Never count a failure as "no change" or as a zero-action effect.
- **Intervention class and NA status:**
  - The cross-chunk action-context interventions (AutoVLA H7/H8/H9/P1A/H32/P1B, RN/RV, matched-prefix B2) are **NA** for π₀-FAST: there is no previous action in its prompt.
  - Only within-chunk prefix interventions (lower-frequency tokens → higher-frequency suffix) exist.
  - For flow/diffusion π₀ and π₀.₅ checkpoints (`pi0_libero`, `pi05_libero`), all token interventions are **NA**. They are not measured zeros.
- **Code needed: UNVERIFIED effort, estimated 150–300 lines of JAX.** `Pi0FAST.sample_actions` is a jitted `lax.while_loop` with greedy argmax. Forced-prefix decoding needs a custom sampler that prefills the prefix plus forced tokens into the KV cache and then continues the loop. A gate like the AutoVLA one is mandatory:
  - untouched;
  - forced-prefix with the model's own tokens (no-op);
  - same-token self-replacement;
  - cached vs full recompute.

## 6. Decode-failure fallbacks: yes, two, and neither is a physical zero action

1. `FASTTokenizer.extract_actions` (openpi `tokenizer.py:119-134`): if the decoded text lacks `"Action: "`, it returns `np.zeros((action_horizon, action_dim))`.
2. `UniversalActionProcessor.decode` (HF `ec4d7aa`): if BPE decode, reshape or the shape assert fails (wrong coefficient count), it prints an error and sets **all DCT coefficients to 0**. The IDCT of zeros is a zero chunk.

**Both zeros are in normalised space.** The output transforms then apply:
- `Unnormalize` (quantile): x = 0 maps to (q01 + q99)/2 per dim;
- `AbsoluteActions` (LIBERO `extra_delta_transform`): adds the current state on dims 0–5.

So a decode failure executes a **state-dependent, non-zero "midpoint" action**. With the norm stats above, the dim-3 midpoint is −2.40 before the state is added back. It is not a stop and not a no-op.

**Further silent paths:**
- The text round trip (PaliGemma `decode`, then `split("Action: ")[1].split("|")[0].strip()`, then re-`encode`) may not be an identity on the action-token ids. **UNVERIFIED.** It is a cheap CPU check: it needs the ~4 MB PaliGemma tokenizer and the small FAST tokenizer files.
- A missing `"|"` keeps the whole tail.
- The decoder prints to stdout and does not raise, so failure rates must be counted explicitly. LeRobot added `validate_action_token_prefix`, a sign that this happens in practice.

## 7. Is there a comparable model in the same environment?

| Candidate | Environment | Token interventions | Status / caveat |
|---|---|---|---|
| OpenVLA-7B LIBERO-Spatial (HF sha `962318c…`, 15.1 GB) | LIBERO (our local stack) | Within-step dim-binning tokens (no action history either) | **Our OpenVLA path is UNVERIFIED** (B3: no parity / no-op gate; 20% baseline rerun disagreement unexplained). The weights are **no longer on local disk** (HF cache holds only a 1.5 KB stub), so it needs a 15 GB re-download. Different backbone (Llama-2 7B), tokenizer (256 bins) and data. |
| π₀ `pi0_libero` (12.0 GB), π₀.₅ `pi05_libero` | LIBERO, same openpi stack | **NA** (flow matching) | Usable for behaviour-level comparison only. A π₀ vs π₀-FAST difference is **not** a pure architecture effect: different heads, objectives and checkpoints. |
| LeRobot `pi0fast-libero` (7.7 GB, PyTorch) | LIBERO | Within-chunk FAST (custom tokenizer) | Third party, different recipe. Would fit our torch tooling, but it is not the PI model. |
| `paligemma_binning_droid_jointpos(_polaris)` vs `pi0_fast_droid_jointpos(_polaris)` | DROID / PolaRiS (Isaac Sim) | Binning vs FAST, same PaliGemma backbone | The scientifically cleanest tokenizer contrast, but PolaRiS needs Isaac Sim plus Gaussian-splat scenes. **Not feasible in this container** (disk, no Isaac). |

Out of scope this round, per §6 of the plan: **RoboFlamingo, RT-2, Octo.** Not evaluated.

## 8. Cost estimates (one RTX 5090, 32 GB). **Estimates only, not measured**

**GPU memory:**
- ~2.9 B params in bf16 is about 6 GB of weights.
- Activations plus KV cache are small at batch 1: prefix of ~3×256 image tokens plus ≤180 text tokens, plus ≤256 decode steps.
- The README says inference needs "> 8 GB". Plan for **8–10 GB** at batch 1 with preallocation off.
- That fits next to the external 13–26 GB process **only if** that process stays below ~22 GB.

**Runtime:**
- FAST paper v1: π₀-FAST takes "approximately 750 ms" per 1 s chunk on an RTX 4090, against ~100 ms for π₀. Their chunks were longer than LIBERO's 10×7.
- For LIBERO, expect roughly 20–60 action tokens per chunk, about 0.3–1.5 s per inference. If the checkpoint never emits EOS (§2), every call runs 256 steps, about 4–6 s (UNVERIFIED).
- LIBERO-Spatial: max 220 steps, replan every 5 (`examples/libero/main.py`), so ≤ 44 inferences per episode. That is about 0.3–4 min of model time per episode, plus Mesa software rendering (CPU).
- **Baseline pilot** (10 tasks × 10 inits = 100 episodes): about 1–7 GPU-h.
- **Full suite** (500 episodes): about 5–35 GPU-h.
- **Open-loop token interventions** on logged observations: per anchor, 1 teacher-forced pass plus 1 suffix regeneration per arm. That is ≈ 1–5 s per arm-anchor, so about 1–3 GPU-h for ~1–2k arm-anchors.
- **Closed-loop interventions** cost about the same as rollouts.

**Disk (required vs 3.1 GB free now):**

| Item | Size |
|---|---|
| `pi0_fast_libero` checkpoint | 10.85 GB |
| openpi Python 3.11 venv (jax[cuda12] + torch + transformers + lerobot deps, per `uv.lock`) | ~8–12 GB (UNVERIFIED) |
| uv cache | duplicates the venv unless cleaned or a copy link-mode is used |
| PaliGemma + FAST tokenizers | < 10 MB |
| LIBERO client | reuse the existing `/root/VLA/openvla` stack (+ `openpi-client`, small) |
| Rollout logs and frames | a few GB depending on frame saving |
| **Total** | **≈ 20–25 GB free needed** |
| Optional: OpenVLA comparator re-download | +15 GB |
| Optional: π₀ `pi0_libero` | +12 GB |

**Fine-tuning:** **not needed** for LIBERO, because the checkpoint exists. If a LIBERO-specific FAST tokenizer or a retrained checkpoint were wanted, the README lists full fine-tuning at > 70 GB GPU memory (A100/H100) and LoRA at > 22.5 GB. LoRA is possible on the 5090 only without the external process, and it would create a **new, non-official checkpoint**. Not recommended. A–C have priority.

## 9. Minimum smoke test before any π₀-FAST experiment (once disk is freed)

1. Pin openpi `215abfb`, HF `physical-intelligence/fast@ec4d7aa`, and the bucket object md5s. Record `jax`/`jaxlib` versions and the CUDA driver.
2. CPU only:
   - FAST encode/decode round trip on LIBERO dataset chunks;
   - PaliGemma text round-trip identity on action ids;
   - per-token span tables;
   - decoder-only replacement effects;
   - fallback trigger rate for length-changing edits.
3. GPU, about 10 fixed observations:
   - CPU-vs-GPU logit/token parity;
   - repeat with `--xla_gpu_autotune_level=3` (JAX #39336);
   - greedy repeat identity;
   - EOS emission and decode length.
4. Intervention gate: no-op forced prefix, self-replacement, and cached vs full recompute must give token identity.
5. Baseline pilot of 100 LIBERO-Spatial episodes on fixed init states. Report SR with a cluster/pair-appropriate interval (A1 lessons). Do not cite the deleted README numbers as its expectation.

## 10. WorldVLA (side note; not a candidate this round)

**Repository state:**
- `github.com/alibaba-damo-academy/WorldVLA` now **redirects to `RynnVLA-002`**.
- The main history was rewritten: it starts at `02acea6` "init" on 2025-11-18, with "history cleanup" at `6f6100d` on 2025-11-22.
- The original WorldVLA commits (2025-06-23…2025-07-05) **are no longer in the upstream repo**. They survive in forks, e.g. `Stars1233/WorldVLA@6a9af08`. That is third-party hosting of the upstream commits, and the commit hashes are listed in §0.

**Original-paper code (fork snapshot, `worldvla/`):**
- The training solver `pretrain_solver_awm_w_ck.py` and the eval solvers `eval_solver_libero_{256,512}.py` use `ChameleonXLLMXForConditionalGeneration_ck`.
- Its `forward(..., att_mask=True)` defaults to building `generate_att_mask_3` in **both the training and the inference branches**. Starting from a causal mask, action blocks after the last image block cannot attend to earlier action blocks.
- This matches paper v1: "current actions rely solely on textual and visual inputs, while prohibiting access to prior actions".

**Current RynnVLA-002 main (`e548ccc`):**
- In `rynnvla-002/model/modeling_xllmx_chameleon.py`, the mask functions still exist, but every call site is commented out and training hard-codes `attention_mask = None`.
- So current main does **not** apply the WorldVLA action mask by default. Results from current main must not be attributed to the original masked model.

**Checkpoints:**
- HF `Alibaba-DAMO-Academy/WorldVLA@5952d61` has one checkpoint per LIBERO suite at each of two resolutions (`model_256/*`, `model_512/*`, ~14.0 GB each).
- Their `args.json` contains no attention-mask flag (`ablation: "0"`, `mask_image_logits: false`).
- **No public matched masked/unmasked checkpoint pair is verified.** The paper's Table 3 compares "default mask" and "proposed mask" (e.g. Long-suite SR 16.9% vs 49.3%), but no unmasked-trained weights are released.
- Turning the mask off at inference on a mask-trained checkpoint is a train/test mismatch, not a matched pair.
- Even whether the released checkpoints were trained with the mask is **UNVERIFIED**. The solver default suggests they were.

**Relevance:** under the mask, the earlier-action-block context channel is removed by construction. Interventions on prior action blocks for a masked model are therefore **structurally NA**, not measured zero effects.

## 11. Open questions (blocking or for the lead)

1. **Disk.** Who frees ≥ 25 GB (or ≥ 40 GB with the OpenVLA comparator), and from which outputs? Do not delete anything without the user's decision.
2. Provenance and baseline SR of `pi0_fast_libero`: which training step, and does it reproduce ~96% on Spatial? Answer by measuring it (§9.5).
3. JAX 0.5.3 on sm_120: does it run, and is it free of the #39336 miscompile? (§9.3)
4. EOS emission of the 2025-05-28 checkpoint: this decides a 2–10× runtime difference.
5. Is the PaliGemma text round trip an identity on FAST action ids? (CPU, cheap)
6. Does a within-chunk-only intervention family answer a question the paper needs? AutoVLA's main claims concern **cross-step action context**, which π₀-FAST does not have. A π₀-FAST result would test a different mechanism (low → high frequency within one chunk). It should be framed as such, never as a replication.
7. Comparator: OpenVLA (path UNVERIFIED, B3 open) or none. Treat a π₀ vs π₀-FAST behaviour difference as checkpoint/method-confounded.
