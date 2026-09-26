# Correct Perception, Wrong Action — ORION misalignment PoC

Research question:

> **Can a driving VLA encode scene-level perceptual information correctly and still emit an
> inconsistent action — and where inside the model does perception decouple from action?**

The analysis is *failure-conditioned*. It is not "success vs failure on average" and it is
not "does the CoT match the trajectory". Both the control and the main failure group have
**correct observable perception**, so a layer-wise difference between them cannot be
explained by the scene having been misperceived.

```
visual evidence → layer-wise hidden representation → planning token → trajectory
                                         ▲
                          where does this link break?
```

---

## 1. Environment

| | |
|---|---|
| Host | 2 × RTX 5090 (32.6 GB, Blackwell **sm_120**) |
| conda env | `orion` — python 3.10, torch 2.8.0+cu128 |
| ORION | `xiaomi-mlab/Orion` @ `657f3cf`, at `/root/VLA/orion` |
| Checkpoint | official `Orion.pth` (38.5 GB) + `pretrain_qformer` (14 GB) |
| Data | Bench2Drive official **val split** (50 clips) + Chat-B2D `val` (12,806 frames) |

### Deliberate deviations from ORION's pins (and why)

ORION pins `python 3.8 + torch 2.4.1+cu118`. **CUDA 11.8 cannot target sm_120 at all**, and
torch ≥ 2.7 (required for Blackwell) needs python ≥ 3.9. Everything else is kept at the
official version (`transformers 4.31.0`, `peft 0.12.0`, `diffusers 0.32.2`, numpy < 2).

Three compatibility fixes were needed; all are documented and none change model behaviour:

1. **`mmcv._ext` build** — the toolchain image has a dangling `/usr/local/cuda` symlink, so
   `nvcc` 12.8 is installed into the conda env and `CUDA_HOME`/`CPATH` point at
   `$CONDA_PREFIX/targets/x86_64-linux` (where conda puts `cusolverDn.h`).
   Built with `TORCH_CUDA_ARCH_LIST=12.0`.
2. **One upstream C++ line** — `mmcv/ops/csrc/pytorch/cpu/nms_rotated.cpp:62` used the
   removed `Tensor::type()` inside `AT_DISPATCH`. Changed to `scalar_type()`, which
   dispatches to identical kernels. Recorded in `patches/0001-nms_rotated-scalar_type.patch`.
   **This is the only modified file in the ORION repo.**
3. **`analysis/numpy_compat.py`** — restores `np.bool` / `np.float` / … which the vendored
   pipeline still uses (removed in numpy ≥ 1.24). Each alias is restored to exactly the
   numpy 1.23 value, so behaviour matches the pinned version.

Everything else is a wrapper, a monkey-patch, or a new config — the repo is otherwise
untouched.

### Rebuilding the environment

```bash
/root/VLA/tools/setup_env.sh        # conda env + base deps
/root/VLA/tools/setup_env2.sh       # numpy<2 pin, flash-attn 2.8.3 (prebuilt wheel)
/root/VLA/tools/build_mmcv.sh       # mmcv._ext for sm_120
python /root/VLA/tools/prepare_infos_val.py --workers 16   # b2d_infos_val.pkl
```

---

## 2. Configuration choice (important)

| config | reasoning text? | fits in 32 GB? |
|---|---|---|
| `orion_stage3_infer.py` | ✗ `desc_qa=False` drops every reasoning round | ✗ fp32 |
| `orion_stage3_fp16.py` | ✗ same | ✓ |
| `orion_stage3_cot.py` | ✓ 3 rounds | ✗ fp32 (38.5 GB checkpoint) |
| **`configs/orion_poc_cot_fp16.py`** | ✓ | ✓ |

Our config inherits the CoT config and overrides only `fp16_infer`/`fp16_eval` (matching
`orion_stage3_fp16.py`) plus absolute data roots. `fp16_eval=True` is essential: with
`fp16_infer` alone, `simple_test_pts` skips the GT branch (`orion.py:741`) and never reads
`ego_fut_trajs`.

The model answers three questions per frame: **scene description**, **critical objects**,
**driving behaviour + reasons**.

---

## 3. Subset and sharding

`scripts/build_subset.py` — **no random frame sampling**:

* temporal stride 6 inside each clip (adjacent near-duplicates never both selected);
* scenario-stratified quotas, floored so rare scenarios survive;
* **all pedestrian-hazard frames force-included** (470 of the 497 in the whole val split);
* frames within 35 of a clip edge dropped — the dataset zero-pads `ego_fut_trajs` there, so
  A labels would be computed against padding;
* clips assigned **whole** to one shard (greedy LPT balance), asserted disjoint.

Result: **1,809 frames / 49 clips / 42 scenarios**, shards 907 + 902.

Traffic lights are deliberately **not** force-included: red+green are ~27% of frames, so
preserving them all would invert the natural class balance and make the traffic-light probe
trivially separable — an artefact of sampling, not a property of the model.

> **Note on the pedestrian prior.** Because pedestrian frames are preserved wholesale, their
> rate in the subset (~26%) is far above the natural 3.9%. AUPRC for pedestrian is therefore
> relative to the *subset* prior and is not comparable to a natural-prior AUPRC.

---

## 4. Temporal fidelity

ORION carries a temporal memory bank in `pts_bbox_head` / `map_head`
(`orion_head.pre_update_memory`). Running only the sub-sampled frames would give it a
0.6 s-spaced history instead of the 0.1 s-spaced history it was evaluated with.

So the runner walks **every** frame of each clip in order; for non-selected frames it runs
the vision/detection/map path only (which is what updates the memory) and skips the LLM.
`history_query` — the only memory input that could have come from the language side — is
produced by `memory_decoder_cq` from the detection memory (`orion_head.py:783`), **not** by
the LLM, so skipping the LLM leaves the memory chain exact.

---

## 5. Hidden-state extraction

`inference_ego` is rebound at runtime (`analysis/orion_hooks.py`) to pass
`output_hidden_states=True` and apply the upstream `<waypoint_ego>` mask to **every** layer.
It returns exactly the upstream value, so ORION's own trajectory is unchanged.

Observed shape: **33 tensors (embedding + 32 blocks) × 4096-d**, 1 waypoint position/frame,
stored fp16 (~0.3 MB/frame).

> Resolving the module is subtle: `lm_head` is `PeftModelForCausalLM → LoraModel →
> LlavaLlamaForCausalLM`. A `hasattr(m, "inference_ego")` test stops one level too early
> (PEFT forwards attribute lookups) and HF's `base_model` property jumps straight to the
> decoder body — both give **logits (32001-d) instead of hidden states (4096-d)**. We search
> the module tree by class name instead.

---

## 6. P / R / A labelling

GT and model output are parsed by the **same** functions (`analysis/chatb2d_parser.py`), so
no asymmetric extraction bias can masquerade as a perception error.

* **P** — traffic light, leading vehicle, pedestrian hazard, critical-object side, critical-object motion.
  The traffic light is compared via `traffic_light_obj`, derived from the critical-object
  list on **both** sides, because the CoT config never asks the model a traffic-light
  question; presence is the primary check and colour is scored only when both sides name
  one (colour inside an object clause disagrees with Chat-B2D's dedicated round in ~11% of
  frames even GT-vs-GT).
* **R** — normalised longitudinal intent from the free-form reasoning; STOP/DECELERATE are
  accepted as agreeing (recorded separately so the strict count is recoverable).
  **R is an *observable* reasoning label, not the model's internal causal reasoning.**
* **A** — `ADE`/`FDE` **plus** coarse action semantics **plus** a safety-conditioned check.
  A- if L2 exceeds threshold, **or** the coarse longitudinal action disagrees with GT's,
  **or** the scene imposes a safety requirement the GT trajectory meets and the prediction
  does not. Thresholds are config constants and swept in `threshold_sensitivity.json`.

Undecidable variables are `Unknown` and reported separately, never forced into ±.

> `ego_fut_preds` in the result dict is **already cumulative** (`orion.py:932` cumsums it)
> while `ego_fut_trajs` stays as deltas. The runner converts the prediction back to deltas
> so both share one convention — without this, ADE is inflated several-fold.

---

## 7. Statistics

Clip-grouped splits everywhere (`StratifiedGroupKFold` on `clip_id`), `StandardScaler` fit
on train folds only, `class_weight="balanced"`, linear probes only, ≥3 seeds, bootstrap CIs,
Wilson CIs on proportions, and a **label-shuffled control on the same splits** so chance
level is measured rather than assumed.

---

## 8. Running it

```bash
conda activate orion
cd /root/VLA/vla_misalignment_poc

# 1. subset + deterministic shards
python scripts/build_subset.py --target 1800 --stride 6

# 2. inference + hidden states, one process per GPU (clip-disjoint shards)
CUDA_VISIBLE_DEVICES=0 python scripts/run_orion_hidden_extract.py \
    --split splits/poc_gpu0.json --output outputs/shard_gpu0 --layers all --seed 0 \
    > logs/gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/run_orion_hidden_extract.py \
    --split splits/poc_gpu1.json --output outputs/shard_gpu1 --layers all --seed 0 \
    > logs/gpu1.log 2>&1 &
# or: ./scripts/launch_both_gpus.sh

# 3. merge + validate
python scripts/merge_shards.py

# 4. taxonomy (+ threshold sensitivity)
python scripts/build_failure_taxonomy.py

# 5. layer-wise probes and divergence
python scripts/train_probes.py
python scripts/analyze_transitions.py

# 6. causal intervention (only if the gate in §9 is met)
python scripts/run_activation_patching.py --max-failures 40 --max-controls 40

# 7. figures
python scripts/make_figures.py
```

Per-GPU logs are `logs/gpu0.log` / `logs/gpu1.log`; each shard writes only its own
`outputs/shard_gpu{N}/` directory.

**OOM checklist** (in order, before touching the model): batch stays 1; drop `--layers all`
to a subset; confirm fp16 is active (`fp16=True` in the config); check nothing else holds
the GPU. Do not shrink the architecture.

---

## 9. Gate before the causal stage

Do not over-invest in mechanism analysis unless:

1. enough **P+A-** samples exist (not 1–2 anecdotes),
2. **P+R+A-** recurs,
3. it is not confined to a single scenario.

If these fail, the honest conclusions are a dataset-size problem, a labelling problem, or a
genuinely rare phenomenon — and the report says so.

---

## 10. Why activation patching, not layer-wise planner decoding

Feeding an intermediate `h_l` straight into ORION's planner is **out of distribution** — the
planner only ever saw final-layer statistics — so a degraded trajectory would mostly measure
that shift. Patching instead writes the planning token at layer *l* and lets blocks *l..L*
run normally, so the planner still receives a genuine final-layer vector:

```
bad input → blocks 1..l-1 → REPLACE h_l[<waypoint_ego>] → blocks l..L → planner → trajectory
```

Controls: random direction at matched norm, random layer, and **collateral damage** measured
on already-correct samples.

---

## 11. Layout

```
configs/    orion_poc_cot_fp16.py       CoT + fp16 config (inherits upstream)
splits/     poc_all/gpu0/gpu1.json      deterministic frame lists + subset_stats.json
scripts/    build_subset.py             stratified, clip-aware sampling & sharding
            run_orion_hidden_extract.py inference + layer-wise capture (per GPU)
            merge_shards.py             merge + integrity validation
            build_failure_taxonomy.py   P/R/A table, CIs, threshold sensitivity
            train_probes.py             layer-wise perception & action probes
            analyze_transitions.py      failure-conditioned divergence, confound checks
            run_activation_patching.py  patching / steering / controls
            make_figures.py             figures A–D
analysis/   chatb2d_parser.py           ONE parser for GT and model output
            trajectory_metrics.py       ADE/FDE, action semantics, safety conditions
            orion_hooks.py              runtime hooks (no repo edits)
            probing.py                  clip-grouped linear probes + shuffled controls
            representation.py           group stats, matching, confound correlation
            activation_patching.py      patch/steer context managers
            numpy_compat.py             np.bool etc. shim
patches/    0001-...patch               the single upstream C++ fix
outputs/    shard_gpu0|1/ merged/ taxonomy/ probes/ intervention/
figures/    figA..figD png
```

## 12. Reproducibility

`seed=0` everywhere (`set_random_seed(..., deterministic=True)`); every shard writes
`run_meta.json` with the split, config, checkpoint, layer set, GPU name, torch/numpy
versions, timings and failure counts; `splits/*.json` pin the exact sample IDs.

> **Known stochasticity.** ORION's VAE planner samples noise at inference
> (`orion.py:1646-1650`) and `generate()` uses `do_sample=True, temperature=0.1`, so
> trajectories are not bit-identical across runs. `set_random_seed` fixes the global torch
> seed, which makes a single-process run reproducible, but a re-run with a different frame
> order would differ. Results are reported at the group level for this reason.
