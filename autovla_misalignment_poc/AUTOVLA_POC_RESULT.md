# AutoVLA PoC — is correct-perception / wrong-action architecture-specific?

**Question.** Does the correct-perception / wrong-action signature found in ORION reproduce
in AutoVLA, which has no separate generative planner and emits its trajectory directly as
LM action tokens?

**Answer: Outcome B — the failure profile is architecture- and benchmark-dependent, and
AutoVLA's dominant failure mode is a different one: autoregressive error propagation across
action tokens.** The ORION-style P/R/A taxonomy could not be constructed at all, for a
reason that is itself a finding (§3).

---

## 1. Setup

| | |
|---|---|
| Model | official `AutoVLA_PDMS_89.ckpt` (LoRA already merged), Qwen2.5-VL-3B backbone |
| Config | `config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml` (the released RFT model) |
| Architecture | **36 decoder layers × 2048**, `lm_head` (153713, 2048) incl. 2048 action tokens |
| Action space | codebook (2048, 6, 4, 2); ids 151665–153712; 10 tokens = 10 poses × 0.5 s |
| Data | OpenScene **navtest** (official token list), 28 logs with local camera blobs |
| **Scenes analysed** | **2,748** |
| Inference failures | **0** |
| Runtime | 48.1 min on one GPU (1.05 s/scene), 3.5 GB of hidden states |

Environment deviations from AutoVLA's pins (RTX 5090 / sm_120) are documented in
`README.md`; the ORION env and results were left untouched.

---

## 2. Table A — baseline reproduction

| metric | AutoVLA (navtest, n=2748) |
|---|---|
| ADE (3 s, ORION-comparable window) — median | **0.038 m** |
| L2 @1 s / @2 s / @3 s (n=60 subsample) | 0.092 / 0.195 / 0.381 m |
| action-token top-1 match vs GT tokens | 0.745 mean |

That is extraordinarily accurate, so it was checked for leakage before being believed:

| diagnostic | result | reading |
|---|---|---|
| GT in prompt? | no — cameras, speed, accel, route command only | ✅ |
| per-step token accuracy | 0.867 → 0.650 (monotone decay) | genuine autoregression, not a leak |
| per-step displacement error | 0.053 → 0.650 m (monotone growth) | genuine prediction decay |
| constant-velocity baseline | 1.802 m | model is 16–30× better |
| constant-acceleration baseline | 0.901 m | ” |

The scenes are not trivial (physics baselines are 1–2 m off); the model is simply very
strong on nuPlan, which is smooth real-world driving with a route command and precise ego
dynamics — a much easier prediction problem than Bench2Drive's adversarial CARLA scenarios.

---

## 3. Why the ORION taxonomy cannot be reproduced here (a finding, not a blocker)

**The released checkpoint never reasons.** Chain-of-thought was present in **1 of 2,748**
scenes; the other 2,747 emit a byte-identical stub:

```
<think>
This is a straightforward scenario, and a direct decision can be made.
</think>
<answer>The final output action is: <action_1503>…<action_1364></answer>
```

This is the RFT model behaving exactly as designed — the GRPO config carries a
`cot_penalty` that rewards skipping reasoning on easy scenes, and navtest is easy for it.

Consequences, stated plainly:

* **R (observable reasoning) does not exist** on this split. No P+R+A− cell can be filled.
* **The model states no perception**, so ORION's P ("does the model's critical-object text
  match GT?") has no counterpart. Defining P from a hidden probe instead would make the
  headline question — "is perception decodable in P+ samples?" — true by construction.

So we followed **Plan A**: perception is used as a *scene condition* from dataset ground
truth (navsim `Annotations`), and A+ vs A− is compared within each condition.

---

## 4. Table B — action taxonomy

| | count | rate | 95% CI |
|---|---|---|---|
| **A−** | **227 / 2,748** | **8.3%** | [7.3%, 9.3%] |
| A+ | 2,521 | 91.7% | |
| CoT present | 1 / 2,748 | 0.04% | |

A− rate by GT scene condition (Figure A):

| condition | A− / n | rate |
|---|---|---|
| lead vehicle present | 68 / 1326 | 5.1% |
| lead vehicle absent | 159 / 1422 | **11.2%** |
| lead vehicle close (<15 m) | 23 / 450 | 5.1% |
| pedestrian present | 34 / 287 | **11.8%** |
| pedestrian absent | 193 / 2461 | 7.8% |
| critical object moving | 98 / 1142 | 8.6% |
| critical object static | 116 / 1425 | 8.1% |

By GT longitudinal action: MAINTAIN **13.1%**, DECELERATE 8.4%, ACCELERATE 6.1%,
STOP 4.5%.

**Threshold sensitivity.** A− is essentially fixed at **8.3–8.7%** across the whole
ADE×FDE backstop sweep, including with the geometric backstop removed entirely
(`inf/inf` = 227, 8.3%). So A− here is decided almost purely by the coarse action
semantics, exactly as intended — and it is not a threshold artefact.

> **Caveat on what A− means here.** The dominant failure reasons are adjacent-category
> confusions — `MAINTAIN→DECELERATE` (31), `DECELERATE→MAINTAIN` (30),
> `MAINTAIN→ACCELERATE` (29), `ACCELERATE→MAINTAIN` (24). With ADE median 0.038 m, most
> A− cases are **boundary cases of the coarse label**, not dangerous driving. That is the
> honest reading, and it is why the MAINTAIN row is the worst: MAINTAIN sits between the
> accelerate and decelerate thresholds and is easiest to cross.

---

## 5. Layer-wise probes (Figures B–D)

Grouped by nuPlan log (the analogue of ORION's clip grouping), scaler on train folds only,
3 seeds, shuffled null on the same folds.

**Both action labels are probed**, because reporting only the GT action is circular — A± is
*defined* by GT-vs-own agreement. This check is what uncovered the error in the ORION report
(now corrected there).

At `action_0`, normalised depth in brackets:

| layer | group | lead vehicle | pedestrian | **action: GT** | **action: own** |
|---|---|---|---|---|---|
| L8 (0.22) | A+ | 0.838 | 0.714 | 0.794 | 0.794 |
| | **A−** | 0.714 | 0.721 | **0.479** | **0.648** |
| L24 (0.67) | A+ | 0.909 | 0.742 | 0.921 | 0.921 |
| | **A−** | 0.817 | 0.695 | **0.589** | **0.687** |
| L36 (1.00) | A+ | 0.857 | 0.635 | 0.894 | 0.894 |
| | **A−** | 0.721 | 0.498 | **0.507** | **0.630** |

Findings:

1. **The same GT-vs-own gap as ORION appears**, in the same direction: in A− samples the
   model's own action is more decodable (0.63–0.73) than the correct one (0.48–0.63). The
   representation holds a confident wrong decision rather than nothing.
2. **But the gap is much smaller than ORION's** (AutoVLA ≈ +0.10–0.17; ORION ≈ +0.22–0.49),
   and AutoVLA's own-action decodability never reaches ORION's 0.94–0.96.
3. Perception in A− is *lower* than in A+ (lead vehicle 0.71–0.82 vs 0.84–0.91) — unlike
   ORION, where the failure group matched or beat the control. So AutoVLA failures are
   **partly perceptual**, not cleanly "perception fine, action wrong".

> **Reliability flag.** With only 28 logs, `lead_vehicle` positives sit in 17 logs
> (top-3 share 56%) and `pedestrian` in 13 — both below the cross-scene-generalisation bar
> the ORION study adopted, so these perception numbers are **flagged, not headline**. Only
> `action_gt` clears it (20 logs, top-3 47%). More camera shards would fix this.

---

## 6. Action logit lens (Figure E) — the analysis ORION could not support

Because AutoVLA emits action tokens from the LM vocabulary, `lm_head` applies to every
layer: a **37 × 10 grid** of (layer × action-generation step) per scene.

**6.1 The action decision forms extremely late.** P(argmax = GT token) is ~0 until roughly
L32 and only rises in the final ~4 layers. Median commitment depth — first layer whose
argmax equals the finally generated token — is **L34–35 of 36 (normalised 0.94–0.97)**, and
is the same for A+ and A−. There is no early-layer action signal to intervene on.

**6.2 Accuracy decays along the generated sequence.** Final-layer P(argmax = GT):

| step | 0 | 3 | 6 | 9 |
|---|---|---|---|---|
| A+ | 0.933 | 0.812 | 0.775 | 0.740 |
| A− | 0.740 | 0.414 | 0.295 | **0.247** |

**6.3 Autoregressive error propagation is the dominant mechanism.**

| step | P(error \| all earlier correct) | P(error \| an earlier step wrong) | ratio |
|---|---|---|---|
| 1 | 0.107 (n=2523) | 0.689 (n=225) | 6.4× |
| 2 | 0.076 (n=2254) | 0.729 (n=494) | 9.6× |
| 3 | 0.048 (n=2083) | 0.762 (n=665) | **15.9×** |
| 4 | 0.040 (n=1984) | 0.737 (n=764) | 18.4× |
| 5 | 0.037 (n=1905) | 0.728 (n=843) | 19.7× |

Once one action token is wrong, the next is wrong ~73% of the time versus ~4% when the
prefix is clean. **This is a failure mode ORION structurally cannot have** — it produces one
planning token and hands it to a planner, with no autoregressive action chain to derail.

**6.4 Error-emergence patterns** (per failing step, spec §14 taxonomy):

| pattern | count |
|---|---|
| D — confident in another token, GT never leads | 502 |
| A — GT never acquired at any layer | 367 |
| C — first step correct, later steps collapse (per sample) | 162 |
| B — GT good mid-stack, lost late | 110 |

D+A dominate at the step level, and C is what links steps together.

---

## 7. Table C — ORION vs AutoVLA

| metric | ORION (Bench2Drive) | AutoVLA (navtest) |
|---|---|---|
| architecture | vision → LLM → `<waypoint_ego>` → VAE planner | vision → LLM → action tokens |
| depth × width | 32 × 4096 | 36 × 2048 |
| samples | 1,809 (49 clips) | 2,748 (28 logs) |
| ADE median | 0.686 m | **0.038 m** |
| **action-failure rate** | 44.6% (A−, any P) | **8.3%** |
| P+A− rate | 23.2% | **not constructible** (no model perception) |
| P+A− / all failures | 55.6% | n/a |
| observable reasoning | present (3 CoT rounds) | **absent (1/2748)** |
| perception probe, failures | 0.83–0.91 | 0.71–0.82 *(flagged: 17 logs)* |
| action probe, failures — GT | 0.45–0.79 | 0.48–0.63 |
| action probe, failures — **own** | **0.94–0.96** | 0.63–0.73 |
| divergence / commitment depth | L11–15 (norm. 0.34–0.47) | **L34–35 (norm. 0.94–0.97)** |
| autoregressive action propagation | n/a (single token) | **6–20× amplification** |

---

## 8. Interpretation — Outcome B, with an explicit confound

The pre-registered outcomes map as follows:

* **Not Outcome A.** The ORION signature does not transfer cleanly. Action failure drops
  from 44.6% to 8.3%; perception in failures is *worse* than in successes rather than
  preserved; and the ORION P+A− construction is impossible because the model neither
  reasons nor states what it perceives.
* **Outcome B.** The action interface changes the failure profile. ORION concentrates the
  decision in one planning token that commits by mid-stack (L11–15) and hands a confident
  wrong action to a separate planner. AutoVLA forms its decision in the last ~3 layers and
  then *propagates* it: a single bad token makes the next bad ~73% of the time. These are
  different mechanisms, not different amounts of the same one.
* Partial **Outcome C** as well: AutoVLA failures carry a perceptual component, so
  "P+ → A−" is not the right description of them.

> **The confound, stated plainly.** Architecture and benchmark are not separated in this
> comparison. ORION ran on Bench2Drive (adversarial CARLA scenarios) and AutoVLA on navtest
> (smooth real-world nuPlan). The 5× difference in failure rate is at least partly task
> difficulty — physics baselines are 0.9–1.8 m on navtest, so it is an easier prediction
> problem. **This PoC cannot attribute the difference to architecture alone.** Separating
> them requires running one model on both, or AutoVLA on its CARLA benchmark.

### What genuinely survives from ORION

The GT-vs-own-action gap reproduces in both models, in the same direction. In neither does
the failing representation lack an action; in both it holds a confident wrong one. That is
the cross-architecture piece of "encoded but unused", and it is the part worth building on.

---

## 9. Next steps

1. **Remove the benchmark confound** — AutoVLA on CARLA/Waymo, or a harder navtest slice
   (it is the single largest threat to every claim in §7–8).
2. **More logs** — 28 is too few for cross-log generalisation on the perception probes;
   each extra OpenScene shard is ~4 GB / 5 min.
3. **Teacher-forcing counterfactual** — replace a wrong action token at step *t* with the GT
   token and regenerate. §6.3 is currently correlational; this makes it causal, and it is
   cheap because the propagation effect is so large.
4. Attention-level analysis of how perception tokens feed the final ~4 layers, where §6.1
   shows the decision is actually formed.

## 10. Artefacts

```
outputs/gpu1/records.jsonl        2,748 records + logit lens
outputs/gpu1/hidden/*.npz         37 layers × (prefill + 10 action steps) × 2048, fp16
outputs/taxonomy/                 taxonomy.{json,csv}, summary, threshold sensitivity
outputs/probes/                   layerwise_probes.json
outputs/logit_lens/               logit_lens.json, per_sample_patterns.json
figures/figA..figE                *.png
```

---

# Addendum — First-error teacher-forcing counterfactual

**Question.** Is the first wrong action token a *cause* of the later ones, or do hard scenes
simply keep being wrong? §6.3 showed 4% → 73% but could not separate the two.

## Design

Fully paired, one token apart. For each sample the first action step `t` whose generated
token differs from GT is located, and all arms share the **identical prefix**
(prompt + the model's own generated tokens up to step *t*). They differ only in the single
token written at *t*; after that, generation is handed back to the model — free
autoregressive decoding to the end, same decoding config, nothing downstream forced.

| arm | token written at step *t* |
|---|---|
| `original` | the model's own wrong token (re-generated, so all arms sit on the same prefix) |
| `gt` | the ground-truth token |
| `plausible` | the model's **own runner-up** action token at that position |
| `random` | uniformly random valid action token ≠ GT, ≠ original (3 seeds) |

The `plausible` arm is the control the claim actually needs: a uniformly random token out of
2048 lands far off any drivable trajectory (ADE ≈ 3.1 m), so beating it only shows
"GT ≠ nonsense". The runner-up is an alternative the model itself ranked second.

**n = 400** first-error samples (of 1,027 available) across **28 logs**, 18.5 min on one GPU.

## Results

| Condition | Next-token error | Downstream error | Sequence recovery | ADE | FDE |
|---|---|---|---|---|---|
| original | 69.8% | 72.2% | 11.2% | 0.298 | 0.705 |
| **gt forced** | **18.0%** | **30.2%** | **50.5%** | **0.068** | **0.192** |
| plausible control | 71.5% | 72.9% | 10.2% | 0.333 | 0.780 |
| random | 93.2% | 94.0% | 1.3% | 3.129 | 5.657 |

95% bootstrap CI (over samples) on downstream error: original [68.7, 75.5],
gt [26.6, 33.8], plausible [69.5, 76.2], random [92.8, 95.0].

**The decisive contrast — GT vs the plausible control** (paired bootstrap, 10k):

| metric | GT − plausible | 95% CI | p |
|---|---|---|---|
| next-token error | −0.535 | [−0.588, −0.483] | ≈0 |
| downstream error | −0.428 | [−0.472, −0.384] | ≈0 |
| sequence recovery | +0.403 | [+0.353, +0.453] | ≈0 |
| ADE | −0.264 m | [−0.313, −0.220] | ≈0 |
| coarse action match | +0.110 | [+0.065, +0.155] | ≈0 |

McNemar on next-token error, GT vs plausible: **n01 = 222, n10 = 8, p = 2.1e-55**.
GT vs original: n01 = 215, n10 = 8, p = 2.1e-53.

Coarse action recovery: gt **89.0%**, original 81.0%, plausible 78.0%, random 41.5%.

By first-error position (downstream error rate):

| t | n | original | gt | plausible-equivalent |
|---|---|---|---|---|
| 0 | 90 | 78.0% | 44.0% | ~95% (random) |
| 1 | 103 | 70.6% | 33.6% | |
| 2 | 76 | 73.5% | 26.7% | |
| 3 | 32 | 62.5% | 22.9% | |
| 4 | 30 | 66.7% | 23.3% | |
| 5 | 25 | 76.0% | 33.0% | |
| 6 | 22 | 72.7% | 12.1% | |
| 7 | 22 | 68.2% | 4.5% | |

Correction helps monotonically more the later the first error — with fewer steps left and
more correct context, one fix nearly restores the sequence (t=7: 68.2% → 4.5%).

## Answer

> **Yes. Correcting the first erroneous action token causally prevents subsequent
> action-token errors.**

The pre-registered bar was that GT correction must clearly beat a same-position
intervention. It does, against both controls, and the *plausible* control is the one that
matters: writing the model's own second-choice token leaves downstream error at 72.9% —
**statistically indistinguishable from leaving the original mistake in place (72.2%)**.
So the effect is not "perturbing this position changes things"; it is specific to writing
the *correct* token.

A quantitative decomposition of the original 72.2% downstream error:

* **≈42 pp is causally attributable to the first wrong token** (72.2% → 30.2%)
* **≈30 pp is not** — residual sample difficulty and later independent errors, which one
  correction does not touch.

## Two honest qualifications

1. **The token metric overstates divergence.** In 117 of 400 samples (29%) the original arm
   has >50% downstream token error but ADE < 0.15 m: the model emits GT's own tokens
   *shifted by one step* rather than diverging. Exact-position token error counts that as
   total failure. The ADE column is the truer measure — and it moves the same way
   (0.298 → 0.068), so the conclusion is unchanged.
2. **This is causal about the token chain, not about perception.** It shows action-token
   errors propagate within the decoding sequence. It does **not** show why the first token
   was wrong, which is where the perception-to-action question actually lives.

Artefacts: `outputs/counterfactual/{counterfactual_raw.json, counterfactual_summary.json,
examples.txt}`, `figures/figF_counterfactual.png`.

---

# Addendum 2 — Which late-layer component writes the wrong action?

The decision forms in the last few layers (§6.1) and attention *allocation* does not
distinguish errors from correct steps (`outputs/attention/`). So the remaining candidate is
what the components **write** into the residual stream.

## Method

A Qwen2.5-VL decoder layer touches the residual stream twice:

```
h_in                                   layer input
h_attn = h_in   + self_attn(ln1(h_in))     after attention
h_out  = h_attn + mlp(ln2(h_attn))         after MLP  == layer output
```

Each state is read at the action query position through the model's **final RMSNorm and
lm_head** (fp32 — the bf16 matmul quantises logits to ~1/8, coarser than the effects being
attributed), giving

```
margin  = logit(GT action) − logit(strongest rival)
d_attn  = margin(h_attn) − margin(h_in)
d_mlp   = margin(h_out)  − margin(h_attn)
```

The rival is the token actually generated when that was wrong, and the runner-up when the
generation was correct — the same quantity in both groups. **n = 475** step-0 probes
(221 first-action-error, 254 correct) over 28 logs.

## Result

| Layer | Component | Correct Δmargin | Failure Δmargin | Difference |
|---|---|---:|---:|---:|
| L28 | Attention | +0.0661 | +0.0162 | −0.0499 |
| L28 | **MLP** | +0.0910 | −0.0470 | **−0.1380** \* |
| L29 | Attention | +0.0082 | −0.0017 | −0.0099 |
| L29 | **MLP** | +0.0731 | −0.0787 | **−0.1518** \* |
| L30 | Attention | +0.0109 | +0.0302 | +0.0193 |
| L30 | **MLP** | +0.2015 | −0.0906 | **−0.2921** \* |
| L31 | Attention | +0.0471 | −0.0396 | −0.0867 |
| L31 | **MLP** | +0.1864 | −0.1417 | **−0.3281** \* |
| L32 | Attention | +0.0029 | −0.0114 | −0.0144 |
| L32 | **MLP** | +0.4033 | −0.0648 | **−0.4681** \* |
| L33 | Attention | +0.0399 | −0.0441 | −0.0839 |
| L33 | **MLP** | +0.4572 | −0.2168 | **−0.6740** \* |
| L34 | Attention | +0.1842 | +0.0074 | −0.1769 \* |
| L34 | **MLP** | +0.8975 | −0.3831 | **−1.2806** \* |
| L35 | Attention | +0.0748 | +0.0888 | +0.0140 |
| **L35** | **MLP** | **+3.1289** | **−1.5263** | **−4.6552** \* |

\* significant at BH FDR 5% across the 16-cell grid. Difference = failure − correct.

Summed over the last four layers (L32–L35):

| component | correct | failure | difference | p |
|---|---:|---:|---:|---:|
| Attention | +0.302 | +0.041 | −0.261 | 0.021 |
| **MLP** | **+4.887** | **−2.191** | **−7.078** | ≈0 |

The MLP accounts for **27× more group separation than attention**. Every MLP cell is
significant; only one attention cell (L34) is, and it is 7× smaller than the L34 MLP.

**Where the gap opens.** The failure group enters L32 only 1.41 logits behind; it leaves
L35 8.75 logits behind. **84% of the final margin gap is created inside the last four
layers**, and almost all of it by their MLPs.

## What the MLP is writing (n = 300)

Raw L34/L35 MLP output vectors at the action position:

| | L35 correct | L35 failure |
|---|---:|---:|
| ‖output‖ | 353.7 | 336.8 |
| cos(group means) | \multicolumn{2}{c}{+0.836} |
| projection on (w_GT − w_rival) | **+31.2** | **−15.1** |
| alignment with w_GT | 91.99 | 67.56 |
| alignment with w_rival | 73.03 | **76.42** |

The failure MLP is **not weak or inert** — its norm is 95% of the correct case and it points
in broadly the same region (cos 0.84). What flips is the *balance*: in correct samples its
projection onto the GT-minus-rival direction is **+31.2**, in failures **−15.1**. Alignment
with the GT readout row falls (92.0 → 67.6) while alignment with the rival's row rises
(73.0 → 76.4).

## Answer

> **The first AutoVLA action error emerges primarily from late-layer MLP transformation,
> overwhelmingly concentrated in the final layer's MLP.**

Attention is essentially exonerated on two independent measures: it neither *allocates*
differently (no step-matched span difference, all p > 0.2) nor *writes* differently
(summed Δmargin difference −0.26 vs the MLP's −7.08). The culprit is the L35 MLP, which
contributes **+3.13** logits toward the correct action in successes and **−1.53** against it
in failures — a swing of 4.66 logits, larger than every other component combined.

This also explains the earlier null result on ORION: an intervention on the residual stream
in the middle of the network is acting long before the layer that actually decides.

**Caveats.** (1) This is an attribution, not an ablation: it shows which component moves the
margin, not that suppressing it would fix the trajectory. (2) `d_attn` and `d_mlp` are not
strictly independent — the MLP reads the post-attention state, so attention can influence
the margin indirectly through it. (3) The final-layer MLP is expected to do the most work in
any transformer; the informative part is the *group difference*, not the raw magnitude.

Artefacts: `outputs/decomposition/{decomposition_raw.json, decomposition_summary.json,
mlp_direction.json}`, `figures/figG_late_layer_decomposition.png`.

---

# Addendum 3 — Causal intervention on the L35 MLP

Addendum 2 attributed the first action error to the final-layer MLP. Attribution is not
causation, so this edits that component directly.

## Method

At the step-0 action query position the L35 MLP output is replaced or steered; the resulting
token is taken; then the hook is removed and generation continues freely. One component, one
position, everything downstream is the model's own decoding.

| arm | edit at the action position | uses GT? |
|---|---|---|
| `baseline` | none | – |
| `mean_swap` | output ← mean MLP output of **correct** samples | no |
| `null_swap` | output ← mean MLP output of **failure** samples (negative control) | no |
| `steer` | + α·d̂, d = mean_correct − mean_failure | no |
| `random` | + α·r̂, r random at the **same norm** as d | no |
| `oracle` | + α·(w_GT − w_rival) direction | **yes — ceiling only** |

Group means and d are fit on **14 logs disjoint from the 14 test logs**, so the
intervention never sees the scenes it is scored on. α scales by ‖d‖ = 119.9 (the MLP output
norm is ≈354, so α=1 moves ~34% of it). **n = 94 failures + 60 correct controls.**

## Results

| Arm | step-0 → GT | downstream err | ADE | coarse ok | broke controls |
|---|---:|---:|---:|---:|---:|
| baseline | 3.2% [1,9] | 77.9% | 0.696 | 73.4% | 0% |
| mean_swap | 13.8% [8,22] | 75.4% | 0.806 | 69.1% | **15%** |
| null_swap *(neg. control)* | **14.9%** [9,23] | 74.9% | 0.752 | 71.3% | 10% |
| steer α=1 | 4.3% | 78.5% | 0.648 | 75.5% | 0% |
| random α=1 | 5.3% | 76.4% | 0.687 | 74.5% | 0% |
| steer α=2 | 8.5% | 77.0% | 0.635 | 75.5% | 2% |
| random α=2 | 3.2% | 77.2% | 0.702 | 73.4% | 2% |
| steer α=4 | 11.7% | 75.5% | 0.634 | 73.4% | 2% |
| random α=4 | 9.6% | 75.4% | 0.663 | 73.4% | 0% |
| **oracle α=1** | **89.4%** [82,94] | 44.6% | 0.278 | 77.7% | **92%** |
| **oracle α=2** | **96.8%** [91,99] | 43.3% | 0.206 | 77% | 92% |
| **oracle α=4** | **98.9%** [94,100] | 41.8% | 0.180 | 80% | 92% |

## Two findings, and they point in opposite directions

**1. The L35 MLP is a causally sufficient lever.** Writing the right direction into it flips
the first action token in **89–99%** of failures (McNemar vs baseline p = 1.7e-23 at α=1),
and the effect carries downstream through free generation: downstream error 77.9% → 41.8%,
ADE 0.696 → 0.180. The attribution in Addendum 2 was pointing at a real control point.

**2. No ground-truth-free lever was found.** Every GT-free arm fails its own control:

* `steer` never beats matched `random` — 4.3% vs 5.3% (p=1), 8.5% vs 3.2% (p=0.13),
  11.7% vs 9.6% (p=0.77).
* `mean_swap` (13.8%) is **indistinguishable from `null_swap` (14.9%)** — and null_swap
  writes the *failure*-group mean, which should help least. Both simply destroy the
  sample-specific MLP output; the small gain over baseline is from erasing it, not from the
  direction being right.

**3. The oracle forces rather than repairs.** It breaks **92% of already-correct samples**
(55/60), and it barely moves coarse action (73.4% → 79.8%) because most failures are
adjacent-category boundary cases (Addendum 1). A high fix rate bought at that collateral
cost is not a mitigation.

## Answer

> **The final-layer MLP is where the first action error is written, and editing it is
> sufficient to change the outcome — but what must be written is sample-specific. No shared
> linear direction recovers it.**

This is the same wall the ORION intervention hit, now localised precisely: the failure is
not that we were editing the wrong place (we now know the right place, and it works), but
that the correction is not a single direction in activation space. Difference-of-means
steering has now failed in two architectures, at two different loci, against matched
controls — that estimator should be retired for this problem.

**Caveats.** (1) The oracle uses per-sample GT and is a ceiling, not a method. (2) α up to
4·‖d‖ moves >100% of the MLP output norm, so the steer arm was not starved for magnitude.
(3) Only the step-0 position and only L35 were edited; a distributed edit across the last
few layers was not tried.

Artefacts: `outputs/mlp_intervention/{mlp_intervention_raw.json,
mlp_intervention_summary.json}`, `figures/figH_mlp_intervention.png`.

---

# Addendum 4 — One direction, a few modes, or sample-specific?

Addendum 3 found the L35 MLP is a causally sufficient lever that no *global* direction can
pull. The hypothesis under test here: the correction is not one direction but several,
conditioned on the action.

## The premise does not survive contact with the data

The proposed modes (`STOP→ACCELERATE`, `MAINTAIN→DECELERATE`, …) barely exist at step 0.
Classifying each action token by its own motion primitive (`src/labeling/token_primitive.py`
— displacement of the token's segment, speed relative to the ego's current speed):

| step-0 failure mode | n |
|---|---:|
| **MAINTAIN → MAINTAIN** | **167** |
| ACCELERATE → ACCELERATE | 16 |
| STOP → STOP | 13 |
| MAINTAIN+LEFT → MAINTAIN | 6 |
| ACCELERATE → MAINTAIN | 5 |
| …9 further modes | ≤ 4 each |

**74% of failures keep the same primitive**, and only 2 of 14 modes reach n ≥ 15. Re-cutting
by the *signed* geometric error is even more degenerate: 213/225 (95%) fall in a single
"same speed, same lateral" cell, with displacement error sd of just 0.135 m longitudinal /
0.082 m lateral over the 0.5 s segment.

The reason is the codebook. The predicted token is the GT token's **8th nearest neighbour
(median)** among 2,048 entries, **0.079 m** away, where typical nearest-neighbour spacing is
**0.043 m**. A step-0 "error" is a quantisation tie between adjacent codebook entries, not a
wrong driving decision.

## Geometry of the required corrections

For each failure the correction the readout actually needs is `w_GT − w_predicted`
(n = 225):

| quantity | value |
|---|---|
| mean pairwise cosine | **+0.0019** (median +0.0007) |
| pairs with \|cos\| > 0.5 | 0.5% |
| mean cosine with the global mean direction | +0.080 |
| PCA: PC1 / PC1–3 / PC1–10 | 9.4% / 15.6% / 30.0% |
| participation ratio (effective dimensions) | **53.9** |

Grouped by GT primitive, within-group cosine barely exceeds between-group:

| primitive | n | within | between |
|---|---:|---:|---:|
| MAINTAIN | 173 | **+0.0006** | +0.0019 |
| ACCELERATE | 25 | +0.046 | +0.004 |
| STOP | 15 | −0.034 | +0.000 |
| MAINTAIN+LEFT | 8 | +0.187 | +0.002 |

The dominant mode (MAINTAIN, 77% of failures) shows **no** within-mode coherence at all.

## Mode-conditioned steering (n = 87 failures, 40 controls, log-disjoint fit)

Directions fit per GT primitive on held-out logs; the three mode directions are mutually
near-orthogonal in MLP space (cos −0.005 to −0.031).

| Arm (α=4) | first-token fix | downstream | ADE | broke controls |
|---|---:|---:|---:|---:|
| baseline | 3.4% [1,10] | 77.7% | 0.589 | 0% |
| random | 6.9% [3,14] | 75.9% | 0.574 | 0% |
| wrong mode | 10.3% [6,19] | 72.9% | 0.618 | 0% |
| global | 12.6% [7,21] | 74.7% | 0.588 | 2% |
| **own mode** | **16.1%** [10,25] | 72.9% | 0.498 | 2% |
| *oracle (per-sample GT)* | *98.9%* | *41.8%* | *0.180* | *92%* |

Per mode at α=4:

| Failure mode | n | Global fix | Mode fix | Wrong-mode fix | Random |
|---|---:|---:|---:|---:|---:|
| MAINTAIN | 72 | 11.1% | 13.9% | 9.7% | 6.9% |
| ACCELERATE | 10 | 20.0% | 30.0% | 10.0% | 10.0% |
| STOP | 5 | 20.0% | 20.0% | 20.0% | 0.0% |

Tests (McNemar, paired):

* own mode vs random: 16.1% vs 6.9%, **p = 0.039** — a real but small effect
* own mode vs **wrong mode**: 16.1% vs 10.3%, **p = 0.30** — not distinguishable
* global (12.6%) also beats random, so "global ≈ random" fails too

## Answer

> **No. AutoVLA first-action errors are not governed by a small set of action-conditioned
> late-MLP directions. The required correction is essentially sample-specific.**

The pre-registered signature for the mode hypothesis was
`global ≈ random`, `wrong-mode ≈ random`, `own-mode ≫ random`. Two of the three fail: the
global and wrong-mode directions both beat random, and own-mode does not separate from
wrong-mode (p = 0.30). What survives is a **weak, graded action-conditioned component** —
own mode > global > wrong mode > random, with only the mode-vs-random gap reaching
significance — on top of a correction space with **~54 effective dimensions** and per-sample
directions that are essentially orthogonal.

Put together with the oracle's 98.9%: the lever exists and is precisely located, but the
signal that tells you *which way to pull it* is per-sample, not per-action-class. Any
mitigation will have to predict the direction from the input, not look it up from a mode
table.

**A reframing this forces.** Because step-0 errors are codebook ties 0.079 m apart, the
right question may not be "why is the token wrong" but "why does a 0.08 m tie propagate into
a 0.59 m trajectory error". The counterfactual (Addendum 1) already showed it does; the
tokenisation granularity, not the perception-to-action link, is the lever that story points
at.

**Caveats.** (1) n = 87 tested failures gives ~±8 pp resolution, so a mode effect smaller
than ~15 pp would be invisible. (2) Only 3 modes had enough support; a finer partition of
MAINTAIN was not tried. (3) Directions were fit on 60 correct + ~20 failure samples per
mode; more data would sharpen them, though the near-zero within-mode cosine suggests there
is little to sharpen.

Artefacts: `outputs/mode_steering/{mode_steering_raw.json, mode_steering_summary.json}`,
`figures/figI_mode_directions.png`, `src/labeling/token_primitive.py`.
