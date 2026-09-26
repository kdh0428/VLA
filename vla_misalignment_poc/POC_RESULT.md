# PoC result — Correct Perception, Wrong Action in ORION

**Question.** Can a driving VLA encode scene-level perceptual information correctly and still
emit an inconsistent action — and where inside the model does perception decouple from action?

**Short answer.** Yes — descriptively. In the failure group the scene stays decodable at
0.83–0.91 AUROC through the early and middle decoder, while the *correct* action is not
(0.45–0.79) even though a *confident wrong* action is (0.94–0.96). But the causal test failed: editing that representation recovers
no more trajectories than a random edit of the same size, so the divergence is so far a
**marker, not a demonstrated cause** (§6).

Everything below is from a single reproducible run; commands, seeds and versions are in
`README.md` and in each `run_meta.json`.

---

## 1. Dataset and sample statistics

| | |
|---|---|
| Model | official ORION `Orion.pth`, config `configs/orion_poc_cot_fp16.py` (CoT + fp16) |
| Data | Bench2Drive official **val** split, 50 clips; Chat-B2D `val` annotations |
| Frames available | 12,806 (Chat-B2D val == b2d_infos_val.pkl, exact match) |
| **Frames analysed** | **1,809** over **49 clips / 42 scenarios** |
| Warm-up frames run | 5,771 (vision-only, to keep the temporal memory exact) |
| Inference failures | **0** |
| Hidden states | 33 tensors (embedding + 32 blocks) × 4096, fp16, per frame |
| Merge integrity | 1,809 requested / 1,809 present / 0 duplicates / 0 trajectory defects |

Sampling was stride-6 temporal subsampling + scenario stratification + whole-clip shard
assignment (never random frames). All 470 available pedestrian-hazard frames were kept.

**Open-loop sanity.** ADE median 0.686 m, mean 1.216 m; FDE median 1.182 m. Consistent with
ORION's published 0.68 L2@2s, i.e. the baseline is reproduced rather than degraded.

---

## 2. Failure taxonomy

| group | n | share |
|---|---|---|
| **P+R+A+** (control) | 458 | 25.3% |
| **P+R+A−** (main failure group) | **262** | **14.5%** |
| P−R+A+ | 241 | 13.3% |
| P+R−A+ | 205 | 11.3% |
| P−R+A− | 197 | 10.9% |
| P+R−A− | 157 | 8.7% |
| P−R−A+ | 148 | 8.2% |
| P−R−A− | 136 | 7.5% |
| undecidable (R unparsed) | 5 | 0.3% |

### Prevalence of the target phenomenon

| | count | rate | 95% CI (Wilson) |
|---|---|---|---|
| **P+A−** | **419 / 1,809** | **23.2%** | [21.3%, 25.2%] |
| **P+R+A−** | **262 / 1,804** | **14.5%** | [13.0%, 16.2%] |

**P+A− accounts for 55.6% of all action failures** — the majority of wrong trajectories occur
while observable perception is correct.

### Not an artefact of thresholds

Across a full ADE × FDE sweep the phenomenon never disappears:

| ADE / FDE | 1.5 | 3.0 | 5.0 |
|---|---|---|---|
| **0.5** | 34.9% | 34.9% | 34.9% |
| **1.0** | 29.7% | 27.7% | 27.7% |
| **1.5** | 28.4% | **23.2%** (default) | 22.6% |
| **2.0** | 28.2% | 21.8% | 20.2% |
| **3.0** | 28.2% | 21.4% | 19.1% |

P+A− stays in **19.1%–34.9%**; P+R+A− never drops below 220 samples.

### Not confined to one scenario

P+A− appears in **39 of 42 scenarios**. Highest: EnterActorFlow 70/78 (90%),
YieldToEmergencyVehicle 24/44 (55%), HazardAtSideLaneTwoWays 55/131 (42%). Lowest:
PedestrianCrossing 0/85, VehicleTurningRoute 0/32.

> **Confounder flagged.** EnterActorFlow at 90% is an outlier and probably reflects
> merging manoeuvres where the GT trajectory is one of several acceptable behaviours, so
> an L2/semantics mismatch overstates "wrong". The phenomenon does not depend on it —
> excluding EnterActorFlow entirely still leaves 349 P+A− cases across 38 scenarios.

### Safety-conditioned behaviour

On the 1,135 frames where the scene imposes a hard requirement (red light, close lead
vehicle, or pedestrian hazard), the predicted trajectory satisfies it in only
**56.1%** of cases (CI [53.2%, 59.0%]). The most frequent semantic failures are
**STOP → ACCELERATE** (95 cases) and **STOP → MAINTAIN** (31 flagged unsafe).

**Gate: passed on all three criteria** (sufficient P+A−, recurring P+R+A−, not
scenario-specific).

---

## 3. The main result — perception preserved, action absent

Linear probes on the **planning token** (`<waypoint_ego>`), clip-grouped CV, scaler fit on
train folds only, 3 seeds, label-shuffled null on the same folds.

### 3.1 Perception stays decodable in the failure group

| label | group | L1 | L8 | L16 | L24 | L32 | shuffled |
|---|---|---|---|---|---|---|---|
| `lead_vehicle` | P+R+A+ | 0.80 | 0.86 | 0.86 | 0.89 | 0.82 | 0.48 |
| | **P+R+A−** | **0.87** | **0.86** | **0.91** | 0.86 | 0.76 | 0.43 |
| `critical_motion` | P+R+A+ | 0.85 | 0.95 | 0.93 | 0.91 | 0.87 | 0.52 |
| | **P+R+A−** | **0.83** | **0.91** | 0.83 | 0.81 | 0.73 | 0.60 |

Through the early and middle decoder the failure group encodes the scene **as well as the
control** — for `lead_vehicle` it is actually *higher* at L16 (0.91 vs 0.86). The scene is
not misperceived.

### 3.2 The *correct* action is not encoded — but a *different* action is

> **Correction (2026-09-09).** An earlier version of this section concluded that failure
> samples "do not contain that decision". That inference was wrong, and the error was
> circular: A± is *defined* by agreement between the model's action and the GT action, so
> decoding the GT action necessarily succeeds in A+ and fails in A−. Re-running the probe
> for the model's OWN predicted action settles it.

| label | group | L1 | L8 | L16 | L24 | L32 | shuffled |
|---|---|---|---|---|---|---|---|
| `action_stop_or_slow` (GT) | P+R+A+ | 0.93 | 0.93 | 0.98 | 0.99 | **0.99** | 0.51 |
| | **P+R+A−** | **0.55** | **0.52** | 0.64 | 0.81 | 0.76 | 0.50 |

Decoding the GT action vs the model's own action, same hidden states, same folds:

| layer | ctrl : GT | ctrl : own | **fail : GT** | **fail : own** |
|---|---|---|---|---|
| L8 | 0.939 | 0.939 | 0.452 | **0.688** |
| L16 | 0.983 | 0.983 | 0.619 | **0.939** |
| L24 | 0.991 | 0.991 | 0.792 | **0.946** |
| L32 | 0.991 | 0.991 | 0.746 | **0.963** |

Two things follow.

* In the control group the two labels are *identical by construction* (GT == own), so the
  0.93–0.99 column carries no independent information. It should not be quoted as evidence.
* In the failure group the model's own action is decodable at **0.94–0.96** from L16 on,
  while the GT action sits at 0.45–0.79. The planning token is **not** missing an action
  decision — it holds a confident one that happens to be wrong.

So the honest statement of the dissociation is:

> Perceptual evidence is encoded (0.83–0.91) **and** an action is encoded (0.94–0.96), but
> the encoded action is not the one the evidence implies.

This is a perception-to-action **utilisation** failure, not an information-acquisition
failure. It also makes the negative intervention result (§6) less surprising: if the model
has already committed confidently to a different action by mid-stack, nudging the planning
token along a group-mean axis is unlikely to flip it.

### 3.3 Where the representations split

Failure-vs-control probe and geometry on the planning token, GT-action-matched
(234 pairs, never pairing frames from the same clip):

| layer | failure AUROC (matched) | shuffled | centroid cos | centroid L2 (SD units) | ρ(score, ADE) |
|---|---|---|---|---|---|
| 1 | 0.647 | 0.541 | 0.9995 | 15.2 | +0.10 |
| 8 | 0.655 | 0.549 | 0.9941 | 13.7 | −0.11 |
| 12 | 0.621 | 0.544 | 0.9936 | 17.5 | −0.14 |
| **15** | **0.697** | 0.538 | **0.9591** | **24.5** | −0.22 |
| 20 | 0.709 | 0.530 | 0.9310 | 28.1 | −0.33 |
| 24 | 0.727 | 0.519 | 0.8754 | 28.5 | −0.45 |
| 32 | 0.682 | 0.495 | 0.8364 | 26.4 | −0.41 |

Two distinct events, and they are not the same thing:

* **From L1** the failure is already weakly decodable (AUROC ≈ 0.65 vs 0.54 null) while the
  two clouds are geometrically almost identical (cosine 0.9995). Some failure predisposition
  is present from the start — a scene property, not a late breakdown.
* **At L11–L15 the representations actually separate**: the centroid distance jumps ~45%
  (17.0 → 24.5 SD units) and the cosine drops 0.994 → 0.959. The automatic detector puts the
  geometric transition at **L11** (action-matched) / L9 (raw).

The split then widens monotonically to L24–L32, which is also where perception decodability
in the failure group starts to *fall* (lead vehicle 0.91@L16 → 0.76@L32).

---

## 4. Confounders found and controlled

1. **Group composition.** Control and failure groups occupy largely different scenarios —
   matching on (scenario, GT action) leaves only 12 pairs. We therefore report a
   **GT-action-matched** comparison (234 pairs, primary) and a scenario-matched one
   (43 pairs, low power, inconclusive). The divergence survives action matching.
2. **Is the geometry just restating the error?** ρ(score, ADE) is ≈ 0 for L1–L14 and grows
   to −0.45 by L24. So the *early* separation is not a re-description of trajectory error,
   but the **late-layer geometry partly is**, and should not be read as independent evidence.
3. **Clip-level label concentration — important.** Under clip-grouped CV some labels cannot
   demonstrate cross-scene generalisation at all:

   | label | clips with positives | top-3 clip share | usable |
   |---|---|---|---|
   | `critical_motion_moving` | 49 | 26% | ✅ |
   | `lead_vehicle` | 43 | 27% | ✅ |
   | `critical_side_left` | 42 | 33% | ✅ |
   | `traffic_light_red` | 11 | 56% | ⚠️ flagged |
   | `pedestrian` | 11 | 60% | ⚠️ flagged |

   Traffic-light and pedestrian probes score 0.92–0.99 but their positives sit in ~11 clips,
   so those numbers may encode clip identity. **They are reported but excluded from the
   headline claim**, which rests only on `lead_vehicle` and `critical_motion`. A
   frame-random split would have hidden this entirely.
4. **Traffic light is not asked of the model.** The CoT config has no traffic-light round, so
   the model's light state can only come from its critical-object list. P comparison uses the
   object-derived value on *both* sides; presence is scored always, colour only when both
   sides name one (GT's own two sources disagree on colour in ~11% of frames).
5. **L0 is degenerate by construction** — every sample's `<waypoint_ego>` embedding is the
   same vector, so AUROC is exactly 0.500. A useful sanity check that the pipeline is sound.
6. **Planner stochasticity.** ORION's VAE planner samples noise and `generate()` uses
   `do_sample=True, temperature=0.1`, so trajectories are not bit-reproducible; conclusions
   are reported at group level.

---

## 5. Interpretation

Mapping onto the pre-registered outcomes: **Outcome A for the observational claim**, with
two qualifications (an early-layer signal, and a failed causal test — §6).

* P+R+A− is frequent (14.5%, n=262) and P+A− is the majority of all action failures (55.6%).
* Correct scene information *is* present in the failure group's planning token at
  0.83–0.91 AUROC through the middle decoder — equal to the control.
* The *correct* action is not encoded at those layers (0.45–0.79), **but a confident
  wrong action is** (own-action AUROC 0.94–0.96 from L16). The failure is misalignment
  between the two, not absence of an action decision — see the correction in §3.2.
* The representations geometrically separate from ~L11–L15.

The qualification: a weak failure signal is decodable from L1 onward, so this is not purely
a late-stage breakdown. The honest reading is that some failures are **predisposed by the
scene from the very first layers**, and the representation then never acquires the correct
action, rather than acquiring it and losing it. The *geometric* divergence is nonetheless
mid-network (L11–L15), well before the planner.

This does not yet establish causality — that is what §6 tests.

---

## 6. Causal intervention — **negative**

Activation patching on the planning token at L8/L16/L24/L30, letting the remaining blocks
run normally before the planner:

```
bad input → blocks 1..l-1 → REPLACE h_l[<waypoint_ego>] → blocks l..31 → planner → trajectory
```

25 P+A− failures and 25 already-correct controls, 1,000 interventions, 98.5 min on one GPU.

| mode | recovered (failures) | rate | 95% CI | broke (controls) | rate |
|---|---|---|---|---|---|
| hard patch (donor) | 7 / 100 | 7.0% | [3.4%, 13.7%] | 4 / 100 | 4.0% |
| **steering** (mean-diff) | 15 / 200 | **7.5%** | [4.6%, 12.0%] | 19 / 200 | 9.5% |
| **random direction** (control) | 11 / 200 | **5.5%** | [3.1%, 9.6%] | 17 / 200 | 8.5% |

* steering vs random: **Fisher exact p = 0.544** (OR 1.39)
* hard patch vs random: **p = 0.613** (OR 1.29)
* mean ΔADE on failures: steer −0.196, random −0.116, patch −0.154
* mean ΔADE on controls: steer +0.161, random +0.135, patch +0.131

**No intervention beats its random-direction control.** The interventions are not inert —
they perturb behaviour (4–9.5% of already-correct samples break, ΔADE moves in both groups)
— but the movement is undirected: *any* perturbation slightly reduces error on failures and
slightly increases it on controls, which is regression toward a generic trajectory prior,
not targeted correction.

### A first run that was underpowered, and the fix

The first attempt used a **unit-normalised** steering direction with fixed α. Because the
residual-stream norm grows steeply with depth, that perturbed the state by only

| layer | ‖h‖ | α=1 as % of ‖h‖ (v1) | α=1 with raw mean-diff (v2) |
|---|---|---|---|
| 8 | 10.9 | 9.2% | 9.6% |
| 16 | 26.1 | 3.8% | 21.8% |
| 24 | 50.3 | 2.0% | 33.0% |
| 30 | 95.5 | **1.1%** | 29.2% |

and its donor pool, keyed on (scenario, full GT action), matched only **1 of 25** failures,
so hard patching effectively never ran. Both were fixed (raw mean-difference direction;
donors keyed on the GT longitudinal action, pool 23–366 per class) and the experiment re-run
in full — the table above is the corrected run. The underpowered run is kept at
`outputs/intervention_v1_underpowered/` for the record. **The conclusion did not change.**

### What this does and does not license

* It does **not** show that the mid-network divergence is causally responsible for the wrong
  trajectory. The mean-difference direction is not a causal handle.
* It does **not** show the opposite either. Three limitations bound the test: only the
  `<waypoint_ego>` token was edited (the failure may be distributed across visual/reasoning
  tokens); n = 25 failures gives roughly ±5 pp resolution, so an effect smaller than ~10 pp
  would be invisible; and only one direction family (difference-of-means) was tried.

Layer-wise decoding of intermediate states straight into the planner is deliberately **not**
used as causal evidence: the planner only ever saw final-layer statistics, so that query is
out of distribution.

---

## 7. Verdict and next step

**The descriptive claim is supported; the causal claim is not.**

Supported, with controls:

* P+A− is real, frequent (23.2%, n=419), robust to thresholds (19–35% across the sweep),
  and spread over 39 of 42 scenarios — it is the majority (55.6%) of all action failures.
* In the failure group the scene remains decodable from the planning token at 0.83–0.91
  AUROC through the middle decoder, matching the control — the model is not misperceiving.
* The correct action is at chance (0.48–0.55) in those same layers for the failure group,
  while the control reaches 0.93–0.99. This dissociation is the finding.
* The two groups' planning tokens separate geometrically at **L11–L15**, and this survives
  GT-action matching.

Not supported:

* Editing the planning token — by donor patch or by mean-difference steering — recovers no
  more failures than a random direction of the same magnitude (7.5% vs 5.5%, p = 0.54).
  On this evidence the representational difference is a **marker**, not a demonstrated cause.

**Is a follow-up worth it? Yes, but the design must change.** The observational result is
strong enough to justify continuing, and the negative intervention narrows the search
usefully: whatever produces the wrong action is not recoverable by moving a single token
along the group-mean axis. Concretely:

1. **Widen the intervention surface.** Patch the visual/critical-object token positions and
   the reasoning span, not only `<waypoint_ego>`. The dissociation says the action is missing
   from the planning token, which is as consistent with "it was never written there" as with
   "it is there but wrong" — patching upstream distinguishes these.
2. **Better directions.** Difference-of-means is the weakest estimator; a supervised
   direction (the trained action-probe weight vector, which reaches 0.99 AUROC on controls)
   is the natural next candidate, with an α-sweep and the same random control.
3. **Power.** 25 failures resolve roughly ±5 pp. A real test of a ~10 pp effect needs
   150–200 failure samples; that is ~6–8 GPU-hours with the current pipeline.
4. Only then does selective/risk-gated intervention become meaningful.

Caveats to carry forward: the traffic-light and pedestrian probes need a subset with more
distinct clips before they can support any claim; the late-layer geometric scores partially
track ADE (ρ ≈ −0.45) and cannot stand alone as mechanism evidence; and a weak failure signal
exists from L1, so this is not a purely late-stage phenomenon.

---

## 8. Figures

| file | content |
|---|---|
| `figures/figA_taxonomy.png` | failure-taxonomy distribution |
| `figures/figB_perception_probe.png` | layer-wise perception probe, control vs failure (low-clip-support panels dashed and annotated) |
| `figures/figC_action_divergence.png` | failure decodability, centroid distance, and the ADE-confound check |
| `figures/figC2_perception_vs_action.png` | headline overlay: perception preserved vs action diverging |
| `figures/figD_cases.png` | representative P+A− cases with GT vs predicted trajectories |
