# P0-A 감사 결과: 원고 주장과 코드·raw 대응 (2026-10-08)

Labels: [확인한 사실] = read from code or raw; [계산한 결과] = recomputed in this run; [미확인] = could not be checked; [가설] = interpretation. No existing file was changed and nothing was committed. Details are in claim_to_code.csv (79 rows), metric_spec.md, intervention_trace.jsonl (102 rows), denominator_flow.csv, unresolved_items.md and manifest.json.

## Findings, ordered by impact

**F1. "Reverse" is a different intervention depending on the model and the experiment [확인한 사실].**
- **AutoVLA exp 8 reverse@L** (prev_action_state_patching.py:99-100,174-189). The base is the GT-history context. At every step k≥t*+2, the hidden state at position k−1 is copied from the **separate Normal row in the same batch**.
  - The tiny re-run trace shows this: in rerun_tiny/exp8 alt0 at k=5, the injected token is Normal's 568, while the reverse row's own output was 268.
- **SpatialVLA token level** (svla_offline.py:111) also takes N3 from the Normal row. In **SpatialVLA closed loop**, reverse means executing natural with a perturbed context, which is a different intervention.
- **Impromptu** reinserts the reverse branch's **own** k−1 waypoint (exp_cde.py context()).
- Impact: the cross-model "Reverse" row (Claim 11, paper_cross_model_table) puts different source branches and mechanisms under one name.

**F2. Recent-GT is not "replace one token once". It is a sliding correction re-applied at every step [확인한 사실 + 계산한 결과].**
- Code: action_history_causal.py:210-212; exp 32 uses the same rule.
- It is applied 8−t* times per unit (A− mean 6.8). The token actually changed 4.32 times on average; in 4 units it never changed.
- **Exp 9 win_w** puts GT at **fixed absolute positions** and keeps it in context afterwards (temporal_feedback_window.py:180-191). So win1 (47%) and Recent-GT (93%) are different interventions.
- **Exp 5** is the only true single correction, and it is output-level: GT is part of the executed trajectory. Exps 7–9 and 32 change context only; the executed output is always the model's own (action_history_causal.py:236).
- The wording in EXPERIMENT_SUMMARY l.17-18 and Claim 3 should be corrected.

**F3. SpatialVLA "trajectory divergence (m)" is a cumulative-sum proxy of commanded actions, not the actual TCP [확인한 사실 + 계산한 결과].**
- Code: analyze_svla.py:68-69,85-87; analyze_protection.py:73-74,94. Exp 34 did not save the TCP.
- In exp 35, where both exist, the commanded proxy is about 4× the real TCP motion (0.84–0.95 vs 0.21–0.23 m). For r1e4 reverse vs natural: proxy 0.24 m vs real 0.069 m.
- Affected: exp 34 "0.21–0.23 vs 0.14 m", Claim 10 "+0.09 m", Claim 17 "+0.116 m".
- Exp 35 position error (e.g. +1.84 cm) is the actual paired TCP distance, not a distance to GT.

**F4. Exp 35 "+1.8 cm when the ensemble is removed" [계산한 결과].**
- Raw comparison: r1e1−r1e4, F vs C mean TCP error +1.76 cm, p=0.028 uncorrected.
- After Holm over 5 configurations, p=0.14. Claim 17 says Holm over 4, but the code uses 5 (analyze_protection.py:19).
- Other ensemble-axis contrasts with p<0.05 are not reported (e.g. r1e2 final −1.37 cm, p=0.016).
- "+1.8 cm" in Claim 11 refers to a different number (r4e1 reverse +1.84 cm). "e2=e4" has no equivalence margin.

**F5. Exps 8, 9 and 5 have no main-run raw. Core numbers in Claims 3 (reverse 33.4%), 4 and 5 (47/76/90/96%) are summary-only [확인한 사실].**
- To produce traces, I re-ran 3 units each with the unmodified scripts (5090, about 2 min).
- This re-run validates the intervention structure only, not the original numbers.

**F6. Normal differs across experiments [계산한 결과].**
- Normal token agreement: exp 7 vs exp 6 71.6%; exp 8 88.6% and exp 9 89.2% (summary); exp 32 88.2% (raw), with the A− amplification label differing in 8 of 365 units.
- The model runs in bf16 and the batch numeric path differs between harnesses. Every effect ratio must use that experiment's own Normal. The CI for the 93% effect ratio comes from exp 8, whose Normal is 46.8%.

**F7. A− is a coarse action-class mismatch [확인한 사실].**
- label_A at pra_labels.py:469-485. Thresholds stated for 3 s are applied over 5 s.
- Amplification = A− ∧ FDE5>3 m. OOD exclusion applies per condition only.

**F8. Denominators confirmed [계산한 결과].**
- 2,748 → 2,747 → A− 52 (16 logs); exp 5 = 1,159 (A+ 1,107 / 28 logs); 208 scenes (A+ 26 logs) → 365/1,043 units.
- In exp 6, one A− unit is invalid. Exps 7 and 32 have no exclusions.
- SpatialVLA: 18,975/19,200 units; 9×80 closed-loop episodes; exp 35 2,080+1,280 episodes.

**F9. Raw recomputation matches the summaries exactly [계산한 결과].**
- Exp 7: 47.4/7.1/4.1%, McNemar 2 vs 149, ratio 93.0%.
- Exp 32: +9.0 pp (61 vs 28), mirror −34.2 pp, distances 0.318/0.505/0.458.
- Exp 34/35 analysis.json, curves.json and the pooled file match field by field.

**F10. NAVSIM [확인한 사실; import path 추론].**
- navhard EPDMS was computed with navsim_v2 at 0a380a9 (v2.2-1), which includes the 2025-09-29 human-filter fix 359c7f7 as an ancestor. All 75 runs set human_penalty_filter true.
- Held-out PDMS uses the vendored NAVSIM v1.1.0, which has no human filter, so the fix does not apply. Its upstream base commit is 미확인.
- The package does not record either commit.

**F11. Other items.**
- Exps 4–11: no git evidence of preregistration.
- Exp 34: the reference is the model's own greedy chunk, not GT. Reverse "fully restores" holds only for dev4; amplification is restored only partly (18.6 vs 22.1%).
- Exp 34 natural_mix: the protocol says "same batch", but it ran as a separate block [가설: weak control].
- No code computes exp 35 exposure. The values recomputed from chunk ages are 0.426/0.31/0/0.507/0.660/0.753.

## Completion status
- **Tied to raw:** the main table numbers for exps 4–7, 32, 33, 34 and 35, and their denominators.
- **Summary-only, not counted as complete:** exp 5, exp 8, exp 9, exp 10, and the exp 35 batch/executor checks.
- **Pseudo-code:** every intervention is written out in metric_spec.md.
