## 2026-10-18 개정 (review_followup4 반영)

**우선순위.** `numbers.json`에 review_followup4 행 1,532개(`RF4.*`)를 추가했습니다(`autovla_misalignment_poc/scripts/review_followup4/c/c_numbers.py`). 원래 행의 값은 바꾸지 않았고, 쓰임이 바뀐 행에는 `rf4_note`를 붙였습니다. 이 문서·`tables.tex`와 충돌하면 `numbers.json`과 `outputs/review_followup4/{STATUS, CLAIM_CHANGES, MANUSCRIPT_REVISION}.md`가 우선합니다. 수정한 줄에는 "(2026-10-18 수정)"을 표시했고, 원문은 `outputs/review_followup4/DOC_EDITS_APPLIED.csv`와 git에 있습니다.

**이번에 고친 것 (corrections).**
- **A1 (post-hoc correction):** sign-flip-inversion CIs for paired binary success are invalid at the boundary (zero width ±2.5e−13 or an end pinned at 0). Corrected intervals: Newcombe 10 (OpenVLA states), Rao–Scott design-effect Newcombe / MOVER (SpatialVLA clusters) — `RF4.A1.*`, Table 13. Judgment changes: OpenVLA C − N "CI excludes 0" deleted ([−17.1, +1.0]); OpenVLA per-task CIs are descriptive only; exp34 perp_left coke C − F → ±10 pp unadjusted only; P1-D r2e1 − r1e1 C − F interaction (physical) not established; r1e4 C − F move_near physical ±5 pp lost; task-level CI widths **18–25 pp (r4e1; r1e4 13–20 pp)** instead of 18–29 / 6.9–21.6. P1-D pooled ±10 pp, per-task 3 of 8 and S1–S4 pooled survive.
- **A2:** the correction window keeps its GT positions in all later contexts; nothing is released. Claim 5 wording changed to "three fixed context positions replaced with ground truth and retained in later contexts"; "post-release horizon" renamed "steps after the last added GT position".
- **A3:** Claim 8 (Impromptu) downgraded to descriptive / parity not established (numerical floor 5–6% scene-level verdict flips).
- **A4:** denominators and definitions (F = 105 scenes / 18 drives; P = 2,191 eligible scenes estimated from 353 sampled scenes with weights; "all" in the motion-baseline rows = 208 selected scenes; OpenVLA 92.6% = at least one downstream component ≥ injected bins; commanded vs TCP; heterogeneity not established from opposite point estimates; A−/severe formula with the 3 s thresholds applied at 5 s).
- **B3:** OpenVLA gate NOT_RUN (user decision 2026-10-17): OpenVLA stays descriptive and is excluded from the core evidence.

**새 claim과 표.** Claim 25 (B2 matched-prefix), Claim 26 (B1 numerical sensitivity), Claim 27 (A1 corrected intervals). `tables.tex`: Tables 8–11 captions corrected (machine-generated block), new Tables 12–14 (`c_tables.py`).
