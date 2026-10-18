#!/usr/bin/env python
"""review_followup4 C: apply the review_followup4 factual / wording corrections to the manuscript-facing documents
(2026-10-18). CPU only.

Edits (each logged to outputs/review_followup4/DOC_EDITS_APPLIED.csv with file, line, old text, new text, source):
  1. the 64 A4 wording violations (outputs/review_followup4/A1_A4/A4_WORDING_VIOLATIONS.csv):
       - 57 applied in place to /root/VLA/EXPERIMENT_SUMMARY.md and /root/VLA/paper_quantitative_package/*.md, tables.tex
         (2 of them, tables.tex l. 230 / 240, sit inside the machine-generated review_followup3 block and are applied by
         c_tables.py);
       - 7 in /root/VLA/review_followup3/{CLAIM_EVIDENCE_MATRIX.md, CP2_REPORT.md} are NOT edited (historical round
         reports); their replacements are listed in outputs/review_followup4/MANUSCRIPT_REVISION.md and superseded by
         CLAIM_CHANGES.md.
  2. A1 / A2 / A3 / B2 / B3-driven sentence edits (EXTRA below).
  3. new dated sections: EXPERIMENT_SUMMARY §6g, PAPER_NUMBERS "2026-10-18 개정" + Claims 25-27, MANUSCRIPT_REVISION
     banner + "2026-10-18 revision (review_followup4)", README note.
Every edited line gets the marker "(2026-10-18 수정)" (tables.tex: a LaTeX comment), so the history stays visible; the
old text of every edit is in DOC_EDITS_APPLIED.csv and in git. Each A4 replacement must find its quoted text on the
cited line. Idempotent: an edit whose new text is already present is skipped; sections are inserted once.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python c_doc_fixes.py
"""
import csv
import json
import os

ROOT = "/root/VLA"
PKG = f"{ROOT}/paper_quantitative_package"
R4 = f"{ROOT}/autovla_misalignment_poc/outputs/review_followup4"
ES = f"{ROOT}/EXPERIMENT_SUMMARY.md"
MR = f"{PKG}/MANUSCRIPT_REVISION.md"
PN = f"{PKG}/PAPER_NUMBERS.md"
TX = f"{PKG}/tables.tex"
MK = " (2026-10-18 수정)"
MK_TEX = "  % corrected 2026-10-18 (review_followup4)"
SECTIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doc_sections")

A4 = list(csv.DictReader(open(f"{R4}/A1_A4/A4_WORDING_VIOLATIONS.csv")))

# ---------------------------------------------------------------------------------------------------------------------
# A4: index -> (old text on the cited line (must contain the A4 quoted text), new text). None = handled elsewhere.
P16 = A4[16]["proposed_replacement"]
A4_EDITS = {
    0: ("p·CI는 cluster sign-flip(역산 CI)",
        "p는 cluster sign-flip(δ = 0에서 유효)이고 CI는 사후 수정 구간(Rao–Scott design-effect Newcombe 10, 과제 평균은 MOVER; "
        "원 sign-flip 역산 CI는 이진 쌍 자료의 경계에서 무효, review_followup4 A1)"),
    1: ("기전 실험과 drive가 겹치지 않는 **새 24 drive**(P1-C)에서 재현: 실패 조건부 RN −34.2%p, RV +26.8%p; 모집단 −4.1 / +3.6%p [사전 등록]. 단 day-vehicle 10개는 공유합니다.",
        "기전 실험과 drive가 겹치지 않는 **새 drive**(P1-C; 0 drive 공유, day–vehicle 10 공유)에서 재현 [사전 등록]: 실패 층 F(105 장면 / "
        "18 drive) RN −34.2%p, RV +26.8%p; 평가 집합 E 추정량(적격 2,191 장면, 선정 353 장면 가중, 24 drive) −4.1 / +3.6%p. RN 평균 효과의 "
        "약 60%는 A+(비실패) 층에서 오므로 원래 실패 장면의 복구로만 설명하지 않습니다."),
    2: ("전체에서는 +0.20 m(p = 0.51)로 외삽과 구별되지 않습니다",
        "선정 기전 집합 전체(208 장면 = A− 52 + A+ 156)에서는 +0.20 m [−0.32, +0.56](p = 0.51)로 차이가 검출되지 않았습니다(동등성 미검정)"),
    3: ("한 step 안에서 첫 token을 8 bin 바꾸면 나머지 차원이 93%에서 그 이상 움직이고",
        "한 step 안에서 x token을 ±8 bin(명령 약 2.6 mm/step) 바꾸면 단위의 92.6%(4,741 단위; 초기 상태 군집 CI [90.8, 94.0])에서 하위 "
        "성분(y, z, roll, pitch, yaw; gripper 제외) 중 적어도 하나의 bin 변화가 실제 주입 bin 이상이었고(translation 성분만으로는 72%; "
        "명령값이며 실현 운동·과제 실패가 아님)"),
    4: ("LIBERO 과제 실패에 대한 효과는 closed-loop 기저 변동(자연 재실행끼리도 결과 20% 불일치) 안에 있었습니다",
        "LIBERO 과제 성공률 효과는 검출되지 않았고(C − F 8-bin −3%p [−11.9, +5.9], Newcombe 10; 동등성 미성립), 자연 재실행끼리도 "
        "결과가 20% 달랐습니다(이 20%를 효과 검출 임계값으로 쓰지 않음)"),
    5: ("하위 차원 93% 증폭", "하위 성분 중 적어도 하나가 주입 bin 이상 변한 단위 92.6%(기술적)"),
    6: ("모집단 A− 1.24 → 2.60%로 악화",
        "dev 자연 평가 집합(2,747 장면: A− 52 / A+ 2,695)으로 재가중한 A− 빈도 1.24 → 2.60%로 악화(배포 빈도 아님)"),
    7: ("하위 차원이 δ 이상 움직이는 비율 **92.6% [89.8, 94.4]**",
        "하위 성분(y..yaw; gripper 제외) 중 적어도 하나의 bin 변화 절댓값이 실제 주입 bin(clip 후) 이상인 단위 비율(4,741 단위 = "
        "2,462 프레임 × 부호; bin은 m·rad 혼합) **92.6% [89.8, 94.4]**"),
    8: ("라 pooled 평균에서 상쇄됩니다",
        "지만 과제 간 차이는 검정하지 않았고 과제별 CI가 넓어, 이질성이나 상쇄가 확인된 것은 아닙니다"),
    9: ("전체 +0.20 m [−0.32, +0.56] (p = 0.51)", "선정 208 장면 전체(A− 52 + A+ 156) +0.20 m [−0.32, +0.56] (p = 0.51; 동등성 미검정)"),
    10: ("RN F −34.2 [−38.3, −29.5] / P −4.1", A4[10]["proposed_replacement"]),
    11: ("**모집단 추정량.**", "**평가 집합(E) 추정량.** (이전 표기 '모집단 추정량'; P는 선정 353 장면의 설계 가중 추정)"),
    12: ("과제 효과가 반대 부호로 평균에서 100% 상쇄됩니다",
         "과제별 점추정이 −4.2 / +4.2%p로 부호가 반대지만, 과제 간 차이 −8.3%p [−21.7, +5.0](p = 0.23)는 0을 포함하므로 이질성·상쇄는 "
         "확인되지 않았습니다"),
    13: ("| 전체 | CM +0.20 (n.s.)", A4[13]["proposed_replacement"]),
    14: ("A+와 전체에서는 운동 외삽과 구별되지 않습니다", A4[14]["proposed_replacement"]),
    15: ("새 24 drive에서 재현", "새 drive의 실패 층(105 장면 / 18 drive)에서 재현"),
    16: ("평가 horizon은 5 s(10 token)이며, 그 이상은 모델 분포 밖입니다.",
         "평가 horizon은 5 s(10 token)이며, 그 이상은 모델 분포 밖입니다. " + P16.split("입니다. ", 1)[1]
         + "(원 정의 보존; 3 s 민감도는 P/R/A 비교에만 있음; review_followup4 A4)."),
    17: ("과제별로 반대 부호 효과가 일부 상쇄됩니다", A4[17]["proposed_replacement"]),
    18: ("과제별로는 CI 폭 최대 18–29%p(P1-D r4e1)로 불확실**합니다(r1e4는 6.9–21.6%p) (2026-10-15 수정)",
         "과제별로는 CI 폭 최대 18–25%p(P1-D r4e1; r1e4 13–20%p; A1 사후 수정 구간)로 불확실**합니다(2026-10-15 판의 '18–29%p, r1e4 "
         "6.9–21.6%p'는 경계에서 무효인 sign-flip 역산 CI 기준)"),
    19: ("Both effects replicated in a preregistered test on 24 new drives that share no drive with the development set.",
         "Both effects replicated in a preregistered test on new drives (failure stratum: 105 scenes, 18 drives; no drive shared "
         "with the development set, but days and vehicles partly shared)."),
    20: (A4[20]["current_text"], A4[20]["proposed_replacement"]),
    21: (A4[21]["current_text"], A4[21]["proposed_replacement"]),
    22: ("across all scenes it was not distinguishable from extrapolation",
         "across the 208 selected mechanism scenes no difference from extrapolation was detected (equivalence not tested)"),
    23: (A4[23]["current_text"], A4[23]["proposed_replacement"]),
    24: (A4[24]["current_text"], A4[24]["proposed_replacement"]),
    25: ("On the evaluation population of the new-drive logs (2,191 eligible scenes), the corresponding effects were −4.1 pp "
         "[−6.4, −2.4] and +3.6 pp [+2.2, +5.5];", A4[25]["proposed_replacement"]),
    26: (A4[26]["current_text"], A4[26]["proposed_replacement"]),
    27: ("moved a later dimension by at least 8 bins in 92.6% of units (initial-state-cluster CI [90.8, 94.0])",
         A4[27]["proposed_replacement"]),
    28: ("\"replicated on 24 new drives (days and vehicles partly shared); no external (day/vehicle-disjoint) validation was possible\"",
         A4[28]["proposed_replacement"]),
    29: (A4[29]["current_text"], A4[29]["proposed_replacement"]),
    30: ("New drives (P1-C, pre-registered; 0 drives shared with this set): RN $-34.2$ [$-38.3$, $-29.5$] (18 drives)",
         "New drives (P1-C, pre-registered; failure stratum 105 scenes / 18 drives; 0 drives but 10 day--vehicle combinations "
         "shared with this set): RN $-34.2$ [$-38.3$, $-29.5$]"),
    31: ("within-step 92.6 [89.8, 94.4] (task CI; state CI [90.8, 94.0])",
         "within-step, $\\ge$1 later component $\\ge$ injected bins: 92.6 [89.8, 94.4] (task CI; state CI [90.8, 94.0]; bins, not "
         "physical units)"),
    32: ("pp with sign-flip-inversion 95\\% CI on initial-state clusters",
         "pp with original sign-flip-inversion 95\\% CI on initial-state clusters (invalid at the boundary for paired binary data; "
         "post-hoc corrected Rao--Scott Newcombe CIs in Table~\\ref{tab:rf4_ci})"),
    33: ("task-level CIs span up to 18--29 pp (r4e1; r1e4 6.9--21.6 pp)",
         "task-level CIs span up to 18--25 pp (r4e1; r1e4 13--20 pp; post-hoc corrected intervals, Table~\\ref{tab:rf4_ci})"),
    34: None,   # tables.tex Table 9 row label / caption: c_tables.py
    35: None,   # tables.tex Table 10 caption: c_tables.py
    36: (A4[36]["current_text"], A4[36]["proposed_replacement"]),
    37: (A4[37]["current_text"], A4[37]["proposed_replacement"]),
    38: ("amplifies a downstream dimension in **92.6% [89.8, 94.4]**",
         "changes at least one downstream component (y..yaw; gripper excluded) by at least the effective injected bin shift (bins "
         "mix m and rad; translation components alone 72%) in **92.6% [89.8, 94.4]**"),
    39: ("no task-level effect distinguishable from re-run variability (20% outcome disagreement)",
         "no detected task-level effect (C − F −3 pp [−11.9, +5.9]; equivalence not established), while re-running alone changed "
         "20% of outcomes"),
    40: (A4[40]["current_text"], A4[40]["proposed_replacement"]),
    41: (A4[41]["current_text"], A4[41]["proposed_replacement"]),
    42: (A4[42]["current_text"], A4[42]["proposed_replacement"]),
    43: (A4[43]["current_text"], A4[43]["proposed_replacement"]),
    44: ("across all scenes it was not distinguishable from extrapolation",
         "across the 208 selected mechanism scenes no difference from extrapolation was detected (equivalence not tested)"),
    45: ("across all scenes and in non-failure scenes it was not distinguishable from motion extrapolation",
         "in non-failure scenes and across the 208 selected mechanism scenes no excess over motion extrapolation was detected "
         "(equivalence not tested)"),
    46: (A4[46]["current_text"], A4[46]["proposed_replacement"]),
    47: (A4[47]["current_text"], A4[47]["proposed_replacement"]),
    48: (A4[48]["current_text"], A4[48]["proposed_replacement"]),
    49: (A4[49]["current_text"], A4[49]["proposed_replacement"]),
    50: (A4[50]["current_text"], A4[50]["proposed_replacement"]),
    51: ("궤적 발산", "실행 명령 누적 끝점 거리 (commanded, TCP 아님)"),
    52: (A4[52]["current_text"], A4[52]["proposed_replacement"]),
    53: ("결론(문맥 효과 0, 실행 효과 있음)은 바뀌지 않지만, 논문에서는 과제별 이질성을 밝혀야 합니다.",
         "결론(문맥 효과 검출 안 됨, 실행 효과 있음)은 바뀌지 않지만, 과제별 점추정 차이(coke −37.5 vs move_near −2.5%p)는 이질성 검정 "
         "없이 '집중'이나 이질성으로 쓰지 않습니다."),
    54: ("→ 효과 없음 ✔", "→ 효과 검출 안 됨(동등성 미검정)"),
    55: ("단 A−에서는 등속 외삽 baseline과 구별되지 않는다", "단 등속 외삽 초과는 A−(실패 선택 층)에서만 성립한다(review_followup3 P2)"),
    56: (A4[56]["current_text"], A4[56]["proposed_replacement"]),
}
NOT_EDITED = {"/root/VLA/review_followup3/CLAIM_EVIDENCE_MATRIX.md", "/root/VLA/review_followup3/CP2_REPORT.md"}

# ---------------------------------------------------------------------------------------------------------------------
# A1 / A2 / A3 / B2 / B3-driven edits: (file, old (unique in file), new, source)
EXTRA = [
    (ES, "최종 갱신: 2026-10-13 (",
     "최종 갱신: 2026-10-18 (review_followup4 반영: §6g 신설; A1–A4·B1·B2·B3 근거의 사실 정정은 해당 줄에 '(2026-10-18 수정)'으로 "
     "표시). 이전 갱신 2026-10-13 (", "C (header)"),
    (ES, "연속 3개 문맥 step 교정이 full-history 효과의",
     "고정된 3개 문맥 위치를 GT로 바꾸고 이후 문맥에도 그대로 유지하면(persistent GT, 해제 없음) full-history 효과의", "A2"),
    (ES, "교정 길이와 해제 후 남은 horizon이 함께 변합니다.",
     "교정 위치 수와 마지막 GT 위치 이후 생성 step 수(3개일 때 평균 3.8)가 함께 변합니다. 단위당 GT 위치 2.96개 중 자기 token과 다른 "
     "것은 1.90개입니다(review_followup4 A2).", "A2"),
    (ES, "4. **다른 VLA(OpenVLA-7B)에서도 token 수준 feedback은 재현됩니다**:",
     "4. **OpenVLA-7B(조작, LIBERO-Spatial)에서도 step 안 token 수준 의존성이 관찰됩니다**(기술적: 개입 경로 parity 미검증, B3 미실행, "
     "핵심 증거에서 제외):", "B3"),
    (ES, "효과가 재현됩니다**(사전 등록 판정 Strong, 실험 33):",
     "효과가 관찰됩니다**(기술적·미검증: 사전 등록 판정은 Strong이었지만 batch·decoding 경로 수치만으로 장면 수준 판정이 5–6% "
     "바뀌고 경로 parity가 검증되지 않아 강한 독립 재현으로 세지 않습니다, review_followup4 A3; 실험 33):", "A3"),
    (ES, "| horizon 착시가 아님, 단 해제 후 감쇠 |",
     "| horizon 착시가 아님; 이전 표기 '해제 후 감쇠'는 w개 GT 위치가 문맥에 남아 있는 동안의 재발산(summary 수준) |", "A2"),
    (ES, "방향 > 크기 → Strong replication (궤적 수준 안정 분기는 미재현)",
     "방향 > 크기 → 사전 등록 판정 Strong이었으나 기술적·미검증으로 낮춤(수치 바닥 5–6%, 경로 parity 미검증; 궤적 수준 안정 분기는 미재현)",
     "A3"),
    (ES, "해제 후 free horizon도 5.8 → 2.9 pose로 함께 줄어듭니다",
     "마지막 GT 위치 이후 생성 step(GT 유지; 이전 표기 '해제 후 free horizon')도 5.8 → 2.9 pose로 함께 줄어듭니다", "A2"),
    (ES, "해제 후 25–29% 재발산 — per-unit raw 없음",
     "w개 GT 위치가 문맥에 남아 있는 동안 25–29% 재발산(이전 표기 '해제 후') — per-unit raw 없음", "A2"),
    (ES, "같은 해제 후 horizon으로 맞추면", "마지막 GT 위치 이후 step 수를 맞추면", "A2"),
    (ES, "교정을 멈추면 서서히 재발산합니다(실험 10, summary 수준).",
     "w개 GT 위치를 문맥에 유지한 채 새 GT 추가를 멈추면 일부 재발산합니다(실험 10, summary 수준). GT를 문맥에서 제거하는 해제 실험은 "
     "없습니다(review_followup4 A2).", "A2"),
    (ES, "task 수준(교란이 기저 변동보다 큰 실패를 만들고 교정이 실패를 줄임) **충족** → **Strong replication**.",
     "task 수준(교란이 기저 변동보다 큰 실패를 만들고 교정이 실패를 줄임) **충족** → **Strong replication**(사전 등록 판정, 기록 보존). "
     "review_followup4 A3 이후: 기술적·미검증 — 같은 경로·batch는 bit 단위로 재현되지만 batch 크기·padding·prefill 분할·decoder 구현만 "
     "바꿔도 장면 수준 A−/증폭 판정이 5–6% 바뀌고, 경로별 no-op gate가 없습니다. 큰 대비(−63.4%p, +81.9%p)만 바닥과 함께 기술합니다.",
     "A3"),
    (ES, "교정 해제 후 재발산은 AutoVLA와 달리 거의 없었습니다.",
     "(삭제: Impromptu window도 GT가 이후 문맥에 유지되므로 '해제 후' 비교가 아닙니다, review_followup4 A2/A3.)", "A2/A3"),
    (ES, "→ 직전 action 문맥이 이후 생성을 바꾸는 token 수준 인과 구조는 driving·Qwen에 한정되지 않고 temporally autoregressive VLA 디코더에서 반복됩니다.",
     "→ 직전 action 문맥이 이후 생성을 바꾸는 token 수준 효과는 SpatialVLA(조작)에서도 관찰됐습니다. 모델별 개입이 달라 일반 VLA에 대한 "
     "주장으로 쓰지 않습니다.", "C (scope)"),
    (ES, "  - 해제 후 free step 평균이 5.8 / 4.8 / 3.8 / 2.9로 함께 줄어듭니다.",
     "  - 마지막 GT 위치 이후 step 평균(GT 유지; 이전 표기 '해제 후 free step')이 5.8 / 4.8 / 3.8 / 2.9로 함께 줄어듭니다.", "A2"),
    (ES, "고정 해제 후 horizon(E_post)", "마지막 GT 위치 이후 고정 horizon(E_post; GT 유지)", "A2"),
    (ES, "자기 이전 action을 다시 넣으면 +29%p", "별도 Normal branch token을 넣으면 +29%p", "A4/D2 wording"),
    (ES, "(첫 step이 특별하다는 근거 없음; 해제 후 감쇠)", "(첫 step이 특별하다는 근거 없음; GT 위치는 이후 문맥에 유지, 해제 실험 없음)", "A2"),
    (ES, "- 직전 action 문맥의 token 수준 인과 효과는 다른 autoregressive VLA(OpenVLA, Impromptu, SpatialVLA)에서도 나타납니다.",
     "- 모델별로 다른 개입에서 token 수준 문맥 효과가 관찰됐습니다(SpatialVLA는 복원 검증; OpenVLA·Impromptu는 기술적·미검증으로 강한 "
     "독립 재현으로 세지 않음).", "A3/B3"),
    (ES, "교정 길이와 해제 후 horizon이 함께 변하고, 절대 pose와 해제 후 horizon을 동시에 맞출 수 없습니다(10 token 한계).",
     "교정 위치 수와 마지막 GT 위치 이후 step 수가 함께 변하고(GT는 이후 문맥에 유지되며 해제 실험은 없음), 절대 pose와 그 step 수를 "
     "동시에 맞출 수 없습니다(10 token 한계).", "A2"),
    # ---- PAPER_NUMBERS
    (PN, "  - The post-release free horizon shrinks with w (5.8 → 2.9 poses), so the length effect is partly a post-release-horizon effect.",
     "  - The number of steps after the last added GT position (GT retained; formerly called the 'post-release free horizon') "
     "shrinks with w (5.8 → 2.9 poses), so the length effect is partly a horizon effect.", "A2"),
    (PN, "Late windows have 2 fewer free poses after release. At a common post-release horizon (post-hoc),",
     "Late windows have 2 fewer poses after the last added GT position. At a common horizon after the last GT position (post-hoc),", "A2"),
    (PN, "re-divergence after release 25–29%.",
     "re-divergence while the w corrected GT positions remain in the context 25–29% (formerly 'after release').", "A2"),
    (PN, "**Manuscript-safe:** \"Correcting three consecutive context steps recovered about 90% of the full-history effect (95% CI 81–99%).\"",
     "**Manuscript-safe:** \"Three fixed context positions replaced with ground truth and retained in later contexts recovered about "
     "90% of the full-history correction effect (95% CI 81–99%).\"", "A2"),
    (PN, "- \"A permanent fix.\" Recovery decays after release (Experiment 10, summary-level).",
     "- \"A permanent fix.\" Re-divergence while the GT positions remain in the context (Experiment 10, summary-level); no release "
     "experiment exists.", "A2"),
    (PN, "# Claim 8. Impromptu VLA: Strong replication (preregistered)",
     "# Claim 8. Impromptu VLA: descriptive, parity not established (preregistered verdict 'Strong' kept on record; downgraded by "
     "review_followup4 A3)", "A3"),
    (PN, "and re-inserting the model's own previous action restored it.",
     "and injecting the separate Normal branch's previous token restored it.", "A4/D2 wording"),
    # ---- MANUSCRIPT_REVISION (historical 2026-10-13 section: abstract / contributions / limitations)
    (MR, "reduced trajectory amplification by 40 percentage points (pp) (95% CI 30–49; 13 independent drives;",
     "reduced trajectory amplification in failure-stratum (A−) scenes by 40 percentage points (pp) (95% CI 30–49; 52 scenes, 13 "
     "independent drives;", "A4"),
    (MR, "and re-inserting the model's own previous action restored it (+29 pp).",
     "and injecting the separate Normal branch's previous token into a ground-truth context restored it (+29 pp).", "A4/D2 wording"),
    (MR, "Three consecutive corrected context steps recover about 90% (81–99%) of the full-history effect",
     "Three fixed context positions replaced with ground truth and retained in later contexts recover about 90% (81–99%) of the "
     "full-history effect", "A2"),
    (MR, "- Correction length and post-release horizon change together; absolute pose and post-release horizon cannot both be equalized within the 10-token plan.",
     "- Correction length and the number of steps after the last added ground-truth position change together (the ground-truth "
     "positions stay in the context; no release experiment exists); absolute pose and that horizon cannot both be equalized within "
     "the 10-token plan.", "A2"),
    (MR, "> \"Correcting three consecutive context steps recovered about 90% (scene-weighted; 95% CI 81–99%) of the full-history "
         "correction effect (365 units, 52 scenes, 13 drives for every window length); correction length and the remaining "
         "post-release horizon change together.\"",
     "> \"Three fixed context positions replaced with ground truth and retained in later contexts recovered about 90% "
     "(scene-weighted; 95% CI 81–99%) of the full-history correction effect (365 units, 52 scenes, 13 drives for every window "
     "length); the number of corrected positions and the number of steps generated after the last added ground-truth position "
     "change together.\"", "A2"),
    (MR, "| \"Correcting three consecutive context steps recovered about 90% of the full-history effect (95% CI 81–99%).\" + ",
     "| \"Three fixed context positions replaced with ground truth and retained in later contexts recovered about 90% of the "
     "full-history correction effect (95% CI 81–99%).\" + ", "A2"),
]


def mark(path, line):
    if path.endswith(".tex"):
        return line if MK_TEX in line else line + MK_TEX
    if MK in line:
        return line
    s = line.rstrip()
    if s.startswith("|") and s.endswith("|") and len(s) > 1:
        return s[:-1].rstrip() + MK + " |"
    return s + MK


def apply_line(path, lineno, old, new, log, src, a4_text=None):
    lines = open(path).read().split("\n")
    text = "\n".join(lines)
    if new in text and (old not in text or old in new):
        log.append({"source": src, "file": path, "line": lineno, "status": "already applied", "old": old, "new": new})
        return
    i = lineno - 1 if lineno else -1
    if lineno is None or old not in lines[i]:
        cand = [k for k, l in enumerate(lines) if old in l]
        assert len(cand) == 1, (path, lineno, old[:70], len(cand))
        i = cand[0]
    if a4_text is not None:
        assert a4_text[:40] in lines[i] or a4_text in old, (path, lineno, a4_text[:60])
    assert lines[i].count(old) == 1, (path, i + 1, old[:70])
    lines[i] = mark(path, lines[i].replace(old, new))
    open(path, "w").write("\n".join(lines))
    log.append({"source": src, "file": path, "line": i + 1, "status": "applied", "old": old, "new": new})


def insert_section(path, anchor, text, key, log, before=True):
    s = open(path).read()
    if key in s:
        log.append({"source": "C (section)", "file": path, "line": "", "status": "already present", "old": "", "new": key})
        return
    if anchor is None:
        s = s.rstrip("\n") + "\n\n" + text.rstrip("\n") + "\n"
    else:
        assert s.count(anchor) == 1, (path, anchor)
        s = s.replace(anchor, (text.rstrip("\n") + "\n\n" + anchor) if before else (anchor + "\n\n" + text.rstrip("\n")), 1)
    open(path, "w").write(s)
    log.append({"source": "C (section)", "file": path, "line": "", "status": "inserted", "old": "", "new": key})


def main():
    log = []
    for i, r in enumerate(A4):
        path, lineno = r["file"], int(r["line"])
        if path in NOT_EDITED:
            log.append({"source": f"A4 #{i}", "file": path, "line": lineno, "status": "NOT EDITED (historical review_followup3 report; superseded by CLAIM_CHANGES.md)",
                        "old": r["current_text"], "new": r["proposed_replacement"]})
            continue
        e = A4_EDITS[i]
        if e is None:
            log.append({"source": f"A4 #{i}", "file": path, "line": lineno, "status": "applied by c_tables.py (machine-generated block)",
                        "old": r["current_text"], "new": r["proposed_replacement"]})
            continue
        old, new = e
        apply_line(path, lineno, old, new, log, f"A4 #{i} ({r['rule']})", r["current_text"])
    for path, old, new, src in EXTRA:
        apply_line(path, None, old, new, log, src)
    # ---- sections
    summ = json.load(open(f"{R4}/REPRO_CHECK_summary.json"))
    rz = (f"`reproducibility_minimal.zip` {summ['zip_mb']} MB(이미지·checkpoint·raw logit 제외; 코드, protocol, manifest, judge "
          f"로그, 단위 수준 표, B2 cell·bank, 고정된 공개 revision·checksum `PINS.csv`, 누락 목록 `MISSING.md`). 압축을 푼 사본에서 "
          f"`regen.sh`로 핵심 표({', '.join(summ['targets'])})를 실제로 다시 계산해 원본과 비교했습니다: 파일 {summ['n_files_compared']}개 중 "
          f"byte 동일 {summ['n_identical_bytes']}개, 값 동일 {summ['n_match_values']}개, 불일치·실패 {summ['n_mismatch_or_failed']}개"
          f"(`REPRO_CHECK.md`).")
    rd = lambda f: open(os.path.join(SECTIONS, f)).read().replace("RZIP_SUMMARY", rz)
    insert_section(ES, "## 7. 불안정 rollout 탐지: 이탈 전에 예측 가능한가 (실험 31, P4)", rd("es_6g.md") + "\n---\n",
                   "## 6g. review_followup4", log)
    insert_section(PN, "# Claim 1. Token mismatch ≠ failure", rd("pn_round.md") + "\n---\n", "## 2026-10-18 개정 (review_followup4 반영)", log)
    insert_section(PN, None, rd("pn_claims.md"), "# Claim 25.", log)
    insert_section(MR, "**Why this file exists.**", rd("mr_banner.md"), "**2026-10-18 (review_followup4):**", log)
    insert_section(MR, None, rd("mr_round.md"), "## 2026-10-18 revision (review_followup4)", log)
    insert_section(f"{PKG}/README.md", None, rd("readme_note.md"), "**2026-10-18 개정 (review_followup4)", log)
    # ---- review_followup4/MANUSCRIPT_REVISION.md: template + A4 table from the log
    a4log = {int(x["source"].split("#")[1].split()[0]): x for x in log if x["source"].startswith("A4 #")}
    tab = ["| # | file:line | rule | current text | replacement | status |", "|---|---|---|---|---|---|"]
    esc = lambda t: str(t).replace("|", "\\|").replace("\n", " ")
    for i, r in enumerate(A4):
        x = a4log[i]
        st = ("applied" if x["status"] in ("applied", "already applied") else
              "c_tables" if x["status"].startswith("applied by c_tables") else "not edited")
        f = r["file"].replace("/root/VLA/", "")
        tab.append(f"| {i} | `{f}:{r['line']}` | {esc(r['rule'])} | {esc(r['current_text'])} | {esc(x['new'] if st == 'applied' else r['proposed_replacement'])} | {st} |")
    open(f"{R4}/MANUSCRIPT_REVISION.md", "w").write(rd("rf4_manuscript_revision.md").replace("<!-- A4_TABLE -->", "\n".join(tab)))
    with open(f"{R4}/DOC_EDITS_APPLIED.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "file", "line", "status", "old", "new"])
        w.writeheader()
        w.writerows(log)
    from collections import Counter
    print(json.dumps(Counter(x["status"] for x in log), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
