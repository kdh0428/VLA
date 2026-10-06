# Conflicts / discrepancies — 탐지·선택 (실험 12–24, 31/P4)

등급: **CRITICAL** = 논문 주장이 원자료와 맞지 않음, **MAJOR** = 표현·범위를 고쳐야 함, **MINOR** = 모집단·반올림·문서 차이(주석으로 충분).
경로 `O = /root/VLA/autovla_misalignment_poc/outputs`.

검토 결과 **CRITICAL은 없습니다**. 탐지 AUROC 9개 행 × pre/post는 EXPERIMENT_SUMMARY §7, RESULTS.md, detection.md, detection.json 사이에서 모두 일치합니다(반올림까지).
선택 결과(dev/held-out PDMS, A−, 충돌, F1)도 요약 문서와 summary JSON이 일치합니다.

## MAJOR

**M1. "이탈 전 엔트로피가 같다"는 과장입니다. 차이는 작지만 CI가 0을 배제합니다.**
- 주장: `/root/VLA/CONCLUSIONS.md` §0-7 ("첫 이탈 **전** 엔트로피가 같고"), §9d; `/root/VLA/EXPERIMENT_SUMMARY.md` §4 ("이탈 전 엔트로피가 같고(차이 −0.016, …)"); `O/selection_validation/SELECTION_VALIDATION.md` 요약 bullet.
- 원자료: `O/mechanism_selection_link/heldout/summary.json` `selected_vs_discarded.all_scenes.pre_dev_entropy` diff −0.016 [−0.023, −0.010], n = 1,464 장면(혼합 GPU); dev −0.024 [−0.032, −0.016]; 5090 held-out −0.016 [−0.023, −0.009].
  자연 실패 장면 부분집합에서만 CI가 0을 포함합니다(held-out +0.012 [−0.038, +0.060], dev −0.019 [−0.040, +0.011]).
- 판정: 진짜 충돌(표현 문제). 효과 크기는 이탈 후 차이(−0.40)의 4–6%이므로 결론 방향은 유지되지만 "같다" 대신 "작다(−0.016 vs −0.40)"로 써야 합니다.
  또 요약 문서는 이 차이와 "장면 내 AUROC 0.51"을 한 괄호에 묶는데, 0.51은 선택 vs 버림 비교가 아니라 이탈 후보 중 증폭 여부에 대한 이탈 전 엔트로피 AUROC(`auroc_amplification.pre_dev_entropy.within_scene_mean`)입니다.

**M2. "open-loop 실패 −33%"의 근거·범위를 명시해야 합니다.**
- 주장: `/root/VLA/EXPERIMENT_SUMMARY.md` §1-7 ("open-loop 실패율 −33%, 사전 등록 held-out PDMS +0.0095"); `/root/VLA/CONCLUSIONS.md` §0-6 ("A− 비율 2.15 → 1.45%(−33%)").
- 원자료: `O/expanded_best_of_n/summary_pooled.json` `rules.ranksum.a_minus`: 0.021477 → 0.014464 (98 → 66 / 4,563 장면), diff −0.70%p [−1.07, −0.33], McNemar 23/55, p = 3.8e-4. 상대 −32.6% → "−33%" 맞음.
- 범위 문제: (a) 상대 감소이며 절대 −0.70%p입니다. (b) dev(실험 20) 결과로, 형식적 사전 등록 문서는 없습니다(규칙은 디코딩 전 e6d9dff에 커밋). §1-7 문장이 "사전 등록 held-out PDMS"와 나란히 있어 −33%도 사전 등록 결과로 읽힐 수 있습니다.
  사전 등록 held-out의 open-loop A−는 1.50 → 0.71%, −0.79%p [−1.01, −0.55], 상대 **−53%**(`O/heldout_best_of_n/summary_pooled.json`). (c) "실패율"은 A−이며, 스크립트에서 `failure`라 부르는 강한 실패(A− ∧ FDE5 > 3 m)는 1.80 → 1.18%(−34%)입니다.
  (d) dev 내부 이질성: 3080 Ti 부분(17 log)은 −0.32%p [−0.83, +0.13], n.s.(`O/expanded_best_of_n/gpu0/summary.json`); 5090 부분은 −0.85%p [−1.32, −0.40]. 5090 전용 재실행 dev는 −0.72%p(−34%).
- 판정: 수치 자체는 맞음. 논문에는 "dev −0.70%p (−33% relative; held-out preregistered −0.79%p, −53%)"처럼 둘 다 쓰는 것을 권장합니다.

**M3. "3080 Ti 결과 vs 5090"이라는 이름과 정본.**
- `/root/VLA/EXPERIMENT_SUMMARY.md` 표 2의 "3080 Ti 결과 5090 재실행" 행과 달리, 실험 20–24의 원 실행은 3080 Ti 단독이 아니라 **혼합 GPU**입니다:
  dev 5090 39 log(3,294 장면) + 3080 Ti 17 log(1,269), held-out 5090 30 log(2,823) + 3080 Ti 22 log(1,991) (`EXPANDED_BEST_OF_N.md`, `PREREGISTRATION.md`, `SELECTION_VALIDATION.md` 데이터 표).
  재실행은 3080 Ti 부분만 5090에서 다시 디코딩했습니다(`GPU5090_REANALYSIS.md`는 "혼합 GPU"라고 정확히 씀).
- 정본: EXPERIMENT_SUMMARY §4·CONCLUSIONS §9d는 혼합 GPU(사전 등록 513fdba 대상)를 주 결과로, 5090 전용을 강건성 확인으로 씁니다. 실험 31(P4)은 5090 전용 후보를 씁니다.
  → 논문 표에서 실험 22 수치(혼합)와 실험 31 수치(5090)를 섞으면 N이 다릅니다(아래 m2). 수치 차이는 결론을 바꾸지 않습니다(held-out rank-sum +0.0095 vs +0.0099, F1 +0.0315 vs +0.0316).
- 실험 14, 15는 원 실행이 실제로 3080 Ti 단독이고 5090 재실행이 따로 있습니다. 요약 문서는 3080 Ti 값(실험 15 모집단 1.24 → 2.60%)을 인용; 5090은 1.31 → 2.27%(`O/natural_reference_stabilization_5090/analysis_console.txt`). 같은 방향, 다른 크기.

## MINOR

**m1. P4 "pre" 창은 이탈 token을 포함합니다.** `PROTOCOL.md`("pre = t*까지(포함)")와 `analyze_instability_detection.py`(`slice(0, ts+1)`)는 명확하지만, 요약 문서는 "이탈 전"으로 씁니다.
pre 신호에 유리한 정의이므로 "이탈 전에는 예측되지 않는다"는 결론을 약하게 만들지는 않습니다. 논문에는 "up to and including the deviating step"으로 써야 정확합니다.

**m2. 탐지·기전의 표본 수 차이 = GPU 모집단 차이.** held-out 이탈 후보/증폭: 실험 22 혼합 43,517 / 2,550(`mechanism_selection_link/heldout`, SELECTION_VALIDATION §2) vs P4·실험 22 5090 43,469 / 2,538.
dev: 42,078 / 2,877(혼합) vs 42,095 / 2,885(5090, P4). 이탈 전 엔트로피 장면 내 AUROC 0.513(혼합) vs 0.516(5090). 진짜 충돌 아님.

**m3. 같은 데이터인데 CI가 약간 다름.** `mechanism_selection_link/heldout_5090` `auroc_amplification.pre_dev_entropy` 0.6335 [0.6080, 0.6584] vs P4 `heldout.entropy_pre` 0.6335 [0.6067, 0.6582].
점추정 동일, bootstrap 반복 수만 다름(실험 22 `boot_ci(..., reps=500)`, P4 `reps=1000`).

**m4. 장면 내 AUROC "0.50–0.65 = 우연"에는 검정이 없습니다.** detection.json에 장면 내 AUROC CI·p가 없고, medoid 거리 pre 장면 내 0.649는 0.5와 분명히 다릅니다. 또 범위 0.50–0.65는 후보 분산(0.432, 의미 없음 표시)을 제외한 값입니다.

**m5. 탐지기 간 비교 검정 없음.** "hidden probe가 엔트로피보다 낫지 않다"(0.611–0.631 vs 0.634)와 "disagreement가 가장 강하다"는 점추정·CI 겹침 비교일 뿐 쌍대 AUROC 검정은 not reported.

**m6. 후보 수 "N"의 두 가지 정의.** 실험 18–19의 "N 16" = T 1.0 샘플 16개 + 자연 계획 = 후보 17개(`run_meta.json` `"n": 16`). 실험 23의 "N 16" = 자연 포함 후보 16개(샘플 15개).
추론 비용 "N 16 1.3×"(timing log "16 candidates", `--n 15`)는 후보 16개 기준이며 실제 배포 설정(17개)은 측정되지 않았습니다.

**m7. 추론 비용 표기.** `GPU5090_REANALYSIS.md`: "N 4/8/16 각 1.2×, N 1 1.8–2.0분". 저장소의 log(`O/gpu5090_reanalysis/timing5090_N*.log`)는 N 1 1.8분(1.10 s/장면), N 4/8/16 2.3분(1.36 s/장면) → 경과 기준 1.28×, 장면당 1.24×.
"2.0분" N 1 log는 저장소에 없습니다. 3080 Ti 측정(1.03/1.16/1.31×)의 원 log는 저장소 밖 scratchpad(`/tmp/claude-0/.../scratchpad/timing_N*.log`: 3.1/3.3, 3.3, 3.7, 4.2분)에만 있어 재현 근거가 약합니다. 수치는 보고서와 일치.

**m8. "N 8에서 이득의 85%"의 지표 선택.** EXPERIMENT_SUMMARY §4·CONCLUSIONS §9d는 held-out 혼합 PDMS 기준(0.0081/0.0095 = 85.7%)입니다. 같은 N 8에서 held-out A− 기준은 73%, dev PDMS 기준 79%, 5090 held-out PDMS 87%
(`O/candidate_count_curve/*/summary.json`, `O/pdm_score_best_of_n/*_ncurve/summary.json`). "12–16에서 포화"는 held-out에는 맞지만 dev PDMS는 N 12에서 91%.

**m9. 사전 등록 시점.** 513fdba 커밋 20:12:38 UTC, held-out 디코딩 종료(`heldout_best_of_n/gpu1/records.jsonl` mtime) 20:11:19 UTC, 분석 결과 커밋 e3bc8db 21:48. 사전 등록은 디코딩 후·분석 전입니다.
"결과를 보기 전"은 분석 결과 기준으로만 지지되고, 저장소 기록으로 분석 미실행을 독립 검증할 수는 없습니다. 디코딩 자체는 고정 seed라 규칙과 무관합니다.

**m10. held-out 자연 실패 장면 수 72 vs 73.** 혼합 GPU 72(`pdm_score_best_of_n/heldout`, `heldout_best_of_n/summary_pooled.json`), 5090 전용 73(`*_5090`). GPU5090_REANALYSIS의 "5090 open-loop A− 1.52%"가 이것입니다. 자연 계획 PDMS는 두 경우 모두 0.8893.

**m11. "자연 실패 장면 PDMS +0.11–0.14"(CONCLUSIONS §9d).** held-out rank-sum +0.111 [+0.029, +0.209](72 장면), dev rank-sum +0.139 [+0.082, +0.192](98 장면) — 두 모집단을 합친 범위입니다. 문제없음, 출처 명시만 필요.

**m12. "모든 안전 관련 위반이 줄며"(SELECTION_VALIDATION §1).** 충돌·DA·TTC는 줄지만 승차감(comfort) 위반은 0.12 → 0.15%로 늘었습니다(`pdm_score_best_of_n/heldout/summary.json` `viol_comfort`). comfort를 안전 지표로 보지 않는다면 맞는 문장입니다.

**m13. 충돌률 감소의 통계 검정 없음.** held-out 충돌 27 → 11 → 5건(0.56 → 0.23 → 0.10%)은 평균 위반율만 보고되고 McNemar·CI는 not reported입니다. 논문에서 "충돌을 절반 이하로"는 기술 통계로만 써야 합니다.

**m14. dev 혼합 vs 5090 자연 계획 PDMS 0.8951 vs 0.8961, 후보 수 곡선 N 1 PDMS 0.8952(혼합).** 0.8951 vs 0.8952는 `ncurve` 채점(앞쪽 부분집합 재채점) 반올림 차이로 보이며 결론과 무관합니다.

**m15. 실험 19 GPU 표기.** CONCLUSIONS §11 "실험 14, 15, 18, 19(seed 0–2)는 RTX 3080 Ti"는 N16 T1.0 seed 0–2·4, N4, N8에 해당하고, N32·N16 T0.85/1.15·seed 3은 5090입니다(`run_meta.json`). 후보 수 곡선에서 N 32 행을 쓸 때 GPU가 다름을 표시해야 합니다.
