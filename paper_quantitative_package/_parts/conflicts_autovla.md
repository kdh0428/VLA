# AutoVLA 기전 실험 — 출처 간 불일치 목록

경로 약어: `O/` = `/root/VLA/autovla_misalignment_poc/outputs/`, `ES` = `/root/VLA/EXPERIMENT_SUMMARY.md`, `CC` = `/root/VLA/CONCLUSIONS.md`.
등급: **CRITICAL** = 방향·결론이 바뀜, **MAJOR** = N·CI·p·구조(architecture)·주장 강도, **MINOR** = 반올림·표현·옛 수치.
요약: 원 RESULTS/summary.json과 반대 방향의 결론을 내는 CRITICAL 불일치는 **찾지 못했습니다**. 아래 MAJOR 항목은 논문 문구에서 반드시 고쳐야 할 것들입니다.

---

## MAJOR

### M-1. "같은 거리(8.8 mm)" — 매칭 오차를 perturbation 크기로 잘못 기술
- `ES` §3.2 표 행 6: "같은 거리(8.8 mm) 대안 token".
- 원 결과: 8.8 mm는 **대안 token 거리와 원래 오답 거리의 절대차 중앙값**(매칭 오차)입니다(`O/equal_distance_perturbation/summary.json` `alt_abs_dist_diff_median_m` = 0.00883; 상대차 중앙값 `alt_rel_err_median` = 7.9%). 실제 perturbation 거리는 A− 대안 중앙값 **0.129 m**(평균 0.179 m), A+ 대안 0.087 m (`table.<g>."alt (all)".dist`).
- `CC` §6은 "같은 거리(중앙값 상대차 7.9%, 절대 8.8 mm)"로 올바르게 기술.
- population 차이 아님 — 표현 오류(크기를 약 15배 과소 기술). 논문에는 "perturbation 0.13 m (matched within 8.8 mm / 7.9%)"로.

### M-2. attention mask "효과 없음"은 전체 집합에서만 성립 (결과로 선택된 부분집합에서는 유의한 감소)
- `ES` §3.2 행 7 "attention mask는 효과 없음", `CC` §7 "attention mask는 효과가 없으므로, 기전은 긴 history 참조가 아니라 …".
- 원 결과(`O/action_history_causal/summary.json`, 보고서 `ACTION_HISTORY_CAUSAL.md`):
  - all perturbations A− (365): action-history mask −1.4%p [−8.5, +7.4], p = 0.63; recent-action mask −3.3%p [−9.7, +2.9], p = 0.18 → 효과 없음 ✔.
  - **previously amplified A− (175 단위, 45 장면)**: −16.0%p [−23.7, −6.2], p = 4.3e-06 / −19.4%p [−28.1, −10.1], p = 5.7e-08; A+ previously amplified (43): −37.2 / −48.8%p.
  - original token only A− (52): −11.5%p [−25.5, +5.4], p = 0.11.
- population 차이(부분집합이 equal-distance run의 결과로 선택됨 → 평균회귀 가능성; 그러나 같은 부분집합에서 Normal은 94.3%로 유지). 결론 방향(직전 1 token GT 교정 −81%p가 mask 효과보다 훨씬 큼)은 유지되지만, "mask는 효과 없음"은 "전체 perturbation 집합에서 유의한 효과 없음(증폭이 이미 일어난 단위에서는 −16~−19%p)"로 한정해야 함.

### M-3. "L35 MLP 지배 재현" — 원 보고서 판정은 "△ 부분 재현"
- `ES` §2 행 4 "late action formation, L35 MLP 지배 재현", `CC` §4 "재현된 것: … L35 MLP 지배(attention 대비 7.8–21.5배)".
- 원 결과(`O/natural_fast_mechanism/NATURAL_FAST_MECHANISM.md` §2, `summary.json` `N.verdicts`): step-0 기준 L35 MLP 차이 −1.16 [−2.15, −0.45] → holds; **첫 불일치 step 기준 −0.34 [−0.89, +0.19], `L35_mlp_dominance[traj_first_mismatch].holds = false`** → 보고서 판정 "△ 부분 재현". 또 "7.8–21.5배"는 L32–35 MLP/Attention 합의 비(attention_weak 지표)이지 L35 MLP 단독 지표가 아님(N 첫 불일치 step 11.2×, C 첫 불일치 step 429×는 인용 범위에서 빠짐).
- population 차이(분석 시점 step-0 vs 첫 불일치 step). 주장 강도 과장.

### M-4. 실험 5의 N을 equal-distance set으로 기술
- `ES` §10: "AutoVLA 기전 실험(5–11, 32)은 equal-distance set(208 장면, A− 52 장면)에 기반".
- 원 결과: 실험 5는 **첫 불일치가 있는 모든 장면 1,159개**(A− 52 + A+ 1,107; step-matched 유효 n 944) (`O/first_mismatch_causal/summary.json` `n`; `run_meta.json` `n_ok` 1159). equal-distance set(208)은 실험 6–11, 32에만 해당.
- population 차이(서술 오류). A− 52는 같음.

### M-5. "37 layer" 표기 (구조)
- `ES` §3.2 행 8 "layer patching (37 layer)", `CC` §8 "37개 layer 전수".
- 원 결과: patch 위치 37개 = **embedding 출력 1 + decoder block L0–L35 36개** (`O/prev_action_state_patching/run_meta.json` `layer_keys`; 보고서 "Layer key: emb = block 0 입력, L{i} = decoder block i 출력"). Qwen2.5-VL-3B 언어모델은 36 layer. 논문에는 "embedding + all 36 decoder layers (37 patch sites)".

### M-6. A− 라벨의 재디코딩 안정성 (N/정의)
- 실험 4–11의 A− 52 장면은 **저장된 natural run(arm N)**으로 정의됨.
  - 같은 52 장면을 실험 5 harness로 재생성하면 A− 유지 **82.7%(43/52)**, 17.3%는 A+ (`O/first_mismatch_causal/summary.json` `reproduction`; = original A+ 17.3%).
  - 실험 6 original의 "amplification"(A− ∧ FDE > 3 m)은 67.3% [51, 80] — A− 정의(82.7%)와 다른 지표.
  - 다른 natural run(gpu1, residual만 저장)에서는 A− 46건(step-0 오답 18) (`O/natural_fast_mechanism/summary.json` `G.*`).
  - best-of-N harness(T 0.01)에서 "실패 단위 52개"의 자연 계획 A−: 3080 Ti 63–65% vs 5090 67–71% (`O/gpu5090_reanalysis/GPU5090_REANALYSIS.md` "실험 18–19" 표).
- population/run 차이(서로 다른 디코딩 harness·GPU). 결론을 바꾸지 않지만, 논문에서 "A− 52 scenes"는 "stored natural-run failures (re-decoding retains 83%)"로 정의를 명시해야 함.

### M-7. "효과의 93%가 사라진다"의 분모
- `ES` §1.2 "직전 token 1개만 GT로 바꿔도 효과의 93%가 사라집니다", `CC` §7 "전체 효과의 93%".
- 원 결과: 93% = (Normal − Recent-GT)/(Normal − GT-history) = (47.4 − 7.1)/(47.4 − 4.1) = 40.3/43.3 (`O/action_history_causal/summary.json` 평균; 실험 7은 이 비율과 CI를 보고하지 않음). 같은 비율의 CI는 실험 8에서만 보고: 92.9% [89.3, 99.1] (`O/prev_action_state_patching/summary.json` `groups.A-.all.fraction_of_gt_history_effect.recent_gt`). Normal 증폭률 대비 상대 감소는 85.0%.
- 표현 문제(분모 = GT-history 최대 효과). 논문에는 공식과 CI 출처(실험 8 harness)를 함께.

---

## MINOR

### m-1. A− Normal AR 증폭률 46.6 / 46.8 / 47.1 / 47.4 / 47.9%
- 실험 9 46.6% (`O/temporal_feedback_window`), 실험 8 46.8% (`O/prev_action_state_patching`), 실험 11 47.1% (`O/prev_action_identity_decomposition`), 실험 7 47.4% (`O/action_history_causal`), 실험 32 47.9% (`O/motion_semantics_ablation`). `ES` §3.3은 "47.1–47.9%", `CC` §9b·9c는 47.1%, `CC` §7은 47.4%, `CC` §9 표는 46.6%.
- **같은 population**(365 단위). batch 행 수(7/93/17/9/6)에 따른 bf16 수치 경로 차이; 실험 7 대비 Normal token 재현율 86.5–89.2%. 마찬가지로 GT-history 4.1/4.4%, Recent-GT 7.1/7.4/7.7/7.9%, A+ Normal 4.2/4.3/4.5%.
- 처리: 상대 효과는 반드시 같은 실험의 분자·분모로 계산(예: 실험 9의 "47%"는 46.6 기준).

### m-2. A− ADE/FDE 2.27/6.63 m vs 2.91/8.18 m
- `O/natural_fast_mechanism/summary.json` `N.codebook_amplification.A-.{ade_m,fde_m}` = 2.27/6.63 (**중앙값**, 저장된 natural run).
- `O/first_mismatch_causal/summary.json` `A-.original.{ade5,fde5}.mean` = 2.91/8.18 (**평균**, 재생성 harness; equal-distance original도 동일).
- 같은 52 장면, 다른 통계량(중앙값 vs 평균) + 다른 run. 충돌 아님.

### m-3. natural N 2,747 vs 2,748
- `O/natural_fast_mechanism` n = 2,747; `O/fast_vs_cot_unbiased/FAST_VS_COT.md` "전체 2,748개"; `O/annotations/` 파일 2,748개.
- `autovla_misalignment_poc/REPRODUCTION.md`: 장면 2,748개 중 arm N fork 성공 2,747. population 차이(1 장면 fork 실패).

### m-4. equal-distance full vs strict
- full: A− 대안 313, 증폭 45.0%, 혼합 장면 71.2% (37/52); strict: 대안 292, 증폭 46.2%, 혼합 64.7% (33/51) (`EQUAL_DISTANCE.md` vs `EQUAL_DISTANCE_strict.md`).
- population 차이(strict는 전체 codebook fill 대안 제외, 대안이 모두 빠진 A− 1 장면 제외). 결론 동일. `ES`/`CC`는 full 값만 인용.
- 부수: `EQUAL_DISTANCE_strict.md` 머리말의 "장면당 대안 6.0/5.7개, 상대차 7.9%"는 스크립트가 필터 전 `recs`로 계산한 full 값(`S/analyze_equal_distance.py` `S["alts_per_scene"]`) — strict 값이 아님.
- 부수: strict `summary_strict.json`/`rows_strict.jsonl`이 출력 디렉토리에 없음(재실행으로 복원: `_parts/recomputed/`).

### m-5. equal-distance Q4 AUROC 재현성
- 재실행 시 Q4(교차검증 로지스틱) AUROC가 소수 둘째 자리에서 최대 ±0.01 다름(예: A− 거리+방향 0.62 → 0.63). `REPRODUCTION.md`가 이미 "최대 0.009 차이(solver 수치)"로 기록. 다른 모든 수치는 비트 단위 동일. MINOR.

### m-6. "어느 layer에서 patch해도 3.8–7.4%" vs 후반 layer의 유의한 추가 감소
- `ES` §3.2 행 8, `CC` §8: "어느 layer에서 patch해도 3.8–7.4%".
- 원 결과는 범위 자체는 맞음. 단 내부 layer − Recent-GT 쌍대 차이: L28 −3.3%p p = 4.9e-4, L31 −3.6%p p = 2.4e-4, L35 −3.0%p p = 0.0034 (`PREV_ACTION_STATE_PATCHING.md` §5). "layer에 무관"이 아니라 "차이가 3%p 이내(효과 42.5%p 대비)". 보고서 §1·§7은 이를 정확히 기술.

### m-7. "거리도 방향도 아닌 장면 맥락이 지배" 표현
- `CC` §6. 원 결과: 장면 AUROC 0.80 vs 거리 0.61 (Q4), 장면 내 순위 상관 +0.09 [−0.01, +0.19] → 장면 우세 ✔. 그러나 풀링 Spearman +0.31 [+0.05, +0.56], 최대 거리 5분위 증폭 75.3%(최소 분위 50.7%, 중간 29–31%)로 거리 효과가 0은 아님. 표현 한정 필요.

### m-8. 실험 32 "같은 거리, GT 반대편"의 달성 거리
- `ES`/`CC`: "같은 크기의 오차를 GT 반대편에 두면 47.9 → 13.7%".
- 원 결과(`O/motion_semantics_ablation/analysis.md` 달성 기하): 반대편 행 d(sub, GT) = 0.458 m vs 자기 오차 e = 0.515 m(약 11% 작음); 방향오답 행 d = 0.318 m(크기오답 행 0.505 m보다 작음). 방향 > 크기 결론에는 보수적 방향의 차이지만, "같은 거리"는 근사.

### m-9. small codebook mismatch "NN 순위 8 → 15"
- `CC` §4: "오답 토큰의 NN 순위 중앙값이 8 → 15로 커짐". 원 보고서: 기존 결과(Add.4) 8번째 최근접; 이번 구 정의 그룹 재계산은 7(arm N) / 8(gpu1); 궤적 A− 기준 15. 비교 대상이 혼재(옛 수치). MINOR.

### m-10. 실험 5 GPU 표기
- `FIRST_MISMATCH_CAUSAL.md` 머리말 "GPU1"만 표기; `run_meta.json` gpu = RTX 5090, CUDA_VISIBLE_DEVICES = 1(+PCI_BUS_ID). memory 기록상 PCI_BUS_ID 미설정 시 GPU1 = 3080 Ti이므로 논문에는 run_meta 기준 "RTX 5090"으로.

### m-11. 파일 시각과 커밋
- 실험 5–10 결과 파일은 1541f30(2026-09-26 13:29 UTC)에 커밋, 로컬 `equal_distance_perturbation/records.jsonl`·`summary.json` mtime은 09-27~09-29(새 서버 재실행). `git status` clean → 커밋 내용과 동일하며, 재실행 records로 원 분석을 다시 돌리면 커밋된 summary와 동일(Q4 제외) — 출처 일관. 실험 11 보고서 실행일 09-27, 커밋 09-29(bf15767). 정보성.

### m-12. temporal window 표의 "Full 효과 대비" 기준
- `ES` §3.2 행 9 "1-step 47%, 2-step 76%, 3-step 90%, 4-step 96%"는 **증폭** 기준; FDE 기준은 58 / 82 / 92 / 97% (`TEMPORAL_FEEDBACK_WINDOW.md` §3). `ES` §1 항목 5의 Impromptu 비교("AutoVLA 47/76/90/96%")도 증폭 기준. 지표를 명시해야 함.

### m-13. horizon 통제 "25–29% 재발산"의 모집단
- `ES` §3.2 행 10, `CC` §9: "해제 후 25–29% 재발산". 원 결과: A− t* = 0, N = 4, **해제 시점 오차 ≤ 1 m였던 단위만**(2-step 24.7%, 3-step 25.0%, 4-step 28.8%; Normal 49.0%, Full 12.1%); CI가 넓음(예: 3-step [2.1, 45.5]). population 한정 명시 필요.
