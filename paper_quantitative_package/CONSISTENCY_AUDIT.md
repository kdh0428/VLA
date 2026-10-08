# CONSISTENCY_AUDIT — 요약 문서와 원 결과의 불일치 점검

## 검사 범위
- 저장소에 논문 초안 파일은 없습니다(`*.tex`, `*paper*`, `*draft*` 검색). 그래서 현재 "main summary"인 다음 두 문서를 원 결과와 대조했습니다.
  - `/root/VLA/EXPERIMENT_SUMMARY.md` (ES)
  - `/root/VLA/CONCLUSIONS.md` (CC)
- 각 실험의 RESULTS·PROTOCOL 문서 사이의 불일치도 함께 봤습니다.
- 원 결과 파일은 하나도 수정하지 않았습니다. 고칠 문구는 논문 쪽에 반영할 권고로만 적었습니다.

## 등급
- **CRITICAL**: 결과 방향이나 결론이 바뀌는 오류.
- **MAJOR**: N, CI, p, 구조, 주장 강도에 영향.
- **MINOR**: 반올림, 표현, 옛 수치, population 차이.

## 결론
**CRITICAL 0건.** raw data에서 재계산할 수 있었던 값은 모두 원 분석과 일치했습니다.
- SpatialVLA closed loop 전체
- navhard 225 그룹 평균 11개 조건 (차이 ≤ 6e-17)
- equal-distance (Q4 AUROC ±0.01 제외)
- dev / held-out ADE와 A− / FDE / McNemar

## 요청된 점검 항목

| 항목 | 판정 | 내용 |
|---|---|---|
| 47.1 vs 47.4 (AutoVLA Normal 증폭률) | MINOR (같은 population) | 같은 A− 365 단위를 서로 다른 batch로 재디코딩한 bf16 수치 차이입니다. 실험 9 46.6%, 실험 8 46.8%, 실험 11 47.1%, 실험 7 47.4%, 실험 32 47.9%이고, token 재현율은 86.5–89.2%입니다. 상대 효과는 반드시 같은 실험의 분자·분모로 계산해야 합니다. |
| AutoVLA "93% 제거" | MAJOR (분모 명시 필요) | (47.4 − 7.1) / (47.4 − 4.1) = 40.3 / 43.3 = 93.0%로, **GT-history 효과의 93%**를 뜻합니다. Normal 증폭률 대비 감소는 85.0%입니다. 실험 7은 이 비율의 CI를 내지 않았고, CI는 실험 8에서만 92.9% [89.3, 99.1]로 보고됐습니다. |
| Impromptu 63.4 → 1.3 | MAJOR (지표 명시) | 이 값은 **증폭률**(A− ∧ FDE5 > 3 m)입니다. 같은 행의 A−는 64.5 → 16.3%입니다. RESULTS §5D의 "failure 63 → 0–1%"는 표현 오류입니다. 절대 위치 GT 문맥은 GT 정보를 누출합니다. 절대 효과 −62.2%p [−65.6, −59.1]은 맞습니다. |
| SpatialVLA ensemble 설명 | MAJOR (문서 오류, 결과 무영향) | 실험 34 PROTOCOL §0은 "오래된 예측 가중치 큼"이라고 썼지만, 실제로는 최신 예측이 0.574 / 0.258 / 0.116 / 0.052 순으로 가장 큽니다(step 2–4 합 42.6%). 실험 35 PROTOCOL(8fc20bf)에 정정이 기록돼 있고, 실험 34 파일은 그대로 두었습니다. |
| SpatialVLA natural 85.0 vs 과제별 | MINOR (subset) | 68/80 = 85.0% [77.5, 92.5]이고, 과제별로는 coke 37/40 = 92.5%, move near 31/40 = 77.5%입니다. CI의 양 끝이 우연히 과제별 값과 같은 숫자이므로 혼동하지 않도록 주의해야 합니다. ES §6c에는 과제별 값만 있습니다. |
| SpatialVLA 1-step window 87% | MINOR (반올림) | 재계산하면 (0.05613 − 0.01624) / (0.05613 − 0.00984) = 86.3%입니다. 실험 34 RESULTS와 ES §6c는 87%라고 적었습니다. |
| OpenVLA official vs reproduced | MINOR | 공식 84.7%는 ES §6에만 있고 실험 28 파일에는 없습니다. 재현은 87/100인데 재실행은 81/100이고 불일치가 20%라서, 차이가 noise 범위 안입니다. |
| navhard rank / maxlog 두 번째 절반 재현 | MINOR (판정 일치) | 사전 등록 H1(F1)은 통과했습니다. H2(max log-lik)는 +0.0097 [−0.0037, +0.0235], p = 0.080으로 실패했습니다. rank-sum은 판정 대상이 아니었고 CI 하한이 −1.5e-5로 경계입니다. 전체 max log-lik +0.020은 효과를 처음 발견한 절반을 포함하므로 독립 검증이 아닙니다. |
| 안전 필터 몫 89% | MAJOR (반올림 규칙) | 0.104610 / 0.118313 = **88.4%**이고, 89%는 반올림한 값끼리 나눈 결과입니다. 또 이것은 Shapley가 아니라 "필터 먼저" 순서로 나눈 값입니다. 선택기를 먼저 넣으면 83.4%가 됩니다. |
| DA 기여 86% | MAJOR (분모 명시) | 2인 Shapley로 0.089673 / 0.104610 = 85.7%이며 재현됩니다. 분모는 **필터만의 이득**이고, F1 총 이득 대비로는 75.8%입니다. CC §9h는 단독 효과(+0.087, +0.012)와 Shapley 비율을 섞어 썼는데, Shapley 값은 +0.090과 +0.015입니다. |
| oracle 69% | MAJOR (사후 선택) | 0.121644 / 0.176669 = 68.85%이며 재현됩니다. 다만 분자 F1+maxlog는 결과를 본 뒤 고른 기술적 조건입니다. 사전 등록 F1 기준으로는 67.0%이고, 절반별로 63.8% / 75.0%이며, CI가 없습니다. |
| 탐지 AUROC | 일치 | 9개 탐지기의 pre/post AUROC는 ES §7, RESULTS, detection.json에서 모두 같습니다. 이탈 전 엔트로피의 "선택 = 버림" 표현만 MAJOR입니다(아래 D-M1). |
| 표본 수 | MINOR 몇 건 (population) | navtest 2,747 vs 2,748(fork 실패 1개). 실험 22 혼합 GPU 43,517 / 2,550 vs 실험 31 5090 전용 43,469 / 2,538. 실험 5의 N은 1,159 장면이며 equal-distance 208이 아닙니다(ES §10 오기, MAJOR). held-out 자연 실패 72(혼합) vs 73(5090). |

## 이 패키지를 만들면서 추가로 확인한 항목
- **[MINOR] 실험 35 PROTOCOL의 "+640 에피소드"는 계산 표기 오류입니다.** 규칙이 정한 seed 40–119는 1,280 에피소드이고, 그대로 실행했습니다. 실험 35 RESULTS에 기록했습니다.
- **[MINOR] 실험 35 분석 스크립트를 결과 commit 전에 1건 고쳤습니다.** seed ≥ 40에서 실험 34 비교 부분이 KeyError를 냈습니다. 통계 정의는 그대로입니다.
- **[MINOR] 비교표의 AutoVLA 자연 성능 한 칸에 서로 다른 모집단 값이 섞여 있습니다.** A− 1.9%는 52/2,747이고, FDE5 0.66 m는 500 장면 subset 값입니다. 실험 33 RESULTS §2가 같은 방식으로 섞어 적었습니다. `paper_cross_model_table.csv`에는 두 모집단을 각각 표기했습니다.
- **[MINOR] analyze_svla.py docstring은 "stratified by task"라고 하지만, 코드는 층화하지 않은 에피소드 bootstrap입니다.** 보고된 CI는 모두 층화하지 않은 값입니다. 반면 실험 35 사전 등록은 "과제로 층화"한다고 명시했으므로, 두 실험의 CI 방법이 다릅니다. 실험 35 RESULTS에 이를 함께 적어야 합니다.
- **[MAJOR] SpatialVLA closed loop의 opposite 교란 실패는 pick_coke_can에 집중됩니다.** feedback −37.5%p, p = 6e-5이고, move_near는 −2.5%p입니다. 원 RESULTS에는 합계만 있습니다. 결론(문맥 효과 0, 실행 효과 있음)은 바뀌지 않지만, 논문에서는 과제별 이질성을 밝혀야 합니다.

---

# 상세 (영역별 원문)


## A. AutoVLA 기전

### AutoVLA 기전 실험 — 출처 간 불일치 목록

경로 약어: `O/` = `/root/VLA/autovla_misalignment_poc/outputs/`, `ES` = `/root/VLA/EXPERIMENT_SUMMARY.md`, `CC` = `/root/VLA/CONCLUSIONS.md`.
등급: **CRITICAL** = 방향·결론이 바뀜, **MAJOR** = N·CI·p·구조(architecture)·주장 강도, **MINOR** = 반올림·표현·옛 수치.
요약: 원 RESULTS/summary.json과 반대 방향의 결론을 내는 CRITICAL 불일치는 **찾지 못했습니다**. 아래 MAJOR 항목은 논문 문구에서 반드시 고쳐야 할 것들입니다.

---

#### MAJOR

##### M-1. "같은 거리(8.8 mm)" — 매칭 오차를 perturbation 크기로 잘못 기술
- `ES` §3.2 표 행 6: "같은 거리(8.8 mm) 대안 token".
- 원 결과: 8.8 mm는 **대안 token 거리와 원래 오답 거리의 절대차 중앙값**(매칭 오차)입니다(`O/equal_distance_perturbation/summary.json` `alt_abs_dist_diff_median_m` = 0.00883; 상대차 중앙값 `alt_rel_err_median` = 7.9%). 실제 perturbation 거리는 A− 대안 중앙값 **0.129 m**(평균 0.179 m), A+ 대안 0.087 m (`table.<g>."alt (all)".dist`).
- `CC` §6은 "같은 거리(중앙값 상대차 7.9%, 절대 8.8 mm)"로 올바르게 기술.
- population 차이 아님 — 표현 오류(크기를 약 15배 과소 기술). 논문에는 "perturbation 0.13 m (matched within 8.8 mm / 7.9%)"로.

##### M-2. attention mask "효과 없음"은 전체 집합에서만 성립 (결과로 선택된 부분집합에서는 유의한 감소)
- `ES` §3.2 행 7 "attention mask는 효과 없음", `CC` §7 "attention mask는 효과가 없으므로, 기전은 긴 history 참조가 아니라 …".
- 원 결과(`O/action_history_causal/summary.json`, 보고서 `ACTION_HISTORY_CAUSAL.md`):
  - all perturbations A− (365): action-history mask −1.4%p [−8.5, +7.4], p = 0.63; recent-action mask −3.3%p [−9.7, +2.9], p = 0.18 → 효과 없음 ✔.
  - **previously amplified A− (175 단위, 45 장면)**: −16.0%p [−23.7, −6.2], p = 4.3e-06 / −19.4%p [−28.1, −10.1], p = 5.7e-08; A+ previously amplified (43): −37.2 / −48.8%p.
  - original token only A− (52): −11.5%p [−25.5, +5.4], p = 0.11.
- population 차이(부분집합이 equal-distance run의 결과로 선택됨 → 평균회귀 가능성; 그러나 같은 부분집합에서 Normal은 94.3%로 유지). 결론 방향(직전 1 token GT 교정 −81%p가 mask 효과보다 훨씬 큼)은 유지되지만, "mask는 효과 없음"은 "전체 perturbation 집합에서 유의한 효과 없음(증폭이 이미 일어난 단위에서는 −16~−19%p)"로 한정해야 함.

##### M-3. "L35 MLP 지배 재현" — 원 보고서 판정은 "△ 부분 재현"
- `ES` §2 행 4 "late action formation, L35 MLP 지배 재현", `CC` §4 "재현된 것: … L35 MLP 지배(attention 대비 7.8–21.5배)".
- 원 결과(`O/natural_fast_mechanism/NATURAL_FAST_MECHANISM.md` §2, `summary.json` `N.verdicts`): step-0 기준 L35 MLP 차이 −1.16 [−2.15, −0.45] → holds; **첫 불일치 step 기준 −0.34 [−0.89, +0.19], `L35_mlp_dominance[traj_first_mismatch].holds = false`** → 보고서 판정 "△ 부분 재현". 또 "7.8–21.5배"는 L32–35 MLP/Attention 합의 비(attention_weak 지표)이지 L35 MLP 단독 지표가 아님(N 첫 불일치 step 11.2×, C 첫 불일치 step 429×는 인용 범위에서 빠짐).
- population 차이(분석 시점 step-0 vs 첫 불일치 step). 주장 강도 과장.

##### M-4. 실험 5의 N을 equal-distance set으로 기술
- `ES` §10: "AutoVLA 기전 실험(5–11, 32)은 equal-distance set(208 장면, A− 52 장면)에 기반".
- 원 결과: 실험 5는 **첫 불일치가 있는 모든 장면 1,159개**(A− 52 + A+ 1,107; step-matched 유효 n 944) (`O/first_mismatch_causal/summary.json` `n`; `run_meta.json` `n_ok` 1159). equal-distance set(208)은 실험 6–11, 32에만 해당.
- population 차이(서술 오류). A− 52는 같음.

##### M-5. "37 layer" 표기 (구조)
- `ES` §3.2 행 8 "layer patching (37 layer)", `CC` §8 "37개 layer 전수".
- 원 결과: patch 위치 37개 = **embedding 출력 1 + decoder block L0–L35 36개** (`O/prev_action_state_patching/run_meta.json` `layer_keys`; 보고서 "Layer key: emb = block 0 입력, L{i} = decoder block i 출력"). Qwen2.5-VL-3B 언어모델은 36 layer. 논문에는 "embedding + all 36 decoder layers (37 patch sites)".

##### M-6. A− 라벨의 재디코딩 안정성 (N/정의)
- 실험 4–11의 A− 52 장면은 **저장된 natural run(arm N)**으로 정의됨.
  - 같은 52 장면을 실험 5 harness로 재생성하면 A− 유지 **82.7%(43/52)**, 17.3%는 A+ (`O/first_mismatch_causal/summary.json` `reproduction`; = original A+ 17.3%).
  - 실험 6 original의 "amplification"(A− ∧ FDE > 3 m)은 67.3% [51, 80] — A− 정의(82.7%)와 다른 지표.
  - 다른 natural run(gpu1, residual만 저장)에서는 A− 46건(step-0 오답 18) (`O/natural_fast_mechanism/summary.json` `G.*`).
  - best-of-N harness(T 0.01)에서 "실패 단위 52개"의 자연 계획 A−: 3080 Ti 63–65% vs 5090 67–71% (`O/gpu5090_reanalysis/GPU5090_REANALYSIS.md` "실험 18–19" 표).
- population/run 차이(서로 다른 디코딩 harness·GPU). 결론을 바꾸지 않지만, 논문에서 "A− 52 scenes"는 "stored natural-run failures (re-decoding retains 83%)"로 정의를 명시해야 함.

##### M-7. "효과의 93%가 사라진다"의 분모
- `ES` §1.2 "직전 token 1개만 GT로 바꿔도 효과의 93%가 사라집니다", `CC` §7 "전체 효과의 93%".
- 원 결과: 93% = (Normal − Recent-GT)/(Normal − GT-history) = (47.4 − 7.1)/(47.4 − 4.1) = 40.3/43.3 (`O/action_history_causal/summary.json` 평균; 실험 7은 이 비율과 CI를 보고하지 않음). 같은 비율의 CI는 실험 8에서만 보고: 92.9% [89.3, 99.1] (`O/prev_action_state_patching/summary.json` `groups.A-.all.fraction_of_gt_history_effect.recent_gt`). Normal 증폭률 대비 상대 감소는 85.0%.
- 표현 문제(분모 = GT-history 최대 효과). 논문에는 공식과 CI 출처(실험 8 harness)를 함께.

---

#### MINOR

##### m-1. A− Normal AR 증폭률 46.6 / 46.8 / 47.1 / 47.4 / 47.9%
- 실험 9 46.6% (`O/temporal_feedback_window`), 실험 8 46.8% (`O/prev_action_state_patching`), 실험 11 47.1% (`O/prev_action_identity_decomposition`), 실험 7 47.4% (`O/action_history_causal`), 실험 32 47.9% (`O/motion_semantics_ablation`). `ES` §3.3은 "47.1–47.9%", `CC` §9b·9c는 47.1%, `CC` §7은 47.4%, `CC` §9 표는 46.6%.
- **같은 population**(365 단위). batch 행 수(7/93/17/9/6)에 따른 bf16 수치 경로 차이; 실험 7 대비 Normal token 재현율 86.5–89.2%. 마찬가지로 GT-history 4.1/4.4%, Recent-GT 7.1/7.4/7.7/7.9%, A+ Normal 4.2/4.3/4.5%.
- 처리: 상대 효과는 반드시 같은 실험의 분자·분모로 계산(예: 실험 9의 "47%"는 46.6 기준).

##### m-2. A− ADE/FDE 2.27/6.63 m vs 2.91/8.18 m
- `O/natural_fast_mechanism/summary.json` `N.codebook_amplification.A-.{ade_m,fde_m}` = 2.27/6.63 (**중앙값**, 저장된 natural run).
- `O/first_mismatch_causal/summary.json` `A-.original.{ade5,fde5}.mean` = 2.91/8.18 (**평균**, 재생성 harness; equal-distance original도 동일).
- 같은 52 장면, 다른 통계량(중앙값 vs 평균) + 다른 run. 충돌 아님.

##### m-3. natural N 2,747 vs 2,748
- `O/natural_fast_mechanism` n = 2,747; `O/fast_vs_cot_unbiased/FAST_VS_COT.md` "전체 2,748개"; `O/annotations/` 파일 2,748개.
- `autovla_misalignment_poc/REPRODUCTION.md`: 장면 2,748개 중 arm N fork 성공 2,747. population 차이(1 장면 fork 실패).

##### m-4. equal-distance full vs strict
- full: A− 대안 313, 증폭 45.0%, 혼합 장면 71.2% (37/52); strict: 대안 292, 증폭 46.2%, 혼합 64.7% (33/51) (`EQUAL_DISTANCE.md` vs `EQUAL_DISTANCE_strict.md`).
- population 차이(strict는 전체 codebook fill 대안 제외, 대안이 모두 빠진 A− 1 장면 제외). 결론 동일. `ES`/`CC`는 full 값만 인용.
- 부수: `EQUAL_DISTANCE_strict.md` 머리말의 "장면당 대안 6.0/5.7개, 상대차 7.9%"는 스크립트가 필터 전 `recs`로 계산한 full 값(`S/analyze_equal_distance.py` `S["alts_per_scene"]`) — strict 값이 아님.
- 부수: strict `summary_strict.json`/`rows_strict.jsonl`이 출력 디렉토리에 없음(재실행으로 복원: `_parts/recomputed/`).

##### m-5. equal-distance Q4 AUROC 재현성
- 재실행 시 Q4(교차검증 로지스틱) AUROC가 소수 둘째 자리에서 최대 ±0.01 다름(예: A− 거리+방향 0.62 → 0.63). `REPRODUCTION.md`가 이미 "최대 0.009 차이(solver 수치)"로 기록. 다른 모든 수치는 비트 단위 동일. MINOR.

##### m-6. "어느 layer에서 patch해도 3.8–7.4%" vs 후반 layer의 유의한 추가 감소
- `ES` §3.2 행 8, `CC` §8: "어느 layer에서 patch해도 3.8–7.4%".
- 원 결과는 범위 자체는 맞음. 단 내부 layer − Recent-GT 쌍대 차이: L28 −3.3%p p = 4.9e-4, L31 −3.6%p p = 2.4e-4, L35 −3.0%p p = 0.0034 (`PREV_ACTION_STATE_PATCHING.md` §5). "layer에 무관"이 아니라 "차이가 3%p 이내(효과 42.5%p 대비)". 보고서 §1·§7은 이를 정확히 기술.

##### m-7. "거리도 방향도 아닌 장면 맥락이 지배" 표현
- `CC` §6. 원 결과: 장면 AUROC 0.80 vs 거리 0.61 (Q4), 장면 내 순위 상관 +0.09 [−0.01, +0.19] → 장면 우세 ✔. 그러나 풀링 Spearman +0.31 [+0.05, +0.56], 최대 거리 5분위 증폭 75.3%(최소 분위 50.7%, 중간 29–31%)로 거리 효과가 0은 아님. 표현 한정 필요.

##### m-8. 실험 32 "같은 거리, GT 반대편"의 달성 거리
- `ES`/`CC`: "같은 크기의 오차를 GT 반대편에 두면 47.9 → 13.7%".
- 원 결과(`O/motion_semantics_ablation/analysis.md` 달성 기하): 반대편 행 d(sub, GT) = 0.458 m vs 자기 오차 e = 0.515 m(약 11% 작음); 방향오답 행 d = 0.318 m(크기오답 행 0.505 m보다 작음). 방향 > 크기 결론에는 보수적 방향의 차이지만, "같은 거리"는 근사.

##### m-9. small codebook mismatch "NN 순위 8 → 15"
- `CC` §4: "오답 토큰의 NN 순위 중앙값이 8 → 15로 커짐". 원 보고서: 기존 결과(Add.4) 8번째 최근접; 이번 구 정의 그룹 재계산은 7(arm N) / 8(gpu1); 궤적 A− 기준 15. 비교 대상이 혼재(옛 수치). MINOR.

##### m-10. 실험 5 GPU 표기
- `FIRST_MISMATCH_CAUSAL.md` 머리말 "GPU1"만 표기; `run_meta.json` gpu = RTX 5090, CUDA_VISIBLE_DEVICES = 1(+PCI_BUS_ID). memory 기록상 PCI_BUS_ID 미설정 시 GPU1 = 3080 Ti이므로 논문에는 run_meta 기준 "RTX 5090"으로.

##### m-11. 파일 시각과 커밋
- 실험 5–10 결과 파일은 1541f30(2026-09-26 13:29 UTC)에 커밋, 로컬 `equal_distance_perturbation/records.jsonl`·`summary.json` mtime은 09-27~09-29(새 서버 재실행). `git status` clean → 커밋 내용과 동일하며, 재실행 records로 원 분석을 다시 돌리면 커밋된 summary와 동일(Q4 제외) — 출처 일관. 실험 11 보고서 실행일 09-27, 커밋 09-29(bf15767). 정보성.

##### m-12. temporal window 표의 "Full 효과 대비" 기준
- `ES` §3.2 행 9 "1-step 47%, 2-step 76%, 3-step 90%, 4-step 96%"는 **증폭** 기준; FDE 기준은 58 / 82 / 92 / 97% (`TEMPORAL_FEEDBACK_WINDOW.md` §3). `ES` §1 항목 5의 Impromptu 비교("AutoVLA 47/76/90/96%")도 증폭 기준. 지표를 명시해야 함.

##### m-13. horizon 통제 "25–29% 재발산"의 모집단
- `ES` §3.2 행 10, `CC` §9: "해제 후 25–29% 재발산". 원 결과: A− t* = 0, N = 4, **해제 시점 오차 ≤ 1 m였던 단위만**(2-step 24.7%, 3-step 25.0%, 4-step 28.8%; Normal 49.0%, Full 12.1%); CI가 넓음(예: 3-step [2.1, 45.5]). population 한정 명시 필요.

---

## B. Cross-model

### Cross-model 충돌·불일치 기록 (실험 28 / 33 / 34 + AutoVLA 참조값)

경로 기준: `/root/VLA/autovla_misalignment_poc/` (별도 표기 없으면). 원본 파일은 하나도 수정하지 않았습니다.
등급: **CRITICAL** = 논문 수치가 틀어짐 / **MAJOR** = 해석·비교 가능성에 영향 / **MINOR** = 반올림·문서 표현·subset 차이.

CRITICAL 등급 항목은 없습니다. raw 데이터에서 다시 계산한 값(SpatialVLA closed loop 전체, token step 변경률, ensemble 가중치)은 모두 `analysis.json`·RESULTS.md와 소수점 반올림 수준까지 일치했습니다.

---

#### MAJOR

##### M1. SpatialVLA ensemble 가중치 설명 오류 (실험 34 PROTOCOL.md §0)
- **실험 34 문서 기술**: `outputs/cross_domain_temporal_replication/PROTOCOL.md` §0 "ActionEnsembler(temp −0.8; 오래된 예측일수록 가중치 큼) … 실행 action의 대부분은 앞 step에 조건화된 chunk의 뒤쪽 step(2–4)에서 옵니다."
- **코드 (진실)**: `scripts/cross_domain_temporal/action_ensemble.py`에서 weights = exp(−temp·i), temp = −0.8입니다. 쌓는 순서는 `zip(range(n−1,−1,−1), history)`이고 index 0이 가장 오래된 chunk입니다. 따라서 **가장 최근 예측의 가중치가 가장 큽니다**.
  - 다시 계산한 값 (`_parts/scripts/svla_token_step_change.py`): 현재 chunk step 1 **0.5741**, 1 step 전 chunk step 2 **0.2579**, 2 step 전 chunk step 3 **0.1159**, 3 step 전 chunk step 4 **0.0521**.
  - step 2–4의 합은 **42.6%**로, "대부분"이 아니라 절반 미만입니다.
- **영향**: 실험 34는 공식 adapter 코드(`spatialvla_policy.PolicyState`, `ActionEnsembler(4, −0.8)`)를 그대로 썼으므로 결과에는 영향이 없습니다. 틀린 것은 설명 문장뿐입니다. 다만 논문에서 "ensemble이 feedback을 흡수한다"고 해석할 때는 42.6%라는 정확한 값을 써야 합니다.
- **이미 기록된 곳**: `outputs/spatialvla_feedback_protection_ablation/PROTOCOL.md` §0-1 (실험 35 사전 등록, commit 8fc20bf)에 같은 정정과 가중치 0.57/0.26/0.12/0.05, 43%가 적혀 있습니다. 실험 34 파일은 수정하지 않았습니다.
- **추가 사항 (MINOR)**: `action_ensemble.py`의 주석 "if temp > 0, more recent predictions get exponentially *less* weight"는 temp > 0일 때만 맞는 설명입니다. temp < 0을 쓰는 이 설정에서 오해를 부른 원인으로 보입니다.

##### M2. Impromptu "63.4 → 1.3%"는 *증폭률*이지 *실패율(A−)*이 아님
- `outputs/cross_vla_temporal_replication/analysis.json`
  - `CDE.all_units.means.normal.amp` = 0.6343, `…recent_gt.amp` = 0.0125 (N = 2,398 단위)
  - 차이 `CDE.all_units.vs_normal.recent_gt.amp` = −0.6218 [−0.6557, −0.5913]. McNemar p는 0.0으로 기록돼 있는데, 이는 underflow입니다.
- 같은 행의 **A−는 64.5% → 16.3%** (`…recent_gt.a_minus` = 0.1635, 차이 −48.1%p [−52.3, −44.4], p = 8e-244)로, 1.3%가 아닙니다.
- RESULTS.md §5D에는 "remove cause → failure 63 → 0–1%"라고 적혀 있습니다. `EXPERIMENT_SUMMARY.md` §1-5와 §6b는 "증폭 63.4 → 1.3%"로 정확하게 썼습니다. "failure"라는 표현은 **증폭 정의(A− ∧ FDE5 > 3 m)에 한정**해야 합니다.
- 해석상 제약 두 가지:
  - gt_history(0.0%)와 recent_gt에는 절대 위치 표현 때문에 **GT 정보 누출**이 섞입니다 (RESULTS §8).
  - 그래서 GT 정보 없이 원인만 확인하는 근거는 reverse(+81.9%p)와 near_gt ≈ recent_gt(p = 0.87)입니다.
- 절대 효과: −62.2%p [−65.6, −59.1]. log-cluster bootstrap(28 log, 2,000회, seed 0)으로 구한 값입니다.

##### M3. AutoVLA "reverse"는 다른 개입 방식
- AutoVLA 4.4 → 33.4%는 실험 8 (`outputs/prev_action_state_patching/PREV_ACTION_STATE_PATCHING.md` l.10, l.333)의 **reverse@emb**입니다. GT-history 문맥에 직전 위치의 Normal *embedding*을 삽입하는 residual patch입니다.
- Impromptu와 SpatialVLA의 reverse는 **token/텍스트 문맥 자체**를 바꿉니다. Impromptu는 gt_history 문맥에 자기 waypoint를 넣고, SpatialVLA는 step 2 = R, step 3 = normal로 둡니다.
- 비교표의 "Reverse" 행은 정성적으로만 비교할 수 있습니다. 기준선도 다릅니다: AutoVLA는 gt_history 4.4% (실험 8 harness), Impromptu는 0.0%, SpatialVLA는 full_ref 13.5%입니다.

##### M4. "Amplification" 정의가 모델마다 다름 (직접 비교 불가)
- AutoVLA·Impromptu: A− ∧ FDE5 > 3 m. 궤적 수준이고 open-loop입니다.
- OpenVLA: 하위 차원 중 하나가 |δ| bin 이상 움직이면 증폭입니다. step 안에서 정의됩니다.
- SpatialVLA: ‖Σ_{k=2..4}(v_k − r_k)‖ / ‖v_1 − r_1‖ ≥ 1. 정규화 translation 공간 기준입니다.
- 비교표에는 수치를 함께 적되, 열 간 크기 비교는 금지라고 caption에 명시했습니다.

---

#### MINOR

##### m1. SpatialVLA natural 85.0% vs task별 92.5% / 77.5% — population/subset 차이 (충돌 아님)
- 다시 계산한 값 (`_parts/scripts/svla_closed_loop_paired.py`):
  - 합계 68/80 = 85.0%
  - pick_coke_can 37/40 = 92.5%
  - move_near 31/40 = 77.5%
- 원본 값:
  - `analysis.json` `closed_loop.success.natural` = 0.85 [0.775, 0.925]
  - RESULTS §1은 "85.0% [77.5, 92.5] (coke 92.5%, move near 77.5%)"로 세 값을 모두 적었습니다.
  - `EXPERIMENT_SUMMARY.md` §6c는 task별 값(92.5%, 77.5%)만 적었습니다.
- **주의**: 85.0%의 CI 하한·상한(77.5, 92.5)이 우연히 task별 성공률과 같은 숫자입니다. 둘을 혼동하지 말아야 합니다.

##### m2. OpenVLA 공식 성공률 84.7% vs 재현 87%
- 공식 84.7%는 **`EXPERIMENT_SUMMARY.md` §6에만** 있습니다. 실험 28의 RESULTS.md, PROTOCOL.md, analysis.json에는 없습니다. 외부 논문 값이며 repo 안에서 검증할 수 없습니다.
- 재현값:
  - `outputs/cross_vla_replication/analysis.json` `natural_success.rate` = 0.87 (87/100)
  - 같은 정책을 batch 구성만 바꿔 다시 돌린 natural_rep = 81/100 (`natural_rep_vs_natural.rate_a` = 0.81)
- 즉 재현값 자체가 81–87% 범위로 흔들립니다(결과 불일치 20%). 87%와 84.7%의 차이(+2.3%p)는 이 재실행 변동 안에 있습니다. 진짜 충돌이라기보다 **noise 범위 안의 차이**입니다.

##### m3. SpatialVLA natural_gen: bootstrap CI와 McNemar의 판단이 다름
- `closed_loop.vs_natural.natural_gen`: +8.75%p, CI [+1.25, +16.25]로 0을 제외합니다. 반면 McNemar 9 vs 2, p = 0.0654입니다.
- RESULTS §1은 McNemar 기준으로 "유의하지 않습니다"라고 썼습니다. 논문에는 두 값을 모두 적어야 합니다.

##### m4. SpatialVLA 1-step window 87% vs 다시 계산한 86.3%
- `token_level.all.means.*.dev4`로 계산: (0.05613 − 0.01624) / (0.05613 − 0.00984) = **86.3%** (recomputed: `build_figure_cross_model.py`)
- RESULTS §3E와 `EXPERIMENT_SUMMARY.md`는 "87%"라고 적었습니다. 반올림·전사 차이로 보입니다. 2-step-only 62.6%는 RESULTS의 "63%"와 맞습니다.

##### m5. analyze_svla.py docstring과 코드의 bootstrap 방식이 다름
- docstring에는 "episode bootstrap CI (2,000, seed 0, **stratified by task**)"라고 적혀 있습니다.
- 실제 `boot()`는 (task, seed) cluster를 층화 없이 복원추출합니다(`rng.choice(cl)`, `random.Random(0)`을 호출마다 새로 만듦).
- PROTOCOL §5에는 "에피소드 cluster bootstrap"이라고만 적혀 있습니다. 보고된 CI는 모두 **층화하지 않은** 방식으로 계산됐습니다. 이 패키지의 재계산도 같은 함수를 import해서 썼습니다.

##### m6. OpenVLA 통계 구현과 PROTOCOL의 차이
- **permutation**: PROTOCOL은 "과제 단위 sign-flip, 20,000회"입니다. 코드(`analyze_cross_vla.perm_tasks`)는 2^10 = 1,024개 부호 조합을 **모두 열거하는 exact 검정**입니다. 결과적으로 더 엄밀하지만 문서와 다릅니다.
- **Phase C CI**: `boot_tasks(c_, 1000)`로 1,000회 resample입니다. PROTOCOL의 2,000회는 Phase B 통계에 대한 명시입니다.
- **p 하한 표현**: RESULTS는 "p ≥ 0.12"라고 썼지만, 실제 최소 McNemar p는 corrected_d8의 0.1153입니다 (`vs_natural.corrected_d8.p_mcnemar`). "p ≥ 0.115"가 정확합니다.
- **amendment 시각**: PROTOCOL 본문에는 "2026-10-03 19:40"이라고 적혀 있지만, commit 6d0375a의 시각은 19:32 +0000입니다(시간대 또는 표기 차이로 추정). 이 amendment는 Phase B 일부를 본 뒤에 추가됐고, 문서에도 그렇게 공개돼 있습니다.

##### m7. AutoVLA Normal 증폭률이 실험마다 다름 (harness 실행 차이)
- 실험 7 (`action_history_causal`): 47.4% [38, 57]
- 실험 8 (`prev_action_state_patching`): 46.8%
- 실험 9 (`temporal_feedback_window`): 46.6%
- 세 실험 모두 같은 365 A− 단위를 썼습니다. window 비율 47/76/90/96%는 실험 9의 46.6%와 4.1% 기준입니다. 비교표의 correction 행은 실험 7을 썼습니다.

##### m8. AutoVLA 자연 성능의 모집단 차이
- **A− 1.9%**: 52/2,747 (`outputs/natural_reference_stabilization/NATURAL_REFERENCE.md` l.14, "자연 실패 장면 52 + 나머지 2,695"). 장면 수 2,747과 Impromptu PROTOCOL의 2,748은 1개 차이가 납니다.
- 같은 문서의 모집단 가중 normal 행은 1.24%이고, `fast_vs_cot_unbiased/FAST_VS_COT.md`의 500 장면 subset에서는 A+ 98.8%, 즉 A− 1.2%입니다.
- **FDE5 0.66 m**는 500 장면 subset 값입니다.
- 실험 33 RESULTS §2의 "A− 1.9%, FDE5 0.66 m"는 서로 다른 모집단의 값을 한 칸에 섞어 놓은 것입니다. subset 차이이며 충돌은 아닙니다.

##### m9. Impromptu task-level 비교에서 기준선 정의가 다름
- **교란 후 A+ → A− 49.5%**: 분모는 **pert_m0가 A+**인 단위입니다 (`AB.A.m0.2.flip_to_a_minus_given_m0_ok`, N = 14,424 = 1,803 장면 × 8).
- **재실행 기저 4.6%**: 분모는 **natural이 A+**인 장면입니다 (`AB.A.baseline_flip_natural_ok_to_rep_a_minus`, N = 1,818).
- 두 기준선이 다르지만, 둘 다 RESULTS §4–5에 기준선으로 공개돼 있습니다. paired 기준 효과(+24.8%p [+19.4, +32.2])는 pert_m0 대비 값입니다.

##### m10. AutoVLA 장면 수준 matched-magnitude 수치의 의미
- `EXPERIMENT_SUMMARY.md` §3.2의 "같은 거리(8.8 mm) 대안 token"에서 8.8 mm는 교란의 크기가 아닙니다. 대안 token과 원래 token의 GT 오차 거리 차이의 **중앙값**입니다 (`equal_distance_perturbation/EQUAL_DISTANCE.md` l.3).
- 비교표의 "Perturbation size" 행에는 그 의미대로 적었습니다.

##### m11. SpatialVLA closed-loop task별 이질성 (RESULTS에 없던 값, 충돌 아님)
- 다시 계산한 값 (`spatialvla_closed_loop_paired.csv`): opposite 교란의 실패는 **pick_coke_can에 집중**됩니다.
  - pick_coke_can: feedback −37.5%p [−52.5, −22.5], p = 6.1e-5 / corrected −30.0%p, p = 0.004
  - move_near: feedback −2.5%p, p = 1 / corrected −12.5%p, p = 0.36
- RESULTS에는 합계 값만 있습니다. task별 검정력은 40 에피소드로 낮습니다.

##### m12. 실험 33 RESULTS의 SpatialVLA 제외 사유는 이후 실험으로 무효화됨
- 실험 33 RESULTS §1과 PROTOCOL §0은 SpatialVLA를 "LIBERO checkpoint 비공개, Vulkan 없음"을 이유로 제외했습니다.
- 실험 34에서 fractal checkpoint와 Mesa lavapipe로 SpatialVLA를 실행했습니다. 시점 차이일 뿐 충돌은 아닙니다.

##### m13. 반올림
- 실험 34 RESULTS의 corrected_opposite "63.7%"는 51/80 = 63.75%입니다. 63.8로 쓰는 것이 일반적입니다.
- 이 패키지의 CSV에는 63.75로 적었습니다.

---

## C/D. 탐지·선택 (D-M1 = 이 파일의 M1)

### Conflicts / discrepancies — 탐지·선택 (실험 12–24, 31/P4)

등급: **CRITICAL** = 논문 주장이 원자료와 맞지 않음, **MAJOR** = 표현·범위를 고쳐야 함, **MINOR** = 모집단·반올림·문서 차이(주석으로 충분).
경로 `O = /root/VLA/autovla_misalignment_poc/outputs`.

검토 결과 **CRITICAL은 없습니다**. 탐지 AUROC 9개 행 × pre/post는 EXPERIMENT_SUMMARY §7, RESULTS.md, detection.md, detection.json 사이에서 모두 일치합니다(반올림까지).
선택 결과(dev/held-out PDMS, A−, 충돌, F1)도 요약 문서와 summary JSON이 일치합니다.

#### MAJOR

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

#### MINOR

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

---

## E. navhard

### Conflicts / discrepancies — navhard (실험 25–27, 29, 30)

검사 방법: 각 RESULTS.md·요약 문서의 수치를 분석 JSON과 그룹 점수 파일에서 다시 구한 평균(`_parts/scripts/navhard_verify.py`)과 대조.
225 그룹 평균 11개 조건 모두 JSON과 일치(차이 ≤ 6e-17), 개선/악화/동일 그룹 수도 모두 일치. N(76 log / 225 그룹, 36+40, 105+120, token 450 + 5,462 = 5,912)은 모든 출처에서 같습니다.
**CRITICAL 등급의 불일치(원자료와 다른 수치, 판정 뒤집힘)는 발견되지 않았습니다.**

#### MAJOR

1. **oracle "69%"의 분자가 사전 등록 주 방법이 아님** — `outputs/candidate_oracle/RESULTS.md` "선택 몫", `EXPERIMENT_SUMMARY.md` §5.
   69% = (F1+maxlog − 선택 없음)/(oracle17 − 선택 없음) = 0.121644/0.176669 = 68.85%로 재현됩니다. 그러나 F1+maxlog는 실험 26 사전 등록에서
   "판정 대상이 아닌 기술적 비교"였고, "최선 방법"은 전체 225 그룹 결과를 보고 고른 것입니다. 사전 등록 주 필터 F1 기준으로는 **67.0%**
   (0.118313/0.176669), 남은 여지 +0.0584. 절반별로도 63.8%(첫) vs 75.0%(두 번째)로 차이가 큽니다. 비율에는 CI가 없습니다.
   권고: 논문에는 "F1+maxlog 기준 69% (F1 기준 67%)"로 함께 쓰고, CI가 없음을 밝힐 것.

2. **"89%"는 반올림된 값의 비율; 정확한 값은 88.4%** — `outputs/safety_filter_ablation/RESULTS.md` 핵심 분해, `EXPERIMENT_SUMMARY.md` §5 및 표("~89%").
   (filter_only − normal)/(F1 − normal) = 0.104610/0.118313 = 0.8842. "89%"는 0.105/0.118 = 0.8898에서 나옵니다. 보수인 "rank-sum 11%"도
   정확값은 11.6%(반올림 12%)입니다. 숫자 자체는 원자료와 맞지만 반올림 규칙이 틀려 논문에는 88%(또는 "약 9할")로 써야 합니다.
   또한 이 분할은 "필터 먼저" 한 순서의 분할이며 Shapley가 아닙니다(선택기 먼저 순서면 필터 몫 83.4%, 파생값).

3. **"86%"의 분모는 F1 총 이득이 아니라 필터만의 이득** — `outputs/safety_filter_components/RESULTS.md`, `EXPERIMENT_SUMMARY.md` 결론 8 ("필터 이득의 86%"),
   §5. 2인 Shapley DA = ½(0.086848 + 0.092558) = 0.089673, 분모 both − none = 0.104610 → 85.7%로 **재현됩니다**. 다만 F1 총 이득(+0.1183) 기준이면
   75.8%이므로, "F1 개선의 86%가 DA"로 읽히지 않게 분모를 명시해야 합니다. Shapley 값의 CI·p는 보고되지 않았습니다.
   `CONCLUSIONS.md` §9h는 "주행 가능 영역 제약 +0.087(Shapley 86%), 충돌 제약 +0.012(14%)"로 **단독 한계 효과(+0.087, +0.012)와 Shapley 비율을 섞어** 씁니다.
   Shapley 값은 +0.090, +0.015입니다(RESULTS.md 자체는 올바름). +0.087/0.105 = 83.0%이므로 숫자 쌍이 서로 맞지 않습니다.

#### MINOR

4. **rank/maxlog 두 번째 절반 재현 서술** — 판정 자체는 모든 출처에서 같고 원자료와 일치합니다: H1(F1) 통과, H2(max log-lik) 실패
   (+0.0097 [−0.0037, +0.0235], p 0.080). rank-sum은 사전 등록상 판정 대상이 아니며, 두 번째 절반 CI 하한이 −1.5e-5로 "−0.000"으로 표기됩니다.
   `RESULTS.md` 표의 "[−0.000, +0.040]"은 사실상 0 경계이므로 "기준 미충족(경계)"로 쓰는 것이 정확합니다. `EXPERIMENT_SUMMARY.md` §5 표에는 max log-lik의 전체
   +0.020 [+0.008, +0.033]이 "(두 번째 절반에서 미재현)" 주석과 함께 나오는데, 이 전체 수치는 효과를 발견한 첫 절반을 포함합니다(RESULTS.md는 이를 명시).
   전체 rank-sum은 CI가 0을 넘지만([+0.0028, +0.0372]) permutation p 0.094 / Wilcoxon 0.082로 유의하지 않아 CI와 p의 결론이 엇갈립니다(서로 다른 검정, 모순 아님).

5. **max log-lik 전체 CI 상한 반올림** — JSON `pooled.rules.max_loglik.epdms.ci95[1]` = 0.032483 → 소수 셋째 자리 0.032. `RESULTS.md`와
   `EXPERIMENT_SUMMARY.md`·`CONCLUSIONS.md`는 "+0.033"(넷째 자리 0.0325를 다시 반올림한 이중 반올림). 논문 표는 +0.0325(넷째 자리) 또는 +0.032.

6. **"token의 31%"** — 정확값 1,867/5,912 = 31.58%(첫 절반 31.42%, 두 번째 31.73%). 반올림하면 32%. "약 31–32%" 또는 31.6%로 쓸 것.
   또한 이 비율은 1단계·2단계 token을 합친 비가중 비율이며(EPDMS의 2단계 가중과 다름), 안전 필터의 "모두 걸림" 27.5%(현재 frame 플래그 기준)와는 다른 양입니다.

7. **"이득의 99–104%가 곱셈 항"** — 세 필터 조건 6개 값은 98.7–103.6%(F1+maxlog 2단계 98.7%). 충돌 제약만 조건은 83.3%/90.4%로 범위 밖.
   서술 범위가 세 조건에 한정됨을 명시할 것. 이 비율들은 token 비가중 평균 기준(공식 단계 집계와 다름, 스크립트 docstring에 명시).

8. **계획 도달 거리(plan_reach.json) 출처 스크립트 없음, CI 없음** — `outputs/safety_filter_components/plan_reach.json`(커밋 0ab33a4)을 만든 스크립트가
   `scripts/`, `tools/`에서 확인되지 않습니다. "바뀐 장면에서 −2.5 ~ −2.9 m, 전체 19.1 → 18.5 m(−3%)"는 JSON 값과 일치하지만(−2.47/−2.65/−2.88 m;
   19.134 → 18.543 m, −3.1%) 재현 경로와 불확실성 추정이 없습니다. 논문에는 기술 통계로만 사용.

9. **Shapley 상호작용 표기** — `safety_filter_components/RESULTS.md`는 "DA +0.090 (86%), 충돌 +0.015 (14%), 상호작용 +0.006"을 나란히 적어 합(0.111)이
   필터 이득(0.105)과 다르게 보입니다. 상호작용은 두 Shapley 값에 반씩 이미 포함되어 있습니다(0.0897 + 0.0149 = 0.1046).

10. **실험 25 원보고서에는 p 값 없음** — `navhard_eval/NAVHARD_CLOSED_LOOP.md`는 CI만 보고합니다. 첫 절반 p 값(F1 2.5e-4, max log-lik 0.0072 등)은 실험 26의
    재채점(`navhard_full_comparison.json` `half1`)에서 나온 것이며, Δ·CI는 실험 25와 정확히 같습니다(같은 디코딩, 같은 picks).

11. **실험 29·30 분석 스크립트 커밋 시점** — `scripts/analyze_navhard_conditions.py`와 `navhard_score_decomposition.py`는 프로토콜 커밋(0cffc15, 14:44) 뒤,
    결과 커밋(0ab33a4, 15:17) 전인 af2eb10(15:10)에 커밋되었습니다. 통계 방법은 프로토콜에 미리 적혀 있었고("P2와 동일") 실험 27 스크립트와 같은 구현이지만,
    엄밀히는 스크립트 자체가 프로토콜 커밋에 포함되지 않았습니다. 실험 26 분석 스크립트의 사전 등록 후 변경은 출력 서식 한 줄뿐입니다.

12. **"F1 +0.118 중 필터 +0.105, rank-sum +0.014"의 반올림 합** — 0.105 + 0.014 = 0.119 ≠ 0.118 (정확값 0.10461 + 0.01370 = 0.11831). 반올림 효과일 뿐 원자료는 정확히 더해집니다.

#### 요약 문서 vs 원자료 (일치 확인)

| 요약 문장 | 원자료 | 판정 |
|---|---|---|
| EXPERIMENT_SUMMARY §5 표 11개 조건 EPDMS·Δ·CI·그룹 수 | 4개 JSON | 일치 (max log-lik CI 상한 반올림만 #5) |
| "F1 두 번째 절반 +0.112 [+0.082, +0.140], p 5e-5 재현" | `half2.rules.F1` | 일치 |
| "max log-lik +0.010 [−0.004, +0.024] 미재현" | `half2.rules.max_loglik` | 일치 |
| "필터 자체 +0.105(89%)" | `ablation.json` | 값 일치, 비율 88.4% (#2) |
| "DA 86%, 충돌 14%" | `components.json` | 재현됨 (분모 주의 #3) |
| "oracle 0.405, 69%, 남은 여지 +0.055" | `oracle_comparison.json` | 재현됨 (분자 선택 #1) |
| "token의 31%는 모든 후보 0점" | `token_scores.npy` | 31.6% (#6) |
| "필터 통계 47.6% / 27.6% / 8.6, 48.4% / 27.4% / 8.5" | `filter_stats.json` 두 개 | 일치 |
| "자연 계획이 바뀌는 장면 8.5% / 18% / 20.5%" | `plan_reach.json` `n_changed` / 5,912 = 8.5 / 18.1 / 20.5% | 일치 |

---
