# P0-C: 실험 35 공통 seed 고정축 paired 재분석 — 분석 protocol

- 작성: 2026-10-08. **이 문서는 아래 수치를 계산하기 전에 작성했다.**
- **사전 등록이 아니다.** 실험 35의 결과(RESULTS.md, analysis.json, error_curves.png)는 이미 공개·열람된 상태다. 이 재분석은 이미 본 자료에 대한 사후(post-hoc) 재분석이며, 여기 적은 정의는 "계산 전에 고정한 분석 규칙"일 뿐 확증적 검정의 지위를 갖지 않는다.
- 계획 문서: `/root/VLA/VLA_experiment_plan_20261008.md` §2, §5(P0-C).
- 새 추론·GPU 사용 없음. 원 파일 수정 없음. 출력은 이 디렉토리에만 쓴다.

## 1. 자료

| 자료 | 경로 | 내용 |
|---|---|---|
| 주 자료 | `outputs/spatialvla_feedback_protection_ablation/rollouts/episodes.jsonl` | 6설정(r1e4, r1e2, r1e1, r2e2, r2e1, r4e1) × natural/feedback/corrected/reverse × 2 task × seed 0–39 (+ natural_gen: r1e4, r4e1) |
| 확장 자료 | `.../rollouts_ext/episodes.jsonl` | r1e4, r4e1 × 4조건 × 2 task × seed 40–119 |
| 코드(읽기만) | `scripts/spatialvla_protection/{svla_protection.py, analyze_protection.py}`, `scripts/cross_domain_temporal/{spatialvla_core.py, svla_rollouts.py}` | 실행기·지표·perturb_token 정의 |
| tokenizer | `/root/VLA/spatialvla/spatialvla-4b-224-sft-fractal` processor (CPU 로드, 모델 가중치 불필요) | translation token → 정규화 xyz / 물리 world vector, clipping 판정 |

평가 단위 = episode (task, seed). 조건·설정은 모두 같은 (task, seed)의 초기 상태에서 실행됐으므로 episode 안에서 paired다.

## 2. 비교 (모든 설정 간 비교는 seed 0–39, 80 episode, 두 task 공통)

설정 내 대비 (각 설정 c):
- `R−N`: reverse 대 natural (실행은 같고 문맥만 교란)
- `F−C`: feedback 대 corrected (실행 교란은 같고 문맥만 다름)

설정 간 대비(difference-in-differences, DiD) — 고정축별 모든 쌍:

| 축 | 대비 (후자 − 전자) |
|---|---|
| e1 고정 (재계획 간격) | r2e1 − r1e1, r4e1 − r1e1, r4e1 − r2e1 |
| r1 고정 (ensemble) | r1e2 − r1e1, r1e4 − r1e1, r1e4 − r1e2 |
| r2 고정 (ensemble) | r2e2 − r2e1 |

총 7개 DiD. 각 DiD는 episode 단위로 `Δ_c2(episode) − Δ_c1(episode)`를 만든 뒤 평균한다(같은 episode의 두 설정 값을 짝지음).

확장 결과(별도 보고): r1e4, r4e1의 seed 0–119(각 240 episode)에서 설정 내 R−N, F−C와 r4e1 − r1e4 DiD(240 대 240 paired). r4e1 − r1e4는 r과 e가 동시에 바뀌므로 고정축 비교가 아니며 재계획만의 효과로 해석하지 않는다. **240 episode 결과를 80 episode 결과와 무대응으로 비교하지 않는다.** seed 0–39 부분과 40–119 부분을 나눈 값도 함께 보고한다.

## 3. 지표 (모두 episode 단위; X 대 Y는 R 대 N 또는 F 대 C)

1. **성공**: 마지막 step의 `done`(기록된 `success`). 대비 = success_X − success_Y (R−N, F−C; %p). 음수 = 문맥 교란(R) 또는 교정 부재(F)가 실패를 늘림.
2. **실제 TCP 위치 편차** d(t) = ‖p_X(t) − p_Y(t)‖ (기록된 `pose[:3]`, 각 step 실행 후, m→cm).
   - `mean_dev`: t ∈ [8, 79] 평균 (RESULTS.md의 "평균 오차"와 같은 정의 — 비교용).
   - `final_dev`: d(79).
   - `max_dev`: max_t d(t), t ∈ [0, 79].
   - `area_dev`: Σ_{t=8}^{79} d(t) (cm·step). mean_dev × 72와 같으므로 독립 지표가 아님을 명시한다.
   - `post_area_dev`: Σ_{t=24}^{79} d(t) (window 이후 지속분).
3. **command proxy**: c_X(t) = Σ_{s≤t} a_X(s)[:3] (실행 world_vector 누적합). `cmd_final` = ‖c_X(79) − c_Y(79)‖ (RESULTS의 "궤적 발산"과 같은 정의), `cmd_max` = max_t ‖c_X(t) − c_Y(t)‖. 실제 TCP가 아니라 명령의 누적이며 접촉·관절 한계·controller를 반영하지 않는다.
4. **실행 action 차이**: ‖a_X(t)[:3] − a_Y(t)[:3]‖의 window(t ∈ [8, 23]) 평균과 전체 평균.
5. **재정렬 시간 (censoring 규칙, 계산 전 고정)**:
   - 정의: t ≥ 24에서 d(t), …, d(t+3) ≤ 2 cm를 만족하는 첫 t (RESULTS와 동일, τ = 2 cm, hold = 4).
   - 시작 가능한 마지막 t는 76. 그때까지 만족하지 않으면 **미재정렬 = 실패로 보고 T = 80으로 censor**한다.
   - 조기 종료(episode 길이 < 80; truncation): 남은 step을 관측할 수 없으므로 같은 방식으로 T = 80(미재정렬)로 취급한다. 길이 분포를 manifest에 기록한다(데이터 점검 결과 2,080 + 1,280개 모두 80 step이었으나, 규칙은 그대로 둔다).
   - 보고: `realigned`(재정렬 여부 비율), `realign_t80`(censor 값 80을 넣은 평균 = 80 step까지의 제한 평균), 재정렬된 episode의 중앙값(기술 통계만).
6. **실패 형태(보조, 기술)**: coke의 never_grasped / dropped, move_near 실패 유형 — 조건별 count만.

## 4. dose / 노출 / clipping (설정별, 조건별; 기술 통계)

- 교란 chunk 수: t ∈ [8, 23]에 생성된 chunk 수 (r1 = 16, r2 = 8, r4 = 4가 예상값 — 실제 기록으로 셈).
- 무효 교란: p = g인 chunk 비율.
- 요청 거리 0.3 대비 실제 정규화 거리 ‖v(p) − v(g)‖ (평균·중앙·최소·최대), 목표점 v(g) − 0.3·v̂(g)와 v(p)의 양자화 오차, |v(g)| < 0.3이라 목표가 원점을 넘어 방향이 뒤집히는 비율.
- 물리 step-1 translation 이동 ‖w(p) − w(g)‖ (m/step), 그 중 실제 실행된 몫 = (그 step에서 age-0 예측의 ensemble 가중치) × ‖w(p) − w(g)‖ (F, C만; R은 0). episode 누적 = 직접 실행 dose.
- 문맥 노출: 교란 chunk의 예측 중 age ≥ 1(문맥에 조건화된 step 2–4)이 실행 action에 들어간 가중치의 합(F, R). age 0의 rotation·gripper는 같은 step 안의 문맥 영향(F, R)으로 별도 합.
- 실행 섭동량: window 안 ‖a_X(t) − a_N(t)‖[:3] 합/평균(X ∈ F, C, R; 같은 설정 natural과 비교). C 대 N은 직접 실행 교란 + 이후 궤적 분기, R 대 N은 문맥만.
- clipping: (a) p·g translation token의 구면→직교 변환이 [-1, 1]을 벗어나 tokenizer가 clip하는 비율, (b) 정규화 xyz가 경계(±1)에 닿는 비율, (c) 기록된 exec token이 `perturb_token(g)`의 재계산과 일치하는지(구현 점검).
- sticky gripper: |a(t)[6]| > 0.5인 step 수, window 안과 전체의 sticky 시작 횟수, natural 대비 시작 시점이 달라진 episode 비율.
- 기록 점검: 실행 world_vector = ensemble 가중 평균(preds, ages)인지 재계산 오차.

## 5. 통계

- **CI**: task로 층화한 episode bootstrap, 2,000회, `numpy.random.default_rng(0)`. 매 replicate에서 task별로 (task, seed) episode를 복원 추출하고, **그 episode의 모든 설정·조건 값을 함께** 사용한다(모든 지표·대비가 같은 resample index를 공유). 95% percentile CI.
- **설정 내 대비**:
  - 성공 R−N, F−C: exact McNemar (이산 쌍의 이항 검정, 양측).
  - 궤적 지표(편차는 정의상 ≥ 0): 설정 내에서는 0과의 검정을 하지 않는다. 평균과 CI만 보고하고, r1e4·r4e1은 natural_gen 대 natural(재실행 noise) 값을 기준선으로 함께 보인다.
- **DiD**:
  - 성공: episode별 DiD ∈ {−2, …, 2}에 sign-flip permutation(10,000회, seed 0, 양측) + bootstrap CI.
  - 연속 지표: paired Wilcoxon signed-rank(양측, zero_method="wilcox") + sign-flip permutation + bootstrap CI.
  - 재정렬 시간: censor 값 80 포함 차이에 Wilcoxon; 재정렬 여부 차이(0/1)는 sign-flip.
- **다중 비교**:
  - 1차 family: (지표, 대비 유형) 하나마다 7개 DiD를 하나의 family로 보고 Holm 보정 p를 보고한다.
  - 전체 민감도: 모든 DiD 검정(지표 × {R−N, F−C} × 7)에 Benjamini–Hochberg FDR q를 함께 보고한다.
  - 설정 내 성공 McNemar 12개(6설정 × 2대비)에 Holm 보정 p.
  - 주 지표로 미리 지정: 성공 DiD와 `area_dev` DiD. 나머지는 보조.
  - 이 재분석의 어떤 p도 확증적 결과로 쓰지 않는다(사후 재분석).
- **task별 결과**: 모든 설정 내 대비와 DiD를 task별로도 계산(task 안 episode bootstrap, 같은 seed; p는 같은 검정).
- 비유의성을 효과 0으로 쓰지 않는다. CI 폭으로 정밀도를 함께 해석한다.

## 6. +1.8 cm 확인

RESULTS.md의 "ensemble 제거 +1.8 cm"가 어떤 지표·비교·검정·보정인지 `analyze_protection.py` 코드로 확인하고, 같은 값을 재계산해 대조한다. 이어 같은 지표를 고정축 DiD(r1e1 ↔ r1e2 ↔ r1e4)와 Holm 보정으로 다시 보고한다.

## 7. 남는 교란 요인

- r에 따라 교란 chunk 수(16/8/4)와 window 종료 위치의 노출이 다르다. dose_table에서 정량하고, e1 축 DiD는 "재계획 간격 + 교란 dose"의 복합 효과로만 해석한다. dose로 사후 정규화한 효과는 결론에 쓰지 않는다(P1-D에서 문맥 예산 고정 / 실행 섭동량 통제 설계로 분리).
- ensemble 축(r 고정)은 교란 chunk 수가 같지만, 실행 가중치(age-0 가중치)가 달라 직접 실행 dose와 문맥 노출이 함께 바뀐다. 이것도 dose_table에 같이 둔다.

## 8. 산출물

`manifest.json`, `condition_summary.csv`, `paired_effects.csv`, `did_effects.csv`, `dose_table.csv`, `extension_240.csv`, `figures/*.png`, `analysis.md`(한국어; 확인한 사실/계산한 결과/미확인/가설 구분). 분석 코드는 `code/` 아래에 둔다.

## 부록 (계산 후 추가, 2026-10-08) — protocol 이탈 기록

- 1차 계산 후 dose_table에서 r1e1의 sticky gripper 시작 횟수가 feedback과 corrected 사이에서 크게 다른 것을 보고, **window 안 sticky 시작 횟수 차이(`sticky_onsets_win_diff`)와 sticky step 수 차이(`sticky_steps_diff`)를 설정 내 paired 지표(Wilcoxon)와 DiD 지표로 추가**했다. 이 두 지표는 위 §3–5를 작성할 때 정하지 않았으므로 탐색적 결과로만 보고하며, BH 전체 q 계산 family에 포함됐다(포함 여부가 다른 지표의 Holm family-7 p는 바꾸지 않는다).
- 그 밖의 정의·검정·family는 바꾸지 않았다.
