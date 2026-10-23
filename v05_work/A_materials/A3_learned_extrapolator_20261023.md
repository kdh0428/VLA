# A3. AutoVLA 학습형 비자기회귀(non-AR) 운동 외삽기 (learned prefix extrapolator, "E_prefix")

작성 2026-10-23, CPU 전용 조사. 모든 값은 아래 파일에서 직접 읽었다. 이 문서는 새로 학습하거나 평가하지 않았다.

## 0. 위치와 버전
- 사전 등록 프로토콜: `/root/VLA/autovla_misalignment_poc/outputs/review_followup3/P2/LEARNED_EXTRAPOLATOR_PROTOCOL.md`
  - sha256 `b7e49a6e610b3ad62194ba344cf3175a54f77f917dce43984e92000b55bad6f5`
  - 잠금 시각 2026-10-14T14:44:28Z (`extrap_protocol_lock.json`). 잠금은 학습 데이터 추출, 학습, 모든 P2-A 신규 기준선 결과보다 앞선다.
  - 학습·평가 스크립트는 이 파일이 바뀌면 실행을 거부한다(`check_lock`).
- 스크립트: `/root/VLA/autovla_misalignment_poc/scripts/review_followup3/`. 네 파일 모두 커밋 `5b158f2`("review_followup3 P2 … non-AR learned extrapolator …")에서 마지막으로 바뀌었다. 작업 트리 blob은 HEAD blob과 같다(미커밋 변경 없음).

| 파일 | 역할 | sha256 (작업 트리) | git blob |
|---|---|---|---|
| `p2_extrap_common.py` | 특징량/타깃/조립 함수 | `8cfcf491cf869d6773815147fdfa9eac5b5590284995c5afc5fee48a6053c62e` (잠금 파일과 일치) | `8c6a5751…` |
| `p2_extrap_data.py` | trainval 스트리밍, 분할, 샘플 생성 | `a8141ee51b8a31d24bc34574a81ea32a736f9902a85d388342061a072cdecd32` | `58c60ba0…` |
| `p2_extrap_train.py` | 그리드/최종 학습, 선택 | `47efd816591e7c09fc795086b1d30ff5079e7898bed088f14f930cc004b07901` | `18021cc5…` |
| `p2_extrap_eval.py` | P1-A 단위 평가 | `8638bb5d4e141dec45332a2767c2f8b31d67fbc9ed82ca12d04a69889d2b28ea` | `310c5b53…` |

- 산출물 디렉터리: `/root/VLA/autovla_misalignment_poc/outputs/review_followup3/P2/`. 하위 항목:
  - `_extrap_data/`: `samples.npz`, `trainval_ego_frames.parquet`
  - `_extrap_models/`: 가중치 `.pt`, 학습 로그 `*_log.csv`, `selected.json`
  - `extrap_*.csv/json`
  - `LEARNED_EXTRAPOLATOR_RESULTS.csv`
- 성격: 이것은 작은 **운동(motion) 모델**이다. VLA의 비자기회귀 버전이 아니며, 프로토콜이 그렇게 부르는 것을 금지한다. 사후(post-hoc) 대조 분석이며, 사전 등록은 P1-A 롤아웃이 이미 존재한 뒤에 이루어졌다.

## 1. 입력 특징량
**주 입력 (E-prefix)**: 79차원. CM과 같은 정보, 즉 디코딩된 계획 포즈 0..j와 위치 j만 담는다.
- 포즈 j 좌표계에서 본 원점과 포즈 0..j: 10 슬롯 × (x/10 m, y/10 m, heading rad)
- 0..j 스텝의 몸체(body) 증분: 10 슬롯 × (fwd/10, left/10, dyaw)
- 슬롯 마스크 10과 j의 one-hot 9
- j 이후 슬롯은 0으로 채우고 마스크한다. 함수는 j 이후를 읽지 않으며, 이는 쓰레기 값 단위 테스트로 강제된다(`extrap_tests_output.txt`).

**보조 변형 (E-prefix+ego)**: 84차원 = 주 입력 + 속도 |v|/10 + |a|/2 + 주행 명령 one-hot(left/straight/right).
- AutoVLA 프롬프트가 받는 ego 상태와 같다.
- CM보다 정보가 많으므로 별도 행으로 보고한다. 원고에서는 secondary다(supplement Sec. D).

**사용하지 않는 것**: 이미지, 지도, 다른 에이전트, 경로, 평가 시점의 GT 미래, 모델 출력, 평가 suffix, 그리고 개발/평가 장면 전부.

## 2. 출력
- 한 번의 forward로 suffix 포즈 j+1..9를 동시에 예측한다(비자기회귀, 출력이 입력으로 되먹임되지 않음).
- 출력은 9 슬롯 × (x/10, y/10, heading)이고 포즈 j 좌표계 기준이다. 9−j를 넘는 슬롯은 마스크한다.
- 예측 포즈는 decoder 좌표계로 되돌린다(`assemble`).

## 3. 구조와 손실 (`p2_extrap_train.py`)
- **MLP**: in → H → H → H → 27. GELU 활성화, dropout과 정규화층은 없다.
- 입력은 학습 분할의 mean/std로 표준화한다(std 하한 1e-3). 통계는 모델 버퍼에 저장된다.
- **손실**: 유효 suffix 슬롯에 대한 마스크 MSE. (x/10, y/10, heading) 제곱오차를 합한 뒤 마스크 합으로 나눈다.
- **최적화**:
  - AdamW, weight decay 1e-4, batch 2048
  - 15 epoch 고정, cosine decay to 0, warm-up 없음
  - early stopping 없음 → 마지막 epoch 가중치 사용
- CPU 전용(torch threads 8, `nice -n 19`).

## 4. 하이퍼파라미터·체크포인트 선택
- 그리드는 H ∈ {256, 512} × peak lr ∈ {1e-3, 3e-4}다. 주 입력, seed 0으로 4회 학습했다.
- 기준: 검증 드라이브의 마지막 epoch 평균 위치오차(m). 평가(P1-A) 데이터는 쓰지 않는다.
- 결과(`extrap_hparam_selection.csv`):

  | H | lr | val 평균 위치오차 (m) | 선택 |
  |---|---|---|---|
  | 512 | 1e-3 | 0.7241 | 선택 |
  | 256 | 1e-3 | 0.7344 | |
  | 512 | 3e-4 | 0.7491 | |
  | 256 | 3e-4 | 0.7658 | |

- 선택된 (H=512, lr=1e-3)을 두 변형과 모든 seed에 쓴다(`_extrap_models/selected.json`).
- 체크포인트는 항상 15 epoch의 최종 가중치다. 검증 손실로 epoch를 고르지 않는다.

## 5. Seed
- 최종 모델은 변형마다 seed {0, 1, 2}로 학습했다. seed는 초기화와 batch 순서를 정한다.
- 주 추정치는 단위별 Δ_E를 3 seed 평균한 뒤 Δ_model − mean_seed(Δ_E)로 계산한다.
- 최종 검증 위치오차(`_extrap_models/final_*_log.csv`, epoch 15):
  - prefix: s0 0.7241, s1 0.7247, s2 0.7254 m
  - prefix_ego: s0 0.5973, s1 0.5959, s2 0.5969 m
- 모델 sha256은 `extrap_run_meta.json`에 있다. 예: final_prefix_s0 `6b4d830c…`.

## 6. 학습·검증 데이터와 드라이브 분할
- **원천**: OpenScene v1.1 trainval 메타데이터. 이것은 AutoVLA 체크포인트의 학습 분할이다.
  - 공개 tarball(dataset commit a76f840b, 7.05 GB)을 디스크에 쓰지 않고 스트리밍했다.
  - 1,310 로그, 723,019 프레임에서 ego 필드만 남겼다.
  - 스트림 sha256 `193138ac…`, 122.7 s (`_extrap_data/trainval_ego_frames_provenance.json`)
- **드라이브 분리** (`extrap_drive_separation.json`):
  - 적격 trainval 드라이브는 171개다.
  - OpenScene test 분할(개발 20, P1-C 26, P5 후보 46, P1-A 평가 19 드라이브)과 공유하는 드라이브는 0개이고, 제외된 드라이브도 0개다.
  - 검증 드라이브: `rng(20261014).permutation`을 정렬된 드라이브 id에 적용하고 앞쪽 ceil(0.15n)개를 뽑았다 → **26개**. 나머지 **145개**가 학습 드라이브다. train/val 드라이브 중복은 0이다.
  - 기술적으로 남는 공유: 날짜 1개(2021.10.06)와 차량 10대(veh-14/26/28/35/38/39/40/42/45/52)가 P1-A 평가와 겹친다. 요구 조건은 드라이브 분리뿐이며, 이 공유는 보고만 한다.
- **샘플**:
  - 짝수 프레임(stride 2) 중 이후 10프레임이 같은 로그에 있고 각 Δt가 [0.45, 0.55] s인 계획 프레임만 쓴다.
  - 후보 355,955개 중 3,233개를 Δt/gap 때문에 제외했다. 0.5 s당 15 m를 넘는 스텝으로 제외된 것은 0개다.
  - 남은 계획 프레임은 **352,722**개다: 학습 296,080 / 검증 56,642.
  - GT 미래는 AutoVLA GT→token 투영(numpy 이식)으로 토큰화한 뒤 다시 디코딩해 입력과 타깃으로 쓴다. 따라서 평가 prefix와 같은 양자화 표현이다. 양자화 잔차는 중앙값 0.023 m, p99 0.194 m다.
  - 계획 프레임마다 j = 0..8의 9행을 만든다. 학습 **2,664,720행**, 검증 **509,778행**이다(`.pt`의 `n_train_rows`/`n_val_rows`).

## 7. 평가와 원고 수치
- **평가 대상**: P1-A 단위(family G, E, GF1, GF3; 5,553 단위). 지표는 같은 φ(M1 = 5 s endpoint 변위 norm)다.
- **방법**: 장면 가중, drive-cluster bootstrap(B = 10,000), 정확 drive sign-flip.
- **분포 밖(OOD) 점검**: 학습 행에 대한 cKDTree 최근접 거리로 판정한다(`extrap_coverage.csv`). 범위 밖 단위를 뺀 민감도 분석도 보고한다.
- **원고 수치** (family G, M1, S0 설정, `review_followup5/B_NUMERIC/B_NUMERIC_RESULTS.csv`; `vla0_work/pkg/paper/motion_excess_intervals.csv`):
  - F (A−, 364 단위 / 52 장면 / 13 드라이브): 초과 +1.97 m, 95% drive bootstrap [1.10, 2.83], raw p = 0.0044
  - 선택 합집합 (1,453 / 208 / 19): −0.34 m [−0.78, −0.01], p = 0.074
- **seed 민감도** (`LEARNED_EXTRAPOLATOR_RESULTS.csv`, A−):
  - seed별 1.961 / 1.946 / 1.989 m, seed SD 0.021
  - 범위 밖 단위를 빼면(332 단위) 1.906 [0.970, 2.841]
- **prefix+ego 변형** (A−): 2.27 [1.50, 3.10]
- **비용** (참고): 최종 학습 1회에 14.4–16.0분(CPU), 평가 5.3분(`extrap_run_meta.json`, elapsed_s 320.4)

## 8. 해석상 주의 (프로토콜 §8, supplement Sec. D)
- 이 대조는 같은 주입 운동 변화에 대한 반응을 비교한다. VLA 구조의 자기회귀 효과를 분리하지 않으며, 사전학습 데이터나 용량도 맞추지 않는다.
- 학습 데이터(trainval, 145 드라이브)는 AutoVLA의 학습 분할이다. 평가 장면과 드라이브는 분리되지만 날짜와 차량 일부는 공유한다.
- 원고(supplement Sec. D)의 "145 drives with 26 validation drives, averaged across three seeds"는 위 파일과 일치한다.
