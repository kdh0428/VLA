# 실험 33: 시간축 action-chunk VLA에서의 previous-action feedback 재현 — 사전 등록

작성: 2026-10-04, 본 실행 전(형식 확인용 smoke test 6–8 장면만 실행). 결과를 보고 정의·threshold·판정 기준을 바꾸지 않습니다.
기존 AutoVLA/navhard 결과·문서·script는 수정하지 않고, 새 결과는 이 디렉토리에만 저장합니다. GPU는 RTX 5090만 사용합니다.

## 0. 모델 선정 (코드 수준 확인)

| 후보 | (1) 여러 token AR 생성 | (2) 앞 token이 뒤 token을 conditioning | (3) chunk 안 시간 순서 | (4) 공개 checkpoint·code | 판정 |
|---|---|---|---|---|---|
| π0-FAST (`lerobot/pi0fast-libero`) | O | O | **X** — FAST는 chunk를 차원별 DCT 후 양자화하고, 저주파부터 flatten한 계수열을 BPE로 압축. token이 시간 step이 아니라 주파수 계수에 대응 | O | 제외 |
| SpatialVLA (`IPEC-COMMUNITY/spatialvla-4b-224-*`) | O (`generate`) | O | O (chunk 4 step × 3 token, `decode_actions`에서 시간순 reshape) | 부분 — LIBERO checkpoint 비공개, 공개된 bridge/fractal용 SimplerEnv는 Vulkan 필요(이 컨테이너에 NVIDIA Vulkan/EGL 없음) | 예비 후보 |
| OpenVLA-7B (실험 28) | O | O (step 안 차원 간) | **X** — 한 step의 7차원만 생성, 다음 step에 이전 action 없음 | O | 이미 수행 |
| **Impromptu VLA 3B** (`aaaaaap/ImpromptuVLAModel/3B_AD`, "3B Base+Impromptu") | O — Qwen2.5-VL causal LM `generate` | O — causal attention, 앞 waypoint 텍스트가 뒤 waypoint를 조건화 | **O** — 미래 10개 waypoint(0.5 s 간격, 5 s)를 시간순 텍스트 `[x, y], [x, y], ...`로 생성 | O — 공개 weight(bf16 7.1 GB), 공개 data-engine code(프롬프트·답 형식) | **채택** |

채택 이유: 조건 4개를 모두 만족하고, AutoVLA와 **같은 NAVSIM navtest 장면·같은 GT·같은 실패 정의(P/R/A A−, 증폭 = A− and FDE5 > 3 m)·같은 5 s horizon**으로 비교할 수 있습니다.

### Impromptu 구조와 AutoVLA와의 차이
- 입력: 현재 frame의 CAM_F0 / CAM_R0 / CAM_L0(장면 3,000여 장의 L0/R0만 navtest shard 0–5에서 선택적으로 다시 받음), 과거 1.5 s의 4개 상태
  (현재 ego 좌표의 위치, 가속도, 속도). 공개 data-engine(`loaders/pipelines`: metadata "3v" + ego_status "x-y" + planning "x-y")의
  프롬프트·답 형식, LLaMA-Factory `qwen2_vl` template, 이미지 최대 262,144 pixel(공개 train/inference 설정)을 그대로 재현.
- 출력: `<PLANNING>Predicted future movement ...: [x1, y1], ..., [x10, y10]</PLANNING>` — 각 waypoint는 **현재 ego 기준 절대 위치**이고
  숫자 하나가 여러 digit token입니다. AutoVLA는 0.5 s마다 **상대 motion** 하나를 codebook token 1개로 냅니다.
- 디코딩: greedy(checkpoint generation_config의 top_k = 1), repetition_penalty 1.05(generation_config 값).

## 1. 데이터와 지표

- 장면: navtest PoC 28 log의 2,748 장면(AutoVLA 기전 실험과 같은 log). log 단위 cluster.
- 궤적 = waypoint 10개(5 s). GT = navsim log의 미래 10 pose(현재 ego 좌표).
- 지표: FDE5, ADE5, **A−**(P/R/A accept 실패, `analyze_action_history.a_eval`), **증폭 = A− and FDE5 > 3 m**,
  **하위 발산** D_k = |w_k(조건) − w_k(같은 장면의 자연 rollout)|, 최종 발산비 r = D_10 / |δ| (주입 편차 대비).
- 기저 변동: 같은 장면을 batch 구성만 바꿔 다시 디코딩한 `natural_rep`와 `natural`의 A−·증폭 불일치율과 D_10.

## 2. Experiment A·B — 작은 초기 편차와 matched-magnitude 분기 (전 2,748 장면)

- 첫 waypoint(0.5 s)를 자연 rollout 값 + δ로 강제하고 나머지 9개는 자유 생성. |δ| ∈ {0.2, 0.5} m, 방향 8개(0°, 45°, …, 315°; 0° = 전방).
  행: natural(단독 batch), natural_rep(교란 행과 같은 batch), **pert_m0**(첫 waypoint를 자연 값 그대로 강제, δ = 0), 교란 16개.
  하위 발산 D_k와 r의 기준은 pert_m0입니다(교란 행과 같은 "텍스트 prefix를 다시 encoding해 이어 쓰기" 경로). smoke test에서
  prefill 한 번과 KV cache 증분 계산 사이의 bf16 차이만으로도 다음 waypoint가 수 cm 달라질 수 있음을 확인했으므로,
  natural vs pert_m0, natural vs natural_rep 차이를 모두 기저 변동으로 보고합니다.
- A: 자연이 A+인 장면에서 교란 후 A−로 바뀐 비율(작은 편차가 항상 실패로 가는가), 자연 재실행 불일치율과 비교.
- B: 같은 장면·같은 |δ|에서 **안정 분기**(A+ 그리고 r ≤ 1)와 **불안정 분기**(증폭 또는 r ≥ 3)가 공존하는 장면 비율;
  log r의 분산 중 장면 간 / 장면 내(방향) 비율(ICC), |δ|와 FDE의 관계, 방향별 평균.

## 3. Experiment C·D·E — conditioning-only 개입 (무작위 300 장면 × 방향 8개, |δ| = 0.5 m, seed 0; δ = 0 기준 단위 1개 추가)

단위 = (장면, 방향). t* = 첫 waypoint(강제한 편차). **실행 궤적은 항상 강제 w̃_1 + 그 행이 생성한 waypoint**이고,
바꾸는 것은 다음 waypoint를 생성할 때의 **문맥**뿐입니다(AutoVLA 실험 7–11과 같은 정의). waypoint를 하나씩 생성합니다.

| 행 | 문맥의 waypoint j (2 ≤ j < k, 강제 w̃_1은 항상 유지) |
|---|---|
| normal | 자기 생성 w_j |
| win1 / win2 / win3 / win4 | j ∈ [2, 1 + w]는 GT g_j, 나머지는 자기 생성 |
| gt_history (full) | 모든 j: GT |
| recent_gt | 마지막(j = k−1)만 GT, 나머지 자기 생성 |
| **reverse** | gt_history 문맥에 마지막(j = k−1)만 그 행의 자기 생성(오차 포함) waypoint |
| near_gt | 마지막 = GT + 0.05 m(무작위 방향) — motion이 GT와 가까운 다른 값 |
| dir_ok_mag_wrong | 마지막 = 직전 문맥 c_{k−2} + GT step motion 방향, 크기 오차 = e (e = 자기 waypoint의 GT 오차) |
| dir_wrong_mag_ok | 마지막 = c_{k−2} + GT step motion 크기, GT와의 거리 = e (방향 오차) |
| random_mag_matched | 마지막 = c_{k−2} + GT step 크기 ±5%의 무작위 방향 motion |

(motion = 문맥에서 직전 두 waypoint의 차이, GT step motion = g_{k−1} − c_{k−2}. 대체는 j = k−1 ≥ 2일 때부터, 즉 waypoint 3부터 영향을 줌.)

## 4. 통계

- log 단위 cluster bootstrap 95% CI(2,000, seed 0); 이진 지표 paired McNemar, 연속 지표 paired Wilcoxon(단위 짝).
- 모든 개입 효과는 natural_rep 대비 기저 변동과 함께 보고합니다. greedy 디코딩이라 seed 변동은 없고, 수치 변동은 batch 구성으로 측정합니다.

## 5. 판정 기준 (미리 고정)

1. **correction**: gt_history(그리고 recent_gt)가 normal보다 증폭률(또는 A−)과 최종 발산 D_10을 유의하게 줄임(p < 0.05, CI가 0 제외).
2. **reverse**: reverse가 gt_history보다 D_10(그리고 증폭률)을 유의하게 늘림.
3. **branch**: matched-magnitude에서 안정·불안정 분기가 공존하는 장면 비율이 자연 재실행 불일치율보다 큼.
4. **window**: win1..win4의 효과가 단조 증가하고 win4 이내에 gt_history 효과의 80% 이상.
5. **direction**: dir_wrong_mag_ok − dir_ok_mag_wrong의 증폭률 또는 D_10 차이 > 0(p < 0.05).

최종 판정:
- **Strong replication**: 1과 2가 성립하고, task 수준(A−/증폭)에서도 교란이 자연 재실행 변동보다 큰 실패를 만들고 correction이 그 실패를 유의하게 줄임.
- **Partial replication**: 1과 2가 waypoint/궤적 수준(D_10)에서 성립하지만 task 수준(A−/증폭) 효과가 불확실.
- **No replication**: 1 또는 2가 성립하지 않음.

## 6. 실행 세부 (smoke test 후, 본 실행 전 추가)

- 답 형식: smoke test에서 `q7_navsim.py` 형식은 학습 형식과 달라(정확도 저하) 공개 data-engine `loaders/pipelines` 형식으로 맞췄습니다
  (현재 프레임 포함 4개 상태, python `round()` 표기, 답 `<PLANNING>…: [x, y], …</PLANNING>`). 텍스트 prefix는 학습 토큰화(`":" + " ["`)와
  같게 이어 붙입니다.
- C·D·E는 waypoint 하나씩 생성하므로, 장면마다 프롬프트 KV cache를 한 번 계산하고 행별 suffix만 넣는 greedy decoder
  (`FastWaypointDecoder`, repetition penalty 1.05를 transformers와 같은 방식으로 적용)를 씁니다. HF `generate`와 마지막 자리(0.01–0.05 m)
  수준의 bf16 차이만 있고, 모든 행·δ = 0 기준 단위가 같은 경로를 쓰므로 비교는 같은 조건입니다.
- C·D·E의 하위 발산은 두 가지로 보고합니다: (주) GT 기준 지표(증폭, A−, FDE5)와 (보조) **같은 행의 δ = 0 기준 단위 대비** 최종 발산
  D_10^row = |w_10(행, δ) − w_10(행, δ = 0)| — 문맥 개입 아래에서 0.5 m 초기 편차가 얼마나 키워지는지.
- A·B의 vision encoder는 장면의 이미지 3장을 한 번만 encoding하고 행마다 복제합니다(이미지별 attention이라 결과 동일, 메모리 절약).
