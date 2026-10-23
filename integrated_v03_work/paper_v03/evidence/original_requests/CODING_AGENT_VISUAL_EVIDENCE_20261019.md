# 코딩 에이전트 추가 요청 — CVPR 원고의 그림·사례·시점별 분석 자료

## 0. 목적과 기존 요청과의 관계

이 요청은 이미 전달한 통계·독립 검증 질문을 **보완하는 추가 작업**이다. 기존 질문의 답변을 대체하거나, 완료된 검증 전체를 반복하지 않는다. 기존 작업과 함께 반환해도 좋지만, 아래 자료 추출은 기존 통계 답변을 기다리지 않고 진행할 수 있다.

현재 논문은 AutoVLA를 주모델로 action-context 개입의 조건부 효과를 분석하고, SpatialVLA로 생성 행동·실제 움직임·성공률의 관계를 보완한다. 원고는 **본문 8쪽, 참고문헌 별도**로 편집 중이다. 설명과 표의 일부를 실제 장면 및 분석 그림으로 바꾸려 한다. 원고 작성자가 서버 접속 없이 그림을 다시 그리고 해석을 검증할 수 있는 **원본 사례 자료 + 수치 데이터 + 재현 코드**를 제공하라.

우선순위는 **V1 실제 주행 사례 → V2 시간별 반응 분석 → V3 효과 분포·추정 대상 비교 → V4 로봇 사례**다. 가능한 항목부터 완성하고, 자료가 없는 항목은 정확한 이유와 함께 반환하라.

## 1. 범위와 기본 원칙

- 기존 저장 자료의 CPU 분석, 디코딩·좌표 복원, 이미지 추출, 그림 제작을 수행한다. **새 GPU 추론·학습, 새 모델·과제 추가는 실행하지 않는다.**
- 기존 모델 출력으로 복원할 수 없는 값은 결측으로 남긴다. 필요하다면 별도의 후속 실행 계획·자원 추정만 작성한다.
- 입력 raw/protocol과 기존 결과는 덮어쓰지 않는다. 새 산출물은 `outputs/cvpr_visual_evidence_20261019/`에 저장한다.
- 탐색 시작점은 현재 작업 저장소의 `review_followup5`, `MANUSCRIPT_AUTHOR_ANSWERS_20261019`, P1-C/B2 및 G-family의 기존 결과다. 경로가 다르면 실제 위치를 기록한다. 이미 확인한 전체 입력 해시·통계 검증을 반복하지 말고 이번에 사용하는 입력만 추적한다.
- 모든 신규 재분석은 **사후 분석**으로 표시한다. 기존 자료의 재분석을 새 독립 재현이나 새 실험으로 부르지 않는다.
- 최종 원고의 framing·수치 정책·8쪽 편집은 원고 작성자가 담당한다. 이 작업에서 원고 전체를 다시 쓰지 않는다.

반드시 보존할 해석:

1. AutoVLA는 **고정 관측에서 생성한 open-loop 계획**이다. 예측 궤적을 실제 차량의 closed-loop 주행 결과로 표시하지 않는다. context 편집값을 이미 emitted된 행동으로 바꿔 그리지 않는다.
2. 실패선택 F와 가중 eligible frame E/E4는 다른 추정 대상이다. E와 E4도 eligibility가 다르다. 전체 선택 장면을 자연 모집단으로 부르지 않는다.
3. GT-free는 perturbation 값·reference에 관한 표현이다. anchor와 strata의 GT 의존성을 숨기지 않는다.
4. GT 세 위치를 이후 문맥에 유지한 R9 결과는 **GT 해제 후 회복 지속**이나 **90% task success**가 아니다. repeated Recent-GT와 retained-window 실험도 구분한다.
5. 신뢰구간은 최신 통계 답변의 사용 범위를 따른다. 답변이 미완료라면 현재 보류된 E/E4 구간은 그리지 않는다. 기존 근사 구간을 검증 완료 구간으로 승격하지 않는다.
6. SpatialVLA의 두 과제 평균 ±10 pp 결과를 과제별 ±10 pp 또는 ±5 pp로 확대하지 않는다. drive/day/vehicle 및 physical-state 의존성도 유지한다.

## 2. 먼저 반환할 자료 가용성 조사

`DATA_AVAILABILITY.csv`를 작성한다. 아래 각 항목에 `AVAILABLE / DERIVABLE_CPU / MISSING / UNVERIFIABLE`과 실제 경로·행 수·복원 방법을 적는다.

- scene/unit/perturbation/drive/log/day/vehicle 식별자, cohort/stratum, eligibility, scene 표집 가중치.
- 원본 카메라 프레임, camera ID, timestamp, 필요한 camera calibration.
- 모델 입력에 사용한 관측, 자연 생성 토큰, 강제 토큰, GT 토큰, 개입별 emitted 토큰과 context 편집 기록.
- 개입별 시간 순서의 예측 pose/action, GT trajectory, motion baseline의 unperturbed/perturbed trajectory.
- SpatialVLA의 초기 물리 상태 ID·state hash, task/instruction, seed, config/arm, 관측 프레임, 생성 행동, 실제 TCP, 실행 행동, 성공 여부.

원본 모델 출력에서 CPU로 재구성한 자료와 실험 중 직접 저장한 자료를 구별한다. 원본 파일이 없고 집계 수치만 있는 경우를 명시한다. 자료가 없다는 이유로 다른 실험의 프레임이나 궤적을 조합하지 않는다.

## V1. 실제 주행 장면과 조건별 예측 궤적

### V1-a. 사용할 대비와 사례 선정

- 가능하면 P1-C B2의 **NAT_REF 대 NAT_NSRC**를 사용한다. 같은 emitted prefix 뒤에서 latest context의 GT/Normal-source 차이를 보여주는 사례다.
- 이 대비의 원본 frame 또는 trace가 없으면 P1-C repeated RN(Recent-GT 대 Normal)을 대안으로 사용한다. fallback은 효과 크기가 아닌 **자료 가용성**으로 결정하고 기록한다.
- Reverse-N을 보여주면 반드시 해당 Full-GT reference와 짝지어 별도 표시한다. 서로 다른 schedule의 곡선을 하나의 matched-prefix 비교로 묶지 않는다.
- 실제 cohort 및 대비를 선택한 뒤, 사례 이미지를 고르기 전에 `CASE_SELECTION_RULE.md`에 선정 규칙을 저장한다. 사후 자료를 사용하는 결정적 선정 규칙이며 사전등록이라고 부르지 않는다.

사례는 개선·변화가 작은 경우·악화를 모두 포함한다. 다음 기본 규칙을 사용하되 구현 불가능하면 이유와 대체 규칙을 먼저 적는다.

1. 원래 정의된 F cohort 안에서 조건 쌍이 완성된 모든 unit을 후보로 삼는다. 시각 자료 가용 여부에 따른 제외 수를 따로 보고한다.
2. 각 unit의 `FDE5(treatment) - FDE5(reference)`를 만들고, scene 안의 원래 적격 perturbation들을 평균한다. treatment는 NAT_REF 또는 Recent-GT처럼 GT 교정을 포함하는 arm으로 정의하고 명시한다.
3. scene의 severe-event 차이가 음수·0·양수인 그룹을 우선 구분한다. 중간 그룹은 **severe-event 판정이 바뀌지 않았다는 뜻**이며, 연속 FDE나 궤적이 같다는 뜻은 아니다.
4. 각 그룹에서 FDE 차이의 중앙값에 가장 가까운 scene을 최대 2개 고른다. drive 중복을 줄이는 결정적 규칙과 stable ID tie-break를 적용하고, 장면 안 unit도 동일한 방식으로 선정한다.
5. 총 최대 6개를 제공하고, 본문용으로 그룹별 1개를 제안한다. 어느 그룹이 비어 있으면 그대로 보고한다. 그림이 좋아 보이도록 다른 cohort에서 보충하지 않는다.

그룹별 후보 수·비율과 최종 사례의 FDE 차이 백분위수를 제공한다. 사례는 설명용으로 선택된 것이며 모집단 대표 표본이나 독립 검정 자료가 아님을 표시한다.

### V1-b. 사례별 제공물

- 실제 입력 카메라 frame PNG/JPG. 같은 관측을 쓴 조건들은 하나의 공통 입력 frame으로 표시한다.
- GT와 각 조건의 예측 궤적을 동일 좌표·축척으로 보여주는 BEV PDF/SVG와 고해상도 PNG.
- 원점·좌표계·단위·시간 간격, 개입 anchor, 공유 emitted prefix, 이후 생성 suffix를 표시한다. 편집된 context와 emitted trajectory를 구별한다.
- 카메라 투영은 올바른 calibration이 있을 때만 수행한다. 없으면 원본 frame과 BEV를 나란히 둔다. 차선·장애물·미래 frame을 추정해서 실제 기록처럼 추가하지 않는다.
- token ID와 개입값, condition별 pose 배열, GT 배열, FDE5/severe-event 값, cohort/IDs/source hash가 포함된 사례 JSON/NPZ 및 CSV.
- 영문 caption 초안: 무엇이 고정됐고 무엇이 편집됐는지, 예측 계획이라는 점, 사례 선정 규칙을 짧게 포함한다.

사진은 실험 원본을 사용한다. 생성형 이미지나 다른 장면의 사진으로 실험 결과를 대체하지 않는다. 원본 frame이 없으면 궤적만 제공하고 해당 제약을 표시한다.

## V2. 개입 이후 시간별 반응과 운동 baseline 비교

### V2-a. raw 시계열 추출

G-family에서 같은 unit과 계산 설정의 unperturbed/perturbed 계획 쌍을 추출한다. 기본 설정 S0를 우선 제공하고, 이미 저장된 SE/SF trace가 있으면 같은 코드로 추출한다. 이 목적의 재추론은 하지 않는다.

`AUTOVLA_TIME_SERIES.csv` 또는 동등한 압축 표를 반환한다. 최소 열:

`cohort, stratum, scene_id, unit_id, perturbation_id, drive_id, day_id, vehicle_id, component_id, scene_weight, setting, arm, baseline, absolute_step, anchor_step, step_after_anchor, horizon_s, x_m, y_m, yaw_rad, token_id, gt_x_m, gt_y_m, source_ref`

실제 저장되지 않은 열은 빈 값과 결측 사유를 둔다. generation position과 environment timestep을 혼용하지 않는다. anchor가 서로 다른 실험은 따로 분석하며, 실제 시간 간격이 확인될 때만 초 단위로 표시한다.

### V2-b. 핵심 분석·그림

각 이용 가능한 horizon h에서 다음을 계산한다.

```text
D_model(h) = ||P_model(h) - U_model(h)||_2
D_b(h)     = ||P_b(h) - U_b(h)||_2
E_b(h)     = D_model(h) - D_b(h)
```

- P/U는 같은 개입 정의와 source를 사용한 perturbed/unperturbed 쌍이다. baseline도 두 입력에 대한 반응끼리 비교한다. baseline의 절대 GT 오차를 D_model에서 빼지 않는다.
- 좌표 누적과 constant-motion baseline을 우선 포함하고, 기존 learned extrapolator의 시계열 출력이 저장됐거나 동일 모델의 CPU 평가로 복원 가능하면 추가한다. 실제 전체 h를 제공하지 못하는 baseline은 종점 결과만 남긴다.
- `D_model(h)`, `D_b(h)`와 `E_b(h)`를 패널로 나누어 F와 전체 selected mechanism cohort를 함께 보여준다. 뒤쪽 cohort를 자연 모집단으로 표기하지 않는다.
- 기존 GT 오차를 계산할 수 있으면 `||P(h)-GT(h)||`와 `||U(h)-GT(h)||`를 별도 패널/표로 제공한다. **반응 크기와 정확도 변화는 별도 지표**다.
- unit→scene 평균→명시한 cohort 가중 집계 순서를 지킨다. horizon별 유효 unit/scene/drive 수를 제공한다. anchor 차이·조기 종료로 표본이 바뀌면 공통 horizon/공통 sample 분석과 관측 가능한 전체 sample 분석을 구분한다.
- 실제 기록이 없는 시간점을 보간해 관측치처럼 만들거나, 결측·조기 종료 뒤를 0으로 채우지 않는다.
- 새 시간별 곡선은 기본적으로 **기술적 사후 분석**이다. 유의한 h를 찾아 primary 결과로 바꾸지 않는다. 구간을 제공하면 추정 대상·군집축·방법을 적고, pointwise 구간을 simultaneous confidence band로 부르지 않는다. 타당한 구간 근거가 없으면 점추정 곡선과 표본 수만 제공해도 된다.
- 현행 severe-event는 5초의 정의를 유지한다. 3 m 기준을 모든 짧은 horizon에 그대로 적용해 새로운 실패율 곡선을 만들지 않는다.

최종 5초 값은 동일 cohort·동일 가중치의 기존 결과와 대조한다. 차이가 나면 반올림 전 수치, 원인, 영향받는 claim을 보고한다. 시계열이 없으면 종점에서 중간 곡선을 역산하지 말고 `BLOCKED_TIME_TRACE_MISSING`으로 반환한다.

### V2-c. 교정 schedule 시각화

repeated Recent-GT와 고정된 세 위치를 이후 문맥에 유지한 R9의 차이를 한눈에 보여주는 작은 시간축 도식을 제공한다. 기존 로그에 실제 window별 데이터가 있으면 window 폭/노출량에 따른 결과도 반환하되, eligibility·교정 횟수·조건별 표본 수가 같지 않으면 명시한다.

Normal·win3·Full-GT의 세 집계 값만 있을 때는 이산 점/표로 표시한다. 없는 window의 값을 연결해 효율 곡선을 만들거나, GT 교정 해제 이후의 지속 효과를 추정하지 않는다.

## V3. 효과 분포와 추정 대상의 차이를 보여주는 수치

아래 데이터를 반환하고, 가능하면 본문용 1개 composite figure를 제안한다.

1. repeated RN/RV: development failure cohort, P1-C F, P1-C weighted E의 효과와 표본 수.
2. matched-prefix C_GT/C_NAT/INT: F와 weighted E4의 효과. C_PROV의 원고 표기는 C_src이며 값·source가 함께 바뀌는 대비라는 설명을 유지한다.
3. 가능하면 scene별 효과와 drive별 요약. drive 내부에서는 scene 가중치를 보존하고, 전체 효과를 drive 단순평균으로 바꾸지 않는다. 그래프의 drive 점들은 이질성 설명용이며 shared day/vehicle에서 독립 표본으로 취급하지 않는다.

`EFFECTS_FOR_PLOTTING.csv`에는 다음을 포함한다.

`claim_id, contrast, cohort, estimand, outcome, unit, sign_definition, estimate, ci_low, ci_high, ci_status, ci_method, cluster_axis, p_raw, p_adjusted, family, alpha, n_units, n_scenes, n_drives, n_components, weighting, source_ref, source_hash`

가중 E/E4의 구간은 앞서 요청한 통계 답변에서 허용된 경우에만 추가한다. 그렇지 않으면 점추정만 그리고 구간 보류를 표시한다. day/vehicle component 민감도에서 약해지는 결론도 그림 설명에 유지한다. 현재 집계 수치의 재시각화와 새 unit-level 분석을 분명히 구분한다.

## V4. SpatialVLA: 실제 장면·TCP·성공률을 연결

- P1-D의 같은 task/config/초기 물리 상태로 짝지어진 N/F/C/R을 우선 사용한다. 최신 정의와 arm mapping을 README에 적는다.
- 과제별 원본 시작 frame, 가능하면 실제 기록의 중간/끝 frame을 제공한다. 조건별 frame은 시뮬레이션 시간 또는 실제 행동 step으로 정렬하고 정책 호출 번호만으로 정렬하지 않는다.
- 생성 action, executor에 보낸 action, 실제 TCP pose를 별도 필드로 내보낸다. translation은 m, rotation은 명시한 rad/quaternion 규약을 사용하고 다른 단위를 하나의 norm으로 합치지 않는다.
- R−N과 C−F의 paired TCP 차이 및 episode success를 추출한다. trajectory 변화와 success의 의미가 다르다는 점을 같은 composite panel에서 읽을 수 있게 한다. 전체/과제별 성공률은 선택한 정성 사례가 아니라 원래 전체 적격 표본으로 계산한다.
- 물리 초기 상태 ID/hash, seed, instruction, task, config, arm, success, episode 길이, 실제 TCP 가용 여부를 `SPATIAL_EPISODES.csv`에 보존한다. 각 timestep의 자료는 `SPATIAL_TIME_SERIES.csv` 또는 NPZ/Parquet로 제공한다.
- move_near의 동일한 10개 물리 배치가 instruction/seed 때문에 독립 상태로 늘어나지 않게 한다. coke의 물리 상태 정의는 기존 audit에 맞춘다.
- state restoration이 검증되지 않은 legacy 실험은 그 제약을 표시한다. 실제 TCP가 없는 실험 34에 commanded action을 TCP처럼 대입하지 않는다. 가능한 P1-D 사례부터 완성한다.
- 로봇 사례도 개선·유사 성공 결과·악화 범위를 가능한 한 포함하고 선정 규칙 및 후보 수를 남긴다. RGB가 없으면 TCP·success 분석만 제공한다.

## 3. 그림 파일과 재현성

- 정량 그림은 편집 가능한 PDF 또는 SVG와 PNG를 함께 제공한다. 사진은 원본 PNG/JPG를 보존한다. 단일 column 약 3.25 inch, 양 column 약 6.9 inch에서 라벨을 읽을 수 있게 제작한다.
- 그래프에는 표본 수, 단위, cohort, reference와 개입 방향, 구간 종류를 명확히 적는다. 다른 metric을 같은 축에 겹치지 않는다.
- 재현 스크립트와 plot input CSV/NPZ를 제공한다. 그림 파일만 반환하지 않는다. 색·crop·축 범위를 조건 사이에 일관되게 유지한다.
- `SOURCE_MANIFEST.csv`에 사용한 raw의 상대경로·hash·생성/재구성 여부·원본 commit 또는 실행 ID를 적는다. 서버 절대경로만 있는 링크를 전달용 자료로 삼지 않는다.
- 최소 검증: paired ID와 emitted-prefix 일치, 좌표계/단위, raw→scene→전체 집계, 종점 수치 재현, 최종 사례와 전체 후보 manifest의 연결. 테스트 개수 채우기나 완료된 통계 audit 반복은 하지 않는다.
- 확인할 수 없는 state/source/time 정합성은 `UNVERIFIABLE`로 남긴다. 새로운 오류를 찾으면 변경 전후 수치와 영향 범위를 별도 보고한다.

## 4. 반환할 패키지와 종료 조건

`CVPR_VISUAL_EVIDENCE_20261019.zip`으로 묶어 반환하라. 다음 파일/폴더를 포함한다.

- `START_HERE.md`: V1–V4별 `READY / PARTIAL / BLOCKED`, 핵심 발견, 본문 후보 그림, 제한, 실제 CPU/GPU 사용량.
- `DATA_AVAILABILITY.csv`, `MISSING.md`: 없는 자료의 정확한 이름·필요 열·가능한 대체와 후속 비용.
- `ANALYSIS_PLAN.md`, `CASE_SELECTION_RULE.md`: 수행 전 고정한 이번 사후 분석·선정 규칙과 이후 변경 내역.
- `CASE_CANDIDATES.csv`, `SELECTED_CASES.csv`: 전체 적격 후보, 제외 사유, 그룹/순위/선정 여부, ID와 metrics.
- `AUTOVLA_TIME_SERIES.*`, `TIME_SERIES_SUMMARY.csv`, `EFFECTS_FOR_PLOTTING.csv`, 이용 가능한 scene/drive별 효과표.
- `SPATIAL_EPISODES.csv`, `SPATIAL_TIME_SERIES.*`: 가용 범위만 포함하고 없는 항목은 상태를 기록.
- `cases/`: 선택된 사례의 원본 frame, GT·조건별 궤적, token/context 기록, 메타데이터.
- `figures/`: 정량 PDF/SVG 및 PNG, 정성 composite. `FIGURE_CAPTIONS.md`에 영문 caption과 각 그림이 뒷받침하는 주장·해석 한계를 작성.
- `scripts/`, `RUN_COMMANDS.md`, `SOURCE_MANIFEST.csv`, `NUMERICAL_CHECKS.md`: CPU 재현 명령, 사용 입력과 수치 대조.

원본 전체 영상/대규모 데이터셋은 묶지 않는다. 선택 사례의 필요한 frame과 plot 재현용 수치는 패키지 자체에 포함한다. 압축 후에도 너무 크면 영상 제외·수치 압축으로 줄이고, 무엇을 제외했는지 적는다.

모든 항목을 억지로 READY로 만들 필요는 없다. **가용한 자료를 실제로 추출·분석하고, 막힌 항목의 정확한 이유를 반환하면 완료**다. 새 GPU 작업이 꼭 필요하다면 이 패키지와 분리해 데이터·목표·실행 시간 추정이 있는 후속 제안만 작성한다. commit/push는 실제 수행한 경우에만 hash와 상태를 보고한다.
