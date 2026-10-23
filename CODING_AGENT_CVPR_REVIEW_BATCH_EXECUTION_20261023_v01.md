# CODING AGENT: CVPR 리뷰 대응 통합 실행 프롬프트
작성일: 2026-10-23 KST
문서 버전: v01
기준 원고: 최신 서버 적용 v04 / CVPR_OVERLEAF_20261023_v04.zip
목표 산출물: 신규 GPU 실험 + 분석 + v05 검토용 원고·PDF + 통합 결과 ZIP

## 0. 요청 방식과 완료 범위

이번 요청은 계획 작성으로 끝내지 말고, 아래에 명시한 신규 GPU 실험과 분석을 실제로 수행하는 요청이다. 사용자는 필요한 작업을 한 번에 모아 맡기기를 원한다. 실험 사이의 의존성·validation gate·재개·결과 정리는 에이전트가 내부에서 처리한다. 정상적인 단계 전환마다 사용자에게 새 지시를 요구하지 않는다.

다만 신규 결과를 확인한 뒤에만 정할 수 있는 최종 과학적 해석은 마지막의 REVIEW_AFTER_RESULTS에 남긴다. 기존 결과를 유지할지, 어떤 새 결과를 초록의 대표 수치로 사용할지, equivalence·일반화·기여 크기를 어디까지 주장할지의 최종 결정은 이번 실행에서 확정하지 않는다.

이번에 끝낼 것:
- 신규 GPU 실험의 실행과 원시 기록 저장
- 사전에 고정한 분석과 불확실성·coverage 보고
- 기존 분석의 가시화, 명칭·표기·관련 연구·계산 비용 수정
- 새 결과의 중립적인 표·그림과 results 초안
- 본문 8쪽의 v05 검토용 PDF, 전체 보충자료, Overleaf ZIP
- 결과 확인 후 검토할 항목과 근거를 한 파일로 정리

실행 결과가 약하거나 반대여도 분석·기록·수치 검증·검토용 원고 패키지까지 완료한다. 한 모델·실험이 실행 불가능하더라도 독립적으로 가능한 작업을 계속한다. 실행하지 않은 작업을 완료했다고 표시하지 않는다.

## 1. 사용할 기준 자료

현재 서버의 기존 프로젝트와 실험 환경을 재사용한다. 원고 기준은 최신 v04 서버 적용본이다. 제공된 Overleaf ZIP과 서버 원고가 다르면 변경 목록을 기록하고, 최신 서버 적용본의 검증된 실험 코드·원시 자료를 보존한다.

가능한 자료:
- CVPR_OVERLEAF_20261023_v04.zip
- CVPR_V04_SERVER_APPLY_RESULT_20261023_v01.zip
- 기존 v04 재현 자료와 AutoVLA / VLA-0 / SpatialVLA 원시 기록
- 첨부 리뷰: 붙여넣은 텍스트 (1)(20261023-103344).txt
- CVPR_REVIEW_RESPONSE_AND_BATCH_PLAN_20261023_v01.md (배경 자료, 이 실행 프롬프트가 우선)

Overleaf 기준 ZIP의 확인된 SHA-256:
26a829bf77ee3a97c4704c3c070ab7c8ca7e75e8294b80caf301fc16234f97e0

기존 자료를 가진 같은 코딩 에이전트라면 이 프롬프트만 추가로 받아도 진행한다. 파일명만 보고 폴더 구조·누락 파일·실험 완료 여부를 추정하지 말고 실제 파일을 확인한다.

지정된 저장 경로·프로젝트 지침이 있으면 우선 준수한다. 신규 파일에는 날짜와 버전을 넣는다. v04와 원시 기록을 덮어쓰지 않고 v05 작업 디렉터리에서 진행한다.

## 2. 자원·표본·실행 순서

### 2.1 자원과 로그

기존 서버·세션에 설정된 GPU 시간·자원 상한이 있으면 우선 적용한다. 별도 상한이 없다면 아래의 유한한 표본 수와 donor search cap을 기본 실행 범위로 삼고, 임의의 추가 모델 학습이나 무한 donor 탐색으로 범위를 늘리지 않는다.

GPU 실험 시작 전에 GPU 이름·VRAM·사용 중인 작업·라이브러리 버전·코드 commit을 기록한다. 실제 inference는 eval/inference mode에서 수행한다. 짧은 smoke의 실측 비용으로 각 실험의 GPU 점유시간과 예상 완료 시간을 기록한 뒤 정해진 실행을 계속한다.

기존 상한 때문에 기본 표본을 완료할 수 없으면:
- 새 outcome을 보기 전에 축소한 고정 표본 수를 기록한다.
- 과제별 균형을 우선하고, 독립 state/layout 수를 줄이면서 seed 반복만 늘리지 않는다.
- 정해진 모든 단위의 완료를 목표로 하며, 유의한 결과가 나올 때 멈추거나 표본을 늘리지 않는다.
- 정밀도가 부족하면 결과를 기술적 추정으로 보고하고, 검정력 부족을 표시한다.
- 일부 분석만 가능해도 완료된 자료와 blocker를 최종 ZIP에 포함한다.

CPU 처리·렌더링 시간과 GPU 점유시간을 구분한다. 동시 GPU 프로세스 수는 VRAM 여유와 native 경로 재현성을 고려한다. 조건별 batching 차이가 결과와 뒤섞이지 않도록 조건 순서와 batch 규칙을 고정한다.

### 2.2 내부 순서

A: 자료·실행 환경 확인 및 기존 근거 재배치
→ B: VLA-0 donor / repetition-penalty 통제
→ C: SpatialVLA 동일-within-call full-episode 평가
→ D: intermediate-context 보조 분석
→ E: 준비된 경우의 AutoVLA 폐루프·독립 자료 확인
→ F: 분석·검토용 원고·최종 패키지

A의 문헌·원고 작업은 GPU 실험과 독립적으로 진행할 수 있다. GPU가 부족할 때 우선순위는 B와 C의 주 분석, 이어서 D와 C의 부가 execution-setting 비교, E의 신규 GPU 평가 순이다. E는 먼저 비용이 작은 실행 가능성 확인을 수행하고, 현재 준비된 동일 정책 폐루프가 있다면 중요도를 재평가해 그 근거를 기록한다.

A–F를 모두 이번 요청의 작업 목록에 넣고, 실제 수행 여부를 추적한다. 단계별로 사용자에게 결과를 가져가 다음 프롬프트를 요청하지 않는다.

## A. 기존 근거 확인 및 바로 가능한 원고 수정

### A1. 이미 존재하는 분석을 새 실험으로 반복하지 않기

v04에서 확인된 근거:
- 본문 5.1·5.2와 보충자료 J/Table S9에 공식 NAVSIM v1 PDMS가 있다.
- 보충자료 J/Table S10 및 인접 문단에 FDE 임계값 2/3/4/5m 분석이 있다. 표에는 2/3/5m, 일부 4m 값은 문단에 있다.
- VLA-0 계산 비용은 보충자료 L에 있다.
- SpatialVLA matched follow-up은 새 intervention output 이전에 donor 규칙을 정한 연구다.
- AR-VLA는 references.bib에 있으나 현재 본문에서는 인용되지 않는다.

이 항목은 원시 결과와 기존 생성 스크립트를 찾아 가시화한다. 변경하지 않은 GPU 실험 전체를 무조건 재실행하지 않는다. 기존 회귀 검사는 adapter나 분석 코드를 바꿨을 때 필요한 범위로 수행한다.

기존 PDMS 예:
- RN: F +0.072 [0.034, 0.111], E +0.023 [-0.004, 0.049].
- C_NAT: F -0.055 [-0.093, -0.017], E4 -0.026 [-0.046, -0.007].

원본 기록과 부호·집단·horizon을 대조한다. RN은 교정, C_NAT은 GT를 Normal-source 값으로 바꾸는 대비라 부호가 다르다. PDMS는 4초, 기존 사건의 FDE는 5초다. 해당 구간은 원고의 approximate drive-cluster 구간이며 day/vehicle 독립 확인으로 격상하지 않는다.

### A2. 원고의 명확한 수정

- severe trajectory event를 “rule-rejected high-FDE plan” 또는 정의가 드러나는 중립적 명칭으로 통일한다. 기존 식·사전 임계값은 유지한다. 표준 정의와 다른 “miss rate”라고 이름만 바꾸지 않는다.
- 본문 주행 표에 PDMS와 ADE/FDE, 기존 사건 결과를 함께 배치한다.
- 실패 선택 F와 적격 프레임 E/E4 결과를 나란히 보여 준다. E/E4 역시 early mismatch에 조건부인 모집단임을 유지한다.
- AutoVLA plan geometry는 plan-space response control로 명확히 구분하고, 실제 실행 command/TCP와 같은 종류의 실증처럼 묶지 않는다.
- Figure 2에 네 주요 cell의 작은 실제 토큰 예시와 emitted action / conditioning copy 구분을 추가한다.
- 정확/근사/보류된 추론, independent unit, 가정을 estimand별 간결한 표로 정리한다.
- VLA-0 공식 loop의 3-step delay는 경로 검증의 예측 결과이지 학습된 모델 특성 발견이라고 설명하지 않는다.
- SpatialVLA within-call과 기존 cross-step 성공률 실험을 섞어 하나의 효과로 추정하지 않는다.
- 자세한 cluster 계보·기존 discovery 시행착오·체크 횟수 일부는 보충자료로 이동해 새 근거를 위한 지면을 확보한다.
- 기존 LLM Usage Statement는 보존한다. 이번 사용 사실에 맞는 확인 가능한 도구 정보만 갱신한다.

계산 비용은 model calls, sampling/forcing calls, wall time, GPU 점유시간, peak VRAM, hardware/renderer를 가능 범위에서 표로 모은다. 로그가 없는 값은 미확인 또는 추정 방식으로 표시한다.

learned extrapolator의 입력·MLP 구조·loss·학습/검증 drive 수·checkpoint 선택·seed를 재현 가능한 수준으로 기록한다.

## B. 신규 GPU 실험 1: VLA-0 repetition penalty와 donor 통제

### B1. 질문과 기본 패널

질문: native decoder의 latest/older 반응 차이가 반복 패널티와 donor 입력 크기에 어느 정도 의존하는가?

가능하면 기존 149 observations / 30 LIBERO tasks의 정확한 입력·자연 prefix·donor bank를 재사용한다. 이 입력은 기존 결과를 알고 있는 재분석 표본이다. 독립 확인 자료 또는 기존 실험의 사전등록으로 부르지 않는다.

기본 조건:
- penalty = native 1.05
- penalty = 1.0
- arm = SELF / O / L / OL
- 동일한 자연 prefix와 donor를 두 penalty 조건에서 고정
- SELF는 강제로 되돌려 주는 자연 conditioning copy
- O/L/OL은 기존 편집 span·출력 parser·response 정의 유지

공식 모델/체크포인트의 실제 generation config에서 penalty 값을 다시 확인한다. 1.05가 아니면 조용히 덮어쓰지 말고 실제 설정과 차이를 기록한다.

prefix는 고정된 조건이므로 penalty=1.0에서 첫 토큰부터 자연 생성한 전체 출력이 원래 prefix와 같아야 한다고 요구하지 않는다. 각 penalty 설정에서 동일 prefix에 조건부인 기준 continuation을 다시 계산하고, 그 설정 안에서 SELF parity를 확인한다. arm별로 다른 penalty를 적용하지 않는다.

### B2. validation과 기록

실제 native incremental/forced path를 사용한다. 기존 self-patch를 통과한 adapter를 우선 재사용한다.

변경한 설정마다 작은 validation subset에서:
- adapter와 해당 설정의 native conditional continuation 일치
- SELF와 조건부 자연 reference의 token/EOS 일치
- cache 초기화와 logits processor state 정합성
- 설정별 입력 prefix·seed·donor hash 일치

짧은 진단 subset에 processor 전 raw logits, processor 후 logits, token mask, 적용된 penalty/history state를 저장한다. 같은 token set이라도 모든 processor 효과가 통제됐다고 단정하지 않는다.

### B3. 분석

고정한 원래 response D로:
- 각 penalty의 A_L, A_O, A_L−A_O
- penalty 설정 간 paired contrast 차이
- task별 효과와 task-cluster 구간
- 기존 matched-distance 및 same-token-set 부분집합의 coverage와 보조 결과

기존 baseline penalty 결과의 허용 오차를 먼저 정하고, 원래 기록과 불일치하면 원인을 분류한다. 기존 결과와 새 결과를 무조건 덮어 평균하지 않는다.

반복 패널티가 native decoder의 일부라는 점과, 이를 제거한 결과가 학습된 문맥 의존성에 관한 보조 질문이라는 점을 구분한다. 결과가 줄어들어도 버그·반증으로 즉시 단정하지 않는다.

### B4. donor 보강

먼저 기존 bank에서 각 L/O donor의 다음 정보를 정리한다:
- semantic distance 및 성분별 translation/rotation 변화
- token 길이, Hamming/편집량, token-set 변화
- candidate ID, source seed, 수용/거절 사유

원래 거리·token 공통 지지가 부족하면, 새 continuation을 보기 전에 다음 follow-up 규칙을 고정한다:
- 동일 관측에서 최대 64개의 T=0.7 donor sample
- 출력 반응을 보지 않고 첫 joint-valid candidate를 선택
- 기존 길이·외부 span 보존 조건과 0.8≤d_L/d_O≤1.25를 적용
- 가능한 token 변경 유형 균형을 별도 보고
- 후보를 찾지 못하면 SOURCE_MISSING
- hard cap 이후 추가 탐색이나 유리한 donor 재선택 금지

새 규칙의 패널은 원래 bank replay 패널과 별도 연구로 분석한다. 입력 거리와 token 조건을 완전히 맞출 수 없으면 달성한 균형·수용률·제외 특성을 그대로 보고한다.

## C. 신규 GPU 실험 2: SpatialVLA 동일 within-call 개입의 full episode

### C1. 이 실험이 채울 공백

현재 within-call L/O 연구는 decoding을 측정했고, 기존 성공률 연구는 별도 cross-step intervention이다. 이번에는 같은 within-call 개입에서 decoding, executor, TCP/EE, full-task outcome을 함께 기록한다.

기존 cross-step 조건의 Reverse R과 혼동하지 않도록 이번 self-patch arm 이름은 SELF를 사용한다.

### C2. task/state 선택

기본 목표:
- 3개 tasks
- task당 20개의 서로 다른 초기 물리 state/layout
- 기존 coke·move_near 외 과제 1개 이상
- task당 N/L/O full episodes
- SELF full-episode validation은 task당 사전 선택한 2개 states

기본 계획의 최대 full episodes는 N/L/O 180개 + SELF validation 6개다. 이는 검정력 보장 수가 아니다. 실제로 독립 layout이 20개 없는 task를 20개라고 보고하지 않는다.

서랍 과제는 기존 15 reset scenarios가 7 pose groups를 공유했다. 지시문·로봇 외형·seed 변경만으로 새 독립 물리 배치라고 세지 않는다. 중복 물리 상태를 묶고, task별 실제로 확보한 layout 수를 보고한다. 반복 seed를 늘려 부족한 layout 수를 채웠다고 하지 않는다.

세 번째 과제는 설치된 task registry와 체크포인트 지원 범위를 확인한 뒤, 신규 outcome을 보기 전에 고정한다. 작업 가능성·상태 다양성·공식 종료 기준을 근거로 고르고 L/O 효과 방향으로 task를 선택하지 않는다. 불가능한 목표는 실제 task/state inventory와 blocker로 기록한다.

새 task/layout은 가능한 범위에서 기존 discovery·validation과 분리하고, reused 항목을 명시한다.

### C3. 개입

- 기존 SpatialVLA within-call adapter, official processor, fresh-cache 정책 사용
- 자연 chunk의 a1/a2는 emitted prefix로 유지
- L은 conditioning copy의 a2, O는 a1만 변경
- suffix a3/a4의 continuation을 생성
- 기존처럼 자연 gripper를 유지하고 translation/rotation donor를 사용
- primary donor: 같은 관측의 공통 sampled chunk에서 first joint-valid distance-matched candidate
- T=0.7, 최대 64 samples, 0.8≤d_L/d_O≤1.25
- donor 선택에 suffix 반응·성공 여부를 사용하지 않음

개입은 사전에 고정한 한 call에서 한 번만 적용한다. 기본 anchor는 zero-based policy call 8이다. settling step, environment step, policy call, token generation index를 구분해 기록한다. 개입 전 종료·invalid output·SOURCE_MISSING의 처리 규칙을 사전에 고정한다.

동일 초기 state에서 공통 natural trajectory를 따라 anchor 직전까지 진행하고, simulator·controller·gripper·ensemble buffer 등 이후 실행에 영향을 주는 상태를 보존한다. N/SELF/L/O의 anchor 관측과 emitted prefix가 같아야 한다.

편집된 conditioning token을 이미 emitted된 실행 prefix에 덮어쓰지 않는다. 단, 디코딩된 suffix는 공식 executor를 통해 실제로 전달한다. 그 경로와 적용 가중치를 기록한다.

### C4. 실행 설정

주 분석에는 현재 official SimplerEnv adapter의 기본 execution setting을 포함한다. 실제 config를 기록하고, 논문·과거 프롬프트의 표기만 믿고 r1e4라고 가정하지 않는다.

서로 다른 arm은 개입 이후 자기 state와 관측으로 replanning한다. 자연 rollout의 저장 카메라만 재생한 것을 정책 폐루프 결과로 부르지 않는다.

전체 task horizon 또는 공식 종료 기준까지 실행한다. 모든 arm의 horizon·early stopping·success latch 규칙을 통일한다. 짧은 48-step 진단 성공 flag를 full-task success로 보고하지 않는다.

부가 execution-setting 비교는 주 실험이 완료되고 자원이 남을 때 수행한다:
- 사전 선택한 task당 최대 5개 layouts
- r4e1처럼 chunk suffix가 더 직접 전달되는 설정
- official setting과 별도로 paired response와 delivered dose를 보고
- 두 설정의 feedback 빈도·ensemble 차이를 함께 기록

### C5. path gate와 결측

최소한 다음을 확인한 뒤 본 실행:
- SELF continuation token/EOS parity
- N/SELF command와 simulator 상태의 결정성
- emitted prefix 보존
- edited suffix → executed command 전달
- state restore 및 controller/ensemble 복원
- shape/type/token parser 유효성

실행 중 gate가 깨진 관측은 통과한 것처럼 남기지 않는다. 해당 branch를 분리하고 원인을 고친 뒤 수정 이후 자료를 구분한다. 다른 독립 작업은 계속한다.

SOURCE_MISSING 또는 적용 실패 단위를 분석에서 지울 때 전체 attempted 수·task/state 분포와 제외 사유를 보고한다. 주 결과가 donor-eligible 집단의 효과이면 그 모집단을 명시한다. no-op을 포함한 “개입 시도” 모집단 요약을 추가할 때는 실제 측정과 동일성에 따른 재사용·파생 값을 구분하고, 두 estimand를 혼합하지 않는다.

### C6. 같은 unit에서 수집할 것

- 첫 개입 call의 L/O continuation distance 및 donor distance
- raw/generated/forced token IDs와 EOS
- 자연 chunk와 편집 chunk
- 실행한 command, 각 chunk의 전달 가중치/실행 dose
- simulator time/step, EE/TCP pose, 실제 관측
- 최초 command 차이 step, 최초 운동 차이 step
- 고정 horizon별 paired TCP/EE 거리 및 전체 시계열
- task 종료·성공·timeout·invalid output·미적용 사유
- task/layout/pose-group/state/seed/instruction ID
- state restore·SELF·replay 검사의 결과와 허용 오차

### C7. 분석

신규 결과를 보기 전에 primary contrasts와 집계 규칙을 JSON/MD로 고정한다.

기본 분석:
- task별 decoding L/O 반응 및 차이
- task별 executed command·TCP/EE 반응
- task별 full-episode success L−N, O−N, L−O
- 실제 독립 layout/pose group으로 clustering
- task 평균을 보일 때 task별 결과도 나란히 제시
- donor 수용·적용·완료 흐름
- 같은 unit의 decoding·motion·success를 연결한 도표

새 결과를 기존 cross-step success와 pooling하지 않는다. 3개 task에서 관찰한 결과를 임의의 모든 task에 대한 일반화로 표현하지 않는다.

Equivalence 검정은 이번 기본 실행의 필수 목표가 아니다. 우선 paired 효과와 불확실성을 보고한다. 비유의를 invariance라고 쓰지 않는다. 기존 ±10pp·사후 ±5pp margin을 새 연구에 자동 상속하지 않는다. 최종 equivalence 주장 채택은 REVIEW_AFTER_RESULTS에 남긴다.

## D. 신규 GPU 보조 분석: intermediate-context 고정의 영향

B와 C의 주 실험을 우선한다. 고정된 입력과 adapter를 재사용해 저비용으로 가능한 범위에서 수행한다.

질문: O에서 intermediate action을 자연값으로 고정하는 설계가 L/O 차이 해석에 얼마나 영향을 주는가?

기본 패널은 VLA-0 30 tasks에서 task당 기존 유효 관측 1개를 ID순으로 사전 선택한다. donor·prefix가 유효하지 않은 경우 다음 유효 ID를 쓰되 개입 출력으로 선택하지 않는다.

조건:
- bridge-fixed: 현재처럼 O 뒤의 intermediate conditioning action을 자연값으로 고정
- bridge-regenerated: O를 변경한 뒤 intermediate conditioning copy를 다시 생성하고 같은 suffix 구간 비교

이미 emitted된 실제 행동은 유지한다. bridge-regenerated는 변경된 conditioning 값의 수·경로가 다르므로 같은 dose의 L/O 실험이나 자연 간접효과 식별로 해석하지 않는다.

비교 가능한 동일 suffix 범위를 사용하고, intermediate 변화량·suffix 반응을 저장한다. 결과는 별도 보조 분석이며 주 분석과 pooling하지 않는다. 새 경로의 parity를 확보할 수 없으면 BLOCKED로 기록하고 그 수치를 핵심 근거로 사용하지 않는다.

## E. AutoVLA 폐루프 및 독립성: 확인 가능한 범위에서 실행

### E1. 폐루프 준비 상태 확인

같은 AutoVLA checkpoint의 실행 가능한 정책 loop가 현재 서버에 있는지 확인한다:
- simulator/renderer와 카메라 입력
- checkpoint·codebook
- ego state/제어기/경로 추종
- 정책이 실제 이동 이후의 자기 관측으로 replanning하는 경로
- 해당 benchmark의 공식 평가 스크립트

현재 원고의 AutoVLA_PDMS_89를 바로 CARLA에서 실행할 수 있다고 가정하지 않는다. 새 모델 재학습·대규모 simulator 이식은 이번 요청의 필수 완료 조건이 아니다. 준비되지 않았으면 정확한 blocker와 최소 후속 필요사항을 남기고 B/C/F를 계속한다.

준비된 경우에는 기존 route registry에서 deterministic한 기준으로 최대 12개 routes를 먼저 고정하고, N/SELF validation 후 실행한다. route 수가 부족하면 실제 수를 보고한다. 가능한 조건은 동일 route/weather/seed로 paired evaluation한다.

주행 intervention은 현재 코드의 실제 구현과 decoder/executor를 먼저 확인해 기존 frozen-source context edit를 재사용한다. 관측 feedback 이후 GT를 donor로 쓰는 경우 그 GT의 정의·정합성을 먼저 확인한다. 분기 이후 reference log의 GT를 해당 branch의 expert라고 부르지 않는다. GT가 정합하지 않으면 검증된 model-sampled donor를 쓰는 별도 prospective protocol로 구분한다. 이를 기존 oracle 교정 효과의 재현이라고 부르지 않는다.

공식 route completion, infractions, benchmark score와 적용률을 기록한다. 유의성·성공 방향으로 route를 추가/제거하지 않는다. 작은 표본이면 기술적으로 보고한다. PDMS나 pseudo-simulation 결과를 실제 정책 폐루프 결과로 이름만 바꾸지 않는다.

### E2. 독립 자료 확인

새 입력·GT·day/vehicle metadata가 있는지 inventory를 확인한다. 결과 보기 전에 기존 development와 P1-C에 대해 day와 vehicle이 모두 분리되는지 graph로 확인한다.

그런 자료가 있으면 기존 frozen matched-prefix protocol을 옮길 수 있는지 확인하고, resource limit 안에서 deterministic ID순의 고정된 소규모 패널을 실행한다. sample 수·선택 규칙·대상 모집단을 먼저 기록한다. training overlap은 확인되는 범위에서 별도 명시한다.

자료가 없으면 기존 6개 component의 효과·leave-component-out 기술 요약까지만 만들고 external validation이라고 부르지 않는다. bootstrap 횟수·seed·drive ID를 늘려 독립 자료를 확보했다고 하지 않는다. p값을 유리하게 만들기 위해 대비 family나 cluster 정의를 바꾸지 않는다.

## F. 문헌·분석·원고·패키지 완성

### F1. 관련 연구

공식 논문/프로젝트를 직접 확인하고 task·context·학습/추론·평가 수준의 차이를 설명한다. 아래 연구를 모두 새 실험 baseline으로 설치할 필요는 없다.

- WorldVLA: https://arxiv.org/abs/2506.21539
- AR-VLA: https://arxiv.org/abs/2603.10126
- AC²-VLA: https://arxiv.org/abs/2601.19634
- CAC-VLA: https://arxiv.org/abs/2607.04816
- ACoT-VLA: https://openaccess.thecvf.com/content/CVPR2026/html/Zhong_ACoT-VLA_Action_Chain-of-Thought_for_Vision-Language-Action_Models_CVPR_2026_paper.html
- DeepThinkVLA: https://arxiv.org/abs/2511.15669
- TacitVLA: https://arxiv.org/abs/2610.17540
- Think at 5 Hz, Act at 20 Hz: https://arxiv.org/abs/2607.15621
- SimLingo: https://openaccess.thecvf.com/content/CVPR2025/html/Renz_SimLingo_Vision-Only_Closed-Loop_Autonomous_Driving_with_Language-Action_Alignment_CVPR_2025_paper.html
- NAVSIM v1: https://arxiv.org/abs/2406.15349
- NAVSIM v1/v2: https://github.com/autonomousvision/navsim
- AutoVLA: https://github.com/ucla-mobility/AutoVLA

BibTeX에 들어 있다는 이유만으로 인용을 완료했다고 하지 않는다. 본문에서 실제로 인용되는지 확인한다.

표기·저자·연도·version을 공식 metadata에 맞춘다. DeepThinkVLA는 2025 최초 제출·2026 수정본이므로 인용하는 판본에 맞춘다. 새로운 문헌의 전문을 확인하지 못하면 확인한 범위까지만 비교한다.

비교할 축:
- WorldVLA의 action 의존·mask와 controlled within-call edit의 차이
- AR-VLA의 장기간 action memory와 같은 call의 위치별 contrast의 차이
- AC²/CAC/ACoT의 계산 배분·latent/action reasoning 구조와 frozen token prefix 개입의 차이
- DeepThink/Tacit의 reasoning 학습·추론 의존성과 action-token conditioning의 차이
- perception cache staleness·closed-loop driving과 같은 call의 action-token 위치의 차이

※ 작업 문서와 최종 보고서는 한국어, 기존 논문 본문은 영어로 작성한다.

### F2. 통계와 검증

- protocol/seed/state/donor/contrast를 output 전에 고정한 시점과 hash를 기록한다.
- 기존 알려진 결과의 사후 재분석과 새 prospective follow-up을 분리한다.
- unit·weight·cluster·반복수·결측을 표마다 명시한다.
- 기존 exact/approximate/withheld 정책을 사후 임의로 대체하지 않는다.
- 결과 표·그림은 raw records에서 생성한다.
- 수치 재계산, denominator, duplicate state, self-patch, cache, dtype, executor 전달을 필요한 범위에서 확인한다.
- 같은 unit의 시간 step·arm·call을 독립 n으로 세지 않는다.
- 미래 관측에서 원래 demonstration을 분기 후 branch의 GT로 사용하지 않는다.
- 부트스트랩 횟수가 독립 물리 상태 수를 늘려 주는 것은 아니다.
- 조건부 decoding 반응·실행 운동·성공률은 서로 다른 결과다.
- 결과가 작거나 반대인 항목도 반환한다.

### F3. v05 검토용 원고

이번에 반영:
- A의 명확한 정정·기존 근거 가시화·관련 연구
- 확인한 새 수치의 중립적 results 초안
- 모든 task별 결과·coverage·한계
- 검토에 필요한 후보 표·그림
- 기존 전체 보충자료와 새 분석의 통합

결과 확인 후까지 보류:
- 새 결과를 초록의 대표 수치로 고르는 결정
- 제목·기여·최종 결론의 대폭 수정
- backbone/task 일반화의 최종 범위
- success invariance/equivalence/utility 개선의 채택
- 어떤 신규 패널을 본문 핵심으로 고를지의 과학적 판단

v05는 검토용 candidate로 표시한다. 아직 검증되지 않은 강한 결론을 본문에 확정적으로 쓰지 않는다. 새로운 결과를 반영한 표·그림은 ID를 안정적으로 부여해 추후 교체가 가능하게 한다.

공식 스타일을 유지하며 본문 8쪽을 맞춘다. 글자·여백을 임의로 줄이지 않는다. 필요한 상세 기록을 보충자료로 옮긴다. 전체 supplementary를 보존하며 추가분만 따로 내서 기존 A–H 등을 누락하지 않는다.

main.tex와 supplementary.tex를 깨끗한 환경에서 각각 컴파일하고 페이지 경계를 확인한다. 표·그림·참고문헌 링크·미정의 citation·겹침·잘림을 렌더링해 확인한다. 최종 원고·PDF·ZIP 사이의 수치와 파일 버전이 같아야 한다.

## 3. 이후 결과 확인에서 다시 검토할 항목

REVIEW_AFTER_RESULTS_20261023_v01.md에 아래 질문별로 근거 파일·표·그림·미해결 사항만 모은다. 답을 맞춰 놓고 유리한 근거를 선택하지 않는다.

1. penalty=1.0에서도 위치 차이가 유지되는가? native decoder와 모델 자체 의존성을 어디까지 구분할 수 있는가?
2. donor 크기·token 변경량·수용 결측이 어떤 결론을 제한하는가?
3. 동일 within-call decoding 반응이 실제 command와 운동으로 전달되는가?
4. 그 결과가 task success와 얼마나 연결되는가? 표본·구간으로 어떤 주장을 지지할 수 있는가?
5. bridge-fixed/regenerated 차이는 현재 위치 비교의 해석을 어떻게 바꾸는가?
6. 실제 폐루프 주행 또는 day/vehicle 독립 확인이 추가됐는가? 없으면 주행의 비중과 제목 범위를 어떻게 조정할 것인가?
7. 어떤 결과를 본문에 둘지, 초록·기여·결론을 어디까지 수정할지?
8. 기존 관련 연구 대비 새로 밝혀진 평가 판단이 충분히 명확한가?
9. 추가 실험이 정말 필요한지, 이미 충분한지? 필요하면 미해결 질문과 최소 설계만 제시한다.

이 항목의 존재를 이유로 A–F 실행·분석·기계적 수정·PDF 빌드를 미루지 않는다.

## 4. 반환물: 한 번에 검토 가능한 ZIP 하나

대표 반환 파일:
CVPR_REVIEW_BATCH_RESULT_20261023_v01.zip

ZIP 안에:
- README_RESULT_20261023_v01.md: 수행·미수행·핵심 수치·blocker·실제 GPU 비용
- TASK_STATUS_20261023_v01.json: A–F 세부 작업 상태와 근거 경로
- REVIEW_AFTER_RESULTS_20261023_v01.md
- 실행 전에 고정한 protocol과 변경 이력
- 코드 patch/commit·환경·실행 명령·resume 방법
- 원시/행 단위 결과, task/layout/state ID, donor/processor 기록, 필요한 replay 자료
- 결과 CSV/JSON 및 표·그림 생성 스크립트
- 전체 v05 검토용 TeX·figures·bibliography
- main/supplement PDF 및 build/render QA
- CVPR_OVERLEAF_20261023_v05.zip
- 검증 보고와 파일 해시

이미 서버에 있는 과거 대형 raw ZIP을 중복 복사할 필요는 없다. 다만 manifest에 정확한 파일명·크기·SHA-256·보관 경로를 기록하고, 이번 신규 결과를 재검토하는 데 필요한 자료는 빠뜨리지 않는다.

상태 값:
DONE / PARTIAL / BLOCKED / DEFERRED_REVIEW

GPU를 돌리지 않은 계획은 DONE이 아니다. DEFERRED_REVIEW는 위의 과학적 최종 판단에만 사용한다. 실행 불가 실험은 BLOCKED, 예산 등으로 일부 완료된 실험은 PARTIAL로 표시한다.

마지막 답변에는 실제 GPU 실행 범위·독립 unit 수·통과한 핵심 gate·미실행 항목·결과 ZIP 위치를 짧게 보고한다. 점수 상승이나 채택을 예측해 실험의 성공 기준으로 삼지 않는다.

