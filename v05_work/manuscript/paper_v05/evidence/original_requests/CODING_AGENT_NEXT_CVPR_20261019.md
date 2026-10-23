# 코딩 에이전트 실행 프롬프트 — CVPR용 남은 검증 마무리

## 목표와 종료 조건

현재 논문은 **AutoVLA의 조건부 action-context 효과를 통제된 개입으로 분석하는 연구**다. 최신 출발점은 `review_followup5`와 `MANUSCRIPT_AUTHOR_ANSWERS_20261019.zip`이다. 이번에는 아래 세 질문에 답하고, 원고 작성자가 곧바로 반영할 수 있는 근거 패키지를 반환하라.

1. 가중 eligible frame E/E4의 구간을 현재 자료에서 어느 수준으로 제시할 수 있는가?
2. matched-prefix interaction의 구간이 여러 조건의 공동 결과와 상위 군집을 제대로 반영하는가?
3. 기존 개발·평가 자료와 **수집일과 차량이 모두 분리된** 추가 검증 자료를 실제로 확보할 수 있는가?

각 질문에 `ANSWERED`, `APPROXIMATE_ONLY`, `BLOCKED` 중 하나의 결론을 주고 작업을 종료하라. 막힌 질문을 해결했다고 꾸미거나, 유의성을 얻을 때까지 방법·표본·margin을 바꾸지 마라. 이번 답변을 다시 받으면 원고 통합은 원고 작성자가 수행한다.

## 이번 작업의 경계

- 기본 실행 범위: **기존 자료의 CPU 분석, 새 독립 데이터의 가능 여부 확인, 필요 시 실행 가능한 후속 프로토콜·코드 준비**.
- **이번 프롬프트 자체로 새 GPU 추론·학습을 실행하지 않는다.** 새 독립 자료의 사용 가능성·표본 수·시간이 확인되지 않은 상태에서 실행을 약속하지 않는다.
- OpenVLA, Impromptu, π0/π0-FAST 등 새 모델 확장은 이번 작업에 포함하지 않는다.
- 기존 입력·출력·protocol·raw를 덮어쓰지 않는다. 새 결과는 `outputs/cvpr_next_validation_20261019/`에 저장한다.
- 이미 확인된 42개 입력 해시와 9개 CSV 재현을 전부 반복하지 않는다. **새로 사용하는 입력과 새 산출물**만 검증한다.
- 완성된 논문 `.tex`는 원고 작성자가 관리한다. 서버에 원고가 없으면 탐색을 반복하지 말고 제공된 CVPR source ZIP의 내용을 사용하라.
- 외부 계정 생성, 유료 자원 사용, 대규모/라이선스 제한 데이터 다운로드, 외부 메시지 발송은 수행하지 않는다.

## 원고 형식 조건과 역할

원고는 공식 CVPR review style로 본문·참고문헌 PDF와 보충자료 PDF를 분리한다. 확인 가능한 CVPR 2026 규정은 **그림·표를 포함한 본문 최대 8쪽, 참고문헌만 있는 추가 페이지 허용**이다. 8쪽을 반드시 채울 필요는 없다. 여백·글자 크기·줄 간격을 줄여 맞추지 않는다. 보충자료를 읽지 않아도 핵심 방법과 주요 근거·한계가 본문에서 이해되어야 한다.

2026-10-19 기준 CVPR 2027 CFP는 확인되지만 상세 AuthorGuidelines URL은 404였다. 제공되는 원고는 공식 2026 v1 author kit를 사용해 2027 제출을 준비한 초안이다. **2027 규정 확인 완료라고 쓰지 마라.** 제출 직전 해당 연도 규정·템플릿을 재확인해야 한다.

공식 출처:
- https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines
- https://github.com/cvpr-org/author-kit
- https://cvpr.thecvf.com/Conferences/2027/CallForPapers

에이전트가 반환할 표는 본문용 짧은 표와 보충자료용 세부 표로 구분하라. 본문을 직접 늘리거나 논문의 framing을 임의로 바꾸지 마라.

## 반드시 유지할 사실과 해석

- 주모델: AutoVLA. SpatialVLA는 closed-loop generation/TCP/success를 연결하는 보조 근거다.
- repeated development RN −39.8 pp, P1-C F RN −34.2 pp, 가중 E RN −4.1 pp는 **서로 다른 estimand**다.
- E는 2,191 eligible scenes를 353개 sampled scenes로 평가하며, F는 105 scenes / 748 units / 18 drives다. 전체 P1-C는 24 drives다.
- B2 E4는 t*≤4 등의 별도 domain으로, 1,775 eligible scenes를 312개 sampled scenes로 평가한다. F는 100 scenes / 711 units / 18 drives다.
- B2 primary older factor는 **한 위치**, primary anchor a=t*+3이다. 고정된 emitted prefix와 frozen source bank를 쓰며, 편집한 context 위치는 이후에도 유지된다.
- C_GT = GT_NSRC − GT_REF; C_NAT = NAT_NSRC − NAT_REF; INT = C_GT − C_NAT; C_PROV = GT_CSRC − GT_NSRC.
- F C_NAT=+25.5 pp. drive/day–vehicle 검정에서는 유지되지만, 6개 day/vehicle connected component에서 Holm p=0.125다. 이는 새 분석으로 지워야 할 오타가 아니다.
- C_PROV의 component raw p=0.0625이며, C_GT/C_NAT의 0.03125와 다르다.
- GT-free는 perturbation 값·reference의 의미다. anchor t*와 strata는 GT 정의에 의존한다.
- G-family S0/SE/SF의 43,590 rows, six controls × two alternative settings의 12-test family는 이미 수행됐다. 이를 새 실험으로 다시 요구하지 않는다.
- CI simulation은 nominal 95%에서 최소 약 93% coverage를 보였다. “95% CI 검증 완료”라는 표현은 금지한다. Development RV 등 직접 simulation하지 않은 대비를 검증했다고 쓰지 않는다.
- 군집 sign-flip의 exact enumeration은 zero-null에서 독립·sign-symmetric cluster totals라는 가정하에 유효하다. randomization에 근거한 무가정 검정으로 부르지 않는다.
- R9 90.1%는 GT 세 위치를 **계속 유지**한 효과의 비율이다. GT 제거 후 지속 회복, 90% task success, 90% 하한으로 부르지 않는다.
- SpatialVLA는 수정된 근사 방법으로 두 과제 평균 ±10 pp만 지지한다. ±5 pp와 과제별 결론을 확대하지 않는다.
- “10 component이면 충분”이라는 표본 보장은 없다. half-effect power는 두 대비를 합치면 가정에 따라 약 32–95%였다.

## A. 통계 질문을 한 번의 CPU 작업으로 마무리

### A1. 먼저 추정 대상과 현재 계산을 점검하라

`ESTIMAND_AND_METHOD_AUDIT.md`에 다음을 명확하게 쓰라.

1. 각 결과가 **관측된 집합의 기술적 효과**, 주어진 eligible frame의 유한모집단 효과, 새 drive를 포함하는 초모집단 효과 중 무엇을 추정하는가?
2. E/E4의 F 전수 포함, S 층별 표집, inverse-inclusion weights, scene 평균 후 군집 집계가 코드에 어떻게 반영되는가?
3. zero-null p와 효과 크기의 CI는 어떤 서로 다른 가정에 의존하는가? CI 보류가 p의 일반적 유효성 보장이 아님을 설명하라.
4. 기존 simulation이 scene 구조를 고정하고 replicate별 조건부 참값을 사용한 범위와, 새 drive 모집단에 대한 보장 사이의 차이를 설명하라.

관련 입력: P1-C `analysis/units.parquet`, Stage-1 eligible frame/strata/inclusion records, B2 cell-level outputs, 각 scene의 weights/drive/day/vehicle 정보, 기존 CI audit와 final-number outputs. 파일 이름만 같고 hash가 다른 입력을 섞지 마라.

### A2. 가중 E/E4 구간의 정직한 평가

새 simulation이 필요하면 결과를 보기 전에 DGP·estimand·methods·performance를 `STATISTICAL_CHECK_PLAN.md`로 고정하라. R을 늘리는 것만으로 기존 구조의 한계가 해결된다고 가정하지 마라.

- 실제 표집 구조와 분석 가중치, 군집 크기 불균형, drive 간 이질성을 반영해야 한다. 분석 대상이 새 drive 모집단이면 새로운 drive/scene의 변동을 모델에 포함하라.
- RN과 RV는 기준 arm이 다르다. 모든 대비에 Normal-arm rate 하나를 재사용하지 마라.
- 가상의 유한모집단을 사용하는 경우 전체 모집단의 참값과 표집 과정이 분리되어야 한다. clipping 전 target과 실제 생성된 참값을 구분하라.
- 사전 계획에 양·음·zero·관측 규모보다 줄인 효과를 포함하라. 모든 parameter 값은 가정임을 표시하고 실제로 추정된 값과 구분하라.
- 후보 방법은 기존 cluster-t와 합리적인 대안으로 제한하고 사유를 설명하라. 다수 방법을 돌려 현재 데이터에 가장 유리한 구간을 고르지 마라.
- 비교 가능한 핵심 시나리오는 가능하면 5,000회 이상 실행하고 coverage, bias, width, zero-width, 실패율, MC SE와 MC interval을 모두 반환하라. 계산 비용 때문에 줄이면 실제 반복 수와 미완료 범위를 표시하라.
- nominal 수준을 충족하지 못하면 `APPROXIMATE_ONLY` 또는 `WITHHOLD`를 유지하라. 불확실성 평가가 불충분한 것을 “어떤 CI도 불가능”이라고 일반화하지 마라.

최종 산출물은 대비별 `interval_method / evidence_scope / actual_coverage / MC_uncertainty / recommended_wording / limitations`가 있는 `FINAL_INTERVAL_POLICY.csv`다. 충분한 근거를 만들 수 없으면 새 구간을 억지로 복구하지 않고, 논문에 쓸 수 있는 한계 문장으로 종료하라.

### A3. INT는 joint outcome 구조를 확인하라

- 같은 unit의 4개 primary cell 결과를 함께 보존한 차이의 차이를 만든 후 scene→cluster로 집계하는지 확인하라.
- 두 contrast의 독립적인 simulation 결과를 빼서 INT의 coverage를 검증했다고 하지 마라.
- paired joint outcomes, between-drive variation, weights를 반영하는 생성 모형을 타당하게 정의할 수 있는 경우에만 INT interval의 별도 simulation을 수행하라.
- 정의가 어려우면 **기존 INT 구간은 미검증 sensitivity**라고 확정하고 종료해도 된다. “상호작용 없음”, “효과 동일”, “older context 무관”이라는 결론을 얻으려 하지 마라. 사후 equivalence margin도 만들지 마라.

### A4. 통계 작업의 종료 기준

한 번의 작업에서 질문별 사용 가능 범위와 근거를 확정하라. CI가 완전히 검증되어야만 모든 작업을 끝낼 수 있다고 생각하지 마라. 기존 결과를 바꾸는 오류가 발견되면 변경 전/후 값, 원인, 영향을 받는 claim ID를 표로 반환하라. p<0.05 또는 coverage>0.95를 얻기 위한 반복 수정은 하지 마라.

## B. 독립 검증 데이터의 실제 가능 여부를 확정

`review_followup5/C_DISJOINT`는 이미 검토한 자료에서 `BLOCKED_NO_DISJOINT_DATA`였다. **같은 navtest/navhard/trainval/private-test 목록을 다시 전수 조사하는 것만으로 새 작업을 끝내지 마라.** 새로 추가된 파일·메타데이터·명시적으로 제공된 다른 수집 자료가 있는지 확인하고 기존 조사 대비 delta를 보여라.

독립성은 새 log 이름이나 새로운 day–vehicle 조합만으로 주장할 수 없다. 평가하려는 전체 신규 cohort가 기존 development 및 P1-C의 **day 집합과 vehicle 집합 모두와 disjoint**여야 한다. 신규 cohort 내부 공유는 별도 connected components로 묶는다.

확인 항목:

1. scene/log/drive/day/vehicle의 식별자와 중복 여부. 식별 불가능은 disjoint가 아니다.
2. AutoVLA 입력, GT trajectory, codebook·checkpoint·evaluator 호환성.
3. 알려진 training/fine-tuning/checkpoint 선택과의 overlap. 확인하지 못한 것은 `UNKNOWN`으로 둔다.
4. F/S와 t* eligibility를 적용하기 전 후보 개수, 적용 후 frame 개수, 각 independent component의 개수.
5. 자료 사용 조건과 현재 로컬 가용성. 다운로드하지 않은 자료는 확보 완료가 아니다.

산출물:
- `DISJOINT_DELTA_INVENTORY.csv`
- `DISJOINT_FEASIBILITY.md`
- 가능할 때만 scene 수준 후보 manifest와 overlap 검증 코드/결과.

새 적격 자료가 없으면 `BLOCKED_NO_NEW_DISJOINT_DATA`로 확정하고, 필요한 메타데이터·입력·GT를 구체적으로 적어라. 기존 shared-day/vehicle 결과를 새 독립 재현으로 이름 바꾸거나, training split을 검증 없이 사용하지 마라. 자료가 없다는 이유로 다른 모델·다른 과제를 임의 추가하지 마라.

## C. 데이터가 실제로 있을 때만 GPU 실행 준비

B에서 적격성이 확인되면 **GPU 실행 없이** 다음 파일을 준비하라.

- `DISJOINT_REPLICATION_PROTOCOL.md`: 질문, domain, 표집, primary contrasts, outcome, cluster axis, family, exclusions, 종료 규칙.
- `disjoint_manifest.json` 및 모든 입력 hash.
- 기존 검증된 inference path를 재사용하는 runner와 CPU dry-run/manifest tests.
- 검증 gate 목록: no-op, 동일 값 self-patch, frozen-bank identity, context/output 분리, emitted-prefix lock, cache 재계산, 동일 batch/seed/CRN.
- 최소 계획은 AutoVLA matched-prefix **C_NAT**의 외부 재현을 우선하며, RN/RV 추가의 비용·과학적 목적을 따로 보여라. family를 좁히는 이유는 독립 평가 전에 고정해야 한다.
- 표본 수는 기존 10-component 가정을 그대로 복사하지 말고, 줄인 효과와 component 내 상관의 민감도를 고려하라. 최소 exact p의 해상도와 목표 power는 별개다. joint Holm power를 계산하지 않았으면 그렇게 쓰지 마라.
- `RESOURCE_ESTIMATE.md`: GPU 모델·예상 시간·메모리·디스크, 실제 계측과 외삽의 구분, 재개/중단 방법.
- `RUN_NEW_GPU=false`를 기본값으로 둔 실행 명령. 독립 자료와 프로토콜 검토 뒤 실행할 준비 상태를 `READY_FOR_REVIEW`로 반환하라.

자료가 없으면 이 단계는 `BLOCKED_BY_DATA`이며, 기존 cohort 재추론을 대체로 실행하지 않는다.

## D. 원고 작성자에게 반환할 결과

다음을 `CVPR_NEXT_VALIDATION_20261019.zip` 하나로 묶어 반환하라.

1. `DECISION_SUMMARY.md`: 세 질문별 답, 근거, 미해결 이유, 다음 단계. 첫 화면에서 CPU 작업으로 답한 것과 새 데이터가 필요한 것을 구분.
2. `ESTIMAND_AND_METHOD_AUDIT.md`, `STATISTICAL_CHECK_PLAN.md`, 실제 simulation code/commands/results, `FINAL_INTERVAL_POLICY.csv`.
3. `DISJOINT_DELTA_INVENTORY.csv`, `DISJOINT_FEASIBILITY.md`, 해당 시 C의 준비 파일.
4. `PAPER_READY_NUMBERS.csv`: 실제 변경·추가된 행만. claim ID, cohort, estimand, outcome, sign, estimate, CI status/method, p/adjustment/family/alpha, units/scenes/drives/components, source hash 포함.
5. `MANUSCRIPT_ACTIONS.md`: 본문에 유지할 문장, 보충자료로 이동할 내용, 삭제할 해석, 제한 문장. 본문용 신규 표는 최대 1개로 압축하고 상세 표는 보충자료용으로 제공.
6. `MISSING.md`, `INPUT_HASHES.csv`, `RUN_COMMANDS.md`.

마지막 답변은 새 GPU 사용량 0임을 명시하고, 통계 구간의 결론과 독립 데이터의 상태를 먼저 설명하라. 실제로 commit/push한 경우에만 hash와 상태를 보고하라. 결과가 음성·불확실·차단이어도 그대로 반환하면 이번 작업은 완료다.
