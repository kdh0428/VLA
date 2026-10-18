## 6g. review_followup4 (2026-10-18 수정): 통계 구간 정정, window 의미, 수치 민감도, matched-prefix 진단

AUTOVLA_NEXT_STEPS_20261018 실행 결과입니다. 주 모델은 AutoVLA이고, SpatialVLA·OpenVLA는 서로 다른 개입·실행 수준의 보완 증거입니다.

- 결과 위치: `autovla_misalignment_poc/outputs/review_followup4/`. 최종 상태 `STATUS.md`, 주장 변경 `CLAIM_CHANGES.md`, 원고 문장 `MANUSCRIPT_REVISION.md`, 재현 명령 `RUN_COMMANDS.md`, 재현 ZIP `reproducibility_minimal.zip`과 재계산 결과 `REPRO_CHECK.md`, 파일별 hash `SOURCE_MANIFEST.csv`.
- commit: 5523ce3(A2/A3), 2a31ad7(A1/A4), 552b3ac(B1 결과, B2 프로토콜 사전 등록), 28f1bfd(FEASIBILITY), c7d1bba(B2 결과). 이번 마무리 문서는 commit하지 않았습니다.
- **역사 기록 보존.** review_followup3 CP2에서 P3/P4를 실행하지 않기로 한 결정은 그대로 둡니다. B2는 별도 프로토콜을 가진 새 round이며 P3를 소급 DONE으로 바꾸지 않습니다.
- 원고 `.tex`는 저장소에 없습니다. 문장별 수정안과 기계 생성 표(`paper_quantitative_package/tables.tex` Table 12–14)만 제공합니다.
- 계산: A1·A2·A4·A5·C는 CPU만. GPU(RTX 5090)는 A3 Impromptu parity 소규모 실행(약 3분), B1(18.3분), B2 gate·pilot·본 실행(dev 52.9분, P1-C 약 86분)뿐입니다. 새 효과 실험은 B2 하나입니다.

**A1 — 성공률 CI 퇴화 (사후 수정, 원 분석 보존)**

- 원인: `stats_core.signflip_ci`와 sign-flip TOST는 이동된 cluster 합이 0에 대칭이라고 가정합니다. 차이가 {−1, 0, 1}이고 0이 많은 쌍 이진 자료에서는 δ = 0에서만 성립하므로, 모든 차이가 0이면 구간이 동점 허용오차 ±2.4958e−13으로 붕괴합니다(bit 단위 재현). δ = 0의 sign-flip / exact McNemar p와 Holm family는 유효하고 그대로입니다.
- 영향: OpenVLA 과제별 150행 중 11행이 폭 0, 75행이 한쪽 끝 0 고정; SpatialVLA 819행 중 37행이 끝 0 고정.
- 방법 선택(실제 자료 재계산 전에 규칙 고정): 독립 쌍은 Newcombe 10(최소 coverage 0.951), 군집 반복은 Rao–Scott design-effect Newcombe(과제 평균·상호작용은 MOVER). 두 방법 모두 폭 0 구간을 내지 않습니다.
- 판정 변경 36행(미러·clustering 중복 포함):
  - OpenVLA C − N 8-bin "95% CI가 0을 제외"는 동점 artefact였습니다: [−16.7, −7e−14] → [−17.1, +1.0](p = 0.115와 일치). R − N_rep의 보정 전 ±10%p 동등은 사라졌습니다(p_eq 0.049 → 0.052).
  - OpenVLA 과제별 CI(10 상태)는 인용하지 않습니다. 새 구간은 ±0.18–0.28로 정보가 없고 기술적으로만 씁니다.
  - SpatialVLA 실험 34 r1e4/perp_left coke C − F "±10%p 동등(Holm)" → 보정 전만(Holm 0.147). P1-D r2e1 − r1e1 C − F 상호작용(pooled, physical) Holm 0.045 → 0.051로 동등 불성립. P1-D r1e4 C − F move_near physical의 ±5%p(보정 전) 동등과 [−2.6, +4.3]은 artefact였습니다([−6.9, +8.6]).
  - **유지:** P1-D pooled ±10%p 동등(4개 대비, 두 clustering, Holm), 과제별 3/8, S1–S4 pooled ±10%p.
  - 과제별 CI 폭: "18–29%p(r4e1; r1e4 6.9–21.6%p)" → **18–25%p(r4e1; r1e4 13–20%p)**.
- 비유의는 동등성이 아니며, 폭 0 구간은 같음의 근거가 아닙니다. AutoVLA 쪽 같은 구성의 6개 행(P1-C "below MME" 규칙 포함)은 CI 끝이 아니라 p로 읽습니다.

**A2 — 교정 window의 의미 (GT 유지, 해제 없음)**

- 코드·실제 forward trace·저장 기록 모두에서, window는 고정 위치 t*+1 … t*+w에 GT를 쓰고 **이후 모든 생성 step의 문맥에 그대로 유지**합니다(gate trace 4,136/4,136 전이에서 유지, 제거 0건; 실험 9 저장 기록 23,936/23,936행이 규칙과 일치). window가 끝나면 새 GT 위치가 추가되지 않을 뿐, 기존 GT는 제거되지 않습니다.
- 허용 문장: "Three fixed context positions replaced with ground truth and retained in later contexts recovered about 90% of the full-history correction effect." 함께 보고: scene 가중 90.1% [81.2, 98.9], A− 365 단위 / 52 장면 / 13 drive, 분모 = 같은 run의 Normal − GT-history(41.7%p), 단위당 GT 위치 2.96개(자기 token과 다른 것 1.90개), 마지막 GT 위치 이후 생성 step 평균 3.8.
- P1-B의 "한 번만 보이고 바로 자기 token으로 교체되는" GT(transient)는 78% [64, 87], 유지되는 GT는 89% [79.5, 98.0], 차이 +4.8%p [+2.9, +7.8](사후, 326 단위 / 12 drive; 유지와 노출 횟수가 섞임).
- 삭제: "3 step 교정 후 GT 정보를 제거해도 효과 유지", "이후 생성은 자기 token만 참고", "짧은 일시적 교정으로 지속 회복 입증", "해제 후 horizon/감쇠/재발산" 표현(→ "마지막 GT 위치 이후 step", "GT가 남아 있는 동안의 재발산"). 별도 GT 제거 실험 없이 해제 결과를 만들지 않았습니다.

**A3 — Impromptu 재현성 원인 (기술적·미검증으로 낮춤)**

- 같은 경로·같은 batch는 bit 단위로 재현됩니다(exp_cde 재실행 108/108행, B = 1 반복 4/4). batch 크기, left padding, prefill/incremental 분할, decoder 구현 중 하나만 바꿔도 출력이 달라집니다. 원인은 bf16 수치 차이가 greedy digit 디코딩의 근소 동점에서 0.01 m 끝자리를 뒤집고, 모델이 그 값을 다시 문맥으로 증폭하는 것입니다.
- 문서의 "no-op parity" 23/300, 237/2,728은 같은 조건 반복이 아니었습니다(구현 차이, batch 구성 차이). 장면 수준 A−/증폭 판정은 수치만으로 5–6% 바뀝니다.
- 결론: Impromptu는 **기술적 보완 증거**(경로 parity 미검증)이며 강한 독립 재현으로 세지 않습니다. 바닥을 크게 넘는 대비(교정 −63.4%p, reverse +81.9%p)만 바닥과 함께 기술하고, 작은 대비(win1 −15.4%p, 방향 − 크기 +15.6%p 등)는 재현이라 부르지 않습니다.

**A4 — 분모·지표·범위**

| 집합 | 장면 | 쓸 수 있는 이름 |
|---|---|---|
| dev 자연 평가 | 2,747 | A− 52 = 1.89%는 평가 집합 빈도(배포 실패율 아님) |
| dev 기전 집합 | 208 선정(A− 52 + A+ 156; A− 비율 25%) | 기전 대비는 A− 층(52 장면 / 13 drive) 조건부 |
| P1-C 평가 대상 E | 적격 2,191 | 평가 집합(E) 추정량 |
| P1-C 개입 표본 | 선정 353(F 105 + S 248), 설계 가중 | RN −4.1 / RV +3.6%p는 이 353 장면의 가중 추정 |
| P1-C 실패 층 F | 105 장면 / **18 drive** | "24 independent failure drives"라고 쓰지 않음 |
| 운동 baseline "전체" | 208 선정 | 2,747·2,191·모집단 아님 |

- RN 평가 집합 효과의 약 60%는 A+(비실패) 층 기여입니다. OpenVLA 92.6%는 "하위 성분 중 **적어도 하나**가 주입 bin 이상" 지표이고, bin은 m·rad 혼합(8 bin = x 2.64 mm, yaw 5.18 mrad 등), 명령값이며 실현 운동·과제 실패가 아닙니다. SpatialVLA의 m 단위 값은 명령 누적, 실제 TCP는 cm 단위입니다. 과제별 반대 부호 점추정만으로 이질성·상쇄를 확정하지 않습니다. P1-C는 drive-disjoint이지 day/vehicle-disjoint가 아닙니다.
- A−/severe 식: A− = NOT `pra_labels.label_A`(종·횡방향 범주 규칙), severe = A− ∧ FDE5 > 3 m(pose 10). 이 임계값은 3 s horizon용으로 문서화됐지만 5 s 계획에 적용됐습니다(원 정의 보존).
- 64개 문장 위반 중 57개를 이 문서와 `paper_quantitative_package`에 줄 단위로 고쳤고(해당 줄에 "(2026-10-18 수정)"), 나머지 7개는 역사 보고서(`review_followup3/CLAIM_EVIDENCE_MATRIX.md`, `CP2_REPORT.md`)라 고치지 않고 `MANUSCRIPT_REVISION.md`에 수정안만 적었습니다. 전체 목록: `outputs/review_followup4/DOC_EDITS_APPLIED.csv`.

**B1 — AutoVLA 수치 민감도 (GPU 18.3분, 56 dev 장면)**

| 설정 | suffix 변경 (row-run) | severe 판정 변경 |
|---|---|---|
| 같은 process 반복 | 0 (bit 단위 동일) | 0 |
| batch 형태만 | 8.9% | 0.26% |
| eager attention | 26.4% | 1.2% |
| fp32 연산 | 25.0% | 0.7% |

- RN / Full-GT / RV 대비는 어느 설정에서도 material하지 않았습니다(severe 변화 ≤ 0.036, FDE5 대비 변화 ≤ 0.16 m vs 효과 1.5–1.9 m).
- P1-A self-reference 대비(OC − U, OC_refR − OC, OC_refH − OC)의 0 근처 FDE5 차이는 수치 설정에 따라 부호가 바뀌었습니다 → "수치 구현 변동과 구별되지 않음"으로 제한합니다. +4.37 m와 A− 외삽 초과는 바닥보다 훨씬 커서 유지합니다.
- 정상 수치 변동을 개입 효과에서 빼지 않고 옆에 보고합니다.

**B2 — AutoVLA matched-prefix factorial (새 효과 실험; 사전 등록 552b3ac, 결과 c7d1bba)**

- 설계: 하나의 편집된 prefix(이전 prefix = GT 또는 matched natural; 최신 block = GT / frozen Normal-source / frozen corrected-source)를 만들고 **이후에도 고정 위치에 유지**한 채 suffix를 B = 1로 생성. 모든 cell이 같은 관측·지시·anchor·source bank·디코딩. identity 검사 100%(P1-C 4,160, dev 2,630 unit-anchor), P1-C 입력 검사 2,160/2,160.
- 실패 층 F(P1-C 100 장면 / 711 단위 / 18 drive; severe, Holm):

| 대비 | F | E4 (Hajek, 312 장면 / 24 drive) |
|---|---|---|
| C_GT = GT_NSRC − GT_REF | **+25.3%p** [+20.5, +29.3] | +3.3%p [+1.0, +5.6], Holm 0.051 (α 0.025에서 비유의) |
| C_NAT = NAT_NSRC − NAT_REF | **+25.5%p** [+21.6, +29.0] | **+4.0%p** [+2.7, +5.7], Holm 6.5e-5 |
| INT = C_GT − C_NAT | −0.2%p [−3.8, +3.1] | −0.7%p [−2.2, +0.8] |
| C_PROV = GT_CSRC − GT_NSRC | **−15.8%p** [−20.1, −10.2] | −2.2%p [−4.2, −0.2], Holm 0.13 |

- 해석: 이 단발(one-shot) 일정에서 효과는 **최신 문맥 token의 값**을 따르며, 이전 GT prefix의 존재에 의존하지 않습니다(INT ≈ 0; 이전 값이 실제로 다른 단위에서도 −1.7%p [−5.4, +1.9]). 실패 층 효과는 크고 E4 평균 효과는 작습니다. E4는 t* ≤ 4 영역이며 원래 2,191 장면 추정량이 아닙니다. dev 코호트(A− 49 장면 / 13 drive)도 같은 패턴입니다(C_GT +28.5, C_NAT +28.4, INT +0.1%p).
- 범위: GT cell이 있으므로 **oracle-conditioned diagnostic**입니다(online 복구·배포 개선 아님). 편집은 유지되므로 해제 실험이 아니고, 기존 drive의 새 개입 arm이므로 외부 검증이 아닙니다. 반복 Recent-GT / Reverse-N 결과와 섞지 않습니다. C_PROV는 값의 차이이지 순수 provenance 효과가 아닙니다. 민감도 anchor(n_old = 2)의 P 수준 INT +2.7%p(p = 0.0046)는 사전 등록 검정이 아닌 민감도 결과입니다.

**B3 — OpenVLA 최소 경로 검증: NOT_RUN** (사용자 결정 2026-10-17; LIBERO-Spatial checkpoint가 디스크에 없음; 여유 디스크 약 2.3–3 GB). OpenVLA는 기술적·미검증으로 두고 핵심 증거에서 제외합니다. 20% 재실행 불일치를 "효과 검출 임계값 20%p"로 쓰지 않습니다(`OPENVLA_GATE.md`).

**§6 — π₀-FAST 적합성(feasibility only).** 공식 `pi0_fast_libero` checkpoint는 공개 bucket에 있으나 이 장비에서는 디스크(약 20–25 GB 필요)와 RTX 5090(sm_120)의 JAX 동작 미확인 때문에 지금은 불가합니다. π₀-FAST는 문맥에 action history가 없어 step 간 action-context 개입은 NA입니다(측정된 0이 아님). WorldVLA의 mask 적용/미적용 checkpoint 쌍은 확인되지 않았습니다(`FEASIBILITY.md`).

**재현 ZIP (A5).** RZIP_SUMMARY

**주장 변경 요약** (`CLAIM_CHANGES.md`):
- 유지(좁힘): 실패 층 반복 Recent-GT/Normal-branch 주입 효과(dev 13 drive, P1-C F 18 drive), B2 최신 token 값 효과(INT ≈ 0, oracle-conditioned), 평가 집합 효과 −4.1 / +3.6%p(353 선정 가중, RN 60% A+)와 B2 E4 +4.0 / +3.3%p, GT 없는 +4.37 m와 A−에서만의 외삽 초과, SpatialVLA 생성·TCP 변화와 두 과제 평균 ±10%p(수정 구간).
- 좁힘: 90% → "GT로 바꾸고 유지한 3개 고정 위치"; P1-A self-reference 0 근처 FDE 차이는 수치 바닥과 구별 불가.
- 낮춤: Impromptu, OpenVLA → 기술적·미검증, 핵심 증거 제외.
- 삭제: 해제·일시 교정·자기 token만 참조 주장, "GT history가 필요", 순수 provenance, online 교정, 외부 재현, OpenVLA 과제별 결론과 C − N "CI 0 제외", 일반 VLA·보편 원인·모델 간 구조 순위·π₀ vs π₀-FAST 구조 효과.
