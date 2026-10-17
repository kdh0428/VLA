# CP2 결정 (2026-10-15, 사용자)

- **A5:** 재검증한다. SpatialVLA executor fingerprint를 policy state와 shadow-cf 장부 기록으로 분리해 hash하고, 같은 20 단위에서 null arm만 다시 실행한다(약 40 에피소드, RTX 5090). 판정 규칙: 분리한 policy-state hash, 실행 action, 시뮬레이터 상태가 bit 단위로 같아야 PASS(허용 오차 0, 사전 기준과 같음). 장부 기록은 보고만 한다. 원래의 엄격 기준 결과(0/20)와 A5 사후 검사(20/20)는 그대로 보존한다.
- **Stage 3:** O0만 실행한다. 실험 32 DM 스크립트와 P1-B 시작 위치 스크립트를 AutoVLA parity 관문 manifest에 추가하고, 기존 PROTOCOL_P1의 대조·허용 오차·판정 규칙을 그대로 적용한다.
- **미실행 (주장 축소로 대응):** O1 P3 AutoVLA, O2 P3 Impromptu, O3 P4. REVIEW_COVERAGE와 STATUS에 "미해결 — 주장 축소로 대응"으로 기록한다.
- **P4 정밀도 목표:** 실행한다면 과제별 ±10%p로 승인한다. 이번에는 실행하지 않는다.
