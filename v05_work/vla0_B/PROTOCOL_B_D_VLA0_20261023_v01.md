# PROTOCOL B/D — VLA-0 repetition-penalty / donor control and bridge regeneration (pre-specified)
작성: 2026-10-23 (신규 GPU 출력 생성 전에 고정; 파일 SHA-256은 PROTOCOL_HASHES_20261023.json에 기록)

## 공통
- 모델: ankgoyal/vla0-libero@6c62a511 model_last (bf16, B=1, sdpa), 코드 NVlabs/vla0@b78db19, 기존 adapter `vla0_work/scripts/vla0_adapter.py`(변경 없음, 같은 ForcePrefix 강제 디코딩 경로).
- checkpoint generation_config.json 확인: repetition_penalty = 1.05 (native), temperature 1e-6, do_sample true → native policy는 generate_temperature=0으로 greedy.
- penalty 조건 p ∈ {1.05 (native), 1.0}. penalty는 `model.generation_config.repetition_penalty`로만 설정(native `QwenActor.forward`와 같은 경로). arm별로 다른 penalty를 쓰지 않음. 조건 순서: 관측별로 p=1.05 → p=1.0, arm 순서 SELF, O, L, OL. B=1, batching 없음.
- 패널: 기존 pilot records(`vla0_work/run/pilot/records.jsonl`)의 status OK 149 observations / 30 LIBERO tasks. 이 입력은 기존 결과를 아는 재분석 표본이다(독립 확인 아님).
- 고정 입력: 저장된 관측 npz, 기존 자연 reference의 처음 3 vectors(21 integers) prefix, 기존 donor bank(records의 donors[2], donors[3]). 입력 hash(rgb_sha, prefix ids sha, donor vector sha)를 기록하고 기존 기록과 일치 확인.
- SELF = 자연 prefix를 conditioning copy로 강제(편집 없음). O = vector 2 교체, L = vector 3 교체, OL = 둘 다. suffix = vectors 4–8 (35 integers).
- response D (기존과 동일): arm suffix와 같은 penalty의 SELF suffix의 6 motion dims 평균 절대차 / 1000. A_L, A_O, A_L−A_O; OL은 기술적.
- 집계: 관측 pooled mean, task-cluster bootstrap 10,000 (percentile 95%), seed 20261023. penalty 간 paired contrast: Δ(L−O)_{1.05} − Δ(L−O)_{1.0}, 같은 관측 단위 paired, task-cluster bootstrap.
- 부분집합(기존 정의 유지): matched source distance (0.8 ≤ d_L/d_O ≤ 1.25, d = 6 motion dims 평균 |Δ|/1000), same prefix token sets (R/O/L prefix token set 동일). 보조 결과로만 보고.

## B baseline 허용 오차 (출력 보기 전 고정)
- p=1.05 replay는 기존 기록과 token-exact가 기대값. 허용 오차: arm ids 완전 일치. 불일치 시 분류: (i) prefix/donor/input hash 불일치, (ii) 결정성(같은 세션 SELF 2회 불일치), (iii) 하드웨어/커널 차이에 의한 bf16 tie(첫 불일치 토큰의 top-2 logit gap < 0.125), (iv) 기타. 불일치 관측은 기존 값과 평균하지 않고 두 값을 별도 보고하며, 이번 세션 값으로 penalty 비교를 한다(같은 세션 paired).

## B validation (설정마다; 패널의 obs_id 정렬 처음 8개 관측)
1. native 일치: 해당 p에서 `QwenActor.forward(get_action, T=0)` text == adapter 자유 생성 text.
2. SELF parity: 해당 p 자유 생성의 처음 21 integers를 강제 prefix로 넣은 연속 == 해당 p 자유 생성의 나머지(token/EOS). 추가로 자유 생성 prefix가 기존 자연 prefix와 같으면 고정 prefix SELF와도 비교.
3. 결정성: 고정 prefix SELF를 2회 실행해 token 일치.
4. cache/processor: 각 generate 호출은 새 cache(HF 기본 DynamicCache, 호출 간 공유 없음). effective repetition_penalty 값과 logits processor 목록을 기록.
5. 입력 hash 일치.
- 짧은 진단 subset(같은 8개 관측, SELF·L arm): suffix 처음 8 토큰에 대해 processor 전 raw logits top-10, processor 후 top-10, 선택 토큰의 raw/processed 값, mask(-inf) 개수, penalty history(이전 토큰 id 집합 크기, 선택 후보 중 penalty를 받은 토큰 여부) 저장.
- gate: 1–5 모두 PASS여야 본 실행 결과를 primary로 사용. 실패 시 원인 수정 후 수정 이후 자료를 구분.

## B4 donor follow-up (별도 연구, prospective)
- 관측마다 p=1.05 native decoder, T=0.7로 batch 8개씩 최대 8 batch(64 samples). seed = sha256("vla0_v05_donor:"+obs_id+":"+b)[:4] little-endian.
- 후보는 생성 순서(batch, index)대로 검사하고 출력 반응을 보지 않고 첫 joint-valid candidate를 선택:
  complete 56 integers, in range, EOS; candidate vector 2 ≠ 자연 vector 2, vector 3 ≠ 자연 vector 3; O, L, OL 각각의 편집 prefix가 기존 길이·외부 span 동일 조건 만족; d_O > 0, d_L > 0, 0.8 ≤ d_L/d_O ≤ 1.25.
- 찾지 못하면 SOURCE_MISSING. hard cap 이후 추가 탐색·재선택 없음. 수용 후보를 찾은 batch에서 탐색 중단.
- 선택된 donor로 O, L, OL arm을 두 penalty에서 실행(SELF는 B main 결과 재사용, 같은 세션·같은 p).
- 보고: 수용률, 탐색 sample 수, d_O/d_L, token Hamming/편집량, token-set 변화 유형(R/O/L token set 동일 여부), 성분별 translation/rotation 변화, 제외 특성.

## D bridge 보조 분석 (별도 보조, p=1.05 native만)
- 패널: 30 tasks에서 task당 기존 OK 관측 중 obs_id 정렬 첫 관측(donor 2 유효). 무효면 다음 ID.
- bridge-fixed = 기존 O arm(vector 2 교체, vector 3 자연값 고정; B main p=1.05 결과 재사용).
- bridge-regenerated = vectors 1 자연 + vector 2 donor (14 integers)만 강제하고 vector 3부터 모델이 다시 생성. suffix 비교 범위는 동일한 vectors 4–8.
- parity: vectors 1–2 자연값 14 integers 강제 연속이 기존 reference의 vector 3–8과 token 일치해야 함(BRIDGE_SELF). 실패 관측은 BLOCKED로 분리.
- 저장: 재생성 vector 3, 자연 vector 3과의 변화량(6 dims |Δ|/1000), suffix 반응 D(동일 정의). bridge-regenerated−bridge-fixed paired 차이와 task-cluster bootstrap(30 tasks, 1 obs/task → 관측 bootstrap과 동일).
- 해석 제한: bridge-regenerated는 변경된 conditioning 값의 수·경로가 다르므로 같은 dose의 L/O 비교나 자연 간접효과 식별이 아니다. 주 분석과 pooling하지 않음.

## GPU 상한 (이번 세션 자체 설정, 출력 전 고정)
- GPU: physical GPU 1 (RTX 5090), CUDA_DEVICE_ORDER=PCI_BUS_ID. 실행 순서 validate → main → D(bridge) → B4(donor). 누적 상한: main 종료 ≤ 120, D 종료 ≤ 135, B4 종료 ≤ 225 GPU-min (B4 ≤ 약 90). 초과 시 진행 중 관측을 마치고 PARTIAL로 기록(관측 순서 = obs_id 정렬, 축소 시 해당 시점까지의 관측만 보고).
