# 실험 31 (P4): 불안정 rollout 탐지 baseline — 실행 전 작성 (2026-10-03)

데이터: 실험 20–24의 5090 디코딩 후보(새 샘플링 없음).
dev = `expanded_best_of_n/gpu1` + `expanded_best_of_n_5090/gpu1` (56 log), held-out = `heldout_best_of_n/gpu1` + `heldout_best_of_n_5090/gpu1` (52 log).
모집단: GT token에서 t* < 9에 처음 벗어나는 후보(실험 22와 같음). 라벨: 증폭(A− and FDE5 > 3 m).

추가 GPU 작업(RTX 5090): `teacher_force_candidates.py` — 같은 seed로 stub을 재생성하고 기록된 후보 token을 teacher forcing해
step별 top-1/top-2 확률 margin과 hidden state(layer 18, 36)를 얻습니다(기록된 log-prob와 일치 여부를 함께 기록).

탐지기와 방향(미리 고정, 튜닝 없음). pre = t*까지(포함), post = t*+1 .. 9:
entropy(높을수록 불안정), log-likelihood(낮을수록), margin(낮을수록), 후보 궤적 분산(높을수록, 장면 단위),
pairwise disagreement(이 후보와 나머지 16개의 평균 거리), medoid 거리, hidden-state logistic / ridge / MLP probe.
probe는 dev에서만 학습(layer와 정규화는 log 단위 3-fold CV로 dev에서 선택) 후 held-out에 한 번 적용합니다.

지표(held-out 주, dev 참고): pooled AUROC·AUPRC(log 단위 bootstrap 95% CI, 1,000회), 장면 내 AUROC(양쪽 클래스가 있는 장면 평균).
비교 기준: 실험 22의 entropy pre-deviation 장면 내 AUROC ≈ 0.51, rank-sum 장면 내 ≈ 0.85–0.87 / post-deviation entropy ≈ 0.77–0.79.
핵심 질문: 불안정 rollout은 이탈 전에 예측 가능한가, 이탈 이후에만 구별 가능한가.
