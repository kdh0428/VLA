# Driving VLA — perception-to-action misalignment & action-token error amplification

AutoVLA(Qwen2.5-VL-3B / nuPlan·navsim)와 ORION(Bench2Drive)에서
**인식은 맞는데 행동이 틀어지는 현상**과 **작은 action-token 편차가 궤적 실패로 증폭되는 기전**을 분석한 PoC입니다.

> **새 서버에서 이어서 실험하려면 [HANDOFF.md](HANDOFF.md)** 부터 보세요 (환경 구축, 함정, 다음 실험 후보).
>
> **결론 요약은 [CONCLUSIONS.md](CONCLUSIONS.md)** 를 보세요. 실험 24건의 질문·수치·해석이 한 문서에 정리돼 있습니다.

## 구조

```
autovla_misalignment_poc/     AutoVLA 실험 (주 분석)
  scripts/                    실험 및 분석 스크립트
  outputs/<실험명>/
    *.md                      실험 보고서 (결론·표·통계)
    figures/*.png             figure
    summary.json              집계 결과
    run_meta.json             실행 메타데이터 (GPU, seed, 조건)
    narrative.json            보고서 서술 원문
vla_misalignment_poc/         ORION 실험
pra_comparison/               ORION vs AutoVLA P/R/A 실패 분해 (동일 라벨링 함수)
tools/, reports/              보조 스크립트 및 문서
```

주요 실험 보고서:

| 실험 | 보고서 |
|---|---|
| P/R/A 실패 분해 | `pra_comparison/outputs/PRA_COMPARISON.md` |
| CoT 인과 개입 | `autovla_misalignment_poc/outputs/cot_intervention/COT_INTERVENTION.md` |
| Fast vs CoT (무편향 500장면) | `.../fast_vs_cot_unbiased/FAST_VS_COT.md` |
| Natural/Fast 기전 재검증 | `.../natural_fast_mechanism/NATURAL_FAST_MECHANISM.md` |
| 첫 mismatch 인과 | `.../first_mismatch_causal/FIRST_MISMATCH_CAUSAL.md` |
| 등거리 perturbation | `.../equal_distance_perturbation/EQUAL_DISTANCE.md` |
| Action history 인과 | `.../action_history_causal/ACTION_HISTORY_CAUSAL.md` |
| Layer-wise state patching | `.../prev_action_state_patching/PREV_ACTION_STATE_PATCHING.md` |
| Temporal feedback window | `.../temporal_feedback_window/TEMPORAL_FEEDBACK_WINDOW.md` |
| Horizon-controlled window | `.../horizon_controlled_window/HORIZON_CONTROLLED_WINDOW.md` |
| 직전 토큰 identity 분해 (motion vs embedding) | `.../prev_action_identity_decomposition/PREV_ACTION_IDENTITY.md` |
| 비-oracle 참조 안정화 (CTRA, 이전 계획) | `.../reference_stabilization/REFERENCE_STABILIZATION.md` |
| Receding-horizon 재계획 (pseudo closed loop) | `.../receding_horizon_replanning/RECEDING_HORIZON.md` |
| 합의 참조 (여러 과거 계획) | `.../consensus_reference_stabilization/CONSENSUS_REFERENCE.md` |
| 배포 조건 안정화 (교정 시점 없음) | `.../natural_reference_stabilization/NATURAL_REFERENCE.md` |
| PDM-Closed 참조 | `.../pdm_reference_stabilization/PDM_REFERENCE.md` |
| seed·온도 강건성 | `.../robustness_reference_stabilization/ROBUSTNESS.md` |
| Best-of-N 선택 (N 8, T 0.7) | `.../best_of_n_selection/BEST_OF_N.md` |
| Best-of-N 선택 (N 16–32, T 1.0–1.3, 5 seed) | `.../best_of_n_selection_n16_T1/BEST_OF_N_16.md` |
| 표본 확대: 새 log 56개에서 best-of-N 확인 | `.../expanded_best_of_n/EXPANDED_BEST_OF_N.md` |
| 검증: PDM Score, 기전 연결, 후보 수, 안전 필터 (사전 등록 held-out) | `.../selection_validation/SELECTION_VALIDATION.md` |

## 저장소에 포함하지 않은 것

용량 때문에 제외했으며, 스크립트로 재생성됩니다.

- 원시 rollout 데이터: `records.jsonl`, `units.jsonl` (실험당 수십~수백 MB)
- 저장된 activation tensor: `outputs/full_extract/tensors/*.npz`
- 데이터셋·체크포인트: nuPlan/navsim 로그와 센서 데이터, AutoVLA/ORION 모델 가중치, 두 모델의 원본 repo clone

## 실행 환경

- AutoVLA: conda env `autovla`, transformers 4.49, 체크포인트 `AutoVLA_PDMS_89.ckpt`, 데이터셋 `dataset/nuplan/navtest_poc`
- GPU 사용 실험은 모두 `CUDA_VISIBLE_DEVICES=1` (RTX 5090) 고정
- 디코딩 설정은 원본 config(`qwen2.5-vl-3B-nuplan-grpo-cot.yaml`)의 `inference.sample`을 그대로 사용 (do_sample, T=0.01)

실행 순서는 `CONCLUSIONS.md`의 "재현" 절에 있습니다.
