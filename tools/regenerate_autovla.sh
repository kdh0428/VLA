#!/bin/bash
# 새 서버에서 AutoVLA PoC의 원시 결과(저장소에 없는 records.jsonl / tensors)를 순서대로 재생성합니다.
# 선행: tools/download_autovla_assets.sh, tools/setup_autovla_env.sh
#
#   bash tools/regenerate_autovla.sh            # 전체 (5090 기준 약 9-10시간)
#   STAGES="ed ah exp11" bash tools/regenerate_autovla.sh   # 일부 단계만
#
# 단계와 의존 관계 (앞 단계의 outputs/*/records.jsonl 을 뒤 단계가 읽음):
#   pre     장면 전처리 2748개                              (CPU, 12분)
#   full    full_extract, arm N 텐서만 저장 (~18 GB)         (4.8 h)   <- 모든 GPU 실험의 입력
#   check   git HEAD 수치와 개수 단위 재현 비교               (CPU)
#   ed      equal_distance_perturbation (실험 6)             (6분)     <- 실험 7-14의 단위 정의
#   ah      action_history_causal (실험 7)                  (20분)
#   exp11   prev_action_identity_decomposition             (6분)
#   frames  이전/미래 frame 장면 + 로그 pose                  (CPU, 15분)
#   exp12   reference_stabilization                        (11분)
#   exp13   receding_horizon_replanning                    (2 h)
#   exp14   reference_stabilization, 합의 참조               (30분)
#   pdm     PDM-Closed 참조 궤적 (natural 장면 선택 포함)       (CPU, 15분)
#   exp15   natural_reference_stabilization (배포 조건)        (50분)
#   exp16   reference_stabilization, PDM 참조                (11분)
#   exp17   seed·온도 강건성 5회                              (50분)
#   exp18   best_of_n_selection N 8 T 0.7                  (15분)
#   exp19   best_of_n_selection N·T 곡선 + N 16 T 1.0 seed 0-4 (2.5시간)
#
# GPU: 모든 실험은 CUDA_VISIBLE_DEVICES=1 (RTX 5090, CUDA_DEVICE_ORDER=PCI_BUS_ID)로 고정돼 있습니다.
# 다른 GPU 구성이면 GPU=0 VLA_ANY_GPU=1 로 실행하세요 (수치가 소폭 달라질 수 있음; 한 실험 안의 비교는 쌍대라 유효).
set -uo pipefail
source "$(conda info --base 2>/dev/null || echo /root/miniforge3)/etc/profile.d/conda.sh"
conda activate autovla
export CUDA_VISIBLE_DEVICES=${GPU:-1}
POC=/root/VLA/autovla_misalignment_poc
cd $POC
STAGES=${STAGES:-"pre full check ed ah exp11 frames exp12 exp13 exp14 pdm exp15 exp16 exp17 exp18 exp19"}
step() { echo; echo "######## $(date '+%F %T') $*"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }

has pre    && { step pre;    python scripts/preprocess_scenes.py || exit 1; }
has full   && { step full;   python scripts/full_extract.py --tensor-arms N > outputs/full_extract/run.log 2>&1 || exit 1; tail -1 outputs/full_extract/run.log; }
has check  && { step check;  python scripts/check_full_extract_repro.py; }
has ed     && { step ed;     python scripts/equal_distance_perturbation.py || exit 1; python scripts/analyze_equal_distance.py > /dev/null; }
has ah     && { step ah;     python scripts/action_history_causal.py || exit 1; python scripts/analyze_action_history.py > /dev/null; }
has exp11  && { step exp11;  python scripts/prev_action_identity_decomposition.py || exit 1; python scripts/analyze_prev_action_identity.py; python scripts/make_identity_figures.py; }
has frames && { step frames; python scripts/build_frame_scenes.py || exit 1; }
has exp12  && { step exp12;  python scripts/reference_stabilization.py || exit 1; python scripts/analyze_reference_stabilization.py; }
has exp13  && { step exp13;  python scripts/receding_horizon_replanning.py || exit 1; python scripts/analyze_receding_horizon.py; }
has exp14  && { step exp14;  python scripts/reference_stabilization.py --sources gt,prev,prev2,prev3,consmed,consmean,consmedkin \
                                  --windows w4,all --output outputs/consensus_reference_stabilization || exit 1
                             python scripts/analyze_reference_stabilization.py --run outputs/consensus_reference_stabilization; }
has pdm    && { step pdm;    python - <<'PY'
import json, random
P = "outputs"
recs = [json.loads(l) for l in open(f"{P}/full_extract/records.jsonl")]
am = set(json.load(open(f"{P}/full_extract/repro_check.json"))["arms"]["N"]["aminus_tokens"])
logof = {r["token"]: r["log_name"] for r in recs}
ok = [r["token"] for r in recs if not r["arms"]["N"]["cot_present"] and not r["arms"]["N"]["runaway_action_tokens"]]
rest = sorted(set(ok) - am); random.Random(0).shuffle(rest)
sel = [{"token": t, "log": logof[t], "group": "A-"} for t in sorted(am)] + [{"token": t, "log": logof[t], "group": "random"} for t in rest[:300]]
json.dump(sel, open(f"{P}/frame_index/natural_units.json", "w"))
PY
                             python scripts/build_frame_scenes.py --units outputs/frame_index/natural_units.json --index-name natural_index.json --no-future || exit 1
                             python scripts/pdm_reference.py || exit 1; }
has exp15  && { step exp15;  python scripts/natural_reference_stabilization.py || exit 1; python scripts/analyze_natural_reference_stabilization.py; }
has exp16  && { step exp16;  python scripts/reference_stabilization.py --sources gt,prev,pdm,prevpdm --windows w4,all --output outputs/pdm_reference_stabilization || exit 1
                             python scripts/analyze_reference_stabilization.py --run outputs/pdm_reference_stabilization; }
has exp17  && { step exp17;  for cfg in "0.01 1" "0.01 2" "0.5 0" "0.5 1" "0.5 2"; do set -- $cfg
                                 python scripts/reference_stabilization.py --sources gt,prev --windows w4,all --temperature $1 --seed $2 \
                                     --output outputs/robustness_reference_stabilization/T$1_seed$2 || exit 1
                                 python scripts/analyze_reference_stabilization.py --run outputs/robustness_reference_stabilization/T$1_seed$2; done; }
has exp18  && { step exp18;  python scripts/best_of_n_selection.py || exit 1; python scripts/analyze_best_of_n.py; }
has exp19  && { step exp19;  for cfg in "16 1.0 0 outputs/best_of_n_selection_n16_T1" "16 1.0 1 outputs/best_of_n_selection_n16_T1_seed1" \
                                        "16 1.0 2 outputs/best_of_n_selection_n16_T1_seed2" "32 1.0 0 outputs/best_of_n_selection_n32_T1.0" \
                                        "32 1.3 0 outputs/best_of_n_selection_n32_T1.3" \
                                        "16 1.0 3 outputs/best_of_n_selection_n16_T1_seed3" "16 1.0 4 outputs/best_of_n_selection_n16_T1_seed4" \
                                        "16 0.85 0 outputs/best_of_n_selection_n16_T0.85" "16 1.15 0 outputs/best_of_n_selection_n16_T1.15" \
                                        "4 1.0 0 outputs/best_of_n_selection_n4_T1.0" "8 1.0 0 outputs/best_of_n_selection_n8_T1.0"; do set -- $cfg
                                 python scripts/best_of_n_selection.py --n $1 --temperature $2 --seed $3 --output $4 || exit 1
                                 python scripts/analyze_best_of_n.py --run $4; done
                             python scripts/pool_best_of_n.py outputs/best_of_n_selection_n16_T1 outputs/best_of_n_selection_n16_T1_seed{1,2,3,4}; }
step done
# 참고: 재생성 결과가 저장소의 기존 파일(summary.json, *.md)을 덮어씁니다. 차이는 git diff 로 확인하세요.
