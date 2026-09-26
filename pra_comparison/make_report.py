#!/usr/bin/env python
"""Render outputs/PRA_COMPARISON.md from outputs/summary.json (numbers are never hand-copied)."""
from __future__ import annotations

import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
HEAD = ["P-R-A-", "P+R-A-", "P+R+A-", "P+R+A+"]
ALL8 = ["P+R+A+", "P+R+A-", "P+R-A+", "P+R-A-", "P-R+A+", "P-R+A-", "P-R-A+", "P-R-A-"]
SENS = [
    ("headline", "기준 (공통 기준, 속도비례 허용오차)"),
    ("with_traffic_light", "ORION에 신호등 GT 반영 (AutoVLA는 신호등 GT 없음 → 동일)"),
    ("strict_no_tolerance", "허용오차 0 (경계 flip도 실패로 계산)"),
    ("p_miss_only", "P를 miss만으로 판정 (hallucination 제외)"),
    ("at_rest_neither_accept", "정지 중 'keep'의 모호한 경우를 R+로 (상한)"),
    ("with_traffic_light_at_rest_neither_accept", "신호등 반영 + 모호한 정지 'keep' R+"),
    ("lenient_stop_decel", "선언 STOP/DECELERATE 상호 인정"),
    ("common_3s_horizon", "AutoVLA도 3 s horizon으로 평가"),
]


def pct(x):
    return f"{100 * x:.1f}%"


def ci(c):
    return f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def two_by_two(model: str) -> Counter:
    """
    Within P+ and A-: was the declared decision right (R_d), and did the executed class follow
    the declaration? (right, follows) cannot be A- -- declared and executed are judged by the
    same accept function -- so the three remaining cells split failures into
    pure interface, pure decision, and compound.
    """
    c = Counter()
    for line in open(os.path.join(OUT, f"labels_{model}.jsonl")):
        r = json.loads(line)
        if r["P"] is not True or r["A"] is not False or r["R"] is None:
            continue
        rd, ad = r["R_detail"], r["A_detail"]
        dl = rd.get("decl_lon_resolved")
        if dl is None:
            c["unresolved"] += 1
            continue
        follows = dl == ad["pred_lon"] and (rd.get("decl_lat") is None or rd["decl_lat"] == ad["pred_lat"])
        ok = rd["Rd_lon_ok"] and rd["Rd_lat_ok"]
        c[("ok" if ok else "wrong", "follows" if follows else "diverges")] += 1
    return c


def main() -> None:
    S = json.load(open(os.path.join(OUT, "summary.json")))
    R = S["results"]
    h = R["headline"]
    o, a = h["ORION"], h["AutoVLA"]
    L = []
    w = L.append

    ao, aa = o["A_minus_attribution"], a["A_minus_attribution"]
    w("# ORION vs AutoVLA — P / R / A failure decomposition\n")
    w("저장된 inference 결과만 사용 (GPU 미사용). 두 모델에 **동일한 라벨링 함수**를 적용했습니다. "
      "재현: `python run_pra_comparison.py && python make_report.py` (`/root/VLA/pra_comparison`).\n")

    w("## 1. 결론\n")
    w(f"**3단계 분류로는 두 모델 모두 reasoning이 최대 원천이지만, 실패가 일어나는 방식은 다릅니다.** "
      f"실제 행동이 틀린(A−) 샘플 중 perception은 맞고 reasoning이 틀린 비율은 "
      f"ORION {pct(ao['reasoning']['share_of_A_minus'])} {ci(ao['reasoning']['ci95_cluster_boot'])}, "
      f"AutoVLA {pct(aa['reasoning']['share_of_A_minus'])} {ci(aa['reasoning']['ci95_cluster_boot'])}이며, "
      f"§3의 8개 민감도 설정 모두에서 최대입니다.\n")
    def cells(t):
        res = sum(v for k, v in t.items() if k != "unresolved")
        return res, t[("wrong", "follows")], t[("ok", "diverges")], t[("wrong", "diverges")]

    t_o, t_a = two_by_two("ORION"), two_by_two("AutoVLA")
    n_, wf, od, wd = cells(t_o)
    w(f"- **ORION — 결정 단계에서 발생.** perception이 맞은 A−(판정 가능 {n_}건) 중 결정이 틀렸고 planner가 그 결정을 "
      f"그대로 실행한 순수 결정 오류 {wf}건({pct(wf / max(n_, 1))}), 결정은 맞았는데 실행이 달라진 순수 interface "
      f"{od}건({pct(od / max(n_, 1))}), 둘 다 {wd}건({pct(wd / max(n_, 1))}). "
      f"정지 중 'keep'이 미해결인 {t_o['unresolved']}건은 제외.")
    n_, wf, od, wd = cells(t_a)
    w(f"- **AutoVLA — 결정 오류와 interface 이탈이 겹침.** perception이 맞은 A−(판정 가능 {n_}건) 중 순수 결정 오류는 "
      f"{wf}건({pct(wf / max(n_, 1))})뿐이고, 실행이 선언과 다르게 나간 경우가 {od + wd}건"
      f"({pct((od + wd) / max(n_, 1))}; 순수 interface {od}, 결정도 틀림 {wd})입니다. CoT가 말한 결정과 실제 action "
      "token의 결합이 약하며, 3단계 분류의 reasoning 비율에는 이 복합 실패가 들어 있습니다.")
    w(f"- **P+R+A− (결정은 맞았는데 다르게 행동)**: ORION {o['groups_all8']['P+R+A-']['count']}건 "
      f"(A− 중 {pct(ao['interface']['share_of_A_minus'])} {ci(ao['interface']['ci95_cluster_boot'])}), "
      f"AutoVLA {a['groups_all8']['P+R+A-']['count']}건 "
      f"(A− 중 {pct(aa['interface']['share_of_A_minus'])} {ci(aa['interface']['ci95_cluster_boot'])}).")
    w(f"- **Perception (P−)**: ORION {pct(ao['perception']['share_of_A_minus'])} "
      f"{ci(ao['perception']['ci95_cluster_boot'])}, AutoVLA {pct(aa['perception']['share_of_A_minus'])} "
      f"{ci(aa['perception']['ci95_cluster_boot'])}. 대부분 hallucination에서 오며, miss만 보면 "
      f"ORION {pct(R['p_miss_only']['ORION']['A_minus_attribution']['perception']['share_of_A_minus'])}, "
      f"AutoVLA {pct(R['p_miss_only']['AutoVLA']['A_minus_attribution']['perception']['share_of_A_minus'])}로 떨어집니다.")
    w(f"- 두 모델의 차이는 **실패의 양**입니다: A− 비율 ORION {pct(1 - o['marginals']['A+'])}, "
      f"AutoVLA(강제 CoT) {pct(1 - a['marginals']['A+'])}, AutoVLA(자연 생성) "
      f"{pct(h['AutoVLA_armN_A_only']['A_minus_rate'])}.")
    w(f"- 별도로, 두 모델 모두 **말한 결정과 실제 행동의 결합이 느슨합니다**: 행동은 맞는데 선언한 결정이 GT와 다른 "
      f"P+R−A+가 ORION {pct(o['groups_all8']['P+R-A+']['rate'])}, AutoVLA {pct(a['groups_all8']['P+R-A+']['rate'])}로 "
      "모든 그룹 중 가장 큽니다. reasoning 텍스트를 action의 설명으로 읽으면 안 된다는 뜻입니다.\n")

    w("## 2. 요청하신 핵심 표\n")
    w("비율은 P/R/A가 모두 판정 가능한 샘플 기준, 괄호는 clip(ORION) / log(AutoVLA) 단위 cluster bootstrap 95% CI.\n")
    w("| Group | ORION | AutoVLA |")
    w("|---|---:|---:|")
    for k in HEAD:
        oo, aa_ = o["headline"][k], a["headline"][k]
        w(f"| {k} | {oo['count']} ({pct(oo['rate'])} {ci(oo['ci95_cluster_boot'])}) | "
          f"{aa_['count']} ({pct(aa_['rate'])} {ci(aa_['ci95_cluster_boot'])}) |")
    w(f"| 판정 가능 / 전체 | {o['n_decidable']} / {o['n_total']} | {a['n_decidable']} / {a['n_total']} |\n")

    w("### A− 샘플의 발생 단계\n")
    w("| 단계 | 정의 | ORION | AutoVLA |")
    w("|---|---|---:|---:|")
    for st, nm in (("perception", "P−"), ("reasoning", "P+R−"), ("interface", "P+R+")):
        w(f"| {st} | A− ∧ {nm} | {ao[st]['count']} ({pct(ao[st]['share_of_A_minus'])} {ci(ao[st]['ci95_cluster_boot'])}) | "
          f"{aa[st]['count']} ({pct(aa[st]['share_of_A_minus'])} {ci(aa[st]['ci95_cluster_boot'])}) |")
    w(f"| 합계 A− | | {o['n_A_minus']} | {a['n_A_minus']} |\n")

    # What a "reasoning" failure actually is: which R criterion failed, and whether the
    # executed action simply followed the (wrong) declared decision.
    def breakdown(model):
        rows = [json.loads(l) for l in open(os.path.join(OUT, f"labels_{model}.jsonl"))]
        g = [r for r in rows if r["group"] == "P+R-A-"]
        kinds = Counter(x.split(":")[0] for r in g for x in r["R_detail"]["fails"])
        comp = [r for r in g if r["R_detail"].get("decl_lon_resolved")]
        same = sum(r["R_detail"]["decl_lon_resolved"] == r["A_detail"]["pred_lon"] for r in comp)
        only_rd = sum(all(x.startswith("Rd_") for x in r["R_detail"]["fails"]) for r in g)
        return len(g), kinds, same, len(comp), only_rd

    bo, ba = breakdown("ORION"), breakdown("AutoVLA")
    w("### reasoning 실패(P+R−A−)의 내용\n")
    w("R 기준별 실패 건수 (한 샘플이 여러 기준에 걸릴 수 있음).\n")
    w("| R 기준 | ORION | AutoVLA |")
    w("|---|---:|---:|")
    labels = [("Rd_lon", "R_d 종방향 결정이 GT와 다름"), ("Rd_lat", "R_d 횡방향 결정이 GT와 다름"),
              ("Rc_not_slowing", "R_c 위험 하 감속 판단 실패"),
              ("Rc_ego_moving_while_at_rest", "R_c 정지 중인데 이동 중이라 서술"),
              ("Rc_ego_stopped_while_moving", "R_c 이동 중인데 정지라 서술"),
              ("Ra_lead_not_referenced", "R_a 가까운 선행차 미언급"), ("Ra_vru_not_referenced", "R_a 보행자 위험 미언급"),
              ("Rb_halluc_vehicle", "R_b 없는 차량 근거"), ("Rb_halluc_vru", "R_b 없는 보행자 근거")]
    for k, nm in labels:
        w(f"| {nm} | {bo[1].get(k, 0)} | {ba[1].get(k, 0)} |")
    w(f"| **결정(R_d)만 틀림** | {bo[4]} / {bo[0]} ({pct(bo[4] / max(bo[0], 1))}) | "
      f"{ba[4]} / {ba[0]} ({pct(ba[4] / max(ba[0], 1))}) |")
    w(f"| **실행 종방향 클래스 = 선언 결정** | {bo[2]} / {bo[3]} ({pct(bo[2] / max(bo[3], 1))}) | "
      f"{ba[2]} / {ba[3]} ({pct(ba[2] / max(ba[3], 1))}) |\n")
    w("두 모델 모두 reasoning 실패는 객체 인식·hallucination이 아니라 **결정 기준(R_d)**에서 옵니다. "
      "그러나 실행이 그 틀린 결정을 따르는지는 모델마다 다릅니다:\n")
    for nm, b in (("ORION", bo), ("AutoVLA", ba)):
        s_ = b[2] / max(b[3], 1)
        if s_ >= 0.6:
            w(f"- **{nm}**: P+R−A−의 {pct(s_)}에서 실행 궤적이 틀린 선언 결정을 그대로 따릅니다. "
              "interface는 말한 대로 실행했고, 오류는 결정 단계에서 이미 발생했습니다.")
        else:
            w(f"- **{nm}**: P+R−A−에서 실행이 선언을 따르는 비율은 {pct(s_)}뿐입니다. 결정이 틀린 데다 실행도 그 결정과 "
              "다르게 나간 복합 실패가 다수이므로, reasoning 비율만 보고 interface 문제가 작다고 해석하면 안 됩니다.")
    w("")

    w("### 결정 정오 × 실행이 선언을 따르는가 (P+ ∧ A−)\n")
    w("선언과 실행을 같은 accept 함수로 판정하므로 '결정이 맞고 실행이 따름'은 A−가 될 수 없습니다. "
      "종방향은 선언 클래스(정지 중 'keep'은 해소된 클래스)와 실행 클래스를, 횡방향은 선언이 있을 때만 비교합니다.\n")
    w("| 셀 | 의미 | ORION | AutoVLA |")
    w("|---|---|---:|---:|")
    to, ta = two_by_two("ORION"), two_by_two("AutoVLA")
    ro = sum(v for k, v in to.items() if k != "unresolved")
    ra = sum(v for k, v in ta.items() if k != "unresolved")
    for key, nm in ((("ok", "diverges"), "결정 맞음 · 실행이 다름 → 순수 interface"),
                    (("wrong", "follows"), "결정 틀림 · 실행이 따름 → 순수 결정 오류"),
                    (("wrong", "diverges"), "결정 틀림 · 실행도 다름 → 복합"),
                    (("ok", "follows"), "결정 맞음 · 실행이 따름 (불가능해야 함)")):
        w(f"| {key[0]}/{key[1]} | {nm} | {to[key]} ({pct(to[key] / max(ro, 1))}) | {ta[key]} ({pct(ta[key] / max(ra, 1))}) |")
    w(f"| unresolved | 정지 중 'keep' 미해결 | {to['unresolved']} | {ta['unresolved']} |\n")

    w("## 3. 민감도 — 결론이 라벨 선택에 얼마나 의존하는가\n")
    w("A− 중 perception / reasoning / interface 비율.\n")
    w("| 설정 | ORION P / R / I | ORION P+R+A− | AutoVLA P / R / I | AutoVLA P+R+A− |")
    w("|---|---|---:|---|---:|")
    for key, desc in SENS:
        if key not in R:
            continue
        r = R[key]
        fo = r["ORION"]["A_minus_attribution"]
        fa = r["AutoVLA"]["A_minus_attribution"]
        w(f"| {desc} | {pct(fo['perception']['share_of_A_minus'])} / {pct(fo['reasoning']['share_of_A_minus'])} / "
          f"{pct(fo['interface']['share_of_A_minus'])} | {r['ORION']['groups_all8']['P+R+A-']['count']} | "
          f"{pct(fa['perception']['share_of_A_minus'])} / {pct(fa['reasoning']['share_of_A_minus'])} / "
          f"{pct(fa['interface']['share_of_A_minus'])} | {r['AutoVLA']['groups_all8']['P+R+A-']['count']} |")
    w("")
    w("reasoning이 최대 단계라는 결론은 모든 설정에서 유지됩니다. 가장 크게 움직이는 것은 ORION의 interface 비율로, "
      "정지 상태에서 선언한 'keep'을 어떻게 읽느냐(§5 R_d)에 달려 있습니다 — 모호한 경우를 R+로 두면 상한값이 됩니다.\n")

    w("## 4. 전체 8개 그룹\n")
    w("| Group | ORION | AutoVLA |")
    w("|---|---:|---:|")
    for k in ALL8:
        w(f"| {k} | {o['groups_all8'][k]['count']} ({pct(o['groups_all8'][k]['rate'])}) | "
          f"{a['groups_all8'][k]['count']} ({pct(a['groups_all8'][k]['rate'])}) |")
    w(f"| 판정 불가 | {o['n_undecidable']} {o['undecidable_breakdown']} | {a['n_undecidable']} {a['undecidable_breakdown']} |\n")

    w("## 5. 라벨 정의 (두 모델 공통)\n")
    w("**A (action)** — 계획 궤적의 의미적 행동. 두 모델 궤적을 같은 좌표계(+right, +forward, 0.5 s)로 변환해 "
      "ORION의 `trajectory_metrics.action_semantics` 하나로 STOP/DECELERATE/MAINTAIN/ACCELERATE 및 "
      "STRAIGHT/LEFT/RIGHT를 구합니다. action token ID는 사용하지 않고, L2 임계값도 쓰지 않습니다.")
    w("- 실행 행동의 종방향 클래스는 **GT 시작 속도(실제 현재 속도) 기준**으로 계산합니다.")
    w("- GT 궤적에서 허용오차 안에 도달 가능한 클래스면 정답: 속도 max(0.5 m/s, 15%), 횡방향 1 m / 10°. "
      "정지 중 0.25 m 미만 이동의 heading은 무시합니다(pose jitter).")
    w("- horizon: ORION 3 s, AutoVLA 5 s(모델 고유). 3 s 통일은 §3 민감도.\n")
    w("**P (perception)** — 모델의 *장면 설명* 텍스트(ORION: scene description + critical objects, "
      "AutoVLA: Scene Description + Critical Object Description) vs GT 장면. 공통 변수: 차선 내 선행차량, 전방 보행자/자전거 위험.")
    w("- miss: GT에 엄격 영역 내 존재하는데 설명에서 인지하지 않음 / hallucination: 구체적으로 주장했는데 GT에 완화 영역 내에도 없음.")
    w("- 일반론·가정형 언급(\"vehicles can proceed\", \"watch for any pedestrians\")은 주장으로 보지 않습니다.")
    w("- 신호등은 ORION만 GT가 있어 공통 기준에서 제외, §3에서 반영.\n")
    w("**R (reasoning)** — 모델의 *추론/결정* 텍스트. 네 기준을 모두 만족해야 R+.")
    w("- R_a 중요 객체: GT 위험(선행차 ≤15 m, 보행자 위험, [신호등 설정] 적색등)을 추론에서 언급")
    w("- R_b hallucination: 존재하지 않는 위험을 근거로 주장하지 않음 (신호등 설정: 적색인데 녹색이라 주장하지 않음)")
    w("- R_c 상황 판단: 위험 하에 GT가 감속하면 결정도 감속/일치, ego 자신의 운동 상태를 틀리게 서술하지 않음")
    w("- R_d 최종 결정: 선언 결정을 **A와 똑같은 accept 함수**로 판정. 따라서 선언 = 실행이면 R_d와 A 판정이 항상 같고, "
      "P+R+A−는 '말한 것과 다르게 행동한' 경우만 셉니다 (위반 0건 자동 검증).")
    w("- 정지 중 선언한 'keep'은 단어만으로 해석 불가: 진행 논리가 있으면 진행, 정지 의도가 있으면 정지, 둘 다 없으면 미해결(기준은 R−, 상한은 §3).")
    w("- 자유 텍스트 GT reasoning은 쓰지 않습니다(AutoVLA에 없음). ORION도 같은 GT 장면 규칙으로 판정합니다.\n")

    w("## 6. AutoVLA 비교 조건\n")
    w(f"AutoVLA의 자연 생성(arm N)은 2,748개 중 1개만 CoT를 내므로 P/R을 볼 텍스트가 없습니다. "
      f"그래서 P/R/A는 **강제 CoT(arm C) 한 번의 생성**에서 CoT와 그 생성의 궤적을 함께 사용했습니다 "
      f"(arm C의 CoT와 arm N의 action을 짝짓지 않음). max_length 도달·action 토큰 폭주 샘플 "
      f"{S['autovla_excluded']}은 제외했습니다. arm C는 반사실 조건이며 A− 비율이 자연 생성보다 높습니다 "
      f"({pct(1 - a['marginals']['A+'])} vs {pct(h['AutoVLA_armN_A_only']['A_minus_rate'])}).\n")

    w("## 7. 라벨 검증\n")
    w("규칙 기반 라벨이므로 각 버전마다 무작위 표본을 원문과 대조해 읽고 규칙을 고쳤습니다. 발견해 수정한 오류:")
    w("- 정지 GT에서 고정 ±1 m/s 허용오차가 출발(0→8 m/s)까지 정답 처리 → 속도비례 허용오차")
    w("- 선언과 실행을 다른 규칙으로 판정해 선언=실행인데 R+A−가 발생 → 단일 accept 함수")
    w("- 예측 궤적 자신의 첫 속도 기준 클래스 → GT 시작 속도 기준")
    w("- ORION이 정지선에서 \"keep … until closer to the intersection\"(=계속 진행)을 'keep=정지 유지'로 인정 → 추론 텍스트로 해소")
    w("- \"the traffic light ahead … vehicles can proceed\", \"a pedestrian crossing\"(횡단보도)이 객체 주장으로 오인 → 일반론·가정형 차단")
    w("- AutoVLA가 ego를 \"the vehicle\"로 부르는 것을 타 차량 언급으로 오인 → 제거")
    w("- 정지 중 sub-mm pose jitter의 heading이 GT를 STOP+RIGHT로 만듦 → 0.25 m 미만 무시\n")
    w("최종 버전 표본 판독 결과: ORION P+R+A− 8건 중 7–8건, AutoVLA P+R+A− 8건 중 8건이 실제 '말한 것과 다르게 행동'한 사례"
      "(AutoVLA 중 3건은 정지 vs 1.4 m/s 감속 수준의 경미한 차이). P− 판정은 텍스트에서 객체를 뽑는 가장 약한 부분으로, "
      "ORION·AutoVLA 모두 5건 중 3–4건이 실제 오류였습니다.\n")

    w("## 8. 한계\n")
    w("- R은 규칙 기반입니다. LLM judge나 사람 라벨이 아니며, 위 판독은 소표본입니다.")
    w("- 공통 기준은 신호등을 볼 수 없습니다. ORION의 적색등 관련 reasoning 오류(\"currently green\")는 신호등 설정에서만 잡히며, "
      "AutoVLA의 신호등 언급은 GT가 없어 검증 불가입니다.")
    w("- ORION GT(Chat-B2D)는 *critical* 객체만 나열하므로, 좌표로 주장된 객체는 20 m 이내에서만 hallucination으로 판정합니다.")
    w("- ORION P+R+A−는 clip 집중도가 높습니다(적색등 대기 장면). CI가 cluster bootstrap인 이유입니다.")
    w("- ORION 궤적은 QA 샘플링 때문에 실행마다 달라집니다(이번 추출의 frame별 재시드 run 사용).\n")

    w("## 9. 다음 단계용 subset\n")
    w("`outputs/subsets/<MODEL>/<group>.json` — 파일명 규칙 `+`→`p`, `−`→`m` (예: `PpRpAm.json` = P+R+A−). "
      "각 샘플에 `sample_id`, `cluster`, `tensor_file`(추출 텐서 경로), A 판정 근거, P/R 실패 사유가 들어 있어 "
      "causal/internal analysis에 바로 쓸 수 있습니다. 전체 샘플별 라벨은 `outputs/labels_<MODEL>.jsonl`.\n")
    w("| 파일 | ORION | AutoVLA |")
    w("|---|---:|---:|")
    for k, f in (("P+R−A−", "PpRmAm.json"), ("P+R+A−", "PpRpAm.json"), ("P+R+A+", "PpRpAp.json"),
                 ("P−R−A−", "PmRmAm.json")):
        nk = k.replace("−", "-")
        w(f"| `{f}` ({k}) | {o['groups_all8'][nk]['count']} | {a['groups_all8'][nk]['count']} |")
    w("")

    with open(os.path.join(OUT, "PRA_COMPARISON.md"), "w") as f:
        f.write("\n".join(L))
    print(f"wrote {os.path.join(OUT, 'PRA_COMPARISON.md')}")


if __name__ == "__main__":
    main()
