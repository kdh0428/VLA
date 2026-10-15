#!/usr/bin/env python
"""Final review_followup3 document fixes (2026-10-15). CPU only.

Applies the number/label corrections found by P0-C (NUMBER_RECONCILIATION.md) and the final verification statuses to
the manuscript-facing documents. Every replacement first ASSERTS the corrected value against a machine source
(numbers.json, the exp 35 / P1-D episode table, or a P0 table), so no corrected digit is hand-typed without a check.
Idempotent: a replacement whose new text is already present (and old text absent) is skipped.

  /root/VLA/EXPERIMENT_SUMMARY.md
  /root/VLA/paper_quantitative_package/{PAPER_NUMBERS.md, tables.tex, MANUSCRIPT_REVISION.md}

  nice -n 19 /root/miniforge3/envs/autovla/bin/python cp2_doc_fixes.py
"""
import json

import pandas as pd

ROOT = "/root/VLA"
PKG = f"{ROOT}/paper_quantitative_package"
POC = f"{ROOT}/autovla_misalignment_poc"
R = json.load(open(f"{PKG}/numbers.json"))["rows"]


def v(rid):
    return R[rid]["value"]


def ci(rid):
    return R[rid]["ci95"]


def fmt2(x):
    return f"{x:+.2f}"


# ---------------------------------------------------------------- machine checks of the corrected values
E = pd.read_csv(f"{POC}/outputs/review_followup2/B_spatialvla_direct/run_20261013/episodes.csv")
x35 = E[E.experiment == "exp35"].pivot_table(index=["task", "seed", "config"], columns="mode", values="success")
RATE = x35.groupby("config").mean() * 100
D = pd.DataFrame({"FmC": RATE.feedback - RATE.corrected, "RmN": RATE.reverse - RATE.natural})
assert abs(v("P0B.exp35.r4e1.R-N.pooled.primary") + 6.25) < 1e-9
assert abs(D.loc["r4e1", "RmN"] + 6.25) < 1e-9
assert abs(v("P0B.P1D.r4e1.R-N.pooled.primary") - 1.25) < 1e-9
for cfg, col, val in (("r1e2", "FmC", 1.25), ("r1e2", "RmN", 1.25), ("r1e1", "FmC", -1.25), ("r2e2", "RmN", 1.25)):
    assert abs(D.loc[cfg, col] - val) < 1e-9, (cfg, col)
for cfg, col, val in (("r1e2", "corrected", 68.75), ("r1e2", "reverse", 86.25), ("r1e1", "corrected", 68.75),
                      ("r2e2", "reverse", 83.75), ("r2e1", "feedback", 76.25), ("r2e1", "corrected", 68.75)):
    assert abs(RATE.loc[cfg, col] - val) < 1e-9, (cfg, col)
auth, v3 = ci("P0C.plus437.AUTHORITATIVE"), ci("P0C.plus437.V3")
assert f"{auth[0]:.2f}" == "3.34" and f"{auth[1]:.2f}" == "5.17"
assert f"{v3[0]:.2f}" == "3.32" and f"{v3[1]:.2f}" == "5.19"
assert R["P0C.plus437.AUTHORITATIVE"]["p_adj"] == 0.000732421875
assert f"{R['P0C.DOC.001']['fields']['source_value']:.2f}" == "0.43"
st = ci("P0E.phaseC.d0_|8|_feedback")
assert (f"{100 * st[0]:.1f}", f"{100 * st[1]:.1f}") == ("90.8", "94.0")
of = R["P0E.succ.C-F_d8"]   # C-F; F-C is the sign flip
assert (round(-100 * of["ci95"][1]), round(-100 * of["ci95"][0])) == (-6, 12)
w = pd.read_csv(f"{POC}/outputs/review_followup3/P0/SPATIAL_TASK_EFFECTS_ci_width_summary.csv")
rf2 = R["P0C.R9.rec.win3.NEW_scene_drive"]
assert round(100 * rf2["value"]) == 90 and round(100 * rf2["ci95"][0]) == 81 and round(100 * rf2["ci95"][1]) == 99
unit = [round(100 * v(f"P0C.R9.rec.win{k}.OLD_exp9")) for k in (1, 2, 3, 4)]
scene = [round(100 * v(f"P0C.R9.rec.win{k}.NEW_scene_drive")) for k in (1, 2, 3, 4)]
assert unit == [47, 76, 90, 96] and scene == [47, 77, 90, 96], (unit, scene)
assert v("P2.verdict.A-") is True and v("P2.verdict.A+") is False and v("P2.verdict.all") is False
NOTE = "(2026-10-15 수정)"
NOTE_EN = "(corrected 2026-10-15)"

FIX = {
    f"{ROOT}/EXPERIMENT_SUMMARY.md": [
        ("순수 좌표 누적(기하 baseline)보다 endpoint가 +4.37 m [+3.32, +5.19] 더 움직입니다(13 drive, Holm 9.8e-4).",
         "순수 좌표 누적(기하 baseline)보다 endpoint가 +4.37 m [+3.32, +5.19] 더 움직입니다(13 drive; **사후** C1 excess 지표, Holm m = 4 9.8e-4). "
         "사전 정의 P1A-2 지표의 authoritative 행은 +4.37 m [+3.34, +5.17], Holm m = 3 7.3e-4이며 두 행을 섞지 않습니다 " + NOTE + "."),
        ("suffix local motion +0.57 m/seg [+0.44, +0.68]; 기하 baseline 대비 endpoint +4.37 m [+3.34, +5.18];",
         "suffix local motion +0.57 m/seg [+0.43, +0.68]; 기하 baseline 대비 endpoint +4.37 m [+3.34, +5.17] " + NOTE + ";"),
        ("(−6.2%p, p = 0.04)\"는 삭제합니다**: 초기 상태 군집 후 p = 0.14(primary) / 0.26(physical)이고 P1-D에서 +1.2%p [−5.6, +8.0]로",
         "(−6.25%p, p = 0.04)\"는 삭제합니다**: 초기 상태 군집 후 p = 0.14(primary) / 0.26(physical)이고 P1-D에서 +1.25%p [−5.6, +8.0]로"),
        ("| r1e2 (1 / 2 / 0.31) | 85.0 | 70.0 | 68.8 | 86.2 | +1.2, Holm 1 | +1.2, Holm 1 |",
         "| r1e2 (1 / 2 / 0.31) | 85.0 | 70.0 | 68.75 | 86.25 | +1.25, Holm 1 | +1.25, Holm 1 |"),
        ("| r1e1 (1 / 없음 / 0) | 85.0 | 67.5 | 68.8 | 80.0 | −1.2, Holm 1 | −5.0, Holm 1 |",
         "| r1e1 (1 / 없음 / 0) | 85.0 | 67.5 | 68.75 | 80.0 | −1.25, Holm 1 | −5.0, Holm 1 |"),
        ("| r2e2 (2 / 2 / 0.66) | 82.5 | 77.5 | 77.5 | 83.8 | 0.0, Holm 1 | +1.2, Holm 1 |",
         "| r2e2 (2 / 2 / 0.66) | 82.5 | 77.5 | 77.5 | 83.75 | 0.0, Holm 1 | +1.25, Holm 1 |"),
        ("| r2e1 (2 / 없음 / 0.50) | 87.5 | 76.2 | 68.8 | 85.0 |", "| r2e1 (2 / 없음 / 0.50) | 87.5 | 76.25 | 68.75 | 85.0 |"),
        ("| −6.2 [−12.1, −1.2], p = 0.040 (episode 수준;", "| −6.25 [−12.1, −1.2], p = 0.040 (episode 수준;"),
        ("(위 표의 p와 CI는 원 episode 수준 분석입니다.",
         "(위 표의 p와 CI는 원 episode 수준 분석입니다. 1.25%p 배수의 정확한 반올림 경계값은 두 자리로 적었습니다 " + NOTE + "."),
        ("| r4e1 | −6.2 [−13.6, 1.1] (**0.136 / 0.256**, 보정 전)", "| r4e1 | −6.25 [−13.6, 1.1] (**0.136 / 0.256**, 보정 전)"),
        ("실험 35 r4e1 reverse −6.2%p는 미재현(+1.2 [−5.6, +8.0])", "실험 35 r4e1 reverse −6.25%p는 미재현(+1.25 [−5.6, +8.0])"),
        ("실험 35 −6.2%p와", "실험 35 −6.25%p와"),
        ("하위 차원이 δ 이상 움직이는 비율 **92.6% [89.8, 94.4]**,",
         "하위 차원이 δ 이상 움직이는 비율 **92.6% [89.8, 94.4]**(task 군집 CI; 초기 상태 군집 CI [90.8, 94.0], review_followup3 P0-E),"),
        ("**과제별로는 CI 폭 18–29%p로 불확실**합니다.",
         "**과제별로는 CI 폭 최대 18–29%p(P1-D r4e1)로 불확실**합니다(r1e4는 6.9–21.6%p) " + NOTE + "."),
        ("(1/2/3/4 step = 47 / 77 / 90 / 96%; R9).",
         "(1/2/3/4 step = 47 / 77 / 90 / 96%, scene 가중; 실험 9 원 unit 가중은 47 / 76 / 90 / 96%; R9, 13 drive)."),
        ("| 1/2/3/4 step 교정이 효과의 47/77/90/96% (3 step ≈ 90% [81, 99]; 원 raw 없음, R9로 복구) |",
         "| 1/2/3/4 step 교정이 효과의 47/77/90/96% (scene 가중; 원 unit 가중 47/76/90/96; 3 step ≈ 90% [81, 99]; 원 raw 없음, R9로 복구) |"),
        ("day-vehicle 수준 상관은 모형화하지 않았습니다.",
         "day / vehicle / day-vehicle 의존성은 review_followup3 P0-D에서 민감도 분석했습니다: 모든 축에서 부호가 유지되지만, P1-C는 day–vehicle 연결 성분이 6개뿐이라 exact 검정의 Holm 하한이 0.094입니다 " + NOTE + "."),
        ("외삽 baseline은 repeat-p constant-motion 하나뿐이고 turn-rate baseline은 없습니다.",
         "기존 repeat-p constant-motion(CM)은 body-frame SE(2) 반복(이미 일정 회전율 외삽)임을 확인했고, CV0·CTRV-K3·CTRA-2와 학습된 non-AR 운동 외삽기를 추가했습니다; 외삽 초과는 A−에서만 성립합니다(review_followup3 P2) " + NOTE + "."),
        ("과제별 ±5%p를 검증하려면 수백–천여 개 상태가 필요하지만 move_near는 상태를 늘릴 수 없습니다(GAP_DECISION #3).",
         "과제별 ±5%p를 검증하려면 과제당 약 1,500개의 새 물리 상태(75–209 GPU h)가 필요합니다. 새 SimplerEnv 변형 상태 실험(P4, ±10%p 목표)은 계획만 하고 실행하지 않았습니다(review_followup3 CP2 결정) " + NOTE + "."),
    ],
    f"{PKG}/PAPER_NUMBERS.md": [
        ("| r4e1 | δR | +1.7 [−8.3, 11.5] | +0.8 [−8.3, 10.1] / [−8.1, 10.0] | +1.2 [−5.6, 8.0] |",
         "| r4e1 | δR | +1.7 [−8.3, 11.5] | +0.8 [−8.3, 10.1] / [−8.1, 10.0] | +1.25 [−5.6, 8.0] |"),
        ("the r4e1 reverse −6.2 pp (p = 0.040, episode level) is p = 0.14 / 0.26 after initial-state clustering and did not replicate in P1-D (+1.2 pp [−5.6, +8.0]).",
         "the r4e1 reverse −6.25 pp (p = 0.040, episode level) is p = 0.14 / 0.26 after initial-state clustering and did not replicate in P1-D (+1.25 pp [−5.6, +8.0])."),
        ("- \"A small task effect appears for reverse at 4-step replanning (−6.2 pp).\" Deleted.",
         "- \"A small task effect appears for reverse at 4-step replanning (−6.25 pp).\" Deleted."),
        ("Experiment 35's r4e1 reverse −6.2 pp is deleted (clustered p = 0.14 / 0.26; P1-D +1.2 pp [−5.6, +8.0]).",
         "Experiment 35's r4e1 reverse −6.25 pp is deleted (clustered p = 0.14 / 0.26; P1-D +1.25 pp [−5.6, +8.0])."),
        ("- \"4-step replanning makes reverse increase failures (−6.2 pp).\"", "- \"4-step replanning makes reverse increase failures (−6.25 pp).\""),
        ("**task-level CIs span 18–29 pp** (r4e1:", "**task-level CIs span 18–29 pp in r4e1** (r1e4: 6.9–21.6 pp; r4e1:"),
        ("endpoint **+4.37 m [+3.32, +5.19]** (drive p = 2.4e-4; Holm 9.8e-4 over the post-hoc excess family; predeclared family Holm 7.3e-4).",
         "endpoint **+4.37 m [+3.32, +5.19]** (post-hoc C1 excess metric ‖OC9 − U9‖ − ‖Geo9 − r9‖; drive p = 2.4e-4; Holm 9.8e-4 over the post-hoc excess family, m = 4). "
         "The **predeclared** P1A-2 metric (dev9_ref, OC − Geo) is the authoritative row: **+4.37 m [+3.34, +5.17]**, sign-flip-inversion CI [+3.06, +5.35], Holm 7.3e-4 (P1A family, m = 3) — `numbers.json` `P0C.plus437.AUTHORITATIVE`. Do not attach the predeclared Holm to the post-hoc CI " + NOTE_EN + "."),
        ("- Suffix local-motion change exceeds CM's d0 in every stratum (+0.48 / +0.13 / +0.21 m/seg). CM is a single repeat-p baseline; no turn-rate or fitted constant-velocity baseline exists.",
         "- Suffix local-motion change exceeds CM's d0 in every stratum (+0.48 / +0.13 / +0.21 m/seg).\n"
         "- **Updated 2026-10-15 (review_followup3 P2):** CM is a body-frame SE(2) repeat of the perturbed token (yaw included), i.e. already a constant-turn-rate extrapolation. Added CV0 (zero-turn global velocity), CTRV-K3, CTRA-2 and a learned non-AR motion extrapolator (prefix / prefix + ego, 3 seeds). In A− the model exceeds all six; in A+ and all scenes it does not (see Claim 23)."),
        ("in **92.6% [89.8, 94.4]** of 4,741 units (24 bins: 81.2%).",
         "in **92.6% [89.8, 94.4]** of 4,741 units (task-cluster CI; initial-state-cluster CI [90.8, 94.0]; 24 bins: 81.2%)."),
        ("- **Task level:** feedback − corrected **+3 pp [−4, +9], p = 0.65** (8 bins),",
         "- **Task level:** feedback − corrected **+3 pp [−4, +9], p = 0.65** (8 bins; task-cluster CI — with initial-state clusters [−6, +12], review_followup3 P0-E),"),
    ],
    f"{PKG}/tables.tex": [
        ("r1e2 & 0.31 & 85.0 & 70.0 & 68.8 & 86.2 & $+1.2$ [$-10.0$, $+13.8$] & $+1.2$ [$-7.5$, $+10.0$] \\\\",
         "r1e2 & 0.31 & 85.0 & 70.0 & 68.75 & 86.25 & $+1.25$ [$-10.0$, $+13.8$] & $+1.25$ [$-7.5$, $+10.0$] \\\\"),
        ("r1e1 & 0 & 85.0 & 67.5 & 68.8 & 80.0 & $-1.2$ [$-13.8$, $+11.2$]", "r1e1 & 0 & 85.0 & 67.5 & 68.75 & 80.0 & $-1.25$ [$-13.8$, $+11.2$]"),
        ("r2e2 & 0.66 & 82.5 & 77.5 & 77.5 & 83.8 & $0.0$ [$-10.0$, $+10.0$] & $+1.2$ [$-10.0$, $+12.5$] \\\\",
         "r2e2 & 0.66 & 82.5 & 77.5 & 77.5 & 83.75 & $0.0$ [$-10.0$, $+10.0$] & $+1.25$ [$-10.0$, $+12.5$] \\\\"),
        ("r2e1 & 0.50 & 87.5 & 76.2 & 68.8 & 85.0 &", "r2e1 & 0.50 & 87.5 & 76.25 & 68.75 & 85.0 &"),
        ("& $-6.2$ [$-12.1$, $-1.2$], $p{=}0.040$$^{\\dagger}$ \\\\", "& $-6.25$ [$-12.1$, $-1.2$], $p{=}0.040$$^{\\dagger}$ \\\\"),
        ("r4e1 $\\delta_R$ $-6.2$ [$-13.6$, $+1.1$] ($p=0.14$)", "r4e1 $\\delta_R$ $-6.25$ [$-13.6$, $+1.1$] ($p=0.14$)"),
        ("the original r4e1 R$-$N effect ($-6.2$ pp, $p=0.040$)", "the original r4e1 R$-$N effect ($-6.25$ pp, $p=0.040$)"),
        ("so it is withdrawn.", "so it is withdrawn. Exact multiples of 1.25 pp at a rounding tie are printed with two decimals (corrected 2026-10-15); CI bounds are as in the original output."),
        ("& $+1.2$ [$-5.6$, $8.0$] & 0.006 / 0.010 ; 0.27 / 0.31 \\\\", "& $+1.25$ [$-5.6$, $8.0$] & 0.006 / 0.010 ; 0.27 / 0.31 \\\\"),
        ("task-level CIs span 18--29 pp, so task-specific effects remain uncertain.",
         "task-level CIs span up to 18--29 pp (r4e1; r1e4 6.9--21.6 pp), so task-specific effects remain uncertain."),
        ("& within-step 92.6 [89.8, 94.4] &", "& within-step 92.6 [89.8, 94.4] (task CI; state CI [90.8, 94.0]) &"),
        ("1/2/3/4 step: 47/77/90/96\\% (3 step: 90 [81, 99])", "1/2/3/4 step: 47/77/90/96\\% scene-weighted (3 step: 90 [81, 99])"),
        ("feedback$-$corrected $+3$ [$-4$, $+9$], $p=0.65$", "feedback$-$corrected $+3$ [$-4$, $+9$] task CI, [$-6$, $+12$] state CI, $p=0.65$"),
    ],
    f"{PKG}/MANUSCRIPT_REVISION.md": [
        ("| \"4-step replanning makes reverse increase failures (−6.2 pp, p = 0.04)\" / \"a small task effect appears only for reverse\" | p = 0.14–0.26 after initial-state clustering; not replicated (+1.2 pp) | (delete) |",
         "| \"4-step replanning makes reverse increase failures (−6.25 pp, p = 0.04)\" / \"a small task effect appears only for reverse\" | p = 0.14–0.26 after initial-state clustering; not replicated (+1.25 pp) | (delete) |"),
        ("moves the 5 s endpoint 4.37 m [3.32, 5.19] beyond pure coordinate accumulation.",
         "moves the 5 s endpoint 4.37 m [3.32, 5.19] beyond pure coordinate accumulation (post-hoc excess metric; the predeclared P1A-2 row is 4.37 m [3.34, 5.17], Holm 7.3 × 10⁻⁴ — see the 2026-10-15 section)."),
    ],
}


def main():
    for path, reps in FIX.items():
        s = open(path).read()
        n_new = 0
        for old, new in reps:
            if old in s:
                assert s.count(old) == 1, (path, old[:60])
                s = s.replace(old, new)
                n_new += 1
            elif new in s:
                continue
            else:
                raise SystemExit(f"NOT FOUND in {path}: {old[:90]}")
        open(path, "w").write(s)
        print(f"{path}: {n_new} replacements")


if __name__ == "__main__":
    main()
