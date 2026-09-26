#!/usr/bin/env python
"""Render outputs/natural_fast_mechanism/NATURAL_FAST_MECHANISM.md from summary.json."""
from __future__ import annotations

import json
import os

POC = "/root/VLA/autovla_misalignment_poc"
OUT = os.path.join(POC, "outputs/natural_fast_mechanism")

# Earlier results, quoted from AUTOVLA_POC_RESULT.md. Those analyses ran on the NATURAL run
# (outputs/gpu1) with failure = step-0 token ID != GT, not on forced CoT.
PRIOR = {
    "perception": "A− lead vehicle AUROC 0.71–0.82 vs A+ 0.84–0.91 → 실패군에서 낮음 (§5, 신뢰도 플래그)",
    "late": "commitment L34–35 (A+/A− 동일), 최종 margin 격차의 84%가 L32–35에서 생성 (§6, Add.2)",
    "l35": "L35 MLP 정상 +3.13 / 실패 −1.53, 차이 −4.66 — 16개 칸 중 최대 (Add.2)",
    "attn": "L32–35 합: MLP 차이 −7.08 vs Attention −0.26 → 27× (Add.2)",
    "codebook": "step-0 오답 토큰 = GT의 8번째 최근접(중앙값), 0.079 m (NN 간격 0.043 m) (Add.4)",
    "amplification": "P(오답|이전 오답) 73% vs 깨끗한 prefix 4% (최대 19.7×); 첫 오답 교정 시 downstream 오류 −42pp (Add.1)",
}


def f(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def ci(d, nd=2, key="diff"):
    if d is None:
        return "–"
    return f"{d[key]:+.{nd}f} [{d['ci95'][0]:+.{nd}f}, {d['ci95'][1]:+.{nd}f}]"


def yes(v):
    if v is None:
        return "판정 불가"
    return "✅" if v.get("holds") else "❌"


def main() -> None:
    S = json.load(open(os.path.join(OUT, "summary.json")))
    N, G, C = S.get("N"), S.get("G"), S.get("C")
    W = []
    w = W.append

    w("# AutoVLA — Natural/Fast 실제 실패에서의 메커니즘 재검증\n")
    w("GPU 미사용. 저장된 tensor만 사용했습니다. 실패(A−)는 **궤적의 coarse action** 기준(P/R/A 비교와 동일한 accept 규칙, 5 s)이며 token ID는 실패 정의에 쓰지 않았습니다.\n")
    w("## 0. 전제 정정\n")
    w("기존 AutoVLA 메커니즘 결과(probe, logit lens, Addendum 1–4)는 **Forced-CoT가 아니라 natural run(`outputs/gpu1`)**에서 나왔고, 실패를 "
      "**step-0 action token ID ≠ GT**로 정의했습니다. Forced-CoT(arm C)는 이후 full extraction과 P/R/A 비교에서만 쓰였습니다. "
      "그래서 이 보고서의 비교축은 두 개입니다: (a) **기존 결과**(natural + token-ID 정의), (b) **Forced-CoT**(arm C, 이번과 같은 파이프라인·궤적 정의).\n")

    w("## 1. 데이터\n")
    w("| 소스 | 설명 | n | 궤적 A− | A− 중 step-0 토큰 오답 | A+ 중 step-0 토큰 오답 |")
    w("|---|---|---:|---:|---:|---:|")
    for key, name, desc in (("N", "Natural Fast (arm N)", "full extraction, 층별 attn/mlp 성분 저장 — **주 분석**"),
                            ("G", "Natural run (gpu1)", "원래 natural run, residual만 저장 — 재현 확인"),
                            ("C", "Forced-CoT (arm C)", "같은 파이프라인 — 비교용")):
        s = S.get(key)
        if s:
            w(f"| {name} | {desc} | {s['n']} | {s['n_Aminus']} | {s['n_Aminus_step0_token_wrong']} | {s['n_Aplus_step0_token_wrong']} |")
    w("")
    if N:
        w(f"궤적 기준 natural 실패 {N['n_Aminus']}건 중 step-0 토큰이 틀린 것은 {N['n_Aminus_step0_token_wrong']}건뿐이고, "
          f"step-0 토큰이 틀렸는데 궤적은 맞은 A+가 {N['n_Aplus_step0_token_wrong']}건입니다. **token-ID 실패와 궤적 실패는 대부분 다른 샘플**입니다.\n")

    # ------------------------------------------------------------------ summary table
    def dec(s, g):
        return (s or {}).get("decomposition", {}).get(g, {})

    def cell_perception(s):
        if not s:
            return "–"
        lv = s["perception"].get("lead_vehicle", {}).get("layers", {})
        parts = []
        for l in (32, 36):
            x = lv.get(l) or lv.get(str(l))
            if x and x["auroc_Aplus"] is not None:
                parts.append(f"L{l} A+ {f(x['auroc_Aplus'])} / A− {f(x['auroc_Aminus'])}")
        return "; ".join(parts) or "A− 표본 부족"

    def cell_late(s):
        d = dec(s, "traj_step0")
        if "commit_layer_median" not in d:
            return "–"
        return (f"commit L{d['commit_layer_median']['ok']:.0f}/L{d['commit_layer_median']['fail']:.0f} (A+/A−), "
                f"격차 {100 * (d['share_gap_created_L32_L35'] or 0):.0f}% L32–35")

    def cell_l35(s, g):
        d = dec(s, g)
        if "cells" not in d:
            return "–"
        c = d["cells"]["L35_mlp"]
        return (f"정상 {f(c['ok']['mean'])} / 실패 {f(c['fail']['mean'])}, 차이 {ci(c['diff'])}; 최대 칸 {d['largest_cell']}")

    def cell_attn(s, g):
        d = dec(s, g)
        if "sum_L32_35" not in d:
            return "–"
        m, a = d["sum_L32_35"]["mlp"]["diff"], d["sum_L32_35"]["attn"]["diff"]
        return f"MLP {m['diff']:+.2f} vs Attn {a['diff']:+.2f} ({d['mlp_over_attn_separation']:.1f}×)"

    def cell_cb(s, grp="A-"):
        if not s or grp not in s["codebook_amplification"]:
            return "–"
        x = s["codebook_amplification"][grp]
        return (f"NN 순위 중앙값 {x['nn_rank_of_wrong_token']['median']:.0f}, {x['wrong_token_displacement_m']['median']:.3f} m, "
                f"순위≤10 {100 * x['nn_rank_of_wrong_token']['share_rank_le10']:.0f}%")

    def cell_amp(s):
        if not s or "A-" not in s["codebook_amplification"]:
            return "–"
        x = s["codebook_amplification"]
        a = x["A-"]
        return (f"FDE/첫 오차 {a['fde_over_first_step_error_median']:.1f}×, 전파 {100 * x['propagation']['p_wrong_given_clean_prefix']:.0f}%→"
                f"{100 * x['propagation']['p_wrong_given_earlier_wrong']:.0f}% ({x['propagation']['ratio']:.1f}×)")

    vN = (N or {}).get("verdicts", {})
    vG = (G or {}).get("verdicts", {})
    vC = (C or {}).get("verdicts", {})

    def rep(*vs):
        vals = [v.get("holds") for v in vs if v]
        if not vals:
            return "판정 불가"
        if all(vals):
            return "✅ 재현"
        if not any(vals):
            return "❌ 재현 안 됨"
        return "△ 부분 재현"

    w("## 2. 요약\n")
    w("| Finding | 기존 결과 (natural + token-ID) | Forced-CoT (arm C, 궤적 A−) | Natural Fast (arm N, 궤적 A−) | 재현 여부 |")
    w("|---|---|---|---|---|")
    w(f"| Perception preserved | {PRIOR['perception']} | {cell_perception(C)} {yes(vC.get('perception_preserved'))} | "
      f"{cell_perception(N)} {yes(vN.get('perception_preserved'))} | {rep(vN.get('perception_preserved'), vG.get('perception_preserved'))} |")
    w(f"| Late action formation | {PRIOR['late']} | {cell_late(C)} {yes(vC.get('late_action_formation'))} | "
      f"{cell_late(N)} {yes(vN.get('late_action_formation'))} | {rep(vN.get('late_action_formation'), vG.get('late_action_formation'))} |")
    w(f"| L35 MLP dominance | {PRIOR['l35']} | {cell_l35(C, 'traj_step0')} {yes(vC.get('L35_mlp_dominance[traj_step0]'))} | "
      f"step0: {cell_l35(N, 'traj_step0')} {yes(vN.get('L35_mlp_dominance[traj_step0]'))}<br>첫 불일치 step: {cell_l35(N, 'traj_first_mismatch')} "
      f"{yes(vN.get('L35_mlp_dominance[traj_first_mismatch]'))} | "
      f"{rep(vN.get('L35_mlp_dominance[traj_step0]'), vN.get('L35_mlp_dominance[traj_first_mismatch]'))} |")
    w(f"| Attention weak | {PRIOR['attn']} | {cell_attn(C, 'traj_step0')} {yes(vC.get('attention_weak[traj_step0]'))} | "
      f"step0: {cell_attn(N, 'traj_step0')} {yes(vN.get('attention_weak[traj_step0]'))}<br>첫 불일치 step: {cell_attn(N, 'traj_first_mismatch')} "
      f"{yes(vN.get('attention_weak[traj_first_mismatch]'))} | "
      f"{rep(vN.get('attention_weak[traj_step0]'), vN.get('attention_weak[traj_first_mismatch]'))} |")
    w(f"| Small codebook mismatch | {PRIOR['codebook']} | {cell_cb(C)} {yes(vC.get('small_codebook_mismatch'))} | "
      f"{cell_cb(N)} {yes(vN.get('small_codebook_mismatch'))}<br>(구 정의 그룹: {cell_cb(N, 'step-0 token wrong (old definition)')}) | "
      f"{rep(vN.get('small_codebook_mismatch'), vG.get('small_codebook_mismatch'))} |")
    w(f"| Error amplification | {PRIOR['amplification']} | {cell_amp(C)} {yes(vC.get('error_amplification'))} | "
      f"{cell_amp(N)} {yes(vN.get('error_amplification'))} | {rep(vN.get('error_amplification'), vG.get('error_amplification'))} |")
    w("")
    w("재현 여부는 Natural Fast(arm N)의 사전 규칙 판정과, 가능한 항목은 원래 natural run(gpu1) 판정을 함께 본 것입니다. 규칙은 `scripts/natural_fast_mechanism.py` 상단에 실행 전 고정했습니다.\n")

    # ------------------------------------------------------------------ details
    w("## 3. 상세\n")
    w("### 3.1 Perception 보존 (step-0 action query에서 GT perception 선형 probe, log 단위 5-fold)\n")
    w("| 소스 | 변수 | n (양성) | A− n (양성) | L32 AUROC A+ / A− | L36 AUROC A+ / A− | L36 격차 A−−A+ [CI] |")
    w("|---|---|---:|---:|---|---|---|")
    for key, s in (("N", N), ("G", G), ("C", C)):
        if not s:
            continue
        for var, pv in s["perception"].items():
            L32 = pv["layers"].get("32") or pv["layers"].get(32) or {}
            L36 = pv["layers"].get("36") or pv["layers"].get(36) or {}
            gap = L36.get("gap_Aminus_minus_Aplus")
            w(f"| {key} | {var} | {pv['n']} ({pv['n_pos']}) | {pv['n_fail']} ({pv['n_fail_pos']}) | "
              f"{f(L32.get('auroc_Aplus'))} / {f(L32.get('auroc_Aminus'))} | {f(L36.get('auroc_Aplus'))} / {f(L36.get('auroc_Aminus'))} | "
              f"{ci(gap) if gap else '–'} |")
    w("")

    w("### 3.2 정상/실패 action representation이 갈라지는 층 (logit lens: GT − rival margin)\n")
    w("| 소스 | 그룹 | n 실패/정상 | commit 층 중앙값 정상/실패 | 격차가 유의하게 음수가 되어 유지되는 첫 층 | L32 입력 격차 | 최종 격차 | L32–35에서 생긴 비율 |")
    w("|---|---|---|---|---:|---:|---:|---:|")
    for key, s in (("N", N), ("G", G), ("C", C)):
        if not s:
            continue
        for g in ("traj_step0", "traj_first_mismatch", "token_step0"):
            d = s["decomposition"].get(g, {})
            if "commit_layer_median" not in d:
                continue
            w(f"| {key} | {g} | {d['n_fail']}/{d['n_ok']} | L{d['commit_layer_median']['ok']:.0f} / L{d['commit_layer_median']['fail']:.0f} | "
              f"{d['first_layer_diff_ci_below0_and_stays']} | {f(d['gap_at_L32_input'])} | {f(d['final_gap'])} | "
              f"{100 * (d['share_gap_created_L32_L35'] or 0):.0f}% |")
    w("")

    w("### 3.3–3.4 층·성분별 margin 기여 (실패 − 정상, log cluster bootstrap)\n")
    for key, s in (("N", N), ("C", C)):
        if not s:
            continue
        for g in ("traj_step0", "traj_first_mismatch", "token_step0"):
            d = s["decomposition"].get(g, {})
            if "cells" not in d:
                continue
            w(f"**{key} · {g}** (실패 {d['n_fail']}, 정상 {d['n_ok']}, 실패 로그 {d['n_logs_fail']})\n")
            w("| 층 | Attn 정상 | Attn 실패 | Attn 차이 [CI] | MLP 정상 | MLP 실패 | MLP 차이 [CI] |")
            w("|---|---:|---:|---|---:|---:|---|")
            for l in range(28, 36):
                a, m = d["cells"][f"L{l}_attn"], d["cells"][f"L{l}_mlp"]
                w(f"| L{l} | {f(a['ok']['mean'])} | {f(a['fail']['mean'])} | {ci(a['diff'])} | {f(m['ok']['mean'])} | {f(m['fail']['mean'])} | {ci(m['diff'])} |")
            sm = d["sum_L32_35"]
            w(f"\nL32–35 합: Attention 차이 {ci(sm['attn']['diff'])}, MLP 차이 {ci(sm['mlp']['diff'])} → MLP/Attn {d['mlp_over_attn_separation']:.1f}×, "
              f"L35 MLP가 전체 성분 차이 합의 {100 * d['L35_mlp_share_of_total_component_diff']:.0f}%\n")

    w("### 3.5–3.6 첫 토큰 불일치의 codebook 크기와 최종 오차\n")
    w("| 소스 | 그룹 | n | 오답 토큰 변위 거리 (중앙값) | 무작위 쌍 대비 백분위 | NN 순위 중앙값 | 순위 ≤10 | 첫 불일치 step 위치 오차 | ADE | FDE | FDE/첫 오차 | Spearman(첫 거리, FDE) |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for key, s in (("N", N), ("G", G), ("C", C)):
        if not s:
            continue
        cb = s["codebook_amplification"]
        for grp in ("A-", "A+ with a token mismatch", "step-0 token wrong (old definition)"):
            x = cb.get(grp)
            if not x:
                continue
            sp = x["spearman_first_displacement_vs_fde"]
            spt = "–" if not sp else (f"{sp['rho']:+.2f}" + (f" [{sp['ci95'][0]:+.2f}, {sp['ci95'][1]:+.2f}]" if sp["ci95"] else ""))
            w(f"| {key} | {grp} | {x['n']} | {x['wrong_token_displacement_m']['median']:.3f} m | {x['wrong_token_displacement_m']['percentile_vs_random_pairs']:.1f} | "
              f"{x['nn_rank_of_wrong_token']['median']:.0f} | {100 * x['nn_rank_of_wrong_token']['share_rank_le10']:.0f}% | "
              f"{x['pose_error_at_first_mismatch_m']:.3f} m | {x['ade_m']:.2f} | {x['fde_m']:.2f} | {x['fde_over_first_step_error_median']:.1f}× | {spt} |")
        pr = cb["propagation"]
        w(f"| {key} | 전파(전체) | | P(오답 | 깨끗한 prefix) {100 * pr['p_wrong_given_clean_prefix']:.1f}% | P(오답 | 이전 오답) {100 * pr['p_wrong_given_earlier_wrong']:.1f}% | {pr['ratio']:.1f}× | | | | | | |")
    w(f"\n무작위 토큰 쌍 변위 거리: 중앙값 {S['N']['codebook_amplification']['random_pair_displacement_m']['median']:.2f} m, "
      f"10백분위 {S['N']['codebook_amplification']['random_pair_displacement_m']['p10']:.2f} m.\n")

    w("## 4. 저장된 tensor로는 할 수 없어 실행하지 않은 분석 (GPU 재추출 필요)\n")
    w("- **Natural 경로의 MLP 내부 활성**(gate_proj / up_proj, L28–35): arm C에만 저장. L35 MLP neuron 수준 분석은 natural에서 불가.")
    w("- **Natural 경로의 attention map / Q·K·V**(L28–35): arm C에만 저장. 'attention allocation이 정상/실패를 구분하지 않는다'는 기존 결과의 natural 재검증은 불가 — 이번 표의 Attention 항목은 **residual 기여(attn_out)만** 근거입니다.")
    w("- **Natural 경로의 visual token hidden state**: arm C heavy 250개에만 저장. perception은 action query 위치의 선형 probe로만 봤고, visual token 수준 perception 보존은 확인 불가.")
    w("- **인과 개입**(L35 MLP patch, 첫 오답 teacher-forcing)을 **궤적 기준 natural 실패**에 다시 적용하는 것: 생성이 필요하므로 미실행. 기존 인과 결과(Add.1, Add.3)는 token-ID 실패에서만 성립이 확인된 상태.")
    w("- **gpu1 natural run의 성분 분해**: residual만 저장돼 attn/mlp 분리 불가 — 성분 분석은 arm N으로만 수행.\n")
    w("## 5. 한계\n")
    w("- Natural 궤적 실패는 드뭅니다(arm N 약 2%). A− 표본이 작아 CI가 넓고, perception probe는 A− 안에 양성이 적은 변수에서 AUROC가 정의되지 않습니다.")
    w("- 궤적 실패 중 다수는 step-0 토큰이 맞습니다. step-0 분석은 그 샘플들에서 '실패를 만든 토큰'을 보지 못하므로, 첫 불일치 step 기준(step-matched 대조) 결과를 함께 봐야 합니다.")
    w("- 이것은 attribution이며 인과 검증이 아닙니다.")

    with open(os.path.join(OUT, "NATURAL_FAST_MECHANISM.md"), "w") as fh:
        fh.write("\n".join(W))
    print(f"wrote {os.path.join(OUT, 'NATURAL_FAST_MECHANISM.md')}")


if __name__ == "__main__":
    main()
