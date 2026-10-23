#!/usr/bin/env python
"""review_followup5: wording / number corrections in the manuscript-facing documents (2026-10-18). CPU only.
Reuses apply_line / insert_section from scripts/review_followup4/c/c_doc_fixes.py with the marker
"(2026-10-18 수정, review_followup5)". Log: outputs/review_followup5/DOC_EDITS_APPLIED.csv. Idempotent.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python c5_doc_fixes.py
"""
import csv
import importlib.util
import os

spec = importlib.util.spec_from_file_location("c4", "/root/VLA/autovla_misalignment_poc/scripts/review_followup4/c/c_doc_fixes.py")
C4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C4)
C4.MK = " (2026-10-18 수정, review_followup5)"
C4.MK_TEX = "  % corrected 2026-10-18 (review_followup5)"
ROOT = "/root/VLA"
PKG = f"{ROOT}/paper_quantitative_package"
R5 = f"{ROOT}/autovla_misalignment_poc/outputs/review_followup5"
ES, PN, MR = f"{ROOT}/EXPERIMENT_SUMMARY.md", f"{PKG}/PAPER_NUMBERS.md", f"{PKG}/MANUSCRIPT_REVISION.md"
SEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doc_sections")

EDITS = [
    (ES, "그러나 **±5%p 동등성은 성립하지 않습니다**(TOST p = 0.09).",
     "원 ±5%p TOST(p = 0.09, sign-flip)는 이진 자료에서 무효였고, 사후 cluster-t TOST로는 ±5%p 안(p = 0.027, 보정 전, 사후 margin)이지만 확증적 동등성으로 쓰지 않습니다.",
     "A_STATS"),
    (ES, "±5%p 동등성 미성립, TOST p = 0.09)",
     "원 sign-flip TOST p = 0.09는 무효; 사후 cluster-t TOST p = 0.027(보정 전·사후 margin, 확증 아님))", "A_STATS"),
    (ES, "+6.2%p [+0.6, +12.1], sign-flip p = 0.073으로",
     "+6.2%p (bootstrap [+0.6, +12.1]; 검증된 cluster-t 구간 [−0.1, +12.4]), sign-flip p = 0.073으로", "A_STATS"),
    (ES, "RN P −4.1%p [−6.4, −2.4], RV P +3.6%p [+2.2, +5.5]입니다(P1-C 사전 등록 판정을 정확히 재현).",
     "RN P −4.1%p, RV P +3.6%p입니다(P1-C 사전 등록 판정을 정확히 재현; sign-flip p 5.7e-6 / 9.5e-7; drive bootstrap 구간 [−6.4, −2.4] / [+2.2, +5.5]는 이 가중 구조에서 coverage가 확인되지 않아 추론용으로 쓰지 않음).",
     "A_STATS"),
    (ES, "바닥을 크게 넘는 대비(교정 −63.4%p, reverse +81.9%p)만 바닥과 함께 기술하고,",
     "교정 −63.4%p, reverse +81.9%p도 기술적으로만 씁니다(관측된 5–6% 판정 불일치율은 조건별 수치 편향의 상한이 아니므로 \"바닥보다 커서 강건\"이라고 쓰지 않음).",
     "A2 (Impromptu wording)"),
    (ES, "효과는 **최신 문맥 token의 값**을 따르며, 이전 GT prefix의 존재에 의존하지 않습니다(INT ≈ 0; 이전 값이 실제로 다른 단위에서도 −1.7%p [−5.4, +1.9]).",
     "앞선 비교 위치(primary anchor에서는 t*+1 **한 위치**)를 모델 생성값으로 유지해도 직전 문맥 token을 GT로 교정하는 큰 효과가 관찰됐고, 두 조건의 효과 차이는 INT −0.2%p [−3.8, +3.1]입니다(이전 값이 실제로 다른 단위 −1.7%p [−5.4, +1.9]; older-prefix 주효과 F +2.0 ~ +4.8%p). 차이가 유의하지 않다는 것은 효과가 같다거나 오래된 문맥과 독립이라는 뜻이 아닙니다.",
     "A2 (B2 wording)"),
    (PN, "# Claim 25. The latest context value matters; no dependence on an older GT prefix in a one-shot schedule (new 2026-10-18; review_followup4 B2, preregistered 552b3ac)",
     "# Claim 25. A large latest-token correction effect also when the earlier compared position keeps the model's own value (new 2026-10-18; review_followup4 B2, preregistered 552b3ac; re-worded by review_followup5)",
     "A2 (B2 wording)"),
    (PN, "This oracle-conditioned diagnostic indicates that, in a one-shot schedule, the effect tracks the value of the latest context token rather than an older ground-truth prefix; across the B2 eligible domain the effect was small (+3–4 pp).",
     "The earlier factor is a single context position; the interaction is an estimate with its uncertainty, not evidence that the effects are equal or independent of older context. This is an oracle-conditioned diagnostic; across the B2 eligible domain the effect was small (+3–4 pp; interval not validated, see review_followup5 A_STATS).",
     "A2 (B2 wording)"),
    (PN, "- **A3:** Claim 8 (Impromptu) downgraded to descriptive / parity not established (numerical floor 5–6% scene-level verdict flips).",
     "- **A3:** Claim 8 (Impromptu) downgraded to descriptive / parity not established (5–6% scene-level verdict flips from numerics alone; not an upper bound on condition-specific bias).",
     "A2 (Impromptu wording)"),
    (PN, "  - RN **−4.1 pp [−6.4, −2.4]**;", "  - RN **−4.1 pp** (sign-flip p 5.7e-6; drive bootstrap [−6.4, −2.4], coverage not established);", "A_STATS"),
    (MR, "so in this one-shot schedule the effect tracks the value of the most recent context token rather than the presence of an older ground-truth prefix.",
     "and the earlier compared position is a single context position (difference estimate reported with its uncertainty; not equality).",
     "A2 (B2 wording)"),
    (MR, "by margins well above the 5–6% scene-level flip rate caused by batch and decoding-path numerics alone.\"",
     "(descriptive: batch and decoding-path numerics alone flipped 5–6% of scene-level verdicts, which is not an upper bound on condition-specific numerical bias).\"",
     "A2 (Impromptu wording)"),
]


def main():
    log = []
    for path, old, new, src in EDITS:
        C4.apply_line(path, None, old, new, log, src)
    rd = lambda f: open(os.path.join(SEC, f)).read()
    C4.insert_section(ES, "## 7. 불안정 rollout 탐지: 이탈 전에 예측 가능한가 (실험 31, P4)", rd("es_6h.md") + "\n---\n", "## 6h. review_followup5", log)
    C4.insert_section(PN, "# Claim 1. Token mismatch ≠ failure", rd("pn_round5.md") + "\n---\n", "## 2026-10-18 개정 2 (review_followup5 반영)", log)
    C4.insert_section(PN, None, rd("pn_claim28.md"), "# Claim 28.", log)
    C4.insert_section(MR, None, rd("mr_round5.md"), "## 2026-10-18 revision 2 (review_followup5)", log)
    C4.insert_section(f"{PKG}/README.md", None, rd("readme_note5.md"), "**2026-10-18 개정 2 (review_followup5)", log)
    s = open(ES).read()
    old = "최종 갱신: 2026-10-18 (review_followup4 반영:"
    if old in s and "review_followup5 반영" not in s:
        s = s.replace(old, "최종 갱신: 2026-10-18 (review_followup5 반영: §6h 신설; 해당 줄 '(2026-10-18 수정, review_followup5)'). 같은 날 앞선 갱신 (review_followup4 반영:", 1)
        open(ES, "w").write(s)
        log.append({"source": "C (header)", "file": ES, "line": 3, "status": "applied", "old": old, "new": "review_followup5 header"})
    with open(f"{R5}/DOC_EDITS_APPLIED.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "file", "line", "status", "old", "new"])
        w.writeheader(); w.writerows(log)
    from collections import Counter
    print(Counter(x["status"] for x in log))


if __name__ == "__main__":
    main()
