#!/usr/bin/env python
"""Writes paper_cross_model_table.csv / .tex. All numbers transcribed from analysis.json / RESULTS / recomputed CSV
(see _parts/source_map_crossmodel.md). No new statistics."""
import csv
P = "/root/VLA/autovla_misalignment_poc/outputs/paper_quantitative_package/"
H = ["row", "AutoVLA", "Impromptu VLA 3B", "OpenVLA-7B", "SpatialVLA-4B"]
R = [
 ["Backbone", "Qwen2.5-VL-3B", "Qwen2.5-VL-3B (full fine-tune)", "Llama-2 7B", "PaliGemma2-3B (Gemma2) + SigLIP + Ego3D"],
 ["Domain", "Driving", "Driving", "Tabletop manipulation (sim)", "Tabletop manipulation (sim)"],
 ["Benchmark", "NAVSIM navtest PoC (28 logs)", "NAVSIM navtest PoC (28 logs, 2,728 parsable scenes)", "LIBERO-Spatial (10 tasks x 10 inits = 100 eps)", "SimplerEnv Google Robot pick_coke_can + move_near (2 x 40 seeds = 80 eps)"],
 ["Natural performance", "A- 1.9% (52/2,747); FDE5 0.66 m [0.48, 0.87] (500-scene subset)", "A- 33.4% [26.2, 38.0]; amp 31.5%; FDE5 5.75 m [5.17, 6.27] (N=2,728)", "Success 87/100 = 87% (official 84.7%, summary only)", "Success 68/80 = 85.0% [77.5, 92.5] (coke 37/40 92.5%, move near 31/40 77.5%; official 0.86/0.78)"],
 ["Natural rerun disagreement / noise", "Not reported (no natural-rerun control)", "A- disagreement 5.9% [5.2, 6.9]; A+->A- 4.6% [3.7, 5.5]; final-wp shift 0.90 m", "Outcome disagreement 20/100 = 20%; 81% vs 87%, -6 pp [-14, +2], McNemar p=0.26", "natural_gen: disagreement 11/80 = 13.8%; +8.8 pp [+1.2, +16.2], McNemar p=0.065; endpoint 0.140 m [0.100, 0.185]; natural_mix 0/80"],
 ["Action representation", "Codebook token (2,048), 1 token / 0.5 s relative motion", "Absolute [x, y] waypoint as digit text (~12 tokens / waypoint)", "7 discretized dims (x,y,z,roll,pitch,yaw,grip), 1 token each", "3 tokens / step: translation (spherical 4,096 bins), rotation (16^3), gripper (2)"],
 ["Action chunk length", "10 steps (5 s)", "10 waypoints (5 s)", "1 step (no chunk)", "4 steps"],
 ["Temporal AR?", "Yes", "Yes", "No (within-step dimension AR only)", "Yes (within 4-step chunk)"],
 ["Previous-step action reconditioned?", "Yes (in-chunk context)", "Yes (in-chunk context)", "No (next step sees observation + instruction only)", "Within chunk yes; across control steps no (fresh chunk each step)"],
 ["Replanning frequency", "N/A (open-loop single plan per scene)", "N/A (open-loop single plan per scene)", "Every control step", "Every control step"],
 ["Temporal ensemble?", "No", "No", "No", "Yes: last 4 chunks, weights newest-first 0.574/0.258/0.116/0.052 (steps 2-4 = 42.6%)"],
 ["Perturbation size", "1 codebook token at first mismatch, GT-distance matched (median |d dist| 8.8 mm)", "First waypoint +0.2 / 0.5 m x 8 dirs (A/B); 0.5 m (C-E)", "x (or y) token +-8 / +-24 bins", "Step-1 translation token shifted by d=0.15/0.30 (token), d=0.30 (closed loop; perp_left, opposite), normalized space"],
 ["Intervention N", "365 A- units / 52 scenes (+1,043 A+ units)", "A/B 2,728 scenes x 16; C-E 2,398 units / 300 scenes / 28 logs", "Token 2,462 frames (4,741 x-units); closed loop 100 eps x 6 conds", "Token 18,975 units / 1,600 frames / 80 eps; closed loop 80 eps x 9 conds"],
 ["Natural/feedback amplification", "47.4% [38, 57] (A- & FDE5>3 m)", "63.4% [60.6, 66.7] (A- & FDE5>3 m)", "Within-step 92.6% [89.8, 94.4] (x, 8 bins); 81.2% (24 bins)", "Token A>=1: 22.1% [19.5, 24.7]; step-2/3/4 token change 55.6/36.0/27.9%"],
 ["Corrected amplification", "Recent-GT 7.1% [4, 10]; GT-history 4.1% [1, 8]", "recent_gt 1.3% [0.8, 1.8]; gt_history 0.0%", "0% (by construction: downstream sees natural token)", "recent_ref 13.7% [11.8, 15.8]; full_ref 13.5% [11.7, 15.4]"],
 ["Reverse amplification", "33.4% [26.3, 43.2] (reverse@emb; from 4.4%)", "81.9% [80.4, 84.2] (from 0.0%)", "= feedback (identical downstream tokens)", "18.6% [16.3, 20.9]; step-4 dev 0.057 (full_ref 0.010, normal 0.056)"],
 ["Correction absolute effect", "-40.3 pp [-50.3, -30.0], p=8e-42", "-62.2 pp [-65.6, -59.1], McNemar p=0.0 (underflow)", "Token: N/A (deterministic); task feedback-corrected +3 pp [-4, +9] p=0.65 (8), +5 pp [-4, +14] p=0.36 (24)", "Token: recent_ref-normal -8.3 pp [-9.3, -7.4], p=7e-261; task corrected-feedback +5.0 pp [-1.2, +12.5] p=0.29 (left), -1.2 pp [-13.8, +12.5] p=1 (opposite)"],
 ["Reverse absolute effect", "+29.0 pp [+21.1, +38.0], p=7e-31", "+81.9 pp [+80.4, +84.2]; D10 +40.5 m [+37.3, +43.9]", "Task reverse-natural -3 pp [-9, +3] p=0.58 (8), -8 pp [-15, 0] p=0.13 (24)", "Token: reverse-full_ref +5.1 pp [+4.4, +5.8], p=2e-162; task reverse-natural -1.2 pp [-11.2, +8.8] p=1 (opp), +5.0 pp [-3.8, +13.8] p=0.39 (left)"],
 ["Critical window", "1/2/3/4 step = 47/76/90/96% of full", "1/2/3/4 step = 24/39/91/95% of full", "N/A (no temporal axis)", "1 step 86% (RESULTS: 87%), 2-step-only 63%; >2 not measurable (chunk 4)"],
 ["Direction > magnitude effect", "+9.0 pp [+3.1, +18.7], p=6e-4", "+15.6 pp [+9.0, +24.3], p=2e-31", "N/A (not tested)", "+1.6 pp [+0.4, +2.8], p=7e-5 (weak); near_ref-ref +4.3 pp (identity-sensitive)"],
 ["Task-level effect?", "Open-loop trajectory failure: yes; closed loop not tested", "Open-loop A-: +24.8 pp [+19.4, +32.2] at 0.2 m (vs rerun 5.9%): yes", "No: all |diff| <= 8 pp, p >= 0.115, within rerun noise (20%)", "Context: no (feedback-corrected +1.2 pp [-12.5, +13.8], p=1, opposite); executed action: yes (corrected-natural -21.2 pp [-33.8, -7.5], p=0.006, opposite)"],
 ["Preregistration result", "N/A (discovery experiments; exp 32 protocol af2eb10)", "038a6d3: crit 1,2,4,5 + task pass; crit 3 fail -> Strong", "6b9e8ad (+ amendment 6d0375a); no Strong/Partial criteria defined", "f578eae: token (1)-(3) pass; closed-loop (1),(2) fail -> Partial"],
 ["Classification", "Reference (original finding)", "Strong replication", "Architectural control", "Partial replication (token Strong, task none)"],
]
with open(P + "paper_cross_model_table.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(H); w.writerows(R)
def tex(s):
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"), ("^", "\\^{}"), ("<=", "$\\le$"), (">=", "$\\ge$"), ("+-", "$\\pm$"), ("->", "$\\rightarrow$"), ("<1e-300", "$<10^{-300}$"), ("~", "$\\sim$")):
        s = s.replace(a, b)
    return s
L = ["% requires \\usepackage{booktabs,tabularx,graphicx}", "\\begin{table*}[t]", "\\centering", "\\scriptsize",
     "\\caption{Cross-model replication of previous-action temporal feedback. Values are rates with 95\\% CI in brackets; differences are paired "
     "(same scene/unit/episode under both conditions). CIs: cluster bootstrap, 2{,}000 resamples, seed 0 (AutoVLA and Impromptu: log clusters; "
     "OpenVLA closed loop: task clusters, 10 tasks; OpenVLA token level: task clusters, 1{,}000 resamples; SpatialVLA: episode clusters). "
     "Binary paired tests are exact McNemar; continuous paired tests Wilcoxon signed-rank. N per model: AutoVLA 365 A$-$ units / 52 scenes; "
     "Impromptu 2{,}728 scenes (A/B) and 2{,}398 units / 300 scenes (C--E); OpenVLA 100 episodes $\\times$ 6 conditions and 2{,}462 frames; "
     "SpatialVLA 80 episodes $\\times$ 9 conditions and 18{,}975 token units. ``Amplification'' is model-specific (driving: A$-$ and FDE$_5>3$\\,m; "
     "OpenVLA: a downstream dimension moves $\\ge|\\delta|$ bins; SpatialVLA: accumulated step-2--4 translation deviation $\\ge$ injected) and is not comparable across columns.}",
     "\\label{tab:cross_model}", "\\resizebox{\\textwidth}{!}{%",
     "\\begin{tabular}{p{2.6cm}p{3.6cm}p{3.6cm}p{3.6cm}p{4.2cm}}", "\\toprule",
     " & ".join(tex(h) for h in H) + " \\\\", "\\midrule"]
for r in R:
    L.append(" & ".join(tex(x) for x in r) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
open(P + "paper_cross_model_table.tex", "w").write("\n".join(L) + "\n")
print(len(R), "rows")
