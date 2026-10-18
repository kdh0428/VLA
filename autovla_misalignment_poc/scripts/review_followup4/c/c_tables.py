#!/usr/bin/env python
"""review_followup4 C: machine-generated LaTeX tables (2026-10-18). CPU only.

1. Regenerates the review_followup3 block of paper_quantitative_package/tables.tex
     % >>> BEGIN machine-generated review_followup3 (cp2_tables.py) >>>  ...  % <<< END machine-generated review_followup3 <<<
   by calling the unchanged functions of scripts/review_followup3/cp2/cp2_tables.py and applying the A4/A1 caption and
   label corrections listed in PATCHES (each must match exactly once; values are untouched; original CIs stay).
2. Writes a new block after it
     % >>> BEGIN machine-generated review_followup4 (c_tables.py) >>>  ...  % <<< END machine-generated review_followup4 <<<
   Table 12  AutoVLA B2 matched-prefix factorial (RF4.B2.*)
   Table 13  post-hoc corrected paired CIs: SpatialVLA P1-D per task (RS-Newcombe; MOVER for the two-task mean) and
             OpenVLA state level (Newcombe 10) next to the original sign-flip-inversion CIs (RF4.A1.*)
   Table 14  AutoVLA B1 numerical sensitivity of the paired contrasts (RF4.B1.stab.*)
Everything outside the two marker blocks is left untouched. Re-runnable (idempotent).

  nice -n 19 /root/miniforge3/envs/autovla/bin/python c_tables.py
"""
import importlib.util
import json
import re

PKG = "/root/VLA/paper_quantitative_package"
TEX = f"{PKG}/tables.tex"
CP2 = "/root/VLA/autovla_misalignment_poc/scripts/review_followup3/cp2/cp2_tables.py"
B3 = "% >>> BEGIN machine-generated review_followup3 (cp2_tables.py) >>>"
E3 = "% <<< END machine-generated review_followup3 <<<"
B4 = "% >>> BEGIN machine-generated review_followup4 (c_tables.py) >>>"
E4 = "% <<< END machine-generated review_followup4 <<<"

spec = importlib.util.spec_from_file_location("cp2_tables", CP2)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
R = T.R
n1, n2, pfmt = T.n1, T.n2, T.pfmt
MARK = " (corrected 2026-10-18, review\\_followup4)"

PATCHES = {
    "table8": [
        ("P = Hajek-weighted estimand over the 2{,}191 eligible scenes of the 64 P1-C logs: an \\emph{evaluation-population} "
         "estimand, not a deployment effect.",
         "P = Hajek-weighted estimand over the 2{,}191 eligible scenes of the 64 P1-C logs, estimated from 353 sampled "
         "scenes with design weights (24 drives): an \\emph{evaluation-frame} estimand, not a deployment effect; about 60\\% "
         "of the RN P effect comes from the non-failure (A+) stratum. F = failure stratum, 105 scenes / 18 drives" + MARK + "."),
    ],
    "table9": [
        ("E = small learned non-AR motion extrapolator", "L = small learned non-AR motion extrapolator"),
        ("Last column: exceeds every control (locked rule).}",
         "Last column: exceeds every control (locked rule). \\emph{all} = the 208 selected mechanism scenes (52 A$-$ + 156 "
         "A+), not the natural set or a population; a non-significant excess is not equivalence. L is unrelated to the P1-C "
         "target set E" + MARK + ".}"),
        ("CTRA-2 & E prefix & E prefix+ego & beyond all", "CTRA-2 & L prefix & L prefix+ego & beyond all"),
        ("\nall & ", "\nall (208 selected) & "),
    ],
    "table10": [
        ("pp, sign-flip-inversion 95\\% CI on initial-state clusters",
         "pp, original sign-flip-inversion 95\\% CI on initial-state clusters (kept for the record; invalid at the "
         "boundary for paired binary data; post-hoc corrected Rao--Scott Newcombe CIs in Table~\\ref{tab:rf4_ci}" + MARK + ")"),
    ],
    "table11": [
        ("the arms are not identical to HF \\texttt{generate} output.}",
         "the arms are not identical to HF \\texttt{generate} output. A verified code path is not a validation of an "
         "estimand or interval (review\\_followup4 A1); B1: batch shape alone changes 9\\% of free-run suffixes, attention "
         "backend / fp32 arithmetic about 25\\%" + MARK + ".}"),
    ],
}


def patched(name, s):
    for old, new in PATCHES.get(name, []):
        assert s.count(old) == 1, (name, old[:80], s.count(old))
        s = s.replace(old, new)
    return s


def pp(x):
    return n1(100 * x)


def ci_pp(ci):
    return f"[{pp(ci[0])}, {pp(ci[1])}]"


def old_end(x):
    """original sign-flip-inversion end; a value pinned at 0 by the tie tolerance is shown as such."""
    return "0 (pinned)" if abs(x) < 1e-9 else pp(x)


def table12():
    lines = []
    lab = {"C_GT": "$C_\\text{GT}$ = GT\\_NSRC $-$ GT\\_REF", "C_NAT": "$C_\\text{NAT}$ = NAT\\_NSRC $-$ NAT\\_REF",
           "INT": "INT = $C_\\text{GT}-C_\\text{NAT}$", "C_PROV": "$C_\\text{PROV}$ = GT\\_CSRC $-$ GT\\_NSRC"}
    for c in ("C_GT", "C_NAT", "INT", "C_PROV"):
        cells = []
        for est in ("F", "P"):
            r = R[f"RF4.B2.p1c.a1.severe.{c}.all.{est}.drive"]
            fl = r["fields"]
            dec = "rej." if fl["reject"] is True else "n.r."
            cells.append(f"{pp(r['value'])} {ci_pp(r['ci95'])} & {pfmt(r['p_raw'])} / {pfmt(r['p_adj'])} {dec}")
        dv = R[f"RF4.B2.dev.a1.severe.{c}.all.F.drive"]
        lines.append(f"{lab[c]} & " + " & ".join(cells) + f" & {pp(dv['value'])} {ci_pp(dv['ci95'])} \\\\")
    fF = R["RF4.B2.p1c.a1.severe.C_GT.all.F.drive"]["fields"]
    fP = R["RF4.B2.p1c.a1.severe.C_GT.all.P.drive"]["fields"]
    fD = R["RF4.B2.dev.a1.severe.C_GT.all.F.drive"]["fields"]
    io = R["RF4.B2.p1c.a1.severe.INT.older_values_differ.F.drive"]
    sP = R["RF4.B2.p1c.a2.severe.INT.all.P.drive"]
    return ("\\begin{table}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n"
            "\\caption{AutoVLA (driving, open-loop 5\\,s plan): matched-prefix factorial (review\\_followup4 B2; protocol "
            "committed before the main runs, 552b3ac; machine-generated from \\texttt{numbers.json} rows \\texttt{RF4.B2.*}). "
            "One edited prefix per unit-anchor: the older prefix is GT or the matched natural prefix (GT\\_/NAT\\_), the latest "
            "block a$-$1 holds GT (REF), the frozen Normal-source token (NSRC) or the frozen corrected-source token (CSRC); "
            "the edit is \\emph{retained} at fixed positions for all later steps (one-shot, not a release test). Outcome: "
            "severe (A$-$ and FDE$_{5\\,s}>3$\\,m), pp; scene-weighted; drive-cluster bootstrap 95\\% CI; exact drive "
            "sign-flip $p$ / Holm $p$ within the preregistered family ($m=4$; F $\\alpha=0.05$, P $\\alpha=0.025$); rej.\\ = "
            f"H0 rejected. F = P1-C failure stratum ({fF['n_scenes']} scenes / {fF['n_units']} units / {fF['n_clusters']} drives); "
            f"P = Hajek estimand over the B2 eligible domain E4 (t$^*\\le4$; {fP['n_scenes']} sampled scenes / {fP['n_units']} "
            f"units / {fP['n_clusters']} drives; weights recomputed), not the original 2{{,}}191-scene estimand. Dev column: "
            f"development A$-$ stratum ({fD['n_scenes']} scenes / {fD['n_clusters']} drives; descriptive). INT restricted to "
            f"units whose older value differs from GT: F {pp(io['value'])} {ci_pp(io['ci95'])}. Sensitivity anchor "
            f"(n$_\\text{{old}}=2$): P-level INT {pp(sP['value'])} {ci_pp(sP['ci95'])}, $p={pfmt(sP['p_raw'])}$ (not "
            "preregistered as a test). Every GT-containing cell is an oracle-conditioned diagnostic, not an online method; "
            "existing drives, not external validation; NSRC vs CSRC differences are value differences, not pure provenance.}\n"
            "\\label{tab:rf4_matched_prefix}\n\\begin{tabular}{lccccc}\n\\toprule\n"
            "Contrast & F: effect [95\\% CI] & F: $p$ / Holm & P (E4): effect [95\\% CI] & P: $p$ / Holm & dev A$-$ \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def table13():
    lines = []
    for cfg in ("r1e4", "r4e1"):
        for c, lab in (("R-N", "$\\delta_R$"), ("C-F", "$\\delta_C$")):
            cells = []
            for task in ("coke", "move_near"):
                r = R[f"RF4.A1.P0B.P1D.{cfg}.{c}.{task}.primary"]
                fl = r["fields"]
                cell = f"{pp(r['value'])} {ci_pp(r['ci95'])}"
                if task == "move_near":
                    cell += f" ({ci_pp(R[f'RF4.A1.P0B.P1D.{cfg}.{c}.{task}.physical']['ci95'])})"
                cells.append(cell)
                cells.append(f"{fl['new_p_equiv_m10_holm']:.2g} / {fl['new_p_equiv_m5_holm']:.2g}")
            pr = R[f"RF4.A1.P0B.P1D.{cfg}.{c}.pooled.primary"]
            lines.append(f"{cfg} & {lab} & " + " & ".join(cells) + f" & {pp(pr['value'])} {ci_pp(pr['ci95'])} \\\\")
    ov = []
    for rid, lab in (("P0E.succ.C-F_d8", "C$-$F 8 bins"), ("P0E.succ.C-F_d24", "C$-$F 24 bins"),
                     ("P0E.succ.C-N_d8", "C$-$N 8 bins"), ("P0E.succ.R-Nrep_d8", "R$-$N$_\\text{rep}$ 8 bins")):
        r = R[f"RF4.A1.{rid}"]
        fl = r["fields"]
        ov.append(f"OpenVLA & {lab} & \\multicolumn{{2}}{{c}}{{{fl['table_a11_b10_c01_d00']}: {pp(r['value'])}, original "
                  f"[{old_end(fl['old_ci95_lo'])}, {old_end(fl['old_ci95_hi'])}]}} & \\multicolumn{{2}}{{c}}{{corrected {ci_pp(r['ci95'])}, "
                  f"$p_\\text{{eq}}$ $\\pm10$ {fl['new_p_equiv_m10']:.2g} (unadj.)}} & $p={pfmt(r['p_raw'])}$ \\\\")
    return ("\\begin{table}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n"
            "\\caption{Post-hoc corrected paired success-rate intervals (review\\_followup4 A1; machine-generated from "
            "\\texttt{numbers.json} rows \\texttt{RF4.A1.*}; the original sign-flip-inversion intervals stay in Table~"
            "\\ref{tab:rf3_spatial} and in the original rows). Top: SpatialVLA-4B (manipulation, closed loop, SimplerEnv "
            "Google Robot), P1-D per-task direct effects, pp, Rao--Scott design-effect Newcombe-10 95\\% CI on initial-state "
            "clusters (move\\_near primary states; physical-layout CI in parentheses, 10 layouts); two-task mean by MOVER; "
            "TOST $p$ by inversion of the corrected interval, Holm within the original family ($m=4$), $\\pm10$ pp / $\\pm5$ "
            "pp. Bottom: OpenVLA-7B (manipulation, closed loop, LIBERO-Spatial; 100 initial states; paired table "
            "both/only-a/only-b/neither), Newcombe-10 95\\% CI; descriptive (intervention path not parity-verified). The "
            "difference-from-0 $p$ (sign-flip / exact McNemar at $\\delta=0$) is valid and unchanged. Method chosen by a "
            "coverage rule fixed before the real data were recomputed. Per-task OpenVLA intervals (10 states) are "
            "uninformative and descriptive only; a zero-width interval is never evidence of equality.}\n"
            "\\label{tab:rf4_ci}\n\\begin{tabular}{llccccc}\n\\toprule\n"
            "Config & Contrast & coke & $p_\\text{eq}$ $\\pm10$ / $\\pm5$ & move\\_near & $p_\\text{eq}$ $\\pm10$ / $\\pm5$ & "
            "two-task mean \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\midrule\n" + "\n".join(ov) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def table14():
    lines = []
    for c, lab in (("RN", "RN (Recent-GT $-$ Normal)"), ("FULL", "Full-GT $-$ Normal"), ("RV", "RV (Reverse-N $-$ Full-GT)"),
                   ("PERT", "P1-A OC $-$ U"), ("SELF_R", "P1-A OC$_\\text{refR}-$OC"), ("SELF_H", "P1-A OC$_\\text{refH}-$OC")):
        sc = R[f"RF4.B1.scale.{c}"]["fields"]
        cells = []
        mat = []
        for s in ("SB", "SE", "SF"):
            fl = R[f"RF4.B1.stab.{c}.{s}"]["fields"]
            cells.append(f"{fl['delta_severe']:+.3f} / {fl['delta_fde5_m']:+.2f}".replace("-", "$-$"))
            mat.append(bool(fl["material"]))
        lines.append(f"{lab} & " + f"{sc['mean_d_severe']:+.3f} / {sc['mean_d_fde5_m']:+.2f}".replace("-", "$-$") + " & " + " & ".join(cells)
                     + f" & {'yes' if any(mat) else 'no'} \\\\")
    return ("\\begin{table}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\caption{AutoVLA (driving, open-loop 5\\,s plan): numerical sensitivity of paired contrasts (review\\_followup4 "
            "B1; machine-generated from \\texttt{numbers.json} rows \\texttt{RF4.B1.*}). 56 development scenes (Stage-0 rule, "
            "take 20 per t$^*$ stratum), T = 0.01, seed 0, RTX 5090. S0 = reference (bf16, sdpa, experiment batch); SB = batch "
            "shape only; SE = eager attention (co-changes mask construction and softmax precision); SF = fp32 arithmetic on the "
            "bf16-rounded weights. Entries: change of the unit-mean paired contrast vs S0, severe (proportion) / FDE$_{5\\,s}$ "
            "(m); first column = S0 contrast. Material = pre-fixed rule (sign change, $|\\Delta\\text{severe}|>0.05$ or "
            "$|\\Delta\\text{FDE}|>25\\%$). The same-process repeat is bitwise. A sensitivity of the numerical implementation, "
            "not a new effect estimate; numeric variation is reported next to the effects and never subtracted.}\n"
            "\\label{tab:rf4_numerics}\n\\begin{tabular}{lccccc}\n\\toprule\n"
            "Contrast & S0 & SB & SE & SF & material \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main():
    rf3 = "\n".join([B3, "% Generated from numbers.json; do not edit by hand. Regenerate with cp2_tables.py "
                         "(captions/labels corrected 2026-10-18 by review_followup4/c/c_tables.py).",
                     "% ---------------------------------------------------------------- Table 8", patched("table8", T.table8()),
                     "% ---------------------------------------------------------------- Table 9", patched("table9", T.table9()),
                     "% ---------------------------------------------------------------- Table 10", patched("table10", T.table10()),
                     "% ---------------------------------------------------------------- Table 11", patched("table11", T.table11()),
                     E3])
    rf4 = "\n".join([B4, "% Generated from numbers.json (RF4.* rows); do not edit by hand. Regenerate with "
                         "scripts/review_followup4/c/c_tables.py (2026-10-18).",
                     "% ---------------------------------------------------------------- Table 12", table12(),
                     "% ---------------------------------------------------------------- Table 13", table13(),
                     "% ---------------------------------------------------------------- Table 14", table14(), E4])
    s = open(TEX).read()
    assert B3 in s and E3 in s
    s = re.sub(re.escape(B3) + ".*?" + re.escape(E3), lambda m: rf3, s, flags=re.S)
    if B4 in s:
        s = re.sub(re.escape(B4) + ".*?" + re.escape(E4), lambda m: rf4, s, flags=re.S)
    else:
        s = s.replace(E3, E3 + "\n\n" + rf4, 1)
    open(TEX, "w").write(s)
    assert s.count("\\begin{table}") == s.count("\\end{table}")
    assert s.count("\\begin{tabular}") == s.count("\\end{tabular}")
    print(rf4)


if __name__ == "__main__":
    main()
