#!/usr/bin/env python
"""Machine-generated LaTeX tables from paper_quantitative_package/numbers.json (review_followup3, 2026-10-15).

Writes the block between the markers
  % >>> BEGIN machine-generated review_followup3 (cp2_tables.py) >>>
  % <<< END machine-generated review_followup3 <<<
at the end of paper_quantitative_package/tables.tex (replaced in place on re-run; nothing else in the file is touched).
Tables: 8 AutoVLA population / conditional effects; 9 motion-extrapolation controls (P2); 10 SpatialVLA P1-D per-task
direct effects; 11 verification (no-op / restore) summary. Values at an exact rounding tie are printed with 2 decimals.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python cp2_tables.py
"""
import json
import re

PKG = "/root/VLA/paper_quantitative_package"
TEX = f"{PKG}/tables.tex"
BEGIN = "% >>> BEGIN machine-generated review_followup3 (cp2_tables.py) >>>"
END = "% <<< END machine-generated review_followup3 <<<"
R = json.load(open(f"{PKG}/numbers.json"))["rows"]


def n1(x):
    """one decimal, two decimals at an exact .x5 tie (avoid half-even ambiguity)."""
    if abs(x * 20 - round(x * 20)) < 1e-9 and abs(x * 10 - round(x * 10)) > 1e-9:
        return f"{x:+.2f}".replace("-", "$-$") if x else "0.00"
    s = f"{x:+.1f}"
    return "0.0" if s in ("+0.0", "-0.0") else s.replace("-", "$-$")


def n2(x):
    return f"{x:+.2f}".replace("-", "$-$")


def pfmt(p):
    if p is None:
        return "--"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{p:.2g}" if p < 0.1 else f"{p:.2f}"


def eff_pp(rid, scale=100.0):
    r = R[rid]
    lo, hi = r["ci95"]
    return f"{n1(scale * r['value'])} [{n1(scale * lo)}, {n1(scale * hi)}]"


def table8():
    rows = []
    for coh, est, lab in (("DEV", "Am", "dev A$-$ (13 drives)"), ("P1C", "F", "P1-C F (new drives)"),
                          ("P1C", "P", "P1-C P (eval.-population)")):
        for c in ("RN", "RV", "DM"):
            rid = f"P0A.{coh}.{c}.{est}"
            r = R[rid]
            f = r["fields"]
            rows.append(f"{lab} & {c} & {eff_pp(rid)} & {pfmt(r['p_raw'])} & {pfmt(r['p_adj'])} & "
                        f"{f['n_scenes']} / {f['n_units']} & {f['n_drives']} \\\\")
    prev = []
    for rid, lab in (("P0A.PREV.DEV.Aminus.2747.drive", "dev natural, A$-$"),
                     ("P0A.PREV.P1C.Aminus.valid.drive", "P1-C stage 1, A$-$")):
        if rid in R:
            r = R[rid]
            f = r["fields"]
            prev.append(f"{lab}: {f['events']}/{f['n']} = {100 * r['value']:.2f}\\% [{100 * r['ci95'][0]:.2f}, {100 * r['ci95'][1]:.2f}]")
    return ("\\begin{table}[t]\n\\centering\n\\small\n"
            "\\caption{AutoVLA (driving, open-loop 5\\,s plan): conditional and evaluation-population effects "
            "(review\\_followup3 P0-A; machine-generated from \\texttt{numbers.json} rows \\texttt{P0A.*}). "
            "RN = Recent-GT$-$Normal (repeated sliding correction), RV = Reverse@emb$-$Full (Normal-branch token injection), "
            "DM = direction$-$magnitude error. pp, scene-weighted paired difference, drive-cluster bootstrap 95\\% CI, exact drive "
            "sign-flip $p$; Holm within the original family. P = Hajek-weighted estimand over the 2{,}191 eligible scenes of the "
            "64 P1-C logs: an \\emph{evaluation-population} estimand, not a deployment effect. Code paths verified by the "
            "no-op/cache gate on the verification manifests (H7, H8, H32); runs outside the manifests are not retro-validated. "
            "A$-$ frequency (evaluation set, not deployment): " + "; ".join(prev) + ".}\n"
            "\\label{tab:rf3_population}\n\\begin{tabular}{llcccc c}\n\\toprule\n"
            "Cohort & Contrast & Effect [95\\% CI] & $p$ & Holm $p$ & scenes / units & drives \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def table9():
    comps = [("CM", "P2A.excess.G.{g}.end.CM"), ("CV0", "P2A.excess.G.{g}.end.CV0"),
             ("CTRV-K3", "P2A.excess.G.{g}.end.CTRV-K3"), ("CTRA-2", "P2A.excess.G.{g}.end.CTRA-2"),
             ("E prefix", "P2B.excess.G.{g}.end.E_prefix.primary"), ("E prefix+ego", "P2B.excess.G.{g}.end.E_prefix_ego.primary")]
    lines = []
    for g in ("A-", "A+", "all"):
        cells = []
        for _, tmpl in comps:
            r = R[tmpl.format(g=g)]
            lo, hi = r["ci95"]
            star = "$^{*}$" if (r["p_adj"] is not None and r["p_adj"] < 0.05 and (lo > 0 or hi < 0)) else ""
            cells.append(f"{n2(r['value'])} [{n2(lo)}, {n2(hi)}]{star}")
        verdict = "yes" if R[f"P2.verdict.{g}"]["value"] else "no"
        lines.append(f"{g.replace('-', '$-$')} & " + " & ".join(cells) + f" & {verdict} \\\\")
    return ("\\begin{table}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n"
            "\\caption{AutoVLA (driving, open-loop): endpoint response beyond motion extrapolation "
            "(review\\_followup3 P2; post hoc on P1-A rollouts, GT-free family G; machine-generated from \\texttt{numbers.json} "
            "rows \\texttt{P2A.*}, \\texttt{P2B.*}, \\texttt{P2.verdict.*}). Entries: $\\Delta_{\\text{model}}-\\Delta_{\\text{control}}$ "
            "in m (M1 endpoint displacement), drive-cluster bootstrap 95\\% CI; $^{*}$ Holm $p<0.05$ and CI excludes 0 "
            "(kinematic: $m=4$; learned: $m=2$). CM = the original repeat-p constant-motion baseline, a body-frame SE(2) repeat "
            "(already constant turn rate); CV0 = zero-turn global velocity; CTRV-K3 = turn rate and speed averaged over the "
            "last 3 steps; CTRA-2 = constant turn rate and acceleration; E = small learned non-AR motion extrapolator (MLP, "
            "3 seeds, trained on drive-disjoint OpenScene trainval GT; not a non-AR version of the VLA). Last column: exceeds "
            "every control (locked rule).}\n\\label{tab:rf3_motion}\n"
            "\\begin{tabular}{lccccccc}\n\\toprule\nStratum & CM & CV0 & CTRV-K3 & CTRA-2 & E prefix & E prefix+ego & beyond all \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def table10():
    lines = []
    for cfg in ("r1e4", "r4e1"):
        for c, lab in (("R-N", "$\\delta_R$"), ("C-F", "$\\delta_C$")):
            cells = []
            for task in ("coke", "move_near"):
                r = R[f"P0B.P1D.{cfg}.{c}.{task}.primary"]
                f = r["fields"]
                lo, hi = f["ci95_signflip_inv_pp"]
                cell = f"{n1(f['value_pp'])} [{n1(lo)}, {n1(hi)}]"
                if task == "move_near":
                    ph = R[f"P0B.P1D.{cfg}.{c}.{task}.physical"]["fields"]["ci95_signflip_inv_pp"]
                    cell += f" ([{n1(ph[0])}, {n1(ph[1])}])"
                cells.append(cell)
                cells.append(f"{f['p_equiv_m10_holm']:.2g} / {f['p_equiv_m5_holm']:.2g}")
            pooled = R[f"P0B.P1D.{cfg}.{c}.pooled.primary"]["fields"]
            lo, hi = pooled["ci95_signflip_inv_pp"]
            lines.append(f"{cfg} & {lab} & " + " & ".join(cells) + f" & {n1(pooled['value_pp'])} [{n1(lo)}, {n1(hi)}] \\\\")
    return ("\\begin{table}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n"
            "\\caption{SpatialVLA-4B (manipulation, closed loop, SimplerEnv Google Robot): per-task success direct effects, P1-D "
            "(review\\_followup3 P0-B, post hoc; machine-generated from \\texttt{numbers.json} rows \\texttt{P0B.P1D.*}). pp, "
            "sign-flip-inversion 95\\% CI on initial-state clusters (move\\_near: primary states; physical-layout CI in parentheses, "
            "10 layouts). TOST $p$ Holm-adjusted within the original family ($m=4$), $\\pm10$ pp (original margin) / $\\pm5$ pp "
            "(post-hoc). Per-task effects are reported in full; the two-task mean (last column) does not replace them. "
            "Restore machinery verified on 20 audit units (split-fingerprint null-arm check 20/20); P1-D runs outside the audit "
            "units are not retro-validated.}\n\\label{tab:rf3_spatial}\n"
            "\\begin{tabular}{llccccc}\n\\toprule\nConfig & Contrast & coke & $p_\\text{eq}$ $\\pm10$ / $\\pm5$ & move\\_near & "
            "$p_\\text{eq}$ $\\pm10$ / $\\pm5$ & two-task mean \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def table11():
    ver = {k.split(".")[-1]: R[k]["value"] for k in R if k.startswith("P1.AV.verdict.")}
    fails = {}
    for k, r in R.items():
        if k.startswith("P1.AV.") and not k.startswith("P1.AV.verdict"):
            p = r["code_path"]
            fails[p] = fails.get(p, 0) + int(r["value"])
    av = " ".join(f"{p}: {ver[p]} ({fails.get(p, 0)} failing rows);" for p in ("H7", "H8", "H9", "P1A", "H32", "P1B"))

    def sv(ctrl, col):
        t, n = R.get(f"P1.SV.{ctrl}.{col}_true"), R.get(f"P1.SV.{ctrl}.{col}_n")
        return f"{t['value']}/{n['value']}" if t and n else "--"
    rows = [("C1 hook installed, no edit", sv("C1", "pass")), ("C5a same-process rerun", sv("C5a", "pass")),
            ("C5b separate-process rerun", sv("C5b", "pass")), ("C3 / P1-C restore (branch N at $T_0$)", sv("C3_restore", "pass")),
            ("P1-C branch restore F/C/R up to $T_0$", sv("P1C_branch_restore", "pass")),
            ("C2 null arm, strict combined hash (historical)", sv("C2", "pass_strict_prefixed_rule")),
            ("C2 null arm, post-hoc A5 policy-state check (historical)", sv("C2", "pass_policy_state_A5")),
            ("C2 null arm, split fingerprint (binding)", sv("C2_A5_split", "pass")),
            ("C4 analogue: B=1 decode = batch-8 chunk (report)", sv("C4_t0_report", "b1_equal_batch8")),
            ("C4 analogue: HF generate = batch-8 chunk (report)", sv("C4_t0_report", "generate_equal_batch8"))]
    body = "\n".join(f"{a} & {b} \\\\" for a, b in rows)
    return ("\\begin{table}[t]\n\\centering\n\\small\n"
            "\\caption{Intervention-machinery verification (review\\_followup3 P1; machine-generated from \\texttt{numbers.json} "
            "rows \\texttt{P1.*}). AutoVLA no-op/cache gate (controls c1--c5, greedy and $T=0.01$, $T_\\text{exact}=0$, "
            "$T_\\text{equiv}=0.75$ logits): " + av + " Stage-0 manifest 21 path-units (H7/H8/H9/P1A) and O0 extension 18 "
            "path-units (H32/P1B). SpatialVLA restore audit below (20 units = 2 tasks $\\times$ 5 states $\\times$ 2 configs; "
            "tolerance 0). Impromptu and OpenVLA intervention paths were not parity-verified. Runs outside the verification "
            "manifests are not retro-validated; the arms are not identical to HF \\texttt{generate} output.}\n"
            "\\label{tab:rf3_verification}\n\\begin{tabular}{lc}\n\\toprule\nSpatialVLA control & pass / units \\\\\n\\midrule\n"
            + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main():
    block = "\n".join([BEGIN, "% Generated from numbers.json; do not edit by hand. Regenerate with cp2_tables.py.",
                       "% ---------------------------------------------------------------- Table 8", table8(),
                       "% ---------------------------------------------------------------- Table 9", table9(),
                       "% ---------------------------------------------------------------- Table 10", table10(),
                       "% ---------------------------------------------------------------- Table 11", table11(), END])
    s = open(TEX).read()
    if BEGIN in s:
        s = re.sub(re.escape(BEGIN) + ".*?" + re.escape(END), lambda m: block, s, flags=re.S)
    else:
        s = s.rstrip("\n") + "\n\n" + block + "\n"
    open(TEX, "w").write(s)
    assert s.count("\\begin{table}") == s.count("\\end{table}")
    assert s.count("\\begin{tabular}") == s.count("\\end{tabular}")
    print(block)


if __name__ == "__main__":
    main()
