#!/usr/bin/env python
"""review_followup5: numbers.json RF5 rows + tables.tex review_followup5 block (2026-10-18). CPU only. Idempotent.

numbers.json: adds RF5.AS.* (A_STATS binary rows), RF5.B2C.* (B2 cluster axes), RF5.BN.* (B_NUMERIC), RF5.C / RF5.D status
rows, `_meta.review_followup5`, and a non-destructive `rf5_note` on affected existing rows (values unchanged).
tables.tex: regenerates the review_followup3 block with one more caption note on Table 8 (P-level CIs not validated) via
scripts/review_followup4/c/c_tables.py, keeps the review_followup4 block, and writes
  % >>> BEGIN machine-generated review_followup5 (c5_tables.py) >>> ... % <<< END machine-generated review_followup5 <<<
with Table 15 (AutoVLA validated intervals), Table 16 (B2 cluster axes), Table 17 (B_NUMERIC).

  nice -n 19 /root/miniforge3/envs/autovla/bin/python c5_tables.py
"""
import importlib.util
import json
import math
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
R5 = f"{POC}/outputs/review_followup5"
PKG = "/root/VLA/paper_quantitative_package"
NJ, TEX = f"{PKG}/numbers.json", f"{PKG}/tables.tex"
DRIVE = {"domain": "driving", "model": "AutoVLA (Qwen2.5-VL-3B)", "eval_mode": "open-loop 5 s plan (navtest)"}
ROUND = "review_followup5 (2026-10-18)"
B5 = "% >>> BEGIN machine-generated review_followup5 (c5_tables.py) >>>"
E5 = "% <<< END machine-generated review_followup5 <<<"


def clean(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def lst(s):
    if isinstance(s, str) and s.startswith("["):
        return [None if x.strip() in ("nan", "") else float(x) for x in s.strip("[]").split(",")]
    return s


def row(rid, **kw):
    b = {"block": "RF5", "round": ROUND, "status_caveat": "", "status_original": "", "fields": {}, **DRIVE}
    b.update(kw)
    return rid, b


def numbers():
    J = json.load(open(NJ))
    R = J["rows"]
    for k in [k for k in R if k.startswith("RF5.")]:
        del R[k]
    for v in R.values():
        v.pop("rf5_note", None)
    new = []
    a = pd.read_csv(f"{R5}/A_STATS/AUTOVLA_CI_AUDIT.csv")
    for i, r in a[a.outcome_type.str.startswith("binary")].iterrows():
        new.append(row(f"RF5.AS.{i:03d}", item=f"A_STATS: {r.source.split('/')[-1]} | {r.row}", value=clean(r.estimate),
                       unit="proportion (paired difference)", ci95=lst(r.new_ci95), ci95_method=r.new_method,
                       p_raw=clean(r["p_signflip_delta0 (unchanged, valid)"]), p_adj=clean(r.holm_original), family=None, code_path=None,
                       status="VERIFIED_MACHINERY_MANIFEST_SCOPE" if r.dataset != "SUB" else "VERIFIED_MACHINERY_MANIFEST_SCOPE",
                       analysis_type="POST-HOC CORRECTION (review_followup5 A_STATS); interval chosen by a coverage study on the real cluster structure",
                       package_use="usable" if r.new_method == "cluster_t" else "estimate and p only (interval NR)",
                       source_file="outputs/review_followup5/A_STATS/AUTOVLA_CI_AUDIT.csv", source_key=f"row {i}",
                       fields={k: clean(r[k]) for k in ("source", "row", "stratum", "cluster", "G", "old_ci95_signflip_inv",
                                                        "bootstrap_ci95 (reported primary)", "old_degeneracy", "margin",
                                                        "old_margin_verdict", "new_margin_verdict", "old_decision", "new_decision", "change")}))
    b = pd.read_csv(f"{R5}/A_B2_CLUSTER/B2_CLUSTER_RESULTS.csv")
    for i, r in b.iterrows():
        new.append(row(f"RF5.B2C.{r.contrast}.{r.estimand}.{r.cluster_axis}", item=f"B2 {r.contrast} ({r.estimand}) on {r.cluster_axis} clusters",
                       value=clean(r.theta), unit="proportion", ci95=lst(r.boot_ci95) if isinstance(r.boot_ci95, str) else None,
                       ci95_method=("drive / day-vehicle bootstrap 95% (reference; see A_STATS)" if r.cluster_axis != "component" else "NR"),
                       p_raw=clean(r.p_signflip_exact), p_adj=clean(r.holm_p), family=(r.family if isinstance(r.family, str) and r.family else None),
                       code_path="B2", status="VERIFIED_IN_RUN_ALL_UNITS",
                       analysis_type="review_followup5 A_B2_CLUSTER re-analysis (no new GPU run); oracle-conditioned diagnostic",
                       package_use="usable (state cluster axis)", source_file="outputs/review_followup5/A_B2_CLUSTER/B2_CLUSTER_RESULTS.csv",
                       source_key=f"row {i}", fields={k: clean(r[k]) for k in ("G", "G_informative", "p_floor", "alpha", "reject_H0", "n_scenes", "n_units")}))
    n = pd.read_csv(f"{R5}/B_NUMERIC/B_NUMERIC_RESULTS.csv")
    for i, r in n.iterrows():
        new.append(row(f"RF5.BN.{r.setting}.{r.group}.{r.quantity}", item=f"B_NUMERIC {r.quantity} ({r.group}, setting {r.setting})",
                       value=clean(r.est_scene), unit="m", ci95=[clean(r.ci95_lo), clean(r.ci95_hi)], ci95_method="drive-cluster bootstrap 95% (B = 10,000)",
                       p_raw=clean(r.p_signflip), p_adj=clean(r.p_holm), family=(r.family if isinstance(r.family, str) and r.family else None),
                       code_path="P1A", status="VERIFIED_MACHINERY_MANIFEST_SCOPE",
                       status_caveat="P1A path (Stage-0 gate PASS); this run: smoke PASS, S0 = stored raw 14,530/14,530 rows",
                       analysis_type="review_followup5 B_NUMERIC (GPU re-inference S0 / SE / SF; protocol locked before the run)",
                       package_use="usable", source_file="outputs/review_followup5/B_NUMERIC/B_NUMERIC_RESULTS.csv", source_key=f"row {i}",
                       fields={k: clean(r[k]) for k in ("n_units", "n_scenes", "n_drives", "role", "baseline")}))
    new.append(row("RF5.C.status", item="C_DISJOINT independent day/vehicle data", value="BLOCKED_NO_DISJOINT_DATA", unit="status",
                   status="NOT_RUN", analysis_type="inventory + overlap audit + power plan", package_use="not available",
                   source_file="outputs/review_followup5/C_DISJOINT/C_DISJOINT_REPORT.md", source_key="Status"))
    new.append(row("RF5.D.status", item="D_RELEASE GT retain vs release", value="NOT_RUN (optional; protocol draft only)", unit="status",
                   status="NOT_RUN", analysis_type="protocol draft", package_use="not available",
                   source_file="outputs/review_followup5/D_RELEASE/RELEASE_PROTOCOL_DRAFT.md", source_key="whole file"))
    for rid, v in new:
        R[rid] = v
    notes = {}
    for k in R:
        if re.match(r"P0A\.P1C\.(RN|RV|DM)\.P$", k) or (k.startswith("RF4.B2.p1c") and ".P." in k):
            notes[k] = ("review_followup5 A_STATS: no interval method reached nominal coverage for the weighted P structure; "
                        "report the estimate and the delta = 0 sign-flip p; the stored bootstrap CI is not validated.")
        elif re.match(r"P0A\.(P1C\.(RN|RV|DM)\.F|DEV\.(RN|RV|DM)\.Am)$", k):
            notes[k] = "review_followup5 A_STATS: validated interval = cluster-robust t (see RF5.AS.* / Table 15)."
    for k, t in notes.items():
        R[k]["rf5_note"] = t
    J["_meta"]["review_followup5"] = {"created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                      "generator": f"{POC}/scripts/review_followup5/c5_tables.py", "n_rf5_rows": len(new),
                                      "n_rows_with_rf5_note": len(notes), "protocol_sha256": json.load(open(f"{R5}/PROTOCOL_LOCK.json"))["sha256"]}
    json.dump(J, open(NJ, "w"), indent=1, default=str)
    import csv
    with open(f"{PKG}/numbers_sources.csv", newline="") as fh:
        base = list(csv.reader(fh))
    hdr, body = base[0], [x for x in base[1:] if not x[0].startswith("RF5.")]
    with open(f"{PKG}/numbers_sources.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(hdr); w.writerows(body)
        for rid, r in new:
            ci = r.get("ci95") or [None, None]
            w.writerow([rid, r["block"], r["domain"], r["model"], r["eval_mode"], r.get("code_path"), r.get("value"), r.get("unit"),
                        ci[0], ci[1], r.get("ci95_method"), r.get("p_raw"), r.get("p_adj"), r.get("family"), r.get("status"),
                        r.get("package_use"), r.get("analysis_type"), r.get("source_file"), r.get("source_key")])
    print("RF5 rows", len(new), "notes", len(notes))


def pp(x):
    return ("$-$" if x < 0 else "+") + f"{abs(100 * x):.1f}"


def m2(x):
    return ("$-$" if x < 0 else "+") + f"{abs(x):.2f}"


def pf(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if p < 1e-3:
        a, e = f"{p:.1e}".split("e")
        return f"${a}\\times10^{{{int(e)}}}$"
    return f"{p:.3g}"


def tables():
    a = pd.read_csv(f"{R5}/A_STATS/AUTOVLA_CI_AUDIT.csv")
    pop = a[a.source.str.contains("POPULATION_EFFECTS")].set_index("row")
    L15 = []
    for rid, lab in (("P0A.DEV.RN.Am", "dev A$-$ RN"), ("P0A.DEV.RV.Am", "dev A$-$ RV"), ("P0A.DEV.DM.Am", "dev A$-$ DM"),
                     ("P0A.P1C.RN.F", "P1-C F RN"), ("P0A.P1C.RV.F", "P1-C F RV"), ("P0A.P1C.DM.F", "P1-C F DM"),
                     ("P0A.P1C.RN.P", "P1-C P RN"), ("P0A.P1C.RV.P", "P1-C P RV"), ("P0A.P1C.DM.P", "P1-C P DM")):
        r = pop.loc[rid]
        bt = lst(r["bootstrap_ci95 (reported primary)"]); nc = lst(r.new_ci95)
        newc = f"[{pp(nc[0])}, {pp(nc[1])}]" if nc and nc[0] is not None and not math.isnan(nc[0]) else "NR"
        L15.append(f"{lab} & {pp(r.estimate)} & [{pp(bt[0])}, {pp(bt[1])}] & {newc} & {pf(r['p_signflip_delta0 (unchanged, valid)'])} & {int(r.G)} \\\\")
    t15 = ("\\begin{table}[t]\n\\centering\n\\small\n\\caption{AutoVLA (driving, open-loop 5\\,s plan): validated intervals for the "
           "binary failure-outcome contrasts (review\\_followup5 A\\_STATS, post hoc; machine-generated from \\texttt{numbers.json} rows "
           "\\texttt{RF5.AS.*}). pp; scene-weighted. A coverage study on the real cluster structures (written before the rows were "
           "recomputed) selected the cluster-robust $t$ interval ($G-1$ df) for the development A$-$ stratum and the P1-C failure stratum; "
           "the drive percentile bootstrap reported so far under-covered (simulated minimum 0.83--0.91 for 95\\%). For the weighted "
           "evaluation-frame estimand P no method reached nominal coverage (NR): report the estimate and the drive sign-flip $p$ at 0 "
           "(exact enumeration under the sign-symmetry null), which also governs the preregistered replication decisions (unchanged).}\n"
           "\\label{tab:rf5_intervals}\n\\begin{tabular}{lccccc}\n\\toprule\nContrast & Effect & bootstrap 95\\% (old) & cluster-$t$ 95\\% "
           "(validated) & $p$ (sign-flip) & drives \\\\\n\\midrule\n" + "\n".join(L15) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    b = pd.read_csv(f"{R5}/A_B2_CLUSTER/B2_CLUSTER_RESULTS.csv")
    b = b[b.family_member]
    L16 = []
    for c in ("C_GT", "C_NAT", "INT", "C_PROV"):
        for est in ("F", "P"):
            cells = []
            for ax in ("drive", "day_vehicle", "component"):
                r = b[(b.contrast == c) & (b.estimand == est) & (b.cluster_axis == ax)].iloc[0]
                cells.append(f"{int(r.G)} / {pf(r.p_signflip_exact)} / {pf(r.holm_p)}")
            L16.append(f"{c.replace('_', chr(92) + '_')} & {est} & {pp(r.theta)} & " + " & ".join(cells) + " \\\\")
    t16 = ("\\begin{table}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\caption{AutoVLA B2 matched-prefix contrasts "
           "under three cluster axes (review\\_followup5 A\\_B2\\_CLUSTER; machine-generated from \\texttt{RF5.B2C.*}). Primary anchor, "
           "severe, scene-weighted; entries: clusters $G$ / exact sign-flip $p$ at 0 / Holm $p$ within the original family ($m=4$; F "
           "$\\alpha=0.05$, P $\\alpha=0.025$). Components: connected day--vehicle components of the 312 P1-C scenes (6); the smallest "
           "attainable $p$ is $2/2^6=0.031$ and the smallest attainable Holm $p$ is 0.125, so no contrast can be significant at that "
           "level; all six components share the sign of C\\_GT and C\\_NAT in F. INT is an estimate of the difference, not evidence of "
           "equality; the earlier factor is one context position. Ground-truth-conditioned diagnostic on re-run existing drives.}\n"
           "\\label{tab:rf5_b2_clusters}\n\\begin{tabular}{llcccc}\n\\toprule\nContrast & Est. & Effect (pp) & drive & day--vehicle & "
           "component \\\\\n\\midrule\n" + "\n".join(L16) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    n = pd.read_csv(f"{R5}/B_NUMERIC/B_NUMERIC_RESULTS.csv")
    L17 = []
    nm = {"Geo": "Geo (coord.\\ accumulation)", "CM": "CM", "CV0": "CV0", "CTRV-K3": "CTRV-K3", "CTRA-2": "CTRA-2", "E_prefix": "learned extrapolator"}
    for bl in ("Geo", "CM", "CV0", "CTRV-K3", "CTRA-2", "E_prefix"):
        cells = []
        for s in ("S0", "SE", "SF"):
            r = n[(n.setting == s) & (n.group == "A-") & (n.quantity == f"E_{bl}")].iloc[0]
            c = f"{m2(r.est_scene)} [{m2(r.ci95_lo)}, {m2(r.ci95_hi)}]"
            if s != "S0":
                c += f" ({pf(r.p_holm)})"
            cells.append(c)
        L17.append(f"{nm[bl]} & " + " & ".join(cells) + " \\\\")
    dl = [n[(n.setting == s) & (n.group == "A-") & (n.quantity == "Delta")].iloc[0] for s in ("SE", "SF")]
    L17.append(f"\\midrule\nsetting change $\\Delta$ (paired) & -- & {m2(dl[0].est_scene)} [{m2(dl[0].ci95_lo)}, {m2(dl[0].ci95_hi)}] & "
               f"{m2(dl[1].est_scene)} [{m2(dl[1].ci95_lo)}, {m2(dl[1].ci95_hi)}] \\\\")
    t17 = ("\\begin{table}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\caption{AutoVLA (driving, open-loop): ground-truth-free "
           "endpoint excess in failure-selected (A$-$) scenes re-inferred under alternative computation settings (review\\_followup5 "
           "B\\_NUMERIC; machine-generated from \\texttt{RF5.BN.*}). Family G, 364 units / 52 scenes / 13 drives (all 1{,}453 units / 208 "
           "selected scenes were run). S0 = original computation (bf16, SDPA, batch 10; reproduced token-for-token), SE = eager "
           "attention, SF = fp32 arithmetic on the bf16-rounded weights; same units, reference, perturbation, batch and random numbers. "
           "Entries: $\\Delta_{\\text{model}}-\\Delta_{\\text{baseline}}$ (m, endpoint displacement), drive bootstrap 95\\% CI, Holm $p$ "
           "over the 12 SE/SF tests in parentheses. Last row: paired change of the excess vs S0 (identical for all baselines); not an "
           "equivalence test. Baseline responses are fixed (computed from the reference and the perturbation only).}\n"
           "\\label{tab:rf5_numerics}\n\\begin{tabular}{lccc}\n\\toprule\nBaseline & S0 & SE & SF \\\\\n\\midrule\n" + "\n".join(L17) +
           "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    blk = "\n".join([B5, "% Generated from review_followup5 outputs / numbers.json RF5 rows; regenerate with scripts/review_followup5/c5_tables.py.",
                     "% ---------------------------------------------------------------- Table 15", t15,
                     "% ---------------------------------------------------------------- Table 16", t16,
                     "% ---------------------------------------------------------------- Table 17", t17, E5])
    # rf3 block: one more caption note on Table 8 (via c_tables.py)
    spec = importlib.util.spec_from_file_location("c4t", f"{POC}/scripts/review_followup4/c/c_tables.py")
    C4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(C4)
    C4.PATCHES["table8"].append(("F = failure stratum, 105 scenes / 18 drives (corrected 2026-10-18, review\\_followup4).",
                                 "F = failure stratum, 105 scenes / 18 drives (corrected 2026-10-18, review\\_followup4). The bootstrap "
                                 "CIs are not validated for the weighted P rows (review\\_followup5 A\\_STATS; validated cluster-$t$ "
                                 "intervals for F and dev A$-$ in Table~\\ref{tab:rf5_intervals})."))
    C4.main()
    s = open(TEX).read()
    if B5 in s:
        s = re.sub(re.escape(B5) + ".*?" + re.escape(E5), lambda m: blk, s, flags=re.S)
    else:
        s = s.rstrip("\n") + "\n\n" + blk + "\n"
    open(TEX, "w").write(s)
    assert s.count("\\begin{table}") == s.count("\\end{table}") and s.count("\\begin{tabular}") == s.count("\\end{tabular}")
    print(blk[-1500:])


if __name__ == "__main__":
    numbers()
    tables()
