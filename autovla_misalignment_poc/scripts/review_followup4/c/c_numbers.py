#!/usr/bin/env python
"""review_followup4 C: extend paper_quantitative_package/numbers.json with review_followup4 rows (2026-10-18).

CPU only. Reads the committed review_followup4 outputs (never modified) and the existing numbers.json (written by
scripts/review_followup3/cp2/cp2_numbers.py), then

  * appends rows with ids  RF4.A1.* (post-hoc corrected paired CIs, A1), RF4.A2.* (window semantics), RF4.A3.* (Impromptu
    parity diagnosis), RF4.A4.* (cohort / estimand map), RF4.B1.* (numerical sensitivity), RF4.B2.* (matched-prefix
    factorial), RF4.B3.* (OpenVLA gate NOT_RUN), RF4.FEAS.* (pi0-FAST feasibility);
  * adds a non-destructive field `rf4_note` to existing rows whose use changes (values, CIs and the original fields are
    left exactly as they were; history is kept);
  * adds `_meta.review_followup4` (inputs sha256, counts, commits). The original `_meta` keys are not changed.

Idempotent: every RF4.* row and every `rf4_note` is removed and rebuilt on each run. No number is typed by hand: every
value is copied from the file named in `source_file` / `source_key`.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python c_numbers.py
"""
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone

import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
R4 = f"{POC}/outputs/review_followup4"
PKG = "/root/VLA/paper_quantitative_package"
OUT_JSON = f"{PKG}/numbers.json"
OUT_CSV = f"{PKG}/numbers_sources.csv"

DRIVE = {"domain": "driving", "model": "AutoVLA (Qwen2.5-VL-3B)", "eval_mode": "open-loop 5 s plan (navtest)"}
IMPR = {"domain": "driving", "model": "Impromptu VLA 3B", "eval_mode": "open-loop 5 s waypoints (navtest)"}
SVLA = {"domain": "manipulation", "model": "SpatialVLA-4B", "eval_mode": "closed loop, SimplerEnv Google Robot"}
OVLA = {"domain": "manipulation", "model": "OpenVLA-7B", "eval_mode": "closed loop, LIBERO-Spatial"}
PI0F = {"domain": "manipulation", "model": "pi0-FAST (candidate)", "eval_mode": "-"}
AUDIT = {"domain": "documentation", "model": "-", "eval_mode": "-"}

ST_VER = "VERIFIED_MACHINERY_MANIFEST_SCOPE"
ST_B2 = "VERIFIED_IN_RUN_ALL_UNITS"
ST_NOTVER = "NOT_VERIFIED"
ST_AUDIT = "AUDIT"
ST_NOTRUN = "NOT_RUN"
ST_FEAS = "FEASIBILITY_ONLY"
CAV_B2 = ("B2 new-arm gate PASS (gate_stage0, 16 unit-anchors, controls G1-G6) and identity checks on 100% of main "
          "unit-anchors (noop = NAT_NSRC, NAT_NSRC = bank Normal, pre-anchor executed tokens identical, prefix sha = bank "
          "sha); P1-C input check 2,160/2,160 Normal tokens equal stored P1-C H7; RTX 5090, bf16, B = 1 per cell")
USE = {ST_VER: "usable (state the manifest-scope caveat)", ST_B2: "usable (oracle-conditioned diagnostic; state scope)",
       ST_NOTVER: "descriptive only", ST_AUDIT: "documentation fix", ST_NOTRUN: "not available (not run)",
       ST_FEAS: "planning only (no model run)"}

ROUND = "review_followup4 (2026-10-17/18)"
COMMITS = {"A2_A3": "5523ce3", "A1_A4": "2a31ad7", "B1_B2_protocol": "552b3ac", "FEASIBILITY": "28f1bfd",
           "B2_results": "c7d1bba"}


def f(x):
    if x is None or isinstance(x, bool):
        return x
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return x if x.strip() else None
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return x


def clean(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and math.isnan(v):
            v = None
        elif hasattr(v, "item"):
            v = v.item()
        out[k] = v
    return out


def ci_list(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    if isinstance(s, (list, tuple)):
        return [f(s[0]), f(s[1])]
    s = str(s).strip().strip("[]")
    a, b = s.split(",")
    return [float(a), float(b)]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rel(p):
    return os.path.relpath(p, POC)


def row(rid, **kw):
    base = {"row_id": rid, "block": "RF4", "item": None, "value": None, "unit": None, "ci95": None, "ci95_method": None,
            "p_raw": None, "p_adj": None, "family": None, "code_path": None, "status": None, "status_caveat": "",
            "analysis_type": None, "status_original": "", "package_use": None, "source_file": None,
            "source_key": None, "fields": {}, "round": ROUND}
    base.update(kw)
    base["package_use"] = base["package_use"] or USE.get(base["status"], "")
    return base


# ------------------------------------------------------------------------------------------------------------- A1
def a1_rows(R):
    src = f"{R4}/A1_A4/CI_VALIDITY_AUDIT.csv"
    d = pd.read_csv(src)
    out, notes = [], {}
    for i, r in d.iterrows():
        if pd.isna(r.row_id):
            continue
        rid = r.row_id
        old = R[rid]
        tags = OVLA if rid.startswith("P0E") else SVLA
        status = old["status"]
        out.append(row(f"RF4.A1.{rid}", item=f"A1 post-hoc corrected paired CI for {rid} ({r.block})",
                       value=f(r.estimate), unit="proportion (paired difference)",
                       ci95=[f(r.new_ci95_lo), f(r.new_ci95_hi)], ci95_method=f"{r.new_ci_method} (A1 selection rule)",
                       p_raw=f(r.new_p), p_adj=f(r.new_p_holm), family=r.family if not pd.isna(r.family) else None,
                       **tags, code_path=old.get("code_path"), status=status, status_caveat=old.get("status_caveat", ""),
                       analysis_type=f"POST-HOC CORRECTION (review_followup4 A1); {r.use}",
                       package_use=("descriptive only (10 states per task)" if rid.startswith("P0E.succ_task")
                                    else old.get("package_use")),
                       source_file=rel(src), source_key=f"row {i} (row_id={rid}, clustering={r.clustering})",
                       fields=clean({k: r[k] for k in ("table_a11_b10_c01_d00", "estimand", "old_ci_method", "old_ci95_lo",
                                                       "old_ci95_hi", "old_ci_degeneracy", "new_ci_method", "new_p_method",
                                                       "old_p_equiv_m10", "new_p_equiv_m10", "old_p_equiv_m10_holm",
                                                       "new_p_equiv_m10_holm", "old_p_equiv_m5", "new_p_equiv_m5",
                                                       "old_p_equiv_m5_holm", "new_p_equiv_m5_holm",
                                                       "old_equivalence_judgment_m10", "new_equivalence_judgment_m10",
                                                       "old_equivalence_judgment_m5", "new_equivalence_judgment_m5",
                                                       "old_ci_excludes_0", "new_ci_excludes_0", "newcombe10_ci95",
                                                       "tango_ci95", "n_units", "n_clusters", "clustering", "status")})))
        if rid.startswith("P0E.succ_task"):
            notes[rid] = ("review_followup4 A1: the stored sign-flip-inversion CI is invalid for this all-/mostly-concordant "
                          "paired binary table (zero width or end pinned at 0); never cite it. Corrected Newcombe-10 CI in "
                          f"RF4.A1.{rid}; per-task OpenVLA rows are descriptive only.")
        else:
            notes[rid] = f"review_followup4 A1: post-hoc corrected CI / TOST in RF4.A1.{rid}; original kept here unchanged."
    src = f"{R4}/A1_A4/A1_JUDGMENT_CHANGES.csv"
    j = pd.read_csv(src)
    for i, r in j.iterrows():
        tags = OVLA if str(r.block).startswith("OpenVLA") else SVLA
        out.append(row(f"RF4.A1.chg.{i:02d}", item=f"A1 judgment change: {r.block} {r.experiment if not pd.isna(r.experiment) else ''} "
                       f"{r.config if not pd.isna(r.config) else ''} {r.contrast} {r.task_level} {r.clustering} ({r.judgment})",
                       value=f"{r.old} -> {r.new}", unit="judgment", ci95=ci_list(r.new_ci95),
                       ci95_method="post-hoc corrected (Newcombe 10 / Rao-Scott Newcombe / MOVER)",
                       p_raw=None, p_adj=f(r.new_p_equiv_holm), family=r.family, **tags, code_path=None,
                       status=ST_NOTVER if tags is OVLA or str(r.experiment) == "exp35" else ST_VER,
                       analysis_type=f"POST-HOC CORRECTION (review_followup4 A1); {r.use}",
                       source_file=rel(src), source_key=f"row {i}", fields=clean(r.to_dict())))
    return out, notes


# ------------------------------------------------------------------------------------------------------------- A2
def a2_rows():
    out = []
    src = f"{R4}/A2_A3/a2_checks.json"
    c = json.load(open(src))
    g, nd = c["gate_H9"], c["ninety_percent_definition"]
    items = [("gate_H9.gt_retained", "H9 gate traces: GT position stays GT in the next step", g["gt_positions_retained_next_step"],
              "transitions", f"of {g['gt_positions_retained_next_step']} (removal events {g['gt_removed_events']})", "gate_H9"),
             ("gate_H9.removal_events", "H9 gate traces: GT removal events", g["gt_removed_events"], "events", "", "gate_H9"),
             ("exp9.offsets_eq_rule", "exp 9 stored final-step GT offsets equal the persistent rule", c["exp9"]["stored_offsets_eq_rule"],
              "rows", f"of {c['exp9']['rows_checked']}", "exp9"),
             ("p1b.persistent_retained", "P1-B P_gt_* GT positions retained in every later step",
              c["p1b_persistent"]["gt_retained_all_later_steps"], "positions", f"of {c['p1b_persistent']['gt_positions']}", "p1b_persistent"),
             ("p1b.transient_once", "P1-B T_gt_* GT positions seen only while most recent",
              c["p1b_transient"]["gt_seen_only_when_most_recent"], "positions", f"of {c['p1b_transient']['gt_positions']}", "p1b_transient"),
             ("win3.mean_corrected_positions", "win3: corrected (GT) context positions per unit", nd["win3_mean_corrected_positions"],
              "positions", "365 A- units", "ninety_percent_definition"),
             ("win3.mean_effective_overwrites", "win3: positions whose GT differs from the own token", nd["win3_mean_effective_overwrites"],
              "positions", "365 A- units", "ninety_percent_definition"),
             ("win3.mean_steps_after_last_gt", "win3: generated steps after the last added GT position (GT retained)",
              nd["win3_mean_post_release_steps"], "steps", "formerly called 'post-release steps'", "ninety_percent_definition"),
             ("win3.mean_steps_generated", "steps generated after t*", nd["mean_steps_generated_after_tstar"], "steps", "", "ninety_percent_definition")]
    for k, item, v, unit, note, key in items:
        out.append(row(f"RF4.A2.{k}", item=f"A2 window semantics: {item}", value=f(v), unit=unit, **DRIVE, code_path="H9",
                       status=ST_VER, status_caveat="audit of executed traces / stored records (CPU)", analysis_type=f"AUDIT (A2); {note}",
                       source_file=rel(src), source_key=key))
    src = f"{R4}/A2_A3/a2_p1b_transient_vs_persistent.json"
    t = json.load(open(src))
    for sub in ("common_tstar_le2", "all"):
        if sub not in t:
            continue
        s = t[sub]
        for k in ("amplification_fraction_P_gt_s1_w3", "amplification_fraction_T_gt_s1_w3", "amplification_T_minus_P",
                  "fde5_T_minus_P"):
            if k not in s:
                continue
            out.append(row(f"RF4.A2.p1b.{sub}.{k}", item=f"A2 P1-B transient vs persistent ({sub}): {k}",
                           value=f(s[k]["est"]), unit=s[k].get("unit", "fraction"), ci95=ci_list(s[k]["ci95"]),
                           ci95_method="drive-cluster bootstrap 95% (B = 10,000)", **DRIVE, code_path="P1B", status=ST_VER,
                           analysis_type="POST-HOC (A2); transient vs persistent confounds retention with exposure count",
                           source_file=rel(src), source_key=f"{sub}.{k}",
                           fields={"n_units": s.get("n_units"), "n_scenes": s.get("n_scenes"), "n_drives": s.get("n_drives")}))
    return out


# ------------------------------------------------------------------------------------------------------------- A3
def a3_rows():
    out = []
    src = f"{R4}/A2_A3/a3_parity_summary.json"
    a = json.load(open(src))
    for r in a["rows"]:
        cid = r["comparison_id"]
        n = r.get("n")
        k = r.get("n_exact_equal_0.01m", r.get("n_exact_equal"))
        out.append(row(f"RF4.A3.{cid}", item=f"A3 Impromptu parity: {r['description']}", value=f(k), unit=f"exact matches of {n}",
                       **IMPR, code_path="exp33 (cross_vla_temporal)", status=ST_NOTVER,
                       status_caveat="Impromptu path not parity-gated; numerical floor measured (5-6% scene-level A-/amp flips)",
                       analysis_type="DESCRIPTIVE / UNVERIFIED (A3 diagnosis)", source_file=rel(src),
                       source_key=f"rows[comparison_id={cid}]", fields=clean(r)))
    return out


# ------------------------------------------------------------------------------------------------------------- A4
def a4_rows():
    src = f"{R4}/A1_A4/COHORT_ESTIMAND_MAP.csv"
    d = pd.read_csv(src)
    out = []
    for i, r in d.iterrows():
        out.append(row(f"RF4.A4.{r['id']}", item=f"A4 cohort / estimand map: {r['item']}", value=r["value"], unit="set / definition",
                       **DRIVE if not str(r["cohort"]).lower().startswith(("openvla", "spatialvla")) else AUDIT,
                       status=ST_AUDIT, analysis_type="AUDIT (A4 cohort-estimand map)", source_file=rel(src),
                       source_key=f"row {i} (id={r['id']})", fields=clean(r.to_dict())))
    return out


# ------------------------------------------------------------------------------------------------------------- B1
def b1_rows():
    src = f"{R4}/B/B1/NUMERICAL_SENSITIVITY.csv"
    d = pd.read_csv(src)
    out = []
    for i, r in d.iterrows():
        if r.section == "row_variation":
            rid = f"RF4.B1.row.{r.setting}.{r.path}"
            val, unit, item = f(r.suffix_changed_rate), "share of row-runs with a changed suffix", "row-level suffix change vs S0"
        elif r.section == "intervention_scale_S0":
            rid = f"RF4.B1.scale.{r.contrast}"
            val, unit, item = f(r.endpoint_shift_mean_m), "m (mean endpoint shift)", "intervention scale on the same units (S0)"
        else:
            rid = f"RF4.B1.stab.{r.contrast}.{r.setting}"
            val, unit, item = f(r.delta_fde5_m), "m (delta of mean d_FDE5 vs S0)", "paired-contrast stability"
        out.append(row(rid, item=f"B1 numerical sensitivity: {item} ({r.section})", value=val, unit=unit, **DRIVE,
                       code_path=str(r.path) if not pd.isna(r.path) else None, status=ST_VER,
                       status_caveat="56 dev scenes (Stage-0 rule, take 20), t001, seed 0; RTX 5090",
                       analysis_type="SENSITIVITY of the numerical implementation (B1); not a new effect estimate"
                                     + ("; material = " + str(r.material) if r.section == "contrast_stability" else ""),
                       source_file=rel(src), source_key=f"row {i}", fields=clean(r.to_dict())))
    return out


# ------------------------------------------------------------------------------------------------------------- B2
def b2_rows():
    src = f"{R4}/B/MATCHED_PREFIX_RESULTS.csv"
    d = pd.read_csv(src)
    out = []
    for i, r in d.iterrows():
        rid = f"RF4.B2.{r.cohort}.a{int(r.n_old)}.{r.metric}.{r.contrast}.{r.subset}.{r.estimand}.{r.cluster}"
        prim = (not pd.isna(r.family)) and r.cohort == "p1c"
        at = ("PREREGISTERED primary family (552b3ac)" if prim else
              "preregistered development cohort (descriptive)" if r.cohort == "dev" else
              "preregistered secondary / sensitivity (no family)")
        out.append(row(rid, item=f"B2 matched-prefix factorial: {r.contrast} on {r.metric} ({r.cohort}, estimand {r.estimand}, "
                                 f"anchor {r.anchor_rule}, subset {r.subset}, {r.cluster} clusters)",
                       value=f(r.theta), unit=("proportion" if r.metric in ("severe", "A_plus", "tok_a_changed", "suffix_changed_frac")
                                               else "rad" if "dyaw" in r.metric else "m"),
                       ci95=ci_list(r.boot_ci95), ci95_method=f"{r.cluster}-cluster bootstrap 95%",
                       p_raw=f(r.p_signflip), p_adj=f(r.holm_p), family=(None if pd.isna(r.family) else
                                                                         f"{r.family} (Holm m = 4, alpha {r.family_alpha})"),
                       **DRIVE, code_path="B2 (b2_matched_prefix.py)", status=ST_B2, status_caveat=CAV_B2,
                       analysis_type=f"{at}; ORACLE-CONDITIONED DIAGNOSTIC; one-shot single retained edit (not release); "
                                     "existing drives (not external)",
                       source_file=rel(src), source_key=f"row {i}",
                       fields=clean({k: r[k] for k in ("cohort", "n_old", "anchor_rule", "n_scenes", "n_units", "n_clusters",
                                                       "weighting", "p_cluster_t", "clt_ci95", "boot_ci90",
                                                       "n_informative_clusters", "family_alpha", "reject")})))
    for coh in ("p1c", "dev"):
        s = f"{R4}/B/B2/{coh}_analysis/matched_prefix_descriptives.csv"
        dd = pd.read_csv(s)
        for i, r in dd.iterrows():
            out.append(row(f"RF4.B2.{coh}.a{int(r.n_old)}.denominator.{r.stratum}", item=f"B2 denominators ({coh}, n_old {r.n_old}, {r.stratum})",
                           value=int(r.units), unit="unit-anchors", **DRIVE, code_path="B2", status=ST_B2, status_caveat=CAV_B2,
                           analysis_type="denominator / bank shares (descriptive)", source_file=rel(s), source_key=f"row {i}",
                           fields=clean(r.to_dict())))
    return out


# ------------------------------------------------------------------------------------------------------------- B3 / FEAS
def b3_feas_rows():
    out = [row("RF4.B3.OPENVLA_GATE", item="B3 OpenVLA minimal path verification (untouched / hook no-op / self-patch / "
                                           "cached vs full recompute)", value="NOT_RUN", unit="status", **OVLA,
               code_path="cross_vla (OpenVLA)", status=ST_NOTRUN,
               status_caveat="user decision 2026-10-17; checkpoint openvla/openvla-7b-finetuned-libero-spatial "
                             "(rev 962318cec55ac10993ff0f5f43eda9a270b4c873) not on disk; ~2.3-3 GB free disk",
               analysis_type="OpenVLA stays descriptive / unverified and is excluded from core evidence",
               source_file="outputs/review_followup4/OPENVLA_GATE.md", source_key="whole file"),
           row("RF4.FEAS.pi0fast", item="pi0-FAST feasibility (section 6)", value="feasible with conditions; not run",
               unit="verdict", **PI0F, status=ST_FEAS,
               status_caveat="disk ~20-25 GB needed; JAX on RTX 5090 (sm_120) unverified; cross-step action-context "
                             "interventions NA (no action history in context); WorldVLA mask pair unverified",
               analysis_type="FEASIBILITY ONLY (no download of weights, no model run)",
               source_file="outputs/review_followup4/FEASIBILITY.md", source_key="Verdict")]
    return out


def main():
    J = json.load(open(OUT_JSON))
    R = J["rows"]
    # idempotent: drop previous RF4 rows / notes
    for k in [k for k in R if k.startswith("RF4.")]:
        del R[k]
    for v in R.values():
        v.pop("rf4_note", None)
    a1, notes = a1_rows(R)
    rows = a1 + a2_rows() + a3_rows() + a4_rows() + b1_rows() + b2_rows() + b3_feas_rows()
    ids = [r["row_id"] for r in rows]
    dup = {x for x in ids if ids.count(x) > 1}
    assert not dup, sorted(dup)[:10]
    # notes on existing rows (non-destructive)
    win_note = ("review_followup4 A2: the window keeps the w GT positions in every later context (persistent GT, no "
                "release); 'post-release horizon' = steps after the last added GT position. Allowed wording: 'Three fixed "
                "context positions replaced with ground truth and retained in later contexts recovered about 90% ...'")
    for k in R:
        if k.startswith(("P0C.R9.", "P0C.P1B.")):
            notes.setdefault(k, win_note)
        if k.startswith(("P0A.P1C.", "P0A.DECOMP.")) and k.endswith((".P", "RN", "RV", "DM")):
            notes.setdefault(k, "review_followup4 A4: P = evaluation-frame (E) estimand over 2,191 eligible scenes, estimated "
                                "from 353 sampled scenes with design weights (24 drives); F = 105 scenes / 18 drives.")
        if k.startswith("P2A.") or k.startswith("P2B.") or k.startswith("P2.verdict"):
            if ".all." in k or k.endswith(".all"):
                notes.setdefault(k, "review_followup4 A4: 'all' = the 208 selected mechanism scenes (52 A- + 156 A+), not "
                                    "2,747 / 2,191 scenes or a population; non-significance is not equivalence.")
        if k.startswith("P0E.phaseC"):
            notes.setdefault(k, "review_followup4 A4: 92.6% = share of (frame, sign) units in which AT LEAST ONE downstream "
                                "component (y..yaw, gripper excluded) moved >= the effective injected bins; bins mix m and rad; "
                                "commanded, not realised motion; not task failure.")
    for k, n in notes.items():
        if k in R:
            R[k]["rf4_note"] = n
    for r in rows:
        R[r["row_id"]] = {k: v for k, v in r.items() if k != "row_id"}
    inputs = [f"{R4}/A1_A4/CI_VALIDITY_AUDIT.csv", f"{R4}/A1_A4/A1_JUDGMENT_CHANGES.csv", f"{R4}/A1_A4/COHORT_ESTIMAND_MAP.csv",
              f"{R4}/A2_A3/a2_checks.json", f"{R4}/A2_A3/a2_p1b_transient_vs_persistent.json",
              f"{R4}/A2_A3/a3_parity_summary.json", f"{R4}/B/B1/NUMERICAL_SENSITIVITY.csv",
              f"{R4}/B/MATCHED_PREFIX_RESULTS.csv", f"{R4}/B/B2/p1c_analysis/matched_prefix_descriptives.csv",
              f"{R4}/B/B2/dev_analysis/matched_prefix_descriptives.csv"]
    counts = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
    J["_meta"]["review_followup4"] = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": f"{POC}/scripts/review_followup4/c/c_numbers.py", "n_rf4_rows": len(rows),
        "n_rows_total": len(R) - 0, "n_rows_with_rf4_note": sum(1 for v in R.values() if "rf4_note" in v),
        "rf4_status_counts": counts,
        "status_rule": {ST_B2: CAV_B2, ST_NOTRUN: "not run (B3, user decision 2026-10-17)",
                        ST_FEAS: "feasibility check only"},
        "commits": COMMITS, "inputs_sha256": {rel(p): sha(p) for p in inputs},
        "note": ("Original rows and their values are unchanged; rows whose use changed carry 'rf4_note'. RF4.A1.* hold the "
                 "post-hoc corrected CIs (the original columns stay in the original rows). RF4.B2.* are oracle-conditioned "
                 "diagnostics from a new round (not a retroactive DONE of CP2 P3).")}
    json.dump(J, open(OUT_JSON, "w"), indent=1, default=str)
    # numbers_sources.csv: rewrite base rows (unchanged) + RF4 rows
    with open(OUT_CSV, newline="") as fh:
        base = [r for r in csv.reader(fh)]
    hdr, body = base[0], [r for r in base[1:] if not r[0].startswith("RF4.")]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(body)
        for r in rows:
            ci = r["ci95"] or [None, None]
            w.writerow([r["row_id"], r["block"], r["domain"], r["model"], r["eval_mode"], r["code_path"], r["value"],
                        r["unit"], ci[0], ci[1], r["ci95_method"], r["p_raw"], r["p_adj"], r["family"], r["status"],
                        r["package_use"], r["analysis_type"], r["source_file"], r["source_key"]])
    print(json.dumps(J["_meta"]["review_followup4"], indent=1)[:3000])


if __name__ == "__main__":
    main()
