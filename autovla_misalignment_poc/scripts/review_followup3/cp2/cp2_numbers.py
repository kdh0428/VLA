#!/usr/bin/env python
"""CP2 machine-merged number bank (2026-10-14). CPU only; reads existing outputs, writes two NEW files only:

  /root/VLA/paper_quantitative_package/numbers.json          every row: id, value, CI, p, n, status, tags, source
  /root/VLA/paper_quantitative_package/numbers_sources.csv   flat row -> source file / key mapping

Inputs (never modified):
  P0  outputs/review_followup3/P0/numbers_p0.json                        (594 rows, generator p0_manifest_numbers.py)
  P1  outputs/review_followup3/P1/NOOP_CACHE_AUDIT.csv, gate_stage0/judge.log, STATE_RESTORE_AUDIT.csv
  P2  outputs/review_followup3/P2/MOTION_BASELINE_RESULTS.csv, LEARNED_EXTRAPOLATOR_RESULTS.csv, p2_verdict.csv,
      extrap_coverage.csv

Status rule (CP2; AutoVLA Stage-0 gate PASS for H7/H8/H9/P1A on the 21 manifest path-units, commit 47efb95):
  AutoVLA rows whose code path is H7 (Recent-GT/Full-GT), H8 (Reverse-N), H9 (windows/Full-GT) or P1A (self-reference,
  coordinate accumulation, rollout-saving path = C1/P2 input) -> VERIFIED_MACHINERY_MANIFEST_SCOPE. Caveat: runs outside
  the 21-unit manifest are not retro-validated by the gate.
  AutoVLA rows on paths NOT in the manifest (H32 = exp 32 / P1-C DM; P1-B start-position script) stay PROVISIONAL.
  AutoVLA natural-pass rows (selection flow, prevalence, inventory) use no intervention machinery -> GATE_NOT_APPLICABLE.
  SpatialVLA P1-D rows -> PENDING_DECISION_A5 (P1: PASS on actions/state; strict C2 FAIL on the executor bookkeeping
  hash; A5 post-hoc policy-state check 20/20; human decision pending). Exp 35 rows -> NOT_VERIFIED (no restore).
  OpenVLA rows -> NOT_VERIFIED (spec only, no parity run; post-hoc re-analysis of stored records).
No number is typed by hand: every value is copied from the input file named in source_file / source_key.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python cp2_numbers.py
"""
import csv
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone

import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
R3 = f"{POC}/outputs/review_followup3"
PKG = "/root/VLA/paper_quantitative_package"
OUT_JSON = f"{PKG}/numbers.json"
OUT_CSV = f"{PKG}/numbers_sources.csv"

DRIVE = {"domain": "driving", "model": "AutoVLA (Qwen2.5-VL-3B)", "eval_mode": "open-loop 5 s plan (navtest)"}
SVLA = {"domain": "manipulation", "model": "SpatialVLA-4B", "eval_mode": "closed loop, SimplerEnv Google Robot"}
OVLA = {"domain": "manipulation", "model": "OpenVLA-7B", "eval_mode": "closed loop, LIBERO-Spatial"}
OVLA_OFF = {"domain": "manipulation", "model": "OpenVLA-7B", "eval_mode": "offline within-step decode (LIBERO-Spatial frames)"}
AUDIT = {"domain": "documentation", "model": "-", "eval_mode": "-"}

ST_VER = "VERIFIED_MACHINERY_MANIFEST_SCOPE"
ST_PROV = "PROVISIONAL"
ST_NA = "GATE_NOT_APPLICABLE"
ST_PEND = "PENDING_DECISION_A5"
ST_NOTVER = "NOT_VERIFIED"
ST_AUDIT = "AUDIT"
CAVEAT_VER = ("AutoVLA Stage-0 gate PASS (commit 47efb95) covers the 21 manifest path-units at HEAD 817d166 + harness "
              "0ef9334 (RTX 5090, bf16); the original full runs outside the manifest are not retro-validated")
CAVEAT_PROV = ("code path not in the Stage-0 manifest (H32 = exp 32 / P1-C DM, or the P1-B start-position script); "
               "not verified by the gate")
CAVEAT_NA = "natural HF-generate pass / data inventory: no intervention machinery involved"
CAVEAT_PEND = ("SpatialVLA P1 audit: PASS on executed actions and simulator state; strict pre-fixed C2 rule FAIL 0/20 "
               "(executor bookkeeping hash incl. shadow-cf dict); A5 post-hoc policy-state check 20/20; human decision "
               "pending (AMENDMENT_P1_01 A5); the 20-unit audit does not retro-validate P1-D runs")
CAVEAT_EXP35 = "exp 35: no state restore / state hash; not covered by the SpatialVLA P1 audit"
CAVEAT_OVLA = "OpenVLA intervention path: spec only (INTERVENTION_SPEC M-O1), no parity run; post-hoc re-analysis"

USE = {ST_VER: "usable (state the manifest-scope caveat)", ST_NA: "usable",
       ST_PROV: "do not enter as final (PROVISIONAL)", ST_PEND: "hold final wording until the A5 decision",
       ST_NOTVER: "descriptive only", ST_AUDIT: "documentation fix"}


def f(x):
    if x is None:
        return None
    try:
        if isinstance(x, str) and x.strip() == "":
            return None
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return x


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------------------------ P0
def p0_path(rid, r):
    """AutoVLA code path of a P0 row (None for non-AutoVLA rows)."""
    if rid.startswith(("P0A.FLOW", "P0A.PREV", "P0D.DISJ")):
        return "natural"
    if rid.startswith("P0C.plus437"):
        return "P1A"
    if rid.startswith("P0C.R9"):
        return "H9"
    if rid.startswith("P0C.P1B"):
        return "P1B"
    if rid.startswith("P0A.DECOMP."):
        return {"RN": "H7", "RV": "H8", "DM": "H32"}[rid.split(".")[2]]
    if rid.startswith(("P0A.P1C.", "P0A.DEV.")):
        return {"RN": "H7", "RV": "H8", "DM": "H32"}[rid.split(".")[2]]
    if rid.startswith("P0D-"):
        c = r.get("contrast", "")
        if "Recent" in c or " RN " in f" {c} ":
            return "H7"
        if "Reverse" in c or " RV " in f" {c} ":
            return "H8"
        if "Direction" in c or " DM " in f" {c} ":
            return "H32"
        raise ValueError(f"cannot map {rid}: {c}")
    return None


CI_ORDER = [("ci95_boot", "drive-cluster bootstrap 95%"), ("ci95_signflip_inv_pp", "sign-flip inversion 95% (states)"),
            ("ci95_signflip", "sign-flip inversion 95%"), ("ci95_state_boot", "initial-state bootstrap 95%"),
            ("ci95", "95% (see source; bootstrap for +4.37 m rows)"), ("ci95_signflip_inv", "sign-flip inversion 95%"),
            ("ci95_pp", "95% (see source)"), ("ci95_boot_pp", "bootstrap 95%")]


def p0_rows():
    src = f"{R3}/P0/numbers_p0.json"
    rows = json.load(open(src))["rows"]
    out = []
    for rid, r in rows.items():
        path = p0_path(rid, r)
        st0 = r.get("status", "")
        analysis = st0.split(";", 1)[1].strip() if ";" in st0 else ("" if st0.startswith("PROVISIONAL") else st0)
        if rid.startswith("P0C.DOC"):
            tags, status, cav = AUDIT, ST_AUDIT, ""
        elif rid.startswith("P0B"):
            tags = SVLA
            if ".exp35." in rid:
                status, cav = ST_NOTVER, CAVEAT_EXP35
            else:
                status, cav = ST_PEND, CAVEAT_PEND
        elif rid.startswith("P0E"):
            tags = OVLA if rid.startswith("P0E.succ") else OVLA_OFF
            status, cav = ST_NOTVER, CAVEAT_OVLA
        else:
            tags = DRIVE
            if path in ("H7", "H8", "H9", "P1A"):
                status, cav = ST_VER, CAVEAT_VER
            elif path in ("H32", "P1B"):
                status, cav = ST_PROV, CAVEAT_PROV
            else:
                status, cav = ST_NA, CAVEAT_NA
                if rid.startswith("P0D.DISJ"):
                    cav += "; expected-scene columns use P1-C natural-pass rates (planning approximation)"
        value, unit = r.get("value"), r.get("unit")
        if value is None and "value_pp" in r:
            value, unit = r["value_pp"], "pp"
        if value is None and "value_m" in r:
            value, unit = r["value_m"], "m"
        ci, ci_m = None, None
        for k, name in CI_ORDER:
            if isinstance(r.get(k), list) and any(v is not None for v in r[k]):
                ci, ci_m = r[k], f"{name} [{k}]"
                break
        p_adj = r.get("p_holm")
        out.append({"row_id": rid, "block": "P0", "item": r.get("item"), "value": f(value), "unit": unit,
                    "ci95": ci, "ci95_method": ci_m, "p_raw": f(r.get("p_raw")), "p_adj": f(p_adj),
                    "family": r.get("family") or r.get("holm_family"), **tags, "code_path": path, "status": status,
                    "status_caveat": cav, "analysis_type": analysis, "status_original": st0,
                    "package_use": USE[status], "source_file": r.get("source_path"),
                    "source_key": f"outputs/review_followup3/P0/numbers_p0.json:rows['{rid}']",
                    "fields": r})
    return out


# ------------------------------------------------------------------------------------------------ P1
def p1_rows():
    out = []
    src = f"{R3}/P1/NOOP_CACHE_AUDIT.csv"
    d = pd.read_csv(src)
    for i, r in d.iterrows():
        rid = f"P1.AV.{r.control}.{r.path}.{r.decoding}"
        out.append({"row_id": rid, "block": "P1", "item": "AutoVLA Stage-0 no-op/cache parity (control x path x decoding)",
                    "value": int(r.n_fail), "unit": "failing rows", "ci95": None, "ci95_method": None, "p_raw": None,
                    "p_adj": None, "family": None, **DRIVE, "code_path": r.path, "status": ST_AUDIT,
                    "status_caveat": "verification, manifest scope (21 path-units)", "analysis_type": "P1-B gate",
                    "status_original": "", "package_use": USE[ST_AUDIT], "source_file": "outputs/review_followup3/P1/NOOP_CACHE_AUDIT.csv",
                    "source_key": f"row {i} (control={r.control}, path={r.path}, decoding={r.decoding})",
                    "fields": {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in r.items()}})
    jl = f"{R3}/P1/gate_stage0/judge.log"
    m = re.search(r"(\{[^}]*'H7'[^}]*\})\s+(\d+) failures", open(jl).read())
    verdicts = json.loads(m.group(1).replace("'", '"'))
    for p, v in verdicts.items():
        out.append({"row_id": f"P1.AV.verdict.{p}", "block": "P1", "item": "AutoVLA Stage-0 gate verdict per path",
                    "value": v, "unit": "verdict", "ci95": None, "ci95_method": None, "p_raw": None, "p_adj": None,
                    "family": None, **DRIVE, "code_path": p, "status": ST_AUDIT,
                    "status_caveat": CAVEAT_VER, "analysis_type": "P1-B gate (PROTOCOL_P1 §6)", "status_original": "",
                    "package_use": USE[ST_AUDIT], "source_file": "outputs/review_followup3/P1/gate_stage0/judge.log",
                    "source_key": f"judge dict['{p}'] (total failures {m.group(2)})", "fields": {}})
    src = f"{R3}/P1/STATE_RESTORE_AUDIT.csv"
    s = pd.read_csv(src)
    for ctrl, g in s.groupby("control"):
        agg = {"n_units": len(g)}
        for col in ("pass", "pass_strict_prefixed_rule", "pass_policy_state_A5", "actions_bytes_equal",
                    "b1_equal_batch8", "generate_equal_batch8", "restore_ok"):
            v = g[col].dropna()
            if len(v):
                agg[f"{col}_true"] = int(v.astype(bool).sum())
                agg[f"{col}_n"] = int(len(v))
        agg["tcp_maxabs_m_max"] = f(g.tcp_maxabs_m.max())
        for k, v in agg.items():
            if k == "n_units":
                continue
            out.append({"row_id": f"P1.SV.{ctrl}.{k}", "block": "P1",
                        "item": "SpatialVLA P1-B/P1-C restore audit (aggregated over arm comparisons)",
                        "value": v, "unit": "count" if not k.endswith("_m_max") else "m", "ci95": None,
                        "ci95_method": None, "p_raw": None, "p_adj": None, "family": None, **SVLA, "code_path": "P1-D runner",
                        "status": ST_AUDIT, "status_caveat": CAVEAT_PEND if ctrl == "C2" else "20-unit audit scope",
                        "analysis_type": "P1 audit (PROTOCOL_P1 §8)" + ("; A5 post hoc" if "A5" in k else ""),
                        "status_original": "", "package_use": USE[ST_AUDIT],
                        "source_file": "outputs/review_followup3/P1/STATE_RESTORE_AUDIT.csv",
                        "source_key": f"control=={ctrl}: {k} (n_units {agg['n_units']})", "fields": {}})
    return out


# ------------------------------------------------------------------------------------------------ P2
def p2_common(r, analysis):
    return {"block": "P2", **DRIVE, "code_path": "P1A", "status": ST_VER,
            "status_caveat": CAVEAT_VER + "; inputs = P1-A rollouts (rollout-saving path verified by c5d / save_path)",
            "analysis_type": analysis, "status_original": r.get("status", ""), "package_use": USE[ST_VER]}


def clean(r):
    return {k: (None if (isinstance(v, float) and math.isnan(v)) else (v.item() if hasattr(v, "item") else v))
            for k, v in r.items()}


def p2_rows():
    out = []
    d = pd.read_csv(f"{R3}/P2/MOTION_BASELINE_RESULTS.csv")
    for i, r in d.iterrows():
        rid = f"P2A.{r.row_type}.{r.family}.{r.group}.{r.metric}.{r.baseline}"
        priv = r.baseline == "Geo"
        out.append({"row_id": rid, "item": f"P2-A motion baseline: {r.quantity} ({r.metric_desc})",
                    "value": f(r.est_scene), "unit": "m" if r.metric in ("end", "dref", "dgt") else ("m/0.5 s" if r.metric == "sfx" else "m*s"),
                    "ci95": [f(r.ci95_lo), f(r.ci95_hi)], "ci95_method": "drive-cluster bootstrap 95% (B=10,000)",
                    "p_raw": f(r.p_signflip), "p_adj": None if priv else f(r.p_holm_m4),
                    "family": None if (priv or r.row_type == "level") else f"Holm m=4 kinematic baselines within ({r.family},{r.group},{r.metric})",
                    **p2_common(r, f"POST-HOC (spec locked before new-baseline results); {r.role}"
                                + ("; Geo = privileged control, outside Holm" if priv else "")),
                    "source_file": "outputs/review_followup3/P2/MOTION_BASELINE_RESULTS.csv",
                    "source_key": f"row {i} (row_type={r.row_type}, family={r.family}, group={r.group}, metric={r.metric}, baseline={r.baseline})",
                    "fields": clean(r.to_dict())})
    d = pd.read_csv(f"{R3}/P2/LEARNED_EXTRAPOLATOR_RESULTS.csv")
    short = {"primary (all units, seed-mean Delta_E)": "primary", "sensitivity: out-of-range units excluded": "oor_excl",
             "level Delta_E (seed mean)": "level"}
    for i, r in d.iterrows():
        a = short.get(r.analysis, f"seed{int(r.train_seed)}" if not pd.isna(r.train_seed) else r.analysis)
        rid = f"P2B.{r.row_type}.{r.family}.{r.group}.{r.metric}.{r.comparator.replace('E:', 'E_')}.{a}"
        out.append({"row_id": rid, "item": f"P2-B learned non-AR motion extrapolator: {r.quantity} ({r.metric_desc})",
                    "value": f(r.est_scene), "unit": "m" if r.metric in ("end", "dref", "dgt") else ("m/0.5 s" if r.metric == "sfx" else "m*s"),
                    "ci95": [f(r.ci95_lo), f(r.ci95_hi)], "ci95_method": "drive-cluster bootstrap 95% (B=10,000)",
                    "p_raw": f(r.p_signflip), "p_adj": f(r.p_holm_m2),
                    "family": f"Holm m=2 extrapolators within ({r.family},{r.group},{r.metric})" if a == "primary" else None,
                    **p2_common(r, "protocol locked 2026-10-14T14:44:28Z before training/evaluation; " + r.analysis),
                    "source_file": "outputs/review_followup3/P2/LEARNED_EXTRAPOLATOR_RESULTS.csv",
                    "source_key": f"row {i} (row_type={r.row_type}, family={r.family}, group={r.group}, metric={r.metric}, comparator={r.comparator}, analysis={r.analysis})",
                    "fields": clean(r.to_dict())})
    d = pd.read_csv(f"{R3}/P2/p2_verdict.csv")
    for i, r in d.iterrows():
        out.append({"row_id": f"P2.verdict.{r.stratum}", "item": "P2 combined rule (family G, M1): beyond motion extrapolation",
                    "value": bool(r.beyond_motion_extrapolation), "unit": "bool", "ci95": None, "ci95_method": None,
                    "p_raw": None, "p_adj": None, "family": None,
                    **p2_common(r, "locked decision rule (P2-A spec 2.4 + P2-B protocol §8)"),
                    "source_file": "outputs/review_followup3/P2/p2_verdict.csv", "source_key": f"row {i} (stratum={r.stratum})",
                    "fields": clean(r.to_dict())})
    d = pd.read_csv(f"{R3}/P2/extrap_coverage.csv")
    d = d[d.table != "per_j_reference"]
    for i, r in d.iterrows():
        out.append({"row_id": f"P2B.coverage.{r.family}.{r.group}", "item": "P2-B prefix coverage: share of units out of range",
                    "value": f(r.share_unit_out_of_range), "unit": "proportion", "ci95": None, "ci95_method": None,
                    "p_raw": None, "p_adj": None, "family": None,
                    **p2_common(r, "descriptive coverage (locked rule)"),
                    "source_file": "outputs/review_followup3/P2/extrap_coverage.csv",
                    "source_key": f"row {i} (family={r.family}, group={r.group})", "fields": clean(r.to_dict())})
    return out


def main():
    rows = p0_rows() + p1_rows() + p2_rows()
    ids = [r["row_id"] for r in rows]
    dup = {x for x in ids if ids.count(x) > 1}
    assert not dup, f"duplicate row ids: {sorted(dup)[:10]}"
    inputs = [f"{R3}/P0/numbers_p0.json", f"{R3}/P1/NOOP_CACHE_AUDIT.csv", f"{R3}/P1/gate_stage0/judge.log",
              f"{R3}/P1/STATE_RESTORE_AUDIT.csv", f"{R3}/P2/MOTION_BASELINE_RESULTS.csv",
              f"{R3}/P2/LEARNED_EXTRAPOLATOR_RESULTS.csv", f"{R3}/P2/p2_verdict.csv", f"{R3}/P2/extrap_coverage.csv"]
    counts = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
    meta = {"created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "checkpoint": "CP2",
            "generator": f"{POC}/scripts/review_followup3/cp2/cp2_numbers.py", "n_rows": len(rows),
            "status_counts": counts,
            "status_rule": {ST_VER: CAVEAT_VER, ST_PROV: CAVEAT_PROV, ST_NA: CAVEAT_NA, ST_PEND: CAVEAT_PEND,
                            ST_NOTVER: f"{CAVEAT_EXP35}; {CAVEAT_OVLA}", ST_AUDIT: "verification / document checks"},
            "package_use": USE,
            "commits": {"P1_protocol": "0ef9334", "P0": "e9907fc", "P2": "5b158f2", "P1_results": "47efb95"},
            "inputs_sha256": {os.path.relpath(p, POC): sha(p) for p in inputs},
            "note": "No hand-typed numbers. 'fields' holds the full source record. CI = 95% CI by ci95_method."}
    json.dump({"_meta": meta, "rows": {r["row_id"]: {k: v for k, v in r.items() if k != "row_id"} for r in rows}},
              open(OUT_JSON, "w"), indent=1, default=str)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "block", "domain", "model", "eval_mode", "code_path", "value", "unit", "ci95_lo", "ci95_hi",
                    "ci95_method", "p_raw", "p_adj", "family", "status", "package_use", "analysis_type", "source_file",
                    "source_key"])
        for r in rows:
            ci = r["ci95"] or [None, None]
            w.writerow([r["row_id"], r["block"], r["domain"], r["model"], r["eval_mode"], r["code_path"], r["value"],
                        r["unit"], ci[0], ci[1], r["ci95_method"], r["p_raw"], r["p_adj"], r["family"], r["status"],
                        r["package_use"], r["analysis_type"], r["source_file"], r["source_key"]])
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
