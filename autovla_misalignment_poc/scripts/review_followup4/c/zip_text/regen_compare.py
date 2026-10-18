#!/usr/bin/env python
"""Re-run the scripts behind the core statistics tables inside the unzipped ZIP and compare with the shipped originals.

  python regen_compare.py <unzipped>/VLA <python>

Writes <unzipped>/REGEN_RESULTS.csv and .json (one row per compared file). Original outputs are moved to
<unzipped>/_original_outputs/ before each script runs, so a script that fails cannot leave the original in place and
pass by accident.
"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

V, PY = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(V)
POC = f"{V}/autovla_misalignment_poc"
S, O = f"{POC}/scripts", f"{POC}/outputs"
ORIG = f"{HERE}/_original_outputs"
B2 = f"{O}/review_followup4/B/B2"

TARGETS = [
    # name, cwd, argv, outputs (relative to POC), what it regenerates
    ("A1_coverage_selection", f"{S}/review_followup4", ["a1_coverage_sim.py"],
     ["outputs/review_followup4/A1_A4/A1_METHOD_SELECTION.json", "outputs/review_followup4/A1_A4/A1_COVERAGE_SIM_iid.csv",
      "outputs/review_followup4/A1_A4/A1_COVERAGE_SIM_cluster.csv", "outputs/review_followup4/A1_A4/A1_TOST_SIZE_SIM.csv"],
     "A1 coverage study and pre-specified CI method selection"),
    ("A1_corrected_CIs", f"{S}/review_followup4", ["a1_ci_audit.py"],
     ["outputs/review_followup4/A1_A4/CI_VALIDITY_AUDIT.csv", "outputs/review_followup4/A1_A4/A1_JUDGMENT_CHANGES.csv",
      "outputs/review_followup4/A1_A4/A1_OPENVLA_2x2.csv", "outputs/review_followup4/A1_A4/A1_DEGENERACY_DEMO.json"],
     "A1 old vs corrected paired CIs, TOST, Holm, judgment changes; OpenVLA paired 2x2 tables"),
    ("B2_p1c", f"{S}/review_followup4/b",
     ["b2_analyze.py", "--run", f"{B2}/p1c_main", "--cohort", "p1c", "--cohort-manifest", f"{B2}/p1c_cohort_t4.json",
      "--out", f"{B2}/p1c_analysis"],
     ["outputs/review_followup4/B/B2/p1c_analysis/MATCHED_PREFIX_RESULTS.csv",
      "outputs/review_followup4/B/B2/p1c_analysis/matched_prefix_descriptives.csv",
      "outputs/review_followup4/B/B2/p1c_analysis/unit_contrasts.parquet"],
     "B2 matched-prefix contrasts, P1-C cohort (F / S / P(E4))"),
    ("B2_dev", f"{S}/review_followup4/b",
     ["b2_analyze.py", "--run", f"{B2}/dev_main", "--cohort", "dev", "--out", f"{B2}/dev_analysis"],
     ["outputs/review_followup4/B/B2/dev_analysis/MATCHED_PREFIX_RESULTS.csv",
      "outputs/review_followup4/B/B2/dev_analysis/matched_prefix_descriptives.csv",
      "outputs/review_followup4/B/B2/dev_analysis/unit_contrasts.parquet"],
     "B2 matched-prefix contrasts, development cohort"),
    ("P0A_population", f"{S}/review_followup3", ["p0_a_population.py"],
     ["outputs/review_followup3/P0/POPULATION_EFFECTS.csv", "outputs/review_followup3/P0/POPULATION_selection_flow.csv",
      "outputs/review_followup3/P0/POPULATION_prevalence.csv", "outputs/review_followup3/P0/POPULATION_decomposition.csv",
      "outputs/review_followup3/P0/POPULATION_p1c_denominators.csv"],
     "P0-A selection flow, A- frequencies, conditional (F) and evaluation-frame (P) effects, P = F + S decomposition"),
    ("P2_motion_baselines", f"{S}/review_followup3", ["p2_motion_baselines.py"],
     ["outputs/review_followup3/P2/MOTION_BASELINE_RESULTS.csv", "outputs/review_followup3/P2/p2_units.parquet",
      "outputs/review_followup3/P2/baseline_fit_diagnostics.csv"],
     "P2 motion-baseline table (CM / CV0 / CTRV-K3 / CTRA-2 / Geo)"),
]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load(p):
    if p.endswith(".csv"):
        return pd.read_csv(p)
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    return json.load(open(p))


VOLATILE = {"runtime_s", "created_utc", "elapsed_s", "wall_s", "run"}   # timing / location only


def close(a, b, rtol=1e-9, atol=1e-12):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a if k not in VOLATILE)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    return a == b


def norm(v):
    """text form of a cell; NaN -> '<NA>'; the unzipped location is mapped back to /root/VLA (absolute input paths
    that a script writes into a 'source' column differ only by location)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "<NA>"
    s = json.dumps(list(map(str, v))) if isinstance(v, (list, np.ndarray)) else str(v)
    return s.replace(V, "/root/VLA")


def compare(new, old):
    if sha(new) == sha(old):
        return "IDENTICAL_BYTES", ""
    a, b = load(new), load(old)
    if isinstance(a, pd.DataFrame):
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            return "MISMATCH", f"shape/columns {a.shape} vs {b.shape}"
        bad = []
        for c in a.columns:
            x, y = a[c], b[c]
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                ok = np.isclose(x.astype(float), y.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True)
                if not ok.all():
                    d = np.nanmax(np.abs(x.astype(float) - y.astype(float)))
                    bad.append(f"{c} ({int((~ok).sum())} cells, max |diff| {d:.3g})")
            else:
                xs, ys = x.map(norm), y.map(norm)
                n = int((xs != ys).sum())
                if n:
                    bad.append(f"{c} ({n} cells differ as text)")
        return ("MATCH_VALUES", "byte difference only (formatting / row metadata)") if not bad else ("MISMATCH", "; ".join(bad)[:900])
    return ("MATCH_VALUES", "json values equal within rtol 1e-9 (timing/location keys ignored: " + ", ".join(sorted(VOLATILE)) + ")") if close(a, b) else ("MISMATCH", "json values differ")


def main():
    fast = os.environ.get("FAST") == "1"
    rows = []
    for name, cwd, argv, outs, what in TARGETS:
        if fast and name == "A1_coverage_selection":
            continue
        for rel in outs:
            src, dst = f"{POC}/{rel}", f"{ORIG}/{rel}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(src):
                shutil.move(src, dst)
        t0 = time.time()
        p = subprocess.run([PY] + argv, cwd=cwd, capture_output=True, text=True)
        dt = time.time() - t0
        open(f"{HERE}/regen_{name}.log", "w").write(p.stdout + "\n--- stderr ---\n" + p.stderr)
        for rel in outs:
            new, old = f"{POC}/{rel}", f"{ORIG}/{rel}"
            if p.returncode != 0 or not os.path.exists(new):
                verdict, note = "NOT_REGENERATED", f"exit {p.returncode}; see regen_{name}.log"
            elif not os.path.exists(old):
                verdict, note = "NO_ORIGINAL", ""
            else:
                verdict, note = compare(new, old)
            rows.append({"target": name, "what": what, "file": rel, "verdict": verdict, "note": note,
                         "seconds": round(dt, 1), "sha256_original": sha(old) if os.path.exists(old) else "",
                         "sha256_regenerated": sha(new) if os.path.exists(new) else ""})
            print(f"{name:22s} {verdict:16s} {rel} {note}", flush=True)
    pd.DataFrame(rows).to_csv(f"{HERE}/REGEN_RESULTS.csv", index=False)
    json.dump(rows, open(f"{HERE}/REGEN_RESULTS.json", "w"), indent=1)


if __name__ == "__main__":
    main()
