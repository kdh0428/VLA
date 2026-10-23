#!/usr/bin/env python
"""Re-run the scripts behind the core statistics tables inside the unzipped ZIP and compare with the shipped originals.
(review_followup5 version: text cells holding numeric lists such as CI "[lo, hi]" are parsed and compared numerically.)

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

R5 = "outputs/review_followup5"
TARGETS = [
    # name, cwd, argv, outputs (relative to POC), what it regenerates
    ("A_STATS_coverage", f"{S}/review_followup5", ["a_stats_coverage.py"],
     [f"{R5}/A_STATS/A_COVERAGE_SIM.csv", f"{R5}/A_STATS/A_METHOD_SELECTION.json"],
     "coverage study on the real AutoVLA cluster structures and the pre-specified interval selection"),
    ("A_STATS_audit", f"{S}/review_followup5", ["a_stats_audit.py"],
     [f"{R5}/A_STATS/AUTOVLA_CI_AUDIT.csv", f"{R5}/A_STATS/a_stats_audit_summary.json"],
     "old vs new AutoVLA intervals / margin verdicts / MME decisions"),
    ("A_B2_CLUSTER", f"{S}/review_followup5", ["a_b2_cluster.py"],
     [f"{R5}/A_B2_CLUSTER/B2_CLUSTER_RESULTS.csv", f"{R5}/A_B2_CLUSTER/COMPONENT_MAP.csv",
      f"{R5}/A_B2_CLUSTER/B2_CLUSTER_CONTRIBUTIONS.csv"],
     "B2 contrasts on drive / day-vehicle / connected-component clusters with Holm"),
    ("B_NUMERIC_analysis", f"{S}/review_followup5", ["b_numeric_analyze.py"],
     [f"{R5}/B_NUMERIC/B_NUMERIC_RESULTS.csv", f"{R5}/B_NUMERIC/B_NUMERIC_MATERIALITY.csv",
      f"{R5}/B_NUMERIC/b_numeric_units.parquet"],
     "GT-free endpoint excess over every baseline under S0 / SE / SF, paired setting changes, materiality (from the GPU raw)"),
    ("C_power", f"{S}/review_followup5", ["c_power.py"], [f"{R5}/C_DISJOINT/POWER_PRECISION_PLAN.csv"],
     "component-level power / precision plan for a day-and-vehicle-disjoint replication"),
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


def numlist(s):
    """parse a text cell holding a number or a list of numbers; None if it is not numeric."""
    try:
        v = json.loads(s.replace("nan", "NaN").replace("inf", "Infinity"))
    except Exception:
        return None
    v = v if isinstance(v, list) else [v]
    try:
        return np.asarray([np.nan if x is None else float(x) for x in v], float)
    except (TypeError, ValueError):
        return None


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
                diff = xs != ys
                n_num = 0
                for i in np.where(diff.values)[0]:            # numeric lists stored as text, e.g. "[0.12, 0.34]"
                    a_, b_ = numlist(xs.iloc[i]), numlist(ys.iloc[i])
                    if a_ is not None and b_ is not None and len(a_) == len(b_) and \
                            np.allclose(a_, b_, rtol=1e-9, atol=1e-12, equal_nan=True):
                        diff.iloc[i] = False; n_num += 1
                n = int(diff.sum())
                if n:
                    bad.append(f"{c} ({n} cells differ)")
        return ("MATCH_VALUES", "byte difference only (formatting / row metadata)") if not bad else ("MISMATCH", "; ".join(bad)[:900])
    return ("MATCH_VALUES", "json values equal within rtol 1e-9 (timing/location keys ignored: " + ", ".join(sorted(VOLATILE)) + ")") if close(a, b) else ("MISMATCH", "json values differ")


def main():
    fast = os.environ.get("FAST") == "1"
    rows = []
    for name, cwd, argv, outs, what in TARGETS:
        if fast and name == "A_STATS_coverage":
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
