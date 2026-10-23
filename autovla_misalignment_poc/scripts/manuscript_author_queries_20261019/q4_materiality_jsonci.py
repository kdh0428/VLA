#!/usr/bin/env python
"""Author query Q4: re-write review_followup5 B_NUMERIC_MATERIALITY.csv with the CI as a JSON list of Python floats
(parsed with json.loads, never eval). Values unchanged. Output: B_NUMERIC_MATERIALITY_jsonci.csv.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python q4_materiality_jsonci.py
"""
import json
import pandas as pd

SRC = "/root/VLA/autovla_misalignment_poc/outputs/review_followup5/B_NUMERIC/B_NUMERIC_MATERIALITY.csv"
OUT = "/root/VLA/autovla_misalignment_poc/outputs/manuscript_author_queries_20261019/B_NUMERIC_MATERIALITY_jsonci.csv"
d = pd.read_csv(SRC)
d["Delta_ci95"] = [json.dumps([float(x) for x in json.loads(s)]) for s in d.Delta_ci95]
d.to_csv(OUT, index=False)
print(d.Delta_ci95.head(2).tolist())
