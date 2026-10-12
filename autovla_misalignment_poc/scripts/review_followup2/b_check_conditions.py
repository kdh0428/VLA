#!/usr/bin/env python
"""
B1 (review_followup2, 2026-10-13): verify from the RECORDS what each condition put in execution and in context.
POST-HOC, CPU only, read-only. For every chunk generated inside the intervention schedule:
  exec_step1 = executed step-1 translation token, ctx_step1 = context step-1 translation token (rec["ctx"] if logged,
  else = exec), nat1 = greedy step-1 translation token g at that observation.
Counts per experiment x config x mode: exec==g, exec!=g, ctx==g, ctx!=g (and P1-D: exec==p / ctx==p with the recorded
p, offset != 0 for FD/CD). Also counts chunks outside the schedule that carry any intervention (must be 0).
Writes condition_record_check.csv in the run dir.
  nice -n 19 /root/miniforge3/envs/autovla/bin/python b_check_conditions.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import pandas as pd

from b_extract import SRC, OUT, parse_cond


def sched(exp, t):
    return (t in (8, 12, 16, 20)) if exp == "P1D" else (8 <= t < 24)


def main():
    C = defaultdict(lambda: defaultdict(int))
    for exp, paths in SRC.items():
        for p in paths:
            for line in open(p):
                ep = json.loads(line)
                cfg, dr, mode = parse_cond(exp, ep)
                k = (exp, cfg, dr, mode)
                for r in ep["rec"]:
                    if "exec" not in r:
                        continue
                    e1 = r["exec"][0][0]
                    c1 = r["ctx"][0][0] if r.get("ctx") else e1
                    g = r["nat1"]
                    if mode == "gen":
                        continue
                    if sched(exp, r["t"]) and r["t"] >= 8:
                        C[k]["chunks_in_schedule"] += 1
                        C[k]["exec==g"] += int(e1 == g)
                        C[k]["ctx==g"] += int(c1 == g)
                        C[k]["suffix_logged_differs_from_exec"] += int(bool(r.get("ctx")) and r["ctx"][1:] != r["exec"][1:])
                        if "inj" in r:
                            C[k]["exec==p"] += int(e1 == r["inj"]["p"])
                            C[k]["ctx==p"] += int(c1 == r["inj"]["p"])
                            C[k]["offset_nonzero"] += int(any(abs(x) > 0 for x in r["inj"]["offset"]))
                    else:
                        C[k]["chunks_outside"] += 1
                        C[k]["outside_with_intervention"] += int(e1 != g or c1 != g)
    rows = [dict(experiment=k[0], config=k[1], direction=k[2], mode=k[3], **v) for k, v in sorted(C.items())]
    df = pd.DataFrame(rows).fillna(0)
    df.to_csv(os.path.join(OUT, "condition_record_check.csv"), index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
