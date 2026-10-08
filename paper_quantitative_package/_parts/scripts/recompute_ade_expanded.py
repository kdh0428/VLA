#!/usr/bin/env python
"""
Recompute ADE5 (in addition to A-/FDE5) for the expanded / held-out best-of-N populations with EXACTLY the
method of scripts/analyze_expanded_best_of_n.py (same pick(), same unit_metrics(), same stub_cot exclusion,
same log-cluster bootstrap: random.Random(0), 2000 resamples, percentile 2.5/97.5; Wilcoxon on paired diffs).
ADE5 = unit_metrics()["ade5"] (mean L2 over the 10 waypoints), an existing metric of analyze_action_history.py.
A-/FDE5 are recomputed too, only as a check that this helper reproduces summary_pooled.json.
CPU only, no torch. Writes nothing outside paper_quantitative_package/_parts/.

  nice -n 19 python recompute_ade_expanded.py <out_json> <run_dir> [<run_dir> ...]
"""
import json, os, random, sys
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest, wilcoxon

SCRIPTS = "/root/VLA/autovla_misalignment_poc/scripts"
sys.path.insert(0, SCRIPTS)
from analyze_action_history import unit_metrics   # noqa: E402
from analyze_best_of_n import pick                # noqa: E402

RULES = ["normal", "max_loglik", "min_entropy", "ranksum", "medoid", "oracle"]


def main():
    out, runs = sys.argv[1], sys.argv[2:]
    units = []
    for run in runs:
        for l in open(os.path.join(run, "records.jsonl")):
            r = json.loads(l)
            if r.get("stub_cot"):
                continue
            r["prev_traj"] = None
            u = {"token": r["token"], "log": r["log"]}
            for rule in RULES:
                x = r["candidates"][pick(r, rule)]
                m = unit_metrics(x["action_idx"], x["trajectory_pred"], x["entropy_steps"], r["gt"], r["trajectory_gt"], 0)
                u[rule] = {"a_minus": float(not m["recovery"]), "failure": float(m["amplification"]), "fde5": m["fde5"], "ade5": m["ade5"]}
            units.append(u)
    by = defaultdict(list)
    for u in units:
        by[u["log"]].append(u)
    logs = sorted(by)
    rng = random.Random(0)
    boots = [[rng.choice(logs) for _ in logs] for _ in range(2000)]
    S = {"runs": runs, "n_scenes": len(units), "n_logs": len(logs),
         "n_natural_a_minus": int(sum(u["normal"]["a_minus"] for u in units)), "rules": {}}
    for rule in RULES:
        o = {}
        for m in ("a_minus", "failure", "fde5", "ade5"):
            d = [u[rule][m] - u["normal"][m] for u in units]
            bs = [np.mean([u[rule][m] - u["normal"][m] for lg in b for u in by[lg]]) for b in boots]
            e = {"mean": float(np.mean([u[rule][m] for u in units])), "diff": float(np.mean(d)),
                 "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
            if m in ("a_minus", "failure"):
                g = sum(1 for u in units if u[rule][m] > u["normal"][m]); ls = sum(1 for u in units if u[rule][m] < u["normal"][m])
                e["count_rule"] = int(sum(u[rule][m] for u in units))
                e["mcnemar"] = {"rule_only": g, "normal_only": ls, "p": 1.0 if g + ls == 0 else float(binomtest(g, g + ls, 0.5).pvalue)}
            else:
                dd = np.array(d)
                e["wilcoxon_p"] = float(wilcoxon(dd).pvalue) if np.any(dd != 0) else 1.0
            o[m] = e
        S["rules"][rule] = o
    json.dump(S, open(out, "w"), indent=1)
    for rule in RULES:
        x = S["rules"][rule]
        print(f"{rule:12s} A- {100*x['a_minus']['mean']:.2f}% ({x['a_minus']['count_rule']}) dA- {100*x['a_minus']['diff']:+.2f} "
              f"[{100*x['a_minus']['ci95'][0]:+.2f},{100*x['a_minus']['ci95'][1]:+.2f}] p {x['a_minus']['mcnemar']['p']:.2g} | "
              f"ADE {x['ade5']['mean']:.3f} d {x['ade5']['diff']:+.3f} [{x['ade5']['ci95'][0]:+.3f},{x['ade5']['ci95'][1]:+.3f}] p {x['ade5']['wilcoxon_p']:.2g} | "
              f"FDE {x['fde5']['mean']:.3f} d {x['fde5']['diff']:+.3f} [{x['fde5']['ci95'][0]:+.3f},{x['fde5']['ci95'][1]:+.3f}] p {x['fde5']['wilcoxon_p']:.2g}")


if __name__ == "__main__":
    main()
