#!/usr/bin/env python
"""
Official NAVSIM v2 two-stage scoring (run_pdm_score_from_submission.py, unchanged) that additionally saves the
per-group scores the final EPDMS averages, so a log-cluster bootstrap is possible. The official
calculate_individual_mapping_scores is wrapped: for each (orig_token, prev_token) group we store
stage-1 score, Gaussian-weighted stage-2 score and the group score (group1 + group2) / 2, exactly as it computes them.

  GROUP_SCORES_OUT=<groups.json> PYTHONPATH=/root/VLA/navsim_v2 python scripts/navhard_group_scores.py <hydra overrides>
"""
import json
import os
import sys

sys.path.insert(0, "/root/VLA/navsim_v2/navsim/planning/script")
import run_pdm_score_from_submission as R  # noqa: E402

_orig = R.calculate_individual_mapping_scores


def _wrapped(df, all_mappings):
    out = []
    for (o, p), pairs in all_mappings.items():
        g = []
        for s1, s2 in ((o, [x[0] for x in pairs if len(x) > 0]), (p, [x[1] for x in pairs if len(x) > 1])):
            a = R.calculate_weighted_average_score(df[df["token"] == s1])["score"]
            b = R.calculate_weighted_average_score(df[df["token"].isin(s2)])["score"]
            g.append({"stage1_token": s1, "stage1": float(a), "stage2": float(b), "n_stage2": len(s2)})
        out.append({"orig_token": o, "prev_token": p, "group_score": (g[0]["stage1"] * g[0]["stage2"] + g[1]["stage1"] * g[1]["stage2"]) / 2,
                    "parts": g})
    json.dump(out, open(os.environ["GROUP_SCORES_OUT"], "w"))
    return _orig(df, all_mappings)


R.calculate_individual_mapping_scores = _wrapped

if __name__ == "__main__":
    R.main()
