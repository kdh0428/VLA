#!/usr/bin/env python
"""
Oracle picks from the official per-token scores of every candidate (P3, outputs/candidate_oracle/PROTOCOL.md).

For candidate k the official CSV of the run scoring "candidate k everywhere" is read (k = 0: the `normal` run);
oracle17 = argmax over k = 0..16 of the token's official score, oracle16 = argmax over k = 1..16 (ties -> lowest k).
Also writes the per-token score matrix for analysis. Analysis-only upper bound (uses the scored future).

  python scripts/navhard_oracle_picks.py <out_dir> <csv_cand00> <csv_cand01> ... <csv_cand16>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

if __name__ == "__main__":
    out, csvs = sys.argv[1], sys.argv[2:]
    assert len(csvs) == 17, len(csvs)
    os.makedirs(out, exist_ok=True)
    S = {}
    for k, p in enumerate(csvs):
        df = pd.read_csv(p)
        df = df[~df["token"].astype(str).str.startswith("extended_pdm_score")]
        S[k] = dict(zip(df["token"].astype(str), df["score"].astype(float)))
    toks = sorted(set.intersection(*(set(v) for v in S.values())))
    M = np.array([[S[k][t] for k in range(17)] for t in toks])          # (tokens, 17)
    picks = {t: {"oracle17": int(np.argmax(M[i])), "oracle16": int(1 + np.argmax(M[i, 1:]))} for i, t in enumerate(toks)}
    json.dump(picks, open(os.path.join(out, "oracle_picks.json"), "w"))
    np.save(os.path.join(out, "token_scores.npy"), M); json.dump(toks, open(os.path.join(out, "token_scores_tokens.json"), "w"))
    print(json.dumps({"tokens": len(toks), "mean_score_natural": float(M[:, 0].mean()), "mean_oracle17_token": float(M.max(1).mean()),
                      "mean_oracle16_token": float(M[:, 1:].max(1).mean()), "oracle17_is_natural": float(np.mean(M.argmax(1) == 0))}))
